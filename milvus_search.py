from milvus_client import MilvusVectorDB
from utils import get_embedding, rerank_documents
import json
import uuid
from bs4 import BeautifulSoup
import re
# 初始化logger，支持文件和控制台，文件用rotate模式
import logging
from logging.handlers import RotatingFileHandler
# 设置日志记录器
logger = logging.getLogger('milvus_search_logger')
logger.setLevel(logging.INFO)
# 创建一个处理器，用于写入日志文件，设置最大文件大小为10MB
handler = RotatingFileHandler('milvus_search.log', maxBytes=10*1024*1024, backupCount=5)
# 创建一个处理器，用于输出到控制台
console_handler = logging.StreamHandler()
# 设置处理器的日志级别
handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)
# 创建一个格式化器，并将其添加到处理器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# 将处理器添加到记录器
logger.addHandler(handler)
logger.addHandler(console_handler)

class MilvusDocCache:

    def __init__(self, engine: MilvusVectorDB, pivot_id: int, expansion_budget: int = 200):
        self.engine = engine
        self.cache = {}
        self.pivot_id = pivot_id
        self.expansion_budget = expansion_budget
        self.start_position = pivot_id
        self.end_position = pivot_id
        self.init = self._init_cache()


    def _init_cache(self):
        if self.pivot_id:
            id_start = self.pivot_id - int(self.expansion_budget/2)
            id_end = self.pivot_id + int(self.expansion_budget/2)
            query_results = self.engine.query_by_id_range(
                start_id=id_start,
                end_id=id_end,
            )
            for entity in query_results:
                self.cache[entity.get('doc_id')] = entity
            self.start_position = query_results[0].get('id', self.pivot_id)
            self.end_position = query_results[-1].get('id', self.pivot_id)
            return True
        else:
            return False
        
    def initialized(self):
        return self.init
    
    def get_entity(self, doc_id: str):
        """
        Get the entity from the cache or fetch it from the database if not cached.
        :param doc_id: The document ID to retrieve.
        :return: The entity corresponding to the document ID.
        """
        if doc_id in self.cache:
            return self.cache[doc_id]
        else:
            return None
        
    def do_expansion(self):
        """
        Expand the cache by fetching entities around the pivot ID.
        :return: A list of entities fetched from the database.
        """
        if not self.initialized():
            return False
        
        id_start = self.start_position - int(self.expansion_budget/2)
        id_end = self.end_position + int(self.expansion_budget/2)
        upper_results = self.engine.query_by_id_range(
            start_id=id_start,
            end_id=self.start_position,
        )

        lower_results = self.engine.query_by_id_range(
            start_id=self.end_position,
            end_id=id_end,
        )

        # refresh the cache
        for entity in upper_results + lower_results:
            self.cache[entity.get('doc_id')] = entity
        if upper_results and len(upper_results) > 0:
            self.start_position = upper_results[0].get('id', self.pivot_id)
        if lower_results and len(lower_results) > 0:
            self.end_position = lower_results[-1].get('id', self.pivot_id)


class MilvusSearch:
    def __init__(self, uri:str, token: str, collection_name: str):
        self.milvus_db = MilvusVectorDB(uri=uri, token=token, collection_name=collection_name)

    def search_sparse(self, query: str, limit: int = 10) -> list:
        """
        Perform a sparse search in the Milvus database.

        :param query: The query string to search for.
        :param top_k: The number of top results to return.
        :return: A list of search results.
        """
        return self.milvus_db.search_by_text_sparse(query_text=[query],limit=limit)

    def search_dense(self, embedding: list[float], limit: int = 10) -> list:
        """
        Perform a dense search in the Milvus database.

        :param query: The query string to search for.
        :param top_k: The number of top results to return.
        :return: A list of search results.
        """

        return self.milvus_db.search_by_text_dense(query_vectors=[embedding],limit=limit)
    
    def rrf_fusion(self, results_sparse: list, results_dense: list, k: int = 60,
                   sparse_weight: float = 0.4, dense_weight: float = 0.6) -> list:
        """
        Perform RRF fusion on the search results.

        :param results_sparse: The sparse search results.
        :param results_dense: The dense search results.
        :param k: The parameter for RRF fusion.
        :return: A list of fused search results.
        """
        scores = {}
        all_results = {}
        for rank, result in enumerate(results_sparse):
            entity = result.get('entity',None)
            if entity:
                doc_id = entity.get('doc_id')
                if doc_id:  # 确保doc_id不为空
                    scores[doc_id] = scores.get(doc_id, 0) + sparse_weight / (k + rank + 1)
                    all_results[doc_id] = result

        for rank, result in enumerate(results_dense):
            entity = result.get('entity',None)
            if entity:
                doc_id = entity.get('doc_id')
                if doc_id:  # 确保doc_id不为空
                    # 关键修正：无论doc_id是否已存在，都要累积分数
                    scores[doc_id] = scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
                    
                    # 只有当doc_id不在all_results中时才添加result
                    # 这样避免覆盖，保留第一次遇到的完整result
                    if doc_id not in all_results:
                        all_results[doc_id] = result
        

        # 按RRF分数排序（从高到低）
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(all_results[doc_id], score) for doc_id, score in sorted_items]
    
    def allocate_document_budgets(self, rrf_fusion_results: list, total_budget: int = 4000,
                                   strategy: str = "balanced") -> list:
        """
        Allocate document budgets based on RRF fusion results.

        :param rrf_fusion_results: The RRF fusion results.
        :param total_budget: The total budget to allocate.
        :return: A list of documents with allocated budgets.
        """
        if len(rrf_fusion_results) == 0:
            return {}

        grouped_results, total_token_results = self._group_results_by_document(rrf_fusion_results)

        num_documents = len(grouped_results)
        if num_documents == 0:
            return {}
        allocated_budgets = {}
        if strategy == "balanced":
            # 平均分配预算
            budget_per_document = total_budget // num_documents
            allocated_budgets = {doc_id: budget_per_document for doc_id in grouped_results.keys()}
        elif strategy == "proportional":
            # 按照RRF分数比例分配预算
            total_score = sum(score for _, score in rrf_fusion_results)
            for doc_id, score in grouped_results.items():
                allocated_budgets[doc_id] = int((score / total_score) * total_budget)
        else:
            pass
        return allocated_budgets, total_token_results, grouped_results

    def _group_results_by_document(self, rrf_fusion_results: list) -> dict:
        """
        Group RRF fusion results by document ID.
        :param rrf_fusion_results: The RRF fusion results.
        :return: A dictionary where keys are document IDs and values are lists of results.
        """
        grouped_results = {}
        total_token_results = {}
        for doc, score in rrf_fusion_results:
            entity = doc.get('entity', {})
            doc_id = entity.get('doc_id')
            doc_meta = entity.get('doc_meta', {})

            if doc_id:
                doc_id_prefix = doc_id.split('_')[0]  # 获取doc_id的前缀
                doc_actual_token_length = doc_meta.get('total_token_count', 0)
                total_token_results[doc_id_prefix] = doc_actual_token_length
                if doc_id_prefix not in grouped_results:
                    grouped_results[doc_id_prefix] = score
                else:
                    grouped_results[doc_id_prefix] += score

        return grouped_results, total_token_results

    def result_merge_to_segments(self, rrf_fusion_results: list, doc_budget: dict, rrf_doc_id_total_score: dict) -> list:
        """
        Merge RRF fusion results into segments.

        :param rrf_fusion_results: The RRF fusion results.
        :return: A list of merged segments.
        """
        doc_prefix_segments = {}
        cache_dict = {}
        for result_data, score in rrf_fusion_results:
            entity = result_data.get('entity', {})
            if not entity:
                continue
            doc_id = entity.get('doc_id')
            if not doc_id:
                continue
            doc_id_prefix = doc_id.split('_')[0]
            cache_item = cache_dict.get(doc_id_prefix, None)
            if cache_item is None:
                cache_item = {}
                cache_item['poviots'] = [entity.get('id')]
            else:
                cache_item['poviots'].append(entity.get('id'))
            cache_dict[doc_id_prefix] = cache_item

        # 初始化cache
        for doc_id_prefix, cache_item in cache_dict.items():
            if 'poviots' not in cache_item:
                cache_item['poviots'] = []
                cache_item['cache'] = None
                cache_dict[doc_id_prefix] = cache_item
            else:
                pivots = cache_item['poviots']
                avg_pivot = int(sum(pivots) / len(pivots))
                cache_engine = MilvusDocCache(
                    engine=self.milvus_db,
                    pivot_id=avg_pivot,
                    expansion_budget=300
                )
                if cache_engine.initialized():
                    cache_item['cache'] = cache_engine
                else:
                    cache_item['cache'] = None

                cache_dict[doc_id_prefix] = cache_item

        
        for result_data, score in rrf_fusion_results:
            entity = result_data.get('entity', {})
            if not entity:
                continue
            doc_id = entity.get('doc_id')
            doc_id_prefix = doc_id.split('_')[0]  # 获取doc_id的前缀
            doc_score = rrf_doc_id_total_score.get(doc_id_prefix)
            doc_budget_value = doc_budget.get(doc_id_prefix)
            doc_meta = entity.get('doc_meta')
            doc_prefix_cache = cache_dict.get(doc_id_prefix)
            doc_prefix_cache = doc_prefix_cache.get('cache', None)

            segments = doc_prefix_segments.get(doc_id_prefix, [])
            doc_id_budget_value = int(score/ doc_score * doc_budget_value)
            segment = self.doc_id_expansion(doc_entity=entity, budget=doc_id_budget_value, cache_dict=doc_prefix_cache)
            segments.append(segment)

            doc_prefix_segments[doc_id_prefix] = segments
        # merge segments
        # doc_prefix_segments = {k: self._merge_segments(v) for k, v in doc_prefix_segments.items()}

        doc_prefix_context = {}
        for doc_id_prefix, segments in doc_prefix_segments.items():
            if not segments:
                continue
            # 重新计算每个段的起始和结束位置
            context_list = []
            for start, end, index in segments:
                L, R = index - 1, index + 1
                doc_id = f"{doc_id_prefix}_{index}"
                doc_prefix_cache = cache_dict.get(doc_id_prefix)
                doc_prefix_cache = doc_prefix_cache.get('cache', None)

                entity = doc_prefix_cache.get_entity(doc_id)
                if entity is None:
                    logger.info(f"Document {doc_id} not found in cache, expanding...")
                    doc_prefix_cache.do_expansion()
                entity = doc_prefix_cache.get_entity(doc_id)
                if entity is None:
                    logger.warning(f"Document {doc_id} still not found in cache after expansion, querying from Milvus.")
                    entity = self.milvus_db.query_by_doc_ids(doc_ids=[doc_id])[0]
                doc_meta = entity.get('doc_meta', {})
                doc_type = doc_meta.get('type', '')
                if doc_type == 'ImageOCR':
                    html_content = '<div type="start">' + '<div type="ImageOCR">' + doc_meta.get('ocr_html_content', '') + '</div>' + '</div>'
                else:
                    html_content = '<div type="start">' + doc_meta.get('html_content', '') + '</div>'
                
                
                brief_context = entity.get('text','')
                total_content = entity.get('text','')
                brief_context_html = html_content
                total_context_html = html_content
                while L >= start or R <= end:
                    if L >= start:
                        l_doc_id = f"{doc_id_prefix}_{L}"
                        l_entity = doc_prefix_cache.get_entity(l_doc_id)
                        if l_entity is None:
                            logger.info(f"Document {l_doc_id} not found in cache, expanding...")
                            doc_prefix_cache.do_expansion()
                        l_entity = doc_prefix_cache.get_entity(l_doc_id)
                        if l_entity is None:
                            logger.warning(f"Document {l_doc_id} still not found in cache after expansion, querying from Milvus.")
                            l_entity = self.milvus_db.query_by_doc_ids(doc_ids=[l_doc_id])[0]
                        l_doc_meta = l_entity.get('doc_meta', {})
                        doc_type = l_doc_meta.get('type', '')
                        if doc_type == 'ImageOCR':
                            html_content = '<div type="ImageOCR">' + l_doc_meta.get('ocr_html_content', '') + '</div>'
                        else:
                            html_content = l_doc_meta.get('html_content', '')
                        
                        total_content = l_entity.get('text', '') + total_content
                        total_context_html = html_content + total_context_html
                        if start - L < 4:
                            brief_context = l_entity.get('text', '') + brief_context
                            brief_context_html = html_content + brief_context_html
                        L -= 1
                    if R <= end:
                        r_doc_id = f"{doc_id_prefix}_{R}"
                        r_entity = doc_prefix_cache.get_entity(r_doc_id)
                        if r_entity is None:
                            logger.info(f"Document {r_doc_id} not found in cache, expanding...")
                            doc_prefix_cache.do_expansion()
                        r_entity = doc_prefix_cache.get_entity(r_doc_id)
                        if r_entity is None:
                            logger.warning(f"Document {r_doc_id} still not found in cache after expansion, querying from Milvus.")
                            r_entity = self.milvus_db.query_by_doc_ids(doc_ids=[r_doc_id])[0]
                        r_doc_meta = r_entity.get('doc_meta', {})
                        doc_type = r_doc_meta.get('type', '')
                        if doc_type == 'ImageOCR':
                            html_content = '<div type="ImageOCR">' + r_doc_meta.get('ocr_html_content', '') + '</div>'
                        else:
                            html_content = r_doc_meta.get('html_content', '')
                        
                        total_content += r_entity.get('text', '')
                        total_context_html += html_content
                        if R - start < 4:
                            brief_context += r_entity.get('text', '')
                            brief_context_html += html_content
                        R += 1


                context_list.append({
                    "brief_context": brief_context,
                    "total_context": total_content,
                    "brief_context_html": brief_context_html,
                    "total_context_html": total_context_html,
                    "file_name": doc_meta.get('file_name', ''),
                })

            doc_prefix_context[doc_id_prefix] = context_list

        return doc_prefix_context
    
    def _merge_segments(self, segments: list) -> list:
        """
        Merge overlapping segments.

        :param segments: A list of segments to merge.
        :return: A list of merged segments.
        """
        if not segments:
            return []

        segments.sort(key=lambda x: x[0])
        merged_segments = []
        current_start, current_end, doc_id_index = segments[0]
        for start, end, index in segments[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged_segments.append((current_start, current_end, index))
                current_start, current_end = start, end
        merged_segments.append((current_start, current_end, (current_start + current_end) // 2))
        return merged_segments

    def doc_id_expansion(self, doc_entity: dict, budget:int, cache_dict:MilvusDocCache) -> tuple[tuple, dict]:
        """
        Expand the document ID to include related documents based on the budget.

        :param doc_id: The document ID to expand.
        :param budget: The budget for expansion.
        :param cache_dict: A dictionary to cache results.
        :return: A tuple containing expanded document data and updated cache dictionary.
        """
        doc_id = doc_entity.get('doc_id')
        if not doc_id:
            return ()
        doc_id_prefix = doc_id.split('_')[0]
        doc_id_index = int(doc_id.split('_')[1])
        left = right = doc_id_index
        doc_meta = doc_entity.get('doc_meta', {})
        total_part_count = doc_meta.get('total_store_len', 0)
        L, R = doc_id_index - 1, doc_id_index + 1

        consumed_budget = doc_meta.get('token_length', 0)
        while consumed_budget < budget and (L >= 0 or R < total_part_count):
            if L >= 0:
                left = L
                L -= 1
                doc_id = f"{doc_id_prefix}_{left}"
                l_entity = cache_dict.get_entity(doc_id)
                if l_entity is None:
                    logger.info(f"Document {doc_id} not found in cache, expanding...")
                    cache_dict.do_expansion()
                
                l_entity = cache_dict.get_entity(doc_id)
                if l_entity is None:
                    logger.warning(f"Document {doc_id} still not found in cache after expansion, querying from Milvus.")
                    l_entity = self.milvus_db.query_by_doc_ids(doc_ids=[doc_id])[0]
                doc_meta = l_entity.get('doc_meta', {})
                entity_len = doc_meta.get('token_length', 0)
                if consumed_budget + entity_len >= budget:
                    break
                consumed_budget += entity_len
            if consumed_budget < budget and R < total_part_count:
                right = R
                R += 1
                doc_id = f"{doc_id_prefix}_{right}"
                r_entity = cache_dict.get_entity(doc_id)
                if r_entity is None:
                    logger.info(f"Document {doc_id} not found in cache, expanding...")
                    cache_dict.do_expansion()
                r_entity = cache_dict.get_entity(doc_id)
                if r_entity is None:
                    logger.warning(f"Document {doc_id} still not found in cache after expansion, querying from Milvus.")
                    r_entity = self.milvus_db.query_by_doc_ids(doc_ids=[doc_id])[0]
                doc_meta = r_entity.get('doc_meta', {})
                entity_len = doc_meta.get('token_length', 0)
                if consumed_budget + entity_len >= budget:
                    break
                consumed_budget += entity_len

        return (left, right, doc_id_index)

def fix_table_structure_advanced(html_content):
    """
    高级版本：更精确地修复表格结构
    """
    if not html_content or '<tr>' not in html_content:
        return html_content
    
    # 分割HTML内容，处理每个部分
    parts = []
    current_pos = 0
    
    # 找到所有<tr>标签的位置
    tr_pattern = r'<tr(?:[^>]*)?>.*?</tr>'
    tr_matches = list(re.finditer(tr_pattern, html_content, re.DOTALL | re.IGNORECASE))
    
    if not tr_matches:
        return html_content
    
    i = 0
    while i < len(tr_matches):
        # 添加当前位置之前的内容
        if tr_matches[i].start() > current_pos:
            parts.append(html_content[current_pos:tr_matches[i].start()])
        
        # 收集连续的tr标签
        tr_group = []
        j = i
        while j < len(tr_matches):
            tr_group.append(tr_matches[j].group())
            j += 1
            
            # 检查下一个tr是否紧邻（允许空白字符）
            if j < len(tr_matches):
                between_content = html_content[tr_matches[j-1].end():tr_matches[j].start()].strip()
                if between_content and not re.match(r'^\s*$', between_content):
                    break
            else:
                break
        
        # 为tr组添加table包装
        tr_content = ''.join(tr_group)
        parts.append(f'<table>{tr_content}</table>')
        
        # 更新位置
        current_pos = tr_matches[j-1].end()
        i = j
    
    # 添加剩余内容
    if current_pos < len(html_content):
        parts.append(html_content[current_pos:])
    
    return ''.join(parts)

if __name__ == "__main__":
    work_dir = '/Users/baoshiqiu/code/file_process_for_llm/search'
    with open(f'{work_dir}/config.json', 'r') as f:
        config = json.load(f)
    milvus_uri = config.get('milvus_uri', 'localhost')
    milvus_token = config.get('milvus_token', 'your_milvus_token')
    collection_names = config.get('collection_names', ['your_collection_name'])
    embedding_model = config.get('embedding_model', 'bge-m3')
    embedding_base_url = config.get('embedding_base_url', 'http://localhost:6001/v1')
    embedding_model_token = config.get('embedding_model_token', 'your_embedding_model_token')
    rerank_base_url = config.get('rerank_base_url')
    rerank_model_token = config.get('rerank_model_token')
    rerank_model = config.get('rerank_model')

    milvus_search = MilvusSearch(uri=milvus_uri, token=milvus_token, collection_name=collection_names[0])

    query_text = "MOJITO"
    doc_ids = set()
    doc_names = set()
    results_sparse = milvus_search.search_sparse(query=query_text, limit=5)


    results_dense = []
    embedding_success, embedding_results, message = get_embedding(
        text=[query_text],
        model=embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_model_token)
    if embedding_success:
        results_dense = milvus_search.search_dense(embedding=embedding_results[0], limit=5)

    rrf_result = milvus_search.rrf_fusion(results_sparse=results_sparse, results_dense=results_dense)
    # print((rrf_result))

    allocated_budgets, total_token_results, doc_total_score = milvus_search.allocate_document_budgets(rrf_fusion_results=rrf_result, total_budget=6000, strategy="proportional")
    for doc_id, budget in allocated_budgets.items():
        # print(f"Document ID: {doc_id}, Allocated Budget: {budget}, Total Token Length: {total_token_results.get(doc_id, 0)}, Document Total Score: {doc_total_score.get(doc_id, 0)}")
        if doc_id in total_token_results:
            if total_token_results[doc_id] < budget:
                # print(f"Warning: Allocated budget {budget} exceeds total token length {total_token_results[doc_id]} for document {doc_id}. will be adjusted.")
                allocated_budgets[doc_id] = total_token_results[doc_id] * 1.25
    # print(f"Allocated Budgets: {allocated_budgets}")
    search_re = milvus_search.result_merge_to_segments(rrf_fusion_results=rrf_result, doc_budget=allocated_budgets, rrf_doc_id_total_score=doc_total_score)
    # print(search_re)
    # rerank
    document_list = []
    doc_rerank_index = {}
    for doc_id ,result in search_re.items():
        for segment in result:
            brief_context = segment.get('brief_context', '')
            document_list.append(brief_context)
            rerank_id = f"{len(document_list)-1}"
            doc_rerank_index[rerank_id] = segment
            

    
    flag, results_rerank, msg = rerank_documents(
        url=rerank_base_url,
        query=query_text,
        documents=document_list,
        model=rerank_model,
        token=rerank_model_token
    )
    for idx, result in enumerate(results_rerank):
        rerank_id = f"{idx}"
        segment = doc_rerank_index[rerank_id]
        brief_content = segment.get('total_context_html', '')
        print(f'{brief_content}')
        print('------------------------------')


