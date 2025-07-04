from pymilvus import MilvusClient
from typing import List, Dict, Any, Optional, Union
import json


class MilvusVectorDB:
    def __init__(self, uri: str = "http://milvus-standalone:19530", token = None, collection_name: str = "llm_default"):
        """
        初始化Milvus向量数据库客户端
        
        Args:
            uri: Milvus服务地址
            collection_name: 集合名称
        """
        self.client = MilvusClient(uri=uri,token=token)
        self.collection_name = collection_name
    
    def insert_data(self, 
                   data: List[Dict[str, Any]], 
                   batch_size: int = 1000,
                   show_progress: bool = True) -> Dict[str, Any]:
        """
        批量插入数据到Milvus集合
        
        注意：由于主键id设置为auto_id=True，插入时不需要提供id字段，系统会自动生成
        
        Args:
            data: 要插入的数据列表，每个元素包含doc_id, text, text_dense, image_dense, doc_meta等字段
                  id字段会自动生成，text_sparse字段会由BM25函数自动生成
            batch_size: 批次大小，默认1000条
            show_progress: 是否显示进度信息
        
        Returns:
            插入结果信息，包含自动生成的ID列表
        """
        if not data:
            return {"success": False, "error": "数据列表为空"}
        
        try:
            # 验证数据格式
            validated_data = self._validate_insert_data(data)
            total_count = len(validated_data)
            inserted_count = 0
            all_inserted_ids = []
            
            if show_progress:
                print(f"开始批量插入 {total_count} 条数据，批次大小: {batch_size}")
            
            # 分批插入
            for i in range(0, total_count, batch_size):
                batch_data = validated_data[i:i + batch_size]
                
                # 执行批次插入
                result = self.client.insert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
                
                batch_inserted = result.get("insert_count", len(batch_data))
                inserted_count += batch_inserted
                
                # 收集自动生成的ID
                if "ids" in result:
                    all_inserted_ids.extend(result["ids"])
                
                if show_progress:
                    print(f"已插入 {inserted_count}/{total_count} 条数据 "
                          f"(当前批次: {batch_inserted} 条)")
            
            return {
                "success": True,
                "total_count": total_count,
                "inserted_count": inserted_count,
                "inserted_ids": all_inserted_ids,
                "batch_size": batch_size,
                "message": f"批量插入成功，共插入 {inserted_count} 条数据"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "inserted_count": inserted_count if 'inserted_count' in locals() else 0,
                "message": "批量插入失败"
            }
    
    def upsert_data(self, 
                   data: List[Dict[str, Any]], 
                   batch_size: int = 1000,
                   show_progress: bool = True) -> Dict[str, Any]:
        """
        批量更新插入数据
        
        注意：由于使用auto_id，upsert操作需要明确指定要更新的记录ID
        建议先查询获取ID，然后进行更新操作
        
        Args:
            data: 要更新插入的数据列表，必须包含id字段用于更新现有记录
            batch_size: 批次大小
            show_progress: 是否显示进度信息
        
        Returns:
            更新插入结果信息
        """
        if not data:
            return {"success": False, "error": "数据列表为空"}
        
        try:
            validated_data = self._validate_upsert_data(data)
            total_count = len(validated_data)
            upserted_count = 0
            
            if show_progress:
                print(f"开始批量更新插入 {total_count} 条数据，批次大小: {batch_size}")
            
            # 分批更新插入
            for i in range(0, total_count, batch_size):
                batch_data = validated_data[i:i + batch_size]
                
                result = self.client.upsert(
                    collection_name=self.collection_name,
                    data=batch_data
                )
                
                batch_upserted = result.get("upsert_count", len(batch_data))
                upserted_count += batch_upserted
                
                if show_progress:
                    print(f"已更新插入 {upserted_count}/{total_count} 条数据 "
                          f"(当前批次: {batch_upserted} 条)")
            
            return {
                "success": True,
                "total_count": total_count,
                "upserted_count": upserted_count,
                "batch_size": batch_size,
                "message": f"批量更新插入成功，共处理 {upserted_count} 条数据"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "upserted_count": upserted_count if 'upserted_count' in locals() else 0,
                "message": "批量更新插入失败"
            }
    
    def bulk_insert_from_files(self, 
                              file_paths: List[str],
                              file_type: str = "json") -> Dict[str, Any]:
        """
        从文件批量导入数据（适用于大规模数据导入）
        
        Args:
            file_paths: 数据文件路径列表
            file_type: 文件类型，支持 "json", "parquet", "csv"
        
        Returns:
            导入结果信息
        """
        try:
            # 使用Milvus的bulk_insert功能进行大规模数据导入
            job_id = self.client.bulk_insert(
                collection_name=self.collection_name,
                files=file_paths,
                file_type=file_type
            )
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"批量导入作业已提交，作业ID: {job_id}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "批量导入失败"
            }
    
    def search_by_text_dense(self, 
                           query_vectors: List[List[float]], 
                           limit: int = 10,
                           output_fields: Optional[List[str]] = None,
                           filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        使用文本密集向量进行相似性搜索
        
        Args:
            query_vectors: 查询向量列表（1024维）
            limit: 返回结果数量限制
            output_fields: 要返回的字段列表
            filter_expr: 过滤表达式
        
        Returns:
            搜索结果列表
        """
        if output_fields is None:
            output_fields = ["id", "text", "doc_id", "doc_meta"]

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=query_vectors,
                anns_field="text_dense",
                search_params={"metric_type": "IP"},
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr
            )
            
            return self._format_search_results(results)
        except Exception as e:
            return [{"error": str(e)}]
    
    def search_by_image_dense(self, 
                            query_vectors: List[List[float]], 
                            limit: int = 10,
                            output_fields: Optional[List[str]] = None,
                            filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        使用图像密集向量进行相似性搜索
        
        Args:
            query_vectors: 查询向量列表（512维）
            limit: 返回结果数量限制
            output_fields: 要返回的字段列表
            filter_expr: 过滤表达式
        
        Returns:
            搜索结果列表
        """
        if output_fields is None:
            output_fields = ["id", "text", "doc_id", "doc_meta"]

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=query_vectors,
                anns_field="image_dense",
                search_params={"metric_type": "IP"},
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr
            )
            
            return self._format_search_results(results)
        except Exception as e:
            return [{"error": str(e)}]
    
    def search_by_text_sparse(self, 
                            query_text: List[str], 
                            limit: int = 10,
                            output_fields: Optional[List[str]] = None,
                            filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        使用文本稀疏向量（BM25）进行搜索
        
        Args:
            query_text: 查询文本列表
            limit: 返回结果数量限制
            output_fields: 要返回的字段列表
            filter_expr: 过滤表达式
        
        Returns:
            搜索结果列表
        """
        if output_fields is None:
            output_fields = ["id", "text","doc_id", "doc_meta"]
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=query_text,
                anns_field="text_sparse",
                search_params={"metric_type": "BM25"},
                limit=limit,
                output_fields=output_fields,
                filter=filter_expr
            )
            
            return self._format_search_results(results)
        except Exception as e:
            return [{"error": str(e)}]
    
    
    def query_by_doc_ids(self, doc_ids: List[str], output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据文档ID查询数据
        
        Args:
            doc_ids: 文档ID列表
            output_fields: 要返回的字段列表
        
        Returns:
            查询结果列表
        """
        if output_fields is None:
            output_fields = ["id", "doc_id", "text", "doc_meta"]
        
        try:
            # 构建过滤条件
            doc_ids_str = ', '.join([f'"{doc_id}"' for doc_id in doc_ids])
            filter_expr = f"doc_id in [{doc_ids_str}]"
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def query_by_doc_id_prefix(self, doc_id_prefix: str, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据文档ID前缀查询数据
        
        Args:
            doc_id_prefix: 文档ID前缀
            output_fields: 要返回的字段列表
        
        Returns:
            查询结果列表
        """
        if output_fields is None:
            output_fields = ["id", "doc_id", "text", "doc_meta"]
        
        try:
            # 使用like语法进行前缀匹配
            filter_expr = f'doc_id like "{doc_id_prefix}%"'
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def query_by_doc_id_prefixes(self, doc_id_prefixes: List[str], output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据多个文档ID前缀查询数据
        
        Args:
            doc_id_prefixes: 文档ID前缀列表
            output_fields: 要返回的字段列表
        
        Returns:
            查询结果列表
        """
        if output_fields is None:
            output_fields = ["id", "doc_id", "text", "doc_meta"]
        
        try:
            # 构建多个前缀的OR条件
            like_conditions = [f'doc_id like "{prefix}%"' for prefix in doc_id_prefixes]
            filter_expr = " or ".join(like_conditions)
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def delete_by_doc_ids(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        根据文档ID删除数据
        
        Args:
            doc_ids: 要删除的文档ID列表
        
        Returns:
            删除结果信息
        """
        try:
            doc_ids_str = ', '.join([f'"{doc_id}"' for doc_id in doc_ids])
            filter_expr = f"doc_id in [{doc_ids_str}]"
            
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
            
            return {
                "success": True,
                "delete_count": result.get("delete_count", 0),
                "message": "数据删除成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "数据删除失败"
            }
    
    def delete_by_doc_id_prefix(self, doc_id_prefix: str) -> Dict[str, Any]:
        """
        根据文档ID前缀删除数据
        
        Args:
            doc_id_prefix: 要删除的文档ID前缀
        
        Returns:
            删除结果信息
        """
        try:
            # 使用like语法进行前缀匹配删除
            filter_expr = f'doc_id like "{doc_id_prefix}%"'
            
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
            
            return {
                "success": True,
                "delete_count": result.get("delete_count", 0),
                "doc_id_prefix": doc_id_prefix,
                "message": f"根据前缀 '{doc_id_prefix}' 删除数据成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"根据前缀 '{doc_id_prefix}' 删除数据失败"
            }
    
    def delete_by_doc_id_prefixes(self, doc_id_prefixes: List[str]) -> Dict[str, Any]:
        """
        根据多个文档ID前缀删除数据
        
        Args:
            doc_id_prefixes: 要删除的文档ID前缀列表
        
        Returns:
            删除结果信息
        """
        try:
            # 构建多个前缀的OR条件
            like_conditions = [f'doc_id like "{prefix}%"' for prefix in doc_id_prefixes]
            filter_expr = " or ".join(like_conditions)
            
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
            
            return {
                "success": True,
                "delete_count": result.get("delete_count", 0),
                "doc_id_prefixes": doc_id_prefixes,
                "message": f"根据前缀列表删除数据成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "根据前缀列表删除数据失败"
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            集合统计信息
        """
        try:
            stats = self.client.describe_collection(collection_name=self.collection_name)
            return stats
        except Exception as e:
            return {"error": str(e)}
    
    def get_batch_insert_recommendations(self, data_size: int, vector_dim: int) -> Dict[str, Any]:
        """
        根据数据规模获取批量插入建议
        
        Args:
            data_size: 数据总量
            vector_dim: 向量维度
        
        Returns:
            批量插入建议
        """
        # 估算内存使用（简化计算）
        estimated_memory_per_record = (vector_dim * 4 + 1000) / 1024 / 1024  # MB
        
        if data_size < 1000:
            batch_size = min(data_size, 100)
            suggestion = "小数据集，建议较小批次"
        elif data_size < 100000:
            batch_size = 1000
            suggestion = "中等数据集，建议标准批次"
        else:
            batch_size = 5000
            suggestion = "大数据集，建议较大批次以提高效率"
        
        return {
            "recommended_batch_size": batch_size,
            "estimated_memory_per_batch": batch_size * estimated_memory_per_record,
            "estimated_total_time": (data_size / batch_size) * 0.5,  # 假设每批次0.5秒
            "suggestion": suggestion
        }
    
    def _validate_insert_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证插入数据格式（用于auto_id模式）
        
        Args:
            data: 原始数据
        
        Returns:
            验证后的数据
        """
        validated_data = []
        
        for item in data:
            validated_item = {}
            
            # 注意：不需要id字段，因为设置了auto_id=True
            # 如果数据中包含id字段，会被忽略
            
            # 必需字段验证
            if "doc_id" not in item:
                raise ValueError("缺少必需字段: doc_id")
            validated_item["doc_id"] = str(item["doc_id"])[:256]  # 限制长度
            
            if "text" not in item:
                raise ValueError("缺少必需字段: text")
            validated_item["text"] = str(item["text"])[:1000]  # 限制长度
            
            if "text_dense" not in item:
                raise ValueError("缺少必需字段: text_dense")
            if len(item["text_dense"]) != 1024:
                raise ValueError("text_dense向量维度必须为1024")
            validated_item["text_dense"] = item["text_dense"]
            
            if "image_dense" not in item:
                raise ValueError("缺少必需字段: image_dense")
            if len(item["image_dense"]) != 512:
                raise ValueError("image_dense向量维度必须为512")
            validated_item["image_dense"] = item["image_dense"]
            
            # 可选字段
            if "doc_meta" in item and item["doc_meta"] is not None:
                if isinstance(item["doc_meta"], dict):
                    validated_item["doc_meta"] = item["doc_meta"]
                else:
                    validated_item["doc_meta"] = json.loads(item["doc_meta"])
            
            validated_data.append(validated_item)
        
        return validated_data
    
    def _validate_upsert_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证更新插入数据格式（需要包含ID）
        
        Args:
            data: 原始数据
        
        Returns:
            验证后的数据
        """
        validated_data = []
        
        for item in data:
            validated_item = {}
            
            # upsert操作需要明确的ID
            if "id" not in item:
                raise ValueError("upsert操作缺少必需字段: id")
            validated_item["id"] = int(item["id"])
            
            # 其他字段验证与insert_data相同
            if "doc_id" not in item:
                raise ValueError("缺少必需字段: doc_id")
            validated_item["doc_id"] = str(item["doc_id"])[:256]
            
            if "text" not in item:
                raise ValueError("缺少必需字段: text")
            validated_item["text"] = str(item["text"])[:1000]
            
            if "text_dense" not in item:
                raise ValueError("缺少必需字段: text_dense")
            if len(item["text_dense"]) != 1024:
                raise ValueError("text_dense向量维度必须为1024")
            validated_item["text_dense"] = item["text_dense"]
            
            if "image_dense" not in item:
                raise ValueError("缺少必需字段: image_dense")
            if len(item["image_dense"]) != 512:
                raise ValueError("image_dense向量维度必须为512")
            validated_item["image_dense"] = item["image_dense"]
            
            if "doc_meta" in item and item["doc_meta"] is not None:
                if isinstance(item["doc_meta"], dict):
                    validated_item["doc_meta"] = item["doc_meta"]
                else:
                    validated_item["doc_meta"] = json.loads(item["doc_meta"])
            
            validated_data.append(validated_item)
        
        return validated_data
    
    def _format_search_results(self, results) -> List[Dict[str, Any]]:
        """
        格式化搜索结果
        
        Args:
            results: 原始搜索结果
        
        Returns:
            格式化后的结果
        """
        formatted_results = []
        
        for result in results:
            for hit in result:
                formatted_hit = {
                    "id": hit.get("id"),
                    "distance": hit.get("distance"),
                    "score": hit.get("score"),
                    "entity": hit.get("entity", {})
                }
                formatted_results.append(formatted_hit)
        
        return formatted_results
    
    def query_by_id_range(self, 
                         start_id: int, 
                         end_id: int, 
                         output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据ID范围查询数据
        
        Args:
            start_id: 起始ID（包含）
            end_id: 结束ID（包含）
            output_fields: 要返回的字段列表
        
        Returns:
            查询结果列表
        """
        if output_fields is None:
            output_fields = ["id", "doc_id", "text", "doc_meta"]
        
        try:
            # 构建ID范围过滤条件
            filter_expr = f"id >= {start_id} and id <= {end_id}"
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            return [{"error": str(e)}]

    def query_by_ids(self, 
                    ids: List[int], 
                    output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据ID列表查询数据
        
        Args:
            ids: ID列表
            output_fields: 要返回的字段列表
        
        Returns:
            查询结果列表
        """
        if output_fields is None:
            output_fields = ["id", "doc_id", "text", "doc_meta"]
        
        try:
            # 构建ID列表过滤条件
            ids_str = ', '.join([str(id_val) for id_val in ids])
            filter_expr = f"id in [{ids_str}]"
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields
            )
            
            return results
        except Exception as e:
            return [{"error": str(e)}]

