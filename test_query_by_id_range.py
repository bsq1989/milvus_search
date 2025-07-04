from milvus_client import MilvusVectorDB
import json


if __name__ == "__main__":
    work_dir = '/Users/baoshiqiu/code/file_process_for_llm/search'
    with open(f'{work_dir}/config.json', 'r') as f:
        config = json.load(f)
    milvus_uri = config.get('milvus_uri', 'localhost')
    milvus_token = config.get('milvus_token', 'your_milvus_token')


    milvus_db = MilvusVectorDB(
        uri=milvus_uri,
        token=milvus_token)
    
    id_range = (457861707833668244,457861707833668252)
    query_results = milvus_db.query_by_id_range(
        start_id=id_range[0],
        end_id=id_range[1],
    )
    print(f'Query results for ID range {id_range}:')
    for result in query_results:
        print(result)
        print('-' * 50)