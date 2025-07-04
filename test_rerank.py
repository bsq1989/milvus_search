from utils import rerank_documents
import json
# 使用示例
if __name__ == "__main__":
    with open('./config.json', 'r') as f:
        config = json.load(f)

    rerank_base_url = config.get('rerank_base_url')
    rerank_model_token = config.get('rerank_model_token')
    rerank_model = config.get('rerank_model')
    documents = [
            "The capital of Brazil is Brasilia.",
            "The capital of France is Paris.",
            "Horses and cows are both animals"
        ]
    # 本地API示例
    success, results, error = rerank_documents(
        url=rerank_base_url,
        query="What is the capital of France?",
        documents=documents,
        model=rerank_model,
        token=rerank_model_token
    )
    
    if success:
        print("重排序成功:")
        for result in results:
            print(result)
            print(f"Index: {result['index']}, Score: {result['relevance_score']:.4f}")
            print(f"Text: {documents[result['index']]}")
            print("-" * 50)
    else:
        print(f"重排序失败: {error}")