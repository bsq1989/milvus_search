from openai import OpenAI


import requests
import json
from typing import List, Dict, Tuple, Optional

def get_embedding(text, model="bge-m3",base_url="http://localhost:6001/v1",api_key=None):
    """
    Get embeddings for the provided text using OpenAI's API
    
    Args:
        text (str): The input text to get embeddings for
        model (str): The embedding model to use
        base_url (str): The base URL for the API
        api_key (str): The API key for authentication

    Returns:
        list: The embedding vector
    """
    # Initialize the client
    # Replace with your API key or set it as an environment variable
    # Get API key and base URL from environment variables or use defaults
    
   
    # Initialize the client with custom base URL
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
    
        # Get the embedding from OpenAI
        response = client.embeddings.create(
            input=text,
            model=model
        )
        
        # 判断response 有效性
        if not response or not hasattr(response, 'data') or not response.data:
            return False, [], 'no embedding data found'
        # Extract the embedding from the response
        embedding_results = [
            item.embedding for item in response.data
        ]
        return True, embedding_results, 'get embedding success'
    except Exception as e:
        return False ,[], f'get embedding failed: {str(e)}'
    



def rerank_documents(
    url: str,
    query: str,
    documents: List[str],
    model: str,
    token: Optional[str] = None
) -> Tuple[bool, List[Dict], str]:
    """
    使用rerank API对文档进行重排序
    
    Args:
        url: API地址，如 'http://127.0.0.1:8000/v1/rerank' 或 'https://api.siliconflow.cn/v1/rerank'
        query: 查询文本
        documents: 文档列表
        model: 模型名称，如 'BAAI/bge-reranker-base'
        token: API token（可选，某些API需要）
    
    Returns:
        Tuple[bool, List[Dict], str]: (成功标志, results列表, 错误信息)
        成功时返回 (True, results, "")
        失败时返回 (False, [], error_message)
    """
    
    # 构建请求头
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # 如果提供了token，添加认证头
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    # 构建请求数据
    payload = {
        "model": model,
        "query": query,
        "documents": documents
    }
    
    try:
        # 发送POST请求
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30  # 30秒超时
        )
        
        # 检查响应状态码
        if response.status_code == 200:
            result_data = response.json()
            # 提取results字段
            results = result_data.get('results', [])
            return True, results, ""
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            return False, [], error_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"请求异常: {str(e)}"
        return False, [], error_msg
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析错误: {str(e)}"
        return False, [], error_msg
    except Exception as e:
        error_msg = f"未知错误: {str(e)}"
        return False, [], error_msg

