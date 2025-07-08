#!/usr/bin/env python3
"""
MilvusSearch 实例池简单使用示例
"""

import json
import time
import threading
from milvus_search import get_global_pool, shutdown_global_pool

def simple_pool_demo():
    """简单的实例池使用演示"""
    print("🚀 MilvusSearch 实例池演示")
    print("=" * 50)
    
    # 模拟配置（请根据实际情况修改）
    milvus_uri = "your_milvus_uri"
    milvus_token = "your_milvus_token" 
    collection_name = "your_collection"
    
    # 获取全局实例池
    pool = get_global_pool(max_pool_size=3, max_idle_time=60)
    
    print("📊 初始池状态:")
    status = pool.get_pool_status()
    print(f"   总实例数: {status['total_instances']}")
    print(f"   使用中: {status['in_use_instances']}")
    print(f"   空闲: {status['idle_instances']}")
    
    print("\n🔄 测试1: 基本使用模式")
    try:
        # 方式1: 手动管理
        print("   获取实例...")
        instance = pool.get_instance(milvus_uri, milvus_token, collection_name)
        print(f"   ✅ 成功获取实例: {type(instance).__name__}")
        
        # 模拟使用
        time.sleep(0.1)
        
        # 释放实例
        pool.release_instance(milvus_uri, milvus_token, collection_name)
        print("   ✅ 实例已释放")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    print("\n🎯 测试2: 上下文管理器（推荐）")
    try:
        with pool.get_instance_context(milvus_uri, milvus_token, collection_name) as search_instance:
            print(f"   ✅ 在上下文中获取实例: {type(search_instance).__name__}")
            # 这里可以执行实际的搜索操作
            # results = search_instance.search_sparse("查询文本", limit=10)
            time.sleep(0.1)
        print("   ✅ 上下文结束，实例自动释放")
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    print("\n🏃‍♂️ 测试3: 并发使用")
    def worker(worker_id):
        try:
            with pool.get_instance_context(milvus_uri, milvus_token, collection_name) as search_instance:
                print(f"   Worker-{worker_id}: 获取到实例")
                time.sleep(0.2)  # 模拟搜索操作
                print(f"   Worker-{worker_id}: 工作完成")
        except Exception as e:
            print(f"   Worker-{worker_id}: 错误 {e}")
    
    # 启动3个并发工作者
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待完成
    for t in threads:
        t.join()
    
    print("   ✅ 所有并发工作者完成")
    
    print("\n📊 最终池状态:")
    status = pool.get_pool_status()
    print(f"   总实例数: {status['total_instances']}")
    print(f"   使用中: {status['in_use_instances']}")
    print(f"   空闲: {status['idle_instances']}")
    
    print("\n🔍 实例详情:")
    for key, info in status['instances'].items():
        print(f"   {key}: 使用中={info['in_use']}, 空闲时间={info['idle_time']:.1f}秒")
    
    print("\n🧹 清理池...")
    shutdown_global_pool()
    print("✅ 演示完成!")

def real_search_demo():
    """真实搜索演示（需要有效的配置）"""
    print("\n🔍 真实搜索演示")
    print("=" * 30)
    
    try:
        # 尝试加载配置
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        milvus_uri = config.get('milvus_uri')
        milvus_token = config.get('milvus_token')
        collection_names = config.get('collection_names', [])
        
        if not all([milvus_uri, milvus_token, collection_names]):
            print("❌ 配置不完整，跳过真实搜索演示")
            return
        
        pool = get_global_pool()
        query_text = "测试查询"
        
        print(f"📝 查询: {query_text}")
        
        with pool.get_instance_context(milvus_uri, milvus_token, collection_names[0]) as search_instance:
            # 执行稀疏搜索
            results_sparse = search_instance.search_sparse(query=query_text, limit=5)
            print(f"✅ 稀疏搜索结果数量: {len(results_sparse)}")
            
            # 如果有稠密搜索的embedding
            try:
                from utils import get_embedding
                embedding_success, embedding_results, message = get_embedding(
                    text=[query_text],
                    model=config.get('embedding_model', 'bge-m3'),
                    base_url=config.get('embedding_base_url', 'http://localhost:6001/v1'),
                    api_key=config.get('embedding_model_token', '')
                )
                
                if embedding_success:
                    results_dense = search_instance.search_dense(embedding=embedding_results[0], limit=5)
                    print(f"✅ 稠密搜索结果数量: {len(results_dense)}")
                else:
                    print(f"⚠️  embedding生成失败: {message}")
                    
            except ImportError:
                print("⚠️  utils模块不可用，跳过稠密搜索")
        
        shutdown_global_pool()
        
    except FileNotFoundError:
        print("❌ 找不到config.json文件，跳过真实搜索演示")
    except Exception as e:
        print(f"❌ 真实搜索演示错误: {e}")

if __name__ == "__main__":
    # 运行简单演示
    simple_pool_demo()
    
    # 如果配置可用，运行真实搜索演示
    real_search_demo()
