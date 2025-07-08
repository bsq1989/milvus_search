#!/usr/bin/env python3
"""
MilvusSearch 实例池使用示例和测试
"""

from milvus_search import get_global_pool, shutdown_global_pool
import json
import threading
import time

def load_config():
    """加载配置文件"""
    work_dir = '/Users/baoshiqiu/code/file_process_for_llm/search'
    with open(f'{work_dir}/config.json', 'r') as f:
        config = json.load(f)
    return config

def test_basic_pool_usage():
    """测试基本的池使用功能"""
    print("=== 测试基本池使用功能 ===")
    
    config = load_config()
    milvus_uri = config.get('milvus_uri', 'localhost')
    milvus_token = config.get('milvus_token', 'your_milvus_token')
    collection_names = config.get('collection_names', ['your_collection_name'])
    
    # 获取实例池
    pool = get_global_pool(max_pool_size=3, max_idle_time=60)
    
    try:
        # 测试获取实例
        print("1. 获取第一个实例...")
        instance1 = pool.get_instance(milvus_uri, milvus_token, collection_names[0])
        print(f"   获取实例成功: {type(instance1).__name__}")
        
        # 查看池状态
        status = pool.get_pool_status()
        print(f"   池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")
        
        # 测试获取相同配置的实例（应该创建新实例，因为第一个还在使用中）
        print("2. 获取第二个实例（相同配置）...")
        instance2 = pool.get_instance(milvus_uri, milvus_token, collection_names[0])
        print(f"   获取实例成功: {type(instance2).__name__}")
        
        status = pool.get_pool_status()
        print(f"   池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")
        
        # 释放第一个实例
        print("3. 释放第一个实例...")
        pool.release_instance(milvus_uri, milvus_token, collection_names[0])
        
        status = pool.get_pool_status()
        print(f"   池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")
        
        # 获取第三个实例（应该复用已释放的实例）
        print("4. 获取第三个实例（应该复用）...")
        instance3 = pool.get_instance(milvus_uri, milvus_token, collection_names[0])
        print(f"   获取实例成功: {type(instance3).__name__}")
        
        status = pool.get_pool_status()
        print(f"   池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")
        
        # 清理
        pool.release_instance(milvus_uri, milvus_token, collection_names[0])  # 释放instance2
        pool.release_instance(milvus_uri, milvus_token, collection_names[0])  # 释放instance3
        
        print("5. 测试完成")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

def test_context_manager():
    """测试上下文管理器"""
    print("\n=== 测试上下文管理器 ===")
    
    config = load_config()
    milvus_uri = config.get('milvus_uri', 'localhost')
    milvus_token = config.get('milvus_token', 'your_milvus_token')
    collection_names = config.get('collection_names', ['your_collection_name'])
    
    pool = get_global_pool()
    
    try:
        print("1. 使用上下文管理器获取实例...")
        with pool.get_instance_context(milvus_uri, milvus_token, collection_names[0]) as search_instance:
            print(f"   在上下文中获取到实例: {type(search_instance).__name__}")
            
            # 查看池状态
            status = pool.get_pool_status()
            print(f"   池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")
            
            # 模拟一些工作
            time.sleep(0.1)
            
        print("2. 退出上下文管理器后...")
        status = pool.get_pool_status()
        print(f"   池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

def test_concurrent_access():
    """测试并发访问"""
    print("\n=== 测试并发访问 ===")
    
    config = load_config()
    milvus_uri = config.get('milvus_uri', 'localhost')
    milvus_token = config.get('milvus_token', 'your_milvus_token')
    collection_names = config.get('collection_names', ['your_collection_name'])
    
    pool = get_global_pool(max_pool_size=3)
    results = []
    
    def worker(worker_id):
        """工作线程函数"""
        try:
            print(f"   Worker {worker_id} 开始")
            with pool.get_instance_context(milvus_uri, milvus_token, collection_names[0]) as search_instance:
                print(f"   Worker {worker_id} 获取到实例")
                
                # 模拟一些工作
                time.sleep(0.2)
                
                # 这里可以执行实际的搜索操作
                # results = search_instance.search_sparse("test query", limit=5)
                
                results.append(f"Worker {worker_id} 完成")
                print(f"   Worker {worker_id} 完成工作")
                
        except Exception as e:
            print(f"   Worker {worker_id} 发生错误: {e}")
            results.append(f"Worker {worker_id} 错误: {e}")
    
    print("1. 启动5个并发工作线程...")
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("2. 所有工作线程已完成")
    print(f"   结果: {results}")
    
    # 查看最终池状态
    status = pool.get_pool_status()
    print(f"3. 最终池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")

def test_pool_limits():
    """测试池大小限制"""
    print("\n=== 测试池大小限制 ===")
    
    config = load_config()
    milvus_uri = config.get('milvus_uri', 'localhost')
    milvus_token = config.get('milvus_token', 'your_milvus_token')
    collection_names = config.get('collection_names', ['your_collection_name'])
    
    # 创建一个小的池进行测试
    from milvus_search import MilvusSearchPool
    test_pool = MilvusSearchPool(max_pool_size=2, max_idle_time=30)
    
    try:
        instances = []
        
        print("1. 尝试获取超过池大小限制的实例...")
        
        # 获取实例直到达到限制
        for i in range(3):
            try:
                instance = test_pool.get_instance(milvus_uri, milvus_token, f"{collection_names[0]}_{i}")
                instances.append((milvus_uri, milvus_token, f"{collection_names[0]}_{i}"))
                print(f"   成功获取实例 {i+1}")
                
                status = test_pool.get_pool_status()
                print(f"   池状态 - 总数: {status['total_instances']}, 使用中: {status['in_use_instances']}")
                
            except RuntimeError as e:
                print(f"   预期的错误: {e}")
                break
        
        print("2. 释放一个实例后再次尝试...")
        if instances:
            uri, token, collection = instances[0]
            test_pool.release_instance(uri, token, collection)
            print("   释放了一个实例")
            
            # 现在应该能够获取新实例
            try:
                instance = test_pool.get_instance(milvus_uri, milvus_token, f"{collection_names[0]}_new")
                print("   成功获取新实例")
            except Exception as e:
                print(f"   获取新实例失败: {e}")
        
        # 清理
        test_pool.shutdown()
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")

def main():
    """主函数"""
    print("开始 MilvusSearch 实例池测试")
    print("=" * 50)
    
    try:
        # 运行各种测试
        test_basic_pool_usage()
        test_context_manager()
        test_concurrent_access()
        test_pool_limits()
        
        print("\n" + "=" * 50)
        print("所有测试完成")
        
    except Exception as e:
        print(f"测试过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理全局池
        print("\n关闭全局实例池...")
        shutdown_global_pool()
        print("测试结束")

if __name__ == "__main__":
    main()
