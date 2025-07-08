# MilvusSearch 实例池使用指南

## 概述

`MilvusSearchPool` 是一个用于管理和复用 `MilvusSearch` 实例的池化管理器。它可以：

- 避免频繁创建和销毁 Milvus 连接
- 支持连接复用，提高性能
- 自动管理实例生命周期
- 支持并发访问
- 提供连接池大小控制

## 主要特性

1. **实例复用**: 相同连接参数的实例会被复用
2. **自动清理**: 空闲时间超过阈值的实例会被自动清理
3. **线程安全**: 支持多线程并发访问
4. **池大小控制**: 可以限制池中的最大实例数量
5. **上下文管理**: 提供上下文管理器，自动管理实例获取和释放

## 使用方法

### 1. 基本使用

```python
from milvus_search import get_global_pool

# 获取全局实例池
pool = get_global_pool(max_pool_size=10, max_idle_time=300)

# 获取实例
search_instance = pool.get_instance(uri, token, collection_name)

# 使用实例进行搜索
results = search_instance.search_sparse("query text", limit=10)

# 释放实例回池
pool.release_instance(uri, token, collection_name)
```

### 2. 使用上下文管理器（推荐）

```python
from milvus_search import get_global_pool

pool = get_global_pool()

# 使用上下文管理器，自动管理实例生命周期
with pool.get_instance_context(uri, token, collection_name) as search_instance:
    results = search_instance.search_sparse("query text", limit=10)
    # 退出 with 块时，实例会自动释放回池
```

### 3. 并发使用

```python
import threading
from milvus_search import get_global_pool

pool = get_global_pool()

def worker(worker_id):
    with pool.get_instance_context(uri, token, collection_name) as search_instance:
        results = search_instance.search_sparse(f"query_{worker_id}", limit=10)
        print(f"Worker {worker_id} 完成，结果数量: {len(results)}")

# 创建多个线程并发执行
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

## 配置参数

### MilvusSearchPool 初始化参数

- `max_pool_size`: 池中最大实例数量（默认: 10）
- `max_idle_time`: 实例最大空闲时间，秒（默认: 300）

### 全局池函数参数

```python
get_global_pool(max_pool_size=10, max_idle_time=300)
```

## 池状态监控

```python
# 获取池状态信息
status = pool.get_pool_status()
print(f"总实例数: {status['total_instances']}")
print(f"使用中实例数: {status['in_use_instances']}")
print(f"空闲实例数: {status['idle_instances']}")
print(f"最大池大小: {status['max_pool_size']}")

# 查看各个实例的详细状态
for key, info in status['instances'].items():
    print(f"实例 {key}: 使用中={info['in_use']}, 空闲时间={info['idle_time']}秒")
```

## 最佳实践

### 1. 使用上下文管理器

始终优先使用上下文管理器，它会自动处理实例的获取和释放：

```python
# ✅ 推荐
with pool.get_instance_context(uri, token, collection) as search_instance:
    # 使用 search_instance

# ❌ 不推荐（除非有特殊需求）
instance = pool.get_instance(uri, token, collection)
try:
    # 使用 instance
finally:
    pool.release_instance(uri, token, collection)
```

### 2. 合理设置池大小

根据应用的并发需求设置合适的池大小：

```python
# 对于高并发应用
pool = get_global_pool(max_pool_size=20, max_idle_time=600)

# 对于低并发应用
pool = get_global_pool(max_pool_size=5, max_idle_time=300)
```

### 3. 程序退出时清理

在程序退出时清理池资源：

```python
import atexit
from milvus_search import shutdown_global_pool

# 注册退出时清理函数
atexit.register(shutdown_global_pool)
```

### 4. 错误处理

```python
try:
    with pool.get_instance_context(uri, token, collection) as search_instance:
        results = search_instance.search_sparse("query", limit=10)
except RuntimeError as e:
    if "实例池已满" in str(e):
        print("池已满，稍后重试或增加池大小")
    else:
        raise
except Exception as e:
    print(f"搜索过程中发生错误: {e}")
```

## 注意事项

1. **连接标识**: 池使用 `uri + token + collection_name` 作为连接的唯一标识
2. **线程安全**: 池本身是线程安全的，可以在多线程环境中使用
3. **内存管理**: 空闲实例会被自动清理，但池中的实例仍会占用内存
4. **连接限制**: 注意 Milvus 服务器的连接数限制
5. **异常处理**: 当池达到最大大小时，会抛出 `RuntimeError`

## 测试和调试

使用提供的测试脚本验证池功能：

```bash
python test_pool.py
```

测试脚本包含：
- 基本池使用功能测试
- 上下文管理器测试
- 并发访问测试
- 池大小限制测试
