#!/bin/bash

# Milvus Search Service - Curl 测试脚本
# 用于测试 FastAPI 搜索服务的各种功能

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
BASE_URL="http://localhost:10005"
CONTENT_TYPE="application/json"

# 函数：打印带颜色的消息
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 函数：检查服务是否运行
check_service() {
    print_status "检查服务是否运行..."
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs" 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        print_success "服务正在运行 (HTTP $response)"
        return 0
    else
        print_error "服务未运行或无法访问 (HTTP $response)"
        print_warning "请确保服务在 $BASE_URL 上运行"
        return 1
    fi
}

# 函数：测试搜索接口
test_search() {
    local query="$1"
    local test_name="$2"
    
    print_status "测试: $test_name"
    print_status "查询: $query"
    
    # 构建请求体
    local request_body=$(cat <<EOF
{
    "query": "$query"
}
EOF
)
    
    echo "请求体:"
    echo "$request_body" | jq '.' 2>/dev/null || echo "$request_body"
    echo ""
    
    # 发送请求
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: $CONTENT_TYPE" \
        -d "$request_body" \
        "$BASE_URL/search")
    
    # 分离响应体和状态码
    response_body=$(echo "$response" | sed '$d')
    status_code=$(echo "$response" | tail -n1)
    
    echo "响应状态码: $status_code"
    echo "响应内容:"
    
    if [ "$status_code" = "200" ]; then
        print_success "搜索成功"
        # 尝试格式化 JSON 输出
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    elif [ "$status_code" = "404" ]; then
        print_warning "未找到结果"
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    else
        print_error "搜索失败 (HTTP $status_code)"
        echo "$response_body"
    fi
    
    echo ""
    echo "=================================="
    echo ""
}

# 函数：测试健康检查
test_health() {
    print_status "测试健康检查接口..."
    
    response=$(curl -s -w "\n%{http_code}" "$BASE_URL/docs")
    response_body=$(echo "$response" | sed '$d')
    status_code=$(echo "$response" | tail -n1)
    
    if [ "$status_code" = "200" ]; then
        print_success "健康检查通过 (HTTP $status_code)"
    else
        print_error "健康检查失败 (HTTP $status_code)"
    fi
    
    echo ""
}

# 函数：测试 API 文档
test_docs() {
    print_status "测试 API 文档接口..."
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs")
    
    if [ "$response" = "200" ]; then
        print_success "API 文档可访问 (HTTP $response)"
        print_status "API 文档地址: $BASE_URL/docs"
    else
        print_error "API 文档不可访问 (HTTP $response)"
    fi
    
    echo ""
}

# 函数：测试 OpenAPI Schema
test_openapi() {
    print_status "测试 OpenAPI Schema..."
    
    response=$(curl -s -w "\n%{http_code}" "$BASE_URL/openapi.json")
    response_body=$(echo "$response" | sed '$d')
    status_code=$(echo "$response" | tail -n1)
    
    if [ "$status_code" = "200" ]; then
        print_success "OpenAPI Schema 可访问 (HTTP $status_code)"
        echo "Schema 信息:"
        echo "$response_body" | jq '.info' 2>/dev/null || echo "无法解析 JSON"
    else
        print_error "OpenAPI Schema 不可访问 (HTTP $status_code)"
    fi
    
    echo ""
}

# 函数：性能测试
performance_test() {
    local query="$1"
    local iterations="${2:-5}"
    
    print_status "性能测试 - 执行 $iterations 次搜索"
    print_status "查询: $query"
    
    local request_body=$(cat <<EOF
{
    "query": "$query"
}
EOF
)
    
    local total_time=0
    local success_count=0
    
    for i in $(seq 1 $iterations); do
        print_status "执行第 $i 次请求..."
        
        start_time=$(date +%s.%N)
        
        response=$(curl -s -w "%{http_code}" \
            -X POST \
            -H "Content-Type: $CONTENT_TYPE" \
            -d "$request_body" \
            "$BASE_URL/search")
        
        end_time=$(date +%s.%N)
        
        status_code=$(echo "$response" | tail -c 4)
        elapsed=$(echo "$end_time - $start_time" | bc -l)
        total_time=$(echo "$total_time + $elapsed" | bc -l)
        
        if [ "$status_code" = "200" ]; then
            success_count=$((success_count + 1))
            print_success "第 $i 次请求成功 (耗时: ${elapsed}s)"
        else
            print_error "第 $i 次请求失败 (HTTP $status_code)"
        fi
    done
    
    if [ $success_count -gt 0 ]; then
        avg_time=$(echo "scale=3; $total_time / $success_count" | bc -l)
        print_success "性能测试完成"
        echo "成功请求: $success_count/$iterations"
        echo "平均响应时间: ${avg_time}s"
        echo "总耗时: ${total_time}s"
    else
        print_error "所有请求都失败了"
    fi
    
    echo ""
}

# 函数：显示使用说明
show_usage() {
    echo "Milvus Search Service - Curl 测试脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -u, --url URL          设置服务地址 (默认: $BASE_URL)"
    echo "  -h, --help            显示此帮助信息"
    echo "  --health              只执行健康检查"
    echo "  --docs                只测试 API 文档"
    echo "  --search QUERY        执行单个搜索测试"
    echo "  --performance QUERY   执行性能测试"
    echo "  --all                 执行所有测试 (默认)"
    echo ""
    echo "示例:"
    echo "  $0 --search \"人工智能\""
    echo "  $0 --url http://example.com:8000"
    echo "  $0 --performance \"机器学习\" "
    echo ""
}

# 主测试函数
run_all_tests() {
    echo "========================================"
    echo "    Milvus Search Service 测试套件"
    echo "========================================"
    echo "服务地址: $BASE_URL"
    echo "时间: $(date)"
    echo "========================================"
    echo ""
    
    # 检查服务状态
    if ! check_service; then
        exit 1
    fi
    
    # 测试 API 文档
    test_docs
    
    # 测试 OpenAPI Schema
    test_openapi
    
    # 测试健康检查
    test_health
    
    # 搜索功能测试
    echo "========================================"
    echo "           搜索功能测试"
    echo "========================================"
    echo ""
    
    # 测试各种搜索场景
    test_search "人工智能" "中文搜索测试"
    test_search "machine learning" "英文搜索测试"
    test_search "深度学习算法" "技术术语搜索"
    test_search "Python编程" "编程语言搜索"
    test_search "数据结构与算法" "计算机科学搜索"
    test_search "blockchain technology" "英文技术搜索"
    test_search "区块链" "中文技术搜索"
    test_search "神经网络" "AI相关搜索"
    test_search "数据库设计" "数据库搜索"
    test_search "云计算架构" "架构相关搜索"
    
    # 边界情况测试
    echo "========================================"
    echo "           边界情况测试"
    echo "========================================"
    echo ""
    
    test_search "" "空查询测试"
    test_search "a" "单字符搜索"
    test_search "这是一个非常长的查询语句用来测试系统对于长文本的处理能力以及性能表现" "长查询测试"
    test_search "!@#$%^&*()" "特殊字符搜索"
    test_search "不存在的超级复杂术语xyz123" "无结果搜索测试"
    
    # 性能测试
    echo "========================================"
    echo "            性能测试"
    echo "========================================"
    echo ""
    
    performance_test "机器学习" 3
    
    echo "========================================"
    echo "            测试完成"
    echo "========================================"
    print_success "所有测试已完成!"
    print_status "查看详细的 API 文档: $BASE_URL/docs"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        --health)
            check_service
            test_health
            exit 0
            ;;
        --docs)
            test_docs
            exit 0
            ;;
        --search)
            if [ -z "$2" ]; then
                print_error "请提供搜索查询"
                exit 1
            fi
            check_service && test_search "$2" "自定义搜索测试"
            exit 0
            ;;
        --performance)
            if [ -z "$2" ]; then
                print_error "请提供搜索查询"
                exit 1
            fi
            check_service && performance_test "$2" 5
            exit 0
            ;;
        --all)
            # 默认行为，什么都不做
            shift
            ;;
        *)
            print_error "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 如果没有指定特定测试，运行所有测试
run_all_tests
