#!/usr/bin/env python3
"""
简化集成测试 - 验证Epic 1基础设施
"""

import asyncio
import time
import sys
import requests
import psycopg2
import redis
from pathlib import Path

def test_infrastructure_services():
    """测试基础设施服务连接"""
    print("🔍 测试基础设施服务...")
    
    results = {}
    
    # 测试PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="historical_text_test",
            user="postgres",
            password="testpass123",
            connect_timeout=5
        )
        conn.close()
        print("  ✅ PostgreSQL 连接成功")
        results["postgresql"] = True
    except Exception as e:
        print(f"  ❌ PostgreSQL 连接失败: {e}")
        results["postgresql"] = False
    
    # 测试Redis
    try:
        r = redis.Redis(host='localhost', port=6380, db=0, socket_connect_timeout=5)
        r.ping()
        print("  ✅ Redis 连接成功")
        results["redis"] = True
    except Exception as e:
        print(f"  ❌ Redis 连接失败: {e}")
        results["redis"] = False
    
    # 测试MinIO (通过HTTP)
    try:
        response = requests.get('http://localhost:9001', timeout=5)
        if response.status_code in [200, 403]:  # 403 表示认证问题但服务正常
            print("  ✅ MinIO 服务可访问")
            results["minio"] = True
        else:
            print(f"  ❌ MinIO 状态异常: {response.status_code}")
            results["minio"] = False
    except Exception as e:
        print(f"  ❌ MinIO 连接失败: {e}")
        results["minio"] = False
    
    # 测试RabbitMQ管理界面
    try:
        response = requests.get('http://localhost:15673', timeout=5)
        if response.status_code in [200, 401]:  # 401表示认证但服务正常
            print("  ✅ RabbitMQ 管理界面可访问")
            results["rabbitmq"] = True
        else:
            print(f"  ❌ RabbitMQ 状态异常: {response.status_code}")
            results["rabbitmq"] = False
    except Exception as e:
        print(f"  ❌ RabbitMQ 连接失败: {e}")
        results["rabbitmq"] = False
    
    return results

def test_microservices():
    """测试微服务连接"""
    print("\n🔍 测试微服务...")
    
    results = {}
    services = {
        "data-source": "http://localhost:8001",
        "data-collection": "http://localhost:8003"
    }
    
    for service_name, base_url in services.items():
        try:
            # 测试健康检查端点
            health_url = f"{base_url}/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ {service_name}服务健康: {data.get('status', 'unknown')}")
                results[service_name] = True
            else:
                print(f"  ❌ {service_name}服务异常: HTTP {response.status_code}")
                results[service_name] = False
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ {service_name}服务不可访问: 连接被拒绝")
            results[service_name] = False
        except requests.exceptions.Timeout:
            print(f"  ❌ {service_name}服务不可访问: 连接超时")
            results[service_name] = False
        except Exception as e:
            print(f"  ❌ {service_name}服务测试失败: {e}")
            results[service_name] = False
    
    return results

def test_docker_containers():
    """测试Docker容器状态"""
    print("\n🔍 测试Docker容器状态...")
    
    import subprocess
    try:
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.dev.yml", "ps", "--format", "table"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("  ✅ Docker Compose服务列表:")
            lines = result.stdout.strip().split('\n')
            container_count = 0
            healthy_count = 0
            
            for line in lines[1:]:  # 跳过标题行
                if line.strip():
                    container_count += 1
                    if "healthy" in line.lower() or "up" in line.lower():
                        healthy_count += 1
                    print(f"    {line}")
            
            print(f"  📊 容器统计: {healthy_count}/{container_count} 运行正常")
            return {"containers": container_count, "healthy": healthy_count}
        else:
            print(f"  ❌ Docker Compose命令失败: {result.stderr}")
            return {"containers": 0, "healthy": 0}
            
    except Exception as e:
        print(f"  ❌ Docker状态检查失败: {e}")
        return {"containers": 0, "healthy": 0}

def main():
    """运行简化集成测试"""
    print("🚀 开始Epic 1简化集成测试")
    print("=" * 50)
    
    # 测试基础设施
    infra_results = test_infrastructure_services()
    
    # 测试微服务
    service_results = test_microservices()
    
    # 测试容器状态
    container_stats = test_docker_containers()
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 集成测试结果汇总:")
    
    # 基础设施得分
    infra_passed = sum(infra_results.values())
    infra_total = len(infra_results)
    print(f"  🏗️  基础设施服务: {infra_passed}/{infra_total} ({infra_passed/infra_total*100:.1f}%)")
    
    # 微服务得分
    service_passed = sum(service_results.values())
    service_total = len(service_results)
    print(f"  🔧 微服务: {service_passed}/{service_total} ({service_passed/service_total*100:.1f}% if service_total > 0 else 0:.1f%)")
    
    # 容器状态
    container_healthy = container_stats["healthy"]
    container_total = container_stats["containers"]
    print(f"  🐳 Docker容器: {container_healthy}/{container_total} 健康")
    
    # 总体评估
    total_passed = infra_passed + service_passed
    total_tests = infra_total + service_total
    overall_rate = total_passed / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n  📈 总体成功率: {overall_rate:.1f}%")
    
    if overall_rate >= 50:
        print("  🎉 基础设施基本可用，支持进一步集成测试")
        return 0
    else:
        print("  ⚠️  基础设施存在较多问题，建议排查服务启动状态")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏸️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 测试执行异常: {e}")
        sys.exit(1)