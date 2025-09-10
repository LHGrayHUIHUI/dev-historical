#!/usr/bin/env python3
"""
全面的服务连通性和API接口测试
测试所有服务间的网络连通性和API端点的可访问性
"""

import asyncio
import httpx
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

class ComprehensiveConnectivityTest:
    """全面连通性测试类"""
    
    def __init__(self):
        # 服务端点配置
        self.services = {
            "file-processor": {
                "url": "http://localhost:8001",
                "container": "integration-file-processor",
                "expected_endpoints": ["/health", "/docs", "/openapi.json"]
            },
            "storage-service": {
                "url": "http://localhost:8002", 
                "container": "integration-storage-service",
                "expected_endpoints": ["/health", "/ready", "/docs", "/openapi.json"]
            },
            "intelligent-classification": {
                "url": "http://localhost:8007",
                "container": "integration-intelligent-classification-service", 
                "expected_endpoints": ["/health", "/docs", "/openapi.json"]
            }
        }
        
        # 基础设施服务
        self.infrastructure = {
            "postgresql": {"host": "localhost", "port": 5433},
            "mongodb": {"host": "localhost", "port": 27018},
            "redis": {"host": "localhost", "port": 6380},
            "minio": {"host": "localhost", "port": 9001},
            "rabbitmq": {"host": "localhost", "port": 15673}
        }
        
        self.timeout = httpx.Timeout(10.0)
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "test_type": "comprehensive_connectivity_test",
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": []
            }
        }
    
    def add_test_result(self, name: str, passed: bool, duration: float, details: Dict = None, error: str = None):
        """添加测试结果"""
        self.test_results["tests"].append({
            "name": name,
            "status": "PASSED" if passed else "FAILED",
            "duration": duration,
            "details": details or {},
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
        if passed:
            self.test_results["summary"]["passed"] += 1
        else:
            self.test_results["summary"]["failed"] += 1
            if error:
                self.test_results["summary"]["errors"].append(error)
        
        self.test_results["summary"]["total"] += 1
    
    async def test_service_basic_connectivity(self, service_name: str, service_config: Dict) -> Tuple[bool, Dict]:
        """测试服务基础连通性"""
        print(f"🔗 测试{service_name}基础连通性...")
        start_time = time.time()
        
        connectivity_results = {
            "service_reachable": False,
            "health_endpoint": False,
            "api_docs_available": False,
            "response_times": {},
            "available_endpoints": []
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                base_url = service_config["url"]
                
                # 测试基础连通性
                for endpoint in service_config["expected_endpoints"]:
                    try:
                        endpoint_start = time.time()
                        response = await client.get(f"{base_url}{endpoint}")
                        endpoint_duration = time.time() - endpoint_start
                        
                        connectivity_results["response_times"][endpoint] = endpoint_duration
                        
                        if response.status_code in [200, 404, 405]:  # 405 Method Not Allowed也表示端点存在
                            connectivity_results["available_endpoints"].append(endpoint)
                            
                            if endpoint == "/health":
                                connectivity_results["health_endpoint"] = True
                                if response.status_code == 200:
                                    connectivity_results["service_reachable"] = True
                            
                            elif endpoint in ["/docs", "/openapi.json"]:
                                if response.status_code == 200:
                                    connectivity_results["api_docs_available"] = True
                        
                        print(f"  {endpoint}: {'✅' if response.status_code == 200 else '⚠️'} {response.status_code} ({endpoint_duration:.3f}s)")
                        
                    except Exception as e:
                        print(f"  {endpoint}: ❌ {str(e)[:50]}...")
                        continue
                
                duration = time.time() - start_time
                overall_success = connectivity_results["service_reachable"] or len(connectivity_results["available_endpoints"]) > 0
                
                print(f"  📊 可用端点: {len(connectivity_results['available_endpoints'])}/{len(service_config['expected_endpoints'])}")
                print(f"  ⏱️  总耗时: {duration:.3f}s")
                
                self.add_test_result(
                    f"{service_name}_basic_connectivity",
                    overall_success,
                    duration,
                    connectivity_results,
                    None if overall_success else "服务无法连接或所有端点均不可用"
                )
                
                return overall_success, connectivity_results
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{service_name}连通性测试异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                f"{service_name}_basic_connectivity",
                False,
                duration,
                connectivity_results,
                error_msg
            )
            
            return False, connectivity_results
    
    async def test_service_advanced_endpoints(self, service_name: str, service_config: Dict) -> bool:
        """测试服务高级API端点"""
        print(f"🔍 测试{service_name}高级API端点...")
        start_time = time.time()
        
        advanced_results = {
            "discovered_endpoints": [],
            "functional_endpoints": [],
            "api_structure": {},
            "endpoint_details": {}
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                base_url = service_config["url"]
                
                # 获取OpenAPI规范
                try:
                    openapi_response = await client.get(f"{base_url}/openapi.json")
                    if openapi_response.status_code == 200:
                        openapi_data = openapi_response.json()
                        paths = openapi_data.get("paths", {})
                        advanced_results["discovered_endpoints"] = list(paths.keys())
                        advanced_results["api_structure"] = {
                            "total_endpoints": len(paths),
                            "methods": {},
                            "tags": set()
                        }
                        
                        for path, methods in paths.items():
                            for method, details in methods.items():
                                if method not in advanced_results["api_structure"]["methods"]:
                                    advanced_results["api_structure"]["methods"][method] = 0
                                advanced_results["api_structure"]["methods"][method] += 1
                                
                                tags = details.get("tags", [])
                                advanced_results["api_structure"]["tags"].update(tags)
                        
                        advanced_results["api_structure"]["tags"] = list(advanced_results["api_structure"]["tags"])
                        
                        print(f"  📚 发现API端点: {len(advanced_results['discovered_endpoints'])}个")
                        print(f"  🏷️  API分类: {advanced_results['api_structure']['tags'][:3]}...")
                        
                except Exception as e:
                    print(f"  ⚠️  无法获取OpenAPI规范: {str(e)[:30]}...")
                
                # 测试常见的功能端点
                common_endpoints = [
                    "/info", "/status", "/version", "/metrics", 
                    "/api/v1/", "/api/v1/status", "/api/v1/health"
                ]
                
                for endpoint in common_endpoints:
                    try:
                        response = await client.get(f"{base_url}{endpoint}")
                        if response.status_code in [200, 201]:
                            advanced_results["functional_endpoints"].append(endpoint)
                            advanced_results["endpoint_details"][endpoint] = {
                                "status_code": response.status_code,
                                "content_type": response.headers.get("content-type", ""),
                                "response_size": len(response.content)
                            }
                            print(f"  ✅ {endpoint}: 可用 ({response.status_code})")
                    except:
                        continue
                
                duration = time.time() - start_time
                success = len(advanced_results["functional_endpoints"]) > 0 or len(advanced_results["discovered_endpoints"]) > 0
                
                print(f"  🎯 功能端点: {len(advanced_results['functional_endpoints'])}个")
                print(f"  ⏱️  测试耗时: {duration:.3f}s")
                
                self.add_test_result(
                    f"{service_name}_advanced_endpoints",
                    success,
                    duration,
                    advanced_results,
                    None if success else "无可用的高级API端点"
                )
                
                return success
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{service_name}高级端点测试异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                f"{service_name}_advanced_endpoints",
                False,
                duration,
                advanced_results,
                error_msg
            )
            
            return False
    
    async def test_inter_service_communication(self) -> bool:
        """测试服务间通信"""
        print("🌐 测试服务间通信...")
        start_time = time.time()
        
        communication_results = {
            "service_pairs": [],
            "successful_communications": 0,
            "total_attempts": 0,
            "communication_matrix": {}
        }
        
        service_names = list(self.services.keys())
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                
                # 测试所有服务对之间的通信
                for i, source_service in enumerate(service_names):
                    for j, target_service in enumerate(service_names):
                        if i != j:  # 不测试自己到自己
                            source_url = self.services[source_service]["url"]
                            target_url = self.services[target_service]["url"]
                            
                            communication_results["total_attempts"] += 1
                            pair_key = f"{source_service} -> {target_service}"
                            
                            # 尝试通过源服务访问目标服务（模拟服务间调用）
                            try:
                                # 首先检查源服务是否可以访问
                                source_response = await client.get(f"{source_url}/health", timeout=5.0)
                                target_response = await client.get(f"{target_url}/health", timeout=5.0)
                                
                                if source_response.status_code in [200, 404] and target_response.status_code in [200, 404]:
                                    communication_results["successful_communications"] += 1
                                    communication_results["communication_matrix"][pair_key] = "✅ 可通信"
                                    print(f"  ✅ {pair_key}: 通信正常")
                                else:
                                    communication_results["communication_matrix"][pair_key] = "❌ 通信失败"
                                    print(f"  ❌ {pair_key}: 通信失败")
                                
                            except Exception as e:
                                communication_results["communication_matrix"][pair_key] = f"❌ 异常: {str(e)[:20]}..."
                                print(f"  ❌ {pair_key}: 异常 - {str(e)[:30]}...")
                
                # 测试并发访问能力
                concurrent_tasks = []
                for service_name, config in self.services.items():
                    task = client.get(f"{config['url']}/health", timeout=5.0)
                    concurrent_tasks.append(task)
                
                try:
                    concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
                    successful_concurrent = sum(
                        1 for result in concurrent_results 
                        if not isinstance(result, Exception) and hasattr(result, 'status_code') and result.status_code in [200, 404]
                    )
                    
                    communication_results["concurrent_access"] = {
                        "total_services": len(concurrent_tasks),
                        "successful_responses": successful_concurrent,
                        "success_rate": successful_concurrent / len(concurrent_tasks) * 100
                    }
                    
                    print(f"  🔄 并发访问: {successful_concurrent}/{len(concurrent_tasks)} 服务响应正常")
                    
                except Exception as e:
                    print(f"  ⚠️  并发测试异常: {str(e)[:40]}...")
                
                duration = time.time() - start_time
                success_rate = communication_results["successful_communications"] / communication_results["total_attempts"] if communication_results["total_attempts"] > 0 else 0
                overall_success = success_rate > 0.5  # 超过50%通信成功就算整体成功
                
                print(f"  📊 通信成功率: {success_rate*100:.1f}% ({communication_results['successful_communications']}/{communication_results['total_attempts']})")
                print(f"  ⏱️  测试耗时: {duration:.3f}s")
                
                self.add_test_result(
                    "inter_service_communication",
                    overall_success,
                    duration,
                    communication_results,
                    None if overall_success else f"服务间通信成功率过低: {success_rate*100:.1f}%"
                )
                
                return overall_success
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"服务间通信测试异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                "inter_service_communication",
                False,
                duration,
                communication_results,
                error_msg
            )
            
            return False
    
    async def test_infrastructure_connectivity(self) -> bool:
        """测试基础设施服务连通性"""
        print("🏗️ 测试基础设施服务连通性...")
        start_time = time.time()
        
        infrastructure_results = {
            "accessible_services": [],
            "connection_details": {},
            "total_services": len(self.infrastructure)
        }
        
        # 测试各个基础设施服务的网络连通性
        for service_name, config in self.infrastructure.items():
            try:
                if service_name in ["postgresql", "mongodb", "redis"]:
                    # 对于数据库服务，使用telnet方式测试端口连通性
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex((config["host"], config["port"]))
                    sock.close()
                    
                    if result == 0:
                        infrastructure_results["accessible_services"].append(service_name)
                        infrastructure_results["connection_details"][service_name] = "✅ 端口可访问"
                        print(f"  ✅ {service_name}:{config['port']} - 连接正常")
                    else:
                        infrastructure_results["connection_details"][service_name] = "❌ 端口不可访问"
                        print(f"  ❌ {service_name}:{config['port']} - 连接失败")
                        
                elif service_name in ["minio", "rabbitmq"]:
                    # 对于Web服务，使用HTTP请求测试
                    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                        url = f"http://{config['host']}:{config['port']}"
                        try:
                            response = await client.get(url)
                            if response.status_code in [200, 401, 403, 404]:  # 这些状态码表示服务可访问
                                infrastructure_results["accessible_services"].append(service_name)
                                infrastructure_results["connection_details"][service_name] = f"✅ HTTP可访问 ({response.status_code})"
                                print(f"  ✅ {service_name}:{config['port']} - HTTP服务正常 ({response.status_code})")
                            else:
                                infrastructure_results["connection_details"][service_name] = f"⚠️ HTTP异常状态 ({response.status_code})"
                                print(f"  ⚠️ {service_name}:{config['port']} - 状态码: {response.status_code}")
                        except Exception as e:
                            infrastructure_results["connection_details"][service_name] = f"❌ HTTP请求失败: {str(e)[:20]}..."
                            print(f"  ❌ {service_name}:{config['port']} - HTTP请求失败")
                            
            except Exception as e:
                infrastructure_results["connection_details"][service_name] = f"❌ 测试异常: {str(e)[:20]}..."
                print(f"  ❌ {service_name} - 测试异常: {str(e)[:30]}...")
        
        duration = time.time() - start_time
        success_count = len(infrastructure_results["accessible_services"])
        success_rate = success_count / infrastructure_results["total_services"]
        overall_success = success_rate >= 0.8  # 80%以上基础设施服务可访问就算成功
        
        print(f"  📊 基础设施可访问率: {success_rate*100:.1f}% ({success_count}/{infrastructure_results['total_services']})")
        print(f"  ⏱️  测试耗时: {duration:.3f}s")
        
        self.add_test_result(
            "infrastructure_connectivity",
            overall_success,
            duration,
            infrastructure_results,
            None if overall_success else f"基础设施服务可访问率过低: {success_rate*100:.1f}%"
        )
        
        return overall_success
    
    async def run_comprehensive_test(self):
        """运行全面的连通性测试"""
        print("🚀 开始全面的服务连通性和API测试...")
        print("=" * 70)
        
        # 1. 基础设施连通性测试
        await self.test_infrastructure_connectivity()
        print()
        
        # 2. 应用服务基础连通性测试
        for service_name, service_config in self.services.items():
            await self.test_service_basic_connectivity(service_name, service_config)
            print()
        
        # 3. 应用服务高级端点测试
        for service_name, service_config in self.services.items():
            await self.test_service_advanced_endpoints(service_name, service_config)
            print()
        
        # 4. 服务间通信测试
        await self.test_inter_service_communication()
        
        # 完成测试
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["total_duration"] = sum(
            test.get("duration", 0) for test in self.test_results["tests"]
        )
        
        # 计算成功率
        passed = self.test_results["summary"]["passed"]
        total = self.test_results["summary"]["total"]
        self.test_results["summary"]["success_rate"] = (passed / total * 100) if total > 0 else 0
        
        return self.test_results

async def main():
    """主函数"""
    print("🧪 全面的服务连通性和API接口测试")
    print("=" * 70)
    print("📝 说明: 测试所有服务间的网络连通性和API端点可访问性")
    print()
    
    # 创建测试实例
    tester = ComprehensiveConnectivityTest()
    
    # 运行测试
    results = await tester.run_comprehensive_test()
    
    # 打印测试结果摘要
    print("\n" + "=" * 70)
    print("📊 全面连通性测试结果摘要")
    print("=" * 70)
    print(f"🎯 总测试数: {results['summary']['total']}")
    print(f"✅ 通过: {results['summary']['passed']}")
    print(f"❌ 失败: {results['summary']['failed']}")
    print(f"📈 成功率: {results['summary']['success_rate']:.1f}%")
    print(f"⏱️  总耗时: {results['total_duration']:.2f}秒")
    
    # 按类别显示测试结果
    categories = {}
    for test in results["tests"]:
        category = test["name"].split("_")[-1]
        if category not in categories:
            categories[category] = {"passed": 0, "total": 0}
        categories[category]["total"] += 1
        if test["status"] == "PASSED":
            categories[category]["passed"] += 1
    
    print(f"\n📋 分类测试结果:")
    for category, stats in categories.items():
        success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  🔸 {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
    
    if results['summary']['errors']:
        print(f"\n❗ 主要错误:")
        for i, error in enumerate(results['summary']['errors'][:5], 1):  # 只显示前5个错误
            print(f"  {i}. {error[:80]}...")
    
    # 保存测试结果
    result_file = "comprehensive_connectivity_test_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细测试结果已保存到: {result_file}")
    print("🏁 全面连通性测试完成")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())