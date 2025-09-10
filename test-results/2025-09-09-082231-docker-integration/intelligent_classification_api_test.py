#!/usr/bin/env python3
"""
intelligent-classification API功能测试
测试智能分类服务的API接口和功能
"""

import asyncio
import aiohttp
import json
import io
from pathlib import Path
from datetime import datetime
import time

class IntelligentClassificationAPITester:
    def __init__(self):
        self.base_url = "http://localhost:8007"
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_type": "intelligent_classification_api_test",
            "tests": []
        }
    
    async def log_test(self, name: str, status: str, details: dict, error: str = None, duration: float = 0):
        """记录测试结果"""
        test_result = {
            "name": name,
            "status": status,
            "duration": duration,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.results["tests"].append(test_result)
        print(f"✅ {name}: {status}" if status == "PASSED" else f"❌ {name}: {status}")
        if error:
            print(f"   错误: {error}")
    
    async def test_service_availability(self, session):
        """测试服务可用性"""
        start_time = asyncio.get_event_loop().time()
        
        endpoints_to_try = [
            "/", "/health", "/ready", "/docs", "/openapi.json",
            "/api/v1/classify", "/api/v1/status", "/api/v1/models"
        ]
        
        details = {
            "service_reachable": False,
            "responding_endpoints": [],
            "non_responding_endpoints": [],
            "connection_errors": {}
        }
        
        for endpoint in endpoints_to_try:
            try:
                # 设置较短的超时时间
                timeout = aiohttp.ClientTimeout(total=5, connect=2)
                async with session.get(f"{self.base_url}{endpoint}", timeout=timeout) as response:
                    details["service_reachable"] = True
                    details["responding_endpoints"].append({
                        "endpoint": endpoint,
                        "status_code": response.status,
                        "content_type": response.headers.get("content-type", ""),
                        "response_size": len(await response.read())
                    })
                    print(f"   ✅ {endpoint}: {response.status}")
                    
            except asyncio.TimeoutError:
                details["non_responding_endpoints"].append(endpoint)
                details["connection_errors"][endpoint] = "连接超时"
                print(f"   ⏱️  {endpoint}: 连接超时")
            except aiohttp.ClientConnectionError as e:
                details["non_responding_endpoints"].append(endpoint)
                details["connection_errors"][endpoint] = f"连接错误: {str(e)}"
                print(f"   ❌ {endpoint}: 连接错误")
            except Exception as e:
                details["non_responding_endpoints"].append(endpoint)
                details["connection_errors"][endpoint] = f"未知错误: {str(e)}"
                print(f"   ❓ {endpoint}: {str(e)}")
        
        duration = asyncio.get_event_loop().time() - start_time
        status = "PASSED" if details["service_reachable"] else "FAILED"
        error_msg = "服务完全无法访问" if not details["service_reachable"] else None
        
        await self.log_test("服务可用性检查", status, details, error_msg, duration)
        return details["service_reachable"]
    
    async def test_container_status(self, session):
        """测试容器状态（通过docker命令）"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 这里我们不能直接执行docker命令，但可以检测网络层面的连接
            details = {
                "port_test_method": "tcp_connection_test",
                "tested_ports": [8007]
            }
            
            # 尝试建立TCP连接来测试端口是否开放
            import socket
            
            for port in details["tested_ports"]:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    details[f"port_{port}_status"] = "open" if result == 0 else "closed"
                    details[f"port_{port}_result_code"] = result
                    
                except Exception as e:
                    details[f"port_{port}_error"] = str(e)
            
            duration = asyncio.get_event_loop().time() - start_time
            port_open = any(details.get(f"port_{p}_status") == "open" for p in details["tested_ports"])
            status = "PASSED" if port_open else "FAILED"
            error_msg = "所有端口都无法连接" if not port_open else None
            
            await self.log_test("容器网络状态", status, details, error_msg, duration)
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("容器网络状态", "FAILED", {}, str(e), duration)
    
    async def test_classification_endpoints(self, session):
        """测试分类相关端点（如果服务可用）"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 尝试获取服务信息
            timeout = aiohttp.ClientTimeout(total=10, connect=3)
            
            test_endpoints = [
                "/api/v1/classify",
                "/api/v1/models", 
                "/api/v1/categories",
                "/api/v1/health"
            ]
            
            details = {"endpoint_tests": {}}
            
            for endpoint in test_endpoints:
                try:
                    # 先尝试GET请求
                    async with session.get(f"{self.base_url}{endpoint}", timeout=timeout) as response:
                        details["endpoint_tests"][endpoint] = {
                            "method": "GET",
                            "status_code": response.status,
                            "accessible": True,
                            "content_type": response.headers.get("content-type", "")
                        }
                        
                        if response.status == 200:
                            try:
                                content = await response.json()
                                details["endpoint_tests"][endpoint]["response_type"] = type(content).__name__
                            except:
                                details["endpoint_tests"][endpoint]["response_type"] = "text"
                        
                        print(f"   ✅ GET {endpoint}: {response.status}")
                        
                except asyncio.TimeoutError:
                    details["endpoint_tests"][endpoint] = {
                        "method": "GET", "accessible": False, "error": "超时"
                    }
                    print(f"   ⏱️  GET {endpoint}: 超时")
                except Exception as e:
                    details["endpoint_tests"][endpoint] = {
                        "method": "GET", "accessible": False, "error": str(e)
                    }
                    print(f"   ❌ GET {endpoint}: {str(e)}")
            
            # 如果/api/v1/classify可用，尝试POST请求
            if details["endpoint_tests"].get("/api/v1/classify", {}).get("accessible"):
                try:
                    test_data = {
                        "text": "这是一个测试文档，用于验证智能分类功能。内容包含历史文本分析。",
                        "options": {"return_confidence": True}
                    }
                    
                    async with session.post(f"{self.base_url}/api/v1/classify",
                                          json=test_data,
                                          timeout=timeout,
                                          headers={"Content-Type": "application/json"}) as response:
                        details["classification_test"] = {
                            "status_code": response.status,
                            "successful": response.status in [200, 201]
                        }
                        
                        if response.status in [200, 201]:
                            result = await response.json()
                            details["classification_test"]["result"] = {
                                "has_classification": "classification" in result or "category" in result,
                                "has_confidence": "confidence" in result,
                                "response_keys": list(result.keys()) if isinstance(result, dict) else []
                            }
                        
                        print(f"   ✅ POST /api/v1/classify: {response.status}")
                        
                except Exception as e:
                    details["classification_test"] = {"error": str(e)}
                    print(f"   ❌ POST /api/v1/classify: {str(e)}")
            
            duration = asyncio.get_event_loop().time() - start_time
            accessible_endpoints = sum(1 for ep in details["endpoint_tests"].values() if ep.get("accessible", False))
            status = "PASSED" if accessible_endpoints > 0 else "FAILED"
            error_msg = "没有端点可访问" if accessible_endpoints == 0 else None
            
            await self.log_test("分类功能端点", status, details, error_msg, duration)
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("分类功能端点", "FAILED", {}, str(e), duration)
    
    async def test_service_startup_status(self, session):
        """测试服务启动状态"""
        start_time = asyncio.get_event_loop().time()
        
        details = {
            "startup_check_method": "multiple_attempts",
            "max_attempts": 3,
            "attempt_interval": 2,
            "attempts": []
        }
        
        for attempt in range(details["max_attempts"]):
            attempt_start = time.time()
            attempt_details = {"attempt_number": attempt + 1}
            
            try:
                timeout = aiohttp.ClientTimeout(total=5, connect=2)
                async with session.get(f"{self.base_url}/health", timeout=timeout) as response:
                    attempt_details.update({
                        "status_code": response.status,
                        "successful": True,
                        "response_time": time.time() - attempt_start,
                        "content_length": len(await response.read())
                    })
                    print(f"   ✅ 尝试 {attempt + 1}: 成功 (HTTP {response.status})")
                    break
                    
            except Exception as e:
                attempt_details.update({
                    "successful": False,
                    "error": str(e),
                    "response_time": time.time() - attempt_start
                })
                print(f"   ❌ 尝试 {attempt + 1}: 失败 - {str(e)}")
                
                if attempt < details["max_attempts"] - 1:
                    await asyncio.sleep(details["attempt_interval"])
            
            details["attempts"].append(attempt_details)
        
        duration = asyncio.get_event_loop().time() - start_time
        successful_attempts = [a for a in details["attempts"] if a.get("successful")]
        status = "PASSED" if successful_attempts else "FAILED"
        error_msg = "多次尝试均失败" if not successful_attempts else None
        
        details["summary"] = {
            "total_attempts": len(details["attempts"]),
            "successful_attempts": len(successful_attempts),
            "success_rate": len(successful_attempts) / len(details["attempts"]) * 100
        }
        
        await self.log_test("服务启动状态", status, details, error_msg, duration)
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始intelligent-classification API功能测试")
        
        async with aiohttp.ClientSession() as session:
            # 首先检查服务可用性
            service_available = await self.test_service_availability(session)
            
            # 无论服务是否可用都执行这些测试
            await self.test_container_status(session)
            await self.test_service_startup_status(session)
            
            # 只有在服务可用时才测试分类端点
            if service_available:
                await self.test_classification_endpoints(session)
            else:
                await self.log_test("分类功能端点", "SKIPPED", 
                                  {"reason": "服务不可用，跳过功能测试"}, 
                                  "服务基础连接失败")
        
        # 生成测试总结
        self.results["end_time"] = datetime.now().isoformat()
        total_tests = len(self.results["tests"])
        passed_tests = len([t for t in self.results["tests"] if t["status"] == "PASSED"])
        failed_tests = len([t for t in self.results["tests"] if t["status"] == "FAILED"])
        skipped_tests = len([t for t in self.results["tests"] if t["status"] == "SKIPPED"])
        
        self.results["summary"] = {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "skipped": skipped_tests,
            "success_rate": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            "errors": [t["error"] for t in self.results["tests"] if t["error"]]
        }
        
        print(f"\n📊 intelligent-classification API测试总结:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   跳过: {skipped_tests}")
        print(f"   成功率: {self.results['summary']['success_rate']}%")
        
        return self.results

async def main():
    tester = IntelligentClassificationAPITester()
    results = await tester.run_all_tests()
    
    # 保存结果
    output_file = Path(__file__).parent / "intelligent_classification_api_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试结果已保存到: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())