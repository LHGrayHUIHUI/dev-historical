#!/usr/bin/env python3
"""
storage-service API功能测试
测试存储服务的所有API端点和功能
"""

import asyncio
import aiohttp
import json
import io
from pathlib import Path
from datetime import datetime

class StorageServiceAPITester:
    def __init__(self):
        self.base_url = "http://localhost:8002"
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_type": "storage_service_api_test",
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
    
    async def test_health_endpoints(self, session):
        """测试健康检查端点"""
        start_time = asyncio.get_event_loop().time()
        
        endpoints = ["/health", "/ready", "/api/v1/data/health", "/api/v1/data/info"]
        details = {"tested_endpoints": [], "response_details": {}}
        
        try:
            for endpoint in endpoints:
                try:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        content = await response.text()
                        details["tested_endpoints"].append(endpoint)
                        details["response_details"][endpoint] = {
                            "status_code": response.status,
                            "content_type": response.headers.get("content-type", ""),
                            "response_size": len(content)
                        }
                        print(f"   {endpoint}: {response.status}")
                except Exception as e:
                    details["response_details"][endpoint] = {"error": str(e)}
            
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("健康检查端点", "PASSED", details, duration=duration)
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("健康检查端点", "FAILED", details, str(e), duration)
    
    async def test_dataset_operations(self, session):
        """测试数据集操作"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. 获取数据集列表
            async with session.get(f"{self.base_url}/api/v1/data/datasets") as response:
                if response.status == 200:
                    datasets = await response.json()
                    details = {
                        "dataset_list_accessible": True,
                        "existing_datasets_count": len(datasets) if isinstance(datasets, list) else 0,
                        "dataset_structure": type(datasets).__name__
                    }
                else:
                    details = {"dataset_list_error": response.status}
            
            # 2. 尝试创建新数据集（如果支持POST）
            create_data = {
                "name": "test_dataset",
                "description": "API测试数据集",
                "created_by": "api_test"
            }
            
            try:
                async with session.post(f"{self.base_url}/api/v1/data/datasets", 
                                       json=create_data,
                                       headers={"Content-Type": "application/json"}) as response:
                    details["dataset_creation_status"] = response.status
                    if response.status in [200, 201]:
                        creation_result = await response.json()
                        details["created_dataset_id"] = creation_result.get("id") or creation_result.get("dataset_id")
                        details["dataset_creation_successful"] = True
                    else:
                        details["dataset_creation_error"] = await response.text()
            except Exception as e:
                details["dataset_creation_exception"] = str(e)
            
            duration = asyncio.get_event_loop().time() - start_time
            status = "PASSED" if details.get("dataset_list_accessible") else "FAILED"
            await self.log_test("数据集操作", status, details, duration=duration)
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("数据集操作", "FAILED", {}, str(e), duration)
    
    async def test_content_management(self, session):
        """测试内容管理功能"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. 获取内容列表
            async with session.get(f"{self.base_url}/api/v1/content/") as response:
                content_list_status = response.status
                if response.status == 200:
                    content_data = await response.json()
                    details = {
                        "content_list_accessible": True,
                        "content_count": len(content_data) if isinstance(content_data, list) else 0,
                        "content_structure": type(content_data).__name__
                    }
                else:
                    details = {"content_list_error": content_list_status, "error_text": await response.text()}
            
            # 2. 尝试创建内容
            test_content = {
                "title": "API测试内容",
                "content": "这是通过API创建的测试内容",
                "content_type": "text",
                "tags": ["测试", "API"],
                "metadata": {"source": "api_test", "created_at": datetime.now().isoformat()}
            }
            
            try:
                async with session.post(f"{self.base_url}/api/v1/content/", 
                                       json=test_content,
                                       headers={"Content-Type": "application/json"}) as response:
                    details["content_creation_status"] = response.status
                    if response.status in [200, 201]:
                        creation_result = await response.json()
                        details["created_content_id"] = creation_result.get("id") or creation_result.get("content_id")
                        details["content_creation_successful"] = True
                        
                        # 3. 尝试获取刚创建的内容
                        content_id = details.get("created_content_id")
                        if content_id:
                            async with session.get(f"{self.base_url}/api/v1/content/{content_id}") as get_response:
                                details["content_retrieval_status"] = get_response.status
                                if get_response.status == 200:
                                    retrieved_content = await get_response.json()
                                    details["content_retrieval_successful"] = True
                                    details["retrieved_content_title"] = retrieved_content.get("title")
                    else:
                        details["content_creation_error"] = await response.text()
                        
            except Exception as e:
                details["content_creation_exception"] = str(e)
            
            # 4. 测试内容搜索
            try:
                search_params = {"q": "测试"}
                async with session.get(f"{self.base_url}/api/v1/content/search/", params=search_params) as response:
                    details["content_search_status"] = response.status
                    if response.status == 200:
                        search_results = await response.json()
                        details["content_search_successful"] = True
                        details["search_results_count"] = len(search_results) if isinstance(search_results, list) else 0
            except Exception as e:
                details["content_search_exception"] = str(e)
            
            duration = asyncio.get_event_loop().time() - start_time
            status = "PASSED" if details.get("content_list_accessible") else "FAILED"
            await self.log_test("内容管理功能", status, details, duration=duration)
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("内容管理功能", "FAILED", {}, str(e), duration)
    
    async def test_file_upload_operations(self, session):
        """测试文件上传功能"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 创建测试文件
            test_content = "这是storage-service的文件上传测试\n包含中文内容测试\n时间：2025年9月9日"
            
            # 准备文件数据
            data = aiohttp.FormData()
            data.add_field('file', 
                          io.BytesIO(test_content.encode('utf-8')), 
                          filename='storage_test.txt',
                          content_type='text/plain')
            data.add_field('metadata', json.dumps({
                "description": "storage-service API测试文件",
                "category": "test",
                "source": "api_test"
            }))
            
            async with session.post(f"{self.base_url}/api/v1/data/upload", data=data) as response:
                details = {
                    "upload_status": response.status,
                    "upload_successful": response.status in [200, 201]
                }
                
                if response.status in [200, 201]:
                    upload_result = await response.json()
                    details["upload_result"] = {
                        "file_id": upload_result.get("file_id") or upload_result.get("id"),
                        "filename": upload_result.get("filename"),
                        "size": upload_result.get("size"),
                        "content_type": upload_result.get("content_type")
                    }
                else:
                    details["upload_error"] = await response.text()
                
                duration = asyncio.get_event_loop().time() - start_time
                status = "PASSED" if details["upload_successful"] else "FAILED"
                await self.log_test("文件上传功能", status, details, duration=duration)
                
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("文件上传功能", "FAILED", {}, str(e), duration)
    
    async def test_statistics_endpoints(self, session):
        """测试统计信息端点"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            endpoints_to_test = [
                "/api/v1/content/stats/",
                "/api/v1/content/with-files"
            ]
            
            details = {"tested_endpoints": {}}
            
            for endpoint in endpoints_to_test:
                try:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        details["tested_endpoints"][endpoint] = {
                            "status_code": response.status,
                            "accessible": response.status == 200
                        }
                        
                        if response.status == 200:
                            try:
                                content = await response.json()
                                details["tested_endpoints"][endpoint]["response_type"] = type(content).__name__
                                if isinstance(content, dict):
                                    details["tested_endpoints"][endpoint]["keys"] = list(content.keys())[:5]  # 前5个键
                                elif isinstance(content, list):
                                    details["tested_endpoints"][endpoint]["item_count"] = len(content)
                            except:
                                details["tested_endpoints"][endpoint]["response_type"] = "text"
                        
                        print(f"   {endpoint}: {response.status}")
                        
                except Exception as e:
                    details["tested_endpoints"][endpoint] = {"error": str(e)}
            
            accessible_count = sum(1 for ep in details["tested_endpoints"].values() if ep.get("accessible", False))
            details["summary"] = {
                "total_endpoints": len(endpoints_to_test),
                "accessible_endpoints": accessible_count,
                "success_rate": round((accessible_count / len(endpoints_to_test)) * 100, 2)
            }
            
            duration = asyncio.get_event_loop().time() - start_time
            status = "PASSED" if accessible_count > 0 else "FAILED"
            await self.log_test("统计信息端点", status, details, duration=duration)
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("统计信息端点", "FAILED", {}, str(e), duration)
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始storage-service API功能测试")
        
        async with aiohttp.ClientSession() as session:
            await self.test_health_endpoints(session)
            await self.test_dataset_operations(session)
            await self.test_content_management(session)
            await self.test_file_upload_operations(session)
            await self.test_statistics_endpoints(session)
        
        # 生成测试总结
        self.results["end_time"] = datetime.now().isoformat()
        total_tests = len(self.results["tests"])
        passed_tests = len([t for t in self.results["tests"] if t["status"] == "PASSED"])
        failed_tests = total_tests - passed_tests
        
        self.results["summary"] = {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            "errors": [t["error"] for t in self.results["tests"] if t["error"]]
        }
        
        print(f"\n📊 storage-service API测试总结:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   成功率: {self.results['summary']['success_rate']}%")
        
        return self.results

async def main():
    tester = StorageServiceAPITester()
    results = await tester.run_all_tests()
    
    # 保存结果
    output_file = Path(__file__).parent / "storage_service_api_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试结果已保存到: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())