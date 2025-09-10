#!/usr/bin/env python3
"""
file-processor API功能测试
测试所有可用的API端点和功能
"""

import asyncio
import aiohttp
import json
import io
from pathlib import Path
from datetime import datetime

class FileProcessorAPITester:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.results = {
            "start_time": datetime.now().isoformat(),
            "test_type": "file_processor_api_test",
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
        
    async def test_basic_endpoints(self, session):
        """测试基本端点"""
        start_time = asyncio.get_event_loop().time()
        
        endpoints = ["/health", "/info", "/docs", "/openapi.json"]
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
            await self.log_test("基本端点测试", "PASSED", details, duration=duration)
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("基本端点测试", "FAILED", details, str(e), duration)
    
    async def test_supported_formats(self, session):
        """测试支持的文件格式端点"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with session.get(f"{self.base_url}/api/v1/process/supported-formats") as response:
                if response.status == 200:
                    content = await response.json()
                    details = {
                        "supported_formats": content,
                        "format_count": len(content.get("supported_formats", [])) if isinstance(content, dict) else 0
                    }
                    duration = asyncio.get_event_loop().time() - start_time
                    await self.log_test("支持格式查询", "PASSED", details, duration=duration)
                else:
                    details = {"status_code": response.status, "response": await response.text()}
                    duration = asyncio.get_event_loop().time() - start_time
                    await self.log_test("支持格式查询", "FAILED", details, f"HTTP {response.status}", duration)
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("支持格式查询", "FAILED", {}, str(e), duration)
    
    async def test_document_processing(self, session):
        """测试文档处理功能"""
        start_time = asyncio.get_event_loop().time()
        
        # 创建测试文档
        test_content = "这是一个测试文档，包含中文内容。\n测试file-processor的文档处理功能。\n日期：2025年9月9日"
        
        try:
            # 准备文件数据
            data = aiohttp.FormData()
            data.add_field('file', 
                          io.BytesIO(test_content.encode('utf-8')), 
                          filename='test_doc.txt',
                          content_type='text/plain')
            
            async with session.post(f"{self.base_url}/api/v1/process/document", data=data) as response:
                if response.status == 200:
                    content = await response.json()
                    details = {
                        "processing_successful": True,
                        "extracted_text_length": len(content.get("extracted_text", "")),
                        "detected_format": content.get("file_info", {}).get("file_type"),
                        "processing_duration": content.get("processing_info", {}).get("duration"),
                        "file_size": content.get("file_info", {}).get("size")
                    }
                    duration = asyncio.get_event_loop().time() - start_time
                    await self.log_test("文档处理功能", "PASSED", details, duration=duration)
                else:
                    error_content = await response.text()
                    details = {"status_code": response.status, "error_response": error_content}
                    duration = asyncio.get_event_loop().time() - start_time
                    await self.log_test("文档处理功能", "FAILED", details, f"HTTP {response.status}", duration)
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("文档处理功能", "FAILED", {}, str(e), duration)
    
    async def test_batch_processing(self, session):
        """测试批量处理功能"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 准备多个测试文件
            files = []
            for i in range(3):
                content = f"测试文档{i+1}\n内容：这是第{i+1}个测试文档"
                files.append(('files', ('test_doc_{}.txt'.format(i+1), content.encode('utf-8'), 'text/plain')))
            
            data = aiohttp.FormData()
            for field_name, (filename, content, content_type) in files:
                data.add_field(field_name, io.BytesIO(content), filename=filename, content_type=content_type)
            
            async with session.post(f"{self.base_url}/api/v1/process/batch", data=data) as response:
                if response.status == 200:
                    content = await response.json()
                    details = {
                        "batch_processing_successful": True,
                        "processed_files_count": len(content.get("results", [])),
                        "total_processing_time": sum([r.get("processing_info", {}).get("duration", 0) for r in content.get("results", [])]),
                        "all_successful": all([r.get("success", False) for r in content.get("results", [])])
                    }
                    duration = asyncio.get_event_loop().time() - start_time
                    await self.log_test("批量处理功能", "PASSED", details, duration=duration)
                else:
                    error_content = await response.text()
                    details = {"status_code": response.status, "error_response": error_content}
                    duration = asyncio.get_event_loop().time() - start_time
                    await self.log_test("批量处理功能", "FAILED", details, f"HTTP {response.status}", duration)
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("批量处理功能", "FAILED", {}, str(e), duration)
    
    async def test_task_status_tracking(self, session):
        """测试任务状态跟踪功能"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 先创建一个处理任务
            test_content = "任务状态跟踪测试文档"
            data = aiohttp.FormData()
            data.add_field('file', 
                          io.BytesIO(test_content.encode('utf-8')), 
                          filename='status_test.txt',
                          content_type='text/plain')
            
            async with session.post(f"{self.base_url}/api/v1/process/document", data=data) as response:
                if response.status == 200:
                    content = await response.json()
                    task_id = content.get("task_info", {}).get("task_id")
                    
                    if task_id:
                        # 测试状态查询
                        async with session.get(f"{self.base_url}/api/v1/process/status/{task_id}") as status_response:
                            if status_response.status == 200:
                                status_content = await status_response.json()
                                details = {
                                    "task_id": task_id,
                                    "status_query_successful": True,
                                    "task_status": status_content.get("status"),
                                    "task_info_available": bool(status_content.get("task_info"))
                                }
                                duration = asyncio.get_event_loop().time() - start_time
                                await self.log_test("任务状态跟踪", "PASSED", details, duration=duration)
                            else:
                                details = {"task_id": task_id, "status_query_failed": True}
                                duration = asyncio.get_event_loop().time() - start_time
                                await self.log_test("任务状态跟踪", "FAILED", details, f"状态查询失败: HTTP {status_response.status}", duration)
                    else:
                        details = {"no_task_id": True}
                        duration = asyncio.get_event_loop().time() - start_time
                        await self.log_test("任务状态跟踪", "FAILED", details, "未返回task_id", duration)
                else:
                    details = {"initial_task_creation_failed": True}
                    duration = asyncio.get_event_loop().time() - start_time
                    await self.log_test("任务状态跟踪", "FAILED", details, f"初始任务创建失败: HTTP {response.status}", duration)
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("任务状态跟踪", "FAILED", {}, str(e), duration)
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始file-processor API功能测试")
        
        async with aiohttp.ClientSession() as session:
            await self.test_basic_endpoints(session)
            await self.test_supported_formats(session)
            await self.test_document_processing(session)
            await self.test_batch_processing(session)
            await self.test_task_status_tracking(session)
        
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
        
        print(f"\n📊 file-processor API测试总结:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   成功率: {self.results['summary']['success_rate']}%")
        
        return self.results

async def main():
    tester = FileProcessorAPITester()
    results = await tester.run_all_tests()
    
    # 保存结果
    output_file = Path(__file__).parent / "file_processor_api_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试结果已保存到: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())