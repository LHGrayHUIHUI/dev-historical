"""
FP-INT-001: 文档处理完整流程集成测试
优先级: P0 - 多组件协作
"""

import asyncio
import json
import aiohttp
import io
from datetime import datetime
from typing import Dict, List, Any


class DocumentProcessingFlowTester:
    """文档处理流程集成测试器"""
    
    def __init__(self):
        self.file_processor_url = "http://localhost:8001"
        self.storage_service_url = "http://localhost:8002"
        self.test_results = []
        
    async def log_test(self, name: str, status: str, details: Dict = None, error: str = None, duration: float = 0):
        """记录测试结果"""
        result = {
            "test_name": name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "details": details or {},
            "error": error
        }
        self.test_results.append(result)
        
        print(f"{'✅' if status == 'PASSED' else '❌'} {name}: {status}")
        if error:
            print(f"   错误: {error}")
        if details and status == "PASSED":
            print(f"   详情: {details}")
    
    async def test_file_processor_health(self, session):
        """测试file-processor健康状态
        
        测试场景: FP-INT-001-001
        验证点: file-processor服务可用性
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with session.get(f"{self.file_processor_url}/health") as response:
                duration = asyncio.get_event_loop().time() - start_time
                
                if response.status == 200:
                    health_data = await response.json()
                    
                    details = {
                        "service_healthy": True,
                        "response_time_ms": round(duration * 1000, 2),
                        "service_info": health_data.get("data", {}),
                        "processors_ready": health_data.get("data", {}).get("components", {}).get("processors", {}).get("status") == "ready"
                    }
                    
                    await self.log_test("file-processor健康检查", "PASSED", details, duration=duration)
                    return True
                else:
                    await self.log_test("file-processor健康检查", "FAILED", 
                                      error=f"健康检查失败: HTTP {response.status}",
                                      duration=duration)
                    return False
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("file-processor健康检查", "FAILED", error=str(e), duration=duration)
            return False
    
    async def test_supported_formats_query(self, session):
        """测试支持格式查询
        
        测试场景: FP-INT-001-002
        验证点: 支持的文件格式获取
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with session.get(f"{self.file_processor_url}/api/v1/process/supported-formats") as response:
                duration = asyncio.get_event_loop().time() - start_time
                
                if response.status == 200:
                    formats_data = await response.json()
                    
                    details = {
                        "formats_query_successful": True,
                        "response_time_ms": round(duration * 1000, 2),
                        "supported_formats": formats_data.get("supported_formats", []),
                        "format_count": len(formats_data.get("supported_formats", [])) if isinstance(formats_data, dict) else 0
                    }
                    
                    await self.log_test("支持格式查询", "PASSED", details, duration=duration)
                    return formats_data.get("supported_formats", [])
                else:
                    await self.log_test("支持格式查询", "FAILED", 
                                      error=f"格式查询失败: HTTP {response.status}",
                                      duration=duration)
                    return []
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("支持格式查询", "FAILED", error=str(e), duration=duration)
            return []
    
    async def test_document_processing(self, session):
        """测试单个文档处理
        
        测试场景: FP-INT-001-003
        验证点: 完整的文档处理流程
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 创建测试文档
            test_content = """历史文本处理集成测试文档

这是一个用于验证文档处理完整流程的测试文档。

内容包括：
1. 中文文本内容
2. 特殊字符：@#$%^&*()
3. 数字和英文：123 ABC test
4. 时间戳：{}

文档目的：验证file-processor的文档处理能力
测试类型：集成测试""".format(datetime.now().isoformat())
            
            # 准备文件数据
            data = aiohttp.FormData()
            data.add_field('file',
                          io.BytesIO(test_content.encode('utf-8')),
                          filename='integration_test_document.txt',
                          content_type='text/plain')
            
            # 发送处理请求
            async with session.post(f"{self.file_processor_url}/api/v1/process/document", data=data) as response:
                duration = asyncio.get_event_loop().time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    # 验证处理结果
                    success = result.get("success", False)
                    extracted_text = result.get("extracted_text", "")
                    processing_info = result.get("processing_info", {})
                    file_info = result.get("file_info", {})
                    
                    # 检查文本提取完整性
                    text_integrity = all([
                        "历史文本处理" in extracted_text,
                        "集成测试文档" in extracted_text,
                        "中文文本内容" in extracted_text,
                        "@#$%^&*()" in extracted_text
                    ])
                    
                    details = {
                        "processing_successful": success,
                        "response_time_ms": round(duration * 1000, 2),
                        "extracted_text_length": len(extracted_text),
                        "text_integrity_check": text_integrity,
                        "file_info": file_info,
                        "processing_duration": processing_info.get("duration", 0),
                        "detected_encoding": processing_info.get("encoding", "unknown")
                    }
                    
                    if success and text_integrity:
                        await self.log_test("文档处理功能", "PASSED", details, duration=duration)
                        return result
                    else:
                        await self.log_test("文档处理功能", "FAILED", 
                                          details, 
                                          error=f"处理不完整: success={success}, integrity={text_integrity}",
                                          duration=duration)
                        return None
                else:
                    error_content = await response.text()
                    await self.log_test("文档处理功能", "FAILED", 
                                      error=f"处理请求失败: HTTP {response.status}, {error_content}",
                                      duration=duration)
                    return None
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("文档处理功能", "FAILED", error=str(e), duration=duration)
            return None
    
    async def test_batch_document_processing(self, session):
        """测试批量文档处理
        
        测试场景: FP-INT-001-004
        验证点: 批量文档处理能力
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 创建多个测试文件
            files_data = []
            for i in range(3):
                content = f"""批量处理测试文档 {i+1}

文档编号：{i+1}
内容：这是第{i+1}个测试文档的内容
特殊内容：{'测试' * (i+1)}
创建时间：{datetime.now().isoformat()}"""
                
                files_data.append(('files', (f'batch_doc_{i+1}.txt', content.encode('utf-8'), 'text/plain')))
            
            data = aiohttp.FormData()
            for field_name, (filename, content, content_type) in files_data:
                data.add_field(field_name, io.BytesIO(content), filename=filename, content_type=content_type)
            
            # 发送批量处理请求
            async with session.post(f"{self.file_processor_url}/api/v1/process/batch", data=data) as response:
                duration = asyncio.get_event_loop().time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    # 验证批量处理结果
                    success = result.get("success", False)
                    results = result.get("results", [])
                    
                    successful_files = len([r for r in results if r.get("success", False)])
                    total_files = len(files_data)
                    
                    details = {
                        "batch_processing_successful": success,
                        "response_time_ms": round(duration * 1000, 2),
                        "total_files": total_files,
                        "successful_files": successful_files,
                        "success_rate": round((successful_files / total_files) * 100, 2) if total_files > 0 else 0,
                        "processing_details": [
                            {
                                "filename": r.get("file_info", {}).get("filename"),
                                "success": r.get("success", False),
                                "text_length": len(r.get("extracted_text", ""))
                            } for r in results
                        ]
                    }
                    
                    if success and successful_files == total_files:
                        await self.log_test("批量文档处理", "PASSED", details, duration=duration)
                        return result
                    else:
                        await self.log_test("批量文档处理", "FAILED", 
                                          details,
                                          error=f"批量处理不完整: {successful_files}/{total_files}成功",
                                          duration=duration)
                        return None
                else:
                    error_content = await response.text()
                    await self.log_test("批量文档处理", "FAILED", 
                                      error=f"批量处理请求失败: HTTP {response.status}, {error_content}",
                                      duration=duration)
                    return None
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("批量文档处理", "FAILED", error=str(e), duration=duration)
            return None
    
    async def test_error_handling(self, session):
        """测试错误处理能力
        
        测试场景: FP-INT-001-005
        验证点: 异常情况和错误处理
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 测试无效文件处理
            invalid_data = aiohttp.FormData()
            invalid_data.add_field('file',
                                 io.BytesIO(b'\x00\x01\x02\x03\x04\x05'),  # 无效二进制数据
                                 filename='invalid_file.bin',
                                 content_type='application/octet-stream')
            
            async with session.post(f"{self.file_processor_url}/api/v1/process/document", data=invalid_data) as response:
                duration = asyncio.get_event_loop().time() - start_time
                
                if response.status in [400, 415, 422]:  # 预期的错误状态码
                    error_result = await response.json()
                    
                    details = {
                        "error_handling_working": True,
                        "response_time_ms": round(duration * 1000, 2),
                        "error_status_code": response.status,
                        "error_response": error_result,
                        "graceful_error_handling": "error" in error_result or "message" in error_result
                    }
                    
                    await self.log_test("错误处理能力", "PASSED", details, duration=duration)
                    return True
                    
                elif response.status == 200:
                    # 如果返回200，检查是否正确标识为错误
                    result = await response.json()
                    success = result.get("success", True)
                    
                    if not success:
                        details = {
                            "error_handling_working": True,
                            "response_time_ms": round(duration * 1000, 2),
                            "error_in_response": True,
                            "error_details": result.get("error", "")
                        }
                        await self.log_test("错误处理能力", "PASSED", details, duration=duration)
                        return True
                    else:
                        await self.log_test("错误处理能力", "FAILED", 
                                          error="无效文件被错误处理为成功",
                                          duration=duration)
                        return False
                else:
                    await self.log_test("错误处理能力", "FAILED", 
                                      error=f"意外的响应状态码: {response.status}",
                                      duration=duration)
                    return False
                    
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("错误处理能力", "FAILED", error=str(e), duration=duration)
            return False
    
    async def test_performance_under_load(self, session):
        """测试负载下的性能表现
        
        测试场景: FP-INT-001-006
        验证点: 并发处理能力和性能稳定性
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 创建并发处理任务
            concurrent_requests = 5
            tasks = []
            
            for i in range(concurrent_requests):
                content = f"并发测试文档 {i+1}\n内容长度测试：{'并发处理' * 20}\n时间戳：{datetime.now().isoformat()}"
                
                async def process_single_doc(doc_content, doc_index):
                    data = aiohttp.FormData()
                    data.add_field('file',
                                  io.BytesIO(doc_content.encode('utf-8')),
                                  filename=f'concurrent_test_{doc_index}.txt',
                                  content_type='text/plain')
                    
                    async with session.post(f"{self.file_processor_url}/api/v1/process/document", data=data) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return {
                                "index": doc_index,
                                "success": result.get("success", False),
                                "processing_time": result.get("processing_info", {}).get("duration", 0),
                                "text_length": len(result.get("extracted_text", ""))
                            }
                        else:
                            return {"index": doc_index, "success": False, "error": resp.status}
                
                task = process_single_doc(content, i+1)
                tasks.append(task)
            
            # 并发执行所有任务
            concurrent_results = await asyncio.gather(*tasks)
            duration = asyncio.get_event_loop().time() - start_time
            
            successful_requests = len([r for r in concurrent_results if r.get("success", False)])
            average_processing_time = sum([r.get("processing_time", 0) for r in concurrent_results if r.get("processing_time")]) / max(len(concurrent_results), 1)
            
            details = {
                "concurrent_requests": concurrent_requests,
                "successful_requests": successful_requests,
                "total_time_seconds": round(duration, 3),
                "average_processing_time": round(average_processing_time, 3),
                "requests_per_second": round(concurrent_requests / duration, 2),
                "success_rate": round((successful_requests / concurrent_requests) * 100, 2),
                "performance_acceptable": duration < 30 and successful_requests >= concurrent_requests * 0.8  # 80%成功率
            }
            
            if successful_requests >= concurrent_requests * 0.8 and duration < 30:
                await self.log_test("负载性能测试", "PASSED", details, duration=duration)
                return True
            else:
                await self.log_test("负载性能测试", "FAILED", 
                                  details,
                                  error=f"性能不达标: {successful_requests}/{concurrent_requests}成功, 用时{duration:.2f}秒",
                                  duration=duration)
                return False
                
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.log_test("负载性能测试", "FAILED", error=str(e), duration=duration)
            return False
    
    async def run_all_tests(self):
        """运行所有文档处理流程集成测试"""
        print("📄 开始执行文档处理流程集成测试...")
        
        async with aiohttp.ClientSession() as session:
            # 依次执行测试
            await self.test_file_processor_health(session)
            await self.test_supported_formats_query(session)
            await self.test_document_processing(session)
            await self.test_batch_document_processing(session)
            await self.test_error_handling(session)
            await self.test_performance_under_load(session)
        
        # 生成测试摘要
        passed_tests = len([t for t in self.test_results if t["status"] == "PASSED"])
        total_tests = len(self.test_results)
        total_duration = sum([t.get("duration", 0) for t in self.test_results])
        
        print(f"\n📊 文档处理流程集成测试摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {total_tests - passed_tests}")
        print(f"   成功率: {round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0}%")
        print(f"   总执行时间: {round(total_duration, 3)}秒")
        
        return self.test_results


async def main():
    tester = DocumentProcessingFlowTester()
    results = await tester.run_all_tests()
    
    # 保存测试结果
    with open("/Users/yjlh/Documents/code/Historical Text Project/test-results/document_processing_flow_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_type": "document_processing_flow_integration",
            "execution_time": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())