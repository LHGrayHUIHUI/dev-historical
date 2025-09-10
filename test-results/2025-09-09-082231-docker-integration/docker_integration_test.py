#!/usr/bin/env python3
"""
Docker服务集成测试
测试file-processor、storage-service和intelligent-classification-service的集成功能
"""

import asyncio
import httpx
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

class DockerIntegrationTest:
    """Docker服务集成测试类"""
    
    def __init__(self):
        self.file_processor_url = "http://localhost:8001"
        self.storage_service_url = "http://localhost:8002" 
        self.intelligent_classification_url = "http://localhost:8007"
        
        self.timeout = httpx.Timeout(30.0)
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "test_type": "docker_integration",
            "services_tested": [
                "file-processor:8001",
                "storage-service:8002", 
                "intelligent-classification-service:8007"
            ],
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
    
    async def test_service_health(self, service_name: str, url: str) -> bool:
        """测试服务健康检查"""
        print(f"🏥 测试{service_name}健康状态...")
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{url}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    duration = time.time() - start_time
                    
                    print(f"  ✅ {service_name}健康检查通过")
                    print(f"  📊 响应时间: {duration:.2f}秒")
                    if isinstance(health_data, dict):
                        print(f"  📝 健康状态: {health_data.get('status', 'unknown')}")
                    
                    self.add_test_result(
                        f"{service_name}_health_check", 
                        True, 
                        duration, 
                        {"health_data": health_data, "response_time": duration}
                    )
                    return True
                else:
                    duration = time.time() - start_time
                    error_msg = f"{service_name}健康检查失败，状态码: {response.status_code}"
                    print(f"  ❌ {error_msg}")
                    
                    self.add_test_result(
                        f"{service_name}_health_check", 
                        False, 
                        duration, 
                        error=error_msg
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{service_name}健康检查异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                f"{service_name}_health_check", 
                False, 
                duration, 
                error=error_msg
            )
            return False
    
    async def test_file_processor_service(self) -> bool:
        """测试文件处理服务功能"""
        print("📁 测试文件处理服务...")
        start_time = time.time()
        
        try:
            # 测试文件上传和处理
            test_content = "这是一个测试文档内容，用于验证文件处理服务的功能。"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 创建测试文件
                files = {"file": ("test.txt", test_content, "text/plain")}
                data = {
                    "extract_text": "true",
                    "validate_format": "true"
                }
                
                response = await client.post(
                    f"{self.file_processor_url}/api/v1/files/process",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    duration = time.time() - start_time
                    
                    print(f"  ✅ 文件处理成功")
                    print(f"  📄 提取文本长度: {len(result.get('extracted_text', ''))}")
                    print(f"  📊 处理时间: {duration:.2f}秒")
                    
                    self.add_test_result(
                        "file_processor_upload_and_process", 
                        True, 
                        duration, 
                        {
                            "extracted_text_length": len(result.get('extracted_text', '')),
                            "file_info": result.get('file_info', {}),
                            "processing_time": duration
                        }
                    )
                    return True
                else:
                    duration = time.time() - start_time
                    error_msg = f"文件处理失败，状态码: {response.status_code}"
                    print(f"  ❌ {error_msg}")
                    
                    self.add_test_result(
                        "file_processor_upload_and_process", 
                        False, 
                        duration, 
                        error=error_msg
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"文件处理服务测试异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                "file_processor_upload_and_process", 
                False, 
                duration, 
                error=error_msg
            )
            return False
    
    async def test_storage_service(self) -> bool:
        """测试存储服务功能"""
        print("💾 测试存储服务...")
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 测试内容创建
                test_data = {
                    "title": "集成测试文档",
                    "content": "这是一个集成测试创建的文档内容。",
                    "content_type": "text",
                    "tags": ["集成测试", "Docker"],
                    "metadata": {
                        "test_type": "docker_integration",
                        "created_by": "integration_test"
                    }
                }
                
                response = await client.post(
                    f"{self.storage_service_url}/api/v1/contents/",
                    json=test_data
                )
                
                if response.status_code == 201:
                    content_result = response.json()
                    content_id = content_result.get('data', {}).get('id')
                    
                    if content_id:
                        # 测试内容检索
                        get_response = await client.get(
                            f"{self.storage_service_url}/api/v1/contents/{content_id}"
                        )
                        
                        if get_response.status_code == 200:
                            duration = time.time() - start_time
                            retrieved_data = get_response.json()
                            
                            print(f"  ✅ 存储服务测试成功")
                            print(f"  📄 内容ID: {content_id}")
                            print(f"  📊 处理时间: {duration:.2f}秒")
                            
                            self.add_test_result(
                                "storage_service_create_and_retrieve", 
                                True, 
                                duration, 
                                {
                                    "content_id": content_id,
                                    "created_content": content_result,
                                    "retrieved_content": retrieved_data,
                                    "processing_time": duration
                                }
                            )
                            return True
                    
                duration = time.time() - start_time
                error_msg = f"存储服务测试失败，无法获取内容ID"
                print(f"  ❌ {error_msg}")
                
                self.add_test_result(
                    "storage_service_create_and_retrieve", 
                    False, 
                    duration, 
                    error=error_msg
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"存储服务测试异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                "storage_service_create_and_retrieve", 
                False, 
                duration, 
                error=error_msg
            )
            return False
    
    async def test_intelligent_classification_service(self) -> bool:
        """测试智能分类服务功能（如果可用）"""
        print("🤖 测试智能分类服务...")
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 首先检查服务是否可用
                health_response = await client.get(f"{self.intelligent_classification_url}/health")
                
                if health_response.status_code == 200:
                    # 测试简单分类请求
                    test_data = {
                        "project_id": "integration-test-001",
                        "text_content": "汉武帝时期的政治改革对后世产生了深远影响。",
                        "return_probabilities": True
                    }
                    
                    response = await client.post(
                        f"{self.intelligent_classification_url}/api/v1/classify/",
                        json=test_data
                    )
                    
                    duration = time.time() - start_time
                    
                    if response.status_code in [200, 201]:
                        result = response.json()
                        print(f"  ✅ 智能分类服务测试成功")
                        print(f"  📊 处理时间: {duration:.2f}秒")
                        print(f"  📝 分类结果: {result.get('data', {})}")
                        
                        self.add_test_result(
                            "intelligent_classification_service_classify", 
                            True, 
                            duration, 
                            {
                                "classification_result": result,
                                "processing_time": duration
                            }
                        )
                        return True
                    else:
                        error_msg = f"分类请求失败，状态码: {response.status_code}"
                        print(f"  ❌ {error_msg}")
                        
                        self.add_test_result(
                            "intelligent_classification_service_classify", 
                            False, 
                            duration, 
                            error=error_msg
                        )
                        return False
                else:
                    duration = time.time() - start_time
                    error_msg = f"智能分类服务不可用，状态码: {health_response.status_code}"
                    print(f"  ⚠️  {error_msg}")
                    
                    self.add_test_result(
                        "intelligent_classification_service_availability", 
                        False, 
                        duration, 
                        error=error_msg
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"智能分类服务测试异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                "intelligent_classification_service_test", 
                False, 
                duration, 
                error=error_msg
            )
            return False
    
    async def test_service_integration(self) -> bool:
        """测试服务间集成功能"""
        print("🔗 测试服务间集成...")
        start_time = time.time()
        
        try:
            # 端到端流程测试
            test_content = "唐朝是中国历史上一个繁荣的朝代，诗歌文化达到巅峰。"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. 使用文件处理服务处理内容
                files = {"file": ("integration_test.txt", test_content, "text/plain")}
                data = {"extract_text": "true"}
                
                file_response = await client.post(
                    f"{self.file_processor_url}/api/v1/files/process",
                    files=files,
                    data=data
                )
                
                if file_response.status_code == 200:
                    file_result = file_response.json()
                    extracted_text = file_result.get('extracted_text', test_content)
                    
                    # 2. 将处理结果存储到存储服务
                    storage_data = {
                        "title": "集成测试-端到端",
                        "content": extracted_text,
                        "content_type": "text",
                        "tags": ["集成测试", "端到端"],
                        "metadata": {
                            "source": "file_processor_integration",
                            "original_filename": "integration_test.txt"
                        }
                    }
                    
                    storage_response = await client.post(
                        f"{self.storage_service_url}/api/v1/contents/",
                        json=storage_data
                    )
                    
                    if storage_response.status_code == 201:
                        storage_result = storage_response.json()
                        duration = time.time() - start_time
                        
                        print(f"  ✅ 服务集成测试成功")
                        print(f"  📄 处理链: 文件处理 → 存储服务")
                        print(f"  📊 总处理时间: {duration:.2f}秒")
                        
                        self.add_test_result(
                            "service_integration_end_to_end", 
                            True, 
                            duration, 
                            {
                                "file_processing_result": file_result,
                                "storage_result": storage_result,
                                "total_processing_time": duration,
                                "pipeline": "file_processor -> storage_service"
                            }
                        )
                        return True
                
                duration = time.time() - start_time
                error_msg = "服务集成测试失败，流程中断"
                print(f"  ❌ {error_msg}")
                
                self.add_test_result(
                    "service_integration_end_to_end", 
                    False, 
                    duration, 
                    error=error_msg
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"服务集成测试异常: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            self.add_test_result(
                "service_integration_end_to_end", 
                False, 
                duration, 
                error=error_msg
            )
            return False
    
    async def run_all_tests(self):
        """运行所有集成测试"""
        print("🚀 开始Docker服务集成测试...")
        print("=" * 60)
        
        # 服务健康检查
        await self.test_service_health("文件处理服务", self.file_processor_url)
        await self.test_service_health("存储服务", self.storage_service_url)
        await self.test_service_health("智能分类服务", self.intelligent_classification_url)
        
        print()
        
        # 个别服务功能测试
        await self.test_file_processor_service()
        await self.test_storage_service()
        await self.test_intelligent_classification_service()
        
        print()
        
        # 服务集成测试
        await self.test_service_integration()
        
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
    print("🧪 Docker服务集成测试")
    print("=" * 60)
    print("📝 说明: 测试file-processor、storage-service、intelligent-classification-service的集成功能")
    print()
    
    # 创建测试实例
    tester = DockerIntegrationTest()
    
    # 运行测试
    results = await tester.run_all_tests()
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("📊 Docker集成测试结果摘要")
    print("=" * 60)
    print(f"🎯 总测试数: {results['summary']['total']}")
    print(f"✅ 通过: {results['summary']['passed']}")
    print(f"❌ 失败: {results['summary']['failed']}")
    print(f"📈 成功率: {results['summary']['success_rate']:.1f}%")
    print(f"⏱️  总耗时: {results['total_duration']:.2f}秒")
    
    if results['summary']['errors']:
        print(f"\n❗ 错误列表:")
        for i, error in enumerate(results['summary']['errors'], 1):
            print(f"  {i}. {error}")
    
    # 保存测试结果
    result_file = "docker_integration_test_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细测试结果已保存到: {result_file}")
    print("🏁 Docker集成测试完成")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())