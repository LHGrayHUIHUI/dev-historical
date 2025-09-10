"""
SS-INT-004: MinIO文件存储集成测试
优先级: P0 - 文件管理系统
"""

import asyncio
import json
import aiohttp
import io
from datetime import datetime
from typing import Dict, Any


class MinIOIntegrationTester:
    """MinIO对象存储集成测试器"""
    
    def __init__(self):
        self.minio_url = "http://localhost:9001"
        self.storage_service_url = "http://localhost:8002"
        self.test_results = []
        self.uploaded_files = []
        
    async def log_test(self, name: str, status: str, details: Dict = None, error: str = None):
        """记录测试结果"""
        result = {
            "test_name": name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "error": error
        }
        self.test_results.append(result)
        
        print(f"{'✅' if status == 'PASSED' else '❌'} {name}: {status}")
        if error:
            print(f"   错误: {error}")
        if details and status == "PASSED":
            print(f"   详情: {details}")
    
    async def test_minio_accessibility(self, session):
        """测试MinIO服务可访问性
        
        测试场景: SS-INT-004-001
        验证点: MinIO服务运行状态和基本连接
        """
        try:
            # 测试MinIO管理界面访问
            async with session.get(self.minio_url) as response:
                minio_accessible = response.status in [200, 403]  # 403也表示服务运行
                
                details = {
                    "minio_service_running": minio_accessible,
                    "response_status": response.status,
                    "service_url": self.minio_url
                }
                
                if minio_accessible:
                    await self.log_test("MinIO服务可访问性", "PASSED", details)
                    return True
                else:
                    await self.log_test("MinIO服务可访问性", "FAILED", 
                                      details, f"MinIO服务不可访问: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            await self.log_test("MinIO服务可访问性", "FAILED", error=str(e))
            return False
    
    async def test_file_upload_via_storage_service(self, session):
        """通过storage-service测试文件上传
        
        测试场景: SS-INT-004-002
        验证点: 文件上传到MinIO的完整流程
        """
        try:
            # 创建测试文件内容
            test_content = """这是MinIO集成测试文件
包含中文内容和特殊字符: @#$%^&*()
测试文件上传功能
创建时间: {}""".format(datetime.now().isoformat())
            
            # 准备文件数据
            data = aiohttp.FormData()
            data.add_field('file',
                          io.BytesIO(test_content.encode('utf-8')),
                          filename='minio_test_file.txt',
                          content_type='text/plain')
            data.add_field('source', 'integration_test')
            data.add_field('metadata', json.dumps({
                "test_type": "minio_integration",
                "description": "MinIO集成测试文件",
                "file_size": len(test_content.encode('utf-8'))
            }))
            
            # 上传文件
            async with session.post(f"{self.storage_service_url}/api/v1/data/upload", 
                                   data=data) as response:
                if response.status in [200, 201]:
                    upload_result = await response.json()
                    
                    file_id = upload_result.get("file_id") or upload_result.get("id")
                    if file_id:
                        self.uploaded_files.append(file_id)
                        
                        details = {
                            "upload_successful": True,
                            "file_id": file_id,
                            "file_size": len(test_content.encode('utf-8')),
                            "filename": "minio_test_file.txt",
                            "storage_location": upload_result.get("storage_path", "unknown")
                        }
                        
                        await self.log_test("文件上传到MinIO", "PASSED", details)
                        return file_id
                    else:
                        await self.log_test("文件上传到MinIO", "FAILED", 
                                          error="上传成功但未返回文件ID")
                        return None
                else:
                    error_content = await response.text()
                    await self.log_test("文件上传到MinIO", "FAILED", 
                                      error=f"上传失败: HTTP {response.status}, {error_content}")
                    return None
                    
        except Exception as e:
            await self.log_test("文件上传到MinIO", "FAILED", error=str(e))
            return None
    
    async def test_file_download_via_storage_service(self, session, file_id):
        """通过storage-service测试文件下载
        
        测试场景: SS-INT-004-003
        验证点: 从MinIO下载文件的完整性
        """
        try:
            if not file_id:
                await self.log_test("文件从MinIO下载", "SKIPPED", error="没有可用的文件ID")
                return False
            
            # 尝试下载文件
            async with session.get(f"{self.storage_service_url}/api/v1/data/files/{file_id}") as response:
                if response.status == 200:
                    downloaded_content = await response.text()
                    
                    # 验证下载内容
                    content_valid = "MinIO集成测试文件" in downloaded_content
                    encoding_valid = "中文内容" in downloaded_content
                    
                    details = {
                        "download_successful": True,
                        "file_id": file_id,
                        "content_size": len(downloaded_content),
                        "content_integrity": content_valid,
                        "encoding_preserved": encoding_valid,
                        "content_type": response.headers.get("content-type", "unknown")
                    }
                    
                    if content_valid and encoding_valid:
                        await self.log_test("文件从MinIO下载", "PASSED", details)
                        return True
                    else:
                        await self.log_test("文件从MinIO下载", "FAILED", 
                                          details, "文件内容完整性检查失败")
                        return False
                        
                else:
                    await self.log_test("文件从MinIO下载", "FAILED", 
                                      error=f"下载失败: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            await self.log_test("文件从MinIO下载", "FAILED", error=str(e))
            return False
    
    async def test_multiple_file_operations(self, session):
        """测试多文件操作
        
        测试场景: SS-INT-004-004
        验证点: 并发文件操作和批量处理
        """
        try:
            # 创建多个测试文件
            num_files = 5
            upload_results = []
            
            for i in range(num_files):
                file_content = f"""批量测试文件 {i+1}
文件索引: {i+1}
内容大小测试: {'X' * (100 * (i+1))}
创建时间: {datetime.now().isoformat()}"""
                
                data = aiohttp.FormData()
                data.add_field('file',
                              io.BytesIO(file_content.encode('utf-8')),
                              filename=f'batch_test_{i+1}.txt',
                              content_type='text/plain')
                data.add_field('source', 'batch_integration_test')
                data.add_field('metadata', json.dumps({
                    "batch_index": i+1,
                    "batch_total": num_files
                }))
                
                async with session.post(f"{self.storage_service_url}/api/v1/data/upload", 
                                       data=data) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        file_id = result.get("file_id") or result.get("id")
                        if file_id:
                            upload_results.append(file_id)
                            self.uploaded_files.append(file_id)
            
            successful_uploads = len(upload_results)
            
            details = {
                "target_files": num_files,
                "successful_uploads": successful_uploads,
                "upload_success_rate": round((successful_uploads / num_files) * 100, 2),
                "uploaded_file_ids": upload_results
            }
            
            if successful_uploads == num_files:
                await self.log_test("多文件操作", "PASSED", details)
                return True
            else:
                await self.log_test("多文件操作", "FAILED", 
                                  details, 
                                  f"批量上传不完整: {successful_uploads}/{num_files}")
                return False
                
        except Exception as e:
            await self.log_test("多文件操作", "FAILED", error=str(e))
            return False
    
    async def test_storage_quota_and_limits(self, session):
        """测试存储配额和限制
        
        测试场景: SS-INT-004-005
        验证点: 文件大小限制和存储配额管理
        """
        try:
            # 创建一个相对较大的测试文件 (1MB)
            large_content = "X" * (1024 * 1024)  # 1MB
            
            data = aiohttp.FormData()
            data.add_field('file',
                          io.BytesIO(large_content.encode('utf-8')),
                          filename='large_file_test.txt',
                          content_type='text/plain')
            data.add_field('source', 'storage_limit_test')
            data.add_field('metadata', json.dumps({
                "file_size_mb": 1,
                "test_purpose": "storage_limit_testing"
            }))
            
            async with session.post(f"{self.storage_service_url}/api/v1/data/upload", 
                                   data=data) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    file_id = result.get("file_id") or result.get("id")
                    if file_id:
                        self.uploaded_files.append(file_id)
                        
                        details = {
                            "large_file_upload": "success",
                            "file_size_mb": 1,
                            "file_id": file_id,
                            "storage_system_handling": "acceptable"
                        }
                        
                        await self.log_test("存储配额和限制", "PASSED", details)
                        return True
                elif response.status == 413:  # Payload Too Large
                    details = {
                        "large_file_upload": "rejected",
                        "file_size_mb": 1,
                        "rejection_reason": "file_too_large",
                        "limit_enforcement": "working"
                    }
                    
                    await self.log_test("存储配额和限制", "PASSED", details)
                    return True
                else:
                    await self.log_test("存储配额和限制", "FAILED", 
                                      error=f"意外响应: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            await self.log_test("存储配额和限制", "FAILED", error=str(e))
            return False
    
    async def cleanup_test_files(self, session):
        """清理测试文件"""
        cleanup_count = 0
        for file_id in self.uploaded_files:
            try:
                async with session.delete(f"{self.storage_service_url}/api/v1/data/files/{file_id}") as response:
                    if response.status in [200, 204, 404]:  # 404也认为是成功（文件已不存在）
                        cleanup_count += 1
            except:
                pass  # 忽略清理错误
        
        print(f"🧹 清理了 {cleanup_count}/{len(self.uploaded_files)} 个测试文件")
    
    async def run_all_tests(self):
        """运行所有MinIO集成测试"""
        print("🗂️ 开始执行MinIO存储集成测试...")
        
        async with aiohttp.ClientSession() as session:
            await self.test_minio_accessibility(session)
            
            # 上传测试
            file_id = await self.test_file_upload_via_storage_service(session)
            
            # 下载测试
            await self.test_file_download_via_storage_service(session, file_id)
            
            # 批量操作测试
            await self.test_multiple_file_operations(session)
            
            # 存储限制测试
            await self.test_storage_quota_and_limits(session)
            
            # 清理测试文件
            await self.cleanup_test_files(session)
        
        # 生成测试摘要
        passed_tests = len([t for t in self.test_results if t["status"] == "PASSED"])
        total_tests = len([t for t in self.test_results if t["status"] != "SKIPPED"])
        skipped_tests = len([t for t in self.test_results if t["status"] == "SKIPPED"])
        
        print(f"\n📊 MinIO存储集成测试摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {total_tests - passed_tests}")
        print(f"   跳过: {skipped_tests}")
        print(f"   成功率: {round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0}%")
        
        return self.test_results


async def main():
    tester = MinIOIntegrationTester()
    results = await tester.run_all_tests()
    
    # 保存测试结果
    with open("/Users/yjlh/Documents/code/Historical Text Project/test-results/minio_integration_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_type": "minio_integration",
            "execution_time": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())