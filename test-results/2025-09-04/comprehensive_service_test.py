#!/usr/bin/env python3
"""
历史文本项目数据源服务 - 全面功能测试脚本
测试架构重构后的所有功能
"""

import os
import sys
import json
import time
import asyncio
import requests
from datetime import datetime
from pathlib import Path

# 添加服务模块路径
service_path = Path(__file__).parent.parent.parent / "services" / "data-source"
sys.path.insert(0, str(service_path))

class ServiceTester:
    """服务测试器"""
    
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "summary": {"total": 0, "passed": 0, "failed": 0}
        }
        self.session = requests.Session()
        
    def log_test(self, test_name, status, details="", error=None):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "status": status,
            "details": details,
            "error": str(error) if error else None,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results["tests"].append(result)
        self.test_results["summary"]["total"] += 1
        
        if status == "PASS":
            self.test_results["summary"]["passed"] += 1
            print(f"✅ {test_name} - {details}")
        else:
            self.test_results["summary"]["failed"] += 1
            print(f"❌ {test_name} - {error}")
    
    def test_health_check(self):
        """测试健康检查接口"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("健康检查", "PASS", f"响应时间: {response.elapsed.total_seconds():.3f}s")
                return True
            else:
                self.log_test("健康检查", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("健康检查", "FAIL", error=e)
            return False
    
    def test_service_info(self):
        """测试服务信息接口"""
        try:
            response = self.session.get(f"{self.base_url}/info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                version = data.get("version", "unknown")
                self.log_test("服务信息", "PASS", f"服务版本: {version}")
                return True
            else:
                self.log_test("服务信息", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("服务信息", "FAIL", error=e)
            return False
    
    def test_api_documentation(self):
        """测试API文档访问"""
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                self.log_test("API文档", "PASS", "Swagger文档可访问")
                return True
            else:
                self.log_test("API文档", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API文档", "FAIL", error=e)
            return False
    
    def test_content_creation(self):
        """测试内容创建功能"""
        try:
            test_content = {
                "title": "测试历史文档",
                "content": "这是一份测试的历史文档内容，用于验证内容管理系统的基本功能。",
                "source": "manual",
                "author": "测试者",
                "keywords": ["测试", "历史"],
                "category": "测试分类"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/content/",
                json=test_content,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    content_id = data["data"].get("id")
                    self.log_test("内容创建", "PASS", f"创建内容ID: {content_id}")
                    return content_id
                else:
                    self.log_test("内容创建", "FAIL", error=data.get("error", "未知错误"))
                    return None
            else:
                self.log_test("内容创建", "FAIL", error=f"HTTP {response.status_code}")
                return None
        except Exception as e:
            self.log_test("内容创建", "FAIL", error=e)
            return None
    
    def test_content_list(self):
        """测试内容列表获取"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/content/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    total = data["data"].get("total", 0)
                    self.log_test("内容列表", "PASS", f"总计内容: {total}条")
                    return True
                else:
                    self.log_test("内容列表", "FAIL", error=data.get("error"))
                    return False
            else:
                self.log_test("内容列表", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("内容列表", "FAIL", error=e)
            return False
    
    def test_content_search(self):
        """测试内容搜索功能"""
        try:
            params = {"keywords": "测试,历史", "page": 1, "size": 10}
            response = self.session.get(
                f"{self.base_url}/api/v1/content/",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    items = len(data["data"].get("items", []))
                    self.log_test("内容搜索", "PASS", f"搜索结果: {items}条")
                    return True
                else:
                    self.log_test("内容搜索", "FAIL", error=data.get("error"))
                    return False
            else:
                self.log_test("内容搜索", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("内容搜索", "FAIL", error=e)
            return False
    
    def test_statistics(self):
        """测试统计信息获取"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/content/statistics/overview",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    stats = data.get("data", {})
                    total_count = stats.get("total_count", 0)
                    self.log_test("统计信息", "PASS", f"统计数据获取成功，总数: {total_count}")
                    return True
                else:
                    self.log_test("统计信息", "FAIL", error=data.get("error"))
                    return False
            else:
                self.log_test("统计信息", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("统计信息", "FAIL", error=e)
            return False
    
    def test_batch_content_creation(self):
        """测试批量内容创建"""
        try:
            batch_data = {
                "contents": [
                    {
                        "title": "批量测试文档1",
                        "content": "这是第一份批量测试文档",
                        "source": "manual",
                        "author": "批量测试者1"
                    },
                    {
                        "title": "批量测试文档2", 
                        "content": "这是第二份批量测试文档",
                        "source": "manual",
                        "author": "批量测试者2"
                    }
                ],
                "batch_name": "功能测试批次",
                "auto_deduplicate": True
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/content/batch",
                json=batch_data,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    created = data["data"].get("created_count", 0)
                    self.log_test("批量内容创建", "PASS", f"创建 {created} 条内容")
                    return True
                else:
                    self.log_test("批量内容创建", "FAIL", error=data.get("error"))
                    return False
            else:
                self.log_test("批量内容创建", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("批量内容创建", "FAIL", error=e)
            return False
    
    def test_file_upload(self):
        """测试文件上传功能"""
        try:
            # 创建测试JSON文件
            test_data = [
                {
                    "title": "文件上传测试文档",
                    "content": "这是通过文件上传方式创建的测试文档",
                    "source": "manual",
                    "author": "文件测试者",
                    "keywords": ["上传", "测试"]
                }
            ]
            
            # 写入临时文件
            test_file_path = "/tmp/test_upload.json"
            with open(test_file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            # 上传文件
            with open(test_file_path, "rb") as f:
                files = {"file": ("test_upload.json", f, "application/json")}
                data = {"batch_name": "文件上传测试", "auto_deduplicate": True}
                
                response = self.session.post(
                    f"{self.base_url}/api/v1/content/upload",
                    files=files,
                    data=data,
                    timeout=15
                )
            
            # 清理临时文件
            os.unlink(test_file_path)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    processed = result["data"].get("processed_count", 0)
                    self.log_test("文件上传", "PASS", f"处理 {processed} 条记录")
                    return True
                else:
                    self.log_test("文件上传", "FAIL", error=result.get("error"))
                    return False
            else:
                self.log_test("文件上传", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("文件上传", "FAIL", error=e)
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始数据源服务全面功能测试...")
        print(f"📍 测试目标: {self.base_url}")
        print("-" * 50)
        
        # 基础服务测试
        self.test_health_check()
        self.test_service_info() 
        self.test_api_documentation()
        
        # API功能测试
        self.test_content_creation()
        self.test_content_list()
        self.test_content_search()
        self.test_statistics()
        self.test_batch_content_creation()
        self.test_file_upload()
        
        # 测试结果总结
        self.test_results["end_time"] = datetime.now().isoformat()
        
        print("-" * 50)
        print("📊 测试结果总结:")
        print(f"  总计测试: {self.test_results['summary']['total']}")
        print(f"  通过测试: {self.test_results['summary']['passed']} ✅")
        print(f"  失败测试: {self.test_results['summary']['failed']} ❌")
        
        success_rate = (self.test_results['summary']['passed'] / 
                       self.test_results['summary']['total']) * 100
        print(f"  成功率: {success_rate:.1f}%")
        
        return self.test_results
    
    def save_results(self, filename):
        """保存测试结果到文件"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        print(f"📄 测试结果已保存到: {filename}")


def main():
    """主函数"""
    tester = ServiceTester()
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 保存结果
    result_file = Path(__file__).parent / f"service_test_results_{datetime.now().strftime('%H%M%S')}.json"
    tester.save_results(result_file)
    
    # 返回退出码
    return 0 if results['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())