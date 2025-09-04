#!/usr/bin/env python3
"""
数据源服务功能测试 - 使用模拟数据库
测试所有API端点和功能，不依赖真实数据库连接
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# 添加服务模块路径
service_path = Path(__file__).parent.parent.parent / "services" / "data-source"
sys.path.insert(0, str(service_path))

# 导入服务模块
from src.main import app
from src.config.settings import get_settings
from src.database.database import get_database_manager

class MockServiceTest:
    """模拟服务测试器"""
    
    def __init__(self):
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "summary": {"total": 0, "passed": 0, "failed": 0}
        }
        self.client = None
        
    def setup_mocks(self):
        """设置模拟依赖"""
        # 模拟数据库管理器
        mock_db_manager = AsyncMock()
        mock_collection = AsyncMock()
        
        # 设置模拟方法
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        mock_db_manager.health_check.return_value = {
            "mongodb": {"status": "connected", "latency": 10.5},
            "redis": {"status": "connected", "latency": 2.3}
        }
        
        # 模拟Redis客户端
        mock_redis = AsyncMock()
        mock_db_manager.get_redis_client.return_value = mock_redis
        
        # 模拟内容数据
        mock_content_data = {
            "_id": "test_id_123",
            "title": "测试历史文档",
            "content": "这是一份测试的历史文档内容",
            "source": "manual",
            "author": "测试者",
            "keywords": ["测试", "历史"],
            "category": "测试分类",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "quality_score": 85
        }
        
        # 设置查询结果
        mock_collection.insert_one.return_value.inserted_id = "test_id_123"
        mock_collection.find.return_value.to_list.return_value = [mock_content_data]
        mock_collection.count_documents.return_value = 1
        mock_collection.find_one.return_value = mock_content_data
        
        # 覆盖依赖注入
        app.dependency_overrides[get_database_manager] = lambda: mock_db_manager
        
        # 创建测试客户端
        self.client = TestClient(app)
        
        return mock_db_manager, mock_collection
    
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
        """测试健康检查端点"""
        try:
            response = self.client.get("/health")
            if response.status_code == 200:
                data = response.json()
                self.log_test("健康检查", "PASS", f"状态: {data.get('status', 'unknown')}")
                return True
            else:
                self.log_test("健康检查", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("健康检查", "FAIL", error=e)
            return False
    
    def test_service_info(self):
        """测试服务信息端点"""
        try:
            response = self.client.get("/info")
            if response.status_code == 200:
                data = response.json()
                service_name = data.get("name", "unknown")
                version = data.get("version", "unknown")
                self.log_test("服务信息", "PASS", f"{service_name} v{version}")
                return True
            else:
                self.log_test("服务信息", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("服务信息", "FAIL", error=e)
            return False
    
    def test_api_documentation(self):
        """测试API文档端点"""
        try:
            response = self.client.get("/docs")
            if response.status_code == 200:
                self.log_test("API文档", "PASS", "Swagger文档可访问")
                return True
            else:
                self.log_test("API文档", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API文档", "FAIL", error=e)
            return False
    
    def test_openapi_schema(self):
        """测试OpenAPI模式"""
        try:
            response = self.client.get("/openapi.json")
            if response.status_code == 200:
                schema = response.json()
                paths_count = len(schema.get("paths", {}))
                self.log_test("OpenAPI模式", "PASS", f"API路径数: {paths_count}")
                return True
            else:
                self.log_test("OpenAPI模式", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("OpenAPI模式", "FAIL", error=e)
            return False
    
    def test_content_creation(self):
        """测试内容创建端点"""
        try:
            test_content = {
                "title": "模拟测试文档",
                "content": "这是一份模拟测试的历史文档内容，用于验证内容管理系统的功能。",
                "source": "manual",
                "author": "测试者",
                "keywords": ["测试", "历史", "模拟"],
                "category": "测试分类"
            }
            
            response = self.client.post("/api/v1/content/", json=test_content)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    content_id = data["data"].get("id")
                    self.log_test("内容创建", "PASS", f"创建内容ID: {content_id}")
                    return True
                else:
                    self.log_test("内容创建", "FAIL", error=data.get("error"))
                    return False
            else:
                self.log_test("内容创建", "FAIL", error=f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("内容创建", "FAIL", error=e)
            return False
    
    def test_content_list(self):
        """测试内容列表端点"""
        try:
            response = self.client.get("/api/v1/content/")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    total = data["data"].get("total", 0)
                    self.log_test("内容列表", "PASS", f"内容总数: {total}")
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
            response = self.client.get("/api/v1/content/", params=params)
            
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
    
    def test_statistics_overview(self):
        """测试统计信息端点"""
        try:
            response = self.client.get("/api/v1/content/statistics/overview")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    stats = data.get("data", {})
                    total_count = stats.get("total_count", 0)
                    self.log_test("统计信息", "PASS", f"统计获取成功，总数: {total_count}")
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
                "batch_name": "模拟测试批次",
                "auto_deduplicate": True
            }
            
            response = self.client.post("/api/v1/content/batch", json=batch_data)
            
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
    
    def test_file_upload_endpoint(self):
        """测试文件上传端点结构"""
        try:
            # 模拟空文件上传测试端点存在性
            files = {"file": ("test.txt", b"", "text/plain")}
            data = {"batch_name": "端点测试"}
            
            response = self.client.post("/api/v1/content/upload", files=files, data=data)
            
            # 即使上传失败，端点应该存在并返回错误响应
            if response.status_code in [200, 400, 422]:
                self.log_test("文件上传端点", "PASS", f"端点可访问 (状态: {response.status_code})")
                return True
            else:
                self.log_test("文件上传端点", "FAIL", error=f"意外状态码: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("文件上传端点", "FAIL", error=e)
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始数据源服务功能测试 (模拟模式)...")
        print("-" * 60)
        
        # 设置模拟环境
        mock_db, mock_collection = self.setup_mocks()
        
        try:
            # 基础端点测试
            self.test_health_check()
            self.test_service_info()
            self.test_api_documentation()
            self.test_openapi_schema()
            
            # API功能测试
            self.test_content_creation()
            self.test_content_list()
            self.test_content_search()
            self.test_statistics_overview()
            self.test_batch_content_creation()
            self.test_file_upload_endpoint()
            
        finally:
            # 清理依赖注入覆盖
            app.dependency_overrides.clear()
        
        # 测试结果总结
        self.test_results["end_time"] = datetime.now().isoformat()
        
        print("-" * 60)
        print("📊 测试结果总结:")
        print(f"  总计测试: {self.test_results['summary']['total']}")
        print(f"  通过测试: {self.test_results['summary']['passed']} ✅")
        print(f"  失败测试: {self.test_results['summary']['failed']} ❌")
        
        if self.test_results['summary']['total'] > 0:
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
    tester = MockServiceTest()
    
    # 运行测试
    results = tester.run_all_tests()
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = Path(__file__).parent / f"mock_test_results_{timestamp}.json"
    tester.save_results(result_file)
    
    # 生成简单报告
    print(f"\n📋 简化测试报告:")
    print(f"测试时间: {results['start_time']} - {results['end_time']}")
    print(f"架构重构后的数据源服务通过了 {results['summary']['passed']}/{results['summary']['total']} 项功能测试")
    
    if results['summary']['failed'] == 0:
        print("🎉 所有核心功能测试通过！服务架构重构成功！")
    else:
        print(f"⚠️  有 {results['summary']['failed']} 项测试失败，需要进一步检查")
    
    # 返回退出码
    return 0 if results['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())