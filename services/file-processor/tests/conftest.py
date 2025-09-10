"""
测试配置文件
提供测试夹具和共享配置
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# 导入应用和依赖
from src.main import app
from src.config.settings import Settings, get_settings
# from src.database.database import DatabaseManager, get_database_manager  # file-processor 服务没有数据库


# 测试配置
@pytest.fixture(scope="session")
def test_settings():
    """测试配置 - file-processor是无状态服务，不需要数据库配置"""
    return Settings(
        service={
            "environment": "testing",
            "secret_key": "test-secret-key", 
            "cors_origins": ["http://testserver"],
            "service_name": "file-processor-test"
        },
        logging={
            "log_level": "DEBUG"
        }
    )


# file-processor 服务是无状态服务，不需要数据库管理器


# 应用依赖注入覆盖
@pytest.fixture
def app_with_mocks(test_settings):
    """带模拟依赖的应用"""
    app.dependency_overrides[get_settings] = lambda: test_settings
    
    yield app
    
    # 清理依赖覆盖
    app.dependency_overrides.clear()


# 同步测试客户端
@pytest.fixture
def client(app_with_mocks):
    """测试客户端"""
    return TestClient(app_with_mocks)


# 异步测试客户端
@pytest.fixture
async def async_client(app_with_mocks):
    """异步测试客户端"""
    async with AsyncClient(app=app_with_mocks, base_url="http://testserver") as client:
        yield client


# 事件循环
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# 测试数据夹具
@pytest.fixture
def sample_content_data():
    """示例内容数据"""
    return {
        "title": "测试文章标题",
        "content": "这是一篇测试文章的内容，包含足够的文本长度以满足验证要求。",
        "source": "manual",
        "author": "测试作者",
        "keywords": ["测试", "文章"],
        "category": "测试分类"
    }




# 数据库清理夹具
@pytest.fixture
async def clean_database():
    """清理测试数据库"""
    # 测试前清理
    yield
    
    # 测试后清理（如果使用真实数据库）
    # try:
    #     db_manager = await get_database_manager()
    #     collection = await db_manager.get_mongodb_collection("contents")
    #     await collection.delete_many({})
    # except Exception:
    #     pass


# 工具函数
@pytest.fixture
def assert_response_structure():
    """断言响应结构的工具函数"""
    def _assert_response_structure(response_data, expected_keys=None):
        """检查API响应的基本结构"""
        assert "success" in response_data
        assert isinstance(response_data["success"], bool)
        
        if response_data["success"]:
            assert "data" in response_data
            assert "message" in response_data
        else:
            assert "error" in response_data
            
        if expected_keys:
            for key in expected_keys:
                assert key in response_data.get("data", {})
    
    return _assert_response_structure


# 异步工具函数
@pytest.fixture
def async_test_helper():
    """异步测试帮助器"""
    class AsyncTestHelper:
        @staticmethod
        async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
            """等待条件满足"""
            import time
            start_time = time.time()
            while time.time() - start_time < timeout:
                if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                    return True
                await asyncio.sleep(interval)
            return False
        
        @staticmethod
        def create_mock_response(data, success=True, message="测试成功"):
            """创建模拟响应"""
            return {
                "success": success,
                "data": data,
                "message": message
            }
    
    return AsyncTestHelper()


# 性能测试夹具
@pytest.fixture
def performance_monitor():
    """性能监控工具"""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.end_time - self.start_time if self.start_time else None
        
        def assert_response_time(self, max_time):
            duration = self.stop()
            assert duration is not None, "性能监控未启动"
            assert duration <= max_time, f"响应时间 {duration:.3f}s 超过期望的 {max_time}s"
    
    return PerformanceMonitor


# 模拟数据生成器
@pytest.fixture
def mock_data_generator():
    """模拟数据生成器"""
    class MockDataGenerator:
        @staticmethod
        def generate_content_batch(count=10):
            """生成批量内容数据"""
            contents = []
            for i in range(count):
                contents.append({
                    "title": f"测试文章标题 {i+1}",
                    "content": f"这是第{i+1}篇测试文章的内容，包含足够的文本长度。",
                    "source": "manual",
                    "author": f"测试作者{i+1}",
                    "keywords": ["测试", f"关键词{i+1}"],
                    "category": "测试分类"
                })
            return contents
        
    
    return MockDataGenerator()