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
from src.database.database import DatabaseManager, get_database_manager
from src.crawler.crawler_manager import CrawlerManager, get_crawler_manager
from src.proxy.proxy_manager import ProxyManager, get_proxy_manager


# 测试配置
@pytest.fixture(scope="session")
def test_settings():
    """测试配置"""
    return Settings(
        service={
            "environment": "testing",
            "secret_key": "test-secret-key",
            "cors_origins": ["http://testserver"]
        },
        database={
            "mongodb_url": "mongodb://localhost:27017",
            "mongodb_db_name": "historical_text_test",
            "redis_url": "redis://localhost:6379/1"  # 使用测试数据库
        },
        logging={
            "log_level": "DEBUG"
        }
    )


# 模拟数据库管理器
@pytest.fixture
async def mock_db_manager():
    """模拟数据库管理器"""
    mock_manager = AsyncMock(spec=DatabaseManager)
    mock_collection = AsyncMock()
    
    # 设置默认返回值
    mock_manager.get_mongodb_collection.return_value = mock_collection
    mock_manager.health_check.return_value = {
        "mongodb": {"status": "connected", "latency": 10.5},
        "redis": {"status": "connected", "latency": 2.3}
    }
    
    # 模拟Redis客户端
    mock_redis = AsyncMock()
    mock_manager.get_redis_client.return_value = mock_redis
    
    return mock_manager


# 模拟爬虫管理器
@pytest.fixture
async def mock_crawler_manager():
    """模拟爬虫管理器"""
    mock_manager = AsyncMock(spec=CrawlerManager)
    mock_manager.tasks = {}
    mock_manager.crawlers = {}
    mock_manager.initialize = AsyncMock()
    mock_manager.cleanup = AsyncMock()
    mock_manager.get_statistics.return_value = {
        "total_tasks": 0,
        "running_tasks": 0,
        "finished_tasks": 0,
        "error_tasks": 0,
        "total_success_items": 0,
        "total_failed_items": 0,
        "overall_success_rate": 0.0
    }
    return mock_manager


# 模拟代理管理器
@pytest.fixture
async def mock_proxy_manager():
    """模拟代理管理器"""
    mock_manager = AsyncMock(spec=ProxyManager)
    mock_manager.proxies = {}
    mock_manager.active_proxies = []
    mock_manager.banned_proxies = set()
    mock_manager.initialize = AsyncMock()
    mock_manager.get_proxy_statistics.return_value = {
        "total_proxies": 0,
        "active_proxies": 0,
        "banned_proxies": 0,
        "quality_distribution": {},
        "average_success_rate": 0.0,
        "providers": []
    }
    return mock_manager


# 应用依赖注入覆盖
@pytest.fixture
def app_with_mocks(test_settings, mock_db_manager, mock_crawler_manager, mock_proxy_manager):
    """带模拟依赖的应用"""
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_database_manager] = lambda: mock_db_manager
    app.dependency_overrides[get_crawler_manager] = lambda: mock_crawler_manager
    app.dependency_overrides[get_proxy_manager] = lambda: mock_proxy_manager
    
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


@pytest.fixture
def sample_crawler_config():
    """示例爬虫配置"""
    return {
        "platform": "toutiao",
        "keywords": ["历史", "文化"],
        "max_pages": 5,
        "interval": 2.0,
        "priority": 5,
        "proxy_enabled": True
    }


@pytest.fixture
def sample_proxy_data():
    """示例代理数据"""
    return {
        "host": "127.0.0.1",
        "port": 8080,
        "protocol": "http"
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
        
        @staticmethod
        def generate_proxy_list(count=5):
            """生成代理列表数据"""
            proxies = []
            for i in range(count):
                proxies.append({
                    "host": f"127.0.0.{i+1}",
                    "port": 8080 + i,
                    "protocol": "http",
                    "quality": "low"
                })
            return proxies
    
    return MockDataGenerator()