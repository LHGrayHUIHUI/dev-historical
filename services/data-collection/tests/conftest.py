"""
pytest配置文件

提供测试夹具和配置
"""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# 设置测试环境变量
os.environ.update({
    "SERVICE_ENVIRONMENT": "testing",
    "DEBUG": "true",
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/test_db",
    "MONGODB_URL": "mongodb://localhost:27017/test_db",
    "REDIS_URL": "redis://localhost:6379/15",
    "RABBITMQ_URL": "amqp://test:test@localhost:5672/test_vhost",
    "SECRET_KEY": "test-secret-key-for-testing-only",
    "JWT_SECRET_KEY": "test-jwt-secret-key",
    "MINIO_ENDPOINT": "localhost:9000",
    "MINIO_ACCESS_KEY": "testkey",
    "MINIO_SECRET_KEY": "testsecret",
    "MINIO_SECURE": "false",
    "VIRUS_SCAN_ENABLED": "false",
    "METRICS_ENABLED": "false"
})

from src.config import get_settings
from src.main import app
from src.models import Base
from src.utils.database import get_database_session


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db_engine():
    """测试数据库引擎"""
    settings = get_settings()
    
    # 创建测试数据库引擎
    engine = create_async_engine(
        settings.database_url,
        echo=True,
        future=True
    )
    
    # 创建所有表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # 清理
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """测试数据库会话"""
    session_factory = sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with session_factory() as session:
        yield session


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """测试客户端"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def temp_file():
    """临时文件夹具"""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as f:
        f.write("This is a test file content.\n测试文件内容。")
        temp_path = f.name
    
    yield temp_path
    
    # 清理
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_pdf_file():
    """示例PDF文件路径"""
    # 这里应该返回一个真实的PDF测试文件路径
    # 在实际项目中，可以在tests/fixtures目录下放置测试文件
    return None


@pytest.fixture
def sample_image_file():
    """示例图像文件路径"""
    # 这里应该返回一个真实的图像测试文件路径
    return None


@pytest.fixture
def mock_settings():
    """模拟设置"""
    return get_settings()


# 异步测试配置
pytest_plugins = ('pytest_asyncio',)