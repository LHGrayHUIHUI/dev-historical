"""
测试配置文件

提供测试夹具和公共配置
"""

import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app
from src.models.database import Base, get_database
from src.services.redis_service import RedisService


# 测试数据库URL
TEST_DATABASE_URL = "postgresql://postgres:password@localhost:5433/test_historical_text_publishing"


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """测试数据库引擎"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    
    # 创建所有表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # 清理
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """测试数据库会话"""
    async_session = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_redis():
    """测试Redis服务"""
    redis_service = RedisService()
    redis_service.redis_url = "redis://localhost:6380/15"  # 使用测试库
    await redis_service.connect()
    
    yield redis_service
    
    # 清理测试数据
    redis = await redis_service.get_redis()
    await redis.flushdb()
    await redis_service.disconnect()


@pytest.fixture
async def test_app(test_db, test_redis):
    """测试应用"""
    app.dependency_overrides[get_database] = lambda: test_db
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
async def test_client(test_app):
    """测试HTTP客户端"""
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sync_test_client(test_app):
    """同步测试客户端"""
    return TestClient(test_app)