"""
数据库连接管理

提供PostgreSQL、Redis等数据库连接的创建、管理和生命周期控制。
支持连接池、事务管理和异步操作。

主要功能：
- PostgreSQL异步连接池
- Redis连接管理
- 数据库会话生命周期
- 连接健康检查
- 自动重连机制

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import StaticPool
from sqlalchemy import text
import redis.asyncio as aioredis

from ..config.settings import get_settings
from ..models.database import Base

# 配置日志
logger = logging.getLogger(__name__)

# 全局配置
settings = get_settings()


class DatabaseManager:
    """
    数据库管理器
    
    统一管理所有数据库连接，包括PostgreSQL和Redis。
    提供连接池管理、健康检查和优雅关闭功能。
    """
    
    def __init__(self):
        """初始化数据库管理器"""
        self._pg_engine: Optional[AsyncEngine] = None
        self._pg_session_factory: Optional[async_sessionmaker] = None
        self._redis_pool: Optional[aioredis.ConnectionPool] = None
        self._redis_client: Optional[aioredis.Redis] = None
        self._initialized = False
        
        logger.info("数据库管理器初始化")
    
    async def initialize(self):
        """
        初始化所有数据库连接
        
        创建连接池、配置会话工厂等。
        """
        if self._initialized:
            return
        
        try:
            await self._init_postgresql()
            await self._init_redis()
            self._initialized = True
            logger.info("数据库管理器初始化成功")
            
        except Exception as e:
            logger.error(f"数据库管理器初始化失败: {str(e)}")
            await self.cleanup()
            raise
    
    async def _init_postgresql(self):
        """初始化PostgreSQL连接"""
        try:
            # 创建异步引擎
            self._pg_engine = create_async_engine(
                settings.database.database_url,
                echo=settings.logging.LOG_SQL_QUERIES,
                pool_size=settings.database.POSTGRES_POOL_SIZE,
                max_overflow=settings.database.POSTGRES_MAX_OVERFLOW,
                pool_pre_ping=True,  # 启用连接健康检查
                pool_recycle=3600,   # 1小时后回收连接
                
                # 连接参数
                connect_args={
                    "server_settings": {
                        "application_name": settings.SERVICE_NAME,
                    },
                    "command_timeout": 30,
                },
                
                # 如果是测试环境，使用StaticPool
                poolclass=StaticPool if settings.is_testing else None,
            )
            
            # 创建会话工厂
            self._pg_session_factory = async_sessionmaker(
                bind=self._pg_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
            
            # 测试连接
            async with self._pg_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("PostgreSQL连接初始化成功")
            
        except Exception as e:
            logger.error(f"PostgreSQL初始化失败: {str(e)}")
            raise
    
    async def _init_redis(self):
        """初始化Redis连接"""
        try:
            # 创建连接池
            self._redis_pool = aioredis.ConnectionPool.from_url(
                settings.redis.redis_url,
                max_connections=settings.redis.REDIS_MAX_CONNECTIONS,
                retry_on_timeout=settings.redis.REDIS_RETRY_ON_TIMEOUT,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
            )
            
            # 创建Redis客户端
            self._redis_client = aioredis.Redis(
                connection_pool=self._redis_pool,
                decode_responses=True,
                protocol=3,
            )
            
            # 测试连接
            await self._redis_client.ping()
            
            logger.info("Redis连接初始化成功")
            
        except Exception as e:
            logger.error(f"Redis初始化失败: {str(e)}")
            raise
    
    async def get_postgres_session(self) -> AsyncSession:
        """
        获取PostgreSQL会话
        
        Returns:
            异步数据库会话
            
        Raises:
            RuntimeError: 数据库管理器未初始化
        """
        if not self._initialized or not self._pg_session_factory:
            raise RuntimeError("数据库管理器未初始化")
        
        return self._pg_session_factory()
    
    async def get_redis_client(self) -> aioredis.Redis:
        """
        获取Redis客户端
        
        Returns:
            Redis客户端实例
            
        Raises:
            RuntimeError: 数据库管理器未初始化
        """
        if not self._initialized or not self._redis_client:
            raise RuntimeError("数据库管理器未初始化")
        
        return self._redis_client
    
    async def health_check(self) -> dict:
        """
        执行健康检查
        
        Returns:
            健康状态字典
        """
        health_status = {
            "postgresql": False,
            "redis": False,
            "overall": False
        }
        
        # 检查PostgreSQL
        try:
            if self._pg_engine:
                async with self._pg_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                health_status["postgresql"] = True
        except Exception as e:
            logger.error(f"PostgreSQL健康检查失败: {str(e)}")
        
        # 检查Redis
        try:
            if self._redis_client:
                await self._redis_client.ping()
                health_status["redis"] = True
        except Exception as e:
            logger.error(f"Redis健康检查失败: {str(e)}")
        
        # 整体状态
        health_status["overall"] = all([
            health_status["postgresql"],
            health_status["redis"]
        ])
        
        return health_status
    
    async def cleanup(self):
        """清理所有数据库连接"""
        logger.info("开始清理数据库连接")
        
        # 清理Redis连接
        if self._redis_client:
            try:
                await self._redis_client.close()
            except Exception as e:
                logger.error(f"关闭Redis客户端失败: {str(e)}")
        
        if self._redis_pool:
            try:
                await self._redis_pool.disconnect()
            except Exception as e:
                logger.error(f"关闭Redis连接池失败: {str(e)}")
        
        # 清理PostgreSQL连接
        if self._pg_engine:
            try:
                await self._pg_engine.dispose()
            except Exception as e:
                logger.error(f"关闭PostgreSQL引擎失败: {str(e)}")
        
        # 重置状态
        self._pg_engine = None
        self._pg_session_factory = None
        self._redis_pool = None
        self._redis_client = None
        self._initialized = False
        
        logger.info("数据库连接清理完成")
    
    @asynccontextmanager
    async def transaction(self):
        """
        事务上下文管理器
        
        提供自动事务管理，支持自动提交和回滚。
        
        Yields:
            数据库会话
        """
        session = await self.get_postgres_session()
        try:
            async with session.begin():
                yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager() -> DatabaseManager:
    """
    获取全局数据库管理器实例
    
    Returns:
        数据库管理器实例
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager


async def get_database_session() -> AsyncSession:
    """
    获取数据库会话
    
    这是一个便捷函数，用于依赖注入。
    
    Returns:
        数据库会话
    """
    db_manager = await get_database_manager()
    return await db_manager.get_postgres_session()


async def get_redis_connection() -> aioredis.Redis:
    """
    获取Redis连接
    
    这是一个便捷函数，用于依赖注入。
    
    Returns:
        Redis客户端
    """
    db_manager = await get_database_manager()
    return await db_manager.get_redis_client()


# 数据库表管理函数

async def create_tables():
    """
    创建所有数据库表
    
    在应用启动时调用，创建所有定义的数据库表。
    """
    try:
        db_manager = await get_database_manager()
        
        async with db_manager._pg_engine.begin() as conn:
            # 创建所有表
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("数据库表创建成功")
        
    except Exception as e:
        logger.error(f"创建数据库表失败: {str(e)}")
        raise


async def drop_tables():
    """
    删除所有数据库表
    
    谨慎使用，通常只在测试环境中使用。
    """
    try:
        db_manager = await get_database_manager()
        
        async with db_manager._pg_engine.begin() as conn:
            # 删除所有表
            await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("数据库表删除成功")
        
    except Exception as e:
        logger.error(f"删除数据库表失败: {str(e)}")
        raise


# 健康检查函数

async def check_database_health() -> bool:
    """
    检查数据库健康状态
    
    Returns:
        数据库是否健康
    """
    try:
        db_manager = await get_database_manager()
        health_status = await db_manager.health_check()
        return health_status["overall"]
    except Exception as e:
        logger.error(f"数据库健康检查失败: {str(e)}")
        return False


# 应用生命周期管理

async def initialize_database():
    """
    初始化数据库连接
    
    在应用启动时调用。
    """
    try:
        await get_database_manager()
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        raise


async def cleanup_database():
    """
    清理数据库连接
    
    在应用关闭时调用。
    """
    global _db_manager
    
    if _db_manager:
        await _db_manager.cleanup()
        _db_manager = None
    
    logger.info("数据库清理完成")


# 测试工具函数

async def reset_database():
    """
    重置数据库
    
    删除所有表并重新创建，仅用于测试环境。
    """
    if not settings.is_testing:
        raise RuntimeError("reset_database只能在测试环境中使用")
    
    await drop_tables()
    await create_tables()
    logger.info("数据库重置完成")


@asynccontextmanager
async def test_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    测试数据库会话上下文管理器
    
    提供独立的测试会话，自动回滚事务。
    
    Yields:
        测试数据库会话
    """
    if not settings.is_testing:
        raise RuntimeError("test_database_session只能在测试环境中使用")
    
    session = await get_database_session()
    transaction = session.begin()
    
    try:
        await transaction.__aenter__()
        yield session
    except Exception:
        await transaction.__aexit__(None, None, None)
        raise
    finally:
        # 在测试中总是回滚
        await session.rollback()
        await session.close()


# 连接池监控

async def get_connection_pool_stats() -> dict:
    """
    获取连接池统计信息
    
    Returns:
        连接池统计字典
    """
    try:
        db_manager = await get_database_manager()
        
        if not db_manager._pg_engine:
            return {"error": "数据库引擎未初始化"}
        
        pool = db_manager._pg_engine.pool
        
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
        
    except Exception as e:
        logger.error(f"获取连接池统计失败: {str(e)}")
        return {"error": str(e)}