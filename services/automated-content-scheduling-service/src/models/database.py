"""
数据库连接和会话管理
支持异步连接池和事务管理
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import event
from sqlalchemy.pool import Pool
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# 设置实例
settings = get_settings()

# 声明基类
Base = declarative_base()

# 异步数据库引擎
engine = None
AsyncSessionLocal = None


def create_engine():
    """创建异步数据库引擎"""
    global engine
    
    if engine is not None:
        return engine
    
    # 创建异步引擎
    engine = create_async_engine(
        settings.database.url.replace("postgresql://", "postgresql+asyncpg://"),
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
        echo=settings.database.echo,
        future=True
    )
    
    # 添加连接池事件监听
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """设置数据库连接参数"""
        if hasattr(dbapi_connection, 'execute'):
            # PostgreSQL特定设置
            dbapi_connection.execute("SET timezone='UTC'")
            
    @event.listens_for(Pool, "connect")
    def receive_connect(dbapi_connection, connection_record):
        """连接建立时的回调"""
        logger.debug(f"New database connection established: {connection_record}")
    
    @event.listens_for(Pool, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        """连接检出时的回调"""
        logger.debug("Connection checked out from pool")
    
    @event.listens_for(Pool, "checkin")
    def receive_checkin(dbapi_connection, connection_record):
        """连接检入时的回调"""
        logger.debug("Connection checked in to pool")
    
    return engine


def create_session_maker():
    """创建会话工厂"""
    global AsyncSessionLocal
    
    if AsyncSessionLocal is not None:
        return AsyncSessionLocal
    
    if engine is None:
        create_engine()
    
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False
    )
    
    return AsyncSessionLocal


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话的异步上下文管理器
    自动处理事务提交和回滚
    """
    if AsyncSessionLocal is None:
        create_session_maker()
    
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("Database session created")
            yield session
            await session.commit()
            logger.debug("Database transaction committed")
        except Exception as e:
            logger.error(f"Database transaction failed: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
            logger.debug("Database session closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI依赖注入用的数据库会话获取器
    """
    async with get_db_session() as session:
        yield session


class DatabaseManager:
    """数据库管理器类"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """初始化数据库连接"""
        try:
            self.engine = create_engine()
            self.session_factory = create_session_maker()
            
            # 测试连接
            async with get_db_session() as session:
                await session.execute("SELECT 1")
            
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_tables(self):
        """创建数据库表"""
        try:
            if self.engine is None:
                await self.initialize()
            
            # 导入所有模型以确保它们被注册到Base.metadata
            from . import scheduling_models, analytics_models
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self):
        """删除数据库表（仅用于测试）"""
        try:
            if self.engine is None:
                await self.initialize()
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def close(self):
        """关闭数据库连接"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    async def health_check(self) -> bool:
        """数据库健康检查"""
        try:
            async with get_db_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """获取连接信息"""
        if not self.engine:
            return {"status": "not_initialized"}
        
        pool = self.engine.pool
        return {
            "status": "connected",
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin()
        }


# 全局数据库管理器实例
db_manager = DatabaseManager()


# 数据库生命周期管理
async def init_database():
    """初始化数据库"""
    await db_manager.initialize()
    await db_manager.create_tables()


async def close_database():
    """关闭数据库连接"""
    await db_manager.close()


# 用于测试的数据库重置功能
async def reset_database():
    """重置数据库（仅用于测试）"""
    if settings.environment.value != "testing":
        raise RuntimeError("Database reset is only allowed in testing environment")
    
    await db_manager.drop_tables()
    await db_manager.create_tables()


# 事务装饰器
def transactional(func):
    """
    事务装饰器，自动处理数据库事务
    用于服务层方法
    """
    async def wrapper(*args, **kwargs):
        async with get_db_session() as session:
            # 将session注入到函数参数中
            if 'session' not in kwargs:
                kwargs['session'] = session
            
            try:
                result = await func(*args, **kwargs)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise
    
    return wrapper