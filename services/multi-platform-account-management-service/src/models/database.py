"""
数据库连接和会话管理

提供异步数据库连接池和会话管理功能
支持PostgreSQL异步操作和连接池优化
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import logging

from ..config.settings import settings

# 创建基础模型类
Base = declarative_base()

# 全局变量
engine = None
async_session_factory = None

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    获取数据库连接URL
    
    Returns:
        str: 数据库连接URL
    """
    return settings.database_url


def create_engine_and_session():
    """
    创建数据库引擎和会话工厂
    
    配置连接池参数和异步引擎选项
    """
    global engine, async_session_factory
    
    if engine is None:
        # 创建异步数据库引擎
        engine = create_async_engine(
            get_database_url(),
            echo=settings.debug,  # 开发模式下打印SQL
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,  # 连接前检查连接有效性
            pool_recycle=3600,   # 1小时后回收连接
            future=True
        )
        
        # 创建会话工厂
        async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        logger.info("数据库引擎和会话工厂创建成功")


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话依赖注入函数
    
    用于FastAPI依赖注入，自动管理会话生命周期
    
    Yields:
        AsyncSession: 数据库会话实例
    """
    if async_session_factory is None:
        create_engine_and_session()
    
    async with async_session_factory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"数据库会话异常: {e}")
            raise
        finally:
            await session.close()


async def create_all_tables():
    """
    创建所有数据表
    
    在应用启动时调用，用于初始化数据库结构
    """
    if engine is None:
        create_engine_and_session()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("所有数据表创建完成")


async def drop_all_tables():
    """
    删除所有数据表
    
    主要用于测试环境的数据清理
    """
    if engine is None:
        create_engine_and_session()
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("所有数据表删除完成")


async def close_database_connection():
    """
    关闭数据库连接
    
    在应用关闭时调用，清理资源
    """
    global engine, async_session_factory
    
    if engine is not None:
        await engine.dispose()
        engine = None
        async_session_factory = None
        logger.info("数据库连接已关闭")


class DatabaseManager:
    """
    数据库管理器类
    
    提供数据库操作的高级接口和事务管理
    """
    
    def __init__(self):
        self.session: AsyncSession = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        if async_session_factory is None:
            create_engine_and_session()
        
        self.session = async_session_factory()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            if exc_type is not None:
                await self.session.rollback()
            else:
                await self.session.commit()
            await self.session.close()
    
    @classmethod
    async def execute_in_transaction(cls, func, *args, **kwargs):
        """
        在事务中执行函数
        
        Args:
            func: 要执行的异步函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            执行结果
        """
        async with cls() as session:
            return await func(session, *args, **kwargs)