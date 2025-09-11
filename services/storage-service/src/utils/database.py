"""
数据库工具模块

提供数据库连接、会话管理和初始化功能
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from ..config import get_settings
from ..models import Base

logger = logging.getLogger(__name__)

# 全局变量存储数据库引擎和会话工厂
_engine: AsyncEngine = None
_session_factory: sessionmaker = None


async def init_database() -> None:
    """初始化数据库连接和会话工厂"""
    global _engine, _session_factory
    
    settings = get_settings()
    
    # 创建异步数据库引擎
    _engine = create_async_engine(
        settings.database_url,
        **settings.database_config,
        poolclass=NullPool if settings.is_testing else None,
    )
    
    # 创建会话工厂
    _session_factory = sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
    )
    
    logger.info("数据库引擎和会话工厂初始化完成")


async def close_database() -> None:
    """关闭数据库连接"""
    global _engine
    
    if _engine:
        await _engine.dispose()
        logger.info("数据库连接已关闭")


def get_database() -> AsyncEngine:
    """获取数据库引擎实例
    
    Returns:
        AsyncEngine: 数据库引擎
    """
    if not _engine:
        raise RuntimeError("数据库未初始化，请先调用 init_database()")
    return _engine


@asynccontextmanager
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话上下文管理器
    
    Yields:
        AsyncSession: 数据库会话
    """
    if not _session_factory:
        raise RuntimeError("数据库未初始化，请先调用 init_database()")
    
    session = _session_factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def check_database_connection() -> bool:
    """检查数据库连接状态
    
    Returns:
        bool: 连接是否正常
    """
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
        
        settings = get_settings()
        
        # 创建临时引擎进行连接测试
        engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        await engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"数据库连接检查失败: {e}")
        return False


async def create_tables() -> None:
    """创建数据库表
    
    注意: 在生产环境中应该使用Alembic迁移
    """
    if not _engine:
        raise RuntimeError("数据库未初始化，请先调用 init_database()")
    
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("数据库表创建完成")


async def drop_tables() -> None:
    """删除数据库表
    
    注意: 谨慎使用，通常只在测试环境使用
    """
    if not _engine:
        raise RuntimeError("数据库未初始化，请先调用 init_database()")
    
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.info("数据库表删除完成")