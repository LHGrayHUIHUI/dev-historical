"""
数据库依赖注入模块 - Database Dependency Injection

为FastAPI控制器提供数据库会话的依赖注入功能
简化数据库操作和事务管理
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .utils.database import get_database_session


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    获取异步数据库会话的依赖注入函数
    
    用于FastAPI的Depends()机制，自动管理数据库会话的生命周期
    
    Yields:
        AsyncSession: 异步数据库会话
        
    使用示例:
        @app.get("/example")
        async def example_endpoint(session: AsyncSession = Depends(get_async_session)):
            # 使用session进行数据库操作
            pass
    """
    async with get_database_session() as session:
        yield session