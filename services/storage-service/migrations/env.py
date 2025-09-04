"""
Alembic数据库迁移环境配置

支持异步SQLAlchemy和自动模型发现
"""

import asyncio
import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入模型
from src.models import Base

# Alembic Config对象，提供访问alembic.ini文件中值的功能
config = context.config

# 解释配置文件进行Python日志记录
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 为'autogenerate'支持添加MetaData对象
target_metadata = Base.metadata

# 其他需要的值从配置中获取
def get_url():
    """获取数据库连接URL"""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """在'offline'模式下运行迁移
    
    这将配置context，只需要一个URL，而不是引擎，
    不过Engine也是可以接受的。通过跳过引擎创建，
    我们甚至不需要DBAPI可用。
    
    调用context.execute()仅输出SQL到STDOUT，
    而不是试图执行它到数据库。
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """运行迁移的核心函数"""
    context.configure(
        connection=connection, 
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在异步模式下运行迁移"""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在'online'模式下运行迁移"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()