"""
数据库连接和初始化模块

负责管理多个数据库的连接：
- PostgreSQL: 关系数据存储
- InfluxDB: 时序数据存储
- ClickHouse: OLAP分析数据库
- Redis: 缓存和会话存储
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from clickhouse_driver import Client as ClickHouseClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

from ..config.settings import settings

logger = logging.getLogger(__name__)

# SQLAlchemy基类
Base = declarative_base()

# 数据库引擎
postgres_engine = None
postgres_session_maker = None

# 全局连接实例
influxdb_client: Optional[InfluxDBClientAsync] = None
clickhouse_client: Optional[ClickHouseClient] = None
redis_client: Optional[redis.Redis] = None


async def init_database():
    """
    初始化所有数据库连接
    
    创建PostgreSQL引擎、InfluxDB客户端、ClickHouse客户端和Redis连接。
    同时创建必要的表结构和初始化数据。
    """
    global postgres_engine, postgres_session_maker
    global influxdb_client, clickhouse_client, redis_client
    
    try:
        logger.info("正在初始化数据库连接...")
        
        # 初始化PostgreSQL
        postgres_engine = create_async_engine(
            settings.database.postgres_url,
            echo=settings.debug,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        postgres_session_maker = async_sessionmaker(
            postgres_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # 创建表结构
        async with postgres_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("PostgreSQL 连接初始化完成")
        
        # 初始化InfluxDB
        influxdb_client = InfluxDBClientAsync(
            url=settings.database.influxdb_url,
            token=settings.database.influxdb_token,
            org=settings.database.influxdb_org
        )
        
        # 验证InfluxDB连接
        health = await influxdb_client.health()
        if health.status == "pass":
            logger.info("InfluxDB 连接初始化完成")
        else:
            logger.warning(f"InfluxDB 健康检查警告: {health.message}")
        
        # 初始化ClickHouse
        clickhouse_client = ClickHouseClient(
            host=settings.database.clickhouse_host,
            port=settings.database.clickhouse_port,
            database=settings.database.clickhouse_database,
            user=settings.database.clickhouse_user,
            password=settings.database.clickhouse_password
        )
        
        # 验证ClickHouse连接
        try:
            clickhouse_client.execute("SELECT 1")
            logger.info("ClickHouse 连接初始化完成")
        except Exception as e:
            logger.warning(f"ClickHouse 连接验证失败: {e}")
        
        # 初始化Redis
        redis_client = redis.from_url(
            settings.database.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20
        )
        
        # 验证Redis连接
        await redis_client.ping()
        logger.info("Redis 连接初始化完成")
        
        # 初始化ClickHouse表结构
        await init_clickhouse_tables()
        
        logger.info("所有数据库连接初始化完成")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


async def init_clickhouse_tables():
    """初始化ClickHouse表结构"""
    
    # 内容性能分析表
    content_performance_ddl = """
    CREATE TABLE IF NOT EXISTS content_performance (
        content_id String,
        platform String,
        publish_time DateTime,
        views UInt64,
        likes UInt64,
        comments UInt64,
        shares UInt64,
        engagement_rate Float64,
        click_through_rate Float64,
        conversion_rate Float64,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (platform, publish_time, content_id)
    PARTITION BY toYYYYMM(publish_time)
    """
    
    # 平台对比分析表
    platform_comparison_ddl = """
    CREATE TABLE IF NOT EXISTS platform_comparison (
        date Date,
        platform String,
        total_content UInt64,
        total_views UInt64,
        total_engagement UInt64,
        avg_engagement_rate Float64,
        top_content_id String,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (date, platform)
    PARTITION BY toYYYYMM(date)
    """
    
    # 用户行为分析表
    user_behavior_ddl = """
    CREATE TABLE IF NOT EXISTS user_behavior (
        user_id String,
        platform String,
        action_type String,
        content_id String,
        timestamp DateTime,
        session_id String,
        device_type String,
        location String,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (user_id, timestamp, platform)
    PARTITION BY toYYYYMM(timestamp)
    """
    
    # 趋势分析聚合表
    trend_analysis_ddl = """
    CREATE TABLE IF NOT EXISTS trend_analysis (
        metric_name String,
        time_period String,
        date Date,
        value Float64,
        growth_rate Float64,
        platform String,
        category String,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree()
    ORDER BY (metric_name, time_period, date, platform)
    PARTITION BY toYYYYMM(date)
    """
    
    try:
        if clickhouse_client:
            clickhouse_client.execute(content_performance_ddl)
            clickhouse_client.execute(platform_comparison_ddl)
            clickhouse_client.execute(user_behavior_ddl)
            clickhouse_client.execute(trend_analysis_ddl)
            logger.info("ClickHouse 表结构初始化完成")
    except Exception as e:
        logger.error(f"ClickHouse 表初始化失败: {e}")


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取PostgreSQL数据库会话
    
    Returns:
        AsyncSession: 异步数据库会话
    """
    if not postgres_session_maker:
        await init_database()
    
    async with postgres_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_influxdb() -> InfluxDBClientAsync:
    """
    获取InfluxDB客户端
    
    Returns:
        InfluxDBClientAsync: InfluxDB异步客户端
    """
    if not influxdb_client:
        await init_database()
    
    return influxdb_client


def get_clickhouse() -> ClickHouseClient:
    """
    获取ClickHouse客户端
    
    Returns:
        ClickHouseClient: ClickHouse客户端
    """
    if not clickhouse_client:
        # ClickHouse是同步客户端，需要在异步环境中谨慎使用
        raise RuntimeError("ClickHouse client not initialized. Call init_database() first.")
    
    return clickhouse_client


async def get_redis() -> redis.Redis:
    """
    获取Redis客户端
    
    Returns:
        redis.Redis: Redis异步客户端
    """
    if not redis_client:
        await init_database()
    
    return redis_client


async def close_databases():
    """关闭所有数据库连接"""
    global postgres_engine, influxdb_client, clickhouse_client, redis_client
    
    try:
        if postgres_engine:
            await postgres_engine.dispose()
            logger.info("PostgreSQL 连接已关闭")
        
        if influxdb_client:
            await influxdb_client.close()
            logger.info("InfluxDB 连接已关闭")
        
        if clickhouse_client:
            clickhouse_client.disconnect()
            logger.info("ClickHouse 连接已关闭")
        
        if redis_client:
            await redis_client.close()
            logger.info("Redis 连接已关闭")
            
    except Exception as e:
        logger.error(f"关闭数据库连接时出错: {e}")


# 数据库健康检查
async def check_database_health() -> dict:
    """
    检查所有数据库的健康状态
    
    Returns:
        dict: 各数据库的健康状态
    """
    health_status = {
        "postgresql": False,
        "influxdb": False,
        "clickhouse": False,
        "redis": False
    }
    
    # PostgreSQL健康检查
    try:
        if postgres_engine:
            async with postgres_engine.connect() as conn:
                await conn.execute("SELECT 1")
            health_status["postgresql"] = True
    except Exception as e:
        logger.warning(f"PostgreSQL 健康检查失败: {e}")
    
    # InfluxDB健康检查
    try:
        if influxdb_client:
            health = await influxdb_client.health()
            health_status["influxdb"] = health.status == "pass"
    except Exception as e:
        logger.warning(f"InfluxDB 健康检查失败: {e}")
    
    # ClickHouse健康检查
    try:
        if clickhouse_client:
            clickhouse_client.execute("SELECT 1")
            health_status["clickhouse"] = True
    except Exception as e:
        logger.warning(f"ClickHouse 健康检查失败: {e}")
    
    # Redis健康检查
    try:
        if redis_client:
            await redis_client.ping()
            health_status["redis"] = True
    except Exception as e:
        logger.warning(f"Redis 健康检查失败: {e}")
    
    return health_status


# 数据库迁移和初始化辅助函数
async def run_migrations():
    """运行数据库迁移"""
    try:
        # 这里可以集成Alembic等迁移工具
        logger.info("数据库迁移完成")
    except Exception as e:
        logger.error(f"数据库迁移失败: {e}")
        raise