"""
数据库连接和配置管理
提供MongoDB和Redis的连接管理，支持异步操作
"""

import asyncio
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis
from pymongo.errors import ConnectionFailure
from loguru import logger
import os
from contextlib import asynccontextmanager


class DatabaseManager:
    """数据库管理器 - 统一管理MongoDB和Redis连接"""
    
    def __init__(self):
        self.mongodb_client: Optional[AsyncIOMotorClient] = None
        self.mongodb_db = None
        self.redis_client: Optional[aioredis.Redis] = None
        self._mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
        self._redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self._db_name = os.getenv('MONGODB_DB_NAME', 'historical_text_data')
        
    async def connect_mongodb(self) -> None:
        """连接MongoDB数据库"""
        try:
            self.mongodb_client = AsyncIOMotorClient(
                self._mongodb_url,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=10
            )
            
            # 测试连接
            await self.mongodb_client.admin.command('ping')
            self.mongodb_db = self.mongodb_client[self._db_name]
            
            logger.info(f"MongoDB连接成功: {self._mongodb_url}")
            
            # 创建索引
            await self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"MongoDB连接失败: {e}")
            raise
        except Exception as e:
            logger.error(f"MongoDB连接异常: {e}")
            raise
    
    async def connect_redis(self) -> None:
        """连接Redis数据库"""
        try:
            self.redis_client = aioredis.from_url(
                self._redis_url,
                encoding='utf-8',
                decode_responses=True,
                max_connections=20,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            
            # 测试连接
            await self.redis_client.ping()
            logger.info(f"Redis连接成功: {self._redis_url}")
            
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    async def _create_indexes(self) -> None:
        """创建数据库索引以优化查询性能"""
        try:
            # 内容集合索引
            content_collection = self.mongodb_db.contents
            
            # 基础索引
            await content_collection.create_index("id", unique=True)
            await content_collection.create_index("status")
            await content_collection.create_index("source")
            await content_collection.create_index("content_type")
            await content_collection.create_index("created_at")
            await content_collection.create_index("updated_at")
            
            # 复合索引
            await content_collection.create_index([("source", 1), ("status", 1)])
            await content_collection.create_index([("created_at", -1), ("status", 1)])
            await content_collection.create_index([("quality_score", -1), ("status", 1)])
            
            # 文本索引用于搜索
            await content_collection.create_index([
                ("title", "text"),
                ("content", "text"),
                ("keywords", "text")
            ])
            
            # 去重索引
            await content_collection.create_index("content_hash", sparse=True)
            await content_collection.create_index("similarity_hash", sparse=True)
            
            # 爬虫任务集合索引
            crawler_collection = self.mongodb_db.crawler_tasks
            await crawler_collection.create_index("task_id", unique=True)
            await crawler_collection.create_index("status")
            await crawler_collection.create_index("platform")
            await crawler_collection.create_index("created_at")
            
            logger.info("数据库索引创建完成")
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            # 不抛出异常，索引创建失败不应该阻止服务启动
    
    async def disconnect(self) -> None:
        """断开数据库连接"""
        if self.mongodb_client:
            self.mongodb_client.close()
            logger.info("MongoDB连接已关闭")
        
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis连接已关闭")
    
    async def get_mongodb_collection(self, collection_name: str):
        """获取MongoDB集合"""
        if not self.mongodb_db:
            raise RuntimeError("MongoDB未连接")
        return self.mongodb_db[collection_name]
    
    async def get_redis_client(self) -> aioredis.Redis:
        """获取Redis客户端"""
        if not self.redis_client:
            raise RuntimeError("Redis未连接")
        return self.redis_client
    
    @asynccontextmanager
    async def mongodb_transaction(self):
        """MongoDB事务上下文管理器"""
        if not self.mongodb_client:
            raise RuntimeError("MongoDB未连接")
        
        async with await self.mongodb_client.start_session() as session:
            async with session.start_transaction():
                yield session
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "mongodb": {"status": "disconnected", "latency": None},
            "redis": {"status": "disconnected", "latency": None}
        }
        
        # MongoDB健康检查
        if self.mongodb_client:
            try:
                import time
                start_time = time.time()
                await self.mongodb_client.admin.command('ping')
                health_status["mongodb"]["status"] = "connected"
                health_status["mongodb"]["latency"] = round((time.time() - start_time) * 1000, 2)
            except Exception as e:
                health_status["mongodb"]["status"] = "error"
                health_status["mongodb"]["error"] = str(e)
        
        # Redis健康检查
        if self.redis_client:
            try:
                import time
                start_time = time.time()
                await self.redis_client.ping()
                health_status["redis"]["status"] = "connected"
                health_status["redis"]["latency"] = round((time.time() - start_time) * 1000, 2)
            except Exception as e:
                health_status["redis"]["status"] = "error"
                health_status["redis"]["error"] = str(e)
        
        return health_status


# 全局数据库管理器实例
db_manager = DatabaseManager()


async def get_database_manager() -> DatabaseManager:
    """获取数据库管理器实例"""
    return db_manager


async def init_database():
    """初始化数据库连接"""
    await db_manager.connect_mongodb()
    await db_manager.connect_redis()


async def close_database():
    """关闭数据库连接"""
    await db_manager.disconnect()