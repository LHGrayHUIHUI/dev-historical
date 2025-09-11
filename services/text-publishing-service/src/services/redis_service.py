"""
Redis缓存服务

提供Redis连接管理和常用缓存操作
支持任务状态缓存、配额管理和限流控制
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import aioredis
from aioredis import Redis

from ..config.settings import settings

logger = logging.getLogger(__name__)


class RedisService:
    """
    Redis服务类
    
    封装Redis操作，提供缓存管理和实时数据存储
    """
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.connection_pool = None
    
    async def connect(self):
        """建立Redis连接"""
        try:
            self.connection_pool = aioredis.ConnectionPool.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                socket_timeout=settings.redis_socket_timeout,
                encoding="utf-8",
                decode_responses=True
            )
            self.redis = aioredis.Redis(connection_pool=self.connection_pool)
            
            # 测试连接
            await self.redis.ping()
            logger.info("Redis连接建立成功")
            
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    async def disconnect(self):
        """关闭Redis连接"""
        if self.connection_pool:
            await self.connection_pool.disconnect()
            logger.info("Redis连接已关闭")
    
    async def get_redis(self) -> Redis:
        """获取Redis客户端实例"""
        if self.redis is None:
            await self.connect()
        return self.redis
    
    # ==================== 任务状态缓存 ====================
    
    async def set_task_status(
        self, 
        task_uuid: str, 
        status: str, 
        progress: int = 0,
        message: str = "",
        ttl: int = 3600
    ):
        """
        设置任务状态缓存
        
        Args:
            task_uuid: 任务UUID
            status: 任务状态
            progress: 进度百分比
            message: 状态消息
            ttl: 过期时间(秒)
        """
        redis = await self.get_redis()
        key = f"task_status:{task_uuid}"
        
        data = {
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        await redis.hset(key, mapping=data)
        await redis.expire(key, ttl)
    
    async def get_task_status(self, task_uuid: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态缓存
        
        Args:
            task_uuid: 任务UUID
            
        Returns:
            Dict[str, Any]: 任务状态信息
        """
        redis = await self.get_redis()
        key = f"task_status:{task_uuid}"
        
        data = await redis.hgetall(key)
        if not data:
            return None
        
        return {
            "task_uuid": task_uuid,
            "status": data.get("status"),
            "progress": int(data.get("progress", 0)),
            "message": data.get("message", ""),
            "updated_at": data.get("updated_at")
        }
    
    async def delete_task_status(self, task_uuid: str):
        """删除任务状态缓存"""
        redis = await self.get_redis()
        key = f"task_status:{task_uuid}"
        await redis.delete(key)
    
    # ==================== 账号配额管理 ====================
    
    async def set_account_quota(
        self, 
        account_id: int, 
        daily_limit: int,
        used_today: int = 0,
        reset_time: Optional[datetime] = None
    ):
        """
        设置账号配额信息
        
        Args:
            account_id: 账号ID
            daily_limit: 每日限制
            used_today: 今日已使用
            reset_time: 重置时间
        """
        redis = await self.get_redis()
        key = f"account_quota:{account_id}"
        
        if reset_time is None:
            # 默认明天凌晨重置
            tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            reset_time = tomorrow
        
        data = {
            "daily_limit": daily_limit,
            "used_today": used_today,
            "reset_time": reset_time.isoformat()
        }
        
        # 设置到明天凌晨过期
        ttl = int((reset_time - datetime.now()).total_seconds())
        
        await redis.hset(key, mapping=data)
        await redis.expire(key, max(ttl, 3600))  # 至少1小时
    
    async def get_account_quota(self, account_id: int) -> Optional[Dict[str, Any]]:
        """
        获取账号配额信息
        
        Args:
            account_id: 账号ID
            
        Returns:
            Dict[str, Any]: 配额信息
        """
        redis = await self.get_redis()
        key = f"account_quota:{account_id}"
        
        data = await redis.hgetall(key)
        if not data:
            return None
        
        return {
            "daily_limit": int(data.get("daily_limit", 0)),
            "used_today": int(data.get("used_today", 0)),
            "reset_time": data.get("reset_time")
        }
    
    async def increment_account_usage(self, account_id: int, increment: int = 1) -> int:
        """
        增加账号使用次数
        
        Args:
            account_id: 账号ID
            increment: 增加数量
            
        Returns:
            int: 更新后的使用次数
        """
        redis = await self.get_redis()
        key = f"account_quota:{account_id}"
        
        new_value = await redis.hincrby(key, "used_today", increment)
        return new_value
    
    # ==================== 平台限流管理 ====================
    
    async def check_platform_rate_limit(
        self, 
        platform_id: int, 
        limit_per_hour: int
    ) -> bool:
        """
        检查平台限流状态
        
        Args:
            platform_id: 平台ID
            limit_per_hour: 每小时限制
            
        Returns:
            bool: 是否在限制内
        """
        redis = await self.get_redis()
        key = f"platform_rate_limit:{platform_id}"
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        window_key = f"{key}:{current_hour.isoformat()}"
        
        current_count = await redis.get(window_key)
        current_count = int(current_count) if current_count else 0
        
        return current_count < limit_per_hour
    
    async def increment_platform_usage(self, platform_id: int) -> int:
        """
        增加平台使用次数
        
        Args:
            platform_id: 平台ID
            
        Returns:
            int: 当前小时的使用次数
        """
        redis = await self.get_redis()
        key = f"platform_rate_limit:{platform_id}"
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        window_key = f"{key}:{current_hour.isoformat()}"
        
        # 增加计数
        new_count = await redis.incr(window_key)
        
        # 设置1小时过期
        if new_count == 1:
            await redis.expire(window_key, 3600)
        
        return new_count
    
    async def get_platform_usage(self, platform_id: int) -> Dict[str, Any]:
        """
        获取平台使用情况
        
        Args:
            platform_id: 平台ID
            
        Returns:
            Dict[str, Any]: 使用情况
        """
        redis = await self.get_redis()
        key = f"platform_rate_limit:{platform_id}"
        
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        window_key = f"{key}:{current_hour.isoformat()}"
        
        current_count = await redis.get(window_key)
        current_count = int(current_count) if current_count else 0
        
        return {
            "current_hour": current_hour.isoformat(),
            "current_requests": current_count,
            "window_key": window_key
        }
    
    # ==================== 平台健康状态 ====================
    
    async def set_platform_health(
        self,
        platform_id: int,
        status: str = "healthy",
        response_time: float = 0.0,
        error_rate: float = 0.0,
        ttl: int = 300  # 5分钟
    ):
        """
        设置平台健康状态
        
        Args:
            platform_id: 平台ID
            status: 健康状态
            response_time: 响应时间(ms)
            error_rate: 错误率
            ttl: 过期时间(秒)
        """
        redis = await self.get_redis()
        key = f"platform_health:{platform_id}"
        
        data = {
            "status": status,
            "last_check": datetime.utcnow().isoformat(),
            "response_time": response_time,
            "error_rate": error_rate
        }
        
        await redis.hset(key, mapping=data)
        await redis.expire(key, ttl)
    
    async def get_platform_health(self, platform_id: int) -> Optional[Dict[str, Any]]:
        """
        获取平台健康状态
        
        Args:
            platform_id: 平台ID
            
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        redis = await self.get_redis()
        key = f"platform_health:{platform_id}"
        
        data = await redis.hgetall(key)
        if not data:
            return None
        
        return {
            "platform_id": platform_id,
            "status": data.get("status"),
            "last_check": data.get("last_check"),
            "response_time": float(data.get("response_time", 0)),
            "error_rate": float(data.get("error_rate", 0))
        }
    
    # ==================== 通用缓存操作 ====================
    
    async def set_cache(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600
    ):
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)
        """
        redis = await self.get_redis()
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        
        await redis.setex(key, ttl, value)
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            Any: 缓存值
        """
        redis = await self.get_redis()
        value = await redis.get(key)
        
        if value is None:
            return None
        
        # 尝试解析JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    async def delete_cache(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        redis = await self.get_redis()
        result = await redis.delete(key)
        return result > 0
    
    async def exists_cache(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在
        """
        redis = await self.get_redis()
        return await redis.exists(key) > 0
    
    async def get_cache_ttl(self, key: str) -> int:
        """
        获取缓存剩余时间
        
        Args:
            key: 缓存键
            
        Returns:
            int: 剩余秒数，-1表示永不过期，-2表示不存在
        """
        redis = await self.get_redis()
        return await redis.ttl(key)
    
    # ==================== 批量操作 ====================
    
    async def set_multiple(self, mapping: Dict[str, Any], ttl: int = 3600):
        """
        批量设置缓存
        
        Args:
            mapping: 键值对映射
            ttl: 过期时间(秒)
        """
        redis = await self.get_redis()
        
        pipe = redis.pipeline()
        for key, value in mapping.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            pipe.setex(key, ttl, value)
        
        await pipe.execute()
    
    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存
        
        Args:
            keys: 键列表
            
        Returns:
            Dict[str, Any]: 键值对结果
        """
        redis = await self.get_redis()
        values = await redis.mget(keys)
        
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[key] = value
            else:
                result[key] = None
        
        return result
    
    async def delete_multiple(self, keys: List[str]) -> int:
        """
        批量删除缓存
        
        Args:
            keys: 键列表
            
        Returns:
            int: 删除的键数量
        """
        redis = await self.get_redis()
        if keys:
            return await redis.delete(*keys)
        return 0


# 全局Redis服务实例
redis_service = RedisService()