"""
依赖注入工具

提供FastAPI应用的依赖注入功能，包括数据库连接、
服务实例、配置获取等常用依赖项的管理。

主要功能：
- 数据库会话管理
- 服务实例创建和缓存
- 配置注入
- 用户认证依赖
- 请求上下文管理

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import logging
from typing import AsyncGenerator, Optional, Dict, Any
import uuid
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
import jwt

from ..config.settings import get_settings, Settings
from ..database.connection import get_database_session, get_redis_connection
from ..services.ocr_service import OCRService
from ..repositories.ocr_repository import OCRRepository
from ..engines.engine_factory import get_engine_factory, OCREngineFactory
from ..utils.cache import CacheManager
from ..utils.message_queue import MessageQueueManager
from ..utils.storage import StorageManager

# 配置日志
logger = logging.getLogger(__name__)

# 全局配置实例
settings = get_settings()


# 基础依赖项

async def get_database_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    数据库会话依赖
    
    提供数据库会话的生命周期管理，确保每个请求
    都有独立的数据库会话，并在请求结束时正确关闭。
    
    Yields:
        数据库会话实例
    """
    session = None
    try:
        session = await get_database_session()
        yield session
    except Exception as e:
        if session:
            await session.rollback()
        logger.error(f"数据库会话异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库连接异常"
        )
    finally:
        if session:
            await session.close()


async def get_redis_connection_dependency() -> AsyncGenerator[aioredis.Redis, None]:
    """
    Redis连接依赖
    
    提供Redis连接的生命周期管理。
    
    Yields:
        Redis连接实例
    """
    redis_client = None
    try:
        redis_client = await get_redis_connection()
        yield redis_client
    except Exception as e:
        logger.error(f"Redis连接异常: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="缓存服务异常"
        )
    finally:
        if redis_client:
            await redis_client.close()


def get_settings_dependency() -> Settings:
    """
    配置依赖
    
    Returns:
        应用配置实例
    """
    return settings


# 服务层依赖项

@lru_cache()
def get_cache_manager() -> CacheManager:
    """
    缓存管理器依赖
    
    Returns:
        缓存管理器实例
    """
    return CacheManager(settings.redis.redis_url)


@lru_cache()
def get_storage_manager() -> StorageManager:
    """
    存储管理器依赖
    
    Returns:
        存储管理器实例
    """
    return StorageManager(
        endpoint=settings.minio.MINIO_ENDPOINT,
        access_key=settings.minio.MINIO_ACCESS_KEY,
        secret_key=settings.minio.MINIO_SECRET_KEY.get_secret_value(),
        secure=settings.minio.MINIO_SECURE,
        bucket_name=settings.minio.MINIO_BUCKET_NAME
    )


@lru_cache()
def get_message_queue_manager() -> MessageQueueManager:
    """
    消息队列管理器依赖
    
    Returns:
        消息队列管理器实例
    """
    return MessageQueueManager(settings.rabbitmq.amqp_url)


def get_engine_factory_dependency() -> OCREngineFactory:
    """
    OCR引擎工厂依赖
    
    Returns:
        引擎工厂实例
    """
    return get_engine_factory()


async def get_ocr_repository(
    db_session: AsyncSession = Depends(get_database_session_dependency)
) -> OCRRepository:
    """
    OCR仓储依赖
    
    Args:
        db_session: 数据库会话
        
    Returns:
        OCR仓储实例
    """
    return OCRRepository(db_session)


async def get_ocr_service(
    repository: OCRRepository = Depends(get_ocr_repository),
    engine_factory: OCREngineFactory = Depends(get_engine_factory_dependency),
    cache_manager: CacheManager = Depends(get_cache_manager),
    storage_manager: StorageManager = Depends(get_storage_manager),
    message_queue: MessageQueueManager = Depends(get_message_queue_manager),
    settings: Settings = Depends(get_settings_dependency)
) -> OCRService:
    """
    OCR服务依赖
    
    Args:
        repository: OCR仓储实例
        engine_factory: 引擎工厂实例
        cache_manager: 缓存管理器
        storage_manager: 存储管理器
        message_queue: 消息队列管理器
        settings: 配置实例
        
    Returns:
        OCR服务实例
    """
    return OCRService(
        repository=repository,
        engine_factory=engine_factory,
        cache_manager=cache_manager,
        storage_manager=storage_manager,
        message_queue=message_queue,
        config=settings.dict()
    )


# 认证和授权依赖项

def get_api_key(
    x_api_key: Optional[str] = Header(None, alias=settings.security.API_KEY_HEADER)
) -> Optional[str]:
    """
    API密钥依赖
    
    Args:
        x_api_key: HTTP头中的API密钥
        
    Returns:
        API密钥或None
    """
    return x_api_key


def get_jwt_token(
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """
    JWT令牌依赖
    
    Args:
        authorization: HTTP Authorization头
        
    Returns:
        JWT令牌或None
    """
    if not authorization:
        return None
    
    # 检查Bearer格式
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
        return token
    except ValueError:
        return None


def verify_jwt_token(
    token: Optional[str] = Depends(get_jwt_token),
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    JWT令牌验证依赖
    
    Args:
        token: JWT令牌
        settings: 配置实例
        
    Returns:
        令牌载荷
        
    Raises:
        HTTPException: 令牌无效时
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证令牌",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = jwt.decode(
            token,
            settings.security.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[settings.security.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌已过期",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"}
        )


def get_current_user(
    payload: Dict[str, Any] = Depends(verify_jwt_token)
) -> Dict[str, Any]:
    """
    当前用户依赖
    
    Args:
        payload: JWT载荷
        
    Returns:
        用户信息
    """
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌中缺少用户信息"
        )
    
    return {
        "user_id": user_id,
        "username": payload.get("username"),
        "roles": payload.get("roles", []),
        "permissions": payload.get("permissions", [])
    }


def require_permissions(*required_permissions: str):
    """
    权限验证依赖工厂
    
    Args:
        *required_permissions: 需要的权限列表
        
    Returns:
        权限验证依赖函数
    """
    def permission_checker(
        user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        user_permissions = set(user.get("permissions", []))
        if not all(perm in user_permissions for perm in required_permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足"
            )
        return user
    
    return permission_checker


# 请求上下文依赖项

def get_request_id(
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID")
) -> str:
    """
    请求ID依赖
    
    Args:
        request: FastAPI请求对象
        x_request_id: HTTP头中的请求ID
        
    Returns:
        请求ID字符串
    """
    if x_request_id:
        return x_request_id
    
    # 如果没有提供请求ID，生成一个新的
    request_id = str(uuid.uuid4())
    
    # 将请求ID存储到请求状态中
    request.state.request_id = request_id
    
    return request_id


def get_client_info(
    request: Request,
    user_agent: Optional[str] = Header(None, alias="User-Agent"),
    x_forwarded_for: Optional[str] = Header(None, alias="X-Forwarded-For")
) -> Dict[str, Any]:
    """
    客户端信息依赖
    
    Args:
        request: FastAPI请求对象
        user_agent: 用户代理字符串
        x_forwarded_for: 转发IP地址
        
    Returns:
        客户端信息字典
    """
    client_host = request.client.host if request.client else "unknown"
    
    # 如果有转发头，使用第一个IP作为真实客户端IP
    if x_forwarded_for:
        client_host = x_forwarded_for.split(",")[0].strip()
    
    return {
        "ip": client_host,
        "user_agent": user_agent,
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params)
    }


# 可选的认证依赖项

def optional_auth(
    token: Optional[str] = Depends(get_jwt_token)
) -> Optional[Dict[str, Any]]:
    """
    可选认证依赖
    
    不会抛出异常，如果没有令牌或令牌无效，返回None
    
    Args:
        token: JWT令牌
        
    Returns:
        用户信息或None
    """
    if not token:
        return None
    
    try:
        payload = jwt.decode(
            token,
            settings.security.JWT_SECRET_KEY.get_secret_value(),
            algorithms=[settings.security.JWT_ALGORITHM]
        )
        return payload
    except jwt.JWTError:
        return None


# 健康检查依赖项

async def check_database_health(
    db_session: AsyncSession = Depends(get_database_session_dependency)
) -> bool:
    """
    数据库健康检查依赖
    
    Args:
        db_session: 数据库会话
        
    Returns:
        数据库是否健康
    """
    try:
        # 执行简单的查询来检查数据库连接
        await db_session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"数据库健康检查失败: {str(e)}")
        return False


async def check_redis_health(
    redis_client: aioredis.Redis = Depends(get_redis_connection_dependency)
) -> bool:
    """
    Redis健康检查依赖
    
    Args:
        redis_client: Redis客户端
        
    Returns:
        Redis是否健康
    """
    try:
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis健康检查失败: {str(e)}")
        return False


# 限流依赖项

class RateLimiter:
    """简单的内存限流器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}
    
    async def is_allowed(self, key: str) -> bool:
        """
        检查是否允许请求
        
        Args:
            key: 限流键（通常是IP地址）
            
        Returns:
            是否允许请求
        """
        import time
        
        now = time.time()
        window_start = now - self.window_seconds
        
        if key not in self.requests:
            self.requests[key] = []
        
        # 清理过期的请求记录
        self.requests[key] = [
            req_time for req_time in self.requests[key] 
            if req_time > window_start
        ]
        
        # 检查是否超过限制
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # 记录当前请求
        self.requests[key].append(now)
        return True


# 全局限流器实例
rate_limiter = RateLimiter(
    max_requests=settings.security.RATE_LIMIT_PER_MINUTE,
    window_seconds=60
)


async def check_rate_limit(
    client_info: Dict[str, Any] = Depends(get_client_info)
) -> bool:
    """
    限流检查依赖
    
    Args:
        client_info: 客户端信息
        
    Returns:
        是否通过限流检查
        
    Raises:
        HTTPException: 超过限流时
    """
    client_ip = client_info["ip"]
    
    if not await rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试"
        )
    
    return True