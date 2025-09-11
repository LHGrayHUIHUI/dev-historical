"""
健康检查API控制器

提供服务健康状态检查接口
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import get_database
from ..services.redis_service import redis_service
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/")
async def health_check():
    """
    基础健康检查
    
    返回服务基本状态
    """
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_database)):
    """
    就绪状态检查
    
    检查数据库和Redis连接状态
    """
    checks = {
        "database": False,
        "redis": False
    }
    
    # 检查数据库连接
    try:
        await db.execute("SELECT 1")
        checks["database"] = True
    except Exception as e:
        logger.error(f"数据库连接检查失败: {e}")
    
    # 检查Redis连接
    try:
        redis = await redis_service.get_redis()
        await redis.ping()
        checks["redis"] = True
    except Exception as e:
        logger.error(f"Redis连接检查失败: {e}")
    
    is_ready = all(checks.values())
    
    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus指标端点
    
    返回基本的服务指标
    """
    # 这里应该返回Prometheus格式的指标
    # 简化实现
    return {
        "service_info": {
            "name": settings.app_name,
            "version": settings.app_version
        },
        "timestamp": datetime.utcnow().isoformat()
    }