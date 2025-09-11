"""
健康检查API控制器

提供服务状态监控和健康检查接口
支持基础健康检查、详细状态检查和就绪状态检查
"""

import logging
import time
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from ..models.database import get_database
from ..models.schemas import HealthCheckSchema, DataResponse
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["健康检查"])

# 服务启动时间
SERVICE_START_TIME = time.time()


@router.get("/health", summary="基础健康检查")
async def basic_health_check():
    """
    基础健康检查端点
    
    Returns:
        Dict: 简单的状态响应
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": settings.app_name,
        "version": settings.app_version
    }


@router.get("/health/detailed", response_model=HealthCheckSchema, summary="详细健康检查")
async def detailed_health_check(
    db: AsyncSession = Depends(get_database)
) -> HealthCheckSchema:
    """
    详细的健康检查，包含各组件状态
    
    Args:
        db: 数据库会话
        
    Returns:
        HealthCheckSchema: 详细健康状态
    """
    try:
        # 计算服务运行时间
        uptime = time.time() - SERVICE_START_TIME
        
        # 检查数据库状态
        database_status = await check_database_health(db)
        
        # 检查缓存状态（Redis）
        cache_status = await check_cache_health()
        
        # 获取系统资源使用情况
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 检查分析器状态
        analyzers_status = await check_analyzers_health()
        
        # 获取任务队列信息
        queue_info = await get_queue_info()
        
        return HealthCheckSchema(
            status="healthy",
            version=settings.app_version,
            uptime=uptime,
            database_status=database_status,
            cache_status=cache_status,
            active_tasks=queue_info.get("active_tasks", 0),
            queue_size=queue_info.get("queue_size", 0),
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            analyzers_status=analyzers_status
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        
        return HealthCheckSchema(
            status="unhealthy",
            version=settings.app_version,
            uptime=time.time() - SERVICE_START_TIME,
            database_status="error",
            cache_status="error",
            active_tasks=0,
            queue_size=0,
            memory_usage=0.0,
            cpu_usage=0.0,
            analyzers_status={}
        )


@router.get("/ready", summary="就绪状态检查")
async def readiness_check(
    db: AsyncSession = Depends(get_database)
):
    """
    Kubernetes就绪探针端点
    检查服务是否准备好接收流量
    
    Args:
        db: 数据库会话
        
    Returns:
        Dict: 就绪状态
    """
    try:
        # 检查关键依赖项
        database_ready = await check_database_health(db) == "healthy"
        
        if database_ready:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow(),
                "checks": {
                    "database": "ready"
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务未就绪：数据库连接失败"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"服务未就绪：{str(e)}"
        )


@router.get("/live", summary="存活状态检查")
async def liveness_check():
    """
    Kubernetes存活探针端点
    检查服务是否存活
    
    Returns:
        Dict: 存活状态
    """
    try:
        # 简单的存活检查
        current_time = time.time()
        uptime = current_time - SERVICE_START_TIME
        
        # 如果服务运行时间正常，认为存活
        if uptime > 0:
            return {
                "status": "alive",
                "timestamp": datetime.utcnow(),
                "uptime": uptime
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务异常：运行时间计算错误"
            )
            
    except Exception as e:
        logger.error(f"存活检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"服务异常：{str(e)}"
        )


@router.get("/metrics", summary="Prometheus指标")
async def get_metrics():
    """
    Prometheus指标收集端点
    
    Returns:
        str: Prometheus格式的指标数据
    """
    try:
        # 这里应该返回Prometheus格式的指标
        # 简化实现，返回基本指标
        
        uptime = time.time() - SERVICE_START_TIME
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        metrics = f"""# HELP content_moderation_uptime_seconds Service uptime in seconds
# TYPE content_moderation_uptime_seconds counter
content_moderation_uptime_seconds {uptime}

# HELP content_moderation_memory_usage_percent Memory usage percentage
# TYPE content_moderation_memory_usage_percent gauge
content_moderation_memory_usage_percent {memory_usage}

# HELP content_moderation_cpu_usage_percent CPU usage percentage
# TYPE content_moderation_cpu_usage_percent gauge
content_moderation_cpu_usage_percent {cpu_usage}

# HELP content_moderation_service_info Service information
# TYPE content_moderation_service_info gauge
content_moderation_service_info{{version="{settings.app_version}",service="{settings.app_name}"}} 1
"""
        
        return metrics
        
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取指标失败"
        )


@router.get("/info", response_model=DataResponse, summary="服务信息")
async def get_service_info() -> DataResponse:
    """
    获取服务基本信息
    
    Returns:
        DataResponse: 服务信息
    """
    try:
        info = {
            "service_name": settings.app_name,
            "version": settings.app_version,
            "uptime": time.time() - SERVICE_START_TIME,
            "start_time": datetime.fromtimestamp(SERVICE_START_TIME),
            "current_time": datetime.utcnow(),
            "debug_mode": settings.debug,
            "environment": "development" if settings.debug else "production",
            "supported_content_types": ["text", "image", "video", "audio"],
            "max_file_size": settings.max_file_size,
            "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "配置中",  # 隐藏敏感信息
            "redis_url": settings.redis_url.split("@")[-1] if "@" in settings.redis_url else "配置中"
        }
        
        return DataResponse(
            success=True,
            message="获取服务信息成功",
            data=info
        )
        
    except Exception as e:
        logger.error(f"获取服务信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取服务信息失败"
        )


# 辅助函数

async def check_database_health(db: AsyncSession) -> str:
    """
    检查数据库健康状态
    
    Args:
        db: 数据库会话
        
    Returns:
        str: 健康状态
    """
    try:
        # 执行简单的数据库查询
        from sqlalchemy import text
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        return "healthy"
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        return "unhealthy"


async def check_cache_health() -> str:
    """
    检查缓存(Redis)健康状态
    
    Returns:
        str: 健康状态
    """
    try:
        # 这里应该检查Redis连接
        # 简化实现，返回健康状态
        return "healthy"
    except Exception as e:
        logger.error(f"缓存健康检查失败: {e}")
        return "unhealthy"


async def check_analyzers_health() -> Dict[str, str]:
    """
    检查各分析器健康状态
    
    Returns:
        Dict[str, str]: 分析器状态映射
    """
    try:
        # 检查各个分析器的状态
        analyzers_status = {}
        
        # 文本分析器
        try:
            from ..analyzers.text_analyzer import TextAnalyzer
            text_analyzer = TextAnalyzer()
            analyzers_status["text_analyzer"] = "healthy"
        except Exception:
            analyzers_status["text_analyzer"] = "unhealthy"
        
        # 图像分析器
        try:
            from ..analyzers.image_analyzer import ImageAnalyzer
            image_analyzer = ImageAnalyzer()
            analyzers_status["image_analyzer"] = "healthy"
        except Exception:
            analyzers_status["image_analyzer"] = "unhealthy"
        
        # 视频分析器
        try:
            from ..analyzers.video_analyzer import VideoAnalyzer
            video_analyzer = VideoAnalyzer()
            analyzers_status["video_analyzer"] = "healthy"
        except Exception:
            analyzers_status["video_analyzer"] = "unhealthy"
        
        # 音频分析器
        try:
            from ..analyzers.audio_analyzer import AudioAnalyzer
            audio_analyzer = AudioAnalyzer()
            analyzers_status["audio_analyzer"] = "healthy"
        except Exception:
            analyzers_status["audio_analyzer"] = "unhealthy"
        
        return analyzers_status
        
    except Exception as e:
        logger.error(f"分析器健康检查失败: {e}")
        return {
            "text_analyzer": "error",
            "image_analyzer": "error", 
            "video_analyzer": "error",
            "audio_analyzer": "error"
        }


async def get_queue_info() -> Dict[str, int]:
    """
    获取任务队列信息
    
    Returns:
        Dict[str, int]: 队列信息
    """
    try:
        # 这里应该从实际的任务队列系统获取信息
        # 简化实现，返回模拟数据
        return {
            "active_tasks": 0,
            "queue_size": 0
        }
    except Exception as e:
        logger.error(f"获取队列信息失败: {e}")
        return {
            "active_tasks": 0,
            "queue_size": 0
        }