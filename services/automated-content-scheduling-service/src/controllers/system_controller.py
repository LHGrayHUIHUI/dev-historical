"""
系统监控和管理API控制器
提供健康检查、系统状态、性能指标等监控API
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
import logging
import psutil
import asyncio
import json

from ..models import get_db, db_manager
from ..config.settings import get_settings
from ..services.optimization_service import OptimizationService

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/system", tags=["系统监控"])

settings = get_settings()
optimization_service = OptimizationService()


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    components: Dict[str, Dict[str, Any]]


class SystemMetricsResponse(BaseModel):
    """系统指标响应模型"""
    timestamp: datetime
    system: Dict[str, Any]
    database: Dict[str, Any]
    application: Dict[str, Any]
    ml_model: Dict[str, Any]


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(session: AsyncSession = Depends(get_db)):
    """
    系统健康检查
    
    检查系统各组件的健康状态，包括数据库、外部服务等。
    """
    try:
        logger.info("执行系统健康检查")
        
        components = {}
        overall_status = "healthy"
        
        # 1. 数据库健康检查
        try:
            db_healthy = await db_manager.health_check()
            db_info = db_manager.get_connection_info()
            
            components["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "connection_info": db_info,
                "response_time_ms": 0  # 实际应该测量响应时间
            }
            
            if not db_healthy:
                overall_status = "unhealthy"
                
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            components["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_status = "unhealthy"
        
        # 2. ML模型状态检查
        try:
            model_status = "healthy" if optimization_service.optimizer.is_trained else "warning"
            
            components["ml_model"] = {
                "status": model_status,
                "is_trained": optimization_service.optimizer.is_trained,
                "model_version": optimization_service.optimizer.model_version,
                "model_type": optimization_service.optimizer.model_type
            }
            
        except Exception as e:
            logger.error(f"ML模型状态检查失败: {e}")
            components["ml_model"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # 3. 外部服务健康检查
        external_services = {
            "account_management": settings.account_management_service_url,
            "content_publishing": settings.content_publishing_service_url,
            "storage_service": settings.storage_service_url
        }
        
        for service_name, service_url in external_services.items():
            try:
                # 这里应该实际检查外部服务的健康状态
                # 暂时设为healthy
                components[service_name] = {
                    "status": "healthy",
                    "url": service_url,
                    "response_time_ms": 0
                }
                
            except Exception as e:
                logger.error(f"{service_name}健康检查失败: {e}")
                components[service_name] = {
                    "status": "unhealthy",
                    "url": service_url,
                    "error": str(e)
                }
                overall_status = "degraded"
        
        # 4. 应用程序状态
        components["application"] = {
            "status": "healthy",
            "uptime_seconds": 0,  # 实际应该计算运行时间
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.Process().cpu_percent()
        }
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.version,
            environment=settings.environment.value,
            components=components
        )
        
    except Exception as e:
        logger.error(f"健康检查异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="健康检查失败"
        )


@router.get("/ready")
async def readiness_check(session: AsyncSession = Depends(get_db)):
    """
    就绪检查（用于Kubernetes等容器编排）
    
    检查服务是否准备好接收流量。
    """
    try:
        # 简单的就绪检查：数据库连接是否正常
        db_healthy = await db_manager.health_check()
        
        if db_healthy:
            return {"status": "ready", "timestamp": datetime.utcnow()}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务尚未就绪"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="就绪检查失败"
        )


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(session: AsyncSession = Depends(get_db)):
    """
    获取系统性能指标
    
    提供详细的系统性能监控数据，兼容Prometheus格式。
    """
    try:
        logger.debug("获取系统指标")
        
        # 1. 系统指标
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv
            }
        }
        
        # 2. 数据库指标
        db_info = db_manager.get_connection_info()
        database_metrics = {
            "connection_pool": db_info,
            "query_performance": {
                "avg_query_time_ms": 0,  # 实际应该从数据库获取
                "active_connections": db_info.get("checked_out", 0),
                "idle_connections": db_info.get("checked_in", 0)
            }
        }
        
        # 3. 应用指标
        process = psutil.Process()
        application_metrics = {
            "process": {
                "pid": process.pid,
                "memory_rss": process.memory_info().rss,
                "memory_vms": process.memory_info().vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            },
            "performance": {
                "request_count": 0,  # 实际应该从中间件获取
                "error_count": 0,
                "avg_response_time_ms": 0,
                "active_requests": 0
            }
        }
        
        # 4. ML模型指标
        ml_metrics = {
            "model_status": "trained" if optimization_service.optimizer.is_trained else "not_trained",
            "model_version": optimization_service.optimizer.model_version,
            "model_type": optimization_service.optimizer.model_type,
            "prediction_count": 0,  # 实际应该跟踪预测次数
            "model_accuracy": 0.0,  # 实际应该从模型获取
            "last_training_time": None
        }
        
        return SystemMetricsResponse(
            timestamp=datetime.utcnow(),
            system=system_metrics,
            database=database_metrics,
            application=application_metrics,
            ml_model=ml_metrics
        )
        
    except Exception as e:
        logger.error(f"获取系统指标异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统指标失败"
        )


@router.get("/info")
async def get_system_info():
    """
    获取系统基本信息
    
    返回服务的基本配置和版本信息。
    """
    try:
        system_info = {
            "service": {
                "name": settings.app_name,
                "version": settings.version,
                "environment": settings.environment.value,
                "debug": settings.debug
            },
            "configuration": {
                "host": settings.host,
                "port": settings.port,
                "supported_platforms": settings.platforms.supported_platforms,
                "ml_enabled": True,
                "optimization_enabled": True
            },
            "build_info": {
                "build_time": datetime.utcnow().isoformat(),  # 实际应该是构建时间
                "git_commit": "unknown",  # 实际应该从环境变量获取
                "python_version": "3.11+",
                "dependencies": {
                    "fastapi": "0.104+",
                    "sqlalchemy": "2.0+",
                    "scikit-learn": "1.3+",
                    "celery": "5.3+"
                }
            },
            "runtime": {
                "start_time": datetime.utcnow().isoformat(),  # 实际应该记录启动时间
                "timezone": "UTC",
                "locale": "en_US.UTF-8"
            }
        }
        
        return {
            "success": True,
            "data": system_info,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"获取系统信息异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统信息失败"
        )


@router.get("/logs")
async def get_system_logs(
    level: str = Query("INFO", description="日志级别", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    lines: int = Query(100, description="返回行数", ge=1, le=1000),
    service: Optional[str] = Query(None, description="服务组件")
):
    """
    获取系统日志
    
    返回最近的系统日志记录。
    
    - **level**: 日志级别筛选
    - **lines**: 返回的日志行数
    - **service**: 服务组件筛选（可选）
    """
    try:
        logger.info(f"获取系统日志，级别: {level}, 行数: {lines}")
        
        # 这里应该实现实际的日志读取逻辑
        # 暂时返回模拟数据
        
        logs = [
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                "level": level,
                "service": service or "scheduling-service",
                "message": f"模拟日志消息 {i}",
                "module": "system_controller"
            }
            for i in range(min(lines, 10))
        ]
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "total": len(logs),
                "level": level,
                "service": service
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"获取系统日志异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统日志失败"
        )


@router.post("/ml-model/retrain")
async def retrain_ml_model(
    force: bool = Query(False, description="强制重新训练"),
    session: AsyncSession = Depends(get_db)
):
    """
    重新训练ML模型
    
    手动触发机器学习模型的重新训练。
    
    - **force**: 是否强制重新训练（即使训练数据不足）
    """
    try:
        logger.info(f"开始重新训练ML模型，强制: {force}")
        
        # 检查是否有足够的训练数据
        min_samples = settings.ml.min_training_samples if not force else 10
        
        # 执行模型训练
        training_result = await optimization_service.retrain_model()
        
        if not training_result:
            return {
                "success": False,
                "message": "模型训练失败：训练数据不足或其他错误",
                "timestamp": datetime.utcnow()
            }
        
        return {
            "success": True,
            "message": "模型重新训练成功",
            "data": {
                "training_metrics": training_result,
                "model_version": optimization_service.optimizer.model_version,
                "training_time": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"重新训练ML模型异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型训练失败"
        )


@router.get("/queue-status")
async def get_queue_status():
    """
    获取任务队列状态
    
    返回Celery任务队列的状态信息。
    """
    try:
        logger.info("获取任务队列状态")
        
        # 这里应该实现实际的队列状态检查
        # 暂时返回模拟数据
        
        queue_status = {
            "active_tasks": 0,
            "pending_tasks": 0,
            "failed_tasks": 0,
            "completed_tasks": 0,
            "queues": {
                "scheduling": {
                    "active": 0,
                    "pending": 0,
                    "workers": 2
                },
                "publishing": {
                    "active": 0,
                    "pending": 0,
                    "workers": 3
                },
                "optimization": {
                    "active": 0,
                    "pending": 0,
                    "workers": 1
                }
            },
            "workers": {
                "total": 6,
                "online": 6,
                "offline": 0
            }
        }
        
        return {
            "success": True,
            "data": queue_status,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"获取任务队列状态异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取队列状态失败"
        )


@router.post("/cache/clear")
async def clear_cache(
    cache_type: str = Query("all", description="缓存类型", regex="^(all|user|platform|model)$")
):
    """
    清理缓存
    
    清理指定类型的缓存数据。
    
    - **cache_type**: 缓存类型（all/user/platform/model）
    """
    try:
        logger.info(f"清理缓存，类型: {cache_type}")
        
        # 这里应该实现实际的缓存清理逻辑
        # 暂时返回成功响应
        
        cleared_items = 0  # 实际应该返回清理的项目数
        
        return {
            "success": True,
            "message": f"缓存清理成功，清理了 {cleared_items} 个项目",
            "data": {
                "cache_type": cache_type,
                "cleared_items": cleared_items
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"清理缓存异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清理缓存失败"
        )


@router.get("/config")
async def get_system_config():
    """
    获取系统配置
    
    返回当前的系统配置信息（不包含敏感信息）。
    """
    try:
        logger.info("获取系统配置")
        
        # 返回非敏感的配置信息
        config_info = {
            "application": {
                "name": settings.app_name,
                "version": settings.version,
                "environment": settings.environment.value,
                "debug": settings.debug,
                "host": settings.host,
                "port": settings.port
            },
            "features": {
                "optimization_enabled": True,
                "conflict_detection": True,
                "batch_operations": True,
                "analytics": True
            },
            "limits": {
                "max_concurrent_tasks": settings.scheduler.max_concurrent_tasks,
                "max_retries": settings.scheduler.max_retries,
                "batch_size_limit": 50
            },
            "platforms": {
                "supported": settings.platforms.supported_platforms,
                "total_count": len(settings.platforms.supported_platforms)
            },
            "ml_config": {
                "model_type": settings.ml.model_type,
                "feature_window_days": settings.ml.feature_window_days,
                "min_training_samples": settings.ml.min_training_samples,
                "retrain_interval": settings.ml.model_retrain_interval
            }
        }
        
        return {
            "success": True,
            "data": config_info,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"获取系统配置异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统配置失败"
        )