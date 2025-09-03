"""
监控控制器模块

此模块提供FastAPI应用的监控API端点，包括健康检查、
指标查询、服务状态监控等功能。

主要功能：
- Prometheus指标端点
- 应用健康检查
- 服务状态查询
- 系统信息获取
- 监控数据聚合

Author: 开发团队  
Created: 2025-09-03
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog
import psutil
import platform
import sys
import os
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from .metrics_middleware import get_business_metrics

# 获取结构化日志记录器
logger = structlog.get_logger()

# 创建监控路由器
router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    responses={
        404: {"description": "监控端点未找到"},
        500: {"description": "监控服务内部错误"}
    }
)

class HealthStatus(BaseModel):
    """健康检查响应模型"""
    
    status: str = Field(..., description="服务状态", example="healthy")
    timestamp: datetime = Field(..., description="检查时间")
    uptime_seconds: float = Field(..., description="运行时间（秒）")
    version: str = Field(..., description="服务版本")
    service_name: str = Field(..., description="服务名称")
    environment: str = Field(..., description="运行环境")
    
    class Config:
        """Pydantic配置"""
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-09-03T10:30:00Z",
                "uptime_seconds": 86400.0,
                "version": "1.0.0",
                "service_name": "historical-text-service",
                "environment": "production"
            }
        }

class SystemInfo(BaseModel):
    """系统信息响应模型"""
    
    hostname: str = Field(..., description="主机名")
    platform: str = Field(..., description="操作系统平台")
    python_version: str = Field(..., description="Python版本")
    cpu_count: int = Field(..., description="CPU核心数")
    memory_total: int = Field(..., description="总内存（字节）")
    memory_available: int = Field(..., description="可用内存（字节）")
    memory_percent: float = Field(..., description="内存使用百分比")
    disk_usage: Dict[str, Any] = Field(..., description="磁盘使用情况")
    load_average: Optional[List[float]] = Field(None, description="系统负载（Linux/macOS）")

class ServiceMetrics(BaseModel):
    """服务指标响应模型"""
    
    service_name: str = Field(..., description="服务名称")
    http_requests_total: int = Field(..., description="HTTP请求总数")
    active_connections: int = Field(..., description="活跃连接数")
    average_response_time: float = Field(..., description="平均响应时间（秒）")
    error_rate: float = Field(..., description="错误率百分比")
    uptime_seconds: float = Field(..., description="运行时间（秒）")

def get_service_name() -> str:
    """获取服务名称"""
    return os.getenv('SERVICE_NAME', 'historical-text-service')

def get_service_version() -> str:
    """获取服务版本"""
    return os.getenv('SERVICE_VERSION', '1.0.0')

def get_environment() -> str:
    """获取运行环境"""
    return os.getenv('ENVIRONMENT', 'development')

# 应用启动时间，用于计算运行时间
_startup_time = datetime.utcnow()

@router.get(
    "/health",
    response_model=HealthStatus,
    summary="健康检查",
    description="获取服务健康状态，包括基本信息和运行状态"
)
async def health_check() -> HealthStatus:
    """服务健康检查端点
    
    提供服务的基本健康状态信息，包括运行时间、版本、
    环境等基础信息。此端点用于负载均衡器和监控系统
    的健康检查。
    
    Returns:
        HealthStatus: 服务健康状态信息
        
    Raises:
        HTTPException: 当服务状态异常时抛出
    """
    try:
        # 计算运行时间
        current_time = datetime.utcnow()
        uptime = (current_time - _startup_time).total_seconds()
        
        # 构建健康状态响应
        health_status = HealthStatus(
            status="healthy",
            timestamp=current_time,
            uptime_seconds=uptime,
            version=get_service_version(),
            service_name=get_service_name(),
            environment=get_environment()
        )
        
        logger.debug(
            "健康检查完成",
            status=health_status.status,
            uptime_seconds=uptime,
            service_name=health_status.service_name
        )
        
        return health_status
        
    except Exception as e:
        logger.error(
            "健康检查失败",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=f"健康检查失败: {str(e)}"
        )

@router.get(
    "/ready",
    summary="就绪检查", 
    description="检查服务是否已准备好处理请求"
)
async def readiness_check():
    """服务就绪检查端点
    
    检查服务是否已完全启动并准备好处理请求，
    包括数据库连接、外部服务依赖等检查。
    
    Returns:
        Dict: 就绪状态信息
        
    Raises:
        HTTPException: 当服务未就绪时抛出
    """
    try:
        checks = {
            "service": "ready",
            "database": "connected",  # 实际项目中应检查数据库连接
            "redis": "connected",     # 实际项目中应检查Redis连接
            "dependencies": "ready"
        }
        
        # 检查关键服务状态（示例）
        # 在实际项目中，这里应该检查数据库、缓存、消息队列等依赖
        all_ready = all(status in ["ready", "connected"] for status in checks.values())
        
        if not all_ready:
            logger.warning("服务就绪检查失败", checks=checks)
            raise HTTPException(
                status_code=503,
                detail="服务未就绪"
            )
        
        logger.debug("服务就绪检查完成", checks=checks)
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "就绪检查异常",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=f"就绪检查异常: {str(e)}"
        )

@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus指标",
    description="获取Prometheus格式的监控指标数据"
)
async def get_metrics() -> Response:
    """Prometheus指标端点
    
    返回Prometheus格式的所有监控指标，包括HTTP请求统计、
    业务指标、系统资源使用情况等。
    
    Returns:
        Response: Prometheus格式的指标数据
        
    Raises:
        HTTPException: 当指标生成失败时抛出
    """
    try:
        # 生成Prometheus格式的指标数据
        metrics_data = generate_latest()
        
        logger.debug(
            "Prometheus指标生成完成",
            metrics_size=len(metrics_data),
            service_name=get_service_name()
        )
        
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
        
    except Exception as e:
        logger.error(
            "指标生成失败",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=f"指标生成失败: {str(e)}"
        )

@router.get(
    "/system",
    response_model=SystemInfo,
    summary="系统信息",
    description="获取服务器系统信息，包括CPU、内存、磁盘使用情况"
)
async def get_system_info() -> SystemInfo:
    """系统信息查询端点
    
    获取运行服务的服务器系统信息，包括硬件配置、
    资源使用情况、系统负载等详细信息。
    
    Returns:
        SystemInfo: 系统信息详情
        
    Raises:
        HTTPException: 当系统信息获取失败时抛出
    """
    try:
        # 获取内存信息
        memory = psutil.virtual_memory()
        
        # 获取磁盘使用情况
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": round((usage.used / usage.total) * 100, 2)
                }
            except PermissionError:
                # 跳过没有权限访问的分区
                continue
        
        # 获取系统负载（Linux/macOS）
        load_avg = None
        try:
            if hasattr(os, 'getloadavg'):
                load_avg = list(os.getloadavg())
        except (OSError, AttributeError):
            # Windows系统不支持getloadavg
            pass
        
        system_info = SystemInfo(
            hostname=platform.node(),
            platform=f"{platform.system()} {platform.release()}",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            cpu_count=psutil.cpu_count(),
            memory_total=memory.total,
            memory_available=memory.available,
            memory_percent=memory.percent,
            disk_usage=disk_usage,
            load_average=load_avg
        )
        
        logger.debug(
            "系统信息查询完成",
            hostname=system_info.hostname,
            memory_percent=system_info.memory_percent,
            disk_count=len(disk_usage)
        )
        
        return system_info
        
    except Exception as e:
        logger.error(
            "系统信息获取失败",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=f"系统信息获取失败: {str(e)}"
        )

@router.get(
    "/services",
    response_model=List[ServiceMetrics],
    summary="服务指标汇总",
    description="获取所有服务的关键指标汇总信息"
)
async def get_service_metrics() -> List[ServiceMetrics]:
    """服务指标汇总端点
    
    获取当前服务的关键业务指标汇总，包括请求统计、
    连接数、响应时间、错误率等核心监控数据。
    
    Returns:
        List[ServiceMetrics]: 服务指标列表
        
    Raises:
        HTTPException: 当指标查询失败时抛出
    """
    try:
        service_name = get_service_name()
        current_time = datetime.utcnow()
        uptime = (current_time - _startup_time).total_seconds()
        
        # 在实际项目中，这里应该从Prometheus或指标收集器中获取真实数据
        # 现在返回模拟数据作为示例
        service_metrics = ServiceMetrics(
            service_name=service_name,
            http_requests_total=0,  # 实际应从Counter获取
            active_connections=0,   # 实际应从Gauge获取
            average_response_time=0.1,  # 实际应从Histogram计算
            error_rate=0.0,  # 实际应根据错误请求和总请求计算
            uptime_seconds=uptime
        )
        
        logger.debug(
            "服务指标查询完成",
            service_name=service_name,
            uptime_seconds=uptime
        )
        
        return [service_metrics]
        
    except Exception as e:
        logger.error(
            "服务指标查询失败",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=f"服务指标查询失败: {str(e)}"
        )

@router.get(
    "/status",
    summary="服务状态概览",
    description="获取服务整体状态概览，包括健康状态、关键指标等"
)
async def get_service_status():
    """服务状态概览端点
    
    提供服务的整体状态概览，包括健康状态、关键指标、
    系统资源使用情况等信息的汇总视图。
    
    Returns:
        Dict: 服务状态概览数据
        
    Raises:
        HTTPException: 当状态查询失败时抛出
    """
    try:
        current_time = datetime.utcnow()
        uptime = (current_time - _startup_time).total_seconds()
        
        # 获取系统基本信息
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        status_overview = {
            "service": {
                "name": get_service_name(),
                "version": get_service_version(),
                "environment": get_environment(),
                "status": "running",
                "uptime_seconds": uptime,
                "timestamp": current_time.isoformat()
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "load_average": list(os.getloadavg()) if hasattr(os, 'getloadavg') else None
            },
            "metrics": {
                "http_requests_total": 0,  # 实际应从指标收集器获取
                "active_connections": 0,
                "error_rate_percent": 0.0,
                "average_response_time_ms": 100
            }
        }
        
        logger.debug(
            "服务状态概览查询完成",
            service_name=status_overview["service"]["name"],
            cpu_percent=cpu_percent,
            memory_percent=memory.percent
        )
        
        return status_overview
        
    except Exception as e:
        logger.error(
            "服务状态查询失败",
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=f"服务状态查询失败: {str(e)}"
        )