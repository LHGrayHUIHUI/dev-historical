"""
系统监控API控制器

提供系统状态监控和API统计的RESTful API接口
处理服务健康检查、性能指标、API使用统计等功能
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import psutil
from datetime import datetime, timedelta

from ..models.database import get_database
from ..models.schemas import SystemStatusSchema, ApiUsageStatsSchema, DataResponse
from ..config.settings import settings

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/v1/system",
    tags=["系统监控"],
    responses={
        503: {"description": "服务不可用"}
    }
)


@router.get(
    "/status",
    response_model=DataResponse[SystemStatusSchema],
    summary="获取系统状态",
    description="获取系统的详细状态信息和性能指标"
)
async def get_system_status(
    db: AsyncSession = Depends(get_database)
):
    """获取系统状态"""
    try:
        # 获取系统性能指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 模拟数据库状态检查
        database_status = "connected"
        redis_status = "connected"
        
        # 模拟业务数据统计
        total_accounts = 0  # 实际应该从数据库查询
        active_accounts = 0
        total_platforms = 5  # 支持的平台数量
        
        # 模拟同步队列状态
        sync_queue_size = 0
        last_sync_time = datetime.utcnow() - timedelta(minutes=5)
        
        status_info = {
            "service_name": settings.app_name,
            "version": "1.0.0",
            "status": "healthy",
            "uptime_seconds": 3600,  # 模拟运行时间
            "database_status": database_status,
            "redis_status": redis_status,
            "total_accounts": total_accounts,
            "active_accounts": active_accounts,
            "total_platforms": total_platforms,
            "sync_queue_size": sync_queue_size,
            "last_sync_time": last_sync_time.isoformat(),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            }
        }
        
        logger.info("获取系统状态信息")
        
        return DataResponse(
            success=True,
            message="获取系统状态成功",
            data=status_info
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统状态失败"
        )


@router.get(
    "/metrics",
    response_model=DataResponse,
    summary="获取性能指标",
    description="获取服务的性能监控指标（Prometheus格式兼容）"
)
async def get_metrics():
    """获取性能指标"""
    try:
        # 模拟性能指标
        metrics = {
            "http_requests_total": 1000,
            "http_request_duration_seconds": 0.15,
            "database_connections_active": 5,
            "database_connections_max": 20,
            "redis_connections_active": 3,
            "sync_tasks_total": 50,
            "sync_tasks_success": 45,
            "sync_tasks_failed": 5,
            "oauth_tokens_active": 25,
            "oauth_tokens_expired": 5,
            "platform_api_requests_total": {
                "weibo": 150,
                "wechat": 100,
                "douyin": 200,
                "toutiao": 80,
                "baijiahao": 70
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("获取性能指标")
        
        return DataResponse(
            success=True,
            message="获取性能指标成功",
            data=metrics
        )
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取性能指标失败"
        )


@router.get(
    "/api-stats",
    response_model=DataResponse,
    summary="获取API使用统计",
    description="获取API接口的调用统计信息"
)
async def get_api_statistics(
    days: int = Query(7, ge=1, le=30, description="统计天数"),
    platform: str = Query(None, description="平台筛选")
):
    """获取API使用统计"""
    try:
        # 模拟API统计数据
        api_stats = []
        
        platforms = [platform] if platform else ["weibo", "wechat", "douyin", "toutiao", "baijiahao"]
        
        for platform_name in platforms:
            stats = {
                "platform_name": platform_name,
                "endpoint": "/api/v1/oauth/authorize",
                "method": "GET",
                "total_requests": 100,
                "error_requests": 5,
                "avg_response_time": 150.0,
                "success_rate": 95.0,
                "date": datetime.utcnow().isoformat()
            }
            api_stats.append(stats)
        
        # 汇总统计
        total_requests = sum(stat["total_requests"] for stat in api_stats)
        total_errors = sum(stat["error_requests"] for stat in api_stats)
        overall_success_rate = ((total_requests - total_errors) / total_requests * 100) if total_requests > 0 else 0
        
        result = {
            "platform_stats": api_stats,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_success_rate": round(overall_success_rate, 2),
            "report_period": f"最近{days}天",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"获取 {days} 天API统计信息")
        
        return DataResponse(
            success=True,
            message="获取API统计成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"获取API统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取API统计失败"
        )


@router.get(
    "/config",
    response_model=DataResponse,
    summary="获取系统配置信息",
    description="获取系统的配置信息（不包含敏感数据）"
)
async def get_system_config():
    """获取系统配置信息"""
    try:
        # 安全的配置信息（不包含敏感数据）
        config_info = {
            "service_name": settings.app_name,
            "version": "1.0.0",
            "debug_mode": settings.debug,
            "database_pool_size": settings.database_pool_size,
            "database_max_overflow": settings.database_max_overflow,
            "supported_platforms": ["weibo", "wechat", "douyin", "toutiao", "baijiahao"],
            "api_version": "v1",
            "cors_enabled": True,
            "rate_limiting_enabled": True,
            "encryption_algorithm": "AES-256",
            "oauth_callback_base": settings.oauth_callback_base_url.split('/')[2],  # 只显示域名
            "features": {
                "account_management": True,
                "oauth_authentication": True,
                "data_synchronization": True,
                "permission_control": True,
                "api_monitoring": True
            },
            "limits": {
                "max_accounts_per_user": 50,
                "sync_batch_size": 50,
                "api_rate_limit": "1000/hour"
            }
        }
        
        logger.info("获取系统配置信息")
        
        return DataResponse(
            success=True,
            message="获取系统配置成功",
            data=config_info
        )
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统配置失败"
        )


@router.get(
    "/logs",
    response_model=DataResponse,
    summary="获取系统日志",
    description="获取系统的最近日志信息"
)
async def get_system_logs(
    level: str = Query("INFO", description="日志级别筛选"),
    lines: int = Query(100, ge=1, le=1000, description="日志行数")
):
    """获取系统日志"""
    try:
        # 模拟日志数据
        logs = []
        for i in range(min(lines, 10)):  # 模拟最多10条日志
            log_entry = {
                "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                "level": level,
                "logger": "account_service",
                "message": f"用户账号操作完成 - {i}",
                "module": "account_controller",
                "function": "add_account"
            }
            logs.append(log_entry)
        
        result = {
            "logs": logs,
            "total_lines": len(logs),
            "level_filter": level,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"获取系统日志，级别: {level}，行数: {lines}")
        
        return DataResponse(
            success=True,
            message="获取系统日志成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取系统日志失败"
        )