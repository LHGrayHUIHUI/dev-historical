"""
数据同步API控制器

提供账号数据同步的RESTful API接口
处理单个账号同步、批量同步、同步日志查询等功能
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
import logging

from ..models.database import get_database
from ..models.schemas import (
    AccountSyncRequestSchema, BatchSyncRequestSchema, AccountSyncLogSchema,
    DataResponse, ListResponse, BaseResponse
)
from ..services.account_service import AccountManagementService
from ..services.oauth_service import OAuthService
from ..services.encryption_service import EncryptionService
from ..utils.exceptions import (
    AccountNotFoundError, SyncError, RateLimitExceededError,
    get_http_status_code, create_error_response
)
from ..config.settings import settings

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/v1/sync",
    tags=["数据同步"],
    responses={
        404: {"description": "账号不存在"},
        429: {"description": "速率限制超出"},
        500: {"description": "同步失败"}
    }
)

# Redis连接池
redis_client = None

async def get_redis_client():
    """获取Redis客户端"""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(settings.redis_url)
    return redis_client

async def get_account_service(
    db: AsyncSession = Depends(get_database),
    redis_client: redis.Redis = Depends(get_redis_client)
) -> AccountManagementService:
    """获取账号管理服务依赖"""
    oauth_service = OAuthService(db, redis_client)
    encryption_service = EncryptionService()
    return AccountManagementService(db, oauth_service, encryption_service, redis_client)


@router.post(
    "/account/{account_id}",
    response_model=DataResponse,
    summary="同步单个账号",
    description="同步指定账号的数据（资料、统计、内容等）"
)
async def sync_account(
    sync_request: AccountSyncRequestSchema,
    account_id: int = Path(..., description="账号ID"),
    background_tasks: BackgroundTasks,
    service: AccountManagementService = Depends(get_account_service)
):
    """
    同步单个账号数据
    
    支持的同步类型：
    - profile: 用户资料信息
    - stats: 统计数据（粉丝数、发布数等）
    - posts: 发布内容
    - followers: 粉丝列表
    - full: 完整同步（包含以上所有）
    """
    try:
        # 验证账号ID匹配
        if sync_request.account_id != account_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="路径中的账号ID与请求体中的账号ID不匹配"
            )
        
        # 添加后台同步任务
        background_tasks.add_task(
            service.sync_account_data,
            account_id=account_id,
            sync_types=sync_request.sync_types,
            force=sync_request.force
        )
        
        logger.info(f"启动账号 {account_id} 数据同步，类型: {sync_request.sync_types}")
        
        return DataResponse(
            success=True,
            message="账号同步任务已启动",
            data={
                "account_id": account_id,
                "sync_types": sync_request.sync_types,
                "force": sync_request.force,
                "status": "started",
                "started_at": __import__('datetime').datetime.utcnow().isoformat()
            }
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"启动账号同步失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="启动账号同步失败"
        )


@router.post(
    "/batch",
    response_model=DataResponse,
    summary="批量同步账号",
    description="批量同步多个账号的数据"
)
async def batch_sync_accounts(
    sync_request: BatchSyncRequestSchema,
    background_tasks: BackgroundTasks,
    service: AccountManagementService = Depends(get_account_service)
):
    """批量同步多个账号数据"""
    try:
        # 添加后台批量同步任务
        background_tasks.add_task(
            service.batch_sync_accounts,
            account_ids=sync_request.account_ids,
            sync_types=sync_request.sync_types
        )
        
        logger.info(f"启动批量同步，账号数量: {len(sync_request.account_ids)}，类型: {sync_request.sync_types}")
        
        return DataResponse(
            success=True,
            message="批量同步任务已启动",
            data={
                "account_ids": sync_request.account_ids,
                "account_count": len(sync_request.account_ids),
                "sync_types": sync_request.sync_types,
                "status": "started",
                "started_at": __import__('datetime').datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"启动批量同步失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="启动批量同步失败"
        )


@router.get(
    "/account/{account_id}/status",
    response_model=DataResponse,
    summary="获取同步状态",
    description="获取指定账号的同步状态信息"
)
async def get_sync_status(
    account_id: int = Path(..., description="账号ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """获取账号同步状态"""
    try:
        status_info = await service.get_sync_status(account_id)
        
        logger.info(f"获取账号 {account_id} 同步状态")
        
        return DataResponse(
            success=True,
            message="获取同步状态成功",
            data=status_info
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"获取同步状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取同步状态失败"
        )


@router.get(
    "/account/{account_id}/logs",
    response_model=ListResponse[AccountSyncLogSchema],
    summary="获取同步日志",
    description="获取指定账号的同步历史日志"
)
async def get_sync_logs(
    account_id: int = Path(..., description="账号ID"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    sync_type: str = Query(None, description="同步类型筛选"),
    status_filter: str = Query(None, description="状态筛选"),
    service: AccountManagementService = Depends(get_account_service)
):
    """获取账号同步日志"""
    try:
        # 构建筛选条件
        filters = {"account_id": account_id}
        if sync_type:
            filters["sync_type"] = sync_type
        if status_filter:
            filters["status"] = status_filter
        
        # 计算偏移量
        offset = (page - 1) * size
        
        # 获取同步日志
        logs_data = await service.get_sync_logs(filters, offset, size)
        
        logger.info(f"获取账号 {account_id} 同步日志，页码: {page}")
        
        return ListResponse(
            success=True,
            message="获取同步日志成功",
            data=logs_data['logs'],
            pagination={
                "page": page,
                "size": size,
                "total": logs_data['total_count'],
                "pages": (logs_data['total_count'] + size - 1) // size
            }
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"获取同步日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取同步日志失败"
        )


@router.delete(
    "/account/{account_id}/cancel",
    response_model=BaseResponse,
    summary="取消同步任务",
    description="取消指定账号正在进行的同步任务"
)
async def cancel_sync(
    account_id: int = Path(..., description="账号ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """取消同步任务"""
    try:
        cancelled = await service.cancel_sync_task(account_id)
        
        if cancelled:
            logger.info(f"取消账号 {account_id} 同步任务成功")
            message = "同步任务已取消"
        else:
            logger.info(f"账号 {account_id} 没有正在进行的同步任务")
            message = "没有正在进行的同步任务"
        
        return BaseResponse(
            success=True,
            message=message
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"取消同步任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="取消同步任务失败"
        )


@router.get(
    "/statistics",
    response_model=DataResponse,
    summary="获取同步统计",
    description="获取系统的同步任务统计信息"
)
async def get_sync_statistics(
    days: int = Query(7, ge=1, le=30, description="统计天数"),
    service: AccountManagementService = Depends(get_account_service)
):
    """获取同步统计信息"""
    try:
        stats = await service.get_sync_statistics(days)
        
        logger.info(f"获取 {days} 天同步统计信息")
        
        return DataResponse(
            success=True,
            message="获取同步统计成功",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"获取同步统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取同步统计失败"
        )


@router.get(
    "/queue/status",
    response_model=DataResponse,
    summary="获取同步队列状态",
    description="获取当前同步任务队列的状态信息"
)
async def get_sync_queue_status(
    service: AccountManagementService = Depends(get_account_service)
):
    """获取同步队列状态"""
    try:
        queue_status = await service.get_sync_queue_status()
        
        logger.info("获取同步队列状态")
        
        return DataResponse(
            success=True,
            message="获取队列状态成功",
            data=queue_status
        )
        
    except Exception as e:
        logger.error(f"获取同步队列状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取队列状态失败"
        )