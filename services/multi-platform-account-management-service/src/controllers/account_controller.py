"""
账号管理API控制器

提供账号的CRUD操作、统计查询等RESTful API接口
处理账号相关的HTTP请求和响应
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
import logging

from ..models.database import get_database
from ..models.schemas import (
    AccountSchema, AccountCreateSchema, AccountUpdateSchema,
    AccountSearchSchema, AccountListResponseSchema, AccountStatsSchema,
    DataResponse, ListResponse, ErrorResponse, BaseResponse
)
from ..services.account_service import AccountManagementService
from ..services.oauth_service import OAuthService
from ..services.encryption_service import EncryptionService
from ..utils.exceptions import (
    AccountNotFoundError, AccountExistsError, PermissionDeniedError,
    ValidationError, get_http_status_code, create_error_response
)
from ..config.settings import settings

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/v1/accounts",
    tags=["账号管理"],
    responses={
        404: {"description": "账号不存在"},
        409: {"description": "账号已存在"},
        422: {"description": "数据验证错误"}
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
    "/",
    response_model=DataResponse[AccountSchema],
    status_code=status.HTTP_201_CREATED,
    summary="添加账号",
    description="通过OAuth认证添加新的社交媒体账号"
)
async def add_account(
    account_data: AccountCreateSchema,
    user_id: int = Query(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """
    添加新账号
    
    通过OAuth授权码交换访问令牌，并创建新的账号记录
    """
    try:
        result = await service.add_account(
            user_id=user_id,
            platform_name=account_data.platform_name,
            auth_code=account_data.auth_code,
            redirect_uri=account_data.redirect_uri
        )
        
        logger.info(f"用户 {user_id} 成功添加 {account_data.platform_name} 账号")
        
        return DataResponse(
            success=True,
            message="账号添加成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"添加账号失败: {e}")
        if isinstance(e, (AccountExistsError, ValidationError)):
            status_code = get_http_status_code(e)
            raise HTTPException(status_code=status_code, detail=create_error_response(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="账号添加失败"
        )


@router.get(
    "/{account_id}",
    response_model=DataResponse[AccountSchema],
    summary="获取账号详情",
    description="根据账号ID获取账号的详细信息"
)
async def get_account(
    account_id: int = Path(..., description="账号ID"),
    user_id: int = Query(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """获取指定账号的详细信息"""
    try:
        account = await service.get_account_by_id(account_id, user_id)
        
        logger.info(f"用户 {user_id} 获取账号 {account_id} 详情")
        
        return DataResponse(
            success=True,
            message="获取账号详情成功",
            data=account
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"获取账号详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取账号详情失败"
        )


@router.put(
    "/{account_id}",
    response_model=DataResponse[AccountSchema],
    summary="更新账号信息",
    description="更新指定账号的基本信息"
)
async def update_account(
    account_data: AccountUpdateSchema,
    account_id: int = Path(..., description="账号ID"),
    user_id: int = Query(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """更新账号信息"""
    try:
        updated_account = await service.update_account(
            account_id=account_id,
            user_id=user_id,
            update_data=account_data.dict(exclude_unset=True)
        )
        
        logger.info(f"用户 {user_id} 更新账号 {account_id} 信息")
        
        return DataResponse(
            success=True,
            message="账号信息更新成功",
            data=updated_account
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"更新账号信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新账号信息失败"
        )


@router.delete(
    "/{account_id}",
    response_model=BaseResponse,
    summary="删除账号",
    description="删除指定的账号及其相关数据"
)
async def delete_account(
    account_id: int = Path(..., description="账号ID"),
    user_id: int = Query(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """删除账号"""
    try:
        await service.delete_account(account_id, user_id)
        
        logger.info(f"用户 {user_id} 删除账号 {account_id}")
        
        return BaseResponse(
            success=True,
            message="账号删除成功"
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"删除账号失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除账号失败"
        )


@router.get(
    "/",
    response_model=ListResponse[AccountListResponseSchema],
    summary="搜索账号列表",
    description="根据搜索条件获取账号列表"
)
async def search_accounts(
    search_params: AccountSearchSchema = Depends(),
    service: AccountManagementService = Depends(get_account_service)
):
    """搜索账号列表"""
    try:
        result = await service.search_accounts(search_params.dict(exclude_unset=True))
        
        logger.info(f"用户 {search_params.user_id} 搜索账号列表")
        
        return ListResponse(
            success=True,
            message="获取账号列表成功",
            data=result['accounts'],
            pagination=result['pagination']
        )
        
    except Exception as e:
        logger.error(f"搜索账号列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取账号列表失败"
        )


@router.get(
    "/{account_id}/stats",
    response_model=DataResponse[AccountStatsSchema],
    summary="获取账号统计信息",
    description="获取指定账号的统计数据"
)
async def get_account_stats(
    account_id: int = Path(..., description="账号ID"),
    user_id: int = Query(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """获取账号统计信息"""
    try:
        stats = await service.get_account_stats(account_id, user_id)
        
        logger.info(f"用户 {user_id} 获取账号 {account_id} 统计信息")
        
        return DataResponse(
            success=True,
            message="获取统计信息成功",
            data=stats
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"获取账号统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取统计信息失败"
        )


@router.get(
    "/user/{user_id}/summary",
    response_model=DataResponse,
    summary="获取用户账号汇总",
    description="获取用户所有账号的汇总统计信息"
)
async def get_user_account_summary(
    user_id: int = Path(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """获取用户账号汇总"""
    try:
        summary = await service.get_user_account_summary(user_id)
        
        logger.info(f"获取用户 {user_id} 账号汇总")
        
        return DataResponse(
            success=True,
            message="获取账号汇总成功",
            data=summary
        )
        
    except Exception as e:
        logger.error(f"获取用户账号汇总失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取账号汇总失败"
        )


@router.post(
    "/{account_id}/validate",
    response_model=DataResponse,
    summary="验证账号状态",
    description="验证指定账号的访问令牌是否有效"
)
async def validate_account(
    account_id: int = Path(..., description="账号ID"),
    user_id: int = Query(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """验证账号状态"""
    try:
        is_valid = await service.validate_account_token(account_id, user_id)
        
        logger.info(f"用户 {user_id} 验证账号 {account_id} 状态: {is_valid}")
        
        return DataResponse(
            success=True,
            message="账号验证完成",
            data={
                "account_id": account_id,
                "is_valid": is_valid,
                "validated_at": __import__('datetime').datetime.utcnow().isoformat()
            }
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"验证账号状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="验证账号状态失败"
        )


@router.post(
    "/{account_id}/refresh-token",
    response_model=DataResponse,
    summary="刷新访问令牌",
    description="刷新指定账号的访问令牌"
)
async def refresh_account_token(
    account_id: int = Path(..., description="账号ID"),
    user_id: int = Query(..., description="用户ID"),
    service: AccountManagementService = Depends(get_account_service)
):
    """刷新访问令牌"""
    try:
        result = await service.refresh_account_token(account_id, user_id)
        
        logger.info(f"用户 {user_id} 刷新账号 {account_id} 访问令牌")
        
        return DataResponse(
            success=True,
            message="访问令牌刷新成功",
            data={
                "account_id": account_id,
                "token_refreshed": True,
                "expires_at": result.get("expires_at"),
                "refreshed_at": __import__('datetime').datetime.utcnow().isoformat()
            }
        )
        
    except AccountNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e)
        )
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=create_error_response(e)
        )
    except Exception as e:
        logger.error(f"刷新访问令牌失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="刷新访问令牌失败"
        )