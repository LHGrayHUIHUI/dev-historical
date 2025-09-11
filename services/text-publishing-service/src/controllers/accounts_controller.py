"""
账号管理API控制器
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import get_database
from ..models.schemas import AccountListResponse
from ..services.account_manager import AccountManager
from ..services.redis_service import redis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/accounts", tags=["Accounts"])


@router.get("", response_model=AccountListResponse)
async def get_accounts(
    platform: Optional[str] = Query(None, description="平台过滤"),
    status: Optional[str] = Query(None, description="状态过滤"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_database)
):
    """获取发布账号列表"""
    try:
        account_manager = AccountManager(db, redis_service)
        result = await account_manager.get_account_list(
            platform=platform,
            status=status,
            page=page,
            page_size=page_size
        )
        
        return AccountListResponse(
            success=True,
            message="获取账号列表成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"获取账号列表失败: {e}")
        return AccountListResponse(
            success=False,
            message="获取账号列表失败",
            data={"accounts": []}
        )