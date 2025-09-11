"""
平台管理API控制器
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.database import get_database
from ..models.publishing_models import PublishingPlatform
from ..models.schemas import PlatformListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/platforms", tags=["Platforms"])


@router.get("", response_model=PlatformListResponse)
async def get_platforms(db: AsyncSession = Depends(get_database)):
    """获取支持的平台列表"""
    try:
        stmt = select(PublishingPlatform).where(PublishingPlatform.is_active == True)
        result = await db.execute(stmt)
        platforms = result.scalars().all()
        
        platform_list = []
        for platform in platforms:
            platform_list.append({
                'id': platform.id,
                'name': platform.platform_name,
                'display_name': platform.display_name,
                'type': platform.platform_type,
                'is_active': platform.is_active,
                'rate_limit': platform.rate_limit_per_hour
            })
        
        return PlatformListResponse(
            success=True,
            message="获取平台列表成功",
            data={"platforms": platform_list}
        )
        
    except Exception as e:
        logger.error(f"获取平台列表失败: {e}")
        return PlatformListResponse(
            success=False,
            message="获取平台列表失败",
            data={"platforms": []}
        )