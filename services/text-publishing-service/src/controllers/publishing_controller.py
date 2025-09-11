"""
发布管理API控制器

提供发布任务的创建、查询、取消等接口
处理多平台发布的核心业务逻辑
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.database import get_database
from ..models.schemas import (
    TaskCreateSchema, BaseResponse, TaskListResponse,
    TaskCreateResponse, TaskStatusSchema, StatsResponseSchema
)
from ..services.publishing_service import PublishingService
from ..services.redis_service import redis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/publish", tags=["Publishing"])


@router.post("", response_model=TaskCreateResponse)
async def create_publishing_task(
    task_data: TaskCreateSchema,
    db: AsyncSession = Depends(get_database)
):
    """
    创建发布任务
    
    支持多平台同时发布，支持定时发布
    """
    try:
        publishing_service = PublishingService(db, redis_service)
        task_uuids = await publishing_service.create_publishing_task(task_data)
        
        return TaskCreateResponse(
            success=True,
            message=f"成功创建 {len(task_uuids)} 个发布任务",
            data={
                "task_uuids": task_uuids,
                "total_tasks": len(task_uuids)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建发布任务失败: {e}")
        raise HTTPException(status_code=500, detail="创建发布任务失败")


@router.get("/tasks/{task_uuid}/status", response_model=BaseResponse)
async def get_task_status(
    task_uuid: str,
    db: AsyncSession = Depends(get_database)
):
    """
    获取任务状态
    
    返回任务的详细状态信息，包括实时进度
    """
    try:
        publishing_service = PublishingService(db, redis_service)
        status_info = await publishing_service.get_task_status(task_uuid)
        
        return BaseResponse(
            success=True,
            message="获取任务状态成功",
            data=status_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取任务状态失败")


@router.post("/tasks/{task_uuid}/cancel", response_model=BaseResponse)
async def cancel_task(
    task_uuid: str,
    db: AsyncSession = Depends(get_database)
):
    """
    取消发布任务
    
    只能取消待处理或处理中的任务
    """
    try:
        publishing_service = PublishingService(db, redis_service)
        success = await publishing_service.cancel_task(task_uuid)
        
        if success:
            return BaseResponse(
                success=True,
                message="任务已取消"
            )
        else:
            raise HTTPException(status_code=400, detail="任务无法取消")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail="取消任务失败")


@router.get("/tasks", response_model=TaskListResponse)
async def get_task_list(
    status: Optional[str] = Query(None, description="任务状态过滤"),
    platform: Optional[str] = Query(None, description="平台过滤"),
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_database)
):
    """
    获取任务列表
    
    支持按状态、平台、日期范围过滤，支持分页
    """
    try:
        # 解析日期参数
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="开始日期格式错误，应为 YYYY-MM-DD")
        
        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="结束日期格式错误，应为 YYYY-MM-DD")
        
        publishing_service = PublishingService(db, redis_service)
        result = await publishing_service.get_task_list(
            status=status,
            platform=platform,
            start_date=start_datetime,
            end_date=end_datetime,
            page=page,
            page_size=page_size
        )
        
        return TaskListResponse(
            success=True,
            message="获取任务列表成功",
            pagination={
                "page": result['pagination']['page'],
                "page_size": result['pagination']['page_size'],
                "total": result['pagination']['total'],
                "pages": result['pagination']['pages']
            },
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取任务列表失败")


@router.get("/statistics", response_model=StatsResponseSchema)
async def get_publishing_statistics(
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
    platform: Optional[str] = Query(None, description="平台名称"),
    db: AsyncSession = Depends(get_database)
):
    """
    获取发布统计数据
    
    返回指定时间范围内的发布统计信息
    """
    try:
        # 解析日期参数
        try:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="日期格式错误，应为 YYYY-MM-DD")
        
        # 验证日期范围
        if start_datetime > end_datetime:
            raise HTTPException(status_code=400, detail="开始日期不能晚于结束日期")
        
        # 限制查询范围（防止查询过大范围影响性能）
        date_range = (end_datetime - start_datetime).days
        if date_range > 90:
            raise HTTPException(status_code=400, detail="查询日期范围不能超过90天")
        
        publishing_service = PublishingService(db, redis_service)
        stats = await publishing_service.get_publishing_statistics(
            start_date=start_datetime,
            end_date=end_datetime,
            platform=platform
        )
        
        return StatsResponseSchema(
            success=True,
            message="获取统计数据成功",
            data=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计数据失败")