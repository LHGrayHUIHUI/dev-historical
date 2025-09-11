"""
调度任务API控制器
提供调度任务的创建、管理、查询等REST API接口
"""
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import logging

from ..models import get_db, TaskStatus, TaskType
from ..services.scheduling_service import (
    ContentSchedulingService, SchedulingRequest, SchedulingStrategy
)
from ..utils.exceptions import (
    SchedulingError, ConflictError, ValidationError
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/scheduling", tags=["调度任务"])

# 初始化服务
scheduling_service = ContentSchedulingService()


# Pydantic模型定义
class CreateSchedulingTaskRequest(BaseModel):
    """创建调度任务请求模型"""
    title: str = Field(..., description="任务标题", max_length=255)
    content_id: str = Field(..., description="内容ID")
    content_body: str = Field(..., description="内容正文")
    target_platforms: List[str] = Field(..., description="目标发布平台列表")
    preferred_time: Optional[datetime] = Field(None, description="偏好发布时间")
    task_type: TaskType = Field(TaskType.SINGLE, description="任务类型")
    recurrence_rule: Optional[str] = Field(None, description="循环规则（RRULE格式）")
    priority: int = Field(5, description="任务优先级（1-10）", ge=1, le=10)
    optimization_enabled: bool = Field(True, description="是否启用智能优化")
    platform_configs: Optional[Dict[str, Any]] = Field(None, description="平台特定配置")
    content_metadata: Optional[Dict[str, Any]] = Field(None, description="内容元数据")
    strategy: SchedulingStrategy = Field(SchedulingStrategy.OPTIMAL_TIME, description="调度策略")


class RescheduleTaskRequest(BaseModel):
    """重新调度任务请求模型"""
    new_scheduled_time: datetime = Field(..., description="新的调度时间")
    reason: str = Field("用户手动重新调度", description="重新调度原因", max_length=255)


class TaskListResponse(BaseModel):
    """任务列表响应模型"""
    tasks: List[Dict[str, Any]]
    total: int
    page: int
    size: int


class SchedulingTaskResponse(BaseModel):
    """调度任务响应模型"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


@router.post("/tasks", response_model=SchedulingTaskResponse, status_code=status.HTTP_201_CREATED)
async def create_scheduling_task(
    request: CreateSchedulingTaskRequest,
    user_id: int = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """
    创建调度任务
    
    创建一个新的内容调度任务，支持智能优化和冲突检测。
    
    - **user_id**: 用户ID
    - **title**: 任务标题
    - **content_id**: 内容ID
    - **content_body**: 内容正文
    - **target_platforms**: 目标发布平台列表
    - **preferred_time**: 偏好发布时间（可选）
    - **task_type**: 任务类型（single/recurring/batch/template）
    - **priority**: 任务优先级（1-10）
    - **optimization_enabled**: 是否启用智能优化
    - **strategy**: 调度策略
    """
    try:
        logger.info(f"创建调度任务请求，用户: {user_id}, 标题: {request.title}")
        
        # 验证平台列表
        if not request.target_platforms:
            raise ValidationError("目标平台列表不能为空")
        
        # 验证循环任务配置
        if request.task_type == TaskType.RECURRING and not request.recurrence_rule:
            raise ValidationError("循环任务必须提供循环规则")
        
        # 创建调度请求
        scheduling_request = SchedulingRequest(
            user_id=user_id,
            content_id=request.content_id,
            title=request.title,
            content_body=request.content_body,
            target_platforms=request.target_platforms,
            preferred_time=request.preferred_time,
            task_type=request.task_type,
            recurrence_rule=request.recurrence_rule,
            priority=request.priority,
            optimization_enabled=request.optimization_enabled,
            platform_configs=request.platform_configs,
            content_metadata=request.content_metadata
        )
        
        # 执行调度
        result = await scheduling_service.schedule_content(scheduling_request, request.strategy)
        
        response_data = {
            "task_id": str(result.task_id),
            "scheduled_time": result.scheduled_time.isoformat(),
            "conflicts": result.conflicts,
            "optimization_applied": result.optimization_applied,
            "optimization_score": result.optimization_score,
            "platform_specific_times": {
                platform: time.isoformat() 
                for platform, time in (result.platform_specific_times or {}).items()
            }
        }
        
        return SchedulingTaskResponse(
            success=True,
            message="调度任务创建成功",
            data=response_data,
            warnings=result.warnings
        )
        
    except ValidationError as e:
        logger.warning(f"调度任务验证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ConflictError as e:
        logger.warning(f"调度冲突: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except SchedulingError as e:
        logger.error(f"调度失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"创建调度任务异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="内部服务错误"
        )


@router.get("/tasks", response_model=TaskListResponse)
async def get_user_tasks(
    user_id: int = Query(..., description="用户ID"),
    status_filter: Optional[List[TaskStatus]] = Query(None, description="状态筛选"),
    page: int = Query(1, description="页码", ge=1),
    size: int = Query(20, description="每页大小", ge=1, le=100),
    session: AsyncSession = Depends(get_db)
):
    """
    获取用户的调度任务列表
    
    分页获取指定用户的调度任务，支持状态筛选。
    
    - **user_id**: 用户ID
    - **status_filter**: 状态筛选（可选）
    - **page**: 页码（默认1）
    - **size**: 每页大小（默认20，最大100）
    """
    try:
        logger.info(f"获取用户任务列表，用户: {user_id}, 页码: {page}")
        
        offset = (page - 1) * size
        
        tasks = await scheduling_service.get_user_scheduled_tasks(
            user_id=user_id,
            status_filter=status_filter,
            limit=size,
            offset=offset
        )
        
        return TaskListResponse(
            tasks=tasks,
            total=len(tasks),  # 实际应该查询总数
            page=page,
            size=size
        )
        
    except Exception as e:
        logger.error(f"获取用户任务列表异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取任务列表失败"
        )


@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_details(
    task_id: UUID,
    user_id: int = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """
    获取任务详情
    
    获取指定任务的详细信息，包括冲突、执行日志等。
    
    - **task_id**: 任务ID
    - **user_id**: 用户ID
    """
    try:
        logger.info(f"获取任务详情，任务: {task_id}, 用户: {user_id}")
        
        # 这里应该实现获取任务详情的逻辑
        # 暂时返回基础响应
        
        return {
            "task_id": str(task_id),
            "message": "获取任务详情功能待实现"
        }
        
    except Exception as e:
        logger.error(f"获取任务详情异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取任务详情失败"
        )


@router.put("/tasks/{task_id}/reschedule", response_model=SchedulingTaskResponse)
async def reschedule_task(
    task_id: UUID,
    request: RescheduleTaskRequest,
    user_id: int = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """
    重新调度任务
    
    修改已存在任务的调度时间。
    
    - **task_id**: 任务ID
    - **user_id**: 用户ID
    - **new_scheduled_time**: 新的调度时间
    - **reason**: 重新调度原因
    """
    try:
        logger.info(f"重新调度任务，任务: {task_id}, 新时间: {request.new_scheduled_time}")
        
        result = await scheduling_service.reschedule_task(
            task_id=task_id,
            new_scheduled_time=request.new_scheduled_time,
            reason=request.reason
        )
        
        response_data = {
            "task_id": str(result.task_id),
            "new_scheduled_time": result.scheduled_time.isoformat(),
            "conflicts": result.conflicts
        }
        
        return SchedulingTaskResponse(
            success=True,
            message="任务重新调度成功",
            data=response_data,
            warnings=result.warnings
        )
        
    except ValueError as e:
        logger.warning(f"重新调度任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"重新调度任务异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="重新调度失败"
        )


@router.delete("/tasks/{task_id}", response_model=Dict[str, Any])
async def cancel_task(
    task_id: UUID,
    user_id: int = Query(..., description="用户ID"),
    reason: str = Query("用户取消", description="取消原因"),
    session: AsyncSession = Depends(get_db)
):
    """
    取消调度任务
    
    取消指定的调度任务。
    
    - **task_id**: 任务ID
    - **user_id**: 用户ID
    - **reason**: 取消原因
    """
    try:
        logger.info(f"取消调度任务，任务: {task_id}, 原因: {reason}")
        
        success = await scheduling_service.cancel_task(task_id, reason)
        
        if success:
            return {
                "success": True,
                "message": "任务已成功取消",
                "task_id": str(task_id)
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="任务不存在"
            )
            
    except ValueError as e:
        logger.warning(f"取消任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"取消任务异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="取消任务失败"
        )


@router.post("/tasks/batch", response_model=List[SchedulingTaskResponse])
async def create_batch_tasks(
    requests: List[CreateSchedulingTaskRequest],
    user_id: int = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """
    批量创建调度任务
    
    一次性创建多个调度任务。
    
    - **user_id**: 用户ID
    - **requests**: 调度任务请求列表
    """
    try:
        logger.info(f"批量创建调度任务，用户: {user_id}, 数量: {len(requests)}")
        
        if len(requests) > 50:  # 限制批量数量
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量创建任务数量不能超过50个"
            )
        
        results = []
        
        for i, request in enumerate(requests):
            try:
                # 创建调度请求
                scheduling_request = SchedulingRequest(
                    user_id=user_id,
                    content_id=request.content_id,
                    title=request.title,
                    content_body=request.content_body,
                    target_platforms=request.target_platforms,
                    preferred_time=request.preferred_time,
                    task_type=request.task_type,
                    recurrence_rule=request.recurrence_rule,
                    priority=request.priority,
                    optimization_enabled=request.optimization_enabled,
                    platform_configs=request.platform_configs,
                    content_metadata=request.content_metadata
                )
                
                # 执行调度
                result = await scheduling_service.schedule_content(
                    scheduling_request, request.strategy
                )
                
                response_data = {
                    "task_id": str(result.task_id),
                    "scheduled_time": result.scheduled_time.isoformat(),
                    "conflicts": result.conflicts,
                    "optimization_applied": result.optimization_applied,
                    "optimization_score": result.optimization_score
                }
                
                results.append(SchedulingTaskResponse(
                    success=True,
                    message=f"第{i+1}个任务创建成功",
                    data=response_data,
                    warnings=result.warnings
                ))
                
            except Exception as e:
                logger.error(f"批量创建第{i+1}个任务失败: {e}")
                results.append(SchedulingTaskResponse(
                    success=False,
                    message=f"第{i+1}个任务创建失败: {str(e)}"
                ))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量创建调度任务异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量创建任务失败"
        )


@router.get("/tasks/{task_id}/conflicts", response_model=List[Dict[str, Any]])
async def get_task_conflicts(
    task_id: UUID,
    user_id: int = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db)
):
    """
    获取任务冲突信息
    
    获取指定任务的所有冲突信息。
    
    - **task_id**: 任务ID
    - **user_id**: 用户ID
    """
    try:
        logger.info(f"获取任务冲突，任务: {task_id}")
        
        # 这里应该实现获取任务冲突的逻辑
        # 暂时返回空列表
        
        return []
        
    except Exception as e:
        logger.error(f"获取任务冲突异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取任务冲突失败"
        )


@router.get("/suggestions", response_model=List[Dict[str, Any]])
async def get_scheduling_suggestions(
    user_id: int = Query(..., description="用户ID"),
    platforms: List[str] = Query(..., description="目标平台"),
    preferred_time: datetime = Query(..., description="偏好时间"),
    flexibility_hours: int = Query(24, description="时间灵活性（小时）", ge=1, le=168),
    session: AsyncSession = Depends(get_db)
):
    """
    获取调度建议
    
    基于用户偏好和冲突检测，提供最佳的调度时间建议。
    
    - **user_id**: 用户ID
    - **platforms**: 目标平台列表
    - **preferred_time**: 偏好时间
    - **flexibility_hours**: 时间灵活性（小时）
    """
    try:
        logger.info(f"获取调度建议，用户: {user_id}, 平台: {platforms}")
        
        suggestions = await scheduling_service.conflict_service.get_conflict_suggestions(
            session=session,
            user_id=user_id,
            target_platforms=platforms,
            preferred_time=preferred_time,
            time_flexibility_hours=flexibility_hours
        )
        
        return suggestions
        
    except Exception as e:
        logger.error(f"获取调度建议异常: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取调度建议失败"
        )