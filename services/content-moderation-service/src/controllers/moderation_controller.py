"""
内容审核API控制器

处理所有审核相关的HTTP请求
提供任务创建、查询、更新等接口
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from datetime import datetime

from ..models.database import get_database
from ..models.schemas import (
    ModerationTaskCreateSchema,
    ModerationTaskUpdateSchema,
    ModerationTaskSchema,
    ModerationResultSchema,
    BatchModerationRequestSchema,
    BatchModerationResponseSchema,
    QuickAnalysisRequestSchema,
    QuickAnalysisResponseSchema,
    TaskSearchSchema,
    TaskListResponseSchema,
    ReviewerActionSchema,
    DataResponse,
    ListResponse,
    ErrorResponse
)
from ..models.moderation_models import ContentType, ModerationStatus
from ..services.moderation_service import ContentModerationService
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/moderation", tags=["内容审核"])

# 创建服务实例
moderation_service = ContentModerationService()


@router.post("/tasks", response_model=DataResponse, summary="创建审核任务")
async def create_moderation_task(
    task_data: ModerationTaskCreateSchema,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    创建新的内容审核任务
    
    Args:
        task_data: 任务创建数据
        db: 数据库会话
        
    Returns:
        DataResponse: 包含任务ID的响应
    """
    try:
        # 准备内容数据
        content_data = {
            "text": task_data.content_text,
            "url": task_data.content_url,
            "platform": task_data.source_platform,
            "user_id": str(task_data.user_id) if task_data.user_id else None,
            "metadata": task_data.metadata or {}
        }
        
        # 提交审核任务
        task_id = await moderation_service.submit_for_moderation(
            content_id=task_data.content_id,
            content_type=task_data.content_type,
            content_data=content_data
        )
        
        return DataResponse(
            success=True,
            message="审核任务创建成功",
            data={"task_id": task_id}
        )
        
    except ValueError as e:
        logger.warning(f"任务创建参数错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"创建审核任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建审核任务失败"
        )


@router.post("/tasks/batch", response_model=BatchModerationResponseSchema, summary="批量创建审核任务")
async def create_batch_moderation_tasks(
    batch_data: BatchModerationRequestSchema,
    db: AsyncSession = Depends(get_database)
) -> BatchModerationResponseSchema:
    """
    批量创建审核任务
    
    Args:
        batch_data: 批量任务数据
        db: 数据库会话
        
    Returns:
        BatchModerationResponseSchema: 批量创建结果
    """
    try:
        total_tasks = len(batch_data.tasks)
        created_tasks = 0
        failed_tasks = 0
        task_ids = []
        errors = []
        
        for i, task_data in enumerate(batch_data.tasks):
            try:
                # 准备内容数据
                content_data = {
                    "text": task_data.content_text,
                    "url": task_data.content_url,
                    "platform": task_data.source_platform,
                    "user_id": str(task_data.user_id) if task_data.user_id else None,
                    "metadata": task_data.metadata or {}
                }
                
                # 提交审核任务
                task_id = await moderation_service.submit_for_moderation(
                    content_id=task_data.content_id,
                    content_type=task_data.content_type,
                    content_data=content_data
                )
                
                task_ids.append(uuid.UUID(task_id))
                created_tasks += 1
                
            except Exception as e:
                failed_tasks += 1
                errors.append({
                    "index": i,
                    "content_id": task_data.content_id,
                    "error": str(e)
                })
                logger.warning(f"批量任务创建失败 [{i}]: {e}")
        
        return BatchModerationResponseSchema(
            total_tasks=total_tasks,
            created_tasks=created_tasks,
            failed_tasks=failed_tasks,
            task_ids=task_ids,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"批量创建审核任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="批量创建审核任务失败"
        )


@router.get("/tasks/{task_id}", response_model=DataResponse, summary="获取审核任务详情")
async def get_moderation_task(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    获取指定审核任务的详情
    
    Args:
        task_id: 任务ID
        db: 数据库会话
        
    Returns:
        DataResponse: 任务详情数据
    """
    try:
        task = await moderation_service.get_task_by_id(str(task_id))
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="审核任务不存在"
            )
        
        task_schema = ModerationTaskSchema.from_orm(task)
        
        return DataResponse(
            success=True,
            message="获取任务详情成功",
            data=task_schema.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取审核任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取审核任务失败"
        )


@router.put("/tasks/{task_id}", response_model=DataResponse, summary="更新审核任务")
async def update_moderation_task(
    task_id: uuid.UUID,
    update_data: ModerationTaskUpdateSchema,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    更新指定审核任务
    
    Args:
        task_id: 任务ID
        update_data: 更新数据
        db: 数据库会话
        
    Returns:
        DataResponse: 更新结果
    """
    try:
        # 获取现有任务
        task = await moderation_service.get_task_by_id(str(task_id))
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="审核任务不存在"
            )
        
        # 更新任务字段
        update_dict = update_data.dict(exclude_unset=True)
        
        if update_dict.get('status'):
            task.update_status(
                status=update_dict['status'],
                result=update_dict.get('manual_result'),
                confidence_score=None,
                risk_level=None,
                violation_types=None
            )
        
        if update_dict.get('reviewer_id'):
            task.reviewer_id = update_dict['reviewer_id']
        
        # 保存到数据库
        await moderation_service._update_task_in_db(task)
        
        return DataResponse(
            success=True,
            message="任务更新成功",
            data={"task_id": str(task_id)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新审核任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新审核任务失败"
        )


@router.get("/tasks", response_model=TaskListResponseSchema, summary="搜索审核任务")
async def search_moderation_tasks(
    content_id: Optional[str] = Query(None, description="内容ID"),
    content_type: Optional[ContentType] = Query(None, description="内容类型"),
    status: Optional[ModerationStatus] = Query(None, description="审核状态"),
    user_id: Optional[uuid.UUID] = Query(None, description="用户ID"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    db: AsyncSession = Depends(get_database)
) -> TaskListResponseSchema:
    """
    搜索审核任务
    
    Args:
        content_id: 内容ID过滤
        content_type: 内容类型过滤
        status: 状态过滤
        user_id: 用户ID过滤
        page: 页码
        size: 每页大小
        db: 数据库会话
        
    Returns:
        TaskListResponseSchema: 任务列表和分页信息
    """
    try:
        # 构建搜索条件
        search_params = {
            "content_id": content_id,
            "content_type": content_type,
            "status": status,
            "user_id": user_id,
            "page": page,
            "size": size
        }
        
        # 执行搜索
        tasks, total_count = await moderation_service.search_tasks(search_params)
        
        # 转换为响应模式
        task_schemas = [ModerationTaskSchema.from_orm(task) for task in tasks]
        
        # 计算分页信息
        total_pages = (total_count + size - 1) // size
        
        pagination = {
            "page": page,
            "size": size,
            "total": total_count,
            "pages": total_pages
        }
        
        return TaskListResponseSchema(
            tasks=task_schemas,
            pagination=pagination
        )
        
    except Exception as e:
        logger.error(f"搜索审核任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="搜索审核任务失败"
        )


@router.get("/tasks/{task_id}/result", response_model=DataResponse, summary="获取审核结果")
async def get_moderation_result(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    获取指定任务的审核结果
    
    Args:
        task_id: 任务ID
        db: 数据库会话
        
    Returns:
        DataResponse: 审核结果数据
    """
    try:
        result = await moderation_service.get_task_result(str(task_id))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="审核结果不存在或任务尚未完成"
            )
        
        return DataResponse(
            success=True,
            message="获取审核结果成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取审核结果失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取审核结果失败"
        )


@router.post("/analyze/quick", response_model=QuickAnalysisResponseSchema, summary="快速内容分析")
async def quick_content_analysis(
    analysis_data: QuickAnalysisRequestSchema
) -> QuickAnalysisResponseSchema:
    """
    快速分析内容，不创建正式任务
    
    Args:
        analysis_data: 分析请求数据
        
    Returns:
        QuickAnalysisResponseSchema: 快速分析结果
    """
    try:
        # 执行快速分析
        result = await moderation_service.quick_analyze(
            content=analysis_data.content,
            content_type=analysis_data.content_type,
            analyzer_types=analysis_data.analyzer_types
        )
        
        # 生成处理建议
        recommendations = []
        
        if result.is_violation:
            if result.risk_level == "critical":
                recommendations.append("建议立即阻止内容发布")
                recommendations.append("需要人工复核")
            elif result.risk_level == "high":
                recommendations.append("建议人工审核后发布")
                recommendations.append("可考虑限制传播范围")
            elif result.risk_level == "medium":
                recommendations.append("建议添加警告标签")
                recommendations.append("可正常发布但需监控")
            else:
                recommendations.append("风险较低，可正常发布")
        else:
            recommendations.append("内容正常，可正常发布")
        
        # 是否建议创建正式任务
        should_create_task = (
            result.is_violation and 
            result.confidence >= 0.5 and 
            result.risk_level in ["high", "critical"]
        )
        
        return QuickAnalysisResponseSchema(
            analysis_result=result,
            recommendations=recommendations,
            should_create_task=should_create_task
        )
        
    except Exception as e:
        logger.error(f"快速分析失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="快速分析失败"
        )


@router.post("/tasks/{task_id}/review", response_model=DataResponse, summary="审核员操作")
async def reviewer_action(
    task_id: uuid.UUID,
    action_data: ReviewerActionSchema,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    审核员对任务执行操作
    
    Args:
        task_id: 任务ID
        action_data: 操作数据
        db: 数据库会话
        
    Returns:
        DataResponse: 操作结果
    """
    try:
        # 获取任务
        task = await moderation_service.get_task_by_id(str(task_id))
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="审核任务不存在"
            )
        
        # 执行审核员操作
        result = await moderation_service.process_reviewer_action(
            task_id=str(task_id),
            action=action_data.action,
            reviewer_notes=action_data.notes,
            custom_result=action_data.custom_result
        )
        
        return DataResponse(
            success=True,
            message=f"审核员操作 '{action_data.action}' 执行成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"审核员操作失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="审核员操作失败"
        )


@router.post("/upload", response_model=DataResponse, summary="上传文件进行审核")
async def upload_file_for_moderation(
    file: UploadFile = File(...),
    content_type: ContentType = Query(..., description="内容类型"),
    source_platform: Optional[str] = Query(None, description="来源平台"),
    user_id: Optional[uuid.UUID] = Query(None, description="用户ID")
) -> DataResponse:
    """
    上传文件并创建审核任务
    
    Args:
        file: 上传的文件
        content_type: 内容类型
        source_platform: 来源平台
        user_id: 用户ID
        
    Returns:
        DataResponse: 包含任务ID的响应
    """
    try:
        # 检查文件大小
        if file.size > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件大小超过限制 ({settings.max_file_size} bytes)"
            )
        
        # 读取文件内容
        file_content = await file.read()
        
        # 生成内容ID
        content_id = f"upload_{uuid.uuid4().hex[:16]}"
        
        # 准备内容数据
        content_data = {
            "file_content": file_content,
            "file_name": file.filename,
            "file_size": file.size,
            "content_type": file.content_type,
            "platform": source_platform,
            "user_id": str(user_id) if user_id else None
        }
        
        # 提交审核任务
        task_id = await moderation_service.submit_for_moderation(
            content_id=content_id,
            content_type=content_type,
            content_data=content_data
        )
        
        return DataResponse(
            success=True,
            message="文件上传成功，审核任务已创建",
            data={
                "task_id": task_id,
                "content_id": content_id,
                "file_name": file.filename
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传审核失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文件上传审核失败"
        )


@router.get("/content/{content_id}/history", response_model=ListResponse, summary="获取内容审核历史")
async def get_content_moderation_history(
    content_id: str,
    db: AsyncSession = Depends(get_database)
) -> ListResponse:
    """
    获取指定内容的所有审核历史
    
    Args:
        content_id: 内容ID
        db: 数据库会话
        
    Returns:
        ListResponse: 审核历史列表
    """
    try:
        history = await moderation_service.get_content_history(content_id)
        
        # 转换为响应模式
        history_data = [
            {
                "task_id": str(task.id),
                "status": task.status,
                "result": task.final_result,
                "confidence": task.confidence_score,
                "risk_level": task.risk_level,
                "created_at": task.created_at,
                "reviewed_at": task.reviewed_at
            }
            for task in history
        ]
        
        return ListResponse(
            success=True,
            message="获取审核历史成功",
            data=history_data
        )
        
    except Exception as e:
        logger.error(f"获取审核历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取审核历史失败"
        )


@router.delete("/tasks/{task_id}", response_model=DataResponse, summary="取消审核任务")
async def cancel_moderation_task(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    取消指定的审核任务
    
    Args:
        task_id: 任务ID
        db: 数据库会话
        
    Returns:
        DataResponse: 取消结果
    """
    try:
        success = await moderation_service.cancel_task(str(task_id))
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="任务无法取消或不存在"
            )
        
        return DataResponse(
            success=True,
            message="审核任务已取消",
            data={"task_id": str(task_id)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消审核任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="取消审核任务失败"
        )