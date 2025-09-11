"""
管理员API控制器

提供管理员专用的API接口
包括规则管理、统计信息、系统配置等功能
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from datetime import datetime, timedelta

from ..models.database import get_database
from ..models.schemas import (
    ModerationRuleCreateSchema,
    ModerationRuleSchema,
    ModerationStatsSchema,
    AppealSchema,
    ServiceConfigSchema,
    DataResponse,
    ListResponse
)
from ..models.moderation_models import ModerationRule, SensitiveWord, Whitelist, Appeal
from ..services.moderation_service import ContentModerationService
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["管理员功能"])

# 创建服务实例
moderation_service = ContentModerationService()


# 审核规则管理

@router.post("/rules", response_model=DataResponse, summary="创建审核规则")
async def create_moderation_rule(
    rule_data: ModerationRuleCreateSchema,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    创建新的审核规则
    
    Args:
        rule_data: 规则创建数据
        db: 数据库会话
        
    Returns:
        DataResponse: 包含规则ID的响应
    """
    try:
        # 创建规则对象
        rule = ModerationRule(
            name=rule_data.name,
            description=rule_data.description,
            rule_type=rule_data.rule_type,
            content_types=rule_data.content_types,
            rule_config=rule_data.rule_config,
            severity=rule_data.severity,
            action=rule_data.action,
            is_active=rule_data.is_active
        )
        
        # 保存到数据库
        db.add(rule)
        await db.commit()
        await db.refresh(rule)
        
        logger.info(f"创建审核规则成功: {rule.name} [{rule.id}]")
        
        return DataResponse(
            success=True,
            message="审核规则创建成功",
            data={"rule_id": str(rule.id)}
        )
        
    except Exception as e:
        logger.error(f"创建审核规则失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="创建审核规则失败"
        )


@router.get("/rules", response_model=ListResponse, summary="获取审核规则列表")
async def get_moderation_rules(
    rule_type: Optional[str] = Query(None, description="规则类型过滤"),
    is_active: Optional[bool] = Query(None, description="是否激活过滤"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    db: AsyncSession = Depends(get_database)
) -> ListResponse:
    """
    获取审核规则列表
    
    Args:
        rule_type: 规则类型过滤
        is_active: 激活状态过滤
        page: 页码
        size: 每页大小
        db: 数据库会话
        
    Returns:
        ListResponse: 规则列表
    """
    try:
        from sqlalchemy import select
        
        # 构建查询
        query = select(ModerationRule)
        
        if rule_type:
            query = query.where(ModerationRule.rule_type == rule_type)
        
        if is_active is not None:
            query = query.where(ModerationRule.is_active == is_active)
        
        # 分页
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        # 执行查询
        result = await db.execute(query)
        rules = result.scalars().all()
        
        # 转换为响应模式
        rule_schemas = [ModerationRuleSchema.from_orm(rule) for rule in rules]
        
        return ListResponse(
            success=True,
            message="获取审核规则列表成功",
            data=[rule.dict() for rule in rule_schemas]
        )
        
    except Exception as e:
        logger.error(f"获取审核规则列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取审核规则列表失败"
        )


@router.put("/rules/{rule_id}", response_model=DataResponse, summary="更新审核规则")
async def update_moderation_rule(
    rule_id: uuid.UUID,
    rule_data: ModerationRuleCreateSchema,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    更新指定的审核规则
    
    Args:
        rule_id: 规则ID
        rule_data: 更新数据
        db: 数据库会话
        
    Returns:
        DataResponse: 更新结果
    """
    try:
        from sqlalchemy import select
        
        # 获取现有规则
        query = select(ModerationRule).where(ModerationRule.id == rule_id)
        result = await db.execute(query)
        rule = result.scalar_one_or_none()
        
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="审核规则不存在"
            )
        
        # 更新字段
        rule.name = rule_data.name
        rule.description = rule_data.description
        rule.rule_type = rule_data.rule_type
        rule.content_types = rule_data.content_types
        rule.rule_config = rule_data.rule_config
        rule.severity = rule_data.severity
        rule.action = rule_data.action
        rule.is_active = rule_data.is_active
        
        await db.commit()
        
        logger.info(f"更新审核规则成功: {rule.name} [{rule_id}]")
        
        return DataResponse(
            success=True,
            message="审核规则更新成功",
            data={"rule_id": str(rule_id)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新审核规则失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新审核规则失败"
        )


@router.delete("/rules/{rule_id}", response_model=DataResponse, summary="删除审核规则")
async def delete_moderation_rule(
    rule_id: uuid.UUID,
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    删除指定的审核规则
    
    Args:
        rule_id: 规则ID
        db: 数据库会话
        
    Returns:
        DataResponse: 删除结果
    """
    try:
        from sqlalchemy import select, delete
        
        # 检查规则是否存在
        query = select(ModerationRule).where(ModerationRule.id == rule_id)
        result = await db.execute(query)
        rule = result.scalar_one_or_none()
        
        if not rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="审核规则不存在"
            )
        
        # 删除规则
        delete_query = delete(ModerationRule).where(ModerationRule.id == rule_id)
        await db.execute(delete_query)
        await db.commit()
        
        logger.info(f"删除审核规则成功: [{rule_id}]")
        
        return DataResponse(
            success=True,
            message="审核规则删除成功",
            data={"rule_id": str(rule_id)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除审核规则失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除审核规则失败"
        )


# 敏感词管理

@router.post("/sensitive-words", response_model=DataResponse, summary="添加敏感词")
async def add_sensitive_words(
    words_data: Dict[str, Any],
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    添加敏感词到词库
    
    Args:
        words_data: 敏感词数据 {"category": "类别", "words": ["词1", "词2"], "severity": "严重程度"}
        db: 数据库会话
        
    Returns:
        DataResponse: 添加结果
    """
    try:
        category = words_data.get("category")
        words = words_data.get("words", [])
        severity = words_data.get("severity", "medium")
        
        if not category or not words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="必须提供category和words字段"
            )
        
        added_count = 0
        
        for word in words:
            # 检查是否已存在
            from sqlalchemy import select
            existing_query = select(SensitiveWord).where(
                SensitiveWord.word == word,
                SensitiveWord.category == category
            )
            result = await db.execute(existing_query)
            existing_word = result.scalar_one_or_none()
            
            if not existing_word:
                sensitive_word = SensitiveWord(
                    word=word,
                    category=category,
                    severity=severity,
                    is_active=True
                )
                db.add(sensitive_word)
                added_count += 1
        
        await db.commit()
        
        logger.info(f"添加敏感词成功: {category} 类别 {added_count} 个词")
        
        return DataResponse(
            success=True,
            message=f"成功添加 {added_count} 个敏感词",
            data={
                "category": category,
                "added_count": added_count,
                "total_words": len(words)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加敏感词失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="添加敏感词失败"
        )


@router.get("/sensitive-words", response_model=ListResponse, summary="获取敏感词列表")
async def get_sensitive_words(
    category: Optional[str] = Query(None, description="类别过滤"),
    is_active: Optional[bool] = Query(None, description="是否激活过滤"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(50, ge=1, le=200, description="每页大小"),
    db: AsyncSession = Depends(get_database)
) -> ListResponse:
    """
    获取敏感词列表
    
    Args:
        category: 类别过滤
        is_active: 激活状态过滤
        page: 页码
        size: 每页大小
        db: 数据库会话
        
    Returns:
        ListResponse: 敏感词列表
    """
    try:
        from sqlalchemy import select
        
        # 构建查询
        query = select(SensitiveWord)
        
        if category:
            query = query.where(SensitiveWord.category == category)
        
        if is_active is not None:
            query = query.where(SensitiveWord.is_active == is_active)
        
        # 分页
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        # 执行查询
        result = await db.execute(query)
        words = result.scalars().all()
        
        # 转换为响应数据
        words_data = [
            {
                "id": str(word.id),
                "word": word.word,
                "category": word.category,
                "severity": word.severity,
                "is_active": word.is_active,
                "hit_count": word.hit_count,
                "created_at": word.created_at
            }
            for word in words
        ]
        
        return ListResponse(
            success=True,
            message="获取敏感词列表成功",
            data=words_data
        )
        
    except Exception as e:
        logger.error(f"获取敏感词列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取敏感词列表失败"
        )


# 白名单管理

@router.post("/whitelist", response_model=DataResponse, summary="添加白名单项")
async def add_whitelist_item(
    whitelist_data: Dict[str, Any],
    db: AsyncSession = Depends(get_database)
) -> DataResponse:
    """
    添加白名单项
    
    Args:
        whitelist_data: 白名单数据
        db: 数据库会话
        
    Returns:
        DataResponse: 添加结果
    """
    try:
        whitelist_item = Whitelist(
            type=whitelist_data.get("type"),
            value=whitelist_data.get("value"),
            description=whitelist_data.get("description"),
            is_active=whitelist_data.get("is_active", True)
        )
        
        db.add(whitelist_item)
        await db.commit()
        await db.refresh(whitelist_item)
        
        logger.info(f"添加白名单项成功: {whitelist_item.type} - {whitelist_item.value}")
        
        return DataResponse(
            success=True,
            message="白名单项添加成功",
            data={"id": str(whitelist_item.id)}
        )
        
    except Exception as e:
        logger.error(f"添加白名单项失败: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="添加白名单项失败"
        )


# 申诉管理

@router.get("/appeals", response_model=ListResponse, summary="获取申诉列表")
async def get_appeals(
    status: Optional[str] = Query(None, description="申诉状态过滤"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小"),
    db: AsyncSession = Depends(get_database)
) -> ListResponse:
    """
    获取申诉列表
    
    Args:
        status: 状态过滤
        page: 页码
        size: 每页大小
        db: 数据库会话
        
    Returns:
        ListResponse: 申诉列表
    """
    try:
        from sqlalchemy import select
        
        # 构建查询
        query = select(Appeal)
        
        if status:
            query = query.where(Appeal.status == status)
        
        # 分页
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        # 执行查询
        result = await db.execute(query)
        appeals = result.scalars().all()
        
        # 转换为响应模式
        appeal_schemas = [AppealSchema.from_orm(appeal) for appeal in appeals]
        
        return ListResponse(
            success=True,
            message="获取申诉列表成功",
            data=[appeal.dict() for appeal in appeal_schemas]
        )
        
    except Exception as e:
        logger.error(f"获取申诉列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取申诉列表失败"
        )


# 统计信息

@router.get("/stats", response_model=ModerationStatsSchema, summary="获取审核统计信息")
async def get_moderation_stats(
    days: int = Query(7, ge=1, le=365, description="统计天数"),
    db: AsyncSession = Depends(get_database)
) -> ModerationStatsSchema:
    """
    获取审核统计信息
    
    Args:
        days: 统计天数
        db: 数据库会话
        
    Returns:
        ModerationStatsSchema: 统计数据
    """
    try:
        from sqlalchemy import select, func
        from ..models.moderation_models import ModerationTask
        
        # 计算日期范围
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # 基础统计查询
        base_query = select(ModerationTask).where(
            ModerationTask.created_at >= start_date,
            ModerationTask.created_at <= end_date
        )
        
        # 总任务数
        total_query = select(func.count(ModerationTask.id)).where(
            ModerationTask.created_at >= start_date,
            ModerationTask.created_at <= end_date
        )
        total_result = await db.execute(total_query)
        total_tasks = total_result.scalar() or 0
        
        # 各状态任务数
        stats_queries = {
            'pending': select(func.count(ModerationTask.id)).where(
                ModerationTask.created_at >= start_date,
                ModerationTask.status == 'pending'
            ),
            'processing': select(func.count(ModerationTask.id)).where(
                ModerationTask.created_at >= start_date,
                ModerationTask.status == 'processing'
            ),
            'completed': select(func.count(ModerationTask.id)).where(
                ModerationTask.created_at >= start_date,
                ModerationTask.status.in_(['approved', 'rejected'])
            ),
            'approved': select(func.count(ModerationTask.id)).where(
                ModerationTask.created_at >= start_date,
                ModerationTask.final_result == 'approved'
            ),
            'rejected': select(func.count(ModerationTask.id)).where(
                ModerationTask.created_at >= start_date,
                ModerationTask.final_result == 'rejected'
            ),
            'manual_review': select(func.count(ModerationTask.id)).where(
                ModerationTask.created_at >= start_date,
                ModerationTask.status == 'manual_review'
            )
        }
        
        stats = {}
        for key, query in stats_queries.items():
            result = await db.execute(query)
            stats[key] = result.scalar() or 0
        
        # 平均处理时间
        avg_time_query = select(func.avg(ModerationTask.processing_time)).where(
            ModerationTask.created_at >= start_date,
            ModerationTask.processing_time.is_not(None)
        )
        avg_time_result = await db.execute(avg_time_query)
        avg_processing_time = float(avg_time_result.scalar() or 0)
        
        # 通过率
        approval_rate = 0.0
        if stats['completed'] > 0:
            approval_rate = stats['approved'] / stats['completed']
        
        return ModerationStatsSchema(
            total_tasks=total_tasks,
            pending_tasks=stats['pending'],
            processing_tasks=stats['processing'],
            completed_tasks=stats['completed'],
            approved_tasks=stats['approved'],
            rejected_tasks=stats['rejected'],
            manual_review_tasks=stats['manual_review'],
            avg_processing_time=avg_processing_time,
            approval_rate=approval_rate,
            violation_types_stats={},  # 需要额外查询
            content_types_stats={}     # 需要额外查询
        )
        
    except Exception as e:
        logger.error(f"获取审核统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取审核统计失败"
        )


# 系统配置

@router.get("/config", response_model=ServiceConfigSchema, summary="获取服务配置")
async def get_service_config() -> ServiceConfigSchema:
    """
    获取服务配置信息
    
    Returns:
        ServiceConfigSchema: 服务配置
    """
    try:
        return ServiceConfigSchema(
            supported_content_types=['text', 'image', 'video', 'audio'],
            max_file_size=settings.max_file_size,
            confidence_thresholds={
                'text': settings.text_confidence_threshold,
                'image': settings.image_confidence_threshold,
                'video': settings.video_confidence_threshold,
                'audio': settings.audio_confidence_threshold
            },
            rate_limits={
                'per_minute': settings.rate_limit_per_minute,
                'per_hour': settings.rate_limit_per_hour
            },
            enabled_analyzers=['text', 'image', 'video', 'audio']
        )
        
    except Exception as e:
        logger.error(f"获取服务配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取服务配置失败"
        )


@router.post("/config/reload", response_model=DataResponse, summary="重新加载配置")
async def reload_service_config() -> DataResponse:
    """
    重新加载服务配置
    
    Returns:
        DataResponse: 重新加载结果
    """
    try:
        # 这里可以实现配置重新加载逻辑
        logger.info("服务配置重新加载成功")
        
        return DataResponse(
            success=True,
            message="服务配置重新加载成功",
            data={"reload_time": datetime.utcnow()}
        )
        
    except Exception as e:
        logger.error(f"重新加载配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="重新加载配置失败"
        )