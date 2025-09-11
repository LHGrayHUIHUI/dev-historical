"""
Pydantic数据模式定义

定义所有API请求和响应的数据结构
提供数据验证和序列化功能
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, validator, Field
from datetime import datetime
from enum import Enum
import uuid

from .moderation_models import ContentType, ModerationStatus, RiskLevel


class BaseResponse(BaseModel):
    """基础响应模式"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")


class PaginationResponse(BaseModel):
    """分页响应模式"""
    page: int = Field(..., ge=1, description="当前页码")
    size: int = Field(..., ge=1, le=100, description="每页大小")
    total: int = Field(..., ge=0, description="总记录数")
    pages: int = Field(..., ge=0, description="总页数")


# 审核任务相关模式

class ModerationTaskCreateSchema(BaseModel):
    """创建审核任务请求模式"""
    content_id: str = Field(..., min_length=1, max_length=255, description="内容ID")
    content_type: ContentType = Field(..., description="内容类型")
    content_url: Optional[str] = Field(None, description="内容URL")
    content_text: Optional[str] = Field(None, description="文本内容")
    source_platform: Optional[str] = Field(None, max_length=100, description="来源平台")
    user_id: Optional[uuid.UUID] = Field(None, description="用户ID")
    priority: Optional[int] = Field(1, ge=1, le=5, description="优先级 1-5")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")
    
    @validator('content_text')
    def validate_content_text(cls, v, values):
        """验证文本内容"""
        content_type = values.get('content_type')
        if content_type == ContentType.TEXT and not v:
            raise ValueError('文本类型必须提供content_text')
        return v
    
    @validator('content_url')
    def validate_content_url(cls, v, values):
        """验证内容URL"""
        content_type = values.get('content_type')
        if content_type in [ContentType.IMAGE, ContentType.VIDEO, ContentType.AUDIO] and not v:
            raise ValueError('多媒体内容必须提供content_url')
        return v


class ModerationTaskUpdateSchema(BaseModel):
    """更新审核任务请求模式"""
    status: Optional[ModerationStatus] = Field(None, description="审核状态")
    manual_result: Optional[Dict[str, Any]] = Field(None, description="人工审核结果")
    reviewer_id: Optional[uuid.UUID] = Field(None, description="审核员ID")
    reviewer_notes: Optional[str] = Field(None, description="审核员备注")


class ViolationDetailSchema(BaseModel):
    """违规详情模式"""
    type: str = Field(..., description="违规类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    description: str = Field(..., description="违规描述")
    evidence: Optional[Dict[str, Any]] = Field(None, description="证据信息")
    location: Optional[Dict[str, Any]] = Field(None, description="位置信息")


class AnalysisResultSchema(BaseModel):
    """分析结果模式"""
    status: str = Field(..., description="分析状态")
    confidence: float = Field(..., ge=0.0, le=1.0, description="整体置信度")
    is_violation: bool = Field(..., description="是否违规")
    risk_level: str = Field(..., description="风险等级")
    violations: List[ViolationDetailSchema] = Field(..., description="违规详情列表")
    processing_time: float = Field(..., ge=0.0, description="处理时间(秒)")
    analyzer_version: str = Field(..., description="分析器版本")
    metadata: Dict[str, Any] = Field(..., description="分析元数据")
    error_message: Optional[str] = Field(None, description="错误消息")


class ModerationTaskSchema(BaseModel):
    """审核任务响应模式"""
    id: uuid.UUID = Field(..., description="任务ID")
    content_id: str = Field(..., description="内容ID")
    content_type: str = Field(..., description="内容类型")
    content_url: Optional[str] = Field(None, description="内容URL")
    content_hash: Optional[str] = Field(None, description="内容哈希值")
    source_platform: Optional[str] = Field(None, description="来源平台")
    user_id: Optional[uuid.UUID] = Field(None, description="用户ID")
    
    status: str = Field(..., description="审核状态")
    auto_result: Optional[Dict[str, Any]] = Field(None, description="自动审核结果")
    manual_result: Optional[Dict[str, Any]] = Field(None, description="人工审核结果")
    final_result: Optional[str] = Field(None, description="最终结果")
    confidence_score: Optional[float] = Field(None, description="置信度分数")
    risk_level: Optional[str] = Field(None, description="风险等级")
    violation_types: Optional[List[str]] = Field(None, description="违规类型列表")
    
    reviewer_id: Optional[uuid.UUID] = Field(None, description="审核员ID")
    processing_time: Optional[float] = Field(None, description="处理时间")
    
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    reviewed_at: Optional[datetime] = Field(None, description="审核时间")
    
    class Config:
        from_attributes = True


class ModerationResultSchema(BaseModel):
    """审核结果响应模式"""
    task_id: uuid.UUID = Field(..., description="任务ID")
    content_id: str = Field(..., description="内容ID")
    result: str = Field(..., description="审核结果")
    confidence: float = Field(..., description="置信度")
    risk_level: str = Field(..., description="风险等级")
    violations: List[ViolationDetailSchema] = Field(..., description="违规详情")
    processing_time: float = Field(..., description="处理时间")
    reviewed_at: datetime = Field(..., description="审核时间")


# 批量处理相关模式

class BatchModerationRequestSchema(BaseModel):
    """批量审核请求模式"""
    tasks: List[ModerationTaskCreateSchema] = Field(..., min_items=1, max_items=100, description="任务列表")
    
    @validator('tasks')
    def validate_tasks_uniqueness(cls, v):
        """验证任务唯一性"""
        content_ids = [task.content_id for task in v]
        if len(content_ids) != len(set(content_ids)):
            raise ValueError('批量任务中存在重复的content_id')
        return v


class BatchModerationResponseSchema(BaseModel):
    """批量审核响应模式"""
    total_tasks: int = Field(..., description="总任务数")
    created_tasks: int = Field(..., description="成功创建的任务数")
    failed_tasks: int = Field(..., description="创建失败的任务数")
    task_ids: List[uuid.UUID] = Field(..., description="创建的任务ID列表")
    errors: List[Dict[str, Any]] = Field(..., description="错误详情")


# 审核规则相关模式

class ModerationRuleCreateSchema(BaseModel):
    """创建审核规则请求模式"""
    name: str = Field(..., min_length=1, max_length=255, description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    rule_type: str = Field(..., description="规则类型")
    content_types: List[str] = Field(..., min_items=1, description="适用内容类型")
    rule_config: Dict[str, Any] = Field(..., description="规则配置")
    severity: str = Field("medium", description="严重程度")
    action: str = Field("flag", description="执行动作")
    is_active: bool = Field(True, description="是否激活")
    
    @validator('rule_type')
    def validate_rule_type(cls, v):
        """验证规则类型"""
        valid_types = ['keyword', 'regex', 'ml_model', 'api_call', 'hash_match']
        if v not in valid_types:
            raise ValueError(f'规则类型必须是以下之一: {valid_types}')
        return v
    
    @validator('severity')
    def validate_severity(cls, v):
        """验证严重程度"""
        valid_levels = ['low', 'medium', 'high', 'critical']
        if v not in valid_levels:
            raise ValueError(f'严重程度必须是以下之一: {valid_levels}')
        return v


class ModerationRuleSchema(BaseModel):
    """审核规则响应模式"""
    id: uuid.UUID = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    rule_type: str = Field(..., description="规则类型")
    content_types: List[str] = Field(..., description="适用内容类型")
    rule_config: Dict[str, Any] = Field(..., description="规则配置")
    severity: str = Field(..., description="严重程度")
    action: str = Field(..., description="执行动作")
    is_active: bool = Field(..., description="是否激活")
    
    created_by: Optional[uuid.UUID] = Field(None, description="创建者ID")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    
    class Config:
        from_attributes = True


# 申诉相关模式

class AppealCreateSchema(BaseModel):
    """创建申诉请求模式"""
    task_id: uuid.UUID = Field(..., description="任务ID")
    reason: str = Field(..., min_length=10, max_length=1000, description="申诉理由")
    evidence: Optional[Dict[str, Any]] = Field(None, description="证据材料")


class AppealSchema(BaseModel):
    """申诉响应模式"""
    id: uuid.UUID = Field(..., description="申诉ID")
    task_id: uuid.UUID = Field(..., description="任务ID")
    user_id: uuid.UUID = Field(..., description="用户ID")
    reason: str = Field(..., description="申诉理由")
    evidence: Optional[Dict[str, Any]] = Field(None, description="证据材料")
    
    status: str = Field(..., description="申诉状态")
    reviewer_id: Optional[uuid.UUID] = Field(None, description="审核员ID")
    reviewer_notes: Optional[str] = Field(None, description="审核员备注")
    
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    reviewed_at: Optional[datetime] = Field(None, description="审核时间")
    
    class Config:
        from_attributes = True


# 统计和报告相关模式

class ModerationStatsSchema(BaseModel):
    """审核统计模式"""
    total_tasks: int = Field(..., description="总任务数")
    pending_tasks: int = Field(..., description="待处理任务数")
    processing_tasks: int = Field(..., description="处理中任务数")
    completed_tasks: int = Field(..., description="已完成任务数")
    approved_tasks: int = Field(..., description="通过的任务数")
    rejected_tasks: int = Field(..., description="拒绝的任务数")
    manual_review_tasks: int = Field(..., description="需人工审核的任务数")
    
    avg_processing_time: float = Field(..., description="平均处理时间(秒)")
    approval_rate: float = Field(..., description="通过率")
    
    violation_types_stats: Dict[str, int] = Field(..., description="违规类型统计")
    content_types_stats: Dict[str, int] = Field(..., description="内容类型统计")


class TaskSearchSchema(BaseModel):
    """任务搜索请求模式"""
    content_id: Optional[str] = Field(None, description="内容ID")
    content_type: Optional[ContentType] = Field(None, description="内容类型")
    status: Optional[ModerationStatus] = Field(None, description="审核状态")
    risk_level: Optional[RiskLevel] = Field(None, description="风险等级")
    user_id: Optional[uuid.UUID] = Field(None, description="用户ID")
    reviewer_id: Optional[uuid.UUID] = Field(None, description="审核员ID")
    source_platform: Optional[str] = Field(None, description="来源平台")
    
    start_date: Optional[datetime] = Field(None, description="开始日期")
    end_date: Optional[datetime] = Field(None, description="结束日期")
    
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页大小")
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", description="排序顺序")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        """验证排序顺序"""
        if v not in ['asc', 'desc']:
            raise ValueError('排序顺序必须是asc或desc')
        return v


class TaskListResponseSchema(BaseModel):
    """任务列表响应模式"""
    tasks: List[ModerationTaskSchema] = Field(..., description="任务列表")
    pagination: PaginationResponse = Field(..., description="分页信息")


# 实时分析模式

class QuickAnalysisRequestSchema(BaseModel):
    """快速分析请求模式"""
    content: str = Field(..., min_length=1, description="待分析内容")
    content_type: ContentType = Field(..., description="内容类型")
    analyzer_types: Optional[List[str]] = Field(None, description="指定分析器类型")
    quick_mode: bool = Field(True, description="是否使用快速模式")


class QuickAnalysisResponseSchema(BaseModel):
    """快速分析响应模式"""
    analysis_result: AnalysisResultSchema = Field(..., description="分析结果")
    recommendations: List[str] = Field(..., description="处理建议")
    should_create_task: bool = Field(..., description="是否建议创建正式任务")


# 审核员相关模式

class ReviewerActionSchema(BaseModel):
    """审核员操作请求模式"""
    task_id: uuid.UUID = Field(..., description="任务ID")
    action: str = Field(..., description="操作类型")  # approve, reject, need_more_info
    notes: Optional[str] = Field(None, description="备注")
    custom_result: Optional[Dict[str, Any]] = Field(None, description="自定义结果")
    
    @validator('action')
    def validate_action(cls, v):
        """验证操作类型"""
        valid_actions = ['approve', 'reject', 'need_more_info']
        if v not in valid_actions:
            raise ValueError(f'操作类型必须是以下之一: {valid_actions}')
        return v


# API响应包装模式

class DataResponse(BaseResponse):
    """数据响应模式"""
    data: Any = Field(..., description="响应数据")


class ListResponse(BaseResponse):
    """列表响应模式"""
    data: List[Any] = Field(..., description="响应数据列表")
    pagination: Optional[PaginationResponse] = Field(None, description="分页信息")


class ErrorResponse(BaseResponse):
    """错误响应模式"""
    error_code: str = Field(..., description="错误代码")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")


# 系统健康检查模式

class HealthCheckSchema(BaseModel):
    """健康检查响应模式"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    uptime: float = Field(..., description="运行时间(秒)")
    
    database_status: str = Field(..., description="数据库状态")
    cache_status: str = Field(..., description="缓存状态")
    
    active_tasks: int = Field(..., description="活跃任务数")
    queue_size: int = Field(..., description="队列大小")
    
    memory_usage: float = Field(..., description="内存使用率")
    cpu_usage: float = Field(..., description="CPU使用率")
    
    analyzers_status: Dict[str, str] = Field(..., description="分析器状态")


# 配置模式

class ServiceConfigSchema(BaseModel):
    """服务配置响应模式"""
    supported_content_types: List[str] = Field(..., description="支持的内容类型")
    max_file_size: int = Field(..., description="最大文件大小")
    confidence_thresholds: Dict[str, float] = Field(..., description="置信度阈值")
    rate_limits: Dict[str, int] = Field(..., description="速率限制")
    enabled_analyzers: List[str] = Field(..., description="启用的分析器")


class ModerationResponseSchema(BaseResponse):
    """审核响应基础模式"""
    pass