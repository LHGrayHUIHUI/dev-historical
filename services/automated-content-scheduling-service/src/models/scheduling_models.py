"""
调度相关的数据模型
包含调度任务、冲突检测、模板管理等核心数据模型
"""
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, JSON, 
    ForeignKey, Enum as SQLEnum, Float, Index, UniqueConstraint,
    CheckConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from .database import Base


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待执行
    SCHEDULED = "scheduled"      # 已调度
    RUNNING = "running"          # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 执行失败
    CANCELLED = "cancelled"     # 已取消
    PAUSED = "paused"          # 已暂停
    RETRYING = "retrying"      # 重试中


class TaskType(str, Enum):
    """任务类型枚举"""
    SINGLE = "single"          # 单次任务
    RECURRING = "recurring"    # 循环任务
    BATCH = "batch"           # 批量任务
    TEMPLATE = "template"     # 模板任务


class ConflictType(str, Enum):
    """冲突类型枚举"""
    TIME_OVERLAP = "time_overlap"           # 时间重叠
    RESOURCE_CONFLICT = "resource_conflict" # 资源冲突
    PLATFORM_LIMIT = "platform_limit"      # 平台限制
    CONTENT_DUPLICATE = "content_duplicate" # 内容重复
    USER_PREFERENCE = "user_preference"     # 用户偏好冲突


class ConflictSeverity(str, Enum):
    """冲突严重程度枚举"""
    LOW = "low"        # 低级冲突
    MEDIUM = "medium"  # 中级冲突
    HIGH = "high"      # 高级冲突
    CRITICAL = "critical" # 严重冲突


class OptimizationStatus(str, Enum):
    """优化状态枚举"""
    NOT_OPTIMIZED = "not_optimized"  # 未优化
    OPTIMIZING = "optimizing"        # 优化中
    OPTIMIZED = "optimized"          # 已优化
    OPTIMIZATION_FAILED = "optimization_failed"  # 优化失败


class SchedulingTask(Base):
    """调度任务主表"""
    __tablename__ = "scheduling_tasks"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="任务ID")
    user_id = Column(Integer, nullable=False, comment="用户ID")
    title = Column(String(255), nullable=False, comment="任务标题")
    description = Column(Text, comment="任务描述")
    
    # 任务配置
    task_type = Column(SQLEnum(TaskType), nullable=False, default=TaskType.SINGLE, comment="任务类型")
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.PENDING, comment="任务状态")
    
    # 内容信息
    content_id = Column(String(255), comment="内容ID（来自存储服务）")
    content_title = Column(String(500), comment="内容标题")
    content_body = Column(Text, comment="内容正文")
    content_metadata = Column(JSON, comment="内容元数据")
    
    # 调度时间
    scheduled_time = Column(DateTime(timezone=True), nullable=False, comment="预定发布时间")
    actual_execution_time = Column(DateTime(timezone=True), comment="实际执行时间")
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 平台配置
    target_platforms = Column(ARRAY(String), comment="目标发布平台列表")
    platform_configs = Column(JSON, comment="平台特定配置")
    
    # 循环任务配置
    recurrence_rule = Column(String(255), comment="循环规则（RRULE格式）")
    recurrence_end_date = Column(DateTime(timezone=True), comment="循环结束日期")
    max_occurrences = Column(Integer, comment="最大执行次数")
    occurrence_count = Column(Integer, default=0, comment="已执行次数")
    
    # 优化配置
    optimization_enabled = Column(Boolean, default=True, comment="是否启用智能优化")
    optimization_status = Column(SQLEnum(OptimizationStatus), default=OptimizationStatus.NOT_OPTIMIZED, comment="优化状态")
    optimization_score = Column(Float, comment="优化得分")
    original_scheduled_time = Column(DateTime(timezone=True), comment="原始调度时间")
    
    # 执行配置
    priority = Column(Integer, default=5, comment="任务优先级（1-10）")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    current_retries = Column(Integer, default=0, comment="当前重试次数")
    retry_delay = Column(Integer, default=300, comment="重试延迟（秒）")
    
    # 模板配置
    template_id = Column(UUID(as_uuid=True), ForeignKey("scheduling_templates.id"), comment="模板ID")
    is_template_instance = Column(Boolean, default=False, comment="是否为模板实例")
    
    # 关联关系
    template = relationship("SchedulingTemplate", back_populates="instances")
    conflicts = relationship("SchedulingConflict", back_populates="task", cascade="all, delete-orphan")
    execution_logs = relationship("TaskExecutionLog", back_populates="task", cascade="all, delete-orphan")
    optimization_logs = relationship("OptimizationLog", back_populates="task", cascade="all, delete-orphan")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_scheduling_tasks_user_id", "user_id"),
        Index("idx_scheduling_tasks_status", "status"),
        Index("idx_scheduling_tasks_scheduled_time", "scheduled_time"),
        Index("idx_scheduling_tasks_created_time", "created_time"),
        Index("idx_scheduling_tasks_platforms", "target_platforms"),
        CheckConstraint("priority >= 1 AND priority <= 10", name="check_priority_range"),
        CheckConstraint("max_retries >= 0", name="check_max_retries"),
        CheckConstraint("occurrence_count >= 0", name="check_occurrence_count"),
    )
    
    def __repr__(self):
        return f"<SchedulingTask(id={self.id}, title='{self.title}', status='{self.status}')>"
    
    @property
    def is_recurring(self) -> bool:
        """是否为循环任务"""
        return self.task_type == TaskType.RECURRING and self.recurrence_rule is not None
    
    @property
    def is_due(self) -> bool:
        """是否到期执行"""
        if self.status != TaskStatus.SCHEDULED:
            return False
        return self.scheduled_time <= datetime.utcnow()
    
    @property
    def can_retry(self) -> bool:
        """是否可以重试"""
        return self.status == TaskStatus.FAILED and self.current_retries < self.max_retries


class SchedulingTemplate(Base):
    """调度模板表"""
    __tablename__ = "scheduling_templates"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="模板ID")
    user_id = Column(Integer, nullable=False, comment="创建用户ID")
    name = Column(String(255), nullable=False, comment="模板名称")
    description = Column(Text, comment="模板描述")
    
    # 模板配置
    template_config = Column(JSON, nullable=False, comment="模板配置JSON")
    default_platforms = Column(ARRAY(String), comment="默认发布平台")
    default_timing = Column(JSON, comment="默认时间配置")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    last_used_time = Column(DateTime(timezone=True), comment="最后使用时间")
    
    # 时间戳
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_public = Column(Boolean, default=False, comment="是否公开")
    
    # 关联关系
    instances = relationship("SchedulingTask", back_populates="template")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_scheduling_templates_user_id", "user_id"),
        Index("idx_scheduling_templates_name", "name"),
        Index("idx_scheduling_templates_usage", "usage_count"),
        UniqueConstraint("user_id", "name", name="uq_user_template_name"),
    )
    
    def __repr__(self):
        return f"<SchedulingTemplate(id={self.id}, name='{self.name}', usage_count={self.usage_count})>"


class SchedulingConflict(Base):
    """调度冲突记录表"""
    __tablename__ = "scheduling_conflicts"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="冲突ID")
    task_id = Column(UUID(as_uuid=True), ForeignKey("scheduling_tasks.id", ondelete="CASCADE"), nullable=False, comment="任务ID")
    
    # 冲突信息
    conflict_type = Column(SQLEnum(ConflictType), nullable=False, comment="冲突类型")
    severity = Column(SQLEnum(ConflictSeverity), nullable=False, comment="严重程度")
    description = Column(Text, comment="冲突描述")
    
    # 冲突对象
    conflicted_task_id = Column(UUID(as_uuid=True), ForeignKey("scheduling_tasks.id"), comment="冲突任务ID")
    conflicted_resource = Column(String(255), comment="冲突资源")
    platform_name = Column(String(50), comment="平台名称")
    
    # 冲突详情
    conflict_details = Column(JSON, comment="冲突详细信息")
    suggested_resolution = Column(JSON, comment="建议解决方案")
    
    # 处理状态
    is_resolved = Column(Boolean, default=False, comment="是否已解决")
    resolution_method = Column(String(255), comment="解决方法")
    resolved_time = Column(DateTime(timezone=True), comment="解决时间")
    resolved_by = Column(Integer, comment="解决人用户ID")
    
    # 时间戳
    detected_time = Column(DateTime(timezone=True), server_default=func.now(), comment="检测时间")
    
    # 关联关系
    task = relationship("SchedulingTask", back_populates="conflicts", foreign_keys=[task_id])
    conflicted_task = relationship("SchedulingTask", foreign_keys=[conflicted_task_id])
    
    # 数据库约束
    __table_args__ = (
        Index("idx_scheduling_conflicts_task_id", "task_id"),
        Index("idx_scheduling_conflicts_type", "conflict_type"),
        Index("idx_scheduling_conflicts_severity", "severity"),
        Index("idx_scheduling_conflicts_platform", "platform_name"),
        Index("idx_scheduling_conflicts_resolved", "is_resolved"),
    )
    
    def __repr__(self):
        return f"<SchedulingConflict(id={self.id}, type='{self.conflict_type}', severity='{self.severity}')>"


class TaskExecutionLog(Base):
    """任务执行日志表"""
    __tablename__ = "task_execution_logs"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="日志ID")
    task_id = Column(UUID(as_uuid=True), ForeignKey("scheduling_tasks.id", ondelete="CASCADE"), nullable=False, comment="任务ID")
    
    # 执行信息
    execution_start_time = Column(DateTime(timezone=True), server_default=func.now(), comment="执行开始时间")
    execution_end_time = Column(DateTime(timezone=True), comment="执行结束时间")
    execution_duration = Column(Float, comment="执行时长（秒）")
    
    # 执行状态
    status = Column(SQLEnum(TaskStatus), nullable=False, comment="执行状态")
    platform_results = Column(JSON, comment="各平台执行结果")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    error_details = Column(JSON, comment="错误详情")
    stack_trace = Column(Text, comment="错误堆栈")
    
    # 执行统计
    successful_platforms = Column(Integer, default=0, comment="成功平台数")
    failed_platforms = Column(Integer, default=0, comment="失败平台数")
    
    # 关联关系
    task = relationship("SchedulingTask", back_populates="execution_logs")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_task_execution_logs_task_id", "task_id"),
        Index("idx_task_execution_logs_status", "status"),
        Index("idx_task_execution_logs_start_time", "execution_start_time"),
    )
    
    def __repr__(self):
        return f"<TaskExecutionLog(id={self.id}, task_id={self.task_id}, status='{self.status}')>"
    
    @property
    def success_rate(self) -> float:
        """执行成功率"""
        total = self.successful_platforms + self.failed_platforms
        return self.successful_platforms / total if total > 0 else 0.0


class OptimizationLog(Base):
    """优化日志表"""
    __tablename__ = "optimization_logs"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="日志ID")
    task_id = Column(UUID(as_uuid=True), ForeignKey("scheduling_tasks.id", ondelete="CASCADE"), nullable=False, comment="任务ID")
    
    # 优化信息
    optimization_type = Column(String(50), nullable=False, comment="优化类型")
    original_scheduled_time = Column(DateTime(timezone=True), nullable=False, comment="原始调度时间")
    optimized_scheduled_time = Column(DateTime(timezone=True), nullable=False, comment="优化后调度时间")
    
    # 优化指标
    optimization_score = Column(Float, comment="优化得分")
    predicted_engagement = Column(Float, comment="预测参与度")
    predicted_reach = Column(Float, comment="预测触达率")
    confidence_score = Column(Float, comment="置信度")
    
    # 优化详情
    optimization_factors = Column(JSON, comment="优化因子")
    model_version = Column(String(50), comment="模型版本")
    model_params = Column(JSON, comment="模型参数")
    
    # 时间戳
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    
    # 关联关系
    task = relationship("SchedulingTask", back_populates="optimization_logs")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_optimization_logs_task_id", "task_id"),
        Index("idx_optimization_logs_type", "optimization_type"),
        Index("idx_optimization_logs_score", "optimization_score"),
    )
    
    def __repr__(self):
        return f"<OptimizationLog(id={self.id}, type='{self.optimization_type}', score={self.optimization_score})>"


class SchedulingQueue(Base):
    """调度队列表"""
    __tablename__ = "scheduling_queues"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="队列项ID")
    task_id = Column(UUID(as_uuid=True), ForeignKey("scheduling_tasks.id", ondelete="CASCADE"), nullable=False, comment="任务ID")
    
    # 队列信息
    queue_name = Column(String(100), nullable=False, comment="队列名称")
    priority = Column(Integer, default=5, comment="优先级")
    
    # 调度信息
    scheduled_execution_time = Column(DateTime(timezone=True), nullable=False, comment="计划执行时间")
    actual_execution_time = Column(DateTime(timezone=True), comment="实际执行时间")
    
    # 状态信息
    status = Column(String(50), default="queued", comment="队列状态")
    retry_count = Column(Integer, default=0, comment="重试次数")
    
    # 时间戳
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关联关系
    task = relationship("SchedulingTask")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_scheduling_queues_execution_time", "scheduled_execution_time"),
        Index("idx_scheduling_queues_priority", "priority"),
        Index("idx_scheduling_queues_status", "status"),
        Index("idx_scheduling_queues_queue_name", "queue_name"),
        UniqueConstraint("task_id", name="uq_task_queue"),
    )
    
    def __repr__(self):
        return f"<SchedulingQueue(id={self.id}, queue='{self.queue_name}', priority={self.priority})>"