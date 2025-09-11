"""
分析报告服务PostgreSQL数据模型

定义所有与分析报告相关的数据库表结构，包括分析任务、
报告模板、生成的报告、数据源配置、告警规则等。
使用SQLAlchemy ORM进行数据建模。
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON,
    Enum, Float, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class AnalysisTaskType(PyEnum):
    """分析任务类型枚举"""
    CONTENT_PERFORMANCE = "content_performance"      # 内容表现分析
    PLATFORM_COMPARISON = "platform_comparison"     # 平台对比分析
    TREND_ANALYSIS = "trend_analysis"               # 趋势分析
    USER_BEHAVIOR = "user_behavior"                 # 用户行为分析
    ANOMALY_DETECTION = "anomaly_detection"         # 异常检测
    FORECAST = "forecast"                          # 预测分析
    CUSTOM = "custom"                              # 自定义分析


class AnalysisTaskStatus(PyEnum):
    """分析任务状态枚举"""
    PENDING = "pending"         # 待执行
    RUNNING = "running"         # 执行中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"     # 已取消
    PAUSED = "paused"          # 已暂停


class AlertSeverity(PyEnum):
    """告警严重程度枚举"""
    LOW = "low"          # 低
    MEDIUM = "medium"    # 中
    HIGH = "high"        # 高
    CRITICAL = "critical" # 严重


class AnalysisTask(Base):
    """分析任务表"""
    
    __tablename__ = "analysis_tasks"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 基本信息
    title = Column(String(255), nullable=False, comment="任务标题")
    description = Column(Text, comment="任务描述")
    task_type = Column(Enum(AnalysisTaskType), nullable=False, comment="任务类型")
    status = Column(
        Enum(AnalysisTaskStatus), 
        nullable=False, 
        default=AnalysisTaskStatus.PENDING,
        comment="任务状态"
    )
    
    # 用户信息
    user_id = Column(String(50), nullable=False, comment="用户ID")
    created_by = Column(String(100), comment="创建者")
    
    # 配置信息
    config = Column(JSON, comment="任务配置JSON")
    parameters = Column(JSON, comment="分析参数JSON")
    filters = Column(JSON, comment="过滤条件JSON")
    
    # 时间范围
    start_date = Column(DateTime, comment="分析开始时间")
    end_date = Column(DateTime, comment="分析结束时间")
    
    # 执行信息
    scheduled_at = Column(DateTime, comment="计划执行时间")
    started_at = Column(DateTime, comment="实际开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    
    # 结果信息
    result_data = Column(JSON, comment="分析结果JSON")
    error_message = Column(Text, comment="错误信息")
    progress = Column(Integer, default=0, comment="执行进度(0-100)")
    
    # 优先级和资源
    priority = Column(Integer, default=5, comment="优先级(1-10)")
    estimated_duration = Column(Integer, comment="预估执行时间(秒)")
    actual_duration = Column(Integer, comment="实际执行时间(秒)")
    
    # 审计字段
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime, 
        nullable=False, 
        default=func.now(), 
        onupdate=func.now(),
        comment="更新时间"
    )
    deleted_at = Column(DateTime, comment="软删除时间")
    
    # 关系
    reports = relationship("GeneratedReport", back_populates="analysis_task")
    
    # 索引
    __table_args__ = (
        Index('idx_analysis_task_user_status', 'user_id', 'status'),
        Index('idx_analysis_task_type_created', 'task_type', 'created_at'),
        Index('idx_analysis_task_scheduled', 'scheduled_at'),
    )


class ReportTemplate(Base):
    """报告模板表"""
    
    __tablename__ = "report_templates"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 基本信息
    name = Column(String(255), nullable=False, comment="模板名称")
    description = Column(Text, comment="模板描述")
    category = Column(String(100), comment="模板分类")
    
    # 用户信息
    user_id = Column(String(50), comment="所属用户ID(NULL表示系统模板)")
    is_public = Column(Boolean, default=False, comment="是否公共模板")
    
    # 模板配置
    template_config = Column(JSON, nullable=False, comment="模板配置JSON")
    chart_configs = Column(JSON, comment="图表配置JSON")
    layout_config = Column(JSON, comment="布局配置JSON")
    
    # 数据配置
    data_sources = Column(JSON, comment="数据源配置JSON")
    default_filters = Column(JSON, comment="默认过滤器JSON")
    
    # 样式配置
    theme = Column(String(50), default="default", comment="主题")
    custom_styles = Column(JSON, comment="自定义样式JSON")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    # 审计字段
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime, 
        nullable=False, 
        default=func.now(), 
        onupdate=func.now(),
        comment="更新时间"
    )
    deleted_at = Column(DateTime, comment="软删除时间")
    
    # 关系
    reports = relationship("GeneratedReport", back_populates="template")
    
    # 索引
    __table_args__ = (
        Index('idx_report_template_user_public', 'user_id', 'is_public'),
        Index('idx_report_template_category', 'category'),
    )


class GeneratedReport(Base):
    """生成的报告表"""
    
    __tablename__ = "generated_reports"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 关联信息
    analysis_task_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("analysis_tasks.id"),
        comment="关联的分析任务ID"
    )
    template_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("report_templates.id"),
        comment="使用的模板ID"
    )
    
    # 基本信息
    title = Column(String(255), nullable=False, comment="报告标题")
    description = Column(Text, comment="报告描述")
    
    # 用户信息
    user_id = Column(String(50), nullable=False, comment="用户ID")
    
    # 报告内容
    content_data = Column(JSON, comment="报告内容数据JSON")
    chart_data = Column(JSON, comment="图表数据JSON")
    summary = Column(Text, comment="报告摘要")
    
    # 时间范围
    report_period_start = Column(DateTime, comment="报告期间开始时间")
    report_period_end = Column(DateTime, comment="报告期间结束时间")
    
    # 文件信息
    file_path = Column(String(500), comment="报告文件路径")
    file_size = Column(Integer, comment="文件大小(字节)")
    file_format = Column(String(20), comment="文件格式")
    
    # 状态信息
    generation_status = Column(String(20), default="generating", comment="生成状态")
    is_shared = Column(Boolean, default=False, comment="是否已分享")
    
    # 访问统计
    view_count = Column(Integer, default=0, comment="查看次数")
    download_count = Column(Integer, default=0, comment="下载次数")
    
    # 审计字段
    generated_at = Column(DateTime, nullable=False, default=func.now(), comment="生成时间")
    expires_at = Column(DateTime, comment="过期时间")
    
    # 关系
    analysis_task = relationship("AnalysisTask", back_populates="reports")
    template = relationship("ReportTemplate", back_populates="reports")
    
    # 索引
    __table_args__ = (
        Index('idx_generated_report_user_generated', 'user_id', 'generated_at'),
        Index('idx_generated_report_task', 'analysis_task_id'),
        Index('idx_generated_report_expires', 'expires_at'),
    )


class DataSource(Base):
    """数据源配置表"""
    
    __tablename__ = "data_sources"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 基本信息
    name = Column(String(255), nullable=False, comment="数据源名称")
    description = Column(Text, comment="数据源描述")
    source_type = Column(String(50), nullable=False, comment="数据源类型")
    
    # 连接配置
    connection_config = Column(JSON, nullable=False, comment="连接配置JSON")
    credentials = Column(JSON, comment="认证信息JSON(加密存储)")
    
    # 数据配置
    schema_config = Column(JSON, comment="数据结构配置JSON")
    refresh_interval = Column(Integer, default=3600, comment="刷新间隔(秒)")
    
    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否启用")
    last_sync_at = Column(DateTime, comment="最后同步时间")
    sync_status = Column(String(20), default="pending", comment="同步状态")
    
    # 用户信息
    user_id = Column(String(50), nullable=False, comment="用户ID")
    
    # 审计字段
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime, 
        nullable=False, 
        default=func.now(), 
        onupdate=func.now(),
        comment="更新时间"
    )
    deleted_at = Column(DateTime, comment="软删除时间")
    
    # 索引
    __table_args__ = (
        Index('idx_data_source_user_active', 'user_id', 'is_active'),
        Index('idx_data_source_type', 'source_type'),
        UniqueConstraint('user_id', 'name', name='uk_data_source_user_name'),
    )


class AlertRule(Base):
    """告警规则表"""
    
    __tablename__ = "alert_rules"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 基本信息
    name = Column(String(255), nullable=False, comment="规则名称")
    description = Column(Text, comment="规则描述")
    
    # 规则配置
    metric_name = Column(String(100), nullable=False, comment="监控指标名称")
    condition = Column(String(50), nullable=False, comment="触发条件")
    threshold_value = Column(Float, nullable=False, comment="阈值")
    
    # 时间窗口
    evaluation_window = Column(Integer, default=300, comment="评估时间窗口(秒)")
    evaluation_frequency = Column(Integer, default=60, comment="评估频率(秒)")
    
    # 告警配置
    severity = Column(Enum(AlertSeverity), nullable=False, comment="严重程度")
    notification_channels = Column(JSON, comment="通知渠道配置JSON")
    
    # 过滤条件
    filters = Column(JSON, comment="过滤条件JSON")
    
    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否启用")
    last_triggered_at = Column(DateTime, comment="最后触发时间")
    trigger_count = Column(Integer, default=0, comment="触发次数")
    
    # 用户信息
    user_id = Column(String(50), nullable=False, comment="用户ID")
    
    # 审计字段
    created_at = Column(DateTime, nullable=False, default=func.now(), comment="创建时间")
    updated_at = Column(
        DateTime, 
        nullable=False, 
        default=func.now(), 
        onupdate=func.now(),
        comment="更新时间"
    )
    deleted_at = Column(DateTime, comment="软删除时间")
    
    # 索引
    __table_args__ = (
        Index('idx_alert_rule_user_active', 'user_id', 'is_active'),
        Index('idx_alert_rule_metric', 'metric_name'),
        Index('idx_alert_rule_last_triggered', 'last_triggered_at'),
    )


class AlertHistory(Base):
    """告警历史表"""
    
    __tablename__ = "alert_history"
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 关联信息
    alert_rule_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("alert_rules.id"),
        nullable=False,
        comment="告警规则ID"
    )
    
    # 告警信息
    triggered_at = Column(DateTime, nullable=False, comment="触发时间")
    resolved_at = Column(DateTime, comment="解决时间")
    
    # 告警数据
    actual_value = Column(Float, comment="实际值")
    threshold_value = Column(Float, comment="阈值")
    severity = Column(Enum(AlertSeverity), nullable=False, comment="严重程度")
    
    # 通知状态
    notification_sent = Column(Boolean, default=False, comment="是否已发送通知")
    notification_channels = Column(JSON, comment="已发送的通知渠道")
    
    # 用户信息
    user_id = Column(String(50), nullable=False, comment="用户ID")
    
    # 索引
    __table_args__ = (
        Index('idx_alert_history_rule_triggered', 'alert_rule_id', 'triggered_at'),
        Index('idx_alert_history_user', 'user_id'),
    )