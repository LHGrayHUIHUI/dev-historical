"""
分析和统计相关的数据模型
包含性能指标、用户行为分析、平台统计等
"""
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, JSON, 
    ForeignKey, Float, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .database import Base


class PlatformMetrics(Base):
    """平台性能指标表"""
    __tablename__ = "platform_metrics"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="指标ID")
    platform_name = Column(String(50), nullable=False, comment="平台名称")
    user_id = Column(Integer, nullable=False, comment="用户ID")
    account_id = Column(String(255), comment="账号ID")
    
    # 时间维度
    metric_date = Column(DateTime(timezone=True), nullable=False, comment="指标日期")
    hour_of_day = Column(Integer, comment="小时（0-23）")
    day_of_week = Column(Integer, comment="星期（0-6）")
    
    # 参与度指标
    total_posts = Column(Integer, default=0, comment="发布总数")
    total_views = Column(Integer, default=0, comment="总浏览量")
    total_likes = Column(Integer, default=0, comment="总点赞数")
    total_shares = Column(Integer, default=0, comment="总分享数")
    total_comments = Column(Integer, default=0, comment="总评论数")
    total_clicks = Column(Integer, default=0, comment="总点击数")
    
    # 计算指标
    engagement_rate = Column(Float, default=0.0, comment="参与率")
    click_through_rate = Column(Float, default=0.0, comment="点击率")
    conversion_rate = Column(Float, default=0.0, comment="转化率")
    reach_rate = Column(Float, default=0.0, comment="触达率")
    
    # 质量指标
    average_post_score = Column(Float, comment="平均内容评分")
    user_satisfaction = Column(Float, comment="用户满意度")
    
    # 时间戳
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_platform_metrics_platform", "platform_name"),
        Index("idx_platform_metrics_user_id", "user_id"),
        Index("idx_platform_metrics_date", "metric_date"),
        Index("idx_platform_metrics_hour", "hour_of_day"),
        Index("idx_platform_metrics_engagement", "engagement_rate"),
        UniqueConstraint("platform_name", "user_id", "account_id", "metric_date", "hour_of_day", 
                        name="uq_platform_metric_key"),
    )
    
    def __repr__(self):
        return f"<PlatformMetrics(platform='{self.platform_name}', date={self.metric_date}, engagement={self.engagement_rate})>"
    
    @property
    def total_interactions(self) -> int:
        """总互动数"""
        return (self.total_likes or 0) + (self.total_shares or 0) + (self.total_comments or 0)
    
    def calculate_engagement_rate(self) -> float:
        """计算参与率"""
        if not self.total_views or self.total_views == 0:
            return 0.0
        return self.total_interactions / self.total_views


class ContentPerformance(Base):
    """内容性能分析表"""
    __tablename__ = "content_performance"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="性能ID")
    task_id = Column(UUID(as_uuid=True), ForeignKey("scheduling_tasks.id", ondelete="CASCADE"), 
                    nullable=False, comment="任务ID")
    content_id = Column(String(255), comment="内容ID")
    
    # 平台信息
    platform_name = Column(String(50), nullable=False, comment="发布平台")
    platform_post_id = Column(String(255), comment="平台内容ID")
    
    # 发布信息
    published_time = Column(DateTime(timezone=True), comment="实际发布时间")
    scheduled_time = Column(DateTime(timezone=True), comment="原计划时间")
    time_variance = Column(Float, comment="时间偏差（秒）")
    
    # 内容特征
    content_type = Column(String(50), comment="内容类型")
    content_length = Column(Integer, comment="内容长度")
    has_media = Column(Boolean, default=False, comment="是否包含媒体")
    media_count = Column(Integer, default=0, comment="媒体数量")
    hashtag_count = Column(Integer, default=0, comment="标签数量")
    mention_count = Column(Integer, default=0, comment="提及数量")
    
    # 性能数据（24小时内）
    views_1h = Column(Integer, default=0, comment="1小时浏览量")
    views_6h = Column(Integer, default=0, comment="6小时浏览量")
    views_24h = Column(Integer, default=0, comment="24小时浏览量")
    
    likes_1h = Column(Integer, default=0, comment="1小时点赞数")
    likes_6h = Column(Integer, default=0, comment="6小时点赞数")
    likes_24h = Column(Integer, default=0, comment="24小时点赞数")
    
    shares_1h = Column(Integer, default=0, comment="1小时分享数")
    shares_6h = Column(Integer, default=0, comment="6小时分享数")
    shares_24h = Column(Integer, default=0, comment="24小时分享数")
    
    comments_1h = Column(Integer, default=0, comment="1小时评论数")
    comments_6h = Column(Integer, default=0, comment="6小时评论数")
    comments_24h = Column(Integer, default=0, comment="24小时评论数")
    
    # 计算指标
    peak_engagement_time = Column(DateTime(timezone=True), comment="峰值参与时间")
    engagement_score = Column(Float, comment="参与度得分")
    virality_score = Column(Float, comment="传播度得分")
    quality_score = Column(Float, comment="质量得分")
    
    # 预测vs实际
    predicted_performance = Column(JSON, comment="预测性能数据")
    actual_performance = Column(JSON, comment="实际性能数据")
    prediction_accuracy = Column(Float, comment="预测准确度")
    
    # 时间戳
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关联关系
    task = relationship("SchedulingTask")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_content_performance_task_id", "task_id"),
        Index("idx_content_performance_platform", "platform_name"),
        Index("idx_content_performance_published_time", "published_time"),
        Index("idx_content_performance_engagement", "engagement_score"),
        UniqueConstraint("task_id", "platform_name", name="uq_content_platform_performance"),
    )
    
    def __repr__(self):
        return f"<ContentPerformance(task_id={self.task_id}, platform='{self.platform_name}', score={self.engagement_score})>"
    
    @property
    def total_engagement_24h(self) -> int:
        """24小时总参与度"""
        return (self.likes_24h or 0) + (self.shares_24h or 0) + (self.comments_24h or 0)


class UserBehaviorPattern(Base):
    """用户行为模式表"""
    __tablename__ = "user_behavior_patterns"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="模式ID")
    user_id = Column(Integer, nullable=False, comment="用户ID")
    platform_name = Column(String(50), comment="平台名称")
    
    # 时间模式
    preferred_hours = Column(ARRAY(Integer), comment="偏好发布小时")
    preferred_days = Column(ARRAY(Integer), comment="偏好发布日期")
    peak_engagement_hours = Column(ARRAY(Integer), comment="峰值参与小时")
    
    # 内容偏好
    preferred_content_types = Column(JSON, comment="偏好内容类型及权重")
    optimal_content_length = Column(JSON, comment="最优内容长度范围")
    media_preference = Column(JSON, comment="媒体偏好配置")
    
    # 频率模式
    average_posting_frequency = Column(Float, comment="平均发布频率（每天）")
    optimal_posting_interval = Column(Integer, comment="最优发布间隔（小时）")
    
    # 性能模式
    average_engagement_rate = Column(Float, comment="平均参与率")
    best_performing_times = Column(JSON, comment="最佳表现时间")
    worst_performing_times = Column(JSON, comment="最差表现时间")
    
    # 统计信息
    total_posts_analyzed = Column(Integer, default=0, comment="分析的总发布数")
    pattern_confidence = Column(Float, comment="模式置信度")
    last_analysis_date = Column(DateTime(timezone=True), comment="最后分析日期")
    
    # 时间戳
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_time = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_user_behavior_user_id", "user_id"),
        Index("idx_user_behavior_platform", "platform_name"),
        Index("idx_user_behavior_confidence", "pattern_confidence"),
        UniqueConstraint("user_id", "platform_name", name="uq_user_platform_behavior"),
    )
    
    def __repr__(self):
        return f"<UserBehaviorPattern(user_id={self.user_id}, platform='{self.platform_name}', confidence={self.pattern_confidence})>"


class MLModelMetrics(Base):
    """机器学习模型指标表"""
    __tablename__ = "ml_model_metrics"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="指标ID")
    model_name = Column(String(100), nullable=False, comment="模型名称")
    model_version = Column(String(50), nullable=False, comment="模型版本")
    
    # 训练信息
    training_start_time = Column(DateTime(timezone=True), comment="训练开始时间")
    training_end_time = Column(DateTime(timezone=True), comment="训练结束时间")
    training_duration = Column(Float, comment="训练时长（秒）")
    training_samples = Column(Integer, comment="训练样本数")
    validation_samples = Column(Integer, comment="验证样本数")
    
    # 模型参数
    model_parameters = Column(JSON, comment="模型参数配置")
    feature_importance = Column(JSON, comment="特征重要性")
    
    # 性能指标
    accuracy_score = Column(Float, comment="准确率")
    precision_score = Column(Float, comment="精确率")
    recall_score = Column(Float, comment="召回率")
    f1_score = Column(Float, comment="F1得分")
    mse_score = Column(Float, comment="均方误差")
    r2_score = Column(Float, comment="R²决定系数")
    
    # 交叉验证指标
    cv_mean_score = Column(Float, comment="交叉验证平均得分")
    cv_std_score = Column(Float, comment="交叉验证标准差")
    
    # 业务指标
    prediction_accuracy = Column(Float, comment="预测准确度")
    improvement_rate = Column(Float, comment="改进率")
    
    # 模型状态
    is_active = Column(Boolean, default=False, comment="是否为活跃模型")
    deployment_time = Column(DateTime(timezone=True), comment="部署时间")
    
    # 时间戳
    created_time = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_ml_model_metrics_name", "model_name"),
        Index("idx_ml_model_metrics_version", "model_version"),
        Index("idx_ml_model_metrics_accuracy", "accuracy_score"),
        Index("idx_ml_model_metrics_active", "is_active"),
        UniqueConstraint("model_name", "model_version", name="uq_model_version"),
    )
    
    def __repr__(self):
        return f"<MLModelMetrics(model='{self.model_name}', version='{self.model_version}', accuracy={self.accuracy_score})>"


class SystemMetrics(Base):
    """系统性能指标表"""
    __tablename__ = "system_metrics"
    
    # 基本信息
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="指标ID")
    metric_type = Column(String(50), nullable=False, comment="指标类型")
    service_name = Column(String(100), comment="服务名称")
    
    # 时间维度
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), comment="时间戳")
    
    # 系统指标
    cpu_usage = Column(Float, comment="CPU使用率（%）")
    memory_usage = Column(Float, comment="内存使用率（%）")
    disk_usage = Column(Float, comment="磁盘使用率（%）")
    network_io = Column(JSON, comment="网络IO统计")
    
    # 数据库指标
    db_connections = Column(Integer, comment="数据库连接数")
    db_query_time = Column(Float, comment="数据库查询平均时间（毫秒）")
    cache_hit_rate = Column(Float, comment="缓存命中率（%）")
    
    # API指标
    api_request_count = Column(Integer, comment="API请求数")
    api_response_time = Column(Float, comment="API平均响应时间（毫秒）")
    api_error_rate = Column(Float, comment="API错误率（%）")
    
    # 任务队列指标
    queue_length = Column(Integer, comment="队列长度")
    task_processing_time = Column(Float, comment="任务平均处理时间（秒）")
    task_success_rate = Column(Float, comment="任务成功率（%）")
    
    # 业务指标
    active_users = Column(Integer, comment="活跃用户数")
    scheduled_tasks = Column(Integer, comment="计划任务数")
    completed_tasks = Column(Integer, comment="完成任务数")
    failed_tasks = Column(Integer, comment="失败任务数")
    
    # 数据库约束
    __table_args__ = (
        Index("idx_system_metrics_type", "metric_type"),
        Index("idx_system_metrics_service", "service_name"),
        Index("idx_system_metrics_timestamp", "timestamp"),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(type='{self.metric_type}', service='{self.service_name}', timestamp={self.timestamp})>"
    
    @property
    def overall_health_score(self) -> float:
        """整体健康得分"""
        factors = []
        
        if self.cpu_usage is not None:
            factors.append(max(0, 100 - self.cpu_usage))
        if self.memory_usage is not None:
            factors.append(max(0, 100 - self.memory_usage))
        if self.api_error_rate is not None:
            factors.append(max(0, 100 - self.api_error_rate))
        if self.task_success_rate is not None:
            factors.append(self.task_success_rate)
            
        return sum(factors) / len(factors) if factors else 0.0