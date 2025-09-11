"""
发布相关的数据模型

定义发布平台、账号、任务和统计相关的SQLAlchemy模型
包含所有数据表的结构定义和关系映射
"""

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, JSON, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from .database import Base


class PublishingPlatform(Base):
    """
    发布平台配置表
    
    存储支持的发布平台基本信息和配置
    """
    __tablename__ = "publishing_platforms"
    
    id = Column(Integer, primary_key=True, index=True, comment="平台ID")
    platform_name = Column(String(50), unique=True, nullable=False, comment="平台名称")
    platform_type = Column(String(20), nullable=False, comment="平台类型")
    display_name = Column(String(100), nullable=False, comment="显示名称")
    api_endpoint = Column(String(255), comment="API接口地址")
    auth_type = Column(String(20), comment="认证类型")
    config_schema = Column(JSON, comment="配置模式")
    is_active = Column(Boolean, default=True, comment="是否激活")
    rate_limit_per_hour = Column(Integer, default=100, comment="每小时限流数量")
    created_at = Column(DateTime, default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系映射
    accounts = relationship("PublishingAccount", back_populates="platform", cascade="all, delete-orphan")
    tasks = relationship("PublishingTask", back_populates="platform")
    stats = relationship("PublishingStats", back_populates="platform")
    
    # 约束
    __table_args__ = (
        CheckConstraint("platform_type IN ('social_media', 'blog', 'news', 'video')", name='ck_platform_type'),
        CheckConstraint("auth_type IN ('oauth2', 'api_key', 'cookie', 'jwt')", name='ck_auth_type'),
        CheckConstraint("rate_limit_per_hour > 0", name='ck_rate_limit_positive'),
        Index('ix_platform_name', 'platform_name'),
        Index('ix_platform_active', 'is_active'),
        {"comment": "发布平台配置表"}
    )
    
    def __repr__(self):
        return f"<PublishingPlatform(id={self.id}, name='{self.platform_name}', active={self.is_active})>"


class PublishingAccount(Base):
    """
    发布账号表
    
    存储各平台的发布账号信息和认证凭据
    """
    __tablename__ = "publishing_accounts"
    
    id = Column(Integer, primary_key=True, index=True, comment="账号ID")
    platform_id = Column(Integer, ForeignKey("publishing_platforms.id"), nullable=False, comment="平台ID")
    account_name = Column(String(100), nullable=False, comment="账号名称")
    account_identifier = Column(String(255), comment="账号标识符")
    auth_credentials = Column(JSON, comment="认证凭据(加密)")
    account_status = Column(String(20), default='active', comment="账号状态")
    daily_quota = Column(Integer, default=50, comment="每日配额")
    used_quota = Column(Integer, default=0, comment="已使用配额")
    last_used_at = Column(DateTime, comment="最后使用时间")
    expires_at = Column(DateTime, comment="凭据过期时间")
    created_at = Column(DateTime, default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系映射
    platform = relationship("PublishingPlatform", back_populates="accounts")
    tasks = relationship("PublishingTask", back_populates="account")
    stats = relationship("PublishingStats", back_populates="account")
    
    # 约束
    __table_args__ = (
        CheckConstraint("account_status IN ('active', 'suspended', 'expired', 'disabled')", name='ck_account_status'),
        CheckConstraint("daily_quota >= 0", name='ck_daily_quota_non_negative'),
        CheckConstraint("used_quota >= 0", name='ck_used_quota_non_negative'),
        UniqueConstraint('platform_id', 'account_identifier', name='uq_platform_account'),
        Index('ix_account_platform', 'platform_id'),
        Index('ix_account_status', 'account_status'),
        Index('ix_account_last_used', 'last_used_at'),
        {"comment": "发布账号表"}
    )
    
    def __repr__(self):
        return f"<PublishingAccount(id={self.id}, name='{self.account_name}', status='{self.account_status}')>"
    
    @property
    def is_available(self) -> bool:
        """检查账号是否可用"""
        return (
            self.account_status == 'active' and
            self.used_quota < self.daily_quota and
            (self.expires_at is None or self.expires_at > datetime.utcnow())
        )
    
    @property
    def quota_remaining(self) -> int:
        """获取剩余配额"""
        return max(0, self.daily_quota - self.used_quota)


class PublishingTask(Base):
    """
    发布任务表
    
    存储所有发布任务的详细信息和状态
    """
    __tablename__ = "publishing_tasks"
    
    id = Column(Integer, primary_key=True, index=True, comment="任务ID")
    task_uuid = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4, comment="任务UUID")
    content_id = Column(Integer, comment="关联内容ID")
    platform_id = Column(Integer, ForeignKey("publishing_platforms.id"), nullable=False, comment="平台ID")
    account_id = Column(Integer, ForeignKey("publishing_accounts.id"), nullable=False, comment="账号ID")
    title = Column(String(500), comment="标题")
    content = Column(Text, nullable=False, comment="发布内容")
    media_urls = Column(JSON, comment="媒体文件URL数组")
    publish_config = Column(JSON, comment="发布配置")
    scheduled_at = Column(DateTime, comment="计划发布时间")
    status = Column(String(20), default='pending', comment="任务状态")
    retry_count = Column(Integer, default=0, comment="重试次数")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    error_message = Column(Text, comment="错误信息")
    platform_post_id = Column(String(255), comment="平台帖子ID")
    published_url = Column(String(500), comment="发布URL")
    published_at = Column(DateTime, comment="实际发布时间")
    created_at = Column(DateTime, default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系映射
    platform = relationship("PublishingPlatform", back_populates="tasks")
    account = relationship("PublishingAccount", back_populates="tasks")
    
    # 约束
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'processing', 'published', 'failed', 'cancelled')", name='ck_task_status'),
        CheckConstraint("retry_count >= 0", name='ck_retry_count_non_negative'),
        CheckConstraint("max_retries >= 0", name='ck_max_retries_non_negative'),
        CheckConstraint("retry_count <= max_retries", name='ck_retry_count_within_limit'),
        Index('ix_task_status', 'status'),
        Index('ix_task_scheduled', 'scheduled_at'),
        Index('ix_task_platform', 'platform_id'),
        Index('ix_task_account', 'account_id'),
        Index('ix_task_created', 'created_at'),
        Index('ix_task_uuid', 'task_uuid'),
        {"comment": "发布任务表"}
    )
    
    def __repr__(self):
        return f"<PublishingTask(id={self.id}, uuid='{self.task_uuid}', status='{self.status}')>"
    
    @property
    def is_pending(self) -> bool:
        """是否为待处理状态"""
        return self.status == 'pending'
    
    @property
    def is_processing(self) -> bool:
        """是否为处理中状态"""
        return self.status == 'processing'
    
    @property
    def is_completed(self) -> bool:
        """是否已完成(成功或失败)"""
        return self.status in ['published', 'failed', 'cancelled']
    
    @property
    def can_retry(self) -> bool:
        """是否可以重试"""
        return (
            self.status == 'failed' and
            self.retry_count < self.max_retries
        )
    
    @property
    def can_cancel(self) -> bool:
        """是否可以取消"""
        return self.status in ['pending', 'processing']
    
    def update_status(
        self, 
        status: str, 
        error_message: Optional[str] = None,
        platform_post_id: Optional[str] = None,
        published_url: Optional[str] = None
    ):
        """
        更新任务状态
        
        Args:
            status: 新状态
            error_message: 错误信息
            platform_post_id: 平台帖子ID
            published_url: 发布URL
        """
        self.status = status
        self.updated_at = func.now()
        
        if status == 'failed':
            self.error_message = error_message
            self.retry_count += 1
        elif status == 'published':
            self.platform_post_id = platform_post_id
            self.published_url = published_url
            self.published_at = func.now()
            self.error_message = None


class PublishingStats(Base):
    """
    发布统计表
    
    存储按日期统计的发布数据和效果指标
    """
    __tablename__ = "publishing_stats"
    
    id = Column(Integer, primary_key=True, index=True, comment="统计ID")
    platform_id = Column(Integer, ForeignKey("publishing_platforms.id"), nullable=False, comment="平台ID")
    account_id = Column(Integer, ForeignKey("publishing_accounts.id"), nullable=False, comment="账号ID")
    date = Column(DateTime, nullable=False, comment="统计日期")
    total_posts = Column(Integer, default=0, comment="总发布数")
    successful_posts = Column(Integer, default=0, comment="成功发布数")
    failed_posts = Column(Integer, default=0, comment="失败发布数")
    engagement_metrics = Column(JSON, comment="互动数据")
    created_at = Column(DateTime, default=func.now(), comment="创建时间")
    
    # 关系映射
    platform = relationship("PublishingPlatform", back_populates="stats")
    account = relationship("PublishingAccount", back_populates="stats")
    
    # 约束
    __table_args__ = (
        CheckConstraint("total_posts >= 0", name='ck_total_posts_non_negative'),
        CheckConstraint("successful_posts >= 0", name='ck_successful_posts_non_negative'),
        CheckConstraint("failed_posts >= 0", name='ck_failed_posts_non_negative'),
        CheckConstraint("successful_posts + failed_posts <= total_posts", name='ck_posts_consistency'),
        UniqueConstraint('platform_id', 'account_id', 'date', name='uq_stats_platform_account_date'),
        Index('ix_stats_platform', 'platform_id'),
        Index('ix_stats_account', 'account_id'),
        Index('ix_stats_date', 'date'),
        {"comment": "发布统计表"}
    )
    
    def __repr__(self):
        return f"<PublishingStats(id={self.id}, date='{self.date}', total={self.total_posts})>"
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_posts == 0:
            return 0.0
        return round(self.successful_posts / self.total_posts * 100, 2)
    
    @property
    def failure_rate(self) -> float:
        """计算失败率"""
        if self.total_posts == 0:
            return 0.0
        return round(self.failed_posts / self.total_posts * 100, 2)
    
    def update_stats(self, successful: bool):
        """
        更新统计数据
        
        Args:
            successful: 是否成功
        """
        self.total_posts += 1
        if successful:
            self.successful_posts += 1
        else:
            self.failed_posts += 1