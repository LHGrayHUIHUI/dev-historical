"""
账号管理相关的数据模型

定义平台、账号、认证信息、权限等相关的SQLAlchemy模型
包含所有数据表的结构定义和关系映射
"""

from sqlalchemy import (
    Column, String, Text, Boolean, DateTime, JSON, Integer,
    ForeignKey, Index, CheckConstraint, DECIMAL, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from .database import Base


class PlatformType(str, Enum):
    """平台类型枚举"""
    SOCIAL_MEDIA = "social_media"
    BLOG = "blog"
    NEWS = "news"
    SHORT_VIDEO = "short_video"
    CONTENT = "content"


class AccountStatus(str, Enum):
    """账号状态枚举"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    ERROR = "error"
    PENDING = "pending"


class VerificationStatus(str, Enum):
    """认证状态枚举"""
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    PENDING_VERIFICATION = "pending_verification"


class AccountType(str, Enum):
    """账号类型枚举"""
    PERSONAL = "personal"
    BUSINESS = "business"
    CREATOR = "creator"
    ORGANIZATION = "organization"


class Platform(Base):
    """
    平台配置表
    
    存储支持的社交媒体平台配置信息
    """
    __tablename__ = "platforms"
    
    id = Column(Integer, primary_key=True, comment="平台ID")
    name = Column(String(100), nullable=False, unique=True, comment="平台名称")
    display_name = Column(String(200), nullable=False, comment="显示名称")
    platform_type = Column(String(50), nullable=False, comment="平台类型")
    api_base_url = Column(String(500), comment="API基础URL")
    oauth_config = Column(JSON, comment="OAuth配置")
    rate_limits = Column(JSON, comment="速率限制配置")
    features = Column(JSON, comment="平台功能特性")
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系映射
    accounts = relationship("Account", back_populates="platform", cascade="all, delete-orphan")
    api_usage_stats = relationship("ApiUsageStats", back_populates="platform")
    
    # 约束
    __table_args__ = (
        CheckConstraint("platform_type IN ('social_media', 'blog', 'news', 'short_video', 'content')", name='ck_platform_type'),
        Index('ix_platform_name', 'name'),
        Index('ix_platform_type', 'platform_type'),
        Index('ix_platform_active', 'is_active'),
        {"comment": "平台配置表"}
    )
    
    def __repr__(self):
        return f"<Platform(id={self.id}, name='{self.name}', type='{self.platform_type}')>"
    
    @property
    def oauth_authorize_url(self) -> Optional[str]:
        """获取OAuth授权URL"""
        if self.oauth_config:
            return self.oauth_config.get('authorize_url')
        return None
    
    @property
    def oauth_token_url(self) -> Optional[str]:
        """获取OAuth令牌URL"""
        if self.oauth_config:
            return self.oauth_config.get('token_url')
        return None


class Account(Base):
    """
    账号表
    
    存储各个平台的账号基本信息
    """
    __tablename__ = "accounts"
    
    id = Column(Integer, primary_key=True, comment="账号ID")
    platform_id = Column(Integer, ForeignKey("platforms.id"), nullable=False, comment="平台ID")
    user_id = Column(Integer, nullable=False, comment="用户ID")
    account_name = Column(String(200), nullable=False, comment="账号名称")
    account_id = Column(String(200), comment="平台账号ID")
    display_name = Column(String(200), comment="显示名称")
    avatar_url = Column(String(500), comment="头像URL")
    bio = Column(Text, comment="个人简介")
    
    # 统计信息
    follower_count = Column(Integer, default=0, comment="粉丝数")
    following_count = Column(Integer, default=0, comment="关注数")
    post_count = Column(Integer, default=0, comment="发布数")
    
    # 账号状态
    verification_status = Column(String(50), default='unverified', comment="认证状态")
    account_type = Column(String(50), default='personal', comment="账号类型")
    status = Column(String(50), default='active', comment="账号状态")
    
    last_sync_at = Column(DateTime(timezone=True), comment="最后同步时间")
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系映射
    platform = relationship("Platform", back_populates="accounts")
    credentials = relationship("AccountCredential", back_populates="account", cascade="all, delete-orphan")
    permissions = relationship("AccountPermission", back_populates="account", cascade="all, delete-orphan")
    sync_logs = relationship("AccountSyncLog", back_populates="account", cascade="all, delete-orphan")
    api_usage_stats = relationship("ApiUsageStats", back_populates="account")
    
    # 约束
    __table_args__ = (
        CheckConstraint("verification_status IN ('unverified', 'verified', 'pending_verification')", name='ck_verification_status'),
        CheckConstraint("account_type IN ('personal', 'business', 'creator', 'organization')", name='ck_account_type'),
        CheckConstraint("status IN ('active', 'suspended', 'expired', 'error', 'pending')", name='ck_account_status'),
        Index('ix_accounts_platform_user', 'platform_id', 'user_id'),
        Index('ix_accounts_status', 'status'),
        Index('ix_accounts_platform_account_id', 'platform_id', 'account_id'),
        {"comment": "账号表"}
    )
    
    def __repr__(self):
        return f"<Account(id={self.id}, platform_id={self.platform_id}, account_name='{self.account_name}')>"
    
    @property
    def is_active(self) -> bool:
        """是否为活跃状态"""
        return self.status == AccountStatus.ACTIVE
    
    @property
    def is_verified(self) -> bool:
        """是否已认证"""
        return self.verification_status == VerificationStatus.VERIFIED
    
    def update_stats(self, stats: Dict[str, int]):
        """
        更新账号统计信息
        
        Args:
            stats: 统计数据字典
        """
        if 'follower_count' in stats:
            self.follower_count = stats['follower_count']
        if 'following_count' in stats:
            self.following_count = stats['following_count']
        if 'post_count' in stats:
            self.post_count = stats['post_count']
        
        self.last_sync_at = func.now()
        self.updated_at = func.now()


class AccountCredential(Base):
    """
    账号认证信息表
    
    存储OAuth令牌等敏感认证信息，加密存储
    """
    __tablename__ = "account_credentials"
    
    id = Column(Integer, primary_key=True, comment="认证信息ID")
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, comment="账号ID")
    access_token = Column(Text, comment="访问令牌")
    refresh_token = Column(Text, comment="刷新令牌")
    token_type = Column(String(50), default='Bearer', comment="令牌类型")
    expires_at = Column(DateTime(timezone=True), comment="过期时间")
    scope = Column(Text, comment="授权范围")
    encrypted_data = Column(JSON, comment="加密存储的敏感信息")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系映射
    account = relationship("Account", back_populates="credentials")
    
    # 约束
    __table_args__ = (
        Index('ix_account_credentials_expires', 'expires_at'),
        Index('ix_account_credentials_account', 'account_id'),
        {"comment": "账号认证信息表"}
    )
    
    def __repr__(self):
        return f"<AccountCredential(id={self.id}, account_id={self.account_id})>"
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return self.expires_at < datetime.utcnow()
    
    @property
    def expires_soon(self, minutes: int = 30) -> bool:
        """是否即将过期"""
        if self.expires_at is None:
            return False
        from datetime import timedelta
        return self.expires_at < datetime.utcnow() + timedelta(minutes=minutes)


class AccountPermission(Base):
    """
    账号权限表
    
    管理用户对账号的操作权限
    """
    __tablename__ = "account_permissions"
    
    id = Column(Integer, primary_key=True, comment="权限ID")
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, comment="账号ID")
    user_id = Column(Integer, nullable=False, comment="用户ID")
    permission_type = Column(String(100), nullable=False, comment="权限类型")
    granted_by = Column(Integer, comment="授权者ID")
    granted_at = Column(DateTime(timezone=True), default=func.now(), comment="授权时间")
    expires_at = Column(DateTime(timezone=True), comment="过期时间")
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    # 关系映射
    account = relationship("Account", back_populates="permissions")
    
    # 约束
    __table_args__ = (
        CheckConstraint("permission_type IN ('read', 'write', 'admin', 'publish', 'manage')", name='ck_permission_type'),
        Index('ix_account_permissions_user', 'user_id', 'is_active'),
        Index('ix_account_permissions_account', 'account_id'),
        {"comment": "账号权限表"}
    )
    
    def __repr__(self):
        return f"<AccountPermission(id={self.id}, account_id={self.account_id}, user_id={self.user_id}, type='{self.permission_type}')>"
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return self.expires_at < datetime.utcnow()


class AccountSyncLog(Base):
    """
    账号同步日志表
    
    记录账号数据同步的历史和结果
    """
    __tablename__ = "account_sync_logs"
    
    id = Column(Integer, primary_key=True, comment="日志ID")
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False, comment="账号ID")
    sync_type = Column(String(50), nullable=False, comment="同步类型")
    status = Column(String(50), nullable=False, comment="同步状态")
    started_at = Column(DateTime(timezone=True), nullable=False, comment="开始时间")
    completed_at = Column(DateTime(timezone=True), comment="完成时间")
    error_message = Column(Text, comment="错误消息")
    sync_data = Column(JSON, comment="同步数据")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    
    # 关系映射
    account = relationship("Account", back_populates="sync_logs")
    
    # 约束
    __table_args__ = (
        CheckConstraint("sync_type IN ('profile', 'stats', 'posts', 'followers', 'full')", name='ck_sync_type'),
        CheckConstraint("status IN ('success', 'failed', 'partial', 'in_progress')", name='ck_sync_status'),
        Index('ix_sync_logs_account_time', 'account_id', 'created_at'),
        Index('ix_sync_logs_status', 'status'),
        {"comment": "账号同步日志表"}
    )
    
    def __repr__(self):
        return f"<AccountSyncLog(id={self.id}, account_id={self.account_id}, type='{self.sync_type}', status='{self.status}')>"
    
    @property
    def duration_seconds(self) -> Optional[int]:
        """同步持续时间（秒）"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None


class ApiUsageStats(Base):
    """
    平台API使用统计表
    
    记录API调用统计数据，用于监控和限流
    """
    __tablename__ = "api_usage_stats"
    
    id = Column(Integer, primary_key=True, comment="统计ID")
    platform_id = Column(Integer, ForeignKey("platforms.id"), nullable=False, comment="平台ID")
    account_id = Column(Integer, ForeignKey("accounts.id"), comment="账号ID")
    endpoint = Column(String(200), nullable=False, comment="API端点")
    method = Column(String(10), nullable=False, comment="HTTP方法")
    status_code = Column(Integer, comment="响应状态码")
    response_time_ms = Column(Integer, comment="响应时间(毫秒)")
    request_date = Column(DateTime(timezone=True), nullable=False, comment="请求日期")
    request_hour = Column(Integer, nullable=False, comment="请求小时")
    request_count = Column(Integer, default=1, comment="请求次数")
    error_count = Column(Integer, default=0, comment="错误次数")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    
    # 关系映射
    platform = relationship("Platform", back_populates="api_usage_stats")
    account = relationship("Account", back_populates="api_usage_stats")
    
    # 约束
    __table_args__ = (
        Index('ix_api_stats_platform_date', 'platform_id', 'request_date'),
        Index('ix_api_stats_account_date', 'account_id', 'request_date'),
        Index('ix_api_stats_endpoint', 'endpoint'),
        {"comment": "平台API使用统计表"}
    )
    
    def __repr__(self):
        return f"<ApiUsageStats(id={self.id}, platform_id={self.platform_id}, endpoint='{self.endpoint}')>"