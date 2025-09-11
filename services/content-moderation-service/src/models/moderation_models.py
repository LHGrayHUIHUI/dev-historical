"""
内容审核相关的数据模型

定义审核任务、规则、敏感词、白名单等相关的SQLAlchemy模型
包含所有数据表的结构定义和关系映射
"""

from sqlalchemy import (
    Column, String, Text, Boolean, DateTime, JSON, 
    ForeignKey, Index, CheckConstraint, DECIMAL, ARRAY, Integer
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from .database import Base


class ContentType(str, Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ModerationStatus(str, Enum):
    """审核状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModerationTask(Base):
    """
    审核任务表
    
    存储所有内容审核任务的详细信息和状态
    """
    __tablename__ = "moderation_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="任务ID")
    content_id = Column(String(255), nullable=False, comment="内容ID")
    content_type = Column(String(50), nullable=False, comment="内容类型")
    content_url = Column(Text, comment="内容URL")
    content_text = Column(Text, comment="文本内容")
    content_hash = Column(String(64), comment="内容哈希值")
    source_platform = Column(String(100), comment="来源平台")
    user_id = Column(UUID(as_uuid=True), comment="用户ID")
    
    status = Column(String(50), default='pending', nullable=False, comment="审核状态")
    auto_result = Column(JSON, comment="自动审核结果")
    manual_result = Column(JSON, comment="人工审核结果")
    final_result = Column(String(50), comment="最终结果")
    confidence_score = Column(DECIMAL(5,4), comment="置信度分数")
    risk_level = Column(String(20), comment="风险等级")
    violation_types = Column(ARRAY(String), comment="违规类型数组")
    
    reviewer_id = Column(UUID(as_uuid=True), comment="审核员ID")
    processing_time = Column(DECIMAL(8,3), comment="处理时间(秒)")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), comment="更新时间")
    reviewed_at = Column(DateTime(timezone=True), comment="审核时间")
    
    # 关系映射
    appeals = relationship("Appeal", back_populates="task", cascade="all, delete-orphan")
    logs = relationship("ModerationLog", back_populates="task", cascade="all, delete-orphan")
    
    # 约束
    __table_args__ = (
        CheckConstraint("content_type IN ('text', 'image', 'video', 'audio')", name='ck_content_type'),
        CheckConstraint("status IN ('pending', 'processing', 'approved', 'rejected', 'manual_review')", name='ck_status'),
        CheckConstraint("risk_level IN ('low', 'medium', 'high', 'critical')", name='ck_risk_level'),
        CheckConstraint("final_result IN ('approved', 'rejected')", name='ck_final_result'),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name='ck_confidence_score'),
        Index('ix_task_content_id', 'content_id'),
        Index('ix_task_status', 'status'),
        Index('ix_task_content_type', 'content_type'),
        Index('ix_task_created_at', 'created_at'),
        Index('ix_task_user_id', 'user_id'),
        Index('ix_task_content_hash', 'content_hash'),
        {"comment": "内容审核任务表"}
    )
    
    def __repr__(self):
        return f"<ModerationTask(id={self.id}, content_type='{self.content_type}', status='{self.status}')>"
    
    @property
    def is_pending(self) -> bool:
        """是否为待处理状态"""
        return self.status == ModerationStatus.PENDING
    
    @property
    def is_processing(self) -> bool:
        """是否为处理中状态"""
        return self.status == ModerationStatus.PROCESSING
    
    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status in [ModerationStatus.APPROVED, ModerationStatus.REJECTED]
    
    @property
    def needs_manual_review(self) -> bool:
        """是否需要人工审核"""
        return self.status == ModerationStatus.MANUAL_REVIEW
    
    def update_status(
        self, 
        status: str, 
        result: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        risk_level: Optional[str] = None,
        violation_types: Optional[List[str]] = None
    ):
        """
        更新任务状态
        
        Args:
            status: 新状态
            result: 审核结果
            confidence_score: 置信度分数
            risk_level: 风险等级
            violation_types: 违规类型
        """
        self.status = status
        self.updated_at = func.now()
        
        if result:
            if status in [ModerationStatus.APPROVED, ModerationStatus.REJECTED]:
                self.final_result = status
            
            # 根据审核类型更新结果
            if self.reviewer_id:
                self.manual_result = result
            else:
                self.auto_result = result
        
        if confidence_score is not None:
            self.confidence_score = confidence_score
        
        if risk_level:
            self.risk_level = risk_level
        
        if violation_types:
            self.violation_types = violation_types
        
        if status in [ModerationStatus.APPROVED, ModerationStatus.REJECTED]:
            self.reviewed_at = func.now()


class ModerationRule(Base):
    """
    审核规则表
    
    存储各种内容审核规则的配置信息
    """
    __tablename__ = "moderation_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="规则ID")
    name = Column(String(255), nullable=False, comment="规则名称")
    description = Column(Text, comment="规则描述")
    rule_type = Column(String(50), nullable=False, comment="规则类型")
    content_types = Column(ARRAY(String), nullable=False, comment="适用内容类型")
    rule_config = Column(JSON, nullable=False, comment="规则配置")
    severity = Column(String(20), default='medium', nullable=False, comment="严重程度")
    action = Column(String(50), default='flag', nullable=False, comment="执行动作")
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    created_by = Column(UUID(as_uuid=True), comment="创建者ID")
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 约束
    __table_args__ = (
        CheckConstraint("rule_type IN ('keyword', 'regex', 'ml_model', 'api_call', 'hash_match')", name='ck_rule_type'),
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='ck_severity'),
        CheckConstraint("action IN ('flag', 'block', 'manual_review', 'auto_reject')", name='ck_action'),
        Index('ix_rule_name', 'name'),
        Index('ix_rule_type', 'rule_type'),
        Index('ix_rule_active', 'is_active'),
        {"comment": "内容审核规则表"}
    )
    
    def __repr__(self):
        return f"<ModerationRule(id={self.id}, name='{self.name}', type='{self.rule_type}')>"
    
    @property
    def is_text_rule(self) -> bool:
        """是否为文本规则"""
        return 'text' in self.content_types
    
    @property
    def is_image_rule(self) -> bool:
        """是否为图像规则"""
        return 'image' in self.content_types
    
    @property
    def is_video_rule(self) -> bool:
        """是否为视频规则"""
        return 'video' in self.content_types
    
    @property
    def is_audio_rule(self) -> bool:
        """是否为音频规则"""
        return 'audio' in self.content_types


class SensitiveWord(Base):
    """
    敏感词库表
    
    存储各类敏感词汇和正则表达式
    """
    __tablename__ = "sensitive_words"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="词汇ID")
    word = Column(String(255), nullable=False, comment="敏感词")
    category = Column(String(100), nullable=False, comment="分类")
    severity = Column(String(20), default='medium', nullable=False, comment="严重程度")
    is_regex = Column(Boolean, default=False, comment="是否为正则表达式")
    is_active = Column(Boolean, default=True, comment="是否激活")
    hit_count = Column(Integer, default=0, comment="命中次数")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 约束
    __table_args__ = (
        CheckConstraint("category IN ('politics', 'violence', 'pornography', 'spam', 'hate', 'drugs', 'gambling')", name='ck_category'),
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')", name='ck_word_severity'),
        Index('ix_word_category', 'category'),
        Index('ix_word_active', 'is_active'),
        Index('ix_word_text', 'word'),
        {"comment": "敏感词库表"}
    )
    
    def __repr__(self):
        return f"<SensitiveWord(id={self.id}, word='{self.word}', category='{self.category}')>"
    
    def increment_hit_count(self):
        """增加命中次数"""
        self.hit_count += 1
        self.updated_at = func.now()


class Whitelist(Base):
    """
    白名单表
    
    存储各种类型的白名单项目
    """
    __tablename__ = "whitelists"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="白名单ID")
    type = Column(String(50), nullable=False, comment="白名单类型")
    value = Column(String(500), nullable=False, comment="白名单值")
    description = Column(Text, comment="描述")
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    created_by = Column(UUID(as_uuid=True), comment="创建者ID")
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    expires_at = Column(DateTime(timezone=True), comment="过期时间")
    
    # 约束
    __table_args__ = (
        CheckConstraint("type IN ('user', 'domain', 'keyword', 'ip', 'hash', 'content_id')", name='ck_whitelist_type'),
        Index('ix_whitelist_type', 'type'),
        Index('ix_whitelist_value', 'value'),
        Index('ix_whitelist_active', 'is_active'),
        {"comment": "白名单表"}
    )
    
    def __repr__(self):
        return f"<Whitelist(id={self.id}, type='{self.type}', value='{self.value}')>"
    
    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return self.expires_at < datetime.utcnow()


class Appeal(Base):
    """
    申诉记录表
    
    存储用户对审核结果的申诉信息
    """
    __tablename__ = "appeals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="申诉ID")
    task_id = Column(UUID(as_uuid=True), ForeignKey("moderation_tasks.id"), nullable=False, comment="任务ID")
    user_id = Column(UUID(as_uuid=True), nullable=False, comment="用户ID")
    reason = Column(Text, nullable=False, comment="申诉理由")
    evidence = Column(JSON, comment="证据材料")
    
    status = Column(String(50), default='pending', nullable=False, comment="申诉状态")
    reviewer_id = Column(UUID(as_uuid=True), comment="审核员ID")
    reviewer_notes = Column(Text, comment="审核员备注")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), comment="更新时间")
    reviewed_at = Column(DateTime(timezone=True), comment="审核时间")
    
    # 关系映射
    task = relationship("ModerationTask", back_populates="appeals")
    
    # 约束
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected', 'processing')", name='ck_appeal_status'),
        Index('ix_appeal_task', 'task_id'),
        Index('ix_appeal_user', 'user_id'),
        Index('ix_appeal_status', 'status'),
        {"comment": "申诉记录表"}
    )
    
    def __repr__(self):
        return f"<Appeal(id={self.id}, task_id={self.task_id}, status='{self.status}')>"
    
    @property
    def is_pending(self) -> bool:
        """是否为待处理状态"""
        return self.status == 'pending'
    
    def update_status(self, status: str, reviewer_id: Optional[uuid.UUID] = None, notes: Optional[str] = None):
        """
        更新申诉状态
        
        Args:
            status: 新状态
            reviewer_id: 审核员ID
            notes: 审核员备注
        """
        self.status = status
        self.updated_at = func.now()
        
        if reviewer_id:
            self.reviewer_id = reviewer_id
        
        if notes:
            self.reviewer_notes = notes
        
        if status in ['approved', 'rejected']:
            self.reviewed_at = func.now()


class ModerationLog(Base):
    """
    审核日志表
    
    记录审核过程中的详细操作日志
    """
    __tablename__ = "moderation_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="日志ID")
    task_id = Column(UUID(as_uuid=True), ForeignKey("moderation_tasks.id"), nullable=False, comment="任务ID")
    action = Column(String(100), nullable=False, comment="操作动作")
    actor_type = Column(String(50), nullable=False, comment="操作者类型")
    actor_id = Column(String(255), comment="操作者ID")
    details = Column(JSON, comment="详细信息")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(String(500), comment="用户代理")
    
    created_at = Column(DateTime(timezone=True), default=func.now(), comment="创建时间")
    
    # 关系映射
    task = relationship("ModerationTask", back_populates="logs")
    
    # 约束
    __table_args__ = (
        CheckConstraint("actor_type IN ('system', 'user', 'admin', 'api', 'ml_model')", name='ck_actor_type'),
        Index('ix_log_task', 'task_id'),
        Index('ix_log_action', 'action'),
        Index('ix_log_actor', 'actor_type', 'actor_id'),
        Index('ix_log_created', 'created_at'),
        {"comment": "审核日志表"}
    )
    
    def __repr__(self):
        return f"<ModerationLog(id={self.id}, action='{self.action}', actor='{self.actor_type}')>"