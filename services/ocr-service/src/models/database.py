"""
OCR服务数据库模型定义

定义OCR任务、结果、配置等相关的数据库模型，
使用SQLAlchemy ORM进行数据库操作。

Author: 开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from enum import Enum

Base = declarative_base()


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 等待中
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"   # 已取消


class OCREngine(str, Enum):
    """OCR引擎枚举"""
    PADDLEOCR = "paddleocr"  # PaddleOCR
    TESSERACT = "tesseract"  # Tesseract
    EASYOCR = "easyocr"      # EasyOCR


class TimestampMixin:
    """时间戳混入类
    
    为模型提供标准的创建时间和更新时间字段
    """
    
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(), 
        nullable=False,
        comment="更新时间"
    )


class OCRTask(Base, TimestampMixin):
    """OCR任务表
    
    存储OCR识别任务的基本信息和处理状态，
    支持单个文件和批量处理任务的管理。
    """
    
    __tablename__ = "ocr_tasks"
    __table_args__ = {
        'comment': 'OCR识别任务表'
    }
    
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        comment="任务ID"
    )
    dataset_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="数据集ID"
    )
    image_path = Column(
        String(500), 
        nullable=False,
        comment="图像文件路径"
    )
    image_size = Column(
        JSONB,
        nullable=True,
        comment="图像尺寸信息 {width: int, height: int}"
    )
    processing_status = Column(
        String(50), 
        default=TaskStatus.PENDING.value, 
        nullable=False,
        index=True,
        comment="处理状态"
    )
    ocr_engine = Column(
        String(50),
        default=OCREngine.PADDLEOCR.value,
        nullable=False,
        comment="使用的OCR引擎"
    )
    confidence_threshold = Column(
        Float,
        default=0.8,
        nullable=False,
        comment="置信度阈值"
    )
    language_codes = Column(
        String(100),
        default="zh,en",
        nullable=False,
        comment="支持的语言代码，逗号分隔"
    )
    preprocessing_config = Column(
        JSONB,
        nullable=True,
        comment="预处理配置参数"
    )
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="处理开始时间"
    )
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="处理完成时间"
    )
    error_message = Column(
        Text,
        nullable=True,
        comment="错误信息"
    )
    created_by = Column(
        UUID(as_uuid=True),
        nullable=False,
        comment="创建者用户ID"
    )
    
    # 关系定义
    ocr_result = relationship(
        "OCRResult", 
        back_populates="task", 
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        """对象字符串表示"""
        return f"<OCRTask(id={self.id}, status={self.processing_status}, engine={self.ocr_engine})>"


class OCRResult(Base, TimestampMixin):
    """OCR识别结果表
    
    存储OCR识别的详细结果，包括提取的文本内容、
    置信度、边界框信息、文本块等详细数据。
    """
    
    __tablename__ = "ocr_results"
    __table_args__ = {
        'comment': 'OCR识别结果表'
    }
    
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        comment="结果ID"
    )
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ocr_tasks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="关联的任务ID"
    )
    text_content = Column(
        Text,
        nullable=False,
        comment="识别的完整文本内容"
    )
    confidence_score = Column(
        Float,
        nullable=False,
        comment="整体置信度分数"
    )
    bounding_boxes = Column(
        JSONB,
        nullable=True,
        comment="文字区域边界框坐标信息"
    )
    text_blocks = Column(
        JSONB,
        nullable=True,
        comment="文本块详细信息"
    )
    language_detected = Column(
        String(50),
        nullable=True,
        comment="检测到的语言"
    )
    word_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="词语数量"
    )
    char_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="字符数量"
    )
    processing_time = Column(
        Float,
        nullable=False,
        comment="处理时间（秒）"
    )
    metadata = Column(
        JSONB,
        nullable=True,
        comment="额外的元数据信息"
    )
    
    # 全文搜索向量（PostgreSQL特性）
    text_vector = Column(
        # 使用生成列进行全文搜索索引
        Text,
        nullable=True,
        comment="全文搜索向量"
    )
    
    # 关系定义
    task = relationship(
        "OCRTask", 
        back_populates="ocr_result"
    )
    
    def __repr__(self):
        """对象字符串表示"""
        return f"<OCRResult(id={self.id}, task_id={self.task_id}, confidence={self.confidence_score:.2f})>"


class OCRConfig(Base, TimestampMixin):
    """OCR配置表
    
    存储不同的OCR引擎配置预设，支持用户自定义
    和管理员预设的配置模板。
    """
    
    __tablename__ = "ocr_configs"
    __table_args__ = {
        'comment': 'OCR配置预设表'
    }
    
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        comment="配置ID"
    )
    name = Column(
        String(100),
        nullable=False,
        comment="配置名称"
    )
    description = Column(
        Text,
        nullable=True,
        comment="配置描述"
    )
    engine = Column(
        String(50),
        nullable=False,
        comment="OCR引擎类型"
    )
    config = Column(
        JSONB,
        nullable=False,
        comment="引擎配置参数JSON"
    )
    is_default = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否为默认配置"
    )
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="是否启用"
    )
    created_by = Column(
        UUID(as_uuid=True),
        nullable=False,
        comment="创建者用户ID"
    )
    
    def __repr__(self):
        """对象字符串表示"""
        return f"<OCRConfig(id={self.id}, name={self.name}, engine={self.engine})>"


# 创建索引
from sqlalchemy import Index, text

# OCR任务相关索引
Index('idx_ocr_tasks_status', OCRTask.processing_status)
Index('idx_ocr_tasks_created_by', OCRTask.created_by)
Index('idx_ocr_tasks_engine', OCRTask.ocr_engine)
Index('idx_ocr_tasks_created_at', OCRTask.created_at.desc())

# OCR结果相关索引
Index('idx_ocr_results_task_id', OCRResult.task_id)
Index('idx_ocr_results_confidence', OCRResult.confidence_score)
Index('idx_ocr_results_char_count', OCRResult.char_count)

# PostgreSQL全文搜索索引（需要在迁移脚本中创建）
# CREATE INDEX idx_ocr_results_text_search ON ocr_results USING GIN(to_tsvector('chinese', text_content));