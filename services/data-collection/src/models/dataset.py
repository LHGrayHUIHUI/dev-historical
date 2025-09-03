"""
数据集模型定义

管理上传的文件和数据集信息
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class Dataset(BaseModel):
    """数据集模型
    
    表示一个完整的数据集，包含文件信息、处理状态等
    """
    
    __tablename__ = "datasets"
    
    # 数据集名称
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
        doc="数据集名称"
    )
    
    # 描述信息
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="数据集描述"
    )
    
    # 关联的数据源ID
    source_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("data_sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="关联的数据源ID"
    )
    
    # 原始文件路径
    file_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        doc="文件在对象存储中的路径"
    )
    
    # 文件大小(字节)
    file_size: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        doc="文件大小(字节)"
    )
    
    # 文件类型
    file_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        doc="文件MIME类型"
    )
    
    # 文件哈希值(用于去重)
    file_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        index=True,
        doc="文件SHA256哈希值"
    )
    
    # 元数据信息JSON
    file_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        doc="文件元数据信息"
    )
    
    # 处理状态 (pending, processing, completed, failed)
    processing_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        index=True,
        doc="处理状态"
    )
    
    # 错误信息
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="处理错误信息"
    )
    
    # 创建者ID
    created_by: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
        doc="创建者用户ID"
    )
    
    # 处理完成时间
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        doc="处理完成时间"
    )
    
    # 文本内容统计
    text_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="提取的文本片段数量"
    )
    
    total_words: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="总词数"
    )
    
    total_chars: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="总字符数"
    )
    
    # 关联关系
    source: Mapped["DataSource"] = relationship(
        "DataSource",
        back_populates="datasets",
        doc="关联的数据源"
    )
    
    text_contents: Mapped[list["TextContent"]] = relationship(
        "TextContent",
        back_populates="dataset",
        cascade="all, delete-orphan",
        doc="提取的文本内容列表"
    )
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"<Dataset(id={self.id}, name='{self.name}', status='{self.processing_status}')>"
    
    @property
    def is_processed(self) -> bool:
        """检查是否已处理完成"""
        return self.processing_status == "completed"
    
    @property
    def is_processing(self) -> bool:
        """检查是否正在处理中"""
        return self.processing_status == "processing"
    
    @property
    def has_error(self) -> bool:
        """检查是否有错误"""
        return self.processing_status == "failed"
    
    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """获取元数据值
        
        Args:
            key: 元数据键名
            default: 默认值
            
        Returns:
            元数据值或默认值
        """
        if not self.file_metadata:
            return default
        return self.file_metadata.get(key, default)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """更新元数据值
        
        Args:
            key: 元数据键名
            value: 元数据值
        """
        if not self.file_metadata:
            self.file_metadata = {}
        self.file_metadata[key] = value
    
    def update_processing_status(self, status: str, error_message: str = None) -> None:
        """更新处理状态
        
        Args:
            status: 新状态
            error_message: 错误信息(可选)
        """
        self.processing_status = status
        if error_message:
            self.error_message = error_message
        if status == "completed":
            self.processed_at = datetime.utcnow()
    
    def calculate_statistics(self) -> None:
        """计算文本统计信息"""
        if self.text_contents:
            self.text_count = len(self.text_contents)
            self.total_words = sum(content.word_count or 0 for content in self.text_contents)
            self.total_chars = sum(content.char_count or 0 for content in self.text_contents)