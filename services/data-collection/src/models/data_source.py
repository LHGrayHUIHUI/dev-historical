"""
数据源模型定义

管理不同的数据源类型和配置信息
"""

from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class DataSource(BaseModel):
    """数据源模型
    
    用于管理不同类型的数据源，如文件上传、API接口、爬虫等
    """
    
    __tablename__ = "data_sources"
    
    # 数据源名称
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="数据源名称"
    )
    
    # 数据源类型 (file_upload, api, crawler, manual)
    type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="数据源类型"
    )
    
    # 描述信息
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="数据源描述"
    )
    
    # 配置信息JSON
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        doc="数据源配置信息"
    )
    
    # 状态 (active, inactive, archived)
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        index=True,
        doc="数据源状态"
    )
    
    # 创建者ID
    created_by: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        doc="创建者用户ID"
    )
    
    # 关联的数据集
    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset",
        back_populates="source",
        cascade="all, delete-orphan",
        doc="关联的数据集列表"
    )
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"<DataSource(id={self.id}, name='{self.name}', type='{self.type}')>"
    
    @property
    def is_active(self) -> bool:
        """检查数据源是否活跃"""
        return self.status == "active"
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键名
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        if not self.config:
            return default
        return self.config.get(key, default)
    
    def update_config(self, key: str, value: Any) -> None:
        """更新配置值
        
        Args:
            key: 配置键名
            value: 配置值
        """
        if not self.config:
            self.config = {}
        self.config[key] = value