"""
数据库基础模型定义

提供所有模型的基础类和通用字段
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有数据库模型的基类"""
    
    # 使用类型提示改进SQLAlchemy 2.0的类型支持
    type_annotation_map = {
        UUID: PGUUID(as_uuid=True),
        datetime: DateTime(timezone=True)
    }


class BaseModel(Base):
    """包含通用字段的基础模型"""
    
    __abstract__ = True
    
    # 主键ID
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        doc="主键ID"
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="创建时间"
    )
    
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        doc="更新时间"
    )
    
    def dict(self, **kwargs) -> dict[str, Any]:
        """转换为字典格式
        
        Returns:
            字典格式的模型数据
        """
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
    
    def __repr__(self) -> str:
        """字符串表示"""
        class_name = self.__class__.__name__
        return f"<{class_name}(id={self.id})>"