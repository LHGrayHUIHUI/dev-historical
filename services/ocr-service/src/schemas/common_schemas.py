"""
通用数据模型schemas

定义通用的请求/响应模型，包括基础响应格式、
错误响应、分页查询等通用数据结构。

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    """
    基础响应模型
    
    所有API响应的基础格式，包含成功状态、
    消息和数据字段。
    """
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """
    错误响应模型
    
    统一的错误响应格式，包含错误代码、
    消息和详细信息。
    """
    success: bool = Field(False, description="请求失败标识")
    error_code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间戳")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginationQuery(BaseModel):
    """
    分页查询模型
    
    用于列表查询接口的分页参数。
    """
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页大小")
    
    @property
    def offset(self) -> int:
        """计算偏移量"""
        return (self.page - 1) * self.size
    
    @property
    def limit(self) -> int:
        """获取限制数量"""
        return self.size


class PaginationResponse(BaseModel):
    """
    分页响应模型
    
    包含分页信息的响应格式。
    """
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页大小")
    pages: int = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")
    
    @classmethod
    def create(
        cls, 
        total: int, 
        page: int, 
        size: int
    ) -> "PaginationResponse":
        """
        创建分页响应
        
        Args:
            total: 总记录数
            page: 当前页码
            size: 每页大小
            
        Returns:
            分页响应实例
        """
        pages = (total + size - 1) // size if total > 0 else 0
        
        return cls(
            total=total,
            page=page,
            size=size,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1
        )


class HealthStatus(BaseModel):
    """
    健康状态模型
    
    用于健康检查接口的响应格式。
    """
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    uptime: float = Field(..., description="运行时间（秒）")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ServiceInfo(BaseModel):
    """
    服务信息模型
    
    包含服务的基本信息和配置。
    """
    name: str = Field(..., description="服务名称")
    version: str = Field(..., description="服务版本")
    description: str = Field(..., description="服务描述")
    environment: str = Field(..., description="运行环境")
    start_time: datetime = Field(..., description="启动时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsData(BaseModel):
    """
    指标数据模型
    
    用于返回服务性能指标和统计信息。
    """
    name: str = Field(..., description="指标名称")
    value: float = Field(..., description="指标值")
    unit: str = Field(..., description="指标单位")
    description: Optional[str] = Field(None, description="指标描述")
    timestamp: datetime = Field(default_factory=datetime.now, description="采集时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }