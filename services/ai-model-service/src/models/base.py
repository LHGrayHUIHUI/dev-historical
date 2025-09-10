"""
基础响应模型
"""

from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, Field

T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    """统一响应格式基类"""
    
    success: bool = Field(..., description="请求是否成功")
    message: str = Field("", description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    error_code: Optional[str] = Field(None, description="错误代码")
    timestamp: float = Field(..., description="响应时间戳")
    
    class Config:
        """Pydantic配置"""
        json_encoders = {
            # 自定义JSON编码器
        }


class ErrorResponse(BaseResponse[None]):
    """错误响应模型"""
    
    success: bool = Field(False, description="请求失败")
    error_code: str = Field(..., description="错误代码")
    error_detail: Optional[dict] = Field(None, description="错误详情")


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应模型"""
    
    items: list[T] = Field(..., description="数据项列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="页面大小")
    total_pages: int = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")