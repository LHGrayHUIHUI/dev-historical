"""
Pydantic模式定义

定义API请求和响应的数据模式
提供数据验证、序列化和文档生成支持
"""

from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing" 
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AccountStatus(str, Enum):
    """账号状态枚举"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    DISABLED = "disabled"


class PlatformType(str, Enum):
    """平台类型枚举"""
    SOCIAL_MEDIA = "social_media"
    BLOG = "blog"
    NEWS = "news"
    VIDEO = "video"


class AuthType(str, Enum):
    """认证类型枚举"""
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    COOKIE = "cookie"
    JWT = "jwt"


# ==================== 基础响应模式 ====================

class BaseResponse(BaseModel):
    """基础响应模式"""
    success: bool = True
    message: str = "操作成功"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationInfo(BaseModel):
    """分页信息模式"""
    page: int = Field(ge=1, description="当前页码")
    page_size: int = Field(ge=1, le=100, description="每页数量")
    total: int = Field(ge=0, description="总记录数")
    pages: int = Field(ge=0, description="总页数")


class PaginationResponse(BaseResponse):
    """分页响应模式"""
    pagination: PaginationInfo


# ==================== 平台相关模式 ====================

class PlatformBase(BaseModel):
    """平台基础模式"""
    platform_name: str = Field(..., max_length=50, description="平台名称")
    platform_type: PlatformType = Field(..., description="平台类型")
    display_name: str = Field(..., max_length=100, description="显示名称")
    api_endpoint: Optional[str] = Field(None, max_length=255, description="API接口地址")
    auth_type: Optional[AuthType] = Field(None, description="认证类型")
    config_schema: Optional[Dict[str, Any]] = Field(None, description="配置模式")
    rate_limit_per_hour: int = Field(default=100, ge=1, description="每小时限流数量")


class PlatformCreateSchema(PlatformBase):
    """创建平台模式"""
    pass


class PlatformUpdateSchema(BaseModel):
    """更新平台模式"""
    display_name: Optional[str] = Field(None, max_length=100, description="显示名称")
    api_endpoint: Optional[str] = Field(None, max_length=255, description="API接口地址")
    auth_type: Optional[AuthType] = Field(None, description="认证类型")
    config_schema: Optional[Dict[str, Any]] = Field(None, description="配置模式")
    is_active: Optional[bool] = Field(None, description="是否激活")
    rate_limit_per_hour: Optional[int] = Field(None, ge=1, description="每小时限流数量")


class PlatformSchema(PlatformBase):
    """平台完整模式"""
    id: int = Field(..., description="平台ID")
    is_active: bool = Field(..., description="是否激活")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    # 统计信息
    total_accounts: Optional[int] = Field(None, description="账号总数")
    active_accounts: Optional[int] = Field(None, description="活跃账号数")
    
    class Config:
        from_attributes = True


# ==================== 账号相关模式 ====================

class AccountBase(BaseModel):
    """账号基础模式"""
    account_name: str = Field(..., max_length=100, description="账号名称")
    account_identifier: Optional[str] = Field(None, max_length=255, description="账号标识符")
    daily_quota: int = Field(default=50, ge=0, description="每日配额")


class AccountCreateSchema(AccountBase):
    """创建账号模式"""
    platform_id: int = Field(..., description="平台ID")
    auth_credentials: Dict[str, Any] = Field(..., description="认证凭据")
    expires_at: Optional[datetime] = Field(None, description="凭据过期时间")


class AccountUpdateSchema(BaseModel):
    """更新账号模式"""
    account_name: Optional[str] = Field(None, max_length=100, description="账号名称")
    account_identifier: Optional[str] = Field(None, max_length=255, description="账号标识符")
    auth_credentials: Optional[Dict[str, Any]] = Field(None, description="认证凭据")
    account_status: Optional[AccountStatus] = Field(None, description="账号状态")
    daily_quota: Optional[int] = Field(None, ge=0, description="每日配额")
    expires_at: Optional[datetime] = Field(None, description="凭据过期时间")


class AccountSchema(AccountBase):
    """账号完整模式"""
    id: int = Field(..., description="账号ID")
    platform_id: int = Field(..., description="平台ID")
    account_status: AccountStatus = Field(..., description="账号状态")
    used_quota: int = Field(..., description="已使用配额")
    last_used_at: Optional[datetime] = Field(None, description="最后使用时间")
    expires_at: Optional[datetime] = Field(None, description="凭据过期时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    # 关联信息
    platform_name: Optional[str] = Field(None, description="平台名称")
    
    # 计算属性
    is_available: Optional[bool] = Field(None, description="是否可用")
    quota_remaining: Optional[int] = Field(None, description="剩余配额")
    
    class Config:
        from_attributes = True


# ==================== 任务相关模式 ====================

class PublishConfig(BaseModel):
    """发布配置模式"""
    tags: Optional[List[str]] = Field(None, description="标签列表")
    category: Optional[str] = Field(None, description="分类")
    visibility: str = Field(default="public", description="可见性")
    
    @validator('visibility')
    def validate_visibility(cls, v):
        valid_options = ['public', 'private', 'friends']
        if v not in valid_options:
            raise ValueError(f'可见性必须是以下之一: {valid_options}')
        return v


class TaskBase(BaseModel):
    """任务基础模式"""
    title: Optional[str] = Field(None, max_length=500, description="标题")
    content: str = Field(..., max_length=10000, description="发布内容")
    media_urls: Optional[List[str]] = Field(None, description="媒体文件URL列表")
    scheduled_at: Optional[datetime] = Field(None, description="计划发布时间")
    publish_config: Optional[PublishConfig] = Field(None, description="发布配置")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('发布内容不能为空')
        return v.strip()
    
    @validator('scheduled_at')
    def validate_scheduled_at(cls, v):
        if v and v <= datetime.utcnow():
            raise ValueError('计划发布时间必须晚于当前时间')
        return v


class TaskCreateSchema(TaskBase):
    """创建任务模式"""
    platforms: List[str] = Field(..., min_items=1, max_items=10, description="目标平台列表")
    content_id: Optional[int] = Field(None, description="关联内容ID")
    
    @validator('platforms')
    def validate_platforms(cls, v):
        if not v:
            raise ValueError('必须选择至少一个发布平台')
        # 去重
        return list(set(v))


class TaskUpdateSchema(BaseModel):
    """更新任务模式"""
    scheduled_at: Optional[datetime] = Field(None, description="计划发布时间")
    max_retries: Optional[int] = Field(None, ge=0, le=10, description="最大重试次数")


class TaskSchema(TaskBase):
    """任务完整模式"""
    id: int = Field(..., description="任务ID")
    task_uuid: UUID = Field(..., description="任务UUID")
    content_id: Optional[int] = Field(None, description="关联内容ID")
    platform_id: int = Field(..., description="平台ID")
    account_id: int = Field(..., description="账号ID")
    status: TaskStatus = Field(..., description="任务状态")
    retry_count: int = Field(..., description="重试次数")
    max_retries: int = Field(..., description="最大重试次数")
    error_message: Optional[str] = Field(None, description="错误信息")
    platform_post_id: Optional[str] = Field(None, description="平台帖子ID")
    published_url: Optional[str] = Field(None, description="发布URL")
    published_at: Optional[datetime] = Field(None, description="实际发布时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    # 关联信息
    platform_name: Optional[str] = Field(None, description="平台名称")
    account_name: Optional[str] = Field(None, description="账号名称")
    
    class Config:
        from_attributes = True


class TaskStatusSchema(BaseModel):
    """任务状态模式"""
    task_uuid: UUID = Field(..., description="任务UUID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: Optional[int] = Field(None, ge=0, le=100, description="进度百分比")
    message: Optional[str] = Field(None, description="状态消息")
    updated_at: datetime = Field(..., description="更新时间")


# ==================== 统计相关模式 ====================

class StatsBase(BaseModel):
    """统计基础模式"""
    total_tasks: int = Field(ge=0, description="总任务数")
    successful_tasks: int = Field(ge=0, description="成功任务数")
    failed_tasks: int = Field(ge=0, description="失败任务数")


class StatsSchema(StatsBase):
    """统计完整模式"""
    platform: str = Field(..., description="平台名称")
    success_rate: float = Field(ge=0, le=100, description="成功率(%)")
    avg_publish_time: float = Field(ge=0, description="平均发布时间(秒)")


class StatsPeriodSchema(BaseModel):
    """统计周期模式"""
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")


class StatsResponseSchema(BaseResponse):
    """统计响应模式"""
    data: Dict[str, Any] = Field(..., description="统计数据")


# ==================== 列表响应模式 ====================

class TaskListResponse(PaginationResponse):
    """任务列表响应"""
    data: Dict[str, Any] = Field(..., description="响应数据")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "获取成功",
                "timestamp": "2024-01-01T00:00:00Z",
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "total": 100,
                    "pages": 5
                },
                "data": {
                    "tasks": []
                }
            }
        }


class PlatformListResponse(BaseResponse):
    """平台列表响应"""
    data: Dict[str, Any] = Field(..., description="响应数据")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "获取成功", 
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {
                    "platforms": []
                }
            }
        }


class AccountListResponse(BaseResponse):
    """账号列表响应"""
    data: Dict[str, Any] = Field(..., description="响应数据")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "获取成功",
                "timestamp": "2024-01-01T00:00:00Z", 
                "data": {
                    "accounts": []
                }
            }
        }


# ==================== 特殊响应模式 ====================

class TaskCreateResponse(BaseResponse):
    """任务创建响应"""
    data: Dict[str, Any] = Field(..., description="响应数据")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "任务创建成功",
                "timestamp": "2024-01-01T00:00:00Z",
                "data": {
                    "task_uuids": ["uuid1", "uuid2"],
                    "total_tasks": 2
                }
            }
        }