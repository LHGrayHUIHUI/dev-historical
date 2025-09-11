"""
Pydantic数据模式定义

定义所有API请求和响应的数据结构
提供数据验证和序列化功能
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, validator, Field
from datetime import datetime
from enum import Enum
import uuid


class BaseResponse(BaseModel):
    """基础响应模式"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")


class PaginationResponse(BaseModel):
    """分页响应模式"""
    page: int = Field(..., ge=1, description="当前页码")
    size: int = Field(..., ge=1, le=100, description="每页大小")
    total: int = Field(..., ge=0, description="总记录数")
    pages: int = Field(..., ge=0, description="总页数")


# 平台相关模式

class PlatformConfigSchema(BaseModel):
    """平台配置模式"""
    id: int = Field(..., description="平台ID")
    name: str = Field(..., description="平台名称")
    display_name: str = Field(..., description="显示名称")
    platform_type: str = Field(..., description="平台类型")
    api_base_url: Optional[str] = Field(None, description="API基础URL")
    oauth_config: Dict[str, Any] = Field(..., description="OAuth配置")
    rate_limits: Dict[str, Any] = Field(..., description="速率限制")
    features: Dict[str, Any] = Field(..., description="平台功能")
    is_active: bool = Field(..., description="是否激活")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        from_attributes = True


class PlatformCreateSchema(BaseModel):
    """创建平台配置请求模式"""
    name: str = Field(..., min_length=1, max_length=100, description="平台名称")
    display_name: str = Field(..., min_length=1, max_length=200, description="显示名称")
    platform_type: str = Field(..., description="平台类型")
    api_base_url: Optional[str] = Field(None, description="API基础URL")
    oauth_config: Dict[str, Any] = Field(..., description="OAuth配置")
    rate_limits: Dict[str, Any] = Field(default_factory=dict, description="速率限制")
    features: Dict[str, Any] = Field(default_factory=dict, description="平台功能")
    is_active: bool = Field(True, description="是否激活")
    
    @validator('platform_type')
    def validate_platform_type(cls, v):
        """验证平台类型"""
        valid_types = ['social_media', 'blog', 'news', 'short_video', 'content']
        if v not in valid_types:
            raise ValueError(f'平台类型必须是以下之一: {valid_types}')
        return v


# 账号相关模式

class AccountSchema(BaseModel):
    """账号信息模式"""
    id: int = Field(..., description="账号ID")
    platform_id: int = Field(..., description="平台ID")
    user_id: int = Field(..., description="用户ID")
    account_name: str = Field(..., description="账号名称")
    account_id: Optional[str] = Field(None, description="平台账号ID")
    display_name: Optional[str] = Field(None, description="显示名称")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    bio: Optional[str] = Field(None, description="个人简介")
    
    # 统计信息
    follower_count: int = Field(..., description="粉丝数")
    following_count: int = Field(..., description="关注数")
    post_count: int = Field(..., description="发布数")
    
    # 状态信息
    verification_status: str = Field(..., description="认证状态")
    account_type: str = Field(..., description="账号类型")
    status: str = Field(..., description="账号状态")
    
    last_sync_at: Optional[datetime] = Field(None, description="最后同步时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    
    # 关联信息
    platform: Optional[PlatformConfigSchema] = Field(None, description="平台信息")
    
    class Config:
        from_attributes = True


class AccountCreateSchema(BaseModel):
    """创建账号请求模式"""
    platform_name: str = Field(..., description="平台名称")
    auth_code: str = Field(..., description="授权码")
    redirect_uri: Optional[str] = Field(None, description="回调URI")
    
    @validator('platform_name')
    def validate_platform_name(cls, v):
        """验证平台名称"""
        supported_platforms = ['weibo', 'wechat', 'douyin', 'toutiao', 'baijiahao']
        if v.lower() not in supported_platforms:
            raise ValueError(f'平台名称必须是以下之一: {supported_platforms}')
        return v.lower()


class AccountUpdateSchema(BaseModel):
    """更新账号请求模式"""
    display_name: Optional[str] = Field(None, max_length=200, description="显示名称")
    bio: Optional[str] = Field(None, description="个人简介")
    account_type: Optional[str] = Field(None, description="账号类型")
    status: Optional[str] = Field(None, description="账号状态")
    
    @validator('account_type')
    def validate_account_type(cls, v):
        """验证账号类型"""
        if v is not None:
            valid_types = ['personal', 'business', 'creator', 'organization']
            if v not in valid_types:
                raise ValueError(f'账号类型必须是以下之一: {valid_types}')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        """验证账号状态"""
        if v is not None:
            valid_statuses = ['active', 'suspended', 'expired', 'error', 'pending']
            if v not in valid_statuses:
                raise ValueError(f'账号状态必须是以下之一: {valid_statuses}')
        return v


class AccountStatsSchema(BaseModel):
    """账号统计信息模式"""
    follower_count: int = Field(..., description="粉丝数")
    following_count: int = Field(..., description="关注数")
    post_count: int = Field(..., description="发布数")
    engagement_rate: Optional[float] = Field(None, description="互动率")
    avg_likes: Optional[int] = Field(None, description="平均点赞数")
    avg_comments: Optional[int] = Field(None, description="平均评论数")
    avg_shares: Optional[int] = Field(None, description="平均分享数")


# 账号同步相关模式

class AccountSyncRequestSchema(BaseModel):
    """账号同步请求模式"""
    account_id: int = Field(..., description="账号ID")
    sync_types: List[str] = Field(default=['profile', 'stats'], description="同步类型")
    force: bool = Field(False, description="是否强制同步")
    
    @validator('sync_types')
    def validate_sync_types(cls, v):
        """验证同步类型"""
        valid_types = ['profile', 'stats', 'posts', 'followers', 'full']
        for sync_type in v:
            if sync_type not in valid_types:
                raise ValueError(f'同步类型必须是以下之一: {valid_types}')
        return v


class AccountSyncLogSchema(BaseModel):
    """账号同步日志模式"""
    id: int = Field(..., description="日志ID")
    account_id: int = Field(..., description="账号ID")
    sync_type: str = Field(..., description="同步类型")
    status: str = Field(..., description="同步状态")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误消息")
    sync_data: Optional[Dict[str, Any]] = Field(None, description="同步数据")
    duration_seconds: Optional[int] = Field(None, description="持续时间(秒)")
    
    class Config:
        from_attributes = True


class BatchSyncRequestSchema(BaseModel):
    """批量同步请求模式"""
    account_ids: List[int] = Field(..., min_items=1, max_items=50, description="账号ID列表")
    sync_types: List[str] = Field(default=['profile', 'stats'], description="同步类型")
    
    @validator('sync_types')
    def validate_sync_types(cls, v):
        """验证同步类型"""
        valid_types = ['profile', 'stats', 'posts', 'followers', 'full']
        for sync_type in v:
            if sync_type not in valid_types:
                raise ValueError(f'同步类型必须是以下之一: {valid_types}')
        return v


# OAuth相关模式

class OAuthUrlSchema(BaseModel):
    """OAuth授权URL模式"""
    authorize_url: str = Field(..., description="授权URL")
    state: str = Field(..., description="状态码")
    expires_at: datetime = Field(..., description="过期时间")


class OAuthCallbackSchema(BaseModel):
    """OAuth回调模式"""
    platform_name: str = Field(..., description="平台名称")
    code: str = Field(..., description="授权码")
    state: str = Field(..., description="状态码")
    error: Optional[str] = Field(None, description="错误信息")
    error_description: Optional[str] = Field(None, description="错误描述")


class TokenRefreshRequestSchema(BaseModel):
    """令牌刷新请求模式"""
    account_id: int = Field(..., description="账号ID")
    force: bool = Field(False, description="是否强制刷新")


# 权限管理相关模式

class AccountPermissionSchema(BaseModel):
    """账号权限模式"""
    id: int = Field(..., description="权限ID")
    account_id: int = Field(..., description="账号ID")
    user_id: int = Field(..., description="用户ID")
    permission_type: str = Field(..., description="权限类型")
    granted_by: Optional[int] = Field(None, description="授权者ID")
    granted_at: datetime = Field(..., description="授权时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    is_active: bool = Field(..., description="是否激活")
    
    class Config:
        from_attributes = True


class GrantPermissionSchema(BaseModel):
    """授权权限请求模式"""
    account_id: int = Field(..., description="账号ID")
    user_id: int = Field(..., description="用户ID")
    permission_type: str = Field(..., description="权限类型")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    
    @validator('permission_type')
    def validate_permission_type(cls, v):
        """验证权限类型"""
        valid_types = ['read', 'write', 'admin', 'publish', 'manage']
        if v not in valid_types:
            raise ValueError(f'权限类型必须是以下之一: {valid_types}')
        return v


# API使用统计相关模式

class ApiUsageStatsSchema(BaseModel):
    """API使用统计模式"""
    platform_name: str = Field(..., description="平台名称")
    endpoint: str = Field(..., description="API端点")
    method: str = Field(..., description="HTTP方法")
    total_requests: int = Field(..., description="总请求数")
    error_requests: int = Field(..., description="错误请求数")
    avg_response_time: float = Field(..., description="平均响应时间(毫秒)")
    success_rate: float = Field(..., description="成功率")
    date: datetime = Field(..., description="统计日期")


class ApiUsageReportSchema(BaseModel):
    """API使用报告模式"""
    platform_stats: List[ApiUsageStatsSchema] = Field(..., description="平台统计")
    total_requests: int = Field(..., description="总请求数")
    total_errors: int = Field(..., description="总错误数")
    overall_success_rate: float = Field(..., description="总体成功率")
    report_period: str = Field(..., description="报告周期")
    generated_at: datetime = Field(..., description="生成时间")


# 搜索和过滤相关模式

class AccountSearchSchema(BaseModel):
    """账号搜索请求模式"""
    user_id: Optional[int] = Field(None, description="用户ID")
    platform_name: Optional[str] = Field(None, description="平台名称")
    account_name: Optional[str] = Field(None, description="账号名称")
    status: Optional[str] = Field(None, description="账号状态")
    account_type: Optional[str] = Field(None, description="账号类型")
    verification_status: Optional[str] = Field(None, description="认证状态")
    
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页大小")
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", description="排序顺序")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        """验证排序顺序"""
        if v not in ['asc', 'desc']:
            raise ValueError('排序顺序必须是asc或desc')
        return v


class AccountListResponseSchema(BaseModel):
    """账号列表响应模式"""
    accounts: List[AccountSchema] = Field(..., description="账号列表")
    pagination: PaginationResponse = Field(..., description="分页信息")


# 系统状态相关模式

class SystemStatusSchema(BaseModel):
    """系统状态模式"""
    service_name: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
    status: str = Field(..., description="服务状态")
    uptime_seconds: float = Field(..., description="运行时间(秒)")
    
    database_status: str = Field(..., description="数据库状态")
    redis_status: str = Field(..., description="Redis状态")
    
    total_accounts: int = Field(..., description="总账号数")
    active_accounts: int = Field(..., description="活跃账号数")
    total_platforms: int = Field(..., description="总平台数")
    
    sync_queue_size: int = Field(..., description="同步队列大小")
    last_sync_time: Optional[datetime] = Field(None, description="最后同步时间")


# API响应包装模式

class DataResponse(BaseResponse):
    """数据响应模式"""
    data: Any = Field(..., description="响应数据")


class ListResponse(BaseResponse):
    """列表响应模式"""
    data: List[Any] = Field(..., description="响应数据列表")
    pagination: Optional[PaginationResponse] = Field(None, description="分页信息")


class ErrorResponse(BaseResponse):
    """错误响应模式"""
    error_code: str = Field(..., description="错误代码")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")