"""
数据模型模块初始化

导出所有数据库模型和Pydantic模式
"""

from .database import Base, get_database_url, create_engine_and_session, get_database
from .account_models import (
    Platform,
    Account,
    AccountCredential,
    AccountPermission,
    AccountSyncLog,
    ApiUsageStats
)
from .schemas import (
    # 基础响应模式
    BaseResponse,
    DataResponse,
    ListResponse,
    ErrorResponse,
    PaginationResponse,
    
    # 平台相关模式
    PlatformConfigSchema,
    PlatformCreateSchema,
    
    # 账号相关模式
    AccountSchema,
    AccountCreateSchema,
    AccountUpdateSchema,
    AccountStatsSchema,
    AccountSearchSchema,
    AccountListResponseSchema,
    
    # 同步相关模式
    AccountSyncRequestSchema,
    AccountSyncLogSchema,
    BatchSyncRequestSchema,
    
    # OAuth相关模式
    OAuthUrlSchema,
    OAuthCallbackSchema,
    TokenRefreshRequestSchema,
    
    # 权限相关模式
    AccountPermissionSchema,
    GrantPermissionSchema,
    
    # 统计相关模式
    ApiUsageStatsSchema,
    ApiUsageReportSchema,
    
    # 系统状态模式
    SystemStatusSchema
)

__all__ = [
    # 数据库相关
    "Base",
    "get_database_url", 
    "create_engine_and_session",
    "get_database",
    
    # SQLAlchemy 模型
    "Platform",
    "Account",
    "AccountCredential",
    "AccountPermission",
    "AccountSyncLog",
    "ApiUsageStats",
    
    # Pydantic 模式 - 基础
    "BaseResponse",
    "DataResponse",
    "ListResponse",
    "ErrorResponse",
    "PaginationResponse",
    
    # Pydantic 模式 - 平台
    "PlatformConfigSchema",
    "PlatformCreateSchema",
    
    # Pydantic 模式 - 账号
    "AccountSchema",
    "AccountCreateSchema",
    "AccountUpdateSchema",
    "AccountStatsSchema",
    "AccountSearchSchema",
    "AccountListResponseSchema",
    
    # Pydantic 模式 - 同步
    "AccountSyncRequestSchema",
    "AccountSyncLogSchema",
    "BatchSyncRequestSchema",
    
    # Pydantic 模式 - OAuth
    "OAuthUrlSchema",
    "OAuthCallbackSchema",
    "TokenRefreshRequestSchema",
    
    # Pydantic 模式 - 权限
    "AccountPermissionSchema",
    "GrantPermissionSchema",
    
    # Pydantic 模式 - 统计
    "ApiUsageStatsSchema",
    "ApiUsageReportSchema",
    
    # Pydantic 模式 - 系统
    "SystemStatusSchema"
]