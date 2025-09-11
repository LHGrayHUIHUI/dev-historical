"""
自定义异常类定义

定义多平台账号管理服务专用的异常类
提供详细的错误信息和错误码
"""

from typing import Optional, Dict, Any


class AccountManagementError(Exception):
    """账号管理基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
        """
        self.message = message
        self.error_code = error_code or "ACCOUNT_ERROR"
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class PlatformNotSupportedError(AccountManagementError):
    """不支持的平台异常"""
    
    def __init__(self, message: str, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="PLATFORM_NOT_SUPPORTED",
            details={"platform_name": platform_name} if platform_name else {}
        )


class OAuthError(AccountManagementError):
    """OAuth认证异常"""
    
    def __init__(self, message: str, oauth_error: str = None, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="OAUTH_ERROR",
            details={
                "oauth_error": oauth_error,
                "platform_name": platform_name
            }
        )


class TokenExpiredError(AccountManagementError):
    """令牌过期异常"""
    
    def __init__(self, message: str, token_type: str = None, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="TOKEN_EXPIRED",
            details={
                "token_type": token_type,
                "platform_name": platform_name
            }
        )


class InvalidTokenError(AccountManagementError):
    """无效令牌异常"""
    
    def __init__(self, message: str, token_type: str = None, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="INVALID_TOKEN",
            details={
                "token_type": token_type,
                "platform_name": platform_name
            }
        )


class AccountNotFoundError(AccountManagementError):
    """账号不存在异常"""
    
    def __init__(self, message: str, account_id: int = None, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="ACCOUNT_NOT_FOUND",
            details={
                "account_id": account_id,
                "platform_name": platform_name
            }
        )


class AccountExistsError(AccountManagementError):
    """账号已存在异常"""
    
    def __init__(self, message: str, account_name: str = None, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="ACCOUNT_EXISTS",
            details={
                "account_name": account_name,
                "platform_name": platform_name
            }
        )


class PermissionDeniedError(AccountManagementError):
    """权限拒绝异常"""
    
    def __init__(self, message: str, user_id: int = None, permission_type: str = None):
        super().__init__(
            message=message,
            error_code="PERMISSION_DENIED",
            details={
                "user_id": user_id,
                "permission_type": permission_type
            }
        )


class SyncError(AccountManagementError):
    """同步异常"""
    
    def __init__(self, message: str, account_id: int = None, sync_type: str = None, 
                 platform_name: str = None):
        super().__init__(
            message=message,
            error_code="SYNC_ERROR",
            details={
                "account_id": account_id,
                "sync_type": sync_type,
                "platform_name": platform_name
            }
        )


class RateLimitExceededError(AccountManagementError):
    """速率限制超出异常"""
    
    def __init__(self, message: str, platform_name: str = None, retry_after: int = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "platform_name": platform_name,
                "retry_after": retry_after
            }
        )


class EncryptionError(AccountManagementError):
    """加密异常"""
    
    def __init__(self, message: str, operation: str = None):
        super().__init__(
            message=message,
            error_code="ENCRYPTION_ERROR",
            details={"operation": operation} if operation else {}
        )


class DecryptionError(AccountManagementError):
    """解密异常"""
    
    def __init__(self, message: str, operation: str = None):
        super().__init__(
            message=message,
            error_code="DECRYPTION_ERROR",
            details={"operation": operation} if operation else {}
        )


class DatabaseError(AccountManagementError):
    """数据库异常"""
    
    def __init__(self, message: str, operation: str = None, table_name: str = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details={
                "operation": operation,
                "table_name": table_name
            }
        )


class ValidationError(AccountManagementError):
    """数据验证异常"""
    
    def __init__(self, message: str, field_name: str = None, field_value: Any = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field_name": field_name,
                "field_value": str(field_value) if field_value is not None else None
            }
        )


class ConfigurationError(AccountManagementError):
    """配置异常"""
    
    def __init__(self, message: str, config_key: str = None, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={
                "config_key": config_key,
                "platform_name": platform_name
            }
        )


class ServiceUnavailableError(AccountManagementError):
    """服务不可用异常"""
    
    def __init__(self, message: str, service_name: str = None, platform_name: str = None):
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            details={
                "service_name": service_name,
                "platform_name": platform_name
            }
        )


class QuotaExceededError(AccountManagementError):
    """配额超出异常"""
    
    def __init__(self, message: str, quota_type: str = None, limit: int = None, 
                 current: int = None):
        super().__init__(
            message=message,
            error_code="QUOTA_EXCEEDED",
            details={
                "quota_type": quota_type,
                "limit": limit,
                "current": current
            }
        )


class BatchOperationError(AccountManagementError):
    """批量操作异常"""
    
    def __init__(self, message: str, failed_items: list = None, 
                 successful_items: list = None):
        super().__init__(
            message=message,
            error_code="BATCH_OPERATION_ERROR",
            details={
                "failed_items": failed_items or [],
                "successful_items": successful_items or [],
                "failed_count": len(failed_items or []),
                "successful_count": len(successful_items or [])
            }
        )


# HTTP状态码映射
EXCEPTION_STATUS_CODES = {
    AccountManagementError: 500,
    PlatformNotSupportedError: 400,
    OAuthError: 401,
    TokenExpiredError: 401,
    InvalidTokenError: 401,
    AccountNotFoundError: 404,
    AccountExistsError: 409,
    PermissionDeniedError: 403,
    SyncError: 500,
    RateLimitExceededError: 429,
    EncryptionError: 500,
    DecryptionError: 500,
    DatabaseError: 500,
    ValidationError: 400,
    ConfigurationError: 500,
    ServiceUnavailableError: 503,
    QuotaExceededError: 429,
    BatchOperationError: 207  # Multi-Status
}


def get_http_status_code(exception: AccountManagementError) -> int:
    """
    获取异常对应的HTTP状态码
    
    Args:
        exception: 异常实例
        
    Returns:
        int: HTTP状态码
    """
    return EXCEPTION_STATUS_CODES.get(type(exception), 500)


def create_error_response(exception: AccountManagementError) -> Dict[str, Any]:
    """
    创建标准化的错误响应
    
    Args:
        exception: 异常实例
        
    Returns:
        Dict: 错误响应字典
    """
    return {
        "success": False,
        "error": exception.to_dict(),
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "status_code": get_http_status_code(exception)
    }