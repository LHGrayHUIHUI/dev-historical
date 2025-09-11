"""
自定义异常类定义
提供调度服务特定的异常处理
"""
from typing import Optional, Any, Dict


class SchedulingServiceError(Exception):
    """调度服务基础异常类"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(SchedulingServiceError):
    """输入验证错误"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['invalid_value'] = value
        super().__init__(message, details)


class SchedulingError(SchedulingServiceError):
    """调度相关错误"""
    pass


class ConflictError(SchedulingServiceError):
    """冲突检测相关错误"""
    pass


class OptimizationError(SchedulingServiceError):
    """优化相关错误"""
    pass


class PlatformIntegrationError(SchedulingServiceError):
    """平台集成错误"""
    
    def __init__(self, message: str, platform: Optional[str] = None, status_code: Optional[int] = None):
        details = {}
        if platform:
            details['platform'] = platform
        if status_code:
            details['status_code'] = status_code
        super().__init__(message, details)


class ModelError(SchedulingServiceError):
    """机器学习模型相关错误"""
    pass


class DatabaseError(SchedulingServiceError):
    """数据库操作错误"""
    pass


class ConfigurationError(SchedulingServiceError):
    """配置错误"""
    pass