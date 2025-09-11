"""
文本发布服务配置管理

统一管理数据库连接、Redis缓存、Celery任务队列等配置
使用Pydantic Settings实现类型安全的配置加载
"""

from typing import List, Optional
from pydantic import BaseSettings, validator
import os


class Settings(BaseSettings):
    """
    应用程序配置类
    
    支持从环境变量和.env文件加载配置
    提供配置验证和默认值设置
    """
    
    # 应用基础配置
    app_name: str = "Text Publishing Service"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    
    # 数据库配置
    database_url: str = "postgresql://postgres:password@localhost:5433/historical_text_publishing"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis配置
    redis_url: str = "redis://localhost:6380/0"
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5
    
    # Celery配置
    celery_broker_url: str = "redis://localhost:6380/1"
    celery_result_backend: str = "redis://localhost:6380/2"
    celery_task_timeout: int = 3600  # 1小时
    celery_max_retries: int = 3
    
    # JWT配置
    jwt_secret_key: str = "your-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    
    # 平台API配置
    weibo_api_key: Optional[str] = None
    weibo_api_secret: Optional[str] = None
    wechat_app_id: Optional[str] = None
    wechat_app_secret: Optional[str] = None
    douyin_client_key: Optional[str] = None
    douyin_client_secret: Optional[str] = None
    
    # 文件上传配置
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = ["image/jpeg", "image/png", "image/gif", "video/mp4", "video/avi"]
    upload_path: str = "/tmp/uploads"
    
    # 发布限制配置
    max_content_length: int = 10000
    max_title_length: int = 500
    max_platforms_per_task: int = 10
    max_tasks_per_user: int = 100
    
    # 监控配置
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    log_level: str = "INFO"
    
    # 安全配置
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    rate_limit_per_minute: int = 60
    
    @validator("database_url")
    def validate_database_url(cls, v):
        """验证数据库连接URL格式"""
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("数据库URL必须以postgresql://或postgres://开头")
        return v
    
    @validator("redis_url")
    def validate_redis_url(cls, v):
        """验证Redis连接URL格式"""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL必须以redis://开头")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是以下之一: {valid_levels}")
        return v.upper()
    
    @validator("allowed_file_types")
    def validate_file_types(cls, v):
        """验证允许的文件类型格式"""
        for file_type in v:
            if "/" not in file_type:
                raise ValueError(f"文件类型格式错误: {file_type}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class PlatformConfig:
    """
    平台配置类
    
    定义各个发布平台的基础配置和限制
    """
    
    PLATFORM_CONFIGS = {
        "weibo": {
            "display_name": "微博",
            "max_content_length": 140,
            "max_images": 9,
            "max_video_size": 100 * 1024 * 1024,  # 100MB
            "rate_limit_per_hour": 100,
            "supported_formats": ["jpg", "jpeg", "png", "gif", "mp4"],
            "api_base_url": "https://api.weibo.com/2",
            "auth_type": "oauth2"
        },
        "wechat": {
            "display_name": "微信公众号",
            "max_content_length": 20000,
            "max_images": 10,
            "max_video_size": 50 * 1024 * 1024,  # 50MB
            "rate_limit_per_hour": 50,
            "supported_formats": ["jpg", "jpeg", "png", "mp4"],
            "api_base_url": "https://api.weixin.qq.com",
            "auth_type": "app_secret"
        },
        "douyin": {
            "display_name": "抖音",
            "max_content_length": 2200,
            "max_images": 12,
            "max_video_size": 200 * 1024 * 1024,  # 200MB
            "rate_limit_per_hour": 200,
            "supported_formats": ["jpg", "jpeg", "png", "mp4", "mov"],
            "api_base_url": "https://open-api.douyin.com",
            "auth_type": "oauth2"
        },
        "toutiao": {
            "display_name": "今日头条",
            "max_content_length": 5000,
            "max_images": 20,
            "max_video_size": 150 * 1024 * 1024,  # 150MB
            "rate_limit_per_hour": 150,
            "supported_formats": ["jpg", "jpeg", "png", "gif", "mp4"],
            "api_base_url": "https://mp.toutiao.com/api",
            "auth_type": "oauth2"
        },
        "baijiahao": {
            "display_name": "百家号",
            "max_content_length": 8000,
            "max_images": 15,
            "max_video_size": 120 * 1024 * 1024,  # 120MB
            "rate_limit_per_hour": 80,
            "supported_formats": ["jpg", "jpeg", "png", "mp4"],
            "api_base_url": "https://baijiahao.baidu.com/builder",
            "auth_type": "oauth2"
        }
    }
    
    @classmethod
    def get_platform_config(cls, platform_name: str) -> dict:
        """
        获取指定平台的配置信息
        
        Args:
            platform_name: 平台名称
            
        Returns:
            dict: 平台配置信息
        """
        return cls.PLATFORM_CONFIGS.get(platform_name, {})
    
    @classmethod
    def get_all_platforms(cls) -> List[str]:
        """
        获取所有支持的平台列表
        
        Returns:
            List[str]: 平台名称列表
        """
        return list(cls.PLATFORM_CONFIGS.keys())
    
    @classmethod
    def is_supported_platform(cls, platform_name: str) -> bool:
        """
        检查平台是否受支持
        
        Args:
            platform_name: 平台名称
            
        Returns:
            bool: 是否支持该平台
        """
        return platform_name in cls.PLATFORM_CONFIGS


# 全局配置实例
settings = Settings()

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "json",
            "class": "logging.FileHandler",
            "filename": "logs/publishing_service.log",
        },
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["default", "file"],
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "sqlalchemy": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        }
    }
}