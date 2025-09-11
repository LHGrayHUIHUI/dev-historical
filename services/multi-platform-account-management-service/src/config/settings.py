"""
多平台账号管理服务配置管理

统一管理数据库连接、Redis缓存、OAuth配置、加密密钥等配置
使用Pydantic Settings实现类型安全的配置加载
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, validator, Field
import os


class Settings(BaseSettings):
    """
    应用程序配置类
    
    支持从环境变量和.env文件加载配置
    提供配置验证和默认值设置
    """
    
    # 应用基础配置
    app_name: str = "Multi-Platform Account Management Service"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8091
    
    # 数据库配置
    database_url: str = "postgresql://postgres:password@localhost:5433/historical_text_accounts"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Redis配置
    redis_url: str = "redis://localhost:6380/13"
    redis_max_connections: int = 100
    redis_socket_timeout: int = 5
    
    # Celery配置
    celery_broker_url: str = "redis://localhost:6380/14"
    celery_result_backend: str = "redis://localhost:6380/15"
    celery_task_timeout: int = 1800  # 30分钟
    celery_max_retries: int = 3
    
    # JWT配置
    jwt_secret_key: str = "account-management-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    
    # 加密配置
    encryption_key: str = "account-encryption-key-32-chars-long!"  # 32字节
    password_salt: str = "account-password-salt"
    
    # OAuth配置
    oauth_callback_base_url: str = "http://localhost:8091/api/v1/oauth/callback"
    oauth_state_expire_seconds: int = 600  # 10分钟
    
    # API限流配置
    rate_limit_per_minute: int = 1000
    rate_limit_per_hour: int = 10000
    
    # 监控配置
    enable_prometheus: bool = True
    prometheus_port: int = 9192
    log_level: str = "INFO"
    
    # 安全配置
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    allowed_hosts: List[str] = ["*"]
    
    # 平台配置
    supported_platforms: List[str] = ["weibo", "wechat", "douyin", "toutiao", "baijiahao"]
    
    # 同步配置
    account_sync_interval_hours: int = 6  # 6小时同步一次
    max_concurrent_syncs: int = 10
    sync_timeout_seconds: int = 300  # 5分钟
    
    # 外部服务配置
    storage_service_url: str = "http://localhost:8002"
    text_publishing_service_url: str = "http://localhost:8089"
    
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
    
    @validator("encryption_key")
    def validate_encryption_key(cls, v):
        """验证加密密钥长度"""
        if len(v.encode()) != 32:
            raise ValueError("加密密钥必须是32字节长度")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class PlatformConfig:
    """
    平台配置管理类
    
    定义各个社交媒体平台的配置信息
    """
    
    # 微博平台配置
    WEIBO_CONFIG = {
        "name": "weibo",
        "display_name": "新浪微博",
        "platform_type": "social_media",
        "api_base_url": "https://api.weibo.com",
        "oauth_config": {
            "authorize_url": "https://api.weibo.com/oauth2/authorize",
            "token_url": "https://api.weibo.com/oauth2/access_token",
            "scopes": ["email", "statuses_to_me_read", "friendships_groups_read"],
            "response_type": "code"
        },
        "rate_limits": {
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "user_timeline_per_hour": 300,
            "post_status_per_hour": 100
        },
        "features": {
            "supports_scheduling": True,
            "max_text_length": 140,
            "supports_images": True,
            "max_images": 9,
            "supports_video": True,
            "max_video_size_mb": 500
        }
    }
    
    # 微信公众号平台配置
    WECHAT_CONFIG = {
        "name": "wechat",
        "display_name": "微信公众号",
        "platform_type": "social_media",
        "api_base_url": "https://api.weixin.qq.com",
        "oauth_config": {
            "authorize_url": "https://open.weixin.qq.com/connect/oauth2/authorize",
            "token_url": "https://api.weixin.qq.com/sns/oauth2/access_token",
            "scopes": ["snsapi_userinfo"],
            "response_type": "code"
        },
        "rate_limits": {
            "requests_per_hour": 2000,
            "requests_per_day": 20000,
            "message_send_per_day": 1000
        },
        "features": {
            "supports_scheduling": True,
            "max_text_length": 2000,
            "supports_images": True,
            "max_images": 8,
            "supports_video": True,
            "max_video_size_mb": 200
        }
    }
    
    # 抖音平台配置
    DOUYIN_CONFIG = {
        "name": "douyin",
        "display_name": "抖音",
        "platform_type": "short_video",
        "api_base_url": "https://open.douyin.com",
        "oauth_config": {
            "authorize_url": "https://open.douyin.com/platform/oauth/connect",
            "token_url": "https://open.douyin.com/oauth/access_token",
            "scopes": ["user_info", "video.list", "video.create"],
            "response_type": "code"
        },
        "rate_limits": {
            "requests_per_hour": 500,
            "requests_per_day": 5000,
            "video_upload_per_day": 50
        },
        "features": {
            "supports_scheduling": False,
            "max_text_length": 55,
            "supports_images": False,
            "supports_video": True,
            "max_video_size_mb": 1024,
            "max_video_duration_seconds": 60
        }
    }
    
    # 今日头条平台配置
    TOUTIAO_CONFIG = {
        "name": "toutiao",
        "display_name": "今日头条",
        "platform_type": "news",
        "api_base_url": "https://mp.toutiao.com/api",
        "oauth_config": {
            "authorize_url": "https://mp.toutiao.com/auth/page/login",
            "token_url": "https://mp.toutiao.com/auth/token",
            "scopes": ["article.publish", "article.manage"],
            "response_type": "code"
        },
        "rate_limits": {
            "requests_per_hour": 800,
            "requests_per_day": 8000,
            "article_publish_per_day": 20
        },
        "features": {
            "supports_scheduling": True,
            "max_text_length": 50000,
            "supports_images": True,
            "max_images": 20,
            "supports_video": True,
            "max_video_size_mb": 2048
        }
    }
    
    # 百家号平台配置
    BAIJIAHAO_CONFIG = {
        "name": "baijiahao",
        "display_name": "百家号",
        "platform_type": "content",
        "api_base_url": "https://baijiahao.baidu.com/builderinner/api",
        "oauth_config": {
            "authorize_url": "https://openapi.baidu.com/oauth/2.0/authorize",
            "token_url": "https://openapi.baidu.com/oauth/2.0/token",
            "scopes": ["public"],
            "response_type": "code"
        },
        "rate_limits": {
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "article_publish_per_day": 30
        },
        "features": {
            "supports_scheduling": True,
            "max_text_length": 100000,
            "supports_images": True,
            "max_images": 30,
            "supports_video": True,
            "max_video_size_mb": 1024
        }
    }
    
    @classmethod
    def get_platform_config(cls, platform_name: str) -> Dict[str, Any]:
        """
        获取指定平台的配置
        
        Args:
            platform_name: 平台名称
            
        Returns:
            Dict[str, Any]: 平台配置
        """
        config_map = {
            "weibo": cls.WEIBO_CONFIG,
            "wechat": cls.WECHAT_CONFIG,
            "douyin": cls.DOUYIN_CONFIG,
            "toutiao": cls.TOUTIAO_CONFIG,
            "baijiahao": cls.BAIJIAHAO_CONFIG
        }
        
        return config_map.get(platform_name.lower(), {})
    
    @classmethod
    def get_all_platforms(cls) -> List[Dict[str, Any]]:
        """
        获取所有支持的平台配置
        
        Returns:
            List[Dict[str, Any]]: 所有平台配置列表
        """
        return [
            cls.WEIBO_CONFIG,
            cls.WECHAT_CONFIG,
            cls.DOUYIN_CONFIG,
            cls.TOUTIAO_CONFIG,
            cls.BAIJIAHAO_CONFIG
        ]
    
    @classmethod
    def get_oauth_config(cls, platform_name: str) -> Dict[str, Any]:
        """
        获取平台OAuth配置
        
        Args:
            platform_name: 平台名称
            
        Returns:
            Dict[str, Any]: OAuth配置
        """
        platform_config = cls.get_platform_config(platform_name)
        return platform_config.get("oauth_config", {})


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
            "filename": "logs/account_management.log",
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