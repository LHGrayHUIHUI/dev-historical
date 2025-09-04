"""
服务配置管理
使用Pydantic Settings进行配置管理，支持环境变量和配置文件
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Dict, Any, Optional
import os
from enum import Enum


class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB连接URL")
    mongodb_db_name: str = Field(default="historical_text_data", description="MongoDB数据库名")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis连接URL")
    
    # 连接池配置
    mongodb_max_pool_size: int = Field(default=50, description="MongoDB最大连接池大小")
    mongodb_min_pool_size: int = Field(default=10, description="MongoDB最小连接池大小")
    redis_max_connections: int = Field(default=20, description="Redis最大连接数")
    
    class Config:
        env_prefix = "DB_"


class ContentSettings(BaseSettings):
    """内容管理配置"""
    # 批量处理设置
    max_batch_size: int = Field(default=100, description="最大批量处理数量")
    max_file_size_mb: int = Field(default=50, description="最大文件大小(MB)")
    
    # 内容验证
    min_content_length: int = Field(default=10, description="最小内容长度")
    max_content_length: int = Field(default=1000000, description="最大内容长度")
    
    # 文件支持
    supported_file_types: List[str] = Field(
        default=["json", "csv"],
        description="支持的文件类型"
    )
    
    # 自动处理
    auto_generate_summary: bool = Field(default=True, description="是否自动生成摘要")
    auto_extract_keywords: bool = Field(default=True, description="是否自动提取关键词")
    auto_deduplicate: bool = Field(default=True, description="是否自动去重")
    
    class Config:
        env_prefix = "CONTENT_"


class ServiceSettings(BaseSettings):
    """服务基础配置"""
    # 服务信息
    service_name: str = Field(default="data-source-service", description="服务名称")
    service_version: str = Field(default="1.0.0", description="服务版本")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="运行环境")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8000, description="服务端口")
    workers: int = Field(default=1, description="工作进程数")
    
    # API配置
    api_prefix: str = Field(default="/api/v1", description="API前缀")
    docs_url: str = Field(default="/docs", description="API文档URL")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI JSON URL")
    
    # CORS配置
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="允许的CORS源"
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="允许的HTTP方法"
    )
    
    # 安全配置
    secret_key: str = Field(default="your-secret-key-change-in-production", description="密钥")
    jwt_algorithm: str = Field(default="HS256", description="JWT算法")
    jwt_expire_minutes: int = Field(default=30, description="JWT过期时间(分钟)")
    
    # 限流配置
    rate_limit_requests: int = Field(default=100, description="速率限制-每分钟请求数")
    rate_limit_window: int = Field(default=60, description="速率限制-时间窗口(秒)")
    
    class Config:
        env_prefix = "SERVICE_"


class LoggingSettings(BaseSettings):
    """日志配置"""
    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="日志格式"
    )
    log_file: Optional[str] = Field(default=None, description="日志文件路径")
    log_rotation: str = Field(default="1 day", description="日志轮转")
    log_retention: str = Field(default="30 days", description="日志保留时间")
    
    # 结构化日志
    json_logs: bool = Field(default=False, description="是否使用JSON格式日志")
    
    class Config:
        env_prefix = "LOG_"


class MonitoringSettings(BaseSettings):
    """监控配置"""
    # Prometheus配置
    enable_metrics: bool = Field(default=True, description="是否启用指标收集")
    metrics_port: int = Field(default=8001, description="指标端口")
    
    # 健康检查
    health_check_interval: int = Field(default=30, description="健康检查间隔(秒)")
    
    # 服务注册
    consul_host: str = Field(default="localhost", description="Consul主机")
    consul_port: int = Field(default=8500, description="Consul端口")
    enable_service_registry: bool = Field(default=True, description="是否启用服务注册")
    
    class Config:
        env_prefix = "MONITOR_"


class Settings(BaseSettings):
    """主配置类 - 聚合所有配置"""
    
    # 子配置
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    content: ContentSettings = Field(default_factory=ContentSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    model_config = {
        "extra": "ignore",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_nested_delimiter": "__"
    }
    
    @validator('service')
    def validate_environment(cls, v):
        """验证环境配置"""
        if v.environment == Environment.PRODUCTION:
            if v.secret_key == "your-secret-key-change-in-production":
                raise ValueError("生产环境必须设置安全的密钥")
        return v
    
    @validator('content')
    def validate_content_settings(cls, v):
        """验证内容配置"""
        if v.min_content_length >= v.max_content_length:
            raise ValueError("最小内容长度必须小于最大内容长度")
        
        if v.max_batch_size <= 0:
            raise ValueError("最大批量处理数量必须大于0")
        
        return v
    
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.service.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.service.environment == Environment.PRODUCTION
    


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


# 便捷访问
def get_database_url() -> str:
    """获取数据库连接URL"""
    return settings.database.mongodb_url


def get_redis_url() -> str:
    """获取Redis连接URL"""
    return settings.database.redis_url


def is_debug_mode() -> bool:
    """是否为调试模式"""
    return settings.logging.log_level == LogLevel.DEBUG