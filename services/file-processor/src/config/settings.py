"""
文件处理服务配置管理
纯文件处理服务配置，无数据库依赖，专注文件处理算法
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


class FileProcessingSettings(BaseSettings):
    """文件处理配置"""
    # 批量处理设置
    max_batch_size: int = Field(default=50, description="最大批量处理数量")
    max_file_size_mb: int = Field(default=100, description="最大文件大小(MB)")
    
    # 文件格式支持
    supported_file_types: List[str] = Field(
        default=["pdf", "docx", "doc", "html", "txt", "jpg", "jpeg", "png", "gif"],
        description="支持的文件类型"
    )
    
    # 处理超时配置
    processing_timeout_seconds: int = Field(default=300, description="文件处理超时时间(秒)")
    ocr_timeout_seconds: int = Field(default=120, description="OCR处理超时时间(秒)")
    
    # 并发处理配置
    max_concurrent_jobs: int = Field(default=10, description="最大并发处理任务数")
    
    # 文件验证
    enable_virus_scan: bool = Field(default=True, description="是否启用病毒扫描")
    enable_format_validation: bool = Field(default=True, description="是否启用文件格式验证")
    
    # OCR配置
    ocr_language: str = Field(default="chi_sim+eng", description="OCR识别语言")
    ocr_confidence_threshold: float = Field(default=0.7, description="OCR置信度阈值")
    
    class Config:
        env_prefix = "FILE_PROCESSING_"


class ServiceSettings(BaseSettings):
    """服务基础配置"""
    # 服务信息
    service_name: str = Field(default="file-processor-service", description="服务名称")
    service_version: str = Field(default="1.0.0", description="服务版本")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="运行环境")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8001, description="服务端口")
    workers: int = Field(default=1, description="工作进程数")
    
    # API配置
    api_prefix: str = Field(default="/api/v1", description="API前缀")
    docs_url: str = Field(default="/docs", description="API文档URL")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI JSON URL")
    
    # CORS配置
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://localhost:8002"],
        description="允许的CORS源"
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="允许的HTTP方法"
    )
    
    # 安全配置
    secret_key: str = Field(default="file-processor-secret-key-change-in-production", description="密钥")
    
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
    metrics_port: int = Field(default=8101, description="指标端口")
    
    # 健康检查
    health_check_interval: int = Field(default=30, description="健康检查间隔(秒)")
    
    # 服务注册
    consul_host: str = Field(default="localhost", description="Consul主机")
    consul_port: int = Field(default=8500, description="Consul端口")
    enable_service_registry: bool = Field(default=True, description="是否启用服务注册")
    
    class Config:
        env_prefix = "MONITOR_"


class ExternalServicesSettings(BaseSettings):
    """外部服务配置"""
    # storage-service配置 (调用方)
    storage_service_url: str = Field(
        default="http://localhost:8002", 
        description="Storage Service URL"
    )
    storage_service_timeout: int = Field(default=30, description="Storage Service超时时间(秒)")
    
    # OCR服务配置 (如果使用外部OCR)
    external_ocr_url: Optional[str] = Field(default=None, description="外部OCR服务URL")
    external_ocr_timeout: int = Field(default=60, description="外部OCR超时时间(秒)")
    
    class Config:
        env_prefix = "EXTERNAL_"


class Settings(BaseSettings):
    """主配置类 - 聚合所有配置 
    
    file-processor是纯文件处理服务：
    - 无数据库依赖 (MongoDB, PostgreSQL, Redis)  
    - 无状态设计
    - 专注文件处理算法
    - 通过API与storage-service协作
    """
    
    # 子配置
    file_processing: FileProcessingSettings = Field(default_factory=FileProcessingSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    external_services: ExternalServicesSettings = Field(default_factory=ExternalServicesSettings)
    
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
            if v.secret_key == "file-processor-secret-key-change-in-production":
                raise ValueError("生产环境必须设置安全的密钥")
        return v
    
    @validator('file_processing')
    def validate_file_processing_settings(cls, v):
        """验证文件处理配置"""
        if v.max_batch_size <= 0:
            raise ValueError("最大批量处理数量必须大于0")
        
        if v.max_file_size_mb <= 0:
            raise ValueError("最大文件大小必须大于0")
        
        if v.processing_timeout_seconds <= 0:
            raise ValueError("处理超时时间必须大于0")
        
        if not (0.0 <= v.ocr_confidence_threshold <= 1.0):
            raise ValueError("OCR置信度阈值必须在0到1之间")
        
        return v
    
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.service.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.service.environment == Environment.PRODUCTION
    
    def get_max_file_size_bytes(self) -> int:
        """获取最大文件大小(字节)"""
        return self.file_processing.max_file_size_mb * 1024 * 1024
    
    def is_file_type_supported(self, file_extension: str) -> bool:
        """检查文件类型是否支持"""
        return file_extension.lower() in [
            ext.lower() for ext in self.file_processing.supported_file_types
        ]


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


# 便捷访问
def get_storage_service_url() -> str:
    """获取Storage Service URL"""
    return settings.external_services.storage_service_url


def is_debug_mode() -> bool:
    """是否为调试模式"""
    return settings.logging.log_level == LogLevel.DEBUG


def get_processing_limits() -> Dict[str, Any]:
    """获取处理限制配置"""
    return {
        "max_file_size_bytes": settings.get_max_file_size_bytes(),
        "max_batch_size": settings.file_processing.max_batch_size,
        "processing_timeout": settings.file_processing.processing_timeout_seconds,
        "max_concurrent_jobs": settings.file_processing.max_concurrent_jobs,
        "supported_file_types": settings.file_processing.supported_file_types
    }