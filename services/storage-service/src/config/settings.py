"""
应用程序配置设置

基于Pydantic的配置管理，支持环境变量和配置验证
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用程序设置配置类"""
    
    # === 服务基础配置 ===
    service_name: str = Field("data-collection-service", description="服务名称")
    service_version: str = Field("1.0.0", description="服务版本")
    service_host: str = Field("0.0.0.0", description="服务监听地址")
    service_port: int = Field(8002, description="服务端口")
    service_environment: str = Field("development", description="运行环境")
    debug: bool = Field(False, description="调试模式")
    
    # === 安全配置 ===
    secret_key: str = Field(..., description="应用程序密钥")
    jwt_secret_key: str = Field(..., description="JWT密钥")
    jwt_algorithm: str = Field("HS256", description="JWT算法")
    jwt_expire_minutes: int = Field(60, description="JWT过期时间(分钟)")
    
    # === 数据库配置 ===
    database_url: str = Field(..., description="PostgreSQL数据库连接URL")
    mongodb_url: str = Field(..., description="MongoDB连接URL")
    redis_url: str = Field(..., description="Redis连接URL")
    
    # === MinIO对象存储配置 ===
    minio_endpoint: str = Field(..., description="MinIO服务端点")
    minio_access_key: str = Field(..., description="MinIO访问密钥")
    minio_secret_key: str = Field(..., description="MinIO秘密密钥")
    minio_bucket_name: str = Field("historical-texts", description="MinIO存储桶名称")
    minio_secure: bool = Field(True, description="是否使用HTTPS")
    
    # === RabbitMQ消息队列配置 ===
    rabbitmq_url: str = Field(..., description="RabbitMQ连接URL")
    
    # === 文件上传配置 ===
    max_file_size: int = Field(100 * 1024 * 1024, description="最大文件大小(字节)")
    max_batch_size: int = Field(50, description="批量上传最大文件数")
    allowed_file_types: List[str] = Field(
        default=[
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "image/jpeg",
            "image/png",
            "image/tiff"
        ],
        description="允许的文件类型"
    )
    
    # === 病毒扫描配置 ===
    virus_scan_enabled: bool = Field(True, description="是否启用病毒扫描")
    clamav_host: str = Field("localhost", description="ClamAV服务器地址")
    clamav_port: int = Field(3310, description="ClamAV服务器端口")
    
    # === OCR配置 ===
    ocr_enabled: bool = Field(True, description="是否启用OCR")
    tesseract_path: str = Field("/usr/bin/tesseract", description="Tesseract可执行文件路径")
    tesseract_languages: List[str] = Field(
        default=["chi_sim", "chi_tra", "eng"],
        description="Tesseract支持的语言"
    )
    
    # === 处理配置 ===
    max_concurrent_extractions: int = Field(5, description="最大并发提取任务数")
    extraction_timeout: int = Field(300, description="提取超时时间(秒)")
    retry_attempts: int = Field(3, description="重试次数")
    upload_concurrency: int = Field(3, description="并发上传数量限制")
    
    # === 监控配置 ===
    metrics_enabled: bool = Field(True, description="是否启用指标收集")
    prometheus_port: int = Field(8003, description="Prometheus指标端口")
    
    # === 日志配置 ===
    log_level: str = Field("INFO", description="日志级别")
    log_format: str = Field("json", description="日志格式")
    json_logs: bool = Field(True, description="是否使用JSON格式日志")
    
    # === Consul服务发现配置 ===
    consul_enabled: bool = Field(True, description="是否启用Consul")
    consul_host: str = Field("localhost", description="Consul服务器地址")
    consul_port: int = Field(8500, description="Consul服务器端口")
    
    # === 代理配置 ===
    proxy_pool_size: int = Field(200, description="代理池大小")
    proxy_check_interval: int = Field(300, description="代理检查间隔(秒)")
    
    @validator('allowed_file_types', pre=True)
    def parse_allowed_file_types(cls, v):
        """解析允许的文件类型配置"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v
    
    @validator('tesseract_languages', pre=True)
    def parse_tesseract_languages(cls, v):
        """解析Tesseract语言配置"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v
    
    @validator('service_environment')
    def validate_environment(cls, v):
        """验证运行环境"""
        allowed_environments = ['development', 'testing', 'staging', 'production']
        if v not in allowed_environments:
            raise ValueError(f'Environment must be one of {allowed_environments}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'Log level must be one of {allowed_levels}')
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """检查是否为开发环境"""
        return self.service_environment == "development"
    
    @property
    def is_production(self) -> bool:
        """检查是否为生产环境"""
        return self.service_environment == "production"
    
    @property
    def is_testing(self) -> bool:
        """检查是否为测试环境"""
        return self.service_environment == "testing"
    
    @property
    def database_config(self) -> dict:
        """获取数据库配置"""
        return {
            "echo": self.debug,
            "future": True,
            # asyncpg不支持pool_size和max_overflow，使用connect_args
            "connect_args": {
                "server_settings": {
                    "application_name": self.service_name,
                }
            }
        }
    
    @property
    def cors_origins(self) -> List[str]:
        """获取CORS允许的源"""
        if self.is_development:
            return ["*"]
        return [
            "https://your-frontend-domain.com",
            "https://admin.your-domain.com"
        ]
    
    @property
    def trusted_hosts(self) -> List[str]:
        """获取信任的主机"""
        if self.is_development or self.is_testing:
            return ["*"]
        return [
            "your-api-domain.com",
            "localhost",
            "127.0.0.1",
            "storage-service",
            "file-processor",
            "intelligent-classification-service"
        ]
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # 使用环境变量前缀
        env_prefix = ""


@lru_cache()
def get_settings() -> Settings:
    """获取应用程序设置实例
    
    使用lru_cache装饰器确保设置只加载一次
    
    Returns:
        Settings: 配置实例
    """
    return Settings()