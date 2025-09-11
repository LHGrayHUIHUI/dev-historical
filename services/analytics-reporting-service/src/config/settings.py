"""
分析报告服务配置管理模块

负责管理服务的所有配置参数，包括数据库连接、缓存配置、
机器学习参数、外部服务配置等。使用Pydantic Settings
确保类型安全和环境变量自动加载。
"""

import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """数据库配置设置"""
    
    # PostgreSQL 主数据库
    postgres_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5439/historical_text_analytics",
        description="PostgreSQL 数据库连接URL"
    )
    
    # InfluxDB 时序数据库
    influxdb_url: str = Field(
        default="http://localhost:8086",
        description="InfluxDB 连接URL"
    )
    influxdb_token: str = Field(
        default="analytics-token",
        description="InfluxDB 访问令牌"
    )
    influxdb_org: str = Field(
        default="historical-text",
        description="InfluxDB 组织名"
    )
    influxdb_bucket: str = Field(
        default="analytics-metrics",
        description="InfluxDB 存储桶"
    )
    
    # ClickHouse OLAP数据库
    clickhouse_host: str = Field(
        default="localhost",
        description="ClickHouse 主机地址"
    )
    clickhouse_port: int = Field(
        default=9000,
        description="ClickHouse 端口"
    )
    clickhouse_database: str = Field(
        default="analytics",
        description="ClickHouse 数据库名"
    )
    clickhouse_user: str = Field(
        default="default",
        description="ClickHouse 用户名"
    )
    clickhouse_password: str = Field(
        default="",
        description="ClickHouse 密码"
    )
    
    # Redis 缓存
    redis_url: str = Field(
        default="redis://localhost:6383/6",
        description="Redis 连接URL"
    )
    
    class Config:
        env_prefix = "DB_"


class MLSettings(BaseSettings):
    """机器学习配置设置"""
    
    # 模型配置
    model_cache_dir: str = Field(
        default="./models",
        description="机器学习模型缓存目录"
    )
    
    # 异常检测配置
    anomaly_detection_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="异常检测置信度阈值"
    )
    
    # 预测配置
    forecast_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="预测天数"
    )
    
    # 训练配置
    min_training_samples: int = Field(
        default=100,
        ge=10,
        description="最小训练样本数"
    )
    
    # 特征工程
    feature_window_days: int = Field(
        default=30,
        ge=1,
        le=180,
        description="特征窗口天数"
    )
    
    class Config:
        env_prefix = "ML_"


class ExternalServiceSettings(BaseSettings):
    """外部服务配置"""
    
    # 存储服务
    storage_service_url: str = Field(
        default="http://localhost:8002",
        description="存储服务URL"
    )
    
    # 内容发布服务
    content_publishing_url: str = Field(
        default="http://localhost:8094",
        description="内容发布服务URL"
    )
    
    # 账号管理服务
    account_management_url: str = Field(
        default="http://localhost:8091",
        description="账号管理服务URL"
    )
    
    # 调度服务
    scheduling_service_url: str = Field(
        default="http://localhost:8095",
        description="自动内容调度服务URL"
    )
    
    class Config:
        env_prefix = "SERVICE_"


class CelerySettings(BaseSettings):
    """Celery异步任务配置"""
    
    # 消息代理
    broker_url: str = Field(
        default="redis://localhost:6383/7",
        description="Celery消息代理URL"
    )
    
    # 结果后端
    result_backend: str = Field(
        default="redis://localhost:6383/8",
        description="Celery结果后端URL"
    )
    
    # 任务配置
    task_serializer: str = Field(
        default="json",
        description="任务序列化格式"
    )
    
    result_serializer: str = Field(
        default="json",
        description="结果序列化格式"
    )
    
    accept_content: List[str] = Field(
        default=["json"],
        description="接受的内容类型"
    )
    
    # 工作进程配置
    worker_concurrency: int = Field(
        default=4,
        ge=1,
        description="工作进程并发数"
    )
    
    class Config:
        env_prefix = "CELERY_"


class ReportSettings(BaseSettings):
    """报告生成配置"""
    
    # 报告存储
    report_storage_path: str = Field(
        default="./reports",
        description="报告文件存储路径"
    )
    
    # 报告缓存
    report_cache_hours: int = Field(
        default=24,
        ge=1,
        description="报告缓存时间(小时)"
    )
    
    # 导出配置
    max_export_rows: int = Field(
        default=100000,
        ge=1000,
        description="最大导出行数"
    )
    
    # 可视化配置
    chart_width: int = Field(
        default=800,
        ge=400,
        description="图表宽度"
    )
    
    chart_height: int = Field(
        default=600,
        ge=300,
        description="图表高度"
    )
    
    class Config:
        env_prefix = "REPORT_"


class Settings(BaseSettings):
    """主配置类"""
    
    # 基础设置
    app_name: str = Field(
        default="Analytics Reporting Service",
        description="应用名称"
    )
    
    app_version: str = Field(
        default="1.0.0",
        description="应用版本"
    )
    
    environment: str = Field(
        default="development",
        description="运行环境"
    )
    
    debug: bool = Field(
        default=True,
        description="调试模式"
    )
    
    host: str = Field(
        default="0.0.0.0",
        description="服务监听地址"
    )
    
    port: int = Field(
        default=8099,
        ge=1000,
        le=65535,
        description="服务端口"
    )
    
    # 安全配置
    secret_key: str = Field(
        default="analytics-secret-key-change-in-production",
        min_length=32,
        description="应用密钥"
    )
    
    # API配置
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="API v1路径前缀"
    )
    
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS允许的源"
    )
    
    # 限流配置
    rate_limit_per_minute: int = Field(
        default=1000,
        ge=10,
        description="每分钟请求限制"
    )
    
    # 日志配置
    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    
    log_format: str = Field(
        default="json",
        description="日志格式"
    )
    
    # 子配置
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    external_services: ExternalServiceSettings = Field(default_factory=ExternalServiceSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    
    @validator('environment')
    def validate_environment(cls, v):
        """验证环境配置"""
        valid_environments = ['development', 'testing', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f'Environment must be one of {valid_environments}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment == "production"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()


# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        settings.ml.model_cache_dir,
        settings.report.report_storage_path,
        "logs",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# 应用启动时调用
ensure_directories()