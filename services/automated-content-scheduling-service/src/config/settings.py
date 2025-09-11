"""
自动内容调度服务配置管理
支持开发、测试、生产环境的完整配置
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
import os
from enum import Enum


class Environment(str, Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    url: str = Field(
        default="postgresql://postgres:password@localhost:5433/historical_text_scheduling",
        description="PostgreSQL数据库连接URL"
    )
    pool_size: int = Field(default=10, description="数据库连接池大小")
    max_overflow: int = Field(default=20, description="连接池最大溢出")
    pool_timeout: int = Field(default=30, description="连接池超时时间(秒)")
    pool_recycle: int = Field(default=3600, description="连接回收时间(秒)")
    echo: bool = Field(default=False, description="是否打印SQL语句")


class RedisSettings(BaseSettings):
    """Redis配置"""
    url: str = Field(
        default="redis://localhost:6379/3",
        description="Redis连接URL"
    )
    max_connections: int = Field(default=20, description="最大连接数")
    retry_on_timeout: bool = Field(default=True, description="超时重试")
    socket_timeout: int = Field(default=30, description="Socket超时时间")
    socket_connect_timeout: int = Field(default=30, description="Socket连接超时")


class CelerySettings(BaseSettings):
    """Celery任务队列配置"""
    broker_url: str = Field(
        default="redis://localhost:6379/4", 
        description="消息队列URL"
    )
    result_backend: str = Field(
        default="redis://localhost:6379/5",
        description="结果存储URL"
    )
    task_serializer: str = Field(default="json", description="任务序列化格式")
    result_serializer: str = Field(default="json", description="结果序列化格式")
    accept_content: List[str] = Field(default=["json"], description="接受的内容类型")
    timezone: str = Field(default="UTC", description="时区设置")
    enable_utc: bool = Field(default=True, description="启用UTC时间")
    
    # 任务执行配置
    task_soft_time_limit: int = Field(default=300, description="任务软超时时间(秒)")
    task_time_limit: int = Field(default=600, description="任务硬超时时间(秒)")
    worker_prefetch_multiplier: int = Field(default=4, description="Worker预取倍数")
    task_routes: Dict[str, Dict[str, str]] = Field(
        default={
            "scheduling.tasks.*": {"queue": "scheduling"},
            "publishing.tasks.*": {"queue": "publishing"},
            "optimization.tasks.*": {"queue": "optimization"}
        },
        description="任务路由配置"
    )


class SchedulerSettings(BaseSettings):
    """调度器配置"""
    # APScheduler配置
    job_defaults: Dict[str, Any] = Field(
        default={
            "coalesce": True,
            "max_instances": 3,
            "misfire_grace_time": 300
        },
        description="任务默认配置"
    )
    executors: Dict[str, Dict[str, Any]] = Field(
        default={
            "default": {
                "type": "threadpool",
                "max_workers": 20
            },
            "processpool": {
                "type": "processpool",
                "max_workers": 5
            }
        },
        description="执行器配置"
    )
    
    # 调度策略配置
    max_concurrent_tasks: int = Field(default=100, description="最大并发任务数")
    task_retry_delay: int = Field(default=300, description="任务重试延迟(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    cleanup_interval: int = Field(default=3600, description="清理间隔(秒)")


class MLSettings(BaseSettings):
    """机器学习优化配置"""
    model_type: str = Field(default="RandomForestRegressor", description="ML模型类型")
    model_params: Dict[str, Any] = Field(
        default={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        },
        description="模型参数"
    )
    
    # 特征工程配置
    feature_window_days: int = Field(default=30, description="特征时间窗口(天)")
    min_training_samples: int = Field(default=100, description="最小训练样本数")
    model_retrain_interval: int = Field(default=86400, description="模型重训练间隔(秒)")
    
    # 优化策略
    optimization_weights: Dict[str, float] = Field(
        default={
            "engagement_rate": 0.4,
            "reach": 0.3,
            "conversion_rate": 0.2,
            "timing_score": 0.1
        },
        description="优化权重配置"
    )


class PlatformSettings(BaseSettings):
    """多平台集成配置"""
    supported_platforms: List[str] = Field(
        default=["weibo", "wechat", "douyin", "toutiao", "baijiahao"],
        description="支持的平台列表"
    )
    
    # 平台API配置
    platform_configs: Dict[str, Dict[str, Any]] = Field(
        default={
            "weibo": {
                "api_base_url": "https://api.weibo.com",
                "rate_limit": 1000,
                "batch_size": 20
            },
            "wechat": {
                "api_base_url": "https://api.weixin.qq.com",
                "rate_limit": 500,
                "batch_size": 10
            },
            "douyin": {
                "api_base_url": "https://open.douyin.com",
                "rate_limit": 800,
                "batch_size": 15
            },
            "toutiao": {
                "api_base_url": "https://open.toutiao.com",
                "rate_limit": 600,
                "batch_size": 10
            },
            "baijiahao": {
                "api_base_url": "https://openapi.baidu.com",
                "rate_limit": 400,
                "batch_size": 10
            }
        },
        description="平台特定配置"
    )


class SecuritySettings(BaseSettings):
    """安全配置"""
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="应用密钥"
    )
    access_token_expire_minutes: int = Field(default=30, description="访问令牌过期时间(分钟)")
    refresh_token_expire_days: int = Field(default=7, description="刷新令牌过期时间(天)")
    
    # API安全
    api_rate_limit: int = Field(default=1000, description="API速率限制(每小时)")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="最大请求大小(字节)")
    
    # CORS配置
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="允许的跨域源"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="允许的HTTP方法"
    )


class MonitoringSettings(BaseSettings):
    """监控配置"""
    enable_metrics: bool = Field(default=True, description="启用指标收集")
    metrics_endpoint: str = Field(default="/metrics", description="指标端点")
    
    # 健康检查
    health_check_interval: int = Field(default=60, description="健康检查间隔(秒)")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )
    
    # 告警配置
    alert_thresholds: Dict[str, float] = Field(
        default={
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "task_failure_rate": 5.0,
            "queue_length": 1000
        },
        description="告警阈值"
    )


class Settings(BaseSettings):
    """主配置类"""
    # 基本配置
    app_name: str = Field(default="Automated Content Scheduling Service", description="应用名称")
    version: str = Field(default="1.0.0", description="应用版本")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="运行环境")
    debug: bool = Field(default=True, description="调试模式")
    
    # 服务配置
    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8095, description="服务端口")
    
    # 组件配置
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    platforms: PlatformSettings = Field(default_factory=PlatformSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # 外部服务URL
    account_management_service_url: str = Field(
        default="http://localhost:8091",
        description="账号管理服务URL"
    )
    content_publishing_service_url: str = Field(
        default="http://localhost:8094", 
        description="内容发布服务URL"
    )
    storage_service_url: str = Field(
        default="http://localhost:8002",
        description="存储服务URL"
    )
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """验证环境设置"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("database")
    def validate_database_config(cls, v, values):
        """验证数据库配置"""
        env = values.get("environment", Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION and v.echo:
            v.echo = False  # 生产环境不输出SQL
        return v
    
    @validator("security")
    def validate_security_config(cls, v, values):
        """验证安全配置"""
        env = values.get("environment", Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION and v.secret_key == "your-secret-key-change-in-production":
            raise ValueError("生产环境必须设置安全的secret_key")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # 环境变量前缀
        env_prefix = "SCHEDULING_"
        
        # 嵌套环境变量支持
        env_nested_delimiter = "__"


# 全局设置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


# 环境特定配置加载
def load_environment_config():
    """根据环境加载特定配置"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "testing":
        # 测试环境配置覆盖
        settings.database.url = "postgresql://test_user:test_pass@localhost:5435/test_scheduling"
        settings.redis.url = "redis://localhost:6379/15"
        settings.debug = True
        settings.monitoring.log_level = "DEBUG"
    
    elif env == "production":
        # 生产环境配置覆盖
        settings.debug = False
        settings.monitoring.log_level = "WARNING"
        settings.database.echo = False
        
    return settings


# 在模块加载时执行环境配置
load_environment_config()