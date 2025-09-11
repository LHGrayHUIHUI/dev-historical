"""
智能文本优化服务配置 - Settings Configuration

提供服务的配置管理，包括数据库连接、外部服务配置、
AI模型设置、性能参数等配置项

核心功能:
1. 基础服务配置 (端口、环境等)
2. Storage Service集成配置
3. AI模型服务集成配置
4. Redis缓存配置
5. 任务队列配置
6. 性能和限制配置
"""

from typing import Optional, List
from pydantic import BaseSettings, Field, validator
import os


class Settings(BaseSettings):
    """
    智能文本优化服务设置
    基于环境变量和默认值的配置管理
    """
    
    # === 服务基础配置 ===
    service_name: str = Field(default="intelligent-text-optimization-service", env="SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="SERVICE_VERSION")
    service_host: str = Field(default="0.0.0.0", env="SERVICE_HOST")
    service_port: int = Field(default=8009, env="SERVICE_PORT")
    service_environment: str = Field(default="development", env="SERVICE_ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # === 日志配置 ===
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="{time} | {level} | {message}", env="LOG_FORMAT")
    
    # === Storage Service配置 ===
    storage_service_url: str = Field(
        default="http://localhost:8002", 
        env="STORAGE_SERVICE_URL",
        description="Storage Service的基础URL"
    )
    storage_service_timeout: int = Field(
        default=30,
        env="STORAGE_SERVICE_TIMEOUT", 
        description="Storage Service请求超时(秒)"
    )
    storage_service_retries: int = Field(
        default=3,
        env="STORAGE_SERVICE_RETRIES",
        description="Storage Service请求重试次数"
    )
    
    # === AI模型服务配置 ===
    ai_model_service_url: str = Field(
        default="http://localhost:8008",
        env="AI_MODEL_SERVICE_URL",
        description="AI模型服务的基础URL"
    )
    ai_model_service_timeout: int = Field(
        default=60,
        env="AI_MODEL_SERVICE_TIMEOUT",
        description="AI模型服务请求超时(秒)"
    )
    
    # === Redis缓存配置 ===
    redis_url: str = Field(
        default="redis://localhost:6379/2", 
        env="REDIS_URL",
        description="Redis连接URL"
    )
    redis_key_prefix: str = Field(
        default="text_optimization:",
        env="REDIS_KEY_PREFIX",
        description="Redis键前缀"
    )
    redis_default_ttl: int = Field(
        default=3600,
        env="REDIS_DEFAULT_TTL",
        description="Redis默认过期时间(秒)"
    )
    
    # === 任务队列配置 ===
    task_queue_url: str = Field(
        default="redis://localhost:6379/3",
        env="TASK_QUEUE_URL",
        description="任务队列连接URL"
    )
    task_queue_name: str = Field(
        default="text_optimization_tasks",
        env="TASK_QUEUE_NAME",
        description="任务队列名称"
    )
    
    # === 安全配置 ===
    secret_key: str = Field(
        default="intelligent-text-optimization-secret-key",
        env="SECRET_KEY",
        description="服务密钥"
    )
    jwt_secret_key: str = Field(
        default="intelligent-text-optimization-jwt-secret",
        env="JWT_SECRET_KEY",
        description="JWT密钥"
    )
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # === 文本优化配置 ===
    # 优化性能限制
    max_content_length: int = Field(
        default=100000,
        env="MAX_CONTENT_LENGTH", 
        description="最大文本长度(字符)"
    )
    max_batch_size: int = Field(
        default=100,
        env="MAX_BATCH_SIZE",
        description="最大批量处理数量"
    )
    optimization_timeout: int = Field(
        default=180,
        env="OPTIMIZATION_TIMEOUT",
        description="单个优化任务超时(秒)"
    )
    batch_processing_timeout: int = Field(
        default=3600,
        env="BATCH_PROCESSING_TIMEOUT",
        description="批量处理超时(秒)"
    )
    
    # 质量评估配置
    quality_assessment_enabled: bool = Field(
        default=True,
        env="QUALITY_ASSESSMENT_ENABLED",
        description="是否启用质量评估"
    )
    min_quality_score: float = Field(
        default=70.0,
        env="MIN_QUALITY_SCORE",
        description="最低质量分数阈值"
    )
    default_quality_threshold: float = Field(
        default=80.0,
        env="DEFAULT_QUALITY_THRESHOLD",
        description="默认质量阈值"
    )
    
    # 优化策略配置
    default_optimization_mode: str = Field(
        default="historical_format",
        env="DEFAULT_OPTIMIZATION_MODE",
        description="默认优化模式"
    )
    max_versions_per_task: int = Field(
        default=5,
        env="MAX_VERSIONS_PER_TASK",
        description="每个任务最大版本数"
    )
    enable_parallel_optimization: bool = Field(
        default=True,
        env="ENABLE_PARALLEL_OPTIMIZATION",
        description="是否启用并行优化"
    )
    
    # === NLP模型配置 ===
    # 中文分词模型
    jieba_dict_path: Optional[str] = Field(
        default=None,
        env="JIEBA_DICT_PATH",
        description="jieba自定义词典路径"
    )
    
    # spaCy模型
    spacy_model: str = Field(
        default="zh_core_web_sm",
        env="SPACY_MODEL", 
        description="spaCy中文模型"
    )
    
    # 缓存配置
    model_cache_dir: str = Field(
        default="/tmp/models",
        env="MODEL_CACHE_DIR",
        description="模型缓存目录"
    )
    
    # === 监控和性能配置 ===
    metrics_enabled: bool = Field(
        default=True,
        env="METRICS_ENABLED",
        description="是否启用指标监控"
    )
    performance_monitoring: bool = Field(
        default=True,
        env="PERFORMANCE_MONITORING",
        description="是否启用性能监控"
    )
    
    # API限制配置
    rate_limit_per_minute: int = Field(
        default=60,
        env="RATE_LIMIT_PER_MINUTE",
        description="每分钟API调用限制"
    )
    concurrent_optimization_limit: int = Field(
        default=10,
        env="CONCURRENT_OPTIMIZATION_LIMIT",
        description="并发优化任务限制"
    )
    
    # === 开发和测试配置 ===
    enable_cors: bool = Field(
        default=True,
        env="ENABLE_CORS",
        description="是否启用CORS"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS",
        description="允许的CORS来源"
    )
    
    # 测试配置
    test_mode: bool = Field(
        default=False,
        env="TEST_MODE",
        description="测试模式"
    )
    mock_ai_responses: bool = Field(
        default=False,
        env="MOCK_AI_RESPONSES", 
        description="是否模拟AI响应"
    )
    
    # === 配置验证器 ===
    @validator("service_port")
    def validate_service_port(cls, v):
        """验证服务端口范围"""
        if not (1024 <= v <= 65535):
            raise ValueError("服务端口必须在1024-65535范围内")
        return v
    
    @validator("storage_service_timeout", "ai_model_service_timeout", "optimization_timeout")
    def validate_timeout(cls, v):
        """验证超时时间"""
        if v <= 0:
            raise ValueError("超时时间必须大于0")
        return v
    
    @validator("max_content_length")
    def validate_content_length(cls, v):
        """验证内容长度限制"""
        if v <= 0:
            raise ValueError("内容长度限制必须大于0")
        return v
    
    @validator("min_quality_score", "default_quality_threshold")
    def validate_quality_score(cls, v):
        """验证质量分数范围"""
        if not (0 <= v <= 100):
            raise ValueError("质量分数必须在0-100范围内")
        return v
    
    @validator("max_batch_size")
    def validate_batch_size(cls, v):
        """验证批量处理大小"""
        if v <= 0:
            raise ValueError("批量处理大小必须大于0")
        return v
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # 配置字段别名
        fields = {
            "service_port": {"env": ["SERVICE_PORT", "API_PORT"]},
            "debug": {"env": ["DEBUG", "DEV_MODE"]},
        }


# 全局设置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取全局设置实例 (单例模式)
    
    Returns:
        Settings: 配置实例
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    重新加载设置配置
    主要用于测试环境或配置热重载
    
    Returns:
        Settings: 新的配置实例
    """
    global _settings
    _settings = None
    return get_settings()


def get_service_info() -> dict:
    """
    获取服务基础信息
    
    Returns:
        dict: 包含服务名称、版本、环境等信息的字典
    """
    settings = get_settings()
    return {
        "service_name": settings.service_name,
        "service_version": settings.service_version,
        "environment": settings.service_environment,
        "debug_mode": settings.debug,
        "host": settings.service_host,
        "port": settings.service_port,
    }


def is_production() -> bool:
    """
    判断是否为生产环境
    
    Returns:
        bool: 是否为生产环境
    """
    settings = get_settings()
    return settings.service_environment.lower() in ["production", "prod"]


def is_development() -> bool:
    """
    判断是否为开发环境
    
    Returns:
        bool: 是否为开发环境
    """
    settings = get_settings()
    return settings.service_environment.lower() in ["development", "dev", "local"]


def is_testing() -> bool:
    """
    判断是否为测试环境
    
    Returns:
        bool: 是否为测试环境
    """
    settings = get_settings()
    return settings.service_environment.lower() in ["testing", "test"] or settings.test_mode