"""
内容质量评估服务配置管理

智能化的内容质量评估系统配置，支持多维度质量评估、趋势分析和基准管理。
通过AI驱动的评估引擎为历史文本内容提供科学、准确的质量反馈机制。
"""

from pydantic import BaseSettings, Field, validator
from typing import List, Dict, Optional
import os
from enum import Enum

class Environment(str, Enum):
    """运行环境"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class ServiceSettings(BaseSettings):
    """服务基础配置"""
    name: str = "content-quality-assessment-service"
    version: str = "1.0.0"
    description: str = "内容质量评估系统 - AI驱动的多维度质量评估服务"
    
    # 服务网络配置
    host: str = Field(default="0.0.0.0", env="SERVICE_HOST")
    port: int = Field(default=8012, env="SERVICE_PORT")
    
    # 运行环境配置
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # 日志配置
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="{time} | {level} | {message}", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class ExternalServicesSettings(BaseSettings):
    """外部服务配置"""
    
    # Storage Service 配置
    storage_service_url: str = Field(default="http://localhost:8002", env="STORAGE_SERVICE_URL")
    storage_service_timeout: int = Field(default=30, env="STORAGE_SERVICE_TIMEOUT")
    storage_service_retries: int = Field(default=3, env="STORAGE_SERVICE_RETRIES")
    
    # AI Model Service 配置
    ai_model_service_url: str = Field(default="http://localhost:8008", env="AI_MODEL_SERVICE_URL") 
    ai_model_service_timeout: int = Field(default=60, env="AI_MODEL_SERVICE_TIMEOUT")
    ai_model_service_retries: int = Field(default=2, env="AI_MODEL_SERVICE_RETRIES")
    
    # Content Quality Control Service 配置 (复用评估能力)
    quality_control_service_url: str = Field(default="http://localhost:8010", env="QUALITY_CONTROL_SERVICE_URL")
    quality_control_service_timeout: int = Field(default=30, env="QUALITY_CONTROL_SERVICE_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class DatabaseSettings(BaseSettings):
    """数据库配置"""
    
    # Redis配置 (缓存评估结果)
    redis_url: str = Field(default="redis://localhost:6379/6", env="REDIS_URL")
    redis_key_prefix: str = Field(default="quality_assessment", env="REDIS_KEY_PREFIX")
    redis_default_ttl: int = Field(default=3600, env="REDIS_DEFAULT_TTL")  # 1小时
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class AssessmentEngineSettings(BaseSettings):
    """评估引擎配置"""
    
    # 核心评估维度配置
    enabled_dimensions: List[str] = Field(
        default=["readability", "accuracy", "completeness", "coherence", "relevance"],
        env="ENABLED_DIMENSIONS"
    )
    
    # 评估性能配置
    max_content_length: int = Field(default=50000, env="MAX_CONTENT_LENGTH")  # 最大内容长度
    assessment_timeout: int = Field(default=30, env="ASSESSMENT_TIMEOUT")  # 评估超时
    max_concurrent_assessments: int = Field(default=10, env="MAX_CONCURRENT_ASSESSMENTS")
    
    # 缓存配置
    cache_assessment_results: bool = Field(default=True, env="CACHE_ASSESSMENT_RESULTS")
    cache_ttl_hours: int = Field(default=24, env="CACHE_TTL_HOURS")
    
    # 批量处理配置
    max_batch_size: int = Field(default=50, env="MAX_BATCH_SIZE")
    batch_processing_timeout: int = Field(default=600, env="BATCH_PROCESSING_TIMEOUT")  # 10分钟
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class QualityMetricsSettings(BaseSettings):
    """质量指标配置"""
    
    # 维度权重配置 (默认权重)
    default_dimension_weights: Dict[str, float] = Field(default={
        "readability": 0.20,      # 可读性权重
        "accuracy": 0.25,         # 准确性权重  
        "completeness": 0.20,     # 完整性权重
        "coherence": 0.20,        # 连贯性权重
        "relevance": 0.15         # 相关性权重
    })
    
    # 评分阈值配置
    grade_thresholds: Dict[str, float] = Field(default={
        "A": 90.0,  # 优秀
        "B": 80.0,  # 良好
        "C": 70.0,  # 中等
        "D": 60.0,  # 及格
        "F": 0.0    # 不及格
    })
    
    # 置信度配置
    min_confidence_threshold: float = Field(default=0.6, env="MIN_CONFIDENCE_THRESHOLD")
    default_confidence: float = Field(default=0.8, env="DEFAULT_CONFIDENCE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class TrendAnalysisSettings(BaseSettings):
    """趋势分析配置"""
    
    # 趋势分析参数
    min_data_points: int = Field(default=5, env="MIN_DATA_POINTS")  # 最少数据点
    analysis_window_days: int = Field(default=30, env="ANALYSIS_WINDOW_DAYS")  # 分析窗口
    trend_significance_threshold: float = Field(default=0.1, env="TREND_SIGNIFICANCE_THRESHOLD")
    
    # 预测配置
    enable_trend_prediction: bool = Field(default=True, env="ENABLE_TREND_PREDICTION")
    prediction_confidence_level: float = Field(default=0.95, env="PREDICTION_CONFIDENCE_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class BenchmarkSettings(BaseSettings):
    """基准管理配置"""
    
    # 默认基准配置
    default_benchmarks: Dict[str, Dict[str, float]] = Field(default={
        "historical_document": {
            "readability": 75.0,
            "accuracy": 85.0, 
            "completeness": 80.0,
            "coherence": 80.0,
            "relevance": 85.0,
            "overall": 80.0
        },
        "academic_paper": {
            "readability": 70.0,
            "accuracy": 90.0,
            "completeness": 85.0, 
            "coherence": 85.0,
            "relevance": 90.0,
            "overall": 85.0
        },
        "educational_content": {
            "readability": 85.0,
            "accuracy": 85.0,
            "completeness": 80.0,
            "coherence": 85.0,
            "relevance": 80.0,
            "overall": 82.0
        }
    })
    
    # 基准管理配置
    max_custom_benchmarks: int = Field(default=100, env="MAX_CUSTOM_BENCHMARKS")
    benchmark_validation_enabled: bool = Field(default=True, env="BENCHMARK_VALIDATION_ENABLED")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class NLPModelSettings(BaseSettings):
    """NLP模型配置"""
    
    # spaCy模型配置
    spacy_model: str = Field(default="zh_core_web_sm", env="SPACY_MODEL")
    spacy_model_cache_dir: str = Field(default="/app/models", env="MODEL_CACHE_DIR")
    
    # 情感分析模型
    sentiment_model: str = Field(default="uer/roberta-base-finetuned-chinanews-chinese", env="SENTIMENT_MODEL")
    
    # 文本统计模型
    enable_textstat: bool = Field(default=True, env="ENABLE_TEXTSTAT")
    
    # 并行处理配置
    max_workers: int = Field(default=4, env="NLP_MAX_WORKERS")
    model_loading_timeout: int = Field(default=60, env="MODEL_LOADING_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class PerformanceSettings(BaseSettings):
    """性能配置"""
    
    # 并发控制
    max_concurrent_assessments: int = Field(default=20, env="MAX_CONCURRENT_ASSESSMENTS")
    assessment_queue_size: int = Field(default=100, env="ASSESSMENT_QUEUE_SIZE")
    
    # 内存管理
    max_memory_usage_mb: int = Field(default=2048, env="MAX_MEMORY_USAGE_MB")
    garbage_collection_interval: int = Field(default=300, env="GC_INTERVAL")  # 5分钟
    
    # 缓存优化
    enable_result_caching: bool = Field(default=True, env="ENABLE_RESULT_CACHING")
    cache_compression: bool = Field(default=True, env="CACHE_COMPRESSION")
    
    # 监控配置
    enable_performance_monitoring: bool = Field(default=True, env="PERFORMANCE_MONITORING")
    metrics_collection_interval: int = Field(default=60, env="METRICS_INTERVAL")  # 1分钟
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class SecuritySettings(BaseSettings):
    """安全配置"""
    
    # API安全
    secret_key: str = Field(default="content-quality-assessment-secret-key", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="content-quality-assessment-jwt-secret", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # 访问控制
    enable_authentication: bool = Field(default=False, env="ENABLE_AUTHENTICATION")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    
    # 数据脱敏
    enable_data_masking: bool = Field(default=True, env="ENABLE_DATA_MASKING")
    log_sensitive_data: bool = Field(default=False, env="LOG_SENSITIVE_DATA")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class APISettings(BaseSettings):
    """API配置"""
    
    # API版本和路径
    api_version: str = Field(default="v1", env="API_VERSION")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # CORS配置
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # API文档配置
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    docs_url: str = Field(default="/docs", env="DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="REDOC_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class Settings(BaseSettings):
    """主配置类"""
    
    # 子配置模块
    service: ServiceSettings = ServiceSettings()
    external_services: ExternalServicesSettings = ExternalServicesSettings()
    database: DatabaseSettings = DatabaseSettings()
    assessment_engine: AssessmentEngineSettings = AssessmentEngineSettings()
    quality_metrics: QualityMetricsSettings = QualityMetricsSettings()
    trend_analysis: TrendAnalysisSettings = TrendAnalysisSettings()
    benchmark: BenchmarkSettings = BenchmarkSettings()
    nlp_models: NLPModelSettings = NLPModelSettings()
    performance: PerformanceSettings = PerformanceSettings()
    security: SecuritySettings = SecuritySettings()
    api: APISettings = APISettings()
    
    @validator('service')
    def validate_service_config(cls, v):
        """验证服务配置"""
        if v.port < 1024 or v.port > 65535:
            raise ValueError("端口号必须在1024-65535范围内")
        return v
    
    @validator('assessment_engine')
    def validate_assessment_config(cls, v):
        """验证评估引擎配置"""
        if v.max_content_length <= 0:
            raise ValueError("最大内容长度必须大于0")
        if v.assessment_timeout <= 0:
            raise ValueError("评估超时时间必须大于0")
        return v
    
    @validator('quality_metrics')
    def validate_quality_metrics(cls, v):
        """验证质量指标配置"""
        # 验证权重总和
        total_weight = sum(v.default_dimension_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"维度权重总和必须等于1.0，当前为{total_weight}")
        
        # 验证评分阈值
        thresholds = list(v.grade_thresholds.values())
        if not all(thresholds[i] >= thresholds[i+1] for i in range(len(thresholds)-1)):
            raise ValueError("评分阈值必须按降序排列")
        
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
# 全局设置实例
settings = Settings()

def get_settings() -> Settings:
    """获取配置实例"""
    return settings