"""
内容合并服务配置管理

该模块管理多内容合并生成服务的所有配置，包括服务设置、
外部服务连接、合并算法参数、AI模型配置等。
"""

from pydantic import BaseSettings, Field, validator
from typing import Optional, Dict, List
import os
from pathlib import Path

class ServiceSettings(BaseSettings):
    """服务基础配置"""
    
    # 基础服务配置
    service_name: str = Field(default="content-merger-service", description="服务名称")
    service_version: str = Field(default="1.0.0", description="服务版本")
    service_host: str = Field(default="0.0.0.0", description="服务主机")
    service_port: int = Field(default=8011, description="服务端口")
    service_environment: str = Field(default="development", description="运行环境")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(default="{time} | {level} | {message}", description="日志格式")
    
    # 开发配置
    debug: bool = Field(default=False, description="调试模式")
    enable_cors: bool = Field(default=True, description="启用CORS")
    test_mode: bool = Field(default=False, description="测试模式")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class ExternalServiceSettings(BaseSettings):
    """外部服务配置"""
    
    # Storage Service配置
    storage_service_url: str = Field(default="http://localhost:8002", description="存储服务URL")
    storage_service_timeout: int = Field(default=60, description="存储服务超时时间(秒)")
    storage_service_retries: int = Field(default=3, description="存储服务重试次数")
    
    # AI Model Service配置
    ai_model_service_url: str = Field(default="http://localhost:8008", description="AI模型服务URL")
    ai_model_service_timeout: int = Field(default=120, description="AI模型服务超时时间(秒)")
    ai_model_service_retries: int = Field(default=2, description="AI模型服务重试次数")
    
    # Redis配置
    redis_url: str = Field(default="redis://localhost:6379/5", description="Redis连接URL")
    redis_key_prefix: str = Field(default="content_merger", description="Redis键前缀")
    redis_default_ttl: int = Field(default=7200, description="Redis默认TTL(秒)")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class MergerAlgorithmSettings(BaseSettings):
    """合并算法配置"""
    
    # 内容处理限制
    max_content_length: int = Field(default=200000, description="最大内容长度")
    max_source_contents: int = Field(default=20, description="最大源内容数量")
    max_batch_size: int = Field(default=50, description="最大批处理大小")
    
    # 合并处理配置
    merge_timeout: int = Field(default=300, description="合并超时时间(秒)")
    analysis_timeout: int = Field(default=60, description="分析超时时间(秒)")
    relationship_timeout: int = Field(default=90, description="关系分析超时时间(秒)")
    
    # 质量控制阈值
    min_quality_threshold: float = Field(default=70.0, description="最低质量阈值")
    consistency_threshold: float = Field(default=80.0, description="一致性阈值")
    completeness_threshold: float = Field(default=75.0, description="完整性阈值")
    
    # 相似度计算配置
    similarity_algorithm: str = Field(default="cosine", description="相似度算法")
    similarity_threshold: float = Field(default=0.3, description="相似度阈值")
    topic_similarity_weight: float = Field(default=0.4, description="主题相似度权重")
    entity_similarity_weight: float = Field(default=0.3, description="实体相似度权重")
    content_similarity_weight: float = Field(default=0.3, description="内容相似度权重")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class AIModelSettings(BaseSettings):
    """AI模型配置"""
    
    # 默认模型配置
    default_model: str = Field(default="gpt-4", description="默认AI模型")
    fallback_model: str = Field(default="gpt-3.5-turbo", description="备用AI模型")
    
    # 生成参数
    default_temperature: float = Field(default=0.3, description="默认温度参数")
    default_max_tokens: int = Field(default=4000, description="默认最大令牌数")
    creativity_temperature: float = Field(default=0.7, description="创意模式温度")
    conservative_temperature: float = Field(default=0.1, description="保守模式温度")
    
    # 提示语配置
    system_prompt_prefix: str = Field(
        default="你是一个专业的历史文献编辑和内容合并专家",
        description="系统提示语前缀"
    )
    
    # NLP模型配置
    spacy_model: str = Field(default="zh_core_web_sm", description="spaCy中文模型")
    sentence_transformer_model: str = Field(
        default="distiluse-base-multilingual-cased",
        description="句子向量模型"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class MergeStrategySettings(BaseSettings):
    """合并策略配置"""
    
    # 时间线合并配置
    timeline_date_extraction: bool = Field(default=True, description="启用时间提取")
    timeline_chronological_order: bool = Field(default=True, description="按时间顺序排序")
    timeline_period_grouping: bool = Field(default=True, description="按时期分组")
    
    # 主题合并配置
    topic_clustering_enabled: bool = Field(default=True, description="启用主题聚类")
    topic_min_cluster_size: int = Field(default=2, description="主题聚类最小大小")
    topic_similarity_threshold: float = Field(default=0.6, description="主题相似度阈值")
    
    # 层次合并配置
    hierarchy_importance_scoring: bool = Field(default=True, description="启用重要性评分")
    hierarchy_depth_limit: int = Field(default=4, description="层次深度限制")
    
    # 逻辑关系配置
    logic_causality_detection: bool = Field(default=True, description="启用因果关系检测")
    logic_sequence_analysis: bool = Field(default=True, description="启用序列分析")
    
    # 补充扩展配置
    supplement_gap_detection: bool = Field(default=True, description="启用内容缺口检测")
    supplement_auto_fill: bool = Field(default=False, description="启用自动填充")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class PerformanceSettings(BaseSettings):
    """性能配置"""
    
    # 并发控制
    max_concurrent_merges: int = Field(default=10, description="最大并发合并数")
    max_concurrent_analyses: int = Field(default=20, description="最大并发分析数")
    
    # 缓存配置
    enable_analysis_cache: bool = Field(default=True, description="启用分析结果缓存")
    enable_similarity_cache: bool = Field(default=True, description="启用相似度缓存")
    cache_expiry_hours: int = Field(default=24, description="缓存过期时间(小时)")
    
    # 资源限制
    max_memory_per_task_mb: int = Field(default=1024, description="每任务最大内存(MB)")
    cleanup_interval_minutes: int = Field(default=30, description="清理间隔(分钟)")
    
    # 监控配置
    metrics_enabled: bool = Field(default=True, description="启用性能指标")
    performance_monitoring: bool = Field(default=True, description="启用性能监控")
    detailed_logging: bool = Field(default=False, description="启用详细日志")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class SecuritySettings(BaseSettings):
    """安全配置"""
    
    # JWT配置
    secret_key: str = Field(description="密钥")
    jwt_secret_key: str = Field(description="JWT密钥")
    jwt_algorithm: str = Field(default="HS256", description="JWT算法")
    jwt_access_token_expire_minutes: int = Field(default=60, description="访问令牌过期时间")
    
    # API安全
    rate_limit_per_minute: int = Field(default=100, description="每分钟请求限制")
    max_request_size_mb: int = Field(default=50, description="最大请求大小(MB)")
    
    @validator('secret_key', 'jwt_secret_key')
    def validate_secrets(cls, v):
        if not v:
            raise ValueError('安全密钥不能为空')
        if len(v) < 32:
            raise ValueError('安全密钥长度至少32字符')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class Settings:
    """统一配置管理器"""
    
    def __init__(self):
        self.service = ServiceSettings()
        self.external_services = ExternalServiceSettings()
        self.merger_algorithm = MergerAlgorithmSettings()
        self.ai_model = AIModelSettings()
        self.merge_strategy = MergeStrategySettings()
        self.performance = PerformanceSettings()
        self.security = SecuritySettings()
    
    @property
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return self.service.service_environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """判断是否为开发环境"""
        return self.service.service_environment.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """判断是否为测试环境"""
        return self.service.service_environment.lower() == "testing"

# 全局配置实例
settings = Settings()

# 导出常用配置
__all__ = [
    'Settings',
    'ServiceSettings', 
    'ExternalServiceSettings',
    'MergerAlgorithmSettings',
    'AIModelSettings',
    'MergeStrategySettings',
    'PerformanceSettings',
    'SecuritySettings',
    'settings'
]