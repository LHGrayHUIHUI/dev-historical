"""
内容质量控制服务配置管理

该模块提供服务的所有配置选项，包括服务设置、外部服务集成、
检测参数、安全配置等。采用 Pydantic Settings 实现类型安全的配置管理。
"""

from pydantic import Field, validator
from pydantic_settings import BaseSettings
from typing import List, Optional
import logging

class Settings(BaseSettings):
    """
    内容质量控制服务配置类
    
    提供服务运行所需的所有配置选项，支持环境变量注入
    """
    
    # === 服务基础配置 ===
    SERVICE_NAME: str = Field(default="content-quality-control-service", description="服务名称")
    SERVICE_VERSION: str = Field(default="1.0.0", description="服务版本")
    SERVICE_HOST: str = Field(default="0.0.0.0", description="服务监听地址")
    SERVICE_PORT: int = Field(default=8010, description="服务端口")
    SERVICE_ENVIRONMENT: str = Field(default="development", description="运行环境")
    DEBUG: bool = Field(default=False, description="调试模式")
    
    # === API配置 ===
    API_V1_PREFIX: str = Field(default="/api/v1", description="API版本前缀")
    API_TITLE: str = Field(default="内容质量控制服务", description="API文档标题")
    API_DESCRIPTION: str = Field(default="提供内容质量检测和合规性审核功能", description="API描述")
    
    # === 外部服务配置 ===
    STORAGE_SERVICE_URL: str = Field(default="http://localhost:8002", description="存储服务地址")
    STORAGE_SERVICE_TIMEOUT: int = Field(default=30, description="存储服务超时时间(秒)")
    STORAGE_SERVICE_RETRIES: int = Field(default=3, description="存储服务重试次数")
    
    # === Redis配置 ===
    REDIS_URL: str = Field(default="redis://localhost:6379/4", description="Redis连接URL")
    REDIS_KEY_PREFIX: str = Field(default="quality_control", description="Redis键前缀")
    REDIS_DEFAULT_TTL: int = Field(default=3600, description="Redis默认过期时间(秒)")
    
    # === 任务队列配置 ===
    TASK_QUEUE_URL: str = Field(default="redis://localhost:6379/5", description="任务队列Redis URL")
    TASK_QUEUE_NAME: str = Field(default="quality_control_tasks", description="任务队列名称")
    
    # === 安全配置 ===
    SECRET_KEY: str = Field(default="your-secret-key-here", description="密钥")
    JWT_SECRET_KEY: str = Field(default="your-jwt-secret-here", description="JWT密钥")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT算法")
    
    # === 质量检测配置 ===
    MAX_CONTENT_LENGTH: int = Field(default=100000, description="最大内容长度(字符)")
    QUALITY_CHECK_TIMEOUT: int = Field(default=30, description="质量检测超时时间(秒)")
    PARALLEL_DETECTION_ENABLED: bool = Field(default=True, description="启用并行检测")
    AUTO_FIX_ENABLED: bool = Field(default=True, description="启用自动修复")
    
    # === 质量评分配置 ===
    GRAMMAR_WEIGHT: float = Field(default=0.25, description="语法检测权重")
    LOGIC_WEIGHT: float = Field(default=0.20, description="逻辑检测权重")
    FORMAT_WEIGHT: float = Field(default=0.15, description="格式检测权重")
    FACTUAL_WEIGHT: float = Field(default=0.20, description="事实检测权重")
    ACADEMIC_WEIGHT: float = Field(default=0.20, description="学术检测权重")
    
    # === 合规检测配置 ===
    COMPLIANCE_CHECK_TIMEOUT: int = Field(default=15, description="合规检测超时时间(秒)")
    SENSITIVE_WORD_CHECK_ENABLED: bool = Field(default=True, description="启用敏感词检测")
    POLICY_CHECK_ENABLED: bool = Field(default=True, description="启用政策检查")
    COPYRIGHT_CHECK_ENABLED: bool = Field(default=True, description="启用版权检查")
    ACADEMIC_INTEGRITY_CHECK_ENABLED: bool = Field(default=True, description="启用学术诚信检查")
    
    # === 风险评分配置 ===
    AUTO_APPROVAL_THRESHOLD: float = Field(default=90.0, description="自动通过质量阈值")
    AUTO_APPROVAL_RISK_THRESHOLD: int = Field(default=2, description="自动通过风险阈值")
    HUMAN_REVIEW_THRESHOLD: float = Field(default=70.0, description="人工审核质量阈值")
    HUMAN_REVIEW_RISK_THRESHOLD: int = Field(default=5, description="人工审核风险阈值")
    
    # === 工作流配置 ===
    DEFAULT_WORKFLOW_NAME: str = Field(default="standard_review", description="默认工作流名称")
    MAX_REVIEW_STEPS: int = Field(default=3, description="最大审核步骤数")
    TASK_ASSIGNMENT_STRATEGY: str = Field(default="round_robin", description="任务分配策略")
    
    # === NLP模型配置 ===
    SPACY_MODEL: str = Field(default="zh_core_web_sm", description="spaCy中文模型")
    JIEBA_DICT_PATH: Optional[str] = Field(default=None, description="jieba词典路径")
    MODEL_CACHE_DIR: str = Field(default="/tmp/models", description="模型缓存目录")
    
    # === 批量处理配置 ===
    MAX_BATCH_SIZE: int = Field(default=100, description="最大批量处理数量")
    BATCH_PROCESSING_TIMEOUT: int = Field(default=1800, description="批量处理超时时间(秒)")
    CONCURRENT_CHECK_LIMIT: int = Field(default=10, description="并发检测限制")
    
    # === 缓存配置 ===
    ENABLE_RESULT_CACHE: bool = Field(default=True, description="启用结果缓存")
    CACHE_TTL_QUALITY_RESULT: int = Field(default=3600, description="质量检测结果缓存时间(秒)")
    CACHE_TTL_COMPLIANCE_RESULT: int = Field(default=1800, description="合规检测结果缓存时间(秒)")
    
    # === 监控配置 ===
    METRICS_ENABLED: bool = Field(default=True, description="启用指标收集")
    PERFORMANCE_MONITORING: bool = Field(default=True, description="启用性能监控")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="健康检查间隔(秒)")
    
    # === 日志配置 ===
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = Field(default="{time} | {level} | {message}", description="日志格式")
    LOG_FILE_PATH: str = Field(default="logs/quality_control.log", description="日志文件路径")
    LOG_ROTATION: str = Field(default="1 day", description="日志轮转设置")
    LOG_RETENTION: str = Field(default="30 days", description="日志保留时间")
    
    # === 开发配置 ===
    ENABLE_CORS: bool = Field(default=True, description="启用CORS")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="CORS允许的源")
    TEST_MODE: bool = Field(default=False, description="测试模式")
    MOCK_EXTERNAL_SERVICES: bool = Field(default=False, description="模拟外部服务")
    
    @validator("SERVICE_PORT")
    def validate_port(cls, v):
        """验证端口号合法性"""
        if not 1 <= v <= 65535:
            raise ValueError("端口号必须在1-65535范围内")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是: {', '.join(valid_levels)}")
        return v.upper()
    
    @validator("TASK_ASSIGNMENT_STRATEGY")
    def validate_assignment_strategy(cls, v):
        """验证任务分配策略"""
        valid_strategies = ["round_robin", "priority", "workload", "expertise"]
        if v not in valid_strategies:
            raise ValueError(f"任务分配策略必须是: {', '.join(valid_strategies)}")
        return v
    
    @validator("GRAMMAR_WEIGHT", "LOGIC_WEIGHT", "FORMAT_WEIGHT", "FACTUAL_WEIGHT", "ACADEMIC_WEIGHT")
    def validate_weights(cls, v):
        """验证权重值范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("权重值必须在0.0-1.0范围内")
        return v
    
    @property
    def total_weights(self) -> float:
        """计算总权重，应该等于1.0"""
        return (self.GRAMMAR_WEIGHT + self.LOGIC_WEIGHT + 
                self.FORMAT_WEIGHT + self.FACTUAL_WEIGHT + self.ACADEMIC_WEIGHT)
    
    @property
    def service_info(self) -> dict:
        """获取服务信息摘要"""
        return {
            "name": self.SERVICE_NAME,
            "version": self.SERVICE_VERSION,
            "environment": self.SERVICE_ENVIRONMENT,
            "debug": self.DEBUG
        }
    
    @property
    def database_config(self) -> dict:
        """获取数据库配置信息"""
        return {
            "redis_url": self.REDIS_URL,
            "redis_prefix": self.REDIS_KEY_PREFIX,
            "redis_ttl": self.REDIS_DEFAULT_TTL
        }
    
    @property
    def external_services_config(self) -> dict:
        """获取外部服务配置"""
        return {
            "storage_service": {
                "url": self.STORAGE_SERVICE_URL,
                "timeout": self.STORAGE_SERVICE_TIMEOUT,
                "retries": self.STORAGE_SERVICE_RETRIES
            }
        }
    
    @property
    def quality_check_config(self) -> dict:
        """获取质量检测配置"""
        return {
            "max_content_length": self.MAX_CONTENT_LENGTH,
            "timeout": self.QUALITY_CHECK_TIMEOUT,
            "parallel_enabled": self.PARALLEL_DETECTION_ENABLED,
            "auto_fix_enabled": self.AUTO_FIX_ENABLED,
            "weights": {
                "grammar": self.GRAMMAR_WEIGHT,
                "logic": self.LOGIC_WEIGHT,
                "format": self.FORMAT_WEIGHT,
                "factual": self.FACTUAL_WEIGHT,
                "academic": self.ACADEMIC_WEIGHT
            }
        }
    
    @property
    def compliance_config(self) -> dict:
        """获取合规检测配置"""
        return {
            "timeout": self.COMPLIANCE_CHECK_TIMEOUT,
            "sensitive_word_enabled": self.SENSITIVE_WORD_CHECK_ENABLED,
            "policy_check_enabled": self.POLICY_CHECK_ENABLED,
            "copyright_check_enabled": self.COPYRIGHT_CHECK_ENABLED,
            "academic_integrity_enabled": self.ACADEMIC_INTEGRITY_CHECK_ENABLED
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

# 创建全局设置实例
settings = Settings()

# 配置日志
def setup_logging():
    """配置日志设置"""
    import sys
    from loguru import logger
    
    # 清除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # 添加文件处理器
    logger.add(
        settings.LOG_FILE_PATH,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip"
    )
    
    return logger

# 初始化日志
logger = setup_logging()