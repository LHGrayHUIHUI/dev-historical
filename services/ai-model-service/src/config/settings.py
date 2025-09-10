"""
AI大模型服务配置管理

基于Pydantic的配置管理，支持环境变量和配置验证
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """AI大模型服务配置类"""
    
    # === 服务基础配置 ===
    service_name: str = Field("ai-model-service", description="服务名称")
    service_version: str = Field("1.0.0", description="服务版本")
    service_host: str = Field("0.0.0.0", description="服务监听地址")
    service_port: int = Field(8007, description="服务端口")
    service_environment: str = Field("development", description="运行环境")
    debug: bool = Field(False, description="调试模式")
    
    # === 安全配置 ===
    secret_key: str = Field(..., description="应用程序密钥")
    jwt_secret_key: str = Field(..., description="JWT密钥")
    jwt_algorithm: str = Field("HS256", description="JWT算法")
    jwt_expire_minutes: int = Field(60, description="JWT过期时间(分钟)")
    encryption_key: str = Field(..., description="API密钥加密密钥")
    
    # === Storage Service 配置 ===
    storage_service_url: str = Field("http://localhost:8002", description="存储服务URL")
    storage_service_timeout: int = Field(30, description="存储服务超时时间(秒)")
    
    # === Redis缓存配置 ===
    redis_url: str = Field("redis://localhost:6380/1", description="Redis连接URL")
    cache_ttl: int = Field(3600, description="缓存过期时间(秒)")
    cache_prefix: str = Field("ai_model:", description="缓存键前缀")
    
    # === AI平台配置 ===
    # OpenAI配置
    openai_api_key: Optional[str] = Field(None, description="OpenAI API密钥")
    openai_organization: Optional[str] = Field(None, description="OpenAI组织ID")
    openai_base_url: str = Field("https://api.openai.com/v1", description="OpenAI API基础URL")
    
    # Claude配置  
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API密钥")
    anthropic_base_url: str = Field("https://api.anthropic.com", description="Anthropic API基础URL")
    
    # 百度文心一言配置
    baidu_api_key: Optional[str] = Field(None, description="百度API密钥")
    baidu_secret_key: Optional[str] = Field(None, description="百度Secret密钥")
    baidu_base_url: str = Field("https://aip.baidubce.com", description="百度API基础URL")
    
    # 阿里通义千问配置
    alibaba_api_key: Optional[str] = Field(None, description="阿里云API密钥")
    alibaba_base_url: str = Field("https://dashscope.aliyuncs.com", description="阿里云API基础URL")
    
    # 腾讯混元配置
    tencent_secret_id: Optional[str] = Field(None, description="腾讯云SecretId")
    tencent_secret_key: Optional[str] = Field(None, description="腾讯云SecretKey")
    tencent_region: str = Field("ap-beijing", description="腾讯云地域")
    
    # 智谱ChatGLM配置
    zhipu_api_key: Optional[str] = Field(None, description="智谱API密钥")
    zhipu_base_url: str = Field("https://open.bigmodel.cn/api/paas/v3", description="智谱API基础URL")
    
    # === 请求配置 ===
    max_concurrent_requests: int = Field(100, description="最大并发请求数")
    request_timeout: int = Field(30, description="请求超时时间(秒)")
    max_retries: int = Field(3, description="最大重试次数")
    retry_delay: float = Field(1.0, description="重试延迟(秒)")
    
    # === 缓存配置 ===
    enable_cache: bool = Field(True, description="是否启用缓存")
    cache_ttl_default: int = Field(3600, description="默认缓存TTL(秒)")
    cache_ttl_models: int = Field(86400, description="模型列表缓存TTL(秒)")
    cache_ttl_accounts: int = Field(3600, description="账号信息缓存TTL(秒)")
    
    # === 路由策略配置 ===
    default_routing_strategy: str = Field("priority", description="默认路由策略")
    enable_load_balancing: bool = Field(True, description="是否启用负载均衡")
    health_check_interval: int = Field(300, description="健康检查间隔(秒)")
    
    # === 成本控制配置 ===
    enable_cost_tracking: bool = Field(True, description="是否启用成本跟踪")
    cost_alert_threshold: float = Field(100.0, description="成本告警阈值(美元)")
    quota_alert_threshold: float = Field(0.8, description="配额告警阈值(比例)")
    
    # === 监控配置 ===
    metrics_enabled: bool = Field(True, description="是否启用指标收集")
    prometheus_port: int = Field(8008, description="Prometheus指标端口")
    
    # === 日志配置 ===
    log_level: str = Field("INFO", description="日志级别")
    log_format: str = Field("json", description="日志格式")
    json_logs: bool = Field(True, description="是否使用JSON格式日志")
    
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
    
    @validator('default_routing_strategy')
    def validate_routing_strategy(cls, v):
        """验证路由策略"""
        allowed_strategies = ['priority', 'round_robin', 'weighted', 'cost_based']
        if v not in allowed_strategies:
            raise ValueError(f'Routing strategy must be one of {allowed_strategies}')
        return v
    
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
            "ai-model-service",
            "storage-service"
        ]
    
    @property
    def available_providers(self) -> Dict[str, bool]:
        """获取可用的AI提供商"""
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "baidu": bool(self.baidu_api_key and self.baidu_secret_key),
            "alibaba": bool(self.alibaba_api_key),
            "tencent": bool(self.tencent_secret_id and self.tencent_secret_key),
            "zhipu": bool(self.zhipu_api_key)
        }
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # 使用环境变量前缀
        env_prefix = "AI_MODEL_"


@lru_cache()
def get_settings() -> Settings:
    """获取应用程序设置实例
    
    使用lru_cache装饰器确保设置只加载一次
    
    Returns:
        Settings: 配置实例
    """
    return Settings()