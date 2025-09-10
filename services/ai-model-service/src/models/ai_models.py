"""
AI模型相关数据模型
"""

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ModelProvider(str, Enum):
    """AI模型提供商枚举"""
    OPENAI = "openai"
    CLAUDE = "claude"
    BAIDU = "baidu"
    ALIBABA = "alibaba"
    TENCENT = "tencent"
    ZHIPU = "zhipu"
    GEMINI = "gemini"
    LOCAL = "local"


class ModelType(str, Enum):
    """模型类型枚举"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class AccountStatus(str, Enum):
    """账号状态枚举"""
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"


class ModelConfig(BaseModel):
    """AI模型配置数据类"""
    
    id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    provider: ModelProvider = Field(..., description="提供商")
    model_id: str = Field(..., description="提供商的模型标识")
    api_endpoint: str = Field(..., description="API端点URL")
    model_type: ModelType = Field(ModelType.CHAT, description="模型类型")
    max_tokens: int = Field(4096, description="最大token数")
    context_window: int = Field(4096, description="上下文窗口大小")
    cost_per_1k_tokens: float = Field(0.0, description="每1K token成本")
    priority: int = Field(1, description="优先级，数值越大优先级越高")
    is_active: bool = Field(True, description="是否活跃")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="模型能力")
    config: Dict[str, Any] = Field(default_factory=dict, description="模型特定配置")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    
    @validator('priority')
    def validate_priority(cls, v):
        """验证优先级"""
        if v < 1 or v > 10:
            raise ValueError("优先级必须在1-10之间")
        return v
    
    @validator('cost_per_1k_tokens')
    def validate_cost(cls, v):
        """验证成本"""
        if v < 0:
            raise ValueError("成本不能为负数")
        return v
    
    class Config:
        """Pydantic配置"""
        use_enum_values = True


class APIAccount(BaseModel):
    """API账号配置数据类"""
    
    id: str = Field(..., description="账号ID")
    provider: ModelProvider = Field(..., description="提供商")
    account_name: str = Field(..., description="账号名称")
    api_key_encrypted: str = Field(..., description="加密的API密钥")
    api_secret_encrypted: Optional[str] = Field(None, description="加密的API密钥")
    organization_id: Optional[str] = Field(None, description="组织ID")
    endpoint_url: Optional[str] = Field(None, description="自定义端点URL")
    quota_limit: int = Field(0, description="配额限制")
    quota_used: int = Field(0, description="已使用配额")
    quota_reset_date: Optional[datetime] = Field(None, description="配额重置日期")
    status: AccountStatus = Field(AccountStatus.ACTIVE, description="账号状态")
    last_used_at: Optional[datetime] = Field(None, description="最后使用时间")
    error_count: int = Field(0, description="错误计数")
    last_error: Optional[str] = Field(None, description="最后错误信息")
    health_score: float = Field(1.0, description="健康评分(0-1)")
    tags: Dict[str, Any] = Field(default_factory=dict, description="标签")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    
    @validator('health_score')
    def validate_health_score(cls, v):
        """验证健康评分"""
        if v < 0 or v > 1:
            raise ValueError("健康评分必须在0-1之间")
        return v
    
    @validator('quota_used')
    def validate_quota_used(cls, v, values):
        """验证配额使用量"""
        if v < 0:
            raise ValueError("配额使用量不能为负数")
        if 'quota_limit' in values and values['quota_limit'] > 0 and v > values['quota_limit']:
            # 允许超出配额，但记录警告
            pass
        return v
    
    class Config:
        """Pydantic配置"""
        use_enum_values = True


class ModelAccountMapping(BaseModel):
    """模型账号关联关系"""
    
    id: str = Field(..., description="关联ID")
    model_id: str = Field(..., description="模型ID")
    account_id: str = Field(..., description="账号ID")
    priority: int = Field(1, description="优先级")
    is_active: bool = Field(True, description="是否活跃")
    created_at: Optional[datetime] = Field(None, description="创建时间")


class RoutingStrategy(BaseModel):
    """路由策略配置"""
    
    id: str = Field(..., description="策略ID")
    name: str = Field(..., description="策略名称")
    description: Optional[str] = Field(None, description="策略描述")
    strategy_type: str = Field(..., description="策略类型")
    config: Dict[str, Any] = Field(..., description="策略配置")
    is_active: bool = Field(True, description="是否活跃")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    
    @validator('strategy_type')
    def validate_strategy_type(cls, v):
        """验证策略类型"""
        allowed_types = ['round_robin', 'weighted', 'priority', 'cost_based', 'health_based']
        if v not in allowed_types:
            raise ValueError(f"策略类型必须是以下之一: {allowed_types}")
        return v


class UsageStatistic(BaseModel):
    """使用统计数据"""
    
    id: str = Field(..., description="统计ID")
    account_id: str = Field(..., description="账号ID")
    model_id: str = Field(..., description="模型ID")
    request_count: int = Field(0, description="请求数量")
    token_count: int = Field(0, description="token数量")
    success_count: int = Field(0, description="成功数量")
    error_count: int = Field(0, description="错误数量")
    total_cost: float = Field(0.0, description="总成本")
    avg_response_time: float = Field(0.0, description="平均响应时间")
    date: datetime = Field(..., description="统计日期")
    hour: Optional[int] = Field(None, description="小时(0-23)")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    
    @validator('hour')
    def validate_hour(cls, v):
        """验证小时"""
        if v is not None and (v < 0 or v > 23):
            raise ValueError("小时必须在0-23之间")
        return v
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
    
    @property
    def error_rate(self) -> float:
        """计算错误率"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


class ModelHealth(BaseModel):
    """模型健康状态"""
    
    model_id: str = Field(..., description="模型ID")
    provider: ModelProvider = Field(..., description="提供商")
    is_available: bool = Field(True, description="是否可用")
    health_score: float = Field(1.0, description="健康评分")
    response_time_avg: float = Field(0.0, description="平均响应时间")
    success_rate: float = Field(1.0, description="成功率")
    error_rate: float = Field(0.0, description="错误率")
    available_accounts: int = Field(0, description="可用账号数")
    last_check: Optional[datetime] = Field(None, description="最后检查时间")
    issues: list[str] = Field(default_factory=list, description="问题列表")
    
    @validator('health_score', 'success_rate', 'error_rate')
    def validate_rates(cls, v):
        """验证比率"""
        if v < 0 or v > 1:
            raise ValueError("比率必须在0-1之间")
        return v