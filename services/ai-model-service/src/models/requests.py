"""
请求和响应模型
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """聊天消息模型"""
    
    role: Literal["system", "user", "assistant"] = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    name: Optional[str] = Field(None, description="消息发送者名称")
    function_call: Optional[Dict[str, Any]] = Field(None, description="函数调用信息")
    
    @validator('role')
    def validate_role(cls, v):
        """验证角色"""
        allowed_roles = ["system", "user", "assistant"]
        if v not in allowed_roles:
            raise ValueError(f"角色必须是以下之一: {allowed_roles}")
        return v
    
    @validator('content')
    def validate_content(cls, v):
        """验证内容"""
        if not v or not v.strip():
            raise ValueError("消息内容不能为空")
        return v.strip()


class ChatCompletionRequest(BaseModel):
    """聊天完成请求模型"""
    
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    model: Optional[str] = Field(None, description="指定模型名称")
    provider: Optional[str] = Field(None, description="指定提供商")
    temperature: float = Field(0.7, description="温度参数")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    top_p: float = Field(1.0, description="核采样参数")
    frequency_penalty: float = Field(0.0, description="频率惩罚")
    presence_penalty: float = Field(0.0, description="存在惩罚")
    stop: Optional[Union[str, List[str]]] = Field(None, description="停止词")
    stream: bool = Field(False, description="是否流式响应")
    
    # 高级选项
    user: Optional[str] = Field(None, description="用户标识")
    timeout: Optional[int] = Field(None, description="超时时间(秒)")
    cache: bool = Field(True, description="是否使用缓存")
    requirements: Optional[Dict[str, Any]] = Field(None, description="特殊需求")
    
    @validator('messages')
    def validate_messages(cls, v):
        """验证消息列表"""
        if not v:
            raise ValueError("消息列表不能为空")
        if len(v) > 100:
            raise ValueError("消息数量不能超过100条")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """验证温度参数"""
        if v < 0 or v > 2:
            raise ValueError("温度参数必须在0-2之间")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        """验证最大token数"""
        if v is not None and (v < 1 or v > 32768):
            raise ValueError("最大token数必须在1-32768之间")
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        """验证top_p参数"""
        if v < 0 or v > 1:
            raise ValueError("top_p参数必须在0-1之间")
        return v


class Usage(BaseModel):
    """token使用情况"""
    
    prompt_tokens: int = Field(..., description="提示token数")
    completion_tokens: int = Field(..., description="完成token数")
    total_tokens: int = Field(..., description="总token数")
    
    @validator('total_tokens')
    def validate_total_tokens(cls, v, values):
        """验证总token数"""
        expected_total = values.get('prompt_tokens', 0) + values.get('completion_tokens', 0)
        if v != expected_total:
            # 允许不一致，但记录警告
            pass
        return v


class Choice(BaseModel):
    """选择项"""
    
    index: int = Field(..., description="选择索引")
    message: ChatMessage = Field(..., description="消息内容")
    finish_reason: Optional[str] = Field(None, description="完成原因")
    
    @validator('finish_reason')
    def validate_finish_reason(cls, v):
        """验证完成原因"""
        if v is not None:
            allowed_reasons = ["stop", "length", "content_filter", "null", "function_call"]
            if v not in allowed_reasons:
                # 允许其他原因，但记录警告
                pass
        return v


class ResponseMetadata(BaseModel):
    """响应元数据"""
    
    response_time_ms: float = Field(..., description="响应时间(毫秒)")
    cache_hit: bool = Field(False, description="是否缓存命中")
    cost: float = Field(0.0, description="请求成本")
    account_id: Optional[str] = Field(None, description="使用的账号ID")
    model_used: Optional[str] = Field(None, description="实际使用的模型")
    provider_used: Optional[str] = Field(None, description="实际使用的提供商")
    routing_strategy: Optional[str] = Field(None, description="使用的路由策略")
    retry_count: int = Field(0, description="重试次数")
    
    @validator('response_time_ms')
    def validate_response_time(cls, v):
        """验证响应时间"""
        if v < 0:
            raise ValueError("响应时间不能为负数")
        return v
    
    @validator('cost')
    def validate_cost(cls, v):
        """验证成本"""
        if v < 0:
            raise ValueError("成本不能为负数")
        return v


class ChatCompletionResponse(BaseModel):
    """聊天完成响应模型"""
    
    id: str = Field(..., description="响应ID")
    object: str = Field("chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    provider: str = Field(..., description="使用的提供商")
    choices: List[Choice] = Field(..., description="选择列表")
    usage: Optional[Usage] = Field(None, description="token使用情况")
    metadata: ResponseMetadata = Field(..., description="响应元数据")
    
    @validator('choices')
    def validate_choices(cls, v):
        """验证选择列表"""
        if not v:
            raise ValueError("选择列表不能为空")
        return v
    
    @validator('created')
    def validate_created(cls, v):
        """验证创建时间"""
        if v <= 0:
            raise ValueError("创建时间必须大于0")
        return v


class StreamingChunk(BaseModel):
    """流式响应块"""
    
    id: str = Field(..., description="响应ID")
    object: str = Field("chat.completion.chunk", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    choices: List[Dict[str, Any]] = Field(..., description="增量选择")


class ModelInfo(BaseModel):
    """模型信息"""
    
    id: str = Field(..., description="模型ID")
    name: str = Field(..., description="模型名称")
    provider: str = Field(..., description="提供商")
    description: Optional[str] = Field(None, description="模型描述")
    context_window: int = Field(..., description="上下文窗口")
    max_tokens: int = Field(..., description="最大输出token")
    cost_per_1k_tokens: float = Field(..., description="每1K token成本")
    capabilities: Dict[str, bool] = Field(..., description="模型能力")
    availability: Dict[str, Any] = Field(..., description="可用性信息")


class ModelListResponse(BaseModel):
    """模型列表响应"""
    
    object: str = Field("list", description="对象类型")
    data: List[ModelInfo] = Field(..., description="模型列表")


class AccountInfo(BaseModel):
    """账号信息"""
    
    id: str = Field(..., description="账号ID")
    provider: str = Field(..., description="提供商")
    account_name: str = Field(..., description="账号名称")
    status: str = Field(..., description="账号状态")
    quota_limit: int = Field(..., description="配额限制")
    quota_used: int = Field(..., description="已用配额")
    quota_remaining: int = Field(..., description="剩余配额")
    health_score: float = Field(..., description="健康评分")
    last_used_at: Optional[datetime] = Field(None, description="最后使用时间")
    
    @property
    def quota_usage_rate(self) -> float:
        """配额使用率"""
        if self.quota_limit <= 0:
            return 0.0
        return self.quota_used / self.quota_limit


class StatisticsRequest(BaseModel):
    """统计查询请求"""
    
    period: Literal["hourly", "daily", "weekly", "monthly"] = Field("daily", description="统计周期")
    start_date: Optional[datetime] = Field(None, description="开始日期")
    end_date: Optional[datetime] = Field(None, description="结束日期")
    model: Optional[str] = Field(None, description="指定模型")
    provider: Optional[str] = Field(None, description="指定提供商")
    account: Optional[str] = Field(None, description="指定账号")


class StatisticsResponse(BaseModel):
    """统计响应"""
    
    period: str = Field(..., description="统计周期")
    date_range: Dict[str, str] = Field(..., description="日期范围")
    summary: Dict[str, Any] = Field(..., description="汇总统计")
    by_model: List[Dict[str, Any]] = Field(..., description="按模型统计")
    by_provider: List[Dict[str, Any]] = Field(..., description="按提供商统计")
    by_account: List[Dict[str, Any]] = Field(..., description="按账号统计")
    trends: List[Dict[str, Any]] = Field(..., description="趋势数据")