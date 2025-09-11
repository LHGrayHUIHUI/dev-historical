"""
智能文本优化数据模型 - Optimization Data Models

定义文本优化服务相关的数据模型，包括请求模型、响应模型、
内部数据结构等，基于Pydantic实现数据验证和序列化

核心模型:
1. 优化请求和响应模型
2. 质量评估模型
3. 版本管理模型
4. 批量处理模型
5. 策略配置模型
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator


class OptimizationType(str, Enum):
    """文本优化类型枚举"""
    POLISH = "polish"           # 文本润色
    EXPAND = "expand"           # 内容扩展
    STYLE_CONVERT = "style_convert"  # 风格转换
    MODERNIZE = "modernize"     # 现代化改写


class OptimizationMode(str, Enum):
    """文本优化模式枚举"""
    HISTORICAL_FORMAT = "historical_format"  # 历史文档格式
    ACADEMIC = "academic"                     # 学术规范
    LITERARY = "literary"                     # 文学性
    SIMPLIFIED = "simplified"                 # 简化表达


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"         # 等待中
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"    # 已取消


class QualityMetrics(BaseModel):
    """质量评估指标模型"""
    overall_score: float = Field(..., ge=0, le=100, description="综合质量分数")
    readability_score: float = Field(..., ge=0, le=100, description="可读性分数") 
    academic_score: float = Field(..., ge=0, le=100, description="学术规范性分数")
    historical_accuracy: float = Field(..., ge=0, le=100, description="历史准确性分数")
    language_quality: float = Field(..., ge=0, le=100, description="语言质量分数")
    structure_score: float = Field(..., ge=0, le=100, description="结构质量分数")
    content_completeness: float = Field(..., ge=0, le=100, description="内容完整性分数")
    
    # 改进指标
    readability_improvement: Optional[float] = Field(None, description="可读性改进幅度")
    academic_improvement: Optional[float] = Field(None, description="学术性改进幅度")
    structure_improvement: Optional[float] = Field(None, description="结构改进幅度")
    
    # 详细分析
    strengths: List[str] = Field(default_factory=list, description="文本优势")
    weaknesses: List[str] = Field(default_factory=list, description="改进建议")
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 2) if v is not None else None
        }


class TokenUsage(BaseModel):
    """Token使用统计模型"""
    prompt_tokens: int = Field(..., ge=0, description="提示词Token数")
    completion_tokens: int = Field(..., ge=0, description="完成Token数")
    total_tokens: int = Field(..., ge=0, description="总Token数")
    estimated_cost: Optional[float] = Field(None, ge=0, description="预估成本")
    
    @validator("total_tokens", always=True)
    def validate_total_tokens(cls, v, values):
        """验证总Token数等于提示词和完成Token数之和"""
        prompt = values.get("prompt_tokens", 0)
        completion = values.get("completion_tokens", 0)
        if v != prompt + completion:
            return prompt + completion
        return v


class OptimizationParameters(BaseModel):
    """文本优化参数配置"""
    target_length: Optional[int] = Field(None, gt=0, description="目标文本长度")
    style_reference: Optional[str] = Field(None, description="风格参考文本")
    preserve_entities: bool = Field(True, description="是否保留历史实体")
    quality_threshold: float = Field(80.0, ge=0, le=100, description="质量阈值")
    custom_instructions: Optional[str] = Field(None, description="自定义优化指令")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="生成温度")
    max_tokens: Optional[int] = Field(None, gt=0, description="最大生成Token数")
    
    # 高级参数
    enable_fact_checking: bool = Field(True, description="启用事实检查")
    preserve_formatting: bool = Field(True, description="保留格式")
    enable_terminology_consistency: bool = Field(True, description="启用术语一致性")


class OptimizationRequest(BaseModel):
    """单个文本优化请求模型"""
    content: str = Field(..., min_length=1, max_length=100000, description="待优化文本内容")
    optimization_type: OptimizationType = Field(..., description="优化类型")
    optimization_mode: OptimizationMode = Field(..., description="优化模式")
    parameters: OptimizationParameters = Field(default_factory=OptimizationParameters, description="优化参数")
    generate_versions: int = Field(1, ge=1, le=5, description="生成版本数量")
    
    # 元数据
    document_id: Optional[str] = Field(None, description="关联文档ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    tags: Optional[Dict[str, Any]] = Field(None, description="标签信息")
    
    @validator("content")
    def validate_content(cls, v):
        """验证内容不为空"""
        if not v.strip():
            raise ValueError("文本内容不能为空")
        return v.strip()


class OptimizationVersion(BaseModel):
    """优化版本模型"""
    version_id: str = Field(default_factory=lambda: str(uuid4()), description="版本ID")
    version_number: int = Field(..., ge=1, description="版本号")
    content: str = Field(..., description="优化后内容")
    title: Optional[str] = Field(None, description="优化后标题")
    summary: Optional[str] = Field(None, description="内容摘要")
    
    # 质量评估
    quality_metrics: QualityMetrics = Field(..., description="质量指标")
    improvements: List[str] = Field(default_factory=list, description="改进说明")
    
    # 技术信息
    model_used: str = Field(..., description="使用的AI模型")
    processing_time_ms: int = Field(..., ge=0, description="处理时间(毫秒)")
    token_usage: TokenUsage = Field(..., description="Token使用情况")
    
    # 元数据
    is_selected: bool = Field(False, description="是否为选中版本")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class OptimizationResult(BaseModel):
    """文本优化结果模型"""
    task_id: str = Field(default_factory=lambda: str(uuid4()), description="任务ID")
    status: TaskStatus = Field(TaskStatus.COMPLETED, description="任务状态")
    versions: List[OptimizationVersion] = Field(..., description="优化版本列表")
    recommended_version: Optional[str] = Field(None, description="推荐版本ID")
    
    # 原始信息
    original_content: str = Field(..., description="原始文本")
    request_parameters: OptimizationParameters = Field(..., description="请求参数")
    
    # 统计信息
    total_versions: int = Field(..., ge=1, description="总版本数")
    average_quality_score: float = Field(..., ge=0, le=100, description="平均质量分数")
    best_quality_score: float = Field(..., ge=0, le=100, description="最佳质量分数")
    total_processing_time_ms: int = Field(..., ge=0, description="总处理时间")
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    
    @validator("total_versions", always=True)
    def validate_total_versions(cls, v, values):
        """验证总版本数与版本列表长度一致"""
        versions = values.get("versions", [])
        if v != len(versions):
            return len(versions)
        return v
    
    @validator("recommended_version", always=True)
    def validate_recommended_version(cls, v, values):
        """验证推荐版本ID的有效性"""
        versions = values.get("versions", [])
        if v and versions:
            version_ids = [version.version_id for version in versions]
            if v not in version_ids:
                # 如果推荐版本无效，选择质量最高的版本
                if versions:
                    best_version = max(versions, key=lambda x: x.quality_metrics.overall_score)
                    return best_version.version_id
        return v


class BatchOptimizationRequest(BaseModel):
    """批量文本优化请求模型"""
    job_name: str = Field(..., min_length=1, max_length=200, description="任务名称")
    document_ids: List[str] = Field(..., min_items=1, max_items=1000, description="文档ID列表")
    optimization_config: Dict[str, Any] = Field(..., description="优化配置")
    
    # 批量配置
    parallel_processing: bool = Field(True, description="是否并行处理")
    max_concurrent_tasks: int = Field(5, ge=1, le=20, description="最大并发任务数")
    priority: int = Field(1, ge=1, le=10, description="任务优先级")
    
    # 通知配置
    notification_email: Optional[str] = Field(None, description="完成通知邮箱")
    webhook_url: Optional[str] = Field(None, description="Webhook回调URL")
    
    # 元数据
    user_id: str = Field(..., description="用户ID")
    tags: Optional[Dict[str, Any]] = Field(None, description="标签信息")
    
    @validator("document_ids")
    def validate_document_ids(cls, v):
        """验证文档ID列表不为空且无重复"""
        if not v:
            raise ValueError("文档ID列表不能为空")
        if len(v) != len(set(v)):
            raise ValueError("文档ID列表中不能有重复项")
        return v


class BatchOptimizationStatus(BaseModel):
    """批量优化任务状态模型"""
    job_id: str = Field(..., description="任务ID")
    job_name: str = Field(..., description="任务名称")
    status: TaskStatus = Field(..., description="任务状态")
    
    # 进度信息
    total_documents: int = Field(..., ge=0, description="文档总数")
    completed_documents: int = Field(..., ge=0, description="已完成文档数")
    failed_documents: int = Field(..., ge=0, description="失败文档数")
    progress_percentage: float = Field(..., ge=0, le=100, description="进度百分比")
    
    # 时间信息
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    estimated_completion_time: Optional[datetime] = Field(None, description="预计完成时间")
    estimated_remaining_time: Optional[str] = Field(None, description="预计剩余时间")
    
    # 结果统计
    results: Optional[Dict[str, Any]] = Field(None, description="结果统计")
    error_summary: Optional[List[str]] = Field(None, description="错误摘要")
    
    # 元数据
    user_id: str = Field(..., description="用户ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    
    @validator("progress_percentage", always=True)
    def calculate_progress(cls, v, values):
        """计算进度百分比"""
        total = values.get("total_documents", 0)
        completed = values.get("completed_documents", 0)
        if total > 0:
            return round((completed / total) * 100, 2)
        return 0


class VersionComparison(BaseModel):
    """版本对比模型"""
    version1: OptimizationVersion = Field(..., description="版本1")
    version2: OptimizationVersion = Field(..., description="版本2")
    
    # 差异信息
    differences: List[Dict[str, Any]] = Field(default_factory=list, description="内容差异")
    similarity_score: float = Field(..., ge=0, le=100, description="相似度分数")
    
    # 质量对比
    quality_comparison: Dict[str, Any] = Field(default_factory=dict, description="质量对比")
    improvement_areas: List[str] = Field(default_factory=list, description="改进领域")
    recommendation: Optional[str] = Field(None, description="推荐建议")


class OptimizationStrategy(BaseModel):
    """优化策略配置模型"""
    strategy_id: str = Field(default_factory=lambda: str(uuid4()), description="策略ID")
    name: str = Field(..., description="策略名称")
    description: Optional[str] = Field(None, description="策略描述")
    
    # 策略配置
    optimization_type: OptimizationType = Field(..., description="优化类型")
    optimization_mode: OptimizationMode = Field(..., description="优化模式")
    target_text_types: List[str] = Field(default_factory=list, description="适用文本类型")
    
    # 提示词模板
    system_prompt: str = Field(..., description="系统提示词")
    prompt_template: str = Field(..., description="提示词模板")
    
    # AI模型配置
    preferred_model: str = Field(..., description="首选模型")
    fallback_models: List[str] = Field(default_factory=list, description="备用模型")
    model_parameters: Dict[str, Any] = Field(default_factory=dict, description="模型参数")
    
    # 质量配置
    quality_thresholds: Dict[str, float] = Field(default_factory=dict, description="质量阈值")
    quality_weights: Dict[str, float] = Field(default_factory=dict, description="质量权重")
    
    # 统计信息
    usage_count: int = Field(0, ge=0, description="使用次数")
    success_rate: float = Field(0, ge=0, le=100, description="成功率")
    avg_quality_improvement: float = Field(0, description="平均质量提升")
    avg_processing_time_ms: int = Field(0, ge=0, description="平均处理时间")
    
    # 元数据
    is_active: bool = Field(True, description="是否激活")
    is_default: bool = Field(False, description="是否为默认策略")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")


class ApiResponse(BaseModel):
    """统一API响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(default="", description="响应消息")
    data: Optional[Union[Dict[str, Any], List[Any]]] = Field(None, description="响应数据")
    error: Optional[Dict[str, Any]] = Field(None, description="错误信息")
    
    # 元数据
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="请求ID")
    
    @classmethod
    def success_response(cls, data: Any = None, message: str = "操作成功"):
        """创建成功响应"""
        return cls(success=True, data=data, message=message)
    
    @classmethod
    def error_response(cls, message: str, error_details: Optional[Dict] = None):
        """创建错误响应"""
        return cls(success=False, message=message, error=error_details)


class HealthStatus(BaseModel):
    """健康检查状态模型"""
    service_name: str = Field(..., description="服务名称")
    service_version: str = Field(..., description="服务版本")
    status: str = Field(..., description="服务状态")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="检查时间")
    
    # 依赖服务状态
    dependencies: Dict[str, bool] = Field(default_factory=dict, description="依赖服务状态")
    
    # 系统资源状态
    system_info: Optional[Dict[str, Any]] = Field(None, description="系统信息")
    
    # 性能指标
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="性能指标")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }