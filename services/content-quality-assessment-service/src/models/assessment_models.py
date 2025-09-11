"""
内容质量评估数据模型

定义质量评估系统的所有数据模型，包括评估请求、结果、趋势分析、基准管理等。
支持多维度质量评估、趋势跟踪和基准对比功能。
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import uuid

# ==================== 枚举定义 ====================

class QualityDimension(str, Enum):
    """质量评估维度"""
    READABILITY = "readability"          # 可读性
    ACCURACY = "accuracy"                # 准确性  
    COMPLETENESS = "completeness"        # 完整性
    COHERENCE = "coherence"              # 连贯性
    RELEVANCE = "relevance"              # 相关性
    ORIGINALITY = "originality"          # 原创性 (扩展维度)
    AUTHORITY = "authority"              # 权威性 (扩展维度)
    TIMELINESS = "timeliness"           # 时效性 (扩展维度)

class ContentType(str, Enum):
    """内容类型"""
    HISTORICAL_DOCUMENT = "historical_document"    # 历史文档
    ACADEMIC_PAPER = "academic_paper"              # 学术论文
    NARRATIVE_TEXT = "narrative_text"              # 叙述文本
    REFERENCE_MATERIAL = "reference_material"      # 参考资料
    EDUCATIONAL_CONTENT = "educational_content"    # 教育内容

class AssessmentStatus(str, Enum):
    """评估状态"""
    PENDING = "pending"          # 等待中
    PROCESSING = "processing"    # 处理中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 已取消

class TrendDirection(str, Enum):
    """趋势方向"""
    IMPROVING = "improving"      # 改善中
    DECLINING = "declining"      # 下降中
    STABLE = "stable"           # 稳定

class QualityGrade(str, Enum):
    """质量等级"""
    A = "A"  # 优秀 (90-100)
    B = "B"  # 良好 (80-89)
    C = "C"  # 中等 (70-79)
    D = "D"  # 及格 (60-69)
    F = "F"  # 不及格 (0-59)

# ==================== 基础模型 ====================

class TokenUsage(BaseModel):
    """Token使用情况"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ProcessingMetrics(BaseModel):
    """处理指标"""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

class AssessmentError(BaseModel):
    """评估错误信息"""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# ==================== 质量指标模型 ====================

class QualityMetric(BaseModel):
    """质量指标"""
    dimension: QualityDimension
    score: float = Field(..., ge=0.0, le=100.0, description="评分 0-100")
    weight: float = Field(..., ge=0.0, le=1.0, description="权重 0-1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")
    details: Dict[str, float] = Field(default_factory=dict, description="详细子指标")
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="原始分析数据")
    
    @validator('score')
    def validate_score(cls, v):
        """验证评分范围"""
        if not 0.0 <= v <= 100.0:
            raise ValueError("评分必须在0-100范围内")
        return round(v, 2)
    
    @validator('weight')
    def validate_weight(cls, v):
        """验证权重范围"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("权重必须在0-1范围内")
        return round(v, 3)

class ReadabilityMetric(QualityMetric):
    """可读性指标详情"""
    avg_sentence_length: Optional[float] = None
    vocab_diversity: Optional[float] = None
    char_complexity: Optional[float] = None
    syntax_complexity: Optional[float] = None
    flesch_score: Optional[float] = None

class AccuracyMetric(QualityMetric):
    """准确性指标详情"""
    factual_consistency: Optional[float] = None
    grammar_accuracy: Optional[float] = None
    terminology_usage: Optional[float] = None
    citation_accuracy: Optional[float] = None
    spell_check_score: Optional[float] = None

class CompletenessMetric(QualityMetric):
    """完整性指标详情"""
    structure_completeness: Optional[float] = None
    information_completeness: Optional[float] = None
    logic_completeness: Optional[float] = None
    missing_elements: Optional[List[str]] = None

class CoherenceMetric(QualityMetric):
    """连贯性指标详情"""
    logical_flow: Optional[float] = None
    transition_quality: Optional[float] = None
    argument_consistency: Optional[float] = None
    narrative_coherence: Optional[float] = None

class RelevanceMetric(QualityMetric):
    """相关性指标详情"""
    topic_relevance: Optional[float] = None
    audience_relevance: Optional[float] = None
    contextual_relevance: Optional[float] = None
    keyword_relevance: Optional[float] = None

# ==================== 评估请求和结果 ====================

class QualityAssessmentRequest(BaseModel):
    """质量评估请求"""
    content: str = Field(..., min_length=1, max_length=50000)
    content_type: ContentType
    content_id: str = Field(..., min_length=1)
    assessment_id: Optional[str] = Field(default=None)
    
    # 可选配置
    custom_weights: Optional[Dict[QualityDimension, float]] = None
    enabled_dimensions: Optional[List[QualityDimension]] = None
    target_audience: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None
    
    # 处理选项
    enable_caching: bool = Field(default=True)
    cache_ttl_hours: Optional[int] = Field(default=24, ge=1, le=168)  # 1-168小时
    priority: int = Field(default=5, ge=1, le=10)  # 1最高，10最低
    
    @validator('assessment_id', pre=True, always=True)
    def generate_assessment_id(cls, v):
        """生成评估ID"""
        if v is None:
            return f"qa_{uuid.uuid4().hex[:12]}"
        return v
    
    @validator('custom_weights')
    def validate_custom_weights(cls, v):
        """验证自定义权重"""
        if v is not None:
            total_weight = sum(v.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"权重总和必须等于1.0，当前为{total_weight}")
        return v

class QualityAssessmentResult(BaseModel):
    """质量评估结果"""
    assessment_id: str
    content_id: str
    content_type: ContentType
    
    # 评估结果
    overall_score: float = Field(..., ge=0.0, le=100.0)
    grade: QualityGrade
    metrics: List[QualityMetric]
    
    # 分析报告
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # 评估元数据
    assessment_time: datetime = Field(default_factory=datetime.now)
    processing_duration: float = Field(..., ge=0.0)
    model_versions: Dict[str, str] = Field(default_factory=dict)
    
    # 处理状态
    status: AssessmentStatus = AssessmentStatus.COMPLETED
    error_info: Optional[AssessmentError] = None
    processing_metrics: Optional[ProcessingMetrics] = None
    
    @validator('overall_score')
    def validate_overall_score(cls, v):
        """验证综合评分"""
        return round(v, 2)

class BatchAssessmentRequest(BaseModel):
    """批量评估请求"""
    requests: List[QualityAssessmentRequest] = Field(..., min_items=1, max_items=50)
    batch_id: Optional[str] = Field(default=None)
    parallel_processing: bool = Field(default=True)
    max_concurrent: int = Field(default=10, ge=1, le=20)
    timeout_minutes: int = Field(default=30, ge=5, le=120)
    
    @validator('batch_id', pre=True, always=True)
    def generate_batch_id(cls, v):
        """生成批次ID"""
        if v is None:
            return f"batch_{uuid.uuid4().hex[:12]}"
        return v

class BatchAssessmentResult(BaseModel):
    """批量评估结果"""
    batch_id: str
    total_requests: int
    completed_count: int
    failed_count: int
    results: List[QualityAssessmentResult]
    failed_requests: List[Dict[str, Any]] = Field(default_factory=list)
    batch_start_time: datetime
    batch_end_time: Optional[datetime] = None
    total_duration: Optional[float] = None

# ==================== 趋势分析模型 ====================

class QualityTrend(BaseModel):
    """质量趋势"""
    content_id: str
    dimension: Optional[QualityDimension] = None  # None表示整体趋势
    
    # 趋势指标
    trend_direction: TrendDirection
    trend_strength: float = Field(..., ge=0.0, le=1.0)
    slope: float
    r_squared: float = Field(..., ge=0.0, le=1.0)
    
    # 预测信息
    prediction_score: float = Field(..., ge=0.0, le=100.0)
    confidence_interval: Tuple[float, float]
    next_assessment_recommended: Optional[datetime] = None
    
    # 统计信息
    data_points_count: int = Field(..., ge=1)
    analysis_period_days: int = Field(..., ge=1)
    average_score: float = Field(..., ge=0.0, le=100.0)
    score_variance: float = Field(..., ge=0.0)

class QualityTrendAnalysis(BaseModel):
    """质量趋势分析"""
    analysis_id: str = Field(default_factory=lambda: f"trend_{uuid.uuid4().hex[:12]}")
    content_id: str
    analysis_period: Tuple[datetime, datetime]
    
    # 趋势结果
    overall_trend: QualityTrend
    dimension_trends: List[QualityTrend] = Field(default_factory=list)
    
    # 分析报告
    improvement_suggestions: List[str] = Field(default_factory=list)
    risk_alerts: List[str] = Field(default_factory=list)
    trend_summary: str = ""
    
    # 预测信息
    next_assessment_date: Optional[datetime] = None
    predicted_performance: Optional[Dict[str, float]] = None
    
    # 分析元数据
    analysis_time: datetime = Field(default_factory=datetime.now)
    min_data_points_met: bool = True
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)

# ==================== 基准管理模型 ====================

class QualityBenchmark(BaseModel):
    """质量基准"""
    benchmark_id: str = Field(default_factory=lambda: f"benchmark_{uuid.uuid4().hex[:12]}")
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    
    # 基准配置
    content_type: ContentType
    target_audience: str = Field(..., min_length=1, max_length=100)
    
    # 标准设置
    dimension_standards: Dict[QualityDimension, float] = Field(..., min_items=1)
    overall_standard: float = Field(..., ge=0.0, le=100.0)
    
    # 基准元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    is_active: bool = Field(default=True)
    is_default: bool = Field(default=False)
    
    # 使用统计
    usage_count: int = Field(default=0, ge=0)
    last_used: Optional[datetime] = None
    
    @validator('dimension_standards')
    def validate_dimension_standards(cls, v):
        """验证维度标准"""
        for dimension, standard in v.items():
            if not 0.0 <= standard <= 100.0:
                raise ValueError(f"维度{dimension}的标准{standard}必须在0-100范围内")
        return v

class BenchmarkComparison(BaseModel):
    """基准对比"""
    comparison_id: str = Field(default_factory=lambda: f"comp_{uuid.uuid4().hex[:12]}")
    content_id: str
    benchmark_id: str
    assessment_result: QualityAssessmentResult
    
    # 对比结果
    comparison_details: Dict[QualityDimension, Dict[str, float]]
    meets_standard: bool
    compliance_score: float = Field(..., ge=0.0, le=100.0)
    
    # 差距分析
    improvement_gaps: List[str] = Field(default_factory=list)
    priority_improvements: List[str] = Field(default_factory=list)
    estimated_effort: Optional[str] = None  # 预估改进工作量
    
    # 对比元数据
    comparison_time: datetime = Field(default_factory=datetime.now)
    benchmark_version: str = "1.0"

# ==================== 仪表板模型 ====================

class QualityDashboardData(BaseModel):
    """质量仪表板数据"""
    content_id: str
    dashboard_id: str = Field(default_factory=lambda: f"dash_{uuid.uuid4().hex[:12]}")
    
    # 当前状态
    latest_assessment: Optional[QualityAssessmentResult] = None
    current_grade: Optional[QualityGrade] = None
    overall_score: Optional[float] = None
    
    # 历史趋势
    score_history: List[Dict[str, Any]] = Field(default_factory=list)
    trend_analysis: Optional[QualityTrendAnalysis] = None
    
    # 基准对比
    benchmark_comparisons: List[BenchmarkComparison] = Field(default_factory=list)
    compliance_status: Optional[bool] = None
    
    # 统计信息
    total_assessments: int = Field(default=0, ge=0)
    average_score: Optional[float] = None
    best_score: Optional[float] = None
    worst_score: Optional[float] = None
    improvement_rate: Optional[float] = None
    
    # 改进建议
    top_recommendations: List[str] = Field(default_factory=list)
    urgent_issues: List[str] = Field(default_factory=list)
    
    # 仪表板元数据
    generated_at: datetime = Field(default_factory=datetime.now)
    data_period: Tuple[datetime, datetime]
    refresh_interval_minutes: int = Field(default=60, ge=5, le=1440)

# ==================== 响应模型 ====================

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True
    message: str = "操作成功"
    timestamp: datetime = Field(default_factory=datetime.now)

class AssessmentResponse(BaseResponse):
    """评估响应"""
    data: Optional[QualityAssessmentResult] = None

class BatchAssessmentResponse(BaseResponse):
    """批量评估响应"""
    data: Optional[BatchAssessmentResult] = None

class TrendAnalysisResponse(BaseResponse):
    """趋势分析响应"""
    data: Optional[QualityTrendAnalysis] = None

class BenchmarkResponse(BaseResponse):
    """基准响应"""
    data: Optional[QualityBenchmark] = None

class ComparisonResponse(BaseResponse):
    """对比响应"""
    data: Optional[BenchmarkComparison] = None

class DashboardResponse(BaseResponse):
    """仪表板响应"""
    data: Optional[QualityDashboardData] = None

class ListResponse(BaseResponse):
    """列表响应"""
    data: List[Any] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    has_next: bool = False

# ==================== 健康检查模型 ====================

class HealthStatus(BaseModel):
    """健康状态"""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    dependencies: Dict[str, bool] = Field(default_factory=dict)
    performance_metrics: Optional[Dict[str, float]] = None
    system_info: Optional[Dict[str, Any]] = None