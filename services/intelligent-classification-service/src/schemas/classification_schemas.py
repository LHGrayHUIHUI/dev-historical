"""
智能分类服务数据模型和Pydantic Schemas
定义API请求响应格式和数据验证规则
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


# ============ 枚举类型定义 ============

class ClassificationType(str, Enum):
    """分类类型枚举"""
    TOPIC = "topic"  # 主题分类
    ERA = "era"  # 时代分类
    DOCUMENT_TYPE = "document_type"  # 文档类型分类
    IMPORTANCE = "importance"  # 重要性评级
    SENTIMENT = "sentiment"  # 情感分析
    GENRE = "genre"  # 体裁分类


class ModelType(str, Enum):
    """机器学习模型类型"""
    SVM = "svm"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    BERT = "bert"
    ROBERTA = "roberta"


class FeatureExtractorType(str, Enum):
    """特征提取器类型"""
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    BERT = "bert"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProjectStatus(str, Enum):
    """项目状态"""
    ACTIVE = "active"
    TRAINING = "training"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# ============ 基础响应模型 ============

class BaseResponse(BaseModel):
    """统一API响应格式"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    

# ============ 项目管理相关模型 ============

class ClassificationProjectCreate(BaseModel):
    """创建分类项目请求"""
    name: str = Field(..., min_length=1, max_length=200, description="项目名称")
    description: Optional[str] = Field(None, max_length=1000, description="项目描述")
    classification_type: ClassificationType = Field(..., description="分类类型")
    domain: Optional[str] = Field(None, max_length=100, description="领域")
    language: str = Field("zh", description="语言")
    custom_labels: Optional[List[str]] = Field(None, description="自定义标签")
    ml_model_config: Optional[Dict[str, Any]] = Field(None, description="模型配置")
    ml_training_config: Optional[Dict[str, Any]] = Field(None, description="训练配置")
    ml_feature_config: Optional[Dict[str, Any]] = Field(None, description="特征配置")


class ClassificationProject(BaseModel):
    """分类项目信息"""
    id: str = Field(..., description="项目ID")
    name: str = Field(..., description="项目名称")
    description: Optional[str] = Field(None, description="项目描述")
    classification_type: ClassificationType = Field(..., description="分类类型")
    domain: Optional[str] = Field(None, description="领域")
    language: str = Field(..., description="语言")
    status: ProjectStatus = Field(..., description="项目状态")
    class_labels: List[str] = Field(..., description="分类标签")
    ml_model_config: Dict[str, Any] = Field(..., description="模型配置")
    ml_training_config: Dict[str, Any] = Field(..., description="训练配置")
    ml_feature_config: Dict[str, Any] = Field(..., description="特征配置")
    created_by: Optional[str] = Field(None, description="创建者")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


# ============ 训练数据相关模型 ============

class TrainingDataCreate(BaseModel):
    """添加训练数据请求"""
    project_id: str = Field(..., description="项目ID")
    text_content: str = Field(..., min_length=1, max_length=10000, description="文本内容")
    true_label: str = Field(..., description="真实标签")
    document_id: Optional[str] = Field(None, description="文档ID")
    label_confidence: float = Field(1.0, ge=0.0, le=1.0, description="标签置信度")
    data_source: str = Field("manual", description="数据来源")


class TrainingDataBatch(BaseModel):
    """批量添加训练数据请求"""
    project_id: str = Field(..., description="项目ID")
    training_data: List[Dict[str, Any]] = Field(..., min_items=1, description="训练数据列表")


class TrainingData(BaseModel):
    """训练数据信息"""
    id: str = Field(..., description="训练数据ID")
    project_id: str = Field(..., description="项目ID")
    document_id: Optional[str] = Field(None, description="文档ID")
    text_content: str = Field(..., description="文本内容")
    true_label: str = Field(..., description="真实标签")
    label_confidence: float = Field(..., description="标签置信度")
    data_source: str = Field(..., description="数据来源")
    text_features: Optional[Dict[str, Any]] = Field(None, description="文本特征")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    created_at: datetime = Field(..., description="创建时间")


# ============ 模型训练相关模型 ============

class ModelTrainingRequest(BaseModel):
    """模型训练请求"""
    project_id: str = Field(..., description="项目ID")
    model_type: ModelType = Field(..., description="模型类型")
    feature_extractor: FeatureExtractorType = Field(..., description="特征提取器")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="超参数")
    training_config: Optional[Dict[str, Any]] = Field(None, description="训练配置")


class ModelInfo(BaseModel):
    """模型信息"""
    id: str = Field(..., description="模型ID")
    project_id: str = Field(..., description="项目ID")
    model_name: str = Field(..., description="模型名称")
    model_type: ModelType = Field(..., description="模型类型")
    feature_extractor: FeatureExtractorType = Field(..., description="特征提取器")
    model_version: str = Field(..., description="模型版本")
    model_path: Optional[str] = Field(None, description="模型文件路径")
    hyperparameters: Dict[str, Any] = Field(..., description="超参数")
    status: TaskStatus = Field(..., description="模型状态")
    is_active: bool = Field(..., description="是否为活跃模型")
    training_data_size: Optional[int] = Field(None, description="训练数据量")
    training_time: Optional[float] = Field(None, description="训练时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class TrainingMetrics(BaseModel):
    """训练指标"""
    accuracy: float = Field(..., description="准确率")
    precision: float = Field(..., description="精确率")
    recall: float = Field(..., description="召回率")
    f1_score: float = Field(..., description="F1分数")
    cv_mean: float = Field(..., description="交叉验证均值")
    cv_std: float = Field(..., description="交叉验证标准差")
    confusion_matrix: List[List[int]] = Field(..., description="混淆矩阵")
    classification_report: Dict[str, Any] = Field(..., description="分类报告")


class ModelTrainingResponse(BaseModel):
    """模型训练响应"""
    model_id: str = Field(..., description="模型ID")
    project_id: str = Field(..., description="项目ID")
    training_metrics: TrainingMetrics = Field(..., description="训练指标")
    training_time: float = Field(..., description="训练时间")
    model_path: str = Field(..., description="模型路径")
    status: TaskStatus = Field(..., description="训练状态")


# ============ 文档分类相关模型 ============

class ClassificationRequest(BaseModel):
    """文档分类请求"""
    project_id: str = Field(..., description="项目ID")
    text_content: str = Field(..., min_length=1, max_length=10000, description="文本内容")
    document_id: Optional[str] = Field(None, description="文档ID")
    model_id: Optional[str] = Field(None, description="指定模型ID")
    return_probabilities: bool = Field(True, description="是否返回概率分布")
    return_explanation: bool = Field(True, description="是否返回分类解释")


class BatchClassificationRequest(BaseModel):
    """批量分类请求"""
    project_id: str = Field(..., description="项目ID")
    documents: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="文档列表")
    model_id: Optional[str] = Field(None, description="指定模型ID")
    return_probabilities: bool = Field(True, description="是否返回概率分布")
    return_explanation: bool = Field(False, description="是否返回分类解释")


class ClassificationResult(BaseModel):
    """分类结果"""
    task_id: str = Field(..., description="任务ID")
    document_id: Optional[str] = Field(None, description="文档ID")
    predicted_label: str = Field(..., description="预测标签")
    confidence_score: float = Field(..., description="置信度")
    probability_distribution: Optional[Dict[str, float]] = Field(None, description="概率分布")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="特征重要性")
    explanation: Optional[str] = Field(None, description="分类解释")
    processing_time: float = Field(..., description="处理时间")
    model_info: Dict[str, Any] = Field(..., description="使用的模型信息")


class BatchClassificationResult(BaseModel):
    """批量分类结果"""
    batch_task_id: str = Field(..., description="批量任务ID")
    total_documents: int = Field(..., description="总文档数")
    successful_classifications: int = Field(..., description="成功分类数")
    failed_classifications: int = Field(..., description="失败分类数")
    results: List[ClassificationResult] = Field(..., description="分类结果列表")
    processing_time: float = Field(..., description="总处理时间")
    statistics: Dict[str, Any] = Field(..., description="统计信息")


# ============ 模型性能相关模型 ============

class ModelPerformanceRequest(BaseModel):
    """模型性能查询请求"""
    model_id: str = Field(..., description="模型ID")
    include_usage_stats: bool = Field(True, description="是否包含使用统计")
    include_detailed_metrics: bool = Field(False, description="是否包含详细指标")


class UsageStatistics(BaseModel):
    """使用统计"""
    total_predictions: int = Field(..., description="总预测次数")
    avg_confidence: float = Field(..., description="平均置信度")
    avg_processing_time: float = Field(..., description="平均处理时间")
    confidence_distribution: Dict[str, int] = Field(..., description="置信度分布")
    label_distribution: Dict[str, int] = Field(..., description="标签分布")
    daily_usage: Dict[str, int] = Field(..., description="日使用量")
    error_rate: float = Field(..., description="错误率")


class ModelPerformanceResponse(BaseModel):
    """模型性能响应"""
    model_info: ModelInfo = Field(..., description="模型信息")
    training_metrics: TrainingMetrics = Field(..., description="训练指标")
    usage_statistics: Optional[UsageStatistics] = Field(None, description="使用统计")
    performance_trend: Optional[Dict[str, List[float]]] = Field(None, description="性能趋势")
    comparison_with_baseline: Optional[Dict[str, float]] = Field(None, description="与基线对比")


# ============ 任务管理相关模型 ============

class TaskInfo(BaseModel):
    """任务信息"""
    id: str = Field(..., description="任务ID")
    project_id: str = Field(..., description="项目ID")
    task_type: str = Field(..., description="任务类型")
    status: TaskStatus = Field(..., description="任务状态")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="任务进度")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    processing_time: Optional[float] = Field(None, description="处理时间")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    error_message: Optional[str] = Field(None, description="错误信息")


# ============ 项目统计相关模型 ============

class ProjectStatistics(BaseModel):
    """项目统计信息"""
    project_id: str = Field(..., description="项目ID")
    total_training_data: int = Field(..., description="训练数据总数")
    total_models: int = Field(..., description="模型总数")
    total_predictions: int = Field(..., description="预测总数")
    active_models: int = Field(..., description="活跃模型数")
    label_distribution: Dict[str, int] = Field(..., description="标签分布")
    model_performance: Dict[str, float] = Field(..., description="模型性能")
    recent_activity: List[Dict[str, Any]] = Field(..., description="最近活动")
    data_quality_score: float = Field(..., description="数据质量分数")


# ============ 系统配置相关模型 ============

class SystemConfig(BaseModel):
    """系统配置"""
    supported_classification_types: List[str] = Field(..., description="支持的分类类型")
    supported_model_types: List[str] = Field(..., description="支持的模型类型")
    supported_feature_extractors: List[str] = Field(..., description="支持的特征提取器")
    predefined_labels: Dict[str, List[str]] = Field(..., description="预定义标签")
    max_text_length: int = Field(..., description="最大文本长度")
    max_batch_size: int = Field(..., description="最大批量大小")
    performance_thresholds: Dict[str, float] = Field(..., description="性能阈值")


# ============ 健康检查相关模型 ============

class HealthCheck(BaseModel):
    """健康检查响应"""
    service: str = Field(..., description="服务名称")
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    timestamp: datetime = Field(..., description="检查时间")
    dependencies: Dict[str, str] = Field(..., description="依赖服务状态")
    system_info: Dict[str, Any] = Field(..., description="系统信息")


# ============ 数据验证器 ============

# 在Pydantic v2中，validator已被移除，字段验证应在模型内部定义


# ============ 模型导出和导入 ============

class ModelExportRequest(BaseModel):
    """模型导出请求"""
    model_id: str = Field(..., description="模型ID")
    export_format: str = Field("pickle", description="导出格式")
    include_metadata: bool = Field(True, description="是否包含元数据")
    include_training_data: bool = Field(False, description="是否包含训练数据")


class ModelImportRequest(BaseModel):
    """模型导入请求"""
    project_id: str = Field(..., description="项目ID")
    model_file: str = Field(..., description="模型文件路径")
    model_name: str = Field(..., description="模型名称")
    replace_existing: bool = Field(False, description="是否替换现有模型")


# ============ A/B测试相关模型 ============

class ABTestRequest(BaseModel):
    """A/B测试请求"""
    project_id: str = Field(..., description="项目ID")
    model_a_id: str = Field(..., description="模型A ID")
    model_b_id: str = Field(..., description="模型B ID")
    test_data: List[Dict[str, Any]] = Field(..., description="测试数据")
    test_name: str = Field(..., description="测试名称")


class ABTestResult(BaseModel):
    """A/B测试结果"""
    test_id: str = Field(..., description="测试ID")
    model_a_performance: Dict[str, float] = Field(..., description="模型A性能")
    model_b_performance: Dict[str, float] = Field(..., description="模型B性能")
    winner: str = Field(..., description="获胜模型")
    confidence_level: float = Field(..., description="置信水平")
    recommendation: str = Field(..., description="推荐")
    detailed_comparison: Dict[str, Any] = Field(..., description="详细对比")