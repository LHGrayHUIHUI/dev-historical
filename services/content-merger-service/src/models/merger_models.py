"""
内容合并服务数据模型

定义内容合并服务使用的所有数据结构，包括请求响应模型、
合并策略配置、质量评估结果等。
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import uuid

# 枚举定义

class MergeStrategy(str, Enum):
    """合并策略枚举"""
    TIMELINE = "timeline"         # 时间线整合
    TOPIC = "topic"              # 主题归并  
    HIERARCHY = "hierarchy"       # 层次整合
    LOGIC = "logic"              # 逻辑关系
    SUPPLEMENT = "supplement"     # 补充扩展

class MergeMode(str, Enum):
    """合并模式枚举"""
    COMPREHENSIVE = "comprehensive"  # 全面合并
    SELECTIVE = "selective"         # 选择性合并
    SUMMARY = "summary"             # 摘要合并
    EXPANSION = "expansion"         # 扩展合并

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"           # 待处理
    ANALYZING = "analyzing"       # 分析中
    MERGING = "merging"          # 合并中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"            # 失败

class RelationshipType(str, Enum):
    """内容关系类型枚举"""
    SIMILAR = "similar"           # 相似
    SUPPLEMENT = "supplement"     # 补充
    CONTRADICTION = "contradiction"  # 矛盾
    SEQUENCE = "sequence"         # 序列
    CAUSAL = "causal"            # 因果

class QualityDimension(str, Enum):
    """质量维度枚举"""
    CONSISTENCY = "consistency"   # 一致性
    COMPLETENESS = "completeness" # 完整性  
    FLUENCY = "fluency"          # 流畅性
    ORIGINALITY = "originality"   # 原创性
    FACTUAL_ACCURACY = "factual_accuracy"  # 事实准确性

# 基础数据模型

class ContentItem(BaseModel):
    """内容项数据模型"""
    id: str = Field(description="内容ID")
    title: str = Field(description="内容标题")
    content: str = Field(description="内容文本")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="内容元数据")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="内容分析结果")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('内容不能为空')
        if len(v) > 100000:
            raise ValueError('内容长度不能超过100000字符')
        return v

class TemporalInfo(BaseModel):
    """时间信息模型"""
    time_period: Optional[str] = Field(description="时间段")
    specific_dates: List[str] = Field(default_factory=list, description="具体日期")
    temporal_order: Optional[int] = Field(description="时间顺序")
    time_score: Optional[float] = Field(description="时间分数")

class Topic(BaseModel):
    """主题模型"""
    topic: str = Field(description="主题名称")
    relevance: float = Field(ge=0.0, le=1.0, description="相关性分数")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    importance: Optional[float] = Field(ge=0.0, le=1.0, description="重要性")

class Entity(BaseModel):
    """实体模型"""
    name: str = Field(description="实体名称")
    type: str = Field(description="实体类型")
    importance: float = Field(ge=0.0, le=1.0, description="重要性")
    mentions: List[int] = Field(default_factory=list, description="提及位置")
    confidence: Optional[float] = Field(ge=0.0, le=1.0, description="置信度")

class ContentAnalysis(BaseModel):
    """内容分析结果模型"""
    topics: List[Topic] = Field(default_factory=list, description="主题列表")
    entities: List[Entity] = Field(default_factory=list, description="实体列表")
    temporal_info: Optional[TemporalInfo] = Field(description="时间信息")
    key_points: List[str] = Field(default_factory=list, description="关键点")
    sentiment_score: Optional[float] = Field(ge=-1.0, le=1.0, description="情感分数")
    complexity_score: Optional[float] = Field(ge=0.0, le=10.0, description="复杂度分数")

class ContentRelationship(BaseModel):
    """内容关系模型"""
    source_content_id: str = Field(description="源内容ID")
    target_content_id: str = Field(description="目标内容ID")
    relationship_type: RelationshipType = Field(description="关系类型")
    similarity_score: float = Field(ge=0.0, le=1.0, description="相似度分数")
    relationship_strength: float = Field(ge=0.0, le=1.0, description="关系强度")
    relationship_data: Dict[str, Any] = Field(default_factory=dict, description="关系详细数据")

# 请求模型

class MergeConfig(BaseModel):
    """合并配置模型"""
    target_length: Optional[int] = Field(gt=0, description="目标长度")
    target_style: Optional[str] = Field(description="目标风格")
    preserve_entities: bool = Field(default=True, description="保留实体")
    merge_overlaps: bool = Field(default=True, description="合并重叠内容")
    quality_threshold: float = Field(default=85.0, ge=0.0, le=100.0, description="质量阈值")
    enable_creativity: bool = Field(default=False, description="启用创意模式")
    custom_instructions: Optional[str] = Field(description="自定义指令")

class MergeRequest(BaseModel):
    """合并请求模型"""
    source_content_ids: List[str] = Field(min_items=2, max_items=20, description="源内容ID列表")
    strategy: MergeStrategy = Field(description="合并策略")
    mode: MergeMode = Field(description="合并模式")
    config: MergeConfig = Field(default_factory=MergeConfig, description="合并配置")
    title: Optional[str] = Field(description="合并结果标题")
    user_id: Optional[str] = Field(description="用户ID")
    
    @validator('source_content_ids')
    def validate_source_content_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('源内容ID不能重复')
        return v

class BatchMergeRequest(BaseModel):
    """批量合并请求模型"""
    content_groups: List[List[str]] = Field(min_items=1, description="内容组列表")
    merge_config: MergeConfig = Field(description="合并配置")
    job_name: Optional[str] = Field(description="任务名称")
    user_id: Optional[str] = Field(description="用户ID")
    
    @validator('content_groups')
    def validate_content_groups(cls, v):
        for group in v:
            if len(group) < 2:
                raise ValueError('每个内容组至少需要2个内容')
            if len(group) > 10:
                raise ValueError('每个内容组最多包含10个内容')
        return v

class RelationshipAnalysisRequest(BaseModel):
    """关系分析请求模型"""
    content_ids: List[str] = Field(min_items=2, max_items=20, description="内容ID列表")
    analysis_types: List[str] = Field(
        default=["similarity", "temporal", "topic", "entity"], 
        description="分析类型"
    )

class MergePreviewRequest(BaseModel):
    """合并预览请求模型"""
    content_ids: List[str] = Field(min_items=2, description="内容ID列表")
    strategy: MergeStrategy = Field(description="合并策略")
    preview_sections: int = Field(default=3, ge=1, le=10, description="预览章节数")

# 响应模型

class MergeSection(BaseModel):
    """合并章节模型"""
    title: str = Field(description="章节标题")
    content: str = Field(description="章节内容")
    source_contents: List[str] = Field(description="源内容ID列表")
    merge_type: str = Field(description="合并类型")
    word_count: int = Field(ge=0, description="字数")

class MergeTransition(BaseModel):
    """合并过渡模型"""
    from_section: int = Field(ge=0, description="起始章节索引")
    to_section: int = Field(ge=0, description="目标章节索引")
    transition_text: str = Field(description="过渡文本")
    transition_type: str = Field(description="过渡类型")

class MergeStructure(BaseModel):
    """合并结构模型"""
    sections: List[MergeSection] = Field(description="章节列表")
    transitions: List[MergeTransition] = Field(default_factory=list, description="过渡列表")
    total_word_count: int = Field(ge=0, description="总字数")

class QualityMetrics(BaseModel):
    """质量指标模型"""
    overall_score: float = Field(ge=0.0, le=100.0, description="总体分数")
    consistency_score: float = Field(ge=0.0, le=100.0, description="一致性分数")
    completeness_score: float = Field(ge=0.0, le=100.0, description="完整性分数")
    fluency_score: float = Field(ge=0.0, le=100.0, description="流畅性分数")
    originality_score: float = Field(ge=0.0, le=100.0, description="原创性分数")
    factual_accuracy: float = Field(ge=0.0, le=100.0, description="事实准确性")

class TokenUsage(BaseModel):
    """令牌使用情况模型"""
    prompt_tokens: int = Field(ge=0, description="提示令牌数")
    completion_tokens: int = Field(ge=0, description="完成令牌数")
    total_tokens: int = Field(ge=0, description="总令牌数")

class MergeMetadata(BaseModel):
    """合并元数据模型"""
    strategy_used: str = Field(description="使用的策略")
    source_count: int = Field(ge=2, description="源内容数量")
    processing_time_ms: int = Field(ge=0, description="处理时间(毫秒)")
    ai_model_used: Optional[str] = Field(description="使用的AI模型")
    token_usage: Optional[TokenUsage] = Field(description="令牌使用情况")
    merge_plan: Optional[Dict[str, Any]] = Field(description="合并计划")

class MergeResult(BaseModel):
    """合并结果模型"""
    title: str = Field(description="合并标题")
    content: str = Field(description="合并内容")
    structure: MergeStructure = Field(description="合并结构")
    quality_metrics: QualityMetrics = Field(description="质量指标")
    merge_metadata: MergeMetadata = Field(description="合并元数据")

class ProcessStep(BaseModel):
    """处理步骤模型"""
    step: str = Field(description="步骤名称")
    completed_at: datetime = Field(description="完成时间")
    duration_ms: int = Field(ge=0, description="持续时间(毫秒)")
    success: bool = Field(description="是否成功")
    details: Optional[Dict[str, Any]] = Field(description="步骤详情")

class TaskProgress(BaseModel):
    """任务进度模型"""
    task_id: str = Field(description="任务ID")
    status: TaskStatus = Field(description="任务状态")
    progress_percentage: int = Field(ge=0, le=100, description="进度百分比")
    current_step: Optional[str] = Field(description="当前步骤")
    estimated_remaining_time: Optional[str] = Field(description="预计剩余时间")
    steps_completed: List[ProcessStep] = Field(default_factory=list, description="已完成步骤")
    error_message: Optional[str] = Field(description="错误消息")

class SimilarityMatrix(BaseModel):
    """相似度矩阵模型"""
    content_ids: List[str] = Field(description="内容ID列表")
    matrix: List[List[float]] = Field(description="相似度矩阵")
    
    @validator('matrix')
    def validate_matrix(cls, v, values):
        if 'content_ids' in values:
            n = len(values['content_ids'])
            if len(v) != n:
                raise ValueError('矩阵行数必须等于内容数量')
            for row in v:
                if len(row) != n:
                    raise ValueError('矩阵列数必须等于内容数量')
        return v

class TopicOverlap(BaseModel):
    """主题重叠模型"""
    topic: str = Field(description="主题名称")
    contents: List[str] = Field(description="涉及内容ID")
    overlap_score: float = Field(ge=0.0, le=1.0, description="重叠分数")

class EntityConnection(BaseModel):
    """实体连接模型"""
    entity: str = Field(description="实体名称")
    contents: List[str] = Field(description="涉及内容ID")
    connection_strength: float = Field(ge=0.0, le=1.0, description="连接强度")

class RelationshipAnalysis(BaseModel):
    """关系分析结果模型"""
    similarity_matrix: SimilarityMatrix = Field(description="相似度矩阵")
    temporal_order: List[Dict[str, Any]] = Field(description="时间顺序")
    topic_overlaps: List[TopicOverlap] = Field(description="主题重叠")
    entity_connections: List[EntityConnection] = Field(description="实体连接")

class MergeRecommendation(BaseModel):
    """合并推荐模型"""
    strategy: MergeStrategy = Field(description="推荐策略")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    reason: str = Field(description="推荐理由")

class PreviewSection(BaseModel):
    """预览章节模型"""
    title: str = Field(description="章节标题")
    preview_content: str = Field(description="预览内容")
    estimated_length: int = Field(ge=0, description="预估长度")

class MergePreview(BaseModel):
    """合并预览模型"""
    title: str = Field(description="预期标题")
    sections: List[PreviewSection] = Field(description="预览章节")
    estimated_quality: float = Field(ge=0.0, le=100.0, description="预估质量")
    estimated_processing_time: str = Field(description="预估处理时间")

# 响应包装模型

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(description="是否成功")
    message: Optional[str] = Field(description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")

class MergeTaskResponse(BaseResponse):
    """合并任务响应模型"""
    data: Optional[Dict[str, Any]] = Field(description="任务数据")

class MergeResultResponse(BaseResponse):
    """合并结果响应模型"""
    data: Optional[MergeResult] = Field(description="合并结果")

class RelationshipAnalysisResponse(BaseResponse):
    """关系分析响应模型"""
    data: Optional[Dict[str, Any]] = Field(description="分析结果")

class MergePreviewResponse(BaseResponse):
    """合并预览响应模型"""
    data: Optional[Dict[str, Any]] = Field(description="预览数据")

class TaskProgressResponse(BaseResponse):
    """任务进度响应模型"""
    data: Optional[TaskProgress] = Field(description="进度数据")

class BatchJobResponse(BaseResponse):
    """批量任务响应模型"""
    data: Optional[Dict[str, Any]] = Field(description="批量任务数据")

# 错误模型

class MergeError(Exception):
    """合并错误基类"""
    
    def __init__(self, message: str, error_code: str = "MERGE_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ContentAnalysisError(MergeError):
    """内容分析错误"""
    
    def __init__(self, message: str):
        super().__init__(message, "CONTENT_ANALYSIS_ERROR")

class MergeStrategyError(MergeError):
    """合并策略错误"""
    
    def __init__(self, message: str):
        super().__init__(message, "MERGE_STRATEGY_ERROR")

class QualityAssessmentError(MergeError):
    """质量评估错误"""
    
    def __init__(self, message: str):
        super().__init__(message, "QUALITY_ASSESSMENT_ERROR")

class AIServiceError(MergeError):
    """AI服务错误"""
    
    def __init__(self, message: str):
        super().__init__(message, "AI_SERVICE_ERROR")