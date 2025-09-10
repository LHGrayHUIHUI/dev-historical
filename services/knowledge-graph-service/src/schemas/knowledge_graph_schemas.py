"""
知识图谱构建服务数据模型
定义请求响应模型和业务实体模型
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


# 枚举类型定义
class EntityType(str, Enum):
    """实体类型枚举"""
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    EVENT = "EVENT"
    TIME = "TIME"
    CONCEPT = "CONCEPT"
    OBJECT = "OBJECT"
    WORK = "WORK"


class RelationType(str, Enum):
    """关系类型枚举"""
    BORN_IN = "出生于"
    DIED_IN = "死于"
    WORKED_AT = "任职于"
    LOCATED_IN = "位于"
    FOUNDED = "创建"
    INFLUENCED = "影响"
    PARTICIPATED_IN = "参与"
    BELONGS_TO = "属于"
    RULED = "统治"
    INHERITED = "继承"
    LEARNED_FROM = "师从"
    CONTAINS = "包含"


class ExtractionMethod(str, Enum):
    """抽取方法枚举"""
    SPACY_NER = "spacy_ner"
    BERT_NER = "bert_ner"
    JIEBA_NER = "jieba_ner"
    PATTERN_BASED = "pattern_based"
    DEPENDENCY_PARSING = "dependency_parsing"
    RULE_BASED = "rule_based"


class GraphTaskStatus(str, Enum):
    """图谱任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Language(str, Enum):
    """语言类型枚举"""
    CHINESE = "zh"
    ENGLISH = "en"


# 基础响应模型
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


# 知识图谱项目模型
class GraphProject(BaseModel):
    """知识图谱项目模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., max_length=200, description="项目名称")
    description: Optional[str] = Field(None, description="项目描述")
    domain: str = Field(..., max_length=100, description="领域")
    language: Language = Field(default=Language.CHINESE, description="语言")
    entity_types: List[EntityType] = Field(default_factory=list, description="实体类型列表")
    relation_types: List[RelationType] = Field(default_factory=list, description="关系类型列表")
    created_by: Optional[str] = Field(None, description="创建者ID")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class CreateGraphProjectRequest(BaseModel):
    """创建图谱项目请求"""
    name: str = Field(..., max_length=200, description="项目名称")
    description: Optional[str] = Field(None, description="项目描述")
    domain: str = Field(..., max_length=100, description="领域")
    language: Language = Field(default=Language.CHINESE, description="语言")
    entity_types: List[EntityType] = Field(default_factory=list, description="实体类型列表")
    relation_types: List[RelationType] = Field(default_factory=list, description="关系类型列表")
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('项目名称不能为空')
        return v.strip()


class CreateGraphProjectResponse(BaseResponse):
    """创建图谱项目响应"""
    project_id: str = Field(..., description="项目ID")
    project: GraphProject = Field(..., description="项目信息")


# 实体模型
class Entity(BaseModel):
    """实体模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., max_length=500, description="实体名称")
    entity_type: EntityType = Field(..., description="实体类型")
    aliases: List[str] = Field(default_factory=list, description="别名列表")
    description: Optional[str] = Field(None, description="实体描述")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    confidence_score: float = Field(ge=0.0, le=1.0, description="置信度")
    mention_count: int = Field(default=1, ge=1, description="提及次数")
    source_documents: List[str] = Field(default_factory=list, description="来源文档ID列表")
    
    class Config:
        use_enum_values = True


class ExtractedEntity(BaseModel):
    """抽取的实体模型"""
    name: str = Field(..., description="实体名称")
    entity_type: EntityType = Field(..., description="实体类型")
    start_pos: int = Field(ge=0, description="开始位置")
    end_pos: int = Field(ge=0, description="结束位置")
    confidence_score: float = Field(ge=0.0, le=1.0, description="置信度")
    context: str = Field(default="", description="上下文")
    
    @validator('end_pos')
    def end_pos_must_be_greater_than_start_pos(cls, v, values):
        if 'start_pos' in values and v <= values['start_pos']:
            raise ValueError('结束位置必须大于开始位置')
        return v


# 关系模型
class Relation(BaseModel):
    """关系模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject_entity_id: str = Field(..., description="主体实体ID")
    predicate: RelationType = Field(..., description="关系类型")
    object_entity_id: str = Field(..., description="客体实体ID")
    confidence_score: float = Field(ge=0.0, le=1.0, description="置信度")
    context: str = Field(default="", description="关系上下文")
    source_sentence: str = Field(default="", description="来源句子")
    source_document_id: Optional[str] = Field(None, description="来源文档ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")
    
    class Config:
        use_enum_values = True


class ExtractedRelation(BaseModel):
    """抽取的关系模型"""
    subject_entity: ExtractedEntity = Field(..., description="主体实体")
    predicate: RelationType = Field(..., description="关系类型")
    object_entity: ExtractedEntity = Field(..., description="客体实体")
    confidence_score: float = Field(ge=0.0, le=1.0, description="置信度")
    context: str = Field(default="", description="关系上下文")
    source_sentence: str = Field(default="", description="来源句子")


# 实体抽取请求响应
class EntityExtractionRequest(BaseModel):
    """实体抽取请求"""
    project_id: str = Field(..., description="项目ID")
    text: str = Field(..., max_length=10000, description="待抽取文本")
    document_id: Optional[str] = Field(None, description="文档ID")
    method: ExtractionMethod = Field(default=ExtractionMethod.SPACY_NER, description="抽取方法")
    config: Dict[str, Any] = Field(default_factory=dict, description="抽取配置")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('待抽取文本不能为空')
        return v.strip()


class EntityExtractionResponse(BaseResponse):
    """实体抽取响应"""
    task_id: str = Field(..., description="任务ID")
    entities_found: int = Field(ge=0, description="发现的实体数量")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="抽取的实体列表")
    processing_time: float = Field(ge=0.0, description="处理时间（秒）")


# 关系抽取请求响应
class RelationExtractionRequest(BaseModel):
    """关系抽取请求"""
    project_id: str = Field(..., description="项目ID")
    text: str = Field(..., max_length=10000, description="待抽取文本")
    document_id: Optional[str] = Field(None, description="文档ID")
    method: ExtractionMethod = Field(default=ExtractionMethod.PATTERN_BASED, description="抽取方法")
    config: Dict[str, Any] = Field(default_factory=dict, description="抽取配置")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('待抽取文本不能为空')
        return v.strip()


class RelationExtractionResponse(BaseResponse):
    """关系抽取响应"""
    task_id: str = Field(..., description="任务ID")
    relations_found: int = Field(ge=0, description="发现的关系数量")
    relations: List[ExtractedRelation] = Field(default_factory=list, description="抽取的关系列表")
    processing_time: float = Field(ge=0.0, description="处理时间（秒）")


# 批量处理请求响应
class BatchExtractionRequest(BaseModel):
    """批量抽取请求"""
    project_id: str = Field(..., description="项目ID")
    documents: List[Dict[str, str]] = Field(..., description="文档列表，包含id和text")
    entity_extraction: bool = Field(default=True, description="是否进行实体抽取")
    relation_extraction: bool = Field(default=True, description="是否进行关系抽取")
    entity_method: ExtractionMethod = Field(default=ExtractionMethod.SPACY_NER, description="实体抽取方法")
    relation_method: ExtractionMethod = Field(default=ExtractionMethod.PATTERN_BASED, description="关系抽取方法")
    config: Dict[str, Any] = Field(default_factory=dict, description="抽取配置")
    
    @validator('documents')
    def documents_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('文档列表不能为空')
        for doc in v:
            if 'id' not in doc or 'text' not in doc:
                raise ValueError('每个文档必须包含id和text字段')
        return v


class BatchExtractionResponse(BaseResponse):
    """批量抽取响应"""
    batch_id: str = Field(..., description="批次ID")
    total_documents: int = Field(ge=0, description="文档总数")
    processed_documents: int = Field(ge=0, description="已处理文档数")
    failed_documents: int = Field(ge=0, description="失败文档数")
    total_entities: int = Field(ge=0, description="总实体数")
    total_relations: int = Field(ge=0, description="总关系数")
    processing_time: float = Field(ge=0.0, description="处理时间（秒）")


# 图谱构建请求响应
class GraphConstructionRequest(BaseModel):
    """图谱构建请求"""
    project_id: str = Field(..., description="项目ID")
    include_entities: bool = Field(default=True, description="是否包含实体")
    include_relations: bool = Field(default=True, description="是否包含关系")
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="最小置信度")
    max_nodes: int = Field(default=1000, ge=1, le=10000, description="最大节点数")
    config: Dict[str, Any] = Field(default_factory=dict, description="构建配置")


class GraphConstructionResponse(BaseResponse):
    """图谱构建响应"""
    task_id: str = Field(..., description="任务ID")
    nodes_count: int = Field(ge=0, description="节点数量")
    edges_count: int = Field(ge=0, description="边数量")
    processing_time: float = Field(ge=0.0, description="处理时间（秒）")


# 图谱查询请求响应
class GraphQueryRequest(BaseModel):
    """图谱查询请求"""
    project_id: str = Field(..., description="项目ID")
    query_type: str = Field(..., description="查询类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    limit: int = Field(default=100, ge=1, le=1000, description="结果限制")
    offset: int = Field(default=0, ge=0, description="结果偏移")


class GraphNode(BaseModel):
    """图节点模型"""
    id: str = Field(..., description="节点ID")
    label: str = Field(..., description="节点标签")
    entity_type: str = Field(..., description="实体类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="节点属性")


class GraphEdge(BaseModel):
    """图边模型"""
    id: str = Field(..., description="边ID")
    source: str = Field(..., description="源节点ID")
    target: str = Field(..., description="目标节点ID")
    relation_type: str = Field(..., description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="边属性")


class GraphQueryResponse(BaseResponse):
    """图谱查询响应"""
    nodes: List[GraphNode] = Field(default_factory=list, description="节点列表")
    edges: List[GraphEdge] = Field(default_factory=list, description="边列表")
    total_count: int = Field(ge=0, description="总数量")
    query_time: float = Field(ge=0.0, description="查询时间（秒）")


# 图谱分析请求响应
class GraphAnalysisRequest(BaseModel):
    """图谱分析请求"""
    project_id: str = Field(..., description="项目ID")
    analysis_type: str = Field(..., description="分析类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="分析参数")


class GraphMetrics(BaseModel):
    """图谱指标模型"""
    node_count: int = Field(ge=0, description="节点数量")
    edge_count: int = Field(ge=0, description="边数量")
    density: float = Field(ge=0.0, le=1.0, description="密度")
    average_degree: float = Field(ge=0.0, description="平均度数")
    clustering_coefficient: float = Field(ge=0.0, le=1.0, description="聚类系数")
    connected_components: int = Field(ge=0, description="连通分量数")


class GraphAnalysisResponse(BaseResponse):
    """图谱分析响应"""
    analysis_type: str = Field(..., description="分析类型")
    metrics: GraphMetrics = Field(..., description="图谱指标")
    results: Dict[str, Any] = Field(default_factory=dict, description="分析结果")
    processing_time: float = Field(ge=0.0, description="处理时间（秒）")


# 任务状态模型
class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str = Field(..., description="任务ID")
    status: GraphTaskStatus = Field(..., description="任务状态")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="进度百分比")
    message: str = Field(default="", description="状态消息")
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误消息")


class TaskStatusResponse(BaseResponse):
    """任务状态响应"""
    task: TaskStatus = Field(..., description="任务状态")


# 分页响应模型
class PaginatedResponse(BaseResponse):
    """分页响应模型"""
    page: int = Field(ge=1, description="当前页码")
    page_size: int = Field(ge=1, description="每页大小")
    total_count: int = Field(ge=0, description="总数量")
    total_pages: int = Field(ge=0, description="总页数")


class EntityListResponse(PaginatedResponse):
    """实体列表响应"""
    entities: List[Entity] = Field(default_factory=list, description="实体列表")


class RelationListResponse(PaginatedResponse):
    """关系列表响应"""
    relations: List[Relation] = Field(default_factory=list, description="关系列表")


# 概念挖掘模型
class ConceptMiningRequest(BaseModel):
    """概念挖掘请求"""
    project_id: str = Field(..., description="项目ID")
    documents: List[str] = Field(..., description="文档ID列表")
    num_topics: int = Field(default=10, ge=1, le=100, description="主题数量")
    method: str = Field(default="lda", description="挖掘方法")
    config: Dict[str, Any] = Field(default_factory=dict, description="挖掘配置")


class ConceptTopic(BaseModel):
    """概念主题模型"""
    id: str = Field(..., description="主题ID")
    name: str = Field(..., description="主题名称")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    weight: float = Field(ge=0.0, le=1.0, description="主题权重")
    documents: List[str] = Field(default_factory=list, description="相关文档")


class ConceptMiningResponse(BaseResponse):
    """概念挖掘响应"""
    task_id: str = Field(..., description="任务ID")
    topics: List[ConceptTopic] = Field(default_factory=list, description="主题列表")
    processing_time: float = Field(ge=0.0, description="处理时间（秒）")


# 统计信息模型
class GraphStatistics(BaseModel):
    """图谱统计信息"""
    project_count: int = Field(ge=0, description="项目数量")
    entity_count: int = Field(ge=0, description="实体数量")
    relation_count: int = Field(ge=0, description="关系数量")
    document_count: int = Field(ge=0, description="文档数量")
    
    entity_type_distribution: Dict[str, int] = Field(default_factory=dict, description="实体类型分布")
    relation_type_distribution: Dict[str, int] = Field(default_factory=dict, description="关系类型分布")
    
    average_confidence_score: float = Field(ge=0.0, le=1.0, description="平均置信度")
    processing_statistics: Dict[str, Any] = Field(default_factory=dict, description="处理统计")


class GraphStatisticsResponse(BaseResponse):
    """图谱统计响应"""
    statistics: GraphStatistics = Field(..., description="统计信息")
    generated_at: datetime = Field(default_factory=datetime.now, description="生成时间")