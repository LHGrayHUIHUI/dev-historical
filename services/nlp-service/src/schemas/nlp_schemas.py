"""
NLP服务数据模型Schema
Pydantic模型定义，用于API请求和响应数据验证
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from uuid import UUID
from datetime import datetime


# ============ 枚举类型 ============

class ProcessingType(str, Enum):
    """NLP处理类型"""
    SEGMENTATION = "segmentation"
    POS_TAGGING = "pos_tagging"
    NER = "ner"
    SENTIMENT = "sentiment"
    KEYWORDS = "keywords"
    SUMMARY = "summary"
    SIMILARITY = "similarity"
    BATCH = "batch"


class ProcessingStatus(str, Enum):
    """处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Language(str, Enum):
    """支持的语言"""
    CHINESE = "zh"
    ENGLISH = "en"
    CLASSICAL_CHINESE = "zh-classical"


class NLPEngine(str, Enum):
    """NLP引擎类型"""
    SPACY = "spacy"
    JIEBA = "jieba"
    HANLP = "hanlp"
    TRANSFORMERS = "transformers"


class SentimentLabel(str, Enum):
    """情感标签"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# ============ 基础响应模型 ============

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = False
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


# ============ 请求模型 ============

class TextProcessRequest(BaseModel):
    """文本处理请求"""
    text: str = Field(..., max_length=1000000, description="待处理文本")
    processing_type: ProcessingType = Field(..., description="处理类型")
    language: Language = Field(default=Language.CHINESE, description="文本语言")
    engine: Optional[NLPEngine] = Field(default=None, description="指定NLP引擎")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="处理配置")
    async_mode: bool = Field(default=False, description="是否异步处理")
    dataset_id: Optional[str] = Field(default=None, description="关联数据集ID")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('文本内容不能为空')
        return v


class BatchProcessRequest(BaseModel):
    """批量文本处理请求"""
    texts: List[str] = Field(..., max_items=50, description="文本列表")
    processing_type: ProcessingType = Field(..., description="处理类型")
    language: Language = Field(default=Language.CHINESE, description="文本语言")
    engine: Optional[NLPEngine] = Field(default=None, description="指定NLP引擎")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="处理配置")
    dataset_id: Optional[str] = Field(default=None, description="关联数据集ID")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        if not v:
            raise ValueError('文本列表不能为空')
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'第{i+1}个文本内容不能为空')
        return v


class SegmentationConfig(BaseModel):
    """分词配置"""
    method: str = Field(default="jieba", description="分词方法")
    enable_parallel: bool = Field(default=True, description="启用并行处理")
    custom_dict: Optional[str] = Field(default=None, description="自定义词典")
    hmm: bool = Field(default=True, description="启用HMM模式")


class KeywordConfig(BaseModel):
    """关键词提取配置"""
    method: str = Field(default="textrank", description="提取方法")
    top_k: int = Field(default=20, ge=1, le=100, description="关键词数量")
    min_word_len: int = Field(default=2, ge=1, description="最小词长")


class SummaryConfig(BaseModel):
    """摘要生成配置"""
    method: str = Field(default="extractive", description="摘要方法")
    max_sentences: int = Field(default=5, ge=1, le=20, description="最大句子数")
    compression_ratio: float = Field(default=0.3, ge=0.1, le=0.9, description="压缩比例")


# ============ 响应数据模型 ============

class WordInfo(BaseModel):
    """词汇信息"""
    text: str = Field(..., description="词汇文本")
    start: int = Field(..., description="起始位置")
    end: int = Field(..., description="结束位置")
    pos: Optional[str] = Field(default=None, description="词性")
    frequency: Optional[int] = Field(default=None, description="词频")
    score: Optional[float] = Field(default=None, description="分数")


class EntityInfo(BaseModel):
    """实体信息"""
    text: str = Field(..., description="实体文本")
    label: str = Field(..., description="实体标签")
    start: int = Field(..., description="起始位置")
    end: int = Field(..., description="结束位置")
    confidence: float = Field(..., ge=0, le=1, description="置信度")


class KeywordInfo(BaseModel):
    """关键词信息"""
    word: str = Field(..., description="关键词")
    score: float = Field(..., ge=0, le=1, description="重要性分数")
    frequency: int = Field(..., ge=0, description="出现频次")


class SentimentInfo(BaseModel):
    """情感信息"""
    label: SentimentLabel = Field(..., description="情感标签")
    score: float = Field(..., ge=-1, le=1, description="情感分数")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    emotions: Optional[Dict[str, float]] = Field(default=None, description="详细情感")


# ============ 处理结果模型 ============

class SegmentationResult(BaseModel):
    """分词结果"""
    original_text: str = Field(..., description="原始文本")
    segmented_text: str = Field(..., description="分词后文本")
    words: List[WordInfo] = Field(..., description="词汇列表")
    word_count: int = Field(..., description="总词数")
    unique_word_count: int = Field(..., description="唯一词数")
    method: str = Field(..., description="分词方法")


class PosTaggingResult(BaseModel):
    """词性标注结果"""
    words_with_pos: List[WordInfo] = Field(..., description="带词性的词汇")
    pos_distribution: Dict[str, int] = Field(..., description="词性分布")
    method: str = Field(..., description="标注方法")


class NERResult(BaseModel):
    """命名实体识别结果"""
    entities: List[EntityInfo] = Field(..., description="实体列表")
    entity_types: Dict[str, int] = Field(..., description="实体类型统计")
    model: str = Field(..., description="使用的模型")


class SentimentResult(BaseModel):
    """情感分析结果"""
    sentiment: SentimentInfo = Field(..., description="情感信息")
    model: str = Field(..., description="使用的模型")


class KeywordResult(BaseModel):
    """关键词提取结果"""
    keywords: List[KeywordInfo] = Field(..., description="关键词列表")
    method: str = Field(..., description="提取方法")


class SummaryResult(BaseModel):
    """文本摘要结果"""
    original_length: int = Field(..., description="原文长度")
    summary_text: str = Field(..., description="摘要内容")
    summary_length: int = Field(..., description="摘要长度")
    compression_ratio: float = Field(..., description="压缩比")
    method: str = Field(..., description="摘要方法")


class TextSimilarityResult(BaseModel):
    """文本相似度结果"""
    text1: str = Field(..., description="文本1")
    text2: str = Field(..., description="文本2")
    similarity_score: float = Field(..., ge=0, le=1, description="相似度分数")
    method: str = Field(..., description="计算方法")


# ============ 任务模型 ============

class NLPTask(BaseModel):
    """NLP任务信息"""
    task_id: str = Field(..., description="任务ID")
    dataset_id: Optional[str] = Field(default=None, description="数据集ID")
    processing_type: ProcessingType = Field(..., description="处理类型")
    processing_status: ProcessingStatus = Field(..., description="处理状态")
    language: Language = Field(..., description="语言")
    engine: str = Field(..., description="使用的引擎")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置")
    text_length: int = Field(..., description="文本长度")
    processing_time: Optional[float] = Field(default=None, description="处理时间（秒）")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")


class NLPTaskResponse(BaseResponse):
    """NLP任务响应"""
    task: NLPTask = Field(..., description="任务信息")


class NLPTaskListResponse(BaseResponse):
    """NLP任务列表响应"""
    tasks: List[NLPTask] = Field(..., description="任务列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="页码")
    size: int = Field(..., description="页大小")


# ============ 处理响应模型 ============

class ProcessingResponse(BaseResponse):
    """处理响应基类"""
    task_id: str = Field(..., description="任务ID")
    processing_type: ProcessingType = Field(..., description="处理类型")
    processing_time: float = Field(..., description="处理时间")


class SegmentationResponse(ProcessingResponse):
    """分词响应"""
    result: SegmentationResult = Field(..., description="分词结果")


class PosTaggingResponse(ProcessingResponse):
    """词性标注响应"""
    result: PosTaggingResult = Field(..., description="词性标注结果")


class NERResponse(ProcessingResponse):
    """命名实体识别响应"""
    result: NERResult = Field(..., description="NER结果")


class SentimentResponse(ProcessingResponse):
    """情感分析响应"""
    result: SentimentResult = Field(..., description="情感分析结果")


class KeywordResponse(ProcessingResponse):
    """关键词提取响应"""
    result: KeywordResult = Field(..., description="关键词结果")


class SummaryResponse(ProcessingResponse):
    """文本摘要响应"""
    result: SummaryResult = Field(..., description="摘要结果")


class SimilarityResponse(ProcessingResponse):
    """相似度计算响应"""
    result: TextSimilarityResult = Field(..., description="相似度结果")


class BatchProcessingResponse(ProcessingResponse):
    """批量处理响应"""
    results: List[Dict[str, Any]] = Field(..., description="批量结果")
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")


# ============ 引擎信息模型 ============

class NLPEngineInfo(BaseModel):
    """NLP引擎信息"""
    name: str = Field(..., description="引擎名称")
    version: str = Field(..., description="版本号")
    supported_languages: List[str] = Field(..., description="支持语言")
    supported_functions: List[str] = Field(..., description="支持功能")
    description: str = Field(..., description="描述")


class NLPEnginesResponse(BaseResponse):
    """NLP引擎列表响应"""
    engines: List[NLPEngineInfo] = Field(..., description="引擎列表")


# ============ 统计信息模型 ============

class NLPStatistics(BaseModel):
    """NLP处理统计"""
    total_tasks: int = Field(..., description="总任务数")
    completed_tasks: int = Field(..., description="完成任务数")
    failed_tasks: int = Field(..., description="失败任务数")
    processing_tasks: int = Field(..., description="处理中任务数")
    avg_processing_time: float = Field(..., description="平均处理时间")
    processing_type_stats: Dict[str, int] = Field(..., description="处理类型统计")
    language_stats: Dict[str, int] = Field(..., description="语言统计")
    engine_stats: Dict[str, int] = Field(..., description="引擎使用统计")


class NLPStatisticsResponse(BaseResponse):
    """NLP统计响应"""
    statistics: NLPStatistics = Field(..., description="统计信息")


# ============ 健康检查模型 ============

class ServiceInfo(BaseModel):
    """服务信息"""
    service_name: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
    status: str = Field(..., description="状态")
    uptime: float = Field(..., description="运行时间（秒）")
    available_engines: List[str] = Field(..., description="可用引擎")
    processing_capabilities: List[str] = Field(..., description="处理能力")


class HealthCheckResponse(BaseResponse):
    """健康检查响应"""
    service_info: ServiceInfo = Field(..., description="服务信息")
    dependencies: Dict[str, str] = Field(..., description="依赖服务状态")