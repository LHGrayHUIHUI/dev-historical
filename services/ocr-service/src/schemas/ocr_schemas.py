"""
OCR相关数据模型schemas

定义OCR识别相关的请求/响应模型，包括识别请求、
结果响应、任务查询等数据结构。

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, HttpUrl, validator
from datetime import datetime
from enum import Enum

from .common_schemas import BaseResponse, PaginationQuery


class OCREngineEnum(str, Enum):
    """OCR引擎类型枚举"""
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"


class TaskStatusEnum(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OCRRequest(BaseModel):
    """
    OCR识别请求模型
    
    用于单图像OCR识别的请求参数。
    """
    engine: OCREngineEnum = Field(
        OCREngineEnum.PADDLEOCR,
        description="OCR引擎类型"
    )
    confidence_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )
    language_codes: str = Field(
        "zh,en",
        description="语言代码，逗号分隔"
    )
    async_mode: bool = Field(
        False,
        description="是否使用异步模式"
    )
    preprocessing: bool = Field(
        True,
        description="是否启用图像预处理"
    )
    postprocessing: bool = Field(
        True,
        description="是否启用文本后处理"
    )
    dataset_id: Optional[str] = Field(
        None,
        description="关联的数据集ID"
    )
    
    @validator('language_codes')
    def validate_language_codes(cls, v):
        """验证语言代码格式"""
        if not v or not isinstance(v, str):
            raise ValueError("语言代码不能为空")
        
        codes = [code.strip() for code in v.split(',')]
        if not codes or any(not code for code in codes):
            raise ValueError("语言代码格式无效")
        
        return v


class OCRBatchRequest(BaseModel):
    """
    批量OCR识别请求模型
    
    用于批量图像OCR识别的请求参数。
    """
    engine: OCREngineEnum = Field(
        OCREngineEnum.PADDLEOCR,
        description="OCR引擎类型"
    )
    confidence_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )
    language_codes: str = Field(
        "zh,en",
        description="语言代码"
    )
    preprocessing: bool = Field(
        True,
        description="是否启用图像预处理"
    )
    postprocessing: bool = Field(
        True,
        description="是否启用文本后处理"
    )
    max_concurrent: int = Field(
        4,
        ge=1,
        le=10,
        description="最大并发处理数"
    )
    dataset_id: Optional[str] = Field(
        None,
        description="关联的数据集ID"
    )


class OCRUrlRequest(BaseModel):
    """
    URL图像OCR识别请求模型
    
    用于从URL获取图像进行OCR识别。
    """
    image_url: HttpUrl = Field(
        ...,
        description="图像URL地址"
    )
    engine: OCREngineEnum = Field(
        OCREngineEnum.PADDLEOCR,
        description="OCR引擎类型"
    )
    confidence_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )
    language_codes: str = Field(
        "zh,en",
        description="语言代码"
    )
    async_mode: bool = Field(
        False,
        description="是否使用异步模式"
    )
    preprocessing: bool = Field(
        True,
        description="是否启用图像预处理"
    )
    postprocessing: bool = Field(
        True,
        description="是否启用文本后处理"
    )


class BoundingBoxData(BaseModel):
    """
    边界框数据模型
    
    表示文字区域的位置信息。
    """
    points: List[List[float]] = Field(
        ...,
        description="边界框坐标点"
    )
    confidence: float = Field(
        ...,
        description="边界框置信度"
    )
    shape_type: str = Field(
        "rectangle",
        description="形状类型"
    )
    x_min: float = Field(..., description="最小X坐标")
    y_min: float = Field(..., description="最小Y坐标")
    x_max: float = Field(..., description="最大X坐标")
    y_max: float = Field(..., description="最大Y坐标")
    width: float = Field(..., description="宽度")
    height: float = Field(..., description="高度")
    area: float = Field(..., description="面积")


class TextBlockData(BaseModel):
    """
    文本块数据模型
    
    表示识别出的单个文本块信息。
    """
    text: str = Field(..., description="文本内容")
    bbox: BoundingBoxData = Field(..., description="边界框信息")
    confidence: float = Field(..., description="文本置信度")
    language: Optional[str] = Field(None, description="语言标识")
    angle: float = Field(0.0, description="文本角度")
    length: int = Field(..., description="文本长度")
    word_count: int = Field(..., description="词语数量")


class OCRResultData(BaseModel):
    """
    OCR识别结果数据模型
    
    包含完整的OCR识别结果信息。
    """
    task_id: str = Field(..., description="任务ID")
    status: TaskStatusEnum = Field(..., description="任务状态")
    text_content: str = Field(..., description="完整文本内容")
    confidence_score: float = Field(..., description="整体置信度")
    language_detected: Optional[str] = Field(None, description="检测到的语言")
    char_count: int = Field(..., description="字符数量")
    word_count: int = Field(..., description="词语数量")
    processing_time: float = Field(..., description="处理时间（秒）")
    text_blocks: Optional[List[TextBlockData]] = Field(None, description="文本块列表")
    bounding_boxes: Optional[List[BoundingBoxData]] = Field(None, description="边界框列表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据信息")
    async_mode: bool = Field(False, description="是否为异步模式")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class OCRResponse(BaseResponse[OCRResultData]):
    """
    OCR识别响应模型
    
    单图像OCR识别的响应格式。
    """
    pass


class OCRBatchResultData(BaseModel):
    """
    批量OCR结果数据模型
    
    包含批量处理的任务信息。
    """
    task_ids: List[str] = Field(..., description="任务ID列表")
    total_tasks: int = Field(..., description="总任务数")
    status: TaskStatusEnum = Field(..., description="批量任务状态")
    max_concurrent: int = Field(..., description="最大并发数")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OCRBatchResponse(BaseResponse[OCRBatchResultData]):
    """
    批量OCR识别响应模型
    
    批量图像OCR识别的响应格式。
    """
    pass


class OCRTaskData(BaseModel):
    """
    OCR任务数据模型
    
    包含任务的详细信息和状态。
    """
    task_id: str = Field(..., description="任务ID")
    status: TaskStatusEnum = Field(..., description="任务状态")
    engine: str = Field(..., description="使用的OCR引擎")
    confidence_threshold: float = Field(..., description="置信度阈值")
    language_codes: str = Field(..., description="语言代码")
    created_at: Optional[datetime] = Field(None, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误信息")
    result: Optional[Dict[str, Any]] = Field(None, description="识别结果")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class OCRTaskResponse(BaseResponse[OCRTaskData]):
    """
    OCR任务响应模型
    
    查询单个OCR任务的响应格式。
    """
    pass


class OCRResultQuery(PaginationQuery):
    """
    OCR结果查询模型
    
    用于批量查询OCR任务的查询参数。
    """
    status: Optional[TaskStatusEnum] = Field(
        None,
        description="任务状态过滤"
    )
    engine: Optional[OCREngineEnum] = Field(
        None,
        description="引擎类型过滤"
    )
    start_date: Optional[datetime] = Field(
        None,
        description="开始日期过滤"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="结束日期过滤"
    )
    created_by: Optional[str] = Field(
        None,
        description="创建者过滤"
    )
    dataset_id: Optional[str] = Field(
        None,
        description="数据集ID过滤"
    )
    min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="最小置信度过滤"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class OCRStatsData(BaseModel):
    """
    OCR统计数据模型
    
    包含OCR服务的统计信息。
    """
    total_tasks: int = Field(..., description="总任务数")
    completed_tasks: int = Field(..., description="完成任务数")
    failed_tasks: int = Field(..., description="失败任务数")
    pending_tasks: int = Field(..., description="待处理任务数")
    processing_tasks: int = Field(..., description="处理中任务数")
    success_rate: float = Field(..., description="成功率")
    avg_processing_time: float = Field(..., description="平均处理时间")
    avg_confidence: float = Field(..., description="平均置信度")
    total_characters: int = Field(..., description="总字符数")
    total_processing_time: float = Field(..., description="总处理时间")
    engine_usage: Dict[str, int] = Field(..., description="引擎使用统计")
    daily_stats: List[Dict[str, Any]] = Field(..., description="每日统计")


class OCRStatsResponse(BaseResponse[OCRStatsData]):
    """
    OCR统计响应模型
    
    OCR服务统计信息的响应格式。
    """
    pass


class SearchOCRRequest(BaseModel):
    """
    OCR文本搜索请求模型
    
    用于在OCR结果中搜索文本内容。
    """
    query: str = Field(..., min_length=1, description="搜索查询")
    user_id: Optional[str] = Field(None, description="用户ID过滤")
    limit: int = Field(20, ge=1, le=100, description="结果限制")
    offset: int = Field(0, ge=0, description="偏移量")
    min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="最小置信度过滤"
    )


class SearchResultData(BaseModel):
    """
    搜索结果数据模型
    
    包含搜索到的OCR结果信息。
    """
    task_id: str = Field(..., description="任务ID")
    text_content: str = Field(..., description="文本内容")
    confidence_score: float = Field(..., description="置信度")
    char_count: int = Field(..., description="字符数量")
    created_at: datetime = Field(..., description="创建时间")
    rank: float = Field(..., description="搜索相关度")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchOCRResponse(BaseResponse[List[SearchResultData]]):
    """
    OCR文本搜索响应模型
    
    文本搜索结果的响应格式。
    """
    pass