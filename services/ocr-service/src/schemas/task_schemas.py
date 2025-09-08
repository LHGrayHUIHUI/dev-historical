"""
任务相关数据模型schemas

定义OCR任务管理相关的请求/响应模型，包括任务创建、
更新、列表查询等数据结构。

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from uuid import UUID

from .common_schemas import BaseResponse, PaginationQuery
from .ocr_schemas import TaskStatusEnum, OCREngineEnum


class TaskCreateRequest(BaseModel):
    """
    任务创建请求模型
    
    用于创建新的OCR任务。
    """
    image_path: str = Field(..., description="图像文件路径")
    dataset_id: Optional[UUID] = Field(None, description="数据集ID")
    ocr_engine: OCREngineEnum = Field(
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
    preprocessing_config: Optional[Dict[str, Any]] = Field(
        None,
        description="预处理配置参数"
    )
    
    @validator('language_codes')
    def validate_language_codes(cls, v):
        """验证语言代码格式"""
        if not v or not isinstance(v, str):
            raise ValueError("语言代码不能为空")
        return v


class TaskUpdateRequest(BaseModel):
    """
    任务更新请求模型
    
    用于更新已存在的OCR任务。
    """
    processing_status: Optional[TaskStatusEnum] = Field(
        None,
        description="处理状态"
    )
    error_message: Optional[str] = Field(
        None,
        description="错误信息"
    )
    started_at: Optional[datetime] = Field(
        None,
        description="开始时间"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="完成时间"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class TaskData(BaseModel):
    """
    任务数据模型
    
    包含OCR任务的完整信息。
    """
    id: UUID = Field(..., description="任务ID")
    dataset_id: Optional[UUID] = Field(None, description="数据集ID")
    image_path: str = Field(..., description="图像路径")
    image_size: Optional[Dict[str, int]] = Field(None, description="图像尺寸")
    processing_status: TaskStatusEnum = Field(..., description="处理状态")
    ocr_engine: str = Field(..., description="OCR引擎")
    confidence_threshold: float = Field(..., description="置信度阈值")
    language_codes: str = Field(..., description="语言代码")
    preprocessing_config: Optional[Dict[str, Any]] = Field(None, description="预处理配置")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_by: UUID = Field(..., description="创建者ID")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class TaskResponse(BaseResponse[TaskData]):
    """
    单个任务响应模型
    
    查询或操作单个任务的响应格式。
    """
    pass


class TaskListData(BaseModel):
    """
    任务列表数据模型
    
    包含任务列表和分页信息。
    """
    tasks: List[TaskData] = Field(..., description="任务列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页")
    size: int = Field(..., description="每页大小")
    pages: int = Field(..., description="总页数")


class TaskListResponse(BaseResponse[TaskListData]):
    """
    任务列表响应模型
    
    任务列表查询的响应格式。
    """
    pass


class TaskQuery(PaginationQuery):
    """
    任务查询模型
    
    用于任务列表查询的参数。
    """
    status: Optional[TaskStatusEnum] = Field(
        None,
        description="状态过滤"
    )
    engine: Optional[OCREngineEnum] = Field(
        None,
        description="引擎过滤"
    )
    dataset_id: Optional[UUID] = Field(
        None,
        description="数据集过滤"
    )
    created_by: Optional[UUID] = Field(
        None,
        description="创建者过滤"
    )
    start_date: Optional[datetime] = Field(
        None,
        description="开始日期过滤"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="结束日期过滤"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v) if v else None
        }


class TaskStatsData(BaseModel):
    """
    任务统计数据模型
    
    包含任务相关的统计信息。
    """
    total_tasks: int = Field(..., description="总任务数")
    status_distribution: Dict[str, int] = Field(..., description="状态分布")
    engine_distribution: Dict[str, int] = Field(..., description="引擎分布")
    avg_processing_time: float = Field(..., description="平均处理时间")
    success_rate: float = Field(..., description="成功率")
    hourly_stats: List[Dict[str, Any]] = Field(..., description="小时统计")
    daily_stats: List[Dict[str, Any]] = Field(..., description="日统计")


class TaskStatsResponse(BaseResponse[TaskStatsData]):
    """
    任务统计响应模型
    
    任务统计信息的响应格式。
    """
    pass


class TaskBatchOperationRequest(BaseModel):
    """
    批量任务操作请求模型
    
    用于批量操作多个任务。
    """
    task_ids: List[UUID] = Field(..., min_items=1, description="任务ID列表")
    operation: str = Field(..., description="操作类型")
    parameters: Optional[Dict[str, Any]] = Field(None, description="操作参数")
    
    @validator('operation')
    def validate_operation(cls, v):
        """验证操作类型"""
        allowed_operations = [
            'cancel', 'retry', 'delete', 'update_status'
        ]
        if v not in allowed_operations:
            raise ValueError(f"不支持的操作类型: {v}")
        return v


class TaskBatchOperationResult(BaseModel):
    """
    批量操作结果模型
    
    包含批量操作的结果信息。
    """
    task_id: UUID = Field(..., description="任务ID")
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    error: Optional[str] = Field(None, description="错误信息")
    
    class Config:
        json_encoders = {
            UUID: lambda v: str(v)
        }


class TaskBatchOperationResponse(BaseResponse[List[TaskBatchOperationResult]]):
    """
    批量操作响应模型
    
    批量任务操作的响应格式。
    """
    pass