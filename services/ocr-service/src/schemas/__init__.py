"""
数据模型schemas模块

定义所有API请求和响应的Pydantic模型，用于数据验证、
序列化和自动生成API文档。

主要功能：
- API请求/响应模型定义
- 数据验证和类型检查
- 自动生成OpenAPI文档
- 统一的错误响应格式

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from .common_schemas import BaseResponse, ErrorResponse, PaginationQuery
from .ocr_schemas import (
    OCRRequest, OCRResponse, OCRBatchRequest, OCRBatchResponse,
    OCRTaskResponse, OCRUrlRequest, OCRResultQuery
)
from .task_schemas import TaskCreateRequest, TaskUpdateRequest, TaskListResponse
from .config_schemas import ConfigResponse, ConfigUpdateRequest

__all__ = [
    # 通用schemas
    'BaseResponse', 'ErrorResponse', 'PaginationQuery',
    
    # OCR相关schemas  
    'OCRRequest', 'OCRResponse', 'OCRBatchRequest', 'OCRBatchResponse',
    'OCRTaskResponse', 'OCRUrlRequest', 'OCRResultQuery',
    
    # 任务相关schemas
    'TaskCreateRequest', 'TaskUpdateRequest', 'TaskListResponse',
    
    # 配置相关schemas
    'ConfigResponse', 'ConfigUpdateRequest'
]