"""
数据采集服务API数据模式

定义请求和响应的数据结构
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# 基础响应模式
class BaseResponse(BaseModel):
    """基础响应模式"""
    success: bool = Field(description="操作是否成功")
    message: Optional[str] = Field(None, description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")


class PaginatedResponse(BaseResponse):
    """分页响应模式"""
    page: int = Field(description="当前页码")
    size: int = Field(description="每页大小")
    total: int = Field(description="总记录数")
    total_pages: int = Field(description="总页数")


# 文件上传相关模式
class FileUploadMetadata(BaseModel):
    """文件上传元数据"""
    title: Optional[str] = Field(None, description="文档标题")
    author: Optional[str] = Field(None, description="作者")
    description: Optional[str] = Field(None, description="描述")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    language: Optional[str] = Field("zh-cn", description="文档语言")
    category: Optional[str] = Field(None, description="文档分类")


class BatchUploadMetadata(BaseModel):
    """批量上传元数据"""
    batch_name: str = Field(description="批次名称")
    description: Optional[str] = Field(None, description="批次描述")
    category: Optional[str] = Field(None, description="批次分类")
    tags: List[str] = Field(default_factory=list, description="批次标签")


class FileInfo(BaseModel):
    """文件信息"""
    filename: str = Field(description="文件名")
    size: int = Field(description="文件大小(字节)")
    type: str = Field(description="文件MIME类型")
    hash: str = Field(description="文件哈希值")


class UploadResult(BaseModel):
    """单个文件上传结果"""
    dataset_id: str = Field(description="数据集ID")
    filename: str = Field(description="文件名")
    file_size: int = Field(description="文件大小")
    upload_status: str = Field(description="上传状态")
    processing_status: str = Field(description="处理状态")
    is_duplicate: bool = Field(False, description="是否为重复文件")
    estimated_processing_time: Optional[int] = Field(None, description="预估处理时间(秒)")


class UploadResponse(BaseResponse):
    """文件上传响应"""
    data: UploadResult = Field(description="上传结果数据")


class BatchUploadFileResult(BaseModel):
    """批量上传单个文件结果"""
    filename: str = Field(description="文件名")
    status: str = Field(description="处理状态")
    dataset_id: Optional[str] = Field(None, description="数据集ID")
    message: Optional[str] = Field(None, description="结果消息")
    error: Optional[str] = Field(None, description="错误信息")
    is_duplicate: bool = Field(False, description="是否为重复文件")


class BatchUploadResult(BaseModel):
    """批量上传结果"""
    batch_id: str = Field(description="批次ID")
    total_files: int = Field(description="总文件数")
    successful_uploads: int = Field(description="成功上传数")
    failed_uploads: int = Field(description="失败上传数")
    uploaded_files: List[BatchUploadFileResult] = Field(description="成功上传的文件列表")
    failed_files: List[BatchUploadFileResult] = Field(description="失败上传的文件列表")


class BatchUploadResponse(BaseResponse):
    """批量上传响应"""
    data: BatchUploadResult = Field(description="批量上传结果数据")


# 数据集相关模式
class DatasetCreateRequest(BaseModel):
    """数据集创建请求"""
    name: str = Field(description="数据集名称")
    description: Optional[str] = Field(None, description="数据集描述")
    source_id: UUID = Field(description="数据源ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class DatasetUpdateRequest(BaseModel):
    """数据集更新请求"""
    name: Optional[str] = Field(None, description="数据集名称")
    description: Optional[str] = Field(None, description="数据集描述")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class TextContentInfo(BaseModel):
    """文本内容信息"""
    id: UUID = Field(description="内容ID")
    title: Optional[str] = Field(None, description="标题")
    content: str = Field(description="文本内容")
    page_number: int = Field(description="页码")
    word_count: int = Field(description="词数")
    char_count: int = Field(description="字符数")
    language: Optional[str] = Field(None, description="语言")
    confidence: Optional[float] = Field(None, description="置信度")
    created_at: datetime = Field(description="创建时间")
    
    class Config:
        from_attributes = True


class DatasetInfo(BaseModel):
    """数据集信息"""
    id: UUID = Field(description="数据集ID")
    name: str = Field(description="数据集名称")
    description: Optional[str] = Field(None, description="描述")
    source_id: UUID = Field(description="数据源ID")
    file_path: Optional[str] = Field(None, description="文件路径")
    file_size: Optional[int] = Field(None, description="文件大小")
    file_type: Optional[str] = Field(None, description="文件类型")
    file_hash: Optional[str] = Field(None, description="文件哈希")
    processing_status: str = Field(description="处理状态")
    error_message: Optional[str] = Field(None, description="错误信息")
    text_count: int = Field(0, description="文本片段数")
    total_words: int = Field(0, description="总词数")
    total_chars: int = Field(0, description="总字符数")
    created_by: UUID = Field(description="创建者ID")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    processed_at: Optional[datetime] = Field(None, description="处理完成时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    text_contents: Optional[List[TextContentInfo]] = Field(None, description="文本内容列表")
    
    class Config:
        from_attributes = True


class DatasetResponse(BaseResponse):
    """数据集响应"""
    data: DatasetInfo = Field(description="数据集信息")


class DatasetListResult(BaseModel):
    """数据集列表结果"""
    items: List[DatasetInfo] = Field(description="数据集列表")
    total: int = Field(description="总数")
    page: int = Field(description="当前页")
    size: int = Field(description="页大小")
    total_pages: int = Field(description="总页数")


class DatasetListResponse(BaseResponse):
    """数据集列表响应"""
    data: DatasetListResult = Field(description="数据集列表结果")


# 处理状态相关模式
class ProcessingStatusInfo(BaseModel):
    """处理状态信息"""
    dataset_id: UUID = Field(description="数据集ID")
    processing_status: str = Field(description="处理状态")
    error_message: Optional[str] = Field(None, description="错误信息")
    text_count: int = Field(0, description="文本片段数")
    total_words: int = Field(0, description="总词数")
    total_chars: int = Field(0, description="总字符数")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    processed_at: Optional[datetime] = Field(None, description="处理完成时间")


class ProcessingStatusResponse(BaseResponse):
    """处理状态响应"""
    data: ProcessingStatusInfo = Field(description="处理状态信息")


# 统计信息相关模式
class StatisticsInfo(BaseModel):
    """统计信息"""
    total_datasets: int = Field(description="总数据集数")
    processing_datasets: int = Field(description="处理中数据集数")
    completed_datasets: int = Field(description="已完成数据集数")
    failed_datasets: int = Field(description="失败数据集数")
    total_file_size: int = Field(description="总文件大小")
    total_text_count: int = Field(description="总文本数")
    total_words: int = Field(description="总词数")
    total_chars: int = Field(description="总字符数")


class StatisticsResponse(BaseResponse):
    """统计信息响应"""
    data: StatisticsInfo = Field(description="统计信息")


# 错误响应模式
class ErrorDetail(BaseModel):
    """错误详情"""
    code: str = Field(description="错误代码")
    message: str = Field(description="错误消息")
    field: Optional[str] = Field(None, description="错误字段")


class ErrorResponse(BaseResponse):
    """错误响应"""
    success: bool = Field(False, description="操作失败")
    error_code: str = Field(description="错误代码")
    error_message: str = Field(description="错误消息")
    details: Optional[List[ErrorDetail]] = Field(None, description="错误详情列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "error_message": "请求参数验证失败",
                "details": [
                    {
                        "code": "VALUE_ERROR",
                        "message": "文件大小超过限制",
                        "field": "file_size"
                    }
                ],
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }