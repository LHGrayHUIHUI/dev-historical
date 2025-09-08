"""
配置相关数据模型schemas

定义OCR配置管理相关的请求/响应模型，包括引擎配置、
系统设置、预设管理等数据结构。

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from uuid import UUID

from .common_schemas import BaseResponse
from .ocr_schemas import OCREngineEnum


class EngineConfigData(BaseModel):
    """
    引擎配置数据模型
    
    包含单个OCR引擎的配置信息。
    """
    id: UUID = Field(..., description="配置ID")
    name: str = Field(..., description="配置名称")
    description: Optional[str] = Field(None, description="配置描述")
    engine: OCREngineEnum = Field(..., description="引擎类型")
    config: Dict[str, Any] = Field(..., description="配置参数")
    is_default: bool = Field(False, description="是否为默认配置")
    is_active: bool = Field(True, description="是否启用")
    created_by: UUID = Field(..., description="创建者ID")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ConfigCreateRequest(BaseModel):
    """
    配置创建请求模型
    
    用于创建新的引擎配置。
    """
    name: str = Field(..., min_length=1, max_length=100, description="配置名称")
    description: Optional[str] = Field(None, max_length=500, description="配置描述")
    engine: OCREngineEnum = Field(..., description="引擎类型")
    config: Dict[str, Any] = Field(..., description="配置参数")
    is_default: bool = Field(False, description="是否为默认配置")
    is_active: bool = Field(True, description="是否启用")
    
    @validator('config')
    def validate_config(cls, v, values):
        """验证配置参数"""
        if not isinstance(v, dict) or not v:
            raise ValueError("配置参数不能为空")
        return v


class ConfigUpdateRequest(BaseModel):
    """
    配置更新请求模型
    
    用于更新已存在的引擎配置。
    """
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="配置名称")
    description: Optional[str] = Field(None, max_length=500, description="配置描述")
    config: Optional[Dict[str, Any]] = Field(None, description="配置参数")
    is_default: Optional[bool] = Field(None, description="是否为默认配置")
    is_active: Optional[bool] = Field(None, description="是否启用")


class ConfigResponse(BaseResponse[EngineConfigData]):
    """
    单个配置响应模型
    
    查询或操作单个配置的响应格式。
    """
    pass


class ConfigListData(BaseModel):
    """
    配置列表数据模型
    
    包含配置列表和相关信息。
    """
    configs: List[EngineConfigData] = Field(..., description="配置列表")
    total: int = Field(..., description="总数量")
    by_engine: Dict[str, int] = Field(..., description="按引擎分组统计")
    active_count: int = Field(..., description="启用的配置数量")
    default_configs: List[str] = Field(..., description="默认配置ID列表")


class ConfigListResponse(BaseResponse[ConfigListData]):
    """
    配置列表响应模型
    
    配置列表查询的响应格式。
    """
    pass


class PreprocessingConfigData(BaseModel):
    """
    预处理配置数据模型
    
    定义图像预处理的详细配置。
    """
    grayscale: bool = Field(True, description="转换为灰度")
    denoise: bool = Field(True, description="去噪处理")
    enhance_contrast: bool = Field(True, description="增强对比度")
    deskew: bool = Field(True, description="倾斜校正")
    binarize: bool = Field(False, description="二值化")
    resize: bool = Field(False, description="尺寸调整")
    scale_factor: float = Field(1.0, gt=0.1, le=5.0, description="缩放因子")
    
    # 去噪参数
    denoise_method: str = Field("bilateral", description="去噪方法")
    denoise_strength: float = Field(0.5, ge=0.0, le=1.0, description="去噪强度")
    
    # 对比度参数
    contrast_alpha: float = Field(1.2, ge=0.5, le=3.0, description="对比度增益")
    contrast_beta: float = Field(10, ge=-50, le=50, description="亮度偏移")
    
    # 二值化参数
    binarize_method: str = Field("otsu", description="二值化方法")
    threshold_value: int = Field(127, ge=0, le=255, description="阈值")


class PostprocessingConfigData(BaseModel):
    """
    后处理配置数据模型
    
    定义文本后处理的详细配置。
    """
    normalize_whitespace: bool = Field(True, description="规范化空白字符")
    normalize_punctuation: bool = Field(True, description="规范化标点符号")
    traditional_to_simplified: bool = Field(False, description="繁体转简体")
    simplified_to_traditional: bool = Field(False, description="简体转繁体")
    remove_suspicious_chars: bool = Field(True, description="移除可疑字符")
    correct_ocr_errors: bool = Field(True, description="纠正OCR错误")
    preserve_line_breaks: bool = Field(True, description="保留换行符")
    min_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="最小置信度阈值")


class SystemConfigData(BaseModel):
    """
    系统配置数据模型
    
    包含OCR服务的系统级配置。
    """
    max_file_size: int = Field(..., description="最大文件大小（字节）")
    max_batch_size: int = Field(..., description="最大批量处理数量")
    max_concurrent_tasks: int = Field(..., description="最大并发任务数")
    task_timeout: int = Field(..., description="任务超时时间（秒）")
    temp_file_cleanup_interval: int = Field(..., description="临时文件清理间隔（秒）")
    enable_preprocessing: bool = Field(True, description="默认启用预处理")
    enable_postprocessing: bool = Field(True, description="默认启用后处理")
    default_engine: OCREngineEnum = Field(..., description="默认OCR引擎")
    default_confidence_threshold: float = Field(..., description="默认置信度阈值")
    default_language_codes: str = Field(..., description="默认语言代码")
    log_level: str = Field("INFO", description="日志级别")
    enable_metrics: bool = Field(True, description="启用性能指标")
    metrics_retention_days: int = Field(30, description="指标保留天数")


class SystemConfigResponse(BaseResponse[SystemConfigData]):
    """
    系统配置响应模型
    
    系统配置查询的响应格式。
    """
    pass


class SystemConfigUpdateRequest(BaseModel):
    """
    系统配置更新请求模型
    
    用于更新系统配置。
    """
    max_file_size: Optional[int] = Field(None, gt=0, description="最大文件大小")
    max_batch_size: Optional[int] = Field(None, gt=0, le=100, description="最大批量大小")
    max_concurrent_tasks: Optional[int] = Field(None, gt=0, le=50, description="最大并发任务")
    task_timeout: Optional[int] = Field(None, gt=0, description="任务超时时间")
    temp_file_cleanup_interval: Optional[int] = Field(None, gt=0, description="清理间隔")
    enable_preprocessing: Optional[bool] = Field(None, description="启用预处理")
    enable_postprocessing: Optional[bool] = Field(None, description="启用后处理")
    default_engine: Optional[OCREngineEnum] = Field(None, description="默认引擎")
    default_confidence_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="默认置信度阈值"
    )
    default_language_codes: Optional[str] = Field(None, description="默认语言代码")
    log_level: Optional[str] = Field(None, description="日志级别")
    enable_metrics: Optional[bool] = Field(None, description="启用指标")
    metrics_retention_days: Optional[int] = Field(
        None, gt=0, le=365, description="指标保留天数"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        if v is not None:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if v.upper() not in valid_levels:
                raise ValueError(f"无效的日志级别: {v}")
        return v.upper() if v else v


class EngineInfoData(BaseModel):
    """
    引擎信息数据模型
    
    包含OCR引擎的详细信息。
    """
    name: str = Field(..., description="引擎名称")
    version: Optional[str] = Field(None, description="引擎版本")
    initialized: bool = Field(..., description="是否已初始化")
    supported_languages: List[str] = Field(..., description="支持的语言")
    features: List[str] = Field(..., description="支持的特性")
    config_schema: Dict[str, Any] = Field(..., description="配置模式")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="性能指标")


class EngineInfoResponse(BaseResponse[EngineInfoData]):
    """
    引擎信息响应模型
    
    引擎信息查询的响应格式。
    """
    pass


class EngineListResponse(BaseResponse[List[EngineInfoData]]):
    """
    引擎列表响应模型
    
    所有引擎信息的响应格式。
    """
    pass


class ConfigValidationRequest(BaseModel):
    """
    配置验证请求模型
    
    用于验证引擎配置的有效性。
    """
    engine: OCREngineEnum = Field(..., description="引擎类型")
    config: Dict[str, Any] = Field(..., description="配置参数")


class ConfigValidationResult(BaseModel):
    """
    配置验证结果模型
    
    包含配置验证的结果信息。
    """
    valid: bool = Field(..., description="配置是否有效")
    errors: List[str] = Field(default_factory=list, description="验证错误列表")
    warnings: List[str] = Field(default_factory=list, description="验证警告列表")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


class ConfigValidationResponse(BaseResponse[ConfigValidationResult]):
    """
    配置验证响应模型
    
    配置验证的响应格式。
    """
    pass


class PresetConfigData(BaseModel):
    """
    预设配置数据模型
    
    包含常用的预设配置选项。
    """
    name: str = Field(..., description="预设名称")
    description: str = Field(..., description="预设描述")
    engine: OCREngineEnum = Field(..., description="引擎类型")
    config: Dict[str, Any] = Field(..., description="配置参数")
    use_cases: List[str] = Field(..., description="适用场景")
    recommended: bool = Field(False, description="是否推荐")


class PresetConfigResponse(BaseResponse[List[PresetConfigData]]):
    """
    预设配置响应模型
    
    预设配置列表的响应格式。
    """
    pass