"""
图像处理服务数据模型Schema
Pydantic模型定义，用于API请求和响应数据验证
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime
import uuid


# ============ 枚举类型 ============

class ProcessingType(str, Enum):
    """图像处理类型"""
    ENHANCE = "enhance"
    DENOISE = "denoise"
    DESKEW = "deskew"
    RESIZE = "resize"
    FORMAT_CONVERT = "format_convert"
    SUPER_RESOLUTION = "super_resolution"
    AUTO_ENHANCE = "auto_enhance"
    QUALITY_ASSESSMENT = "quality_assessment"
    BATCH = "batch"


class ProcessingStatus(str, Enum):
    """处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImageFormat(str, Enum):
    """图像格式"""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"
    PDF = "pdf"


class ProcessingEngine(str, Enum):
    """图像处理引擎"""
    OPENCV = "opencv"
    PILLOW = "pillow"
    SKIMAGE = "skimage"


class DenoiseMethod(str, Enum):
    """去噪方法"""
    BILATERAL = "bilateral"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    NLM = "nlm"


class InterpolationMethod(str, Enum):
    """插值方法"""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    LANCZOS = "lanczos"


class ResizeMethod(str, Enum):
    """尺寸调整方法"""
    SCALE = "scale"
    FIXED_SIZE = "fixed_size"
    MAX_DIMENSION = "max_dimension"


class ThresholdMethod(str, Enum):
    """阈值方法"""
    OTSU = "otsu"
    ADAPTIVE = "adaptive"
    FIXED = "fixed"


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


# ============ 图像信息模型 ============

class ImageSize(BaseModel):
    """图像尺寸"""
    width: int = Field(..., ge=1, description="图像宽度")
    height: int = Field(..., ge=1, description="图像高度")


class ImageInfo(BaseModel):
    """图像基本信息"""
    size: ImageSize = Field(..., description="图像尺寸")
    channels: int = Field(..., ge=1, le=4, description="颜色通道数")
    format: str = Field(..., description="图像格式")
    file_size: int = Field(..., ge=0, description="文件大小（字节）")
    color_mode: str = Field(..., description="颜色模式")


class QualityMetrics(BaseModel):
    """图像质量指标"""
    brightness_score: float = Field(..., ge=0, le=1, description="亮度评分")
    contrast_score: float = Field(..., ge=0, le=1, description="对比度评分")
    sharpness_score: float = Field(..., ge=0, le=1, description="清晰度评分")
    noise_level: float = Field(..., ge=0, le=1, description="噪声水平")
    blur_level: float = Field(..., ge=0, le=1, description="模糊程度")
    skew_angle: float = Field(..., description="倾斜角度（度）")
    text_region_ratio: float = Field(..., ge=0, le=1, description="文本区域占比")
    overall_quality: float = Field(..., ge=0, le=1, description="整体质量评分")


# ============ 配置模型 ============

class EnhanceConfig(BaseModel):
    """图像增强配置"""
    adjust_brightness: bool = Field(default=False, description="是否调整亮度")
    brightness_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="亮度因子")
    adjust_contrast: bool = Field(default=False, description="是否调整对比度")
    contrast_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="对比度因子")
    gamma_correction: bool = Field(default=False, description="是否进行伽马校正")
    gamma: float = Field(default=1.0, ge=0.1, le=3.0, description="伽马值")
    histogram_equalization: bool = Field(default=False, description="是否直方图均衡化")
    clahe: bool = Field(default=False, description="是否使用CLAHE")
    clahe_clip_limit: float = Field(default=2.0, ge=1.0, le=10.0, description="CLAHE裁剪限制")
    clahe_tile_size: Tuple[int, int] = Field(default=(8, 8), description="CLAHE瓦片大小")
    sharpen: bool = Field(default=False, description="是否锐化")
    sharpen_strength: float = Field(default=1.0, ge=0.1, le=3.0, description="锐化强度")


class DenoiseConfig(BaseModel):
    """去噪配置"""
    method: DenoiseMethod = Field(default=DenoiseMethod.BILATERAL, description="去噪方法")
    # 双边滤波参数
    bilateral_d: int = Field(default=9, ge=3, le=15, description="邻域直径")
    bilateral_sigma_color: float = Field(default=75, ge=10, le=200, description="颜色空间标准差")
    bilateral_sigma_space: float = Field(default=75, ge=10, le=200, description="坐标空间标准差")
    # 高斯滤波参数
    gaussian_kernel_size: int = Field(default=5, ge=3, le=15, description="高斯核大小")
    gaussian_sigma: float = Field(default=1.0, ge=0.1, le=5.0, description="高斯标准差")
    # 中值滤波参数
    median_kernel_size: int = Field(default=5, ge=3, le=15, description="中值滤波核大小")
    # 非局部均值参数
    nlm_h: float = Field(default=10, ge=1, le=30, description="滤波强度")
    nlm_template_size: int = Field(default=7, ge=3, le=15, description="模板窗口大小")
    nlm_search_size: int = Field(default=21, ge=7, le=35, description="搜索窗口大小")


class DeskewConfig(BaseModel):
    """倾斜校正配置"""
    auto_detect: bool = Field(default=True, description="是否自动检测角度")
    manual_angle: Optional[float] = Field(default=None, ge=-45, le=45, description="手动指定角度")
    angle_threshold: float = Field(default=0.5, ge=0.1, le=5.0, description="角度阈值")
    interpolation: InterpolationMethod = Field(default=InterpolationMethod.CUBIC, description="插值方法")


class ResizeConfig(BaseModel):
    """尺寸调整配置"""
    method: ResizeMethod = Field(default=ResizeMethod.SCALE, description="调整方法")
    scale_factor: float = Field(default=1.0, ge=0.1, le=10.0, description="缩放因子")
    target_width: Optional[int] = Field(default=None, ge=1, description="目标宽度")
    target_height: Optional[int] = Field(default=None, ge=1, description="目标高度")
    max_dimension: int = Field(default=2048, ge=100, description="最大尺寸")
    interpolation: InterpolationMethod = Field(default=InterpolationMethod.CUBIC, description="插值方法")
    maintain_aspect_ratio: bool = Field(default=True, description="是否保持宽高比")


class FormatConvertConfig(BaseModel):
    """格式转换配置"""
    target_format: ImageFormat = Field(..., description="目标格式")
    quality: int = Field(default=95, ge=1, le=100, description="输出质量")
    # 二值化配置（转换为黑白图像时）
    enable_binarization: bool = Field(default=False, description="是否二值化")
    threshold_method: ThresholdMethod = Field(default=ThresholdMethod.OTSU, description="阈值方法")
    threshold_value: int = Field(default=127, ge=0, le=255, description="固定阈值")
    # PNG特定配置
    png_compression: int = Field(default=6, ge=0, le=9, description="PNG压缩级别")
    # TIFF特定配置
    tiff_compression: str = Field(default="lzw", description="TIFF压缩方式")


class SuperResolutionConfig(BaseModel):
    """超分辨率配置"""
    scale_factor: int = Field(default=2, ge=2, le=4, description="放大倍数")
    model_name: str = Field(default="ESRGAN", description="模型名称")
    use_gpu: bool = Field(default=False, description="是否使用GPU")


class ProcessingConfig(BaseModel):
    """图像处理配置"""
    enhance: Optional[EnhanceConfig] = Field(default=None, description="增强配置")
    denoise: Optional[DenoiseConfig] = Field(default=None, description="去噪配置")
    deskew: Optional[DeskewConfig] = Field(default=None, description="校正配置")
    resize: Optional[ResizeConfig] = Field(default=None, description="调整配置")
    format_convert: Optional[FormatConvertConfig] = Field(default=None, description="转换配置")
    super_resolution: Optional[SuperResolutionConfig] = Field(default=None, description="超分辨率配置")


# ============ 请求模型 ============

class ImageProcessingRequest(BaseModel):
    """图像处理请求"""
    processing_type: ProcessingType = Field(..., description="处理类型")
    config: Optional[ProcessingConfig] = Field(default=None, description="处理配置")
    engine: Optional[ProcessingEngine] = Field(default=None, description="指定处理引擎")
    async_mode: bool = Field(default=False, description="是否异步处理")
    dataset_id: Optional[str] = Field(default=None, description="关联数据集ID")
    priority: int = Field(default=5, ge=1, le=10, description="任务优先级")
    callback_url: Optional[str] = Field(default=None, description="回调URL")
    
    @validator('config')
    def validate_config(cls, v, values):
        """验证配置与处理类型的匹配"""
        if not v:
            return v
        
        processing_type = values.get('processing_type')
        if processing_type == ProcessingType.ENHANCE and not v.enhance:
            raise ValueError('增强处理需要enhance配置')
        elif processing_type == ProcessingType.DENOISE and not v.denoise:
            raise ValueError('去噪处理需要denoise配置')
        elif processing_type == ProcessingType.DESKEW and not v.deskew:
            raise ValueError('校正处理需要deskew配置')
        elif processing_type == ProcessingType.RESIZE and not v.resize:
            raise ValueError('尺寸调整需要resize配置')
        elif processing_type == ProcessingType.FORMAT_CONVERT and not v.format_convert:
            raise ValueError('格式转换需要format_convert配置')
        elif processing_type == ProcessingType.SUPER_RESOLUTION and not v.super_resolution:
            raise ValueError('超分辨率需要super_resolution配置')
        
        return v


class BatchProcessingRequest(BaseModel):
    """批量图像处理请求"""
    image_paths: List[str] = Field(..., min_items=1, max_items=20, description="图像路径列表")
    processing_type: ProcessingType = Field(..., description="处理类型")
    config: Optional[ProcessingConfig] = Field(default=None, description="处理配置")
    engine: Optional[ProcessingEngine] = Field(default=None, description="指定处理引擎")
    dataset_id: Optional[str] = Field(default=None, description="关联数据集ID")
    priority: int = Field(default=5, ge=1, le=10, description="任务优先级")
    callback_url: Optional[str] = Field(default=None, description="回调URL")


class QualityAssessmentRequest(BaseModel):
    """图像质量评估请求"""
    compare_with_reference: bool = Field(default=False, description="是否与参考图像比较")
    reference_image_path: Optional[str] = Field(default=None, description="参考图像路径")


# ============ 处理结果模型 ============

class ProcessingResult(BaseModel):
    """图像处理结果基类"""
    task_id: str = Field(..., description="任务ID")
    processing_type: ProcessingType = Field(..., description="处理类型")
    engine: str = Field(..., description="使用的引擎")
    original_image_info: ImageInfo = Field(..., description="原始图像信息")
    processed_image_info: ImageInfo = Field(..., description="处理后图像信息")
    original_image_path: str = Field(..., description="原始图像路径")
    processed_image_path: str = Field(..., description="处理后图像路径")
    processing_time: float = Field(..., description="处理时间（秒）")
    quality_before: Optional[QualityMetrics] = Field(default=None, description="处理前质量")
    quality_after: Optional[QualityMetrics] = Field(default=None, description="处理后质量")
    config_used: Dict[str, Any] = Field(default_factory=dict, description="使用的配置")


class EnhancementResult(ProcessingResult):
    """图像增强结果"""
    enhancement_metrics: Dict[str, float] = Field(default_factory=dict, description="增强指标")


class DenoiseResult(ProcessingResult):
    """去噪结果"""
    noise_reduction_ratio: float = Field(..., description="噪声降低比例")


class DeskewResult(ProcessingResult):
    """倾斜校正结果"""
    detected_angle: float = Field(..., description="检测到的倾斜角度")
    correction_angle: float = Field(..., description="校正角度")


class ResizeResult(ProcessingResult):
    """尺寸调整结果"""
    scale_factor_applied: float = Field(..., description="实际应用的缩放因子")


class FormatConvertResult(ProcessingResult):
    """格式转换结果"""
    source_format: str = Field(..., description="源格式")
    target_format: str = Field(..., description="目标格式")
    file_size_change: float = Field(..., description="文件大小变化比例")


class BatchProcessingResult(BaseModel):
    """批量处理结果"""
    batch_id: str = Field(..., description="批次ID")
    total_images: int = Field(..., description="总图像数")
    processed_count: int = Field(..., description="已处理数量")
    failed_count: int = Field(..., description="失败数量")
    success_rate: float = Field(..., description="成功率")
    total_processing_time: float = Field(..., description="总处理时间")
    results: List[ProcessingResult] = Field(..., description="处理结果列表")
    failed_images: List[Dict[str, Any]] = Field(default_factory=list, description="失败图像信息")


# ============ 任务模型 ============

class ImageProcessingTask(BaseModel):
    """图像处理任务"""
    task_id: str = Field(..., description="任务ID")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    dataset_id: Optional[str] = Field(default=None, description="数据集ID")
    processing_type: ProcessingType = Field(..., description="处理类型")
    status: ProcessingStatus = Field(..., description="任务状态")
    engine: str = Field(..., description="处理引擎")
    priority: int = Field(..., description="任务优先级")
    original_image_path: str = Field(..., description="原始图像路径")
    processed_image_path: Optional[str] = Field(default=None, description="处理后图像路径")
    config: Dict[str, Any] = Field(default_factory=dict, description="处理配置")
    progress: int = Field(default=0, ge=0, le=100, description="处理进度")
    processing_time: Optional[float] = Field(default=None, description="处理时间")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")


# ============ 响应模型 ============

class ImageProcessingResponse(BaseResponse):
    """图像处理响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    result: Optional[ProcessingResult] = Field(default=None, description="处理结果")


class BatchProcessingResponse(BaseResponse):
    """批量处理响应"""
    batch_id: str = Field(..., description="批次ID")
    status: str = Field(..., description="批次状态")
    total_images: int = Field(..., description="总图像数")
    result: Optional[BatchProcessingResult] = Field(default=None, description="批量结果")


class TaskStatusResponse(BaseResponse):
    """任务状态响应"""
    task: ImageProcessingTask = Field(..., description="任务信息")


class TaskListResponse(BaseResponse):
    """任务列表响应"""
    tasks: List[ImageProcessingTask] = Field(..., description="任务列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="页码")
    size: int = Field(..., description="页大小")


class QualityAssessmentResponse(BaseResponse):
    """质量评估响应"""
    quality_metrics: QualityMetrics = Field(..., description="质量指标")
    assessment_time: float = Field(..., description="评估耗时")
    recommendations: List[str] = Field(default_factory=list, description="改进建议")


# ============ 引擎信息模型 ============

class ProcessingEngineInfo(BaseModel):
    """图像处理引擎信息"""
    name: str = Field(..., description="引擎名称")
    version: str = Field(..., description="版本号")
    supported_formats: List[str] = Field(..., description="支持格式")
    supported_operations: List[str] = Field(..., description="支持操作")
    gpu_support: bool = Field(..., description="是否支持GPU")
    description: str = Field(..., description="描述")


class EnginesResponse(BaseResponse):
    """引擎列表响应"""
    engines: List[ProcessingEngineInfo] = Field(..., description="引擎列表")


# ============ 统计信息模型 ============

class ProcessingStatistics(BaseModel):
    """图像处理统计"""
    total_tasks: int = Field(..., description="总任务数")
    completed_tasks: int = Field(..., description="完成任务数")
    failed_tasks: int = Field(..., description="失败任务数")
    processing_tasks: int = Field(..., description="处理中任务数")
    avg_processing_time: float = Field(..., description="平均处理时间")
    processing_type_stats: Dict[str, int] = Field(..., description="处理类型统计")
    engine_stats: Dict[str, int] = Field(..., description="引擎使用统计")
    format_stats: Dict[str, int] = Field(..., description="格式统计")
    quality_improvement_avg: float = Field(..., description="平均质量提升")


class StatisticsResponse(BaseResponse):
    """统计响应"""
    statistics: ProcessingStatistics = Field(..., description="统计信息")


# ============ 服务信息模型 ============

class ServiceInfo(BaseModel):
    """服务信息"""
    service_name: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
    status: str = Field(..., description="状态")
    uptime: float = Field(..., description="运行时间（秒）")
    available_engines: List[str] = Field(..., description="可用引擎")
    supported_formats: List[str] = Field(..., description="支持格式")
    processing_capabilities: List[str] = Field(..., description="处理能力")
    gpu_available: bool = Field(..., description="是否有GPU可用")
    max_image_size: int = Field(..., description="最大图像大小")
    max_concurrent_tasks: int = Field(..., description="最大并发任务数")


class HealthCheckResponse(BaseResponse):
    """健康检查响应"""
    service_info: ServiceInfo = Field(..., description="服务信息")
    dependencies: Dict[str, str] = Field(..., description="依赖服务状态")