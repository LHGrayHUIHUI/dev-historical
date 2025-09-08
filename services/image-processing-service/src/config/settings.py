"""
图像处理服务配置管理
无状态图像处理微服务配置
数据存储通过storage-service完成
"""

from pydantic_settings import BaseSettings
from typing import Dict, Any, List, Tuple
import os


class Settings(BaseSettings):
    """图像处理服务配置（无状态架构）"""
    
    # 基础服务配置
    service_name: str = "image-processing-service"
    service_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8005  # 使用8005端口避免与其他服务冲突
    api_prefix: str = "/api/v1"
    workers: int = 1
    
    # Storage Service配置（统一数据管理）
    storage_service_url: str = "http://localhost:8002"
    storage_service_timeout: int = 120  # 图像处理可能需要更长时间
    storage_service_retries: int = 3
    
    # File Processor配置（可选）
    file_processor_url: str = "http://localhost:8001"
    
    # OCR Service配置（可选，处理结果可能用于OCR）
    ocr_service_url: str = "http://localhost:8003"
    
    # NLP Service配置（可选）
    nlp_service_url: str = "http://localhost:8004"
    
    # 图像处理引擎配置
    default_processing_engine: str = "opencv"  # opencv, pillow, skimage
    max_image_size: int = 100 * 1024 * 1024  # 100MB
    max_batch_size: int = 20
    max_image_dimension: int = 8192  # 最大图像尺寸
    image_task_timeout: int = 300  # 5分钟
    max_concurrent_tasks: int = 5
    
    # 支持的图像格式
    supported_input_formats: List[str] = [
        "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"
    ]
    supported_output_formats: List[str] = [
        "jpg", "jpeg", "png", "bmp", "tiff", "webp", "pdf"
    ]
    
    # 处理功能开关
    enable_image_enhancement: bool = True
    enable_noise_reduction: bool = True
    enable_deskewing: bool = True
    enable_resizing: bool = True
    enable_format_conversion: bool = True
    enable_super_resolution: bool = False  # 需要深度学习模型
    enable_auto_enhancement: bool = True
    enable_quality_assessment: bool = True
    
    # 图像增强配置
    default_brightness_range: Tuple[float, float] = (0.5, 2.0)
    default_contrast_range: Tuple[float, float] = (0.5, 2.0)
    default_gamma_range: Tuple[float, float] = (0.5, 2.0)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    # 去噪配置
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    median_kernel_size: int = 5
    nlm_h: float = 10
    nlm_template_window_size: int = 7
    nlm_search_window_size: int = 21
    
    # 倾斜校正配置
    deskew_angle_threshold: float = 0.5
    deskew_max_angle: float = 45.0
    hough_threshold: int = 100
    
    # 尺寸调整配置
    resize_interpolation: str = "cubic"  # nearest, linear, cubic, lanczos
    maintain_aspect_ratio: bool = True
    
    # 格式转换配置
    jpeg_quality: int = 95
    png_compression: int = 6
    tiff_compression: str = "lzw"
    
    # 超分辨率配置
    sr_scale_factors: List[int] = [2, 3, 4]
    sr_model_path: str = "./models/esrgan"
    
    # GPU配置
    use_gpu: bool = False  # 在生产环境中可能开启
    gpu_device: int = 0
    
    # 临时文件配置
    temp_dir: str = "/tmp/image-processing-service"
    temp_file_cleanup_interval: int = 3600
    temp_file_max_age: int = 7200
    
    # 缓存配置（本地缓存，不使用外部Redis）
    enable_cache: bool = True
    cache_max_size: int = 500  # 图像占用内存较大
    cache_ttl: int = 1800  # 30分钟
    
    # 质量评估配置
    quality_metrics: List[str] = [
        "brightness", "contrast", "sharpness", "noise_level", 
        "blur_level", "skew_angle", "text_region_ratio"
    ]
    overall_quality_weights: Dict[str, float] = {
        "brightness": 0.15,
        "contrast": 0.20,
        "sharpness": 0.30,
        "noise_level": 0.20,
        "blur_level": 0.15
    }
    
    # 批量处理配置
    batch_processing_enabled: bool = True
    batch_chunk_size: int = 5
    batch_progress_update_interval: int = 10  # 秒
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = ""
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    enable_json_logs: bool = False
    log_request_id: bool = True
    
    # 开发环境特定配置
    debug_show_error_details: bool = True
    debug_reload_on_change: bool = True
    debug_save_intermediate_results: bool = False
    
    class Config:
        env_prefix = "IMAGE_PROCESSING_SERVICE_"
        env_file = ".env"


# 全局配置实例
settings = Settings()


def get_processing_engine_config() -> Dict[str, Any]:
    """获取图像处理引擎配置"""
    return {
        "opencv": {
            "interpolation_methods": {
                "nearest": 0,  # cv2.INTER_NEAREST
                "linear": 1,   # cv2.INTER_LINEAR
                "cubic": 2,    # cv2.INTER_CUBIC
                "lanczos": 4   # cv2.INTER_LANCZOS4
            },
            "border_types": {
                "constant": 0,    # cv2.BORDER_CONSTANT
                "reflect": 2,     # cv2.BORDER_REFLECT
                "replicate": 1,   # cv2.BORDER_REPLICATE
                "wrap": 3         # cv2.BORDER_WRAP
            }
        },
        "pillow": {
            "resampling_filters": {
                "nearest": 0,     # PIL.Image.NEAREST
                "linear": 1,      # PIL.Image.BILINEAR
                "cubic": 3,       # PIL.Image.BICUBIC
                "lanczos": 1      # PIL.Image.LANCZOS
            }
        },
        "skimage": {
            "preserve_range": True,
            "multichannel": True,
            "anti_aliasing": True
        }
    }


def get_enhancement_config() -> Dict[str, Any]:
    """获取图像增强配置"""
    return {
        "brightness": {
            "min_factor": settings.default_brightness_range[0],
            "max_factor": settings.default_brightness_range[1],
            "default_factor": 1.0
        },
        "contrast": {
            "min_factor": settings.default_contrast_range[0],
            "max_factor": settings.default_contrast_range[1],
            "default_factor": 1.0
        },
        "gamma": {
            "min_value": settings.default_gamma_range[0],
            "max_value": settings.default_gamma_range[1],
            "default_value": 1.0
        },
        "clahe": {
            "clip_limit": settings.clahe_clip_limit,
            "tile_grid_size": settings.clahe_tile_grid_size
        },
        "sharpening": {
            "kernel": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
            "strength": 1.0
        }
    }


def get_denoise_config() -> Dict[str, Any]:
    """获取去噪配置"""
    return {
        "bilateral": {
            "d": settings.bilateral_d,
            "sigma_color": settings.bilateral_sigma_color,
            "sigma_space": settings.bilateral_sigma_space
        },
        "gaussian": {
            "kernel_size": settings.gaussian_kernel_size,
            "sigma": settings.gaussian_sigma
        },
        "median": {
            "kernel_size": settings.median_kernel_size
        },
        "nlm": {
            "h": settings.nlm_h,
            "template_window_size": settings.nlm_template_window_size,
            "search_window_size": settings.nlm_search_window_size
        }
    }


def get_format_config() -> Dict[str, Any]:
    """获取格式转换配置"""
    return {
        "jpeg": {
            "quality": settings.jpeg_quality,
            "optimize": True,
            "progressive": True
        },
        "png": {
            "compression": settings.png_compression,
            "optimize": True
        },
        "tiff": {
            "compression": settings.tiff_compression
        },
        "webp": {
            "quality": settings.jpeg_quality,
            "method": 6
        }
    }


def get_feature_config() -> Dict[str, bool]:
    """获取功能开关配置"""
    return {
        "image_enhancement": settings.enable_image_enhancement,
        "noise_reduction": settings.enable_noise_reduction,
        "deskewing": settings.enable_deskewing,
        "resizing": settings.enable_resizing,
        "format_conversion": settings.enable_format_conversion,
        "super_resolution": settings.enable_super_resolution,
        "auto_enhancement": settings.enable_auto_enhancement,
        "quality_assessment": settings.enable_quality_assessment,
        "batch_processing": settings.batch_processing_enabled
    }