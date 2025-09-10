"""
图像处理服务数据模型单元测试

测试图像处理服务的核心数据结构，包括：
- 请求和响应模型验证
- 枚举类型定义
- 配置模型验证
- 边界条件处理

作者: Claude (测试架构师)
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
from datetime import datetime
from uuid import uuid4
from pydantic import ValidationError
from typing import Dict, Any

from src.schemas.image_schemas import (
    ProcessingType, ProcessingStatus, ImageFormat, ProcessingEngine,
    DenoiseMethod, InterpolationMethod, ResizeMethod, ThresholdMethod,
    BaseResponse, ErrorResponse, ImageSize, ImageInfo, QualityMetrics,
    EnhanceConfig, DenoiseConfig, DeskewConfig, ResizeConfig, FormatConvertConfig,
    SuperResolutionConfig, ProcessingConfig, ImageProcessingRequest,
    BatchProcessingRequest, ProcessingResult, ImageProcessingTask
)


class TestEnumTypes:
    """枚举类型测试"""
    
    def test_processing_type_enum(self):
        """测试图像处理类型枚举"""
        assert ProcessingType.ENHANCE == "enhance"
        assert ProcessingType.DENOISE == "denoise"
        assert ProcessingType.DESKEW == "deskew"
        assert ProcessingType.RESIZE == "resize"
        assert ProcessingType.FORMAT_CONVERT == "format_convert"
        assert ProcessingType.SUPER_RESOLUTION == "super_resolution"
        assert ProcessingType.AUTO_ENHANCE == "auto_enhance"
        assert ProcessingType.QUALITY_ASSESSMENT == "quality_assessment"
        assert ProcessingType.BATCH == "batch"
        
        # 测试枚举转换
        assert ProcessingType("enhance") == ProcessingType.ENHANCE
        
        # 测试无效值
        with pytest.raises(ValueError):
            ProcessingType("invalid_type")
    
    def test_processing_status_enum(self):
        """测试处理状态枚举"""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.CANCELLED == "cancelled"
    
    def test_image_format_enum(self):
        """测试图像格式枚举"""
        assert ImageFormat.JPEG == "jpeg"
        assert ImageFormat.PNG == "png"
        assert ImageFormat.TIFF == "tiff"
        assert ImageFormat.BMP == "bmp"
        assert ImageFormat.WEBP == "webp"
        assert ImageFormat.PDF == "pdf"
    
    def test_processing_engine_enum(self):
        """测试图像处理引擎枚举"""
        assert ProcessingEngine.OPENCV == "opencv"
        assert ProcessingEngine.PILLOW == "pillow"
        assert ProcessingEngine.SKIMAGE == "skimage"
    
    def test_denoise_method_enum(self):
        """测试去噪方法枚举"""
        assert DenoiseMethod.BILATERAL == "bilateral"
        assert DenoiseMethod.GAUSSIAN == "gaussian"
        assert DenoiseMethod.MEDIAN == "median"
        assert DenoiseMethod.NLM == "nlm"
    
    def test_interpolation_method_enum(self):
        """测试插值方法枚举"""
        assert InterpolationMethod.NEAREST == "nearest"
        assert InterpolationMethod.LINEAR == "linear"
        assert InterpolationMethod.CUBIC == "cubic"
        assert InterpolationMethod.LANCZOS == "lanczos"
    
    def test_resize_method_enum(self):
        """测试尺寸调整方法枚举"""
        assert ResizeMethod.SCALE == "scale"
        assert ResizeMethod.FIXED_SIZE == "fixed_size"
        assert ResizeMethod.MAX_DIMENSION == "max_dimension"
    
    def test_threshold_method_enum(self):
        """测试阈值方法枚举"""
        assert ThresholdMethod.OTSU == "otsu"
        assert ThresholdMethod.ADAPTIVE == "adaptive"
        assert ThresholdMethod.FIXED == "fixed"


class TestBasicModels:
    """基础模型测试"""
    
    def test_image_size_creation(self):
        """测试图像尺寸模型创建"""
        size = ImageSize(width=800, height=600)
        
        assert size.width == 800
        assert size.height == 600
        
        # 测试无效尺寸
        with pytest.raises(ValidationError):
            ImageSize(width=0, height=600)
            
        with pytest.raises(ValidationError):
            ImageSize(width=-100, height=600)
    
    def test_image_info_creation(self):
        """测试图像信息模型创建"""
        size = ImageSize(width=1920, height=1080)
        info = ImageInfo(
            size=size,
            channels=3,
            format="jpeg",
            file_size=2048576,
            color_mode="RGB"
        )
        
        assert info.size.width == 1920
        assert info.size.height == 1080
        assert info.channels == 3
        assert info.format == "jpeg"
        assert info.file_size == 2048576
        assert info.color_mode == "RGB"
        
        # 测试无效通道数
        with pytest.raises(ValidationError):
            ImageInfo(
                size=size, channels=0, format="jpeg", 
                file_size=1024, color_mode="RGB"
            )
        
        with pytest.raises(ValidationError):
            ImageInfo(
                size=size, channels=5, format="jpeg", 
                file_size=1024, color_mode="RGB"
            )
    
    def test_quality_metrics_creation(self):
        """测试图像质量指标模型创建"""
        metrics = QualityMetrics(
            brightness_score=0.8,
            contrast_score=0.75,
            sharpness_score=0.9,
            noise_level=0.1,
            blur_level=0.05,
            skew_angle=2.5,
            text_region_ratio=0.6,
            overall_quality=0.85
        )
        
        assert metrics.brightness_score == 0.8
        assert metrics.contrast_score == 0.75
        assert metrics.overall_quality == 0.85
        
        # 测试边界值
        with pytest.raises(ValidationError):
            QualityMetrics(
                brightness_score=1.5,  # 超出范围
                contrast_score=0.75, sharpness_score=0.9,
                noise_level=0.1, blur_level=0.05, skew_angle=2.5,
                text_region_ratio=0.6, overall_quality=0.85
            )


class TestConfigurationModels:
    """配置模型测试"""
    
    def test_enhance_config_creation(self):
        """测试图像增强配置创建"""
        config = EnhanceConfig(
            adjust_brightness=True,
            brightness_factor=1.2,
            adjust_contrast=True,
            contrast_factor=1.1,
            gamma_correction=True,
            gamma=0.8,
            sharpen=True,
            sharpen_strength=1.5
        )
        
        assert config.adjust_brightness is True
        assert config.brightness_factor == 1.2
        assert config.contrast_factor == 1.1
        assert config.gamma == 0.8
        assert config.sharpen_strength == 1.5
        
        # 测试默认值
        default_config = EnhanceConfig()
        assert default_config.adjust_brightness is False
        assert default_config.brightness_factor == 1.0
        assert default_config.clahe_tile_size == (8, 8)
    
    def test_enhance_config_validation(self):
        """测试增强配置验证"""
        # 测试边界值
        with pytest.raises(ValidationError):
            EnhanceConfig(brightness_factor=0.05)  # 小于最小值
        
        with pytest.raises(ValidationError):
            EnhanceConfig(brightness_factor=5.0)  # 大于最大值
        
        with pytest.raises(ValidationError):
            EnhanceConfig(clahe_clip_limit=15.0)  # 大于最大值
    
    def test_denoise_config_creation(self):
        """测试去噪配置创建"""
        config = DenoiseConfig(
            method=DenoiseMethod.BILATERAL,
            bilateral_d=9,
            bilateral_sigma_color=75,
            bilateral_sigma_space=75
        )
        
        assert config.method == DenoiseMethod.BILATERAL
        assert config.bilateral_d == 9
        assert config.bilateral_sigma_color == 75
        
        # 测试不同方法的配置
        gaussian_config = DenoiseConfig(
            method=DenoiseMethod.GAUSSIAN,
            gaussian_kernel_size=7,
            gaussian_sigma=2.0
        )
        
        assert gaussian_config.method == DenoiseMethod.GAUSSIAN
        assert gaussian_config.gaussian_kernel_size == 7
        assert gaussian_config.gaussian_sigma == 2.0
    
    def test_resize_config_creation(self):
        """测试尺寸调整配置创建"""
        config = ResizeConfig(
            method=ResizeMethod.FIXED_SIZE,
            target_width=800,
            target_height=600,
            interpolation=InterpolationMethod.CUBIC,
            maintain_aspect_ratio=False
        )
        
        assert config.method == ResizeMethod.FIXED_SIZE
        assert config.target_width == 800
        assert config.target_height == 600
        assert config.interpolation == InterpolationMethod.CUBIC
        assert config.maintain_aspect_ratio is False
        
        # 测试缩放配置
        scale_config = ResizeConfig(
            method=ResizeMethod.SCALE,
            scale_factor=2.0
        )
        
        assert scale_config.method == ResizeMethod.SCALE
        assert scale_config.scale_factor == 2.0
    
    def test_format_convert_config_creation(self):
        """测试格式转换配置创建"""
        config = FormatConvertConfig(
            target_format=ImageFormat.PNG,
            quality=90,
            enable_binarization=True,
            threshold_method=ThresholdMethod.OTSU
        )
        
        assert config.target_format == ImageFormat.PNG
        assert config.quality == 90
        assert config.enable_binarization is True
        assert config.threshold_method == ThresholdMethod.OTSU
    
    def test_processing_config_combination(self):
        """测试组合处理配置"""
        enhance_config = EnhanceConfig(adjust_brightness=True, brightness_factor=1.1)
        resize_config = ResizeConfig(method=ResizeMethod.SCALE, scale_factor=0.5)
        
        config = ProcessingConfig(
            enhance=enhance_config,
            resize=resize_config
        )
        
        assert config.enhance is not None
        assert config.resize is not None
        assert config.denoise is None
        assert config.enhance.brightness_factor == 1.1
        assert config.resize.scale_factor == 0.5


class TestRequestModels:
    """请求模型测试"""
    
    def test_image_processing_request_basic(self):
        """测试基本图像处理请求"""
        request = ImageProcessingRequest(
            processing_type=ProcessingType.ENHANCE
        )
        
        assert request.processing_type == ProcessingType.ENHANCE
        assert request.config is None
        assert request.engine is None
        assert request.async_mode is False
        assert request.priority == 5
    
    def test_image_processing_request_with_config(self):
        """测试带配置的图像处理请求"""
        enhance_config = EnhanceConfig(
            adjust_brightness=True,
            brightness_factor=1.2
        )
        processing_config = ProcessingConfig(enhance=enhance_config)
        
        request = ImageProcessingRequest(
            processing_type=ProcessingType.ENHANCE,
            config=processing_config,
            engine=ProcessingEngine.PILLOW,
            async_mode=True,
            priority=8
        )
        
        assert request.processing_type == ProcessingType.ENHANCE
        assert request.config is not None
        assert request.config.enhance is not None
        assert request.config.enhance.brightness_factor == 1.2
        assert request.engine == ProcessingEngine.PILLOW
        assert request.async_mode is True
        assert request.priority == 8
    
    def test_request_config_validation(self):
        """测试请求配置验证"""
        # 增强处理但没有增强配置
        resize_config = ResizeConfig(method=ResizeMethod.SCALE, scale_factor=2.0)
        processing_config = ProcessingConfig(resize=resize_config)
        
        with pytest.raises(ValidationError) as exc_info:
            ImageProcessingRequest(
                processing_type=ProcessingType.ENHANCE,
                config=processing_config
            )
        
        assert "增强处理需要enhance配置" in str(exc_info.value)
    
    def test_batch_processing_request(self):
        """测试批量处理请求"""
        image_paths = ["/path/image1.jpg", "/path/image2.jpg", "/path/image3.jpg"]
        
        request = BatchProcessingRequest(
            image_paths=image_paths,
            processing_type=ProcessingType.RESIZE
        )
        
        assert len(request.image_paths) == 3
        assert request.image_paths[0] == "/path/image1.jpg"
        assert request.processing_type == ProcessingType.RESIZE
        assert request.priority == 5
    
    def test_batch_request_validation(self):
        """测试批量请求验证"""
        # 空路径列表
        with pytest.raises(ValidationError):
            BatchProcessingRequest(
                image_paths=[],
                processing_type=ProcessingType.RESIZE
            )
        
        # 超过最大数量
        too_many_paths = [f"/path/image{i}.jpg" for i in range(25)]
        with pytest.raises(ValidationError) as exc_info:
            BatchProcessingRequest(
                image_paths=too_many_paths,
                processing_type=ProcessingType.RESIZE
            )
        
        assert "at most 20 items" in str(exc_info.value)


class TestResultModels:
    """结果模型测试"""
    
    def test_processing_result_creation(self):
        """测试处理结果创建"""
        original_size = ImageSize(width=800, height=600)
        processed_size = ImageSize(width=1600, height=1200)
        
        original_info = ImageInfo(
            size=original_size, channels=3, format="jpeg",
            file_size=1024000, color_mode="RGB"
        )
        processed_info = ImageInfo(
            size=processed_size, channels=3, format="png",
            file_size=2048000, color_mode="RGB"
        )
        
        result = ProcessingResult(
            task_id="task_123",
            processing_type=ProcessingType.RESIZE,
            engine="pillow",
            original_image_info=original_info,
            processed_image_info=processed_info,
            original_image_path="/input/image.jpg",
            processed_image_path="/output/image.png",
            processing_time=2.5
        )
        
        assert result.task_id == "task_123"
        assert result.processing_type == ProcessingType.RESIZE
        assert result.engine == "pillow"
        assert result.processing_time == 2.5
        assert result.original_image_info.size.width == 800
        assert result.processed_image_info.size.width == 1600
    
    def test_task_model_creation(self):
        """测试任务模型创建"""
        task = ImageProcessingTask(
            task_id="task_456",
            processing_type=ProcessingType.DENOISE,
            status=ProcessingStatus.PROCESSING,
            engine="opencv",
            priority=7,
            original_image_path="/input/noisy.jpg",
            progress=75
        )
        
        assert task.task_id == "task_456"
        assert task.processing_type == ProcessingType.DENOISE
        assert task.status == ProcessingStatus.PROCESSING
        assert task.engine == "opencv"
        assert task.priority == 7
        assert task.progress == 75
        assert task.processed_image_path is None
        assert task.error_message is None


class TestEdgeCasesAndValidation:
    """边界条件和验证测试"""
    
    def test_priority_validation(self):
        """测试优先级验证"""
        # 有效优先级
        request = ImageProcessingRequest(
            processing_type=ProcessingType.ENHANCE,
            priority=1
        )
        assert request.priority == 1
        
        request = ImageProcessingRequest(
            processing_type=ProcessingType.ENHANCE,
            priority=10
        )
        assert request.priority == 10
        
        # 无效优先级
        with pytest.raises(ValidationError):
            ImageProcessingRequest(
                processing_type=ProcessingType.ENHANCE,
                priority=0
            )
        
        with pytest.raises(ValidationError):
            ImageProcessingRequest(
                processing_type=ProcessingType.ENHANCE,
                priority=11
            )
    
    def test_progress_validation(self):
        """测试进度验证"""
        # 有效进度
        task = ImageProcessingTask(
            task_id="test", processing_type=ProcessingType.ENHANCE,
            status=ProcessingStatus.PROCESSING, engine="test",
            priority=5, original_image_path="/test", progress=50
        )
        assert task.progress == 50
        
        # 边界值
        task.progress = 0
        assert task.progress == 0
        
        task.progress = 100
        assert task.progress == 100
        
        # 无效进度
        with pytest.raises(ValidationError):
            ImageProcessingTask(
                task_id="test", processing_type=ProcessingType.ENHANCE,
                status=ProcessingStatus.PROCESSING, engine="test",
                priority=5, original_image_path="/test", progress=-1
            )
        
        with pytest.raises(ValidationError):
            ImageProcessingTask(
                task_id="test", processing_type=ProcessingType.ENHANCE,
                status=ProcessingStatus.PROCESSING, engine="test",
                priority=5, original_image_path="/test", progress=101
            )
    
    def test_enum_string_conversion(self):
        """测试枚举字符串转换"""
        # 测试字符串直接传入
        request = ImageProcessingRequest(
            processing_type="enhance",  # 字符串
            engine="pillow"  # 字符串
        )
        
        assert request.processing_type == ProcessingType.ENHANCE
        assert request.engine == ProcessingEngine.PILLOW
    
    def test_config_parameter_ranges(self):
        """测试配置参数范围"""
        # 测试各种配置的边界值
        
        # 增强配置边界值
        config = EnhanceConfig(
            brightness_factor=0.1,  # 最小值
            contrast_factor=3.0,    # 最大值
            gamma=0.1,             # 最小值
            clahe_clip_limit=10.0   # 最大值
        )
        assert config.brightness_factor == 0.1
        assert config.contrast_factor == 3.0
        
        # 去噪配置边界值
        denoise_config = DenoiseConfig(
            bilateral_d=3,           # 最小值
            bilateral_sigma_color=200, # 最大值
            gaussian_kernel_size=15,   # 最大值
            nlm_h=30                  # 最大值
        )
        assert denoise_config.bilateral_d == 3
        assert denoise_config.bilateral_sigma_color == 200
    
    def test_complex_config_combinations(self):
        """测试复杂配置组合"""
        # 创建所有类型的配置
        enhance_config = EnhanceConfig(
            adjust_brightness=True,
            brightness_factor=1.1,
            clahe=True,
            clahe_clip_limit=3.0
        )
        
        denoise_config = DenoiseConfig(
            method=DenoiseMethod.NLM,
            nlm_h=15,
            nlm_template_size=7
        )
        
        resize_config = ResizeConfig(
            method=ResizeMethod.MAX_DIMENSION,
            max_dimension=2048,
            interpolation=InterpolationMethod.LANCZOS
        )
        
        format_config = FormatConvertConfig(
            target_format=ImageFormat.PNG,
            quality=95,
            enable_binarization=True
        )
        
        # 组合所有配置
        full_config = ProcessingConfig(
            enhance=enhance_config,
            denoise=denoise_config,
            resize=resize_config,
            format_convert=format_config
        )
        
        assert full_config.enhance is not None
        assert full_config.denoise is not None
        assert full_config.resize is not None
        assert full_config.format_convert is not None
        
        # 验证配置值
        assert full_config.enhance.clahe_clip_limit == 3.0
        assert full_config.denoise.method == DenoiseMethod.NLM
        assert full_config.resize.max_dimension == 2048
        assert full_config.format_convert.target_format == ImageFormat.PNG