"""
图像处理服务核心功能单元测试
"""

import pytest
import numpy as np
from PIL import Image
import io
from pathlib import Path

from src.services.image_processing_service import ImageProcessingService
from src.schemas.image_schemas import ProcessingType, Engine


class TestImageProcessingService:
    """图像处理服务测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.service = ImageProcessingService()
    
    def test_assess_quality(self, sample_image: bytes):
        """测试图像质量评估"""
        quality_metrics = self.service.assess_quality(sample_image)
        
        # 验证所有质量指标都存在
        expected_metrics = [
            'brightness', 'contrast', 'sharpness', 'noise_level', 
            'blur_metric', 'skew_angle'
        ]
        
        for metric in expected_metrics:
            assert metric in quality_metrics
            assert isinstance(quality_metrics[metric], (int, float))
        
        # 验证数值范围
        assert 0 <= quality_metrics['brightness'] <= 255
        assert quality_metrics['contrast'] >= 0
        assert quality_metrics['sharpness'] >= 0
        assert 0 <= quality_metrics['noise_level'] <= 1
        assert quality_metrics['blur_metric'] >= 0
        assert -90 <= quality_metrics['skew_angle'] <= 90
    
    def test_enhance_image_basic(self, sample_image: bytes):
        """测试基础图像增强"""
        config = {
            "brightness_factor": 1.2,
            "contrast_factor": 1.1,
            "sharpness_factor": 1.0
        }
        
        enhanced_image = self.service.enhance_image(
            sample_image, 
            config=config,
            engine=Engine.PILLOW
        )
        
        # 验证返回的是字节数据
        assert isinstance(enhanced_image, bytes)
        assert len(enhanced_image) > 0
        
        # 验证可以重新加载为图像
        img = Image.open(io.BytesIO(enhanced_image))
        assert img.size == (100, 100)  # 原始图像尺寸
    
    def test_denoise_image(self, noisy_image: bytes):
        """测试图像去噪"""
        config = {"denoise_strength": 10}
        
        denoised = self.service.denoise_image(
            noisy_image,
            config=config,
            engine=Engine.OPENCV
        )
        
        assert isinstance(denoised, bytes)
        assert len(denoised) > 0
    
    def test_deskew_image(self, skewed_image: bytes):
        """测试图像倾斜校正"""
        config = {"angle_threshold": 1.0}
        
        deskewed = self.service.deskew_image(
            skewed_image,
            config=config,
            engine=Engine.OPENCV
        )
        
        assert isinstance(deskewed, bytes)
        assert len(deskewed) > 0
    
    def test_resize_image(self, sample_image: bytes):
        """测试图像尺寸调整"""
        config = {
            "width": 200,
            "height": 150,
            "interpolation": "lanczos",
            "maintain_aspect_ratio": False
        }
        
        resized = self.service.resize_image(
            sample_image,
            config=config,
            engine=Engine.PILLOW
        )
        
        assert isinstance(resized, bytes)
        
        # 验证尺寸
        img = Image.open(io.BytesIO(resized))
        assert img.size == (200, 150)
    
    def test_resize_maintain_aspect_ratio(self, sample_image: bytes):
        """测试保持宽高比的尺寸调整"""
        config = {
            "width": 200,
            "height": 300,  # 不同宽高比
            "maintain_aspect_ratio": True
        }
        
        resized = self.service.resize_image(
            sample_image,
            config=config,
            engine=Engine.PILLOW
        )
        
        img = Image.open(io.BytesIO(resized))
        # 应该保持1:1的宽高比，取较小的尺寸
        assert img.size == (200, 200)
    
    def test_convert_format(self, sample_image: bytes):
        """测试格式转换"""
        config = {
            "output_format": "png",
            "quality": 95
        }
        
        converted = self.service.convert_format(
            sample_image,
            config=config,
            engine=Engine.PILLOW
        )
        
        assert isinstance(converted, bytes)
        
        # 验证格式
        img = Image.open(io.BytesIO(converted))
        assert img.format == 'PNG'
    
    def test_auto_enhance(self, sample_image: bytes):
        """测试智能自动增强"""
        config = {
            "target_brightness": 128,
            "target_contrast": 50,
            "adaptive": True
        }
        
        enhanced = self.service.auto_enhance(
            sample_image,
            config=config,
            engine=Engine.OPENCV
        )
        
        assert isinstance(enhanced, bytes)
        assert len(enhanced) > 0
    
    @pytest.mark.asyncio
    async def test_process_image_sync(self, sample_image: bytes):
        """测试同步图像处理"""
        result = await self.service.process_image(
            image_data=sample_image,
            processing_type=ProcessingType.ENHANCE,
            engine=Engine.PILLOW,
            config={
                "brightness_factor": 1.1,
                "contrast_factor": 1.1
            },
            async_processing=False
        )
        
        # 验证结果结构
        assert "processed_image" in result
        assert "quality_before" in result
        assert "quality_after" in result
        assert "processing_time" in result
        assert "metadata" in result
        
        # 验证处理后的图像数据
        assert isinstance(result["processed_image"], bytes)
        assert len(result["processed_image"]) > 0
        
        # 验证质量评估
        assert isinstance(result["quality_before"], dict)
        assert isinstance(result["quality_after"], dict)
        
        # 验证处理时间
        assert isinstance(result["processing_time"], float)
        assert result["processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_batch_process_images(self, batch_image_paths: list[str]):
        """测试批量图像处理"""
        config = {"brightness_factor": 1.2}
        
        results = await self.service.batch_process_images(
            image_paths=batch_image_paths,
            processing_type=ProcessingType.ENHANCE,
            engine=Engine.PILLOW,
            config=config,
            max_workers=2
        )
        
        # 验证结果数量
        assert len(results) == len(batch_image_paths)
        
        # 验证每个结果
        for i, result in enumerate(results):
            assert "image_path" in result
            assert result["image_path"] == batch_image_paths[i]
            
            if "error" not in result:
                # 成功处理的结果
                assert "processed_image" in result
                assert "processing_time" in result
                assert isinstance(result["processed_image"], bytes)
    
    def test_invalid_image_data(self):
        """测试无效图像数据处理"""
        with pytest.raises(Exception):
            self.service.assess_quality(b"invalid image data")
    
    def test_unsupported_processing_type(self, sample_image: bytes):
        """测试不支持的处理类型"""
        # 这个测试可能需要根据实际实现调整
        with pytest.raises((ValueError, AttributeError)):
            self.service.enhance_image(
                sample_image,
                config={},
                engine="unsupported_engine"
            )
    
    def test_empty_config(self, sample_image: bytes):
        """测试空配置处理"""
        # 应该使用默认参数
        enhanced = self.service.enhance_image(
            sample_image,
            config={},
            engine=Engine.PILLOW
        )
        
        assert isinstance(enhanced, bytes)
        assert len(enhanced) > 0
    
    def test_extreme_config_values(self, sample_image: bytes):
        """测试极端配置值"""
        config = {
            "brightness_factor": 0.1,  # 非常暗
            "contrast_factor": 3.0,    # 高对比度
            "sharpness_factor": 2.0    # 高锐度
        }
        
        # 应该不会崩溃，虽然结果可能不太好
        enhanced = self.service.enhance_image(
            sample_image,
            config=config,
            engine=Engine.PILLOW
        )
        
        assert isinstance(enhanced, bytes)
        assert len(enhanced) > 0