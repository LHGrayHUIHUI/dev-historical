"""
OCR引擎简化集成测试

简化的集成测试，避免复杂的异步fixture问题
测试核心集成功能

作者: Quinn (测试架构师)
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import os
from PIL import Image
import numpy as np

from src.engines.engine_factory import OCREngineFactory, OCREngineType
from src.engines.base_engine import BaseOCREngine, OCRResult, BoundingBox, TextBlock


class SimpleIntegrationEngine(BaseOCREngine):
    """用于集成测试的简单模拟引擎"""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.engine_name = "SimpleIntegrationEngine"
        self.engine_version = "1.0.0"
    
    async def initialize(self) -> bool:
        self.is_initialized = True
        return True
    
    async def recognize_async(self, image, **kwargs) -> OCRResult:
        bbox = BoundingBox([(0, 0), (100, 0), (100, 30), (0, 30)], 0.9)
        block = TextBlock("集成测试文本", bbox, 0.9, "zh")
        
        return OCRResult(
            text_content="集成测试文本",
            text_blocks=[block],
            confidence_score=0.9,
            engine_name=self.engine_name,
            processing_time=0.1
        )
    
    def get_supported_languages(self) -> list:
        return ["zh", "en"]
    
    def get_engine_info(self) -> dict:
        return {
            "name": self.engine_name,
            "version": self.engine_version,
            "languages": self.get_supported_languages()
        }


class TestSimpleIntegration:
    """简化的集成测试"""
    
    @pytest.mark.asyncio
    async def test_engine_factory_basic_integration(self):
        """测试引擎工厂基本集成功能"""
        factory = OCREngineFactory()
        
        # 替换引擎类为测试引擎
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = SimpleIntegrationEngine
        
        try:
            # 创建引擎
            engine = await factory.create_engine(
                engine_type=OCREngineType.PADDLEOCR,
                instance_name="integration_test_engine"
            )
            
            # 验证引擎创建成功
            assert engine is not None
            assert engine.is_initialized
            assert engine.engine_name == "SimpleIntegrationEngine"
            
            # 测试引擎识别功能
            test_image = Image.new('RGB', (200, 100), color='white')
            result = await engine.recognize_async(test_image)
            
            # 验证识别结果
            assert isinstance(result, OCRResult)
            assert result.text_content == "集成测试文本"
            assert result.confidence_score == 0.9
            
            # 测试引擎列表功能
            engine_list = await factory.list_engines()
            assert len(engine_list) == 1
            assert engine_list[0]['instance_name'] == "integration_test_engine"
            
            # 测试健康检查
            health_results = await factory.health_check_all()
            assert health_results["integration_test_engine"] is True
            
        finally:
            # 恢复原始类并清理
            factory.ENGINE_CLASSES = original_classes
            await factory.cleanup()
    
    @pytest.mark.asyncio
    async def test_multiple_engines_creation(self):
        """测试创建多个引擎实例"""
        factory = OCREngineFactory()
        
        # 替换引擎类
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = SimpleIntegrationEngine
        factory.ENGINE_CLASSES[OCREngineType.TESSERACT] = SimpleIntegrationEngine
        
        try:
            # 创建多个引擎
            engine1 = await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="engine1"
            )
            
            engine2 = await factory.create_engine(
                OCREngineType.TESSERACT,
                instance_name="engine2"
            )
            
            # 验证两个引擎都创建成功
            assert engine1.is_initialized
            assert engine2.is_initialized
            
            # 验证引擎数量
            engine_list = await factory.list_engines()
            assert len(engine_list) == 2
            
            # 验证健康检查
            health_results = await factory.health_check_all()
            assert len(health_results) == 2
            assert all(health_results.values())
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            await factory.cleanup()
    
    @pytest.mark.asyncio
    async def test_engine_performance_metrics_integration(self):
        """测试性能指标集成"""
        factory = OCREngineFactory()
        
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = SimpleIntegrationEngine
        
        try:
            engine = await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="perf_engine"
            )
            
            # 执行识别任务
            test_image = Image.new('RGB', (150, 75), color='white')
            result = await engine.recognize_async(test_image)
            
            # 更新性能指标
            factory.update_performance_metrics(
                "perf_engine",
                processing_time=result.processing_time,
                success=True,
                confidence=result.confidence_score
            )
            
            # 验证性能指标
            metrics = factory.get_performance_metrics("perf_engine")
            assert metrics is not None
            assert metrics.total_recognitions == 1
            assert metrics.success_count == 1
            assert metrics.success_rate == 1.0
            
            # 获取所有性能指标
            all_metrics = factory.get_all_performance_metrics()
            assert "perf_engine" in all_metrics
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            await factory.cleanup()
    
    @pytest.mark.asyncio
    async def test_image_format_processing(self):
        """测试不同图像格式处理"""
        engine = SimpleIntegrationEngine()
        await engine.initialize()
        
        # 测试PIL图像
        pil_image = Image.new('RGB', (100, 50), color='red')
        result = await engine.recognize_async(pil_image)
        assert isinstance(result, OCRResult)
        assert result.text_content == "集成测试文本"
        
        # 测试numpy数组
        np_image = np.array(pil_image)
        result = await engine.recognize_async(np_image)
        assert isinstance(result, OCRResult)
        
        # 测试文件路径
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            pil_image.save(f.name, 'PNG')
            
            try:
                result = await engine.recognize_async(f.name)
                assert isinstance(result, OCRResult)
                
            finally:
                os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_engine_configuration_integration(self):
        """测试引擎配置集成"""
        config = {
            "language": "zh",
            "confidence_threshold": 0.8
        }
        
        engine = SimpleIntegrationEngine(config)
        await engine.initialize()
        
        # 验证配置
        assert engine.config["language"] == "zh"
        assert engine.config["confidence_threshold"] == 0.8
        
        # 测试配置更新
        new_config = {
            "language": "en", 
            "confidence_threshold": 0.9
        }
        
        result = engine.update_config(new_config)
        assert result is True
        assert engine.config["language"] == "en"
        assert engine.config["confidence_threshold"] == 0.9
    
    @pytest.mark.asyncio 
    async def test_concurrent_processing(self):
        """测试并发处理能力"""
        factory = OCREngineFactory()
        
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = SimpleIntegrationEngine
        
        try:
            # 创建引擎
            engine = await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="concurrent_engine"
            )
            
            # 创建多个并发任务
            test_image = Image.new('RGB', (100, 50), color='blue')
            tasks = []
            
            for i in range(5):
                task = engine.recognize_async(test_image)
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks)
            
            # 验证结果
            assert len(results) == 5
            for result in results:
                assert isinstance(result, OCRResult)
                assert result.text_content == "集成测试文本"
                assert result.confidence_score == 0.9
                
        finally:
            factory.ENGINE_CLASSES = original_classes
            await factory.cleanup()
    
    def test_engine_info_integration(self):
        """测试引擎信息集成"""
        engine = SimpleIntegrationEngine()
        
        # 测试引擎信息
        info = engine.get_engine_info()
        assert info["name"] == "SimpleIntegrationEngine"
        assert info["version"] == "1.0.0"
        assert "languages" in info
        
        # 测试支持的语言
        languages = engine.get_supported_languages()
        assert "zh" in languages
        assert "en" in languages
    
    def test_factory_supported_engines(self):
        """测试工厂支持的引擎类型"""
        supported = OCREngineFactory.get_supported_engines()
        
        assert "paddleocr" in supported
        assert "tesseract" in supported
        assert "easyocr" in supported
        
        # 测试引擎类型验证
        assert OCREngineFactory.validate_engine_type("paddleocr") is True
        assert OCREngineFactory.validate_engine_type("invalid_engine") is False