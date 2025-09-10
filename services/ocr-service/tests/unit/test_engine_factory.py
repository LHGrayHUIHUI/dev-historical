"""
OCR引擎工厂单元测试

测试OCR引擎工厂的核心功能，包括：
- 引擎创建和管理
- 性能指标监控
- 健康检查机制
- 引擎选择策略

作者: Quinn (测试架构师)
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.engines.engine_factory import (
    OCREngineFactory, OCREngineType, EnginePerformanceMetrics,
    get_engine_factory, create_engine_from_config
)
from src.engines.base_engine import BaseOCREngine, OCRResult, BoundingBox, TextBlock


class MockOCREngine(BaseOCREngine):
    """用于测试的模拟OCR引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.engine_name = "MockOCREngine"
        self.engine_version = "1.0.0"
        self._should_fail_init = config and config.get("fail_init", False)
        self._should_fail_health = config and config.get("fail_health", False)
    
    async def initialize(self) -> bool:
        if self._should_fail_init:
            return False
        self.is_initialized = True
        return True
    
    async def recognize_async(self, image, **kwargs) -> OCRResult:
        bbox = BoundingBox([(0, 0), (100, 0), (100, 30), (0, 30)], 0.9)
        block = TextBlock("模拟文本", bbox, 0.9, "zh")
        
        return OCRResult(
            text_content="模拟文本",
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
    
    async def health_check(self) -> bool:
        if self._should_fail_health:
            return False
        return await super().health_check()


class TestEnginePerformanceMetrics:
    """引擎性能指标测试"""
    
    def test_create_performance_metrics(self):
        """测试创建性能指标"""
        metrics = EnginePerformanceMetrics(engine_name="test_engine")
        
        assert metrics.engine_name == "test_engine"
        assert metrics.total_recognitions == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.avg_processing_time == 0.0
        assert metrics.success_rate == 0.0
        
    def test_update_metrics_success(self):
        """测试更新成功指标"""
        metrics = EnginePerformanceMetrics(engine_name="test_engine")
        
        # 第一次成功
        metrics.update_metrics(processing_time=1.0, success=True, confidence=0.9)
        
        assert metrics.total_recognitions == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.avg_processing_time == 1.0
        assert metrics.avg_confidence == 0.9
        assert metrics.success_rate == 1.0
        
        # 第二次成功
        metrics.update_metrics(processing_time=2.0, success=True, confidence=0.8)
        
        assert metrics.total_recognitions == 2
        assert metrics.success_count == 2
        assert metrics.avg_processing_time == 1.5
        assert abs(metrics.avg_confidence - 0.85) < 1e-10  # 浮点数精度比较
        assert metrics.success_rate == 1.0
        
    def test_update_metrics_failure(self):
        """测试更新失败指标"""
        metrics = EnginePerformanceMetrics(engine_name="test_engine")
        
        # 一次成功，一次失败
        metrics.update_metrics(processing_time=1.0, success=True, confidence=0.9)
        metrics.update_metrics(processing_time=0.5, success=False, confidence=0.0)
        
        assert metrics.total_recognitions == 2
        assert metrics.success_count == 1
        assert metrics.failure_count == 1
        assert metrics.avg_processing_time == 0.75
        assert metrics.avg_confidence == 0.9  # 只考虑成功的识别
        assert metrics.success_rate == 0.5
        
    def test_metrics_to_dict(self):
        """测试指标字典转换"""
        metrics = EnginePerformanceMetrics(engine_name="test_engine")
        metrics.update_metrics(processing_time=1.0, success=True, confidence=0.9)
        
        result = metrics.to_dict()
        
        assert result['engine_name'] == "test_engine"
        assert result['total_recognitions'] == 1
        assert result['success_count'] == 1
        assert result['avg_processing_time'] == 1.0
        assert result['success_rate'] == 1.0
        assert 'last_used' in result


class TestOCREngineFactory:
    """OCR引擎工厂测试"""
    
    @pytest.fixture
    def factory(self):
        """创建测试用的引擎工厂"""
        return OCREngineFactory()
    
    def test_factory_initialization(self, factory):
        """测试工厂初始化"""
        assert factory._default_engine_type == OCREngineType.PADDLEOCR
        assert len(factory._engine_instances) == 0
        assert len(factory._engine_configs) == 0
        assert len(factory._performance_metrics) == 0
        
    def test_get_supported_engines(self):
        """测试获取支持的引擎列表"""
        supported = OCREngineFactory.get_supported_engines()
        
        assert "paddleocr" in supported
        assert "tesseract" in supported
        assert "easyocr" in supported
        
    def test_validate_engine_type(self):
        """测试验证引擎类型"""
        assert OCREngineFactory.validate_engine_type("paddleocr")
        assert OCREngineFactory.validate_engine_type("TESSERACT")
        assert OCREngineFactory.validate_engine_type("EasyOCR")
        assert not OCREngineFactory.validate_engine_type("invalid_engine")
        
    def test_set_default_engine_type(self, factory):
        """测试设置默认引擎类型"""
        # 使用枚举设置
        factory.set_default_engine_type(OCREngineType.TESSERACT)
        assert factory._default_engine_type == OCREngineType.TESSERACT
        
        # 使用字符串设置
        factory.set_default_engine_type("easyocr")
        assert factory._default_engine_type == OCREngineType.EASYOCR
        
    @pytest.mark.asyncio
    async def test_create_mock_engine(self, factory):
        """测试创建模拟引擎（绕过真实引擎依赖）"""
        # 替换引擎类映射以使用模拟引擎
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            config = {"test_param": "test_value"}
            engine = await factory.create_engine(
                OCREngineType.PADDLEOCR, 
                config, 
                "test_instance"
            )
            
            assert isinstance(engine, MockOCREngine)
            assert engine.is_initialized
            assert "test_instance" in factory._engine_instances
            assert "test_instance" in factory._performance_metrics
            
        finally:
            # 恢复原始映射
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_create_engine_failure(self, factory):
        """测试引擎创建失败"""
        # 替换引擎类映射以使用会失败的模拟引擎
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            config = {"fail_init": True}
            
            with pytest.raises(RuntimeError, match="引擎初始化失败"):
                await factory.create_engine(
                    OCREngineType.PADDLEOCR,
                    config,
                    "failing_instance"
                )
                
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_get_or_create_engine(self, factory):
        """测试获取或创建引擎"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            # 第一次调用应该创建引擎
            engine1 = await factory.get_or_create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="shared_instance"
            )
            
            # 第二次调用应该返回相同的实例
            engine2 = await factory.get_or_create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="shared_instance"
            )
            
            assert engine1 is engine2
            assert len(factory._engine_instances) == 1
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_get_default_engine(self, factory):
        """测试获取默认引擎"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            engine = await factory.get_default_engine()
            
            assert isinstance(engine, MockOCREngine)
            assert engine.is_initialized
            assert "default_paddleocr" in factory._engine_instances
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_remove_engine(self, factory):
        """测试移除引擎"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            # 先创建引擎
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="temp_instance"
            )
            
            assert "temp_instance" in factory._engine_instances
            
            # 移除引擎
            result = await factory.remove_engine("temp_instance")
            
            assert result is True
            assert "temp_instance" not in factory._engine_instances
            assert "temp_instance" not in factory._engine_configs
            assert "temp_instance" not in factory._performance_metrics
            
            # 再次移除应该返回False
            result = await factory.remove_engine("temp_instance")
            assert result is False
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_list_engines(self, factory):
        """测试列出引擎"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            # 创建两个引擎实例
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                {"param1": "value1"},
                "instance1"
            )
            
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                {"param2": "value2"},
                "instance2"
            )
            
            engines = await factory.list_engines()
            
            assert len(engines) == 2
            
            instance_names = [engine['instance_name'] for engine in engines]
            assert "instance1" in instance_names
            assert "instance2" in instance_names
            
            # 检查引擎信息
            for engine_info in engines:
                assert 'engine_type' in engine_info
                assert 'initialized' in engine_info
                assert 'config' in engine_info
                assert 'performance' in engine_info
                
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_health_check_all(self, factory):
        """测试所有引擎健康检查"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            # 创建正常引擎和会失败的引擎
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                {},
                "healthy_instance"
            )
            
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                {"fail_health": True},
                "unhealthy_instance"
            )
            
            results = await factory.health_check_all()
            
            assert len(results) == 2
            assert results["healthy_instance"] is True
            assert results["unhealthy_instance"] is False
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, factory):
        """测试性能指标跟踪"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="perf_instance"
            )
            
            # 更新性能指标
            factory.update_performance_metrics("perf_instance", 1.5, True, 0.9)
            factory.update_performance_metrics("perf_instance", 2.0, True, 0.8)
            factory.update_performance_metrics("perf_instance", 0.5, False, 0.0)
            
            metrics = factory.get_performance_metrics("perf_instance")
            
            assert metrics is not None
            assert metrics.total_recognitions == 3
            assert metrics.success_count == 2
            assert metrics.failure_count == 1
            assert metrics.success_rate == 2/3
            assert metrics.avg_processing_time == (1.5 + 2.0 + 0.5) / 3
            
            # 测试获取所有指标
            all_metrics = factory.get_all_performance_metrics()
            assert "perf_instance" in all_metrics
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_select_best_engine(self, factory):
        """测试选择最佳引擎"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            # 创建多个引擎实例
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="fast_engine"
            )
            
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="accurate_engine"
            )
            
            # 设置不同的性能指标
            # fast_engine: 更快但准确率低
            factory.update_performance_metrics("fast_engine", 0.5, True, 0.7)
            factory.update_performance_metrics("fast_engine", 0.6, True, 0.8)
            
            # accurate_engine: 较慢但准确率高
            factory.update_performance_metrics("accurate_engine", 2.0, True, 0.95)
            factory.update_performance_metrics("accurate_engine", 1.8, True, 0.92)
            
            # 按处理时间选择（fast_engine应该获胜）
            best_name, best_engine = await factory.select_best_engine("avg_processing_time")
            assert best_name == "fast_engine"
            
            # 按置信度选择（accurate_engine应该获胜）
            best_name, best_engine = await factory.select_best_engine("avg_confidence")
            assert best_name == "accurate_engine"
            
            # 按成功率选择（都是100%成功率，应该返回其中一个）
            best_name, best_engine = await factory.select_best_engine("success_rate")
            assert best_name in ["fast_engine", "accurate_engine"]
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            
    @pytest.mark.asyncio
    async def test_cleanup(self, factory):
        """测试清理功能"""
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            # 创建一些引擎实例
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="cleanup_test1"
            )
            
            await factory.create_engine(
                OCREngineType.PADDLEOCR,
                instance_name="cleanup_test2"
            )
            
            assert len(factory._engine_instances) == 2
            
            # 执行清理
            await factory.cleanup()
            
            assert len(factory._engine_instances) == 0
            assert len(factory._engine_configs) == 0
            assert len(factory._performance_metrics) == 0
            
        finally:
            factory.ENGINE_CLASSES = original_classes


class TestGlobalFactory:
    """全局工厂函数测试"""
    
    def test_get_engine_factory(self):
        """测试获取全局引擎工厂"""
        factory1 = get_engine_factory()
        factory2 = get_engine_factory()
        
        # 应该返回相同的实例
        assert factory1 is factory2
        assert isinstance(factory1, OCREngineFactory)
        
    @pytest.mark.asyncio
    async def test_create_engine_from_config(self):
        """测试从配置创建引擎"""
        # 替换全局工厂中的引擎类
        factory = get_engine_factory()
        original_classes = factory.ENGINE_CLASSES.copy()
        factory.ENGINE_CLASSES[OCREngineType.PADDLEOCR] = MockOCREngine
        
        try:
            config = {
                "engine_type": "paddleocr",
                "instance_name": "config_test_instance",
                "param1": "value1",
                "param2": "value2"
            }
            
            engine = await create_engine_from_config(config)
            
            assert isinstance(engine, MockOCREngine)
            assert engine.is_initialized
            
            # 检查配置是否正确传递
            engines_list = await factory.list_engines()
            config_engine = next(
                e for e in engines_list 
                if e['instance_name'] == 'config_test_instance'
            )
            
            assert config_engine['config']['param1'] == 'value1'
            assert config_engine['config']['param2'] == 'value2'
            
        finally:
            factory.ENGINE_CLASSES = original_classes
            await factory.cleanup()


class TestEngineTypes:
    """引擎类型枚举测试"""
    
    def test_engine_type_values(self):
        """测试引擎类型值"""
        assert OCREngineType.PADDLEOCR.value == "paddleocr"
        assert OCREngineType.TESSERACT.value == "tesseract"
        assert OCREngineType.EASYOCR.value == "easyocr"
        
    def test_engine_type_string_conversion(self):
        """测试引擎类型字符串转换"""
        assert OCREngineType("paddleocr") == OCREngineType.PADDLEOCR
        assert OCREngineType("tesseract") == OCREngineType.TESSERACT
        assert OCREngineType("easyocr") == OCREngineType.EASYOCR
        
        with pytest.raises(ValueError):
            OCREngineType("invalid_engine")