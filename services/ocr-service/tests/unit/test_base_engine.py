"""
OCR基础引擎单元测试

测试OCR引擎抽象基类的核心功能，包括：
- 数据结构（BoundingBox, TextBlock, OCRResult）
- BaseOCREngine抽象类的通用方法
- 配置管理和验证
- 图像预处理功能

作者: Quinn (测试架构师)
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import os

from src.engines.base_engine import (
    BoundingBox, TextBlock, OCRResult, BaseOCREngine
)


class TestBoundingBox:
    """边界框数据结构测试"""
    
    def test_create_rectangle_bbox(self):
        """测试创建矩形边界框"""
        points = [(0, 0), (100, 0), (100, 50), (0, 50)]
        bbox = BoundingBox(points=points, confidence=0.95)
        
        assert bbox.points == points
        assert bbox.confidence == 0.95
        assert bbox.shape_type == "rectangle"
        
    def test_bbox_properties(self):
        """测试边界框属性计算"""
        points = [(10, 20), (110, 20), (110, 70), (10, 70)]
        bbox = BoundingBox(points=points, confidence=0.85)
        
        assert bbox.x_min == 10
        assert bbox.y_min == 20
        assert bbox.x_max == 110
        assert bbox.y_max == 70
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000
        
    def test_polygon_bbox(self):
        """测试多边形边界框"""
        points = [(0, 10), (50, 0), (100, 10), (90, 60), (10, 60)]
        bbox = BoundingBox(
            points=points, 
            confidence=0.90, 
            shape_type="polygon"
        )
        
        assert bbox.shape_type == "polygon"
        assert bbox.x_min == 0
        assert bbox.x_max == 100
        assert bbox.y_min == 0
        assert bbox.y_max == 60
        
    def test_bbox_to_dict(self):
        """测试边界框字典转换"""
        points = [(0, 0), (50, 0), (50, 30), (0, 30)]
        bbox = BoundingBox(points=points, confidence=0.88)
        
        result = bbox.to_dict()
        
        assert result['points'] == points
        assert result['confidence'] == 0.88
        assert result['width'] == 50
        assert result['height'] == 30
        assert 'area' in result
        
    def test_empty_bbox(self):
        """测试空边界框"""
        points = []
        
        with pytest.raises(ValueError):
            bbox = BoundingBox(points=points)
            # 访问属性时应该抛出异常
            _ = bbox.x_min


class TestTextBlock:
    """文本块数据结构测试"""
    
    def test_create_text_block(self):
        """测试创建文本块"""
        points = [(0, 0), (100, 0), (100, 30), (0, 30)]
        bbox = BoundingBox(points=points, confidence=0.92)
        
        block = TextBlock(
            text="测试文本",
            bbox=bbox,
            confidence=0.90,
            language="zh"
        )
        
        assert block.text == "测试文本"
        assert block.confidence == 0.90
        assert block.language == "zh"
        assert block.length == 4
        assert not block.is_empty
        
    def test_empty_text_block(self):
        """测试空文本块"""
        points = [(0, 0), (50, 0), (50, 20), (0, 20)]
        bbox = BoundingBox(points=points, confidence=0.1)
        
        block = TextBlock(text="", bbox=bbox, confidence=0.1)
        
        assert block.is_empty
        assert block.length == 0
        assert block.word_count == 0
        
    def test_text_block_cleanup(self):
        """测试文本块内容清理"""
        points = [(0, 0), (80, 0), (80, 25), (0, 25)]
        bbox = BoundingBox(points=points, confidence=0.85)
        
        # 测试None文本处理
        block1 = TextBlock(text=None, bbox=bbox, confidence=0.85)
        assert block1.text == ""
        
        # 测试空白文本清理
        block2 = TextBlock(text="  \n  测试文本  \n  ", bbox=bbox, confidence=0.85)
        assert block2.text == "测试文本"
        
    def test_text_block_properties(self):
        """测试文本块属性"""
        points = [(0, 0), (120, 0), (120, 40), (0, 40)]
        bbox = BoundingBox(points=points, confidence=0.88)
        
        block = TextBlock(
            text="这是一个测试 文档",
            bbox=bbox,
            confidence=0.88,
            language="zh",
            angle=5.0
        )
        
        assert block.word_count == 2  # 按空格分割
        assert block.angle == 5.0
        
    def test_text_block_to_dict(self):
        """测试文本块字典转换"""
        points = [(5, 5), (95, 5), (95, 35), (5, 35)]
        bbox = BoundingBox(points=points, confidence=0.93)
        
        block = TextBlock(
            text="字典转换测试",
            bbox=bbox,
            confidence=0.93,
            language="zh",
            char_details=[{"char": "字", "confidence": 0.95}]
        )
        
        result = block.to_dict()
        
        assert result['text'] == "字典转换测试"
        assert result['confidence'] == 0.93
        assert result['language'] == "zh"
        assert result['length'] == 6
        assert 'bbox' in result
        assert 'char_details' in result


class TestOCRResult:
    """OCR结果数据结构测试"""
    
    def test_create_ocr_result(self):
        """测试创建OCR结果"""
        # 创建测试文本块
        bbox1 = BoundingBox([(0, 0), (50, 0), (50, 20), (0, 20)], 0.95)
        bbox2 = BoundingBox([(60, 0), (110, 0), (110, 20), (60, 20)], 0.90)
        
        blocks = [
            TextBlock("测试", bbox1, 0.95, "zh"),
            TextBlock("文档", bbox2, 0.90, "zh")
        ]
        
        result = OCRResult(
            text_content="测试文档",
            text_blocks=blocks,
            confidence_score=0.925,
            language_detected="zh",
            processing_time=1.5,
            engine_name="TestEngine",
            image_size=(200, 100)
        )
        
        assert result.text_content == "测试文档"
        assert len(result.text_blocks) == 2
        assert result.confidence_score == 0.925
        assert result.char_count == 4
        assert result.word_count == 2  # 两个文本块，每个块算作1个词
        assert result.block_count == 2
        assert not result.is_empty
        
    def test_auto_generate_text_content(self):
        """测试自动生成文本内容"""
        bbox1 = BoundingBox([(0, 0), (40, 0), (40, 20), (0, 20)], 0.92)
        bbox2 = BoundingBox([(0, 25), (60, 25), (60, 45), (0, 45)], 0.88)
        
        blocks = [
            TextBlock("第一行", bbox1, 0.92, "zh"),
            TextBlock("第二行", bbox2, 0.88, "zh")
        ]
        
        # 不提供text_content，让系统自动生成
        result = OCRResult(
            text_content="",  # 空文本
            text_blocks=blocks,
            confidence_score=0.0  # 0分让系统自动计算
        )
        
        # 验证自动生成的内容
        assert "第一行" in result.text_content
        assert "第二行" in result.text_content
        assert result.confidence_score == 0.9  # (0.92 + 0.88) / 2
        
    def test_filter_by_confidence(self):
        """测试按置信度筛选文本"""
        bbox1 = BoundingBox([(0, 0), (30, 0), (30, 20), (0, 20)], 0.95)
        bbox2 = BoundingBox([(35, 0), (65, 0), (65, 20), (35, 20)], 0.60)
        bbox3 = BoundingBox([(70, 0), (100, 0), (100, 20), (70, 20)], 0.85)
        
        blocks = [
            TextBlock("高质量", bbox1, 0.95, "zh"),
            TextBlock("低质量", bbox2, 0.60, "zh"),
            TextBlock("中质量", bbox3, 0.85, "zh")
        ]
        
        result = OCRResult(
            text_content="高质量低质量中质量",
            text_blocks=blocks,
            confidence_score=0.8
        )
        
        # 筛选置信度 >= 0.8 的文本
        filtered_text = result.get_text_by_confidence(min_confidence=0.8)
        
        assert "高质量" in filtered_text
        assert "低质量" not in filtered_text
        assert "中质量" in filtered_text
        
    def test_empty_ocr_result(self):
        """测试空OCR结果"""
        result = OCRResult(
            text_content="",
            text_blocks=[],
            confidence_score=0.0
        )
        
        assert result.is_empty
        assert result.char_count == 0
        assert result.word_count == 0
        assert result.block_count == 0
        
    def test_ocr_result_to_dict(self):
        """测试OCR结果字典转换"""
        bbox = BoundingBox([(0, 0), (100, 0), (100, 25), (0, 25)], 0.90)
        block = TextBlock("测试转换", bbox, 0.90, "zh")
        
        result = OCRResult(
            text_content="测试转换",
            text_blocks=[block],
            confidence_score=0.90,
            language_detected="zh",
            engine_name="TestEngine",
            processing_time=0.8
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['text_content'] == "测试转换"
        assert result_dict['confidence_score'] == 0.90
        assert result_dict['language_detected'] == "zh"
        assert result_dict['processing_time'] == 0.8
        assert 'text_blocks' in result_dict
        assert 'bounding_boxes' in result_dict


class TestBaseOCREngine:
    """OCR基础引擎测试"""
    
    class MockOCREngine(BaseOCREngine):
        """用于测试的模拟OCR引擎"""
        
        async def initialize(self) -> bool:
            self.is_initialized = True
            self.engine_version = "1.0.0"
            return True
            
        async def recognize_async(self, image, **kwargs) -> OCRResult:
            bbox = BoundingBox([(0, 0), (100, 0), (100, 30), (0, 30)], 0.9)
            block = TextBlock("模拟文本", bbox, 0.9, "zh")
            
            return OCRResult(
                text_content="模拟文本",
                text_blocks=[block],
                confidence_score=0.9,
                engine_name=self.engine_name,
                engine_version=self.engine_version
            )
            
        def get_supported_languages(self) -> list:
            return ["zh", "en"]
            
        def get_engine_info(self) -> dict:
            return {
                "name": self.engine_name,
                "version": self.engine_version,
                "languages": self.get_supported_languages()
            }
            
        def get_required_config_fields(self) -> list:
            return ["language", "confidence_threshold"]
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        config = {"language": "zh", "confidence_threshold": 0.8}
        engine = self.MockOCREngine(config=config)
        
        assert engine.config == config
        assert not engine.is_initialized
        assert engine.engine_name == "MockOCREngine"
        
    @pytest.mark.asyncio
    async def test_async_initialize(self):
        """测试异步初始化"""
        engine = self.MockOCREngine()
        
        result = await engine.initialize()
        
        assert result is True
        assert engine.is_initialized
        assert engine.engine_version == "1.0.0"
        
    @pytest.mark.asyncio
    async def test_health_check(self):
        """测试健康检查"""
        engine = self.MockOCREngine()
        
        # 未初始化时的健康检查
        result = await engine.health_check()
        assert result is True
        assert engine.is_initialized
        
        # 已初始化时的健康检查
        result = await engine.health_check()
        assert result is True
        
    def test_config_validation(self):
        """测试配置验证"""
        engine = self.MockOCREngine()
        
        # 有效配置
        valid_config = {
            "language": "zh",
            "confidence_threshold": 0.8
        }
        assert engine.validate_config(valid_config) is True
        
        # 无效配置（缺少必需字段）
        invalid_config = {"language": "zh"}
        assert engine.validate_config(invalid_config) is False
        
    def test_config_update(self):
        """测试配置更新"""
        initial_config = {"language": "zh", "confidence_threshold": 0.8}
        engine = self.MockOCREngine(config=initial_config)
        
        # 有效更新（包含所有必需字段）
        new_config = {"language": "zh", "confidence_threshold": 0.9}
        result = engine.update_config(new_config)
        
        assert result is True
        assert engine.config["confidence_threshold"] == 0.9
        assert engine.config["language"] == "zh"  # 保持原有配置
        
    def test_image_preprocessing_pil(self):
        """测试PIL图像预处理"""
        engine = self.MockOCREngine()
        
        # 创建测试图像
        test_image = Image.new('RGB', (100, 50), color='white')
        
        result = engine._preprocess_image(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 100, 3)  # height, width, channels
        
    def test_image_preprocessing_numpy(self):
        """测试numpy数组预处理"""
        engine = self.MockOCREngine()
        
        # 创建测试数组
        test_array = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        result = engine._preprocess_image(test_array)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, test_array)
        
    def test_image_preprocessing_file_path(self):
        """测试文件路径预处理"""
        engine = self.MockOCREngine()
        
        # 创建临时图像文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            test_image = Image.new('RGB', (80, 40), color='red')
            test_image.save(f.name, 'PNG')
            
            try:
                result = engine._preprocess_image(f.name)
                
                assert isinstance(result, np.ndarray)
                assert result.shape == (40, 80, 3)
                
            finally:
                os.unlink(f.name)
                
    def test_image_preprocessing_invalid_format(self):
        """测试无效图像格式预处理"""
        engine = self.MockOCREngine()
        
        with pytest.raises(ValueError, match="不支持的图像格式"):
            engine._preprocess_image({"invalid": "format"})
            
    def test_create_default_result(self):
        """测试创建默认结果"""
        engine = self.MockOCREngine()
        
        result = engine._create_default_result("测试错误")
        
        assert isinstance(result, OCRResult)
        assert result.is_empty
        assert result.confidence_score == 0.0
        assert result.engine_name == "MockOCREngine"
        assert result.metadata['error'] == "测试错误"
        
    @pytest.mark.asyncio
    async def test_async_recognition(self):
        """测试异步识别功能"""
        engine = self.MockOCREngine()
        await engine.initialize()
        
        test_image = Image.new('RGB', (200, 100), color='white')
        result = await engine.recognize_async(test_image)
        
        assert isinstance(result, OCRResult)
        assert result.text_content == "模拟文本"
        assert result.confidence_score == 0.9
        assert result.engine_name == "MockOCREngine"
        
    def test_engine_info(self):
        """测试引擎信息获取"""
        engine = self.MockOCREngine()
        engine.engine_version = "1.0.0"
        
        info = engine.get_engine_info()
        
        assert info['name'] == "MockOCREngine"
        assert info['version'] == "1.0.0"
        assert "zh" in info['languages']
        assert "en" in info['languages']
        
    def test_supported_languages(self):
        """测试支持的语言列表"""
        engine = self.MockOCREngine()
        
        languages = engine.get_supported_languages()
        
        assert isinstance(languages, list)
        assert "zh" in languages
        assert "en" in languages
        
    def test_engine_repr(self):
        """测试引擎字符串表示"""
        engine = self.MockOCREngine()
        
        repr_str = repr(engine)
        
        assert "MockOCREngine" in repr_str
        assert "initialized=False" in repr_str
        
        # 初始化后再测试
        engine.is_initialized = True
        repr_str = repr(engine)
        assert "initialized=True" in repr_str