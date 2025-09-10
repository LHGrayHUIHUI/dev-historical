"""
OCR服务测试配置文件
设置测试环境、夹具和通用测试工具
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock
import tempfile
import os
from PIL import Image
import numpy as np

# 测试配置
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_ocr_engine():
    """模拟OCR引擎"""
    engine = AsyncMock()
    engine.extract_text.return_value = "测试文本内容"
    engine.detect_language.return_value = "zh"
    engine.get_confidence.return_value = 0.95
    engine.get_bounding_boxes.return_value = [
        {"text": "测试", "bbox": [0, 0, 100, 50], "confidence": 0.95},
        {"text": "文本", "bbox": [100, 0, 200, 50], "confidence": 0.92}
    ]
    return engine


@pytest.fixture
def sample_image():
    """创建示例测试图像"""
    # 创建一个简单的测试图像
    image = Image.new('RGB', (400, 200), color='white')
    return image


@pytest.fixture
def sample_image_file():
    """创建临时图像文件"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        image = Image.new('RGB', (400, 200), color='white')
        image.save(f.name, 'PNG')
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_chinese_image():
    """创建包含中文文本的测试图像"""
    image = Image.new('RGB', (600, 200), color='white')
    # 这里应该创建包含中文文本的图像，但为了简化，只返回白色图像
    return image


@pytest.fixture
def sample_ancient_text_image():
    """创建包含古代文本的测试图像"""
    image = Image.new('RGB', (800, 400), color='white')
    # 这里应该创建包含古代文本的图像，但为了简化，只返回白色图像
    return image


@pytest.fixture
def mock_paddleocr():
    """模拟PaddleOCR引擎"""
    mock = Mock()
    mock.ocr.return_value = [[
        [[0, 0, 100, 0, 100, 50, 0, 50], ("测试文本", 0.95)]
    ]]
    return mock


@pytest.fixture
def mock_tesseract():
    """模拟Tesseract引擎"""
    mock = Mock()
    mock.image_to_string.return_value = "测试文本"
    mock.image_to_data.return_value = {
        'text': ['', '测试', '文本'],
        'conf': ['-1', '95', '92'],
        'left': [0, 0, 100],
        'top': [0, 0, 0],
        'width': [400, 100, 100],
        'height': [200, 50, 50]
    }
    return mock


@pytest.fixture
def mock_easyocr():
    """模拟EasyOCR引擎"""
    mock = Mock()
    mock.readtext.return_value = [
        ([[0, 0], [100, 0], [100, 50], [0, 50]], "测试", 0.95),
        ([[100, 0], [200, 0], [200, 50], [100, 50]], "文本", 0.92)
    ]
    return mock


@pytest.fixture
def test_config():
    """测试配置"""
    return {
        "ocr_engines": ["paddleocr", "tesseract", "easyocr"],
        "default_engine": "paddleocr",
        "languages": ["chi_sim", "eng"],
        "confidence_threshold": 0.8,
        "max_image_size": 10485760,  # 10MB
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "preprocessing_enabled": True,
        "enhancement_enabled": True
    }


@pytest.fixture
def performance_benchmark():
    """性能基准数据"""
    return {
        "single_image_processing_time": 2.0,  # 秒
        "batch_processing_rate": 10,  # 图像/分钟
        "memory_usage_limit": 512,  # MB
        "accuracy_threshold": {
            "modern_text": 0.90,
            "ancient_text": 0.85,
            "handwritten_text": 0.75
        }
    }


@pytest.fixture
def test_data_directory():
    """测试数据目录路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "fixtures", "test_images")


@pytest.fixture
def accuracy_test_cases():
    """准确性测试用例"""
    return [
        {
            "name": "modern_chinese_document",
            "image_path": "modern_chinese.png",
            "expected_text": "这是一个现代中文文档测试",
            "min_accuracy": 0.90
        },
        {
            "name": "ancient_chinese_text",
            "image_path": "ancient_chinese.png", 
            "expected_text": "古代漢字測試文本",
            "min_accuracy": 0.85
        },
        {
            "name": "mixed_language_text",
            "image_path": "mixed_language.png",
            "expected_text": "Mixed 中英文 Text 测试",
            "min_accuracy": 0.80
        }
    ]