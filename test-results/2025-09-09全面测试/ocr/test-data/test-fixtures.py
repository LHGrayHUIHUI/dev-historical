"""
OCR服务测试夹具和模拟数据

基于OCR-service-test-design-20250909.md的comprehensive测试设计，
提供所有测试场景所需的模拟数据、夹具和辅助函数。

Author: Quinn (QA Agent)
Created: 2025-09-09
"""

import base64
import io
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import pytest
import json
from pathlib import Path


class OCRTestFixtures:
    """OCR测试夹具类，提供各种测试数据和模拟对象"""
    
    @staticmethod
    def create_test_image(
        text: str,
        size: Tuple[int, int] = (800, 600),
        background_color: str = "white",
        text_color: str = "black",
        font_size: int = 24
    ) -> bytes:
        """
        创建包含指定文本的测试图像
        
        Args:
            text: 要在图像中显示的文本
            size: 图像大小 (宽度, 高度)
            background_color: 背景颜色
            text_color: 文字颜色
            font_size: 字体大小
            
        Returns:
            bytes: 图像的字节数据
        """
        # 创建图像
        image = Image.new('RGB', size, background_color)
        draw = ImageDraw.Draw(image)
        
        try:
            # 尝试使用默认字体，如果失败则使用PIL默认
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # 计算文字位置（居中）
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # 绘制文字
        draw.text((x, y), text, fill=text_color, font=font)
        
        # 转换为字节
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    @staticmethod
    def create_empty_image(size: Tuple[int, int] = (800, 600)) -> bytes:
        """创建空白图像用于边界情况测试"""
        image = Image.new('RGB', size, 'white')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    @staticmethod
    def create_corrupted_image() -> bytes:
        """创建损坏的图像数据用于错误处理测试"""
        # 创建不完整的JPEG数据
        return b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01'  # JPEG头部但数据不完整
    
    @staticmethod
    def get_test_scenarios() -> Dict[str, Any]:
        """获取所有测试场景配置"""
        return {
            "high_quality_texts": [
                "这是一份清晰的中文文档测试",
                "This is a clear English document test", 
                "混合语言测试 Mixed Language Test 123",
                "发票金额：￥1,234.56 Invoice Amount: $789.00",
                "技术规格 Technical Specifications v2.0.1"
            ],
            "medium_quality_texts": [
                "手机拍照文档质量测试",
                "Slightly blurred text quality test",
                "中等清晰度 Medium clarity test",
                "传真文档 Fax document simulation", 
                "复印件质量 Photocopy quality"
            ],
            "low_quality_texts": [
                "手写笔记测试内容",
                "Faded old document text",
                "倾斜文档内容 Skewed content",
                "低分辨率扫描文字",
                "破损纸张部分文字"
            ],
            "special_characters": [
                "标点符号：，。！？；：''""（）【】",
                "Special chars: @#$%^&*()_+-=[]{}|;:',.<>?",
                "数字测试：0123456789 ￥$€£¥",
                "技术符号：<>{}[]()/*-+=|\\",
                "中文符号：《》「」、·¨ˇˉ˘˙˚"
            ]
        }
    
    @staticmethod
    def get_validation_test_data() -> Dict[str, Any]:
        """获取参数验证测试数据"""
        return {
            "valid_confidence_thresholds": [0.0, 0.1, 0.5, 0.8, 0.9, 1.0],
            "invalid_confidence_thresholds": [-0.1, 1.1, "invalid", None],
            "valid_language_codes": ["zh", "en", "zh,en", "zh,en,ja", "zh, en"],
            "invalid_language_codes": ["", "invalid_lang", "zh,,en", None],
            "valid_engines": ["paddleocr", "tesseract", "easyocr"],
            "invalid_engines": ["unknown_engine", "", None],
            "valid_file_sizes": [1024, 1048576, 10485760],  # 1KB, 1MB, 10MB
            "invalid_file_sizes": [0, 104857600],  # 0B, 100MB
        }


class MockOCRResponse:
    """模拟OCR响应对象"""
    
    def __init__(self, success: bool = True, **kwargs):
        self.success = success
        self.text_content = kwargs.get('text_content', "模拟OCR识别结果")
        self.confidence = kwargs.get('confidence', 0.85)
        self.processing_time = kwargs.get('processing_time', 2.5)
        self.language_detected = kwargs.get('language_detected', "zh,en")
        self.word_count = kwargs.get('word_count', 10)
        self.char_count = kwargs.get('char_count', 50)
        self.bounding_boxes = kwargs.get('bounding_boxes', [])
        self.text_blocks = kwargs.get('text_blocks', [])
        self.error_message = kwargs.get('error_message', None)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "success": self.success,
            "text_content": self.text_content,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "language_detected": self.language_detected,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "bounding_boxes": self.bounding_boxes,
            "text_blocks": self.text_blocks
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result


class MockStorageServiceClient:
    """模拟Storage Service客户端"""
    
    def __init__(self):
        self.tasks = {}
        self.results = {}
        
    async def create_ocr_task(self, task_request) -> str:
        """创建OCR任务"""
        task_id = f"mock_task_{len(self.tasks) + 1:04d}"
        self.tasks[task_id] = {
            "status": "pending",
            "request": task_request,
            "created_at": "2025-09-09T12:00:00Z"
        }
        return task_id
    
    async def update_ocr_task_status(self, task_id: str, status: str, error_message: str = None):
        """更新任务状态"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            if error_message:
                self.tasks[task_id]["error_message"] = error_message
    
    async def save_ocr_result(self, task_id: str, result_data: Dict[str, Any]):
        """保存OCR结果"""
        self.results[task_id] = result_data
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
    
    async def get_ocr_result(self, task_id: str):
        """获取OCR结果"""
        if task_id not in self.tasks:
            return None
            
        task = self.tasks[task_id]
        result = self.results.get(task_id, {})
        
        class MockResult:
            def __init__(self, task_id, task_data, result_data):
                self.task_id = task_id
                self.status = task_data["status"]
                self.text_content = result_data.get("text_content", "")
                self.confidence_scores = result_data.get("confidence", 0.0)
                self.processing_time = result_data.get("processing_time", 0.0)
                self.error_message = task_data.get("error_message")
                self.created_at = task_data.get("created_at")
                self.completed_at = task_data.get("completed_at")
        
        return MockResult(task_id, task, result)


# Pytest夹具定义
@pytest.fixture
def ocr_fixtures():
    """OCR测试夹具"""
    return OCRTestFixtures()


@pytest.fixture
def mock_storage_client():
    """模拟Storage Service客户端夹具"""
    return MockStorageServiceClient()


@pytest.fixture
def test_images(ocr_fixtures):
    """测试图像数据夹具"""
    scenarios = ocr_fixtures.get_test_scenarios()
    images = {}
    
    # 高质量图像
    for i, text in enumerate(scenarios["high_quality_texts"]):
        images[f"high_quality_{i+1}.jpg"] = ocr_fixtures.create_test_image(
            text, size=(1200, 800), font_size=32
        )
    
    # 中等质量图像
    for i, text in enumerate(scenarios["medium_quality_texts"]):
        images[f"medium_quality_{i+1}.jpg"] = ocr_fixtures.create_test_image(
            text, size=(800, 600), font_size=24
        )
    
    # 低质量图像
    for i, text in enumerate(scenarios["low_quality_texts"]):
        images[f"low_quality_{i+1}.jpg"] = ocr_fixtures.create_test_image(
            text, size=(640, 480), font_size=16
        )
    
    # 特殊字符图像
    for i, text in enumerate(scenarios["special_characters"]):
        images[f"special_chars_{i+1}.jpg"] = ocr_fixtures.create_test_image(
            text, size=(1000, 600), font_size=28
        )
    
    # 边界情况图像
    images["empty.jpg"] = ocr_fixtures.create_empty_image()
    images["corrupted.jpg"] = ocr_fixtures.create_corrupted_image()
    
    return images


@pytest.fixture
def validation_test_data(ocr_fixtures):
    """参数验证测试数据夹具"""
    return ocr_fixtures.get_validation_test_data()


@pytest.fixture
def expected_api_responses():
    """预期API响应模板夹具"""
    return {
        "successful_recognition": {
            "success": True,
            "message": "OCR识别完成",
            "data": {
                "text_content": "测试文本内容",
                "confidence": 0.85,
                "processing_time": 2.5,
                "language_detected": "zh,en",
                "word_count": 10,
                "char_count": 50,
                "bounding_boxes": [],
                "text_blocks": [],
                "async_mode": False
            }
        },
        "async_task_created": {
            "success": True,
            "message": "任务已创建，异步处理中",
            "data": {
                "task_id": "test_task_001",
                "status": "processing",
                "async_mode": True
            }
        },
        "task_status_completed": {
            "success": True,
            "message": "任务状态查询成功",
            "data": {
                "task_id": "test_task_001",
                "status": "completed",
                "text_content": "异步任务识别结果",
                "confidence_scores": 0.88,
                "processing_time": 3.2,
                "error_message": None,
                "created_at": "2025-09-09T12:00:00Z",
                "completed_at": "2025-09-09T12:00:03Z"
            }
        },
        "batch_processing_result": {
            "success": True,
            "message": "批量识别完成，成功: 4, 失败: 1",
            "data": {
                "total_files": 5,
                "successful_count": 4,
                "failed_count": 1,
                "results": []
            }
        },
        "available_engines": {
            "success": True,
            "message": "获取可用引擎成功",
            "data": {
                "available_engines": ["paddleocr", "tesseract", "easyocr"],
                "default_engine": "paddleocr",
                "engine_configs": {},
                "total_count": 3
            }
        },
        "health_check_healthy": {
            "success": True,
            "message": "健康检查完成",
            "data": {
                "status": "healthy",
                "engines_status": {
                    "paddleocr": "available",
                    "tesseract": "available",
                    "easyocr": "available"
                },
                "uptime": 3600,
                "version": "2.0.0"
            }
        },
        "error_invalid_file_format": {
            "success": False,
            "message": "只支持图像文件",
            "data": None
        },
        "error_file_too_large": {
            "success": False,
            "message": "文件大小不能超过10MB",
            "data": None
        },
        "error_unsupported_engine": {
            "success": False,
            "message": "不支持的OCR引擎: unknown_engine, 可用引擎: ['paddleocr', 'tesseract', 'easyocr']",
            "data": None
        }
    }


# 辅助函数
def create_mock_upload_file(filename: str, content: bytes, content_type: str):
    """创建模拟的UploadFile对象"""
    from fastapi import UploadFile
    from io import BytesIO
    
    file_obj = BytesIO(content)
    file_obj.seek(0)
    
    upload_file = UploadFile(
        filename=filename,
        file=file_obj,
        size=len(content),
        headers={"content-type": content_type}
    )
    upload_file.content_type = content_type
    return upload_file


def assert_ocr_response_structure(response_data: Dict[str, Any], expected_fields: List[str]):
    """验证OCR响应结构"""
    assert "success" in response_data, "响应缺少success字段"
    assert "message" in response_data, "响应缺少message字段"
    
    if response_data["success"] and "data" in response_data:
        data = response_data["data"]
        for field in expected_fields:
            assert field in data, f"响应数据缺少{field}字段"


def calculate_text_accuracy(expected: str, actual: str) -> float:
    """计算文本识别准确率"""
    if not expected or not actual:
        return 0.0
    
    # 简化的字符级准确率计算
    expected_chars = set(expected)
    actual_chars = set(actual)
    
    intersection = expected_chars & actual_chars
    union = expected_chars | actual_chars
    
    if not union:
        return 1.0
    
    return len(intersection) / len(union)


def generate_test_report(test_results: List[Dict[str, Any]], output_path: str):
    """生成测试报告"""
    report = {
        "test_execution_time": "2025-09-09T12:00:00Z",
        "total_tests": len(test_results),
        "passed_tests": sum(1 for r in test_results if r.get("status") == "passed"),
        "failed_tests": sum(1 for r in test_results if r.get("status") == "failed"),
        "test_results": test_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report