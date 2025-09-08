"""
OCR引擎模块

提供各种OCR引擎的统一接口和实现，包括PaddleOCR、
Tesseract、EasyOCR等主流OCR引擎。

本模块定义了OCR引擎的抽象基类和具体实现类，
为上层服务提供一致的OCR识别接口。

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from .base_engine import BaseOCREngine, OCRResult
from .paddleocr_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine
from .easyocr_engine import EasyOCREngine
from .engine_factory import OCREngineFactory

__all__ = [
    'BaseOCREngine',
    'OCRResult',
    'PaddleOCREngine',
    'TesseractEngine', 
    'EasyOCREngine',
    'OCREngineFactory'
]