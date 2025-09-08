"""
OCR引擎抽象基类

定义OCR引擎的标准接口和数据结构，为所有具体的OCR引擎实现
提供统一的抽象基类。确保所有引擎实现的一致性和可扩展性。

主要功能：
- 定义OCR引擎抽象接口
- 标准化OCR结果数据结构
- 提供通用的错误处理机制
- 定义配置和初始化规范

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """
    文字区域边界框
    
    表示识别到的文字在图像中的位置坐标，
    支持矩形和多边形两种表示方式。
    """
    # 坐标点列表，支持4点矩形或更多点的多边形
    points: List[Tuple[float, float]]
    
    # 置信度分数
    confidence: float = 0.0
    
    # 边界框类型标识
    shape_type: str = "rectangle"  # rectangle, polygon
    
    @property
    def x_min(self) -> float:
        """获取最小X坐标"""
        return min(point[0] for point in self.points)
    
    @property
    def y_min(self) -> float:
        """获取最小Y坐标"""
        return min(point[1] for point in self.points)
    
    @property
    def x_max(self) -> float:
        """获取最大X坐标"""
        return max(point[0] for point in self.points)
    
    @property
    def y_max(self) -> float:
        """获取最大Y坐标"""
        return max(point[1] for point in self.points)
    
    @property
    def width(self) -> float:
        """获取边界框宽度"""
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        """获取边界框高度"""
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        """获取边界框面积（近似）"""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'points': self.points,
            'confidence': self.confidence,
            'shape_type': self.shape_type,
            'x_min': self.x_min,
            'y_min': self.y_min,
            'x_max': self.x_max,
            'y_max': self.y_max,
            'width': self.width,
            'height': self.height,
            'area': self.area
        }


@dataclass
class TextBlock:
    """
    文本块信息
    
    表示OCR识别出的单个文本块，包含文本内容、
    位置信息、置信度等详细信息。
    """
    # 识别的文本内容
    text: str
    
    # 文本块边界框
    bbox: BoundingBox
    
    # 文本块置信度
    confidence: float
    
    # 语言标识
    language: Optional[str] = None
    
    # 文本方向角度（度数）
    angle: float = 0.0
    
    # 字符级别的详细信息（可选）
    char_details: Optional[List[Dict[str, Any]]] = None
    
    # 额外的引擎特定属性
    engine_specific: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """后处理初始化"""
        # 确保文本不为None
        if self.text is None:
            self.text = ""
        
        # 清理文本中的异常字符
        self.text = self.text.strip()
    
    @property
    def length(self) -> int:
        """获取文本长度"""
        return len(self.text)
    
    @property
    def is_empty(self) -> bool:
        """判断文本块是否为空"""
        return len(self.text.strip()) == 0
    
    @property
    def word_count(self) -> int:
        """获取词语数量（简单按空格分割）"""
        return len(self.text.split()) if self.text else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'text': self.text,
            'bbox': self.bbox.to_dict(),
            'confidence': self.confidence,
            'language': self.language,
            'angle': self.angle,
            'length': self.length,
            'word_count': self.word_count,
            'char_details': self.char_details,
            'engine_specific': self.engine_specific
        }


@dataclass
class OCRResult:
    """
    OCR识别结果
    
    封装完整的OCR识别结果，包括文本内容、
    边界框信息、置信度等所有相关数据。
    """
    # 完整的识别文本内容
    text_content: str
    
    # 文本块列表
    text_blocks: List[TextBlock]
    
    # 整体置信度分数
    confidence_score: float
    
    # 检测到的语言
    language_detected: Optional[str] = None
    
    # 处理时间（秒）
    processing_time: float = 0.0
    
    # 使用的引擎名称
    engine_name: str = ""
    
    # 引擎版本信息
    engine_version: Optional[str] = None
    
    # 处理的图像尺寸
    image_size: Optional[Tuple[int, int]] = None
    
    # 额外的元数据
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """后处理初始化"""
        # 确保文本内容不为None
        if self.text_content is None:
            self.text_content = ""
        
        # 如果没有单独提供text_content，从text_blocks合成
        if not self.text_content and self.text_blocks:
            self.text_content = '\n'.join(
                block.text for block in self.text_blocks 
                if not block.is_empty
            )
        
        # 如果没有提供整体置信度，从文本块计算平均值
        if self.confidence_score == 0.0 and self.text_blocks:
            valid_blocks = [block for block in self.text_blocks if not block.is_empty]
            if valid_blocks:
                self.confidence_score = sum(
                    block.confidence for block in valid_blocks
                ) / len(valid_blocks)
    
    @property
    def char_count(self) -> int:
        """获取字符总数"""
        return len(self.text_content)
    
    @property
    def word_count(self) -> int:
        """获取词语总数"""
        return sum(block.word_count for block in self.text_blocks)
    
    @property
    def block_count(self) -> int:
        """获取文本块数量"""
        return len([block for block in self.text_blocks if not block.is_empty])
    
    @property
    def bounding_boxes(self) -> List[Dict[str, Any]]:
        """获取所有边界框信息"""
        return [block.bbox.to_dict() for block in self.text_blocks]
    
    @property
    def is_empty(self) -> bool:
        """判断识别结果是否为空"""
        return len(self.text_content.strip()) == 0
    
    def get_text_by_confidence(self, min_confidence: float = 0.5) -> str:
        """
        根据置信度筛选文本内容
        
        Args:
            min_confidence: 最小置信度阈值
            
        Returns:
            筛选后的文本内容
        """
        filtered_blocks = [
            block for block in self.text_blocks 
            if block.confidence >= min_confidence and not block.is_empty
        ]
        
        return '\n'.join(block.text for block in filtered_blocks)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'text_content': self.text_content,
            'text_blocks': [block.to_dict() for block in self.text_blocks],
            'confidence_score': self.confidence_score,
            'language_detected': self.language_detected,
            'processing_time': self.processing_time,
            'engine_name': self.engine_name,
            'engine_version': self.engine_version,
            'image_size': self.image_size,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'block_count': self.block_count,
            'bounding_boxes': self.bounding_boxes,
            'metadata': self.metadata
        }


class BaseOCREngine(ABC):
    """
    OCR引擎抽象基类
    
    定义所有OCR引擎必须实现的标准接口，包括初始化、
    识别、配置管理等核心功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化OCR引擎
        
        Args:
            config: 引擎配置参数
        """
        self.config = config or {}
        self.is_initialized = False
        self.engine_name = self.__class__.__name__
        self.engine_version = None
        
        # 设置日志
        self.logger = logging.getLogger(f"{__name__}.{self.engine_name}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        异步初始化引擎
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    async def recognize_async(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        **kwargs
    ) -> OCRResult:
        """
        异步执行OCR识别
        
        Args:
            image: 输入图像（支持多种格式）
            **kwargs: 额外的识别参数
            
        Returns:
            OCR识别结果
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的语言列表
        
        Returns:
            支持的语言代码列表
        """
        pass
    
    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """
        获取引擎信息
        
        Returns:
            引擎详细信息
        """
        pass
    
    async def health_check(self) -> bool:
        """
        引擎健康检查
        
        Returns:
            引擎是否正常运行
        """
        try:
            if not self.is_initialized:
                return await self.initialize()
            return True
        except Exception as e:
            self.logger.error(f"引擎健康检查失败: {str(e)}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置参数
        
        Args:
            config: 配置参数
            
        Returns:
            配置是否有效
        """
        # 基础验证逻辑
        required_fields = self.get_required_config_fields()
        
        for field in required_fields:
            if field not in config:
                self.logger.error(f"缺少必需的配置参数: {field}")
                return False
        
        return True
    
    def get_required_config_fields(self) -> List[str]:
        """
        获取必需的配置字段
        
        Returns:
            必需配置字段列表
        """
        # 默认没有必需字段，子类可以重写
        return []
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新引擎配置
        
        Args:
            new_config: 新的配置参数
            
        Returns:
            是否更新成功
        """
        try:
            if self.validate_config(new_config):
                self.config.update(new_config)
                self.logger.info("引擎配置已更新")
                return True
            else:
                self.logger.error("配置验证失败")
                return False
        except Exception as e:
            self.logger.error(f"配置更新失败: {str(e)}")
            return False
    
    def _preprocess_image(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        预处理输入图像，统一格式
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的numpy数组
        """
        try:
            if isinstance(image, (str, Path)):
                # 从文件路径加载图像
                pil_image = Image.open(image)
                return np.array(pil_image)
            
            elif isinstance(image, Image.Image):
                # PIL Image转numpy
                return np.array(image)
            
            elif isinstance(image, np.ndarray):
                # 已经是numpy数组
                return image
            
            else:
                raise ValueError(f"不支持的图像格式: {type(image)}")
                
        except Exception as e:
            self.logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def _create_default_result(
        self, 
        error_message: str = "识别失败"
    ) -> OCRResult:
        """
        创建默认的空结果（用于错误处理）
        
        Args:
            error_message: 错误信息
            
        Returns:
            空的OCR结果
        """
        return OCRResult(
            text_content="",
            text_blocks=[],
            confidence_score=0.0,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            metadata={'error': error_message}
        )
    
    def __repr__(self) -> str:
        """对象字符串表示"""
        return f"<{self.engine_name}(initialized={self.is_initialized})>"