"""
EasyOCR引擎实现

基于EasyOCR的OCR引擎实现，支持80+种语言的即开即用OCR。
EasyOCR基于深度学习，在多语言文本识别方面表现优秀。

主要特性：
- 支持80+种语言
- 基于深度学习的高精度识别  
- GPU加速支持
- 无需额外配置即可使用
- 对倾斜文本有较好的鲁棒性

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import logging

from .base_engine import BaseOCREngine, OCRResult, TextBlock, BoundingBox

# 可选依赖导入
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    easyocr = None

# 配置日志
logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR引擎实现类
    
    封装EasyOCR功能，提供异步接口和针对多语言文本的优化配置。
    特别适合处理多语言混合文档和倾斜文本。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化EasyOCR引擎
        
        Args:
            config: 引擎配置参数
        """
        super().__init__(config)
        
        # 检查依赖
        if not HAS_EASYOCR:
            raise ImportError("EasyOCR未安装，请运行: pip install easyocr")
        
        self.engine_name = "EasyOCR"
        self.engine_version = None
        self.reader = None
        
        # 默认配置参数
        self.default_config = {
            # 语言配置
            "lang_list": ["ch_sim", "en"],     # 支持的语言列表
            
            # GPU配置
            "gpu": True,                       # 使用GPU
            "model_storage_directory": None,   # 模型存储目录
            "user_network_directory": None,    # 用户网络目录
            "download_enabled": True,          # 允许下载模型
            
            # 识别参数
            "decoder": "greedy",               # 解码器类型 greedy/beamsearch
            "beamWidth": 5,                   # beam搜索宽度
            "batch_size": 1,                  # 批处理大小
            "workers": 0,                     # 工作线程数 (0=自动)
            
            # 检测参数  
            "width_ths": 0.7,                 # 宽度阈值
            "height_ths": 0.7,                # 高度阈值
            "detector": True,                 # 启用检测器
            "recognizer": True,               # 启用识别器
            
            # 文本参数
            "paragraph": False,               # 段落模式
            "min_size": 20,                   # 最小文本区域大小
            "text_threshold": 0.7,            # 文本阈值
            "low_text": 0.4,                  # 低文本阈值
            "link_threshold": 0.4,            # 连接阈值
            "canvas_size": 2560,              # 画布大小
            "mag_ratio": 1.,                  # 放大比率
            
            # 后处理参数
            "slope_ths": 0.1,                 # 斜率阈值
            "ycenter_ths": 0.5,               # y中心阈值
            "height_ths": 0.7,                # 高度阈值
            "width_ths": 0.7,                 # 宽度阈值
            "add_margin": 0.1,                # 添加边距
            
            # 输出参数
            "allowlist": None,                # 允许字符列表
            "blocklist": None,                # 禁止字符列表
            "detail": 1,                      # 详细程度 (0-1)
            "rotation_info": None,            # 旋转信息
            
            # 优化参数
            "contrast_ths": 0.1,              # 对比度阈值
            "adjust_contrast": 0.5,           # 对比度调整
            "filter_ths": 0.003,              # 过滤阈值
            "textline_ths": 0.9,              # 文本行阈值
        }
        
        # 合并用户配置
        self.config = {**self.default_config, **self.config}
        
        # 支持的语言列表（EasyOCR官方支持）
        self._supported_languages = [
            'ch_sim', 'ch_tra', 'en', 'ja', 'ko', 'th', 'vi', 'ar',
            'bg', 'cs', 'da', 'de', 'el', 'es', 'et', 'fa', 'fi',
            'fr', 'he', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'lt',
            'lv', 'ms', 'mt', 'nl', 'no', 'pl', 'pt', 'ro', 'rs_cyrillic',
            'rs_latin', 'ru', 'sk', 'sl', 'sq', 'sv', 'sw', 'tr', 'uk', 'ur'
        ]
    
    async def initialize(self) -> bool:
        """
        异步初始化EasyOCR引擎
        
        Returns:
            是否初始化成功
        """
        try:
            if self.is_initialized:
                return True
            
            self.logger.info("开始初始化EasyOCR引擎...")
            
            # 准备初始化参数
            init_params = self._prepare_init_params()
            
            # 在线程池中执行同步初始化（可能需要下载模型）
            loop = asyncio.get_event_loop()
            self.reader = await loop.run_in_executor(
                None, self._init_reader_sync, init_params
            )
            
            if self.reader is None:
                self.logger.error("EasyOCR引擎初始化失败")
                return False
            
            # 获取版本信息
            try:
                self.engine_version = easyocr.__version__
            except:
                self.engine_version = "unknown"
            
            # 预热引擎
            if self.config.get('warmup', True):
                await self._warmup_engine()
            
            self.is_initialized = True
            self.logger.info("EasyOCR引擎初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"EasyOCR引擎初始化失败: {str(e)}")
            return False
    
    def _prepare_init_params(self) -> Dict[str, Any]:
        """
        准备EasyOCR初始化参数
        
        Returns:
            初始化参数字典
        """
        params = {
            'lang_list': self.config['lang_list'],
            'gpu': self.config['gpu'],
            'download_enabled': self.config['download_enabled']
        }
        
        # 可选参数
        if self.config.get('model_storage_directory'):
            params['model_storage_directory'] = self.config['model_storage_directory']
        
        if self.config.get('user_network_directory'):
            params['user_network_directory'] = self.config['user_network_directory']
        
        return params
    
    def _init_reader_sync(self, params: Dict[str, Any]) -> Optional[object]:
        """
        同步初始化EasyOCR Reader
        
        Args:
            params: 初始化参数
            
        Returns:
            EasyOCR Reader实例或None
        """
        try:
            return easyocr.Reader(**params)
        except Exception as e:
            self.logger.error(f"EasyOCR Reader初始化失败: {str(e)}")
            return None
    
    async def _warmup_engine(self):
        """预热引擎"""
        try:
            # 创建测试图像
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
            await self.recognize_async(test_image)
            self.logger.info("EasyOCR引擎预热完成")
        except Exception as e:
            self.logger.warning(f"引擎预热失败: {str(e)}")
    
    async def recognize_async(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        **kwargs
    ) -> OCRResult:
        """
        异步执行OCR识别
        
        Args:
            image: 输入图像
            **kwargs: 额外识别参数
            
        Returns:
            OCR识别结果
        """
        if not self.is_initialized:
            if not await self.initialize():
                return self._create_default_result("引擎初始化失败")
        
        try:
            start_time = time.time()
            
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 准备识别参数
            recognition_params = self._prepare_recognition_params(kwargs)
            
            # 在线程池中执行识别
            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(
                None, self._recognize_sync, processed_image, recognition_params
            )
            
            if not raw_results:
                return self._create_default_result("无识别结果")
            
            # 转换为标准格式
            ocr_result = self._convert_easyocr_results(
                raw_results,
                time.time() - start_time,
                processed_image.shape[:2] if hasattr(processed_image, 'shape') else None
            )
            
            self.logger.info(f"EasyOCR识别完成，耗时: {ocr_result.processing_time:.2f}秒")
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"EasyOCR识别失败: {str(e)}")
            return self._create_default_result(f"识别异常: {str(e)}")
    
    def _prepare_recognition_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备识别参数
        
        Args:
            kwargs: 用户参数
            
        Returns:
            识别参数字典
        """
        params = {}
        
        # 参数映射
        param_keys = [
            'width_ths', 'height_ths', 'decoder', 'beamWidth', 'batch_size',
            'workers', 'allowlist', 'blocklist', 'detail', 'paragraph',
            'min_size', 'text_threshold', 'low_text', 'link_threshold',
            'canvas_size', 'mag_ratio', 'slope_ths', 'ycenter_ths',
            'add_margin', 'contrast_ths', 'adjust_contrast', 'filter_ths'
        ]
        
        for key in param_keys:
            if key in kwargs:
                params[key] = kwargs[key]
            elif key in self.config and self.config[key] is not None:
                params[key] = self.config[key]
        
        # 确保detail参数正确
        params['detail'] = kwargs.get('detail', self.config.get('detail', 1))
        
        return params
    
    def _recognize_sync(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> List[Any]:
        """
        同步执行识别
        
        Args:
            image: 图像数组
            params: 识别参数
            
        Returns:
            原始识别结果
        """
        try:
            # 调用EasyOCR进行识别
            results = self.reader.readtext(image, **params)
            return results
            
        except Exception as e:
            self.logger.error(f"EasyOCR同步识别失败: {str(e)}")
            return []
    
    def _convert_easyocr_results(
        self,
        raw_results: List[Any],
        processing_time: float,
        image_size: Optional[Tuple[int, int]] = None
    ) -> OCRResult:
        """
        转换EasyOCR原始结果为标准格式
        
        Args:
            raw_results: EasyOCR原始结果
            processing_time: 处理时间
            image_size: 图像尺寸
            
        Returns:
            标准OCR结果
        """
        text_blocks = []
        full_text_parts = []
        total_confidence = 0.0
        valid_block_count = 0
        
        try:
            for result_item in raw_results:
                if not result_item or len(result_item) < 2:
                    continue
                
                # EasyOCR结果格式: [bbox_coords, text, confidence]
                if len(result_item) == 3:
                    bbox_coords, text, confidence = result_item
                else:
                    # 处理不同的返回格式
                    bbox_coords = result_item[0]
                    text = result_item[1]
                    confidence = result_item[2] if len(result_item) > 2 else 0.0
                
                if not text or not text.strip():
                    continue
                
                # 创建边界框对象
                bbox_points = [(float(point[0]), float(point[1])) for point in bbox_coords]
                bbox = BoundingBox(
                    points=bbox_points,
                    confidence=float(confidence),
                    shape_type="polygon"
                )
                
                # 检测文本角度（简单估计）
                angle = self._estimate_text_angle(bbox_points)
                
                # 创建文本块对象
                text_block = TextBlock(
                    text=text.strip(),
                    bbox=bbox,
                    confidence=float(confidence),
                    language=self._detect_text_language(text),
                    angle=angle,
                    engine_specific={
                        'bbox_coords': bbox_coords,
                        'original_confidence': confidence,
                        'estimated_angle': angle
                    }
                )
                
                text_blocks.append(text_block)
                full_text_parts.append(text.strip())
                total_confidence += float(confidence)
                valid_block_count += 1
            
            # 计算平均置信度
            avg_confidence = total_confidence / valid_block_count if valid_block_count > 0 else 0.0
            
            # 组合完整文本
            full_text = '\n'.join(full_text_parts)
            
            # 检测主要语言
            detected_language = self._detect_language(full_text)
            
            # 创建OCR结果
            ocr_result = OCRResult(
                text_content=full_text,
                text_blocks=text_blocks,
                confidence_score=avg_confidence,
                language_detected=detected_language,
                processing_time=processing_time,
                engine_name=self.engine_name,
                engine_version=self.engine_version,
                image_size=image_size,
                metadata={
                    'total_blocks': len(text_blocks),
                    'valid_blocks': valid_block_count,
                    'avg_confidence': avg_confidence,
                    'engine_config': {
                        'lang_list': self.config.get('lang_list'),
                        'gpu': self.config.get('gpu'),
                        'detail': self.config.get('detail')
                    }
                }
            )
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"结果转换失败: {str(e)}")
            return self._create_default_result(f"结果转换异常: {str(e)}")
    
    def _estimate_text_angle(self, bbox_points: List[Tuple[float, float]]) -> float:
        """
        估计文本角度
        
        Args:
            bbox_points: 边界框点坐标
            
        Returns:
            估计的角度（度数）
        """
        try:
            if len(bbox_points) < 4:
                return 0.0
            
            # 计算上边和下边的斜率
            p1, p2, p3, p4 = bbox_points[:4]
            
            # 计算上边的角度
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx == 0:
                return 90.0 if dy > 0 else -90.0
            
            import math
            angle = math.degrees(math.atan(dy / dx))
            return angle
            
        except Exception:
            return 0.0
    
    def _detect_text_language(self, text: str) -> str:
        """
        检测单个文本块的语言
        
        Args:
            text: 文本内容
            
        Returns:
            语言代码
        """
        if not text:
            return 'unknown'
        
        # 简单的语言检测逻辑
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        total_chars = len(text)
        
        if chinese_chars > english_chars:
            return 'ch_sim'
        elif english_chars > 0:
            return 'en'
        else:
            return 'unknown'
    
    def _detect_language(self, text: str) -> str:
        """
        检测整体文本语言
        
        Args:
            text: 完整文本
            
        Returns:
            主要语言代码
        """
        if not text:
            return 'unknown'
        
        # 统计各种字符类型
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        total_alpha_chars = chinese_chars + english_chars
        
        if total_alpha_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_alpha_chars
        
        if chinese_ratio > 0.6:
            return 'ch_sim'
        elif chinese_ratio > 0.2:
            return 'mixed'
        else:
            return 'en'
    
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的语言列表
        
        Returns:
            支持的语言代码列表
        """
        return self._supported_languages.copy()
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        获取引擎信息
        
        Returns:
            引擎详细信息
        """
        return {
            'name': self.engine_name,
            'version': self.engine_version,
            'initialized': self.is_initialized,
            'supported_languages': self.get_supported_languages(),
            'features': [
                'text_detection',
                'text_recognition',
                'multi_language',
                'gpu_acceleration',
                'rotation_robust',
                '80_plus_languages',
                'no_training_required'
            ],
            'config': {
                'lang_list': self.config.get('lang_list'),
                'gpu': self.config.get('gpu'),
                'detail': self.config.get('detail'),
                'paragraph': self.config.get('paragraph')
            }
        }
    
    def get_required_config_fields(self) -> List[str]:
        """
        获取必需的配置字段
        
        Returns:
            必需配置字段列表
        """
        return ['lang_list']  # 语言列表是必需的
    
    async def batch_recognize(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        max_concurrent: int = 2  # EasyOCR内存占用较大，降低并发数
    ) -> List[OCRResult]:
        """
        批量识别多个图像
        
        Args:
            images: 图像列表
            max_concurrent: 最大并发数
            
        Returns:
            识别结果列表
        """
        if not self.is_initialized:
            if not await self.initialize():
                return [self._create_default_result("引擎未初始化") for _ in images]
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def recognize_single(image):
            async with semaphore:
                return await self.recognize_async(image)
        
        tasks = [recognize_single(image) for image in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"批量识别异常: {str(result)}")
                processed_results.append(self._create_default_result(f"识别异常: {str(result)}"))
            else:
                processed_results.append(result)
        
        self.logger.info(f"EasyOCR批量识别完成: {len(images)} 个图像")
        return processed_results
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'reader') and self.reader:
            try:
                del self.reader
            except:
                pass