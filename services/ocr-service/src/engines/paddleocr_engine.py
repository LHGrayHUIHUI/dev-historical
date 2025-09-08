"""
PaddleOCR引擎实现

基于百度PaddleOCR的OCR引擎实现，专门优化用于中文古籍识别。
PaddleOCR在中文识别方面表现优秀，支持多种语言和复杂场景。

主要特性：
- 高精度中文识别
- 支持文本检测和识别
- GPU加速支持
- 多语言支持
- 可配置的预处理参数

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
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    PaddleOCR = None

# 配置日志
logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR引擎实现类
    
    封装PaddleOCR功能，提供异步接口和古籍文本优化配置。
    支持检测和识别的分离执行，以及多种自定义参数。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化PaddleOCR引擎
        
        Args:
            config: 引擎配置参数
        """
        super().__init__(config)
        
        # 检查PaddleOCR依赖
        if not HAS_PADDLEOCR:
            raise ImportError("PaddleOCR未安装，请运行: pip install paddleocr")
        
        self.engine_name = "PaddleOCR"
        self.engine_version = None
        self.ocr_engine = None
        
        # 默认配置参数
        self.default_config = {
            # 基础配置
            "use_angle_cls": True,       # 使用文字方向分类器
            "lang": "ch",                # 语言（中文）
            "use_gpu": True,            # 使用GPU加速
            "gpu_mem": 500,             # GPU内存限制(MB)
            "enable_mkldnn": False,     # 启用MKLDNN加速
            "cpu_threads": 10,          # CPU线程数
            "det": True,                # 启用文本检测
            "rec": True,                # 启用文本识别
            "cls": True,                # 启用分类器
            
            # 检测模型配置
            "det_model_dir": None,      # 检测模型路径
            "det_limit_side_len": 960,  # 检测模型输入图像长边限制
            "det_limit_type": 'max',    # 限制类型
            "det_thresh": 0.3,          # 检测阈值
            "det_box_thresh": 0.6,      # 检测框阈值
            "det_unclip_ratio": 1.5,    # 检测框扩展比例
            "use_dilation": False,      # 是否使用膨胀操作
            "det_score_mode": "fast",   # 检测分数模式
            
            # 识别模型配置
            "rec_model_dir": None,      # 识别模型路径
            "rec_image_shape": "3, 48, 320",  # 识别模型输入形状
            "rec_batch_num": 6,         # 识别批处理大小
            "max_text_length": 25,      # 最大文本长度
            "rec_char_dict_path": None, # 字符字典路径
            "use_space_char": True,     # 使用空格字符
            
            # 分类器配置
            "cls_model_dir": None,      # 分类器模型路径
            "cls_image_shape": "3, 48, 192",  # 分类器输入形状
            "label_list": ['0', '180'], # 角度标签列表
            "cls_batch_num": 6,         # 分类批处理大小
            "cls_thresh": 0.9,          # 分类阈值
            
            # 古籍优化配置
            "drop_score": 0.5,          # 丢弃低置信度结果阈值
            "use_onnx": False,          # 使用ONNX推理
            "warmup": True,             # 启用预热
            "benchmark": False,         # 基准测试模式
            
            # 后处理配置
            "merge_no_span_structure": True,  # 合并无跨度结构
            "table_max_len": 488,       # 表格最大长度
            "table_algorithm": "TableAttn",  # 表格算法
        }
        
        # 合并用户配置
        self.config = {**self.default_config, **self.config}
        
        # 支持的语言列表
        self._supported_languages = [
            'ch', 'en', 'french', 'german', 'korean', 'japan',
            'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic',
            'cyrillic', 'devanagari'
        ]
    
    async def initialize(self) -> bool:
        """
        异步初始化PaddleOCR引擎
        
        Returns:
            是否初始化成功
        """
        try:
            if self.is_initialized:
                return True
            
            self.logger.info("开始初始化PaddleOCR引擎...")
            
            # 准备初始化参数
            init_params = self._prepare_init_params()
            
            # 在线程池中执行同步初始化
            loop = asyncio.get_event_loop()
            self.ocr_engine = await loop.run_in_executor(
                None, self._init_paddleocr_sync, init_params
            )
            
            if self.ocr_engine is None:
                self.logger.error("PaddleOCR引擎初始化失败")
                return False
            
            # 预热引擎（可选）
            if self.config.get('warmup', True):
                await self._warmup_engine()
            
            # 获取版本信息
            try:
                import paddle
                self.engine_version = paddle.__version__
            except:
                self.engine_version = "unknown"
            
            self.is_initialized = True
            self.logger.info("PaddleOCR引擎初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"PaddleOCR引擎初始化失败: {str(e)}")
            return False
    
    def _prepare_init_params(self) -> Dict[str, Any]:
        """
        准备PaddleOCR初始化参数
        
        Returns:
            初始化参数字典
        """
        params = {}
        
        # 基础参数映射
        param_mapping = {
            'use_angle_cls': 'use_angle_cls',
            'lang': 'lang', 
            'use_gpu': 'use_gpu',
            'gpu_mem': 'gpu_mem',
            'enable_mkldnn': 'enable_mkldnn',
            'cpu_threads': 'cpu_threads',
            'det': 'det',
            'rec': 'rec',
            'cls': 'cls',
            'det_model_dir': 'det_model_dir',
            'rec_model_dir': 'rec_model_dir',
            'cls_model_dir': 'cls_model_dir',
            'rec_char_dict_path': 'rec_char_dict_path',
            'use_space_char': 'use_space_char',
            'drop_score': 'drop_score',
            'use_onnx': 'use_onnx'
        }
        
        # 复制有效的配置参数
        for config_key, paddle_key in param_mapping.items():
            if config_key in self.config and self.config[config_key] is not None:
                params[paddle_key] = self.config[config_key]
        
        return params
    
    def _init_paddleocr_sync(self, params: Dict[str, Any]) -> Optional[PaddleOCR]:
        """
        同步初始化PaddleOCR实例
        
        Args:
            params: 初始化参数
            
        Returns:
            PaddleOCR实例或None
        """
        try:
            return PaddleOCR(**params)
        except Exception as e:
            self.logger.error(f"PaddleOCR同步初始化失败: {str(e)}")
            return None
    
    async def _warmup_engine(self):
        """预热引擎（使用测试图像）"""
        try:
            # 创建一个简单的测试图像
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
            
            # 执行一次测试识别
            await self.recognize_async(test_image)
            self.logger.info("PaddleOCR引擎预热完成")
            
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
            
            # 在线程池中执行同步识别
            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(
                None, self._recognize_sync, processed_image, recognition_params
            )
            
            if not raw_results:
                return self._create_default_result("无识别结果")
            
            # 转换为标准格式
            ocr_result = self._convert_results(
                raw_results, 
                time.time() - start_time,
                processed_image.shape[:2] if hasattr(processed_image, 'shape') else None
            )
            
            self.logger.info(f"PaddleOCR识别完成，耗时: {ocr_result.processing_time:.2f}秒")
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"PaddleOCR识别失败: {str(e)}")
            return self._create_default_result(f"识别异常: {str(e)}")
    
    def _prepare_recognition_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备识别参数
        
        Args:
            kwargs: 用户传入的参数
            
        Returns:
            处理后的识别参数
        """
        params = {}
        
        # 从配置和kwargs中提取识别相关参数
        recognition_keys = [
            'det_limit_side_len', 'det_thresh', 'det_box_thresh',
            'det_unclip_ratio', 'use_dilation', 'det_score_mode',
            'rec_batch_num', 'cls_thresh', 'drop_score'
        ]
        
        for key in recognition_keys:
            if key in kwargs:
                params[key] = kwargs[key]
            elif key in self.config:
                params[key] = self.config[key]
        
        return params
    
    def _recognize_sync(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> List[List]:
        """
        同步执行识别
        
        Args:
            image: 图像数组
            params: 识别参数
            
        Returns:
            原始识别结果
        """
        try:
            # 调用PaddleOCR进行识别
            results = self.ocr_engine.ocr(image, **params)
            
            # 处理返回结果格式
            if results and isinstance(results, list):
                if len(results) > 0 and isinstance(results[0], list):
                    return results[0]  # 单页结果
                return results
            
            return []
            
        except Exception as e:
            self.logger.error(f"PaddleOCR同步识别失败: {str(e)}")
            return []
    
    def _convert_results(
        self, 
        raw_results: List[List], 
        processing_time: float,
        image_size: Optional[Tuple[int, int]] = None
    ) -> OCRResult:
        """
        转换PaddleOCR原始结果为标准格式
        
        Args:
            raw_results: PaddleOCR原始结果
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
                if not result_item or len(result_item) != 2:
                    continue
                
                # 解析PaddleOCR结果格式: [bbox_coords, (text, confidence)]
                bbox_coords, (text, confidence) = result_item
                
                if not text or not text.strip():
                    continue
                
                # 创建边界框对象
                bbox_points = [(float(point[0]), float(point[1])) for point in bbox_coords]
                bbox = BoundingBox(
                    points=bbox_points,
                    confidence=float(confidence),
                    shape_type="polygon"
                )
                
                # 创建文本块对象
                text_block = TextBlock(
                    text=text.strip(),
                    bbox=bbox,
                    confidence=float(confidence),
                    language=self.config.get('lang', 'ch'),
                    engine_specific={
                        'bbox_coords': bbox_coords,
                        'original_confidence': confidence
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
            
            # 检测语言
            detected_language = self._detect_language(full_text)
            
            # 创建OCR结果对象
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
                        'lang': self.config.get('lang'),
                        'use_gpu': self.config.get('use_gpu'),
                        'drop_score': self.config.get('drop_score')
                    }
                }
            )
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"结果转换失败: {str(e)}")
            return self._create_default_result(f"结果转换异常: {str(e)}")
    
    def _detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            检测到的语言代码
        """
        if not text:
            return self.config.get('lang', 'ch')
        
        # 简单的语言检测逻辑
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        total_chars = len([c for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff'])
        
        if total_chars == 0:
            return self.config.get('lang', 'ch')
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.5:
            return 'ch'
        elif chinese_ratio > 0.1:
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
                'angle_classification',
                'gpu_acceleration',
                'multi_language',
                'chinese_optimization'
            ],
            'config': {
                'use_gpu': self.config.get('use_gpu'),
                'lang': self.config.get('lang'),
                'use_angle_cls': self.config.get('use_angle_cls'),
                'drop_score': self.config.get('drop_score')
            }
        }
    
    def get_required_config_fields(self) -> List[str]:
        """
        获取必需的配置字段
        
        Returns:
            必需配置字段列表
        """
        return ['lang']  # 语言是必需的
    
    async def batch_recognize(
        self, 
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        max_concurrent: int = 4
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
        
        self.logger.info(f"批量识别完成: {len(images)} 个图像")
        return processed_results
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'ocr_engine') and self.ocr_engine:
            try:
                del self.ocr_engine
            except:
                pass