"""
Tesseract引擎实现

基于Google Tesseract OCR引擎的实现，提供高质量的文本识别功能。
Tesseract是开源OCR引擎的标杆，支持多种语言和输出格式。

主要特性：
- 支持100+种语言
- 可配置的页面分段模式
- 丰富的输出格式选项
- 字符和词级别的置信度
- 支持古文和特殊字体

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
import re

from .base_engine import BaseOCREngine, OCRResult, TextBlock, BoundingBox

# 可选依赖导入
try:
    import pytesseract
    from pytesseract import Output
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None
    Output = None

# 配置日志
logger = logging.getLogger(__name__)


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR引擎实现类
    
    封装pytesseract功能，提供异步接口和针对古籍文本的优化配置。
    支持多种页面分段模式和输出格式。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Tesseract引擎
        
        Args:
            config: 引擎配置参数
        """
        super().__init__(config)
        
        # 检查依赖
        if not HAS_TESSERACT:
            raise ImportError("pytesseract未安装，请运行: pip install pytesseract")
        
        self.engine_name = "Tesseract"
        self.engine_version = None
        
        # 默认配置参数
        self.default_config = {
            # 基础语言配置
            "lang": "chi_sim+eng",      # 简体中文+英文
            
            # OCR引擎模式 (oem)
            "oem": 3,                   # 默认，基于LSTM OCR引擎模式
            
            # 页面分段模式 (psm) 
            "psm": 6,                   # 假设文本为单一均匀的块
            
            # 自定义配置字符串
            "config": "--dpi 300",      # DPI设置
            
            # Tesseract参数
            "nice": 0,                  # 进程优先级
            "timeout": 30,              # 超时时间（秒）
            
            # 输出配置
            "output_type": Output.DICT, # 输出格式
            "pandas_config": None,      # pandas配置
            
            # 预处理配置
            "preserve_interword_spaces": 1,  # 保留词间空格
            "tessedit_char_whitelist": "",   # 白名单字符
            "tessedit_char_blacklist": "",   # 黑名单字符
            
            # 古籍优化参数
            "textord_tablefind_good_columns_ratio": 0.8,
            "textord_heavy_nr": 1,
            "textord_show_initial_words": 1,
            "classify_enable_learning": 0,
            "classify_enable_adaptive_matcher": 1,
            
            # 置信度相关
            "tessedit_reject_mode": 0,       # 拒绝模式
            "tessedit_reject_bad_qual_wds": 1,  # 拒绝质量差的词
            "tessedit_reject_row_percent": 40,  # 行拒绝百分比
            "tessedit_reject_doc_percent": 65,  # 文档拒绝百分比
        }
        
        # 合并用户配置
        self.config = {**self.default_config, **self.config}
        
        # 页面分段模式说明
        self.psm_modes = {
            0: "仅方向和脚本检测",
            1: "带方向和脚本检测的自动页面分段",
            2: "自动页面分段，无方向和脚本检测",
            3: "完全自动页面分段，无方向和脚本检测",
            4: "假设可变大小的单列文本",
            5: "假设单一均匀的垂直对齐文本块",
            6: "假设单一均匀的文本块",
            7: "将图像视为单一文本行",
            8: "将图像视为单一单词",
            9: "将图像视为圆圈中的单一单词",
            10: "将图像视为单一字符",
            11: "稀疏文本，按顺序查找尽可能多的文本",
            12: "带方向和脚本检测的稀疏文本",
            13: "原始行，将图像视为单一文本行"
        }
        
        # OEM模式说明
        self.oem_modes = {
            0: "仅传统引擎",
            1: "仅神经网络LSTM引擎", 
            2: "传统+LSTM引擎",
            3: "基于可用引擎的默认"
        }
        
        # 支持的语言（常用）
        self._supported_languages = [
            'chi_sim', 'chi_tra', 'eng', 'jpn', 'kor', 
            'fra', 'deu', 'spa', 'rus', 'ara', 'tha'
        ]
    
    async def initialize(self) -> bool:
        """
        异步初始化Tesseract引擎
        
        Returns:
            是否初始化成功
        """
        try:
            if self.is_initialized:
                return True
            
            self.logger.info("开始初始化Tesseract引擎...")
            
            # 检测Tesseract可执行文件
            loop = asyncio.get_event_loop()
            version_info = await loop.run_in_executor(
                None, self._get_tesseract_version
            )
            
            if not version_info:
                self.logger.error("无法检测到Tesseract可执行文件")
                return False
            
            self.engine_version = version_info
            
            # 验证语言支持
            available_languages = await loop.run_in_executor(
                None, self._get_available_languages
            )
            
            configured_langs = self.config.get('lang', 'eng').split('+')
            for lang in configured_langs:
                if lang not in available_languages:
                    self.logger.warning(f"语言 '{lang}' 不可用，请检查Tesseract语言包")
            
            # 预热引擎
            if self.config.get('warmup', True):
                await self._warmup_engine()
            
            self.is_initialized = True
            self.logger.info("Tesseract引擎初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"Tesseract引擎初始化失败: {str(e)}")
            return False
    
    def _get_tesseract_version(self) -> Optional[str]:
        """
        获取Tesseract版本信息
        
        Returns:
            版本信息字符串或None
        """
        try:
            return pytesseract.get_tesseract_version()
        except Exception as e:
            self.logger.error(f"获取Tesseract版本失败: {str(e)}")
            return None
    
    def _get_available_languages(self) -> List[str]:
        """
        获取可用语言列表
        
        Returns:
            可用语言列表
        """
        try:
            langs = pytesseract.get_languages()
            return langs if langs else []
        except Exception as e:
            self.logger.error(f"获取可用语言失败: {str(e)}")
            return []
    
    async def _warmup_engine(self):
        """预热引擎"""
        try:
            # 创建测试图像
            test_image = Image.new('RGB', (200, 100), color='white')
            await self.recognize_async(test_image)
            self.logger.info("Tesseract引擎预热完成")
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
            pil_image = self._convert_to_pil(image)
            
            # 准备Tesseract配置
            tesseract_config = self._prepare_tesseract_config(kwargs)
            
            # 在线程池中执行识别
            loop = asyncio.get_event_loop()
            raw_data = await loop.run_in_executor(
                None, self._recognize_sync, pil_image, tesseract_config
            )
            
            if not raw_data:
                return self._create_default_result("无识别结果")
            
            # 转换为标准格式
            ocr_result = self._convert_tesseract_results(
                raw_data,
                time.time() - start_time,
                pil_image.size
            )
            
            self.logger.info(f"Tesseract识别完成，耗时: {ocr_result.processing_time:.2f}秒")
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"Tesseract识别失败: {str(e)}")
            return self._create_default_result(f"识别异常: {str(e)}")
    
    def _convert_to_pil(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Image.Image:
        """
        转换图像为PIL Image格式
        
        Args:
            image: 输入图像
            
        Returns:
            PIL Image对象
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        else:
            raise ValueError(f"不支持的图像格式: {type(image)}")
    
    def _prepare_tesseract_config(self, kwargs: Dict[str, Any]) -> str:
        """
        准备Tesseract配置字符串
        
        Args:
            kwargs: 用户参数
            
        Returns:
            配置字符串
        """
        config_parts = []
        
        # 基础配置
        base_config = self.config.get('config', '')
        if base_config:
            config_parts.append(base_config)
        
        # OEM模式
        oem = kwargs.get('oem', self.config.get('oem'))
        if oem is not None:
            config_parts.append(f'--oem {oem}')
        
        # PSM模式
        psm = kwargs.get('psm', self.config.get('psm'))
        if psm is not None:
            config_parts.append(f'--psm {psm}')
        
        # 字符白名单
        whitelist = kwargs.get('tessedit_char_whitelist', 
                              self.config.get('tessedit_char_whitelist'))
        if whitelist:
            config_parts.append(f'-c tessedit_char_whitelist={whitelist}')
        
        # 字符黑名单
        blacklist = kwargs.get('tessedit_char_blacklist',
                              self.config.get('tessedit_char_blacklist'))
        if blacklist:
            config_parts.append(f'-c tessedit_char_blacklist={blacklist}')
        
        # 其他配置参数
        config_params = [
            'preserve_interword_spaces', 'textord_tablefind_good_columns_ratio',
            'textord_heavy_nr', 'classify_enable_learning', 'tessedit_reject_mode'
        ]
        
        for param in config_params:
            value = kwargs.get(param, self.config.get(param))
            if value is not None:
                config_parts.append(f'-c {param}={value}')
        
        return ' '.join(config_parts)
    
    def _recognize_sync(
        self, 
        image: Image.Image, 
        config: str
    ) -> Optional[Dict]:
        """
        同步执行Tesseract识别
        
        Args:
            image: PIL图像
            config: Tesseract配置
            
        Returns:
            识别结果字典
        """
        try:
            # 获取语言配置
            lang = self.config.get('lang', 'eng')
            timeout = self.config.get('timeout', 30)
            
            # 执行OCR识别，获取详细数据
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config=config,
                output_type=Output.DICT,
                timeout=timeout
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Tesseract同步识别失败: {str(e)}")
            return None
    
    def _convert_tesseract_results(
        self,
        raw_data: Dict,
        processing_time: float,
        image_size: Tuple[int, int]
    ) -> OCRResult:
        """
        转换Tesseract原始结果为标准格式
        
        Args:
            raw_data: Tesseract原始数据
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
            # 获取数据字段
            texts = raw_data.get('text', [])
            confidences = raw_data.get('conf', [])
            lefts = raw_data.get('left', [])
            tops = raw_data.get('top', [])
            widths = raw_data.get('width', [])
            heights = raw_data.get('height', [])
            levels = raw_data.get('level', [])
            
            # 处理每个识别项
            for i in range(len(texts)):
                text = texts[i].strip()
                confidence = float(confidences[i])
                
                # 跳过空文本和低置信度结果
                if not text or confidence < 0:
                    continue
                
                # 创建边界框
                left = float(lefts[i])
                top = float(tops[i])
                width = float(widths[i])
                height = float(heights[i])
                
                bbox_points = [
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left, top + height)
                ]
                
                bbox = BoundingBox(
                    points=bbox_points,
                    confidence=confidence / 100.0,  # Tesseract置信度是0-100
                    shape_type="rectangle"
                )
                
                # 创建文本块
                text_block = TextBlock(
                    text=text,
                    bbox=bbox,
                    confidence=confidence / 100.0,
                    language=self.config.get('lang', 'eng'),
                    engine_specific={
                        'level': levels[i],
                        'original_confidence': confidence,
                        'tesseract_index': i
                    }
                )
                
                text_blocks.append(text_block)
                full_text_parts.append(text)
                total_confidence += confidence / 100.0
                valid_block_count += 1
            
            # 计算平均置信度
            avg_confidence = total_confidence / valid_block_count if valid_block_count > 0 else 0.0
            
            # 组合完整文本
            full_text = self._reconstruct_text(raw_data)
            
            # 检测语言
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
                        'lang': self.config.get('lang'),
                        'oem': self.config.get('oem'),
                        'psm': self.config.get('psm')
                    }
                }
            )
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"结果转换失败: {str(e)}")
            return self._create_default_result(f"结果转换异常: {str(e)}")
    
    def _reconstruct_text(self, raw_data: Dict) -> str:
        """
        重建完整文本，保持适当的换行和空格
        
        Args:
            raw_data: Tesseract原始数据
            
        Returns:
            重建的文本
        """
        try:
            texts = raw_data.get('text', [])
            levels = raw_data.get('level', [])
            confidences = raw_data.get('conf', [])
            
            lines = []
            current_line = []
            current_level = None
            
            for i, (text, level, conf) in enumerate(zip(texts, levels, confidences)):
                text = text.strip()
                
                # 跳过空文本和低置信度
                if not text or conf < 0:
                    continue
                
                # 按级别分组文本
                if level == 5:  # 行级别
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = []
                    current_level = level
                elif level == 4:  # 词级别
                    current_line.append(text)
            
            # 添加最后一行
            if current_line:
                lines.append(' '.join(current_line))
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.warning(f"文本重建失败: {str(e)}")
            # fallback: 简单连接所有文本
            texts = raw_data.get('text', [])
            return ' '.join(text.strip() for text in texts if text.strip())
    
    def _detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            检测到的语言
        """
        if not text:
            return self.config.get('lang', 'eng')
        
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return self.config.get('lang', 'eng')
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.5:
            return 'chi_sim'
        elif chinese_ratio > 0.1:
            return 'mixed'
        else:
            return 'eng'
    
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
                'text_recognition',
                'multi_language',
                'configurable_psm',
                'word_confidence',
                'character_whitelist',
                'historical_text_support'
            ],
            'psm_modes': self.psm_modes,
            'oem_modes': self.oem_modes,
            'config': {
                'lang': self.config.get('lang'),
                'oem': self.config.get('oem'),
                'psm': self.config.get('psm'),
                'timeout': self.config.get('timeout')
            }
        }
    
    def get_required_config_fields(self) -> List[str]:
        """
        获取必需的配置字段
        
        Returns:
            必需配置字段列表
        """
        return []  # Tesseract有很好的默认值