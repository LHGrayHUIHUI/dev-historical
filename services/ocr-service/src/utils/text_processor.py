"""
文本后处理工具模块

提供OCR识别结果的后处理功能，包括繁简转换、标点符号规范化、
错误纠正、古文字处理等专门针对历史文献的文本优化功能。

主要功能：
- 繁体/简体中文转换
- 标点符号智能规范化
- OCR错误字符修正
- 古文字和异体字处理
- 文本质量评估
- 文本分段和结构化

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import unicodedata
import difflib

# 可选依赖的导入
try:
    import opencc  # 繁简转换库
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    logging.warning("OpenCC未安装，繁简转换功能将被禁用")

try:
    import jieba
    import jieba.posseg as pseg
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    logging.warning("jieba未安装，分词功能将被禁用")

# 配置日志
logger = logging.getLogger(__name__)


class ConversionType(str, Enum):
    """文字转换类型枚举"""
    TRADITIONAL_TO_SIMPLIFIED = "t2s"      # 繁体转简体
    SIMPLIFIED_TO_TRADITIONAL = "s2t"      # 简体转繁体
    TRADITIONAL_TO_TAIWAN = "t2tw"         # 繁体转台湾正体
    TRADITIONAL_TO_HONGKONG = "t2hk"       # 繁体转香港繁体
    SIMPLIFIED_TO_TAIWAN = "s2tw"          # 简体转台湾正体
    SIMPLIFIED_TO_HONGKONG = "s2hk"        # 简体转香港繁体


class TextNormalizeLevel(str, Enum):
    """文本规范化级别"""
    LIGHT = "light"         # 轻度规范化
    STANDARD = "standard"   # 标准规范化
    AGGRESSIVE = "aggressive" # 激进规范化


@dataclass
class TextQualityMetrics:
    """文本质量评估指标"""
    char_count: int                    # 字符总数
    word_count: int                    # 词语数量
    line_count: int                    # 行数
    paragraph_count: int               # 段落数
    confidence_score: float            # 整体置信度
    error_char_count: int              # 疑似错误字符数
    punctuation_ratio: float           # 标点符号比例
    chinese_char_ratio: float          # 中文字符比例
    suspicious_patterns: List[str]     # 可疑模式列表
    avg_line_length: float            # 平均行长度
    avg_word_length: float            # 平均词长度


class TextProcessor:
    """
    文本后处理器
    
    专门为OCR识别后的古籍文本优化的后处理工具类，
    支持各种文本清洗、规范化、纠错等功能。
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化文本处理器
        
        Args:
            max_workers: 线程池最大工作线程数
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 初始化转换器
        self.converters = {}
        if HAS_OPENCC:
            try:
                self.converters = {
                    ConversionType.TRADITIONAL_TO_SIMPLIFIED: opencc.OpenCC('t2s.json'),
                    ConversionType.SIMPLIFIED_TO_TRADITIONAL: opencc.OpenCC('s2t.json'),
                    ConversionType.TRADITIONAL_TO_TAIWAN: opencc.OpenCC('t2tw.json'),
                    ConversionType.TRADITIONAL_TO_HONGKONG: opencc.OpenCC('t2hk.json'),
                    ConversionType.SIMPLIFIED_TO_TAIWAN: opencc.OpenCC('s2tw.json'),
                    ConversionType.SIMPLIFIED_TO_HONGKONG: opencc.OpenCC('s2hk.json'),
                }
                logger.info("OpenCC繁简转换器初始化成功")
            except Exception as e:
                logger.warning(f"OpenCC初始化失败: {str(e)}")
                self.converters = {}
        
        # 初始化分词器
        if HAS_JIEBA:
            jieba.initialize()
            logger.info("jieba分词器初始化成功")
        
        # 预定义的规则和字典
        self._init_processing_rules()
    
    def _init_processing_rules(self):
        """初始化处理规则和字典"""
        
        # 常见OCR错误字符映射
        self.ocr_error_mappings = {
            # 数字和字母混淆
            '0': 'O', '1': 'l', '5': 'S', '8': 'B',
            'O': '0', 'l': '1', 'S': '5', 'B': '8',
            
            # 中文常见OCR错误
            '未': '末', '末': '未', '己': '已', '已': '己',
            '测': '侧', '侧': '测', '账': '帐', '帐': '账',
            '辩': '辨', '辨': '辩', '做': '作', '作': '做',
            
            # 标点符号错误
            '。': '.', '，': ',', '；': ';', '：': ':',
            '？': '?', '！': '!', '"': '"', '"': '"',
            ''': "'", ''': "'", '（': '(', '）': ')',
            
            # 古文字常见错误
            '万': '萬', '与': '與', '会': '會', '个': '個',
            '国': '國', '来': '來', '时': '時', '实': '實'
        }
        
        # 标点符号映射规则
        self.punctuation_mappings = {
            # 英文标点转中文标点
            '.': '。', ',': '，', ';': '；', ':': '：',
            '?': '？', '!': '！', '"': '"', "'": ''',
            '(': '（', ')': '）', '[': '「', ']': '」',
            '{': '『', '}': '』',
            
            # 规范化重复标点
            '。。': '。', '，，': '，', '？？': '？',
            '！！': '！', '……': '…', '---': '—'
        }
        
        # 可疑字符模式（正则表达式）
        self.suspicious_patterns = [
            r'[^\u4e00-\u9fff\u3400-\u4dbf\s\n\r\t，。；：？！""''（）「」『』《》〈〉【】…—\-]',  # 非中文字符
            r'\d{4,}',                    # 连续4个或更多数字
            r'[A-Za-z]{3,}',              # 连续3个或更多英文字母
            r'[\s]{3,}',                  # 连续3个或更多空白字符
            r'[^\u4e00-\u9fff]{5,}',      # 连续5个或更多非汉字字符
            r'[。，；：！？]{3,}',         # 连续3个或更多标点符号
        ]
        
        # 古文常用字符集
        self.ancient_chinese_chars = set([
            '之', '乎', '者', '也', '矣', '焉', '哉', '兮',
            '於', '以', '为', '其', '而', '则', '若', '夫',
            '然', '故', '所', '与', '及', '乃', '即', '既',
            '且', '又', '亦', '或', '惟', '唯', '但', '只'
        ])
        
        # 现代标点符号集合
        self.modern_punctuations = set('，。；：？！""''（）「」『』《》〈〉【】…—')
        
        # 全角半角映射
        self.fullwidth_mappings = {
            '１': '1', '２': '2', '３': '3', '４': '4', '５': '5',
            '６': '6', '７': '7', '８': '8', '９': '9', '０': '0',
            'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
            'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
            'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
            'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
            'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z',
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
            'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
            'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
            'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
            'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z'
        }
    
    async def process_text_async(
        self,
        text: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, TextQualityMetrics]:
        """
        异步文本后处理主函数
        
        Args:
            text: 输入文本
            config: 处理配置参数
            
        Returns:
            Tuple[处理后文本, 文本质量评估]
        """
        try:
            logger.info("开始异步处理文本")
            
            # 默认配置
            default_config = {
                'normalize_whitespace': True,
                'normalize_punctuation': True,
                'traditional_to_simplified': False,
                'remove_suspicious_chars': True,
                'correct_ocr_errors': True,
                'normalize_level': TextNormalizeLevel.STANDARD,
                'preserve_line_breaks': True,
                'min_confidence_threshold': 0.7
            }
            
            processing_config = {**default_config, **(config or {})}
            
            # 执行处理管道
            processed_text = await self._execute_text_pipeline(text, processing_config)
            
            # 计算文本质量指标
            quality_metrics = await self._calculate_quality_metrics(processed_text)
            
            logger.info(f"文本处理完成，字符数: {quality_metrics.char_count}")
            return processed_text, quality_metrics
            
        except Exception as e:
            logger.error(f"文本处理失败: {str(e)}")
            raise
    
    async def _execute_text_pipeline(
        self,
        text: str,
        config: Dict[str, Any]
    ) -> str:
        """
        执行完整的文本处理流水线
        
        Args:
            text: 输入文本
            config: 处理配置
            
        Returns:
            处理后的文本
        """
        processed_text = text
        
        # 1. 基础清理和空白字符规范化
        if config.get('normalize_whitespace', True):
            processed_text = await self._normalize_whitespace_async(processed_text, config)
        
        # 2. 全角半角字符规范化
        processed_text = await self._normalize_fullwidth_chars_async(processed_text)
        
        # 3. 移除可疑字符
        if config.get('remove_suspicious_chars', True):
            processed_text = await self._remove_suspicious_chars_async(processed_text)
        
        # 4. OCR错误纠正
        if config.get('correct_ocr_errors', True):
            processed_text = await self._correct_ocr_errors_async(processed_text)
        
        # 5. 标点符号规范化
        if config.get('normalize_punctuation', True):
            processed_text = await self._normalize_punctuation_async(processed_text)
        
        # 6. 繁简转换
        if config.get('traditional_to_simplified', False):
            processed_text = await self._convert_traditional_to_simplified_async(processed_text)
        elif config.get('simplified_to_traditional', False):
            processed_text = await self._convert_simplified_to_traditional_async(processed_text)
        
        # 7. 最终清理和格式化
        processed_text = await self._final_cleanup_async(processed_text, config)
        
        return processed_text
    
    async def _normalize_whitespace_async(
        self, 
        text: str, 
        config: Dict[str, Any]
    ) -> str:
        """
        异步规范化空白字符
        
        Args:
            text: 输入文本
            config: 处理配置
            
        Returns:
            规范化后文本
        """
        def _normalize_sync():
            # 统一换行符
            normalized = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # 移除行首行尾空白
            lines = normalized.split('\n')
            lines = [line.strip() for line in lines]
            
            # 处理空行
            if not config.get('preserve_line_breaks', True):
                lines = [line for line in lines if line]
            
            # 规范化行内空白字符
            normalized_lines = []
            for line in lines:
                # 将多个空白字符替换为单个空格
                line = re.sub(r'\s+', ' ', line)
                normalized_lines.append(line)
            
            return '\n'.join(normalized_lines)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _normalize_sync
        )
    
    async def _normalize_fullwidth_chars_async(self, text: str) -> str:
        """
        异步规范化全角字符
        
        Args:
            text: 输入文本
            
        Returns:
            规范化后文本
        """
        def _normalize_sync():
            normalized = text
            for fullwidth, halfwidth in self.fullwidth_mappings.items():
                normalized = normalized.replace(fullwidth, halfwidth)
            return normalized
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _normalize_sync
        )
    
    async def _remove_suspicious_chars_async(self, text: str) -> str:
        """
        异步移除可疑字符
        
        Args:
            text: 输入文本
            
        Returns:
            清理后文本
        """
        def _remove_sync():
            cleaned = text
            
            # 移除控制字符（保留换行符和制表符）
            control_chars = ''.join([chr(i) for i in range(32) if i not in [9, 10, 13]])
            cleaned = ''.join(char for char in cleaned if char not in control_chars)
            
            # 移除零宽字符
            zero_width_chars = [
                '\u200b',  # 零宽空格
                '\u200c',  # 零宽非连字符
                '\u200d',  # 零宽连字符
                '\ufeff',  # 字节顺序标记
            ]
            for char in zero_width_chars:
                cleaned = cleaned.replace(char, '')
            
            return cleaned
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _remove_sync
        )
    
    async def _correct_ocr_errors_async(self, text: str) -> str:
        """
        异步纠正OCR错误
        
        Args:
            text: 输入文本
            
        Returns:
            纠错后文本
        """
        def _correct_sync():
            corrected = text
            
            # 应用预定义的错误映射
            for error_char, correct_char in self.ocr_error_mappings.items():
                corrected = corrected.replace(error_char, correct_char)
            
            # 处理常见的数字字母混淆（在中文语境中）
            # 在中文文本中，独立的数字字母很可能是识别错误
            if self._is_primarily_chinese(corrected):
                # 替换独立的英文字母为相似汉字
                letter_corrections = {
                    r'\bO\b': '零', r'\bl\b': '一', r'\bS\b': '三',
                    r'\bB\b': '八', r'\bA\b': '人', r'\bT\b': '十'
                }
                
                for pattern, replacement in letter_corrections.items():
                    corrected = re.sub(pattern, replacement, corrected)
            
            return corrected
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _correct_sync
        )
    
    async def _normalize_punctuation_async(self, text: str) -> str:
        """
        异步标点符号规范化
        
        Args:
            text: 输入文本
            
        Returns:
            规范化后文本
        """
        def _normalize_sync():
            normalized = text
            
            # 应用标点符号映射
            for english_punct, chinese_punct in self.punctuation_mappings.items():
                normalized = normalized.replace(english_punct, chinese_punct)
            
            # 处理重复标点符号
            normalized = re.sub(r'[。]{2,}', '。', normalized)
            normalized = re.sub(r'[，]{2,}', '，', normalized)
            normalized = re.sub(r'[？]{2,}', '？', normalized)
            normalized = re.sub(r'[！]{2,}', '！', normalized)
            normalized = re.sub(r'[…]{2,}', '…', normalized)
            
            # 修复标点符号间距
            normalized = re.sub(r'\s+([，。；：？！])', r'\1', normalized)  # 移除标点前空格
            normalized = re.sub(r'([，；：])(?!\s)', r'\1 ', normalized)    # 在某些标点后添加空格
            
            return normalized
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _normalize_sync
        )
    
    async def _convert_traditional_to_simplified_async(self, text: str) -> str:
        """
        异步繁体转简体
        
        Args:
            text: 输入文本
            
        Returns:
            转换后文本
        """
        def _convert_sync():
            if not HAS_OPENCC or ConversionType.TRADITIONAL_TO_SIMPLIFIED not in self.converters:
                logger.warning("OpenCC不可用，跳过繁简转换")
                return text
            
            try:
                converter = self.converters[ConversionType.TRADITIONAL_TO_SIMPLIFIED]
                return converter.convert(text)
            except Exception as e:
                logger.error(f"繁简转换失败: {str(e)}")
                return text
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _convert_sync
        )
    
    async def _convert_simplified_to_traditional_async(self, text: str) -> str:
        """
        异步简体转繁体
        
        Args:
            text: 输入文本
            
        Returns:
            转换后文本
        """
        def _convert_sync():
            if not HAS_OPENCC or ConversionType.SIMPLIFIED_TO_TRADITIONAL not in self.converters:
                logger.warning("OpenCC不可用，跳过繁简转换")
                return text
            
            try:
                converter = self.converters[ConversionType.SIMPLIFIED_TO_TRADITIONAL]
                return converter.convert(text)
            except Exception as e:
                logger.error(f"繁简转换失败: {str(e)}")
                return text
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _convert_sync
        )
    
    async def _final_cleanup_async(
        self, 
        text: str, 
        config: Dict[str, Any]
    ) -> str:
        """
        异步最终清理
        
        Args:
            text: 输入文本
            config: 处理配置
            
        Returns:
            最终清理后文本
        """
        def _cleanup_sync():
            cleaned = text
            
            # 移除多余的空行
            cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
            
            # 修整首尾空白
            cleaned = cleaned.strip()
            
            # 确保文本不为空
            if not cleaned:
                logger.warning("处理后文本为空")
                return text  # 返回原始文本
            
            return cleaned
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _cleanup_sync
        )
    
    async def _calculate_quality_metrics(self, text: str) -> TextQualityMetrics:
        """
        异步计算文本质量指标
        
        Args:
            text: 文本内容
            
        Returns:
            文本质量评估指标
        """
        def _calculate_sync():
            # 基础计数统计
            char_count = len(text)
            lines = text.split('\n')
            line_count = len(lines)
            non_empty_lines = [line for line in lines if line.strip()]
            paragraph_count = len([line for line in lines if line.strip() == '']) + 1
            
            # 中文字符统计
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            chinese_char_ratio = chinese_chars / char_count if char_count > 0 else 0
            
            # 标点符号统计
            punctuation_chars = len([char for char in text if char in self.modern_punctuations])
            punctuation_ratio = punctuation_chars / char_count if char_count > 0 else 0
            
            # 词语数量（如果有jieba）
            if HAS_JIEBA:
                words = list(jieba.cut(text))
                word_count = len([word for word in words if word.strip() and len(word) > 1])
                avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            else:
                word_count = len(text.split())
                avg_word_length = char_count / word_count if word_count > 0 else 0
            
            # 平均行长度
            avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0
            
            # 检测可疑模式
            suspicious_patterns = []
            error_char_count = 0
            
            for pattern in self.suspicious_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    suspicious_patterns.append(pattern)
                    error_char_count += sum(len(match) for match in matches)
            
            # 计算整体置信度分数
            confidence_score = self._calculate_confidence_score(
                chinese_char_ratio, punctuation_ratio, error_char_count, char_count
            )
            
            return TextQualityMetrics(
                char_count=char_count,
                word_count=word_count,
                line_count=line_count,
                paragraph_count=paragraph_count,
                confidence_score=confidence_score,
                error_char_count=error_char_count,
                punctuation_ratio=punctuation_ratio,
                chinese_char_ratio=chinese_char_ratio,
                suspicious_patterns=suspicious_patterns,
                avg_line_length=avg_line_length,
                avg_word_length=avg_word_length
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _calculate_sync
        )
    
    def _calculate_confidence_score(
        self,
        chinese_ratio: float,
        punct_ratio: float,
        error_count: int,
        total_chars: int
    ) -> float:
        """
        计算文本整体置信度分数
        
        Args:
            chinese_ratio: 中文字符比例
            punct_ratio: 标点符号比例
            error_count: 错误字符数量
            total_chars: 总字符数
            
        Returns:
            置信度分数 (0-1)
        """
        base_score = 1.0
        
        # 中文字符比例加权
        chinese_weight = min(chinese_ratio * 1.2, 1.0)
        
        # 标点符号比例评估（适度的标点符号是好的）
        if 0.05 <= punct_ratio <= 0.2:
            punct_weight = 1.0
        elif punct_ratio < 0.05:
            punct_weight = 0.8
        else:
            punct_weight = max(0.5, 1.0 - (punct_ratio - 0.2))
        
        # 错误字符惩罚
        if total_chars > 0:
            error_ratio = error_count / total_chars
            error_penalty = max(0.0, 1.0 - error_ratio * 3)
        else:
            error_penalty = 0.0
        
        # 综合计算
        confidence = base_score * chinese_weight * punct_weight * error_penalty
        return max(0.0, min(1.0, confidence))
    
    def _is_primarily_chinese(self, text: str) -> bool:
        """
        判断文本是否主要为中文
        
        Args:
            text: 文本内容
            
        Returns:
            是否主要为中文
        """
        if not text:
            return False
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        return (chinese_chars / total_chars) > 0.6 if total_chars > 0 else False
    
    async def batch_process_texts(
        self,
        texts: List[str],
        config: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 4
    ) -> List[Tuple[str, TextQualityMetrics]]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            config: 处理配置
            max_concurrent: 最大并发数
            
        Returns:
            处理结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(text):
            async with semaphore:
                return await self.process_text_async(text, config)
        
        tasks = [process_single(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        logger.info(f"批量处理完成: {len(texts)} 个文本")
        return results
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# 便捷函数
async def process_text(text: str, **kwargs) -> Tuple[str, TextQualityMetrics]:
    """
    便捷的文本处理函数
    
    Args:
        text: 输入文本
        **kwargs: 处理配置参数
        
    Returns:
        Tuple[处理后文本, 质量指标]
    """
    processor = TextProcessor()
    try:
        return await processor.process_text_async(text, kwargs)
    finally:
        del processor


def create_ancient_text_config() -> Dict[str, Any]:
    """
    创建专门用于古籍文本处理的配置
    
    Returns:
        古籍优化配置字典
    """
    return {
        'normalize_whitespace': True,
        'normalize_punctuation': True,
        'traditional_to_simplified': True,  # 古籍通常需要繁简转换
        'remove_suspicious_chars': True,
        'correct_ocr_errors': True,
        'normalize_level': TextNormalizeLevel.AGGRESSIVE,
        'preserve_line_breaks': True,
        'min_confidence_threshold': 0.6  # 古籍文本容忍度更高
    }