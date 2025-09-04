"""
文本提取器基类

定义文本提取的通用接口和基础功能
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TextExtractor(ABC):
    """文本提取器基类
    
    定义所有文本提取器的通用接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化文本提取器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"{self.__class__.__name__} 初始化完成")
    
    @abstractmethod
    async def extract(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """提取文本内容（抽象方法）
        
        Args:
            file_path: 文件路径
            **kwargs: 其他参数
            
        Returns:
            提取的文本内容列表，每个元素包含:
            - page_number: 页码或段落号
            - content: 文本内容
            - word_count: 词数
            - char_count: 字符数
            - title: 标题(可选)
            - confidence: 置信度(可选)
            - metadata: 其他元数据(可选)
        """
        pass
    
    @abstractmethod
    def supports_file_type(self, file_type: str) -> bool:
        """检查是否支持指定文件类型
        
        Args:
            file_type: MIME类型
            
        Returns:
            是否支持
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """验证文件是否可以处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否可以处理
        """
        try:
            import os
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.logger.error(f"文件不存在: {file_path}")
                return False
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            max_size = self.config.get('max_file_size', 100 * 1024 * 1024)  # 默认100MB
            
            if file_size > max_size:
                self.logger.error(f"文件过大: {file_size} > {max_size}")
                return False
            
            if file_size == 0:
                self.logger.error(f"文件为空: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"文件验证失败: {str(e)}")
            return False
    
    def calculate_text_stats(self, text: str) -> Dict[str, int]:
        """计算文本统计信息
        
        Args:
            text: 文本内容
            
        Returns:
            统计信息字典
        """
        import re
        
        if not text:
            return {"word_count": 0, "char_count": 0}
        
        # 字符数统计
        char_count = len(text)
        
        # 词数统计 - 支持中英文混合
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文单词
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        # 统计数字
        numbers = len(re.findall(r'\b\d+\b', text))
        
        # 总词数 = 中文字符数 + 英文单词数 + 数字数
        word_count = chinese_chars + english_words + numbers
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "chinese_chars": chinese_chars,
            "english_words": english_words,
            "numbers": numbers
        }
    
    def clean_text(self, text: str) -> str:
        """清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        import re
        
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除首尾空白
        text = text.strip()
        
        # 移除特殊控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def detect_language(self, text: str) -> Optional[str]:
        """检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            语言代码，如 'zh-cn', 'en'
        """
        if not text or len(text.strip()) < 10:
            return None
        
        try:
            # 尝试使用langdetect库
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # 确保结果一致性
            return detect(text)
        except ImportError:
            self.logger.warning("langdetect 库未安装，使用简单语言检测")
        except Exception as e:
            self.logger.warning(f"语言检测失败: {str(e)}")
        
        # 简单的语言检测
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return None
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3:
            return 'zh-cn'
        elif chinese_ratio > 0.1:
            return 'zh-en'  # 中英文混合
        else:
            return 'en'
    
    def split_text_by_sentences(self, text: str, max_length: int = 1000) -> List[str]:
        """按句子分割长文本
        
        Args:
            text: 文本内容
            max_length: 最大长度
            
        Returns:
            分割后的文本列表
        """
        if len(text) <= max_length:
            return [text]
        
        import re
        
        # 按句子分割（中英文标点）
        sentences = re.split(r'[。！？；\.\!\?\;]', text)
        
        result = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 如果加上当前句子会超长，先保存当前块
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                result.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence
        
        # 添加最后一块
        if current_chunk:
            result.append(current_chunk.strip())
        
        return result
    
    def create_text_content(
        self,
        content: str,
        page_number: int = 1,
        title: str = None,
        confidence: float = None,
        **metadata
    ) -> Dict[str, Any]:
        """创建标准化的文本内容字典
        
        Args:
            content: 文本内容
            page_number: 页码
            title: 标题
            confidence: 置信度
            **metadata: 其他元数据
            
        Returns:
            标准化的文本内容字典
        """
        # 清理文本
        clean_content = self.clean_text(content)
        
        # 计算统计信息
        stats = self.calculate_text_stats(clean_content)
        
        # 检测语言
        language = self.detect_language(clean_content)
        
        result = {
            "content": clean_content,
            "page_number": page_number,
            "word_count": stats["word_count"],
            "char_count": stats["char_count"],
            "language": language
        }
        
        # 添加可选字段
        if title:
            result["title"] = title
        
        if confidence is not None:
            result["confidence"] = confidence
        
        # 添加元数据
        if metadata:
            result["metadata"] = metadata
        
        return result