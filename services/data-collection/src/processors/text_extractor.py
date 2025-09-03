"""
纯文本提取器

处理纯文本文件的内容提取
"""

import logging
import chardet
from typing import Dict, List, Any

from .base import TextExtractor

logger = logging.getLogger(__name__)


class PlainTextExtractor(TextExtractor):
    """纯文本文件提取器
    
    支持各种编码的纯文本文件提取
    """
    
    SUPPORTED_TYPES = {
        'text/plain',
        'text/txt',
        'application/x-text',
        'text/x-text'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化纯文本提取器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.max_chunk_size = config.get('max_chunk_size', 5000)
        self.auto_detect_encoding = config.get('auto_detect_encoding', True)
        self.fallback_encoding = config.get('fallback_encoding', 'utf-8')
        self.split_by_paragraphs = config.get('split_by_paragraphs', True)
    
    def supports_file_type(self, file_type: str) -> bool:
        """检查是否支持指定文件类型"""
        return file_type in self.SUPPORTED_TYPES
    
    async def extract(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """从纯文本文件提取内容
        
        Args:
            file_path: 文本文件路径
            **kwargs: 其他参数
            
        Returns:
            提取的文本内容列表
        """
        if not self.validate_file(file_path):
            raise ValueError(f"文件验证失败: {file_path}")
        
        try:
            # 检测文件编码
            encoding = await self._detect_encoding(file_path)
            
            # 读取文件内容
            content = await self._read_text_file(file_path, encoding)
            
            # 处理文本内容
            contents = await self._process_text_content(content)
            
            self.logger.info(f"纯文本提取完成: {file_path}, 段落数: {len(contents)}, 编码: {encoding}")
            return contents
            
        except Exception as e:
            self.logger.error(f"纯文本提取失败: {file_path}, 错误: {str(e)}")
            raise
    
    async def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码
        
        Args:
            file_path: 文件路径
            
        Returns:
            检测到的编码
        """
        if not self.auto_detect_encoding:
            return self.fallback_encoding
        
        try:
            # 读取文件的一部分用于编码检测
            with open(file_path, 'rb') as file:
                raw_data = file.read(10240)  # 读取前10KB
            
            # 使用chardet检测编码
            detected = chardet.detect(raw_data)
            
            if detected and detected['encoding']:
                confidence = detected.get('confidence', 0)
                encoding = detected['encoding']
                
                self.logger.debug(f"检测到编码: {encoding}, 置信度: {confidence}")
                
                # 如果置信度足够高，使用检测到的编码
                if confidence > 0.7:
                    return encoding
            
            # 否则使用备用编码
            return self.fallback_encoding
            
        except Exception as e:
            self.logger.warning(f"编码检测失败: {str(e)}, 使用备用编码: {self.fallback_encoding}")
            return self.fallback_encoding
    
    async def _read_text_file(self, file_path: str, encoding: str) -> str:
        """读取文本文件内容
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            
        Returns:
            文件内容
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
            
        except UnicodeDecodeError as e:
            self.logger.warning(f"使用编码 {encoding} 读取失败: {str(e)}")
            
            # 尝试其他常见编码
            fallback_encodings = ['utf-8', 'gb2312', 'gbk', 'gb18030', 'big5', 'iso-8859-1']
            
            for fallback_encoding in fallback_encodings:
                if fallback_encoding == encoding:
                    continue
                
                try:
                    with open(file_path, 'r', encoding=fallback_encoding) as file:
                        content = file.read()
                    
                    self.logger.info(f"使用备用编码成功: {fallback_encoding}")
                    return content
                    
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，使用errors='ignore'
            self.logger.warning("所有编码尝试失败，使用ignore模式")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            return content
    
    async def _process_text_content(self, content: str) -> List[Dict[str, Any]]:
        """处理文本内容
        
        Args:
            content: 原始文本内容
            
        Returns:
            处理后的内容列表
        """
        if not content or not content.strip():
            return []
        
        contents = []
        
        if self.split_by_paragraphs:
            # 按段落分割
            paragraphs = self._split_by_paragraphs(content)
            
            for idx, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    # 如果段落太长，进一步分割
                    if len(paragraph) > self.max_chunk_size:
                        chunks = self.split_text_by_sentences(paragraph, self.max_chunk_size)
                        for chunk_idx, chunk in enumerate(chunks):
                            content_item = self.create_text_content(
                                content=chunk,
                                page_number=idx + 1,
                                title=f"Paragraph {idx + 1} - Part {chunk_idx + 1}",
                                paragraph_index=idx,
                                chunk_index=chunk_idx
                            )
                            contents.append(content_item)
                    else:
                        content_item = self.create_text_content(
                            content=paragraph,
                            page_number=idx + 1,
                            title=f"Paragraph {idx + 1}",
                            paragraph_index=idx
                        )
                        contents.append(content_item)
        else:
            # 不按段落分割，直接按大小分块
            if len(content) > self.max_chunk_size:
                chunks = self.split_text_by_sentences(content, self.max_chunk_size)
                for idx, chunk in enumerate(chunks):
                    content_item = self.create_text_content(
                        content=chunk,
                        page_number=idx + 1,
                        title=f"Text Block {idx + 1}",
                        chunk_index=idx
                    )
                    contents.append(content_item)
            else:
                content_item = self.create_text_content(
                    content=content,
                    page_number=1,
                    title="Text Content"
                )
                contents.append(content_item)
        
        return contents
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """按段落分割文本
        
        Args:
            content: 文本内容
            
        Returns:
            段落列表
        """
        import re
        
        # 先按双换行分割（常见的段落分隔）
        paragraphs = re.split(r'\n\s*\n', content)
        
        # 如果没有双换行，按单换行分割
        if len(paragraphs) == 1:
            paragraphs = content.split('\n')
        
        # 清理空段落和首尾空白
        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = para.strip()
            if cleaned:
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def _detect_text_structure(self, content: str) -> Dict[str, Any]:
        """检测文本结构
        
        Args:
            content: 文本内容
            
        Returns:
            结构信息
        """
        import re
        
        structure = {
            'has_headers': False,
            'has_lists': False,
            'has_code_blocks': False,
            'line_count': 0,
            'paragraph_count': 0,
            'avg_line_length': 0
        }
        
        lines = content.split('\n')
        structure['line_count'] = len(lines)
        
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            structure['avg_line_length'] = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        
        # 检测标题（以#开头或全大写短行）
        header_patterns = [
            r'^#+\s+',  # Markdown风格标题
            r'^[A-Z\s]{3,50}$',  # 全大写短行
            r'^第.*章|第.*节|第.*部分'  # 中文章节
        ]
        
        for line in lines:
            line = line.strip()
            if any(re.match(pattern, line) for pattern in header_patterns):
                structure['has_headers'] = True
                break
        
        # 检测列表
        list_patterns = [
            r'^\s*[\-\*\+]\s+',  # 无序列表
            r'^\s*\d+\.\s+',     # 有序列表
            r'^\s*[a-zA-Z]\.\s+' # 字母列表
        ]
        
        for line in lines:
            if any(re.match(pattern, line) for pattern in list_patterns):
                structure['has_lists'] = True
                break
        
        # 检测代码块
        if '```' in content or re.search(r'^\s{4,}', content, re.MULTILINE):
            structure['has_code_blocks'] = True
        
        # 段落数估算
        structure['paragraph_count'] = len(re.split(r'\n\s*\n', content))
        
        return structure