"""
纯文本处理器
专门处理纯文本文件(.txt)和HTML文件的文本提取
"""

import io
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import chardet
from bs4 import BeautifulSoup
import html

logger = logging.getLogger(__name__)


class TextProcessor:
    """纯文本和HTML处理器
    
    提供纯文本、HTML文件的文本提取功能
    支持多种文本编码的自动检测
    """
    
    def __init__(self):
        """初始化文本处理器"""
        self.supported_extensions = ['.txt', '.html', '.htm']
        logger.info("文本处理器初始化完成")
    
    async def process(
        self, 
        content: bytes, 
        extract_text: bool = True,
        extract_metadata: bool = True,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理文本文件
        
        Args:
            content: 文件的二进制内容
            extract_text: 是否提取文本内容
            extract_metadata: 是否提取元数据
            filename: 文件名（可选）
            
        Returns:
            包含处理结果的字典
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"开始处理文本文件，大小: {len(content)} bytes")
            
            file_ext = Path(filename or "").suffix.lower() if filename else '.txt'
            
            result = {
                "filename": filename or f"unknown{file_ext}",
                "size": len(content),
                "format": "html" if file_ext in ['.html', '.htm'] else "text",
                "text_content": "",
                "metadata": {},
                "encoding": "unknown",
                "processing_time": 0,
                "success": True,
                "warnings": []
            }
            
            # 检测文件编码
            encoding_info = chardet.detect(content)
            detected_encoding = encoding_info.get('encoding', 'utf-8')
            encoding_confidence = encoding_info.get('confidence', 0.0)
            
            result["encoding"] = detected_encoding
            result["encoding_confidence"] = round(encoding_confidence, 3)
            
            logger.debug(f"检测到编码: {detected_encoding} (置信度: {encoding_confidence:.3f})")
            
            # 提取文本内容
            if extract_text:
                try:
                    text_content = await self._extract_text(content, detected_encoding, file_ext)
                    result["text_content"] = text_content
                    logger.info(f"成功提取文本，文本长度: {len(text_content)}")
                    
                except Exception as e:
                    logger.error(f"文本提取失败: {e}")
                    result["text_content"] = ""
                    result["warnings"].append(f"文本提取失败: {str(e)}")
            
            # 提取元数据
            if extract_metadata:
                try:
                    metadata = await self._extract_metadata(content, detected_encoding, file_ext)
                    result["metadata"] = metadata
                    logger.debug(f"成功提取元数据: {list(metadata.keys())}")
                    
                except Exception as e:
                    logger.warning(f"元数据提取失败: {e}")
                    result["warnings"].append(f"元数据提取失败: {str(e)}")
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time"] = round(processing_time, 3)
            
            logger.info(f"文本文件处理完成，处理时间: {processing_time:.3f}秒")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"文本文件处理失败: {e}")
            
            return {
                "filename": filename or "unknown.txt",
                "size": len(content),
                "format": "text",
                "text_content": "",
                "metadata": {},
                "encoding": "unknown",
                "processing_time": round(processing_time, 3),
                "success": False,
                "error": str(e)
            }
    
    async def _extract_text(self, content: bytes, encoding: str, file_ext: str) -> str:
        """提取文本内容
        
        Args:
            content: 文件二进制内容
            encoding: 文件编码
            file_ext: 文件扩展名
            
        Returns:
            提取的文本内容
        """
        try:
            # 解码为文本
            text_content = content.decode(encoding or 'utf-8', errors='replace')
            
            if file_ext in ['.html', '.htm']:
                # HTML文件特殊处理
                return await self._extract_html_text(text_content)
            else:
                # 纯文本文件
                return self._clean_text(text_content)
                
        except Exception as e:
            # 尝试其他编码
            for fallback_encoding in ['utf-8', 'gbk', 'gb2312', 'big5', 'latin1']:
                try:
                    text_content = content.decode(fallback_encoding, errors='replace')
                    logger.warning(f"使用备选编码 {fallback_encoding} 解码成功")
                    
                    if file_ext in ['.html', '.htm']:
                        return await self._extract_html_text(text_content)
                    else:
                        return self._clean_text(text_content)
                        
                except:
                    continue
            
            raise Exception(f"无法解码文本内容: {str(e)}")
    
    async def _extract_html_text(self, html_content: str) -> str:
        """从HTML内容中提取纯文本
        
        Args:
            html_content: HTML源代码
            
        Returns:
            提取的纯文本内容
        """
        try:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除脚本和样式标签
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 提取文本内容
            text_parts = []
            
            # 提取标题
            title = soup.find('title')
            if title and title.string:
                text_parts.append(f"=== 标题 ===\n{title.string.strip()}")
            
            # 提取正文内容
            body = soup.find('body')
            if body:
                body_text = body.get_text(separator='\n', strip=True)
            else:
                body_text = soup.get_text(separator='\n', strip=True)
            
            if body_text:
                text_parts.append(f"=== 正文内容 ===\n{body_text}")
            
            full_text = '\n\n'.join(text_parts)
            return self._clean_text(full_text)
            
        except Exception as e:
            logger.warning(f"HTML解析失败，返回原始文本: {e}")
            # 如果HTML解析失败，至少去除HTML标签
            return self._clean_html_tags(html_content)
    
    def _clean_html_tags(self, html_content: str) -> str:
        """简单去除HTML标签
        
        Args:
            html_content: HTML源代码
            
        Returns:
            去除标签后的文本
        """
        try:
            # 解码HTML实体
            text = html.unescape(html_content)
            
            # 简单的标签去除（不如BeautifulSoup精确，但是备用方案）
            import re
            text = re.sub(r'<[^>]+>', '', text)
            
            return self._clean_text(text)
            
        except Exception as e:
            logger.warning(f"HTML标签清理失败: {e}")
            return html_content
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 标准化换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 去除每行前后的空白
        lines = [line.strip() for line in text.split('\n')]
        
        # 去除过多的空行，保留段落间距
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            else:
                # 只保留一个空行作为段落分隔
                if not prev_empty and cleaned_lines:
                    cleaned_lines.append("")
                prev_empty = True
        
        return '\n'.join(cleaned_lines).strip()
    
    async def _extract_metadata(self, content: bytes, encoding: str, file_ext: str) -> Dict[str, Any]:
        """提取文件元数据
        
        Args:
            content: 文件二进制内容
            encoding: 文件编码
            file_ext: 文件扩展名
            
        Returns:
            包含元数据的字典
        """
        metadata = {
            "file_type": "html" if file_ext in ['.html', '.htm'] else "text",
            "encoding": encoding,
            "size_bytes": len(content)
        }
        
        try:
            # 解码文本以获取更多信息
            text_content = content.decode(encoding or 'utf-8', errors='replace')
            
            # 基本统计
            metadata["character_count"] = len(text_content)
            metadata["line_count"] = text_content.count('\n') + 1
            metadata["word_count"] = len(text_content.split())
            
            if file_ext in ['.html', '.htm']:
                # HTML特定元数据
                metadata.update(await self._extract_html_metadata(text_content))
            
        except Exception as e:
            logger.warning(f"元数据提取过程中发生错误: {e}")
            metadata["extraction_error"] = str(e)
        
        return metadata
    
    async def _extract_html_metadata(self, html_content: str) -> Dict[str, Any]:
        """提取HTML特定的元数据
        
        Args:
            html_content: HTML源代码
            
        Returns:
            HTML元数据字典
        """
        html_metadata = {}
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取title标签
            title = soup.find('title')
            if title and title.string:
                html_metadata["title"] = title.string.strip()
            
            # 提取meta标签信息
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                if meta.get('name') and meta.get('content'):
                    html_metadata[f"meta_{meta.get('name')}"] = meta.get('content')
                elif meta.get('property') and meta.get('content'):
                    html_metadata[f"meta_{meta.get('property')}"] = meta.get('content')
            
            # 统计HTML元素
            html_metadata["tag_counts"] = {
                "paragraphs": len(soup.find_all('p')),
                "headings": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                "links": len(soup.find_all('a')),
                "images": len(soup.find_all('img')),
                "tables": len(soup.find_all('table'))
            }
            
        except Exception as e:
            logger.warning(f"HTML元数据提取失败: {e}")
            html_metadata["html_parsing_error"] = str(e)
        
        return html_metadata
    
    def is_supported(self, filename: str) -> bool:
        """检查文件是否支持
        
        Args:
            filename: 文件名
            
        Returns:
            是否支持该文件格式
        """
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.supported_extensions