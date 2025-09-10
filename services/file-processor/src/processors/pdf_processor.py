"""
PDF文档处理器
专门处理PDF文件的文本提取和元数据获取
"""

import io
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import PyPDF2
import pdfplumber
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFProcessor:
    """PDF文档处理器
    
    提供PDF文件的文本提取、元数据获取等功能
    支持多种PDF格式，包括文本型PDF和扫描型PDF
    """
    
    def __init__(self):
        """初始化PDF处理器"""
        self.supported_extensions = ['.pdf']
        logger.info("PDF处理器初始化完成")
    
    async def process(
        self, 
        content: bytes, 
        extract_text: bool = True,
        extract_metadata: bool = True,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理PDF文件
        
        Args:
            content: PDF文件的二进制内容
            extract_text: 是否提取文本内容
            extract_metadata: 是否提取元数据
            filename: 文件名（可选）
            
        Returns:
            包含处理结果的字典
            
        Raises:
            Exception: 处理过程中发生错误时抛出
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"开始处理PDF文件，大小: {len(content)} bytes")
            
            result = {
                "filename": filename or "unknown.pdf",
                "size": len(content),
                "format": "pdf",
                "processing_method": None,
                "text_content": "",
                "metadata": {},
                "pages": 0,
                "processing_time": 0,
                "success": True,
                "warnings": []
            }
            
            # 使用BytesIO包装二进制内容
            pdf_buffer = io.BytesIO(content)
            
            # 首先尝试使用pdfplumber提取文本（推荐，支持更复杂的布局）
            if extract_text:
                try:
                    text_content, page_count = await self._extract_text_with_pdfplumber(pdf_buffer)
                    result["text_content"] = text_content
                    result["pages"] = page_count
                    result["processing_method"] = "pdfplumber"
                    logger.info(f"pdfplumber成功提取文本，页数: {page_count}，文本长度: {len(text_content)}")
                    
                except Exception as e:
                    logger.warning(f"pdfplumber提取失败，尝试PyPDF2: {e}")
                    # 回退到PyPDF2
                    pdf_buffer.seek(0)  # 重置缓冲区位置
                    try:
                        text_content, page_count = await self._extract_text_with_pypdf2(pdf_buffer)
                        result["text_content"] = text_content
                        result["pages"] = page_count
                        result["processing_method"] = "pypdf2"
                        result["warnings"].append("使用PyPDF2作为备选方案")
                        logger.info(f"PyPDF2成功提取文本，页数: {page_count}，文本长度: {len(text_content)}")
                        
                    except Exception as e2:
                        logger.error(f"PyPDF2也提取失败: {e2}")
                        result["text_content"] = ""
                        result["processing_method"] = "failed"
                        result["warnings"].extend([
                            f"pdfplumber提取失败: {str(e)}",
                            f"PyPDF2提取失败: {str(e2)}"
                        ])
            
            # 提取元数据
            if extract_metadata:
                pdf_buffer.seek(0)  # 重置缓冲区位置
                try:
                    metadata = await self._extract_metadata(pdf_buffer)
                    result["metadata"] = metadata
                    logger.debug(f"成功提取元数据: {list(metadata.keys())}")
                    
                except Exception as e:
                    logger.warning(f"元数据提取失败: {e}")
                    result["warnings"].append(f"元数据提取失败: {str(e)}")
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time"] = round(processing_time, 3)
            
            logger.info(f"PDF处理完成，处理时间: {processing_time:.3f}秒")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"PDF处理失败: {e}")
            
            return {
                "filename": filename or "unknown.pdf",
                "size": len(content),
                "format": "pdf", 
                "text_content": "",
                "metadata": {},
                "pages": 0,
                "processing_time": round(processing_time, 3),
                "success": False,
                "error": str(e)
            }
    
    async def _extract_text_with_pdfplumber(self, pdf_buffer: io.BytesIO) -> tuple[str, int]:
        """使用pdfplumber提取文本
        
        Args:
            pdf_buffer: PDF文件的BytesIO缓冲区
            
        Returns:
            tuple: (提取的文本内容, 页数)
        """
        text_parts = []
        page_count = 0
        
        with pdfplumber.open(pdf_buffer) as pdf:
            page_count = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # 提取页面文本
                    page_text = page.extract_text()
                    
                    if page_text:
                        # 清理文本：去除多余空白，标准化换行
                        cleaned_text = self._clean_extracted_text(page_text)
                        text_parts.append(f"=== 第{page_num}页 ===\n{cleaned_text}")
                        
                    else:
                        logger.debug(f"第{page_num}页未提取到文本")
                        
                except Exception as e:
                    logger.warning(f"第{page_num}页文本提取失败: {e}")
                    continue
        
        full_text = "\n\n".join(text_parts)
        return full_text, page_count
    
    async def _extract_text_with_pypdf2(self, pdf_buffer: io.BytesIO) -> tuple[str, int]:
        """使用PyPDF2提取文本
        
        Args:
            pdf_buffer: PDF文件的BytesIO缓冲区
            
        Returns:
            tuple: (提取的文本内容, 页数)
        """
        text_parts = []
        
        reader = PyPDF2.PdfReader(pdf_buffer)
        page_count = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text()
                
                if page_text:
                    cleaned_text = self._clean_extracted_text(page_text)
                    text_parts.append(f"=== 第{page_num}页 ===\n{cleaned_text}")
                    
                else:
                    logger.debug(f"第{page_num}页未提取到文本")
                    
            except Exception as e:
                logger.warning(f"第{page_num}页文本提取失败: {e}")
                continue
        
        full_text = "\n\n".join(text_parts)
        return full_text, page_count
    
    async def _extract_metadata(self, pdf_buffer: io.BytesIO) -> Dict[str, Any]:
        """提取PDF元数据
        
        Args:
            pdf_buffer: PDF文件的BytesIO缓冲区
            
        Returns:
            包含元数据的字典
        """
        metadata = {}
        
        try:
            reader = PyPDF2.PdfReader(pdf_buffer)
            
            # 基本信息
            metadata["pages"] = len(reader.pages)
            
            # 文档信息
            if reader.metadata:
                raw_metadata = reader.metadata
                
                # 标准元数据字段
                metadata_mappings = {
                    '/Title': 'title',
                    '/Author': 'author', 
                    '/Subject': 'subject',
                    '/Creator': 'creator',
                    '/Producer': 'producer',
                    '/CreationDate': 'creation_date',
                    '/ModDate': 'modification_date',
                    '/Keywords': 'keywords'
                }
                
                for pdf_key, json_key in metadata_mappings.items():
                    if pdf_key in raw_metadata:
                        value = raw_metadata[pdf_key]
                        if value:
                            metadata[json_key] = str(value).strip()
            
            # 加密信息
            metadata["encrypted"] = reader.is_encrypted
            
            # 版本信息
            if hasattr(reader, 'pdf_header'):
                metadata["pdf_version"] = reader.pdf_header
            
            logger.debug(f"成功提取元数据字段: {list(metadata.keys())}")
            
        except Exception as e:
            logger.warning(f"元数据提取过程中发生错误: {e}")
            metadata["extraction_error"] = str(e)
        
        return metadata
    
    def _clean_extracted_text(self, text: str) -> str:
        """清理提取的文本
        
        Args:
            text: 原始提取的文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 标准化换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 去除每行前后的空白
        lines = [line.strip() for line in text.split('\n')]
        
        # 去除空行（但保留段落间距）
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
    
    def is_supported(self, filename: str) -> bool:
        """检查文件是否支持
        
        Args:
            filename: 文件名
            
        Returns:
            是否支持该文件格式
        """
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.supported_extensions