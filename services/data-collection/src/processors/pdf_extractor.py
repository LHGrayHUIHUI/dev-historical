"""
PDF文本提取器

支持从PDF文件中提取文本内容
"""

import logging
from io import BytesIO
from typing import Dict, List, Any

from .base import TextExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(TextExtractor):
    """PDF文本提取器
    
    支持从PDF文件中提取文本，包括处理加密PDF和图像PDF
    """
    
    SUPPORTED_TYPES = {
        'application/pdf',
        'application/x-pdf'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化PDF提取器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.use_pdfplumber = config.get('use_pdfplumber', True)
        self.extract_images = config.get('extract_images', False)
        self.ocr_fallback = config.get('ocr_fallback', True)
    
    def supports_file_type(self, file_type: str) -> bool:
        """检查是否支持指定文件类型"""
        return file_type in self.SUPPORTED_TYPES
    
    async def extract(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """从PDF文件提取文本内容
        
        Args:
            file_path: PDF文件路径
            **kwargs: 其他参数
            
        Returns:
            提取的文本内容列表
        """
        if not self.validate_file(file_path):
            raise ValueError(f"文件验证失败: {file_path}")
        
        contents = []
        
        try:
            # 优先使用pdfplumber（更好的文本提取质量）
            if self.use_pdfplumber:
                contents = await self._extract_with_pdfplumber(file_path)
            else:
                contents = await self._extract_with_pypdf2(file_path)
            
            # 如果提取的内容很少，可能是图像PDF，尝试OCR
            if self.ocr_fallback and self._should_use_ocr(contents):
                self.logger.info(f"PDF文本内容较少，尝试OCR: {file_path}")
                ocr_contents = await self._extract_with_ocr(file_path)
                if ocr_contents and len(ocr_contents) > len(contents):
                    contents = ocr_contents
            
            self.logger.info(f"PDF提取完成: {file_path}, 页数: {len(contents)}")
            return contents
            
        except Exception as e:
            self.logger.error(f"PDF文本提取失败: {file_path}, 错误: {str(e)}")
            raise
    
    async def _extract_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        """使用pdfplumber提取PDF文本
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            提取的文本内容列表
        """
        contents = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        # 提取文本
                        text = page.extract_text()
                        
                        if text and text.strip():
                            # 获取页面元数据
                            page_metadata = {
                                'page_width': page.width,
                                'page_height': page.height,
                                'page_rotation': getattr(page, 'rotation', 0)
                            }
                            
                            # 创建文本内容
                            content_item = self.create_text_content(
                                content=text,
                                page_number=page_num,
                                title=f"Page {page_num}",
                                **page_metadata
                            )
                            
                            contents.append(content_item)
                            
                        # 提取表格（可选）
                        if self.config.get('extract_tables', False):
                            tables = page.extract_tables()
                            for table_idx, table in enumerate(tables):
                                if table:
                                    table_text = self._format_table_as_text(table)
                                    if table_text:
                                        table_content = self.create_text_content(
                                            content=table_text,
                                            page_number=page_num,
                                            title=f"Page {page_num} - Table {table_idx + 1}",
                                            table_index=table_idx
                                        )
                                        contents.append(table_content)
                    
                    except Exception as e:
                        self.logger.warning(f"提取PDF第{page_num}页失败: {str(e)}")
                        continue
            
            return contents
            
        except ImportError:
            self.logger.warning("pdfplumber 未安装，回退到PyPDF2")
            return await self._extract_with_pypdf2(file_path)
    
    async def _extract_with_pypdf2(self, file_path: str) -> List[Dict[str, Any]]:
        """使用PyPDF2提取PDF文本
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            提取的文本内容列表
        """
        contents = []
        
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # 检查是否加密
                if pdf_reader.is_encrypted:
                    # 尝试空密码
                    if not pdf_reader.decrypt(''):
                        self.logger.error(f"PDF文件已加密且无法解密: {file_path}")
                        return contents
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        
                        if text and text.strip():
                            content_item = self.create_text_content(
                                content=text,
                                page_number=page_num,
                                title=f"Page {page_num}"
                            )
                            
                            contents.append(content_item)
                    
                    except Exception as e:
                        self.logger.warning(f"提取PDF第{page_num}页失败: {str(e)}")
                        continue
            
            return contents
            
        except ImportError:
            self.logger.error("PyPDF2 未安装")
            raise ImportError("需要安装 PyPDF2 或 pdfplumber")
    
    async def _extract_with_ocr(self, file_path: str) -> List[Dict[str, Any]]:
        """使用OCR提取PDF图像中的文本
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            提取的文本内容列表
        """
        contents = []
        
        try:
            # 导入OCR相关库
            from PIL import Image
            import pytesseract
            import fitz  # PyMuPDF
            
            # 打开PDF
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                try:
                    page = pdf_document.load_page(page_num)
                    
                    # 将PDF页面转换为图像
                    mat = fitz.Matrix(2.0, 2.0)  # 2倍缩放以提高OCR质量
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # 使用PIL打开图像
                    image = Image.open(BytesIO(img_data))
                    
                    # 设置Tesseract语言
                    languages = self.config.get('ocr_languages', ['chi_sim', 'eng'])
                    lang_string = '+'.join(languages)
                    
                    # OCR识别
                    ocr_config = '--oem 3 --psm 6'  # 使用LSTM引擎，统一文本块模式
                    text = pytesseract.image_to_string(
                        image, 
                        lang=lang_string,
                        config=ocr_config
                    )
                    
                    if text and text.strip():
                        # 计算OCR置信度（简化版本）
                        confidence = self._estimate_ocr_confidence(text)
                        
                        content_item = self.create_text_content(
                            content=text,
                            page_number=page_num + 1,
                            title=f"Page {page_num + 1} (OCR)",
                            confidence=confidence,
                            extraction_method='ocr'
                        )
                        
                        contents.append(content_item)
                
                except Exception as e:
                    self.logger.warning(f"OCR提取第{page_num + 1}页失败: {str(e)}")
                    continue
            
            pdf_document.close()
            return contents
            
        except ImportError as e:
            self.logger.warning(f"OCR依赖库未安装: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"OCR提取失败: {str(e)}")
            return []
    
    def _should_use_ocr(self, contents: List[Dict[str, Any]]) -> bool:
        """判断是否应该使用OCR
        
        Args:
            contents: 已提取的文本内容
            
        Returns:
            是否应该使用OCR
        """
        if not contents:
            return True
        
        # 计算平均字符数
        total_chars = sum(item.get('char_count', 0) for item in contents)
        avg_chars_per_page = total_chars / len(contents) if contents else 0
        
        # 如果平均每页字符数少于50，可能是图像PDF
        return avg_chars_per_page < 50
    
    def _format_table_as_text(self, table: List[List[str]]) -> str:
        """将表格格式化为文本
        
        Args:
            table: 表格数据
            
        Returns:
            格式化后的文本
        """
        if not table:
            return ""
        
        formatted_rows = []
        for row in table:
            if row:
                # 过滤空值并连接
                clean_row = [cell.strip() if cell else "" for cell in row]
                formatted_rows.append(" | ".join(clean_row))
        
        return "\n".join(formatted_rows)
    
    def _estimate_ocr_confidence(self, text: str) -> float:
        """估算OCR置信度
        
        Args:
            text: OCR识别的文本
            
        Returns:
            置信度分数 (0-1)
        """
        if not text:
            return 0.0
        
        # 简单的置信度估算
        import re
        
        # 检查特殊字符比例
        special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return 0.0
        
        special_ratio = special_chars / total_chars
        
        # 基础置信度
        confidence = 0.8
        
        # 特殊字符太多降低置信度
        if special_ratio > 0.3:
            confidence *= 0.6
        elif special_ratio > 0.1:
            confidence *= 0.8
        
        # 文本长度影响
        if total_chars < 10:
            confidence *= 0.5
        elif total_chars < 50:
            confidence *= 0.7
        
        return round(confidence, 3)