"""
HTML文本提取器

从HTML文件中提取纯文本内容
"""

import logging
from typing import Dict, List, Any

from .base import TextExtractor

logger = logging.getLogger(__name__)


class HTMLExtractor(TextExtractor):
    """HTML文本提取器
    
    支持从HTML文件中提取纯文本内容，保留结构信息
    """
    
    SUPPORTED_TYPES = {
        'text/html',
        'application/xhtml+xml',
        'text/xml',
        'application/xml'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化HTML提取器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.extract_links = config.get('extract_links', False)
        self.extract_tables = config.get('extract_tables', True)
        self.preserve_structure = config.get('preserve_structure', True)
        self.remove_scripts = config.get('remove_scripts', True)
        self.remove_styles = config.get('remove_styles', True)
    
    def supports_file_type(self, file_type: str) -> bool:
        """检查是否支持指定文件类型"""
        return file_type in self.SUPPORTED_TYPES
    
    async def extract(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """从HTML文件提取文本内容
        
        Args:
            file_path: HTML文件路径
            **kwargs: 其他参数
            
        Returns:
            提取的文本内容列表
        """
        if not self.validate_file(file_path):
            raise ValueError(f"文件验证失败: {file_path}")
        
        try:
            # 读取HTML内容
            html_content = await self._read_html_file(file_path)
            
            # 解析HTML并提取内容
            contents = await self._extract_from_html(html_content)
            
            self.logger.info(f"HTML提取完成: {file_path}, 内容块数: {len(contents)}")
            return contents
            
        except Exception as e:
            self.logger.error(f"HTML提取失败: {file_path}, 错误: {str(e)}")
            raise
    
    async def _read_html_file(self, file_path: str) -> str:
        """读取HTML文件
        
        Args:
            file_path: HTML文件路径
            
        Returns:
            HTML内容
        """
        try:
            # 尝试检测编码
            import chardet
            
            with open(file_path, 'rb') as file:
                raw_data = file.read()
            
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            
            # 解码HTML内容
            html_content = raw_data.decode(encoding, errors='ignore')
            
            return html_content
            
        except ImportError:
            # 如果chardet不可用，使用默认编码
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"读取HTML文件失败: {str(e)}")
            raise
    
    async def _extract_from_html(self, html_content: str) -> List[Dict[str, Any]]:
        """从HTML内容中提取文本
        
        Args:
            html_content: HTML内容
            
        Returns:
            提取的文本内容列表
        """
        contents = []
        
        try:
            from bs4 import BeautifulSoup
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除脚本和样式标签
            if self.remove_scripts:
                for script in soup(["script", "style"]):
                    script.decompose()
            
            # 提取标题
            title = soup.find('title')
            if title and title.string:
                title_content = self.create_text_content(
                    content=title.string.strip(),
                    page_number=0,
                    title="Document Title",
                    element_type='title'
                )
                contents.append(title_content)
            
            # 提取元数据
            meta_contents = self._extract_meta_data(soup)
            contents.extend(meta_contents)
            
            # 提取主要内容
            main_contents = await self._extract_main_content(soup)
            contents.extend(main_contents)
            
            # 提取表格内容
            if self.extract_tables:
                table_contents = self._extract_tables(soup)
                contents.extend(table_contents)
            
            # 提取链接信息
            if self.extract_links:
                link_contents = self._extract_links_info(soup)
                contents.extend(link_contents)
            
            return contents
            
        except ImportError:
            self.logger.error("BeautifulSoup4 未安装")
            raise ImportError("需要安装 beautifulsoup4 库")
        except Exception as e:
            self.logger.error(f"HTML解析失败: {str(e)}")
            raise
    
    def _extract_meta_data(self, soup: 'BeautifulSoup') -> List[Dict[str, Any]]:
        """提取HTML元数据
        
        Args:
            soup: BeautifulSoup对象
            
        Returns:
            元数据内容列表
        """
        contents = []
        
        # 提取重要的meta标签
        important_meta = ['description', 'keywords', 'author']
        
        for meta_name in important_meta:
            meta_tag = soup.find('meta', attrs={'name': meta_name})
            if not meta_tag:
                meta_tag = soup.find('meta', attrs={'property': f'og:{meta_name}'})
            
            if meta_tag and meta_tag.get('content'):
                content_item = self.create_text_content(
                    content=meta_tag['content'].strip(),
                    page_number=0,
                    title=f"Meta {meta_name.title()}",
                    element_type='meta',
                    meta_name=meta_name
                )
                contents.append(content_item)
        
        return contents
    
    async def _extract_main_content(self, soup: 'BeautifulSoup') -> List[Dict[str, Any]]:
        """提取HTML主要内容
        
        Args:
            soup: BeautifulSoup对象
            
        Returns:
            主要内容列表
        """
        contents = []
        content_index = 1
        
        # 定义重要的内容标签
        content_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section']
        
        for tag in soup.find_all(content_tags):
            text = tag.get_text(strip=True)
            
            if text and len(text) > 10:  # 过滤太短的内容
                # 判断元素类型
                element_type = tag.name
                is_heading = tag.name.startswith('h')
                
                # 获取标签属性
                tag_attrs = {}
                if tag.get('class'):
                    tag_attrs['class'] = ' '.join(tag['class'])
                if tag.get('id'):
                    tag_attrs['id'] = tag['id']
                
                content_item = self.create_text_content(
                    content=text,
                    page_number=content_index,
                    title=text[:50] + "..." if len(text) > 50 and is_heading else None,
                    element_type=element_type,
                    is_heading=is_heading,
                    tag_attributes=tag_attrs
                )
                
                contents.append(content_item)
                content_index += 1
        
        return contents
    
    def _extract_tables(self, soup: 'BeautifulSoup') -> List[Dict[str, Any]]:
        """提取HTML表格
        
        Args:
            soup: BeautifulSoup对象
            
        Returns:
            表格内容列表
        """
        contents = []
        
        for table_idx, table in enumerate(soup.find_all('table')):
            try:
                table_data = []
                
                # 提取表格行
                for row in table.find_all('tr'):
                    row_data = []
                    for cell in row.find_all(['td', 'th']):
                        cell_text = cell.get_text(strip=True)
                        row_data.append(cell_text)
                    
                    if row_data:
                        table_data.append(row_data)
                
                # 格式化表格
                if table_data:
                    table_text = self._format_table_as_text(table_data)
                    
                    content_item = self.create_text_content(
                        content=table_text,
                        page_number=table_idx + 1,
                        title=f"Table {table_idx + 1}",
                        element_type='table',
                        table_index=table_idx
                    )
                    
                    contents.append(content_item)
            
            except Exception as e:
                self.logger.warning(f"提取表格 {table_idx} 失败: {str(e)}")
                continue
        
        return contents
    
    def _extract_links_info(self, soup: 'BeautifulSoup') -> List[Dict[str, Any]]:
        """提取链接信息
        
        Args:
            soup: BeautifulSoup对象
            
        Returns:
            链接信息列表
        """
        contents = []
        
        links = soup.find_all('a', href=True)
        
        if links:
            link_texts = []
            for link in links:
                link_text = link.get_text(strip=True)
                href = link['href']
                
                if link_text and href:
                    link_info = f"{link_text} ({href})"
                    link_texts.append(link_info)
            
            if link_texts:
                links_content = "\n".join(link_texts)
                
                content_item = self.create_text_content(
                    content=links_content,
                    page_number=1,
                    title="Document Links",
                    element_type='links',
                    link_count=len(link_texts)
                )
                
                contents.append(content_item)
        
        return contents
    
    def _format_table_as_text(self, table_data: List[List[str]]) -> str:
        """将表格格式化为文本
        
        Args:
            table_data: 表格数据
            
        Returns:
            格式化后的文本
        """
        if not table_data:
            return ""
        
        # 计算每列的最大宽度
        max_widths = []
        for row in table_data:
            for col_idx, cell in enumerate(row):
                if col_idx >= len(max_widths):
                    max_widths.append(0)
                max_widths[col_idx] = max(max_widths[col_idx], len(str(cell)))
        
        # 格式化表格
        formatted_rows = []
        for row_idx, row in enumerate(table_data):
            formatted_cells = []
            for col_idx, cell in enumerate(row):
                width = max_widths[col_idx] if col_idx < len(max_widths) else 10
                formatted_cells.append(str(cell).ljust(width))
            
            formatted_row = " | ".join(formatted_cells)
            formatted_rows.append(formatted_row)
            
            # 在第一行后添加分隔线
            if row_idx == 0 and len(table_data) > 1:
                separator = " | ".join("-" * width for width in max_widths)
                formatted_rows.append(separator)
        
        return "\n".join(formatted_rows)
    
    def _clean_html_text(self, text: str) -> str:
        """清理HTML文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        import re
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除HTML实体
        html_entities = {
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&#39;': "'"
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        return text.strip()