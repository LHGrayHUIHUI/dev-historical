"""
文本内容模型定义

存储从文件中提取的文本内容
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import BaseModel


class TextContent(BaseModel):
    """文本内容模型
    
    存储从数据集中提取的文本内容，支持全文搜索
    """
    
    __tablename__ = "text_contents"
    
    # 关联的数据集ID
    dataset_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="关联的数据集ID"
    )
    
    # 文本标题
    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        doc="文本标题"
    )
    
    # 文本内容
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="文本内容"
    )
    
    # 语言代码
    language: Mapped[Optional[str]] = mapped_column(
        String(10),
        index=True,
        doc="文本语言代码"
    )
    
    # 文本编码
    encoding: Mapped[Optional[str]] = mapped_column(
        String(20),
        doc="文本编码格式"
    )
    
    # 页码/段落号
    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        doc="页码或段落号"
    )
    
    # 词数统计
    word_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        doc="词数统计"
    )
    
    # 字符数统计
    char_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        doc="字符数统计"
    )
    
    # 提取时间
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="文本提取时间"
    )
    
    # 全文搜索向量(自动生成)
    content_vector: Mapped[Optional[str]] = mapped_column(
        TSVECTOR,
        doc="全文搜索向量"
    )
    
    # 置信度分数(OCR等)
    confidence_score: Mapped[Optional[float]] = mapped_column(
        doc="文本提取置信度分数"
    )
    
    # 质量评分
    quality_score: Mapped[Optional[float]] = mapped_column(
        doc="文本质量评分"
    )
    
    # 关联关系
    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        back_populates="text_contents",
        doc="关联的数据集"
    )
    
    def __repr__(self) -> str:
        """字符串表示"""
        title_preview = self.title[:30] + "..." if self.title and len(self.title) > 30 else self.title
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<TextContent(id={self.id}, title='{title_preview}', content='{content_preview}')>"
    
    @property
    def content_preview(self) -> str:
        """内容预览(前100个字符)"""
        return self.content[:100] + "..." if len(self.content) > 100 else self.content
    
    @property
    def has_high_confidence(self) -> bool:
        """检查是否有高置信度"""
        return self.confidence_score and self.confidence_score >= 0.8
    
    @property
    def has_good_quality(self) -> bool:
        """检查是否有良好质量"""
        return self.quality_score and self.quality_score >= 0.7
    
    def calculate_basic_stats(self) -> None:
        """计算基本统计信息"""
        if self.content:
            self.char_count = len(self.content)
            # 简单的中英文词数统计
            import re
            # 统计中文字符
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', self.content))
            # 统计英文单词
            english_words = len(re.findall(r'\b[a-zA-Z]+\b', self.content))
            # 中文按字符计算，英文按单词计算
            self.word_count = chinese_chars + english_words
    
    def detect_language(self) -> Optional[str]:
        """检测文本语言
        
        Returns:
            语言代码，如 'zh-cn', 'en'
        """
        if not self.content:
            return None
            
        try:
            from langdetect import detect
            detected_lang = detect(self.content)
            self.language = detected_lang
            return detected_lang
        except Exception:
            # 简单的语言检测
            import re
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', self.content))
            total_chars = len(self.content.strip())
            
            if total_chars == 0:
                return None
                
            chinese_ratio = chinese_chars / total_chars
            if chinese_ratio > 0.3:
                self.language = 'zh-cn'
                return 'zh-cn'
            else:
                self.language = 'en'
                return 'en'
    
    def calculate_quality_score(self) -> float:
        """计算文本质量评分
        
        基于以下因素:
        - 文本长度
        - 特殊字符比例
        - 重复内容比例
        - 语言一致性
        
        Returns:
            质量评分 (0-1)
        """
        if not self.content:
            return 0.0
        
        score = 1.0
        content = self.content.strip()
        
        # 长度评分
        if len(content) < 10:
            score *= 0.3
        elif len(content) < 50:
            score *= 0.7
        
        # 特殊字符比例评分
        import re
        special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', content))
        special_ratio = special_chars / len(content) if content else 0
        if special_ratio > 0.5:
            score *= 0.5
        
        # 重复字符评分
        unique_chars = len(set(content))
        repetition_score = unique_chars / len(content) if content else 0
        if repetition_score < 0.3:
            score *= 0.6
        
        # 置信度影响
        if self.confidence_score:
            score *= self.confidence_score
        
        self.quality_score = round(score, 3)
        return self.quality_score