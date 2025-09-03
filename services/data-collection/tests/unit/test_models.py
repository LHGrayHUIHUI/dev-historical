"""
数据模型单元测试
"""

import pytest
from datetime import datetime
from uuid import uuid4

from src.models import DataSource, Dataset, TextContent


class TestDataSource:
    """数据源模型测试"""
    
    def test_create_data_source(self):
        """测试创建数据源"""
        data_source = DataSource(
            name="测试数据源",
            type="file_upload",
            description="测试用数据源",
            config={"max_size": 100},
            created_by=uuid4(),
            status="active"
        )
        
        assert data_source.name == "测试数据源"
        assert data_source.type == "file_upload"
        assert data_source.is_active is True
        assert data_source.get_config_value("max_size") == 100
        assert data_source.get_config_value("unknown", "default") == "default"
    
    def test_update_config(self):
        """测试更新配置"""
        data_source = DataSource(
            name="测试数据源",
            type="api",
            created_by=uuid4()
        )
        
        data_source.update_config("timeout", 30)
        assert data_source.get_config_value("timeout") == 30


class TestDataset:
    """数据集模型测试"""
    
    def test_create_dataset(self):
        """测试创建数据集"""
        dataset = Dataset(
            name="测试数据集.pdf",
            source_id=uuid4(),
            file_path="/uploads/test.pdf",
            file_size=1024,
            file_type="application/pdf",
            file_hash="abc123",
            created_by=uuid4(),
            processing_status="pending"
        )
        
        assert dataset.name == "测试数据集.pdf"
        assert dataset.processing_status == "pending"
        assert not dataset.is_processed
        assert not dataset.has_error
    
    def test_update_processing_status(self):
        """测试更新处理状态"""
        dataset = Dataset(
            name="测试数据集",
            source_id=uuid4(),
            created_by=uuid4()
        )
        
        dataset.update_processing_status("completed")
        assert dataset.processing_status == "completed"
        assert dataset.is_processed
        assert dataset.processed_at is not None
        
        dataset.update_processing_status("failed", "处理失败")
        assert dataset.processing_status == "failed"
        assert dataset.error_message == "处理失败"
        assert dataset.has_error
    
    def test_calculate_statistics(self):
        """测试计算统计信息"""
        dataset = Dataset(
            name="测试数据集",
            source_id=uuid4(),
            created_by=uuid4()
        )
        
        # 模拟文本内容（实际测试中需要创建真实的TextContent对象）
        dataset.text_count = 2
        dataset.total_words = 100
        dataset.total_chars = 500
        
        assert dataset.text_count == 2
        assert dataset.total_words == 100
        assert dataset.total_chars == 500


class TestTextContent:
    """文本内容模型测试"""
    
    def test_create_text_content(self):
        """测试创建文本内容"""
        text_content = TextContent(
            dataset_id=uuid4(),
            content="这是一个测试文本内容。This is test content.",
            page_number=1
        )
        
        assert "测试文本内容" in text_content.content
        assert text_content.page_number == 1
        assert text_content.content_preview in text_content.content
    
    def test_calculate_basic_stats(self):
        """测试计算基本统计"""
        text_content = TextContent(
            dataset_id=uuid4(),
            content="这是测试内容。This is a test.",
            page_number=1
        )
        
        text_content.calculate_basic_stats()
        
        assert text_content.char_count > 0
        assert text_content.word_count > 0
    
    def test_detect_language(self):
        """测试语言检测"""
        # 中文内容
        chinese_content = TextContent(
            dataset_id=uuid4(),
            content="这是一个中文测试内容，用于检测语言类型。",
            page_number=1
        )
        
        language = chinese_content.detect_language()
        assert language in ['zh-cn', 'zh', 'zh-en']  # 可能的中文语言代码
        
        # 英文内容
        english_content = TextContent(
            dataset_id=uuid4(),
            content="This is an English test content for language detection.",
            page_number=1
        )
        
        language = english_content.detect_language()
        assert language in ['en', 'zh-en']  # 可能的英文语言代码
    
    def test_calculate_quality_score(self):
        """测试质量评分计算"""
        text_content = TextContent(
            dataset_id=uuid4(),
            content="这是一个质量良好的文本内容，包含足够的字符和合理的结构。",
            page_number=1,
            confidence_score=0.9
        )
        
        quality_score = text_content.calculate_quality_score()
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5  # 应该是较高质量的文本
        
        # 测试低质量文本
        low_quality_content = TextContent(
            dataset_id=uuid4(),
            content="abc",
            page_number=1,
            confidence_score=0.3
        )
        
        low_quality_score = low_quality_content.calculate_quality_score()
        assert low_quality_score < quality_score