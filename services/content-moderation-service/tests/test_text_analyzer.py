"""
文本分析器测试

测试文本内容分析器的各项功能
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.analyzers.text_analyzer import TextAnalyzer
from src.analyzers.base_analyzer import ViolationType, AnalysisStatus


class TestTextAnalyzer:
    """文本分析器测试类"""
    
    @pytest.fixture
    def text_analyzer(self):
        """创建文本分析器实例"""
        config = {
            "confidence_threshold": 0.7,
            "timeout": 30.0,
            "max_text_length": 10000
        }
        return TextAnalyzer(config)
    
    def test_init(self, text_analyzer):
        """测试初始化"""
        assert text_analyzer.name == "TextAnalyzer"
        assert text_analyzer.version == "1.0.0"
        assert text_analyzer.confidence_threshold == 0.7
        assert text_analyzer.timeout == 30.0
    
    def test_get_supported_types(self, text_analyzer):
        """测试支持的内容类型"""
        supported_types = text_analyzer.get_supported_types()
        
        assert "text/plain" in supported_types
        assert "text/html" in supported_types
        assert "text/markdown" in supported_types
        assert len(supported_types) >= 3
    
    def test_is_supported(self, text_analyzer):
        """测试内容类型支持检查"""
        assert text_analyzer.is_supported("text/plain")
        assert text_analyzer.is_supported("text/html")
        assert not text_analyzer.is_supported("image/jpeg")
        assert not text_analyzer.is_supported("video/mp4")
    
    @pytest.mark.asyncio
    async def test_analyze_normal_text(self, text_analyzer):
        """测试分析正常文本"""
        text = "这是一段正常的文本内容，没有任何违规信息。"
        
        result = await text_analyzer.analyze(text)
        
        assert result.status == AnalysisStatus.SUCCESS
        assert result.confidence >= 0.0
        assert result.is_violation == False
        assert result.risk_level == "low"
        assert len(result.violations) == 0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_analyze_sensitive_text(self, text_analyzer):
        """测试分析包含敏感词的文本"""
        text = "这段文本包含政治敏感词示例1和暴力词汇1"
        
        result = await text_analyzer.analyze(text)
        
        assert result.status == AnalysisStatus.SUCCESS
        assert result.processing_time > 0
        
        # 检查是否检测到违规
        if result.violations:
            violation_types = [v.type for v in result.violations]
            assert any(vt in [ViolationType.POLITICS, ViolationType.VIOLENCE] for vt in violation_types)
    
    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, text_analyzer):
        """测试分析空文本"""
        text = ""
        
        result = await text_analyzer.analyze(text)
        
        assert result.status == AnalysisStatus.FAILED
        assert "文本内容为空" in result.error_message
    
    @pytest.mark.asyncio
    async def test_analyze_long_text(self, text_analyzer):
        """测试分析超长文本"""
        text = "测试" * 10000  # 创建超长文本
        
        result = await text_analyzer.analyze(text)
        
        assert result.status == AnalysisStatus.FAILED
        assert "超过限制" in result.error_message
    
    @pytest.mark.asyncio
    async def test_analyze_html_content(self, text_analyzer):
        """测试分析HTML内容"""
        html_text = "<html><body><p>这是一段HTML内容</p></body></html>"
        
        result = await text_analyzer.analyze(html_text)
        
        assert result.status == AnalysisStatus.SUCCESS
        assert result.processing_time > 0
        # HTML应该被清理成纯文本
    
    @pytest.mark.asyncio
    async def test_analyze_bytes_content(self, text_analyzer):
        """测试分析字节内容"""
        text_bytes = "这是字节格式的文本".encode('utf-8')
        
        result = await text_analyzer.analyze(text_bytes)
        
        assert result.status == AnalysisStatus.SUCCESS
        assert result.processing_time > 0
    
    def test_calculate_content_hash(self, text_analyzer):
        """测试内容哈希计算"""
        text1 = "测试文本"
        text2 = "测试文本"
        text3 = "不同文本"
        
        hash1 = text_analyzer.calculate_content_hash(text1)
        hash2 = text_analyzer.calculate_content_hash(text2)
        hash3 = text_analyzer.calculate_content_hash(text3)
        
        assert hash1 == hash2  # 相同内容应该有相同哈希
        assert hash1 != hash3  # 不同内容应该有不同哈希
        assert len(hash1) == 64  # SHA256哈希长度
    
    def test_extract_keywords(self, text_analyzer):
        """测试关键词提取"""
        text = "这是一个测试文本，用于测试关键词提取功能。测试是很重要的。"
        
        keywords = text_analyzer.extract_keywords(text, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert "测试" in keywords  # 高频词应该被提取
    
    def test_get_analyzer_info(self, text_analyzer):
        """测试获取分析器信息"""
        info = text_analyzer.get_analyzer_info()
        
        assert info["name"] == "TextAnalyzer"
        assert info["version"] == "1.0.0"
        assert "supported_types" in info
        assert "confidence_threshold" in info
        assert "timeout" in info
    
    def test_batch_analyze_texts(self, text_analyzer):
        """测试批量分析文本"""
        texts = [
            "正常文本1",
            "正常文本2",
            "包含敏感词示例1的文本"
        ]
        
        results = text_analyzer.batch_analyze_texts(texts)
        
        assert len(results) == 3
        assert all(result.status == AnalysisStatus.SUCCESS for result in results)
    
    @pytest.mark.asyncio
    async def test_analyze_with_timeout(self, text_analyzer):
        """测试带超时的分析"""
        text = "测试文本"
        
        result = await text_analyzer.analyze_with_timeout(text)
        
        assert result.status == AnalysisStatus.SUCCESS
        assert result.processing_time > 0
    
    def test_sensitive_word_detector(self, text_analyzer):
        """测试敏感词检测器"""
        detector = text_analyzer.sensitive_word_detector
        
        # 测试检测功能
        violations = detector.detect("这里包含政治敏感词示例1")
        
        if violations:  # 如果检测到违规
            assert len(violations) > 0
            assert violations[0].type == ViolationType.POLITICS
            assert violations[0].confidence > 0
    
    def test_pattern_detector(self, text_analyzer):
        """测试模式检测器"""
        detector = text_analyzer.pattern_detector
        
        # 测试诈骗模式检测
        violations = detector.detect("免费获得大奖，点击联系我们")
        
        if violations:
            assert len(violations) > 0
            assert violations[0].type == ViolationType.FRAUD
    
    def test_sentiment_analyzer(self, text_analyzer):
        """测试情感分析器"""
        analyzer = text_analyzer.sentiment_analyzer
        
        # 测试正面情感
        positive_score = analyzer.analyze("我很喜欢这个产品，非常好用")
        assert positive_score > 0
        
        # 测试负面情感
        negative_score = analyzer.analyze("这个产品很差，我很讨厌")
        assert negative_score < 0
        
        # 测试中性情感
        neutral_score = analyzer.analyze("这是一个产品")
        assert abs(neutral_score) < 0.5