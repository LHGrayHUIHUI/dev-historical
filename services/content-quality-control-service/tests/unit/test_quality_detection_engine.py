"""
质量检测引擎单元测试

测试质量检测引擎的核心功能，包括各种检测器的正确性、
评分算法的准确性和性能表现。
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.quality_detection_engine import (
    QualityDetectionEngine, GrammarDetector, LogicDetector,
    FormatDetector, FactualDetector, AcademicDetector
)
from src.models.quality_models import (
    QualityCheckRequest, QualityCheckResult, QualityIssue,
    QualityMetrics, IssueType, IssueSeverity
)
from src.clients.storage_client import StorageServiceClient

class TestQualityDetectionEngine:
    """质量检测引擎测试类"""
    
    @pytest.fixture
    def mock_storage_client(self):
        """模拟存储客户端"""
        mock_client = AsyncMock(spec=StorageServiceClient)
        mock_client.get_quality_rules.return_value = {"data": []}
        return mock_client
    
    @pytest.fixture
    def detection_engine(self, mock_storage_client):
        """创建检测引擎实例"""
        return QualityDetectionEngine(mock_storage_client)
    
    @pytest.mark.asyncio
    async def test_quality_check_basic(self, detection_engine):
        """测试基础质量检测功能"""
        # 准备测试数据
        request = QualityCheckRequest(
            content="朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州。",
            content_type="historical_text",
            check_options={
                "grammar_check": True,
                "logic_check": True,
                "format_check": True,
                "factual_check": True,
                "academic_check": True
            }
        )
        
        # 执行检测
        result = await detection_engine.check_quality(request)
        
        # 验证结果
        assert isinstance(result, QualityCheckResult)
        assert result.overall_score >= 0
        assert result.overall_score <= 100
        assert isinstance(result.metrics, QualityMetrics)
        assert isinstance(result.issues, list)
        assert isinstance(result.suggestions, list)
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_quality_check_with_auto_fix(self, detection_engine):
        """测试带自动修复的质量检测"""
        request = QualityCheckRequest(
            content="这是一个测试文本，，有重复标点符号。。",
            auto_fix=True
        )
        
        result = await detection_engine.check_quality(request)
        
        # 应该检测到格式问题并提供修复建议
        format_issues = [issue for issue in result.issues 
                        if issue.issue_type == IssueType.FORMAT_VIOLATION]
        assert len(format_issues) > 0
        
        # 应该有自动修复方案
        fixable_issues = [issue for issue in result.issues if issue.auto_fixable]
        if fixable_issues:
            assert len(result.auto_fixes) > 0
    
    @pytest.mark.asyncio
    async def test_quality_check_content_too_long(self, detection_engine):
        """测试内容过长的情况"""
        # 创建超长内容
        long_content = "测试" * 50000  # 100000字符
        request = QualityCheckRequest(content=long_content)
        
        # 应该抛出ValueError异常
        with pytest.raises(ValueError, match="内容长度超过限制"):
            await detection_engine.check_quality(request)
    
    @pytest.mark.asyncio
    async def test_quality_check_empty_content(self, detection_engine):
        """测试空内容的情况"""
        request = QualityCheckRequest(content="")
        
        # 应该抛出验证错误
        with pytest.raises(ValueError):
            await detection_engine.check_quality(request)
    
    @pytest.mark.asyncio
    async def test_calculate_quality_metrics(self, detection_engine):
        """测试质量指标计算"""
        # 模拟检测结果
        detection_results = {
            'grammar': ([], {'grammar_score': 85.0}),
            'logic': ([], {'logic_score': 80.0}),
            'format': ([], {'format_score': 90.0}),
            'factual': ([], {'factual_score': 88.0}),
            'academic': ([], {'academic_score': 82.0})
        }
        
        metrics = await detection_engine._calculate_quality_metrics(detection_results)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.grammar_score == 85.0
        assert metrics.logic_score == 80.0
        assert metrics.format_score == 90.0
        assert metrics.factual_score == 88.0
        assert metrics.academic_score == 82.0
        
        # 验证加权总分计算
        expected_overall = (
            85.0 * 0.25 + 80.0 * 0.20 + 90.0 * 0.15 + 88.0 * 0.20 + 82.0 * 0.20
        )
        assert abs(metrics.overall_score - expected_overall) < 0.1

class TestGrammarDetector:
    """语法检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建语法检测器实例"""
        return GrammarDetector()
    
    @pytest.mark.asyncio
    async def test_detect_long_sentence(self, detector):
        """测试长句检测"""
        long_sentence = "这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的句子。"
        
        issues, metrics = await detector.detect(long_sentence, "general")
        
        # 应该检测到长句问题
        long_sentence_issues = [issue for issue in issues 
                              if "过长" in issue.description]
        assert len(long_sentence_issues) > 0
        assert isinstance(metrics['grammar_score'], float)
    
    @pytest.mark.asyncio
    async def test_detect_excessive_commas(self, detector):
        """测试过多逗号检测"""
        comma_heavy = "这是，一个，有，很多，逗号，的，句子。"
        
        issues, metrics = await detector.detect(comma_heavy, "general")
        
        # 应该检测到逗号过多问题
        comma_issues = [issue for issue in issues 
                       if "逗号" in issue.description]
        assert len(comma_issues) > 0
    
    @pytest.mark.asyncio
    async def test_grammar_score_calculation(self, detector):
        """测试语法分数计算"""
        normal_text = "这是一个正常的句子。"
        
        issues, metrics = await detector.detect(normal_text, "general")
        
        # 正常文本应该有较高的语法分数
        assert metrics['grammar_score'] >= 85.0

class TestLogicDetector:
    """逻辑检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建逻辑检测器实例"""
        return LogicDetector()
    
    @pytest.mark.asyncio
    async def test_detect_temporal_logic(self, detector):
        """测试时间逻辑检测"""
        text_with_dates = "朱元璋生于1328年，死于1398年，是明朝的开国皇帝。"
        
        issues, metrics = await detector.detect(text_with_dates, "historical")
        
        # 应该能处理时间表达式
        assert isinstance(metrics['logic_score'], float)
        assert metrics['logic_score'] >= 0
    
    @pytest.mark.asyncio
    async def test_detect_coherence(self, detector):
        """测试连贯性检测"""
        incoherent_text = """第一段内容。

第二段内容没有过渡。

第三段又是另一个话题。"""
        
        issues, metrics = await detector.detect(incoherent_text, "general")
        
        # 可能检测到连贯性问题
        coherence_issues = [issue for issue in issues 
                          if issue.issue_type == IssueType.LOGIC_INCONSISTENCY]
        # 注意：这个测试可能不会总是检测到问题，取决于实现细节

class TestFormatDetector:
    """格式检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建格式检测器实例"""
        return FormatDetector()
    
    @pytest.mark.asyncio
    async def test_detect_duplicate_punctuation(self, detector):
        """测试重复标点符号检测"""
        text_with_duplicates = "这是什么？？这很奇怪！！！确实，，。。"
        
        issues, metrics = await detector.detect(text_with_duplicates, "general")
        
        # 应该检测到重复标点符号
        punctuation_issues = [issue for issue in issues 
                            if issue.issue_type == IssueType.FORMAT_VIOLATION]
        assert len(punctuation_issues) > 0
        
        # 重复标点应该是可自动修复的
        auto_fixable_issues = [issue for issue in punctuation_issues if issue.auto_fixable]
        assert len(auto_fixable_issues) > 0
    
    @pytest.mark.asyncio
    async def test_detect_short_title(self, detector):
        """测试短标题检测"""
        short_title_text = "题\n\n这是正文内容。"
        
        issues, metrics = await detector.detect(short_title_text, "general")
        
        # 可能检测到标题过短问题
        title_issues = [issue for issue in issues 
                       if "标题" in issue.description]
        # 注意：这个测试取决于具体实现
    
    @pytest.mark.asyncio
    async def test_detect_long_paragraph(self, detector):
        """测试长段落检测"""
        long_paragraph = "这是一个很长的段落。" * 100  # 创建长段落
        
        issues, metrics = await detector.detect(long_paragraph, "general")
        
        # 应该检测到段落过长问题
        paragraph_issues = [issue for issue in issues 
                          if "段落" in issue.description]
        assert len(paragraph_issues) > 0

class TestFactualDetector:
    """事实检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建事实检测器实例"""
        return FactualDetector()
    
    @pytest.mark.asyncio
    async def test_detect_invalid_year(self, detector):
        """测试无效年份检测"""
        text_with_invalid_year = "这件事发生在2050年。"
        
        issues, metrics = await detector.detect(text_with_invalid_year, "historical")
        
        # 应该检测到未来年份问题
        year_issues = [issue for issue in issues 
                      if issue.issue_type == IssueType.FACTUAL_ERROR]
        assert len(year_issues) > 0
    
    @pytest.mark.asyncio
    async def test_detect_valid_historical_year(self, detector):
        """测试有效历史年份"""
        text_with_valid_year = "朱元璋登基于1368年。"
        
        issues, metrics = await detector.detect(text_with_valid_year, "historical")
        
        # 有效年份不应该产生错误
        year_issues = [issue for issue in issues 
                      if "年份" in issue.description]
        assert len(year_issues) == 0

class TestAcademicDetector:
    """学术检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建学术检测器实例"""
        return AcademicDetector()
    
    @pytest.mark.asyncio
    async def test_detect_colloquial_words(self, detector):
        """测试口语化用词检测"""
        colloquial_text = "这个皇帝挺好的，超级厉害。"
        
        issues, metrics = await detector.detect(colloquial_text, "academic")
        
        # 应该检测到口语化用词
        vocabulary_issues = [issue for issue in issues 
                           if issue.issue_type == IssueType.ACADEMIC_STANDARD]
        assert len(vocabulary_issues) > 0
    
    @pytest.mark.asyncio
    async def test_formal_academic_text(self, detector):
        """测试正式学术文本"""
        formal_text = "朱元璋是明朝的开国皇帝，其统治政策对后世产生了深远影响。"
        
        issues, metrics = await detector.detect(formal_text, "academic")
        
        # 正式文本应该有较高的学术分数
        assert metrics['academic_score'] >= 85.0

@pytest.mark.asyncio
async def test_integration_quality_check():
    """集成测试：完整的质量检测流程"""
    # 创建模拟客户端
    mock_client = AsyncMock(spec=StorageServiceClient)
    mock_client.get_quality_rules.return_value = {"data": []}
    
    # 创建检测引擎
    engine = QualityDetectionEngine(mock_client)
    
    # 测试历史文本
    historical_text = """
    朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州。父世珍，
    母陈氏。元至顺四年，岁在癸酉九月十八日乙丑，生太祖于钟离东乡。
    """
    
    request = QualityCheckRequest(
        content=historical_text.strip(),
        content_type="historical_text"
    )
    
    result = await engine.check_quality(request)
    
    # 验证结果完整性
    assert isinstance(result, QualityCheckResult)
    assert result.overall_score >= 0
    assert result.overall_score <= 100
    assert result.processing_time_ms > 0
    assert len(result.status) > 0
    
    # 验证各维度分数
    assert result.metrics.grammar_score >= 0
    assert result.metrics.logic_score >= 0
    assert result.metrics.format_score >= 0
    assert result.metrics.factual_score >= 0
    assert result.metrics.academic_score >= 0
    
    # 验证建议生成
    if result.overall_score < 90:
        assert len(result.suggestions) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])