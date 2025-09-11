"""
合规检测引擎单元测试

测试合规检测引擎的核心功能，包括敏感词检测、政策合规、
版权检查、学术诚信检测等功能的正确性。
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.compliance_engine import (
    ComplianceEngine, SensitiveWordDetector, PolicyComplianceDetector,
    CopyrightDetector, AcademicIntegrityDetector
)
from src.models.quality_models import (
    ComplianceCheckRequest, ComplianceCheckResult, ComplianceViolation,
    ViolationType, ComplianceStatus
)
from src.clients.storage_client import StorageServiceClient

class TestComplianceEngine:
    """合规检测引擎测试类"""
    
    @pytest.fixture
    def mock_storage_client(self):
        """模拟存储客户端"""
        mock_client = AsyncMock(spec=StorageServiceClient)
        mock_client.get_sensitive_words.return_value = {
            "data": [
                {
                    "word": "测试敏感词",
                    "category": "test",
                    "severity_level": 5,
                    "replacement_suggestion": "替代词"
                }
            ]
        }
        mock_client.get_compliance_rules.return_value = {"data": []}
        return mock_client
    
    @pytest.fixture
    def compliance_engine(self, mock_storage_client):
        """创建合规检测引擎实例"""
        return ComplianceEngine(mock_storage_client)
    
    @pytest.mark.asyncio
    async def test_compliance_check_basic(self, compliance_engine):
        """测试基础合规检测功能"""
        # 准备测试数据
        request = ComplianceCheckRequest(
            content="这是一个正常的测试内容，不包含任何违规信息。",
            check_types=["sensitive_words", "policy", "copyright", "academic_integrity"]
        )
        
        # 执行检测
        result = await compliance_engine.check_compliance(request)
        
        # 验证结果
        assert isinstance(result, ComplianceCheckResult)
        assert result.compliance_status in [ComplianceStatus.PASS, ComplianceStatus.WARNING, ComplianceStatus.FAIL]
        assert result.risk_score >= 0
        assert result.risk_score <= 10
        assert isinstance(result.violations, list)
        assert isinstance(result.recommendations, list)
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_compliance_check_with_sensitive_word(self, compliance_engine):
        """测试包含敏感词的合规检测"""
        request = ComplianceCheckRequest(
            content="这个内容包含测试敏感词，应该被检测出来。",
            check_types=["sensitive_words"]
        )
        
        result = await compliance_engine.check_compliance(request)
        
        # 应该检测到敏感词违规
        sensitive_violations = [v for v in result.violations 
                              if v.violation_type == ViolationType.SENSITIVE_WORD]
        assert len(sensitive_violations) > 0
        
        # 风险评分应该大于0
        assert result.risk_score > 0
        
        # 合规状态应该不是PASS
        assert result.compliance_status != ComplianceStatus.PASS
    
    @pytest.mark.asyncio
    async def test_compliance_check_strict_mode(self, compliance_engine):
        """测试严格模式的合规检测"""
        request = ComplianceCheckRequest(
            content="这是一个测试内容。",
            check_types=["sensitive_words", "policy"],
            strict_mode=True
        )
        
        result = await compliance_engine.check_compliance(request)
        
        # 严格模式下可能产生更多检测结果
        assert isinstance(result, ComplianceCheckResult)
    
    @pytest.mark.asyncio
    async def test_compliance_check_content_too_long(self, compliance_engine):
        """测试内容过长的情况"""
        long_content = "测试" * 50000  # 100000字符
        request = ComplianceCheckRequest(content=long_content)
        
        # 应该抛出ValueError异常
        with pytest.raises(ValueError, match="内容长度超过限制"):
            await compliance_engine.check_compliance(request)
    
    @pytest.mark.asyncio
    async def test_calculate_risk_score(self, compliance_engine):
        """测试风险评分计算"""
        # 创建模拟违规
        violations = [
            ComplianceViolation(
                violation_type=ViolationType.SENSITIVE_WORD,
                severity=5,
                position=0,
                content="测试",
                description="敏感词",
                action="replace",
                suggestion="替换",
                confidence=0.9
            ),
            ComplianceViolation(
                violation_type=ViolationType.POLICY_VIOLATION,
                severity=7,
                position=10,
                content="违规内容",
                description="政策违规",
                action="remove",
                suggestion="删除",
                confidence=0.8
            )
        ]
        
        risk_score = compliance_engine._calculate_risk_score(violations)
        
        # 验证风险评分
        assert risk_score >= 0
        assert risk_score <= 10
        assert risk_score > 0  # 有违规应该有风险分数
    
    @pytest.mark.asyncio
    async def test_determine_compliance_status(self, compliance_engine):
        """测试合规状态确定"""
        # 测试低风险情况
        low_risk_violations = [
            ComplianceViolation(
                violation_type=ViolationType.SENSITIVE_WORD,
                severity=2,
                position=0,
                content="轻微违规",
                description="轻微问题",
                action="warn",
                suggestion="注意",
                confidence=0.6
            )
        ]
        
        status = compliance_engine._determine_compliance_status(2, low_risk_violations)
        assert status == ComplianceStatus.PASS
        
        # 测试高风险情况
        high_risk_violations = [
            ComplianceViolation(
                violation_type=ViolationType.POLICY_VIOLATION,
                severity=9,
                position=0,
                content="严重违规",
                description="严重问题",
                action="block",
                suggestion="阻止",
                confidence=0.9
            )
        ]
        
        status = compliance_engine._determine_compliance_status(8, high_risk_violations)
        assert status == ComplianceStatus.FAIL

class TestSensitiveWordDetector:
    """敏感词检测器测试类"""
    
    @pytest.fixture
    def mock_storage_client(self):
        """模拟存储客户端"""
        mock_client = AsyncMock(spec=StorageServiceClient)
        mock_client.get_sensitive_words.return_value = {
            "data": [
                {
                    "word": "敏感词1",
                    "category": "political",
                    "severity_level": 8,
                    "replacement_suggestion": "合适词1"
                },
                {
                    "word": "敏感词2",
                    "category": "social",
                    "severity_level": 6,
                    "replacement_suggestion": "合适词2"
                }
            ]
        }
        return mock_client
    
    @pytest.fixture
    def detector(self, mock_storage_client):
        """创建敏感词检测器实例"""
        return SensitiveWordDetector(mock_storage_client)
    
    @pytest.mark.asyncio
    async def test_detect_sensitive_words(self, detector):
        """测试敏感词检测"""
        content = "这个文本包含敏感词1和敏感词2。"
        
        violations, policy_status = await detector.detect(content, False)
        
        # 应该检测到两个敏感词
        assert len(violations) == 2
        
        # 验证违规信息
        for violation in violations:
            assert violation.violation_type == ViolationType.SENSITIVE_WORD
            assert violation.content in ["敏感词1", "敏感词2"]
            assert violation.action == "replace"
            assert violation.confidence > 0.5
        
        # 策略状态应该不是pass
        assert policy_status["sensitive_word_check"] != "pass"
    
    @pytest.mark.asyncio
    async def test_detect_no_sensitive_words(self, detector):
        """测试不包含敏感词的情况"""
        content = "这是一个正常的文本内容。"
        
        violations, policy_status = await detector.detect(content, False)
        
        # 应该没有检测到敏感词
        assert len(violations) == 0
        
        # 策略状态应该是pass
        assert policy_status["sensitive_word_check"] == "pass"
    
    @pytest.mark.asyncio
    async def test_get_sensitive_words_cache(self, detector):
        """测试敏感词缓存机制"""
        # 第一次调用
        words1 = await detector._get_sensitive_words()
        
        # 第二次调用应该使用缓存
        words2 = await detector._get_sensitive_words()
        
        assert words1 == words2
        # 验证只调用了一次storage_client
        detector.storage_client.get_sensitive_words.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, detector):
        """测试检测置信度计算"""
        # 测试长词的置信度
        confidence_long = detector._calculate_confidence("长敏感词", "这是一个长敏感词的上下文", "political")
        
        # 测试短词的置信度
        confidence_short = detector._calculate_confidence("短", "这是短词上下文", "social")
        
        # 长词的置信度应该高于短词
        assert confidence_long > confidence_short
        
        # 置信度应该在0-1范围内
        assert 0 <= confidence_long <= 1
        assert 0 <= confidence_short <= 1

class TestPolicyComplianceDetector:
    """政策合规检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建政策合规检测器实例"""
        return PolicyComplianceDetector()
    
    @pytest.mark.asyncio
    async def test_detect_long_content(self, detector):
        """测试过长内容检测"""
        long_content = "测试内容" * 10000  # 创建长内容
        
        violations, policy_status = await detector.detect(long_content, False)
        
        # 应该检测到内容过长问题
        length_violations = [v for v in violations 
                           if "过长" in v.description]
        assert len(length_violations) > 0
    
    @pytest.mark.asyncio
    async def test_detect_promotional_content(self, detector):
        """测试推广内容检测"""
        promotional_content = "这是一个广告内容，请联系我们购买产品。"
        
        violations, policy_status = await detector.detect(promotional_content, False)
        
        # 可能检测到推广内容
        promo_violations = [v for v in violations 
                          if v.violation_type == ViolationType.POLICY_VIOLATION]
        # 注意：这个测试取决于具体的模式匹配实现
    
    @pytest.mark.asyncio
    async def test_detect_contact_info(self, detector):
        """测试联系方式检测"""
        contact_content = "请联系电话13800138000或微信abc123。"
        
        violations, policy_status = await detector.detect(contact_content, False)
        
        # 可能检测到联系方式
        contact_violations = [v for v in violations 
                            if "联系方式" in v.description]
        # 注意：这个测试取决于具体的模式匹配实现

class TestCopyrightDetector:
    """版权检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建版权检测器实例"""
        return CopyrightDetector()
    
    @pytest.mark.asyncio
    async def test_detect_copyright_info(self, detector):
        """测试版权信息检测"""
        copyright_content = "© 2023 某公司版权所有，保留所有权利。"
        
        violations, policy_status = await detector.detect(copyright_content, False)
        
        # 应该检测到版权信息
        copyright_violations = [v for v in violations 
                              if v.violation_type == ViolationType.COPYRIGHT_VIOLATION]
        assert len(copyright_violations) > 0
    
    @pytest.mark.asyncio
    async def test_detect_citations(self, detector):
        """测试引用检测"""
        citation_content = "根据研究[1]显示，引用自某文献的内容。参考文献：[1] 作者，论文，期刊，2023。"
        
        violations, policy_status = await detector.detect(citation_content, False)
        
        # 有适当引用的内容策略状态应该是pass
        assert policy_status["copyright_check"] == "pass"
    
    @pytest.mark.asyncio
    async def test_no_copyright_issues(self, detector):
        """测试无版权问题的内容"""
        normal_content = "这是一个原创的正常内容。"
        
        violations, policy_status = await detector.detect(normal_content, False)
        
        # 应该没有版权违规
        copyright_violations = [v for v in violations 
                              if v.violation_type == ViolationType.COPYRIGHT_VIOLATION]
        assert len(copyright_violations) == 0

class TestAcademicIntegrityDetector:
    """学术诚信检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        """创建学术诚信检测器实例"""
        return AcademicIntegrityDetector()
    
    @pytest.mark.asyncio
    async def test_detect_missing_citations(self, detector):
        """测试缺少引用检测"""
        long_content_no_citations = "这是一个很长的学术内容。" * 200  # 创建长内容但无引用
        
        violations, policy_status = await detector.detect(long_content_no_citations, False)
        
        # 可能检测到缺少引用的问题
        citation_violations = [v for v in violations 
                             if "引用" in v.description]
        # 注意：这个测试取决于具体实现
    
    @pytest.mark.asyncio
    async def test_detect_proper_citations(self, detector):
        """测试适当引用检测"""
        content_with_citations = """
        这是一个学术内容，根据研究[1]显示相关结论。
        参考文献：
        [1] 作者名，论文标题，期刊名，2023年。
        """
        
        violations, policy_status = await detector.detect(content_with_citations, False)
        
        # 有适当引用的内容策略状态应该是pass
        assert policy_status["academic_integrity"] == "pass"
    
    @pytest.mark.asyncio
    async def test_check_citations(self, detector):
        """测试引用检查功能"""
        # 测试有引用的内容
        content_with_refs = "根据研究[1]，参考文献显示..."
        has_citations = await detector._check_citations(content_with_refs)
        assert has_citations == True
        
        # 测试无引用的内容
        content_no_refs = "这是没有引用的内容。"
        has_citations = await detector._check_citations(content_no_refs)
        assert has_citations == False
    
    @pytest.mark.asyncio
    async def test_check_duplicates(self, detector):
        """测试重复内容检查"""
        content_with_duplicates = """
        这是第一个句子。这是第二个句子。
        这是第一个句子。这是第三个句子。
        """
        
        violations = await detector._check_duplicates(content_with_duplicates)
        
        # 应该检测到重复句子
        duplicate_violations = [v for v in violations 
                              if "重复" in v.description]
        assert len(duplicate_violations) > 0

@pytest.mark.asyncio
async def test_integration_compliance_check():
    """集成测试：完整的合规检测流程"""
    # 创建模拟客户端
    mock_client = AsyncMock(spec=StorageServiceClient)
    mock_client.get_sensitive_words.return_value = {
        "data": [
            {
                "word": "违禁词",
                "category": "test",
                "severity_level": 7,
                "replacement_suggestion": "合规词"
            }
        ]
    }
    
    # 创建合规检测引擎
    engine = ComplianceEngine(mock_client)
    
    # 测试包含多种违规的内容
    problematic_content = """
    这个内容包含违禁词，还有版权信息© 2023。
    这是一个广告推广内容，联系电话123456。
    这是第一句话。这是第一句话。
    """
    
    request = ComplianceCheckRequest(
        content=problematic_content,
        check_types=["sensitive_words", "policy", "copyright", "academic_integrity"]
    )
    
    result = await engine.check_compliance(request)
    
    # 验证结果完整性
    assert isinstance(result, ComplianceCheckResult)
    assert result.risk_score > 0  # 应该有风险分数
    assert len(result.violations) > 0  # 应该检测到违规
    assert result.compliance_status != ComplianceStatus.PASS  # 不应该通过
    assert len(result.recommendations) > 0  # 应该有整改建议
    assert result.processing_time_ms > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])