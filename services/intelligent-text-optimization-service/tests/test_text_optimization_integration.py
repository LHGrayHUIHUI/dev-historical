"""
智能文本优化服务集成测试 - Integration Tests

测试文本优化服务的核心功能集成，包括文本分析、优化引擎、
质量评估、策略管理等模块的协同工作

测试用例:
1. 文本分析功能测试
2. 优化引擎集成测试  
3. 质量评估系统测试
4. 策略管理器测试
5. 完整优化流程测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# 导入被测试的模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.optimization_models import (
    OptimizationType, OptimizationMode, OptimizationRequest, 
    OptimizationParameters, QualityMetrics
)
from services.text_optimization_engine import TextAnalyzer, TextOptimizationEngine
from services.quality_assessor import QualityAssessor
from services.optimization_strategy_manager import OptimizationStrategyManager
from clients.ai_model_service_client import AIModelServiceClient
from clients.storage_service_client import StorageServiceClient


class TestTextAnalyzer:
    """文本分析器测试"""
    
    @pytest.fixture
    def text_analyzer(self):
        """创建文本分析器实例"""
        return TextAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_basic_text(self, text_analyzer):
        """测试基础文本分析"""
        content = "朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州。"
        
        analysis = await text_analyzer.analyze_text(content)
        
        # 验证分析结果结构
        assert 'basic_stats' in analysis
        assert 'complexity' in analysis
        assert 'style' in analysis
        assert 'readability' in analysis
        
        # 验证基础统计
        basic_stats = analysis['basic_stats']
        assert basic_stats['length'] == len(content)
        assert basic_stats['word_count'] > 0
        assert basic_stats['sentence_count'] > 0
        
        # 验证复杂度评分
        assert 0 <= analysis['complexity'] <= 1
        
        # 验证可读性评分
        assert 0 <= analysis['readability'] <= 100
    
    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, text_analyzer):
        """测试空文本分析"""
        content = ""
        
        # 应该返回默认分析结果
        analysis = await text_analyzer._get_basic_analysis(content)
        
        assert analysis['basic_stats']['length'] == 0
        assert analysis['complexity'] == 0.5
        assert analysis['readability'] == 50.0
    
    @pytest.mark.asyncio
    async def test_analyze_classical_chinese(self, text_analyzer):
        """测试文言文分析"""
        content = "昔者仲尼与于蜡宾，事毕，出游于观，之上，喟然而叹。"
        
        analysis = await text_analyzer.analyze_text(content)
        
        # 文言文应有较高复杂度
        assert analysis['complexity'] > 0.3
        
        # 应识别为文言文风格
        assert '文言' in analysis['style'] or analysis['style'] == '未知风格'
    
    @pytest.mark.asyncio
    async def test_complexity_calculation(self, text_analyzer):
        """测试复杂度计算"""
        simple_text = "这是一个简单的现代汉语句子。"
        complex_text = "昔者仲尼与于蜡宾，事毕，出游于观之上，喟然而叹曰。"
        
        simple_complexity = await text_analyzer._calculate_complexity(simple_text)
        complex_complexity = await text_analyzer._calculate_complexity(complex_text)
        
        # 复杂文本应该有更高的复杂度分数
        assert complex_complexity > simple_complexity


class TestQualityAssessor:
    """质量评估器测试"""
    
    @pytest.fixture
    def quality_assessor(self):
        """创建质量评估器实例"""
        return QualityAssessor()
    
    @pytest.mark.asyncio
    async def test_assess_basic_quality(self, quality_assessor):
        """测试基础质量评估"""
        original = "朱元璋，濠州钟离人也。"
        optimized = "明太祖朱元璋，安徽濠州钟离县人。"
        
        metrics = await quality_assessor.assess_quality(
            original_text=original,
            optimized_text=optimized,
            optimization_type=OptimizationType.POLISH,
            optimization_mode=OptimizationMode.HISTORICAL_FORMAT
        )
        
        # 验证质量指标结构
        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.overall_score <= 100
        assert 0 <= metrics.readability_score <= 100
        assert 0 <= metrics.academic_score <= 100
        assert 0 <= metrics.historical_accuracy <= 100
        assert 0 <= metrics.language_quality <= 100
        assert 0 <= metrics.structure_score <= 100
        assert 0 <= metrics.content_completeness <= 100
        
        # 验证优势和不足列表
        assert isinstance(metrics.strengths, list)
        assert isinstance(metrics.weaknesses, list)
    
    @pytest.mark.asyncio
    async def test_historical_accuracy_preservation(self, quality_assessor):
        """测试历史准确性保持"""
        original = "朱元璋生于1328年，死于1398年。"
        optimized_good = "明太祖朱元璋生于1328年，逝世于1398年。"
        optimized_bad = "朱元璋生于元朝末年，死于明朝初期。"
        
        # 测试保持准确性的优化
        good_metrics = await quality_assessor.accuracy_analyzer.assess_historical_accuracy(
            original, optimized_good
        )
        
        # 测试改变准确性的优化
        bad_metrics = await quality_assessor.accuracy_analyzer.assess_historical_accuracy(
            original, optimized_bad
        )
        
        # 保持准确信息的版本应该有更高的历史准确性分数
        assert good_metrics > bad_metrics
    
    @pytest.mark.asyncio
    async def test_readability_assessment(self, quality_assessor):
        """测试可读性评估"""
        readable_text = "这是一段容易理解的现代汉语文本。句子长度适中，用词简单明了。"
        difficult_text = "此乃极难理解之古典文献，句式冗长复杂，词汇艰深晦涩，语法结构错综繁复。"
        
        readable_score = await quality_assessor.readability_analyzer.assess_readability(readable_text)
        difficult_score = await quality_assessor.readability_analyzer.assess_readability(difficult_text)
        
        # 易读文本应该有更高的可读性分数
        assert readable_score > difficult_score


class TestOptimizationStrategyManager:
    """优化策略管理器测试"""
    
    @pytest.fixture
    def mock_storage_client(self):
        """模拟存储服务客户端"""
        client = Mock(spec=StorageServiceClient)
        client.get_optimization_strategies = AsyncMock(return_value={
            'data': [
                {
                    'strategy_id': 'test-strategy-1',
                    'name': '测试策略1',
                    'description': '测试用策略',
                    'optimization_type': 'polish',
                    'optimization_mode': 'historical_format',
                    'system_prompt': '你是专业的文本优化专家',
                    'prompt_template': '请优化以下文本：{content}',
                    'preferred_model': 'test-model',
                    'is_active': True,
                    'usage_count': 0,
                    'success_rate': 0.0,
                    'avg_quality_improvement': 0.0,
                    'avg_processing_time_ms': 0
                }
            ]
        })
        client.get_user_optimization_preferences = AsyncMock(return_value={'data': {}})
        return client
    
    @pytest.fixture
    def strategy_manager(self, mock_storage_client):
        """创建策略管理器实例"""
        return OptimizationStrategyManager(mock_storage_client)
    
    @pytest.mark.asyncio
    async def test_load_strategies(self, strategy_manager, mock_storage_client):
        """测试策略加载"""
        await strategy_manager.load_strategies()
        
        # 验证策略已加载
        assert len(strategy_manager._strategies_cache) >= 1
        
        # 验证存储客户端被调用
        mock_storage_client.get_optimization_strategies.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_select_strategy(self, strategy_manager):
        """测试策略选择"""
        # 模拟文本分析结果
        text_analysis = {
            'content': '测试文本内容',
            'basic_stats': {'length': 100},
            'complexity': 0.5,
            'style': '现代白话文',
            'readability': 75.0
        }
        
        await strategy_manager.load_strategies()
        
        strategy = await strategy_manager.select_strategy(
            text_analysis=text_analysis,
            optimization_type=OptimizationType.POLISH,
            optimization_mode=OptimizationMode.HISTORICAL_FORMAT
        )
        
        # 验证返回了策略
        assert strategy is not None
        assert strategy.optimization_type == OptimizationType.POLISH


class TestTextOptimizationEngine:
    """文本优化引擎测试"""
    
    @pytest.fixture
    def mock_ai_client(self):
        """模拟AI模型客户端"""
        client = Mock(spec=AIModelServiceClient)
        client.select_best_model = AsyncMock(return_value="test-model")
        client.generate_optimization_prompt = AsyncMock(return_value={
            "system_prompt": "你是专业的文本优化专家",
            "user_prompt": "请优化以下文本：测试内容"
        })
        client.optimize_text_with_ai = AsyncMock(return_value={
            "optimized_content": "优化后的测试内容",
            "model_used": "test-model",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 150,
                "total_tokens": 250
            }
        })
        return client
    
    @pytest.fixture
    def mock_storage_client(self):
        """模拟存储服务客户端"""
        client = Mock(spec=StorageServiceClient)
        client.create_optimization_task = AsyncMock(return_value={"task_id": "test-task-id"})
        client.save_optimization_version = AsyncMock(return_value={"version_id": "test-version-id"})
        return client
    
    @pytest.fixture
    def optimization_engine(self, mock_ai_client, mock_storage_client):
        """创建文本优化引擎实例"""
        return TextOptimizationEngine(mock_ai_client, mock_storage_client)
    
    @pytest.mark.asyncio
    async def test_optimize_text_basic(self, optimization_engine, mock_ai_client):
        """测试基础文本优化"""
        request = OptimizationRequest(
            content="朱元璋，濠州钟离人也。",
            optimization_type=OptimizationType.POLISH,
            optimization_mode=OptimizationMode.HISTORICAL_FORMAT,
            generate_versions=1,
            user_id="test-user"
        )
        
        result = await optimization_engine.optimize_text(request)
        
        # 验证优化结果
        assert result.task_id is not None
        assert len(result.versions) == 1
        assert result.versions[0].content is not None
        assert result.versions[0].quality_metrics.overall_score > 0
        
        # 验证AI客户端被调用
        mock_ai_client.select_best_model.assert_called_once()
        mock_ai_client.generate_optimization_prompt.assert_called_once()
        mock_ai_client.optimize_text_with_ai.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimize_text_multiple_versions(self, optimization_engine):
        """测试多版本优化"""
        request = OptimizationRequest(
            content="测试内容",
            optimization_type=OptimizationType.EXPAND,
            optimization_mode=OptimizationMode.ACADEMIC,
            generate_versions=3
        )
        
        result = await optimization_engine.optimize_text(request)
        
        # 验证生成了多个版本
        assert len(result.versions) == 3
        assert result.total_versions == 3
        
        # 验证推荐版本
        assert result.recommended_version is not None
        
        # 验证统计信息
        assert result.average_quality_score > 0
        assert result.best_quality_score > 0


class TestIntegrationFlow:
    """完整集成流程测试"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """模拟所有依赖"""
        ai_client = Mock(spec=AIModelServiceClient)
        ai_client.select_best_model = AsyncMock(return_value="gemini-1.5-pro")
        ai_client.generate_optimization_prompt = AsyncMock(return_value={
            "system_prompt": "你是专业的历史文本优化专家",
            "user_prompt": "请对以下历史文本进行润色：朱元璋，濠州钟离人也。"
        })
        ai_client.optimize_text_with_ai = AsyncMock(return_value={
            "optimized_content": "明太祖朱元璋，安徽濠州钟离县人。",
            "model_used": "gemini-1.5-pro",
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 180,
                "total_tokens": 300
            }
        })
        
        storage_client = Mock(spec=StorageServiceClient)
        storage_client.get_optimization_strategies = AsyncMock(return_value={'data': []})
        storage_client.get_user_optimization_preferences = AsyncMock(return_value={'data': {}})
        storage_client.create_optimization_task = AsyncMock(return_value={"task_id": "integration-test"})
        storage_client.save_optimization_version = AsyncMock(return_value={"version_id": "version-1"})
        
        return ai_client, storage_client
    
    @pytest.mark.asyncio
    async def test_complete_optimization_flow(self, mock_dependencies):
        """测试完整的优化流程"""
        ai_client, storage_client = mock_dependencies
        
        # 创建优化引擎
        optimization_engine = TextOptimizationEngine(ai_client, storage_client)
        
        # 创建优化请求
        request = OptimizationRequest(
            content="朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州。",
            optimization_type=OptimizationType.POLISH,
            optimization_mode=OptimizationMode.HISTORICAL_FORMAT,
            parameters=OptimizationParameters(
                quality_threshold=80.0,
                preserve_entities=True
            ),
            generate_versions=2,
            user_id="integration-test-user"
        )
        
        # 执行优化
        result = await optimization_engine.optimize_text(request)
        
        # 验证完整流程结果
        assert result.task_id is not None
        assert result.status.value == "completed"
        assert len(result.versions) == 2
        assert result.total_versions == 2
        
        # 验证版本内容
        for version in result.versions:
            assert version.content is not None
            assert len(version.content) > 0
            assert version.quality_metrics is not None
            assert version.quality_metrics.overall_score > 0
            assert version.processing_time_ms > 0
            assert version.model_used == "gemini-1.5-pro"
        
        # 验证推荐版本
        assert result.recommended_version is not None
        
        # 验证统计信息
        assert result.average_quality_score > 0
        assert result.best_quality_score > 0
        assert result.total_processing_time_ms > 0
        
        print(f"集成测试完成:")
        print(f"- 任务ID: {result.task_id}")
        print(f"- 生成版本数: {result.total_versions}")
        print(f"- 平均质量分数: {result.average_quality_score}")
        print(f"- 最高质量分数: {result.best_quality_score}")
        print(f"- 总处理时间: {result.total_processing_time_ms}ms")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])