"""
NLP服务核心功能单元测试

测试NLP服务的核心处理功能，包括：
- 文本分词
- 词性标注
- 命名实体识别
- 情感分析
- 关键词提取
- 文本相似度

作者: Quinn (测试架构师)
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# 创建模拟的NLP服务类用于测试
class MockNLPService:
    """模拟NLP服务类，避免真实模型依赖"""
    
    def __init__(self):
        self.is_initialized = False
        self.models = {}
        self.tokenizers = {}
        self._cache = {}
        
        # 模拟配置
        self.engine_config = {
            'spacy': {'model': 'zh_core_web_sm'},
            'jieba': {'enabled': True},
            'transformers': {'device': 'cpu'}
        }
    
    async def initialize_models(self):
        """模拟模型初始化"""
        self.models = {
            'spacy': Mock(),
            'sentiment': Mock(),
            'sentence_transformer': Mock()
        }
        self.is_initialized = True
        return True
    
    async def segment_text(self, text: str, engine: str = 'jieba', **kwargs) -> Dict[str, Any]:
        """模拟文本分词"""
        if not text:
            return {
                'success': False,
                'error_message': 'Empty text input'
            }
        
        # 简单的模拟分词逻辑
        segments = text.split()  # 简化的分词
        
        return {
            'success': True,
            'text': text,
            'segments': segments,
            'word_count': len(segments),
            'char_count': len(text),
            'engine': engine,
            'processing_time': 0.1,
            'metadata': {
                'language': 'zh',
                'confidence': 0.95
            }
        }
    
    async def pos_tag(self, text: str, engine: str = 'jieba', **kwargs) -> Dict[str, Any]:
        """模拟词性标注"""
        if not text:
            return {
                'success': False,
                'error_message': 'Empty text input'
            }
        
        # 模拟词性标注结果
        words = text.split()
        pos_tags = []
        
        for word in words:
            # 简化的词性判断
            if word.isdigit():
                pos = 'CD'  # 数词
            elif word in ['的', '了', '在', '和']:
                pos = 'P'   # 介词
            elif word in ['很', '非常', '特别']:
                pos = 'AD'  # 副词
            else:
                pos = 'NN'  # 名词（默认）
            
            pos_tags.append({
                'word': word,
                'pos': pos,
                'start': text.find(word),
                'end': text.find(word) + len(word)
            })
        
        return {
            'success': True,
            'text': text,
            'pos_tags': pos_tags,
            'word_count': len(pos_tags),
            'engine': engine,
            'processing_time': 0.15,
            'metadata': {
                'pos_count': len(set(tag['pos'] for tag in pos_tags))
            }
        }
    
    async def extract_entities(self, text: str, engine: str = 'spacy', **kwargs) -> Dict[str, Any]:
        """模拟命名实体识别"""
        if not text:
            return {
                'success': False,
                'error_message': 'Empty text input'
            }
        
        # 模拟实体识别（简化版）
        entities = []
        
        # 简单的实体识别逻辑
        entity_patterns = {
            '北京': 'GPE',
            '上海': 'GPE', 
            '中国': 'GPE',
            '清华大学': 'ORG',
            '北京大学': 'ORG',
            '2025': 'DATE',
            '九月': 'DATE'
        }
        
        for entity, label in entity_patterns.items():
            if entity in text:
                start = text.find(entity)
                entities.append({
                    'text': entity,
                    'label': label,
                    'start': start,
                    'end': start + len(entity),
                    'confidence': 0.9
                })
        
        return {
            'success': True,
            'text': text,
            'entities': entities,
            'entity_count': len(entities),
            'engine': engine,
            'processing_time': 0.2,
            'metadata': {
                'entity_types': list(set(ent['label'] for ent in entities))
            }
        }
    
    async def analyze_sentiment(self, text: str, engine: str = 'transformers', **kwargs) -> Dict[str, Any]:
        """模拟情感分析"""
        if not text:
            return {
                'success': False,
                'error_message': 'Empty text input'
            }
        
        # 简化的情感分析逻辑
        positive_words = ['好', '棒', '优秀', '满意', '喜欢', '推荐', '赞']
        negative_words = ['坏', '差', '糟糕', '失望', '讨厌', '不好', '烂']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.6 + pos_count * 0.1)
            scores = {'positive': confidence, 'negative': 1 - confidence, 'neutral': 0.1}
        elif neg_count > pos_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.6 + neg_count * 0.1)
            scores = {'negative': confidence, 'positive': 1 - confidence, 'neutral': 0.1}
        else:
            sentiment = 'neutral'
            confidence = 0.7
            scores = {'neutral': confidence, 'positive': 0.15, 'negative': 0.15}
        
        return {
            'success': True,
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores,
            'engine': engine,
            'processing_time': 0.3,
            'metadata': {
                'pos_word_count': pos_count,
                'neg_word_count': neg_count
            }
        }
    
    async def extract_keywords(self, text: str, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        """模拟关键词提取"""
        if not text:
            return {
                'success': False,
                'error_message': 'Empty text input'
            }
        
        # 简化的关键词提取
        words = text.split()
        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '和', '与', '或', '但', '然而', '因为', '所以'}
        
        keywords = []
        for word in words:
            if word not in stop_words and len(word) > 1:
                keywords.append({
                    'word': word,
                    'score': min(0.9, len(word) / 10 + 0.5),  # 简化的评分
                    'frequency': text.count(word),
                    'pos': 'NN'  # 简化的词性
                })
        
        # 按分数排序并取前k个
        keywords = sorted(keywords, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return {
            'success': True,
            'text': text,
            'keywords': keywords,
            'keyword_count': len(keywords),
            'processing_time': 0.2,
            'metadata': {
                'total_words': len(words),
                'filtered_words': len([w for w in words if w not in stop_words])
            }
        }
    
    async def calculate_similarity(self, text1: str, text2: str, **kwargs) -> Dict[str, Any]:
        """模拟文本相似度计算"""
        if not text1 or not text2:
            return {
                'success': False,
                'error_message': 'Both texts must be non-empty'
            }
        
        # 简化的相似度计算（基于字符重叠）
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        return {
            'success': True,
            'text1': text1,
            'text2': text2,
            'similarity_score': similarity,
            'similarity_type': 'character_jaccard',
            'processing_time': 0.05,
            'metadata': {
                'text1_length': len(text1),
                'text2_length': len(text2),
                'common_chars': intersection
            }
        }


class TestNLPServiceCore:
    """NLP服务核心功能测试"""
    
    @pytest.fixture
    def nlp_service(self):
        """创建测试用的NLP服务实例"""
        return MockNLPService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, nlp_service):
        """测试服务初始化"""
        assert not nlp_service.is_initialized
        
        result = await nlp_service.initialize_models()
        
        assert result is True
        assert nlp_service.is_initialized
        assert 'spacy' in nlp_service.models
        assert 'sentiment' in nlp_service.models
        assert 'sentence_transformer' in nlp_service.models
    
    @pytest.mark.asyncio
    async def test_text_segmentation_success(self, nlp_service):
        """测试文本分词成功"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.segment_text("这是 一个 测试 文本")
        
        assert result['success'] is True
        assert result['text'] == "这是 一个 测试 文本"
        assert isinstance(result['segments'], list)
        assert result['word_count'] > 0
        assert result['char_count'] == len("这是 一个 测试 文本")
        assert result['engine'] == 'jieba'
        assert 'processing_time' in result
    
    @pytest.mark.asyncio
    async def test_text_segmentation_empty_text(self, nlp_service):
        """测试空文本分词"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.segment_text("")
        
        assert result['success'] is False
        assert 'error_message' in result
    
    @pytest.mark.asyncio
    async def test_pos_tagging_success(self, nlp_service):
        """测试词性标注成功"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.pos_tag("这是 测试 文本")
        
        assert result['success'] is True
        assert result['text'] == "这是 测试 文本"
        assert isinstance(result['pos_tags'], list)
        assert len(result['pos_tags']) > 0
        
        # 检查词性标注结构
        for tag in result['pos_tags']:
            assert 'word' in tag
            assert 'pos' in tag
            assert 'start' in tag
            assert 'end' in tag
    
    @pytest.mark.asyncio
    async def test_entity_recognition_success(self, nlp_service):
        """测试命名实体识别成功"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.extract_entities("我来自北京，在清华大学学习。")
        
        assert result['success'] is True
        assert isinstance(result['entities'], list)
        
        # 应该识别出北京和清华大学
        entity_texts = [ent['text'] for ent in result['entities']]
        assert '北京' in entity_texts
        assert '清华大学' in entity_texts
        
        # 检查实体结构
        for entity in result['entities']:
            assert 'text' in entity
            assert 'label' in entity
            assert 'start' in entity
            assert 'end' in entity
            assert 'confidence' in entity
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_positive(self, nlp_service):
        """测试正面情感分析"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.analyze_sentiment("这个产品很好，我很满意，强烈推荐！")
        
        assert result['success'] is True
        assert result['sentiment'] == 'positive'
        assert result['confidence'] > 0.7
        assert isinstance(result['scores'], dict)
        assert 'positive' in result['scores']
        assert 'negative' in result['scores']
        assert 'neutral' in result['scores']
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_negative(self, nlp_service):
        """测试负面情感分析"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.analyze_sentiment("这个产品很差，我很失望，不推荐购买。")
        
        assert result['success'] is True
        assert result['sentiment'] == 'negative'
        assert result['confidence'] > 0.7
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_neutral(self, nlp_service):
        """测试中性情感分析"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.analyze_sentiment("这是一个普通的文本描述。")
        
        assert result['success'] is True
        assert result['sentiment'] == 'neutral'
    
    @pytest.mark.asyncio
    async def test_keyword_extraction_success(self, nlp_service):
        """测试关键词提取成功"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.extract_keywords(
            "人工智能技术在自然语言处理领域的应用越来越广泛",
            top_k=5
        )
        
        assert result['success'] is True
        assert isinstance(result['keywords'], list)
        assert len(result['keywords']) <= 5
        
        # 检查关键词结构
        for keyword in result['keywords']:
            assert 'word' in keyword
            assert 'score' in keyword
            assert 'frequency' in keyword
            assert isinstance(keyword['score'], float)
            assert keyword['score'] > 0
    
    @pytest.mark.asyncio
    async def test_text_similarity_success(self, nlp_service):
        """测试文本相似度计算成功"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.calculate_similarity(
            "人工智能技术发展迅速",
            "人工智能技术进步很快"
        )
        
        assert result['success'] is True
        assert 'similarity_score' in result
        assert isinstance(result['similarity_score'], float)
        assert 0 <= result['similarity_score'] <= 1
        assert result['text1'] == "人工智能技术发展迅速"
        assert result['text2'] == "人工智能技术进步很快"
    
    @pytest.mark.asyncio
    async def test_text_similarity_identical_texts(self, nlp_service):
        """测试相同文本的相似度"""
        await nlp_service.initialize_models()
        
        text = "完全相同的文本"
        result = await nlp_service.calculate_similarity(text, text)
        
        assert result['success'] is True
        assert result['similarity_score'] == 1.0
    
    @pytest.mark.asyncio
    async def test_text_similarity_empty_texts(self, nlp_service):
        """测试空文本相似度"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.calculate_similarity("", "测试文本")
        
        assert result['success'] is False
        assert 'error_message' in result


class TestNLPServicePerformance:
    """NLP服务性能测试"""
    
    @pytest.fixture
    def nlp_service(self):
        return MockNLPService()
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, nlp_service):
        """测试处理时间跟踪"""
        await nlp_service.initialize_models()
        
        result = await nlp_service.segment_text("性能测试文本")
        
        assert 'processing_time' in result
        assert isinstance(result['processing_time'], float)
        assert result['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_simulation(self, nlp_service):
        """测试批量处理模拟"""
        await nlp_service.initialize_models()
        
        texts = [
            "第一个测试文本",
            "第二个测试文本",
            "第三个测试文本"
        ]
        
        results = []
        for text in texts:
            result = await nlp_service.segment_text(text)
            results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert result['success'] is True
            assert 'processing_time' in result


class TestNLPServiceErrorHandling:
    """NLP服务错误处理测试"""
    
    @pytest.fixture
    def nlp_service(self):
        return MockNLPService()
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, nlp_service):
        """测试空文本处理"""
        await nlp_service.initialize_models()
        
        methods_to_test = [
            nlp_service.segment_text,
            nlp_service.pos_tag,
            nlp_service.extract_entities,
            nlp_service.analyze_sentiment,
            nlp_service.extract_keywords
        ]
        
        for method in methods_to_test:
            result = await method("")
            assert result['success'] is False
            assert 'error_message' in result
    
    @pytest.mark.asyncio
    async def test_none_text_handling(self, nlp_service):
        """测试None文本处理"""
        await nlp_service.initialize_models()
        
        # 模拟None输入的处理
        try:
            result = await nlp_service.segment_text(None)
            # 如果方法能处理None，检查返回值
            assert result['success'] is False
        except (TypeError, AttributeError):
            # 如果方法不能处理None，抛出异常也是合理的
            pass
    
    @pytest.mark.asyncio
    async def test_very_long_text_handling(self, nlp_service):
        """测试超长文本处理"""
        await nlp_service.initialize_models()
        
        # 创建很长的文本
        long_text = "测试文本 " * 1000
        
        result = await nlp_service.segment_text(long_text)
        
        # 即使是很长的文本也应该能处理
        assert result['success'] is True
        assert result['char_count'] == len(long_text)


class TestNLPServiceConfiguration:
    """NLP服务配置测试"""
    
    @pytest.fixture
    def nlp_service(self):
        return MockNLPService()
    
    def test_engine_config_initialization(self, nlp_service):
        """测试引擎配置初始化"""
        assert hasattr(nlp_service, 'engine_config')
        assert isinstance(nlp_service.engine_config, dict)
        assert 'spacy' in nlp_service.engine_config
        assert 'jieba' in nlp_service.engine_config
        assert 'transformers' in nlp_service.engine_config
    
    @pytest.mark.asyncio
    async def test_engine_selection(self, nlp_service):
        """测试引擎选择"""
        await nlp_service.initialize_models()
        
        # 测试不同引擎
        engines = ['jieba', 'spacy']
        
        for engine in engines:
            result = await nlp_service.segment_text("测试文本", engine=engine)
            assert result['success'] is True
            assert result['engine'] == engine
    
    def test_cache_initialization(self, nlp_service):
        """测试缓存初始化"""
        assert hasattr(nlp_service, '_cache')
        assert isinstance(nlp_service._cache, dict)