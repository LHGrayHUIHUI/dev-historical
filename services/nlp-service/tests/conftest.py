"""
NLP服务测试配置文件
设置测试环境、夹具和通用测试工具

作者: Quinn (测试架构师)  
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

# 测试配置
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_nlp_service():
    """模拟NLP服务"""
    service = AsyncMock()
    service.is_initialized = True
    service.models = {
        'spacy': Mock(),
        'sentiment': Mock(),
        'sentence_transformer': Mock()
    }
    service.tokenizers = {
        'bert': Mock()
    }
    
    # 模拟各种NLP功能
    service.segment_text.return_value = {
        'success': True,
        'segments': ['这是', '一个', '测试', '文本'],
        'word_count': 4,
        'char_count': 8,
        'processing_time': 0.1
    }
    
    service.analyze_sentiment.return_value = {
        'success': True,
        'sentiment': 'positive',
        'confidence': 0.85,
        'scores': {'positive': 0.85, 'negative': 0.15}
    }
    
    service.extract_keywords.return_value = {
        'success': True,
        'keywords': [
            {'word': '测试', 'score': 0.9, 'pos': 'n'},
            {'word': '文本', 'score': 0.8, 'pos': 'n'}
        ],
        'keyword_count': 2
    }
    
    service.extract_entities.return_value = {
        'success': True,
        'entities': [
            {'text': '北京', 'label': 'GPE', 'start': 0, 'end': 2, 'confidence': 0.95}
        ],
        'entity_count': 1
    }
    
    return service


@pytest.fixture
def sample_texts():
    """测试文本数据"""
    return {
        'modern_chinese': '这是一个现代中文文本处理的测试案例。',
        'ancient_chinese': '古之學者必有師，師者，所以傳道受業解惑也。',
        'mixed_language': 'This is a mixed 中英文 text for testing purposes.',
        'traditional_chinese': '這是一個繁體中文的測試文本。',
        'long_text': '这是一个很长的文本，用来测试NLP服务处理长文本的能力。' * 20,
        'short_text': '短文本',
        'empty_text': '',
        'special_chars': '这个文本包含特殊字符：@#$%^&*()，用来测试特殊字符处理。',
        'numbers_mixed': '今天是2025年9月9日，温度25度。',
        'punctuation_heavy': '这是一个标点符号很多的文本：！？。，；：""''（）【】',
    }


@pytest.fixture 
def sample_ancient_texts():
    """古代文本样本"""
    return {
        'poetry': '床前明月光，疑是地上霜。举头望明月，低头思故乡。',
        'prose': '学而时习之，不亦说乎？有朋自远方来，不亦乐乎？',
        'classical': '天下皆知美之为美，斯恶已；皆知善之为善，斯不善已。',
        'historical': '秦始皇帝者，秦庄襄王子也。庄襄王为秦质子於赵，见吕不韦姬，悦而取之。'
    }


@pytest.fixture
def nlp_test_config():
    """NLP测试配置"""
    return {
        'models': {
            'spacy_model': 'zh_core_web_sm',
            'sentiment_model': 'bert-base-chinese',
            'sentence_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        },
        'engines': {
            'jieba_enabled': True,
            'spacy_enabled': True,
            'transformers_enabled': True
        },
        'processing': {
            'max_text_length': 10000,
            'batch_size': 32,
            'timeout': 30.0
        },
        'features': {
            'enable_pos_tagging': True,
            'enable_ner': True,
            'enable_sentiment_analysis': True,
            'enable_keyword_extraction': True,
            'enable_text_similarity': True,
            'enable_summarization': True
        }
    }


@pytest.fixture
def mock_jieba():
    """模拟jieba分词器"""
    mock = Mock()
    mock.cut.return_value = ['这是', '一个', '测试']
    mock.posseg.cut.return_value = [
        Mock(word='这是', flag='r'),
        Mock(word='一个', flag='m'),
        Mock(word='测试', flag='n')
    ]
    return mock


@pytest.fixture
def mock_spacy_model():
    """模拟spaCy模型"""
    mock_model = Mock()
    
    # 模拟文档对象
    mock_doc = Mock()
    mock_doc.text = '测试文本'
    
    # 模拟token
    mock_token1 = Mock()
    mock_token1.text = '测试'
    mock_token1.pos_ = 'NOUN'
    mock_token1.lemma_ = '测试'
    mock_token1.is_alpha = True
    
    mock_token2 = Mock()
    mock_token2.text = '文本'
    mock_token2.pos_ = 'NOUN'
    mock_token2.lemma_ = '文本'
    mock_token2.is_alpha = True
    
    mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
    
    # 模拟实体
    mock_ent = Mock()
    mock_ent.text = '北京'
    mock_ent.label_ = 'GPE'
    mock_ent.start_char = 0
    mock_ent.end_char = 2
    
    mock_doc.ents = [mock_ent]
    
    mock_model.return_value = mock_doc
    return mock_model


@pytest.fixture
def mock_transformers_pipeline():
    """模拟Transformers pipeline"""
    mock = Mock()
    mock.return_value = [
        {'label': 'POSITIVE', 'score': 0.85}
    ]
    return mock


@pytest.fixture
def mock_sentence_transformer():
    """模拟句子向量模型"""
    mock = Mock()
    mock.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # 模拟向量
    return mock


@pytest.fixture
def performance_benchmarks():
    """性能基准数据"""
    return {
        'text_segmentation': {
            'max_time': 1.0,  # 秒
            'min_accuracy': 0.90
        },
        'sentiment_analysis': {
            'max_time': 2.0,
            'min_accuracy': 0.85
        },
        'keyword_extraction': {
            'max_time': 1.5,
            'min_keywords': 3
        },
        'entity_recognition': {
            'max_time': 2.0,
            'min_precision': 0.80
        },
        'text_similarity': {
            'max_time': 1.0,
            'min_correlation': 0.70
        }
    }


@pytest.fixture
def accuracy_test_cases():
    """准确性测试用例"""
    return [
        {
            'name': 'modern_chinese_segmentation',
            'text': '现代汉语自动分词技术测试',
            'expected_segments': ['现代', '汉语', '自动', '分词', '技术', '测试'],
            'min_accuracy': 0.90
        },
        {
            'name': 'ancient_chinese_analysis',
            'text': '古代漢字文本分析測試',
            'expected_difficulty': 'high',
            'min_accuracy': 0.80
        },
        {
            'name': 'sentiment_classification',
            'text': '这个产品非常好用，我很满意！',
            'expected_sentiment': 'positive',
            'min_confidence': 0.80
        },
        {
            'name': 'entity_extraction',
            'text': '北京大学位于中国北京市海淀区',
            'expected_entities': ['北京大学', '中国', '北京市', '海淀区'],
            'min_precision': 0.75
        }
    ]


@pytest.fixture
def chinese_linguistic_features():
    """中文语言学特征测试"""
    return {
        'word_segmentation': {
            'text': '中文分词是自然语言处理的基础任务',
            'expected_features': ['segmentation_quality', 'boundary_accuracy']
        },
        'pos_tagging': {
            'text': '这是一个简单的词性标注测试',
            'expected_tags': ['DT', 'VC', 'CD', 'JJ', 'DE', 'NN', 'NN', 'NN']
        },
        'traditional_simplified': {
            'traditional': '這是繁體中文測試',
            'simplified': '这是繁体中文测试',
            'conversion_accuracy': 1.0
        }
    }


@pytest.fixture
def error_handling_cases():
    """错误处理测试用例"""
    return [
        {'input': None, 'expected_error': 'invalid_input'},
        {'input': '', 'expected_error': 'empty_text'},
        {'input': 'a' * 20000, 'expected_error': 'text_too_long'},
        {'input': '���非法字符���', 'expected_error': 'encoding_error'},
        {'input': '\x00\x01\x02', 'expected_error': 'invalid_characters'}
    ]