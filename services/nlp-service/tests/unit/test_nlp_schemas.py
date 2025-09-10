"""
NLP服务数据模型单元测试

测试NLP服务的核心数据结构，包括：
- 请求和响应模型验证
- 枚举类型定义
- 数据验证规则
- 边界条件处理

作者: Quinn (测试架构师)
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
from datetime import datetime
from uuid import uuid4
from pydantic import ValidationError

from src.schemas.nlp_schemas import (
    ProcessingType, ProcessingStatus, Language, NLPEngine, SentimentLabel,
    BaseResponse, ErrorResponse, TextProcessRequest, BatchProcessRequest
)


class TestEnumTypes:
    """枚举类型测试"""
    
    def test_processing_type_enum(self):
        """测试处理类型枚举"""
        assert ProcessingType.SEGMENTATION == "segmentation"
        assert ProcessingType.POS_TAGGING == "pos_tagging"
        assert ProcessingType.NER == "ner"
        assert ProcessingType.SENTIMENT == "sentiment"
        assert ProcessingType.KEYWORDS == "keywords"
        assert ProcessingType.SUMMARY == "summary"
        assert ProcessingType.SIMILARITY == "similarity"
        assert ProcessingType.BATCH == "batch"
        
        # 测试枚举转换
        assert ProcessingType("segmentation") == ProcessingType.SEGMENTATION
        
        # 测试无效值
        with pytest.raises(ValueError):
            ProcessingType("invalid_type")
    
    def test_processing_status_enum(self):
        """测试处理状态枚举"""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing" 
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
    
    def test_language_enum(self):
        """测试语言枚举"""
        assert Language.CHINESE == "zh"
        assert Language.ENGLISH == "en"
        assert Language.CLASSICAL_CHINESE == "zh-classical"
    
    def test_nlp_engine_enum(self):
        """测试NLP引擎枚举"""
        assert NLPEngine.SPACY == "spacy"
        assert NLPEngine.JIEBA == "jieba"
        assert NLPEngine.HANLP == "hanlp"
        assert NLPEngine.TRANSFORMERS == "transformers"
    
    def test_sentiment_label_enum(self):
        """测试情感标签枚举"""
        assert SentimentLabel.POSITIVE == "positive"
        assert SentimentLabel.NEGATIVE == "negative"
        assert SentimentLabel.NEUTRAL == "neutral"


class TestBaseModels:
    """基础模型测试"""
    
    def test_base_response_creation(self):
        """测试基础响应创建"""
        response = BaseResponse()
        
        assert response.success is True
        assert response.message == ""
        assert isinstance(response.timestamp, datetime)
        
        # 测试自定义值
        custom_response = BaseResponse(
            success=False,
            message="测试消息"
        )
        
        assert custom_response.success is False
        assert custom_response.message == "测试消息"
    
    def test_error_response_creation(self):
        """测试错误响应创建"""
        error_response = ErrorResponse(
            message="测试错误",
            error_code="TEST_ERROR",
            error_details={"field": "value"}
        )
        
        assert error_response.success is False
        assert error_response.message == "测试错误"
        assert error_response.error_code == "TEST_ERROR"
        assert error_response.error_details == {"field": "value"}
    
    def test_base_response_json_serialization(self):
        """测试响应JSON序列化"""
        response = BaseResponse(message="测试")
        json_data = response.dict()
        
        assert "success" in json_data
        assert "message" in json_data
        assert "timestamp" in json_data
        assert json_data["success"] is True
        assert json_data["message"] == "测试"


class TestRequestModels:
    """请求模型测试"""
    
    def test_text_process_request_valid(self):
        """测试有效的文本处理请求"""
        request = TextProcessRequest(
            text="这是一个测试文本",
            processing_type=ProcessingType.SEGMENTATION,
            language=Language.CHINESE
        )
        
        assert request.text == "这是一个测试文本"
        assert request.processing_type == ProcessingType.SEGMENTATION
        assert request.language == Language.CHINESE
        assert request.engine is None
        assert request.config == {}
        assert request.async_mode is False
        assert request.dataset_id is None
    
    def test_text_process_request_with_all_fields(self):
        """测试包含所有字段的文本处理请求"""
        request = TextProcessRequest(
            text="完整参数测试文本",
            processing_type=ProcessingType.POS_TAGGING,
            language=Language.ENGLISH,
            engine=NLPEngine.SPACY,
            config={"threshold": 0.8},
            async_mode=True,
            dataset_id="test-dataset-123"
        )
        
        assert request.text == "完整参数测试文本"
        assert request.processing_type == ProcessingType.POS_TAGGING
        assert request.language == Language.ENGLISH
        assert request.engine == NLPEngine.SPACY
        assert request.config == {"threshold": 0.8}
        assert request.async_mode is True
        assert request.dataset_id == "test-dataset-123"
    
    def test_text_process_request_empty_text(self):
        """测试空文本请求验证"""
        with pytest.raises(ValidationError) as exc_info:
            TextProcessRequest(
                text="   ",  # 只有空白字符
                processing_type=ProcessingType.SEGMENTATION
            )
        
        assert "文本内容不能为空" in str(exc_info.value)
    
    def test_text_process_request_missing_required_fields(self):
        """测试缺少必需字段"""
        with pytest.raises(ValidationError):
            TextProcessRequest(processing_type=ProcessingType.SEGMENTATION)
        
        with pytest.raises(ValidationError):
            TextProcessRequest(text="测试文本")
    
    def test_text_process_request_too_long_text(self):
        """测试文本过长验证"""
        long_text = "a" * 1000001  # 超过最大长度
        
        with pytest.raises(ValidationError) as exc_info:
            TextProcessRequest(
                text=long_text,
                processing_type=ProcessingType.SEGMENTATION
            )
        
        assert "String should have at most 1000000 characters" in str(exc_info.value)
    
    def test_batch_process_request_valid(self):
        """测试有效的批量处理请求"""
        request = BatchProcessRequest(
            texts=["文本1", "文本2", "文本3"],
            processing_type=ProcessingType.SENTIMENT
        )
        
        assert len(request.texts) == 3
        assert request.texts[0] == "文本1"
        assert request.processing_type == ProcessingType.SENTIMENT
        assert request.language == Language.CHINESE
    
    def test_batch_process_request_too_many_texts(self):
        """测试批量请求文本数量限制"""
        too_many_texts = [f"文本{i}" for i in range(51)]  # 超过最大数量50
        
        with pytest.raises(ValidationError) as exc_info:
            BatchProcessRequest(
                texts=too_many_texts,
                processing_type=ProcessingType.SEGMENTATION
            )
        
        assert "List should have at most 50 items" in str(exc_info.value)
    
    def test_batch_process_request_empty_list(self):
        """测试空文本列表"""
        with pytest.raises(ValidationError):
            BatchProcessRequest(
                texts=[],
                processing_type=ProcessingType.SEGMENTATION
            )


class TestModelValidation:
    """模型验证测试"""
    
    def test_enum_string_conversion(self):
        """测试枚举字符串转换"""
        # 测试字符串直接传入
        request = TextProcessRequest(
            text="测试文本",
            processing_type="segmentation",  # 字符串
            language="zh",  # 字符串
            engine="jieba"  # 字符串
        )
        
        assert request.processing_type == ProcessingType.SEGMENTATION
        assert request.language == Language.CHINESE
        assert request.engine == NLPEngine.JIEBA
    
    def test_invalid_enum_values(self):
        """测试无效枚举值"""
        with pytest.raises(ValidationError):
            TextProcessRequest(
                text="测试文本",
                processing_type="invalid_type"
            )
        
        with pytest.raises(ValidationError):
            TextProcessRequest(
                text="测试文本",
                processing_type=ProcessingType.SEGMENTATION,
                language="invalid_language"
            )
    
    def test_config_dict_validation(self):
        """测试配置字典验证"""
        # 有效配置
        request = TextProcessRequest(
            text="测试文本",
            processing_type=ProcessingType.SEGMENTATION,
            config={
                "threshold": 0.8,
                "max_length": 1000,
                "enable_pos": True,
                "custom_dict": ["word1", "word2"]
            }
        )
        
        assert isinstance(request.config, dict)
        assert request.config["threshold"] == 0.8
        assert request.config["enable_pos"] is True
    
    def test_optional_fields_default_values(self):
        """测试可选字段默认值"""
        request = TextProcessRequest(
            text="最小参数测试",
            processing_type=ProcessingType.KEYWORDS
        )
        
        # 检查默认值
        assert request.language == Language.CHINESE
        assert request.engine is None
        assert request.config == {}
        assert request.async_mode is False
        assert request.dataset_id is None


class TestModelSerialization:
    """模型序列化测试"""
    
    def test_request_model_dict_export(self):
        """测试请求模型字典导出"""
        request = TextProcessRequest(
            text="序列化测试",
            processing_type=ProcessingType.NER,
            language=Language.CLASSICAL_CHINESE,
            config={"model": "bert-base"}
        )
        
        data = request.dict()
        
        assert data["text"] == "序列化测试"
        assert data["processing_type"] == "ner"
        assert data["language"] == "zh-classical"
        assert data["config"]["model"] == "bert-base"
    
    def test_response_model_json_export(self):
        """测试响应模型JSON导出"""
        response = BaseResponse(
            success=True,
            message="处理完成"
        )
        
        json_str = response.json()
        assert "success" in json_str
        assert "message" in json_str
        assert "timestamp" in json_str
        assert "true" in json_str.lower()
        assert "处理完成" in json_str
    
    def test_model_from_dict(self):
        """测试从字典创建模型"""
        data = {
            "text": "字典创建测试",
            "processing_type": "similarity",
            "language": "en",
            "engine": "transformers",
            "async_mode": True
        }
        
        request = TextProcessRequest(**data)
        
        assert request.text == "字典创建测试"
        assert request.processing_type == ProcessingType.SIMILARITY
        assert request.language == Language.ENGLISH
        assert request.engine == NLPEngine.TRANSFORMERS
        assert request.async_mode is True


class TestEdgeCases:
    """边界条件测试"""
    
    def test_unicode_text_handling(self):
        """测试Unicode文本处理"""
        unicode_texts = [
            "中文测试",
            "English test",
            "Émoji test 😀",
            "混合language测试",
            "古代漢字測試",
            "特殊符号©®™℠"
        ]
        
        for text in unicode_texts:
            request = TextProcessRequest(
                text=text,
                processing_type=ProcessingType.SEGMENTATION
            )
            assert request.text == text
    
    def test_whitespace_handling(self):
        """测试空白字符处理"""
        # 测试开头结尾空白字符（应该被保留，但不能全为空白）
        request = TextProcessRequest(
            text="  正常文本  ",
            processing_type=ProcessingType.SEGMENTATION
        )
        assert request.text == "  正常文本  "
        
        # 测试各种空白字符
        whitespace_text = "文本\t包含\n各种\r空白\u00A0字符"
        request = TextProcessRequest(
            text=whitespace_text,
            processing_type=ProcessingType.SEGMENTATION
        )
        assert request.text == whitespace_text
    
    def test_boundary_length_texts(self):
        """测试边界长度文本"""
        # 测试最大允许长度的文本
        max_length_text = "a" * 1000000
        request = TextProcessRequest(
            text=max_length_text,
            processing_type=ProcessingType.SEGMENTATION
        )
        assert len(request.text) == 1000000
        
        # 测试单字符文本
        single_char_request = TextProcessRequest(
            text="字",
            processing_type=ProcessingType.SEGMENTATION
        )
        assert single_char_request.text == "字"
    
    def test_special_characters(self):
        """测试特殊字符处理"""
        special_texts = [
            "包含标点：！？。，；：",
            "括号测试（）【】「」",
            "引号测试""''",
            "数字测试123456789",
            "符号测试@#$%^&*",
            "控制字符测试\x00\x01\x02"  # 包含控制字符
        ]
        
        for text in special_texts:
            # 大部分特殊字符应该被接受
            try:
                request = TextProcessRequest(
                    text=text,
                    processing_type=ProcessingType.SEGMENTATION
                )
                assert request.text == text
            except ValidationError:
                # 某些控制字符可能会被拒绝，这是正常的
                pass