"""
智能分类服务工具模块
提供文本预处理和特征提取功能
"""

from .text_preprocessing import (
    ChineseTextPreprocessor,
    PreprocessingConfig,
    default_preprocessor,
    preprocess_text,
    preprocess_text_to_string
)

from .feature_extraction import (
    BaseFeatureExtractor,
    TfidfFeatureExtractor,
    CountFeatureExtractor,
    FeatureExtractorFactory,
    FeatureCombiner,
    get_default_extractor_config
)

__all__ = [
    # 文本预处理
    'ChineseTextPreprocessor',
    'PreprocessingConfig',
    'default_preprocessor',
    'preprocess_text',
    'preprocess_text_to_string',
    
    # 特征提取
    'BaseFeatureExtractor',
    'TfidfFeatureExtractor',
    'CountFeatureExtractor',
    'FeatureExtractorFactory',
    'FeatureCombiner',
    'get_default_extractor_config'
]