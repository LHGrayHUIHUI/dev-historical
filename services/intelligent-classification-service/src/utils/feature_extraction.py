"""
智能分类服务特征提取模块
简化版本，只支持TF-IDF特征提取
针对中文历史文献优化的特征工程工具
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
import joblib
import os
from pathlib import Path

# 传统机器学习特征提取
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from .text_preprocessing import ChineseTextPreprocessor, PreprocessingConfig
from ..config.settings import settings


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.preprocessor = ChineseTextPreprocessor()
    
    @abstractmethod
    def fit(self, texts: List[str], labels: Optional[List[str]] = None) -> 'BaseFeatureExtractor':
        """训练特征提取器"""
        pass
    
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """提取特征"""
        pass
    
    def fit_transform(self, texts: List[str], labels: Optional[List[str]] = None) -> np.ndarray:
        """训练并提取特征"""
        return self.fit(texts, labels).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return getattr(self, 'feature_names_', [])
    
    def save(self, filepath: str):
        """保存特征提取器"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'extractor': self,
            'config': self.config,
            'is_fitted': self.is_fitted
        }, filepath)
        self.logger.info(f"特征提取器已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseFeatureExtractor':
        """加载特征提取器"""
        data = joblib.load(filepath)
        extractor = data['extractor']
        extractor.config = data['config']
        extractor.is_fitted = data['is_fitted']
        return extractor


class TfidfFeatureExtractor(BaseFeatureExtractor):
    """TF-IDF特征提取器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("tfidf", config)
        
        # 配置参数
        self.max_features = self.config.get('max_features', 5000)
        self.min_df = self.config.get('min_df', 2)
        self.max_df = self.config.get('max_df', 0.8)
        self.ngram_range = tuple(self.config.get('ngram_range', [1, 2]))
        self.use_idf = self.config.get('use_idf', True)
        self.sublinear_tf = self.config.get('sublinear_tf', True)
        self.norm = self.config.get('norm', 'l2')
        
        # 初始化TF-IDF向量化器
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            norm=self.norm,
            token_pattern=r'(?u)\b\w+\b',  # 适用于中文分词结果
            lowercase=False  # 中文不需要转小写
        )
        
    def fit(self, texts: List[str], labels: Optional[List[str]] = None) -> 'TfidfFeatureExtractor':
        """训练TF-IDF特征提取器"""
        if not texts:
            raise ValueError("训练文本不能为空")
            
        # 预处理文本
        processed_texts = []
        for text in texts:
            processed = self.preprocessor.preprocess(text, return_tokens=False)
            processed_texts.append(processed)
        
        # 训练TF-IDF模型
        self.tfidf.fit(processed_texts)
        self.feature_names_ = list(self.tfidf.get_feature_names_out())
        self.is_fitted = True
        
        self.logger.info(f"TF-IDF特征提取器训练完成，特征数量: {len(self.feature_names_)}")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """使用TF-IDF提取特征"""
        if not self.is_fitted:
            raise ValueError("特征提取器未训练，请先调用fit()方法")
            
        if not texts:
            return np.array([])
        
        # 预处理文本
        processed_texts = []
        for text in texts:
            processed = self.preprocessor.preprocess(text, return_tokens=False)
            processed_texts.append(processed)
        
        # 提取TF-IDF特征
        features = self.tfidf.transform(processed_texts).toarray()
        
        self.logger.debug(f"提取了 {len(texts)} 个文本的TF-IDF特征，特征维度: {features.shape}")
        return features
    
    def get_top_words(self, n: int = 20) -> List[Tuple[str, float]]:
        """获取重要词汇及其权重"""
        if not self.is_fitted:
            raise ValueError("特征提取器未训练")
        
        # 计算所有词汇的平均TF-IDF分数
        feature_names = self.get_feature_names()
        if not feature_names:
            return []
        
        # 获取词汇的IDF值作为重要性度量
        idf_scores = self.tfidf.idf_
        word_scores = list(zip(feature_names, idf_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[:n]
    
    def get_feature_importance(self, text: str, n: int = 10) -> List[Tuple[str, float]]:
        """获取单个文本的重要特征"""
        if not self.is_fitted:
            raise ValueError("特征提取器未训练")
        
        # 提取文本特征
        features = self.transform([text])[0]
        feature_names = self.get_feature_names()
        
        # 获取非零特征及其分数
        nonzero_indices = np.nonzero(features)[0]
        if len(nonzero_indices) == 0:
            return []
        
        word_scores = [(feature_names[i], features[i]) for i in nonzero_indices]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[:n]


class CountFeatureExtractor(BaseFeatureExtractor):
    """词频特征提取器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("count", config)
        
        # 配置参数
        self.max_features = self.config.get('max_features', 5000)
        self.min_df = self.config.get('min_df', 2)
        self.max_df = self.config.get('max_df', 0.8)
        self.ngram_range = tuple(self.config.get('ngram_range', [1, 2]))
        
        # 初始化词频向量化器
        self.count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            token_pattern=r'(?u)\b\w+\b',
            lowercase=False
        )
        
    def fit(self, texts: List[str], labels: Optional[List[str]] = None) -> 'CountFeatureExtractor':
        """训练词频特征提取器"""
        if not texts:
            raise ValueError("训练文本不能为空")
            
        # 预处理文本
        processed_texts = []
        for text in texts:
            processed = self.preprocessor.preprocess(text, return_tokens=False)
            processed_texts.append(processed)
        
        # 训练词频模型
        self.count_vectorizer.fit(processed_texts)
        self.feature_names_ = list(self.count_vectorizer.get_feature_names_out())
        self.is_fitted = True
        
        self.logger.info(f"词频特征提取器训练完成，特征数量: {len(self.feature_names_)}")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """使用词频提取特征"""
        if not self.is_fitted:
            raise ValueError("特征提取器未训练，请先调用fit()方法")
            
        if not texts:
            return np.array([])
        
        # 预处理文本
        processed_texts = []
        for text in texts:
            processed = self.preprocessor.preprocess(text, return_tokens=False)
            processed_texts.append(processed)
        
        # 提取词频特征
        features = self.count_vectorizer.transform(processed_texts).toarray()
        
        self.logger.debug(f"提取了 {len(texts)} 个文本的词频特征，特征维度: {features.shape}")
        return features


class FeatureExtractorFactory:
    """特征提取器工厂"""
    
    _extractors = {
        'tfidf': TfidfFeatureExtractor,
        'count': CountFeatureExtractor,
    }
    
    @classmethod
    def create(cls, extractor_type: str, config: Dict[str, Any] = None) -> BaseFeatureExtractor:
        """创建特征提取器"""
        if extractor_type not in cls._extractors:
            available_types = list(cls._extractors.keys())
            raise ValueError(f"不支持的特征提取器类型: {extractor_type}，可用类型: {available_types}")
        
        extractor_class = cls._extractors[extractor_type]
        return extractor_class(config)
    
    @classmethod
    def get_available_extractors(cls) -> List[str]:
        """获取可用的特征提取器类型"""
        return list(cls._extractors.keys())


# 简化的特征组合器
class FeatureCombiner:
    """特征组合器"""
    
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        self.extractors = extractors
        self.feature_names_ = []
        
    def fit_transform(self, texts: List[str], labels: Optional[List[str]] = None) -> np.ndarray:
        """训练并提取组合特征"""
        features_list = []
        
        for extractor in self.extractors:
            features = extractor.fit_transform(texts, labels)
            features_list.append(features)
            
            # 更新特征名称
            extractor_feature_names = [f"{extractor.name}_{name}" 
                                     for name in extractor.get_feature_names()]
            self.feature_names_.extend(extractor_feature_names)
        
        if not features_list:
            return np.array([])
        
        # 水平拼接特征
        combined_features = np.hstack(features_list)
        return combined_features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """提取组合特征"""
        features_list = []
        
        for extractor in self.extractors:
            features = extractor.transform(texts)
            features_list.append(features)
        
        if not features_list:
            return np.array([])
        
        # 水平拼接特征
        combined_features = np.hstack(features_list)
        return combined_features
    
    def get_feature_names(self) -> List[str]:
        """获取组合特征名称"""
        return self.feature_names_


def get_default_extractor_config() -> Dict[str, Dict[str, Any]]:
    """获取默认的特征提取器配置"""
    return {
        'tfidf': {
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.8,
            'ngram_range': [1, 2],
            'use_idf': True,
            'sublinear_tf': True,
            'norm': 'l2'
        },
        'count': {
            'max_features': 3000,
            'min_df': 2,
            'max_df': 0.8,
            'ngram_range': [1, 2]
        }
    }