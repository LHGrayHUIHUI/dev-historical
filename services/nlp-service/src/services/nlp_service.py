"""
NLP核心服务类
无状态NLP文本处理算法，专注于语言分析计算
数据存储通过storage-service完成
"""

import asyncio
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import uuid
from loguru import logger

# NLP核心库
import spacy
import jieba
import jieba.posseg as pseg
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import opencc
from textstat import flesch_reading_ease
from gensim.summarization import keywords, summarize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config.settings import settings, get_nlp_engine_config
from ..schemas.nlp_schemas import *


class NLPService:
    """NLP核心服务类（无状态架构）"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.is_initialized = False
        
        # 配置信息
        self.engine_config = get_nlp_engine_config()
        
        # 繁简转换器
        self.t2s_converter = opencc.OpenCC('t2s')  # 繁体转简体
        self.s2t_converter = opencc.OpenCC('s2t')  # 简体转繁体
        
        # 文本缓存（本地缓存）
        self._cache = {}
        self._cache_max_size = settings.cache_max_size
        
        logger.info("NLP服务初始化开始...")
    
    async def initialize_models(self):
        """异步初始化NLP模型"""
        if self.is_initialized:
            return
        
        try:
            # 并行初始化模型以减少启动时间
            await asyncio.gather(
                self._init_spacy_model(),
                self._init_transformers_models(),
                self._init_sentence_model(),
                self._init_jieba_dict()
            )
            
            self.is_initialized = True
            logger.info("所有NLP模型初始化完成")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise
    
    async def _init_spacy_model(self):
        """初始化spaCy模型"""
        try:
            if settings.spacy_model and settings.enable_pos_tagging:
                self.models['spacy'] = spacy.load(settings.spacy_model)
                logger.info(f"spaCy模型加载完成: {settings.spacy_model}")
        except Exception as e:
            logger.warning(f"spaCy模型加载失败: {str(e)}，将使用其他分词方法")
    
    async def _init_transformers_models(self):
        """初始化Transformers模型"""
        try:
            # 情感分析模型
            if settings.enable_sentiment_analysis:
                device = self.engine_config['transformers']['device']
                self.models['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model=settings.sentiment_model,
                    device=device
                )
                logger.info(f"情感分析模型加载完成: {settings.sentiment_model}")
                
        except Exception as e:
            logger.warning(f"Transformers模型加载失败: {str(e)}")
    
    async def _init_sentence_model(self):
        """初始化句子嵌入模型"""
        try:
            if settings.enable_text_similarity:
                self.models['sentence_transformer'] = SentenceTransformer(settings.sentence_model)
                logger.info(f"句子嵌入模型加载完成: {settings.sentence_model}")
        except Exception as e:
            logger.warning(f"句子嵌入模型加载失败: {str(e)}")
    
    async def _init_jieba_dict(self):
        """初始化jieba词典"""
        try:
            # 启用并行分词
            if settings.jieba_enable_parallel:
                jieba.enable_parallel(settings.jieba_parallel_workers)
            
            # 加载自定义词典
            if settings.jieba_dict_path:
                jieba.load_userdict(settings.jieba_dict_path)
            
            logger.info("jieba分词器初始化完成")
        except Exception as e:
            logger.warning(f"jieba初始化失败: {str(e)}")
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取结果"""
        if not settings.enable_cache:
            return None
        return self._cache.get(key)
    
    def _set_cache(self, key: str, value: Any):
        """设置缓存"""
        if not settings.enable_cache:
            return
        
        # 简单的LRU缓存实现
        if len(self._cache) >= self._cache_max_size:
            # 删除最旧的条目
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    def _preprocess_text(self, text: str, language: str = "zh") -> str:
        """文本预处理"""
        # 基础清理
        text = re.sub(r'\\s+', ' ', text)  # 标准化空白字符
        text = text.strip()
        
        # 繁简转换（如果需要）
        if language == "zh" and any(ord(char) > 0x4e00 and ord(char) < 0x9fff for char in text):
            # 检测是否包含繁体字，如果有则转换为简体
            text = self.t2s_converter.convert(text)
        
        return text
    
    # ============ 分词功能 ============
    
    async def segment_text(
        self,
        text: str,
        method: str = "jieba",
        language: str = "zh",
        config: Optional[Dict] = None
    ) -> SegmentationResult:
        """文本分词"""
        config = config or {}
        
        # 缓存键
        cache_key = f"seg_{method}_{hash(text)}_{language}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        # 预处理文本
        processed_text = self._preprocess_text(text, language)
        
        try:
            if method == "jieba":
                result = await self._segment_with_jieba(processed_text, config)
            elif method == "spacy" and 'spacy' in self.models:
                result = await self._segment_with_spacy(processed_text, config)
            else:
                # 默认使用jieba
                result = await self._segment_with_jieba(processed_text, config)
            
            result.method = method
            
            # 缓存结果
            self._set_cache(cache_key, result)
            
            processing_time = time.time() - start_time
            logger.info(f"分词完成，方法: {method}, 耗时: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"分词失败: {str(e)}")
            raise
    
    async def _segment_with_jieba(self, text: str, config: Dict) -> SegmentationResult:
        """使用jieba分词"""
        # 执行分词
        words_info = []
        segmented_parts = []
        current_pos = 0
        
        # 使用jieba.posseg进行分词和词性标注
        seg_list = pseg.cut(text, HMM=config.get('hmm', True))
        
        for word, pos in seg_list:
            if word.strip():  # 跳过空白词
                start_pos = text.find(word, current_pos)
                end_pos = start_pos + len(word)
                
                words_info.append(WordInfo(
                    text=word,
                    start=start_pos,
                    end=end_pos,
                    pos=pos
                ))
                
                segmented_parts.append(word)
                current_pos = end_pos
        
        return SegmentationResult(
            original_text=text,
            segmented_text=" / ".join(segmented_parts),
            words=words_info,
            word_count=len(words_info),
            unique_word_count=len(set(w.text for w in words_info)),
            method="jieba"
        )
    
    async def _segment_with_spacy(self, text: str, config: Dict) -> SegmentationResult:
        """使用spaCy分词"""
        doc = self.models['spacy'](text)
        
        words_info = []
        segmented_parts = []
        
        for token in doc:
            if not token.is_space and not token.is_punct:
                words_info.append(WordInfo(
                    text=token.text,
                    start=token.idx,
                    end=token.idx + len(token.text),
                    pos=token.pos_
                ))
                segmented_parts.append(token.text)
        
        return SegmentationResult(
            original_text=text,
            segmented_text=" / ".join(segmented_parts),
            words=words_info,
            word_count=len(words_info),
            unique_word_count=len(set(w.text for w in words_info)),
            method="spacy"
        )
    
    # ============ 词性标注功能 ============
    
    async def pos_tagging(
        self,
        text: str,
        method: str = "jieba",
        language: str = "zh",
        config: Optional[Dict] = None
    ) -> PosTaggingResult:
        """词性标注"""
        config = config or {}
        
        # 先进行分词，然后标注词性
        seg_result = await self.segment_text(text, method, language, config)
        
        # 统计词性分布
        pos_distribution = Counter(word.pos for word in seg_result.words if word.pos)
        
        return PosTaggingResult(
            words_with_pos=seg_result.words,
            pos_distribution=dict(pos_distribution),
            method=method
        )
    
    # ============ 命名实体识别功能 ============
    
    async def named_entity_recognition(
        self,
        text: str,
        model: str = "spacy",
        language: str = "zh",
        config: Optional[Dict] = None
    ) -> NERResult:
        """命名实体识别"""
        config = config or {}
        
        # 缓存键
        cache_key = f"ner_{model}_{hash(text)}_{language}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            if model == "spacy" and 'spacy' in self.models:
                result = await self._ner_with_spacy(text, config)
            else:
                # 简单的规则基础NER
                result = await self._ner_with_rules(text, config)
            
            result.model = model
            
            # 缓存结果
            self._set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"NER处理失败: {str(e)}")
            raise
    
    async def _ner_with_spacy(self, text: str, config: Dict) -> NERResult:
        """使用spaCy进行NER"""
        doc = self.models['spacy'](text)
        
        entities = []
        entity_types = Counter()
        
        for ent in doc.ents:
            entities.append(EntityInfo(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0  # spaCy不提供置信度
            ))
            entity_types[ent.label_] += 1
        
        return NERResult(
            entities=entities,
            entity_types=dict(entity_types),
            model="spacy"
        )
    
    async def _ner_with_rules(self, text: str, config: Dict) -> NERResult:
        """基于规则的简单NER"""
        entities = []
        entity_types = Counter()
        
        # 简单的正则表达式规则
        patterns = {
            "DATE": r'\\d{4}年\\d{1,2}月\\d{1,2}日|\\d{4}-\\d{1,2}-\\d{1,2}',
            "PERSON": r'[王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏锺汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文][\\u4e00-\\u9fff]{1,3}',
            "LOCATION": r'[\\u4e00-\\u9fff]{2,8}[省市县区镇村街道路]',
            "ORGANIZATION": r'[\\u4e00-\\u9fff]{2,10}[公司企业集团有限责任股份]',
        }
        
        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(EntityInfo(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8  # 规则匹配的默认置信度
                ))
                entity_types[label] += 1
        
        return NERResult(
            entities=entities,
            entity_types=dict(entity_types),
            model="rules"
        )
    
    # ============ 情感分析功能 ============
    
    async def sentiment_analysis(
        self,
        text: str,
        model: str = "transformers",
        language: str = "zh",
        config: Optional[Dict] = None
    ) -> SentimentResult:
        """情感分析"""
        config = config or {}
        
        # 缓存键
        cache_key = f"sentiment_{model}_{hash(text)}_{language}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            if model == "transformers" and 'sentiment' in self.models:
                result = await self._sentiment_with_transformers(text, config)
            else:
                # 简单的词典基础情感分析
                result = await self._sentiment_with_lexicon(text, config)
            
            result.model = model
            
            # 缓存结果
            self._set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            raise
    
    async def _sentiment_with_transformers(self, text: str, config: Dict) -> SentimentResult:
        """使用Transformers进行情感分析"""
        # 限制文本长度以避免模型限制
        if len(text) > 512:
            text = text[:512]
        
        result = self.models['sentiment'](text)
        sentiment_data = result[0]
        
        # 映射标签
        label_mapping = {
            'POSITIVE': SentimentLabel.POSITIVE,
            'NEGATIVE': SentimentLabel.NEGATIVE,
            'NEUTRAL': SentimentLabel.NEUTRAL,
            'LABEL_1': SentimentLabel.POSITIVE,  # 有些模型使用数字标签
            'LABEL_0': SentimentLabel.NEGATIVE
        }
        
        sentiment_label = label_mapping.get(sentiment_data['label'], SentimentLabel.NEUTRAL)
        sentiment_score = sentiment_data['score']
        
        # 将置信度转换为-1到1的情感分数
        if sentiment_label == SentimentLabel.POSITIVE:
            final_score = sentiment_score
        elif sentiment_label == SentimentLabel.NEGATIVE:
            final_score = -sentiment_score
        else:
            final_score = 0.0
        
        return SentimentResult(
            sentiment=SentimentInfo(
                label=sentiment_label,
                score=final_score,
                confidence=sentiment_score,
                emotions={"positive": sentiment_score if sentiment_label == SentimentLabel.POSITIVE else 1-sentiment_score,
                         "negative": sentiment_score if sentiment_label == SentimentLabel.NEGATIVE else 1-sentiment_score,
                         "neutral": sentiment_score if sentiment_label == SentimentLabel.NEUTRAL else 0.5}
            ),
            model="transformers"
        )
    
    async def _sentiment_with_lexicon(self, text: str, config: Dict) -> SentimentResult:
        """基于词典的简单情感分析"""
        # 简单的情感词典（示例）
        positive_words = ['好', '棒', '优秀', '杰出', '完美', '喜欢', '爱', '赞', '满意', '高兴']
        negative_words = ['坏', '差', '糟糕', '讨厌', '恨', '失望', '愤怒', '悲伤', '痛苦', '难过']
        
        # 分词
        seg_result = await self.segment_text(text, "jieba")
        words = [word.text for word in seg_result.words]
        
        # 计算情感分数
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if positive_count > negative_count:
            sentiment_label = SentimentLabel.POSITIVE
            score = (positive_count - negative_count) / total_words
        elif negative_count > positive_count:
            sentiment_label = SentimentLabel.NEGATIVE  
            score = -(negative_count - positive_count) / total_words
        else:
            sentiment_label = SentimentLabel.NEUTRAL
            score = 0.0
        
        confidence = abs(score) if abs(score) > 0 else 0.5
        
        return SentimentResult(
            sentiment=SentimentInfo(
                label=sentiment_label,
                score=score,
                confidence=confidence
            ),
            model="lexicon"
        )
    
    # ============ 关键词提取功能 ============
    
    async def extract_keywords(
        self,
        text: str,
        method: str = "textrank",
        top_k: int = 20,
        language: str = "zh",
        config: Optional[Dict] = None
    ) -> KeywordResult:
        """关键词提取"""
        config = config or {}
        
        try:
            if method == "textrank":
                result = await self._keywords_with_textrank(text, top_k, config)
            elif method == "tfidf":
                result = await self._keywords_with_tfidf(text, top_k, config)
            else:
                # 默认使用词频方法
                result = await self._keywords_with_frequency(text, top_k, config)
            
            result.method = method
            return result
            
        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            raise
    
    async def _keywords_with_textrank(self, text: str, top_k: int, config: Dict) -> KeywordResult:
        """使用TextRank算法提取关键词"""
        try:
            # 使用gensim的keywords函数
            keyword_scores = keywords(text, ratio=0.3, words=top_k, lemmatize=True, scores=True)
            
            keyword_list = []
            for word, score in keyword_scores:
                # 计算词频
                frequency = text.count(word)
                keyword_list.append(KeywordInfo(
                    word=word,
                    score=float(score),
                    frequency=frequency
                ))
            
            return KeywordResult(
                keywords=keyword_list[:top_k],
                method="textrank"
            )
        except:
            # 如果TextRank失败，降级到词频方法
            return await self._keywords_with_frequency(text, top_k, config)
    
    async def _keywords_with_tfidf(self, text: str, top_k: int, config: Dict) -> KeywordResult:
        """使用TF-IDF提取关键词"""
        # 分词
        seg_result = await self.segment_text(text, "jieba")
        words = [word.text for word in seg_result.words if len(word.text) > 1]
        
        if not words:
            return KeywordResult(keywords=[], method="tfidf")
        
        # 构建文档
        doc_text = " ".join(words)
        
        # TF-IDF计算
        vectorizer = TfidfVectorizer(max_features=top_k * 2, token_pattern=r'(?u)\\b\\w+\\b')
        tfidf_matrix = vectorizer.fit_transform([doc_text])
        
        # 获取特征名和分数
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # 排序并获取top-k
        word_scores = list(zip(feature_names, tfidf_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        keyword_list = []
        for word, score in word_scores[:top_k]:
            frequency = text.count(word)
            keyword_list.append(KeywordInfo(
                word=word,
                score=float(score),
                frequency=frequency
            ))
        
        return KeywordResult(
            keywords=keyword_list,
            method="tfidf"
        )
    
    async def _keywords_with_frequency(self, text: str, top_k: int, config: Dict) -> KeywordResult:
        """基于词频的关键词提取"""
        # 分词
        seg_result = await self.segment_text(text, "jieba")
        
        # 过滤停用词和短词
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        words = [word.text for word in seg_result.words 
                if len(word.text) > 1 and word.text not in stopwords]
        
        # 计算词频
        word_freq = Counter(words)
        
        # 转换为KeywordInfo格式
        keyword_list = []
        for word, freq in word_freq.most_common(top_k):
            # 简单的重要性分数（基于频率）
            score = freq / len(words)
            keyword_list.append(KeywordInfo(
                word=word,
                score=score,
                frequency=freq
            ))
        
        return KeywordResult(
            keywords=keyword_list,
            method="frequency"
        )
    
    # ============ 文本摘要功能 ============
    
    async def text_summarization(
        self,
        text: str,
        method: str = "extractive",
        max_sentences: int = 5,
        compression_ratio: float = 0.3,
        language: str = "zh",
        config: Optional[Dict] = None
    ) -> SummaryResult:
        """文本摘要"""
        config = config or {}
        
        try:
            if method == "extractive":
                result = await self._extractive_summary(text, max_sentences, compression_ratio, config)
            else:
                # 默认抽取式摘要
                result = await self._extractive_summary(text, max_sentences, compression_ratio, config)
            
            result.method = method
            return result
            
        except Exception as e:
            logger.error(f"文本摘要失败: {str(e)}")
            raise
    
    async def _extractive_summary(
        self, 
        text: str, 
        max_sentences: int, 
        compression_ratio: float, 
        config: Dict
    ) -> SummaryResult:
        """抽取式摘要"""
        try:
            # 使用gensim的summarize函数
            summary_text = summarize(text, ratio=compression_ratio, word_count=None)
            
            if not summary_text:
                # 如果gensim失败，使用简单的句子选择
                sentences = re.split(r'[。！？]', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                # 选择前几个句子作为摘要
                selected_sentences = sentences[:min(max_sentences, len(sentences))]
                summary_text = '。'.join(selected_sentences) + '。'
            
        except:
            # 降级方案：选择前几个句子
            sentences = re.split(r'[。！？]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            selected_sentences = sentences[:min(max_sentences, len(sentences))]
            summary_text = '。'.join(selected_sentences) + '。'
        
        return SummaryResult(
            original_length=len(text),
            summary_text=summary_text,
            summary_length=len(summary_text),
            compression_ratio=len(summary_text) / len(text) if len(text) > 0 else 0,
            method="extractive"
        )
    
    # ============ 文本相似度计算功能 ============
    
    async def text_similarity(
        self,
        text1: str,
        text2: str,
        method: str = "sentence_transformer",
        language: str = "zh",
        config: Optional[Dict] = None
    ) -> TextSimilarityResult:
        """文本相似度计算"""
        config = config or {}
        
        try:
            if method == "sentence_transformer" and 'sentence_transformer' in self.models:
                result = await self._similarity_with_sentence_transformer(text1, text2, config)
            elif method == "tfidf":
                result = await self._similarity_with_tfidf(text1, text2, config)
            else:
                # 默认使用TF-IDF
                result = await self._similarity_with_tfidf(text1, text2, config)
            
            result.method = method
            return result
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            raise
    
    async def _similarity_with_sentence_transformer(
        self, 
        text1: str, 
        text2: str, 
        config: Dict
    ) -> TextSimilarityResult:
        """使用句子嵌入计算相似度"""
        # 获取句子嵌入
        embeddings = self.models['sentence_transformer'].encode([text1, text2])
        
        # 计算余弦相似度
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return TextSimilarityResult(
            text1=text1,
            text2=text2,
            similarity_score=float(similarity),
            method="sentence_transformer"
        )
    
    async def _similarity_with_tfidf(self, text1: str, text2: str, config: Dict) -> TextSimilarityResult:
        """使用TF-IDF计算相似度"""
        # 分词
        seg1 = await self.segment_text(text1, "jieba")
        seg2 = await self.segment_text(text2, "jieba")
        
        words1 = " ".join([word.text for word in seg1.words])
        words2 = " ".join([word.text for word in seg2.words])
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b')
        tfidf_matrix = vectorizer.fit_transform([words1, words2])
        
        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return TextSimilarityResult(
            text1=text1,
            text2=text2,
            similarity_score=float(similarity),
            method="tfidf"
        )
    
    # ============ 批量处理功能 ============
    
    async def batch_process(
        self,
        texts: List[str],
        processing_type: ProcessingType,
        config: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """批量文本处理"""
        config = config or {}
        
        # 并发处理以提高效率
        semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
        
        async def process_single_text(text: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    if processing_type == ProcessingType.SEGMENTATION:
                        result = await self.segment_text(text, **config)
                    elif processing_type == ProcessingType.POS_TAGGING:
                        result = await self.pos_tagging(text, **config)
                    elif processing_type == ProcessingType.NER:
                        result = await self.named_entity_recognition(text, **config)
                    elif processing_type == ProcessingType.SENTIMENT:
                        result = await self.sentiment_analysis(text, **config)
                    elif processing_type == ProcessingType.KEYWORDS:
                        result = await self.extract_keywords(text, **config)
                    elif processing_type == ProcessingType.SUMMARY:
                        result = await self.text_summarization(text, **config)
                    else:
                        raise ValueError(f"不支持的处理类型: {processing_type}")
                    
                    return {"success": True, "result": result.dict()}
                    
                except Exception as e:
                    logger.error(f"单个文本处理失败: {str(e)}")
                    return {"success": False, "error": str(e)}
        
        # 并发处理所有文本
        results = await asyncio.gather(
            *[process_single_text(text) for text in texts],
            return_exceptions=True
        )
        
        return results
    
    def get_available_engines(self) -> List[NLPEngineInfo]:
        """获取可用的NLP引擎信息"""
        engines = []
        
        # jieba引擎
        engines.append(NLPEngineInfo(
            name="jieba",
            version="0.42.1",
            supported_languages=["zh"],
            supported_functions=["segmentation", "pos_tagging"],
            description="结巴中文分词"
        ))
        
        # spaCy引擎
        if 'spacy' in self.models:
            engines.append(NLPEngineInfo(
                name="spacy",
                version="3.7.2",
                supported_languages=["zh", "en"],
                supported_functions=["segmentation", "pos_tagging", "ner"],
                description="工业级自然语言处理库"
            ))
        
        # Transformers引擎
        if 'sentiment' in self.models:
            engines.append(NLPEngineInfo(
                name="transformers",
                version="4.35.2",
                supported_languages=["zh", "en"],
                supported_functions=["sentiment"],
                description="预训练深度学习模型"
            ))
        
        # 句子嵌入引擎
        if 'sentence_transformer' in self.models:
            engines.append(NLPEngineInfo(
                name="sentence_transformer",
                version="2.2.2",
                supported_languages=["zh", "en"],
                supported_functions=["similarity"],
                description="句子嵌入和相似度计算"
            ))
        
        return engines
    
    async def cleanup(self):
        """清理资源"""
        logger.info("NLP服务清理资源...")
        
        # 清理模型
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'close'):
                    await model.close()
                elif hasattr(model, '__del__'):
                    del model
            except:
                pass
        
        self.models.clear()
        self.tokenizers.clear()
        self._cache.clear()
        
        # 禁用jieba并行
        if settings.jieba_enable_parallel:
            jieba.disable_parallel()
        
        logger.info("NLP服务资源清理完成")