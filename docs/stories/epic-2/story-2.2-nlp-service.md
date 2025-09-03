# Story 2.2: NLP文本处理服务

## 基本信息
- **Story ID**: 2.2
- **Epic**: Epic 2 - 数据处理和智能分类微服务
- **标题**: NLP文本处理服务
- **优先级**: 高
- **状态**: 待开发
- **预估工期**: 8-10天

## 用户故事
**作为** 文本分析专员  
**我希望** 有一个专业的NLP文本处理服务  
**以便** 对历史文档进行深度语言分析，包括分词、词性标注、命名实体识别、情感分析、关键词提取等功能

## 需求描述
开发专业的NLP文本处理服务，支持古代汉语和现代汉语的语言分析，具备分词、词性标注、命名实体识别、情感分析、关键词提取、文本摘要、语义相似度计算等完整功能。

## 技术实现

### 核心技术栈
- **后端框架**: FastAPI 0.104+ (Python)
- **NLP框架**: 
  - spaCy 3.7+ (主要NLP框架)
  - jieba 0.42+ (中文分词)
  - pkuseg 0.0.25+ (北大分词)
  - LAC 2.1+ (百度词法分析)
- **深度学习**: 
  - transformers 4.35+ (预训练模型)
  - torch 2.1+ (深度学习框架)
  - sentence-transformers 2.2+ (句子嵌入)
- **中文NLP**: 
  - HanLP 2.1+ (多语言NLP)
  - ckiptagger 0.2+ (中研院标注工具)
  - opencc 1.1+ (繁简转换)
- **文本处理**: 
  - textstat 0.7+ (文本统计)
  - wordcloud 1.9+ (词云生成)
  - gensim 4.3+ (主题建模)
- **数据库**: 
  - PostgreSQL (分析结果存储)
  - Redis (缓存和队列)
  - Elasticsearch (全文搜索)
- **消息队列**: RabbitMQ 3.12+

### 数据模型设计

#### NLP任务表 (nlp_tasks)
```sql
CREATE TABLE nlp_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES datasets(id),
    text_content TEXT NOT NULL,
    text_length INTEGER,
    processing_type VARCHAR(50) NOT NULL, -- segmentation, pos_tagging, ner, sentiment, keywords, summary
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    nlp_model VARCHAR(100), -- 使用的NLP模型
    language VARCHAR(10) DEFAULT 'zh', -- zh, en, zh-classical
    config JSONB, -- 处理配置参数
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time FLOAT,
    error_message TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 分词结果表 (segmentation_results)
```sql
CREATE TABLE segmentation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES nlp_tasks(id),
    original_text TEXT NOT NULL,
    segmented_text TEXT NOT NULL, -- 分词后的文本
    word_count INTEGER,
    unique_word_count INTEGER,
    words JSONB, -- 词汇列表及其属性
    segmentation_method VARCHAR(50), -- jieba, pkuseg, lac, hanlp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 词性标注结果表 (pos_tagging_results)
```sql
CREATE TABLE pos_tagging_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES nlp_tasks(id),
    words_with_pos JSONB NOT NULL, -- [{"word": "历史", "pos": "n", "start": 0, "end": 2}]
    pos_distribution JSONB, -- 词性分布统计
    tagging_method VARCHAR(50), -- spacy, hanlp, lac
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 命名实体识别结果表 (ner_results)
```sql
CREATE TABLE ner_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES nlp_tasks(id),
    entities JSONB NOT NULL, -- [{"text": "明朝", "label": "DYNASTY", "start": 10, "end": 12, "confidence": 0.95}]
    entity_types JSONB, -- 实体类型统计
    ner_model VARCHAR(100), -- 使用的NER模型
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 情感分析结果表 (sentiment_results)
```sql
CREATE TABLE sentiment_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES nlp_tasks(id),
    sentiment_label VARCHAR(20), -- positive, negative, neutral
    sentiment_score FLOAT, -- -1.0 到 1.0
    confidence FLOAT,
    emotion_details JSONB, -- 详细情感分析
    sentiment_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 关键词提取结果表 (keyword_results)
```sql
CREATE TABLE keyword_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES nlp_tasks(id),
    keywords JSONB NOT NULL, -- [{"word": "历史", "score": 0.85, "frequency": 5}]
    extraction_method VARCHAR(50), -- tfidf, textrank, yake
    keyword_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 文本摘要结果表 (summary_results)
```sql
CREATE TABLE summary_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES nlp_tasks(id),
    original_length INTEGER,
    summary_text TEXT NOT NULL,
    summary_length INTEGER,
    compression_ratio FLOAT, -- 压缩比例
    summary_method VARCHAR(50), -- extractive, abstractive
    summary_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 服务架构

#### NLP服务主类
```python
# src/services/nlp_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional, Union
import asyncio
import spacy
import jieba
import jieba.posseg as pseg
from LAC import LAC
import hanlp
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from collections import Counter
import uuid
import time

class NLPService:
    def __init__(self):
        # 初始化各种NLP模型
        self.spacy_model = spacy.load("zh_core_web_sm")
        self.lac = LAC(mode='seg')
        self.lac_pos = LAC(mode='lac')
        
        # HanLP模型
        self.hanlp_tokenizer = hanlp.load('FINE_ELECTRA_SMALL_ZH')
        self.hanlp_ner = hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH')
        
        # 情感分析模型
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="uer/roberta-base-finetuned-chinanews-chinese",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # 句子嵌入模型
        self.sentence_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        
        # 数据库和消息队列
        self.db = DatabaseManager()
        self.message_queue = RabbitMQClient()
        
        # 初始化jieba词典
        self._init_jieba_dict()
    
    def _init_jieba_dict(self):
        """
        初始化jieba自定义词典
        """
        # 添加历史相关词汇
        historical_words = [
            '明朝', '清朝', '唐朝', '宋朝', '元朝',
            '皇帝', '大臣', '官员', '百姓',
            '史记', '资治通鉴', '二十四史'
        ]
        
        for word in historical_words:
            jieba.add_word(word, freq=1000, tag='nr')
    
    async def process_text(self, 
                          text: str, 
                          task_id: str,
                          processing_types: List[str],
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文本NLP分析
        
        Args:
            text: 输入文本
            task_id: 任务ID
            processing_types: 处理类型列表
            config: 处理配置
            
        Returns:
            NLP分析结果
        """
        try:
            start_time = time.time()
            results = {}
            
            # 更新任务状态
            await self._update_task_status(task_id, 'processing')
            
            # 文本预处理
            cleaned_text = self._preprocess_text(text, config)
            
            # 根据处理类型执行相应分析
            if 'segmentation' in processing_types:
                results['segmentation'] = await self._segment_text(cleaned_text, config)
            
            if 'pos_tagging' in processing_types:
                results['pos_tagging'] = await self._pos_tagging(cleaned_text, config)
            
            if 'ner' in processing_types:
                results['ner'] = await self._named_entity_recognition(cleaned_text, config)
            
            if 'sentiment' in processing_types:
                results['sentiment'] = await self._sentiment_analysis(cleaned_text, config)
            
            if 'keywords' in processing_types:
                results['keywords'] = await self._extract_keywords(cleaned_text, config)
            
            if 'summary' in processing_types:
                results['summary'] = await self._text_summarization(cleaned_text, config)
            
            if 'similarity' in processing_types:
                results['similarity'] = await self._compute_embeddings(cleaned_text, config)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 保存结果到数据库
            await self._save_nlp_results(task_id, results, processing_time)
            
            # 更新任务状态
            await self._update_task_status(task_id, 'completed')
            
            return {
                'success': True,
                'task_id': task_id,
                'results': results,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"NLP处理失败: {str(e)}")
            await self._update_task_status(task_id, 'failed', str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    def _preprocess_text(self, text: str, config: Dict[str, Any]) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            config: 预处理配置
            
        Returns:
            预处理后的文本
        """
        preprocessing = config.get('preprocessing', {})
        
        # 去除多余空白
        if preprocessing.get('remove_extra_whitespace', True):
            text = re.sub(r'\s+', ' ', text).strip()
        
        # 去除特殊字符
        if preprocessing.get('remove_special_chars', False):
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：""''（）【】]', '', text)
        
        # 繁简转换
        if preprocessing.get('traditional_to_simplified', True):
            import opencc
            converter = opencc.OpenCC('t2s')
            text = converter.convert(text)
        
        # 文本长度限制
        max_length = preprocessing.get('max_length', 10000)
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    async def _segment_text(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        文本分词
        
        Args:
            text: 输入文本
            config: 分词配置
            
        Returns:
            分词结果
        """
        method = config.get('segmentation_method', 'jieba')
        
        if method == 'jieba':
            # jieba分词
            words = list(jieba.cut(text, cut_all=False))
            words_with_pos = [(word, flag) for word, flag in pseg.cut(text)]
        
        elif method == 'lac':
            # LAC分词
            words = self.lac.run(text)
            words_with_pos = list(zip(*self.lac_pos.run(text)))
        
        elif method == 'hanlp':
            # HanLP分词
            words = self.hanlp_tokenizer(text)
            words_with_pos = [(word, 'n') for word in words]  # HanLP需要额外的词性标注
        
        else:
            raise ValueError(f"不支持的分词方法: {method}")
        
        # 过滤停用词
        if config.get('remove_stopwords', True):
            stopwords = self._load_stopwords()
            words = [word for word in words if word not in stopwords and len(word.strip()) > 0]
            words_with_pos = [(word, pos) for word, pos in words_with_pos 
                             if word not in stopwords and len(word.strip()) > 0]
        
        # 统计信息
        word_count = len(words)
        unique_words = list(set(words))
        unique_word_count = len(unique_words)
        
        # 词频统计
        word_freq = Counter(words)
        top_words = word_freq.most_common(20)
        
        return {
            'words': words,
            'words_with_pos': words_with_pos,
            'word_count': word_count,
            'unique_word_count': unique_word_count,
            'word_frequency': dict(word_freq),
            'top_words': top_words,
            'segmentation_method': method
        }
    
    async def _pos_tagging(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        词性标注
        
        Args:
            text: 输入文本
            config: 标注配置
            
        Returns:
            词性标注结果
        """
        method = config.get('pos_method', 'jieba')
        
        if method == 'jieba':
            words_with_pos = [(word, flag) for word, flag in pseg.cut(text)]
        
        elif method == 'lac':
            seg_result = self.lac_pos.run(text)
            words_with_pos = list(zip(seg_result[0], seg_result[1]))
        
        elif method == 'spacy':
            doc = self.spacy_model(text)
            words_with_pos = [(token.text, token.pos_) for token in doc]
        
        else:
            raise ValueError(f"不支持的词性标注方法: {method}")
        
        # 词性分布统计
        pos_counter = Counter([pos for _, pos in words_with_pos])
        
        # 构建详细结果
        detailed_results = []
        char_offset = 0
        for word, pos in words_with_pos:
            start_pos = text.find(word, char_offset)
            if start_pos != -1:
                detailed_results.append({
                    'word': word,
                    'pos': pos,
                    'start': start_pos,
                    'end': start_pos + len(word)
                })
                char_offset = start_pos + len(word)
        
        return {
            'words_with_pos': detailed_results,
            'pos_distribution': dict(pos_counter),
            'total_words': len(words_with_pos),
            'pos_method': method
        }
    
    async def _named_entity_recognition(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        命名实体识别
        
        Args:
            text: 输入文本
            config: NER配置
            
        Returns:
            命名实体识别结果
        """
        method = config.get('ner_method', 'hanlp')
        
        entities = []
        
        if method == 'hanlp':
            # 使用HanLP进行NER
            ner_results = self.hanlp_ner(text)
            for entity_info in ner_results:
                entities.append({
                    'text': entity_info[0],
                    'label': entity_info[1],
                    'start': entity_info[2],
                    'end': entity_info[3],
                    'confidence': 0.9  # HanLP没有直接提供置信度
                })
        
        elif method == 'spacy':
            # 使用spaCy进行NER
            doc = self.spacy_model(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8  # spaCy也没有直接的置信度
                })
        
        # 自定义历史实体识别
        historical_entities = self._extract_historical_entities(text)
        entities.extend(historical_entities)
        
        # 实体类型统计
        entity_types = Counter([entity['label'] for entity in entities])
        
        return {
            'entities': entities,
            'entity_count': len(entities),
            'entity_types': dict(entity_types),
            'ner_method': method
        }
    
    def _extract_historical_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        提取历史相关实体
        
        Args:
            text: 输入文本
            
        Returns:
            历史实体列表
        """
        entities = []
        
        # 朝代识别
        dynasty_pattern = r'(夏|商|周|春秋|战国|秦|汉|三国|晋|南北朝|隋|唐|五代|宋|元|明|清)朝?'
        for match in re.finditer(dynasty_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'DYNASTY',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        # 人名识别（历史人物）
        person_pattern = r'(皇帝|帝|王|公|侯|伯|子|男|太子|皇后|贵妃|丞相|宰相|尚书|御史|将军)'
        for match in re.finditer(person_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'PERSON_TITLE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.85
            })
        
        # 地名识别
        place_pattern = r'(京师|都城|长安|洛阳|开封|临安|大都|北京|南京|西安|州|府|县|郡|道)'
        for match in re.finditer(place_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'PLACE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.80
            })
        
        return entities
    
    async def _sentiment_analysis(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        情感分析
        
        Args:
            text: 输入文本
            config: 情感分析配置
            
        Returns:
            情感分析结果
        """
        try:
            # 使用预训练模型进行情感分析
            result = self.sentiment_pipeline(text)
            
            # 转换为标准格式
            sentiment_label = result[0]['label'].lower()
            confidence = result[0]['score']
            
            # 计算情感分数 (-1到1)
            if sentiment_label == 'positive':
                sentiment_score = confidence
            elif sentiment_label == 'negative':
                sentiment_score = -confidence
            else:
                sentiment_score = 0.0
            
            # 详细情感分析
            emotion_details = self._analyze_emotions(text)
            
            return {
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'emotion_details': emotion_details,
                'text_length': len(text)
            }
            
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'emotion_details': {},
                'error': str(e)
            }
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        详细情感分析
        
        Args:
            text: 输入文本
            
        Returns:
            情感详情
        """
        emotions = {
            'joy': 0.0,
            'anger': 0.0,
            'sadness': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0
        }
        
        # 基于关键词的简单情感分析
        joy_words = ['喜', '乐', '欢', '庆', '祝', '贺', '兴', '悦']
        anger_words = ['怒', '愤', '恨', '恼', '气', '火', '暴', '怒']
        sadness_words = ['悲', '哀', '愁', '忧', '伤', '痛', '泣', '哭']
        fear_words = ['怕', '恐', '惧', '畏', '惊', '慌', '吓', '骇']
        
        text_lower = text.lower()
        
        for word in joy_words:
            emotions['joy'] += text_lower.count(word) * 0.1
        for word in anger_words:
            emotions['anger'] += text_lower.count(word) * 0.1
        for word in sadness_words:
            emotions['sadness'] += text_lower.count(word) * 0.1
        for word in fear_words:
            emotions['fear'] += text_lower.count(word) * 0.1
        
        # 归一化
        max_score = max(emotions.values()) if max(emotions.values()) > 0 else 1
        emotions = {k: min(v / max_score, 1.0) for k, v in emotions.items()}
        
        return emotions
    
    async def _extract_keywords(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        关键词提取
        
        Args:
            text: 输入文本
            config: 关键词提取配置
            
        Returns:
            关键词提取结果
        """
        method = config.get('keyword_method', 'tfidf')
        max_keywords = config.get('max_keywords', 20)
        
        keywords = []
        
        if method == 'tfidf':
            keywords = self._extract_keywords_tfidf(text, max_keywords)
        elif method == 'textrank':
            keywords = self._extract_keywords_textrank(text, max_keywords)
        elif method == 'yake':
            keywords = self._extract_keywords_yake(text, max_keywords)
        
        return {
            'keywords': keywords,
            'keyword_count': len(keywords),
            'extraction_method': method,
            'max_keywords': max_keywords
        }
    
    def _extract_keywords_tfidf(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        """
        使用TF-IDF提取关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        Returns:
            关键词列表
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        import jieba
        
        # 分词
        words = list(jieba.cut(text))
        words = [word for word in words if len(word.strip()) > 1]
        
        if len(words) < 2:
            return []
        
        # 构建文档
        documents = [' '.join(words)]
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=max_keywords)
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # 构建关键词列表
            keywords = []
            for i, score in enumerate(scores):
                if score > 0:
                    keywords.append({
                        'word': feature_names[i],
                        'score': float(score),
                        'frequency': words.count(feature_names[i])
                    })
            
            # 按分数排序
            keywords.sort(key=lambda x: x['score'], reverse=True)
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"TF-IDF关键词提取失败: {str(e)}")
            return []
    
    def _extract_keywords_textrank(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        """
        使用TextRank提取关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        Returns:
            关键词列表
        """
        try:
            import jieba.analyse
            
            # 使用jieba的TextRank
            keywords_with_scores = jieba.analyse.textrank(
                text, 
                topK=max_keywords, 
                withWeight=True
            )
            
            keywords = []
            for word, score in keywords_with_scores:
                keywords.append({
                    'word': word,
                    'score': float(score),
                    'frequency': text.count(word)
                })
            
            return keywords
            
        except Exception as e:
            logger.error(f"TextRank关键词提取失败: {str(e)}")
            return []
    
    def _extract_keywords_yake(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        """
        使用YAKE提取关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        Returns:
            关键词列表
        """
        try:
            import yake
            
            # 配置YAKE
            kw_extractor = yake.KeywordExtractor(
                lan="zh",
                n=3,  # n-gram大小
                dedupLim=0.7,
                top=max_keywords
            )
            
            keywords_with_scores = kw_extractor.extract_keywords(text)
            
            keywords = []
            for score, word in keywords_with_scores:
                keywords.append({
                    'word': word,
                    'score': 1.0 - float(score),  # YAKE分数越低越好，转换为越高越好
                    'frequency': text.count(word)
                })
            
            return keywords
            
        except Exception as e:
            logger.error(f"YAKE关键词提取失败: {str(e)}")
            return []
    
    async def _text_summarization(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        文本摘要
        
        Args:
            text: 输入文本
            config: 摘要配置
            
        Returns:
            文本摘要结果
        """
        method = config.get('summary_method', 'extractive')
        max_length = config.get('max_summary_length', 200)
        compression_ratio = config.get('compression_ratio', 0.3)
        
        if method == 'extractive':
            summary = self._extractive_summarization(text, compression_ratio)
        elif method == 'abstractive':
            summary = await self._abstractive_summarization(text, max_length)
        else:
            raise ValueError(f"不支持的摘要方法: {method}")
        
        return {
            'original_length': len(text),
            'summary_text': summary,
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0,
            'summary_method': method
        }
    
    def _extractive_summarization(self, text: str, compression_ratio: float) -> str:
        """
        抽取式摘要
        
        Args:
            text: 输入文本
            compression_ratio: 压缩比例
            
        Returns:
            摘要文本
        """
        try:
            import jieba.analyse
            from collections import defaultdict
            
            # 分句
            sentences = re.split(r'[。！？；]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            if len(sentences) <= 1:
                return text
            
            # 计算句子重要性分数
            sentence_scores = defaultdict(float)
            
            # 基于关键词的句子评分
            keywords = jieba.analyse.textrank(text, topK=20, withWeight=True)
            keyword_dict = dict(keywords)
            
            for i, sentence in enumerate(sentences):
                score = 0.0
                words = list(jieba.cut(sentence))
                
                for word in words:
                    if word in keyword_dict:
                        score += keyword_dict[word]
                
                # 句子长度惩罚（太短或太长的句子分数降低）
                length_penalty = 1.0
                if len(sentence) < 10:
                    length_penalty = 0.5
                elif len(sentence) > 100:
                    length_penalty = 0.8
                
                sentence_scores[i] = score * length_penalty
            
            # 选择得分最高的句子
            num_sentences = max(1, int(len(sentences) * compression_ratio))
            top_sentences = sorted(sentence_scores.items(), 
                                 key=lambda x: x[1], reverse=True)[:num_sentences]
            
            # 按原文顺序排列
            selected_indices = sorted([idx for idx, _ in top_sentences])
            summary_sentences = [sentences[i] for i in selected_indices]
            
            return '。'.join(summary_sentences) + '。'
            
        except Exception as e:
            logger.error(f"抽取式摘要失败: {str(e)}")
            return text[:200] + '...' if len(text) > 200 else text
    
    async def _abstractive_summarization(self, text: str, max_length: int) -> str:
        """
        生成式摘要
        
        Args:
            text: 输入文本
            max_length: 最大摘要长度
            
        Returns:
            摘要文本
        """
        try:
            # 这里可以集成更高级的生成式摘要模型
            # 目前使用简化版本
            from transformers import pipeline
            
            summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 限制输入长度
            if len(text) > 1024:
                text = text[:1024]
            
            summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"生成式摘要失败: {str(e)}")
            # 回退到抽取式摘要
            return self._extractive_summarization(text, 0.3)
    
    async def _compute_embeddings(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算文本嵌入向量
        
        Args:
            text: 输入文本
            config: 嵌入配置
            
        Returns:
            文本嵌入结果
        """
        try:
            # 计算句子嵌入
            embeddings = self.sentence_model.encode([text])
            embedding_vector = embeddings[0].tolist()
            
            return {
                'embedding_vector': embedding_vector,
                'embedding_dimension': len(embedding_vector),
                'model_name': 'text2vec-base-chinese',
                'text_length': len(text)
            }
            
        except Exception as e:
            logger.error(f"文本嵌入计算失败: {str(e)}")
            return {
                'embedding_vector': [],
                'embedding_dimension': 0,
                'error': str(e)
            }
    
    def _load_stopwords(self) -> set:
        """
        加载停用词表
        
        Returns:
            停用词集合
        """
        # 中文停用词
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '它', '他', '她', '们', '这个', '那个', '什么', '怎么',
            '为什么', '哪里', '哪个', '怎样', '多少', '几个', '第一', '第二', '之后',
            '之前', '以后', '以前', '现在', '当时', '那时', '这时', '已经', '还是',
            '但是', '可是', '然而', '因为', '所以', '如果', '虽然', '尽管', '无论'
        }
        
        return stopwords
        
        # 年号识别
        era_pattern = r'(贞观|开元|天宝|永乐|康熙|乾隆|嘉庆|道光|咸丰|同治|光绪|宣统)\d*年?'
        for match in re.finditer(era_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'ERA',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # 官职识别
        official_pattern = r'(皇帝|皇后|太子|丞相|尚书|侍郎|知府|知县|县令|太守)'
        for match in re.finditer(official_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'OFFICIAL_TITLE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.85
            })
        
        return entities
    
    async def _sentiment_analysis(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        情感分析
        
        Args:
            text: 输入文本
            config: 情感分析配置
            
        Returns:
            情感分析结果
        """
        # 文本长度限制（transformers模型限制）
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        # 使用预训练模型进行情感分析
        result = self.sentiment_pipeline(text)
        
        # 转换标签
        label_mapping = {
            'POSITIVE': 'positive',
            'NEGATIVE': 'negative',
            'NEUTRAL': 'neutral'
        }
        
        sentiment_label = label_mapping.get(result[0]['label'], 'neutral')
        confidence = result[0]['score']
        
        # 计算情感分数（-1到1）
        if sentiment_label == 'positive':
            sentiment_score = confidence
        elif sentiment_label == 'negative':
            sentiment_score = -confidence
        else:
            sentiment_score = 0.0
        
        # 详细情感分析（可以扩展）
        emotion_details = {
            'joy': 0.0,
            'anger': 0.0,
            'sadness': 0.0,
            'fear': 0.0,
            'surprise': 0.0
        }
        
        return {
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'emotion_details': emotion_details
        }
    
    async def _extract_keywords(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        关键词提取
        
        Args:
            text: 输入文本
            config: 关键词提取配置
            
        Returns:
            关键词提取结果
        """
        method = config.get('keyword_method', 'tfidf')
        top_k = config.get('top_k', 10)
        
        if method == 'tfidf':
            keywords = self._extract_keywords_tfidf(text, top_k)
        elif method == 'textrank':
            keywords = self._extract_keywords_textrank(text, top_k)
        elif method == 'yake':
            keywords = self._extract_keywords_yake(text, top_k)
        else:
            raise ValueError(f"不支持的关键词提取方法: {method}")
        
        return {
            'keywords': keywords,
            'keyword_count': len(keywords),
            'extraction_method': method
        }
    
    def _extract_keywords_tfidf(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        使用TF-IDF提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回关键词数量
            
        Returns:
            关键词列表
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 分词
        words = jieba.cut(text)
        text_segmented = ' '.join(words)
        
        # TF-IDF计算
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform([text_segmented])
        
        # 获取特征名称和分数
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # 排序并获取top_k
        word_scores = list(zip(feature_names, tfidf_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        keywords = []
        for word, score in word_scores[:top_k]:
            if score > 0:
                keywords.append({
                    'word': word,
                    'score': float(score),
                    'frequency': text.count(word)
                })
        
        return keywords
    
    def _extract_keywords_textrank(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        使用TextRank提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回关键词数量
            
        Returns:
            关键词列表
        """
        import jieba.analyse
        
        # 使用jieba的TextRank
        keywords_with_scores = jieba.analyse.textrank(text, topK=top_k, withWeight=True)
        
        keywords = []
        for word, score in keywords_with_scores:
            keywords.append({
                'word': word,
                'score': float(score),
                'frequency': text.count(word)
            })
        
        return keywords
    
    def _extract_keywords_yake(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """
        使用YAKE提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回关键词数量
            
        Returns:
            关键词列表
        """
        try:
            import yake
            
            # YAKE关键词提取器
            kw_extractor = yake.KeywordExtractor(
                lan="zh",
                n=3,  # n-gram大小
                dedupLim=0.7,
                top=top_k
            )
            
            keywords_with_scores = kw_extractor.extract_keywords(text)
            
            keywords = []
            for word, score in keywords_with_scores:
                keywords.append({
                    'word': word,
                    'score': 1.0 - float(score),  # YAKE分数越低越好，转换为越高越好
                    'frequency': text.count(word)
                })
            
            return keywords
            
        except ImportError:
            # 如果YAKE未安装，回退到TF-IDF
            return self._extract_keywords_tfidf(text, top_k)
    
    async def _text_summarization(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        文本摘要
        
        Args:
            text: 输入文本
            config: 摘要配置
            
        Returns:
            文本摘要结果
        """
        method = config.get('summary_method', 'extractive')
        max_length = config.get('max_summary_length', 200)
        
        original_length = len(text)
        
        if method == 'extractive':
            summary = self._extractive_summarization(text, max_length)
        elif method == 'abstractive':
            summary = self._abstractive_summarization(text, max_length)
        else:
            raise ValueError(f"不支持的摘要方法: {method}")
        
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        return {
            'summary_text': summary,
            'original_length': original_length,
            'summary_length': summary_length,
            'compression_ratio': compression_ratio,
            'summary_method': method
        }
    
    def _extractive_summarization(self, text: str, max_length: int) -> str:
        """
        抽取式摘要
        
        Args:
            text: 输入文本
            max_length: 最大摘要长度
            
        Returns:
            摘要文本
        """
        # 简单的抽取式摘要：选择包含关键词最多的句子
        sentences = re.split(r'[。！？]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return text[:max_length]
        
        # 提取关键词
        keywords = self._extract_keywords_tfidf(text, 10)
        keyword_set = set([kw['word'] for kw in keywords])
        
        # 计算每个句子的重要性分数
        sentence_scores = []
        for sentence in sentences:
            score = sum(1 for word in jieba.cut(sentence) if word in keyword_set)
            sentence_scores.append((sentence, score))
        
        # 按分数排序
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最重要的句子组成摘要
        summary_sentences = []
        current_length = 0
        
        for sentence, score in sentence_scores:
            if current_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        return '。'.join(summary_sentences) + '。'
    
    def _abstractive_summarization(self, text: str, max_length: int) -> str:
        """
        生成式摘要（简化版本）
        
        Args:
            text: 输入文本
            max_length: 最大摘要长度
            
        Returns:
            摘要文本
        """
        # 这里可以集成更复杂的生成式摘要模型
        # 目前使用简化的方法
        return self._extractive_summarization(text, max_length)
    
    async def _compute_embeddings(self, text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算文本嵌入向量
        
        Args:
            text: 输入文本
            config: 嵌入配置
            
        Returns:
            文本嵌入结果
        """
        # 计算句子嵌入
        embeddings = self.sentence_model.encode([text])
        embedding_vector = embeddings[0].tolist()
        
        return {
            'embedding_vector': embedding_vector,
            'embedding_dimension': len(embedding_vector),
            'model_name': 'text2vec-base-chinese'
        }
    
    def _load_stopwords(self) -> set:
        """
        加载停用词表
        
        Returns:
            停用词集合
        """
        # 中文停用词
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他',
            '她', '它', '们', '这个', '那个', '什么', '怎么', '为什么',
            '\n', '\t', ' ', '\r'
        }
        
        return stopwords
```

### API控制器实现

```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from uuid import UUID
import asyncio
from datetime import datetime

from ..models.nlp_models import (
    NLPAnalysisRequest, NLPBatchRequest, NLPTaskResponse,
    NLPResultResponse, NLPTaskListResponse, NLPSimilarityRequest,
    NLPSimilarityResponse, NLPConfigRequest, ProcessingType
)
from ..services.nlp_service import NLPService
from ..dependencies import get_current_user, get_nlp_service, get_optional_user
from ..models.user import User

router = APIRouter(prefix="/api/v1/nlp", tags=["NLP"])

@router.post("/analyze", response_model=NLPResultResponse)
async def analyze_text(
    request: NLPAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    单文本NLP分析
    
    Args:
        request: NLP分析请求
        background_tasks: 后台任务
        current_user: 当前用户
        nlp_service: NLP服务实例
    
    Returns:
        NLP分析结果
    """
    try:
        # 验证文本长度
        if len(request.text) > 50000:
            raise HTTPException(
                status_code=400,
                detail="文本长度不能超过50000字符"
            )
        
        # 创建NLP任务
        task = await nlp_service.create_task(
            user_id=current_user.id,
            text=request.text,
            processing_types=request.processing_types,
            config=request.config.dict() if request.config else {}
        )
        
        # 异步处理文本
        background_tasks.add_task(
            nlp_service.process_text_async,
            task.id,
            request.text,
            request.processing_types,
            request.config.dict() if request.config else {}
        )
        
        return NLPResultResponse(
            task_id=task.id,
            status=task.status,
            message="NLP分析任务已创建",
            results=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"NLP分析失败: {str(e)}"
        )

@router.post("/batch", response_model=NLPTaskResponse)
async def batch_analyze(
    request: NLPBatchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    批量文本NLP分析
    
    Args:
        request: 批量NLP分析请求
        background_tasks: 后台任务
        current_user: 当前用户
        nlp_service: NLP服务实例
    
    Returns:
        批量任务响应
    """
    try:
        # 验证批量大小
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=400,
                detail="批量处理文本数量不能超过100个"
            )
        
        # 验证每个文本长度
        for i, text in enumerate(request.texts):
            if len(text) > 10000:
                raise HTTPException(
                    status_code=400,
                    detail=f"第{i+1}个文本长度不能超过10000字符"
                )
        
        # 创建批量任务
        batch_task = await nlp_service.create_batch_task(
            user_id=current_user.id,
            texts=request.texts,
            processing_types=request.processing_types,
            config=request.config.dict() if request.config else {}
        )
        
        # 异步处理批量文本
        background_tasks.add_task(
            nlp_service.process_batch_async,
            batch_task.id,
            request.texts,
            request.processing_types,
            request.config.dict() if request.config else {}
        )
        
        return NLPTaskResponse(
            task_id=batch_task.id,
            status=batch_task.status,
            message="批量NLP分析任务已创建",
            total_texts=len(request.texts),
            processed_texts=0
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"批量NLP分析失败: {str(e)}"
        )

@router.get("/tasks/{task_id}", response_model=NLPTaskResponse)
async def get_task_status(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    获取NLP任务状态
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        nlp_service: NLP服务实例
    
    Returns:
        任务状态信息
    """
    try:
        task = await nlp_service.get_task(task_id, current_user.id)
        if not task:
            raise HTTPException(
                status_code=404,
                detail="任务不存在"
            )
        
        return NLPTaskResponse(
            task_id=task.id,
            status=task.status,
            message=task.error_message or "任务正常",
            progress=task.progress,
            created_at=task.created_at,
            completed_at=task.completed_at,
            processing_time=task.processing_time,
            results=task.results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取任务状态失败: {str(e)}"
        )

@router.get("/results", response_model=NLPTaskListResponse)
async def get_nlp_results(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    processing_type: Optional[ProcessingType] = None,
    current_user: User = Depends(get_current_user),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    获取NLP结果列表
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 任务状态过滤
        processing_type: 处理类型过滤
        current_user: 当前用户
        nlp_service: NLP服务实例
    
    Returns:
        NLP结果列表
    """
    try:
        if page_size > 100:
            page_size = 100
        
        tasks, total = await nlp_service.get_user_tasks(
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            status=status,
            processing_type=processing_type.value if processing_type else None
        )
        
        return NLPTaskListResponse(
            tasks=tasks,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取NLP结果列表失败: {str(e)}"
        )

@router.post("/similarity", response_model=NLPSimilarityResponse)
async def calculate_similarity(
    request: NLPSimilarityRequest,
    current_user: User = Depends(get_current_user),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    计算文本相似度
    
    Args:
        request: 相似度计算请求
        current_user: 当前用户
        nlp_service: NLP服务实例
    
    Returns:
        相似度计算结果
    """
    try:
        # 验证文本长度
        if len(request.text1) > 10000 or len(request.text2) > 10000:
            raise HTTPException(
                status_code=400,
                detail="文本长度不能超过10000字符"
            )
        
        similarity_score = await nlp_service.calculate_similarity(
            text1=request.text1,
            text2=request.text2,
            method=request.method
        )
        
        return NLPSimilarityResponse(
            similarity_score=similarity_score,
            method=request.method,
            text1_length=len(request.text1),
            text2_length=len(request.text2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"相似度计算失败: {str(e)}"
        )

@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    删除NLP任务
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        nlp_service: NLP服务实例
    
    Returns:
        删除结果
    """
    try:
        success = await nlp_service.delete_task(task_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail="任务不存在或无权限删除"
            )
        
        return JSONResponse(
            content={"message": "任务删除成功"},
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除任务失败: {str(e)}"
        )

@router.post("/tasks/{task_id}/retry", response_model=NLPTaskResponse)
async def retry_task(
    task_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    重试失败的NLP任务
    
    Args:
        task_id: 任务ID
        background_tasks: 后台任务
        current_user: 当前用户
        nlp_service: NLP服务实例
    
    Returns:
        重试任务响应
    """
    try:
        task = await nlp_service.get_task(task_id, current_user.id)
        if not task:
            raise HTTPException(
                status_code=404,
                detail="任务不存在"
            )
        
        if task.status != "failed":
            raise HTTPException(
                status_code=400,
                detail="只能重试失败的任务"
            )
        
        # 重置任务状态
        await nlp_service.reset_task(task_id)
        
        # 重新处理
        background_tasks.add_task(
            nlp_service.process_text_async,
            task_id,
            task.input_text,
            task.processing_types,
            task.config
        )
        
        return NLPTaskResponse(
            task_id=task_id,
            status="processing",
            message="任务重试中"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"重试任务失败: {str(e)}"
        )
```

### 数据模型定义

```python
# src/models/nlp_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from uuid import UUID
from datetime import datetime

class ProcessingType(str, Enum):
    SEGMENTATION = "segmentation"
    POS_TAGGING = "pos_tagging"
    NER = "ner"
    SENTIMENT = "sentiment"
    KEYWORDS = "keywords"
    SUMMARY = "summary"
    SIMILARITY = "similarity"

class SegmentationMethod(str, Enum):
    JIEBA = "jieba"
    LAC = "lac"
    HANLP = "hanlp"
    PKUSEG = "pkuseg"

class KeywordMethod(str, Enum):
    TFIDF = "tfidf"
    TEXTRANK = "textrank"
    YAKE = "yake"

class SimilarityMethod(str, Enum):
    COSINE = "cosine"
    JACCARD = "jaccard"
    EUCLIDEAN = "euclidean"

class NLPPreprocessingConfig(BaseModel):
    remove_extra_whitespace: bool = True
    remove_special_chars: bool = False
    traditional_to_simplified: bool = True
    remove_stopwords: bool = True
    max_length: int = 10000

class NLPConfig(BaseModel):
    segmentation_method: SegmentationMethod = SegmentationMethod.JIEBA
    pos_method: str = "jieba"
    ner_method: str = "hanlp"
    keyword_method: KeywordMethod = KeywordMethod.TEXTRANK
    top_k: int = Field(default=10, ge=1, le=50)
    max_summary_length: int = Field(default=200, ge=50, le=1000)
    summary_method: str = "extractive"
    preprocessing: NLPPreprocessingConfig = NLPPreprocessingConfig()

class NLPAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    processing_types: List[ProcessingType]
    config: Optional[NLPConfig] = None
    
    @validator('processing_types')
    def validate_processing_types(cls, v):
        if not v:
            raise ValueError('至少需要选择一种处理类型')
        return v

class NLPBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    processing_types: List[ProcessingType]
    config: Optional[NLPConfig] = None
    
    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if len(text) > 10000:
                raise ValueError('批量处理中每个文本长度不能超过10000字符')
        return v

class NLPSimilarityRequest(BaseModel):
    text1: str = Field(..., min_length=1, max_length=10000)
    text2: str = Field(..., min_length=1, max_length=10000)
    method: SimilarityMethod = SimilarityMethod.COSINE

class NLPTaskResponse(BaseModel):
    task_id: UUID
    status: str
    message: str
    progress: Optional[float] = None
    total_texts: Optional[int] = None
    processed_texts: Optional[int] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None

class NLPResultResponse(BaseModel):
    task_id: UUID
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None

class NLPSimilarityResponse(BaseModel):
    similarity_score: float
    method: SimilarityMethod
    text1_length: int
    text2_length: int

class NLPTaskListResponse(BaseModel):
    tasks: List[NLPTaskResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

class NLPConfigRequest(BaseModel):
    config: NLPConfig
```

### 配置管理

```python
# src/config/nlp_config.py
from pydantic import BaseSettings, Field
from typing import Dict, Any, List
import os

class NLPSettings(BaseSettings):
    # 模型配置
    spacy_model: str = Field(default="zh_core_web_sm", env="NLP_SPACY_MODEL")
    hanlp_model: str = Field(default="FINE_ELECTRA_SMALL_ZH", env="NLP_HANLP_MODEL")
    sentiment_model: str = Field(
        default="uer/roberta-base-finetuned-chinanews-chinese",
        env="NLP_SENTIMENT_MODEL"
    )
    sentence_model: str = Field(
        default="shibing624/text2vec-base-chinese",
        env="NLP_SENTENCE_MODEL"
    )
    
    # 处理配置
    max_text_length: int = Field(default=50000, env="NLP_MAX_TEXT_LENGTH")
    max_batch_size: int = Field(default=100, env="NLP_MAX_BATCH_SIZE")
    max_batch_text_length: int = Field(default=10000, env="NLP_MAX_BATCH_TEXT_LENGTH")
    
    # 性能配置
    use_gpu: bool = Field(default=True, env="NLP_USE_GPU")
    gpu_device: int = Field(default=0, env="NLP_GPU_DEVICE")
    max_workers: int = Field(default=4, env="NLP_MAX_WORKERS")
    
    # 缓存配置
    enable_cache: bool = Field(default=True, env="NLP_ENABLE_CACHE")
    cache_ttl: int = Field(default=3600, env="NLP_CACHE_TTL")  # 1小时
    
    # 模型路径
    model_cache_dir: str = Field(default="./models", env="NLP_MODEL_CACHE_DIR")
    custom_dict_path: str = Field(default="./data/custom_dict.txt", env="NLP_CUSTOM_DICT_PATH")
    stopwords_path: str = Field(default="./data/stopwords.txt", env="NLP_STOPWORDS_PATH")
    
    # 默认配置
    default_segmentation_method: str = Field(default="jieba", env="NLP_DEFAULT_SEGMENTATION")
    default_keyword_method: str = Field(default="textrank", env="NLP_DEFAULT_KEYWORD")
    default_top_k: int = Field(default=10, env="NLP_DEFAULT_TOP_K")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 全局配置实例
nlp_settings = NLPSettings()
```

### 依赖注入配置

```python
# src/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime

from .services.nlp_service import NLPService
from .models.user import User
from .config.nlp_config import nlp_settings
from .database import get_db_session

security = HTTPBearer()

# NLP服务实例（单例）
_nlp_service_instance: Optional[NLPService] = None

async def get_nlp_service() -> NLPService:
    """
    获取NLP服务实例
    
    Returns:
        NLP服务实例
    """
    global _nlp_service_instance
    if _nlp_service_instance is None:
        _nlp_service_instance = NLPService()
        await _nlp_service_instance.initialize()
    return _nlp_service_instance

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db_session = Depends(get_db_session)
) -> User:
    """
    获取当前认证用户
    
    Args:
        credentials: JWT认证凭据
        db_session: 数据库会话
    
    Returns:
        当前用户对象
    
    Raises:
        HTTPException: 认证失败时抛出
    """
    try:
        # 解码JWT token
        payload = jwt.decode(
            credentials.credentials,
            nlp_settings.jwt_secret_key,
            algorithms=[nlp_settings.jwt_algorithm]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 检查token是否过期
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="认证凭据已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 从数据库获取用户信息
        user = await db_session.get(User, user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db_session = Depends(get_db_session)
) -> Optional[User]:
    """
    获取可选的当前用户（用于公开接口）
    
    Args:
        credentials: 可选的JWT认证凭据
        db_session: 数据库会话
    
    Returns:
        当前用户对象或None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials, db_session)
    except HTTPException:
        return None
```

### 应用入口点

```python
# src/main.py
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import uvicorn
from typing import Dict, Any

from .controllers.nlp_controller import router as nlp_router
from .services.nlp_service import NLPService
from .config.nlp_config import nlp_settings
from .database import init_database, close_database
from .dependencies import get_nlp_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    Args:
        app: FastAPI应用实例
    """
    # 启动时初始化
    logger.info("正在初始化NLP服务...")
    
    try:
        # 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")
        
        # 初始化NLP服务
        nlp_service = await get_nlp_service()
        logger.info("NLP服务初始化完成")
        
        # 预热模型
        await nlp_service.warmup()
        logger.info("模型预热完成")
        
        yield
        
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        raise
    
    finally:
        # 关闭时清理资源
        logger.info("正在清理资源...")
        
        try:
            # 清理NLP服务
            nlp_service = await get_nlp_service()
            await nlp_service.cleanup()
            logger.info("NLP服务清理完成")
            
            # 关闭数据库连接
            await close_database()
            logger.info("数据库连接关闭完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")

# 创建FastAPI应用
app = FastAPI(
    title="历史文本项目 - NLP服务",
    description="提供文本分词、词性标注、命名实体识别、情感分析等NLP功能",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=nlp_settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加GZip压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    记录HTTP请求日志
    
    Args:
        request: HTTP请求
        call_next: 下一个中间件
    
    Returns:
        HTTP响应
    """
    start_time = time.time()
    
    # 记录请求信息
    logger.info(
        f"请求开始: {request.method} {request.url.path} - "
        f"客户端: {request.client.host if request.client else 'unknown'}"
    )
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 记录响应信息
    logger.info(
        f"请求完成: {request.method} {request.url.path} - "
        f"状态码: {response.status_code} - 处理时间: {process_time:.3f}s"
    )
    
    # 添加处理时间到响应头
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 添加异常处理中间件
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    
    Args:
        request: HTTP请求
        exc: 异常对象
    
    Returns:
        错误响应
    """
    logger.error(f"未处理的异常: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": "服务暂时不可用，请稍后重试",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# 注册路由
app.include_router(nlp_router)

# 健康检查端点
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点
    
    Returns:
        服务健康状态
    """
    try:
        # 检查NLP服务状态
        nlp_service = await get_nlp_service()
        nlp_status = await nlp_service.health_check()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "nlp": nlp_status,
                "database": "healthy"  # 可以添加数据库健康检查
            }
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
        )

# 服务信息端点
@app.get("/info")
async def service_info() -> Dict[str, Any]:
    """
    获取服务信息
    
    Returns:
        服务配置和状态信息
    """
    try:
        nlp_service = await get_nlp_service()
        
        return {
            "service": "NLP文本处理服务",
            "version": "1.0.0",
            "description": "提供文本分词、词性标注、命名实体识别、情感分析等功能",
            "supported_features": [
                "文本分词",
                "词性标注",
                "命名实体识别",
                "情感分析",
                "关键词提取",
                "文本摘要",
                "文本相似度计算"
            ],
            "supported_languages": ["中文", "英文"],
            "models": {
                "segmentation": ["jieba", "lac", "hanlp", "pkuseg"],
                "ner": ["hanlp", "spacy"],
                "sentiment": ["transformers"],
                "keywords": ["tfidf", "textrank", "yake"]
            },
            "limits": {
                "max_text_length": nlp_settings.max_text_length,
                "max_batch_size": nlp_settings.max_batch_size,
                "max_batch_text_length": nlp_settings.max_batch_text_length
            }
        }
        
    except Exception as e:
        logger.error(f"获取服务信息失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "获取服务信息失败"}
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
```

### 前端集成

#### Vue3 NLP分析组件
```vue
<!-- components/NLPAnalyzer.vue -->
<template>
  <div class="nlp-analyzer">
    <el-card class="config-card">
      <template #header>
        <span>NLP分析配置</span>
      </template>
      
      <el-form :model="nlpConfig" label-width="120px">
        <el-form-item label="分析类型">
          <el-checkbox-group v-model="selectedTypes">
            <el-checkbox label="segmentation">分词</el-checkbox>
            <el-checkbox label="pos_tagging">词性标注</el-checkbox>
            <el-checkbox label="ner">命名实体识别</el-checkbox>
            <el-checkbox label="sentiment">情感分析</el-checkbox>
            <el-checkbox label="keywords">关键词提取</el-checkbox>
            <el-checkbox label="summary">文本摘要</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        
        <el-form-item label="分词方法">
          <el-select v-model="nlpConfig.segmentation_method">
            <el-option label="jieba" value="jieba" />
            <el-option label="LAC" value="lac" />
            <el-option label="HanLP" value="hanlp" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="关键词方法">
          <el-select v-model="nlpConfig.keyword_method">
            <el-option label="TF-IDF" value="tfidf" />
            <el-option label="TextRank" value="textrank" />
            <el-option label="YAKE" value="yake" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="关键词数量">
          <el-input-number v-model="nlpConfig.top_k" :min="5" :max="50" />
        </el-form-item>
      </el-form>
    </el-card>
    
    <el-card class="input-card">
      <template #header>
        <span>文本输入</span>
      </template>
      
      <el-input
        v-model="inputText"
        type="textarea"
        :rows="8"
        placeholder="请输入要分析的文本..."
        maxlength="10000"
        show-word-limit
      />
      
      <div class="action-buttons">
        <el-button 
          type="primary" 
          @click="analyzeText"
          :loading="analyzing"
          :disabled="!inputText.trim()"
        >
          开始分析
        </el-button>
        <el-button @click="clearText">清空</el-button>
      </div>
    </el-card>
    
    <!-- 分析结果 -->
    <el-card v-if="analysisResults" class="results-card">
      <template #header>
        <span>分析结果</span>
        <el-button 
          type="text" 
          @click="exportResults" 
          style="float: right; padding: 3px 0"
        >
          导出结果
        </el-button>
      </template>
      
      <el-tabs v-model="activeTab" type="border-card">
        <!-- 分词结果 -->
        <el-tab-pane 
          v-if="analysisResults.segmentation" 
          label="分词" 
          name="segmentation"
        >
          <div class="segmentation-result">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="总词数">
                {{ analysisResults.segmentation.word_count }}
              </el-descriptions-item>
              <el-descriptions-item label="唯一词数">
                {{ analysisResults.segmentation.unique_word_count }}
              </el-descriptions-item>
            </el-descriptions>
            
            <h4 style="margin-top: 20px;">分词结果</h4>
            <div class="word-tags">
              <el-tag 
                v-for="word in analysisResults.segmentation.words" 
                :key="word" 
                style="margin: 2px;"
              >
                {{ word }}
              </el-tag>
            </div>
            
            <h4 style="margin-top: 20px;">高频词汇</h4>
            <el-table 
              :data="analysisResults.segmentation.top_words" 
              size="small" 
              max-height="300"
            >
              <el-table-column prop="0" label="词汇" width="150" />
              <el-table-column prop="1" label="频次" width="100" />
            </el-table>
          </div>
        </el-tab-pane>
        
        <!-- 词性标注结果 -->
        <el-tab-pane 
          v-if="analysisResults.pos_tagging" 
          label="词性标注" 
          name="pos_tagging"
        >
          <el-table 
            :data="analysisResults.pos_tagging.words_with_pos" 
            size="small"
          >
            <el-table-column prop="word" label="词汇" width="150" />
            <el-table-column prop="pos" label="词性" width="100" />
            <el-table-column prop="start" label="起始位置" width="100" />
            <el-table-column prop="end" label="结束位置" width="100" />
          </el-table>
        </el-tab-pane>
        
        <!-- 命名实体识别结果 -->
        <el-tab-pane 
          v-if="analysisResults.ner" 
          label="命名实体" 
          name="ner"
        >
          <el-table 
            :data="analysisResults.ner.entities" 
            size="small"
          >
            <el-table-column prop="text" label="实体" width="150" />
            <el-table-column prop="label" label="类型" width="120" />
            <el-table-column prop="confidence" label="置信度" width="100">
              <template #default="{ row }">
                {{ (row.confidence * 100).toFixed(1) }}%
              </template>
            </el-table-column>
            <el-table-column prop="start" label="起始位置" width="100" />
            <el-table-column prop="end" label="结束位置" width="100" />
          </el-table>
        </el-tab-pane>
        
        <!-- 情感分析结果 -->
        <el-tab-pane 
          v-if="analysisResults.sentiment" 
          label="情感分析" 
          name="sentiment"
        >
          <el-descriptions :column="1" border>
            <el-descriptions-item label="情感倾向">
              <el-tag :type="getSentimentType(analysisResults.sentiment.sentiment_label)">
                {{ getSentimentText(analysisResults.sentiment.sentiment_label) }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="情感分数">
              {{ analysisResults.sentiment.sentiment_score.toFixed(3) }}
            </el-descriptions-item>
            <el-descriptions-item label="置信度">
              {{ (analysisResults.sentiment.confidence * 100).toFixed(1) }}%
            </el-descriptions-item>
          </el-descriptions>
        </el-tab-pane>
        
        <!-- 关键词结果 -->
        <el-tab-pane 
          v-if="analysisResults.keywords" 
          label="关键词" 
          name="keywords"
        >
          <el-table 
            :data="analysisResults.keywords.keywords" 
            size="small"
          >
            <el-table-column prop="word" label="关键词" width="150" />
            <el-table-column prop="score" label="重要性分数" width="120">
              <template #default="{ row }">
                {{ row.score.toFixed(3) }}
              </template>
            </el-table-column>
            <el-table-column prop="frequency" label="频次" width="100" />
          </el-table>
        </el-tab-pane>
        
        <!-- 文本摘要结果 -->
        <el-tab-pane 
          v-if="analysisResults.summary" 
          label="文本摘要" 
          name="summary"
        >
          <el-descriptions :column="2" border>
            <el-descriptions-item label="原文长度">
              {{ analysisResults.summary.original_length }}
            </el-descriptions-item>
            <el-descriptions-item label="摘要长度">
              {{ analysisResults.summary.summary_length }}
            </el-descriptions-item>
            <el-descriptions-item label="压缩比例">
              {{ (analysisResults.summary.compression_ratio * 100).toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="摘要方法">
              {{ analysisResults.summary.summary_method }}
            </el-descriptions-item>
          </el-descriptions>
          
          <h4 style="margin-top: 20px;">摘要内容</h4>
          <el-input 
            v-model="analysisResults.summary.summary_text" 
            type="textarea" 
            :rows="6" 
            readonly
          />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'

interface AnalysisResults {
  segmentation?: any
  pos_tagging?: any
  ner?: any
  sentiment?: any
  keywords?: any
  summary?: any
}

const authStore = useAuthStore()
const inputText = ref('')
const analyzing = ref(false)
const analysisResults = ref<AnalysisResults | null>(null)
const activeTab = ref('segmentation')

const selectedTypes = ref(['segmentation', 'pos_tagging', 'ner', 'sentiment', 'keywords'])

const nlpConfig = reactive({
  segmentation_method: 'jieba',
  pos_method: 'jieba',
  ner_method: 'hanlp',
  keyword_method: 'textrank',
  top_k: 10,
  preprocessing: {
    remove_stopwords: true,
    traditional_to_simplified: true
  }
})

/**
 * 分析文本
 */
const analyzeText = async () => {
  if (!inputText.value.trim()) {
    ElMessage.warning('请输入要分析的文本')
    return
  }
  
  analyzing.value = true
  
  try {
    const response = await fetch(
      `${import.meta.env.VITE_API_BASE_URL}/api/v1/nlp/analyze`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authStore.accessToken}`
        },
        body: JSON.stringify({
          text: inputText.value,
          processing_types: selectedTypes.value,
          config: nlpConfig
        })
      }
    )
    
    if (response.ok) {
      const data = await response.json()
      analysisResults.value = data.results
      
      // 设置默认激活的标签页
      if (selectedTypes.value.length > 0) {
        activeTab.value = selectedTypes.value[0]
      }
      
      ElMessage.success('文本分析完成')
    } else {
      ElMessage.error('文本分析失败')
    }
  } catch (error) {
    ElMessage.error('网络错误，请重试')
  } finally {
    analyzing.value = false
  }
}

/**
 * 清空文本
 */
const clearText = () => {
  inputText.value = ''
  analysisResults.value = null
}

/**
 * 导出结果
 */
const exportResults = () => {
  if (!analysisResults.value) return
  
  const data = JSON.stringify(analysisResults.value, null, 2)
  const blob = new Blob([data], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'nlp_analysis_results.json'
  link.click()
  URL.revokeObjectURL(url)
}

/**
 * 获取情感类型
 * @param sentiment 情感标签
 */
const getSentimentType = (sentiment: string) => {
  const typeMap = {
    'positive': 'success',
    'negative': 'danger',
    'neutral': 'info'
  }
  return typeMap[sentiment as keyof typeof typeMap] || 'info'
}

/**
 * 获取情感文本
 * @param sentiment 情感标签
 */
const getSentimentText = (sentiment: string) => {
  const textMap = {
    'positive': '积极',
    'negative': '消极',
    'neutral': '中性'
  }
  return textMap[sentiment as keyof typeof textMap] || sentiment
}
</script>

<style scoped>
.nlp-analyzer {
  padding: 20px;
}

.config-card,
.input-card,
.results-card {
  margin-bottom: 20px;
}

.action-buttons {
  margin-top: 15px;
  text-align: right;
}

.word-tags {
  margin-top: 10px;
  max-height: 200px;
  overflow-y: auto;
}

.segmentation-result h4 {
  color: #409eff;
  margin-bottom: 10px;
}
</style>
```

## 验收标准

### 功能验收
- [ ] 支持多种分词方法切换
- [ ] 词性标注准确率 > 90%
- [ ] 命名实体识别准确率 > 85%
- [ ] 情感分析准确率 > 80%
- [ ] 关键词提取相关性 > 85%
- [ ] 文本摘要质量良好
- [ ] 批量处理功能正常

### 性能验收
- [ ] 单文本处理时间 < 3秒
- [ ] 并发处理能力 > 20个任务
- [ ] 内存使用稳定
- [ ] GPU加速效果明显

### 准确性验收
- [ ] 古代汉语处理效果良好
- [ ] 历史实体识别准确
- [ ] 文本统计信息正确
- [ ] 结果格式标准化

## 业务价值
- 自动化文本语言分析，提高研究效率
- 支持古代文献的深度语言学分析
- 为AI模型提供结构化的语言特征
- 辅助历史文献的内容理解和分类

## 依赖关系
- **前置条件**: Story 2.1 (OCR文本识别服务)
- **后续依赖**: Story 2.4 (智能分类服务), Story 3.1 (AI大模型服务)

## 风险与缓解
- **风险**: 古代汉语处理准确率低
- **缓解**: 训练专门的古汉语NLP模型
- **风险**: 处理速度慢影响用户体验
- **缓解**: 模型优化 + 异步处理

## API控制器实现

### NLPController

```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
from datetime import datetime

class ProcessingType(str, Enum):
    """NLP处理类型枚举"""
    SEGMENTATION = "segmentation"
    POS_TAGGING = "pos_tagging"
    NER = "ner"
    SENTIMENT = "sentiment"
    KEYWORDS = "keywords"
    SUMMARY = "summary"
    EMBEDDINGS = "embeddings"
    ALL = "all"

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SegmentationMethod(str, Enum):
    """分词方法枚举"""
    JIEBA = "jieba"
    LAC = "lac"
    HANLP = "hanlp"

class POSMethod(str, Enum):
    """词性标注方法枚举"""
    JIEBA = "jieba"
    LAC = "lac"
    SPACY = "spacy"

class NERMethod(str, Enum):
    """命名实体识别方法枚举"""
    HANLP = "hanlp"
    SPACY = "spacy"
    CUSTOM = "custom"

class KeywordMethod(str, Enum):
    """关键词提取方法枚举"""
    TFIDF = "tfidf"
    TEXTRANK = "textrank"
    YAKE = "yake"

class SummaryMethod(str, Enum):
    """摘要方法枚举"""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"

# Pydantic模型定义
class WordInfo(BaseModel):
    """词语信息模型"""
    word: str = Field(..., description="词语")
    pos: Optional[str] = Field(None, description="词性")
    start: int = Field(..., description="起始位置")
    end: int = Field(..., description="结束位置")
    frequency: Optional[int] = Field(None, description="词频")

class EntityInfo(BaseModel):
    """实体信息模型"""
    text: str = Field(..., description="实体文本")
    label: str = Field(..., description="实体标签")
    start: int = Field(..., description="起始位置")
    end: int = Field(..., description="结束位置")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)

class KeywordInfo(BaseModel):
    """关键词信息模型"""
    word: str = Field(..., description="关键词")
    score: float = Field(..., description="重要性分数")
    frequency: int = Field(..., description="词频")

class EmotionDetails(BaseModel):
    """情感详情模型"""
    joy: float = Field(..., description="喜悦", ge=0.0, le=1.0)
    anger: float = Field(..., description="愤怒", ge=0.0, le=1.0)
    sadness: float = Field(..., description="悲伤", ge=0.0, le=1.0)
    fear: float = Field(..., description="恐惧", ge=0.0, le=1.0)
    surprise: float = Field(..., description="惊讶", ge=0.0, le=1.0)
    disgust: float = Field(..., description="厌恶", ge=0.0, le=1.0)

class ProcessingConfig(BaseModel):
    """处理配置模型"""
    # 分词配置
    segmentation_method: SegmentationMethod = Field(SegmentationMethod.JIEBA, description="分词方法")
    enable_stopwords: bool = Field(True, description="是否启用停用词过滤")
    custom_dict: Optional[List[str]] = Field(None, description="自定义词典")
    
    # 词性标注配置
    pos_method: POSMethod = Field(POSMethod.JIEBA, description="词性标注方法")
    
    # 命名实体识别配置
    ner_method: NERMethod = Field(NERMethod.HANLP, description="NER方法")
    enable_historical_entities: bool = Field(True, description="是否启用历史实体识别")
    
    # 关键词提取配置
    keyword_method: KeywordMethod = Field(KeywordMethod.TFIDF, description="关键词提取方法")
    max_keywords: int = Field(20, description="最大关键词数量", ge=1, le=100)
    
    # 摘要配置
    summary_method: SummaryMethod = Field(SummaryMethod.EXTRACTIVE, description="摘要方法")
    max_summary_length: int = Field(200, description="最大摘要长度", ge=50, le=1000)
    compression_ratio: float = Field(0.3, description="压缩比例", ge=0.1, le=0.8)
    
    # 文本预处理配置
    enable_traditional_conversion: bool = Field(True, description="是否启用繁简转换")
    max_text_length: int = Field(10000, description="最大文本长度", ge=100, le=50000)

class NLPRequest(BaseModel):
    """NLP处理请求模型"""
    text: str = Field(..., description="待处理文本", min_length=1, max_length=50000)
    processing_types: List[ProcessingType] = Field(..., description="处理类型列表")
    config: Optional[ProcessingConfig] = Field(None, description="处理配置")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('文本内容不能为空')
        return v.strip()

class NLPBatchRequest(BaseModel):
    """批量NLP处理请求模型"""
    texts: List[str] = Field(..., description="待处理文本列表", min_items=1, max_items=100)
    processing_types: List[ProcessingType] = Field(..., description="处理类型列表")
    config: Optional[ProcessingConfig] = Field(None, description="处理配置")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    @validator('texts')
    def validate_texts(cls, v):
        for text in v:
            if not text.strip():
                raise ValueError('文本内容不能为空')
        return [text.strip() for text in v]

class NLPTask(BaseModel):
    """NLP任务模型"""
    task_id: str = Field(..., description="任务ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    status: TaskStatus = Field(..., description="任务状态")
    processing_types: List[ProcessingType] = Field(..., description="处理类型")
    config: Dict[str, Any] = Field(..., description="处理配置")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误信息")
    progress: float = Field(0.0, description="处理进度", ge=0.0, le=1.0)

class SegmentationResult(BaseModel):
    """分词结果模型"""
    words: List[WordInfo] = Field(..., description="分词结果")
    word_count: int = Field(..., description="总词数")
    unique_word_count: int = Field(..., description="唯一词数")
    top_words: List[WordInfo] = Field(..., description="高频词")
    method: str = Field(..., description="分词方法")

class POSResult(BaseModel):
    """词性标注结果模型"""
    words: List[WordInfo] = Field(..., description="词性标注结果")
    pos_distribution: Dict[str, int] = Field(..., description="词性分布")
    method: str = Field(..., description="标注方法")

class NERResult(BaseModel):
    """命名实体识别结果模型"""
    entities: List[EntityInfo] = Field(..., description="实体列表")
    entity_count: int = Field(..., description="实体总数")
    entity_types: Dict[str, int] = Field(..., description="实体类型分布")
    method: str = Field(..., description="识别方法")

class SentimentResult(BaseModel):
    """情感分析结果模型"""
    sentiment_label: str = Field(..., description="情感标签")
    sentiment_score: float = Field(..., description="情感分数", ge=-1.0, le=1.0)
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    emotion_details: EmotionDetails = Field(..., description="情感详情")
    text_length: int = Field(..., description="文本长度")

class KeywordResult(BaseModel):
    """关键词提取结果模型"""
    keywords: List[KeywordInfo] = Field(..., description="关键词列表")
    keyword_count: int = Field(..., description="关键词数量")
    extraction_method: str = Field(..., description="提取方法")
    max_keywords: int = Field(..., description="最大关键词数")

class SummaryResult(BaseModel):
    """文本摘要结果模型"""
    original_length: int = Field(..., description="原文长度")
    summary_text: str = Field(..., description="摘要文本")
    summary_length: int = Field(..., description="摘要长度")
    compression_ratio: float = Field(..., description="压缩比例")
    summary_method: str = Field(..., description="摘要方法")

class EmbeddingResult(BaseModel):
    """文本嵌入结果模型"""
    embedding_vector: List[float] = Field(..., description="嵌入向量")
    embedding_dimension: int = Field(..., description="向量维度")
    model_name: str = Field(..., description="模型名称")
    text_length: int = Field(..., description="文本长度")

class NLPResult(BaseModel):
    """NLP处理结果模型"""
    result_id: str = Field(..., description="结果ID")
    task_id: str = Field(..., description="任务ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    original_text: str = Field(..., description="原始文本")
    processing_types: List[ProcessingType] = Field(..., description="处理类型")
    
    # 各种处理结果
    segmentation: Optional[SegmentationResult] = Field(None, description="分词结果")
    pos_tagging: Optional[POSResult] = Field(None, description="词性标注结果")
    ner: Optional[NERResult] = Field(None, description="命名实体识别结果")
    sentiment: Optional[SentimentResult] = Field(None, description="情感分析结果")
    keywords: Optional[KeywordResult] = Field(None, description="关键词提取结果")
    summary: Optional[SummaryResult] = Field(None, description="文本摘要结果")
    embeddings: Optional[EmbeddingResult] = Field(None, description="文本嵌入结果")
    
    processing_time: float = Field(..., description="处理时间（秒）")
    created_at: datetime = Field(..., description="创建时间")

class NLPTaskResponse(BaseModel):
    """NLP任务响应模型"""
    task: NLPTask = Field(..., description="任务信息")
    results: Optional[List[NLPResult]] = Field(None, description="处理结果")

class NLPTaskListResponse(BaseModel):
    """NLP任务列表响应模型"""
    tasks: List[NLPTask] = Field(..., description="任务列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="页码")
    page_size: int = Field(..., description="页大小")

class NLPResultResponse(BaseModel):
    """NLP结果响应模型"""
    result: NLPResult = Field(..., description="处理结果")

class NLPSettings(BaseModel):
    """NLP服务设置模型"""
    # 模型配置
    spacy_model: str = Field("zh_core_web_sm", description="spaCy模型")
    hanlp_model: str = Field("hanlp://zh", description="HanLP模型")
    sentence_model: str = Field("text2vec-base-chinese", description="句子嵌入模型")
    sentiment_model: str = Field("cardiffnlp/twitter-roberta-base-sentiment-latest", description="情感分析模型")
    
    # 处理限制
    max_text_length: int = Field(50000, description="最大文本长度")
    max_batch_size: int = Field(100, description="最大批处理大小")
    max_concurrent_tasks: int = Field(10, description="最大并发任务数")
    
    # 缓存配置
    enable_cache: bool = Field(True, description="是否启用缓存")
    cache_ttl: int = Field(3600, description="缓存TTL（秒）")
    
    # 质量控制
    min_confidence_threshold: float = Field(0.5, description="最小置信度阈值")
    enable_quality_check: bool = Field(True, description="是否启用质量检查")

# API控制器
class NLPController:
    """NLP服务API控制器"""
    
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/nlp", tags=["NLP"])
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/analyze", response_model=NLPTaskResponse)
        async def analyze_text(
            request: NLPRequest,
            background_tasks: BackgroundTasks,
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_optional_user)
        ):
            """文本分析"""
            try:
                # 创建任务
                task_id = str(uuid.uuid4())
                task = NLPTask(
                    task_id=task_id,
                    user_id=current_user.user_id if current_user else None,
                    status=TaskStatus.PENDING,
                    processing_types=request.processing_types,
                    config=request.config.dict() if request.config else {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    progress=0.0
                )
                
                # 保存任务到数据库
                await nlp_service.db_manager.save_nlp_task(task.dict())
                
                # 后台处理
                background_tasks.add_task(
                    nlp_service.process_text,
                    task_id,
                    request.text,
                    request.processing_types,
                    request.config.dict() if request.config else {}
                )
                
                return NLPTaskResponse(task=task)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"文本分析失败: {str(e)}")
        
        @self.router.post("/batch", response_model=NLPTaskResponse)
        async def batch_analyze(
            request: NLPBatchRequest,
            background_tasks: BackgroundTasks,
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_optional_user)
        ):
            """批量文本分析"""
            try:
                # 验证批量大小
                if len(request.texts) > 100:
                    raise HTTPException(status_code=400, detail="批量处理文本数量不能超过100")
                
                # 创建批量任务
                task_id = str(uuid.uuid4())
                task = NLPTask(
                    task_id=task_id,
                    user_id=current_user.user_id if current_user else None,
                    status=TaskStatus.PENDING,
                    processing_types=request.processing_types,
                    config=request.config.dict() if request.config else {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    progress=0.0
                )
                
                # 保存任务到数据库
                await nlp_service.db_manager.save_nlp_task(task.dict())
                
                # 后台批量处理
                background_tasks.add_task(
                    nlp_service.process_batch_texts,
                    task_id,
                    request.texts,
                    request.processing_types,
                    request.config.dict() if request.config else {}
                )
                
                return NLPTaskResponse(task=task)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"批量文本分析失败: {str(e)}")
        
        @self.router.get("/tasks/{task_id}", response_model=NLPTaskResponse)
        async def get_task_status(
            task_id: str,
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_optional_user)
        ):
            """获取任务状态"""
            try:
                task = await nlp_service.db_manager.get_nlp_task(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="任务不存在")
                
                # 检查权限
                if current_user and task.get('user_id') != current_user.user_id:
                    raise HTTPException(status_code=403, detail="无权访问此任务")
                
                # 获取结果
                results = None
                if task['status'] == TaskStatus.COMPLETED:
                    results = await nlp_service.db_manager.get_nlp_results_by_task(task_id)
                
                return NLPTaskResponse(
                    task=NLPTask(**task),
                    results=[NLPResult(**result) for result in results] if results else None
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")
        
        @self.router.get("/tasks", response_model=NLPTaskListResponse)
        async def list_tasks(
            page: int = Query(1, ge=1, description="页码"),
            page_size: int = Query(20, ge=1, le=100, description="页大小"),
            status: Optional[TaskStatus] = Query(None, description="任务状态过滤"),
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_optional_user)
        ):
            """获取任务列表"""
            try:
                user_id = current_user.user_id if current_user else None
                tasks, total = await nlp_service.db_manager.list_nlp_tasks(
                    user_id=user_id,
                    status=status,
                    page=page,
                    page_size=page_size
                )
                
                return NLPTaskListResponse(
                    tasks=[NLPTask(**task) for task in tasks],
                    total=total,
                    page=page,
                    page_size=page_size
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")
        
        @self.router.delete("/tasks/{task_id}")
        async def delete_task(
            task_id: str,
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_current_user)
        ):
            """删除任务"""
            try:
                task = await nlp_service.db_manager.get_nlp_task(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="任务不存在")
                
                # 检查权限
                if task.get('user_id') != current_user.user_id:
                    raise HTTPException(status_code=403, detail="无权删除此任务")
                
                # 删除任务和相关结果
                await nlp_service.db_manager.delete_nlp_task(task_id)
                
                return {"message": "任务删除成功"}
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")
        
        @self.router.post("/tasks/{task_id}/retry", response_model=NLPTaskResponse)
        async def retry_task(
            task_id: str,
            background_tasks: BackgroundTasks,
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_current_user)
        ):
            """重试失败的任务"""
            try:
                task = await nlp_service.db_manager.get_nlp_task(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="任务不存在")
                
                # 检查权限
                if task.get('user_id') != current_user.user_id:
                    raise HTTPException(status_code=403, detail="无权重试此任务")
                
                # 检查任务状态
                if task['status'] != TaskStatus.FAILED:
                    raise HTTPException(status_code=400, detail="只能重试失败的任务")
                
                # 重置任务状态
                await nlp_service.db_manager.update_nlp_task_status(
                    task_id, TaskStatus.PENDING, progress=0.0
                )
                
                # 获取原始文本（这里需要从任务配置或其他地方获取）
                # 简化处理，实际应该保存原始文本
                original_text = task.get('original_text', '')
                
                # 后台重新处理
                background_tasks.add_task(
                    nlp_service.process_text,
                    task_id,
                    original_text,
                    task['processing_types'],
                    task['config']
                )
                
                updated_task = await nlp_service.db_manager.get_nlp_task(task_id)
                return NLPTaskResponse(task=NLPTask(**updated_task))
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"重试任务失败: {str(e)}")
        
        @self.router.get("/results/{result_id}", response_model=NLPResultResponse)
        async def get_result(
            result_id: str,
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_optional_user)
        ):
            """获取NLP结果"""
            try:
                result = await nlp_service.db_manager.get_nlp_result(result_id)
                if not result:
                    raise HTTPException(status_code=404, detail="结果不存在")
                
                # 检查权限
                if current_user and result.get('user_id') != current_user.user_id:
                    raise HTTPException(status_code=403, detail="无权访问此结果")
                
                return NLPResultResponse(result=NLPResult(**result))
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")
        
        @self.router.get("/search")
        async def search_results(
            query: str = Query(..., description="搜索查询"),
            processing_type: Optional[ProcessingType] = Query(None, description="处理类型过滤"),
            page: int = Query(1, ge=1, description="页码"),
            page_size: int = Query(20, ge=1, le=100, description="页大小"),
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_optional_user)
        ):
            """搜索NLP结果"""
            try:
                user_id = current_user.user_id if current_user else None
                results, total = await nlp_service.db_manager.search_nlp_results(
                    query=query,
                    processing_type=processing_type,
                    user_id=user_id,
                    page=page,
                    page_size=page_size
                )
                
                return {
                    "results": [NLPResult(**result) for result in results],
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "query": query
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"搜索结果失败: {str(e)}")
        
        @self.router.post("/file-upload")
        async def upload_file(
            file: UploadFile = File(...),
            processing_types: List[ProcessingType] = Query(..., description="处理类型"),
            background_tasks: BackgroundTasks,
            nlp_service: NLPService = Depends(get_nlp_service),
            current_user = Depends(get_current_user)
        ):
            """上传文件进行NLP分析"""
            try:
                # 验证文件类型
                if not file.filename.endswith(('.txt', '.md', '.doc', '.docx')):
                    raise HTTPException(status_code=400, detail="不支持的文件类型")
                
                # 验证文件大小（10MB限制）
                content = await file.read()
                if len(content) > 10 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="文件大小不能超过10MB")
                
                # 提取文本内容
                if file.filename.endswith('.txt'):
                    text = content.decode('utf-8')
                else:
                    # 这里可以集成文档解析库
                    text = content.decode('utf-8')  # 简化处理
                
                # 创建任务
                task_id = str(uuid.uuid4())
                task = NLPTask(
                    task_id=task_id,
                    user_id=current_user.user_id,
                    status=TaskStatus.PENDING,
                    processing_types=processing_types,
                    config={"source_file": file.filename},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    progress=0.0
                )
                
                # 保存任务
                await nlp_service.db_manager.save_nlp_task(task.dict())
                
                # 后台处理
                background_tasks.add_task(
                    nlp_service.process_text,
                    task_id,
                    text,
                    processing_types,
                    {"source_file": file.filename}
                )
                
                return NLPTaskResponse(task=task)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"文件上传处理失败: {str(e)}")
```

## 依赖注入配置

```python
from functools import lru_cache
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime

# 全局服务实例
_nlp_service: Optional[NLPService] = None

@lru_cache()
def get_nlp_service() -> NLPService:
    """
    获取NLP服务单例实例
    
    Returns:
        NLP服务实例
    """
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService()
    return _nlp_service

# JWT认证
security = HTTPBearer(auto_error=False)

class User:
    """用户模型"""
    def __init__(self, user_id: str, username: str, email: str, roles: List[str] = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles or []

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    获取当前认证用户
    
    Args:
        credentials: JWT凭证
        
    Returns:
        用户信息
        
    Raises:
        HTTPException: 认证失败
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # 解码JWT token
        payload = jwt.decode(
            credentials.credentials,
            "your-secret-key",  # 应该从环境变量获取
            algorithms=["HS256"]
        )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭证"
            )
        
        # 检查token是否过期
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="认证凭证已过期"
            )
        
        # 构建用户对象
        user = User(
            user_id=user_id,
            username=payload.get("username", ""),
            email=payload.get("email", ""),
            roles=payload.get("roles", [])
        )
        
        return user
        
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证"
        )

async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """
    获取可选的当前用户（允许匿名访问）
    
    Args:
        credentials: JWT凭证
        
    Returns:
        用户信息或None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

async def validate_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    验证管理员用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        管理员用户
        
    Raises:
        HTTPException: 权限不足
    """
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user
```

## 应用程序入口点

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import uvicorn
from typing import Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用程序生命周期管理
    
    Args:
        app: FastAPI应用实例
    """
    # 启动时初始化
    logger.info("正在启动NLP服务...")
    
    try:
        # 初始化数据库连接
        from core.database import DatabaseManager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("数据库连接初始化完成")
        
        # 初始化NLP服务
        nlp_service = get_nlp_service()
        await nlp_service.initialize()
        logger.info("NLP服务初始化完成")
        
        # 预热模型
        logger.info("正在预热NLP模型...")
        test_text = "这是一个测试文本，用于预热模型。"
        await nlp_service.process_text(
            "warmup-task",
            test_text,
            ["segmentation"],
            {}
        )
        logger.info("模型预热完成")
        
        logger.info("NLP服务启动完成")
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭NLP服务...")
    
    try:
        # 清理NLP服务资源
        nlp_service = get_nlp_service()
        await nlp_service.cleanup()
        logger.info("NLP服务资源清理完成")
        
        # 关闭数据库连接
        db_manager = DatabaseManager()
        await db_manager.close()
        logger.info("数据库连接关闭完成")
        
        logger.info("NLP服务关闭完成")
        
    except Exception as e:
        logger.error(f"服务关闭时出错: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="历史文本项目 - NLP服务",
    description="提供文本分词、词性标注、命名实体识别、情感分析、关键词提取、文本摘要等NLP功能",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加Gzip压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    记录HTTP请求日志
    
    Args:
        request: HTTP请求
        call_next: 下一个中间件
        
    Returns:
        HTTP响应
    """
    start_time = time.time()
    
    # 记录请求信息
    logger.info(
        f"请求开始: {request.method} {request.url} - "
        f"客户端: {request.client.host if request.client else 'unknown'}"
    )
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 记录响应信息
    logger.info(
        f"请求完成: {request.method} {request.url} - "
        f"状态码: {response.status_code} - "
        f"处理时间: {process_time:.3f}s"
    )
    
    # 添加处理时间到响应头
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    
    Args:
        request: HTTP请求
        exc: 异常对象
        
    Returns:
        错误响应
    """
    logger.error(
        f"未处理的异常: {request.method} {request.url} - "
        f"异常类型: {type(exc).__name__} - "
        f"异常信息: {str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": "服务暂时不可用，请稍后重试",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# 注册路由
nlp_controller = NLPController()
app.include_router(nlp_controller.router)

# 健康检查端点
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点
    
    Returns:
        服务健康状态
    """
    try:
        # 检查NLP服务状态
        nlp_service = get_nlp_service()
        service_status = await nlp_service.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "nlp-service",
            "version": "1.0.0",
            "details": service_status
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "nlp-service",
                "error": str(e)
            }
        )

# 服务信息端点
@app.get("/info")
async def service_info() -> Dict[str, Any]:
    """
    获取服务信息
    
    Returns:
        服务详细信息
    """
    return {
        "service": "nlp-service",
        "version": "1.0.0",
        "description": "历史文本项目NLP处理服务",
        "features": [
            "文本分词",
            "词性标注",
            "命名实体识别",
            "情感分析",
            "关键词提取",
            "文本摘要",
            "文本嵌入"
        ],
        "supported_languages": ["zh", "zh-cn"],
        "api_version": "v1",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
```

## 开发任务分解
1. NLP框架集成和配置 (2天)
2. 分词和词性标注功能开发 (2天)
3. 命名实体识别功能开发 (2天)
4. 情感分析和关键词提取 (2天)
5. 文本摘要功能开发 (1天)
6. 前端NLP组件开发 (2天)
7. 性能优化和测试 (1天)