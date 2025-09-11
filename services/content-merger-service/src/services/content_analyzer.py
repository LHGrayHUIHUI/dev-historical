"""
内容分析器

该模块负责分析文本内容的特征，包括主题提取、实体识别、
时间信息分析、关键点提取等功能。
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import jieba
import jieba.analyse
from datetime import datetime

from ..models.merger_models import (
    ContentItem, ContentAnalysis, Topic, Entity, TemporalInfo,
    ContentAnalysisError
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """内容分析器核心类"""
    
    def __init__(self):
        self._initialize_nlp_tools()
    
    def _initialize_nlp_tools(self):
        """初始化NLP工具"""
        try:
            # 初始化jieba
            jieba.initialize()
            
            # 设置用户词典（如果需要）
            self._load_custom_dictionaries()
            
            logger.info("Content analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize content analyzer: {str(e)}")
            raise ContentAnalysisError(f"初始化内容分析器失败: {str(e)}")
    
    def _load_custom_dictionaries(self):
        """加载自定义词典"""
        # 历史人物词典
        historical_figures = [
            "朱元璋", "王安石", "司马迁", "诸葛亮", "李白", "杜甫",
            "孔子", "孟子", "老子", "庄子", "韩愈", "欧阳修"
        ]
        
        # 历史地名词典
        historical_places = [
            "长安", "洛阳", "开封", "临安", "金陵", "燕京", 
            "大都", "应天府", "濠州", "钟离"
        ]
        
        # 历史事件词典
        historical_events = [
            "变法", "改革", "起义", "战争", "政变", "革命",
            "统一", "分裂", "迁都", "建都"
        ]
        
        # 添加到jieba词典
        for word in historical_figures + historical_places + historical_events:
            jieba.add_word(word)
    
    async def analyze_content(self, content: ContentItem) -> Dict[str, Any]:
        """
        分析单个内容的特征
        
        Args:
            content: 要分析的内容项
            
        Returns:
            分析结果字典
        """
        try:
            logger.debug(f"Analyzing content: {content.id}")
            
            # 并行执行各种分析
            tasks = [
                self._extract_topics(content.content),
                self._extract_entities(content.content),
                self._extract_temporal_info(content.content),
                self._extract_key_points(content.content),
                self._analyze_sentiment(content.content),
                self._calculate_complexity(content.content)
            ]
            
            results = await asyncio.gather(*tasks)
            
            analysis_result = {
                'topics': results[0],
                'entities': results[1],
                'temporal_info': results[2],
                'key_points': results[3],
                'sentiment_score': results[4],
                'complexity_score': results[5]
            }
            
            logger.debug(f"Content analysis completed for: {content.id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Content analysis failed for {content.id}: {str(e)}")
            raise ContentAnalysisError(f"内容分析失败: {str(e)}")
    
    async def _extract_topics(self, text: str) -> List[Dict[str, Any]]:
        """提取主题"""
        try:
            # 使用jieba的TF-IDF算法提取关键词
            keywords = jieba.analyse.extract_tags(
                text, 
                topK=10, 
                withWeight=True,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz')  # 只保留名词类
            )
            
            # 基于关键词构建主题
            topics = []
            
            # 历史主题分类
            topic_categories = {
                '政治': ['政治', '政策', '改革', '变法', '政府', '朝廷', '皇帝', '官员'],
                '经济': ['经济', '商业', '农业', '税收', '财政', '贸易', '货币'],
                '文化': ['文化', '教育', '科举', '学术', '艺术', '文学', '诗歌'],
                '军事': ['军事', '战争', '军队', '将军', '战役', '武器', '防御'],
                '社会': ['社会', '民众', '百姓', '人口', '阶级', '风俗', '习惯'],
                '地理': ['地理', '山川', '河流', '城市', '疆域', '交通', '地形']
            }
            
            # 计算主题相关性
            for category, category_words in topic_categories.items():
                relevance = 0.0
                matched_keywords = []
                
                for keyword, weight in keywords:
                    if any(word in keyword for word in category_words):
                        relevance += weight
                        matched_keywords.append(keyword)
                
                if relevance > 0.1:  # 设置阈值
                    topics.append({
                        'topic': category,
                        'relevance': min(relevance, 1.0),
                        'keywords': matched_keywords[:5]  # 最多5个关键词
                    })
            
            # 按相关性排序
            topics.sort(key=lambda x: x['relevance'], reverse=True)
            
            return topics[:5]  # 返回最多5个主题
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            return []
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取命名实体"""
        try:
            entities = []
            
            # 人名实体识别
            person_patterns = [
                r'[一-龥]{2,4}(?:帝|王|公|侯|将军|丞相|太尉|司马|太守)',
                r'(?:皇帝|国王|君主|将军|丞相|宰相|太尉|司马|太守)[一-龥]{2,4}',
                r'[一-龥]{2,3}(?=(?:说|曰|云|言|道|认为|表示))',
            ]
            
            person_entities = set()
            for pattern in person_patterns:
                matches = re.findall(pattern, text)
                person_entities.update(matches)
            
            # 添加已知历史人物
            known_figures = [
                '朱元璋', '王安石', '司马迁', '诸葛亮', '李白', '杜甫',
                '孔子', '孟子', '老子', '庄子', '韩愈', '欧阳修'
            ]
            
            for figure in known_figures:
                if figure in text:
                    person_entities.add(figure)
            
            # 计算人物重要性
            for person in person_entities:
                mentions = [m.start() for m in re.finditer(re.escape(person), text)]
                importance = min(len(mentions) * 0.2, 1.0)
                
                entities.append({
                    'name': person,
                    'type': '人物',
                    'importance': importance,
                    'mentions': mentions,
                    'confidence': 0.8
                })
            
            # 地名实体识别
            place_patterns = [
                r'[一-龥]{2,4}(?:州|府|县|郡|国|城|京|都|镇|村)',
                r'(?:北京|南京|西安|洛阳|开封|临安|金陵|燕京|大都|应天府)',
            ]
            
            place_entities = set()
            for pattern in place_patterns:
                matches = re.findall(pattern, text)
                place_entities.update(matches)
            
            for place in place_entities:
                mentions = [m.start() for m in re.finditer(re.escape(place), text)]
                importance = min(len(mentions) * 0.15, 1.0)
                
                entities.append({
                    'name': place,
                    'type': '地名',
                    'importance': importance,
                    'mentions': mentions,
                    'confidence': 0.7
                })
            
            # 按重要性排序
            entities.sort(key=lambda x: x['importance'], reverse=True)
            
            return entities[:20]  # 返回最多20个实体
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
    
    async def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """提取时间信息"""
        try:
            temporal_info = {
                'time_period': None,
                'specific_dates': [],
                'temporal_order': None,
                'time_score': 0.0
            }
            
            # 朝代时期识别
            dynasties = [
                '夏朝', '商朝', '周朝', '春秋', '战国', '秦朝',
                '汉朝', '三国', '晋朝', '南北朝', '隋朝', '唐朝',
                '五代', '宋朝', '元朝', '明朝', '清朝', '近代', '现代'
            ]
            
            for dynasty in dynasties:
                if dynasty in text:
                    temporal_info['time_period'] = dynasty
                    break
            
            # 具体年份识别
            year_patterns = [
                r'(\d{3,4})年',
                r'公元(\d{3,4})年',
                r'(?:元|贞观|开元|天宝|咸丰|光绪|宣统)(\d{1,2})年',
                r'(?:至正|洪武|永乐|嘉靖|万历|康熙|雍正|乾隆)(\d{1,2})年'
            ]
            
            specific_dates = []
            for pattern in year_patterns:
                matches = re.findall(pattern, text)
                specific_dates.extend([match if isinstance(match, str) else match[0] for match in matches])
            
            temporal_info['specific_dates'] = list(set(specific_dates))
            
            # 计算时间分数用于排序
            if temporal_info['specific_dates']:
                years = []
                for date in temporal_info['specific_dates']:
                    try:
                        year = int(date)
                        if 100 <= year <= 2024:  # 合理的年份范围
                            years.append(year)
                    except ValueError:
                        continue
                
                if years:
                    temporal_info['time_score'] = float(min(years))
            
            return temporal_info
            
        except Exception as e:
            logger.error(f"Temporal info extraction failed: {str(e)}")
            return {}
    
    async def _extract_key_points(self, text: str) -> List[str]:
        """提取关键要点"""
        try:
            # 分句
            sentences = re.split(r'[。！？；]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            key_points = []
            
            # 基于关键词的重要句子识别
            important_keywords = [
                '重要', '关键', '主要', '核心', '根本', '基础',
                '意义', '影响', '作用', '效果', '结果', '后果',
                '原因', '背景', '目的', '目标', '方针', '政策'
            ]
            
            for sentence in sentences:
                score = 0
                
                # 关键词评分
                for keyword in important_keywords:
                    if keyword in sentence:
                        score += 1
                
                # 长度评分（中等长度句子更可能是关键点）
                if 20 <= len(sentence) <= 100:
                    score += 1
                
                # 位置评分（开头和结尾的句子更重要）
                position = sentences.index(sentence)
                if position < 3 or position >= len(sentences) - 3:
                    score += 1
                
                if score >= 2:
                    key_points.append(sentence)
            
            # 使用TF-IDF提取重要句子
            if len(sentences) > 5:
                # 计算句子的TF-IDF分数
                sentence_scores = self._calculate_sentence_importance(sentences)
                
                # 选择得分最高的句子
                top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
                
                for sentence, score in top_sentences[:5]:
                    if sentence not in key_points:
                        key_points.append(sentence)
            
            return key_points[:10]  # 最多10个关键点
            
        except Exception as e:
            logger.error(f"Key points extraction failed: {str(e)}")
            return []
    
    def _calculate_sentence_importance(self, sentences: List[str]) -> Dict[str, float]:
        """计算句子重要性分数"""
        sentence_scores = {}
        
        # 简化的TF-IDF计算
        word_freq = Counter()
        for sentence in sentences:
            words = jieba.lcut(sentence)
            word_freq.update(words)
        
        for sentence in sentences:
            words = jieba.lcut(sentence)
            score = 0.0
            
            for word in words:
                if len(word) > 1 and word_freq[word] > 1:
                    # 简化的TF-IDF: tf * log(N/df)
                    tf = words.count(word) / len(words)
                    df = sum(1 for s in sentences if word in jieba.lcut(s))
                    idf = len(sentences) / df
                    score += tf * idf
            
            sentence_scores[sentence] = score
        
        return sentence_scores
    
    async def _analyze_sentiment(self, text: str) -> float:
        """分析情感倾向"""
        try:
            # 简化的情感分析（基于情感词典）
            positive_words = [
                '好', '优', '善', '美', '佳', '良', '棒', '妙',
                '成功', '胜利', '繁荣', '昌盛', '兴盛', '发达'
            ]
            
            negative_words = [
                '坏', '恶', '差', '劣', '糟', '乱', '败', '衰',
                '失败', '挫折', '困难', '问题', '危机', '灾难'
            ]
            
            words = jieba.lcut(text)
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                return 0.0  # 中性
            
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            return max(-1.0, min(1.0, sentiment_score))
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return 0.0
    
    async def _calculate_complexity(self, text: str) -> float:
        """计算文本复杂度"""
        try:
            # 多个复杂度指标
            
            # 词汇复杂度
            words = jieba.lcut(text)
            unique_words = set(words)
            vocab_complexity = len(unique_words) / len(words) if words else 0
            
            # 句子长度复杂度
            sentences = re.split(r'[。！？；]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences:
                avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
                length_complexity = min(avg_sentence_length / 50, 1.0)  # 归一化到0-1
            else:
                length_complexity = 0
            
            # 标点符号复杂度
            punctuation_count = len(re.findall(r'[，；：""''（）【】]', text))
            punct_complexity = min(punctuation_count / len(text) * 100, 1.0) if text else 0
            
            # 综合复杂度
            complexity = (vocab_complexity * 0.4 + length_complexity * 0.4 + punct_complexity * 0.2) * 10
            
            return min(complexity, 10.0)
            
        except Exception as e:
            logger.error(f"Complexity calculation failed: {str(e)}")
            return 5.0  # 默认中等复杂度
    
    async def analyze_content_similarity(self, content1: str, content2: str) -> float:
        """分析两个内容的相似度"""
        try:
            # 基于词汇重叠的相似度
            words1 = set(jieba.lcut(content1))
            words2 = set(jieba.lcut(content2))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            return jaccard_similarity
            
        except Exception as e:
            logger.error(f"Similarity analysis failed: {str(e)}")
            return 0.0
    
    async def extract_content_structure(self, text: str) -> Dict[str, Any]:
        """提取内容结构"""
        try:
            structure = {
                'paragraphs': [],
                'headings': [],
                'key_sentences': [],
                'organization_type': 'unknown'
            }
            
            # 段落分析
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            structure['paragraphs'] = [
                {
                    'content': para,
                    'length': len(para),
                    'position': i
                }
                for i, para in enumerate(paragraphs)
            ]
            
            # 标题识别
            heading_patterns = [
                r'^[一二三四五六七八九十]+[、．](.+)',
                r'^第[一二三四五六七八九十]+章\s+(.+)',
                r'^[0-9]+[、．]\s*(.+)',
                r'^#+\s*(.+)'  # Markdown标题
            ]
            
            for para in paragraphs:
                for pattern in heading_patterns:
                    match = re.match(pattern, para.strip())
                    if match:
                        structure['headings'].append({
                            'title': match.group(1),
                            'level': 1,
                            'position': para
                        })
                        break
            
            # 组织类型判断
            if any('年' in para for para in paragraphs):
                structure['organization_type'] = 'chronological'
            elif len(structure['headings']) > 2:
                structure['organization_type'] = 'structured'
            else:
                structure['organization_type'] = 'narrative'
            
            return structure
            
        except Exception as e:
            logger.error(f"Structure extraction failed: {str(e)}")
            return {}

# 全局分析器实例
content_analyzer = ContentAnalyzer()