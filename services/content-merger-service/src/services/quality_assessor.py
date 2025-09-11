"""
质量评估器

该模块负责评估合并后内容的质量，包括一致性、完整性、
流畅性、原创性和事实准确性等多个维度的评估。
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from collections import Counter
import jieba
from datetime import datetime

from ..models.merger_models import (
    ContentItem, MergeRequest, QualityMetrics, 
    QualityAssessmentError
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

class QualityAssessor:
    """质量评估器核心类"""
    
    def __init__(self):
        self._initialize_quality_metrics()
    
    def _initialize_quality_metrics(self):
        """初始化质量评估指标"""
        # 质量评估权重配置
        self.quality_weights = {
            'consistency': 0.25,    # 一致性权重
            'completeness': 0.25,   # 完整性权重
            'fluency': 0.20,        # 流畅性权重
            'originality': 0.15,    # 原创性权重
            'factual_accuracy': 0.15 # 事实准确性权重
        }
        
        # 质量阈值配置
        self.quality_thresholds = {
            'excellent': 90.0,
            'good': 80.0,
            'fair': 70.0,
            'poor': 60.0
        }
        
        logger.info("Quality assessor initialized successfully")
    
    async def assess_merge_quality(self, 
                                 source_contents: List[ContentItem],
                                 merged_result: Dict[str, Any], 
                                 request: MergeRequest) -> QualityMetrics:
        """
        评估合并质量
        
        Args:
            source_contents: 源内容列表
            merged_result: 合并结果
            request: 合并请求
            
        Returns:
            质量评估结果
        """
        try:
            logger.info("Starting merge quality assessment")
            
            merged_content = merged_result['content']
            
            # 并行执行各项质量评估
            assessment_tasks = [
                self._assess_consistency(merged_content),
                self._assess_completeness(source_contents, merged_content),
                self._assess_fluency(merged_content),
                self._assess_originality(source_contents, merged_content),
                self._assess_factual_accuracy(source_contents, merged_content)
            ]
            
            results = await asyncio.gather(*assessment_tasks)
            
            # 构建质量指标
            metrics = QualityMetrics(
                consistency_score=results[0],
                completeness_score=results[1],
                fluency_score=results[2],
                originality_score=results[3],
                factual_accuracy=results[4],
                overall_score=self._calculate_overall_score(results)
            )
            
            logger.info(f"Quality assessment completed. Overall score: {metrics.overall_score:.1f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            raise QualityAssessmentError(f"质量评估失败: {str(e)}")
    
    def _calculate_overall_score(self, individual_scores: List[float]) -> float:
        """计算综合质量分数"""
        score_mapping = {
            'consistency': individual_scores[0],
            'completeness': individual_scores[1],
            'fluency': individual_scores[2],
            'originality': individual_scores[3],
            'factual_accuracy': individual_scores[4]
        }
        
        overall_score = sum(
            score_mapping[metric] * weight 
            for metric, weight in self.quality_weights.items()
        )
        
        return min(100.0, max(0.0, overall_score))
    
    async def _assess_consistency(self, content: str) -> float:
        """评估内容一致性"""
        try:
            consistency_score = 100.0
            consistency_issues = 0
            
            # 分句处理
            sentences = re.split(r'[。！？；]', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            
            if len(sentences) < 2:
                return consistency_score
            
            # 1. 时间逻辑一致性检查
            temporal_issues = await self._check_temporal_consistency(sentences)
            consistency_issues += temporal_issues
            
            # 2. 人物角色一致性检查
            character_issues = await self._check_character_consistency(sentences)
            consistency_issues += character_issues
            
            # 3. 事实陈述一致性检查
            factual_issues = await self._check_factual_consistency(sentences)
            consistency_issues += factual_issues
            
            # 4. 语言风格一致性检查
            style_issues = await self._check_style_consistency(sentences)
            consistency_issues += style_issues
            
            # 根据发现的问题数量扣分
            consistency_score = max(0, 100 - consistency_issues * 5)
            
            logger.debug(f"Consistency assessment: {consistency_issues} issues, score: {consistency_score}")
            return consistency_score
            
        except Exception as e:
            logger.error(f"Consistency assessment failed: {str(e)}")
            return 70.0  # 默认分数
    
    async def _check_temporal_consistency(self, sentences: List[str]) -> int:
        """检查时间逻辑一致性"""
        issues = 0
        
        # 提取年份信息
        year_mentions = []
        for i, sentence in enumerate(sentences):
            years = re.findall(r'(\d{3,4})年', sentence)
            for year in years:
                try:
                    year_int = int(year)
                    if 100 <= year_int <= 2024:  # 合理年份范围
                        year_mentions.append((i, year_int, sentence))
                except ValueError:
                    continue
        
        # 检查年份逻辑顺序
        if len(year_mentions) > 1:
            for i in range(len(year_mentions) - 1):
                current_sentence_idx, current_year, _ = year_mentions[i]
                next_sentence_idx, next_year, _ = year_mentions[i + 1]
                
                # 如果后面的句子在文中位置靠后，但年份更早，可能存在时间逻辑问题
                if next_sentence_idx > current_sentence_idx and next_year < current_year - 50:
                    issues += 1
        
        return issues
    
    async def _check_character_consistency(self, sentences: List[str]) -> int:
        """检查人物角色一致性"""
        issues = 0
        
        # 提取人物及其描述
        character_descriptions = {}
        
        for sentence in sentences:
            # 简单的人物识别（可以改进）
            persons = re.findall(r'([一-龥]{2,4})(?:帝|王|公|侯|将军|丞相)', sentence)
            for person in persons:
                if person not in character_descriptions:
                    character_descriptions[person] = []
                character_descriptions[person].append(sentence)
        
        # 检查同一人物的描述是否一致
        for person, descriptions in character_descriptions.items():
            if len(descriptions) > 1:
                # 这里可以添加更复杂的一致性检查逻辑
                # 简化实现：检查是否有明显矛盾的描述
                pass
        
        return issues
    
    async def _check_factual_consistency(self, sentences: List[str]) -> int:
        """检查事实陈述一致性"""
        issues = 0
        
        # 检查数字和事实的一致性
        fact_mentions = {}
        
        for sentence in sentences:
            # 提取数字事实
            numbers = re.findall(r'(\d+)(?:年|人|万|千|百)', sentence)
            for number in numbers:
                context = sentence[:sentence.find(number) + len(number) + 10]
                if context in fact_mentions:
                    if fact_mentions[context] != number:
                        issues += 1
                else:
                    fact_mentions[context] = number
        
        return issues
    
    async def _check_style_consistency(self, sentences: List[str]) -> int:
        """检查语言风格一致性"""
        issues = 0
        
        # 检查语言风格的一致性
        formal_indicators = ['因此', '然而', '此外', '另外', '综上所述', '总之']
        informal_indicators = ['不过', '当然', '肯定', '绝对', '很', '非常']
        
        formal_count = 0
        informal_count = 0
        
        for sentence in sentences:
            for indicator in formal_indicators:
                if indicator in sentence:
                    formal_count += 1
                    break
            
            for indicator in informal_indicators:
                if indicator in sentence:
                    informal_count += 1
                    break
        
        # 如果正式和非正式风格混合过多，认为存在风格不一致
        total_indicators = formal_count + informal_count
        if total_indicators > 0:
            style_ratio = abs(formal_count - informal_count) / total_indicators
            if style_ratio < 0.6:  # 风格混杂
                issues += 1
        
        return issues
    
    async def _assess_completeness(self, source_contents: List[ContentItem], 
                                 merged_content: str) -> float:
        """评估内容完整性"""
        try:
            if not source_contents:
                return 100.0
            
            # 提取源内容的关键信息
            source_key_info = await self._extract_key_information(source_contents)
            
            # 检查合并内容中保留了多少关键信息
            preserved_info = await self._check_preserved_information(
                source_key_info, merged_content
            )
            
            # 计算完整性分数
            if not source_key_info['total_elements']:
                return 100.0
            
            preservation_rate = preserved_info['preserved_count'] / source_key_info['total_elements']
            completeness_score = preservation_rate * 100
            
            # 考虑信息密度
            density_bonus = min(10, len(merged_content) / 1000)  # 内容密度奖励
            completeness_score += density_bonus
            
            logger.debug(f"Completeness assessment: {preservation_rate:.2f} preservation rate, score: {completeness_score:.1f}")
            return min(100.0, completeness_score)
            
        except Exception as e:
            logger.error(f"Completeness assessment failed: {str(e)}")
            return 75.0
    
    async def _extract_key_information(self, source_contents: List[ContentItem]) -> Dict[str, Any]:
        """提取源内容的关键信息"""
        key_info = {
            'entities': set(),
            'topics': set(),
            'key_phrases': set(),
            'dates': set(),
            'total_elements': 0
        }
        
        for content in source_contents:
            text = content.content
            
            # 提取实体
            entities = re.findall(r'[一-龥]{2,4}(?:帝|王|公|侯|将军|丞相)', text)
            key_info['entities'].update(entities)
            
            # 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=10)
            key_info['key_phrases'].update(keywords)
            
            # 提取日期
            dates = re.findall(r'\d{3,4}年', text)
            key_info['dates'].update(dates)
            
            # 从分析结果中提取主题（如果有）
            if hasattr(content, 'analysis') and content.analysis:
                topics = content.analysis.get('topics', [])
                for topic in topics:
                    if isinstance(topic, dict):
                        key_info['topics'].add(topic.get('topic', ''))
                    else:
                        key_info['topics'].add(str(topic))
        
        key_info['total_elements'] = (
            len(key_info['entities']) + 
            len(key_info['topics']) + 
            len(key_info['key_phrases']) + 
            len(key_info['dates'])
        )
        
        return key_info
    
    async def _check_preserved_information(self, source_key_info: Dict[str, Any], 
                                         merged_content: str) -> Dict[str, Any]:
        """检查合并内容中保留的信息"""
        preserved_info = {
            'preserved_count': 0,
            'missing_critical': [],
            'details': {}
        }
        
        # 检查实体保留
        preserved_entities = 0
        for entity in source_key_info['entities']:
            if entity in merged_content:
                preserved_entities += 1
        
        # 检查主题保留
        preserved_topics = 0
        for topic in source_key_info['topics']:
            if topic and any(keyword in merged_content for keyword in topic.split()[:3]):
                preserved_topics += 1
        
        # 检查关键词保留
        preserved_phrases = 0
        for phrase in source_key_info['key_phrases']:
            if phrase in merged_content:
                preserved_phrases += 1
        
        # 检查日期保留
        preserved_dates = 0
        for date in source_key_info['dates']:
            if date in merged_content:
                preserved_dates += 1
        
        preserved_info['preserved_count'] = (
            preserved_entities + preserved_topics + 
            preserved_phrases + preserved_dates
        )
        
        preserved_info['details'] = {
            'entities': f"{preserved_entities}/{len(source_key_info['entities'])}",
            'topics': f"{preserved_topics}/{len(source_key_info['topics'])}",
            'phrases': f"{preserved_phrases}/{len(source_key_info['key_phrases'])}",
            'dates': f"{preserved_dates}/{len(source_key_info['dates'])}"
        }
        
        return preserved_info
    
    async def _assess_fluency(self, content: str) -> float:
        """评估文本流畅性"""
        try:
            fluency_score = 100.0
            
            # 分句处理
            sentences = re.split(r'[。！？；]', content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            
            if not sentences:
                return 0.0
            
            fluency_issues = 0
            
            # 1. 句子长度检查
            length_issues = await self._check_sentence_length(sentences)
            fluency_issues += length_issues
            
            # 2. 连接词使用检查
            transition_issues = await self._check_transitions(sentences)
            fluency_issues += transition_issues
            
            # 3. 重复用词检查
            repetition_issues = await self._check_repetitions(sentences)
            fluency_issues += repetition_issues
            
            # 4. 语法结构检查
            grammar_issues = await self._check_grammar_structure(sentences)
            fluency_issues += grammar_issues
            
            # 计算流畅性分数
            fluency_score = max(0, 100 - fluency_issues * 3)
            
            logger.debug(f"Fluency assessment: {fluency_issues} issues, score: {fluency_score}")
            return fluency_score
            
        except Exception as e:
            logger.error(f"Fluency assessment failed: {str(e)}")
            return 75.0
    
    async def _check_sentence_length(self, sentences: List[str]) -> int:
        """检查句子长度"""
        issues = 0
        
        for sentence in sentences:
            length = len(sentence)
            # 过长或过短的句子影响流畅性
            if length > 150 or length < 10:
                issues += 1
        
        return issues
    
    async def _check_transitions(self, sentences: List[str]) -> int:
        """检查句子间的过渡"""
        issues = 0
        
        transition_words = [
            '因此', '所以', '然而', '但是', '不过', '而且', '此外',
            '另外', '同时', '接着', '然后', '最后', '总之', '综上'
        ]
        
        # 检查过渡词的使用是否合理
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i]
            next_sentence = sentences[i + 1]
            
            # 检查是否需要过渡词但缺少
            if len(current_sentence) > 50 and len(next_sentence) > 50:
                has_transition = any(word in next_sentence[:20] for word in transition_words)
                if not has_transition:
                    # 这里可以添加更复杂的逻辑判断是否真的需要过渡
                    pass
        
        return issues
    
    async def _check_repetitions(self, sentences: List[str]) -> int:
        """检查重复用词"""
        issues = 0
        
        # 统计词汇使用频率
        all_words = []
        for sentence in sentences:
            words = jieba.lcut(sentence)
            all_words.extend([w for w in words if len(w) > 1])
        
        word_freq = Counter(all_words)
        
        # 检查过度重复的词汇
        total_words = len(all_words)
        for word, freq in word_freq.items():
            if freq > 5 and freq / total_words > 0.05:  # 单词出现过于频繁
                issues += 1
        
        return issues
    
    async def _check_grammar_structure(self, sentences: List[str]) -> int:
        """检查语法结构"""
        issues = 0
        
        for sentence in sentences:
            # 检查标点符号使用
            if sentence.count('，') > 5:  # 逗号过多
                issues += 1
            
            # 检查括号匹配
            if sentence.count('（') != sentence.count('）'):
                issues += 1
            
            if sentence.count('"') % 2 != 0:  # 引号不匹配
                issues += 1
        
        return issues
    
    async def _assess_originality(self, source_contents: List[ContentItem], 
                                merged_content: str) -> float:
        """评估原创性"""
        try:
            # 计算与源内容的相似度
            total_similarity = 0.0
            
            for content in source_contents:
                similarity = await self._calculate_text_overlap(
                    content.content, merged_content
                )
                total_similarity += similarity
            
            avg_similarity = total_similarity / len(source_contents) if source_contents else 0
            
            # 原创性分数（相似度越低，原创性越高）
            originality_score = max(0, 100 - avg_similarity * 100)
            
            # 检查创新元素
            innovation_bonus = await self._check_innovation_elements(
                source_contents, merged_content
            )
            
            originality_score += innovation_bonus
            
            logger.debug(f"Originality assessment: {avg_similarity:.2f} similarity, score: {originality_score:.1f}")
            return min(100.0, originality_score)
            
        except Exception as e:
            logger.error(f"Originality assessment failed: {str(e)}")
            return 70.0
    
    async def _calculate_text_overlap(self, source_text: str, merged_text: str) -> float:
        """计算文本重叠度"""
        source_words = set(jieba.lcut(source_text))
        merged_words = set(jieba.lcut(merged_text))
        
        if not source_words:
            return 0.0
        
        overlap = len(source_words & merged_words)
        return overlap / len(source_words)
    
    async def _check_innovation_elements(self, source_contents: List[ContentItem], 
                                       merged_content: str) -> float:
        """检查创新元素"""
        innovation_score = 0.0
        
        # 检查新的连接和过渡
        transition_words = ['因此', '综上所述', '由此可见', '总的来说']
        for word in transition_words:
            if word in merged_content:
                innovation_score += 2.0
        
        # 检查结构化改进
        if '##' in merged_content or '第' in merged_content and '章' in merged_content:
            innovation_score += 5.0
        
        return min(innovation_score, 20.0)  # 最多20分创新奖励
    
    async def _assess_factual_accuracy(self, source_contents: List[ContentItem], 
                                     merged_content: str) -> float:
        """评估事实准确性"""
        try:
            accuracy_score = 100.0
            accuracy_issues = 0
            
            # 检查关键事实的保持
            factual_issues = await self._check_factual_preservation(
                source_contents, merged_content
            )
            accuracy_issues += factual_issues
            
            # 检查时间事实的准确性
            temporal_issues = await self._check_temporal_accuracy(merged_content)
            accuracy_issues += temporal_issues
            
            # 检查数字事实的准确性
            numerical_issues = await self._check_numerical_accuracy(
                source_contents, merged_content
            )
            accuracy_issues += numerical_issues
            
            # 计算准确性分数
            accuracy_score = max(0, 100 - accuracy_issues * 10)
            
            logger.debug(f"Factual accuracy assessment: {accuracy_issues} issues, score: {accuracy_score}")
            return accuracy_score
            
        except Exception as e:
            logger.error(f"Factual accuracy assessment failed: {str(e)}")
            return 80.0
    
    async def _check_factual_preservation(self, source_contents: List[ContentItem], 
                                        merged_content: str) -> int:
        """检查事实保持"""
        issues = 0
        
        # 提取源内容中的关键事实
        source_facts = []
        for content in source_contents:
            # 提取人物-事件关联
            facts = re.findall(r'([一-龥]{2,4})(?:帝|王|公|侯).*?([一-龥]{2,4}(?:年|月))', content.content)
            source_facts.extend(facts)
        
        # 检查这些事实在合并内容中是否得到保持
        for person, event in source_facts:
            if person in merged_content and event not in merged_content:
                issues += 1
        
        return issues
    
    async def _check_temporal_accuracy(self, content: str) -> int:
        """检查时间准确性"""
        issues = 0
        
        # 提取年份
        years = re.findall(r'(\d{3,4})年', content)
        
        for year in years:
            try:
                year_int = int(year)
                # 检查年份的合理性
                if year_int < 1 or year_int > 2024:
                    issues += 1
                # 检查历史年份的合理性
                elif year_int < 1000 and year_int > 0:
                    # 中国历史中的合理年份范围检查
                    pass
            except ValueError:
                issues += 1
        
        return issues
    
    async def _check_numerical_accuracy(self, source_contents: List[ContentItem], 
                                      merged_content: str) -> int:
        """检查数字准确性"""
        issues = 0
        
        # 提取源内容中的数字
        source_numbers = []
        for content in source_contents:
            numbers = re.findall(r'(\d+)(?:万|千|百|十|年|人)', content.content)
            source_numbers.extend(numbers)
        
        # 提取合并内容中的数字
        merged_numbers = re.findall(r'(\d+)(?:万|千|百|十|年|人)', merged_content)
        
        # 检查数字的一致性（简化检查）
        source_set = set(source_numbers)
        merged_set = set(merged_numbers)
        
        # 如果合并内容中出现了源内容中没有的数字，可能存在错误
        new_numbers = merged_set - source_set
        if len(new_numbers) > len(source_numbers) * 0.2:  # 新数字过多
            issues += 1
        
        return issues

# 全局质量评估器实例
quality_assessor = QualityAssessor()