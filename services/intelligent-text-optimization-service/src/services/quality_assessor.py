"""
质量评估器 - Quality Assessor

专门负责文本优化质量的多维度评估，提供准确的质量评分和改进建议
集成多种评估算法和指标，确保评估结果的准确性和可信度

核心功能:
1. 多维度质量评估 (可读性、学术性、准确性等)
2. BLEU、ROUGE等客观指标计算
3. 自定义历史文本质量评估
4. 质量对比分析
5. 改进建议生成
"""

import logging
import asyncio
import math
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import jieba
import jieba.posseg as pseg
from rouge_score import rouge_scorer
from transformers import pipeline

from ..config.settings import get_settings
from ..models.optimization_models import (
    OptimizationType, OptimizationMode, QualityMetrics
)


logger = logging.getLogger(__name__)


class QualityAssessmentError(Exception):
    """质量评估错误"""
    pass


class ReadabilityAnalyzer:
    """
    可读性分析器
    专门评估中文历史文本的可读性
    """
    
    def __init__(self):
        """初始化可读性分析器"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def assess_readability(self, text: str) -> float:
        """
        评估文本可读性
        
        Args:
            text: 待评估文本
            
        Returns:
            可读性分数 (0-100)
        """
        try:
            if not text.strip():
                return 0.0
            
            # 多个可读性指标的综合评估
            sentence_score = await self._assess_sentence_readability(text)
            vocab_score = await self._assess_vocabulary_readability(text)
            structure_score = await self._assess_structure_readability(text)
            
            # 加权综合评分
            readability = (
                sentence_score * 0.4 +
                vocab_score * 0.35 +
                structure_score * 0.25
            )
            
            return min(max(readability, 0), 100)
            
        except Exception as e:
            self._logger.error(f"可读性评估失败: {e}")
            return 50.0  # 返回中等可读性
    
    async def _assess_sentence_readability(self, text: str) -> float:
        """评估句子可读性"""
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if not sentences:
            return 50.0
        
        sentence_lengths = [len(sentence) for sentence in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # 理想的句子长度范围
        if 15 <= avg_length <= 30:
            length_score = 100
        elif 10 <= avg_length < 15 or 30 < avg_length <= 40:
            length_score = 80
        elif 5 <= avg_length < 10 or 40 < avg_length <= 50:
            length_score = 60
        else:
            length_score = 40
        
        # 句子长度一致性
        if sentence_lengths:
            std_dev = math.sqrt(sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths))
            consistency_score = max(0, 100 - std_dev * 2)
        else:
            consistency_score = 50
        
        return (length_score * 0.7 + consistency_score * 0.3)
    
    async def _assess_vocabulary_readability(self, text: str) -> float:
        """评估词汇可读性"""
        words = list(jieba.cut(text))
        if not words:
            return 50.0
        
        # 词汇难度评估
        difficult_chars = set('顷翌卅廿惟乃迨逮')  # 一些较难的古汉语字符
        difficult_count = sum(1 for word in words for char in word if char in difficult_chars)
        difficult_ratio = difficult_count / max(len(''.join(words)), 1)
        
        # 词汇长度评估
        avg_word_length = sum(len(word) for word in words) / len(words)
        if 1.5 <= avg_word_length <= 2.5:
            length_score = 100
        elif 1 <= avg_word_length < 1.5 or 2.5 < avg_word_length <= 3:
            length_score = 80
        else:
            length_score = 60
        
        # 词汇多样性
        unique_words = set(words)
        diversity = len(unique_words) / max(len(words), 1)
        diversity_score = min(diversity * 150, 100)
        
        # 综合评分
        difficulty_penalty = difficult_ratio * 200  # 难字减分
        vocab_score = (length_score * 0.4 + diversity_score * 0.6) - difficulty_penalty
        
        return max(vocab_score, 0)
    
    async def _assess_structure_readability(self, text: str) -> float:
        """评估结构可读性"""
        # 段落结构
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        para_score = min(len(paragraphs) * 20, 100) if paragraphs else 50
        
        # 标点符号使用
        punctuation = '，。；：！？""''（）【】'
        punct_count = sum(text.count(p) for p in punctuation)
        punct_ratio = punct_count / max(len(text), 1)
        punct_score = min(punct_ratio * 500, 100)  # 适当的标点符号密度
        
        # 连接词使用
        connectives = ['因此', '然而', '此外', '同时', '另外', '首先', '其次', '最后', '总之']
        conn_count = sum(text.count(conn) for conn in connectives)
        conn_score = min(conn_count * 15, 80)
        
        return (para_score * 0.3 + punct_score * 0.4 + conn_score * 0.3)


class AcademicQualityAnalyzer:
    """
    学术质量分析器
    评估文本的学术规范性和专业性
    """
    
    def __init__(self):
        """初始化学术质量分析器"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 学术词汇库
        self.academic_terms = {
            '研究词汇': ['研究', '分析', '考察', '探讨', '论述', '阐述', '解释', '说明'],
            '证据词汇': ['根据', '依据', '基于', '据此', '由此', '证明', '表明', '显示'],
            '逻辑词汇': ['因此', '所以', '然而', '但是', '此外', '另外', '同时', '进而'],
            '历史专业': ['史载', '史书', '史料', '文献', '典籍', '记录', '记载', '考证'],
            '时间表达': ['时期', '阶段', '年间', '期间', '初期', '中期', '后期', '末年']
        }
    
    async def assess_academic_quality(self, text: str, optimization_type: OptimizationType) -> float:
        """
        评估学术质量
        
        Args:
            text: 待评估文本
            optimization_type: 优化类型
            
        Returns:
            学术质量分数 (0-100)
        """
        try:
            if not text.strip():
                return 0.0
            
            # 多维度学术质量评估
            terminology_score = await self._assess_terminology_usage(text)
            citation_score = await self._assess_citation_style(text)
            logic_score = await self._assess_logical_structure(text)
            formality_score = await self._assess_formality_level(text)
            
            # 根据优化类型调整权重
            weights = self._get_academic_weights(optimization_type)
            
            academic_quality = (
                terminology_score * weights['terminology'] +
                citation_score * weights['citation'] +
                logic_score * weights['logic'] +
                formality_score * weights['formality']
            )
            
            return min(max(academic_quality, 0), 100)
            
        except Exception as e:
            self._logger.error(f"学术质量评估失败: {e}")
            return 60.0
    
    async def _assess_terminology_usage(self, text: str) -> float:
        """评估专业术语使用"""
        score = 50  # 基础分数
        
        for category, terms in self.academic_terms.items():
            category_score = 0
            for term in terms:
                count = text.count(term)
                if count > 0:
                    category_score += min(count * 5, 20)  # 每个术语最多20分
            
            # 不同类别的权重
            if category == '历史专业':
                score += category_score * 0.3
            elif category == '研究词汇':
                score += category_score * 0.25
            else:
                score += category_score * 0.15
        
        return min(score, 100)
    
    async def _assess_citation_style(self, text: str) -> float:
        """评估引用风格"""
        score = 60  # 基础分数
        
        # 检测引用标志
        citation_indicators = ['据', '根据', '依据', '史载', '记录', '文献', '典籍']
        citation_count = sum(text.count(indicator) for indicator in citation_indicators)
        
        if citation_count > 0:
            score += min(citation_count * 8, 30)
        
        # 检测引号使用 (可能表示直接引用)
        quote_count = text.count('"') + text.count('"') + text.count('"')
        if quote_count >= 2:  # 至少有一对引号
            score += 10
        
        return min(score, 100)
    
    async def _assess_logical_structure(self, text: str) -> float:
        """评估逻辑结构"""
        score = 50
        
        # 逻辑连接词
        logic_words = {
            '因果': ['因此', '所以', '故', '因而', '由此'],
            '转折': ['然而', '但是', '可是', '不过', '虽然'],
            '递进': ['此外', '另外', '而且', '并且', '同时'],
            '总结': ['总之', '综上', '总的来说', '由此可见']
        }
        
        for category, words in logic_words.items():
            category_count = sum(text.count(word) for word in words)
            if category_count > 0:
                score += min(category_count * 8, 15)
        
        # 段落结构逻辑性
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if 2 <= len(paragraphs) <= 5:
            score += 10
        
        return min(score, 100)
    
    async def _assess_formality_level(self, text: str) -> float:
        """评估正式性程度"""
        score = 60
        
        # 正式用词
        formal_words = ['进行', '实施', '执行', '开展', '具备', '拥有', '获得', '取得']
        formal_count = sum(text.count(word) for word in formal_words)
        score += min(formal_count * 5, 20)
        
        # 避免口语化表达
        informal_words = ['特别', '非常', '很', '挺', '蛮', '超', '巨']
        informal_count = sum(text.count(word) for word in informal_words)
        score -= min(informal_count * 10, 30)
        
        return max(score, 0)
    
    def _get_academic_weights(self, optimization_type: OptimizationType) -> Dict[str, float]:
        """根据优化类型获取学术质量权重"""
        if optimization_type == OptimizationType.POLISH:
            return {
                'terminology': 0.3,
                'citation': 0.2,
                'logic': 0.3,
                'formality': 0.2
            }
        elif optimization_type == OptimizationType.EXPAND:
            return {
                'terminology': 0.25,
                'citation': 0.35,
                'logic': 0.25,
                'formality': 0.15
            }
        else:
            return {
                'terminology': 0.25,
                'citation': 0.25,
                'logic': 0.25,
                'formality': 0.25
            }


class HistoricalAccuracyAnalyzer:
    """
    历史准确性分析器
    评估文本优化过程中历史信息的保持情况
    """
    
    def __init__(self):
        """初始化历史准确性分析器"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def assess_historical_accuracy(
        self, 
        original_text: str, 
        optimized_text: str
    ) -> float:
        """
        评估历史准确性
        
        Args:
            original_text: 原始文本
            optimized_text: 优化后文本
            
        Returns:
            历史准确性分数 (0-100)
        """
        try:
            if not original_text.strip() or not optimized_text.strip():
                return 0.0
            
            # 提取和比较关键历史信息
            original_entities = await self._extract_historical_entities(original_text)
            optimized_entities = await self._extract_historical_entities(optimized_text)
            
            # 计算实体保持率
            entity_preservation = await self._calculate_entity_preservation(
                original_entities, optimized_entities
            )
            
            # 评估时间信息一致性
            time_consistency = await self._assess_time_consistency(
                original_text, optimized_text
            )
            
            # 评估数字信息一致性
            number_consistency = await self._assess_number_consistency(
                original_text, optimized_text
            )
            
            # 综合评分
            accuracy = (
                entity_preservation * 0.5 +
                time_consistency * 0.3 +
                number_consistency * 0.2
            )
            
            return min(max(accuracy, 0), 100)
            
        except Exception as e:
            self._logger.error(f"历史准确性评估失败: {e}")
            return 80.0  # 返回较高的默认值
    
    async def _extract_historical_entities(self, text: str) -> Dict[str, List[str]]:
        """提取历史实体"""
        entities = {
            '人名': [],
            '地名': [],
            '朝代': [],
            '官职': [],
            '时间': []
        }
        
        try:
            # 使用jieba词性标注
            words = pseg.cut(text)
            
            for word, flag in words:
                if len(word) > 1:  # 过滤单字
                    if flag == 'nr':  # 人名
                        entities['人名'].append(word)
                    elif flag == 'ns':  # 地名
                        entities['地名'].append(word)
                    elif flag == 'nt':  # 时间
                        entities['时间'].append(word)
            
            # 额外的历史专有名词识别
            await self._identify_historical_terms(text, entities)
            
            return entities
            
        except Exception as e:
            self._logger.warning(f"实体提取失败: {e}")
            return entities
    
    async def _identify_historical_terms(self, text: str, entities: Dict[str, List[str]]):
        """识别历史专有名词"""
        # 朝代识别
        dynasties = ['夏', '商', '周', '秦', '汉', '三国', '晋', '南北朝', '隋', '唐', 
                    '五代', '宋', '元', '明', '清', '春秋', '战国']
        for dynasty in dynasties:
            if dynasty in text and dynasty not in entities['朝代']:
                entities['朝代'].append(dynasty)
        
        # 官职识别
        positions = ['皇帝', '丞相', '太守', '刺史', '县令', '知府', '知县', '将军', 
                    '尚书', '侍郎', '御史', '太尉', '司徒', '司空']
        for position in positions:
            if position in text and position not in entities['官职']:
                entities['官职'].append(position)
    
    async def _calculate_entity_preservation(
        self, 
        original_entities: Dict[str, List[str]], 
        optimized_entities: Dict[str, List[str]]
    ) -> float:
        """计算实体保持率"""
        total_original = sum(len(entities) for entities in original_entities.values())
        
        if total_original == 0:
            return 100.0  # 如果原文没有实体，认为保持完好
        
        preserved_count = 0
        
        for category, orig_entities in original_entities.items():
            opt_entities = optimized_entities.get(category, [])
            
            for entity in orig_entities:
                # 检查实体是否在优化后文本中保持
                if any(entity in opt_entity or opt_entity in entity for opt_entity in opt_entities):
                    preserved_count += 1
                elif entity in ' '.join(opt_entities):  # 模糊匹配
                    preserved_count += 0.5
        
        preservation_rate = preserved_count / total_original
        return preservation_rate * 100
    
    async def _assess_time_consistency(self, original: str, optimized: str) -> float:
        """评估时间信息一致性"""
        try:
            # 提取时间表达
            time_patterns = ['年', '月', '日', '初', '末', '前', '后', '间', '时']
            
            original_times = []
            optimized_times = []
            
            import re
            
            # 提取年代信息
            year_pattern = r'(\d{1,4}年)'
            original_years = re.findall(year_pattern, original)
            optimized_years = re.findall(year_pattern, optimized)
            
            original_times.extend(original_years)
            optimized_times.extend(optimized_years)
            
            # 提取其他时间表达
            for pattern in time_patterns:
                orig_matches = re.findall(r'(\w*' + pattern + r'\w*)', original)
                opt_matches = re.findall(r'(\w*' + pattern + r'\w*)', optimized)
                original_times.extend(orig_matches)
                optimized_times.extend(opt_matches)
            
            if not original_times:
                return 100.0
            
            # 计算时间信息保持率
            preserved_times = 0
            for orig_time in original_times:
                if orig_time in optimized_times:
                    preserved_times += 1
                elif any(orig_time in opt_time or opt_time in orig_time for opt_time in optimized_times):
                    preserved_times += 0.5
            
            consistency = preserved_times / len(original_times)
            return consistency * 100
            
        except Exception as e:
            self._logger.warning(f"时间一致性评估失败: {e}")
            return 90.0
    
    async def _assess_number_consistency(self, original: str, optimized: str) -> float:
        """评估数字信息一致性"""
        try:
            import re
            
            # 提取数字
            number_pattern = r'\d+'
            original_numbers = re.findall(number_pattern, original)
            optimized_numbers = re.findall(number_pattern, optimized)
            
            if not original_numbers:
                return 100.0
            
            # 计算数字保持率
            preserved_numbers = 0
            for orig_num in original_numbers:
                if orig_num in optimized_numbers:
                    preserved_numbers += 1
            
            consistency = preserved_numbers / len(original_numbers)
            return consistency * 100
            
        except Exception as e:
            self._logger.warning(f"数字一致性评估失败: {e}")
            return 90.0


class ObjectiveMetricsCalculator:
    """
    客观指标计算器
    计算BLEU、ROUGE等客观评估指标
    """
    
    def __init__(self):
        """初始化客观指标计算器"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化ROUGE评分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    async def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        Args:
            reference: 参考文本 (原始文本)
            candidate: 候选文本 (优化后文本)
            
        Returns:
            ROUGE分数字典
        """
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            
            return {
                'rouge1_f': scores['rouge1'].fmeasure * 100,
                'rouge1_p': scores['rouge1'].precision * 100,
                'rouge1_r': scores['rouge1'].recall * 100,
                'rouge2_f': scores['rouge2'].fmeasure * 100,
                'rouge2_p': scores['rouge2'].precision * 100,
                'rouge2_r': scores['rouge2'].recall * 100,
                'rougeL_f': scores['rougeL'].fmeasure * 100,
                'rougeL_p': scores['rougeL'].precision * 100,
                'rougeL_r': scores['rougeL'].recall * 100
            }
            
        except Exception as e:
            self._logger.error(f"ROUGE分数计算失败: {e}")
            return {k: 0.0 for k in ['rouge1_f', 'rouge1_p', 'rouge1_r',
                                   'rouge2_f', 'rouge2_p', 'rouge2_r',
                                   'rougeL_f', 'rougeL_p', 'rougeL_r']}
    
    async def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        计算BLEU分数 (简化版本)
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            
        Returns:
            BLEU分数 (0-100)
        """
        try:
            # 分词
            ref_words = list(jieba.cut(reference))
            cand_words = list(jieba.cut(candidate))
            
            if not ref_words or not cand_words:
                return 0.0
            
            # 计算n-gram精确度
            def ngram_precision(ref_words, cand_words, n):
                if len(cand_words) < n:
                    return 0.0
                
                ref_ngrams = [tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)]
                cand_ngrams = [tuple(cand_words[i:i+n]) for i in range(len(cand_words)-n+1)]
                
                if not cand_ngrams:
                    return 0.0
                
                ref_counter = Counter(ref_ngrams)
                cand_counter = Counter(cand_ngrams)
                
                overlap = 0
                for ngram, count in cand_counter.items():
                    overlap += min(count, ref_counter.get(ngram, 0))
                
                return overlap / len(cand_ngrams)
            
            # 计算1-4gram精确度
            precisions = []
            for n in range(1, 5):
                precision = ngram_precision(ref_words, cand_words, n)
                if precision == 0:
                    return 0.0
                precisions.append(precision)
            
            # 几何平均
            bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            
            # 长度惩罚
            bp = min(1.0, math.exp(1 - len(ref_words) / max(len(cand_words), 1)))
            
            return bleu * bp * 100
            
        except Exception as e:
            self._logger.error(f"BLEU分数计算失败: {e}")
            return 0.0


class QualityAssessor:
    """
    质量评估器主类
    整合各种评估组件，提供统一的质量评估接口
    """
    
    def __init__(self):
        """初始化质量评估器"""
        self.settings = get_settings()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化各个评估组件
        self.readability_analyzer = ReadabilityAnalyzer()
        self.academic_analyzer = AcademicQualityAnalyzer()
        self.accuracy_analyzer = HistoricalAccuracyAnalyzer()
        self.metrics_calculator = ObjectiveMetricsCalculator()
    
    async def assess_quality(
        self,
        original_text: str,
        optimized_text: str,
        optimization_type: OptimizationType,
        optimization_mode: OptimizationMode = OptimizationMode.HISTORICAL_FORMAT
    ) -> QualityMetrics:
        """
        全面评估优化质量
        
        Args:
            original_text: 原始文本
            optimized_text: 优化后文本
            optimization_type: 优化类型
            optimization_mode: 优化模式
            
        Returns:
            完整的质量评估结果
        """
        try:
            self._logger.info(f"开始质量评估 (优化类型: {optimization_type.value})")
            
            # 并行执行各项评估
            tasks = [
                self.readability_analyzer.assess_readability(optimized_text),
                self.academic_analyzer.assess_academic_quality(optimized_text, optimization_type),
                self.accuracy_analyzer.assess_historical_accuracy(original_text, optimized_text),
                self._assess_language_quality(optimized_text),
                self._assess_structure_quality(optimized_text),
                self._assess_content_completeness(original_text, optimized_text),
                self.readability_analyzer.assess_readability(original_text)  # 用于计算改进幅度
            ]
            
            results = await asyncio.gather(*tasks)
            
            (readability_score, academic_score, historical_accuracy, 
             language_quality, structure_score, content_completeness, original_readability) = results
            
            # 计算改进幅度
            readability_improvement = readability_score - original_readability
            academic_improvement = 0  # 需要原始学术评分来计算
            structure_improvement = 0  # 需要原始结构评分来计算
            
            # 计算综合评分
            overall_score = await self._calculate_overall_score(
                readability_score, academic_score, historical_accuracy,
                language_quality, structure_score, content_completeness,
                optimization_type
            )
            
            # 分析优势和不足
            strengths, weaknesses = await self._analyze_quality_aspects({
                'readability': readability_score,
                'academic': academic_score,
                'historical_accuracy': historical_accuracy,
                'language_quality': language_quality,
                'structure': structure_score,
                'completeness': content_completeness
            })
            
            # 构建质量评估结果
            quality_metrics = QualityMetrics(
                overall_score=round(overall_score, 2),
                readability_score=round(readability_score, 2),
                academic_score=round(academic_score, 2),
                historical_accuracy=round(historical_accuracy, 2),
                language_quality=round(language_quality, 2),
                structure_score=round(structure_score, 2),
                content_completeness=round(content_completeness, 2),
                readability_improvement=round(readability_improvement, 2) if readability_improvement else None,
                academic_improvement=round(academic_improvement, 2) if academic_improvement else None,
                structure_improvement=round(structure_improvement, 2) if structure_improvement else None,
                strengths=strengths,
                weaknesses=weaknesses
            )
            
            self._logger.info(f"质量评估完成，综合分数: {overall_score:.2f}")
            return quality_metrics
            
        except Exception as e:
            self._logger.error(f"质量评估失败: {e}")
            raise QualityAssessmentError(f"质量评估失败: {str(e)}")
    
    async def _assess_language_quality(self, text: str) -> float:
        """评估语言质量"""
        score = 70  # 基础分数
        
        # 语法检查 (简单规则)
        if '的的' not in text and '了了' not in text and '是是' not in text:
            score += 5
        
        # 词汇重复检查
        words = list(jieba.cut(text))
        word_freq = Counter(words)
        if word_freq:
            max_freq = max(word_freq.values())
            total_words = len(words)
            if max_freq <= total_words * 0.15:  # 单词重复率不超过15%
                score += 10
            else:
                score -= 5
        
        # 句子完整性
        sentences = text.split('。')
        complete_sentences = [s for s in sentences if len(s.strip()) > 5]
        if len(complete_sentences) >= len(sentences) * 0.8:
            score += 10
        
        # 标点符号使用
        punctuation_ratio = sum(text.count(p) for p in '，。；：') / max(len(text), 1)
        if 0.05 <= punctuation_ratio <= 0.15:
            score += 5
        
        return min(score, 100)
    
    async def _assess_structure_quality(self, text: str) -> float:
        """评估结构质量"""
        score = 60
        
        # 段落结构
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if 1 <= len(paragraphs) <= 5:
            score += 15
        elif len(paragraphs) > 5:
            score += 10
        
        # 句子长度分布
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if sentences:
            lengths = [len(s) for s in sentences]
            avg_length = sum(lengths) / len(lengths)
            if 15 <= avg_length <= 35:
                score += 15
        
        # 逻辑连接
        connectives = ['因此', '然而', '此外', '同时', '另外', '首先', '其次', '最后']
        conn_count = sum(text.count(conn) for conn in connectives)
        score += min(conn_count * 3, 15)
        
        return min(score, 100)
    
    async def _assess_content_completeness(self, original: str, optimized: str) -> float:
        """评估内容完整性"""
        try:
            # 长度比例评估
            length_ratio = len(optimized) / max(len(original), 1)
            
            if 0.8 <= length_ratio <= 1.5:
                length_score = 90
            elif 0.6 <= length_ratio <= 2.0:
                length_score = 80
            else:
                length_score = 60
            
            # 关键词保持率
            orig_words = set(jieba.cut(original))
            opt_words = set(jieba.cut(optimized))
            
            # 过滤停用词
            stopwords = {'的', '了', '在', '是', '有', '和', '与', '或', '但', '等', '也', '都', '又', '还'}
            orig_keywords = orig_words - stopwords
            opt_keywords = opt_words - stopwords
            
            if orig_keywords:
                retention_rate = len(orig_keywords & opt_keywords) / len(orig_keywords)
                retention_score = retention_rate * 100
            else:
                retention_score = 100
            
            # 综合评分
            completeness = length_score * 0.4 + retention_score * 0.6
            return min(completeness, 100)
            
        except Exception as e:
            self._logger.warning(f"内容完整性评估失败: {e}")
            return 80.0
    
    async def _calculate_overall_score(
        self,
        readability: float,
        academic: float,
        accuracy: float,
        language: float,
        structure: float,
        completeness: float,
        optimization_type: OptimizationType
    ) -> float:
        """计算综合质量分数"""
        # 根据优化类型设置权重
        if optimization_type == OptimizationType.POLISH:
            weights = {
                'readability': 0.25,
                'academic': 0.2,
                'accuracy': 0.2,
                'language': 0.25,
                'structure': 0.05,
                'completeness': 0.05
            }
        elif optimization_type == OptimizationType.EXPAND:
            weights = {
                'readability': 0.15,
                'academic': 0.2,
                'accuracy': 0.25,
                'language': 0.15,
                'structure': 0.1,
                'completeness': 0.15
            }
        elif optimization_type == OptimizationType.MODERNIZE:
            weights = {
                'readability': 0.35,
                'academic': 0.1,
                'accuracy': 0.2,
                'language': 0.25,
                'structure': 0.05,
                'completeness': 0.05
            }
        else:  # STYLE_CONVERT
            weights = {
                'readability': 0.3,
                'academic': 0.15,
                'accuracy': 0.2,
                'language': 0.2,
                'structure': 0.1,
                'completeness': 0.05
            }
        
        overall = (
            readability * weights['readability'] +
            academic * weights['academic'] +
            accuracy * weights['accuracy'] +
            language * weights['language'] +
            structure * weights['structure'] +
            completeness * weights['completeness']
        )
        
        return min(max(overall, 0), 100)
    
    async def _analyze_quality_aspects(self, scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """分析质量各方面的优势和不足"""
        strengths = []
        weaknesses = []
        
        aspect_names = {
            'readability': '可读性',
            'academic': '学术规范性',
            'historical_accuracy': '历史准确性',
            'language_quality': '语言质量',
            'structure': '文本结构',
            'completeness': '内容完整性'
        }
        
        for aspect, score in scores.items():
            name = aspect_names.get(aspect, aspect)
            
            if score >= 85:
                strengths.append(f"{name}优秀")
            elif score >= 75:
                strengths.append(f"{name}良好")
            elif score < 60:
                weaknesses.append(f"{name}需要改进")
            elif score < 70:
                weaknesses.append(f"{name}有待提升")
        
        if not strengths:
            strengths.append("完成了基础优化")
        if not weaknesses:
            weaknesses.append("整体质量良好")
        
        return strengths, weaknesses