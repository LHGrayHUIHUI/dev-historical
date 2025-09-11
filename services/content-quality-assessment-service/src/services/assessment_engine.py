"""
内容质量评估引擎

核心评估引擎，负责执行多维度内容质量评估，包括可读性、准确性、
完整性、连贯性、相关性等维度的分析和评分。
"""

import spacy
import jieba
import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import numpy as np
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import aioredis

from ..config.settings import settings
from ..models.assessment_models import (
    QualityAssessmentRequest, QualityAssessmentResult, QualityMetric,
    QualityDimension, QualityGrade, AssessmentStatus, ProcessingMetrics,
    ReadabilityMetric, AccuracyMetric, CompletenessMetric, 
    CoherenceMetric, RelevanceMetric, AssessmentError
)
from ..clients.ai_service_client import ai_service_client
from ..clients.storage_client import storage_client

logger = logging.getLogger(__name__)

class ContentQualityAssessmentEngine:
    """内容质量评估引擎"""
    
    def __init__(self):
        # NLP模型初始化
        self.nlp = None
        self.executor = ThreadPoolExecutor(max_workers=settings.nlp_models.max_workers)
        self.redis = None
        
        # 评估配置
        self.enabled_dimensions = settings.assessment_engine.enabled_dimensions
        self.dimension_weights = settings.quality_metrics.default_dimension_weights
        self.grade_thresholds = settings.quality_metrics.grade_thresholds
        
        # 缓存配置
        self.cache_enabled = settings.assessment_engine.cache_assessment_results
        self.cache_ttl = settings.assessment_engine.cache_ttl_hours * 3600
        
        # 性能配置
        self.max_content_length = settings.assessment_engine.max_content_length
        self.assessment_timeout = settings.assessment_engine.assessment_timeout
        
    async def initialize(self):
        """初始化评估引擎"""
        try:
            # 初始化spaCy模型
            await self._load_nlp_model()
            
            # 初始化Redis连接
            if self.cache_enabled:
                await self._initialize_redis()
            
            # 初始化jieba
            await self._initialize_jieba()
            
            logger.info("Assessment engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize assessment engine: {str(e)}")
            raise
    
    async def _load_nlp_model(self):
        """加载NLP模型"""
        try:
            def load_model():
                import spacy
                return spacy.load(settings.nlp_models.spacy_model)
            
            loop = asyncio.get_event_loop()
            self.nlp = await loop.run_in_executor(self.executor, load_model)
            
            logger.info(f"Loaded spaCy model: {settings.nlp_models.spacy_model}")
            
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            raise
    
    async def _initialize_redis(self):
        """初始化Redis连接"""
        try:
            self.redis = await aioredis.from_url(
                settings.database.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # 测试连接
            await self.redis.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed, disabling cache: {str(e)}")
            self.cache_enabled = False
            self.redis = None
    
    async def _initialize_jieba(self):
        """初始化jieba分词"""
        try:
            def init_jieba():
                jieba.initialize()
                return True
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, init_jieba)
            
            logger.info("Jieba initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize jieba: {str(e)}")
            raise
    
    async def assess_quality(self, request: QualityAssessmentRequest) -> QualityAssessmentResult:
        """
        执行内容质量评估
        
        Args:
            request: 评估请求
            
        Returns:
            QualityAssessmentResult: 评估结果
        """
        start_time = datetime.now()
        processing_metrics = ProcessingMetrics(start_time=start_time)
        
        try:
            # 验证请求
            await self._validate_request(request)
            
            # 检查缓存
            if self.cache_enabled and request.enable_caching:
                cached_result = await self._get_cached_result(request)
                if cached_result:
                    logger.info(f"Returning cached result for {request.assessment_id}")
                    return cached_result
            
            # 预处理内容
            processed_content = await self._preprocess_content(request.content)
            
            # 并行执行各维度评估
            metrics = await self._execute_dimensional_assessments(
                processed_content, request
            )
            
            # 应用自定义权重
            if request.custom_weights:
                metrics = self._apply_custom_weights(metrics, request.custom_weights)
            elif request.enabled_dimensions:
                metrics = self._filter_enabled_dimensions(metrics, request.enabled_dimensions)
            
            # 计算综合评分
            overall_score = self._calculate_overall_score(metrics)
            grade = self._determine_grade(overall_score)
            
            # 生成分析报告
            strengths, weaknesses, recommendations = await self._generate_analysis_report(
                metrics, request.content_type
            )
            
            # 记录处理完成时间
            processing_metrics.end_time = datetime.now()
            processing_metrics.duration_seconds = (
                processing_metrics.end_time - processing_metrics.start_time
            ).total_seconds()
            
            # 构建评估结果
            result = QualityAssessmentResult(
                assessment_id=request.assessment_id,
                content_id=request.content_id,
                content_type=request.content_type,
                overall_score=overall_score,
                grade=grade,
                metrics=metrics,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                assessment_time=start_time,
                processing_duration=processing_metrics.duration_seconds,
                model_versions={
                    "spacy": spacy.__version__ if spacy else "unknown",
                    "assessment_engine": "1.0.0",
                    "jieba": "0.42.1"
                },
                status=AssessmentStatus.COMPLETED,
                processing_metrics=processing_metrics
            )
            
            # 缓存结果
            if self.cache_enabled and request.enable_caching:
                await self._cache_result(request, result)
            
            # 保存到存储服务
            try:
                await storage_client.save_assessment_result(result)
            except Exception as e:
                logger.warning(f"Failed to save assessment result: {str(e)}")
            
            logger.info(f"Quality assessment completed for {request.content_id}, "
                       f"score: {overall_score:.2f}, grade: {grade}")
            
            return result
            
        except Exception as e:
            # 记录错误信息
            error_info = AssessmentError(
                error_code="ASSESSMENT_FAILED",
                error_message=str(e),
                error_details={"request_id": request.assessment_id}
            )
            
            processing_metrics.end_time = datetime.now()
            processing_metrics.duration_seconds = (
                processing_metrics.end_time - processing_metrics.start_time
            ).total_seconds()
            
            # 返回失败结果
            result = QualityAssessmentResult(
                assessment_id=request.assessment_id,
                content_id=request.content_id,
                content_type=request.content_type,
                overall_score=0.0,
                grade=QualityGrade.F,
                metrics=[],
                assessment_time=start_time,
                processing_duration=processing_metrics.duration_seconds,
                model_versions={},
                status=AssessmentStatus.FAILED,
                error_info=error_info,
                processing_metrics=processing_metrics
            )
            
            logger.error(f"Quality assessment failed for {request.content_id}: {str(e)}")
            return result
    
    async def _validate_request(self, request: QualityAssessmentRequest):
        """验证评估请求"""
        if len(request.content) > self.max_content_length:
            raise ValueError(f"Content length {len(request.content)} exceeds maximum {self.max_content_length}")
        
        if not request.content.strip():
            raise ValueError("Content cannot be empty")
    
    async def _preprocess_content(self, content: str) -> str:
        """预处理内容"""
        # 清理多余空白字符
        content = re.sub(r'\s+', ' ', content.strip())
        
        # 移除特殊字符但保留必要的标点
        content = re.sub(r'[^\u4e00-\u9fff\w\s.,;:!?""''()【】《》\-]', '', content)
        
        return content
    
    async def _execute_dimensional_assessments(self, 
                                             content: str,
                                             request: QualityAssessmentRequest) -> List[QualityMetric]:
        """并行执行各维度评估"""
        tasks = []
        
        # 基础维度评估
        if QualityDimension.READABILITY in self.enabled_dimensions:
            tasks.append(self._assess_readability(content, request.content_type))
        
        if QualityDimension.ACCURACY in self.enabled_dimensions:
            tasks.append(self._assess_accuracy(content, request.content_type))
        
        if QualityDimension.COMPLETENESS in self.enabled_dimensions:
            tasks.append(self._assess_completeness(content, request.content_type))
        
        if QualityDimension.COHERENCE in self.enabled_dimensions:
            tasks.append(self._assess_coherence(content, request.content_type))
        
        if QualityDimension.RELEVANCE in self.enabled_dimensions:
            tasks.append(self._assess_relevance(content, request.content_type, request.target_audience))
        
        # 执行并行评估
        metrics = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_metrics = []
        for metric in metrics:
            if isinstance(metric, Exception):
                logger.error(f"Dimensional assessment failed: {str(metric)}")
            else:
                valid_metrics.append(metric)
        
        return valid_metrics
    
    async def _assess_readability(self, content: str, content_type: str) -> ReadabilityMetric:
        """评估可读性"""
        try:
            def analyze_readability():
                # spaCy分析
                doc = self.nlp(content)
                
                # 基础统计
                sentences = list(doc.sents)
                sentence_count = len(sentences)
                word_count = len([token for token in doc if not token.is_space])
                
                # 平均句长
                avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
                
                # 词汇多样性
                unique_words = set([token.text.lower() for token in doc if token.is_alpha])
                vocab_diversity = len(unique_words) / word_count if word_count > 0 else 0
                
                # 字符复杂度 (中文字符比例)
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
                char_complexity = chinese_chars / len(content) if len(content) > 0 else 0
                
                # 句法复杂度
                complex_deps = ["nsubj", "dobj", "pobj", "amod", "advmod"]
                syntax_complexity = sum(1 for token in doc if token.dep_ in complex_deps) / word_count if word_count > 0 else 0
                
                return {
                    "avg_sentence_length": avg_sentence_length,
                    "vocab_diversity": vocab_diversity,
                    "char_complexity": char_complexity,
                    "syntax_complexity": syntax_complexity
                }
            
            # 在线程池中执行
            loop = asyncio.get_event_loop()
            analysis_result = await loop.run_in_executor(self.executor, analyze_readability)
            
            # 计算评分
            score = await self._calculate_readability_score(analysis_result, content_type)
            
            # 生成问题和建议
            issues, suggestions = self._generate_readability_feedback(analysis_result)
            
            return ReadabilityMetric(
                dimension=QualityDimension.READABILITY,
                score=score,
                weight=self.dimension_weights.get(QualityDimension.READABILITY, 0.2),
                confidence=0.85,
                details=analysis_result,
                issues=issues,
                suggestions=suggestions,
                avg_sentence_length=analysis_result["avg_sentence_length"],
                vocab_diversity=analysis_result["vocab_diversity"] * 100,
                char_complexity=analysis_result["char_complexity"] * 100,
                syntax_complexity=analysis_result["syntax_complexity"] * 100
            )
            
        except Exception as e:
            logger.error(f"Readability assessment failed: {str(e)}")
            # 返回默认指标
            return ReadabilityMetric(
                dimension=QualityDimension.READABILITY,
                score=50.0,
                weight=self.dimension_weights.get(QualityDimension.READABILITY, 0.2),
                confidence=0.3,
                details={},
                issues=["评估失败"],
                suggestions=["请检查内容格式"]
            )
    
    async def _calculate_readability_score(self, analysis: Dict[str, float], content_type: str) -> float:
        """计算可读性评分"""
        # 基础评分
        base_score = 80.0
        
        # 句长调整 (中文适宜句长15-25字)
        avg_len = analysis["avg_sentence_length"]
        if avg_len > 30:
            base_score -= min(20, (avg_len - 30) * 0.5)
        elif avg_len < 8:
            base_score -= min(15, (8 - avg_len) * 2)
        else:
            base_score += 5  # 句长适中加分
        
        # 词汇多样性调整
        diversity = analysis["vocab_diversity"]
        if diversity > 0.6:
            base_score += 10
        elif diversity < 0.3:
            base_score -= 10
        
        # 复杂度调整
        complexity = analysis["char_complexity"]
        if content_type == "educational_content" and complexity > 0.8:
            base_score -= 15  # 教育内容不宜过于复杂
        elif content_type == "academic_paper" and complexity > 0.9:
            base_score += 5   # 学术论文允许更复杂
        
        return max(0, min(100, base_score))
    
    def _generate_readability_feedback(self, analysis: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """生成可读性反馈"""
        issues = []
        suggestions = []
        
        if analysis["avg_sentence_length"] > 25:
            issues.append("句子平均长度过长，可能影响理解")
            suggestions.append("建议将长句拆分为多个短句")
        
        if analysis["vocab_diversity"] < 0.3:
            issues.append("词汇使用重复性较高")
            suggestions.append("增加词汇多样性，使用同义词替换")
        
        if analysis["char_complexity"] > 0.9:
            issues.append("复杂字符使用过多")
            suggestions.append("适当使用更常见的词汇表达")
        
        return issues, suggestions
    
    async def _assess_accuracy(self, content: str, content_type: str) -> AccuracyMetric:
        """评估准确性"""
        try:
            # 基础语言分析
            grammar_score = await self._check_grammar_basic(content)
            
            # AI辅助准确性分析
            ai_analysis = None
            try:
                ai_result = await ai_service_client.analyze_accuracy(content, content_type)
                ai_analysis = json.loads(ai_result["ai_analysis"])
            except Exception as e:
                logger.warning(f"AI accuracy analysis failed: {str(e)}")
                ai_analysis = {
                    "factual_consistency": 75.0,
                    "logical_coherence": 75.0,
                    "data_accuracy": 75.0,
                    "terminology_correctness": 75.0,
                    "overall_accuracy": 75.0
                }
            
            # 综合评分
            score = (
                ai_analysis.get("overall_accuracy", 75.0) * 0.6 +
                grammar_score * 0.4
            )
            
            details = {
                "factual_consistency": ai_analysis.get("factual_consistency", 75.0),
                "grammar_accuracy": grammar_score,
                "terminology_usage": ai_analysis.get("terminology_correctness", 75.0),
                "logical_coherence": ai_analysis.get("logical_coherence", 75.0)
            }
            
            # 生成反馈
            issues = ai_analysis.get("potential_issues", [])
            suggestions = ai_analysis.get("suggestions", [])
            
            return AccuracyMetric(
                dimension=QualityDimension.ACCURACY,
                score=score,
                weight=self.dimension_weights.get(QualityDimension.ACCURACY, 0.25),
                confidence=0.80,
                details=details,
                issues=issues,
                suggestions=suggestions,
                factual_consistency=details["factual_consistency"],
                grammar_accuracy=details["grammar_accuracy"],
                terminology_usage=details["terminology_usage"]
            )
            
        except Exception as e:
            logger.error(f"Accuracy assessment failed: {str(e)}")
            return AccuracyMetric(
                dimension=QualityDimension.ACCURACY,
                score=50.0,
                weight=self.dimension_weights.get(QualityDimension.ACCURACY, 0.25),
                confidence=0.3,
                details={},
                issues=["评估失败"],
                suggestions=["请检查内容质量"]
            )
    
    async def _check_grammar_basic(self, content: str) -> float:
        """基础语法检查"""
        try:
            def grammar_check():
                doc = self.nlp(content)
                
                # 简单语法规则检查
                issues = 0
                total_tokens = 0
                
                for token in doc:
                    if not token.is_space:
                        total_tokens += 1
                        
                        # 检查基本语法问题
                        if token.pos_ == "PUNCT" and token.head.pos_ == "PUNCT":
                            issues += 1  # 连续标点
                        
                        if token.pos_ == "VERB" and token.dep_ == "ROOT":
                            # 检查主谓一致等
                            pass
                
                # 计算语法准确率
                if total_tokens == 0:
                    return 50.0
                
                error_rate = issues / total_tokens
                grammar_score = max(0, 100 - error_rate * 100)
                
                return grammar_score
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, grammar_check)
            
        except Exception as e:
            logger.error(f"Grammar check failed: {str(e)}")
            return 75.0  # 默认分数
    
    async def _assess_completeness(self, content: str, content_type: str) -> CompletenessMetric:
        """评估完整性"""
        try:
            # AI辅助完整性分析
            ai_analysis = None
            try:
                ai_result = await ai_service_client.analyze_completeness(content, content_type)
                ai_analysis = json.loads(ai_result["ai_analysis"])
            except Exception as e:
                logger.warning(f"AI completeness analysis failed: {str(e)}")
                ai_analysis = {
                    "structure_completeness": 75.0,
                    "information_completeness": 75.0,
                    "logical_completeness": 75.0,
                    "background_sufficiency": 75.0,
                    "overall_completeness": 75.0
                }
            
            # 结构分析
            structure_score = await self._analyze_content_structure(content, content_type)
            
            # 综合评分
            score = (
                ai_analysis.get("overall_completeness", 75.0) * 0.7 +
                structure_score * 0.3
            )
            
            details = {
                "structure_completeness": ai_analysis.get("structure_completeness", 75.0),
                "information_completeness": ai_analysis.get("information_completeness", 75.0),
                "logical_completeness": ai_analysis.get("logical_completeness", 75.0)
            }
            
            # 生成反馈
            issues = ai_analysis.get("missing_elements", [])
            suggestions = ai_analysis.get("suggestions", [])
            
            return CompletenessMetric(
                dimension=QualityDimension.COMPLETENESS,
                score=score,
                weight=self.dimension_weights.get(QualityDimension.COMPLETENESS, 0.20),
                confidence=0.75,
                details=details,
                issues=issues,
                suggestions=suggestions,
                structure_completeness=details["structure_completeness"],
                information_completeness=details["information_completeness"],
                logic_completeness=details["logical_completeness"]
            )
            
        except Exception as e:
            logger.error(f"Completeness assessment failed: {str(e)}")
            return CompletenessMetric(
                dimension=QualityDimension.COMPLETENESS,
                score=50.0,
                weight=self.dimension_weights.get(QualityDimension.COMPLETENESS, 0.20),
                confidence=0.3,
                details={},
                issues=["评估失败"],
                suggestions=["请检查内容结构"]
            )
    
    async def _analyze_content_structure(self, content: str, content_type: str) -> float:
        """分析内容结构"""
        try:
            # 段落分析
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            
            # 基础结构评分
            base_score = 50.0
            
            # 根据内容类型调整期望结构
            if content_type == "academic_paper":
                if paragraph_count >= 5:  # 期望有引言、正文、结论等
                    base_score += 30
                elif paragraph_count >= 3:
                    base_score += 20
            elif content_type == "historical_document":
                if paragraph_count >= 3:
                    base_score += 25
                elif paragraph_count >= 2:
                    base_score += 15
            
            # 段落长度均衡性
            if paragraph_count > 1:
                lengths = [len(p) for p in paragraphs]
                avg_length = sum(lengths) / len(lengths)
                variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
                if variance < avg_length * 0.5:  # 长度相对均衡
                    base_score += 15
            
            return min(100, base_score)
            
        except Exception as e:
            logger.error(f"Structure analysis failed: {str(e)}")
            return 50.0
    
    async def _assess_coherence(self, content: str, content_type: str) -> CoherenceMetric:
        """评估连贯性"""
        try:
            # AI辅助连贯性分析
            ai_analysis = None
            try:
                ai_result = await ai_service_client.analyze_coherence(content, content_type)
                ai_analysis = json.loads(ai_result["ai_analysis"])
            except Exception as e:
                logger.warning(f"AI coherence analysis failed: {str(e)}")
                ai_analysis = {
                    "logical_flow": 75.0,
                    "transition_quality": 75.0,
                    "argument_consistency": 75.0,
                    "narrative_smoothness": 75.0,
                    "overall_coherence": 75.0
                }
            
            # 文本连贯性分析
            coherence_score = await self._analyze_text_coherence(content)
            
            # 综合评分
            score = (
                ai_analysis.get("overall_coherence", 75.0) * 0.7 +
                coherence_score * 0.3
            )
            
            details = {
                "logical_flow": ai_analysis.get("logical_flow", 75.0),
                "transition_quality": ai_analysis.get("transition_quality", 75.0),
                "argument_consistency": ai_analysis.get("argument_consistency", 75.0),
                "narrative_coherence": ai_analysis.get("narrative_smoothness", 75.0)
            }
            
            issues = ai_analysis.get("coherence_issues", [])
            suggestions = ai_analysis.get("suggestions", [])
            
            return CoherenceMetric(
                dimension=QualityDimension.COHERENCE,
                score=score,
                weight=self.dimension_weights.get(QualityDimension.COHERENCE, 0.20),
                confidence=0.80,
                details=details,
                issues=issues,
                suggestions=suggestions,
                logical_flow=details["logical_flow"],
                transition_quality=details["transition_quality"],
                argument_consistency=details["argument_consistency"],
                narrative_coherence=details["narrative_coherence"]
            )
            
        except Exception as e:
            logger.error(f"Coherence assessment failed: {str(e)}")
            return CoherenceMetric(
                dimension=QualityDimension.COHERENCE,
                score=50.0,
                weight=self.dimension_weights.get(QualityDimension.COHERENCE, 0.20),
                confidence=0.3,
                details={},
                issues=["评估失败"],
                suggestions=["请检查内容连贯性"]
            )
    
    async def _analyze_text_coherence(self, content: str) -> float:
        """分析文本连贯性"""
        try:
            def coherence_analysis():
                # 分段分析
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                if len(paragraphs) < 2:
                    return 70.0  # 单段落默认分数
                
                # 使用TF-IDF计算段落间相似度
                vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(paragraphs)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # 计算相邻段落的平均相似度
                    adjacent_similarities = []
                    for i in range(len(paragraphs) - 1):
                        sim = similarity_matrix[i][i + 1]
                        adjacent_similarities.append(sim)
                    
                    avg_similarity = np.mean(adjacent_similarities)
                    
                    # 相似度转换为连贯性评分
                    coherence_score = min(100, avg_similarity * 200)
                    
                    return coherence_score
                    
                except Exception:
                    return 60.0  # 分析失败默认分数
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, coherence_analysis)
            
        except Exception as e:
            logger.error(f"Text coherence analysis failed: {str(e)}")
            return 60.0
    
    async def _assess_relevance(self, content: str, content_type: str, 
                              target_audience: Optional[str] = None) -> RelevanceMetric:
        """评估相关性"""
        try:
            # AI辅助相关性分析
            ai_analysis = None
            try:
                ai_result = await ai_service_client.analyze_relevance(content, content_type, target_audience)
                ai_analysis = json.loads(ai_result["ai_analysis"])
            except Exception as e:
                logger.warning(f"AI relevance analysis failed: {str(e)}")
                ai_analysis = {
                    "topic_relevance": 80.0,
                    "audience_relevance": 75.0,
                    "temporal_relevance": 75.0,
                    "practical_relevance": 70.0,
                    "overall_relevance": 75.0
                }
            
            # 主题相关性分析
            topic_score = await self._analyze_topic_relevance(content, content_type)
            
            # 综合评分
            score = (
                ai_analysis.get("overall_relevance", 75.0) * 0.8 +
                topic_score * 0.2
            )
            
            details = {
                "topic_relevance": ai_analysis.get("topic_relevance", 80.0),
                "audience_relevance": ai_analysis.get("audience_relevance", 75.0),
                "contextual_relevance": ai_analysis.get("temporal_relevance", 75.0),
                "keyword_relevance": topic_score
            }
            
            issues = ai_analysis.get("relevance_gaps", [])
            suggestions = ai_analysis.get("suggestions", [])
            
            return RelevanceMetric(
                dimension=QualityDimension.RELEVANCE,
                score=score,
                weight=self.dimension_weights.get(QualityDimension.RELEVANCE, 0.15),
                confidence=0.75,
                details=details,
                issues=issues,
                suggestions=suggestions,
                topic_relevance=details["topic_relevance"],
                audience_relevance=details["audience_relevance"],
                contextual_relevance=details["contextual_relevance"],
                keyword_relevance=details["keyword_relevance"]
            )
            
        except Exception as e:
            logger.error(f"Relevance assessment failed: {str(e)}")
            return RelevanceMetric(
                dimension=QualityDimension.RELEVANCE,
                score=50.0,
                weight=self.dimension_weights.get(QualityDimension.RELEVANCE, 0.15),
                confidence=0.3,
                details={},
                issues=["评估失败"],
                suggestions=["请检查内容相关性"]
            )
    
    async def _analyze_topic_relevance(self, content: str, content_type: str) -> float:
        """分析主题相关性"""
        try:
            def topic_analysis():
                # 简单的关键词分析
                
                # 历史文档相关关键词
                history_keywords = ['历史', '朝代', '皇帝', '战争', '文化', '传统', '古代', '时期', '年代']
                
                # 学术论文相关关键词
                academic_keywords = ['研究', '分析', '理论', '方法', '结果', '结论', '数据', '实验', '调查']
                
                # 教育内容相关关键词
                education_keywords = ['学习', '教育', '知识', '技能', '培训', '课程', '学生', '教师', '教学']
                
                keywords_map = {
                    'historical_document': history_keywords,
                    'academic_paper': academic_keywords,
                    'educational_content': education_keywords
                }
                
                relevant_keywords = keywords_map.get(content_type, history_keywords)
                
                # 计算关键词匹配度
                matches = sum(1 for keyword in relevant_keywords if keyword in content)
                relevance_score = min(100, (matches / len(relevant_keywords)) * 150)
                
                return relevance_score
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, topic_analysis)
            
        except Exception as e:
            logger.error(f"Topic relevance analysis failed: {str(e)}")
            return 70.0
    
    def _apply_custom_weights(self, metrics: List[QualityMetric], 
                            custom_weights: Dict[QualityDimension, float]) -> List[QualityMetric]:
        """应用自定义权重"""
        for metric in metrics:
            if metric.dimension in custom_weights:
                metric.weight = custom_weights[metric.dimension]
        
        return metrics
    
    def _filter_enabled_dimensions(self, metrics: List[QualityMetric],
                                 enabled_dimensions: List[QualityDimension]) -> List[QualityMetric]:
        """过滤启用的评估维度"""
        return [metric for metric in metrics if metric.dimension in enabled_dimensions]
    
    def _calculate_overall_score(self, metrics: List[QualityMetric]) -> float:
        """计算综合评分"""
        if not metrics:
            return 0.0
        
        total_weighted_score = sum(metric.score * metric.weight for metric in metrics)
        total_weight = sum(metric.weight for metric in metrics)
        
        if total_weight == 0:
            return 0.0
        
        return round(total_weighted_score / total_weight, 2)
    
    def _determine_grade(self, score: float) -> QualityGrade:
        """确定等级评定"""
        for grade, threshold in self.grade_thresholds.items():
            if score >= threshold:
                return QualityGrade(grade)
        
        return QualityGrade.F
    
    async def _generate_analysis_report(self, metrics: List[QualityMetric], 
                                      content_type: str) -> Tuple[List[str], List[str], List[str]]:
        """生成分析报告"""
        try:
            strengths = []
            weaknesses = []
            recommendations = []
            
            # 分析各维度表现
            for metric in metrics:
                if metric.score >= 85:
                    strengths.append(f"{metric.dimension.value}表现优秀 ({metric.score:.1f}分)")
                elif metric.score < 70:
                    weaknesses.append(f"{metric.dimension.value}需要改进 ({metric.score:.1f}分)")
                
                # 收集建议
                recommendations.extend(metric.suggestions)
            
            # 使用AI生成更详细的分析
            try:
                metrics_data = {
                    "metrics": [
                        {
                            "dimension": metric.dimension.value,
                            "score": metric.score,
                            "details": metric.details
                        }
                        for metric in metrics
                    ],
                    "overall_score": self._calculate_overall_score(metrics)
                }
                
                ai_result = await ai_service_client.generate_quality_summary(metrics_data, content_type)
                ai_summary = json.loads(ai_result["summary"])
                
                # 合并AI分析结果
                strengths.extend(ai_summary.get("strengths", []))
                weaknesses.extend(ai_summary.get("weaknesses", []))
                recommendations.extend(ai_summary.get("recommendations", []))
                
            except Exception as e:
                logger.warning(f"AI analysis report generation failed: {str(e)}")
            
            # 去重和限制数量
            strengths = list(set(strengths))[:5]
            weaknesses = list(set(weaknesses))[:5]
            recommendations = list(set(recommendations))[:8]
            
            return strengths, weaknesses, recommendations
            
        except Exception as e:
            logger.error(f"Analysis report generation failed: {str(e)}")
            return [], [], []
    
    # ==================== 缓存功能 ====================
    
    async def _get_cached_result(self, request: QualityAssessmentRequest) -> Optional[QualityAssessmentResult]:
        """获取缓存的评估结果"""
        if not self.redis:
            return None
        
        try:
            cache_key = self._build_cache_key(request)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                result_dict = json.loads(cached_data)
                return QualityAssessmentResult(**result_dict)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached result: {str(e)}")
            return None
    
    async def _cache_result(self, request: QualityAssessmentRequest, 
                          result: QualityAssessmentResult):
        """缓存评估结果"""
        if not self.redis:
            return
        
        try:
            cache_key = self._build_cache_key(request)
            cache_data = result.dict()
            
            # 设置缓存
            ttl = request.cache_ttl_hours * 3600 if request.cache_ttl_hours else self.cache_ttl
            await self.redis.setex(cache_key, ttl, json.dumps(cache_data, default=str))
            
            logger.debug(f"Cached assessment result: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")
    
    def _build_cache_key(self, request: QualityAssessmentRequest) -> str:
        """构建缓存键"""
        # 使用内容哈希作为缓存键的一部分
        import hashlib
        content_hash = hashlib.md5(request.content.encode()).hexdigest()[:12]
        
        cache_key = f"{settings.database.redis_key_prefix}:assessment:{content_hash}:{request.content_type}"
        
        # 包含自定义权重信息
        if request.custom_weights:
            weights_str = "_".join(f"{k.value}:{v}" for k, v in request.custom_weights.items())
            cache_key += f":{hashlib.md5(weights_str.encode()).hexdigest()[:8]}"
        
        return cache_key
    
    # ==================== 清理和关闭 ====================
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self.redis:
                await self.redis.close()
            
            if self.executor:
                self.executor.shutdown(wait=True)
            
            logger.info("Assessment engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Assessment engine cleanup failed: {str(e)}")

# 全局评估引擎实例
assessment_engine = ContentQualityAssessmentEngine()