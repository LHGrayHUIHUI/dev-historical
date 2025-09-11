"""
文本优化引擎 - Text Optimization Engine

智能文本优化服务的核心引擎，负责执行文本优化任务
集成AI模型调用、质量评估、策略管理等功能模块

核心功能:
1. 文本分析和预处理
2. AI模型调用和优化执行
3. 质量评估和版本生成
4. 优化策略应用
5. 结果处理和优化
"""

import asyncio
import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4

import jieba
import spacy
from transformers import pipeline

from ..config.settings import get_settings
from ..models.optimization_models import (
    OptimizationType, OptimizationMode, TaskStatus,
    OptimizationRequest, OptimizationResult, OptimizationVersion,
    OptimizationParameters, QualityMetrics, TokenUsage
)
from ..clients.ai_model_service_client import AIModelServiceClient
from ..clients.storage_service_client import StorageServiceClient


logger = logging.getLogger(__name__)


class TextOptimizationError(Exception):
    """文本优化错误"""
    pass


class TextAnalyzer:
    """
    文本分析器
    负责对输入文本进行各种分析，为优化提供基础信息
    """
    
    def __init__(self):
        """初始化文本分析器"""
        self.settings = get_settings()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化NLP工具
        self._init_nlp_tools()
    
    def _init_nlp_tools(self):
        """初始化NLP工具"""
        try:
            # 初始化jieba分词
            if self.settings.jieba_dict_path:
                jieba.load_userdict(self.settings.jieba_dict_path)
            
            # 初始化spaCy模型 (如果可用)
            try:
                self.nlp = spacy.load(self.settings.spacy_model)
            except OSError:
                self._logger.warning(f"无法加载spaCy模型 {self.settings.spacy_model}，将使用基础分析")
                self.nlp = None
            
            # 初始化情感分析管道 (如果需要)
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-dianping-chinese")
            except Exception:
                self._logger.warning("无法加载情感分析模型，将跳过情感分析")
                self.sentiment_analyzer = None
                
        except Exception as e:
            self._logger.error(f"初始化NLP工具失败: {e}")
    
    async def analyze_text(self, content: str) -> Dict[str, Any]:
        """
        全面分析文本
        
        Args:
            content: 文本内容
            
        Returns:
            文本分析结果
        """
        try:
            analysis = {
                'basic_stats': await self._get_basic_statistics(content),
                'complexity': await self._calculate_complexity(content),
                'style': await self._detect_writing_style(content),
                'entities': await self._extract_entities(content),
                'topics': await self._extract_topics(content),
                'language_features': await self._analyze_language_features(content),
                'readability': await self._assess_readability(content)
            }
            
            # 添加综合评估
            analysis['overall_assessment'] = await self._generate_overall_assessment(analysis)
            
            return analysis
            
        except Exception as e:
            self._logger.error(f"文本分析失败: {e}")
            # 返回基础分析结果
            return await self._get_basic_analysis(content)
    
    async def _get_basic_statistics(self, content: str) -> Dict[str, Any]:
        """获取基础统计信息"""
        words = list(jieba.cut(content))
        sentences = content.split('。')
        paragraphs = content.split('\n\n')
        
        return {
            'length': len(content),
            'character_count': len(content.replace(' ', '')),  # 不含空格字符数
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'avg_sentence_length': len(content) / max(len(sentences), 1),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1)
        }
    
    async def _calculate_complexity(self, content: str) -> float:
        """计算文本复杂度"""
        try:
            # 基于多个指标计算复杂度
            words = list(jieba.cut(content))
            sentences = content.split('。')
            
            # 句子长度复杂度
            avg_sentence_length = len(content) / max(len(sentences), 1)
            length_complexity = min(avg_sentence_length / 50, 1.0)  # 归一化到0-1
            
            # 词汇复杂度 (基于词汇多样性)
            unique_words = set(words)
            vocab_diversity = len(unique_words) / max(len(words), 1)
            vocab_complexity = min(vocab_diversity * 2, 1.0)
            
            # 古文特征复杂度
            classical_chars = set('之乎者也矣焉哉兮')
            classical_ratio = sum(1 for char in content if char in classical_chars) / max(len(content), 1)
            classical_complexity = min(classical_ratio * 10, 1.0)
            
            # 综合复杂度
            complexity = (length_complexity * 0.3 + vocab_complexity * 0.4 + classical_complexity * 0.3)
            return round(complexity, 2)
            
        except Exception as e:
            self._logger.warning(f"复杂度计算失败: {e}")
            return 0.5  # 返回中等复杂度
    
    async def _detect_writing_style(self, content: str) -> str:
        """检测写作风格"""
        try:
            # 基于关键特征检测写作风格
            
            # 文言文特征
            classical_indicators = ['之', '乎', '者', '也', '矣', '焉', '哉', '兮', '其', '於', '与', '以', '而']
            classical_count = sum(content.count(indicator) for indicator in classical_indicators)
            classical_ratio = classical_count / max(len(content), 1)
            
            # 现代文特征
            modern_indicators = ['的', '了', '是', '在', '有', '我', '你', '他', '她', '它']
            modern_count = sum(content.count(indicator) for indicator in modern_indicators)
            modern_ratio = modern_count / max(len(content), 1)
            
            # 学术文体特征
            academic_indicators = ['研究', '分析', '表明', '显示', '证明', '结果', '因此', '然而', '此外']
            academic_count = sum(content.count(indicator) for indicator in academic_indicators)
            academic_ratio = academic_count / max(len(content), 1)
            
            # 历史文献特征
            historical_indicators = ['史载', '据', '考', '云', '记', '志', '传', '纪']
            historical_count = sum(content.count(indicator) for indicator in historical_indicators)
            historical_ratio = historical_count / max(len(content), 1)
            
            # 判断主要风格
            if classical_ratio > 0.05:
                return "文言文"
            elif academic_ratio > 0.01:
                return "学术文体"
            elif historical_ratio > 0.008:
                return "史书体例"
            elif modern_ratio > 0.08:
                return "现代白话文"
            else:
                return "混合文体"
                
        except Exception as e:
            self._logger.warning(f"风格检测失败: {e}")
            return "未知风格"
    
    async def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """提取命名实体"""
        entities = []
        
        try:
            # 使用jieba进行词性标注
            import jieba.posseg as pseg
            words = pseg.cut(content)
            
            for word, flag in words:
                if len(word) > 1:  # 过滤单字词
                    if flag == 'nr':  # 人名
                        entities.append({
                            'text': word,
                            'type': '人名',
                            'confidence': 0.8
                        })
                    elif flag == 'ns':  # 地名
                        entities.append({
                            'text': word,
                            'type': '地名',
                            'confidence': 0.8
                        })
                    elif flag == 'nt':  # 时间
                        entities.append({
                            'text': word,
                            'type': '时间',
                            'confidence': 0.7
                        })
            
            # 使用spaCy进行补充识别 (如果可用)
            if self.nlp:
                doc = self.nlp(content[:1000])  # 限制长度避免性能问题
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_,
                        'confidence': 0.9
                    })
            
            # 去重
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity['text'], entity['type'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            return unique_entities[:20]  # 最多返回20个实体
            
        except Exception as e:
            self._logger.warning(f"实体提取失败: {e}")
            return []
    
    async def _extract_topics(self, content: str) -> List[str]:
        """提取文本主题"""
        try:
            # 基于关键词的简单主题提取
            keywords = []
            words = list(jieba.cut(content))
            
            # 历史主题关键词
            historical_topics = {
                '政治': ['皇帝', '朝廷', '官员', '制度', '政策', '改革', '统治'],
                '军事': ['战争', '军队', '将军', '战役', '兵力', '攻击', '防守'],
                '经济': ['农业', '商业', '贸易', '税收', '货币', '市场', '财政'],
                '文化': ['文学', '艺术', '教育', '宗教', '习俗', '传统', '礼仪'],
                '社会': ['民众', '社会', '阶层', '生活', '风俗', '人民', '百姓'],
                '地理': ['地域', '城市', '河流', '山脉', '边疆', '疆域', '地区']
            }
            
            topics = []
            for topic, topic_keywords in historical_topics.items():
                score = sum(content.count(keyword) for keyword in topic_keywords)
                if score > 0:
                    topics.append((topic, score))
            
            # 按分数排序，返回前5个主题
            topics.sort(key=lambda x: x[1], reverse=True)
            return [topic[0] for topic in topics[:5]]
            
        except Exception as e:
            self._logger.warning(f"主题提取失败: {e}")
            return ['综合']
    
    async def _analyze_language_features(self, content: str) -> Dict[str, Any]:
        """分析语言特征"""
        try:
            features = {}
            
            # 标点符号分析
            punctuation_chars = '，。；：！？""''（）【】'
            punctuation_count = sum(content.count(p) for p in punctuation_chars)
            features['punctuation_ratio'] = punctuation_count / max(len(content), 1)
            
            # 数字比例
            digit_count = sum(1 for char in content if char.isdigit())
            features['digit_ratio'] = digit_count / max(len(content), 1)
            
            # 繁体字比例 (简单检测)
            traditional_chars = '國學時間問題發現現實實際際遇運動動作'
            traditional_count = sum(content.count(char) for char in traditional_chars)
            features['traditional_ratio'] = traditional_count / max(len(content), 1)
            
            # 重复度分析
            words = list(jieba.cut(content))
            word_freq = {}
            for word in words:
                if len(word) > 1:  # 忽略单字词
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            if word_freq:
                max_freq = max(word_freq.values())
                features['max_word_frequency'] = max_freq
                features['vocabulary_diversity'] = len(word_freq) / max(len(words), 1)
            else:
                features['max_word_frequency'] = 0
                features['vocabulary_diversity'] = 0
            
            return features
            
        except Exception as e:
            self._logger.warning(f"语言特征分析失败: {e}")
            return {}
    
    async def _assess_readability(self, content: str) -> float:
        """评估可读性"""
        try:
            sentences = [s.strip() for s in content.split('。') if s.strip()]
            words = list(jieba.cut(content))
            
            if not sentences or not words:
                return 50.0
            
            # 平均句子长度 (字符数)
            avg_sentence_length = len(content) / len(sentences)
            
            # 平均词长
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # 复杂词比例 (长度大于3的词)
            complex_words = [word for word in words if len(word) > 3]
            complex_word_ratio = len(complex_words) / len(words)
            
            # 简化的可读性评分 (0-100分)
            readability = 100 - (avg_sentence_length * 0.5) - (avg_word_length * 5) - (complex_word_ratio * 20)
            
            return max(0, min(100, readability))
            
        except Exception as e:
            self._logger.warning(f"可读性评估失败: {e}")
            return 50.0  # 返回中等可读性
    
    async def _generate_overall_assessment(self, analysis: Dict[str, Any]) -> str:
        """生成综合评估"""
        try:
            basic_stats = analysis.get('basic_stats', {})
            complexity = analysis.get('complexity', 0.5)
            style = analysis.get('style', '未知')
            readability = analysis.get('readability', 50.0)
            
            length = basic_stats.get('length', 0)
            
            assessment_parts = []
            
            # 长度评估
            if length < 100:
                assessment_parts.append("文本较短")
            elif length > 1000:
                assessment_parts.append("文本较长")
            else:
                assessment_parts.append("文本长度适中")
            
            # 复杂度评估
            if complexity > 0.7:
                assessment_parts.append("表达较为复杂")
            elif complexity < 0.3:
                assessment_parts.append("表达相对简单")
            else:
                assessment_parts.append("表达复杂度适中")
            
            # 可读性评估
            if readability > 70:
                assessment_parts.append("可读性较好")
            elif readability < 40:
                assessment_parts.append("可读性有待改善")
            else:
                assessment_parts.append("可读性一般")
            
            # 风格评估
            assessment_parts.append(f"文体风格为{style}")
            
            return "，".join(assessment_parts) + "。"
            
        except Exception as e:
            self._logger.warning(f"生成综合评估失败: {e}")
            return "文本分析完成。"
    
    async def _get_basic_analysis(self, content: str) -> Dict[str, Any]:
        """获取基础分析结果（备用方案）"""
        return {
            'basic_stats': {
                'length': len(content),
                'character_count': len(content.replace(' ', '')),
                'word_count': len(list(jieba.cut(content))),
                'sentence_count': len(content.split('。')),
                'avg_sentence_length': len(content) / max(len(content.split('。')), 1)
            },
            'complexity': 0.5,
            'style': '混合文体',
            'entities': [],
            'topics': ['综合'],
            'language_features': {},
            'readability': 50.0,
            'overall_assessment': '文本分析完成。'
        }


class TextOptimizationEngine:
    """
    文本优化引擎核心类
    负责调用AI模型执行文本优化任务，集成分析、评估等功能
    """
    
    def __init__(self, ai_client: AIModelServiceClient, storage_client: StorageServiceClient):
        """
        初始化文本优化引擎
        
        Args:
            ai_client: AI模型服务客户端
            storage_client: 存储服务客户端
        """
        self.settings = get_settings()
        self.ai_client = ai_client
        self.storage_client = storage_client
        self.text_analyzer = TextAnalyzer()
        
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 缓存已分析的文本
        self._analysis_cache = {}
    
    async def optimize_text(self, request: OptimizationRequest) -> OptimizationResult:
        """
        执行文本优化
        
        Args:
            request: 优化请求
            
        Returns:
            优化结果
        """
        start_time = time.time()
        task_id = str(uuid4())
        
        try:
            self._logger.info(f"开始文本优化任务: {task_id}")
            
            # 1. 文本分析
            text_analysis = await self._get_text_analysis(request.content)
            self._logger.debug(f"文本分析完成: {text_analysis.get('overall_assessment')}")
            
            # 2. 验证优化请求
            await self._validate_optimization_request(request, text_analysis)
            
            # 3. 生成多个版本
            versions = []
            total_processing_time = 0
            
            for i in range(request.generate_versions):
                version = await self._generate_optimization_version(
                    request, text_analysis, i + 1
                )
                versions.append(version)
                total_processing_time += version.processing_time_ms
                
                self._logger.info(f"生成版本 {i + 1}/{request.generate_versions}，质量分数: {version.quality_metrics.overall_score}")
            
            # 4. 选择推荐版本
            recommended_version = self._select_recommended_version(versions)
            
            # 5. 计算统计信息
            quality_scores = [v.quality_metrics.overall_score for v in versions]
            average_quality = sum(quality_scores) / len(quality_scores)
            best_quality = max(quality_scores)
            
            # 6. 构建优化结果
            result = OptimizationResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                versions=versions,
                recommended_version=recommended_version.version_id,
                original_content=request.content,
                request_parameters=request.parameters,
                total_versions=len(versions),
                average_quality_score=round(average_quality, 2),
                best_quality_score=round(best_quality, 2),
                total_processing_time_ms=total_processing_time,
                created_at=datetime.utcnow()
            )
            
            # 7. 保存到存储服务
            if request.user_id:
                await self._save_optimization_result(result, request.user_id)
            
            elapsed_time = (time.time() - start_time) * 1000
            self._logger.info(f"文本优化完成: {task_id}，总耗时: {elapsed_time:.0f}ms")
            
            return result
            
        except Exception as e:
            self._logger.error(f"文本优化失败 (task_id={task_id}): {e}")
            raise TextOptimizationError(f"文本优化失败: {str(e)}")
    
    async def _get_text_analysis(self, content: str) -> Dict[str, Any]:
        """获取文本分析结果（带缓存）"""
        # 生成内容哈希用于缓存
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash in self._analysis_cache:
            self._logger.debug("使用缓存的文本分析结果")
            return self._analysis_cache[content_hash]
        
        # 执行文本分析
        analysis = await self.text_analyzer.analyze_text(content)
        
        # 缓存结果 (限制缓存大小)
        if len(self._analysis_cache) > 100:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
        
        self._analysis_cache[content_hash] = analysis
        return analysis
    
    async def _validate_optimization_request(
        self, 
        request: OptimizationRequest, 
        text_analysis: Dict[str, Any]
    ):
        """验证优化请求的合理性"""
        basic_stats = text_analysis.get('basic_stats', {})
        content_length = basic_stats.get('length', 0)
        
        # 检查内容长度
        if content_length > self.settings.max_content_length:
            raise TextOptimizationError(
                f"文本长度超出限制: {content_length} > {self.settings.max_content_length}"
            )
        
        # 检查内容质量
        if content_length < 10:
            raise TextOptimizationError("文本内容过短，无法进行有效优化")
        
        # 检查优化参数
        params = request.parameters
        if params.target_length and params.target_length > content_length * 3:
            self._logger.warning("目标长度可能过长，优化效果可能受影响")
        
        if params.quality_threshold > 95:
            self._logger.warning("质量阈值设置过高，可能导致优化失败")
    
    async def _generate_optimization_version(
        self,
        request: OptimizationRequest,
        text_analysis: Dict[str, Any],
        version_number: int
    ) -> OptimizationVersion:
        """生成单个优化版本"""
        start_time = time.time()
        
        try:
            # 1. 选择最佳AI模型
            model_name = await self.ai_client.select_best_model(
                optimization_type=request.optimization_type,
                content_length=len(request.content)
            )
            
            # 2. 生成优化提示词
            prompts = await self.ai_client.generate_optimization_prompt(
                original_content=request.content,
                optimization_type=request.optimization_type,
                optimization_mode=request.optimization_mode,
                text_analysis=text_analysis,
                custom_instructions=request.parameters.custom_instructions
            )
            
            # 3. 调用AI模型执行优化
            optimization_result = await self.ai_client.optimize_text_with_ai(
                content=request.content,
                system_prompt=prompts["system_prompt"],
                user_prompt=prompts["user_prompt"],
                model=model_name,
                temperature=request.parameters.temperature or 0.7,
                max_tokens=request.parameters.max_tokens
            )
            
            optimized_content = optimization_result["optimized_content"]
            processing_time = int((time.time() - start_time) * 1000)
            
            # 4. 质量评估
            quality_metrics = await self._assess_quality(
                original_text=request.content,
                optimized_text=optimized_content,
                optimization_type=request.optimization_type,
                text_analysis=text_analysis
            )
            
            # 5. 生成改进说明
            improvements = await self._generate_improvement_summary(
                original=request.content,
                optimized=optimized_content,
                metrics=quality_metrics,
                optimization_type=request.optimization_type
            )
            
            # 6. 生成标题和摘要
            title = await self._generate_title(optimized_content)
            summary = await self._generate_summary(optimized_content)
            
            # 7. 构建版本对象
            version = OptimizationVersion(
                version_number=version_number,
                content=optimized_content,
                title=title,
                summary=summary,
                quality_metrics=quality_metrics,
                improvements=improvements,
                model_used=optimization_result["model_used"],
                processing_time_ms=processing_time,
                token_usage=TokenUsage(**optimization_result["token_usage"]),
                is_selected=(version_number == 1),  # 默认选择第一个版本
                created_at=datetime.utcnow()
            )
            
            return version
            
        except Exception as e:
            self._logger.error(f"生成优化版本失败 (version {version_number}): {e}")
            raise TextOptimizationError(f"生成优化版本失败: {str(e)}")
    
    async def _assess_quality(
        self,
        original_text: str,
        optimized_text: str,
        optimization_type: OptimizationType,
        text_analysis: Dict[str, Any]
    ) -> QualityMetrics:
        """评估优化质量"""
        try:
            # 基础质量评估
            readability_score = await self.text_analyzer._assess_readability(optimized_text)
            original_readability = text_analysis.get('readability', 50.0)
            readability_improvement = readability_score - original_readability
            
            # 学术规范性评估
            academic_score = await self._assess_academic_quality(optimized_text)
            
            # 历史准确性评估
            historical_accuracy = await self._assess_historical_accuracy(
                original_text, optimized_text
            )
            
            # 语言质量评估
            language_quality = await self._assess_language_quality(optimized_text)
            
            # 结构质量评估
            structure_score = await self._assess_structure_quality(optimized_text)
            
            # 内容完整性评估
            content_completeness = await self._assess_content_completeness(
                original_text, optimized_text
            )
            
            # 计算综合评分 (根据优化类型调整权重)
            weights = self._get_quality_weights(optimization_type)
            overall_score = (
                readability_score * weights['readability'] +
                academic_score * weights['academic'] +
                historical_accuracy * weights['historical'] +
                language_quality * weights['language'] +
                structure_score * weights['structure'] +
                content_completeness * weights['completeness']
            )
            
            # 分析优势和改进建议
            strengths, weaknesses = await self._analyze_quality_aspects(
                optimized_text, {
                    'readability': readability_score,
                    'academic': academic_score,
                    'historical_accuracy': historical_accuracy,
                    'language_quality': language_quality,
                    'structure': structure_score,
                    'completeness': content_completeness
                }
            )
            
            return QualityMetrics(
                overall_score=round(overall_score, 2),
                readability_score=round(readability_score, 2),
                academic_score=round(academic_score, 2),
                historical_accuracy=round(historical_accuracy, 2),
                language_quality=round(language_quality, 2),
                structure_score=round(structure_score, 2),
                content_completeness=round(content_completeness, 2),
                readability_improvement=round(readability_improvement, 2),
                strengths=strengths,
                weaknesses=weaknesses
            )
            
        except Exception as e:
            self._logger.error(f"质量评估失败: {e}")
            # 返回默认质量指标
            return QualityMetrics(
                overall_score=75.0,
                readability_score=75.0,
                academic_score=75.0,
                historical_accuracy=75.0,
                language_quality=75.0,
                structure_score=75.0,
                content_completeness=75.0,
                strengths=["优化完成"],
                weaknesses=["需要人工验证"]
            )
    
    def _get_quality_weights(self, optimization_type: OptimizationType) -> Dict[str, float]:
        """根据优化类型获取质量权重"""
        base_weights = {
            'readability': 0.2,
            'academic': 0.2,
            'historical': 0.2,
            'language': 0.2,
            'structure': 0.1,
            'completeness': 0.1
        }
        
        # 根据优化类型调整权重
        if optimization_type == OptimizationType.POLISH:
            base_weights.update({
                'language': 0.3,
                'readability': 0.25,
                'academic': 0.25
            })
        elif optimization_type == OptimizationType.EXPAND:
            base_weights.update({
                'completeness': 0.3,
                'historical': 0.25,
                'structure': 0.2
            })
        elif optimization_type == OptimizationType.MODERNIZE:
            base_weights.update({
                'readability': 0.35,
                'language': 0.3,
                'historical': 0.15
            })
        
        return base_weights
    
    async def _assess_academic_quality(self, text: str) -> float:
        """评估学术质量"""
        # 学术词汇和表达
        academic_indicators = [
            '研究', '分析', '表明', '显示', '证明', '结果', '因此', '然而', '此外',
            '根据', '依据', '基于', '通过', '采用', '运用', '考察', '探讨',
            '史料', '文献', '记载', '考证', '学者', '专家', '研究者'
        ]
        
        score = 60  # 基础分数
        
        for indicator in academic_indicators:
            if indicator in text:
                score += min(text.count(indicator) * 2, 5)  # 每个指标最多加5分
        
        return min(score, 100)
    
    async def _assess_historical_accuracy(self, original: str, optimized: str) -> float:
        """评估历史准确性"""
        try:
            # 提取关键历史信息
            original_entities = await self._extract_key_entities(original)
            optimized_entities = await self._extract_key_entities(optimized)
            
            if not original_entities:
                return 100.0  # 如果原文没有关键实体，认为保持完好
            
            # 计算实体保持率
            preserved_count = 0
            for entity in original_entities:
                if any(entity['text'] in opt_entity['text'] or opt_entity['text'] in entity['text'] 
                      for opt_entity in optimized_entities):
                    preserved_count += 1
            
            preservation_rate = preserved_count / len(original_entities)
            return preservation_rate * 100
            
        except Exception as e:
            self._logger.warning(f"历史准确性评估失败: {e}")
            return 85.0  # 返回较高的默认分数
    
    async def _extract_key_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取关键历史实体"""
        # 这里复用文本分析器的实体提取功能
        entities = await self.text_analyzer._extract_entities(text)
        
        # 过滤出重要的历史实体
        key_entities = []
        for entity in entities:
            if entity['type'] in ['人名', '地名', '时间'] and entity['confidence'] > 0.7:
                key_entities.append(entity)
        
        return key_entities
    
    async def _assess_language_quality(self, text: str) -> float:
        """评估语言质量"""
        score = 70  # 基础分数
        
        # 语法正确性 (简单检测)
        if '的的' not in text and '了了' not in text:
            score += 5
        
        # 表达流畅性
        sentences = text.split('。')
        avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_length <= 50:  # 适中的句子长度
            score += 10
        
        # 用词准确性 (基于重复词检测)
        words = list(jieba.cut(text))
        word_freq = {}
        for word in words:
            if len(word) > 1:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 检查词汇多样性
        if word_freq:
            max_freq = max(word_freq.values())
            if max_freq <= len(words) * 0.1:  # 没有过度重复的词
                score += 10
        
        return min(score, 100)
    
    async def _assess_structure_quality(self, text: str) -> float:
        """评估结构质量"""
        score = 70
        
        # 段落结构
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if 1 <= len(paragraphs) <= 5:  # 合理的段落数量
            score += 10
        
        # 逻辑连接词
        connectives = ['因此', '然而', '此外', '同时', '另外', '首先', '其次', '最后']
        connective_count = sum(text.count(conn) for conn in connectives)
        if connective_count > 0:
            score += min(connective_count * 3, 15)
        
        return min(score, 100)
    
    async def _assess_content_completeness(self, original: str, optimized: str) -> float:
        """评估内容完整性"""
        try:
            # 基于长度比例的简单评估
            length_ratio = len(optimized) / max(len(original), 1)
            
            # 理想的长度比例范围
            if 0.8 <= length_ratio <= 1.5:
                base_score = 90
            elif 0.6 <= length_ratio <= 2.0:
                base_score = 80
            else:
                base_score = 70
            
            # 基于关键词保持率
            original_words = set(jieba.cut(original))
            optimized_words = set(jieba.cut(optimized))
            
            # 过滤停用词
            stopwords = {'的', '了', '在', '是', '有', '和', '与', '或', '但', '等'}
            original_words = original_words - stopwords
            optimized_words = optimized_words - stopwords
            
            if original_words:
                keyword_retention = len(original_words & optimized_words) / len(original_words)
                retention_score = keyword_retention * 100
            else:
                retention_score = 100
            
            # 综合评分
            final_score = (base_score * 0.3 + retention_score * 0.7)
            return min(final_score, 100)
            
        except Exception as e:
            self._logger.warning(f"内容完整性评估失败: {e}")
            return 80.0
    
    async def _analyze_quality_aspects(
        self, 
        text: str, 
        scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """分析质量各方面的优势和不足"""
        strengths = []
        weaknesses = []
        
        # 根据各项分数分析
        for aspect, score in scores.items():
            if score >= 85:
                if aspect == 'readability':
                    strengths.append("文本可读性优秀")
                elif aspect == 'academic':
                    strengths.append("学术规范性良好")
                elif aspect == 'historical_accuracy':
                    strengths.append("历史准确性保持很好")
                elif aspect == 'language_quality':
                    strengths.append("语言表达质量高")
                elif aspect == 'structure':
                    strengths.append("文本结构清晰")
                elif aspect == 'completeness':
                    strengths.append("内容完整性好")
            elif score < 60:
                if aspect == 'readability':
                    weaknesses.append("可读性有待提升")
                elif aspect == 'academic':
                    weaknesses.append("学术规范性需要加强")
                elif aspect == 'historical_accuracy':
                    weaknesses.append("部分历史信息可能有变化")
                elif aspect == 'language_quality':
                    weaknesses.append("语言表达需要改进")
                elif aspect == 'structure':
                    weaknesses.append("文本结构需要优化")
                elif aspect == 'completeness':
                    weaknesses.append("内容完整性需要关注")
        
        # 如果没有明显的优势和不足，给出通用评价
        if not strengths:
            strengths.append("文本优化完成")
        if not weaknesses:
            weaknesses.append("整体质量良好")
        
        return strengths, weaknesses
    
    async def _generate_improvement_summary(
        self,
        original: str,
        optimized: str,
        metrics: QualityMetrics,
        optimization_type: OptimizationType
    ) -> List[str]:
        """生成改进说明"""
        improvements = []
        
        # 根据优化类型生成对应的改进说明
        if optimization_type == OptimizationType.POLISH:
            if metrics.language_quality > 80:
                improvements.append("显著改善了语言表达的流畅性和准确性")
            if metrics.academic_score > 80:
                improvements.append("提升了文本的学术规范性")
            if metrics.readability_improvement and metrics.readability_improvement > 5:
                improvements.append("明显提高了文本的可读性")
                
        elif optimization_type == OptimizationType.EXPAND:
            if len(optimized) > len(original) * 1.2:
                improvements.append("合理扩展了内容，增加了相关细节和背景信息")
            if metrics.content_completeness > 85:
                improvements.append("保持了内容的完整性和一致性")
            if metrics.structure_score > 80:
                improvements.append("改善了文本的整体结构")
                
        elif optimization_type == OptimizationType.STYLE_CONVERT:
            improvements.append("成功转换了文本风格")
            if metrics.readability_score > 80:
                improvements.append("提升了目标读者的阅读体验")
            if metrics.historical_accuracy > 90:
                improvements.append("保持了历史信息的准确性")
                
        elif optimization_type == OptimizationType.MODERNIZE:
            if metrics.readability_improvement and metrics.readability_improvement > 10:
                improvements.append("显著提升了现代读者的理解度")
            improvements.append("成功将古文表达转换为现代汉语")
            if metrics.historical_accuracy > 85:
                improvements.append("保留了重要的历史术语和核心内容")
        
        # 通用改进说明
        if metrics.overall_score > 85:
            improvements.append("整体优化质量优秀")
        elif not improvements:
            improvements.append("完成了文本的基础优化")
        
        return improvements
    
    async def _generate_title(self, content: str) -> Optional[str]:
        """生成优化后文本的标题"""
        try:
            # 简单的标题生成逻辑
            sentences = [s.strip() for s in content.split('。') if s.strip()]
            if sentences:
                # 取第一句作为标题的基础
                first_sentence = sentences[0]
                if len(first_sentence) > 20:
                    # 如果第一句太长，尝试提取关键部分
                    words = list(jieba.cut(first_sentence))
                    if len(words) > 5:
                        title = ''.join(words[:5]) + '...'
                    else:
                        title = first_sentence[:20] + '...'
                else:
                    title = first_sentence
                
                return title
            
            return None
            
        except Exception as e:
            self._logger.warning(f"标题生成失败: {e}")
            return None
    
    async def _generate_summary(self, content: str) -> Optional[str]:
        """生成优化后文本的摘要"""
        try:
            # 简单的摘要生成逻辑
            if len(content) <= 100:
                return content
            
            # 取前100个字符作为摘要
            summary = content[:100]
            
            # 确保在句号处截断
            last_period = summary.rfind('。')
            if last_period > 50:  # 确保摘要有足够的长度
                summary = summary[:last_period + 1]
            else:
                summary = summary + '...'
            
            return summary
            
        except Exception as e:
            self._logger.warning(f"摘要生成失败: {e}")
            return None
    
    def _select_recommended_version(self, versions: List[OptimizationVersion]) -> OptimizationVersion:
        """选择推荐版本"""
        if not versions:
            raise TextOptimizationError("没有可用的优化版本")
        
        # 根据综合质量分数选择最佳版本
        best_version = max(versions, key=lambda v: v.quality_metrics.overall_score)
        
        # 更新选择状态
        for version in versions:
            version.is_selected = (version.version_id == best_version.version_id)
        
        return best_version
    
    async def _save_optimization_result(self, result: OptimizationResult, user_id: str):
        """保存优化结果到存储服务"""
        try:
            # 准备任务数据
            task_data = {
                'task_id': result.task_id,
                'user_id': user_id,
                'optimization_type': 'unknown',  # 需要从请求中获取
                'optimization_mode': 'unknown',
                'status': result.status.value,
                'total_versions': result.total_versions,
                'average_quality_score': result.average_quality_score,
                'best_quality_score': result.best_quality_score,
                'total_processing_time_ms': result.total_processing_time_ms
            }
            
            # 创建优化任务记录
            await self.storage_client.create_optimization_task(task_data)
            
            # 保存每个版本
            for version in result.versions:
                version_data = {
                    'task_id': result.task_id,
                    'version_id': version.version_id,
                    'version_number': version.version_number,
                    'content': version.content,
                    'title': version.title,
                    'quality_metrics': version.quality_metrics.dict(),
                    'improvements': version.improvements,
                    'model_used': version.model_used,
                    'processing_time_ms': version.processing_time_ms,
                    'token_usage': version.token_usage.dict(),
                    'is_selected': version.is_selected
                }
                
                await self.storage_client.save_optimization_version(version_data)
            
            self._logger.info(f"优化结果已保存: {result.task_id}")
            
        except Exception as e:
            self._logger.error(f"保存优化结果失败: {e}")
            # 不抛出异常，避免影响主要流程