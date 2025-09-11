"""
优化策略管理器 - Optimization Strategy Manager

负责管理和选择文本优化策略，根据文本特征和优化需求
自动选择最适合的优化策略和参数配置

核心功能:
1. 策略配置管理和加载
2. 智能策略选择算法
3. 策略性能统计和优化
4. 动态策略调整和学习
5. 用户偏好整合
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..config.settings import get_settings
from ..models.optimization_models import (
    OptimizationType, OptimizationMode, OptimizationStrategy,
    OptimizationParameters
)
from ..clients.storage_service_client import StorageServiceClient


logger = logging.getLogger(__name__)


@dataclass
class StrategyScore:
    """策略评分结果"""
    strategy_id: str
    strategy_name: str
    score: float
    confidence: float
    reasons: List[str]


class StrategySelectionError(Exception):
    """策略选择错误"""
    pass


class TextFeatureAnalyzer:
    """
    文本特征分析器
    为策略选择提供文本特征分析
    """
    
    def __init__(self):
        """初始化文本特征分析器"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze_text_features(self, content: str, text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析文本特征用于策略选择
        
        Args:
            content: 文本内容
            text_analysis: 已有的文本分析结果
            
        Returns:
            策略选择相关的特征字典
        """
        try:
            features = {
                # 基础特征
                'length': len(content),
                'complexity': text_analysis.get('complexity', 0.5),
                'style': text_analysis.get('style', '未知'),
                'readability': text_analysis.get('readability', 50.0),
                
                # 内容类型特征
                'text_type': await self._classify_text_type(content),
                'historical_period': await self._identify_historical_period(content),
                'domain': await self._identify_domain(content),
                
                # 语言特征
                'language_level': await self._assess_language_level(content),
                'formality': await self._assess_formality(content),
                'technical_level': await self._assess_technical_level(content),
                
                # 结构特征
                'structure_type': await self._identify_structure_type(content),
                'has_dialogue': '："' in content or '"' in content,
                'has_citations': any(keyword in content for keyword in ['据', '史载', '记录']),
                
                # 质量特征
                'current_quality': await self._estimate_current_quality(content, text_analysis),
                'improvement_potential': await self._estimate_improvement_potential(content),
            }
            
            return features
            
        except Exception as e:
            self._logger.error(f"文本特征分析失败: {e}")
            return self._get_default_features(content)
    
    async def _classify_text_type(self, content: str) -> str:
        """分类文本类型"""
        # 史书类型特征
        historical_indicators = ['史载', '记录', '传记', '志', '纪', '表']
        if any(indicator in content for indicator in historical_indicators):
            return '史书'
        
        # 文学类型特征
        literary_indicators = ['诗', '词', '赋', '文', '记', '序']
        if any(indicator in content for indicator in literary_indicators):
            return '文学'
        
        # 政治文献特征
        political_indicators = ['诏书', '奏疏', '诰', '令', '制']
        if any(indicator in content for indicator in political_indicators):
            return '政治文献'
        
        # 学术文献特征
        academic_indicators = ['研究', '考证', '论', '说', '辨']
        if any(indicator in content for indicator in academic_indicators):
            return '学术文献'
        
        return '综合'
    
    async def _identify_historical_period(self, content: str) -> str:
        """识别历史时期"""
        period_keywords = {
            '先秦': ['春秋', '战国', '周', '秦'],
            '汉代': ['汉', '西汉', '东汉'],
            '魏晋': ['魏', '晋', '三国'],
            '南北朝': ['南北朝', '南朝', '北朝'],
            '隋唐': ['隋', '唐'],
            '宋代': ['宋', '北宋', '南宋'],
            '元代': ['元', '蒙古'],
            '明代': ['明'],
            '清代': ['清']
        }
        
        for period, keywords in period_keywords.items():
            if any(keyword in content for keyword in keywords):
                return period
        
        return '综合时期'
    
    async def _identify_domain(self, content: str) -> str:
        """识别领域类别"""
        domain_keywords = {
            '政治': ['皇帝', '朝廷', '官员', '政策', '制度'],
            '军事': ['战争', '军队', '将军', '战役', '兵'],
            '经济': ['农业', '商业', '贸易', '税收', '货币'],
            '文化': ['文学', '艺术', '教育', '宗教', '礼仪'],
            '社会': ['民众', '社会', '生活', '风俗', '习俗'],
            '地理': ['地域', '城市', '河流', '山脉', '疆域']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(content.count(keyword) for keyword in keywords)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return '综合'
    
    async def _assess_language_level(self, content: str) -> str:
        """评估语言水平"""
        # 文言文特征
        classical_chars = set('之乎者也矣焉哉兮於与')
        classical_count = sum(1 for char in content if char in classical_chars)
        classical_ratio = classical_count / max(len(content), 1)
        
        if classical_ratio > 0.05:
            return '文言文'
        elif classical_ratio > 0.02:
            return '半文言'
        else:
            return '白话文'
    
    async def _assess_formality(self, content: str) -> str:
        """评估正式程度"""
        formal_indicators = ['进行', '实施', '执行', '开展', '据此', '依据', '基于']
        informal_indicators = ['很', '特别', '非常', '挺', '蛮']
        
        formal_count = sum(content.count(word) for word in formal_indicators)
        informal_count = sum(content.count(word) for word in informal_indicators)
        
        if formal_count > informal_count * 2:
            return '正式'
        elif informal_count > formal_count:
            return '非正式'
        else:
            return '中等'
    
    async def _assess_technical_level(self, content: str) -> str:
        """评估专业技术水平"""
        technical_indicators = ['制度', '体系', '机制', '原理', '方法', '理论']
        technical_count = sum(content.count(word) for word in technical_indicators)
        
        if technical_count > len(content) * 0.01:
            return '高技术'
        elif technical_count > len(content) * 0.005:
            return '中技术'
        else:
            return '低技术'
    
    async def _identify_structure_type(self, content: str) -> str:
        """识别结构类型"""
        # 检查段落结构
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return '单段落'
        elif len(paragraphs) <= 3:
            return '简单结构'
        else:
            return '复杂结构'
    
    async def _estimate_current_quality(self, content: str, text_analysis: Dict[str, Any]) -> float:
        """估算当前质量水平"""
        # 基于现有分析结果估算质量
        readability = text_analysis.get('readability', 50.0)
        complexity = text_analysis.get('complexity', 0.5)
        
        # 简单的质量评估算法
        quality = readability * 0.6 + (1 - complexity) * 40 * 0.4
        return min(max(quality, 0), 100)
    
    async def _estimate_improvement_potential(self, content: str) -> float:
        """估算改进潜力"""
        # 基于内容特征估算改进空间
        length = len(content)
        
        # 较短文本改进潜力较大
        if length < 100:
            return 90.0
        elif length < 500:
            return 75.0
        elif length < 1000:
            return 60.0
        else:
            return 45.0
    
    def _get_default_features(self, content: str) -> Dict[str, Any]:
        """获取默认特征"""
        return {
            'length': len(content),
            'complexity': 0.5,
            'style': '未知',
            'readability': 50.0,
            'text_type': '综合',
            'historical_period': '综合时期',
            'domain': '综合',
            'language_level': '白话文',
            'formality': '中等',
            'technical_level': '低技术',
            'structure_type': '简单结构',
            'has_dialogue': False,
            'has_citations': False,
            'current_quality': 50.0,
            'improvement_potential': 60.0
        }


class StrategySelector:
    """
    策略选择器
    基于文本特征和优化需求选择最佳策略
    """
    
    def __init__(self):
        """初始化策略选择器"""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 策略选择规则权重
        self.feature_weights = {
            'text_type_match': 0.25,
            'optimization_suitability': 0.30,
            'language_level_match': 0.15,
            'domain_expertise': 0.10,
            'historical_accuracy': 0.20
        }
    
    async def score_strategy(
        self,
        strategy: OptimizationStrategy,
        text_features: Dict[str, Any],
        optimization_type: OptimizationType,
        optimization_mode: OptimizationMode,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> StrategyScore:
        """
        为策略评分
        
        Args:
            strategy: 待评分策略
            text_features: 文本特征
            optimization_type: 优化类型
            optimization_mode: 优化模式
            user_preferences: 用户偏好
            
        Returns:
            策略评分结果
        """
        try:
            scores = {}
            reasons = []
            
            # 1. 文本类型匹配度
            type_score, type_reason = await self._score_text_type_match(
                strategy, text_features['text_type']
            )
            scores['text_type_match'] = type_score
            if type_reason:
                reasons.append(type_reason)
            
            # 2. 优化适用性
            opt_score, opt_reason = await self._score_optimization_suitability(
                strategy, optimization_type, optimization_mode
            )
            scores['optimization_suitability'] = opt_score
            if opt_reason:
                reasons.append(opt_reason)
            
            # 3. 语言水平匹配
            lang_score, lang_reason = await self._score_language_level_match(
                strategy, text_features['language_level']
            )
            scores['language_level_match'] = lang_score
            if lang_reason:
                reasons.append(lang_reason)
            
            # 4. 领域专业性
            domain_score, domain_reason = await self._score_domain_expertise(
                strategy, text_features['domain']
            )
            scores['domain_expertise'] = domain_score
            if domain_reason:
                reasons.append(domain_reason)
            
            # 5. 历史准确性要求
            accuracy_score, accuracy_reason = await self._score_historical_accuracy(
                strategy, text_features['historical_period']
            )
            scores['historical_accuracy'] = accuracy_score
            if accuracy_reason:
                reasons.append(accuracy_reason)
            
            # 6. 用户偏好调整
            preference_adjustment = 0.0
            if user_preferences:
                preference_adjustment = await self._apply_user_preferences(
                    strategy, user_preferences
                )
            
            # 计算总分
            weighted_score = sum(
                scores[key] * self.feature_weights[key] 
                for key in scores
            )
            
            # 应用用户偏好调整
            final_score = min(max(weighted_score + preference_adjustment, 0), 100)
            
            # 计算置信度
            confidence = await self._calculate_confidence(scores, text_features)
            
            return StrategyScore(
                strategy_id=strategy.strategy_id,
                strategy_name=strategy.name,
                score=round(final_score, 2),
                confidence=round(confidence, 2),
                reasons=reasons
            )
            
        except Exception as e:
            self._logger.error(f"策略评分失败: {e}")
            return StrategyScore(
                strategy_id=strategy.strategy_id,
                strategy_name=strategy.name,
                score=50.0,
                confidence=0.5,
                reasons=["评分失败，使用默认分数"]
            )
    
    async def _score_text_type_match(self, strategy: OptimizationStrategy, text_type: str) -> Tuple[float, str]:
        """评分文本类型匹配度"""
        target_types = strategy.target_text_types
        
        if not target_types or text_type in target_types:
            return 100.0, f"策略适用于{text_type}类型文本"
        
        # 检查相似类型
        type_similarities = {
            '史书': ['学术文献', '政治文献'],
            '文学': ['综合'],
            '政治文献': ['史书', '学术文献'],
            '学术文献': ['史书', '政治文献'],
            '综合': ['史书', '文学', '政治文献', '学术文献']
        }
        
        similar_types = type_similarities.get(text_type, [])
        for target_type in target_types:
            if target_type in similar_types:
                return 75.0, f"策略部分适用于{text_type}类型"
        
        return 30.0, f"策略与{text_type}类型匹配度较低"
    
    async def _score_optimization_suitability(
        self, 
        strategy: OptimizationStrategy, 
        opt_type: OptimizationType, 
        opt_mode: OptimizationMode
    ) -> Tuple[float, str]:
        """评分优化适用性"""
        # 检查策略是否直接匹配优化类型和模式
        if (strategy.optimization_type == opt_type and 
            strategy.optimization_mode == opt_mode):
            return 100.0, f"策略专为{opt_type.value}+{opt_mode.value}设计"
        
        # 检查优化类型匹配
        if strategy.optimization_type == opt_type:
            return 80.0, f"策略适用于{opt_type.value}优化"
        
        # 检查模式兼容性
        mode_compatibility = {
            OptimizationMode.HISTORICAL_FORMAT: [OptimizationMode.ACADEMIC],
            OptimizationMode.ACADEMIC: [OptimizationMode.HISTORICAL_FORMAT, OptimizationMode.LITERARY],
            OptimizationMode.LITERARY: [OptimizationMode.SIMPLIFIED],
            OptimizationMode.SIMPLIFIED: [OptimizationMode.LITERARY]
        }
        
        compatible_modes = mode_compatibility.get(opt_mode, [])
        if strategy.optimization_mode in compatible_modes:
            return 60.0, f"策略与{opt_mode.value}模式部分兼容"
        
        return 40.0, "策略与所需优化类型匹配度较低"
    
    async def _score_language_level_match(self, strategy: OptimizationStrategy, lang_level: str) -> Tuple[float, str]:
        """评分语言水平匹配"""
        # 从策略参数中获取目标语言水平
        strategy_params = strategy.model_parameters or {}
        target_lang_level = strategy_params.get('target_language_level', '白话文')
        
        if lang_level == target_lang_level:
            return 100.0, f"策略适合{lang_level}水平"
        
        # 语言水平相似性
        level_similarities = {
            '文言文': ['半文言'],
            '半文言': ['文言文', '白话文'],
            '白话文': ['半文言']
        }
        
        similar_levels = level_similarities.get(lang_level, [])
        if target_lang_level in similar_levels:
            return 75.0, f"策略与{lang_level}水平部分兼容"
        
        return 50.0, "语言水平匹配度一般"
    
    async def _score_domain_expertise(self, strategy: OptimizationStrategy, domain: str) -> Tuple[float, str]:
        """评分领域专业性"""
        # 从策略标签中获取领域信息
        tags = strategy.tags or {}
        strategy_domains = tags.get('domains', [])
        
        if domain in strategy_domains:
            return 100.0, f"策略专长于{domain}领域"
        
        # 检查相关领域
        related_domains = {
            '政治': ['社会', '军事'],
            '军事': ['政治'],
            '经济': ['社会'],
            '文化': ['社会'],
            '社会': ['政治', '经济', '文化'],
            '地理': ['政治', '军事']
        }
        
        related = related_domains.get(domain, [])
        for strategy_domain in strategy_domains:
            if strategy_domain in related:
                return 75.0, f"策略在{strategy_domain}领域的经验适用于{domain}"
        
        return 60.0, "领域匹配度一般"
    
    async def _score_historical_accuracy(self, strategy: OptimizationStrategy, historical_period: str) -> Tuple[float, str]:
        """评分历史准确性要求"""
        # 从策略配置中获取历史准确性权重
        quality_weights = strategy.quality_weights or {}
        accuracy_weight = quality_weights.get('historical_accuracy', 0.2)
        
        if accuracy_weight >= 0.3:
            return 100.0, "策略高度重视历史准确性"
        elif accuracy_weight >= 0.2:
            return 80.0, "策略较为重视历史准确性"
        else:
            return 60.0, "策略对历史准确性要求一般"
    
    async def _apply_user_preferences(self, strategy: OptimizationStrategy, preferences: Dict[str, Any]) -> float:
        """应用用户偏好调整"""
        adjustment = 0.0
        
        # 偏好的策略类型
        preferred_strategies = preferences.get('preferred_strategies', [])
        if strategy.strategy_id in preferred_strategies:
            adjustment += 10.0
        
        # 质量阈值偏好
        quality_thresholds = preferences.get('quality_thresholds', {})
        for metric, threshold in quality_thresholds.items():
            strategy_threshold = strategy.quality_thresholds.get(metric, 0)
            if abs(strategy_threshold - threshold) < 10:
                adjustment += 2.0
        
        # 风格偏好
        style_preferences = preferences.get('style_preferences', {})
        strategy_style = strategy.tags.get('style', {}) if strategy.tags else {}
        
        for pref_key, pref_value in style_preferences.items():
            if strategy_style.get(pref_key) == pref_value:
                adjustment += 5.0
        
        return adjustment
    
    async def _calculate_confidence(self, scores: Dict[str, float], text_features: Dict[str, Any]) -> float:
        """计算置信度"""
        # 基于评分分布计算置信度
        score_values = list(scores.values())
        if not score_values:
            return 0.5
        
        avg_score = sum(score_values) / len(score_values)
        score_variance = sum((s - avg_score) ** 2 for s in score_values) / len(score_values)
        
        # 分数越高且方差越小，置信度越高
        base_confidence = min(avg_score / 100, 1.0)
        variance_penalty = min(score_variance / 1000, 0.3)
        
        # 文本特征完整性影响置信度
        feature_completeness = min(len(text_features) / 10, 1.0)
        
        confidence = (base_confidence - variance_penalty) * feature_completeness
        return max(min(confidence, 1.0), 0.1)


class OptimizationStrategyManager:
    """
    优化策略管理器主类
    统一管理策略加载、选择和统计更新
    """
    
    def __init__(self, storage_client: StorageServiceClient):
        """
        初始化策略管理器
        
        Args:
            storage_client: 存储服务客户端
        """
        self.settings = get_settings()
        self.storage_client = storage_client
        self.feature_analyzer = TextFeatureAnalyzer()
        self.strategy_selector = StrategySelector()
        
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 策略缓存
        self._strategies_cache: Dict[str, OptimizationStrategy] = {}
        self._cache_updated_at: Optional[datetime] = None
        
        # 默认策略
        self._default_strategies = {}
        
        # 初始化默认策略
        asyncio.create_task(self._init_default_strategies())
    
    async def load_strategies(self, force_reload: bool = False):
        """
        加载优化策略
        
        Args:
            force_reload: 是否强制重新加载
        """
        try:
            # 检查缓存有效性
            if (not force_reload and self._strategies_cache and 
                self._cache_updated_at and 
                (datetime.utcnow() - self._cache_updated_at).total_seconds() < 300):  # 5分钟缓存
                return
            
            self._logger.info("加载优化策略配置")
            
            # 从存储服务获取策略
            try:
                strategies_response = await self.storage_client.get_optimization_strategies(
                    active_only=True
                )
                
                strategies = strategies_response.get('data', [])
                
                # 更新缓存
                self._strategies_cache = {}
                for strategy_data in strategies:
                    strategy = OptimizationStrategy(**strategy_data)
                    self._strategies_cache[strategy.strategy_id] = strategy
                
                self._cache_updated_at = datetime.utcnow()
                
                self._logger.info(f"成功加载 {len(self._strategies_cache)} 个策略")
                
            except Exception as e:
                self._logger.warning(f"从存储服务加载策略失败: {e}")
                # 使用默认策略
                if not self._strategies_cache:
                    await self._load_default_strategies()
        
        except Exception as e:
            self._logger.error(f"加载策略失败: {e}")
            # 确保有可用的默认策略
            if not self._strategies_cache:
                await self._load_default_strategies()
    
    async def select_strategy(
        self,
        text_analysis: Dict[str, Any],
        optimization_type: OptimizationType,
        optimization_mode: OptimizationMode,
        user_id: Optional[str] = None
    ) -> OptimizationStrategy:
        """
        选择最佳优化策略
        
        Args:
            text_analysis: 文本分析结果
            optimization_type: 优化类型
            optimization_mode: 优化模式
            user_id: 用户ID (用于获取用户偏好)
            
        Returns:
            选中的优化策略
        """
        try:
            self._logger.info(f"选择优化策略 (类型: {optimization_type.value}, 模式: {optimization_mode.value})")
            
            # 确保策略已加载
            await self.load_strategies()
            
            if not self._strategies_cache:
                raise StrategySelectionError("没有可用的优化策略")
            
            # 分析文本特征
            text_features = await self.feature_analyzer.analyze_text_features(
                text_analysis.get('content', ''),
                text_analysis
            )
            
            # 获取用户偏好
            user_preferences = None
            if user_id:
                try:
                    preferences_response = await self.storage_client.get_user_optimization_preferences(user_id)
                    user_preferences = preferences_response.get('data', {})
                except Exception as e:
                    self._logger.warning(f"获取用户偏好失败: {e}")
            
            # 评分所有策略
            strategy_scores = []
            for strategy in self._strategies_cache.values():
                # 过滤不匹配的策略
                if not await self._is_strategy_applicable(strategy, optimization_type, optimization_mode):
                    continue
                
                score = await self.strategy_selector.score_strategy(
                    strategy=strategy,
                    text_features=text_features,
                    optimization_type=optimization_type,
                    optimization_mode=optimization_mode,
                    user_preferences=user_preferences
                )
                
                strategy_scores.append(score)
            
            if not strategy_scores:
                raise StrategySelectionError("没有适用的优化策略")
            
            # 选择评分最高的策略
            best_score = max(strategy_scores, key=lambda x: x.score)
            selected_strategy = self._strategies_cache[best_score.strategy_id]
            
            self._logger.info(f"选择策略: {best_score.strategy_name} (分数: {best_score.score}, 置信度: {best_score.confidence})")
            self._logger.debug(f"选择理由: {', '.join(best_score.reasons)}")
            
            return selected_strategy
            
        except Exception as e:
            self._logger.error(f"策略选择失败: {e}")
            # 返回默认策略
            return await self._get_default_strategy(optimization_type, optimization_mode)
    
    async def update_strategy_performance(
        self,
        strategy_id: str,
        success: bool,
        quality_improvement: float,
        processing_time_ms: int
    ):
        """
        更新策略性能统计
        
        Args:
            strategy_id: 策略ID
            success: 是否成功
            quality_improvement: 质量改进幅度
            processing_time_ms: 处理时间
        """
        try:
            # 更新本地缓存中的策略统计
            if strategy_id in self._strategies_cache:
                strategy = self._strategies_cache[strategy_id]
                strategy.usage_count += 1
                
                if success:
                    # 更新成功率
                    total_attempts = strategy.usage_count
                    current_successes = (strategy.success_rate / 100) * (total_attempts - 1)
                    new_successes = current_successes + 1
                    strategy.success_rate = (new_successes / total_attempts) * 100
                    
                    # 更新平均质量改进
                    current_avg = strategy.avg_quality_improvement
                    strategy.avg_quality_improvement = (
                        (current_avg * (total_attempts - 1) + quality_improvement) / total_attempts
                    )
                
                # 更新平均处理时间
                current_avg_time = strategy.avg_processing_time_ms
                strategy.avg_processing_time_ms = int(
                    (current_avg_time * (strategy.usage_count - 1) + processing_time_ms) / strategy.usage_count
                )
                
                strategy.updated_at = datetime.utcnow()
            
            # 更新存储服务中的统计
            stats_data = {
                'usage_count_increment': 1,
                'success': success,
                'quality_improvement': quality_improvement,
                'processing_time_ms': processing_time_ms
            }
            
            await self.storage_client.update_strategy_statistics(strategy_id, stats_data)
            
            self._logger.debug(f"更新策略性能统计: {strategy_id}")
            
        except Exception as e:
            self._logger.error(f"更新策略性能统计失败: {e}")
    
    async def get_strategy_by_id(self, strategy_id: str) -> Optional[OptimizationStrategy]:
        """
        根据ID获取策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            优化策略或None
        """
        await self.load_strategies()
        return self._strategies_cache.get(strategy_id)
    
    async def get_all_strategies(self) -> List[OptimizationStrategy]:
        """
        获取所有策略
        
        Returns:
            策略列表
        """
        await self.load_strategies()
        return list(self._strategies_cache.values())
    
    async def _is_strategy_applicable(
        self,
        strategy: OptimizationStrategy,
        optimization_type: OptimizationType,
        optimization_mode: OptimizationMode
    ) -> bool:
        """检查策略是否适用"""
        # 检查策略是否激活
        if not strategy.is_active:
            return False
        
        # 检查优化类型匹配
        if strategy.optimization_type != optimization_type:
            # 检查是否有通用策略
            return strategy.optimization_type.value == 'universal'
        
        return True
    
    async def _get_default_strategy(
        self,
        optimization_type: OptimizationType,
        optimization_mode: OptimizationMode
    ) -> OptimizationStrategy:
        """获取默认策略"""
        strategy_key = f"{optimization_type.value}_{optimization_mode.value}"
        
        if strategy_key in self._default_strategies:
            return self._default_strategies[strategy_key]
        
        # 创建通用默认策略
        return OptimizationStrategy(
            name=f"默认{optimization_type.value}策略",
            description=f"默认的{optimization_type.value}优化策略",
            optimization_type=optimization_type,
            optimization_mode=optimization_mode,
            system_prompt="你是一个专业的历史文本优化专家。",
            prompt_template="请对以下文本进行优化：\n\n{content}",
            preferred_model="gemini-1.5-pro",
            is_default=True
        )
    
    async def _init_default_strategies(self):
        """初始化默认策略"""
        try:
            await asyncio.sleep(0.1)  # 避免阻塞
            await self._load_default_strategies()
        except Exception as e:
            self._logger.error(f"初始化默认策略失败: {e}")
    
    async def _load_default_strategies(self):
        """加载默认策略配置"""
        self._logger.info("加载默认策略配置")
        
        # 定义默认策略配置
        default_configs = [
            {
                'name': '历史文本润色策略',
                'optimization_type': OptimizationType.POLISH,
                'optimization_mode': OptimizationMode.HISTORICAL_FORMAT,
                'system_prompt': '你是一个专业的历史文本优化专家，擅长对历史文献进行润色优化。',
                'prompt_template': '''请对以下历史文本进行润色优化：

【优化要求】
1. 保持历史事实和核心内容不变
2. 改善语言表达的流畅性和准确性
3. 统一用词规范，提升文本的学术性
4. 修正语法错误和表达不当之处
5. 保持历史文献的庄重感和严谨性

【原始文本】
{content}

请提供优化后的文本：'''
            },
            {
                'name': '历史文本扩展策略',
                'optimization_type': OptimizationType.EXPAND,
                'optimization_mode': OptimizationMode.HISTORICAL_FORMAT,
                'system_prompt': '你是一个专业的历史文本扩展专家，擅长为历史文献增加相关细节和背景。',
                'prompt_template': '''请对以下历史文本进行扩展优化：

【优化要求】
1. 在保持原文核心内容的基础上增加相关细节
2. 补充必要的历史背景和上下文信息
3. 增强文本的完整性和可读性
4. 添加适当的人物描述和事件细节
5. 确保扩展内容符合历史事实

【原始文本】
{content}

请提供扩展优化后的文本：'''
            },
            {
                'name': '历史文本现代化策略',
                'optimization_type': OptimizationType.MODERNIZE,
                'optimization_mode': OptimizationMode.SIMPLIFIED,
                'system_prompt': '你是一个专业的古文现代化专家，擅长将古文转换为现代汉语。',
                'prompt_template': '''请将以下历史文本进行现代化改写：

【优化要求】
1. 将文言文表达转换为现代汉语
2. 保持历史事实和核心内容不变
3. 使用现代读者容易理解的表达方式
4. 保留必要的历史术语和专有名词
5. 确保改写后的文本准确传达原意

【原始文本】
{content}

请提供现代化改写后的文本：'''
            }
        ]
        
        # 创建默认策略对象
        for config in default_configs:
            strategy = OptimizationStrategy(
                name=config['name'],
                description=f"系统默认的{config['optimization_type'].value}策略",
                optimization_type=config['optimization_type'],
                optimization_mode=config['optimization_mode'],
                system_prompt=config['system_prompt'],
                prompt_template=config['prompt_template'],
                preferred_model="gemini-1.5-pro",
                is_default=True,
                is_active=True
            )
            
            strategy_key = f"{config['optimization_type'].value}_{config['optimization_mode'].value}"
            self._default_strategies[strategy_key] = strategy
            self._strategies_cache[strategy.strategy_id] = strategy
        
        self._logger.info(f"加载了 {len(self._default_strategies)} 个默认策略")