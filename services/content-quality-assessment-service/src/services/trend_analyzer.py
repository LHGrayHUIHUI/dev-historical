"""
质量趋势分析器

分析内容质量的历史趋势，提供趋势预测、改进建议和风险提醒。
支持线性回归、多项式拟合等多种趋势分析方法。
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import settings
from ..models.assessment_models import (
    QualityTrendAnalysis, QualityTrend, QualityDimension,
    TrendDirection, QualityAssessmentResult
)
from ..clients.storage_client import storage_client

logger = logging.getLogger(__name__)

class QualityTrendAnalyzer:
    """质量趋势分析器"""
    
    def __init__(self):
        self.min_data_points = settings.trend_analysis.min_data_points
        self.analysis_window_days = settings.trend_analysis.analysis_window_days
        self.trend_threshold = settings.trend_analysis.trend_significance_threshold
        self.enable_prediction = settings.trend_analysis.enable_trend_prediction
        self.confidence_level = settings.trend_analysis.prediction_confidence_level
    
    async def analyze_quality_trend(self, 
                                   content_id: str, 
                                   start_date: datetime,
                                   end_date: datetime) -> QualityTrendAnalysis:
        """
        分析质量趋势
        
        Args:
            content_id: 内容ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            QualityTrendAnalysis: 趋势分析结果
        """
        try:
            # 获取历史评估数据
            historical_assessments = await self._get_historical_assessments(
                content_id, start_date, end_date
            )
            
            if len(historical_assessments) < self.min_data_points:
                raise ValueError(f"数据点不足，至少需要{self.min_data_points}个评估记录，当前只有{len(historical_assessments)}个")
            
            logger.info(f"Analyzing trend for content {content_id} with {len(historical_assessments)} data points")
            
            # 分析整体趋势
            overall_trend = await self._analyze_overall_trend(historical_assessments)
            
            # 分析各维度趋势
            dimension_trends = []
            for dimension in QualityDimension:
                trend = await self._analyze_dimension_trend(historical_assessments, dimension)
                if trend:
                    dimension_trends.append(trend)
            
            # 生成改进建议和风险提醒
            improvement_suggestions = self._generate_improvement_suggestions(dimension_trends)
            risk_alerts = self._generate_risk_alerts(dimension_trends)
            trend_summary = self._generate_trend_summary(overall_trend, dimension_trends)
            
            # 预测下次评估日期
            next_assessment_date = self._predict_next_assessment_date(historical_assessments)
            
            # 预测未来性能
            predicted_performance = None
            if self.enable_prediction:
                predicted_performance = self._predict_future_performance(
                    overall_trend, dimension_trends
                )
            
            analysis_id = f"trend_{content_id}_{int(datetime.now().timestamp())}"
            
            # 构建分析结果
            analysis = QualityTrendAnalysis(
                analysis_id=analysis_id,
                content_id=content_id,
                analysis_period=(start_date, end_date),
                overall_trend=overall_trend,
                dimension_trends=dimension_trends,
                improvement_suggestions=improvement_suggestions,
                risk_alerts=risk_alerts,
                trend_summary=trend_summary,
                next_assessment_date=next_assessment_date,
                predicted_performance=predicted_performance,
                analysis_time=datetime.now(),
                min_data_points_met=True,
                confidence_level=self.confidence_level
            )
            
            # 保存分析结果
            try:
                await storage_client.save_trend_analysis(analysis)
            except Exception as e:
                logger.warning(f"Failed to save trend analysis: {str(e)}")
            
            logger.info(f"Trend analysis completed for content {content_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Trend analysis failed for content {content_id}: {str(e)}")
            raise
    
    async def _get_historical_assessments(self, 
                                        content_id: str,
                                        start_date: datetime,
                                        end_date: datetime) -> List[Dict[str, Any]]:
        """获取历史评估数据"""
        try:
            # 从storage service获取历史评估数据
            response = await storage_client.get_content_assessments(
                content_id=content_id,
                limit=1000,  # 获取足够多的历史数据
                offset=0,
                start_date=start_date,
                end_date=end_date
            )
            
            assessments = response.get('data', [])
            
            # 确保数据按时间排序
            assessments.sort(key=lambda x: x.get('assessment_time', datetime.now()))
            
            logger.debug(f"Retrieved {len(assessments)} historical assessments for content {content_id}")
            return assessments
            
        except Exception as e:
            logger.error(f"Failed to get historical assessments: {str(e)}")
            return []
    
    async def _analyze_overall_trend(self, assessments: List[Dict]) -> QualityTrend:
        """分析整体质量趋势"""
        scores = [assessment['overall_score'] for assessment in assessments]
        timestamps = [
            assessment['assessment_time'].timestamp() 
            if isinstance(assessment['assessment_time'], datetime)
            else datetime.fromisoformat(assessment['assessment_time']).timestamp()
            for assessment in assessments
        ]
        
        return await self._calculate_trend(
            "overall", scores, timestamps, assessments[0]['content_id']
        )
    
    async def _analyze_dimension_trend(self, 
                                     assessments: List[Dict], 
                                     dimension: QualityDimension) -> Optional[QualityTrend]:
        """分析特定维度趋势"""
        scores = []
        timestamps = []
        
        for assessment in assessments:
            metrics = assessment.get('metrics', [])
            for metric in metrics:
                if metric.get('dimension') == dimension.value:
                    scores.append(metric['score'])
                    assessment_time = assessment['assessment_time']
                    if isinstance(assessment_time, datetime):
                        timestamps.append(assessment_time.timestamp())
                    else:
                        timestamps.append(datetime.fromisoformat(assessment_time).timestamp())
                    break
        
        if len(scores) < self.min_data_points:
            return None
        
        return await self._calculate_trend(
            dimension.value, scores, timestamps, assessments[0]['content_id']
        )
    
    async def _calculate_trend(self, 
                             dimension_name: str, 
                             scores: List[float], 
                             timestamps: List[float], 
                             content_id: str) -> QualityTrend:
        """计算趋势指标"""
        try:
            # 转换为numpy数组
            X = np.array(timestamps).reshape(-1, 1)
            y = np.array(scores)
            
            # 线性回归分析
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            r_squared = model.score(X, y)
            
            # 预测下期得分
            prediction_score = 0.0
            confidence_interval = (0.0, 100.0)
            
            if len(timestamps) > 1:
                # 计算时间间隔
                time_intervals = np.diff(sorted(timestamps))
                avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 86400  # 默认1天
                
                # 预测下一个时间点
                next_timestamp = timestamps[-1] + avg_interval
                prediction_score = model.predict([[next_timestamp]])[0]
                
                # 置信区间估算
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                std_error = np.sqrt(mse)
                
                # 95%置信区间
                confidence_margin = 1.96 * std_error
                confidence_interval = (
                    max(0, prediction_score - confidence_margin),
                    min(100, prediction_score + confidence_margin)
                )
            
            # 判断趋势方向和强度
            trend_direction = TrendDirection.STABLE
            trend_strength = 0.1
            
            if abs(slope) >= self.trend_threshold:
                if slope > 0:
                    trend_direction = TrendDirection.IMPROVING
                else:
                    trend_direction = TrendDirection.DECLINING
                
                # 趋势强度基于斜率和拟合度
                trend_strength = min(1.0, abs(slope) * r_squared / 10)
            
            # 计算统计信息
            average_score = np.mean(scores)
            score_variance = np.var(scores)
            data_points_count = len(scores)
            
            # 计算分析周期天数
            if len(timestamps) > 1:
                period_seconds = max(timestamps) - min(timestamps)
                analysis_period_days = max(1, int(period_seconds / 86400))
            else:
                analysis_period_days = 1
            
            # 预测下次评估时间
            next_assessment_recommended = None
            if trend_direction == TrendDirection.DECLINING and trend_strength > 0.3:
                # 下降趋势建议更频繁评估
                next_assessment_recommended = datetime.fromtimestamp(
                    timestamps[-1] + avg_interval / 2
                )
            elif trend_direction == TrendDirection.IMPROVING and trend_strength > 0.5:
                # 改善趋势可以延长评估间隔
                next_assessment_recommended = datetime.fromtimestamp(
                    timestamps[-1] + avg_interval * 1.5
                )
            else:
                next_assessment_recommended = datetime.fromtimestamp(
                    timestamps[-1] + avg_interval
                )
            
            return QualityTrend(
                content_id=content_id,
                dimension=QualityDimension(dimension_name) if dimension_name != "overall" else None,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=r_squared,
                prediction_score=max(0, min(100, prediction_score)),
                confidence_interval=confidence_interval,
                next_assessment_recommended=next_assessment_recommended,
                data_points_count=data_points_count,
                analysis_period_days=analysis_period_days,
                average_score=average_score,
                score_variance=score_variance
            )
            
        except Exception as e:
            logger.error(f"Trend calculation failed for {dimension_name}: {str(e)}")
            raise
    
    def _generate_improvement_suggestions(self, dimension_trends: List[QualityTrend]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 分析各维度趋势
        declining_dimensions = []
        stable_low_dimensions = []
        
        for trend in dimension_trends:
            if trend.dimension is None:
                continue
                
            if trend.trend_direction == TrendDirection.DECLINING:
                declining_dimensions.append(trend.dimension.value)
            elif (trend.trend_direction == TrendDirection.STABLE and 
                  trend.average_score < 70):
                stable_low_dimensions.append(trend.dimension.value)
        
        # 针对下降维度的建议
        if declining_dimensions:
            suggestions.append(f"以下维度呈下降趋势，需要重点关注：{', '.join(declining_dimensions)}")
            
            for dim in declining_dimensions:
                if dim == "readability":
                    suggestions.append("建议优化文本结构，简化句式，提高可读性")
                elif dim == "accuracy":
                    suggestions.append("建议加强事实核查，完善引用和数据验证")
                elif dim == "completeness":
                    suggestions.append("建议补充缺失信息，完善内容结构")
                elif dim == "coherence":
                    suggestions.append("建议改善段落衔接，增强逻辑连贯性")
                elif dim == "relevance":
                    suggestions.append("建议明确目标受众，增强内容针对性")
        
        # 针对稳定但低分维度的建议
        if stable_low_dimensions:
            suggestions.append(f"以下维度得分偏低但趋势稳定，可进行专项改进：{', '.join(stable_low_dimensions)}")
        
        # 通用建议
        if len(dimension_trends) > 0:
            avg_variance = np.mean([trend.score_variance for trend in dimension_trends])
            if avg_variance > 100:  # 方差较大
                suggestions.append("质量波动较大，建议建立标准化的内容创作流程")
        
        return suggestions[:5]  # 限制建议数量
    
    def _generate_risk_alerts(self, dimension_trends: List[QualityTrend]) -> List[str]:
        """生成风险提醒"""
        alerts = []
        
        # 整体趋势风险
        declining_count = sum(1 for trend in dimension_trends 
                            if trend.trend_direction == TrendDirection.DECLINING)
        
        if declining_count > len(dimension_trends) / 2:
            alerts.append("⚠️ 多个维度呈下降趋势，质量管控存在系统性风险")
        
        # 单维度风险
        for trend in dimension_trends:
            if trend.dimension is None:
                continue
                
            # 急剧下降风险
            if (trend.trend_direction == TrendDirection.DECLINING and 
                trend.trend_strength > 0.5):
                alerts.append(f"⚠️ {trend.dimension.value}维度急剧下降，需立即干预")
            
            # 预测低分风险
            if trend.prediction_score < 60:
                alerts.append(f"⚠️ 预测{trend.dimension.value}维度将降至不合格水平")
            
            # 不稳定风险
            if trend.score_variance > 200:
                alerts.append(f"⚠️ {trend.dimension.value}维度波动过大，质量不稳定")
        
        return alerts[:3]  # 限制提醒数量
    
    def _generate_trend_summary(self, 
                              overall_trend: QualityTrend,
                              dimension_trends: List[QualityTrend]) -> str:
        """生成趋势摘要"""
        try:
            # 整体趋势描述
            if overall_trend.trend_direction == TrendDirection.IMPROVING:
                summary = f"内容质量整体呈上升趋势，平均分{overall_trend.average_score:.1f}分"
            elif overall_trend.trend_direction == TrendDirection.DECLINING:
                summary = f"内容质量整体呈下降趋势，平均分{overall_trend.average_score:.1f}分"
            else:
                summary = f"内容质量整体保持稳定，平均分{overall_trend.average_score:.1f}分"
            
            # 维度表现
            improving_dims = [trend.dimension.value for trend in dimension_trends 
                            if trend.trend_direction == TrendDirection.IMPROVING]
            declining_dims = [trend.dimension.value for trend in dimension_trends 
                            if trend.trend_direction == TrendDirection.DECLINING]
            
            if improving_dims:
                summary += f"，其中{', '.join(improving_dims)}表现改善"
            
            if declining_dims:
                summary += f"，但{', '.join(declining_dims)}需要关注"
            
            # 预测信息
            if overall_trend.prediction_score > overall_trend.average_score + 5:
                summary += "，预期将继续提升"
            elif overall_trend.prediction_score < overall_trend.average_score - 5:
                summary += "，预期可能下降"
            else:
                summary += "，预期保持当前水平"
            
            return summary + "。"
            
        except Exception as e:
            logger.error(f"Failed to generate trend summary: {str(e)}")
            return "趋势分析摘要生成失败。"
    
    def _predict_next_assessment_date(self, assessments: List[Dict]) -> Optional[datetime]:
        """预测下次评估日期"""
        try:
            if len(assessments) < 2:
                return datetime.now() + timedelta(days=7)  # 默认一周后
            
            # 计算评估间隔
            times = []
            for assessment in assessments:
                assessment_time = assessment['assessment_time']
                if isinstance(assessment_time, datetime):
                    times.append(assessment_time)
                else:
                    times.append(datetime.fromisoformat(assessment_time))
            
            times.sort()
            intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
            
            if intervals:
                avg_interval_seconds = np.mean(intervals)
                avg_interval_days = avg_interval_seconds / 86400
                
                # 根据质量趋势调整间隔
                last_assessment_time = times[-1]
                
                # 基础间隔
                next_interval_days = avg_interval_days
                
                # 根据最近的质量趋势调整
                if len(assessments) >= 3:
                    recent_scores = [a['overall_score'] for a in assessments[-3:]]
                    score_trend = recent_scores[-1] - recent_scores[0]
                    
                    if score_trend < -10:  # 显著下降
                        next_interval_days = min(avg_interval_days, 3)  # 最多3天
                    elif score_trend > 10:  # 显著改善
                        next_interval_days = min(avg_interval_days * 1.5, 14)  # 最多2周
                
                return last_assessment_time + timedelta(days=next_interval_days)
            
            return datetime.now() + timedelta(days=7)
            
        except Exception as e:
            logger.error(f"Failed to predict next assessment date: {str(e)}")
            return datetime.now() + timedelta(days=7)
    
    def _predict_future_performance(self, 
                                  overall_trend: QualityTrend,
                                  dimension_trends: List[QualityTrend]) -> Dict[str, float]:
        """预测未来性能"""
        try:
            predictions = {}
            
            # 整体性能预测
            predictions["overall"] = overall_trend.prediction_score
            
            # 各维度预测
            for trend in dimension_trends:
                if trend.dimension:
                    predictions[trend.dimension.value] = trend.prediction_score
            
            # 信心评估
            avg_r_squared = np.mean([trend.r_squared for trend in [overall_trend] + dimension_trends])
            predictions["confidence"] = avg_r_squared * 100
            
            # 改善潜力评估
            improving_count = sum(1 for trend in dimension_trends 
                                if trend.trend_direction == TrendDirection.IMPROVING)
            total_dimensions = len(dimension_trends)
            
            if total_dimensions > 0:
                improvement_potential = (improving_count / total_dimensions) * 100
                predictions["improvement_potential"] = improvement_potential
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to predict future performance: {str(e)}")
            return {}

# 全局趋势分析器实例
trend_analyzer = QualityTrendAnalyzer()