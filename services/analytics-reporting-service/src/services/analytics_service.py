"""
分析服务核心模块

负责数据分析、指标计算、趋势分析等核心业务逻辑。
集成机器学习算法进行异常检测、预测分析和智能洞察。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

from ..models import (
    get_db, get_influxdb, get_clickhouse, get_redis,
    AnalysisTask, AnalysisTaskType, AnalysisTaskStatus,
    AnalyticsMetrics, ContentPerformance, PlatformComparison,
    TrendAnalysis, UserBehaviorInsights, TimeSeriesPoint,
    MetricQuery, AggregationResult
)
from ..config.settings import settings

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    分析服务核心类
    
    提供各种数据分析功能：
    - 内容表现分析
    - 平台对比分析  
    - 趋势分析和预测
    - 用户行为分析
    - 异常检测
    - 智能洞察生成
    """
    
    def __init__(self):
        self.redis_client = None
        self.influxdb_client = None
        self.clickhouse_client = None
        self.ml_models = {}
        self.scalers = {}
        self._model_cache_initialized = False

    async def initialize(self):
        """初始化服务依赖"""
        try:
            self.redis_client = await get_redis()
            self.influxdb_client = await get_influxdb()
            self.clickhouse_client = get_clickhouse()
            
            # 初始化机器学习模型缓存
            await self._initialize_ml_models()
            
            logger.info("AnalyticsService 初始化完成")
        except Exception as e:
            logger.error(f"AnalyticsService 初始化失败: {e}")
            raise

    async def _initialize_ml_models(self):
        """初始化机器学习模型"""
        try:
            model_dir = settings.ml.model_cache_dir
            os.makedirs(model_dir, exist_ok=True)
            
            # 加载预训练模型（如果存在）
            model_files = {
                'anomaly_detector': 'anomaly_model.pkl',
                'trend_predictor': 'trend_model.pkl',
                'engagement_predictor': 'engagement_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(model_dir, filename)
                if os.path.exists(filepath):
                    self.ml_models[model_name] = joblib.load(filepath)
                    logger.info(f"已加载预训练模型: {model_name}")
                else:
                    # 创建新的模型实例
                    if model_name == 'anomaly_detector':
                        self.ml_models[model_name] = IsolationForest(
                            contamination=0.1,
                            random_state=42
                        )
                    else:
                        self.ml_models[model_name] = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42
                        )
                    logger.info(f"已创建新模型实例: {model_name}")
            
            self._model_cache_initialized = True
            
        except Exception as e:
            logger.error(f"机器学习模型初始化失败: {e}")
            raise

    async def analyze_content_performance(
        self, 
        user_id: str,
        content_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        platforms: Optional[List[str]] = None
    ) -> List[ContentPerformance]:
        """
        内容表现分析
        
        Args:
            user_id: 用户ID
            content_ids: 内容ID列表，None表示分析所有内容
            start_date: 开始日期
            end_date: 结束日期
            platforms: 平台列表
            
        Returns:
            内容表现分析结果列表
        """
        try:
            logger.info(f"开始内容表现分析 - 用户: {user_id}")
            
            # 构建查询条件
            filters = {"user_id": user_id}
            if content_ids:
                filters["content_id"] = content_ids
            if platforms:
                filters["platform"] = platforms
            
            # 设置默认时间范围（最近30天）
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # 从ClickHouse查询内容表现数据
            query = self._build_content_performance_query(filters, start_date, end_date)
            results = self.clickhouse_client.execute(query)
            
            # 转换为ContentPerformance对象
            performances = []
            for row in results:
                # 计算表现评分
                performance_score = await self._calculate_performance_score(
                    views=row[3], likes=row[4], comments=row[5], 
                    shares=row[6], engagement_rate=row[7]
                )
                
                performance = ContentPerformance(
                    content_id=row[0],
                    title=row[1],
                    platform=row[2],
                    views=row[3],
                    likes=row[4],
                    comments=row[5],
                    shares=row[6],
                    engagement_rate=row[7],
                    click_through_rate=row[8],
                    conversion_rate=row[9],
                    performance_score=performance_score,
                    publish_time=row[10]
                )
                performances.append(performance)
            
            # 缓存结果
            cache_key = f"content_performance:{user_id}:{hash(str(filters))}"
            await self._cache_result(cache_key, performances, expire=3600)
            
            logger.info(f"内容表现分析完成 - 找到 {len(performances)} 个内容")
            return performances
            
        except Exception as e:
            logger.error(f"内容表现分析失败: {e}")
            raise

    async def analyze_platform_comparison(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        metrics: Optional[List[str]] = None
    ) -> List[PlatformComparison]:
        """
        平台对比分析
        
        Args:
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            metrics: 对比指标列表
            
        Returns:
            平台对比分析结果
        """
        try:
            logger.info(f"开始平台对比分析 - 用户: {user_id}")
            
            # 设置默认时间范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # 从ClickHouse获取平台汇总数据
            query = self._build_platform_comparison_query(user_id, start_date, end_date)
            results = self.clickhouse_client.execute(query)
            
            comparisons = []
            for row in results:
                # 计算指标对比
                metrics_comparison = await self._calculate_platform_metrics(
                    platform=row[0], 
                    total_content=row[1],
                    total_views=row[2],
                    total_engagement=row[3]
                )
                
                comparison = PlatformComparison(
                    platform=row[0],
                    total_content=row[1],
                    total_views=row[2],
                    total_engagement=row[3],
                    avg_engagement_rate=row[4],
                    avg_reach=row[5],
                    top_content_id=row[6] or "",
                    metrics_comparison=metrics_comparison
                )
                comparisons.append(comparison)
            
            # 缓存结果
            cache_key = f"platform_comparison:{user_id}:{start_date.date()}:{end_date.date()}"
            await self._cache_result(cache_key, comparisons, expire=7200)
            
            logger.info(f"平台对比分析完成 - 对比 {len(comparisons)} 个平台")
            return comparisons
            
        except Exception as e:
            logger.error(f"平台对比分析失败: {e}")
            raise

    async def analyze_trends(
        self,
        user_id: str,
        metric_names: List[str],
        time_period: str = "daily",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_forecast: bool = True
    ) -> List[TrendAnalysis]:
        """
        趋势分析
        
        Args:
            user_id: 用户ID
            metric_names: 分析的指标名称列表
            time_period: 时间周期 (daily, weekly, monthly)
            start_date: 开始日期
            end_date: 结束日期
            include_forecast: 是否包含预测
            
        Returns:
            趋势分析结果列表
        """
        try:
            logger.info(f"开始趋势分析 - 用户: {user_id}, 指标: {metric_names}")
            
            # 设置默认时间范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=90)
            if not end_date:
                end_date = datetime.now()
            
            trends = []
            for metric_name in metric_names:
                # 从时序数据库获取历史数据
                time_series_data = await self._get_time_series_data(
                    user_id, metric_name, start_date, end_date, time_period
                )
                
                if not time_series_data:
                    logger.warning(f"未找到指标 {metric_name} 的数据")
                    continue
                
                # 计算趋势方向和增长率
                trend_direction, growth_rate = self._calculate_trend_metrics(time_series_data)
                
                # 异常检测
                anomalies = await self._detect_anomalies(time_series_data, metric_name)
                
                # 预测未来趋势
                forecast_data = None
                if include_forecast and len(time_series_data) >= settings.ml.min_training_samples:
                    forecast_data = await self._forecast_trend(
                        time_series_data, metric_name, days=settings.ml.forecast_days
                    )
                
                # 季节性分析
                seasonality = await self._analyze_seasonality(time_series_data)
                
                trend = TrendAnalysis(
                    metric_name=metric_name,
                    time_period=time_period,
                    trend_direction=trend_direction,
                    growth_rate=growth_rate,
                    data_points=time_series_data,
                    forecast=forecast_data,
                    seasonality=seasonality,
                    anomalies=anomalies
                )
                trends.append(trend)
            
            # 缓存结果
            cache_key = f"trend_analysis:{user_id}:{hash(str(metric_names))}:{time_period}"
            await self._cache_result(cache_key, trends, expire=3600)
            
            logger.info(f"趋势分析完成 - 分析了 {len(trends)} 个指标")
            return trends
            
        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            raise

    async def analyze_user_behavior(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> UserBehaviorInsights:
        """
        用户行为分析
        
        Args:
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            用户行为洞察结果
        """
        try:
            logger.info(f"开始用户行为分析 - 用户: {user_id}")
            
            # 设置默认时间范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # 从ClickHouse查询用户行为数据
            behavior_query = self._build_user_behavior_query(user_id, start_date, end_date)
            behavior_data = self.clickhouse_client.execute(behavior_query)
            
            if not behavior_data:
                logger.warning(f"未找到用户 {user_id} 的行为数据")
                return UserBehaviorInsights(
                    active_users=0, new_users=0, returning_users=0,
                    avg_session_duration=0.0, avg_pages_per_session=0.0,
                    top_content=[], popular_times=[], 
                    user_segments={}, behavior_patterns={}
                )
            
            # 计算基础指标
            active_users = len(set(row[0] for row in behavior_data))
            
            # 分析会话数据
            session_stats = await self._analyze_session_data(behavior_data)
            
            # 识别热门内容
            top_content = await self._identify_top_content(behavior_data)
            
            # 分析活跃时段
            popular_times = await self._analyze_popular_times(behavior_data)
            
            # 用户分群
            user_segments = await self._segment_users(behavior_data)
            
            # 行为模式分析
            behavior_patterns = await self._analyze_behavior_patterns(behavior_data)
            
            insights = UserBehaviorInsights(
                active_users=active_users,
                new_users=session_stats.get("new_users", 0),
                returning_users=session_stats.get("returning_users", 0),
                avg_session_duration=session_stats.get("avg_duration", 0.0),
                avg_pages_per_session=session_stats.get("avg_pages", 0.0),
                top_content=top_content,
                popular_times=popular_times,
                user_segments=user_segments,
                behavior_patterns=behavior_patterns
            )
            
            # 缓存结果
            cache_key = f"user_behavior:{user_id}:{start_date.date()}:{end_date.date()}"
            await self._cache_result(cache_key, insights, expire=7200)
            
            logger.info("用户行为分析完成")
            return insights
            
        except Exception as e:
            logger.error(f"用户行为分析失败: {e}")
            raise

    async def detect_anomalies(
        self,
        user_id: str,
        metric_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        异常检测
        
        Args:
            user_id: 用户ID
            metric_name: 指标名称
            start_date: 开始日期
            end_date: 结束日期
            threshold: 异常阈值
            
        Returns:
            异常点列表
        """
        try:
            logger.info(f"开始异常检测 - 指标: {metric_name}")
            
            # 获取时序数据
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            time_series_data = await self._get_time_series_data(
                user_id, metric_name, start_date, end_date, "daily"
            )
            
            if len(time_series_data) < 10:
                logger.warning(f"数据点太少，无法进行异常检测: {len(time_series_data)}")
                return []
            
            # 使用机器学习模型检测异常
            anomalies = await self._detect_anomalies(
                time_series_data, metric_name, threshold
            )
            
            logger.info(f"异常检测完成 - 发现 {len(anomalies)} 个异常点")
            return anomalies
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            raise

    # ===== 私有辅助方法 =====

    def _build_content_performance_query(
        self, 
        filters: Dict[str, Any], 
        start_date: datetime, 
        end_date: datetime
    ) -> str:
        """构建内容表现查询SQL"""
        
        where_conditions = [
            f"publish_time >= '{start_date.isoformat()}'",
            f"publish_time <= '{end_date.isoformat()}'"
        ]
        
        for key, value in filters.items():
            if isinstance(value, list):
                values_str = "', '".join(value)
                where_conditions.append(f"{key} IN ('{values_str}')")
            else:
                where_conditions.append(f"{key} = '{value}'")
        
        query = f"""
        SELECT 
            content_id,
            title,
            platform,
            views,
            likes,
            comments,
            shares,
            engagement_rate,
            click_through_rate,
            conversion_rate,
            publish_time
        FROM content_performance
        WHERE {' AND '.join(where_conditions)}
        ORDER BY publish_time DESC
        LIMIT 1000
        """
        
        return query

    def _build_platform_comparison_query(
        self, 
        user_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> str:
        """构建平台对比查询SQL"""
        
        query = f"""
        SELECT 
            platform,
            count(*) as total_content,
            sum(views) as total_views,
            sum(likes + comments + shares) as total_engagement,
            avg(engagement_rate) as avg_engagement_rate,
            avg(views * engagement_rate) as avg_reach,
            argMax(content_id, engagement_rate) as top_content_id
        FROM content_performance
        WHERE publish_time >= '{start_date.isoformat()}'
            AND publish_time <= '{end_date.isoformat()}'
        GROUP BY platform
        ORDER BY total_engagement DESC
        """
        
        return query

    def _build_user_behavior_query(
        self, 
        user_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> str:
        """构建用户行为查询SQL"""
        
        query = f"""
        SELECT 
            user_id,
            platform,
            action_type,
            content_id,
            timestamp,
            session_id,
            device_type,
            location
        FROM user_behavior
        WHERE timestamp >= '{start_date.isoformat()}'
            AND timestamp <= '{end_date.isoformat()}'
        ORDER BY timestamp DESC
        LIMIT 10000
        """
        
        return query

    async def _calculate_performance_score(
        self, 
        views: int, 
        likes: int, 
        comments: int, 
        shares: int, 
        engagement_rate: float
    ) -> float:
        """计算内容表现评分"""
        
        # 基础权重
        weights = {
            'views': 0.3,
            'likes': 0.25,
            'comments': 0.25,
            'shares': 0.2
        }
        
        # 标准化处理
        max_values = {
            'views': 1000000,  # 假设最大浏览量
            'likes': 50000,    # 假设最大点赞数
            'comments': 10000, # 假设最大评论数
            'shares': 10000    # 假设最大分享数
        }
        
        normalized_values = {
            'views': min(views / max_values['views'], 1.0),
            'likes': min(likes / max_values['likes'], 1.0),
            'comments': min(comments / max_values['comments'], 1.0),
            'shares': min(shares / max_values['shares'], 1.0)
        }
        
        # 计算加权评分
        score = sum(
            normalized_values[metric] * weight 
            for metric, weight in weights.items()
        )
        
        # 结合参与度进行调整
        score = score * (1 + engagement_rate)
        
        return min(score * 100, 100.0)  # 转换为0-100分制

    async def _calculate_platform_metrics(
        self, 
        platform: str, 
        total_content: int, 
        total_views: int, 
        total_engagement: int
    ) -> Dict[str, float]:
        """计算平台指标对比"""
        
        metrics = {
            'content_density': total_content,
            'view_performance': total_views / max(total_content, 1),
            'engagement_performance': total_engagement / max(total_views, 1),
            'efficiency_score': total_engagement / max(total_content, 1)
        }
        
        return metrics

    def _calculate_trend_metrics(
        self, 
        time_series_data: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """计算趋势指标"""
        
        if len(time_series_data) < 2:
            return "stable", 0.0
        
        values = [point['value'] for point in time_series_data]
        
        # 计算线性趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # 确定趋势方向
        if abs(slope) < 0.01:  # 阈值可调整
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # 计算增长率（基于首末值）
        if values[0] != 0:
            growth_rate = ((values[-1] - values[0]) / abs(values[0])) * 100
        else:
            growth_rate = 0.0
        
        return trend_direction, growth_rate

    async def _get_time_series_data(
        self, 
        user_id: str, 
        metric_name: str, 
        start_date: datetime, 
        end_date: datetime, 
        time_period: str
    ) -> List[Dict[str, Any]]:
        """从InfluxDB获取时序数据"""
        
        try:
            # 构建InfluxDB查询
            query = f'''
            from(bucket: "{settings.database.influxdb_bucket}")
                |> range(start: {start_date.isoformat()}, stop: {end_date.isoformat()})
                |> filter(fn: (r) => r["_measurement"] == "{metric_name}")
                |> filter(fn: (r) => r["user_id"] == "{user_id}")
                |> aggregateWindow(every: 1{time_period[0]}, fn: mean, createEmpty: false)
                |> yield(name: "mean")
            '''
            
            query_api = self.influxdb_client.query_api()
            result = await query_api.query(query)
            
            data_points = []
            for table in result:
                for record in table.records:
                    data_points.append({
                        'timestamp': record.get_time(),
                        'value': record.get_value() or 0.0
                    })
            
            return sorted(data_points, key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.error(f"获取时序数据失败: {e}")
            return []

    async def _detect_anomalies(
        self, 
        time_series_data: List[Dict[str, Any]], 
        metric_name: str, 
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """使用机器学习检测异常点"""
        
        try:
            if not self._model_cache_initialized:
                await self._initialize_ml_models()
            
            values = np.array([point['value'] for point in time_series_data]).reshape(-1, 1)
            
            # 使用隔离森林检测异常
            model = self.ml_models.get('anomaly_detector')
            if model is None:
                return []
            
            # 如果模型未训练，先训练
            if not hasattr(model, 'offset_'):
                model.fit(values)
                # 保存模型
                model_path = os.path.join(settings.ml.model_cache_dir, 'anomaly_model.pkl')
                joblib.dump(model, model_path)
            
            # 预测异常
            predictions = model.predict(values)
            anomaly_scores = model.score_samples(values)
            
            anomalies = []
            for i, (prediction, score) in enumerate(zip(predictions, anomaly_scores)):
                if prediction == -1:  # 异常点
                    anomalies.append({
                        'timestamp': time_series_data[i]['timestamp'],
                        'value': time_series_data[i]['value'],
                        'anomaly_score': float(score),
                        'severity': 'high' if score < -0.5 else 'medium'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return []

    async def _forecast_trend(
        self, 
        time_series_data: List[Dict[str, Any]], 
        metric_name: str, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """预测未来趋势"""
        
        try:
            if len(time_series_data) < settings.ml.min_training_samples:
                return []
            
            # 准备训练数据
            df = pd.DataFrame(time_series_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['hour'] = df['timestamp'].dt.hour
            df['weekday'] = df['timestamp'].dt.weekday
            
            # 特征工程
            features = ['day_of_year', 'hour', 'weekday']
            X = df[features].values
            y = df['value'].values
            
            # 训练模型
            model = self.ml_models.get('trend_predictor', RandomForestRegressor())
            model.fit(X, y)
            
            # 生成预测时间点
            last_timestamp = df['timestamp'].iloc[-1]
            future_timestamps = [
                last_timestamp + timedelta(days=i) for i in range(1, days + 1)
            ]
            
            # 预测未来值
            forecast_data = []
            for timestamp in future_timestamps:
                future_features = np.array([[
                    timestamp.dayofyear,
                    timestamp.hour,
                    timestamp.weekday()
                ]])
                
                predicted_value = model.predict(future_features)[0]
                
                forecast_data.append({
                    'timestamp': timestamp,
                    'value': max(predicted_value, 0.0)  # 确保非负值
                })
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"趋势预测失败: {e}")
            return []

    async def _analyze_seasonality(
        self, 
        time_series_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """分析季节性模式"""
        
        try:
            if len(time_series_data) < 14:  # 至少两周数据
                return None
            
            df = pd.DataFrame(time_series_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 按小时和星期几分组分析
            df['hour'] = df['timestamp'].dt.hour
            df['weekday'] = df['timestamp'].dt.weekday
            df['month'] = df['timestamp'].dt.month
            
            seasonality = {
                'hourly_pattern': df.groupby('hour')['value'].mean().to_dict(),
                'weekly_pattern': df.groupby('weekday')['value'].mean().to_dict(),
                'monthly_pattern': df.groupby('month')['value'].mean().to_dict()
            }
            
            return seasonality
            
        except Exception as e:
            logger.error(f"季节性分析失败: {e}")
            return None

    async def _analyze_session_data(
        self, 
        behavior_data: List[Tuple]
    ) -> Dict[str, Any]:
        """分析会话数据"""
        
        try:
            df = pd.DataFrame(behavior_data, columns=[
                'user_id', 'platform', 'action_type', 'content_id', 
                'timestamp', 'session_id', 'device_type', 'location'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 按会话分组计算统计
            session_stats = df.groupby(['user_id', 'session_id']).agg({
                'timestamp': ['min', 'max', 'count'],
                'content_id': 'nunique'
            }).reset_index()
            
            # 计算会话时长（分钟）
            session_stats['duration'] = (
                session_stats[('timestamp', 'max')] - 
                session_stats[('timestamp', 'min')]
            ).dt.total_seconds() / 60
            
            # 识别新用户和回访用户
            user_sessions = df.groupby('user_id')['session_id'].nunique()
            new_users = (user_sessions == 1).sum()
            returning_users = (user_sessions > 1).sum()
            
            return {
                'new_users': new_users,
                'returning_users': returning_users,
                'avg_duration': session_stats['duration'].mean(),
                'avg_pages': session_stats[('content_id', 'nunique')].mean()
            }
            
        except Exception as e:
            logger.error(f"会话数据分析失败: {e}")
            return {}

    async def _identify_top_content(
        self, 
        behavior_data: List[Tuple]
    ) -> List[str]:
        """识别热门内容"""
        
        try:
            df = pd.DataFrame(behavior_data, columns=[
                'user_id', 'platform', 'action_type', 'content_id', 
                'timestamp', 'session_id', 'device_type', 'location'
            ])
            
            # 计算内容热度（基于交互次数）
            content_popularity = df['content_id'].value_counts()
            
            return content_popularity.head(10).index.tolist()
            
        except Exception as e:
            logger.error(f"热门内容识别失败: {e}")
            return []

    async def _analyze_popular_times(
        self, 
        behavior_data: List[Tuple]
    ) -> List[int]:
        """分析热门时段"""
        
        try:
            df = pd.DataFrame(behavior_data, columns=[
                'user_id', 'platform', 'action_type', 'content_id', 
                'timestamp', 'session_id', 'device_type', 'location'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            
            # 计算每小时的活动量
            hourly_activity = df['hour'].value_counts().sort_index()
            
            return hourly_activity.head(5).index.tolist()
            
        except Exception as e:
            logger.error(f"热门时段分析失败: {e}")
            return []

    async def _segment_users(
        self, 
        behavior_data: List[Tuple]
    ) -> Dict[str, int]:
        """用户分群分析"""
        
        try:
            df = pd.DataFrame(behavior_data, columns=[
                'user_id', 'platform', 'action_type', 'content_id', 
                'timestamp', 'session_id', 'device_type', 'location'
            ])
            
            # 基于设备类型分群
            device_segments = df['device_type'].value_counts().to_dict()
            
            # 基于平台偏好分群
            user_platform_pref = df.groupby('user_id')['platform'].apply(
                lambda x: x.value_counts().index[0]
            )
            platform_segments = user_platform_pref.value_counts().to_dict()
            
            return {
                **{f"device_{k}": v for k, v in device_segments.items()},
                **{f"platform_{k}": v for k, v in platform_segments.items()}
            }
            
        except Exception as e:
            logger.error(f"用户分群失败: {e}")
            return {}

    async def _analyze_behavior_patterns(
        self, 
        behavior_data: List[Tuple]
    ) -> Dict[str, Any]:
        """行为模式分析"""
        
        try:
            df = pd.DataFrame(behavior_data, columns=[
                'user_id', 'platform', 'action_type', 'content_id', 
                'timestamp', 'session_id', 'device_type', 'location'
            ])
            
            # 行为类型分布
            action_distribution = df['action_type'].value_counts().to_dict()
            
            # 用户活跃度分布
            user_activity = df.groupby('user_id').size()
            activity_stats = {
                'high_activity': (user_activity > user_activity.quantile(0.8)).sum(),
                'medium_activity': ((user_activity > user_activity.quantile(0.4)) & 
                                   (user_activity <= user_activity.quantile(0.8))).sum(),
                'low_activity': (user_activity <= user_activity.quantile(0.4)).sum()
            }
            
            return {
                'action_distribution': action_distribution,
                'activity_distribution': activity_stats
            }
            
        except Exception as e:
            logger.error(f"行为模式分析失败: {e}")
            return {}

    async def _cache_result(
        self, 
        cache_key: str, 
        data: Any, 
        expire: int = 3600
    ):
        """缓存分析结果"""
        
        try:
            if self.redis_client:
                # 将结果序列化为JSON
                import json
                
                # 处理不可序列化的对象
                def serialize_obj(obj):
                    if hasattr(obj, 'dict'):
                        return obj.dict()
                    elif isinstance(obj, (datetime, UUID)):
                        return str(obj)
                    else:
                        return str(obj)
                
                serialized_data = json.dumps(data, default=serialize_obj)
                await self.redis_client.setex(cache_key, expire, serialized_data)
                
        except Exception as e:
            logger.warning(f"缓存结果失败: {e}")

    async def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """获取缓存的分析结果"""
        
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    import json
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"获取缓存结果失败: {e}")
        
        return None