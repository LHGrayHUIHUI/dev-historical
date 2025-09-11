"""
ML优化服务
基于机器学习的智能调度时间优化和性能预测
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import numpy as np
import pickle
import os
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ML相关导入
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..models import (
    SchedulingTask, PlatformMetrics, ContentPerformance, UserBehaviorPattern,
    MLModelMetrics, get_db_session
)
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    original_time: datetime
    optimized_time: datetime
    optimization_score: float
    confidence_score: float
    predicted_metrics: Dict[str, float]
    optimization_factors: Dict[str, Any]
    model_version: str
    model_params: Dict[str, Any]


@dataclass
class PredictionFeatures:
    """预测特征数据类"""
    hour_of_day: int
    day_of_week: int
    month: int
    is_weekend: bool
    content_length: int
    has_media: bool
    hashtag_count: int
    historical_engagement_rate: float
    platform_avg_engagement: float
    user_avg_engagement: float
    recent_post_frequency: float
    time_since_last_post: int
    competitive_post_count: int
    seasonal_factor: float


class ScheduleOptimizer:
    """调度优化器类"""
    
    def __init__(self, model_type: str = "RandomForestRegressor"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_version = "1.0.0"
        self.is_trained = False
        
        # 模型存储路径
        self.model_dir = Path("models/optimization")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程池用于CPU密集型操作
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def initialize_model(self):
        """初始化机器学习模型"""
        try:
            # 尝试加载已存在的模型
            if await self._load_existing_model():
                logger.info("成功加载现有ML模型")
                return
            
            # 创建新模型
            await self._create_new_model()
            logger.info("创建新的ML模型")
            
        except Exception as e:
            logger.error(f"ML模型初始化失败: {e}")
            # 创建基础模型作为后备
            self._create_fallback_model()
    
    async def _load_existing_model(self) -> bool:
        """加载已存在的模型"""
        model_file = self.model_dir / "scheduler_model.pkl"
        scaler_file = self.model_dir / "scaler.pkl"
        meta_file = self.model_dir / "model_meta.json"
        
        if not all(f.exists() for f in [model_file, scaler_file, meta_file]):
            return False
        
        try:
            # 在线程池中执行IO操作
            loop = asyncio.get_event_loop()
            
            # 加载模型和缩放器
            self.model = await loop.run_in_executor(
                self.executor, self._load_pickle_file, model_file
            )
            self.scaler = await loop.run_in_executor(
                self.executor, self._load_pickle_file, scaler_file
            )
            
            # 加载元数据
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                self.model_version = meta.get('version', '1.0.0')
                self.feature_names = meta.get('feature_names', [])
                self.is_trained = meta.get('is_trained', False)
            
            return True
            
        except Exception as e:
            logger.warning(f"加载模型失败: {e}")
            return False
    
    def _load_pickle_file(self, file_path: Path):
        """线程安全的pickle文件加载"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    async def _create_new_model(self):
        """创建新的机器学习模型"""
        # 设置模型参数
        model_params = settings.ml.model_params
        
        if self.model_type == "RandomForestRegressor":
            self.model = RandomForestRegressor(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 创建特征缩放器
        self.scaler = StandardScaler()
        
        # 定义特征名称
        self.feature_names = [
            'hour_of_day', 'day_of_week', 'month', 'is_weekend',
            'content_length', 'has_media', 'hashtag_count',
            'historical_engagement_rate', 'platform_avg_engagement',
            'user_avg_engagement', 'recent_post_frequency',
            'time_since_last_post', 'competitive_post_count',
            'seasonal_factor'
        ]
        
        self.is_trained = False
    
    def _create_fallback_model(self):
        """创建后备模型（简单规则）"""
        logger.warning("使用后备优化规则")
        self.model = None
        self.is_trained = False
    
    async def train_model(self, min_samples: int = None) -> Dict[str, float]:
        """训练机器学习模型"""
        if min_samples is None:
            min_samples = settings.ml.min_training_samples
        
        try:
            # 获取训练数据
            training_data = await self._collect_training_data()
            
            if len(training_data) < min_samples:
                logger.warning(f"训练数据不足 ({len(training_data)} < {min_samples})，跳过训练")
                return {}
            
            # 准备特征和标签
            X, y = await self._prepare_training_data(training_data)
            
            if len(X) == 0:
                logger.warning("没有有效的训练数据")
                return {}
            
            # 在线程池中执行训练
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                self.executor, self._train_model_sync, X, y
            )
            
            # 保存模型
            await self._save_model()
            
            # 记录模型指标
            await self._log_model_metrics(metrics, len(training_data))
            
            logger.info(f"模型训练完成，准确率: {metrics.get('r2_score', 0):.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return {}
    
    def _train_model_sync(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """同步执行模型训练（在线程池中运行）"""
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练测试分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = self.model.predict(X_test)
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        
        self.is_trained = True
        
        return {
            'mse': mse,
            'r2_score': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(X),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    async def _collect_training_data(self) -> List[Dict[str, Any]]:
        """收集训练数据"""
        training_data = []
        
        try:
            async with get_db_session() as session:
                # 获取最近30天的内容性能数据
                cutoff_date = datetime.utcnow() - timedelta(days=settings.ml.feature_window_days)
                
                result = await session.execute(
                    select(ContentPerformance, SchedulingTask)
                    .join(SchedulingTask, ContentPerformance.task_id == SchedulingTask.id)
                    .where(
                        and_(
                            ContentPerformance.published_time >= cutoff_date,
                            ContentPerformance.engagement_score.isnot(None),
                            ContentPerformance.views_24h > 0
                        )
                    )
                    .limit(10000)  # 限制数据量
                )
                
                for performance, task in result:
                    training_data.append({
                        'performance': performance,
                        'task': task
                    })
                
        except Exception as e:
            logger.error(f"收集训练数据失败: {e}")
            
        return training_data
    
    async def _prepare_training_data(
        self, 
        training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据的特征和标签"""
        features = []
        labels = []
        
        for data in training_data:
            try:
                performance = data['performance']
                task = data['task']
                
                # 提取特征
                feature_vector = await self._extract_features_for_task(task, performance)
                if feature_vector is None:
                    continue
                
                # 标签（参与度得分）
                label = performance.engagement_score or 0.0
                
                features.append(feature_vector)
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"处理训练数据时出错: {e}")
                continue
        
        return np.array(features), np.array(labels)
    
    async def _extract_features_for_task(
        self,
        task: SchedulingTask,
        performance: Optional[ContentPerformance] = None
    ) -> Optional[List[float]]:
        """为任务提取特征向量"""
        try:
            published_time = performance.published_time if performance else task.scheduled_time
            
            # 基础时间特征
            features = [
                published_time.hour,  # 小时
                published_time.weekday(),  # 星期
                published_time.month,  # 月份
                1 if published_time.weekday() >= 5 else 0,  # 是否周末
            ]
            
            # 内容特征
            content_length = len(task.content_body) if task.content_body else 0
            has_media = 1 if (task.content_metadata and 
                            task.content_metadata.get('has_media')) else 0
            hashtag_count = task.content_metadata.get('hashtag_count', 0) if task.content_metadata else 0
            
            features.extend([content_length, has_media, hashtag_count])
            
            # 历史性能特征
            async with get_db_session() as session:
                # 用户历史参与率
                user_avg_engagement = await self._get_user_avg_engagement(
                    session, task.user_id
                )
                
                # 平台平均参与率
                platform_avg_engagement = await self._get_platform_avg_engagement(
                    session, task.target_platforms
                )
                
                # 最近发布频率
                recent_frequency = await self._get_recent_post_frequency(
                    session, task.user_id
                )
                
                # 距离上次发布时间
                time_since_last = await self._get_time_since_last_post(
                    session, task.user_id, published_time
                )
                
                # 竞争性内容数量
                competitive_count = await self._get_competitive_post_count(
                    session, published_time
                )
            
            # 季节性因子
            seasonal_factor = self._calculate_seasonal_factor(published_time)
            
            features.extend([
                user_avg_engagement,
                platform_avg_engagement,
                recent_frequency,
                time_since_last,
                competitive_count,
                seasonal_factor
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
    
    async def _get_user_avg_engagement(
        self, 
        session: AsyncSession, 
        user_id: int
    ) -> float:
        """获取用户平均参与率"""
        try:
            result = await session.execute(
                select(func.avg(ContentPerformance.engagement_score))
                .join(SchedulingTask, ContentPerformance.task_id == SchedulingTask.id)
                .where(
                    and_(
                        SchedulingTask.user_id == user_id,
                        ContentPerformance.engagement_score.isnot(None)
                    )
                )
            )
            return result.scalar() or 0.0
        except Exception:
            return 0.0
    
    async def _get_platform_avg_engagement(
        self,
        session: AsyncSession,
        platforms: List[str]
    ) -> float:
        """获取平台平均参与率"""
        if not platforms:
            return 0.0
        
        try:
            result = await session.execute(
                select(func.avg(ContentPerformance.engagement_score))
                .where(
                    and_(
                        ContentPerformance.platform_name.in_(platforms),
                        ContentPerformance.engagement_score.isnot(None)
                    )
                )
            )
            return result.scalar() or 0.0
        except Exception:
            return 0.0
    
    async def _get_recent_post_frequency(
        self,
        session: AsyncSession,
        user_id: int
    ) -> float:
        """获取最近发布频率"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            result = await session.execute(
                select(func.count(SchedulingTask.id))
                .where(
                    and_(
                        SchedulingTask.user_id == user_id,
                        SchedulingTask.scheduled_time >= cutoff_date
                    )
                )
            )
            count = result.scalar() or 0
            return count / 7.0  # 每天平均发布数
        except Exception:
            return 0.0
    
    async def _get_time_since_last_post(
        self,
        session: AsyncSession,
        user_id: int,
        current_time: datetime
    ) -> int:
        """获取距离上次发布的时间（小时）"""
        try:
            result = await session.execute(
                select(func.max(SchedulingTask.scheduled_time))
                .where(
                    and_(
                        SchedulingTask.user_id == user_id,
                        SchedulingTask.scheduled_time < current_time
                    )
                )
            )
            last_post_time = result.scalar()
            
            if not last_post_time:
                return 168  # 默认一周
            
            delta = current_time - last_post_time
            return int(delta.total_seconds() / 3600)
        except Exception:
            return 24  # 默认24小时
    
    async def _get_competitive_post_count(
        self,
        session: AsyncSession,
        target_time: datetime
    ) -> int:
        """获取目标时间附近的竞争性内容数量"""
        try:
            # 查看前后1小时的发布数量
            start_time = target_time - timedelta(hours=1)
            end_time = target_time + timedelta(hours=1)
            
            result = await session.execute(
                select(func.count(SchedulingTask.id))
                .where(
                    and_(
                        SchedulingTask.scheduled_time.between(start_time, end_time),
                        SchedulingTask.status != 'cancelled'
                    )
                )
            )
            return result.scalar() or 0
        except Exception:
            return 0
    
    def _calculate_seasonal_factor(self, target_time: datetime) -> float:
        """计算季节性因子"""
        # 简单的季节性计算，基于一年中的天数
        day_of_year = target_time.timetuple().tm_yday
        seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.25)
        return seasonal_factor
    
    async def predict_optimal_time(
        self,
        user_id: int,
        platforms: List[str],
        content_metadata: Dict[str, Any],
        time_window_hours: int = 24
    ) -> datetime:
        """预测最优发布时间"""
        if not self.is_trained:
            # 如果模型未训练，返回基于规则的推荐时间
            return await self._rule_based_optimal_time(user_id, platforms)
        
        try:
            current_time = datetime.utcnow()
            best_time = current_time
            best_score = -1.0
            
            # 在未来时间窗口内每小时测试一次
            for hour_offset in range(1, time_window_hours + 1):
                candidate_time = current_time + timedelta(hours=hour_offset)
                
                # 创建模拟任务用于特征提取
                mock_task = self._create_mock_task(
                    user_id, platforms, content_metadata, candidate_time
                )
                
                # 预测该时间的参与度
                predicted_score = await self._predict_engagement_score(mock_task)
                
                if predicted_score > best_score:
                    best_score = predicted_score
                    best_time = candidate_time
            
            return best_time
            
        except Exception as e:
            logger.error(f"预测最优时间失败: {e}")
            return await self._rule_based_optimal_time(user_id, platforms)
    
    async def _predict_engagement_score(self, task: SchedulingTask) -> float:
        """预测任务的参与度得分"""
        try:
            # 提取特征
            features = await self._extract_features_for_task(task)
            if not features:
                return 0.0
            
            # 在线程池中执行预测
            loop = asyncio.get_event_loop()
            score = await loop.run_in_executor(
                self.executor, self._predict_score_sync, features
            )
            
            return max(0.0, score)  # 确保非负数
            
        except Exception as e:
            logger.error(f"预测参与度得分失败: {e}")
            return 0.0
    
    def _predict_score_sync(self, features: List[float]) -> float:
        """同步执行预测（在线程池中运行）"""
        if not self.is_trained or self.model is None or self.scaler is None:
            return 0.0
        
        # 标准化特征
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # 预测
        prediction = self.model.predict(X_scaled)[0]
        return float(prediction)
    
    def _create_mock_task(
        self,
        user_id: int,
        platforms: List[str],
        content_metadata: Dict[str, Any],
        scheduled_time: datetime
    ) -> SchedulingTask:
        """创建模拟任务用于预测"""
        return SchedulingTask(
            user_id=user_id,
            title="预测任务",
            content_body=content_metadata.get('content_body', ''),
            target_platforms=platforms,
            content_metadata=content_metadata,
            scheduled_time=scheduled_time
        )
    
    async def _rule_based_optimal_time(
        self,
        user_id: int,
        platforms: List[str]
    ) -> datetime:
        """基于规则的最优时间推荐"""
        try:
            async with get_db_session() as session:
                # 查询用户的历史最佳发布时间
                result = await session.execute(
                    select(UserBehaviorPattern)
                    .where(UserBehaviorPattern.user_id == user_id)
                    .limit(1)
                )
                
                pattern = result.scalar_one_or_none()
                
                if pattern and pattern.best_performing_times:
                    best_hours = pattern.best_performing_times.get('hours', [])
                    if best_hours:
                        # 选择最佳小时中最近的一个
                        current_time = datetime.utcnow()
                        current_hour = current_time.hour
                        
                        # 找到下一个最佳小时
                        for hour in sorted(best_hours):
                            if hour > current_hour:
                                target_time = current_time.replace(
                                    hour=hour, minute=0, second=0, microsecond=0
                                )
                                return target_time
                        
                        # 如果今天没有更好的时间，选择明天的第一个最佳小时
                        next_day = current_time + timedelta(days=1)
                        return next_day.replace(
                            hour=best_hours[0], minute=0, second=0, microsecond=0
                        )
                
                # 默认推荐时间（基于一般经验）
                default_hours = [9, 12, 15, 18, 20]  # 一般的最佳发布时间
                current_time = datetime.utcnow()
                current_hour = current_time.hour
                
                for hour in default_hours:
                    if hour > current_hour:
                        return current_time.replace(
                            hour=hour, minute=0, second=0, microsecond=0
                        )
                
                # 如果当天没有好时间，推荐明天早上9点
                tomorrow = current_time + timedelta(days=1)
                return tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)
                
        except Exception as e:
            logger.error(f"基于规则的时间推荐失败: {e}")
            # 最后的后备方案：2小时后
            return datetime.utcnow() + timedelta(hours=2)
    
    async def _save_model(self):
        """保存模型到文件"""
        try:
            loop = asyncio.get_event_loop()
            
            # 保存模型
            model_file = self.model_dir / "scheduler_model.pkl"
            await loop.run_in_executor(
                self.executor, self._save_pickle_file, self.model, model_file
            )
            
            # 保存缩放器
            scaler_file = self.model_dir / "scaler.pkl"
            await loop.run_in_executor(
                self.executor, self._save_pickle_file, self.scaler, scaler_file
            )
            
            # 保存元数据
            meta_file = self.model_dir / "model_meta.json"
            meta_data = {
                'version': self.model_version,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'created_at': datetime.utcnow().isoformat()
            }
            
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def _save_pickle_file(self, obj, file_path: Path):
        """线程安全的pickle文件保存"""
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    
    async def _log_model_metrics(
        self,
        metrics: Dict[str, float],
        training_samples: int
    ):
        """记录模型训练指标"""
        try:
            async with get_db_session() as session:
                model_metrics = MLModelMetrics(
                    model_name="ScheduleOptimizer",
                    model_version=self.model_version,
                    training_start_time=datetime.utcnow(),
                    training_end_time=datetime.utcnow(),
                    training_samples=training_samples,
                    model_parameters=settings.ml.model_params,
                    feature_importance=metrics.get('feature_importance', {}),
                    r2_score=metrics.get('r2_score'),
                    mse_score=metrics.get('mse'),
                    cv_mean_score=metrics.get('cv_mean'),
                    cv_std_score=metrics.get('cv_std'),
                    is_active=True
                )
                
                session.add(model_metrics)
                await session.commit()
                
        except Exception as e:
            logger.error(f"记录模型指标失败: {e}")


class OptimizationService:
    """优化服务主类"""
    
    def __init__(self):
        self.optimizer = ScheduleOptimizer()
        self.settings = get_settings()
        
    async def initialize(self):
        """初始化服务"""
        await self.optimizer.initialize_model()
    
    async def optimize_scheduling_time(
        self,
        user_id: int,
        platforms: List[str],
        content_metadata: Dict[str, Any],
        preferred_time: Optional[datetime] = None
    ) -> OptimizationResult:
        """优化调度时间"""
        original_time = preferred_time or datetime.utcnow()
        
        try:
            # 预测最优时间
            optimized_time = await self.optimizer.predict_optimal_time(
                user_id, platforms, content_metadata
            )
            
            # 如果用户有偏好时间，限制优化范围
            if preferred_time:
                max_deviation = timedelta(hours=4)  # 最多偏离4小时
                if abs((optimized_time - preferred_time).total_seconds()) > max_deviation.total_seconds():
                    # 在允许范围内选择最佳时间
                    optimized_time = await self._constrained_optimization(
                        user_id, platforms, content_metadata, preferred_time, max_deviation
                    )
            
            # 计算优化指标
            optimization_score = await self._calculate_optimization_score(
                original_time, optimized_time, user_id, platforms
            )
            
            # 预测性能指标
            predicted_metrics = await self._predict_performance_metrics(
                user_id, platforms, content_metadata, optimized_time
            )
            
            return OptimizationResult(
                original_time=original_time,
                optimized_time=optimized_time,
                optimization_score=optimization_score,
                confidence_score=0.8,  # 基于模型训练结果的置信度
                predicted_metrics=predicted_metrics,
                optimization_factors={
                    'time_shift_hours': (optimized_time - original_time).total_seconds() / 3600,
                    'optimization_method': 'ml_prediction' if self.optimizer.is_trained else 'rule_based'
                },
                model_version=self.optimizer.model_version,
                model_params=self.settings.ml.model_params
            )
            
        except Exception as e:
            logger.error(f"优化调度时间失败: {e}")
            # 返回原始时间作为后备
            return OptimizationResult(
                original_time=original_time,
                optimized_time=original_time,
                optimization_score=0.0,
                confidence_score=0.0,
                predicted_metrics={},
                optimization_factors={'error': str(e)},
                model_version=self.optimizer.model_version,
                model_params={}
            )
    
    async def _constrained_optimization(
        self,
        user_id: int,
        platforms: List[str],
        content_metadata: Dict[str, Any],
        preferred_time: datetime,
        max_deviation: timedelta
    ) -> datetime:
        """在约束范围内进行优化"""
        start_time = preferred_time - max_deviation
        end_time = preferred_time + max_deviation
        
        best_time = preferred_time
        best_score = -1.0
        
        # 每小时测试一次
        current_time = start_time
        while current_time <= end_time:
            mock_task = self.optimizer._create_mock_task(
                user_id, platforms, content_metadata, current_time
            )
            score = await self.optimizer._predict_engagement_score(mock_task)
            
            if score > best_score:
                best_score = score
                best_time = current_time
            
            current_time += timedelta(hours=1)
        
        return best_time
    
    async def _calculate_optimization_score(
        self,
        original_time: datetime,
        optimized_time: datetime,
        user_id: int,
        platforms: List[str]
    ) -> float:
        """计算优化得分"""
        try:
            # 创建模拟任务
            mock_task_original = self.optimizer._create_mock_task(
                user_id, platforms, {}, original_time
            )
            mock_task_optimized = self.optimizer._create_mock_task(
                user_id, platforms, {}, optimized_time
            )
            
            # 预测两个时间点的性能
            original_score = await self.optimizer._predict_engagement_score(mock_task_original)
            optimized_score = await self.optimizer._predict_engagement_score(mock_task_optimized)
            
            # 计算改进百分比
            if original_score > 0:
                improvement = (optimized_score - original_score) / original_score * 100
                return max(0.0, min(100.0, improvement))  # 限制在0-100之间
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算优化得分失败: {e}")
            return 0.0
    
    async def _predict_performance_metrics(
        self,
        user_id: int,
        platforms: List[str],
        content_metadata: Dict[str, Any],
        scheduled_time: datetime
    ) -> Dict[str, float]:
        """预测性能指标"""
        try:
            mock_task = self.optimizer._create_mock_task(
                user_id, platforms, content_metadata, scheduled_time
            )
            
            engagement_score = await self.optimizer._predict_engagement_score(mock_task)
            
            # 基于参与度得分估算其他指标
            base_views = 1000  # 假设基础浏览量
            estimated_views = base_views * (1 + engagement_score / 100)
            
            return {
                'engagement_rate': engagement_score,
                'estimated_views': estimated_views,
                'estimated_likes': estimated_views * 0.05,
                'estimated_shares': estimated_views * 0.02,
                'estimated_comments': estimated_views * 0.01,
                'reach': estimated_views * 0.3
            }
            
        except Exception as e:
            logger.error(f"预测性能指标失败: {e}")
            return {}
    
    async def retrain_model(self) -> Dict[str, float]:
        """重新训练模型"""
        return await self.optimizer.train_model()
    
    async def predict_optimal_time(
        self,
        user_id: int,
        platforms: List[str],
        content_metadata: Dict[str, Any]
    ) -> datetime:
        """预测最优发布时间"""
        return await self.optimizer.predict_optimal_time(
            user_id, platforms, content_metadata
        )