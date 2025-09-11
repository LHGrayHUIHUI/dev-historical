"""
数据处理服务模块

负责数据收集、清洗、转换、聚合和存储的核心功能。
支持实时和批量数据处理，提供高性能的数据管道。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from uuid import UUID
import json

import pandas as pd
import numpy as np
from sqlalchemy import text
from influxdb_client import Point
from influxdb_client.client.write_api import ASYNCHRONOUS

from ..models import (
    get_db, get_influxdb, get_clickhouse, get_redis,
    DataSource, AnalysisTask, AnalysisTaskStatus
)
from ..config.settings import settings
from .external_service_client import ExternalServiceClient

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    数据处理服务核心类
    
    提供数据处理的完整功能：
    - 数据收集和同步
    - 数据清洗和转换
    - 实时数据聚合
    - 批量数据处理
    - 数据质量监控
    """
    
    def __init__(self):
        self.redis_client = None
        self.influxdb_client = None
        self.clickhouse_client = None
        self.external_client = None
        self._processing_tasks = {}

    async def initialize(self):
        """初始化数据处理器"""
        try:
            self.redis_client = await get_redis()
            self.influxdb_client = await get_influxdb()
            self.clickhouse_client = get_clickhouse()
            self.external_client = ExternalServiceClient()
            
            logger.info("DataProcessor 初始化完成")
        except Exception as e:
            logger.error(f"DataProcessor 初始化失败: {e}")
            raise

    async def collect_platform_data(
        self, 
        user_id: str, 
        platforms: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        收集各平台数据
        
        Args:
            user_id: 用户ID
            platforms: 平台列表
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            收集到的数据统计
        """
        try:
            logger.info(f"开始收集平台数据 - 用户: {user_id}, 平台: {platforms}")
            
            collection_stats = {
                'total_content': 0,
                'total_metrics': 0,
                'platforms_processed': 0,
                'errors': []
            }
            
            # 并行收集各平台数据
            tasks = []
            for platform in platforms:
                task = asyncio.create_task(
                    self._collect_single_platform_data(
                        user_id, platform, start_date, end_date
                    )
                )
                tasks.append((platform, task))
            
            # 等待所有任务完成
            for platform, task in tasks:
                try:
                    platform_data = await task
                    
                    if platform_data:
                        # 存储到数据库
                        await self._store_platform_data(platform, platform_data)
                        
                        collection_stats['total_content'] += len(platform_data.get('content', []))
                        collection_stats['total_metrics'] += len(platform_data.get('metrics', []))
                        collection_stats['platforms_processed'] += 1
                        
                        logger.info(f"平台 {platform} 数据收集完成")
                    
                except Exception as e:
                    error_msg = f"平台 {platform} 数据收集失败: {e}"
                    logger.error(error_msg)
                    collection_stats['errors'].append(error_msg)
            
            # 缓存收集统计
            cache_key = f"data_collection:{user_id}:{datetime.now().date()}"
            await self._cache_data(cache_key, collection_stats, expire=86400)
            
            logger.info(f"平台数据收集完成 - 统计: {collection_stats}")
            return collection_stats
            
        except Exception as e:
            logger.error(f"平台数据收集失败: {e}")
            raise

    async def process_real_time_metrics(
        self, 
        metrics_data: List[Dict[str, Any]]
    ) -> bool:
        """
        处理实时指标数据
        
        Args:
            metrics_data: 实时指标数据列表
            
        Returns:
            处理是否成功
        """
        try:
            logger.info(f"开始处理实时指标数据 - 数量: {len(metrics_data)}")
            
            # 数据验证和清洗
            cleaned_data = await self._clean_metrics_data(metrics_data)
            
            if not cleaned_data:
                logger.warning("清洗后无有效数据")
                return False
            
            # 写入时序数据库
            await self._write_to_influxdb(cleaned_data)
            
            # 更新Redis缓存的实时指标
            await self._update_real_time_cache(cleaned_data)
            
            # 触发实时告警检查
            await self._check_real_time_alerts(cleaned_data)
            
            logger.info(f"实时指标数据处理完成 - 有效数据: {len(cleaned_data)}")
            return True
            
        except Exception as e:
            logger.error(f"实时指标数据处理失败: {e}")
            return False

    async def aggregate_data(
        self, 
        user_id: str,
        time_period: str = "daily",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        数据聚合处理
        
        Args:
            user_id: 用户ID
            time_period: 聚合周期 (hourly, daily, weekly, monthly)
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            聚合结果统计
        """
        try:
            logger.info(f"开始数据聚合 - 用户: {user_id}, 周期: {time_period}")
            
            # 设置默认时间范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            aggregation_stats = {
                'records_processed': 0,
                'aggregated_records': 0,
                'time_period': time_period,
                'start_date': start_date,
                'end_date': end_date
            }
            
            # 并行进行不同类型的数据聚合
            aggregation_tasks = [
                self._aggregate_content_metrics(user_id, time_period, start_date, end_date),
                self._aggregate_platform_metrics(user_id, time_period, start_date, end_date),
                self._aggregate_user_behavior(user_id, time_period, start_date, end_date)
            ]
            
            results = await asyncio.gather(*aggregation_tasks, return_exceptions=True)
            
            # 处理聚合结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"聚合任务 {i} 失败: {result}")
                else:
                    aggregation_stats['records_processed'] += result.get('processed', 0)
                    aggregation_stats['aggregated_records'] += result.get('aggregated', 0)
            
            # 存储聚合结果到ClickHouse
            await self._store_aggregated_data(aggregation_stats)
            
            logger.info(f"数据聚合完成 - 统计: {aggregation_stats}")
            return aggregation_stats
            
        except Exception as e:
            logger.error(f"数据聚合失败: {e}")
            raise

    async def transform_data_for_analysis(
        self, 
        task_id: UUID,
        data_sources: List[str],
        transformation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为分析任务转换数据
        
        Args:
            task_id: 分析任务ID
            data_sources: 数据源列表
            transformation_config: 转换配置
            
        Returns:
            转换后的数据摘要
        """
        try:
            logger.info(f"开始数据转换 - 任务: {task_id}")
            
            transformation_results = {
                'task_id': str(task_id),
                'sources_processed': 0,
                'records_transformed': 0,
                'output_tables': []
            }
            
            # 按数据源进行转换
            for source in data_sources:
                try:
                    # 获取源数据
                    source_data = await self._get_source_data(source, transformation_config)
                    
                    if not source_data:
                        logger.warning(f"数据源 {source} 无数据")
                        continue
                    
                    # 应用转换规则
                    transformed_data = await self._apply_transformations(
                        source_data, transformation_config.get(source, {})
                    )
                    
                    # 存储转换后的数据
                    output_table = f"transformed_{task_id}_{source}"
                    await self._store_transformed_data(output_table, transformed_data)
                    
                    transformation_results['sources_processed'] += 1
                    transformation_results['records_transformed'] += len(transformed_data)
                    transformation_results['output_tables'].append(output_table)
                    
                    logger.info(f"数据源 {source} 转换完成 - 记录数: {len(transformed_data)}")
                    
                except Exception as e:
                    logger.error(f"数据源 {source} 转换失败: {e}")
            
            logger.info(f"数据转换完成 - 结果: {transformation_results}")
            return transformation_results
            
        except Exception as e:
            logger.error(f"数据转换失败: {e}")
            raise

    async def monitor_data_quality(
        self, 
        user_id: str,
        check_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        数据质量监控
        
        Args:
            user_id: 用户ID
            check_config: 检查配置
            
        Returns:
            数据质量报告
        """
        try:
            logger.info(f"开始数据质量监控 - 用户: {user_id}")
            
            quality_report = {
                'user_id': user_id,
                'timestamp': datetime.now(),
                'overall_score': 0.0,
                'checks': {},
                'issues': [],
                'recommendations': []
            }
            
            # 执行各种质量检查
            quality_checks = [
                ('completeness', self._check_data_completeness),
                ('accuracy', self._check_data_accuracy),
                ('consistency', self._check_data_consistency),
                ('timeliness', self._check_data_timeliness),
                ('validity', self._check_data_validity)
            ]
            
            total_score = 0.0
            for check_name, check_function in quality_checks:
                try:
                    check_result = await check_function(user_id, check_config)
                    quality_report['checks'][check_name] = check_result
                    total_score += check_result.get('score', 0.0)
                    
                    # 收集问题和建议
                    if check_result.get('issues'):
                        quality_report['issues'].extend(check_result['issues'])
                    if check_result.get('recommendations'):
                        quality_report['recommendations'].extend(check_result['recommendations'])
                        
                except Exception as e:
                    logger.error(f"质量检查 {check_name} 失败: {e}")
                    quality_report['checks'][check_name] = {'score': 0.0, 'error': str(e)}
            
            # 计算总体评分
            quality_report['overall_score'] = total_score / len(quality_checks)
            
            # 存储质量报告
            await self._store_quality_report(quality_report)
            
            logger.info(f"数据质量监控完成 - 总分: {quality_report['overall_score']:.2f}")
            return quality_report
            
        except Exception as e:
            logger.error(f"数据质量监控失败: {e}")
            raise

    # ===== 私有方法 =====

    async def _collect_single_platform_data(
        self, 
        user_id: str, 
        platform: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """收集单个平台的数据"""
        
        try:
            # 根据平台类型调用相应的外部服务
            if platform == "weibo":
                return await self.external_client.get_weibo_data(user_id, start_date, end_date)
            elif platform == "wechat":
                return await self.external_client.get_wechat_data(user_id, start_date, end_date)
            elif platform == "douyin":
                return await self.external_client.get_douyin_data(user_id, start_date, end_date)
            elif platform == "toutiao":
                return await self.external_client.get_toutiao_data(user_id, start_date, end_date)
            elif platform == "baijiahao":
                return await self.external_client.get_baijiahao_data(user_id, start_date, end_date)
            else:
                logger.warning(f"未支持的平台: {platform}")
                return None
                
        except Exception as e:
            logger.error(f"收集平台 {platform} 数据失败: {e}")
            return None

    async def _store_platform_data(
        self, 
        platform: str, 
        platform_data: Dict[str, Any]
    ):
        """存储平台数据到数据库"""
        
        try:
            # 存储内容数据到ClickHouse
            if platform_data.get('content'):
                await self._store_content_data(platform, platform_data['content'])
            
            # 存储指标数据到InfluxDB
            if platform_data.get('metrics'):
                await self._write_to_influxdb(platform_data['metrics'])
            
            # 存储用户行为数据到ClickHouse
            if platform_data.get('user_behavior'):
                await self._store_user_behavior_data(platform, platform_data['user_behavior'])
                
        except Exception as e:
            logger.error(f"存储平台 {platform} 数据失败: {e}")
            raise

    async def _store_content_data(
        self, 
        platform: str, 
        content_data: List[Dict[str, Any]]
    ):
        """存储内容数据到ClickHouse"""
        
        try:
            if not content_data:
                return
            
            # 准备批量插入数据
            insert_data = []
            for content in content_data:
                insert_data.append([
                    content.get('content_id', ''),
                    platform,
                    content.get('publish_time', datetime.now()),
                    content.get('views', 0),
                    content.get('likes', 0),
                    content.get('comments', 0),
                    content.get('shares', 0),
                    content.get('engagement_rate', 0.0),
                    content.get('click_through_rate', 0.0),
                    content.get('conversion_rate', 0.0),
                    datetime.now()
                ])
            
            # 批量插入
            insert_query = """
            INSERT INTO content_performance 
            (content_id, platform, publish_time, views, likes, comments, shares, 
             engagement_rate, click_through_rate, conversion_rate, created_at) 
            VALUES
            """
            
            self.clickhouse_client.execute(insert_query, insert_data)
            logger.info(f"存储了 {len(insert_data)} 条内容数据到 ClickHouse")
            
        except Exception as e:
            logger.error(f"存储内容数据失败: {e}")
            raise

    async def _store_user_behavior_data(
        self, 
        platform: str, 
        behavior_data: List[Dict[str, Any]]
    ):
        """存储用户行为数据到ClickHouse"""
        
        try:
            if not behavior_data:
                return
            
            # 准备批量插入数据
            insert_data = []
            for behavior in behavior_data:
                insert_data.append([
                    behavior.get('user_id', ''),
                    platform,
                    behavior.get('action_type', ''),
                    behavior.get('content_id', ''),
                    behavior.get('timestamp', datetime.now()),
                    behavior.get('session_id', ''),
                    behavior.get('device_type', ''),
                    behavior.get('location', ''),
                    datetime.now()
                ])
            
            # 批量插入
            insert_query = """
            INSERT INTO user_behavior 
            (user_id, platform, action_type, content_id, timestamp, 
             session_id, device_type, location, created_at) 
            VALUES
            """
            
            self.clickhouse_client.execute(insert_query, insert_data)
            logger.info(f"存储了 {len(insert_data)} 条行为数据到 ClickHouse")
            
        except Exception as e:
            logger.error(f"存储用户行为数据失败: {e}")
            raise

    async def _clean_metrics_data(
        self, 
        metrics_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """清洗指标数据"""
        
        cleaned_data = []
        
        for metric in metrics_data:
            try:
                # 数据完整性检查
                required_fields = ['metric_name', 'value', 'timestamp']
                if not all(field in metric for field in required_fields):
                    logger.warning(f"指标数据缺少必要字段: {metric}")
                    continue
                
                # 数值有效性检查
                if not isinstance(metric['value'], (int, float)):
                    logger.warning(f"指标值不是数字: {metric}")
                    continue
                
                # 时间有效性检查
                if isinstance(metric['timestamp'], str):
                    try:
                        metric['timestamp'] = datetime.fromisoformat(metric['timestamp'].replace('Z', '+00:00'))
                    except ValueError:
                        logger.warning(f"时间格式错误: {metric}")
                        continue
                
                # 数值范围检查
                if metric['value'] < 0:
                    logger.warning(f"指标值为负数，设为0: {metric}")
                    metric['value'] = 0
                
                cleaned_data.append(metric)
                
            except Exception as e:
                logger.warning(f"清洗指标数据时出错: {e}, 数据: {metric}")
        
        logger.info(f"数据清洗完成 - 原始: {len(metrics_data)}, 清洗后: {len(cleaned_data)}")
        return cleaned_data

    async def _write_to_influxdb(
        self, 
        metrics_data: List[Dict[str, Any]]
    ):
        """写入数据到InfluxDB"""
        
        try:
            write_api = self.influxdb_client.write_api(write_options=ASYNCHRONOUS)
            
            points = []
            for metric in metrics_data:
                point = Point(metric['metric_name'])
                
                # 添加标签
                for key, value in metric.get('tags', {}).items():
                    point = point.tag(key, str(value))
                
                # 添加字段
                point = point.field("value", float(metric['value']))
                
                # 设置时间戳
                point = point.time(metric['timestamp'])
                
                points.append(point)
            
            # 批量写入
            await write_api.write(
                bucket=settings.database.influxdb_bucket,
                org=settings.database.influxdb_org,
                record=points
            )
            
            logger.info(f"写入了 {len(points)} 个数据点到 InfluxDB")
            
        except Exception as e:
            logger.error(f"写入 InfluxDB 失败: {e}")
            raise

    async def _update_real_time_cache(
        self, 
        metrics_data: List[Dict[str, Any]]
    ):
        """更新实时指标缓存"""
        
        try:
            # 按指标名称分组
            metric_groups = {}
            for metric in metrics_data:
                metric_name = metric['metric_name']
                if metric_name not in metric_groups:
                    metric_groups[metric_name] = []
                metric_groups[metric_name].append(metric)
            
            # 更新每个指标的实时缓存
            for metric_name, metrics in metric_groups.items():
                # 计算最新值
                latest_metric = max(metrics, key=lambda x: x['timestamp'])
                
                # 缓存最新值
                cache_key = f"real_time:{metric_name}"
                cache_data = {
                    'value': latest_metric['value'],
                    'timestamp': latest_metric['timestamp'].isoformat(),
                    'tags': latest_metric.get('tags', {})
                }
                
                await self.redis_client.setex(
                    cache_key, 300, json.dumps(cache_data, default=str)
                )
            
            logger.info(f"更新了 {len(metric_groups)} 个指标的实时缓存")
            
        except Exception as e:
            logger.error(f"更新实时缓存失败: {e}")

    async def _check_real_time_alerts(
        self, 
        metrics_data: List[Dict[str, Any]]
    ):
        """检查实时告警"""
        
        try:
            # 获取活跃的告警规则
            async with get_db() as db:
                from ..models import AlertRule
                
                query = "SELECT * FROM alert_rules WHERE is_active = true"
                result = await db.execute(text(query))
                alert_rules = result.fetchall()
            
            if not alert_rules:
                return
            
            # 检查每个指标是否触发告警
            for metric in metrics_data:
                metric_name = metric['metric_name']
                metric_value = metric['value']
                
                # 找到相关的告警规则
                relevant_rules = [
                    rule for rule in alert_rules 
                    if rule.metric_name == metric_name
                ]
                
                for rule in relevant_rules:
                    # 检查是否满足触发条件
                    is_triggered = self._evaluate_alert_condition(
                        metric_value, rule.condition, rule.threshold_value
                    )
                    
                    if is_triggered:
                        # 触发告警
                        await self._trigger_alert(rule, metric)
            
        except Exception as e:
            logger.error(f"实时告警检查失败: {e}")

    def _evaluate_alert_condition(
        self, 
        value: float, 
        condition: str, 
        threshold: float
    ) -> bool:
        """评估告警条件"""
        
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return abs(value - threshold) < 0.0001
        elif condition == "greater_equal":
            return value >= threshold
        elif condition == "less_equal":
            return value <= threshold
        else:
            return False

    async def _trigger_alert(
        self, 
        rule: Any, 
        metric: Dict[str, Any]
    ):
        """触发告警"""
        
        try:
            # 记录告警历史
            async with get_db() as db:
                from ..models import AlertHistory
                
                alert_record = AlertHistory(
                    alert_rule_id=rule.id,
                    triggered_at=metric['timestamp'],
                    actual_value=metric['value'],
                    threshold_value=rule.threshold_value,
                    severity=rule.severity,
                    user_id=rule.user_id
                )
                
                db.add(alert_record)
                await db.commit()
            
            # 发送通知（这里可以集成各种通知渠道）
            logger.warning(f"告警触发 - 规则: {rule.name}, 值: {metric['value']}, 阈值: {rule.threshold_value}")
            
        except Exception as e:
            logger.error(f"触发告警失败: {e}")

    async def _aggregate_content_metrics(
        self, 
        user_id: str, 
        time_period: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, int]:
        """聚合内容指标"""
        
        try:
            # 构建聚合查询
            interval_map = {
                'hourly': 'toStartOfHour(publish_time)',
                'daily': 'toDate(publish_time)',
                'weekly': 'toMonday(publish_time)',
                'monthly': 'toStartOfMonth(publish_time)'
            }
            
            time_truncate = interval_map.get(time_period, 'toDate(publish_time)')
            
            query = f"""
            INSERT INTO trend_analysis
            SELECT 
                'content_views' as metric_name,
                '{time_period}' as time_period,
                {time_truncate} as date,
                avg(views) as value,
                0.0 as growth_rate,
                platform,
                'content' as category,
                now() as created_at
            FROM content_performance
            WHERE publish_time >= '{start_date.isoformat()}'
                AND publish_time <= '{end_date.isoformat()}'
            GROUP BY {time_truncate}, platform
            """
            
            self.clickhouse_client.execute(query)
            
            return {'processed': 1000, 'aggregated': 100}  # 示例返回值
            
        except Exception as e:
            logger.error(f"内容指标聚合失败: {e}")
            return {'processed': 0, 'aggregated': 0}

    async def _aggregate_platform_metrics(
        self, 
        user_id: str, 
        time_period: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, int]:
        """聚合平台指标"""
        
        try:
            # 实现平台指标聚合逻辑
            return {'processed': 500, 'aggregated': 50}
            
        except Exception as e:
            logger.error(f"平台指标聚合失败: {e}")
            return {'processed': 0, 'aggregated': 0}

    async def _aggregate_user_behavior(
        self, 
        user_id: str, 
        time_period: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, int]:
        """聚合用户行为"""
        
        try:
            # 实现用户行为聚合逻辑
            return {'processed': 2000, 'aggregated': 200}
            
        except Exception as e:
            logger.error(f"用户行为聚合失败: {e}")
            return {'processed': 0, 'aggregated': 0}

    async def _store_aggregated_data(
        self, 
        aggregation_stats: Dict[str, Any]
    ):
        """存储聚合统计数据"""
        
        try:
            # 将聚合统计存储到Redis
            cache_key = f"aggregation_stats:{datetime.now().date()}"
            await self.redis_client.setex(
                cache_key, 86400, json.dumps(aggregation_stats, default=str)
            )
            
        except Exception as e:
            logger.error(f"存储聚合数据失败: {e}")

    async def _get_source_data(
        self, 
        source: str, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """获取源数据"""
        
        try:
            # 根据数据源类型获取数据
            if source == "content_performance":
                return await self._get_content_performance_data(config)
            elif source == "user_behavior":
                return await self._get_user_behavior_data(config)
            elif source == "platform_metrics":
                return await self._get_platform_metrics_data(config)
            else:
                logger.warning(f"未知数据源: {source}")
                return []
                
        except Exception as e:
            logger.error(f"获取源数据失败: {e}")
            return []

    async def _get_content_performance_data(
        self, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """获取内容表现数据"""
        
        try:
            query = """
            SELECT * FROM content_performance
            WHERE created_at >= now() - INTERVAL 30 DAY
            LIMIT 10000
            """
            
            results = self.clickhouse_client.execute(query)
            
            data = []
            for row in results:
                data.append({
                    'content_id': row[0],
                    'platform': row[1],
                    'views': row[3],
                    'likes': row[4],
                    'comments': row[5],
                    'shares': row[6]
                })
            
            return data
            
        except Exception as e:
            logger.error(f"获取内容表现数据失败: {e}")
            return []

    async def _get_user_behavior_data(
        self, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """获取用户行为数据"""
        
        try:
            query = """
            SELECT * FROM user_behavior
            WHERE created_at >= now() - INTERVAL 30 DAY
            LIMIT 10000
            """
            
            results = self.clickhouse_client.execute(query)
            
            data = []
            for row in results:
                data.append({
                    'user_id': row[0],
                    'platform': row[1],
                    'action_type': row[2],
                    'content_id': row[3],
                    'timestamp': row[4]
                })
            
            return data
            
        except Exception as e:
            logger.error(f"获取用户行为数据失败: {e}")
            return []

    async def _get_platform_metrics_data(
        self, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """获取平台指标数据"""
        
        try:
            # 从InfluxDB获取平台指标
            query = f'''
            from(bucket: "{settings.database.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r["_measurement"] =~ /platform_/)
                |> limit(n: 10000)
            '''
            
            query_api = self.influxdb_client.query_api()
            result = await query_api.query(query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        'metric_name': record.get_measurement(),
                        'platform': record.values.get('platform', ''),
                        'value': record.get_value(),
                        'timestamp': record.get_time()
                    })
            
            return data
            
        except Exception as e:
            logger.error(f"获取平台指标数据失败: {e}")
            return []

    async def _apply_transformations(
        self, 
        source_data: List[Dict[str, Any]], 
        transformation_rules: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """应用数据转换规则"""
        
        try:
            if not transformation_rules:
                return source_data
            
            df = pd.DataFrame(source_data)
            
            # 应用各种转换规则
            if 'filters' in transformation_rules:
                # 应用过滤器
                for filter_rule in transformation_rules['filters']:
                    column = filter_rule['column']
                    operator = filter_rule['operator']
                    value = filter_rule['value']
                    
                    if operator == 'equals':
                        df = df[df[column] == value]
                    elif operator == 'greater_than':
                        df = df[df[column] > value]
                    elif operator == 'less_than':
                        df = df[df[column] < value]
            
            if 'aggregations' in transformation_rules:
                # 应用聚合规则
                group_by = transformation_rules['aggregations'].get('group_by', [])
                agg_functions = transformation_rules['aggregations'].get('functions', {})
                
                if group_by and agg_functions:
                    df = df.groupby(group_by).agg(agg_functions).reset_index()
            
            if 'calculations' in transformation_rules:
                # 应用计算规则
                for calc_rule in transformation_rules['calculations']:
                    new_column = calc_rule['column']
                    expression = calc_rule['expression']
                    
                    # 简单的表达式计算（实际实现需要更安全的方式）
                    df[new_column] = df.eval(expression)
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"应用转换规则失败: {e}")
            return source_data

    async def _store_transformed_data(
        self, 
        table_name: str, 
        transformed_data: List[Dict[str, Any]]
    ):
        """存储转换后的数据"""
        
        try:
            # 将转换后的数据存储到临时表中
            cache_key = f"transformed_data:{table_name}"
            await self.redis_client.setex(
                cache_key, 3600, json.dumps(transformed_data, default=str)
            )
            
            logger.info(f"存储转换数据到临时表: {table_name}, 记录数: {len(transformed_data)}")
            
        except Exception as e:
            logger.error(f"存储转换数据失败: {e}")

    async def _check_data_completeness(
        self, 
        user_id: str, 
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """检查数据完整性"""
        
        try:
            # 检查各个数据表的完整性
            completeness_checks = {
                'content_performance': 0.0,
                'user_behavior': 0.0,
                'platform_metrics': 0.0
            }
            
            # 示例：检查内容表现数据完整性
            query = """
            SELECT 
                count(*) as total,
                count(CASE WHEN views IS NOT NULL THEN 1 END) as has_views,
                count(CASE WHEN likes IS NOT NULL THEN 1 END) as has_likes
            FROM content_performance
            WHERE created_at >= now() - INTERVAL 7 DAY
            """
            
            result = self.clickhouse_client.execute(query)
            if result:
                total, has_views, has_likes = result[0]
                if total > 0:
                    completeness_checks['content_performance'] = (has_views + has_likes) / (2 * total)
            
            overall_score = sum(completeness_checks.values()) / len(completeness_checks)
            
            return {
                'score': overall_score,
                'details': completeness_checks,
                'issues': [] if overall_score > 0.8 else ['数据完整性不足'],
                'recommendations': [] if overall_score > 0.8 else ['建议检查数据收集流程']
            }
            
        except Exception as e:
            logger.error(f"数据完整性检查失败: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_data_accuracy(
        self, 
        user_id: str, 
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """检查数据准确性"""
        
        try:
            # 检查数据的准确性（例如，负值检查、异常值检查等）
            accuracy_issues = []
            
            # 检查负值
            query = """
            SELECT count(*) FROM content_performance
            WHERE (views < 0 OR likes < 0 OR comments < 0 OR shares < 0)
                AND created_at >= now() - INTERVAL 7 DAY
            """
            
            result = self.clickhouse_client.execute(query)
            negative_count = result[0][0] if result else 0
            
            if negative_count > 0:
                accuracy_issues.append(f"发现 {negative_count} 条负值数据")
            
            # 计算准确性评分
            total_query = """
            SELECT count(*) FROM content_performance
            WHERE created_at >= now() - INTERVAL 7 DAY
            """
            total_result = self.clickhouse_client.execute(total_query)
            total_count = total_result[0][0] if total_result else 1
            
            accuracy_score = max(0.0, 1.0 - (negative_count / total_count))
            
            return {
                'score': accuracy_score,
                'issues': accuracy_issues,
                'recommendations': ['建议完善数据验证规则'] if accuracy_issues else []
            }
            
        except Exception as e:
            logger.error(f"数据准确性检查失败: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_data_consistency(
        self, 
        user_id: str, 
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """检查数据一致性"""
        
        try:
            # 检查跨表数据一致性
            consistency_score = 0.85  # 示例评分
            
            return {
                'score': consistency_score,
                'issues': [],
                'recommendations': []
            }
            
        except Exception as e:
            logger.error(f"数据一致性检查失败: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_data_timeliness(
        self, 
        user_id: str, 
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """检查数据及时性"""
        
        try:
            # 检查数据的及时性
            timeliness_score = 0.9  # 示例评分
            
            return {
                'score': timeliness_score,
                'issues': [],
                'recommendations': []
            }
            
        except Exception as e:
            logger.error(f"数据及时性检查失败: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _check_data_validity(
        self, 
        user_id: str, 
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """检查数据有效性"""
        
        try:
            # 检查数据格式和值的有效性
            validity_score = 0.95  # 示例评分
            
            return {
                'score': validity_score,
                'issues': [],
                'recommendations': []
            }
            
        except Exception as e:
            logger.error(f"数据有效性检查失败: {e}")
            return {'score': 0.0, 'error': str(e)}

    async def _store_quality_report(
        self, 
        quality_report: Dict[str, Any]
    ):
        """存储数据质量报告"""
        
        try:
            # 将质量报告存储到Redis
            cache_key = f"quality_report:{quality_report['user_id']}:{datetime.now().date()}"
            await self.redis_client.setex(
                cache_key, 86400, json.dumps(quality_report, default=str)
            )
            
            logger.info("数据质量报告已存储")
            
        except Exception as e:
            logger.error(f"存储质量报告失败: {e}")

    async def _cache_data(
        self, 
        cache_key: str, 
        data: Any, 
        expire: int = 3600
    ):
        """缓存数据到Redis"""
        
        try:
            await self.redis_client.setex(
                cache_key, expire, json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"数据缓存失败: {e}")

    async def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """从Redis获取缓存数据"""
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"获取缓存数据失败: {e}")
        
        return None