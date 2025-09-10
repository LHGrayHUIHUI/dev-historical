"""
使用统计跟踪器

负责记录和分析AI模型的使用情况、成本统计和性能指标
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..config.settings import get_settings
from .storage_client import get_storage_client


@dataclass
class UsageRecord:
    """使用记录"""
    model_id: str
    model_name: str
    provider: str
    account_id: str
    account_name: str
    user_id: Optional[str] = None
    message_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error_message: Optional[str] = None
    response_time_ms: float = 0.0
    routing_strategy: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class UsageTracker:
    """
    使用统计跟踪器
    记录AI模型调用的详细统计信息
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._logger = logging.getLogger(__name__)
        self.batch_records: List[UsageRecord] = []
        self.batch_size = 100
        self.flush_interval = 300  # 5分钟
        self.is_running = False
        self._flush_task = None
    
    async def initialize(self):
        """初始化跟踪器"""
        if self.is_running:
            return
        
        self.is_running = True
        # 启动定期刷新任务
        self._flush_task = asyncio.create_task(self._periodic_flush())
        self._logger.info("Usage tracker initialized")
    
    async def record_usage(self, usage_data: Dict[str, Any]):
        """
        记录使用情况
        
        Args:
            usage_data: 使用数据字典
        """
        try:
            record = UsageRecord(
                model_id=usage_data.get('model_id', ''),
                model_name=usage_data.get('model_name', ''),
                provider=usage_data.get('provider', ''),
                account_id=usage_data.get('account_id', ''),
                account_name=usage_data.get('account_name', ''),
                user_id=usage_data.get('user_id'),
                message_count=usage_data.get('message_count', 0),
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0),
                success=usage_data.get('success', True),
                error_message=usage_data.get('error_message'),
                response_time_ms=usage_data.get('response_time_ms', 0.0),
                routing_strategy=usage_data.get('routing_strategy')
            )
            
            self.batch_records.append(record)
            
            # 如果批次满了，立即刷新
            if len(self.batch_records) >= self.batch_size:
                await self._flush_records()
                
        except Exception as e:
            self._logger.error(f"Failed to record usage: {e}")
    
    async def _periodic_flush(self):
        """定期刷新记录"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                if self.batch_records:
                    await self._flush_records()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in periodic flush: {e}")
    
    async def _flush_records(self):
        """刷新记录到存储系统"""
        if not self.batch_records:
            return
        
        try:
            storage_client = await get_storage_client()
            
            # 转换为存储格式
            records_data = []
            for record in self.batch_records:
                record_dict = {
                    'model_id': record.model_id,
                    'model_name': record.model_name,
                    'provider': record.provider,
                    'account_id': record.account_id,
                    'account_name': record.account_name,
                    'user_id': record.user_id,
                    'message_count': record.message_count,
                    'prompt_tokens': record.prompt_tokens,
                    'completion_tokens': record.completion_tokens,
                    'total_tokens': record.total_tokens,
                    'success': record.success,
                    'error_message': record.error_message,
                    'response_time_ms': record.response_time_ms,
                    'routing_strategy': record.routing_strategy,
                    'timestamp': record.timestamp.isoformat()
                }
                records_data.append(record_dict)
            
            # 批量写入存储
            await storage_client.batch_create_usage_records(records_data)
            
            self._logger.info(f"Flushed {len(self.batch_records)} usage records")
            self.batch_records.clear()
            
        except Exception as e:
            self._logger.error(f"Failed to flush usage records: {e}")
    
    async def get_usage_summary(self, 
                              period: str = '24h',
                              provider: Optional[str] = None,
                              model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取使用统计摘要
        
        Args:
            period: 统计周期 ('1h', '24h', '7d', '30d')
            provider: 可选的提供商筛选
            model_id: 可选的模型筛选
            
        Returns:
            Dict[str, Any]: 统计摘要
        """
        try:
            storage_client = await get_storage_client()
            
            # 构建筛选条件
            filters = {'period': period}
            if provider:
                filters['provider'] = provider
            if model_id:
                filters['model_id'] = model_id
            
            # 获取统计数据
            stats_result = await storage_client.get_usage_statistics(**filters)
            
            # 构建摘要
            summary = {
                'period': period,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0.0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'avg_response_time': 0.0,
                'provider_breakdown': {},
                'model_breakdown': {},
                'error_breakdown': {},
                'top_users': []
            }
            
            data = stats_result.get('data', {})
            
            if data:
                summary.update({
                    'total_requests': data.get('total_requests', 0),
                    'successful_requests': data.get('total_success', 0),
                    'failed_requests': data.get('total_errors', 0),
                    'success_rate': data.get('success_rate', 0.0),
                    'total_tokens': data.get('total_tokens', 0),
                    'total_cost': data.get('total_cost', 0.0),
                    'avg_response_time': data.get('avg_response_time', 0.0),
                    'provider_breakdown': data.get('provider_stats', {}),
                    'model_breakdown': data.get('model_stats', {}),
                    'error_breakdown': data.get('error_stats', {}),
                    'top_users': data.get('top_users', [])
                })
            
            return summary
            
        except Exception as e:
            self._logger.error(f"Failed to get usage summary: {e}")
            return {'error': str(e)}
    
    async def get_cost_analysis(self, 
                              period: str = '30d',
                              group_by: str = 'provider') -> Dict[str, Any]:
        """
        获取成本分析
        
        Args:
            period: 分析周期
            group_by: 分组方式 ('provider', 'model', 'user')
            
        Returns:
            Dict[str, Any]: 成本分析结果
        """
        try:
            storage_client = await get_storage_client()
            
            # 获取成本统计数据
            cost_result = await storage_client.get_cost_analysis(
                period=period,
                group_by=group_by
            )
            
            analysis = {
                'period': period,
                'group_by': group_by,
                'total_cost': 0.0,
                'cost_breakdown': {},
                'cost_trend': [],
                'top_cost_items': []
            }
            
            data = cost_result.get('data', {})
            if data:
                analysis.update({
                    'total_cost': data.get('total_cost', 0.0),
                    'cost_breakdown': data.get('breakdown', {}),
                    'cost_trend': data.get('trend', []),
                    'top_cost_items': data.get('top_items', [])
                })
            
            return analysis
            
        except Exception as e:
            self._logger.error(f"Failed to get cost analysis: {e}")
            return {'error': str(e)}
    
    async def get_performance_metrics(self, 
                                    period: str = '24h',
                                    model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取性能指标
        
        Args:
            period: 分析周期
            model_id: 可选的模型筛选
            
        Returns:
            Dict[str, Any]: 性能指标
        """
        try:
            storage_client = await get_storage_client()
            
            filters = {'period': period}
            if model_id:
                filters['model_id'] = model_id
            
            perf_result = await storage_client.get_performance_metrics(**filters)
            
            metrics = {
                'period': period,
                'avg_response_time': 0.0,
                'p50_response_time': 0.0,
                'p95_response_time': 0.0,
                'p99_response_time': 0.0,
                'throughput_per_minute': 0.0,
                'error_rate': 0.0,
                'timeout_rate': 0.0,
                'model_performance': {}
            }
            
            data = perf_result.get('data', {})
            if data:
                metrics.update({
                    'avg_response_time': data.get('avg_response_time', 0.0),
                    'p50_response_time': data.get('p50_response_time', 0.0),
                    'p95_response_time': data.get('p95_response_time', 0.0),
                    'p99_response_time': data.get('p99_response_time', 0.0),
                    'throughput_per_minute': data.get('throughput_per_minute', 0.0),
                    'error_rate': data.get('error_rate', 0.0),
                    'timeout_rate': data.get('timeout_rate', 0.0),
                    'model_performance': data.get('model_breakdown', {})
                })
            
            return metrics
            
        except Exception as e:
            self._logger.error(f"Failed to get performance metrics: {e}")
            return {'error': str(e)}
    
    async def stop(self):
        """停止跟踪器"""
        self.is_running = False
        
        # 取消定期刷新任务
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # 刷新剩余记录
        if self.batch_records:
            await self._flush_records()
        
        self._logger.info("Usage tracker stopped")


# 全局使用跟踪器实例
_usage_tracker_instance = None

async def get_usage_tracker() -> UsageTracker:
    """
    获取使用跟踪器实例 (单例模式)
    
    Returns:
        UsageTracker: 使用跟踪器实例
    """
    global _usage_tracker_instance
    if _usage_tracker_instance is None:
        _usage_tracker_instance = UsageTracker()
        await _usage_tracker_instance.initialize()
    return _usage_tracker_instance