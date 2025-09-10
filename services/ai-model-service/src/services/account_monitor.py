"""
账号健康监控服务

监控API账号的可用性、性能和配额使用情况
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..models.ai_models import APIAccount, AccountStatus
from ..config.settings import get_settings
from .storage_client import get_storage_client


class AccountHealthMonitor:
    """
    账号健康监控服务
    监控API账号的可用性和性能
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._logger = logging.getLogger(__name__)
        self.monitoring_tasks = {}
        self.is_running = False
    
    async def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            self._logger.warning("Monitoring is already running")
            return
        
        self.is_running = True
        self._logger.info("Starting account health monitoring")
        
        # 启动定期监控任务
        asyncio.create_task(self._periodic_health_check())
        asyncio.create_task(self._periodic_quota_check())
        asyncio.create_task(self._periodic_cleanup())
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        
        # 取消所有监控任务
        for task in self.monitoring_tasks.values():
            if not task.done():
                task.cancel()
        
        self.monitoring_tasks.clear()
        self._logger.info("Account health monitoring stopped")
    
    async def _periodic_health_check(self):
        """定期健康检查"""
        while self.is_running:
            try:
                await self.monitor_all_accounts_health()
                await asyncio.sleep(self.settings.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Periodic health check failed: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟再重试
    
    async def _periodic_quota_check(self):
        """定期配额检查"""
        while self.is_running:
            try:
                await self.check_quota_alerts()
                await asyncio.sleep(3600)  # 每小时检查一次配额
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Periodic quota check failed: {e}")
                await asyncio.sleep(300)  # 出错后等待5分钟再重试
    
    async def _periodic_cleanup(self):
        """定期清理过期数据"""
        while self.is_running:
            try:
                await self.cleanup_expired_data()
                await asyncio.sleep(86400)  # 每天清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Periodic cleanup failed: {e}")
                await asyncio.sleep(3600)  # 出错后等待1小时再重试
    
    async def monitor_all_accounts_health(self):
        """监控所有账号健康状态"""
        try:
            storage_client = await get_storage_client()
            accounts_data = await storage_client.get_api_accounts()
            
            tasks = []
            for account_data in accounts_data:
                account = APIAccount(**account_data)
                task = asyncio.create_task(self._check_single_account_health(account))
                tasks.append(task)
            
            # 并发检查所有账号
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count
            
            self._logger.info(f"Health check completed: {success_count} success, {error_count} errors")
            
        except Exception as e:
            self._logger.error(f"Failed to monitor accounts health: {e}")
    
    async def _check_single_account_health(self, account: APIAccount) -> float:
        """检查单个账号的健康状态"""
        try:
            # 获取最近的使用统计
            stats = await self._get_recent_stats(account.id)
            
            # 计算健康评分（0-1之间）
            health_score = self._calculate_health_score(account, stats)
            
            # 更新账号健康评分
            await self._update_account_health(account.id, health_score)
            
            # 检查是否需要状态更新
            new_status = self._determine_account_status(account, health_score, stats)
            if new_status != account.status:
                await self._update_account_status(account.id, new_status)
            
            return health_score
            
        except Exception as e:
            self._logger.error(f"Health check failed for account {account.account_name}: {e}")
            
            # 增加错误计数
            await self._increment_error_count(account.id)
            return 0.0
    
    def _calculate_health_score(self, account: APIAccount, stats: Dict[str, Any]) -> float:
        """计算健康评分"""
        # 基础权重
        weights = {
            'success_rate': 0.4,      # 成功率权重40%
            'response_time': 0.3,     # 响应时间权重30%
            'error_rate': 0.2,        # 错误率权重20%
            'quota_usage': 0.1        # 配额使用率权重10%
        }
        
        # 成功率评分
        success_rate = stats.get('success_rate', 1.0)
        success_score = success_rate * weights['success_rate']
        
        # 响应时间评分（5秒内为满分）
        avg_response_time = stats.get('avg_response_time', 0)
        response_score = max(0, 1 - avg_response_time / 5000) * weights['response_time']
        
        # 错误率评分
        error_rate = stats.get('error_rate', 0)
        error_score = max(0, 1 - error_rate) * weights['error_rate']
        
        # 配额使用率评分
        quota_usage = account.quota_used / max(account.quota_limit, 1)
        quota_score = max(0, 1 - quota_usage) * weights['quota_usage']
        
        # 综合评分
        health_score = success_score + response_score + error_score + quota_score
        
        # 额外惩罚因子
        penalties = 0
        
        # 连续错误惩罚
        if account.error_count > 10:
            penalties += 0.1
        
        # 长时间未使用惩罚
        if account.last_used_at:
            hours_unused = (datetime.now() - account.last_used_at).total_seconds() / 3600
            if hours_unused > 168:  # 7天未使用
                penalties += 0.1
        
        # 应用惩罚
        health_score = max(0, health_score - penalties)
        
        return min(1.0, health_score)
    
    def _determine_account_status(self, account: APIAccount, health_score: float, stats: Dict[str, Any]) -> AccountStatus:
        """确定账号状态"""
        # 配额检查
        quota_usage = account.quota_used / max(account.quota_limit, 1)
        if quota_usage >= 1.0:
            return AccountStatus.QUOTA_EXCEEDED
        
        # 健康评分检查
        if health_score < 0.3:
            return AccountStatus.ERROR
        elif health_score < 0.5:
            return AccountStatus.RATE_LIMITED
        
        # 错误率检查
        error_rate = stats.get('error_rate', 0)
        if error_rate > 0.5:  # 错误率超过50%
            return AccountStatus.ERROR
        
        # 响应时间检查
        avg_response_time = stats.get('avg_response_time', 0)
        if avg_response_time > 10000:  # 响应时间超过10秒
            return AccountStatus.RATE_LIMITED
        
        return AccountStatus.ACTIVE
    
    async def _get_recent_stats(self, account_id: str) -> Dict[str, Any]:
        """获取账号最近的统计数据"""
        try:
            storage_client = await get_storage_client()
            stats_result = await storage_client.get_usage_statistics(
                account_id=account_id,
                period='24h'
            )
            
            data = stats_result.get('data', {})
            
            if not data or data.get('total_requests', 0) == 0:
                return {
                    'success_rate': 1.0,
                    'error_rate': 0.0,
                    'avg_response_time': 0,
                    'total_requests': 0
                }
            
            return {
                'success_rate': data.get('total_success', 0) / data.get('total_requests', 1),
                'error_rate': data.get('total_errors', 0) / data.get('total_requests', 1),
                'avg_response_time': data.get('avg_response_time', 0),
                'total_requests': data.get('total_requests', 0),
                'total_cost': data.get('total_cost', 0.0)
            }
            
        except Exception as e:
            self._logger.warning(f"Failed to get stats for account {account_id}: {e}")
            return {
                'success_rate': 0.5,
                'error_rate': 0.5,
                'avg_response_time': 5000,
                'total_requests': 0
            }
    
    async def _update_account_health(self, account_id: str, health_score: float):
        """更新账号健康评分"""
        try:
            storage_client = await get_storage_client()
            await storage_client.update_account_health(account_id, health_score)
            
        except Exception as e:
            self._logger.error(f"Failed to update health for account {account_id}: {e}")
    
    async def _update_account_status(self, account_id: str, status: AccountStatus):
        """更新账号状态"""
        try:
            storage_client = await get_storage_client()
            await storage_client.update_api_account(
                account_id, 
                {'status': status.value, 'updated_at': datetime.now().isoformat()}
            )
            
            self._logger.info(f"Account {account_id} status updated to {status.value}")
            
        except Exception as e:
            self._logger.error(f"Failed to update status for account {account_id}: {e}")
    
    async def _increment_error_count(self, account_id: str):
        """增加错误计数"""
        try:
            storage_client = await get_storage_client()
            account_data = await storage_client.get_api_account(account_id)
            
            new_error_count = account_data.get('error_count', 0) + 1
            await storage_client.update_api_account(
                account_id,
                {
                    'error_count': new_error_count,
                    'last_error': f"Health check failed at {datetime.now().isoformat()}",
                    'updated_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self._logger.error(f"Failed to increment error count for account {account_id}: {e}")
    
    async def check_quota_alerts(self):
        """检查配额告警"""
        try:
            storage_client = await get_storage_client()
            accounts_data = await storage_client.get_api_accounts(status='active')
            
            alert_threshold = self.settings.quota_alert_threshold
            alerts = []
            
            for account_data in accounts_data:
                account = APIAccount(**account_data)
                
                if account.quota_limit > 0:
                    usage_rate = account.quota_used / account.quota_limit
                    
                    if usage_rate >= alert_threshold:
                        alert = {
                            'account_id': account.id,
                            'account_name': account.account_name,
                            'provider': account.provider.value,
                            'usage_rate': usage_rate,
                            'quota_used': account.quota_used,
                            'quota_limit': account.quota_limit,
                            'alert_type': 'quota_warning' if usage_rate < 1.0 else 'quota_exceeded'
                        }
                        alerts.append(alert)
            
            if alerts:
                self._logger.warning(f"Quota alerts detected: {len(alerts)} accounts")
                await self._send_quota_alerts(alerts)
                
        except Exception as e:
            self._logger.error(f"Failed to check quota alerts: {e}")
    
    async def _send_quota_alerts(self, alerts: List[Dict[str, Any]]):
        """发送配额告警"""
        # 这里可以集成告警系统，如邮件、钉钉、企业微信等
        for alert in alerts:
            self._logger.warning(
                f"Quota alert: {alert['account_name']} ({alert['provider']}) "
                f"usage: {alert['usage_rate']:.1%} ({alert['quota_used']}/{alert['quota_limit']})"
            )
    
    async def cleanup_expired_data(self):
        """清理过期数据"""
        try:
            # 这里可以清理过期的统计数据、日志等
            self._logger.info("Cleanup expired data completed")
            
        except Exception as e:
            self._logger.error(f"Failed to cleanup expired data: {e}")
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状况摘要"""
        try:
            storage_client = await get_storage_client()
            accounts_data = await storage_client.get_api_accounts()
            
            summary = {
                'total_accounts': len(accounts_data),
                'status_distribution': {},
                'health_distribution': {
                    'excellent': 0,  # > 0.8
                    'good': 0,       # 0.6 - 0.8
                    'warning': 0,    # 0.4 - 0.6
                    'critical': 0    # < 0.4
                },
                'quota_alerts': 0,
                'provider_distribution': {},
                'average_health_score': 0.0
            }
            
            total_health = 0
            alert_threshold = self.settings.quota_alert_threshold
            
            for account_data in accounts_data:
                account = APIAccount(**account_data)
                
                # 状态分布
                status = account.status.value
                summary['status_distribution'][status] = summary['status_distribution'].get(status, 0) + 1
                
                # 健康评分分布
                health_score = account.health_score
                total_health += health_score
                
                if health_score > 0.8:
                    summary['health_distribution']['excellent'] += 1
                elif health_score > 0.6:
                    summary['health_distribution']['good'] += 1
                elif health_score > 0.4:
                    summary['health_distribution']['warning'] += 1
                else:
                    summary['health_distribution']['critical'] += 1
                
                # 提供商分布
                provider = account.provider.value
                summary['provider_distribution'][provider] = summary['provider_distribution'].get(provider, 0) + 1
                
                # 配额告警
                if account.quota_limit > 0:
                    usage_rate = account.quota_used / account.quota_limit
                    if usage_rate >= alert_threshold:
                        summary['quota_alerts'] += 1
            
            # 平均健康评分
            summary['average_health_score'] = total_health / len(accounts_data) if accounts_data else 0.0
            
            return summary
            
        except Exception as e:
            self._logger.error(f"Failed to get health summary: {e}")
            return {'error': str(e)}


# 全局监控实例
_account_monitor_instance = None

async def get_account_monitor() -> AccountHealthMonitor:
    """获取账号监控实例"""
    global _account_monitor_instance
    if _account_monitor_instance is None:
        _account_monitor_instance = AccountHealthMonitor()
    return _account_monitor_instance