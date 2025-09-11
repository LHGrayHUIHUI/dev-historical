"""
多平台集成服务
与多平台账号管理服务和内容发布服务集成
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from uuid import UUID
import logging
import asyncio
import aiohttp
from dataclasses import dataclass

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PublishRequest:
    """发布请求数据类"""
    platform_name: str
    account_id: str
    content_title: str
    content_body: str
    content_metadata: Dict[str, Any]
    scheduled_time: datetime
    priority: int = 5


@dataclass
class PublishResult:
    """发布结果数据类"""
    platform_name: str
    success: bool
    platform_post_id: Optional[str] = None
    error_message: Optional[str] = None
    published_time: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None


class PlatformIntegrationService:
    """平台集成服务类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 外部服务URL
        self.account_service_url = self.settings.account_management_service_url
        self.publishing_service_url = self.settings.content_publishing_service_url
        self.storage_service_url = self.settings.storage_service_url
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'Content-Type': 'application/json'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def get_user_accounts(
        self, 
        user_id: int, 
        platforms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        获取用户的平台账号列表
        
        Args:
            user_id: 用户ID
            platforms: 平台筛选列表
            
        Returns:
            List[Dict]: 账号列表
        """
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._fetch_user_accounts(session, user_id, platforms)
            else:
                return await self._fetch_user_accounts(self.session, user_id, platforms)
                
        except Exception as e:
            logger.error(f"获取用户账号失败: {e}")
            return []
    
    async def _fetch_user_accounts(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        platforms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """内部方法：获取用户账号"""
        
        url = f"{self.account_service_url}/api/v1/accounts/"
        params = {"user_id": user_id}
        
        if platforms:
            params["platforms"] = ",".join(platforms)
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('accounts', [])
            else:
                logger.error(f"账号服务响应错误: {response.status}")
                return []
    
    async def validate_accounts_status(
        self,
        user_id: int,
        account_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        验证账号状态
        
        Args:
            user_id: 用户ID
            account_ids: 账号ID列表
            
        Returns:
            Dict: 账号状态信息
        """
        account_statuses = {}
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._validate_accounts_status(session, user_id, account_ids)
            else:
                return await self._validate_accounts_status(self.session, user_id, account_ids)
                
        except Exception as e:
            logger.error(f"验证账号状态失败: {e}")
            # 返回默认状态
            for account_id in account_ids:
                account_statuses[account_id] = {
                    'status': 'unknown',
                    'valid': False,
                    'error': str(e)
                }
            
        return account_statuses
    
    async def _validate_accounts_status(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        account_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """内部方法：验证账号状态"""
        
        account_statuses = {}
        
        # 并发检查所有账号状态
        tasks = [
            self._check_single_account_status(session, user_id, account_id)
            for account_id in account_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for account_id, result in zip(account_ids, results):
            if isinstance(result, Exception):
                account_statuses[account_id] = {
                    'status': 'error',
                    'valid': False,
                    'error': str(result)
                }
            else:
                account_statuses[account_id] = result
        
        return account_statuses
    
    async def _check_single_account_status(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        account_id: str
    ) -> Dict[str, Any]:
        """检查单个账号状态"""
        
        url = f"{self.account_service_url}/api/v1/accounts/{account_id}"
        params = {"user_id": user_id}
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                account_data = await response.json()
                account = account_data.get('account', {})
                
                return {
                    'status': account.get('status', 'unknown'),
                    'valid': account.get('status') == 'active',
                    'platform': account.get('platform_name'),
                    'display_name': account.get('display_name'),
                    'last_sync': account.get('last_sync_time')
                }
            else:
                return {
                    'status': 'not_found',
                    'valid': False,
                    'error': f'HTTP {response.status}'
                }
    
    async def publish_content(
        self,
        publish_requests: List[PublishRequest]
    ) -> List[PublishResult]:
        """
        发布内容到多个平台
        
        Args:
            publish_requests: 发布请求列表
            
        Returns:
            List[PublishResult]: 发布结果列表
        """
        results = []
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._publish_content(session, publish_requests)
            else:
                return await self._publish_content(self.session, publish_requests)
                
        except Exception as e:
            logger.error(f"发布内容失败: {e}")
            # 返回失败结果
            for request in publish_requests:
                results.append(PublishResult(
                    platform_name=request.platform_name,
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    async def _publish_content(
        self,
        session: aiohttp.ClientSession,
        publish_requests: List[PublishRequest]
    ) -> List[PublishResult]:
        """内部方法：发布内容"""
        
        # 并发发布到所有平台
        tasks = [
            self._publish_to_single_platform(session, request)
            for request in publish_requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for request, result in zip(publish_requests, results):
            if isinstance(result, Exception):
                final_results.append(PublishResult(
                    platform_name=request.platform_name,
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _publish_to_single_platform(
        self,
        session: aiohttp.ClientSession,
        request: PublishRequest
    ) -> PublishResult:
        """发布内容到单个平台"""
        
        try:
            # 调用内容发布服务
            url = f"{self.publishing_service_url}/api/v1/publish"
            
            payload = {
                "platform_name": request.platform_name,
                "account_id": request.account_id,
                "content": {
                    "title": request.content_title,
                    "body": request.content_body,
                    "metadata": request.content_metadata
                },
                "scheduled_time": request.scheduled_time.isoformat(),
                "priority": request.priority
            }
            
            async with session.post(url, json=payload) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    publish_data = data.get('result', {})
                    
                    return PublishResult(
                        platform_name=request.platform_name,
                        success=True,
                        platform_post_id=publish_data.get('post_id'),
                        published_time=datetime.fromisoformat(
                            publish_data.get('published_time', request.scheduled_time.isoformat())
                        ),
                        metrics=publish_data.get('initial_metrics', {})
                    )
                else:
                    error_data = await response.json()
                    return PublishResult(
                        platform_name=request.platform_name,
                        success=False,
                        error_message=error_data.get('error', f'HTTP {response.status}')
                    )
                    
        except Exception as e:
            logger.error(f"发布到平台 {request.platform_name} 失败: {e}")
            return PublishResult(
                platform_name=request.platform_name,
                success=False,
                error_message=str(e)
            )
    
    async def get_content_performance(
        self,
        platform_post_ids: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取内容性能数据
        
        Args:
            platform_post_ids: 平台和内容ID列表 [{"platform": "weibo", "post_id": "123"}]
            
        Returns:
            Dict: 性能数据
        """
        performance_data = {}
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._get_content_performance(session, platform_post_ids)
            else:
                return await self._get_content_performance(self.session, platform_post_ids)
                
        except Exception as e:
            logger.error(f"获取内容性能数据失败: {e}")
            
        return performance_data
    
    async def _get_content_performance(
        self,
        session: aiohttp.ClientSession,
        platform_post_ids: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """内部方法：获取内容性能数据"""
        
        performance_data = {}
        
        # 并发获取所有内容的性能数据
        tasks = [
            self._get_single_content_performance(session, item)
            for item in platform_post_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for item, result in zip(platform_post_ids, results):
            key = f"{item['platform']}_{item['post_id']}"
            
            if isinstance(result, Exception):
                performance_data[key] = {
                    'platform': item['platform'],
                    'post_id': item['post_id'],
                    'error': str(result),
                    'success': False
                }
            else:
                performance_data[key] = result
        
        return performance_data
    
    async def _get_single_content_performance(
        self,
        session: aiohttp.ClientSession,
        item: Dict[str, str]
    ) -> Dict[str, Any]:
        """获取单个内容的性能数据"""
        
        try:
            # 调用存储服务获取性能数据
            url = f"{self.storage_service_url}/api/v1/analytics/content-performance"
            params = {
                "platform": item['platform'],
                "post_id": item['post_id']
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    performance = data.get('performance', {})
                    
                    return {
                        'platform': item['platform'],
                        'post_id': item['post_id'],
                        'success': True,
                        'metrics': {
                            'views': performance.get('views', 0),
                            'likes': performance.get('likes', 0),
                            'shares': performance.get('shares', 0),
                            'comments': performance.get('comments', 0),
                            'engagement_rate': performance.get('engagement_rate', 0.0),
                            'reach': performance.get('reach', 0)
                        },
                        'last_updated': performance.get('last_updated')
                    }
                else:
                    return {
                        'platform': item['platform'],
                        'post_id': item['post_id'],
                        'success': False,
                        'error': f'HTTP {response.status}'
                    }
                    
        except Exception as e:
            logger.error(f"获取内容性能数据失败 {item}: {e}")
            return {
                'platform': item['platform'],
                'post_id': item['post_id'],
                'success': False,
                'error': str(e)
            }
    
    async def get_platform_analytics(
        self,
        user_id: int,
        platforms: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        获取平台分析数据
        
        Args:
            user_id: 用户ID
            platforms: 平台列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 分析数据
        """
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._get_platform_analytics(
                        session, user_id, platforms, start_date, end_date
                    )
            else:
                return await self._get_platform_analytics(
                    self.session, user_id, platforms, start_date, end_date
                )
                
        except Exception as e:
            logger.error(f"获取平台分析数据失败: {e}")
            return {}
    
    async def _get_platform_analytics(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        platforms: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """内部方法：获取平台分析数据"""
        
        url = f"{self.storage_service_url}/api/v1/analytics/platform-metrics"
        params = {
            "user_id": user_id,
            "platforms": ",".join(platforms),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('analytics', {})
            else:
                logger.error(f"平台分析服务响应错误: {response.status}")
                return {}
    
    async def sync_account_data(
        self,
        user_id: int,
        account_ids: List[str],
        sync_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        同步账号数据
        
        Args:
            user_id: 用户ID
            account_ids: 账号ID列表
            sync_types: 同步类型 ["profile", "stats", "posts", "followers"]
            
        Returns:
            Dict: 同步结果
        """
        if sync_types is None:
            sync_types = ["profile", "stats"]
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._sync_account_data(session, user_id, account_ids, sync_types)
            else:
                return await self._sync_account_data(self.session, user_id, account_ids, sync_types)
                
        except Exception as e:
            logger.error(f"同步账号数据失败: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _sync_account_data(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        account_ids: List[str],
        sync_types: List[str]
    ) -> Dict[str, Any]:
        """内部方法：同步账号数据"""
        
        url = f"{self.account_service_url}/api/v1/sync/batch"
        payload = {
            "user_id": user_id,
            "account_ids": account_ids,
            "sync_types": sync_types
        }
        
        async with session.post(url, json=payload) as response:
            if response.status in [200, 202]:
                data = await response.json()
                return data.get('result', {})
            else:
                error_data = await response.json()
                return {
                    'success': False,
                    'error': error_data.get('error', f'HTTP {response.status}')
                }
    
    async def get_platform_rate_limits(self, platforms: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        获取平台速率限制信息
        
        Args:
            platforms: 平台列表
            
        Returns:
            Dict: 速率限制信息
        """
        rate_limits = {}
        
        # 从配置中获取平台限制信息
        for platform in platforms:
            platform_config = self.settings.platforms.platform_configs.get(platform, {})
            rate_limits[platform] = {
                'hourly_limit': platform_config.get('rate_limit', 1000),
                'batch_size': platform_config.get('batch_size', 10),
                'min_interval_seconds': platform_config.get('min_interval_seconds', 1),
                'supports_scheduling': platform_config.get('supports_scheduling', True)
            }
        
        return rate_limits
    
    async def check_platform_availability(self, platforms: List[str]) -> Dict[str, bool]:
        """
        检查平台可用性
        
        Args:
            platforms: 平台列表
            
        Returns:
            Dict: 平台可用性状态
        """
        availability = {}
        
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    return await self._check_platform_availability(session, platforms)
            else:
                return await self._check_platform_availability(self.session, platforms)
                
        except Exception as e:
            logger.error(f"检查平台可用性失败: {e}")
            # 默认所有平台不可用
            for platform in platforms:
                availability[platform] = False
                
        return availability
    
    async def _check_platform_availability(
        self,
        session: aiohttp.ClientSession,
        platforms: List[str]
    ) -> Dict[str, bool]:
        """内部方法：检查平台可用性"""
        
        availability = {}
        
        # 检查账号管理服务的健康状态
        try:
            url = f"{self.account_service_url}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                account_service_healthy = response.status == 200
        except:
            account_service_healthy = False
        
        # 检查发布服务的健康状态
        try:
            url = f"{self.publishing_service_url}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                publishing_service_healthy = response.status == 200
        except:
            publishing_service_healthy = False
        
        # 基于服务健康状态确定平台可用性
        for platform in platforms:
            # 简化判断：如果相关服务健康，则平台可用
            availability[platform] = account_service_healthy and publishing_service_healthy
        
        return availability