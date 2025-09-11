"""
外部服务客户端模块

负责与其他微服务和外部API的集成，包括：
- 存储服务
- 内容发布服务
- 账号管理服务
- 自动调度服务
- 第三方平台API
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

import httpx
from httpx import AsyncClient, RequestError, HTTPStatusError

from ..config.settings import settings

logger = logging.getLogger(__name__)


class ExternalServiceClient:
    """
    外部服务客户端
    
    提供与各种外部服务和API的统一接口，包括：
    - HTTP客户端管理
    - 请求重试和错误处理
    - 响应缓存和优化
    - 服务健康检查
    """
    
    def __init__(self):
        self.http_client: Optional[AsyncClient] = None
        self.service_urls = {
            'storage': settings.external_services.storage_service_url,
            'content_publishing': settings.external_services.content_publishing_url,
            'account_management': settings.external_services.account_management_url,
            'scheduling': settings.external_services.scheduling_service_url
        }
        self._service_health_cache = {}

    async def initialize(self):
        """初始化外部服务客户端"""
        try:
            # 创建HTTP客户端
            timeout = httpx.Timeout(
                connect=10.0,  # 连接超时
                read=30.0,     # 读取超时
                write=10.0,    # 写入超时
                pool=60.0      # 连接池超时
            )
            
            self.http_client = AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100
                ),
                headers={
                    'User-Agent': 'Analytics-Reporting-Service/1.0',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            )
            
            logger.info("ExternalServiceClient 初始化完成")
            
        except Exception as e:
            logger.error(f"ExternalServiceClient 初始化失败: {e}")
            raise

    async def close(self):
        """关闭HTTP客户端"""
        if self.http_client:
            await self.http_client.aclose()

    # ===== 存储服务集成 =====

    async def get_content_data(
        self, 
        user_id: str,
        content_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        从存储服务获取内容数据
        
        Args:
            user_id: 用户ID
            content_ids: 内容ID列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            内容数据列表
        """
        try:
            logger.info(f"从存储服务获取内容数据 - 用户: {user_id}")
            
            # 构建查询参数
            params = {'user_id': user_id}
            if content_ids:
                params['content_ids'] = ','.join(content_ids)
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()
            
            # 发起请求
            url = f"{self.service_urls['storage']}/api/v1/contents"
            response = await self._make_request('GET', url, params=params)
            
            if response and response.get('success'):
                content_data = response.get('data', [])
                logger.info(f"获取到 {len(content_data)} 条内容数据")
                return content_data
            
            return []
            
        except Exception as e:
            logger.error(f"获取内容数据失败: {e}")
            return []

    async def get_file_data(
        self, 
        user_id: str,
        file_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        从存储服务获取文件数据
        
        Args:
            user_id: 用户ID
            file_types: 文件类型列表
            
        Returns:
            文件数据列表
        """
        try:
            logger.info(f"从存储服务获取文件数据 - 用户: {user_id}")
            
            params = {'user_id': user_id}
            if file_types:
                params['file_types'] = ','.join(file_types)
            
            url = f"{self.service_urls['storage']}/api/v1/files"
            response = await self._make_request('GET', url, params=params)
            
            if response and response.get('success'):
                file_data = response.get('data', [])
                logger.info(f"获取到 {len(file_data)} 条文件数据")
                return file_data
            
            return []
            
        except Exception as e:
            logger.error(f"获取文件数据失败: {e}")
            return []

    # ===== 内容发布服务集成 =====

    async def get_publishing_data(
        self, 
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        从内容发布服务获取发布数据
        
        Args:
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            发布数据列表
        """
        try:
            logger.info(f"从发布服务获取发布数据 - 用户: {user_id}")
            
            params = {'user_id': user_id}
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()
            
            url = f"{self.service_urls['content_publishing']}/api/v1/publications"
            response = await self._make_request('GET', url, params=params)
            
            if response and response.get('success'):
                publishing_data = response.get('data', [])
                logger.info(f"获取到 {len(publishing_data)} 条发布数据")
                return publishing_data
            
            return []
            
        except Exception as e:
            logger.error(f"获取发布数据失败: {e}")
            return []

    # ===== 账号管理服务集成 =====

    async def get_account_data(
        self, 
        user_id: str,
        platforms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        从账号管理服务获取账号数据
        
        Args:
            user_id: 用户ID
            platforms: 平台列表
            
        Returns:
            账号数据列表
        """
        try:
            logger.info(f"从账号管理服务获取账号数据 - 用户: {user_id}")
            
            params = {'user_id': user_id}
            if platforms:
                params['platforms'] = ','.join(platforms)
            
            url = f"{self.service_urls['account_management']}/api/v1/accounts"
            response = await self._make_request('GET', url, params=params)
            
            if response and response.get('success'):
                account_data = response.get('data', [])
                logger.info(f"获取到 {len(account_data)} 条账号数据")
                return account_data
            
            return []
            
        except Exception as e:
            logger.error(f"获取账号数据失败: {e}")
            return []

    # ===== 自动调度服务集成 =====

    async def get_scheduling_data(
        self, 
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        从调度服务获取调度数据
        
        Args:
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            调度数据列表
        """
        try:
            logger.info(f"从调度服务获取调度数据 - 用户: {user_id}")
            
            params = {'user_id': user_id}
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()
            
            url = f"{self.service_urls['scheduling']}/api/v1/tasks"
            response = await self._make_request('GET', url, params=params)
            
            if response and response.get('success'):
                scheduling_data = response.get('data', [])
                logger.info(f"获取到 {len(scheduling_data)} 条调度数据")
                return scheduling_data
            
            return []
            
        except Exception as e:
            logger.error(f"获取调度数据失败: {e}")
            return []

    # ===== 第三方平台API集成 =====

    async def get_weibo_data(
        self, 
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取微博平台数据
        
        Args:
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            微博数据字典
        """
        try:
            logger.info(f"获取微博平台数据 - 用户: {user_id}")
            
            # 首先从账号管理服务获取微博账号信息
            account_data = await self.get_account_data(user_id, ['weibo'])
            
            if not account_data:
                logger.warning(f"用户 {user_id} 未配置微博账号")
                return None
            
            # 模拟微博API数据（实际应调用真实的微博API）
            weibo_data = {
                'platform': 'weibo',
                'content': [
                    {
                        'content_id': f'weibo_{i}',
                        'title': f'微博内容 {i}',
                        'publish_time': datetime.now() - timedelta(days=i),
                        'views': np.random.randint(1000, 50000),
                        'likes': np.random.randint(10, 1000),
                        'comments': np.random.randint(5, 500),
                        'shares': np.random.randint(1, 100),
                        'engagement_rate': np.random.uniform(0.01, 0.1)
                    }
                    for i in range(1, 11)
                ],
                'metrics': [
                    {
                        'metric_name': 'weibo_views',
                        'value': np.random.randint(10000, 100000),
                        'timestamp': datetime.now() - timedelta(hours=i),
                        'tags': {'platform': 'weibo', 'user_id': user_id}
                    }
                    for i in range(24)
                ],
                'user_behavior': [
                    {
                        'user_id': f'weibo_user_{i}',
                        'platform': 'weibo',
                        'action_type': np.random.choice(['view', 'like', 'comment', 'share']),
                        'content_id': f'weibo_{np.random.randint(1, 11)}',
                        'timestamp': datetime.now() - timedelta(minutes=i*10),
                        'session_id': f'session_{i}',
                        'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
                        'location': 'China'
                    }
                    for i in range(100)
                ]
            }
            
            return weibo_data
            
        except Exception as e:
            logger.error(f"获取微博数据失败: {e}")
            return None

    async def get_wechat_data(
        self, 
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """获取微信公众号数据"""
        
        try:
            import numpy as np
            
            logger.info(f"获取微信公众号数据 - 用户: {user_id}")
            
            # 模拟微信公众号API数据
            wechat_data = {
                'platform': 'wechat',
                'content': [
                    {
                        'content_id': f'wechat_{i}',
                        'title': f'微信公众号文章 {i}',
                        'publish_time': datetime.now() - timedelta(days=i*2),
                        'views': np.random.randint(500, 20000),
                        'likes': np.random.randint(20, 500),
                        'comments': np.random.randint(5, 100),
                        'shares': np.random.randint(10, 200),
                        'engagement_rate': np.random.uniform(0.02, 0.15)
                    }
                    for i in range(1, 6)
                ],
                'metrics': [
                    {
                        'metric_name': 'wechat_reads',
                        'value': np.random.randint(5000, 50000),
                        'timestamp': datetime.now() - timedelta(hours=i),
                        'tags': {'platform': 'wechat', 'user_id': user_id}
                    }
                    for i in range(24)
                ],
                'user_behavior': [
                    {
                        'user_id': f'wechat_user_{i}',
                        'platform': 'wechat',
                        'action_type': np.random.choice(['read', 'like', 'share', 'collect']),
                        'content_id': f'wechat_{np.random.randint(1, 6)}',
                        'timestamp': datetime.now() - timedelta(minutes=i*15),
                        'session_id': f'wx_session_{i}',
                        'device_type': 'mobile',
                        'location': 'China'
                    }
                    for i in range(80)
                ]
            }
            
            return wechat_data
            
        except Exception as e:
            logger.error(f"获取微信公众号数据失败: {e}")
            return None

    async def get_douyin_data(
        self, 
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """获取抖音数据"""
        
        try:
            import numpy as np
            
            logger.info(f"获取抖音数据 - 用户: {user_id}")
            
            # 模拟抖音API数据
            douyin_data = {
                'platform': 'douyin',
                'content': [
                    {
                        'content_id': f'douyin_{i}',
                        'title': f'抖音视频 {i}',
                        'publish_time': datetime.now() - timedelta(days=i),
                        'views': np.random.randint(10000, 500000),
                        'likes': np.random.randint(500, 10000),
                        'comments': np.random.randint(50, 1000),
                        'shares': np.random.randint(100, 2000),
                        'engagement_rate': np.random.uniform(0.05, 0.3)
                    }
                    for i in range(1, 8)
                ],
                'metrics': [
                    {
                        'metric_name': 'douyin_plays',
                        'value': np.random.randint(50000, 500000),
                        'timestamp': datetime.now() - timedelta(hours=i),
                        'tags': {'platform': 'douyin', 'user_id': user_id}
                    }
                    for i in range(24)
                ],
                'user_behavior': [
                    {
                        'user_id': f'douyin_user_{i}',
                        'platform': 'douyin',
                        'action_type': np.random.choice(['play', 'like', 'comment', 'share', 'follow']),
                        'content_id': f'douyin_{np.random.randint(1, 8)}',
                        'timestamp': datetime.now() - timedelta(minutes=i*5),
                        'session_id': f'dy_session_{i}',
                        'device_type': 'mobile',
                        'location': 'China'
                    }
                    for i in range(200)
                ]
            }
            
            return douyin_data
            
        except Exception as e:
            logger.error(f"获取抖音数据失败: {e}")
            return None

    async def get_toutiao_data(
        self, 
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """获取今日头条数据"""
        
        try:
            import numpy as np
            
            logger.info(f"获取今日头条数据 - 用户: {user_id}")
            
            # 模拟今日头条API数据
            toutiao_data = {
                'platform': 'toutiao',
                'content': [
                    {
                        'content_id': f'toutiao_{i}',
                        'title': f'今日头条文章 {i}',
                        'publish_time': datetime.now() - timedelta(days=i),
                        'views': np.random.randint(5000, 100000),
                        'likes': np.random.randint(50, 2000),
                        'comments': np.random.randint(10, 500),
                        'shares': np.random.randint(20, 800),
                        'engagement_rate': np.random.uniform(0.02, 0.12)
                    }
                    for i in range(1, 12)
                ],
                'metrics': [
                    {
                        'metric_name': 'toutiao_reads',
                        'value': np.random.randint(20000, 200000),
                        'timestamp': datetime.now() - timedelta(hours=i),
                        'tags': {'platform': 'toutiao', 'user_id': user_id}
                    }
                    for i in range(24)
                ],
                'user_behavior': [
                    {
                        'user_id': f'toutiao_user_{i}',
                        'platform': 'toutiao',
                        'action_type': np.random.choice(['read', 'like', 'comment', 'share', 'collect']),
                        'content_id': f'toutiao_{np.random.randint(1, 12)}',
                        'timestamp': datetime.now() - timedelta(minutes=i*8),
                        'session_id': f'tt_session_{i}',
                        'device_type': np.random.choice(['mobile', 'desktop']),
                        'location': 'China'
                    }
                    for i in range(150)
                ]
            }
            
            return toutiao_data
            
        except Exception as e:
            logger.error(f"获取今日头条数据失败: {e}")
            return None

    async def get_baijiahao_data(
        self, 
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """获取百家号数据"""
        
        try:
            import numpy as np
            
            logger.info(f"获取百家号数据 - 用户: {user_id}")
            
            # 模拟百家号API数据
            baijiahao_data = {
                'platform': 'baijiahao',
                'content': [
                    {
                        'content_id': f'bjh_{i}',
                        'title': f'百家号文章 {i}',
                        'publish_time': datetime.now() - timedelta(days=i*2),
                        'views': np.random.randint(3000, 80000),
                        'likes': np.random.randint(30, 1500),
                        'comments': np.random.randint(5, 300),
                        'shares': np.random.randint(10, 500),
                        'engagement_rate': np.random.uniform(0.015, 0.1)
                    }
                    for i in range(1, 9)
                ],
                'metrics': [
                    {
                        'metric_name': 'baijiahao_views',
                        'value': np.random.randint(15000, 150000),
                        'timestamp': datetime.now() - timedelta(hours=i),
                        'tags': {'platform': 'baijiahao', 'user_id': user_id}
                    }
                    for i in range(24)
                ],
                'user_behavior': [
                    {
                        'user_id': f'bjh_user_{i}',
                        'platform': 'baijiahao',
                        'action_type': np.random.choice(['view', 'like', 'comment', 'share']),
                        'content_id': f'bjh_{np.random.randint(1, 9)}',
                        'timestamp': datetime.now() - timedelta(minutes=i*12),
                        'session_id': f'bjh_session_{i}',
                        'device_type': np.random.choice(['mobile', 'desktop']),
                        'location': 'China'
                    }
                    for i in range(120)
                ]
            }
            
            return baijiahao_data
            
        except Exception as e:
            logger.error(f"获取百家号数据失败: {e}")
            return None

    # ===== 服务健康检查 =====

    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        检查外部服务健康状态
        
        Args:
            service_name: 服务名称
            
        Returns:
            健康状态字典
        """
        try:
            if service_name not in self.service_urls:
                return {'healthy': False, 'error': 'Unknown service'}
            
            # 检查缓存
            cache_key = f"health_{service_name}"
            if cache_key in self._service_health_cache:
                cache_time, health_data = self._service_health_cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=1):
                    return health_data
            
            # 发起健康检查请求
            url = f"{self.service_urls[service_name]}/health"
            response = await self._make_request('GET', url, timeout=5)
            
            health_data = {
                'healthy': response is not None,
                'response_time': 0,  # 实际应测量响应时间
                'timestamp': datetime.now().isoformat()
            }
            
            if response:
                health_data.update(response)
            
            # 缓存结果
            self._service_health_cache[cache_key] = (datetime.now(), health_data)
            
            return health_data
            
        except Exception as e:
            logger.error(f"服务 {service_name} 健康检查失败: {e}")
            return {'healthy': False, 'error': str(e)}

    async def check_all_services_health(self) -> Dict[str, Any]:
        """检查所有外部服务的健康状态"""
        
        health_results = {}
        
        # 并行检查所有服务
        tasks = [
            self.check_service_health(service_name)
            for service_name in self.service_urls.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, service_name in enumerate(self.service_urls.keys()):
            if isinstance(results[i], Exception):
                health_results[service_name] = {
                    'healthy': False,
                    'error': str(results[i])
                }
            else:
                health_results[service_name] = results[i]
        
        return health_results

    # ===== 私有辅助方法 =====

    async def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        发起HTTP请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            params: 查询参数
            json_data: JSON数据
            timeout: 超时时间
            retries: 重试次数
            
        Returns:
            响应数据或None
        """
        
        if not self.http_client:
            await self.initialize()
        
        for attempt in range(retries + 1):
            try:
                # 构建请求参数
                request_kwargs = {}
                if params:
                    request_kwargs['params'] = params
                if json_data:
                    request_kwargs['json'] = json_data
                if timeout:
                    request_kwargs['timeout'] = timeout
                
                # 发起请求
                response = await self.http_client.request(method, url, **request_kwargs)
                
                # 检查响应状态
                response.raise_for_status()
                
                # 解析JSON响应
                return response.json()
                
            except HTTPStatusError as e:
                logger.warning(f"HTTP错误 {e.response.status_code}: {url}")
                if e.response.status_code < 500 or attempt == retries:
                    # 客户端错误或最后一次重试，不再重试
                    break
                    
            except RequestError as e:
                logger.warning(f"请求错误: {e}")
                if attempt == retries:
                    break
                    
            except Exception as e:
                logger.error(f"请求异常: {e}")
                if attempt == retries:
                    break
            
            # 等待后重试
            if attempt < retries:
                wait_time = 2 ** attempt  # 指数退避
                logger.info(f"第 {attempt + 1} 次重试，等待 {wait_time} 秒...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"请求失败，已尝试 {retries + 1} 次: {url}")
        return None

    async def _batch_request(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        批量发起HTTP请求
        
        Args:
            requests: 请求列表，每个元素包含method, url等参数
            
        Returns:
            响应结果列表
        """
        
        tasks = []
        for req in requests:
            task = self._make_request(
                method=req['method'],
                url=req['url'],
                params=req.get('params'),
                json_data=req.get('json_data'),
                timeout=req.get('timeout'),
                retries=req.get('retries', 3)
            )
            tasks.append(task)
        
        # 并行执行所有请求
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量请求中的异常: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results