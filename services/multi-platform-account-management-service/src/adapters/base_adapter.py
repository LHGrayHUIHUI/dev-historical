"""
平台适配器基类

定义所有平台适配器的统一接口
提供通用的API调用、错误处理和数据格式化功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import logging
from datetime import datetime
import aiohttp
from ..utils.exceptions import (
    PlatformNotSupportedError,
    RateLimitExceededError,
    ServiceUnavailableError,
    ValidationError
)

logger = logging.getLogger(__name__)


class BasePlatformAdapter(ABC):
    """
    平台适配器基类
    
    定义所有平台适配器必须实现的接口方法
    提供通用的HTTP请求、错误处理和数据转换功能
    """
    
    def __init__(self, platform_config: Dict[str, Any]):
        """
        初始化平台适配器
        
        Args:
            platform_config: 平台配置信息
        """
        self.platform_name = platform_config.get('name')
        self.api_base_url = platform_config.get('api_base_url')
        self.oauth_config = platform_config.get('oauth_config', {})
        self.rate_limits = platform_config.get('rate_limits', {})
        self.features = platform_config.get('features', {})
        self.is_active = platform_config.get('is_active', True)
        
        # HTTP客户端配置
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # 速率限制状态
        self._last_request_time = None
        self._request_count = 0
        self._rate_limit_window_start = None
    
    @property
    @abstractmethod
    def platform_type(self) -> str:
        """平台类型"""
        pass
    
    @property
    @abstractmethod  
    def supported_sync_types(self) -> List[str]:
        """支持的同步类型"""
        pass
    
    @abstractmethod
    async def get_user_profile(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取用户资料信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID(可选)
            
        Returns:
            Dict: 标准化的用户资料信息
        """
        pass
    
    @abstractmethod
    async def get_user_stats(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取用户统计信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID(可选)
            
        Returns:
            Dict: 标准化的用户统计信息
        """
        pass
    
    @abstractmethod
    async def get_user_posts(self, access_token: str, user_id: str = None, 
                           limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取用户发布内容
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID(可选)
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的用户发布内容
        """
        pass
    
    async def get_followers(self, access_token: str, user_id: str = None,
                          limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取粉丝列表(可选实现)
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID(可选)
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的粉丝列表
        """
        # 默认实现返回不支持
        return {
            'followers': [],
            'total_count': 0,
            'has_more': False,
            'message': f'{self.platform_name} 平台不支持粉丝列表获取'
        }
    
    async def validate_token(self, access_token: str) -> bool:
        """
        验证访问令牌有效性
        
        Args:
            access_token: 访问令牌
            
        Returns:
            bool: 令牌是否有效
        """
        try:
            # 通过获取用户信息来验证令牌
            await self.get_user_profile(access_token)
            return True
        except Exception as e:
            logger.warning(f"{self.platform_name} 令牌验证失败: {e}")
            return False
    
    async def _make_request(self, method: str, url: str, headers: Dict[str, str] = None,
                          params: Dict[str, Any] = None, data: Dict[str, Any] = None,
                          json_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        发起HTTP请求的通用方法
        
        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            params: URL参数
            data: 表单数据
            json_data: JSON数据
            
        Returns:
            Dict: 响应数据
            
        Raises:
            RateLimitExceededError: 速率限制超出
            ServiceUnavailableError: 服务不可用
        """
        # 检查速率限制
        await self._check_rate_limit()
        
        # 设置默认请求头
        if headers is None:
            headers = {}
        
        headers.setdefault('User-Agent', 'Historical-Text-Account-Manager/1.0')
        headers.setdefault('Accept', 'application/json')
        
        # 重试逻辑
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        data=data,
                        json=json_data
                    ) as response:
                        
                        # 更新请求统计
                        self._update_request_stats()
                        
                        # 处理速率限制
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 60))
                            raise RateLimitExceededError(
                                f"{self.platform_name} API速率限制超出",
                                platform_name=self.platform_name,
                                retry_after=retry_after
                            )
                        
                        # 处理服务不可用
                        if response.status in [502, 503, 504]:
                            raise ServiceUnavailableError(
                                f"{self.platform_name} API服务不可用",
                                platform_name=self.platform_name
                            )
                        
                        # 获取响应内容
                        content_type = response.headers.get('Content-Type', '')
                        if 'application/json' in content_type:
                            response_data = await response.json()
                        else:
                            response_text = await response.text()
                            response_data = {'text': response_text}
                        
                        # 检查API错误
                        if response.status >= 400:
                            error_msg = self._extract_error_message(response_data)
                            logger.error(f"{self.platform_name} API错误 ({response.status}): {error_msg}")
                            
                            if response.status == 401:
                                from ..utils.exceptions import InvalidTokenError
                                raise InvalidTokenError(
                                    f"访问令牌无效: {error_msg}",
                                    platform_name=self.platform_name
                                )
                            elif response.status == 403:
                                from ..utils.exceptions import PermissionDeniedError
                                raise PermissionDeniedError(
                                    f"权限不足: {error_msg}",
                                    permission_type="api_access"
                                )
                            else:
                                raise ServiceUnavailableError(
                                    f"API请求失败: {error_msg}",
                                    platform_name=self.platform_name
                                )
                        
                        logger.debug(f"{self.platform_name} API请求成功: {method} {url}")
                        return response_data
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.warning(f"{self.platform_name} API请求失败，{delay}秒后重试: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{self.platform_name} API请求最终失败: {e}")
        
        # 重试耗尽后抛出最后的异常
        raise ServiceUnavailableError(
            f"{self.platform_name} API请求失败: {str(last_exception)}",
            platform_name=self.platform_name
        )
    
    async def _check_rate_limit(self):
        """检查速率限制"""
        if not self.rate_limits:
            return
        
        current_time = datetime.utcnow()
        window_seconds = self.rate_limits.get('window_seconds', 3600)
        max_requests = self.rate_limits.get('max_requests', 1000)
        
        # 初始化或重置窗口
        if (self._rate_limit_window_start is None or 
            (current_time - self._rate_limit_window_start).total_seconds() >= window_seconds):
            self._rate_limit_window_start = current_time
            self._request_count = 0
        
        # 检查是否超出限制
        if self._request_count >= max_requests:
            remaining_time = window_seconds - (current_time - self._rate_limit_window_start).total_seconds()
            raise RateLimitExceededError(
                f"{self.platform_name} API速率限制超出",
                platform_name=self.platform_name,
                retry_after=int(remaining_time)
            )
        
        # 请求间隔限制
        min_interval = self.rate_limits.get('min_interval_seconds', 0)
        if (self._last_request_time and min_interval > 0):
            elapsed = (current_time - self._last_request_time).total_seconds()
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
    
    def _update_request_stats(self):
        """更新请求统计"""
        self._last_request_time = datetime.utcnow()
        self._request_count += 1
    
    def _extract_error_message(self, response_data: Dict[str, Any]) -> str:
        """从响应中提取错误消息"""
        # 通用错误字段
        error_fields = ['error_description', 'error', 'message', 'msg', 'text']
        
        for field in error_fields:
            if field in response_data:
                error_value = response_data[field]
                if isinstance(error_value, str):
                    return error_value
                elif isinstance(error_value, dict) and 'message' in error_value:
                    return error_value['message']
        
        return str(response_data)
    
    def _standardize_user_profile(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化用户资料数据格式
        
        子类应该重写此方法以适配平台特定的数据格式
        """
        return {
            'user_id': raw_data.get('id', raw_data.get('uid', '')),
            'username': raw_data.get('screen_name', raw_data.get('name', '')),
            'display_name': raw_data.get('display_name', raw_data.get('nickname', '')),
            'avatar_url': raw_data.get('profile_image_url', raw_data.get('avatar_url', '')),
            'bio': raw_data.get('description', raw_data.get('bio', '')),
            'verification_status': 'verified' if raw_data.get('verified', False) else 'unverified',
            'account_type': raw_data.get('account_type', 'personal'),
            'location': raw_data.get('location', ''),
            'website_url': raw_data.get('url', ''),
            'created_at': raw_data.get('created_at'),
            'raw_data': raw_data
        }
    
    def _standardize_user_stats(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化用户统计数据格式
        
        子类应该重写此方法以适配平台特定的数据格式
        """
        return {
            'follower_count': raw_data.get('followers_count', 0),
            'following_count': raw_data.get('friends_count', 0),
            'post_count': raw_data.get('statuses_count', 0),
            'like_count': raw_data.get('favourites_count', 0),
            'engagement_rate': raw_data.get('engagement_rate'),
            'last_post_time': raw_data.get('last_post_time'),
            'raw_data': raw_data
        }
    
    def _standardize_posts(self, raw_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        标准化用户发布内容格式
        
        子类应该重写此方法以适配平台特定的数据格式
        """
        standardized_posts = []
        
        for post in raw_posts:
            standardized_post = {
                'post_id': post.get('id', ''),
                'content': post.get('text', post.get('content', '')),
                'created_at': post.get('created_at'),
                'like_count': post.get('favorite_count', 0),
                'comment_count': post.get('comment_count', 0),
                'share_count': post.get('repost_count', 0),
                'media_urls': post.get('media_urls', []),
                'hashtags': post.get('hashtags', []),
                'mentions': post.get('mentions', []),
                'raw_data': post
            }
            standardized_posts.append(standardized_post)
        
        return {
            'posts': standardized_posts,
            'total_count': len(standardized_posts),
            'has_more': False  # 子类应该设置正确的值
        }
    
    def get_platform_info(self) -> Dict[str, Any]:
        """获取平台信息"""
        return {
            'name': self.platform_name,
            'type': self.platform_type,
            'api_base_url': self.api_base_url,
            'supported_sync_types': self.supported_sync_types,
            'features': self.features,
            'rate_limits': self.rate_limits,
            'is_active': self.is_active
        }