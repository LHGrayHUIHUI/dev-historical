"""
OAuth 认证服务

提供多平台OAuth 2.0认证功能
支持授权码交换、令牌刷新、授权URL生成等核心OAuth流程
"""

import asyncio
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode, parse_qs, urlparse
import aiohttp
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..config.settings import settings
from ..models.account_models import Platform
from ..utils.exceptions import (
    OAuthError, 
    PlatformNotSupportedError,
    TokenExpiredError,
    InvalidTokenError
)

logger = logging.getLogger(__name__)


class OAuthService:
    """
    OAuth认证服务类
    
    负责处理多个社交媒体平台的OAuth 2.0认证流程
    提供授权URL生成、令牌交换、令牌刷新等功能
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        """
        初始化OAuth服务
        
        Args:
            db_session: 数据库会话
            redis_client: Redis客户端
        """
        self.db_session = db_session
        self.redis_client = redis_client
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # OAuth状态存储TTL (30分钟)
        self.state_ttl = 1800
        
    async def generate_authorization_url(self, platform_name: str, user_id: int, 
                                       redirect_uri: str = None) -> Dict[str, Any]:
        """
        生成平台的OAuth授权URL
        
        Args:
            platform_name: 平台名称
            user_id: 用户ID
            redirect_uri: 回调URI
            
        Returns:
            Dict: 包含authorize_url和state的字典
            
        Raises:
            PlatformNotSupportedError: 不支持的平台
            OAuthError: OAuth配置错误
        """
        try:
            # 获取平台配置
            platform = await self._get_platform_config(platform_name)
            oauth_config = platform.oauth_config
            
            if not oauth_config or 'authorize_url' not in oauth_config:
                raise OAuthError(f"平台 {platform_name} 的OAuth配置不完整")
            
            # 生成状态码
            state = secrets.token_urlsafe(32)
            
            # 构建授权参数
            params = {
                'client_id': oauth_config['client_id'],
                'redirect_uri': redirect_uri or settings.oauth_callback_base_url,
                'response_type': 'code',
                'state': state,
                'scope': oauth_config.get('scope', '')
            }
            
            # 添加平台特定参数
            if platform_name == 'wechat':
                params['appid'] = oauth_config['client_id']
                params.pop('client_id')
            elif platform_name == 'douyin':
                params['client_key'] = oauth_config['client_id']
                params.pop('client_id')
            
            # 构建授权URL
            authorize_url = f"{oauth_config['authorize_url']}?{urlencode(params)}"
            
            # 存储状态信息到Redis
            state_data = {
                'platform_name': platform_name,
                'user_id': user_id,
                'redirect_uri': redirect_uri,
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(seconds=self.state_ttl)).isoformat()
            }
            
            await self.redis_client.setex(
                f"oauth_state:{state}",
                self.state_ttl,
                json.dumps(state_data)
            )
            
            logger.info(f"为用户 {user_id} 生成 {platform_name} 授权URL")
            
            return {
                'authorize_url': authorize_url,
                'state': state,
                'expires_at': datetime.utcnow() + timedelta(seconds=self.state_ttl)
            }
            
        except Exception as e:
            logger.error(f"生成授权URL失败: {e}")
            if isinstance(e, (PlatformNotSupportedError, OAuthError)):
                raise
            raise OAuthError(f"生成授权URL失败: {str(e)}")
    
    async def exchange_code_for_token(self, platform_name: str, code: str, 
                                    state: str = None, redirect_uri: str = None) -> Dict[str, Any]:
        """
        使用授权码交换访问令牌
        
        Args:
            platform_name: 平台名称
            code: 授权码
            state: 状态码
            redirect_uri: 回调URI
            
        Returns:
            Dict: 令牌信息
            
        Raises:
            OAuthError: 令牌交换失败
            InvalidTokenError: 无效的授权码或状态
        """
        try:
            # 验证状态码
            if state:
                state_data = await self._validate_oauth_state(state)
                if state_data['platform_name'] != platform_name:
                    raise InvalidTokenError("状态码与平台不匹配")
            
            # 获取平台配置
            platform = await self._get_platform_config(platform_name)
            oauth_config = platform.oauth_config
            
            if not oauth_config or 'token_url' not in oauth_config:
                raise OAuthError(f"平台 {platform_name} 的令牌URL配置不完整")
            
            # 准备令牌交换参数
            token_params = {
                'client_id': oauth_config['client_id'],
                'client_secret': oauth_config['client_secret'],
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': redirect_uri or settings.oauth_callback_base_url
            }
            
            # 平台特定参数调整
            if platform_name == 'wechat':
                token_params['appid'] = token_params.pop('client_id')
                token_params['secret'] = token_params.pop('client_secret')
            elif platform_name == 'douyin':
                token_params['client_key'] = token_params.pop('client_id')
                token_params['client_secret'] = oauth_config['client_secret']
            
            # 发起令牌交换请求
            token_data = await self._make_token_request(
                oauth_config['token_url'], 
                token_params,
                platform_name
            )
            
            # 获取用户信息
            user_info = await self._get_user_info(platform_name, token_data['access_token'])
            
            # 合并令牌和用户信息
            result = {
                **token_data,
                'user_info': user_info,
                'platform_name': platform_name
            }
            
            # 清理状态信息
            if state:
                await self.redis_client.delete(f"oauth_state:{state}")
            
            logger.info(f"成功交换 {platform_name} 平台令牌")
            return result
            
        except Exception as e:
            logger.error(f"令牌交换失败: {e}")
            if isinstance(e, (OAuthError, InvalidTokenError)):
                raise
            raise OAuthError(f"令牌交换失败: {str(e)}")
    
    async def refresh_access_token(self, platform_name: str, refresh_token: str) -> Dict[str, Any]:
        """
        刷新访问令牌
        
        Args:
            platform_name: 平台名称
            refresh_token: 刷新令牌
            
        Returns:
            Dict: 新的令牌信息
            
        Raises:
            TokenExpiredError: 刷新令牌过期
            OAuthError: 刷新失败
        """
        try:
            # 获取平台配置
            platform = await self._get_platform_config(platform_name)
            oauth_config = platform.oauth_config
            
            if not oauth_config or 'token_url' not in oauth_config:
                raise OAuthError(f"平台 {platform_name} 的令牌URL配置不完整")
            
            # 准备刷新令牌参数
            refresh_params = {
                'client_id': oauth_config['client_id'],
                'client_secret': oauth_config['client_secret'],
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token'
            }
            
            # 平台特定参数调整
            if platform_name == 'wechat':
                refresh_params['appid'] = refresh_params.pop('client_id')
                refresh_params['secret'] = refresh_params.pop('client_secret')
            
            # 发起令牌刷新请求
            token_data = await self._make_token_request(
                oauth_config['token_url'],
                refresh_params,
                platform_name
            )
            
            logger.info(f"成功刷新 {platform_name} 平台令牌")
            return {
                **token_data,
                'platform_name': platform_name
            }
            
        except Exception as e:
            logger.error(f"令牌刷新失败: {e}")
            if isinstance(e, (TokenExpiredError, OAuthError)):
                raise
            raise OAuthError(f"令牌刷新失败: {str(e)}")
    
    async def validate_access_token(self, platform_name: str, access_token: str) -> bool:
        """
        验证访问令牌有效性
        
        Args:
            platform_name: 平台名称
            access_token: 访问令牌
            
        Returns:
            bool: 令牌是否有效
        """
        try:
            # 通过获取用户信息来验证令牌
            user_info = await self._get_user_info(platform_name, access_token)
            return user_info is not None
        except Exception as e:
            logger.warning(f"令牌验证失败: {e}")
            return False
    
    async def revoke_token(self, platform_name: str, access_token: str) -> bool:
        """
        撤销访问令牌
        
        Args:
            platform_name: 平台名称
            access_token: 访问令牌
            
        Returns:
            bool: 撤销是否成功
        """
        try:
            # 获取平台配置
            platform = await self._get_platform_config(platform_name)
            oauth_config = platform.oauth_config
            
            # 检查是否支持令牌撤销
            revoke_url = oauth_config.get('revoke_url')
            if not revoke_url:
                logger.warning(f"平台 {platform_name} 不支持令牌撤销")
                return False
            
            # 准备撤销参数
            revoke_params = {
                'token': access_token,
                'token_type_hint': 'access_token'
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(revoke_url, data=revoke_params) as response:
                    success = response.status == 200
                    
            if success:
                logger.info(f"成功撤销 {platform_name} 平台令牌")
            else:
                logger.warning(f"撤销 {platform_name} 平台令牌失败")
                
            return success
            
        except Exception as e:
            logger.error(f"撤销令牌失败: {e}")
            return False
    
    async def _get_platform_config(self, platform_name: str) -> Platform:
        """获取平台配置"""
        result = await self.db_session.execute(
            select(Platform).where(
                Platform.name == platform_name,
                Platform.is_active == True
            )
        )
        
        platform = result.scalar_one_or_none()
        if not platform:
            raise PlatformNotSupportedError(f"不支持的平台: {platform_name}")
            
        return platform
    
    async def _validate_oauth_state(self, state: str) -> Dict[str, Any]:
        """验证OAuth状态码"""
        state_json = await self.redis_client.get(f"oauth_state:{state}")
        if not state_json:
            raise InvalidTokenError("无效的状态码或状态已过期")
        
        state_data = json.loads(state_json)
        
        # 检查过期时间
        expires_at = datetime.fromisoformat(state_data['expires_at'])
        if datetime.utcnow() > expires_at:
            raise InvalidTokenError("状态码已过期")
        
        return state_data
    
    async def _make_token_request(self, token_url: str, params: Dict[str, Any], 
                                platform_name: str) -> Dict[str, Any]:
        """发起令牌请求"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # 根据平台选择请求方式
            if platform_name in ['wechat', 'douyin']:
                # GET 请求
                async with session.get(token_url, params=params) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        raise OAuthError(f"令牌请求失败: HTTP {response.status}")
                    
                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError:
                        # 微信返回的可能是查询字符串格式
                        if platform_name == 'wechat':
                            data = dict(parse_qs(response_text))
                            # 处理查询字符串解析结果
                            data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v 
                                   for k, v in data.items()}
                        else:
                            raise OAuthError(f"无效的响应格式: {response_text}")
            else:
                # POST 请求 (标准OAuth)
                async with session.post(token_url, data=params) as response:
                    if response.status != 200:
                        raise OAuthError(f"令牌请求失败: HTTP {response.status}")
                    
                    data = await response.json()
            
            # 检查错误响应
            if 'error' in data:
                error_desc = data.get('error_description', data['error'])
                raise OAuthError(f"OAuth错误: {error_desc}")
            
            # 检查必需字段
            if 'access_token' not in data:
                raise OAuthError("响应中缺少访问令牌")
            
            # 标准化响应格式
            result = {
                'access_token': data['access_token'],
                'token_type': data.get('token_type', 'Bearer'),
                'expires_in': data.get('expires_in'),
                'refresh_token': data.get('refresh_token'),
                'scope': data.get('scope', '')
            }
            
            # 计算过期时间
            if result['expires_in']:
                result['expires_at'] = datetime.utcnow() + timedelta(seconds=int(result['expires_in']))
            
            return result
    
    async def _get_user_info(self, platform_name: str, access_token: str) -> Dict[str, Any]:
        """获取用户信息"""
        # 获取平台配置
        platform = await self._get_platform_config(platform_name)
        oauth_config = platform.oauth_config
        
        user_info_url = oauth_config.get('user_info_url')
        if not user_info_url:
            logger.warning(f"平台 {platform_name} 没有配置用户信息URL")
            return {}
        
        # 准备请求头或参数
        if platform_name == 'wechat':
            # 微信需要openid，这里简化处理
            params = {'access_token': access_token, 'openid': 'placeholder'}
        else:
            params = {}
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'User-Agent': 'Historical-Text-Account-Manager/1.0'
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                if platform_name == 'wechat':
                    async with session.get(user_info_url, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                else:
                    async with session.get(user_info_url, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
            
            logger.warning(f"获取 {platform_name} 用户信息失败")
            return {}
            
        except Exception as e:
            logger.error(f"获取用户信息异常: {e}")
            return {}