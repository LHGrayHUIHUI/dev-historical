"""
微信平台适配器

实现微信公众号API的具体调用逻辑
处理微信特定的数据格式和API接口
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from .base_adapter import BasePlatformAdapter

logger = logging.getLogger(__name__)


class WeChatAdapter(BasePlatformAdapter):
    """
    微信平台适配器
    
    实现微信公众号API的接口调用和数据处理
    主要支持公众号信息获取、粉丝管理等功能
    """
    
    @property
    def platform_type(self) -> str:
        return "social_media"
    
    @property
    def supported_sync_types(self) -> List[str]:
        return ["profile", "stats", "followers"]  # 微信不支持获取发布内容历史
    
    async def get_user_profile(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取微信公众号基本信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID（对于公众号，通常是openid）
            
        Returns:
            Dict: 标准化的用户资料信息
        """
        try:
            # 如果提供了openid，获取用户基本信息
            if user_id:
                return await self._get_wechat_user_info(access_token, user_id)
            else:
                # 获取公众号基本信息
                return await self._get_account_info(access_token)
                
        except Exception as e:
            logger.error(f"获取微信用户资料失败: {e}")
            raise
    
    async def get_user_stats(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取微信公众号统计信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            
        Returns:
            Dict: 标准化的用户统计信息
        """
        try:
            # 获取用户增减数据
            user_summary = await self._get_user_summary(access_token)
            
            # 获取累计用户数据
            user_cumulate = await self._get_user_cumulate(access_token)
            
            # 整合统计信息
            stats_data = {
                'follower_count': user_cumulate.get('cumulate_user', 0),
                'following_count': 0,  # 微信公众号不适用
                'post_count': 0,  # 需要单独API获取
                'like_count': 0,  # 微信不提供总赞数
                'daily_new_followers': user_summary.get('new_user', 0),
                'daily_unfollows': user_summary.get('cancel_user', 0),
                'last_updated': datetime.utcnow().isoformat(),
                'platform_specific': {
                    'net_user': user_summary.get('net_user', 0),  # 净增关注人数
                    'cumulate_user': user_cumulate.get('cumulate_user', 0)  # 总用户量
                }
            }
            
            logger.info("成功获取微信公众号统计信息")
            return stats_data
            
        except Exception as e:
            logger.error(f"获取微信公众号统计失败: {e}")
            raise
    
    async def get_user_posts(self, access_token: str, user_id: str = None,
                           limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取微信公众号图文消息（有限制）
        
        注意：微信API对历史消息获取有严格限制
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的发布内容
        """
        try:
            # 微信API限制，只能获取最近的素材
            posts_data = await self._get_material_list(access_token, "news", offset, limit)
            
            # 标准化数据格式
            standardized_posts = self._standardize_wechat_posts(posts_data.get('item', []))
            
            result = {
                'posts': standardized_posts,
                'total_count': posts_data.get('total_count', 0),
                'has_more': posts_data.get('item_count', 0) == limit,
                'message': '微信平台仅支持获取永久素材中的图文消息'
            }
            
            logger.info(f"成功获取微信公众号 {len(standardized_posts)} 条图文消息")
            return result
            
        except Exception as e:
            logger.error(f"获取微信图文消息失败: {e}")
            # 返回空结果而不是抛出异常
            return {
                'posts': [],
                'total_count': 0,
                'has_more': False,
                'message': f'获取微信图文消息失败: {str(e)}'
            }
    
    async def get_followers(self, access_token: str, user_id: str = None,
                          limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取微信公众号粉丝列表
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的粉丝列表
        """
        try:
            # 获取用户openid列表
            openid_list = await self._get_user_list(access_token, limit, offset)
            
            if not openid_list.get('data', {}).get('openid'):
                return {
                    'followers': [],
                    'total_count': openid_list.get('total', 0),
                    'has_more': False,
                    'message': '暂无粉丝数据'
                }
            
            # 批量获取用户详细信息
            followers = []
            openids = openid_list['data']['openid'][:limit]  # 限制数量
            
            # 分批获取用户信息（微信API限制单次最多100个）
            batch_size = 100
            for i in range(0, len(openids), batch_size):
                batch_openids = openids[i:i + batch_size]
                batch_users = await self._batch_get_user_info(access_token, batch_openids)
                
                for user_info in batch_users:
                    if user_info.get('subscribe') == 1:  # 关注状态
                        follower = {
                            'user_id': user_info.get('openid', ''),
                            'username': user_info.get('nickname', ''),
                            'display_name': user_info.get('nickname', ''),
                            'avatar_url': user_info.get('headimgurl', ''),
                            'verified': False,  # 微信粉丝无认证概念
                            'follower_count': 0,  # 不可获取
                            'following_count': 0,  # 不可获取
                            'location': f"{user_info.get('country', '')} {user_info.get('province', '')} {user_info.get('city', '')}".strip(),
                            'gender': '男' if user_info.get('sex') == 1 else ('女' if user_info.get('sex') == 2 else '未知'),
                            'language': user_info.get('language', ''),
                            'subscribe_time': datetime.fromtimestamp(user_info.get('subscribe_time', 0)).isoformat() if user_info.get('subscribe_time') else None,
                            'raw_data': user_info
                        }
                        followers.append(follower)
            
            result = {
                'followers': followers,
                'total_count': openid_list.get('total', 0),
                'has_more': openid_list.get('count', 0) == limit and openid_list.get('next_openid'),
                'next_cursor': openid_list.get('next_openid') if openid_list.get('next_openid') else None
            }
            
            logger.info(f"成功获取微信公众号 {len(followers)} 个粉丝信息")
            return result
            
        except Exception as e:
            logger.error(f"获取微信粉丝列表失败: {e}")
            raise
    
    async def _get_account_info(self, access_token: str) -> Dict[str, Any]:
        """获取公众号基本信息"""
        # 微信没有直接获取公众号信息的API，这里返回模拟数据
        return {
            'user_id': 'wechat_account',
            'username': '微信公众号',
            'display_name': '微信公众号',
            'avatar_url': '',
            'bio': '微信公众号账户',
            'verification_status': 'unverified',
            'account_type': 'business',
            'location': '',
            'website_url': '',
            'created_at': datetime.utcnow().isoformat(),
            'last_updated': datetime.utcnow().isoformat(),
            'platform_specific': {
                'account_type': 'subscription',  # 订阅号/服务号
                'verify_type': 'unverified'
            },
            'raw_data': {'access_token': access_token[:10] + '...'}  # 隐藏完整token
        }
    
    async def _get_wechat_user_info(self, access_token: str, openid: str) -> Dict[str, Any]:
        """获取微信用户基本信息"""
        url = "https://api.weixin.qq.com/cgi-bin/user/info"
        params = {
            'access_token': access_token,
            'openid': openid,
            'lang': 'zh_CN'
        }
        
        response = await self._make_request('GET', url, params=params)
        
        return {
            'user_id': response.get('openid', ''),
            'username': response.get('nickname', ''),
            'display_name': response.get('nickname', ''),
            'avatar_url': response.get('headimgurl', ''),
            'bio': '',
            'verification_status': 'unverified',
            'account_type': 'personal',
            'location': f"{response.get('country', '')} {response.get('province', '')} {response.get('city', '')}".strip(),
            'website_url': '',
            'created_at': datetime.fromtimestamp(response.get('subscribe_time', 0)).isoformat() if response.get('subscribe_time') else None,
            'last_updated': datetime.utcnow().isoformat(),
            'platform_specific': {
                'sex': response.get('sex', 0),  # 1:男性，2:女性，0:未知
                'language': response.get('language', ''),
                'subscribe': response.get('subscribe', 0),
                'subscribe_scene': response.get('subscribe_scene', ''),
                'qr_scene': response.get('qr_scene', 0),
                'qr_scene_str': response.get('qr_scene_str', '')
            },
            'raw_data': response
        }
    
    async def _get_user_summary(self, access_token: str) -> Dict[str, Any]:
        """获取用户增减数据（最近7天）"""
        from datetime import timedelta
        
        url = "https://api.weixin.qq.com/datacube/getusersummary"
        
        # 获取最近一天的数据
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        data = {
            'begin_date': yesterday,
            'end_date': yesterday
        }
        
        headers = {'Content-Type': 'application/json'}
        params = {'access_token': access_token}
        
        try:
            response = await self._make_request('POST', url, headers=headers, params=params, json_data=data)
            
            if response.get('list'):
                return response['list'][0]  # 返回最近一天的数据
            else:
                return {'new_user': 0, 'cancel_user': 0, 'net_user': 0}
                
        except Exception as e:
            logger.warning(f"获取用户增减数据失败: {e}")
            return {'new_user': 0, 'cancel_user': 0, 'net_user': 0}
    
    async def _get_user_cumulate(self, access_token: str) -> Dict[str, Any]:
        """获取累计用户数据"""
        from datetime import timedelta
        
        url = "https://api.weixin.qq.com/datacube/getusercumulate"
        
        # 获取最近一天的数据
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        data = {
            'begin_date': yesterday,
            'end_date': yesterday
        }
        
        headers = {'Content-Type': 'application/json'}
        params = {'access_token': access_token}
        
        try:
            response = await self._make_request('POST', url, headers=headers, params=params, json_data=data)
            
            if response.get('list'):
                return response['list'][0]  # 返回最近一天的数据
            else:
                return {'cumulate_user': 0}
                
        except Exception as e:
            logger.warning(f"获取累计用户数据失败: {e}")
            return {'cumulate_user': 0}
    
    async def _get_material_list(self, access_token: str, material_type: str = "news", 
                               offset: int = 0, count: int = 20) -> Dict[str, Any]:
        """获取素材列表"""
        url = "https://api.weixin.qq.com/cgi-bin/material/batchget_material"
        
        data = {
            'type': material_type,
            'offset': offset,
            'count': min(count, 20)  # 微信API限制单次最多20条
        }
        
        headers = {'Content-Type': 'application/json'}
        params = {'access_token': access_token}
        
        response = await self._make_request('POST', url, headers=headers, params=params, json_data=data)
        return response
    
    async def _get_user_list(self, access_token: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """获取用户列表"""
        url = "https://api.weixin.qq.com/cgi-bin/user/get"
        
        params = {
            'access_token': access_token,
            'next_openid': ''  # 微信使用next_openid分页
        }
        
        response = await self._make_request('GET', url, params=params)
        return response
    
    async def _batch_get_user_info(self, access_token: str, openids: List[str]) -> List[Dict[str, Any]]:
        """批量获取用户信息"""
        url = "https://api.weixin.qq.com/cgi-bin/user/info/batchget"
        
        user_list = [{'openid': openid, 'lang': 'zh_CN'} for openid in openids]
        data = {'user_list': user_list}
        
        headers = {'Content-Type': 'application/json'}
        params = {'access_token': access_token}
        
        response = await self._make_request('POST', url, headers=headers, params=params, json_data=data)
        return response.get('user_info_list', [])
    
    def _standardize_wechat_posts(self, raw_posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化微信图文消息数据"""
        standardized_posts = []
        
        for item in raw_posts:
            content = item.get('content', {})
            news_item = content.get('news_item', [])
            
            if news_item:
                # 微信图文消息可能包含多篇文章
                for article in news_item:
                    post = {
                        'post_id': item.get('media_id', ''),
                        'content': article.get('digest', ''),  # 摘要作为内容
                        'title': article.get('title', ''),
                        'created_at': datetime.fromtimestamp(item.get('update_time', 0)).isoformat() if item.get('update_time') else None,
                        'like_count': 0,  # 微信不提供
                        'comment_count': 0,  # 微信不提供
                        'share_count': 0,  # 微信不提供
                        'media_urls': [article.get('thumb_url', '')] if article.get('thumb_url') else [],
                        'hashtags': [],
                        'mentions': [],
                        'url': article.get('url', ''),
                        'platform_specific': {
                            'author': article.get('author', ''),
                            'content_source_url': article.get('content_source_url', ''),
                            'show_cover_pic': article.get('show_cover_pic', 0),
                            'thumb_media_id': article.get('thumb_media_id', ''),
                            'need_open_comment': article.get('need_open_comment', 0),
                            'only_fans_can_comment': article.get('only_fans_can_comment', 0)
                        },
                        'raw_data': {
                            'media_id': item.get('media_id'),
                            'update_time': item.get('update_time'),
                            'article': article
                        }
                    }
                    standardized_posts.append(post)
        
        return standardized_posts