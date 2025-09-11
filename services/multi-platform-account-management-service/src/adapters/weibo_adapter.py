"""
微博平台适配器

实现微博API的具体调用逻辑
处理微博特定的数据格式和API接口
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from .base_adapter import BasePlatformAdapter

logger = logging.getLogger(__name__)


class WeiboAdapter(BasePlatformAdapter):
    """
    微博平台适配器
    
    实现微博API v2的接口调用和数据处理
    支持用户信息获取、微博内容同步等功能
    """
    
    @property
    def platform_type(self) -> str:
        return "social_media"
    
    @property
    def supported_sync_types(self) -> List[str]:
        return ["profile", "stats", "posts", "followers"]
    
    async def get_user_profile(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取微博用户资料信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID，如果不提供则获取当前认证用户信息
            
        Returns:
            Dict: 标准化的用户资料信息
        """
        try:
            # 构建API URL
            if user_id:
                url = f"{self.api_base_url}/2/users/show.json"
                params = {'uid': user_id}
            else:
                url = f"{self.api_base_url}/2/account/get_uid.json"
                params = {}
            
            # 设置请求头
            headers = {
                'Authorization': f'OAuth2 {access_token}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # 如果没有指定用户ID，先获取当前用户ID
            if not user_id:
                uid_response = await self._make_request('GET', url, headers=headers, params=params)
                user_id = uid_response.get('uid')
                if not user_id:
                    raise ValueError("无法获取用户ID")
                
                # 获取用户详细信息
                url = f"{self.api_base_url}/2/users/show.json"
                params = {'uid': user_id}
            
            # 获取用户信息
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 标准化数据格式
            profile_data = self._standardize_weibo_profile(response)
            
            logger.info(f"成功获取微博用户 {user_id} 的资料信息")
            return profile_data
            
        except Exception as e:
            logger.error(f"获取微博用户资料失败: {e}")
            raise
    
    async def get_user_stats(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取微博用户统计信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            
        Returns:
            Dict: 标准化的用户统计信息
        """
        try:
            # 先获取用户基本信息（包含统计数据）
            profile_data = await self.get_user_profile(access_token, user_id)
            
            # 提取统计信息
            stats_data = {
                'follower_count': profile_data['raw_data'].get('followers_count', 0),
                'following_count': profile_data['raw_data'].get('friends_count', 0),
                'post_count': profile_data['raw_data'].get('statuses_count', 0),
                'like_count': profile_data['raw_data'].get('favourites_count', 0),
                'verified_count': profile_data['raw_data'].get('verified_count', 0),
                'last_updated': datetime.utcnow().isoformat(),
                'platform_specific': {
                    'bi_followers_count': profile_data['raw_data'].get('bi_followers_count', 0),
                    'verified_type': profile_data['raw_data'].get('verified_type', 0),
                    'verified_reason': profile_data['raw_data'].get('verified_reason', ''),
                    'online_status': profile_data['raw_data'].get('online_status', 0)
                }
            }
            
            logger.info(f"成功获取微博用户 {user_id} 的统计信息")
            return stats_data
            
        except Exception as e:
            logger.error(f"获取微博用户统计失败: {e}")
            raise
    
    async def get_user_posts(self, access_token: str, user_id: str = None,
                           limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取微博用户发布内容
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            limit: 限制数量(1-200)
            offset: 页码(从1开始)
            
        Returns:
            Dict: 标准化的用户发布内容
        """
        try:
            # 参数验证
            limit = min(max(limit, 1), 200)
            page = max(offset // limit + 1, 1)
            
            # 构建API URL和参数
            url = f"{self.api_base_url}/2/statuses/user_timeline.json"
            params = {
                'count': limit,
                'page': page,
                'trim_user': 0,  # 返回完整用户信息
                'include_entities': 1  # 包含实体信息
            }
            
            if user_id:
                params['uid'] = user_id
            
            # 设置请求头
            headers = {
                'Authorization': f'OAuth2 {access_token}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # 获取微博列表
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 标准化数据格式
            posts_data = self._standardize_weibo_posts(response.get('statuses', []))
            posts_data.update({
                'total_count': response.get('total_number', 0),
                'has_more': len(response.get('statuses', [])) == limit,
                'next_cursor': str(page + 1) if len(response.get('statuses', [])) == limit else None
            })
            
            logger.info(f"成功获取微博用户 {user_id} 的 {len(posts_data['posts'])} 条微博")
            return posts_data
            
        except Exception as e:
            logger.error(f"获取微博用户发布内容失败: {e}")
            raise
    
    async def get_followers(self, access_token: str, user_id: str = None,
                          limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取微博粉丝列表
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的粉丝列表
        """
        try:
            # 参数验证
            limit = min(max(limit, 1), 200)
            cursor = offset
            
            # 构建API URL和参数
            url = f"{self.api_base_url}/2/friendships/followers.json"
            params = {
                'count': limit,
                'cursor': cursor,
                'trim_status': 1  # 不返回微博信息
            }
            
            if user_id:
                params['uid'] = user_id
            
            # 设置请求头
            headers = {
                'Authorization': f'OAuth2 {access_token}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # 获取粉丝列表
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 标准化粉丝数据
            followers = []
            for user_data in response.get('users', []):
                follower = {
                    'user_id': str(user_data.get('id', '')),
                    'username': user_data.get('screen_name', ''),
                    'display_name': user_data.get('name', ''),
                    'avatar_url': user_data.get('profile_image_url', ''),
                    'verified': user_data.get('verified', False),
                    'follower_count': user_data.get('followers_count', 0),
                    'following_count': user_data.get('friends_count', 0),
                    'location': user_data.get('location', ''),
                    'description': user_data.get('description', ''),
                    'created_at': user_data.get('created_at'),
                    'raw_data': user_data
                }
                followers.append(follower)
            
            result = {
                'followers': followers,
                'total_count': response.get('total_number', 0),
                'has_more': response.get('next_cursor', 0) > 0,
                'next_cursor': str(response.get('next_cursor', 0)) if response.get('next_cursor', 0) > 0 else None
            }
            
            logger.info(f"成功获取微博用户 {user_id} 的 {len(followers)} 个粉丝")
            return result
            
        except Exception as e:
            logger.error(f"获取微博粉丝列表失败: {e}")
            raise
    
    def _standardize_weibo_profile(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化微博用户资料数据"""
        return {
            'user_id': str(raw_data.get('id', raw_data.get('idstr', ''))),
            'username': raw_data.get('screen_name', ''),
            'display_name': raw_data.get('name', ''),
            'avatar_url': raw_data.get('avatar_large', raw_data.get('profile_image_url', '')),
            'bio': raw_data.get('description', ''),
            'verification_status': 'verified' if raw_data.get('verified', False) else 'unverified',
            'account_type': self._determine_weibo_account_type(raw_data),
            'location': raw_data.get('location', ''),
            'website_url': raw_data.get('url', ''),
            'gender': raw_data.get('gender', 'n'),  # m:男性, f:女性, n:未知
            'created_at': raw_data.get('created_at'),
            'last_updated': datetime.utcnow().isoformat(),
            'platform_specific': {
                'verified_type': raw_data.get('verified_type', 0),
                'verified_reason': raw_data.get('verified_reason', ''),
                'urank': raw_data.get('urank', 0),
                'mbrank': raw_data.get('mbrank', 0),
                'mbtype': raw_data.get('mbtype', 0),
                'class': raw_data.get('class', 0),
                'lang': raw_data.get('lang', 'zh-cn')
            },
            'raw_data': raw_data
        }
    
    def _determine_weibo_account_type(self, raw_data: Dict[str, Any]) -> str:
        """根据微博数据确定账号类型"""
        verified_type = raw_data.get('verified_type', 0)
        
        if verified_type == 0:
            return 'personal'
        elif verified_type in [1, 2, 3, 4, 5]:
            return 'business'  # 机构认证
        elif verified_type in [200, 220]:
            return 'creator'   # 个人认证
        else:
            return 'organization'
    
    def _standardize_weibo_posts(self, raw_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """标准化微博发布内容数据"""
        standardized_posts = []
        
        for post in raw_posts:
            # 提取媒体URL
            media_urls = []
            pic_urls = post.get('pic_urls', [])
            for pic in pic_urls:
                if isinstance(pic, dict) and 'thumbnail_pic' in pic:
                    # 获取原图URL
                    original_url = pic['thumbnail_pic'].replace('thumbnail', 'large')
                    media_urls.append(original_url)
            
            # 提取话题标签
            hashtags = []
            entities = post.get('entities', {})
            if 'hashtags' in entities:
                hashtags = [tag['text'] for tag in entities['hashtags']]
            
            # 提取用户提及
            mentions = []
            if 'user_mentions' in entities:
                mentions = [{'username': mention['screen_name'], 'user_id': str(mention['id'])} 
                           for mention in entities['user_mentions']]
            
            standardized_post = {
                'post_id': str(post.get('id', post.get('idstr', ''))),
                'content': post.get('text', ''),
                'created_at': post.get('created_at'),
                'like_count': post.get('attitudes_count', 0),
                'comment_count': post.get('comments_count', 0),
                'share_count': post.get('reposts_count', 0),
                'media_urls': media_urls,
                'hashtags': hashtags,
                'mentions': mentions,
                'source': post.get('source', ''),
                'geo': post.get('geo'),
                'is_repost': bool(post.get('retweeted_status')),
                'platform_specific': {
                    'mlevel': post.get('mlevel', 0),
                    'visible': post.get('visible', {}),
                    'darwin_tags': post.get('darwin_tags', []),
                    'hot_weibo_tags': post.get('hot_weibo_tags', []),
                    'text_tag_tips': post.get('text_tag_tips', [])
                },
                'raw_data': post
            }
            
            # 如果是转发微博，添加原微博信息
            if post.get('retweeted_status'):
                standardized_post['repost_info'] = {
                    'original_post_id': str(post['retweeted_status'].get('id', '')),
                    'original_content': post['retweeted_status'].get('text', ''),
                    'original_user': {
                        'user_id': str(post['retweeted_status']['user'].get('id', '')),
                        'username': post['retweeted_status']['user'].get('screen_name', '')
                    }
                }
            
            standardized_posts.append(standardized_post)
        
        return {
            'posts': standardized_posts,
            'platform': 'weibo'
        }