"""
抖音平台适配器

实现抖音开放平台API的具体调用逻辑
处理抖音特定的数据格式和API接口
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from .base_adapter import BasePlatformAdapter

logger = logging.getLogger(__name__)


class DouyinAdapter(BasePlatformAdapter):
    """
    抖音平台适配器
    
    实现抖音开放平台API的接口调用和数据处理
    支持用户信息获取、视频内容同步、粉丝管理等功能
    """
    
    @property
    def platform_type(self) -> str:
        return "short_video"
    
    @property
    def supported_sync_types(self) -> List[str]:
        return ["profile", "stats", "posts", "followers"]
    
    async def get_user_profile(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取抖音用户资料信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户openid
            
        Returns:
            Dict: 标准化的用户资料信息
        """
        try:
            # 构建API URL
            url = f"{self.api_base_url}/oauth/userinfo/"
            
            # 设置请求参数
            params = {
                'access_token': access_token,
                'open_id': user_id if user_id else None
            }
            
            # 移除空值参数
            params = {k: v for k, v in params.items() if v is not None}
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取用户信息
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if response.get('error_code') != 0:
                error_msg = response.get('description', '获取用户信息失败')
                raise ValueError(f"抖音API错误: {error_msg}")
            
            user_data = response.get('data', {})
            
            # 标准化数据格式
            profile_data = self._standardize_douyin_profile(user_data)
            
            logger.info(f"成功获取抖音用户 {user_id} 的资料信息")
            return profile_data
            
        except Exception as e:
            logger.error(f"获取抖音用户资料失败: {e}")
            raise
    
    async def get_user_stats(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取抖音用户统计信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户openid
            
        Returns:
            Dict: 标准化的用户统计信息
        """
        try:
            # 先获取用户基本信息
            profile_data = await self.get_user_profile(access_token, user_id)
            
            # 获取视频数据统计
            video_stats = await self._get_video_statistics(access_token, user_id)
            
            # 整合统计信息
            stats_data = {
                'follower_count': profile_data['raw_data'].get('follower_count', 0),
                'following_count': profile_data['raw_data'].get('following_count', 0),
                'post_count': video_stats.get('total_videos', 0),
                'like_count': video_stats.get('total_likes', 0),
                'view_count': video_stats.get('total_views', 0),
                'comment_count': video_stats.get('total_comments', 0),
                'share_count': video_stats.get('total_shares', 0),
                'last_updated': datetime.utcnow().isoformat(),
                'platform_specific': {
                    'aweme_count': profile_data['raw_data'].get('aweme_count', 0),
                    'total_favorited': profile_data['raw_data'].get('total_favorited', 0),
                    'verification_type': profile_data['raw_data'].get('verification_type', 0),
                    'enterprise_verify_reason': profile_data['raw_data'].get('enterprise_verify_reason', '')
                }
            }
            
            logger.info(f"成功获取抖音用户 {user_id} 的统计信息")
            return stats_data
            
        except Exception as e:
            logger.error(f"获取抖音用户统计失败: {e}")
            raise
    
    async def get_user_posts(self, access_token: str, user_id: str = None,
                           limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取抖音用户发布的视频内容
        
        Args:
            access_token: 访问令牌
            user_id: 用户openid
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的用户发布内容
        """
        try:
            # 参数验证
            limit = min(max(limit, 1), 20)  # 抖音API限制单次最多20条
            
            # 构建API URL
            url = f"{self.api_base_url}/video/list/"
            
            # 设置请求参数
            params = {
                'access_token': access_token,
                'open_id': user_id if user_id else None,
                'cursor': offset,
                'count': limit
            }
            
            # 移除空值参数
            params = {k: v for k, v in params.items() if v is not None}
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取视频列表
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if response.get('error_code') != 0:
                error_msg = response.get('description', '获取视频列表失败')
                raise ValueError(f"抖音API错误: {error_msg}")
            
            data = response.get('data', {})
            videos = data.get('list', [])
            
            # 标准化数据格式
            posts_data = self._standardize_douyin_posts(videos)
            posts_data.update({
                'total_count': data.get('total', 0),
                'has_more': data.get('has_more', False),
                'next_cursor': str(data.get('cursor', 0)) if data.get('has_more') else None
            })
            
            logger.info(f"成功获取抖音用户 {user_id} 的 {len(posts_data['posts'])} 个视频")
            return posts_data
            
        except Exception as e:
            logger.error(f"获取抖音用户发布内容失败: {e}")
            raise
    
    async def get_followers(self, access_token: str, user_id: str = None,
                          limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取抖音粉丝列表
        
        注意：抖音开放平台对粉丝列表获取有限制，可能需要特殊权限
        
        Args:
            access_token: 访问令牌
            user_id: 用户openid
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的粉丝列表
        """
        try:
            # 构建API URL
            url = f"{self.api_base_url}/fans/list/"
            
            # 设置请求参数
            params = {
                'access_token': access_token,
                'open_id': user_id if user_id else None,
                'cursor': offset,
                'count': min(limit, 20)
            }
            
            # 移除空值参数
            params = {k: v for k, v in params.items() if v is not None}
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取粉丝列表
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if response.get('error_code') != 0:
                error_msg = response.get('description', '获取粉丝列表失败')
                # 如果是权限问题，返回空结果而不是抛出异常
                if '权限' in error_msg or 'permission' in error_msg.lower():
                    logger.warning(f"抖音粉丝列表获取权限不足: {error_msg}")
                    return {
                        'followers': [],
                        'total_count': 0,
                        'has_more': False,
                        'message': '需要特殊权限才能获取粉丝列表'
                    }
                raise ValueError(f"抖音API错误: {error_msg}")
            
            data = response.get('data', {})
            fans_list = data.get('list', [])
            
            # 标准化粉丝数据
            followers = []
            for fan_data in fans_list:
                follower = {
                    'user_id': fan_data.get('open_id', ''),
                    'username': fan_data.get('nickname', ''),
                    'display_name': fan_data.get('nickname', ''),
                    'avatar_url': fan_data.get('avatar', ''),
                    'verified': fan_data.get('is_verified', False),
                    'follower_count': 0,  # 抖音不提供粉丝的粉丝数
                    'following_count': 0,  # 抖音不提供粉丝的关注数
                    'location': fan_data.get('city', ''),
                    'description': fan_data.get('signature', ''),
                    'gender': '男' if fan_data.get('gender') == 1 else ('女' if fan_data.get('gender') == 2 else '未知'),
                    'platform_specific': {
                        'country': fan_data.get('country', ''),
                        'province': fan_data.get('province', ''),
                        'language': fan_data.get('language', ''),
                        'verification_type': fan_data.get('verification_type', 0)
                    },
                    'raw_data': fan_data
                }
                followers.append(follower)
            
            result = {
                'followers': followers,
                'total_count': data.get('total', 0),
                'has_more': data.get('has_more', False),
                'next_cursor': str(data.get('cursor', 0)) if data.get('has_more') else None
            }
            
            logger.info(f"成功获取抖音用户 {user_id} 的 {len(followers)} 个粉丝")
            return result
            
        except Exception as e:
            logger.error(f"获取抖音粉丝列表失败: {e}")
            # 对于权限问题，返回空结果
            if '权限' in str(e) or 'permission' in str(e).lower():
                return {
                    'followers': [],
                    'total_count': 0,
                    'has_more': False,
                    'message': '需要特殊权限才能获取粉丝列表'
                }
            raise
    
    async def _get_video_statistics(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """获取视频统计数据"""
        try:
            # 获取用户的视频列表，计算总体统计
            videos_response = await self.get_user_posts(access_token, user_id, limit=20, offset=0)
            videos = videos_response.get('posts', [])
            
            total_likes = sum(video.get('like_count', 0) for video in videos)
            total_comments = sum(video.get('comment_count', 0) for video in videos)
            total_shares = sum(video.get('share_count', 0) for video in videos)
            total_views = sum(video.get('view_count', 0) for video in videos)
            
            return {
                'total_videos': len(videos),
                'total_likes': total_likes,
                'total_comments': total_comments,
                'total_shares': total_shares,
                'total_views': total_views
            }
            
        except Exception as e:
            logger.warning(f"获取抖音视频统计失败: {e}")
            return {
                'total_videos': 0,
                'total_likes': 0,
                'total_comments': 0,
                'total_shares': 0,
                'total_views': 0
            }
    
    def _standardize_douyin_profile(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化抖音用户资料数据"""
        return {
            'user_id': raw_data.get('open_id', raw_data.get('union_id', '')),
            'username': raw_data.get('nickname', ''),
            'display_name': raw_data.get('nickname', ''),
            'avatar_url': raw_data.get('avatar', ''),
            'bio': raw_data.get('description', raw_data.get('signature', '')),
            'verification_status': 'verified' if raw_data.get('is_verified', False) else 'unverified',
            'account_type': self._determine_douyin_account_type(raw_data),
            'location': f"{raw_data.get('country', '')} {raw_data.get('province', '')} {raw_data.get('city', '')}".strip(),
            'website_url': '',  # 抖音不提供网站链接
            'gender': '男' if raw_data.get('gender') == 1 else ('女' if raw_data.get('gender') == 2 else '未知'),
            'created_at': None,  # 抖音不提供注册时间
            'last_updated': datetime.utcnow().isoformat(),
            'platform_specific': {
                'union_id': raw_data.get('union_id', ''),
                'e_account_role': raw_data.get('e_account_role', ''),
                'avatar_larger': raw_data.get('avatar_larger', ''),
                'captcha_url': raw_data.get('captcha_url', ''),
                'desc_url': raw_data.get('desc_url', ''),
                'display_name': raw_data.get('display_name', ''),
                'error_code': raw_data.get('error_code', 0)
            },
            'raw_data': raw_data
        }
    
    def _determine_douyin_account_type(self, raw_data: Dict[str, Any]) -> str:
        """根据抖音数据确定账号类型"""
        e_account_role = raw_data.get('e_account_role', '')
        
        if 'enterprise' in e_account_role.lower() or raw_data.get('verification_type') == 1:
            return 'business'
        elif raw_data.get('is_verified', False):
            return 'creator'
        else:
            return 'personal'
    
    def _standardize_douyin_posts(self, raw_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """标准化抖音视频内容数据"""
        standardized_posts = []
        
        for video in raw_posts:
            # 提取媒体URL
            media_urls = []
            video_info = video.get('video', {})
            if video_info.get('play_addr', {}).get('url_list'):
                media_urls.extend(video_info['play_addr']['url_list'])
            
            # 提取封面图
            cover_urls = []
            if video_info.get('cover', {}).get('url_list'):
                cover_urls.extend(video_info['cover']['url_list'])
            
            # 提取音乐信息
            music_info = video.get('music', {})
            music_title = music_info.get('title', '')
            
            # 提取话题标签（从文本中解析）
            content_text = video.get('desc', '')
            hashtags = self._extract_hashtags_from_text(content_text)
            
            standardized_post = {
                'post_id': video.get('item_id', ''),
                'content': content_text,
                'created_at': datetime.fromtimestamp(video.get('create_time', 0)).isoformat() if video.get('create_time') else None,
                'like_count': video.get('statistics', {}).get('digg_count', 0),
                'comment_count': video.get('statistics', {}).get('comment_count', 0),
                'share_count': video.get('statistics', {}).get('share_count', 0),
                'view_count': video.get('statistics', {}).get('play_count', 0),
                'media_urls': media_urls,
                'cover_urls': cover_urls,
                'hashtags': hashtags,
                'mentions': [],  # 需要从文本中解析
                'duration': video_info.get('duration', 0),
                'platform_specific': {
                    'is_top': video.get('is_top', 0),
                    'item_id': video.get('item_id', ''),
                    'video_status': video.get('video_status', 0),
                    'music_title': music_title,
                    'music_author': music_info.get('author', ''),
                    'video_width': video_info.get('width', 0),
                    'video_height': video_info.get('height', 0),
                    'ratio': video_info.get('ratio', ''),
                    'download_count': video.get('statistics', {}).get('download_count', 0),
                    'forward_count': video.get('statistics', {}).get('forward_count', 0)
                },
                'raw_data': video
            }
            
            standardized_posts.append(standardized_post)
        
        return {
            'posts': standardized_posts,
            'platform': 'douyin'
        }
    
    def _extract_hashtags_from_text(self, text: str) -> List[str]:
        """从文本中提取话题标签"""
        import re
        
        # 抖音话题标签格式：#话题名称#
        hashtag_pattern = r'#([^#]+)#'
        hashtags = re.findall(hashtag_pattern, text)
        
        return [tag.strip() for tag in hashtags if tag.strip()]