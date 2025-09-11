"""
今日头条平台适配器

实现今日头条开放平台API的具体调用逻辑  
处理头条特定的数据格式和API接口
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from .base_adapter import BasePlatformAdapter

logger = logging.getLogger(__name__)


class ToutiaoAdapter(BasePlatformAdapter):
    """
    今日头条平台适配器
    
    实现今日头条开放平台API的接口调用和数据处理
    支持用户信息获取、文章内容同步、粉丝管理等功能
    """
    
    @property
    def platform_type(self) -> str:
        return "news"
    
    @property
    def supported_sync_types(self) -> List[str]:
        return ["profile", "stats", "posts"]  # 头条不支持粉丝列表获取
    
    async def get_user_profile(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取今日头条用户资料信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            
        Returns:
            Dict: 标准化的用户资料信息
        """
        try:
            # 构建API URL
            url = f"{self.api_base_url}/oauth/userinfo/"
            
            # 设置请求参数
            params = {
                'access_token': access_token
            }
            
            if user_id:
                params['open_id'] = user_id
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取用户信息
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if response.get('error_code') != 0:
                error_msg = response.get('description', '获取用户信息失败')
                raise ValueError(f"头条API错误: {error_msg}")
            
            user_data = response.get('data', {})
            
            # 标准化数据格式
            profile_data = self._standardize_toutiao_profile(user_data)
            
            logger.info(f"成功获取头条用户 {user_id} 的资料信息")
            return profile_data
            
        except Exception as e:
            logger.error(f"获取头条用户资料失败: {e}")
            raise
    
    async def get_user_stats(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取今日头条用户统计信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            
        Returns:
            Dict: 标准化的用户统计信息
        """
        try:
            # 先获取用户基本信息
            profile_data = await self.get_user_profile(access_token, user_id)
            
            # 获取文章统计数据
            article_stats = await self._get_article_statistics(access_token, user_id)
            
            # 整合统计信息
            stats_data = {
                'follower_count': profile_data['raw_data'].get('fans_count', 0),
                'following_count': 0,  # 头条不提供关注数
                'post_count': article_stats.get('total_articles', 0),
                'like_count': article_stats.get('total_likes', 0),
                'view_count': article_stats.get('total_views', 0),
                'comment_count': article_stats.get('total_comments', 0),
                'share_count': article_stats.get('total_shares', 0),
                'last_updated': datetime.utcnow().isoformat(),
                'platform_specific': {
                    'media_id': profile_data['raw_data'].get('media_id', ''),
                    'approved_time': profile_data['raw_data'].get('approved_time', ''),
                    'avatar_url_large': profile_data['raw_data'].get('avatar_url_large', ''),
                    'is_media': profile_data['raw_data'].get('is_media', False)
                }
            }
            
            logger.info(f"成功获取头条用户 {user_id} 的统计信息")
            return stats_data
            
        except Exception as e:
            logger.error(f"获取头条用户统计失败: {e}")
            raise
    
    async def get_user_posts(self, access_token: str, user_id: str = None,
                           limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取今日头条用户发布的文章内容
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 标准化的用户发布内容
        """
        try:
            # 参数验证
            limit = min(max(limit, 1), 20)  # 头条API限制单次最多20条
            page = offset // limit + 1
            
            # 构建API URL
            url = f"{self.api_base_url}/article/list/"
            
            # 设置请求参数
            params = {
                'access_token': access_token,
                'page': page,
                'page_size': limit
            }
            
            if user_id:
                params['open_id'] = user_id
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取文章列表
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if response.get('error_code') != 0:
                error_msg = response.get('description', '获取文章列表失败')
                raise ValueError(f"头条API错误: {error_msg}")
            
            data = response.get('data', {})
            articles = data.get('list', [])
            
            # 标准化数据格式
            posts_data = self._standardize_toutiao_posts(articles)
            posts_data.update({
                'total_count': data.get('total', 0),
                'has_more': len(articles) == limit and data.get('total', 0) > offset + limit,
                'next_cursor': str(page + 1) if len(articles) == limit else None
            })
            
            logger.info(f"成功获取头条用户 {user_id} 的 {len(posts_data['posts'])} 篇文章")
            return posts_data
            
        except Exception as e:
            logger.error(f"获取头条用户发布内容失败: {e}")
            raise
    
    async def get_followers(self, access_token: str, user_id: str = None,
                          limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取今日头条粉丝列表
        
        注意：今日头条开放平台目前不支持粉丝列表获取
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 空的粉丝列表（不支持）
        """
        logger.info("今日头条平台不支持粉丝列表获取")
        return {
            'followers': [],
            'total_count': 0,
            'has_more': False,
            'message': '今日头条平台不支持粉丝列表获取'
        }
    
    async def _get_article_statistics(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """获取文章统计数据"""
        try:
            # 获取用户的文章列表，计算总体统计
            articles_response = await self.get_user_posts(access_token, user_id, limit=20, offset=0)
            articles = articles_response.get('posts', [])
            
            total_likes = sum(article.get('like_count', 0) for article in articles)
            total_comments = sum(article.get('comment_count', 0) for article in articles)
            total_shares = sum(article.get('share_count', 0) for article in articles)
            total_views = sum(article.get('view_count', 0) for article in articles)
            
            return {
                'total_articles': len(articles),
                'total_likes': total_likes,
                'total_comments': total_comments,
                'total_shares': total_shares,
                'total_views': total_views
            }
            
        except Exception as e:
            logger.warning(f"获取头条文章统计失败: {e}")
            return {
                'total_articles': 0,
                'total_likes': 0,
                'total_comments': 0,
                'total_shares': 0,
                'total_views': 0
            }
    
    async def get_article_detail(self, access_token: str, article_id: str) -> Dict[str, Any]:
        """
        获取文章详细信息
        
        Args:
            access_token: 访问令牌
            article_id: 文章ID
            
        Returns:
            Dict: 文章详细信息
        """
        try:
            # 构建API URL
            url = f"{self.api_base_url}/article/detail/"
            
            # 设置请求参数
            params = {
                'access_token': access_token,
                'article_id': article_id
            }
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取文章详情
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if response.get('error_code') != 0:
                error_msg = response.get('description', '获取文章详情失败')
                raise ValueError(f"头条API错误: {error_msg}")
            
            article_data = response.get('data', {})
            
            logger.info(f"成功获取头条文章 {article_id} 的详细信息")
            return article_data
            
        except Exception as e:
            logger.error(f"获取头条文章详情失败: {e}")
            raise
    
    def _standardize_toutiao_profile(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化今日头条用户资料数据"""
        return {
            'user_id': raw_data.get('open_id', ''),
            'username': raw_data.get('display_name', raw_data.get('nickname', '')),
            'display_name': raw_data.get('display_name', raw_data.get('nickname', '')),
            'avatar_url': raw_data.get('avatar_url', ''),
            'bio': raw_data.get('description', ''),
            'verification_status': 'verified' if raw_data.get('is_verified', False) else 'unverified',
            'account_type': self._determine_toutiao_account_type(raw_data),
            'location': '',  # 头条不提供地理位置信息
            'website_url': '',  # 头条不提供网站链接
            'gender': '未知',  # 头条不提供性别信息
            'created_at': raw_data.get('approved_time'),  # 使用审核通过时间作为创建时间
            'last_updated': datetime.utcnow().isoformat(),
            'platform_specific': {
                'media_id': raw_data.get('media_id', ''),
                'union_id': raw_data.get('union_id', ''),
                'client_key': raw_data.get('client_key', ''),
                'avatar_url_large': raw_data.get('avatar_url_large', ''),
                'is_media': raw_data.get('is_media', False),
                'approved_time': raw_data.get('approved_time', ''),
                'scope': raw_data.get('scope', [])
            },
            'raw_data': raw_data
        }
    
    def _determine_toutiao_account_type(self, raw_data: Dict[str, Any]) -> str:
        """根据头条数据确定账号类型"""
        if raw_data.get('is_media', False):
            return 'creator'  # 媒体账号
        elif raw_data.get('is_verified', False):
            return 'business'  # 认证账号
        else:
            return 'personal'  # 个人账号
    
    def _standardize_toutiao_posts(self, raw_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """标准化今日头条文章内容数据"""
        standardized_posts = []
        
        for article in raw_posts:
            # 提取媒体URL
            media_urls = []
            if article.get('image_list'):
                media_urls.extend(article['image_list'])
            if article.get('thumb_image_list'):
                media_urls.extend(article['thumb_image_list'])
            
            # 提取标签
            tags = article.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            
            # 确定内容类型
            article_type = article.get('item_type', 1)  # 1:文章, 2:图片, 3:视频
            content_type_map = {1: 'article', 2: 'image', 3: 'video'}
            content_type = content_type_map.get(article_type, 'article')
            
            standardized_post = {
                'post_id': article.get('item_id', ''),
                'content': article.get('content', article.get('abstract', '')),
                'title': article.get('title', ''),
                'created_at': datetime.fromtimestamp(article.get('publish_time', 0)).isoformat() if article.get('publish_time') else None,
                'like_count': article.get('digg_count', 0),
                'comment_count': article.get('comment_count', 0),
                'share_count': article.get('share_count', 0),
                'view_count': article.get('read_count', 0),
                'media_urls': media_urls,
                'hashtags': tags,
                'mentions': [],  # 头条不提供用户提及信息
                'url': article.get('article_url', ''),
                'content_type': content_type,
                'platform_specific': {
                    'item_type': article.get('item_type', 1),
                    'media_id': article.get('media_id', ''),
                    'article_id': article.get('article_id', ''),
                    'status': article.get('status', ''),
                    'reject_reason': article.get('reject_reason', ''),
                    'is_original': article.get('is_original', False),
                    'video_id': article.get('video_id', ''),
                    'video_poster_url': article.get('video_poster_url', ''),
                    'video_duration': article.get('video_duration', 0),
                    'abstract': article.get('abstract', ''),
                    'source_url': article.get('source_url', ''),
                    'category': article.get('category', ''),
                    'label': article.get('label', ''),
                    'keywords': article.get('keywords', [])
                },
                'raw_data': article
            }
            
            standardized_posts.append(standardized_post)
        
        return {
            'posts': standardized_posts,
            'platform': 'toutiao'
        }