"""
百家号平台适配器

实现百家号开放平台API的具体调用逻辑
处理百家号特定的数据格式和API接口
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from .base_adapter import BasePlatformAdapter

logger = logging.getLogger(__name__)


class BaijiahaoAdapter(BasePlatformAdapter):
    """
    百家号平台适配器
    
    实现百家号开放平台API的接口调用和数据处理
    支持用户信息获取、文章内容同步等功能
    """
    
    @property
    def platform_type(self) -> str:
        return "content"
    
    @property
    def supported_sync_types(self) -> List[str]:
        return ["profile", "stats", "posts"]  # 百家号不支持粉丝列表获取
    
    async def get_user_profile(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取百家号用户资料信息
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            
        Returns:
            Dict: 标准化的用户资料信息
        """
        try:
            # 构建API URL
            url = f"{self.api_base_url}/rest/2.0/cambrian/app_info"
            
            # 设置请求参数
            params = {
                'access_token': access_token
            }
            
            if user_id:
                params['app_id'] = user_id
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取用户信息
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if 'error_code' in response and response['error_code'] != 0:
                error_msg = response.get('error_msg', '获取用户信息失败')
                raise ValueError(f"百家号API错误: {error_msg}")
            
            user_data = response.get('data', response)  # 百家号的数据结构可能直接在根层级
            
            # 标准化数据格式
            profile_data = self._standardize_baijiahao_profile(user_data)
            
            logger.info(f"成功获取百家号用户 {user_id} 的资料信息")
            return profile_data
            
        except Exception as e:
            logger.error(f"获取百家号用户资料失败: {e}")
            raise
    
    async def get_user_stats(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """
        获取百家号用户统计信息
        
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
            
            # 获取粉丝统计（如果API支持）
            fan_stats = await self._get_fan_statistics(access_token, user_id)
            
            # 整合统计信息
            stats_data = {
                'follower_count': fan_stats.get('total_fans', 0),
                'following_count': 0,  # 百家号不提供关注数
                'post_count': article_stats.get('total_articles', 0),
                'like_count': article_stats.get('total_likes', 0),
                'view_count': article_stats.get('total_views', 0),
                'comment_count': article_stats.get('total_comments', 0),
                'share_count': article_stats.get('total_shares', 0),
                'last_updated': datetime.utcnow().isoformat(),
                'platform_specific': {
                    'app_id': profile_data['raw_data'].get('app_id', ''),
                    'name': profile_data['raw_data'].get('name', ''),
                    'type': profile_data['raw_data'].get('type', ''),
                    'avatar': profile_data['raw_data'].get('avatar', ''),
                    'description': profile_data['raw_data'].get('description', '')
                }
            }
            
            logger.info(f"成功获取百家号用户 {user_id} 的统计信息")
            return stats_data
            
        except Exception as e:
            logger.error(f"获取百家号用户统计失败: {e}")
            raise
    
    async def get_user_posts(self, access_token: str, user_id: str = None,
                           limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取百家号用户发布的文章内容
        
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
            limit = min(max(limit, 1), 20)  # 百家号API限制单次最多20条
            start = offset
            
            # 构建API URL
            url = f"{self.api_base_url}/rest/2.0/cambrian/article/list"
            
            # 设置请求参数
            params = {
                'access_token': access_token,
                'start': start,
                'count': limit
            }
            
            if user_id:
                params['app_id'] = user_id
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取文章列表
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if 'error_code' in response and response['error_code'] != 0:
                error_msg = response.get('error_msg', '获取文章列表失败')
                raise ValueError(f"百家号API错误: {error_msg}")
            
            data = response.get('data', {})
            articles = data.get('list', []) if isinstance(data, dict) else response.get('list', [])
            
            # 标准化数据格式
            posts_data = self._standardize_baijiahao_posts(articles)
            posts_data.update({
                'total_count': data.get('total', len(articles)) if isinstance(data, dict) else len(articles),
                'has_more': len(articles) == limit,
                'next_cursor': str(offset + limit) if len(articles) == limit else None
            })
            
            logger.info(f"成功获取百家号用户 {user_id} 的 {len(posts_data['posts'])} 篇文章")
            return posts_data
            
        except Exception as e:
            logger.error(f"获取百家号用户发布内容失败: {e}")
            raise
    
    async def get_followers(self, access_token: str, user_id: str = None,
                          limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        获取百家号粉丝列表
        
        注意：百家号开放平台目前不支持粉丝列表获取
        
        Args:
            access_token: 访问令牌
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            Dict: 空的粉丝列表（不支持）
        """
        logger.info("百家号平台不支持粉丝列表获取")
        return {
            'followers': [],
            'total_count': 0,
            'has_more': False,
            'message': '百家号平台不支持粉丝列表获取'
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
            logger.warning(f"获取百家号文章统计失败: {e}")
            return {
                'total_articles': 0,
                'total_likes': 0,
                'total_comments': 0,
                'total_shares': 0,
                'total_views': 0
            }
    
    async def _get_fan_statistics(self, access_token: str, user_id: str = None) -> Dict[str, Any]:
        """获取粉丝统计数据"""
        try:
            # 构建API URL（如果API支持）
            url = f"{self.api_base_url}/rest/2.0/cambrian/fans/portrait"
            
            # 设置请求参数
            params = {
                'access_token': access_token
            }
            
            if user_id:
                params['app_id'] = user_id
            
            # 设置请求头
            headers = {
                'Content-Type': 'application/json'
            }
            
            # 获取粉丝统计
            response = await self._make_request('GET', url, headers=headers, params=params)
            
            # 检查响应数据
            if 'error_code' in response and response['error_code'] != 0:
                logger.warning(f"获取百家号粉丝统计失败: {response.get('error_msg')}")
                return {'total_fans': 0}
            
            data = response.get('data', {})
            return {
                'total_fans': data.get('total_fans', 0)
            }
            
        except Exception as e:
            logger.warning(f"获取百家号粉丝统计失败: {e}")
            return {'total_fans': 0}
    
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
            url = f"{self.api_base_url}/rest/2.0/cambrian/article/detail"
            
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
            if 'error_code' in response and response['error_code'] != 0:
                error_msg = response.get('error_msg', '获取文章详情失败')
                raise ValueError(f"百家号API错误: {error_msg}")
            
            article_data = response.get('data', response)
            
            logger.info(f"成功获取百家号文章 {article_id} 的详细信息")
            return article_data
            
        except Exception as e:
            logger.error(f"获取百家号文章详情失败: {e}")
            raise
    
    def _standardize_baijiahao_profile(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化百家号用户资料数据"""
        return {
            'user_id': raw_data.get('app_id', ''),
            'username': raw_data.get('name', ''),
            'display_name': raw_data.get('name', ''),
            'avatar_url': raw_data.get('avatar', ''),
            'bio': raw_data.get('description', ''),
            'verification_status': 'verified' if raw_data.get('is_verified', False) else 'unverified',
            'account_type': self._determine_baijiahao_account_type(raw_data),
            'location': '',  # 百家号不提供地理位置信息
            'website_url': '',  # 百家号不提供网站链接
            'gender': '未知',  # 百家号不提供性别信息
            'created_at': raw_data.get('create_time'),  # 创建时间
            'last_updated': datetime.utcnow().isoformat(),
            'platform_specific': {
                'app_id': raw_data.get('app_id', ''),
                'name': raw_data.get('name', ''),
                'type': raw_data.get('type', ''),
                'avatar': raw_data.get('avatar', ''),
                'description': raw_data.get('description', ''),
                'province': raw_data.get('province', ''),
                'city': raw_data.get('city', ''),
                'create_time': raw_data.get('create_time', ''),
                'status': raw_data.get('status', ''),
                'fans_num': raw_data.get('fans_num', 0),
                'follow_num': raw_data.get('follow_num', 0)
            },
            'raw_data': raw_data
        }
    
    def _determine_baijiahao_account_type(self, raw_data: Dict[str, Any]) -> str:
        """根据百家号数据确定账号类型"""
        account_type = raw_data.get('type', '')
        
        if 'enterprise' in account_type.lower() or 'company' in account_type.lower():
            return 'business'  # 企业账号
        elif 'media' in account_type.lower() or 'creator' in account_type.lower():
            return 'creator'  # 创作者/媒体账号
        else:
            return 'personal'  # 个人账号
    
    def _standardize_baijiahao_posts(self, raw_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """标准化百家号文章内容数据"""
        standardized_posts = []
        
        for article in raw_posts:
            # 提取媒体URL
            media_urls = []
            if article.get('image_list'):
                media_urls.extend(article['image_list'])
            if article.get('cover_images'):
                media_urls.extend(article['cover_images'])
            
            # 提取标签
            tags = article.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            
            # 提取关键词作为hashtags
            keywords = article.get('keywords', [])
            if isinstance(keywords, str):
                keywords = [keywords]
            
            all_tags = list(set(tags + keywords))  # 去重合并
            
            # 确定内容类型
            article_type = article.get('type', 'article')  # 默认为文章
            
            standardized_post = {
                'post_id': article.get('article_id', article.get('id', '')),
                'content': article.get('content', article.get('summary', '')),
                'title': article.get('title', ''),
                'created_at': article.get('publish_time') or article.get('create_time'),
                'like_count': article.get('like_num', 0),
                'comment_count': article.get('comment_num', 0),
                'share_count': article.get('share_num', 0),
                'view_count': article.get('read_num', 0),
                'media_urls': media_urls,
                'hashtags': all_tags,
                'mentions': [],  # 百家号不提供用户提及信息
                'url': article.get('article_url', ''),
                'content_type': article_type,
                'platform_specific': {
                    'article_id': article.get('article_id', ''),
                    'app_id': article.get('app_id', ''),
                    'status': article.get('status', ''),
                    'reject_reason': article.get('reject_reason', ''),
                    'is_original': article.get('is_original', False),
                    'category': article.get('category', ''),
                    'subcategory': article.get('subcategory', ''),
                    'summary': article.get('summary', ''),
                    'cover_images': article.get('cover_images', []),
                    'video_duration': article.get('video_duration', 0),
                    'video_poster': article.get('video_poster', ''),
                    'source_url': article.get('source_url', ''),
                    'publish_status': article.get('publish_status', ''),
                    'failed_reason': article.get('failed_reason', '')
                },
                'raw_data': article
            }
            
            standardized_posts.append(standardized_post)
        
        return {
            'posts': standardized_posts,
            'platform': 'baijiahao'
        }