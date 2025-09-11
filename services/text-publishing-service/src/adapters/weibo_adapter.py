"""
微博平台适配器

实现微博API的认证和发布功能
支持文本、图片发布
"""

import logging
from typing import Dict, Any, Optional, List
from .base_adapter import PlatformAdapter, PlatformAPIError

logger = logging.getLogger(__name__)


class WeiboAdapter(PlatformAdapter):
    """
    微博平台适配器
    
    实现微博API的具体发布逻辑
    """
    
    def __init__(self):
        super().__init__()
        self.platform_name = "weibo"
        self.api_base_url = "https://api.weibo.com/2"
        self.access_token = None
        
        # 微博平台限制
        self.max_content_length = 140
        self.max_images = 9
        self.supported_formats = ['jpg', 'jpeg', 'png', 'gif']
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        微博OAuth2认证
        
        Args:
            credentials: 包含access_token的认证信息
            
        Returns:
            bool: 认证是否成功
        """
        try:
            access_token = credentials.get('access_token')
            if not access_token:
                logger.error("微博认证缺少access_token")
                return False
            
            # 验证access_token有效性
            verify_url = f"{self.api_base_url}/account/get_uid.json"
            params = {'access_token': access_token}
            
            async with self:
                response = await self._make_request(
                    method='GET',
                    url=verify_url,
                    params=params
                )
                
                if 'uid' in response:
                    self.access_token = access_token
                    logger.info(f"微博认证成功，用户ID: {response['uid']}")
                    return True
                else:
                    logger.error("微博认证失败：未获取到用户ID")
                    return False
                    
        except PlatformAPIError as e:
            logger.error(f"微博认证API错误: {e}")
            return False
        except Exception as e:
            logger.error(f"微博认证异常: {e}")
            return False
    
    async def publish_content(
        self,
        content: str,
        title: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        发布微博内容
        
        Args:
            content: 微博文本内容
            title: 标题（微博不使用）
            media_urls: 图片URL列表
            config: 发布配置
            
        Returns:
            Dict[str, Any]: 发布结果
        """
        try:
            # 验证认证状态
            if not self.access_token:
                raise PlatformAPIError("未完成微博认证")
            
            # 验证内容
            self._validate_content(content, self.max_content_length)
            
            # 验证媒体文件
            if media_urls:
                self._validate_media(media_urls, self.max_images, self.supported_formats)
            
            async with self:
                if media_urls:
                    # 带图片的微博
                    return await self._publish_with_images(content, media_urls, config)
                else:
                    # 纯文本微博
                    return await self._publish_text_only(content, config)
                    
        except (ValueError, PlatformAPIError) as e:
            return self._format_publish_result(
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"微博发布异常: {e}")
            return self._format_publish_result(
                success=False,
                error=f"发布异常: {str(e)}"
            )
    
    async def _publish_text_only(
        self, 
        content: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        发布纯文本微博
        
        Args:
            content: 文本内容
            config: 发布配置
            
        Returns:
            Dict[str, Any]: 发布结果
        """
        url = f"{self.api_base_url}/statuses/update.json"
        
        data = {
            'access_token': self.access_token,
            'status': content
        }
        
        # 处理发布配置
        if config:
            # 设置可见性
            visibility = config.get('visibility', 'public')
            if visibility == 'private':
                data['visible'] = 0  # 仅自己可见
            elif visibility == 'friends':
                data['visible'] = 1  # 好友可见
            
            # 设置地理位置
            if 'location' in config:
                data['lat'] = config['location'].get('lat')
                data['long'] = config['location'].get('lng')
        
        response = await self._make_request(
            method='POST',
            url=url,
            data=data
        )
        
        # 解析响应
        if 'id' in response:
            post_url = f"https://weibo.com/{response.get('user', {}).get('id', '')}/{response['id']}"
            return self._format_publish_result(
                success=True,
                post_id=str(response['id']),
                url=post_url,
                raw_response=response
            )
        else:
            return self._format_publish_result(
                success=False,
                error="微博发布失败，未返回有效ID",
                raw_response=response
            )
    
    async def _publish_with_images(
        self, 
        content: str, 
        media_urls: List[str], 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        发布带图片的微博
        
        Args:
            content: 文本内容
            media_urls: 图片URL列表
            config: 发布配置
            
        Returns:
            Dict[str, Any]: 发布结果
        """
        # 微博API需要先上传图片，再发布微博
        # 这里简化处理，假设图片已经可以通过URL访问
        
        url = f"{self.api_base_url}/statuses/upload_url_text.json"
        
        data = {
            'access_token': self.access_token,
            'status': content,
            'url': media_urls[0]  # 微博API一次只能传一张图片
        }
        
        # 处理发布配置
        if config:
            visibility = config.get('visibility', 'public')
            if visibility == 'private':
                data['visible'] = 0
            elif visibility == 'friends':
                data['visible'] = 1
        
        response = await self._make_request(
            method='POST',
            url=url,
            data=data
        )
        
        # 解析响应
        if 'id' in response:
            post_url = f"https://weibo.com/{response.get('user', {}).get('id', '')}/{response['id']}"
            return self._format_publish_result(
                success=True,
                post_id=str(response['id']),
                url=post_url,
                raw_response=response
            )
        else:
            return self._format_publish_result(
                success=False,
                error="微博图片发布失败",
                raw_response=response
            )
    
    async def get_post_metrics(self, post_id: str) -> Dict[str, Any]:
        """
        获取微博数据指标
        
        Args:
            post_id: 微博ID
            
        Returns:
            Dict[str, Any]: 指标数据
        """
        try:
            if not self.access_token:
                return super().get_post_metrics(post_id)
            
            url = f"{self.api_base_url}/statuses/show.json"
            params = {
                'access_token': self.access_token,
                'id': post_id
            }
            
            async with self:
                response = await self._make_request(
                    method='GET',
                    url=url,
                    params=params
                )
                
                return {
                    'post_id': post_id,
                    'views': response.get('attitudes_count', 0),  # 点赞数作为浏览量参考
                    'likes': response.get('attitudes_count', 0),
                    'comments': response.get('comments_count', 0),
                    'shares': response.get('reposts_count', 0),
                    'engagement_rate': self._calculate_engagement_rate(response),
                    'created_at': response.get('created_at'),
                    'source': response.get('source')
                }
                
        except Exception as e:
            logger.error(f"获取微博指标失败: {e}")
            return super().get_post_metrics(post_id)
    
    def _calculate_engagement_rate(self, post_data: Dict[str, Any]) -> float:
        """
        计算互动率
        
        Args:
            post_data: 微博数据
            
        Returns:
            float: 互动率
        """
        likes = post_data.get('attitudes_count', 0)
        comments = post_data.get('comments_count', 0)
        shares = post_data.get('reposts_count', 0)
        
        # 假设粉丝数作为基数
        followers = post_data.get('user', {}).get('followers_count', 1)
        
        if followers <= 0:
            return 0.0
        
        engagement = (likes + comments + shares) / followers * 100
        return round(engagement, 2)
    
    async def delete_post(self, post_id: str) -> bool:
        """
        删除微博
        
        Args:
            post_id: 微博ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            if not self.access_token:
                return False
            
            url = f"{self.api_base_url}/statuses/destroy.json"
            data = {
                'access_token': self.access_token,
                'id': post_id
            }
            
            async with self:
                response = await self._make_request(
                    method='POST',
                    url=url,
                    data=data
                )
                
                return 'id' in response
                
        except Exception as e:
            logger.error(f"删除微博失败: {e}")
            return False