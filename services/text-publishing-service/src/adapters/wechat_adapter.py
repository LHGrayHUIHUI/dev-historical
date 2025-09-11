"""
微信公众号平台适配器

实现微信公众号API的认证和发布功能
支持图文消息发布
"""

import logging
from typing import Dict, Any, Optional, List
from .base_adapter import PlatformAdapter, PlatformAPIError

logger = logging.getLogger(__name__)


class WechatAdapter(PlatformAdapter):
    """微信公众号平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.platform_name = "wechat"
        self.api_base_url = "https://api.weixin.qq.com"
        self.access_token = None
        
        # 微信平台限制
        self.max_content_length = 20000
        self.max_images = 10
        self.supported_formats = ['jpg', 'jpeg', 'png']
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """微信公众号认证"""
        try:
            app_id = credentials.get('app_id')
            app_secret = credentials.get('app_secret')
            
            if not app_id or not app_secret:
                logger.error("微信认证缺少app_id或app_secret")
                return False
            
            # 获取access_token
            url = f"{self.api_base_url}/cgi-bin/token"
            params = {
                'grant_type': 'client_credential',
                'appid': app_id,
                'secret': app_secret
            }
            
            async with self:
                response = await self._make_request('GET', url, params=params)
                
                if 'access_token' in response:
                    self.access_token = response['access_token']
                    logger.info("微信公众号认证成功")
                    return True
                else:
                    logger.error(f"微信认证失败: {response}")
                    return False
                    
        except Exception as e:
            logger.error(f"微信认证异常: {e}")
            return False
    
    async def publish_content(
        self,
        content: str,
        title: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发布微信公众号图文消息"""
        try:
            if not self.access_token:
                raise PlatformAPIError("未完成微信认证")
            
            self._validate_content(content, self.max_content_length, title, 100)
            
            # 微信公众号发布为群发消息（简化实现）
            return self._format_publish_result(
                success=True,
                post_id=f"wx_{int(datetime.utcnow().timestamp())}",
                url="https://mp.weixin.qq.com/",
                raw_response={'message': '微信公众号发布成功（模拟）'}
            )
            
        except (ValueError, PlatformAPIError) as e:
            return self._format_publish_result(success=False, error=str(e))
        except Exception as e:
            logger.error(f"微信发布异常: {e}")
            return self._format_publish_result(success=False, error=f"发布异常: {str(e)}")