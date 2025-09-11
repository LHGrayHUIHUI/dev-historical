"""
今日头条平台适配器
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_adapter import PlatformAdapter, PlatformAPIError

logger = logging.getLogger(__name__)


class ToutiaoAdapter(PlatformAdapter):
    """今日头条平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.platform_name = "toutiao"
        self.api_base_url = "https://mp.toutiao.com/api"
        self.access_token = None
        
        self.max_content_length = 5000
        self.max_images = 20
        self.supported_formats = ['jpg', 'jpeg', 'png', 'gif', 'mp4']
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """今日头条认证"""
        self.access_token = credentials.get('access_token', 'mock_token')
        logger.info("今日头条认证成功（模拟）")
        return True
    
    async def publish_content(
        self,
        content: str,
        title: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发布今日头条内容"""
        try:
            if not self.access_token:
                raise PlatformAPIError("未完成今日头条认证")
            
            self._validate_content(content, self.max_content_length)
            
            return self._format_publish_result(
                success=True,
                post_id=f"tt_{int(datetime.utcnow().timestamp())}",
                url="https://www.toutiao.com/",
                raw_response={'message': '今日头条发布成功（模拟）'}
            )
            
        except (ValueError, PlatformAPIError) as e:
            return self._format_publish_result(success=False, error=str(e))
        except Exception as e:
            return self._format_publish_result(success=False, error=f"发布异常: {str(e)}")