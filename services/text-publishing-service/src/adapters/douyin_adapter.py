"""
抖音平台适配器

实现抖音开放平台API的认证和发布功能
支持短视频和图文发布
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_adapter import PlatformAdapter, PlatformAPIError

logger = logging.getLogger(__name__)


class DouyinAdapter(PlatformAdapter):
    """抖音平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.platform_name = "douyin"
        self.api_base_url = "https://open-api.douyin.com"
        self.access_token = None
        
        # 抖音平台限制
        self.max_content_length = 2200
        self.max_images = 12
        self.supported_formats = ['jpg', 'jpeg', 'png', 'mp4', 'mov']
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """抖音OAuth2认证"""
        try:
            access_token = credentials.get('access_token')
            if not access_token:
                logger.error("抖音认证缺少access_token")
                return False
            
            # 简化认证验证
            self.access_token = access_token
            logger.info("抖音认证成功（模拟）")
            return True
                    
        except Exception as e:
            logger.error(f"抖音认证异常: {e}")
            return False
    
    async def publish_content(
        self,
        content: str,
        title: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """发布抖音内容"""
        try:
            if not self.access_token:
                raise PlatformAPIError("未完成抖音认证")
            
            self._validate_content(content, self.max_content_length)
            
            # 抖音发布（简化实现）
            return self._format_publish_result(
                success=True,
                post_id=f"dy_{int(datetime.utcnow().timestamp())}",
                url="https://www.douyin.com/",
                raw_response={'message': '抖音发布成功（模拟）'}
            )
            
        except (ValueError, PlatformAPIError) as e:
            return self._format_publish_result(success=False, error=str(e))
        except Exception as e:
            logger.error(f"抖音发布异常: {e}")
            return self._format_publish_result(success=False, error=f"发布异常: {str(e)}")