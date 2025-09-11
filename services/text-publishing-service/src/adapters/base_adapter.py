"""
平台适配器基类

定义各平台发布接口的统一规范
提供通用的认证、发布和错误处理机制
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class PlatformAdapter(ABC):
    """
    平台适配器基类
    
    定义各平台发布接口的统一规范
    子类需要实现具体的平台逻辑
    """
    
    def __init__(self):
        self.platform_name = ""
        self.api_base_url = ""
        self.timeout = 30
        self.max_retries = 3
        self.headers = {
            'User-Agent': 'Text-Publishing-Service/1.0',
            'Content-Type': 'application/json'
        }
        self.session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.aclose()
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        平台认证
        
        Args:
            credentials: 认证凭据
            
        Returns:
            bool: 认证是否成功
        """
        pass
    
    @abstractmethod
    async def publish_content(
        self, 
        content: str,
        title: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        发布内容
        
        Args:
            content: 内容文本
            title: 标题
            media_urls: 媒体文件URL列表
            config: 发布配置
            
        Returns:
            Dict[str, Any]: 发布结果
        """
        pass
    
    async def get_post_metrics(self, post_id: str) -> Dict[str, Any]:
        """
        获取帖子数据指标
        
        Args:
            post_id: 帖子ID
            
        Returns:
            Dict[str, Any]: 指标数据
        """
        return {
            'post_id': post_id,
            'views': 0,
            'likes': 0,
            'comments': 0,
            'shares': 0,
            'engagement_rate': 0.0
        }
    
    async def delete_post(self, post_id: str) -> bool:
        """
        删除帖子
        
        Args:
            post_id: 帖子ID
            
        Returns:
            bool: 是否成功删除
        """
        logger.warning(f"平台 {self.platform_name} 不支持删除帖子")
        return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        发起HTTP请求
        
        Args:
            method: 请求方法
            url: 请求URL
            data: 请求数据
            params: 查询参数
            headers: 请求头
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        if not self.session:
            raise RuntimeError("会话未初始化，请使用 async with 语句")
        
        try:
            request_headers = {**self.headers}
            if headers:
                request_headers.update(headers)
            
            response = await self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=request_headers
            )
            
            response.raise_for_status()
            
            # 记录请求日志
            logger.debug(f"{method} {url} - {response.status_code}")
            
            return response.json() if response.content else {}
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP错误 {e.response.status_code}: {e.response.text}")
            raise PlatformAPIError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            logger.error(f"请求错误: {e}")
            raise PlatformAPIError(f"请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise PlatformAPIError(f"未知错误: {str(e)}")
    
    def _validate_content(
        self, 
        content: str, 
        max_length: int,
        title: Optional[str] = None,
        max_title_length: Optional[int] = None
    ):
        """
        验证内容格式
        
        Args:
            content: 内容
            max_length: 最大长度
            title: 标题
            max_title_length: 最大标题长度
        """
        if not content or not content.strip():
            raise ValueError("内容不能为空")
        
        if len(content) > max_length:
            raise ValueError(f"内容长度超过限制 ({max_length} 字符)")
        
        if title and max_title_length and len(title) > max_title_length:
            raise ValueError(f"标题长度超过限制 ({max_title_length} 字符)")
    
    def _validate_media(
        self, 
        media_urls: List[str], 
        max_count: int,
        supported_formats: List[str]
    ):
        """
        验证媒体文件
        
        Args:
            media_urls: 媒体URL列表
            max_count: 最大数量
            supported_formats: 支持的格式
        """
        if len(media_urls) > max_count:
            raise ValueError(f"媒体文件数量超过限制 ({max_count})")
        
        for url in media_urls:
            # 简单的格式检查
            file_ext = url.split('.')[-1].lower()
            if file_ext not in supported_formats:
                raise ValueError(f"不支持的文件格式: {file_ext}")
    
    def _format_publish_result(
        self,
        success: bool,
        post_id: Optional[str] = None,
        url: Optional[str] = None,
        error: Optional[str] = None,
        raw_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        格式化发布结果
        
        Args:
            success: 是否成功
            post_id: 帖子ID
            url: 发布URL
            error: 错误信息
            raw_response: 原始响应
            
        Returns:
            Dict[str, Any]: 标准化结果
        """
        result = {
            'success': success,
            'platform': self.platform_name,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if success:
            result.update({
                'post_id': post_id,
                'url': url,
                'message': '发布成功'
            })
        else:
            result.update({
                'error': error or '发布失败',
                'message': '发布失败'
            })
        
        if raw_response:
            result['raw_response'] = raw_response
        
        return result


class PlatformAPIError(Exception):
    """平台API错误"""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code