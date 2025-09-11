"""
Storage Service客户端 - Storage Service Client

用于AI模型服务与Storage Service通信的HTTP客户端
实现模型配置的获取、创建、更新和删除操作

核心功能:
1. AI模型配置CRUD操作
2. 系统提示语管理
3. 模型状态查询和统计
4. 错误处理和重试机制
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from ..config.settings import get_settings


logger = logging.getLogger(__name__)


class StorageServiceError(Exception):
    """Storage Service客户端错误"""
    pass


class AIModelConfigResponse(BaseModel):
    """AI模型配置响应模型"""
    id: str
    alias: str
    provider: str
    model_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    has_api_key: bool
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4096
    context_window: Optional[int] = None
    default_temperature: float = 0.7
    default_top_p: Optional[float] = None
    is_local: bool = False
    is_streaming_supported: bool = True
    # 多模态支持
    multimodal_capabilities: Dict[str, Any] = {}
    status: str
    priority: int = 0
    tags: Optional[Dict[str, Any]] = None
    last_used_at: Optional[str] = None
    usage_count: int = 0
    error_count: int = 0
    request_timeout: Optional[int] = None
    max_retries: int = 3
    rate_limit_per_minute: Optional[int] = None
    created_at: str
    updated_at: str


class AIModelConfigRequest(BaseModel):
    """AI模型配置请求模型"""
    alias: str = Field(..., description="模型别名")
    provider: str = Field(..., description="AI提供商")
    model_name: str = Field(..., description="模型名称")
    display_name: Optional[str] = Field(None, description="显示名称")
    description: Optional[str] = Field(None, description="描述")
    api_key: Optional[str] = Field(None, description="API密钥")
    api_base: Optional[str] = Field(None, description="API基础URL")
    api_version: Optional[str] = Field(None, description="API版本")
    max_tokens: int = Field(4096, description="最大token数")
    context_window: Optional[int] = Field(None, description="上下文窗口")
    default_temperature: float = Field(0.7, description="默认温度")
    default_top_p: Optional[float] = Field(None, description="默认top_p")
    is_local: bool = Field(False, description="是否本地模型")
    is_streaming_supported: bool = Field(True, description="是否支持流式")
    # 多模态支持
    supports_files: bool = Field(False, description="支持文件")
    supports_images: bool = Field(False, description="支持图片")
    supports_videos: bool = Field(False, description="支持视频")
    supports_audio: bool = Field(False, description="支持音频")
    max_file_size_mb: Optional[int] = Field(None, description="最大文件大小MB")
    # 其他配置
    priority: int = Field(0, description="优先级")
    tags: Optional[Dict[str, Any]] = Field(None, description="标签")
    request_timeout: Optional[int] = Field(None, description="请求超时")
    max_retries: int = Field(3, description="最大重试次数")
    rate_limit_per_minute: Optional[int] = Field(None, description="速率限制")


class StorageServiceClient:
    """
    Storage Service HTTP客户端
    
    提供与Storage Service通信的接口，用于管理AI模型配置
    支持异步操作、错误处理和重试机制
    """
    
    def __init__(self):
        """初始化Storage Service客户端"""
        self.settings = get_settings()
        self.base_url = self.settings.storage_service_url
        self.timeout = self.settings.storage_service_timeout
        self.max_retries = 3
        
        # HTTP客户端配置
        self.client_config = {
            "timeout": httpx.Timeout(self.timeout),
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> Dict[str, Any]:
        """
        执行HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            params: 查询参数
            retries: 重试次数
            
        Returns:
            响应数据
            
        Raises:
            StorageServiceError: 请求失败时抛出
        """
        url = f"{self.base_url}{endpoint}"
        self._logger.debug(f"Making {method} request to {url}")
        
        for attempt in range(retries + 1):
            try:
                # 简化客户端配置以避免连接问题
                timeout = httpx.Timeout(self.timeout)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 201:
                        return response.json()
                    elif response.status_code == 404:
                        raise StorageServiceError(f"资源未找到: {endpoint}")
                    elif response.status_code == 422:
                        error_data = response.json()
                        raise StorageServiceError(f"请求验证失败: {error_data}")
                    else:
                        response.raise_for_status()
                        
            except httpx.ConnectError as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"Storage Service连接失败，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise StorageServiceError(f"无法连接到Storage Service: {e}")
            except httpx.TimeoutException as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"Storage Service请求超时，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise StorageServiceError(f"Storage Service请求超时: {e}")
            except Exception as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"Storage Service请求失败，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise StorageServiceError(f"Storage Service请求失败: {e}")
        
        raise StorageServiceError("达到最大重试次数")
    
    async def get_all_models(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """
        获取所有AI模型配置
        
        Args:
            skip: 跳过条数
            limit: 限制条数
            
        Returns:
            包含模型列表的字典
        """
        try:
            params = {"skip": skip, "limit": limit}
            response = await self._make_request("GET", "/api/v1/ai-models/configs", params=params)
            return response
        except Exception as e:
            self._logger.error(f"获取模型列表失败: {e}")
            raise
    
    async def get_model_by_alias(self, alias: str) -> AIModelConfigResponse:
        """
        根据别名获取模型配置
        
        Args:
            alias: 模型别名
            
        Returns:
            模型配置
        """
        try:
            response = await self._make_request("GET", f"/api/v1/ai-models/configs/by-alias/{alias}")
            return AIModelConfigResponse(**response)
        except Exception as e:
            self._logger.error(f"获取模型配置失败 (alias={alias}): {e}")
            raise
    
    async def create_model_config(self, config_data: AIModelConfigRequest) -> AIModelConfigResponse:
        """
        创建模型配置
        
        Args:
            config_data: 模型配置数据
            
        Returns:
            创建的模型配置
        """
        try:
            data = config_data.model_dump(exclude_unset=True)
            response = await self._make_request("POST", "/api/v1/ai-models/configs", data=data)
            return AIModelConfigResponse(**response)
        except Exception as e:
            self._logger.error(f"创建模型配置失败: {e}")
            raise
    
    async def update_model_config(self, model_id: str, updates: Dict[str, Any]) -> AIModelConfigResponse:
        """
        更新模型配置
        
        Args:
            model_id: 模型ID
            updates: 更新数据
            
        Returns:
            更新后的模型配置
        """
        try:
            response = await self._make_request("PUT", f"/api/v1/ai-models/configs/{model_id}", data=updates)
            return AIModelConfigResponse(**response)
        except Exception as e:
            self._logger.error(f"更新模型配置失败 (id={model_id}): {e}")
            raise
    
    async def delete_model_config(self, model_id: str) -> bool:
        """
        删除模型配置
        
        Args:
            model_id: 模型ID
            
        Returns:
            删除是否成功
        """
        try:
            await self._make_request("DELETE", f"/api/v1/ai-models/configs/{model_id}")
            return True
        except Exception as e:
            self._logger.error(f"删除模型配置失败 (id={model_id}): {e}")
            return False
    
    async def get_active_models(self) -> List[AIModelConfigResponse]:
        """
        获取所有激活的模型配置
        
        Returns:
            激活的模型配置列表
        """
        try:
            response = await self._make_request("GET", "/api/v1/ai-models/active")
            return [AIModelConfigResponse(**model) for model in response.get("models", [])]
        except Exception as e:
            self._logger.error(f"获取激活模型失败: {e}")
            raise
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """
        获取模型统计信息
        
        Returns:
            统计信息字典
        """
        try:
            response = await self._make_request("GET", "/api/v1/ai-models/statistics")
            return response
        except Exception as e:
            self._logger.error(f"获取模型统计失败: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        检查Storage Service健康状态
        
        Returns:
            健康状态
        """
        try:
            await self._make_request("GET", "/health")
            return True
        except Exception as e:
            self._logger.warning(f"Storage Service健康检查失败: {e}")
            return False