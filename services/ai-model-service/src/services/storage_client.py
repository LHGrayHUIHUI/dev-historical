"""
Storage Service 客户端

与storage-service (端口8002)通信的HTTP客户端
所有数据库操作都通过这个客户端进行
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date

import httpx
from pydantic import BaseModel

from ..config.settings import get_settings
from ..models.ai_models import ModelConfig, APIAccount, UsageStatistic


class StorageServiceError(Exception):
    """Storage Service错误"""
    pass


class StorageClient:
    """Storage Service HTTP客户端"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.storage_service_url
        self.timeout = self.settings.storage_service_timeout
        self.client = None
        self._logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()
    
    async def connect(self):
        """建立连接"""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"{self.settings.service_name}/{self.settings.service_version}"
                }
            )
    
    async def disconnect(self):
        """关闭连接"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发起HTTP请求"""
        if not self.client:
            await self.connect()
        
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            
            # 检查响应格式
            data = response.json()
            if isinstance(data, dict) and 'success' in data:
                if not data['success']:
                    raise StorageServiceError(data.get('message', 'Storage service error'))
                return data.get('data', {})
            
            return data
            
        except httpx.HTTPStatusError as e:
            self._logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise StorageServiceError(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            self._logger.error(f"Request error: {str(e)}")
            raise StorageServiceError(f"Request error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            raise StorageServiceError(f"Unexpected error: {str(e)}")
    
    # === AI模型相关接口 ===
    
    async def get_ai_models(self, provider: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """获取AI模型列表"""
        params = {}
        if provider:
            params['provider'] = provider
        if active_only:
            params['active_only'] = 'true'
        
        return await self._make_request('GET', '/api/v1/ai-models', params=params)
    
    async def get_ai_model(self, model_id: str) -> Dict[str, Any]:
        """获取单个AI模型"""
        return await self._make_request('GET', f'/api/v1/ai-models/{model_id}')
    
    async def create_ai_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建AI模型"""
        return await self._make_request('POST', '/api/v1/ai-models', json=model_data)
    
    async def update_ai_model(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新AI模型"""
        return await self._make_request('PUT', f'/api/v1/ai-models/{model_id}', json=model_data)
    
    async def delete_ai_model(self, model_id: str) -> bool:
        """删除AI模型"""
        await self._make_request('DELETE', f'/api/v1/ai-models/{model_id}')
        return True
    
    # === API账号相关接口 ===
    
    async def get_api_accounts(self, provider: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取API账号列表"""
        params = {}
        if provider:
            params['provider'] = provider
        if status:
            params['status'] = status
        
        return await self._make_request('GET', '/api/v1/api-accounts', params=params)
    
    async def get_api_account(self, account_id: str) -> Dict[str, Any]:
        """获取单个API账号"""
        return await self._make_request('GET', f'/api/v1/api-accounts/{account_id}')
    
    async def create_api_account(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建API账号"""
        return await self._make_request('POST', '/api/v1/api-accounts', json=account_data)
    
    async def update_api_account(self, account_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新API账号"""
        return await self._make_request('PUT', f'/api/v1/api-accounts/{account_id}', json=account_data)
    
    async def update_account_health(self, account_id: str, health_score: float) -> Dict[str, Any]:
        """更新账号健康评分"""
        return await self._make_request('PATCH', f'/api/v1/api-accounts/{account_id}/health', 
                                      json={'health_score': health_score})
    
    async def update_account_usage(self, account_id: str, quota_used: int) -> Dict[str, Any]:
        """更新账号使用量"""
        return await self._make_request('PATCH', f'/api/v1/api-accounts/{account_id}/usage',
                                      json={'quota_used': quota_used})
    
    async def delete_api_account(self, account_id: str) -> bool:
        """删除API账号"""
        await self._make_request('DELETE', f'/api/v1/api-accounts/{account_id}')
        return True
    
    # === 模型账号映射接口 ===
    
    async def get_model_account_mappings(self, model_id: Optional[str] = None, 
                                       account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取模型账号映射"""
        params = {}
        if model_id:
            params['model_id'] = model_id
        if account_id:
            params['account_id'] = account_id
        
        return await self._make_request('GET', '/api/v1/model-account-mappings', params=params)
    
    async def create_model_account_mapping(self, mapping_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建模型账号映射"""
        return await self._make_request('POST', '/api/v1/model-account-mappings', json=mapping_data)
    
    async def delete_model_account_mapping(self, mapping_id: str) -> bool:
        """删除模型账号映射"""
        await self._make_request('DELETE', f'/api/v1/model-account-mappings/{mapping_id}')
        return True
    
    # === 路由策略接口 ===
    
    async def get_routing_strategies(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """获取路由策略"""
        params = {}
        if active_only:
            params['active_only'] = 'true'
        
        return await self._make_request('GET', '/api/v1/routing-strategies', params=params)
    
    async def get_routing_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """获取单个路由策略"""
        return await self._make_request('GET', f'/api/v1/routing-strategies/{strategy_id}')
    
    async def create_routing_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建路由策略"""
        return await self._make_request('POST', '/api/v1/routing-strategies', json=strategy_data)
    
    async def update_routing_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新路由策略"""
        return await self._make_request('PUT', f'/api/v1/routing-strategies/{strategy_id}', json=strategy_data)
    
    # === 使用统计接口 ===
    
    async def record_usage(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录使用统计"""
        return await self._make_request('POST', '/api/v1/usage-statistics', json=usage_data)
    
    async def get_usage_statistics(self, 
                                 account_id: Optional[str] = None,
                                 model_id: Optional[str] = None,
                                 start_date: Optional[date] = None,
                                 end_date: Optional[date] = None,
                                 period: str = '24h') -> Dict[str, Any]:
        """获取使用统计"""
        params = {'period': period}
        if account_id:
            params['account_id'] = account_id
        if model_id:
            params['model_id'] = model_id
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()
        
        return await self._make_request('GET', '/api/v1/usage-statistics', params=params)
    
    # === 请求日志接口 ===
    
    async def log_request(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录请求日志"""
        return await self._make_request('POST', '/api/v1/ai-request-logs', json=log_data)
    
    async def get_request_logs(self,
                             account_id: Optional[str] = None,
                             model_id: Optional[str] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """获取请求日志"""
        params = {'limit': limit}
        if account_id:
            params['account_id'] = account_id
        if model_id:
            params['model_id'] = model_id
        if start_time:
            params['start_time'] = start_time.isoformat()
        if end_time:
            params['end_time'] = end_time.isoformat()
        
        return await self._make_request('GET', '/api/v1/ai-request-logs', params=params)
    
    # === 健康检查接口 ===
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return await self._make_request('GET', '/health')


# 全局客户端实例
_storage_client_instance = None

async def get_storage_client() -> StorageClient:
    """获取Storage Service客户端实例"""
    global _storage_client_instance
    if _storage_client_instance is None:
        _storage_client_instance = StorageClient()
        await _storage_client_instance.connect()
    return _storage_client_instance

async def close_storage_client():
    """关闭Storage Service客户端"""
    global _storage_client_instance
    if _storage_client_instance:
        await _storage_client_instance.disconnect()
        _storage_client_instance = None