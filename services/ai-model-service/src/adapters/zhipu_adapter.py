"""
智谱AI平台适配器
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, AsyncIterator

import httpx

from .base_adapter import BaseAdapter, AdapterError
from ..models.ai_models import ModelConfig, APIAccount
from ..models.requests import ChatMessage


class ZhipuAdapter(BaseAdapter):
    """智谱AI平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.clients = {}  # 账号ID -> 客户端实例的缓存
    
    def _get_client(self, account_config: APIAccount) -> httpx.AsyncClient:
        """获取智谱AI客户端实例"""
        account_id = account_config.id
        
        if account_id not in self.clients:
            api_key = self._decrypt_api_key(account_config.api_key_encrypted)
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # 自定义端点URL (默认使用官方API)
            base_url = account_config.endpoint_url or 'https://open.bigmodel.cn/api/paas/v4'
            
            self.clients[account_id] = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=30.0
            )
        
        return self.clients[account_id]
    
    async def chat_completion(self,
                            model_config: ModelConfig,
                            account_config: APIAccount,
                            messages: List[ChatMessage],
                            **kwargs) -> Dict[str, Any]:
        """智谱AI聊天完成接口"""
        # 验证输入
        self._validate_messages(messages)
        self._validate_model_config(model_config)
        self._validate_account_config(account_config)
        
        # 应用限制
        kwargs = self._apply_request_limits(kwargs, model_config)
        
        # 记录请求
        self._log_request(model_config.name, len(messages), **kwargs)
        
        try:
            client = self._get_client(account_config)
            start_time = time.time()
            
            # 转换消息格式
            zhipu_messages = self._convert_messages_to_zhipu(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'messages': zhipu_messages,
                'stream': False
            }
            
            # 添加可选参数
            if kwargs.get('temperature') is not None:
                request_params['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['top_p'] = kwargs['top_p']
            
            if kwargs.get('max_tokens'):
                request_params['max_tokens'] = kwargs['max_tokens']
            
            if kwargs.get('stop'):
                request_params['stop'] = kwargs['stop']
            
            # 发起请求
            response = await client.post('/chat/completions', json=request_params)
            response.raise_for_status()
            
            response_data = response.json()
            response_time_ms = (time.time() - start_time) * 1000
            
            # 检查响应中的错误
            if 'error' in response_data:
                error_info = response_data['error']
                error_msg = error_info.get('message', '未知错误')
                self._log_response(model_config.name, response_time_ms, False)
                raise AdapterError(f"智谱API错误: {error_msg}")
            
            self._log_response(model_config.name, response_time_ms, True)
            
            # 转换为标准格式
            return self._convert_zhipu_response_to_standard(
                response_data, model_config, response_time_ms
            )
            
        except httpx.HTTPStatusError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            
            error_msg = self._extract_zhipu_error(e)
            if e.response.status_code == 401:
                self._logger.error(f"Zhipu authentication error: {error_msg}")
                raise AdapterError(f"认证失败: {error_msg}")
            elif e.response.status_code == 429:
                self._logger.error(f"Zhipu rate limit: {error_msg}")
                raise AdapterError(f"请求频率限制: {error_msg}")
            else:
                self._logger.error(f"Zhipu HTTP error: {e.response.status_code} - {error_msg}")
                raise AdapterError(f"智谱请求失败: {error_msg}")
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            self._logger.error(f"Zhipu request failed: {e}")
            raise AdapterError(f"智谱请求失败: {self._extract_error_message(e)}")
    
    async def chat_completion_stream(self,
                                   model_config: ModelConfig,
                                   account_config: APIAccount,
                                   messages: List[ChatMessage],
                                   **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """智谱AI流式聊天完成接口"""
        # 验证输入
        self._validate_messages(messages)
        self._validate_model_config(model_config)
        self._validate_account_config(account_config)
        
        # 应用限制
        kwargs = self._apply_request_limits(kwargs, model_config)
        
        try:
            client = self._get_client(account_config)
            
            # 转换消息格式
            zhipu_messages = self._convert_messages_to_zhipu(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'messages': zhipu_messages,
                'stream': True
            }
            
            # 添加可选参数
            if kwargs.get('temperature') is not None:
                request_params['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['top_p'] = kwargs['top_p']
            
            if kwargs.get('max_tokens'):
                request_params['max_tokens'] = kwargs['max_tokens']
            
            response_id = f"zhipu-{uuid.uuid4().hex[:10]}"
            
            # 发起流式请求
            async with client.stream('POST', '/chat/completions', 
                                   json=request_params) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        
                        if data_str.strip() == '[DONE]':
                            yield self._create_stream_chunk(
                                response_id=response_id,
                                model_name=model_config.name,
                                delta_content="",
                                finish_reason='stop'
                            )
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            # 检查错误
                            if 'error' in chunk_data:
                                error_info = chunk_data['error']
                                error_msg = error_info.get('message', '未知错误')
                                raise AdapterError(f"智谱流式API错误: {error_msg}")
                            
                            choices = chunk_data.get('choices', [])
                            if choices:
                                choice = choices[0]
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                                finish_reason = choice.get('finish_reason')
                                
                                if content:
                                    yield self._create_stream_chunk(
                                        response_id=response_id,
                                        model_name=model_config.name,
                                        delta_content=content,
                                        finish_reason=finish_reason
                                    )
                                
                                if finish_reason:
                                    break
                                
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self._logger.error(f"Zhipu stream request failed: {e}")
            raise AdapterError(f"智谱流式请求失败: {self._extract_error_message(e)}")
    
    async def list_models(self, account_config: APIAccount) -> List[Dict[str, Any]]:
        """获取智谱AI可用模型列表"""
        try:
            # 智谱AI的已知模型列表
            known_models = [
                {
                    'id': 'glm-4',
                    'name': 'GLM-4',
                    'description': 'GLM-4 - 第四代GLM模型',
                    'provider': 'zhipu',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'glm-4-air',
                    'name': 'GLM-4 Air',
                    'description': 'GLM-4 Air - 轻量版本',
                    'provider': 'zhipu',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'glm-4-airx',
                    'name': 'GLM-4 AirX',
                    'description': 'GLM-4 AirX - 增强版本',
                    'provider': 'zhipu',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'glm-4-flash',
                    'name': 'GLM-4 Flash',
                    'description': 'GLM-4 Flash - 快速版本',
                    'provider': 'zhipu',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'glm-3-turbo',
                    'name': 'GLM-3 Turbo',
                    'description': 'GLM-3 Turbo - 第三代高速版本',
                    'provider': 'zhipu',
                    'type': 'chat',
                    'created': None
                }
            ]
            
            return known_models
            
        except Exception as e:
            self._logger.error(f"Failed to list Zhipu models: {e}")
            raise AdapterError(f"获取智谱模型列表失败: {self._extract_error_message(e)}")
    
    def _convert_messages_to_zhipu(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为智谱格式"""
        zhipu_messages = []
        
        for message in messages:
            zhipu_message = {
                'role': message.role,
                'content': message.content
            }
            
            # 添加名称（如果有）
            if message.name:
                zhipu_message['name'] = message.name
            
            # 添加函数调用（如果有）
            if message.function_call:
                zhipu_message['function_call'] = message.function_call
            
            zhipu_messages.append(zhipu_message)
        
        return zhipu_messages
    
    def _convert_zhipu_response_to_standard(self, 
                                          response_data: Dict[str, Any], 
                                          model_config: ModelConfig, 
                                          response_time_ms: float) -> Dict[str, Any]:
        """转换智谱响应为标准格式"""
        choices = []
        
        for choice_data in response_data.get('choices', []):
            message_data = choice_data.get('message', {})
            
            message = ChatMessage(
                role=message_data.get('role', 'assistant'),
                content=message_data.get('content', '')
            )
            
            standard_choice = self._create_standard_choice(
                index=choice_data.get('index', 0),
                message=message,
                finish_reason=choice_data.get('finish_reason', 'stop')
            )
            
            choices.append(standard_choice)
        
        # 使用量统计
        usage = {}
        usage_data = response_data.get('usage', {})
        if usage_data:
            usage = self._create_standard_usage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0)
            )
        
        return self._create_standard_response(
            response_id=response_data.get('id', f"zhipu-{uuid.uuid4().hex[:10]}"),
            model_name=model_config.name,
            provider_name='zhipu',
            choices=choices,
            usage=usage,
            response_time_ms=response_time_ms
        )
    
    def _extract_zhipu_error(self, error: httpx.HTTPStatusError) -> str:
        """提取智谱API错误信息"""
        try:
            error_data = error.response.json()
            error_info = error_data.get('error', {})
            error_msg = error_info.get('message', str(error))
            error_code = error_info.get('code', '')
            
            if error_code:
                return f"[{error_code}] {error_msg}"
            return error_msg
        except:
            return str(error)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """清理资源"""
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()