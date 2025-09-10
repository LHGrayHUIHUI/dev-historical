"""
百度文心一言平台适配器
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


class BaiduAdapter(BaseAdapter):
    """百度文心一言平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.clients = {}  # 账号ID -> 客户端实例的缓存
        self.access_tokens = {}  # 账号ID -> access_token的缓存
        self.token_expires = {}  # 账号ID -> token过期时间的缓存
    
    def _get_client(self, account_config: APIAccount) -> httpx.AsyncClient:
        """获取百度客户端实例"""
        account_id = account_config.id
        
        if account_id not in self.clients:
            # 自定义端点URL (默认使用官方API)
            base_url = account_config.endpoint_url or 'https://aip.baidubce.com'
            
            self.clients[account_id] = httpx.AsyncClient(
                base_url=base_url,
                timeout=30.0
            )
        
        return self.clients[account_id]
    
    async def _get_access_token(self, account_config: APIAccount) -> str:
        """获取百度API访问令牌"""
        account_id = account_config.id
        current_time = time.time()
        
        # 检查是否有有效的token
        if (account_id in self.access_tokens and 
            account_id in self.token_expires and 
            current_time < self.token_expires[account_id]):
            return self.access_tokens[account_id]
        
        try:
            # 解密API密钥信息
            api_key = self._decrypt_api_key(account_config.api_key_encrypted)
            # 假设secret_key存储在organization_id字段中
            secret_key = account_config.organization_id or ""
            
            client = self._get_client(account_config)
            
            # 获取access_token
            token_url = f"/oauth/2.0/token"
            params = {
                'grant_type': 'client_credentials',
                'client_id': api_key,
                'client_secret': secret_key
            }
            
            response = await client.post(token_url, params=params)
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)
            
            if not access_token:
                raise AdapterError("无法获取百度API访问令牌")
            
            # 缓存token (提前5分钟过期)
            self.access_tokens[account_id] = access_token
            self.token_expires[account_id] = current_time + expires_in - 300
            
            return access_token
            
        except Exception as e:
            self._logger.error(f"Failed to get Baidu access token: {e}")
            raise AdapterError(f"获取百度访问令牌失败: {self._extract_error_message(e)}")
    
    async def chat_completion(self,
                            model_config: ModelConfig,
                            account_config: APIAccount,
                            messages: List[ChatMessage],
                            **kwargs) -> Dict[str, Any]:
        """百度文心一言聊天完成接口"""
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
            access_token = await self._get_access_token(account_config)
            start_time = time.time()
            
            # 转换消息格式
            baidu_messages = self._convert_messages_to_baidu(messages)
            
            # 构建请求参数
            request_params = {
                'messages': baidu_messages,
                'stream': False
            }
            
            # 添加可选参数
            if kwargs.get('temperature') is not None:
                request_params['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['top_p'] = kwargs['top_p']
            
            if kwargs.get('max_tokens'):
                request_params['max_output_tokens'] = kwargs['max_tokens']
            
            if kwargs.get('stop'):
                request_params['stop'] = kwargs['stop']
            
            if kwargs.get('user'):
                request_params['user_id'] = kwargs['user']
            
            # 确定API端点 (不同模型使用不同的端点)
            endpoint = self._get_model_endpoint(model_config.model_id)
            api_url = f"{endpoint}?access_token={access_token}"
            
            # 发起请求
            response = await client.post(api_url, json=request_params)
            response.raise_for_status()
            
            response_data = response.json()
            response_time_ms = (time.time() - start_time) * 1000
            
            # 检查响应中的错误
            if 'error_code' in response_data:
                error_msg = response_data.get('error_msg', '未知错误')
                self._log_response(model_config.name, response_time_ms, False)
                raise AdapterError(f"百度API错误: {error_msg}")
            
            self._log_response(model_config.name, response_time_ms, True)
            
            # 转换为标准格式
            return self._convert_baidu_response_to_standard(
                response_data, model_config, response_time_ms
            )
            
        except httpx.HTTPStatusError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            
            error_msg = self._extract_baidu_error(e)
            if e.response.status_code == 401:
                self._logger.error(f"Baidu authentication error: {error_msg}")
                raise AdapterError(f"认证失败: {error_msg}")
            elif e.response.status_code == 429:
                self._logger.error(f"Baidu rate limit: {error_msg}")
                raise AdapterError(f"请求频率限制: {error_msg}")
            else:
                self._logger.error(f"Baidu HTTP error: {e.response.status_code} - {error_msg}")
                raise AdapterError(f"百度请求失败: {error_msg}")
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            self._logger.error(f"Baidu request failed: {e}")
            raise AdapterError(f"百度请求失败: {self._extract_error_message(e)}")
    
    async def chat_completion_stream(self,
                                   model_config: ModelConfig,
                                   account_config: APIAccount,
                                   messages: List[ChatMessage],
                                   **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """百度文心一言流式聊天完成接口"""
        # 验证输入
        self._validate_messages(messages)
        self._validate_model_config(model_config)
        self._validate_account_config(account_config)
        
        # 应用限制
        kwargs = self._apply_request_limits(kwargs, model_config)
        
        try:
            client = self._get_client(account_config)
            access_token = await self._get_access_token(account_config)
            
            # 转换消息格式
            baidu_messages = self._convert_messages_to_baidu(messages)
            
            # 构建请求参数
            request_params = {
                'messages': baidu_messages,
                'stream': True
            }
            
            # 添加可选参数
            if kwargs.get('temperature') is not None:
                request_params['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['top_p'] = kwargs['top_p']
            
            if kwargs.get('max_tokens'):
                request_params['max_output_tokens'] = kwargs['max_tokens']
            
            # 确定API端点
            endpoint = self._get_model_endpoint(model_config.model_id)
            api_url = f"{endpoint}?access_token={access_token}"
            
            response_id = f"baidu-{uuid.uuid4().hex[:10]}"
            
            # 发起流式请求
            async with client.stream('POST', api_url, json=request_params) as response:
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
                            if 'error_code' in chunk_data:
                                error_msg = chunk_data.get('error_msg', '未知错误')
                                raise AdapterError(f"百度流式API错误: {error_msg}")
                            
                            content = chunk_data.get('result', '')
                            is_end = chunk_data.get('is_end', False)
                            
                            if content:
                                yield self._create_stream_chunk(
                                    response_id=response_id,
                                    model_name=model_config.name,
                                    delta_content=content,
                                    finish_reason='stop' if is_end else None
                                )
                            
                            if is_end:
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self._logger.error(f"Baidu stream request failed: {e}")
            raise AdapterError(f"百度流式请求失败: {self._extract_error_message(e)}")
    
    async def list_models(self, account_config: APIAccount) -> List[Dict[str, Any]]:
        """获取百度文心一言可用模型列表"""
        try:
            # 百度文心一言的已知模型列表
            known_models = [
                {
                    'id': 'ernie-4.0-turbo-8k',
                    'name': 'ERNIE 4.0 Turbo',
                    'description': 'ERNIE 4.0 Turbo - 高性能版本',
                    'provider': 'baidu',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'ernie-3.5-8k',
                    'name': 'ERNIE 3.5',
                    'description': 'ERNIE 3.5 - 平衡版本',
                    'provider': 'baidu',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'ernie-turbo-8k',
                    'name': 'ERNIE Turbo',
                    'description': 'ERNIE Turbo - 快速版本',
                    'provider': 'baidu',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'ernie-speed-8k',
                    'name': 'ERNIE Speed',
                    'description': 'ERNIE Speed - 极速版本',
                    'provider': 'baidu',
                    'type': 'chat',
                    'created': None
                }
            ]
            
            return known_models
            
        except Exception as e:
            self._logger.error(f"Failed to list Baidu models: {e}")
            raise AdapterError(f"获取百度模型列表失败: {self._extract_error_message(e)}")
    
    def _get_model_endpoint(self, model_id: str) -> str:
        """根据模型ID获取对应的API端点"""
        endpoints = {
            'ernie-4.0-turbo-8k': '/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-4.0-turbo-8k',
            'ernie-3.5-8k': '/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-8k',
            'ernie-turbo-8k': '/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-turbo-8k',
            'ernie-speed-8k': '/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-8k'
        }
        
        return endpoints.get(model_id, '/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions')
    
    def _convert_messages_to_baidu(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为百度格式"""
        baidu_messages = []
        
        for message in messages:
            # 百度API不支持system消息，将其转换为user消息
            role = message.role
            if role == 'system':
                role = 'user'
            
            baidu_message = {
                'role': role,
                'content': message.content
            }
            
            baidu_messages.append(baidu_message)
        
        return baidu_messages
    
    def _convert_baidu_response_to_standard(self, 
                                          response_data: Dict[str, Any], 
                                          model_config: ModelConfig, 
                                          response_time_ms: float) -> Dict[str, Any]:
        """转换百度响应为标准格式"""
        choices = []
        
        result_text = response_data.get('result', '')
        
        message = ChatMessage(
            role='assistant',
            content=result_text
        )
        
        standard_choice = self._create_standard_choice(
            index=0,
            message=message,
            finish_reason='stop'
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
            response_id=response_data.get('id', f"baidu-{uuid.uuid4().hex[:10]}"),
            model_name=model_config.name,
            provider_name='baidu',
            choices=choices,
            usage=usage,
            response_time_ms=response_time_ms
        )
    
    def _extract_baidu_error(self, error: httpx.HTTPStatusError) -> str:
        """提取百度API错误信息"""
        try:
            error_data = error.response.json()
            error_msg = error_data.get('error_msg', str(error))
            error_code = error_data.get('error_code', '')
            
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