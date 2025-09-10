"""
阿里云通义千问平台适配器
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


class AlibabaAdapter(BaseAdapter):
    """阿里云通义千问平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.clients = {}  # 账号ID -> 客户端实例的缓存
    
    def _get_client(self, account_config: APIAccount) -> httpx.AsyncClient:
        """获取阿里云客户端实例"""
        account_id = account_config.id
        
        if account_id not in self.clients:
            api_key = self._decrypt_api_key(account_config.api_key_encrypted)
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # 自定义端点URL (默认使用官方API)
            base_url = account_config.endpoint_url or 'https://dashscope.aliyuncs.com/api/v1'
            
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
        """阿里云通义千问聊天完成接口"""
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
            alibaba_messages = self._convert_messages_to_alibaba(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'input': {
                    'messages': alibaba_messages
                },
                'parameters': {}
            }
            
            # 添加参数
            if kwargs.get('temperature') is not None:
                request_params['parameters']['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['parameters']['top_p'] = kwargs['top_p']
            
            if kwargs.get('max_tokens'):
                request_params['parameters']['max_tokens'] = kwargs['max_tokens']
            
            if kwargs.get('stop'):
                request_params['parameters']['stop'] = kwargs['stop']
            
            if kwargs.get('user'):
                request_params['parameters']['user'] = kwargs['user']
            
            # 发起请求
            response = await client.post('/services/aigc/text-generation/generation', 
                                       json=request_params)
            response.raise_for_status()
            
            response_data = response.json()
            response_time_ms = (time.time() - start_time) * 1000
            
            # 检查响应状态
            if response_data.get('status_code') != 200:
                error_msg = response_data.get('message', '未知错误')
                self._log_response(model_config.name, response_time_ms, False)
                raise AdapterError(f"阿里云API错误: {error_msg}")
            
            self._log_response(model_config.name, response_time_ms, True)
            
            # 转换为标准格式
            return self._convert_alibaba_response_to_standard(
                response_data, model_config, response_time_ms
            )
            
        except httpx.HTTPStatusError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            
            error_msg = self._extract_alibaba_error(e)
            if e.response.status_code == 401:
                self._logger.error(f"Alibaba authentication error: {error_msg}")
                raise AdapterError(f"认证失败: {error_msg}")
            elif e.response.status_code == 429:
                self._logger.error(f"Alibaba rate limit: {error_msg}")
                raise AdapterError(f"请求频率限制: {error_msg}")
            else:
                self._logger.error(f"Alibaba HTTP error: {e.response.status_code} - {error_msg}")
                raise AdapterError(f"阿里云请求失败: {error_msg}")
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            self._logger.error(f"Alibaba request failed: {e}")
            raise AdapterError(f"阿里云请求失败: {self._extract_error_message(e)}")
    
    async def chat_completion_stream(self,
                                   model_config: ModelConfig,
                                   account_config: APIAccount,
                                   messages: List[ChatMessage],
                                   **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """阿里云通义千问流式聊天完成接口"""
        # 验证输入
        self._validate_messages(messages)
        self._validate_model_config(model_config)
        self._validate_account_config(account_config)
        
        # 应用限制
        kwargs = self._apply_request_limits(kwargs, model_config)
        
        try:
            client = self._get_client(account_config)
            
            # 转换消息格式
            alibaba_messages = self._convert_messages_to_alibaba(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'input': {
                    'messages': alibaba_messages
                },
                'parameters': {
                    'incremental_output': True  # 启用增量输出
                }
            }
            
            # 添加参数
            if kwargs.get('temperature') is not None:
                request_params['parameters']['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['parameters']['top_p'] = kwargs['top_p']
            
            if kwargs.get('max_tokens'):
                request_params['parameters']['max_tokens'] = kwargs['max_tokens']
            
            # 设置流式传输头
            headers = {'X-DashScope-SSE': 'enable'}
            
            response_id = f"alibaba-{uuid.uuid4().hex[:10]}"
            
            # 发起流式请求
            async with client.stream('POST', '/services/aigc/text-generation/generation', 
                                   json=request_params, headers=headers) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith('data:'):
                        data_str = line[5:].strip()  # 移除 'data:' 前缀
                        
                        if data_str == '[DONE]':
                            yield self._create_stream_chunk(
                                response_id=response_id,
                                model_name=model_config.name,
                                delta_content="",
                                finish_reason='stop'
                            )
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            # 检查状态
                            if chunk_data.get('status_code') != 200:
                                error_msg = chunk_data.get('message', '未知错误')
                                raise AdapterError(f"阿里云流式API错误: {error_msg}")
                            
                            output = chunk_data.get('output', {})
                            text = output.get('text', '')
                            finish_reason = output.get('finish_reason')
                            
                            if text:
                                yield self._create_stream_chunk(
                                    response_id=response_id,
                                    model_name=model_config.name,
                                    delta_content=text,
                                    finish_reason=finish_reason
                                )
                            
                            if finish_reason:
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self._logger.error(f"Alibaba stream request failed: {e}")
            raise AdapterError(f"阿里云流式请求失败: {self._extract_error_message(e)}")
    
    async def list_models(self, account_config: APIAccount) -> List[Dict[str, Any]]:
        """获取阿里云通义千问可用模型列表"""
        try:
            # 阿里云通义千问的已知模型列表
            known_models = [
                {
                    'id': 'qwen-plus',
                    'name': 'Qwen Plus',
                    'description': 'Qwen Plus - 高性能版本',
                    'provider': 'alibaba',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'qwen-turbo',
                    'name': 'Qwen Turbo',
                    'description': 'Qwen Turbo - 快速版本',
                    'provider': 'alibaba',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'qwen-max',
                    'name': 'Qwen Max',
                    'description': 'Qwen Max - 最强版本',
                    'provider': 'alibaba',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'qwen-max-1201',
                    'name': 'Qwen Max 1201',
                    'description': 'Qwen Max 1201 - 增强版本',
                    'provider': 'alibaba',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'qwen-max-longcontext',
                    'name': 'Qwen Max Long Context',
                    'description': 'Qwen Max Long Context - 长上下文版本',
                    'provider': 'alibaba',
                    'type': 'chat',
                    'created': None
                }
            ]
            
            return known_models
            
        except Exception as e:
            self._logger.error(f"Failed to list Alibaba models: {e}")
            raise AdapterError(f"获取阿里云模型列表失败: {self._extract_error_message(e)}")
    
    def _convert_messages_to_alibaba(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为阿里云格式"""
        alibaba_messages = []
        
        for message in messages:
            alibaba_message = {
                'role': message.role,
                'content': message.content
            }
            
            # 添加名称（如果有）
            if message.name:
                alibaba_message['name'] = message.name
            
            alibaba_messages.append(alibaba_message)
        
        return alibaba_messages
    
    def _convert_alibaba_response_to_standard(self, 
                                            response_data: Dict[str, Any], 
                                            model_config: ModelConfig, 
                                            response_time_ms: float) -> Dict[str, Any]:
        """转换阿里云响应为标准格式"""
        choices = []
        
        output = response_data.get('output', {})
        text = output.get('text', '')
        finish_reason = output.get('finish_reason', 'stop')
        
        message = ChatMessage(
            role='assistant',
            content=text
        )
        
        standard_choice = self._create_standard_choice(
            index=0,
            message=message,
            finish_reason=finish_reason
        )
        
        choices.append(standard_choice)
        
        # 使用量统计
        usage = {}
        usage_data = response_data.get('usage', {})
        if usage_data:
            usage = self._create_standard_usage(
                prompt_tokens=usage_data.get('input_tokens', 0),
                completion_tokens=usage_data.get('output_tokens', 0)
            )
        
        return self._create_standard_response(
            response_id=response_data.get('request_id', f"alibaba-{uuid.uuid4().hex[:10]}"),
            model_name=model_config.name,
            provider_name='alibaba',
            choices=choices,
            usage=usage,
            response_time_ms=response_time_ms
        )
    
    def _extract_alibaba_error(self, error: httpx.HTTPStatusError) -> str:
        """提取阿里云API错误信息"""
        try:
            error_data = error.response.json()
            error_msg = error_data.get('message', str(error))
            error_code = error_data.get('code', '')
            
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