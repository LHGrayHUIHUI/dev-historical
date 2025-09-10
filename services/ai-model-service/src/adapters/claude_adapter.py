"""
Claude (Anthropic) 平台适配器
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


class ClaudeAdapter(BaseAdapter):
    """Claude平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.clients = {}  # 账号ID -> 客户端实例的缓存
    
    def _get_client(self, account_config: APIAccount) -> httpx.AsyncClient:
        """获取Claude客户端实例"""
        account_id = account_config.id
        
        if account_id not in self.clients:
            api_key = self._decrypt_api_key(account_config.api_key_encrypted)
            
            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            # 自定义端点URL (默认使用官方API)
            base_url = account_config.endpoint_url or 'https://api.anthropic.com'
            
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
        """Claude聊天完成接口"""
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
            claude_messages = self._convert_messages_to_claude(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'max_tokens': kwargs.get('max_tokens', 1000),
                'messages': claude_messages,
                'stream': False
            }
            
            # 添加可选参数
            if kwargs.get('temperature') is not None:
                request_params['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['top_p'] = kwargs['top_p']
            
            if kwargs.get('stop'):
                request_params['stop_sequences'] = kwargs['stop']
            
            # 系统消息处理 (Claude有专门的system参数)
            system_messages = [msg for msg in messages if msg.role == 'system']
            if system_messages:
                request_params['system'] = system_messages[0].content
                # 从messages中移除system消息
                claude_messages = [msg for msg in claude_messages if msg['role'] != 'system']
                request_params['messages'] = claude_messages
            
            # 发起请求
            response = await client.post('/v1/messages', json=request_params)
            response.raise_for_status()
            
            response_data = response.json()
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, True)
            
            # 转换为标准格式
            return self._convert_claude_response_to_standard(
                response_data, model_config, response_time_ms
            )
            
        except httpx.HTTPStatusError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            
            error_msg = self._extract_claude_error(e)
            if e.response.status_code == 401:
                self._logger.error(f"Claude authentication error: {error_msg}")
                raise AdapterError(f"认证失败: {error_msg}")
            elif e.response.status_code == 429:
                self._logger.error(f"Claude rate limit: {error_msg}")
                raise AdapterError(f"请求频率限制: {error_msg}")
            elif e.response.status_code == 400:
                self._logger.error(f"Claude bad request: {error_msg}")
                raise AdapterError(f"请求参数错误: {error_msg}")
            else:
                self._logger.error(f"Claude HTTP error: {e.response.status_code} - {error_msg}")
                raise AdapterError(f"Claude请求失败: {error_msg}")
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            self._logger.error(f"Claude request failed: {e}")
            raise AdapterError(f"Claude请求失败: {self._extract_error_message(e)}")
    
    async def chat_completion_stream(self,
                                   model_config: ModelConfig,
                                   account_config: APIAccount,
                                   messages: List[ChatMessage],
                                   **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Claude流式聊天完成接口"""
        # 验证输入
        self._validate_messages(messages)
        self._validate_model_config(model_config)
        self._validate_account_config(account_config)
        
        # 应用限制
        kwargs = self._apply_request_limits(kwargs, model_config)
        
        try:
            client = self._get_client(account_config)
            
            # 转换消息格式
            claude_messages = self._convert_messages_to_claude(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'max_tokens': kwargs.get('max_tokens', 1000),
                'messages': claude_messages,
                'stream': True
            }
            
            # 添加可选参数
            if kwargs.get('temperature') is not None:
                request_params['temperature'] = kwargs['temperature']
            
            if kwargs.get('top_p') is not None:
                request_params['top_p'] = kwargs['top_p']
            
            if kwargs.get('stop'):
                request_params['stop_sequences'] = kwargs['stop']
            
            # 系统消息处理
            system_messages = [msg for msg in messages if msg.role == 'system']
            if system_messages:
                request_params['system'] = system_messages[0].content
                claude_messages = [msg for msg in claude_messages if msg['role'] != 'system']
                request_params['messages'] = claude_messages
            
            # 发起流式请求
            response_id = f"claude-{uuid.uuid4().hex[:10]}"
            
            async with client.stream('POST', '/v1/messages', json=request_params) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            if chunk_data.get('type') == 'content_block_delta':
                                delta = chunk_data.get('delta', {})
                                content = delta.get('text', '')
                                
                                if content:
                                    yield self._create_stream_chunk(
                                        response_id=response_id,
                                        model_name=model_config.name,
                                        delta_content=content,
                                        finish_reason=None
                                    )
                            
                            elif chunk_data.get('type') == 'message_stop':
                                yield self._create_stream_chunk(
                                    response_id=response_id,
                                    model_name=model_config.name,
                                    delta_content="",
                                    finish_reason='stop'
                                )
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self._logger.error(f"Claude stream request failed: {e}")
            raise AdapterError(f"Claude流式请求失败: {self._extract_error_message(e)}")
    
    async def list_models(self, account_config: APIAccount) -> List[Dict[str, Any]]:
        """获取Claude可用模型列表"""
        try:
            # Claude不提供公开的模型列表API，返回已知的Claude模型
            known_models = [
                {
                    'id': 'claude-3-5-sonnet-20241022',
                    'name': 'Claude 3.5 Sonnet',
                    'description': 'Claude 3.5 Sonnet - 平衡性能和速度',
                    'provider': 'claude',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'claude-3-opus-20240229',
                    'name': 'Claude 3 Opus',
                    'description': 'Claude 3 Opus - 最强大的模型',
                    'provider': 'claude',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'claude-3-haiku-20240307',
                    'name': 'Claude 3 Haiku',
                    'description': 'Claude 3 Haiku - 快速且经济',
                    'provider': 'claude',
                    'type': 'chat',
                    'created': None
                }
            ]
            
            return known_models
            
        except Exception as e:
            self._logger.error(f"Failed to list Claude models: {e}")
            raise AdapterError(f"获取Claude模型列表失败: {self._extract_error_message(e)}")
    
    def _convert_messages_to_claude(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为Claude格式"""
        claude_messages = []
        
        for message in messages:
            # Claude的消息格式比较简单
            claude_message = {
                'role': message.role,
                'content': message.content
            }
            
            claude_messages.append(claude_message)
        
        return claude_messages
    
    def _convert_claude_response_to_standard(self, 
                                           response_data: Dict[str, Any], 
                                           model_config: ModelConfig, 
                                           response_time_ms: float) -> Dict[str, Any]:
        """转换Claude响应为标准格式"""
        choices = []
        
        # Claude响应格式: content是一个数组
        content_blocks = response_data.get('content', [])
        content_text = ''
        
        for block in content_blocks:
            if block.get('type') == 'text':
                content_text += block.get('text', '')
        
        message = ChatMessage(
            role='assistant',
            content=content_text
        )
        
        standard_choice = self._create_standard_choice(
            index=0,
            message=message,
            finish_reason=response_data.get('stop_reason', 'stop')
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
            response_id=response_data.get('id', f"claude-{uuid.uuid4().hex[:10]}"),
            model_name=model_config.name,
            provider_name='claude',
            choices=choices,
            usage=usage,
            response_time_ms=response_time_ms
        )
    
    def _extract_claude_error(self, error: httpx.HTTPStatusError) -> str:
        """提取Claude API错误信息"""
        try:
            error_data = error.response.json()
            error_msg = error_data.get('error', {}).get('message', str(error))
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