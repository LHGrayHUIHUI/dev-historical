"""
OpenAI平台适配器
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, AsyncIterator

import httpx
import openai
from openai import AsyncOpenAI

from .base_adapter import BaseAdapter, AdapterError
from ..models.ai_models import ModelConfig, APIAccount
from ..models.requests import ChatMessage


class OpenAIAdapter(BaseAdapter):
    """OpenAI平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.clients = {}  # 账号ID -> 客户端实例的缓存
    
    def _get_client(self, account_config: APIAccount) -> AsyncOpenAI:
        """获取OpenAI客户端实例"""
        account_id = account_config.id
        
        if account_id not in self.clients:
            api_key = self._decrypt_api_key(account_config.api_key_encrypted)
            
            client_kwargs = {
                'api_key': api_key,
                'timeout': 30.0,
                'max_retries': 2
            }
            
            # 自定义端点URL
            if account_config.endpoint_url:
                client_kwargs['base_url'] = account_config.endpoint_url
            
            # 组织ID
            if account_config.organization_id:
                client_kwargs['organization'] = account_config.organization_id
            
            self.clients[account_id] = AsyncOpenAI(**client_kwargs)
        
        return self.clients[account_id]
    
    async def chat_completion(self,
                            model_config: ModelConfig,
                            account_config: APIAccount,
                            messages: List[ChatMessage],
                            **kwargs) -> Dict[str, Any]:
        """OpenAI聊天完成接口"""
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
            openai_messages = self._convert_messages_to_openai(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'messages': openai_messages,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
                'top_p': kwargs.get('top_p', 1.0),
                'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
                'presence_penalty': kwargs.get('presence_penalty', 0.0),
                'stream': False
            }
            
            # 停止词
            if kwargs.get('stop'):
                request_params['stop'] = kwargs['stop']
            
            # 用户标识
            if kwargs.get('user'):
                request_params['user'] = kwargs['user']
            
            # 发起请求
            response = await client.chat.completions.create(**request_params)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, True)
            
            # 转换为标准格式
            return self._convert_openai_response_to_standard(
                response, model_config, response_time_ms
            )
            
        except openai.APITimeoutError as e:
            self._logger.error(f"OpenAI API timeout: {e}")
            raise AdapterError(f"请求超时: {self._extract_error_message(e)}")
        
        except openai.RateLimitError as e:
            self._logger.error(f"OpenAI rate limit: {e}")
            raise AdapterError(f"请求频率限制: {self._extract_error_message(e)}")
        
        except openai.AuthenticationError as e:
            self._logger.error(f"OpenAI authentication error: {e}")
            raise AdapterError(f"认证失败: {self._extract_error_message(e)}")
        
        except openai.PermissionDeniedError as e:
            self._logger.error(f"OpenAI permission denied: {e}")
            raise AdapterError(f"权限不足: {self._extract_error_message(e)}")
        
        except openai.BadRequestError as e:
            self._logger.error(f"OpenAI bad request: {e}")
            raise AdapterError(f"请求参数错误: {self._extract_error_message(e)}")
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            self._logger.error(f"OpenAI request failed: {e}")
            raise AdapterError(f"OpenAI请求失败: {self._extract_error_message(e)}")
    
    async def chat_completion_stream(self,
                                   model_config: ModelConfig,
                                   account_config: APIAccount,
                                   messages: List[ChatMessage],
                                   **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """OpenAI流式聊天完成接口"""
        # 验证输入
        self._validate_messages(messages)
        self._validate_model_config(model_config)
        self._validate_account_config(account_config)
        
        # 应用限制
        kwargs = self._apply_request_limits(kwargs, model_config)
        
        try:
            client = self._get_client(account_config)
            
            # 转换消息格式
            openai_messages = self._convert_messages_to_openai(messages)
            
            # 构建请求参数
            request_params = {
                'model': model_config.model_id,
                'messages': openai_messages,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
                'top_p': kwargs.get('top_p', 1.0),
                'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
                'presence_penalty': kwargs.get('presence_penalty', 0.0),
                'stream': True
            }
            
            # 停止词
            if kwargs.get('stop'):
                request_params['stop'] = kwargs['stop']
            
            # 用户标识
            if kwargs.get('user'):
                request_params['user'] = kwargs['user']
            
            # 发起流式请求
            response_stream = await client.chat.completions.create(**request_params)
            response_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"
            
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    finish_reason = chunk.choices[0].finish_reason
                    
                    yield self._create_stream_chunk(
                        response_id=response_id,
                        model_name=model_config.name,
                        delta_content=content,
                        finish_reason=finish_reason
                    )
                    
                    if finish_reason:
                        break
            
        except Exception as e:
            self._logger.error(f"OpenAI stream request failed: {e}")
            raise AdapterError(f"OpenAI流式请求失败: {self._extract_error_message(e)}")
    
    async def list_models(self, account_config: APIAccount) -> List[Dict[str, Any]]:
        """获取OpenAI可用模型列表"""
        try:
            client = self._get_client(account_config)
            
            models_response = await client.models.list()
            models = []
            
            # 过滤聊天模型
            chat_model_prefixes = ['gpt-3.5', 'gpt-4']
            
            for model in models_response.data:
                model_id = model.id
                
                if any(model_id.startswith(prefix) for prefix in chat_model_prefixes):
                    models.append({
                        'id': model_id,
                        'name': model_id,
                        'description': f"OpenAI {model_id}",
                        'provider': 'openai',
                        'type': 'chat',
                        'created': getattr(model, 'created', None)
                    })
            
            return models
            
        except Exception as e:
            self._logger.error(f"Failed to list OpenAI models: {e}")
            raise AdapterError(f"获取OpenAI模型列表失败: {self._extract_error_message(e)}")
    
    def _convert_messages_to_openai(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为OpenAI格式"""
        openai_messages = []
        
        for message in messages:
            openai_message = {
                'role': message.role,
                'content': message.content
            }
            
            # 添加名称
            if message.name:
                openai_message['name'] = message.name
            
            # 添加函数调用
            if message.function_call:
                openai_message['function_call'] = message.function_call
            
            openai_messages.append(openai_message)
        
        return openai_messages
    
    def _convert_openai_response_to_standard(self, 
                                           response, 
                                           model_config: ModelConfig, 
                                           response_time_ms: float) -> Dict[str, Any]:
        """转换OpenAI响应为标准格式"""
        choices = []
        
        for choice in response.choices:
            message = ChatMessage(
                role=choice.message.role,
                content=choice.message.content or ""
            )
            
            standard_choice = self._create_standard_choice(
                index=choice.index,
                message=message,
                finish_reason=choice.finish_reason
            )
            
            choices.append(standard_choice)
        
        # 使用量统计
        usage = {}
        if response.usage:
            usage = self._create_standard_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
        
        return self._create_standard_response(
            response_id=response.id,
            model_name=model_config.name,
            provider_name='openai',
            choices=choices,
            usage=usage,
            response_time_ms=response_time_ms
        )
    
    def __del__(self):
        """清理资源"""
        # OpenAI客户端会自动清理，这里不需要特殊处理
        pass