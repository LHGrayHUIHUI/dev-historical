"""
AI平台适配器基类
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, AsyncIterator
from datetime import datetime

from ..models.ai_models import ModelConfig, APIAccount
from ..models.requests import ChatMessage, Usage


class AdapterError(Exception):
    """适配器错误"""
    pass


class BaseAdapter(ABC):
    """AI平台适配器基类"""
    
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def chat_completion(self,
                            model_config: ModelConfig,
                            account_config: APIAccount,
                            messages: List[ChatMessage],
                            **kwargs) -> Dict[str, Any]:
        """
        聊天完成接口
        
        Args:
            model_config: 模型配置
            account_config: 账号配置
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 标准化响应
        """
        pass
    
    @abstractmethod
    async def chat_completion_stream(self,
                                   model_config: ModelConfig,
                                   account_config: APIAccount,
                                   messages: List[ChatMessage],
                                   **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """
        流式聊天完成接口
        
        Args:
            model_config: 模型配置
            account_config: 账号配置
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            Dict[str, Any]: 流式响应块
        """
        pass
    
    @abstractmethod
    async def list_models(self, account_config: APIAccount) -> List[Dict[str, Any]]:
        """
        获取可用模型列表
        
        Args:
            account_config: 账号配置
            
        Returns:
            List[Dict[str, Any]]: 模型列表
        """
        pass
    
    def _decrypt_api_key(self, encrypted_key: str) -> str:
        """解密API密钥"""
        # 这里应该使用实际的解密逻辑
        # 暂时返回原始值
        return encrypted_key
    
    def _create_standard_response(self,
                                 response_id: str,
                                 model_name: str,
                                 provider_name: str,
                                 choices: List[Dict[str, Any]],
                                 usage: Dict[str, int],
                                 response_time_ms: float) -> Dict[str, Any]:
        """创建标准化响应"""
        return {
            'id': response_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model_name,
            'provider': provider_name,
            'choices': choices,
            'usage': usage,
            'response_time_ms': response_time_ms
        }
    
    def _create_standard_choice(self,
                              index: int,
                              message: ChatMessage,
                              finish_reason: str = 'stop') -> Dict[str, Any]:
        """创建标准化选择项"""
        return {
            'index': index,
            'message': {
                'role': message.role,
                'content': message.content
            },
            'finish_reason': finish_reason
        }
    
    def _create_standard_usage(self,
                             prompt_tokens: int,
                             completion_tokens: int) -> Dict[str, int]:
        """创建标准化使用统计"""
        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }
    
    def _create_stream_chunk(self,
                           response_id: str,
                           model_name: str,
                           delta_content: str,
                           index: int = 0,
                           finish_reason: str = None) -> Dict[str, Any]:
        """创建流式响应块"""
        return {
            'id': response_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model_name,
            'choices': [{
                'index': index,
                'delta': {'content': delta_content} if delta_content else {},
                'finish_reason': finish_reason
            }]
        }
    
    def _log_request(self, model_name: str, messages_count: int, **kwargs):
        """记录请求日志"""
        self._logger.debug(f"Request to {model_name}: {messages_count} messages, kwargs: {kwargs}")
    
    def _log_response(self, model_name: str, response_time_ms: float, success: bool = True):
        """记录响应日志"""
        if success:
            self._logger.debug(f"Response from {model_name}: {response_time_ms:.2f}ms")
        else:
            self._logger.warning(f"Failed response from {model_name}: {response_time_ms:.2f}ms")
    
    def _extract_error_message(self, error: Exception) -> str:
        """提取错误信息"""
        error_msg = str(error)
        
        # 去除敏感信息
        sensitive_keywords = ['api_key', 'token', 'secret', 'password']
        for keyword in sensitive_keywords:
            if keyword in error_msg.lower():
                return "Authentication or authorization error"
        
        return error_msg
    
    def _validate_messages(self, messages: List[ChatMessage]):
        """验证消息格式"""
        if not messages:
            raise AdapterError("Messages cannot be empty")
        
        for i, message in enumerate(messages):
            if not message.content or not message.content.strip():
                raise AdapterError(f"Message {i} content cannot be empty")
            
            if message.role not in ['system', 'user', 'assistant']:
                raise AdapterError(f"Invalid message role: {message.role}")
    
    def _validate_model_config(self, model_config: ModelConfig):
        """验证模型配置"""
        if not model_config.api_endpoint:
            raise AdapterError("API endpoint is required")
        
        if model_config.max_tokens <= 0:
            raise AdapterError("Max tokens must be positive")
    
    def _validate_account_config(self, account_config: APIAccount):
        """验证账号配置"""
        if not account_config.api_key_encrypted:
            raise AdapterError("API key is required")
        
        if account_config.status.value != 'active':
            raise AdapterError(f"Account is not active: {account_config.status.value}")
    
    def _apply_request_limits(self, kwargs: Dict[str, Any], model_config: ModelConfig) -> Dict[str, Any]:
        """应用请求限制"""
        # 限制最大token数
        if 'max_tokens' in kwargs:
            kwargs['max_tokens'] = min(kwargs['max_tokens'], model_config.max_tokens)
        else:
            kwargs['max_tokens'] = min(1000, model_config.max_tokens)
        
        # 限制温度参数
        if 'temperature' in kwargs:
            kwargs['temperature'] = max(0, min(2, kwargs['temperature']))
        
        # 限制top_p参数
        if 'top_p' in kwargs:
            kwargs['top_p'] = max(0, min(1, kwargs['top_p']))
        
        return kwargs