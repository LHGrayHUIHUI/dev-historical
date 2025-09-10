"""
Google Gemini 平台适配器
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


class GeminiAdapter(BaseAdapter):
    """Google Gemini平台适配器"""
    
    def __init__(self):
        super().__init__()
        self.clients = {}  # 账号ID -> 客户端实例的缓存
    
    def _get_client(self, account_config: APIAccount) -> httpx.AsyncClient:
        """获取Gemini客户端实例"""
        account_id = account_config.id
        
        if account_id not in self.clients:
            api_key = self._decrypt_api_key(account_config.api_key_encrypted)
            
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': api_key
            }
            
            # 自定义端点URL (默认使用官方API)
            base_url = account_config.endpoint_url or 'https://generativelanguage.googleapis.com'
            
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
        """Gemini聊天完成接口"""
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
            
            # 转换消息格式为Gemini格式
            gemini_contents = self._convert_messages_to_gemini(messages)
            
            # 构建请求参数
            request_params = {
                'contents': gemini_contents
            }
            
            # 添加生成配置
            generation_config = {}
            if kwargs.get('temperature') is not None:
                generation_config['temperature'] = kwargs['temperature']
            
            if kwargs.get('max_tokens'):
                generation_config['maxOutputTokens'] = kwargs['max_tokens']
            
            if kwargs.get('top_p') is not None:
                generation_config['topP'] = kwargs['top_p']
            
            if kwargs.get('stop'):
                generation_config['stopSequences'] = kwargs['stop']
            
            if generation_config:
                request_params['generationConfig'] = generation_config
            
            # 确定API端点
            endpoint = f"/v1beta/models/{model_config.model_id}:generateContent"
            
            # 发起请求
            response = await client.post(endpoint, json=request_params)
            response.raise_for_status()
            
            response_data = response.json()
            response_time_ms = (time.time() - start_time) * 1000
            
            # 检查响应中的错误
            if 'error' in response_data:
                error_info = response_data['error']
                error_msg = error_info.get('message', '未知错误')
                self._log_response(model_config.name, response_time_ms, False)
                raise AdapterError(f"Gemini API错误: {error_msg}")
            
            self._log_response(model_config.name, response_time_ms, True)
            
            # 转换为标准格式
            return self._convert_gemini_response_to_standard(
                response_data, model_config, response_time_ms
            )
            
        except httpx.HTTPStatusError as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            
            error_msg = self._extract_gemini_error(e)
            if e.response.status_code == 400:
                self._logger.error(f"Gemini bad request: {error_msg}")
                raise AdapterError(f"请求参数错误: {error_msg}")
            elif e.response.status_code == 401:
                self._logger.error(f"Gemini authentication error: {error_msg}")
                raise AdapterError(f"认证失败: {error_msg}")
            elif e.response.status_code == 429:
                self._logger.error(f"Gemini rate limit: {error_msg}")
                raise AdapterError(f"请求频率限制: {error_msg}")
            else:
                self._logger.error(f"Gemini HTTP error: {e.response.status_code} - {error_msg}")
                raise AdapterError(f"Gemini请求失败: {error_msg}")
        
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._log_response(model_config.name, response_time_ms, False)
            self._logger.error(f"Gemini request failed: {e}")
            raise AdapterError(f"Gemini请求失败: {self._extract_error_message(e)}")
    
    async def chat_completion_stream(self,
                                   model_config: ModelConfig,
                                   account_config: APIAccount,
                                   messages: List[ChatMessage],
                                   **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Gemini流式聊天完成接口"""
        # 验证输入
        self._validate_messages(messages)
        self._validate_model_config(model_config)
        self._validate_account_config(account_config)
        
        # 应用限制
        kwargs = self._apply_request_limits(kwargs, model_config)
        
        try:
            client = self._get_client(account_config)
            
            # 转换消息格式
            gemini_contents = self._convert_messages_to_gemini(messages)
            
            # 构建请求参数
            request_params = {
                'contents': gemini_contents
            }
            
            # 添加生成配置
            generation_config = {}
            if kwargs.get('temperature') is not None:
                generation_config['temperature'] = kwargs['temperature']
            
            if kwargs.get('max_tokens'):
                generation_config['maxOutputTokens'] = kwargs['max_tokens']
            
            if kwargs.get('top_p') is not None:
                generation_config['topP'] = kwargs['top_p']
            
            if generation_config:
                request_params['generationConfig'] = generation_config
            
            # 确定流式API端点
            endpoint = f"/v1beta/models/{model_config.model_id}:streamGenerateContent"
            
            response_id = f"gemini-{uuid.uuid4().hex[:10]}"
            
            # 发起流式请求
            async with client.stream('POST', endpoint, json=request_params) as response:
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
                                raise AdapterError(f"Gemini流式API错误: {error_msg}")
                            
                            # 解析候选响应
                            candidates = chunk_data.get('candidates', [])
                            if candidates and candidates[0].get('content'):
                                content = candidates[0]['content']
                                parts = content.get('parts', [])
                                if parts and parts[0].get('text'):
                                    text = parts[0]['text']
                                    finish_reason = candidates[0].get('finishReason')
                                    
                                    yield self._create_stream_chunk(
                                        response_id=response_id,
                                        model_name=model_config.name,
                                        delta_content=text,
                                        finish_reason=finish_reason.lower() if finish_reason else None
                                    )
                                    
                                    if finish_reason:
                                        break
                                
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self._logger.error(f"Gemini stream request failed: {e}")
            raise AdapterError(f"Gemini流式请求失败: {self._extract_error_message(e)}")
    
    async def list_models(self, account_config: APIAccount) -> List[Dict[str, Any]]:
        """获取Gemini可用模型列表"""
        try:
            # Gemini的已知模型列表
            known_models = [
                {
                    'id': 'gemini-2.0-flash',
                    'name': 'Gemini 2.0 Flash',
                    'description': 'Gemini 2.0 Flash - 最新快速版本',
                    'provider': 'gemini',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'gemini-1.5-pro',
                    'name': 'Gemini 1.5 Pro',
                    'description': 'Gemini 1.5 Pro - 专业版本',
                    'provider': 'gemini',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'gemini-1.5-flash',
                    'name': 'Gemini 1.5 Flash',
                    'description': 'Gemini 1.5 Flash - 快速版本',
                    'provider': 'gemini',
                    'type': 'chat',
                    'created': None
                },
                {
                    'id': 'gemini-pro',
                    'name': 'Gemini Pro',
                    'description': 'Gemini Pro - 基础版本',
                    'provider': 'gemini',
                    'type': 'chat',
                    'created': None
                }
            ]
            
            return known_models
            
        except Exception as e:
            self._logger.error(f"Failed to list Gemini models: {e}")
            raise AdapterError(f"获取Gemini模型列表失败: {self._extract_error_message(e)}")
    
    def _convert_messages_to_gemini(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换消息格式为Gemini格式"""
        gemini_contents = []
        
        for message in messages:
            # Gemini使用role: "user" 或 "model" (相当于assistant)
            role = message.role
            if role == 'assistant':
                role = 'model'
            elif role == 'system':
                # 系统消息转换为用户消息
                role = 'user'
            
            gemini_content = {
                'role': role,
                'parts': [
                    {
                        'text': message.content
                    }
                ]
            }
            
            gemini_contents.append(gemini_content)
        
        return gemini_contents
    
    def _convert_gemini_response_to_standard(self, 
                                           response_data: Dict[str, Any], 
                                           model_config: ModelConfig, 
                                           response_time_ms: float) -> Dict[str, Any]:
        """转换Gemini响应为标准格式"""
        choices = []
        
        candidates = response_data.get('candidates', [])
        for i, candidate in enumerate(candidates):
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            
            # 合并所有parts的文本
            text_content = ''
            for part in parts:
                if part.get('text'):
                    text_content += part['text']
            
            message = ChatMessage(
                role='assistant',
                content=text_content
            )
            
            finish_reason = candidate.get('finishReason', 'STOP')
            # 转换Gemini的finish_reason为OpenAI格式
            if finish_reason == 'STOP':
                finish_reason = 'stop'
            elif finish_reason == 'MAX_TOKENS':
                finish_reason = 'length'
            elif finish_reason == 'SAFETY':
                finish_reason = 'content_filter'
            else:
                finish_reason = 'stop'
            
            standard_choice = self._create_standard_choice(
                index=i,
                message=message,
                finish_reason=finish_reason
            )
            
            choices.append(standard_choice)
        
        # 使用量统计
        usage = {}
        usage_metadata = response_data.get('usageMetadata', {})
        if usage_metadata:
            usage = self._create_standard_usage(
                prompt_tokens=usage_metadata.get('promptTokenCount', 0),
                completion_tokens=usage_metadata.get('candidatesTokenCount', 0)
            )
        
        return self._create_standard_response(
            response_id=f"gemini-{uuid.uuid4().hex[:10]}",
            model_name=model_config.name,
            provider_name='gemini',
            choices=choices,
            usage=usage,
            response_time_ms=response_time_ms
        )
    
    def _extract_gemini_error(self, error: httpx.HTTPStatusError) -> str:
        """提取Gemini API错误信息"""
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