"""
简化的Gemini适配器（用于Docker测试）
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any

import httpx

from .base_adapter import BaseAdapter, AdapterError


class SimpleGeminiAdapter:
    """简化的Gemini适配器，专用于Docker测试"""
    
    def __init__(self):
        self.api_key = "AIzaSyCrpXFxpEbsKjrHOCQ0oR2dUtMRjys3_-w"
        self.api_base = "https://generativelanguage.googleapis.com/v1beta"
    
    def _convert_messages_to_gemini(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """将OpenAI格式消息转换为Gemini格式"""
        gemini_messages = []
        
        for message in messages:
            # 处理ChatMessage对象或字典
            if hasattr(message, 'role') and hasattr(message, 'content'):
                role = message.role
                content = message.content
            else:
                role = message.get("role")
                content = message.get("content", "")
            
            if role == "system":
                # Gemini doesn't have system role, merge with first user message
                if gemini_messages and gemini_messages[-1].get("role") == "user":
                    gemini_messages[-1]["parts"][0]["text"] = f"{content}\n\n{gemini_messages[-1]['parts'][0]['text']}"
                else:
                    gemini_messages.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
            elif role == "user":
                gemini_messages.append({
                    "role": "user", 
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",  # Gemini uses "model" for assistant
                    "parts": [{"text": content}]
                })
        
        return gemini_messages
    
    async def chat_completion(self, model_config: Any, account_config: Any, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Gemini聊天完成"""
        start_time = time.time()
        
        try:
            # 转换消息格式
            gemini_messages = self._convert_messages_to_gemini(messages)
            
            # 构建请求
            model_name = getattr(model_config, 'model_id', 'gemini-1.5-flash')
            url = f"{self.api_base}/models/{model_name}:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'x-goog-api-key': self.api_key
            }
            
            payload = {
                "contents": gemini_messages,
                "generationConfig": {
                    "maxOutputTokens": getattr(model_config, 'max_tokens', 1000),
                    "temperature": kwargs.get('temperature', 0.7),
                    "topP": kwargs.get('top_p', 1.0)
                }
            }
            
            # 发送请求
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 转换为OpenAI格式
                    choices = []
                    if 'candidates' in data and data['candidates']:
                        for i, candidate in enumerate(data['candidates']):
                            content = ""
                            if 'content' in candidate and 'parts' in candidate['content']:
                                content = candidate['content']['parts'][0].get('text', '')
                            
                            choices.append({
                                "index": i,
                                "message": {
                                    "role": "assistant",
                                    "content": content
                                },
                                "finish_reason": candidate.get('finishReason', 'stop').lower()
                            })
                    
                    # 使用统计
                    usage = {}
                    if 'usageMetadata' in data:
                        metadata = data['usageMetadata']
                        usage = {
                            "prompt_tokens": metadata.get('promptTokenCount', 0),
                            "completion_tokens": metadata.get('candidatesTokenCount', 0),
                            "total_tokens": metadata.get('totalTokenCount', 0)
                        }
                    
                    return {
                        "id": f"gemini-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "provider": "gemini",
                        "choices": choices,
                        "usage": usage,
                        "response_time_ms": response_time_ms
                    }
                
                else:
                    # 处理错误
                    error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {"error": response.text}
                    return {
                        "error": f"Gemini API error ({response.status_code}): {error_data}",
                        "status_code": response.status_code,
                        "response_time_ms": response_time_ms
                    }
                    
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return {
                "error": f"Request failed: {str(e)}",
                "response_time_ms": response_time_ms
            }
    
    async def chat_completion_stream(self, model_config: Any, account_config: Any, messages: List[Dict[str, Any]], **kwargs):
        """流式聊天完成（暂未实现）"""
        yield {"error": "Streaming not implemented in simplified adapter"}