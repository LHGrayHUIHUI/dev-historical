"""
聊天API控制器
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from ..models.requests import ChatRequest, ChatStreamRequest
from ..services.ai_service import get_ai_service, AIServiceError


class ChatController:
    """聊天API控制器"""
    
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
        self._logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.post("/completions")
        async def chat_completions(request: ChatRequest) -> Dict[str, Any]:
            """
            聊天完成接口
            
            接收聊天请求，返回AI模型的响应
            """
            try:
                self._logger.info(f"Received chat completion request: model={request.model_name}, "
                                f"provider={request.provider}, messages_count={len(request.messages)}")
                
                # 验证请求
                self._validate_chat_request(request)
                
                # 调用AI服务
                ai_service = await get_ai_service()
                response = await ai_service.chat_completion(request)
                
                self._logger.info(f"Chat completion successful: "
                                f"response_id={response.get('id')}, "
                                f"response_time={response.get('response_time_ms')}ms")
                
                return response
                
            except AIServiceError as e:
                self._logger.error(f"AI service error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            
            except Exception as e:
                self._logger.error(f"Unexpected error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
        
        @self.router.post("/completions/stream")
        async def chat_completions_stream(request: ChatStreamRequest):
            """
            流式聊天完成接口
            
            接收聊天请求，返回流式AI模型响应
            """
            try:
                self._logger.info(f"Received stream chat request: model={request.model_name}, "
                                f"provider={request.provider}, messages_count={len(request.messages)}")
                
                # 验证请求
                self._validate_stream_request(request)
                
                # 调用AI服务
                ai_service = await get_ai_service()
                
                # 创建流式响应生成器
                async def stream_generator():
                    try:
                        async for chunk in ai_service.chat_completion_stream(request):
                            # 格式化为SSE格式
                            chunk_json = self._format_sse_chunk(chunk)
                            yield f"data: {chunk_json}\n\n"
                        
                        # 发送结束标记
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        # 在流中发送错误
                        error_chunk = {"error": {"message": str(e), "type": "stream_error"}}
                        error_json = self._format_sse_chunk(error_chunk)
                        yield f"data: {error_json}\n\n"
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/plain; charset=utf-8"
                    }
                )
                
            except AIServiceError as e:
                self._logger.error(f"AI service error in stream: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            
            except Exception as e:
                self._logger.error(f"Unexpected error in stream completion: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
    
    def _validate_chat_request(self, request: ChatRequest):
        """验证聊天请求"""
        if not request.messages:
            raise HTTPException(status_code=400, detail="消息列表不能为空")
        
        # 验证消息格式
        for i, message in enumerate(request.messages):
            if not message.content or not message.content.strip():
                raise HTTPException(status_code=400, detail=f"消息{i}内容不能为空")
            
            if message.role not in ['system', 'user', 'assistant']:
                raise HTTPException(status_code=400, detail=f"无效的消息角色: {message.role}")
        
        # 验证参数范围
        if 'temperature' in request.parameters:
            temp = request.parameters['temperature']
            if not 0 <= temp <= 2:
                raise HTTPException(status_code=400, detail="temperature参数必须在0-2之间")
        
        if 'top_p' in request.parameters:
            top_p = request.parameters['top_p']
            if not 0 <= top_p <= 1:
                raise HTTPException(status_code=400, detail="top_p参数必须在0-1之间")
        
        if 'max_tokens' in request.parameters:
            max_tokens = request.parameters['max_tokens']
            if max_tokens <= 0:
                raise HTTPException(status_code=400, detail="max_tokens参数必须大于0")
    
    def _validate_stream_request(self, request: ChatStreamRequest):
        """验证流式请求"""
        # 使用相同的验证逻辑
        chat_request = ChatRequest(
            model_name=request.model_name,
            provider=request.provider,
            messages=request.messages,
            parameters=request.parameters,
            requirements=request.requirements,
            user_id=request.user_id,
            priority=request.priority
        )
        self._validate_chat_request(chat_request)
    
    def _format_sse_chunk(self, chunk: Dict[str, Any]) -> str:
        """格式化SSE数据块"""
        import json
        try:
            return json.dumps(chunk, ensure_ascii=False)
        except Exception as e:
            self._logger.error(f"Failed to format SSE chunk: {e}")
            return json.dumps({"error": {"message": "格式化响应失败", "type": "format_error"}})


# 创建控制器实例
def create_chat_controller() -> ChatController:
    """创建聊天控制器实例"""
    return ChatController()