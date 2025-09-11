"""
聊天API控制器 - Chat API Controller

提供标准的AI聊天完成接口，支持多种AI模型提供商
兼容OpenAI ChatGPT API格式，支持同步和流式响应

主要功能:
- 统一的聊天完成API接口
- 多AI提供商支持(Gemini、OpenAI、Claude等)
- 流式和非流式响应模式  
- 请求验证和错误处理
- 详细的中文日志记录
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from ..models.requests import ChatCompletionRequest, ChatCompletionRequest as ChatStreamRequest
from ..services.ai_service_simplified import get_ai_service


class ChatController:
    """
    聊天API控制器 - Chat API Controller
    
    核心功能说明:
    1. 提供标准聊天完成接口 - 兼容OpenAI API格式
    2. 支持多AI模型智能路由 - 自动选择最佳可用模型
    3. 流式和非流式响应 - 支持实时对话和批量处理
    4. 统一错误处理 - 提供中文错误信息和标准HTTP状态码
    5. 请求参数验证 - 确保请求格式正确和参数有效
    
    支持的AI提供商:
    - Google Gemini (gemini-1.5-flash, gemini-1.5-pro)  
    - OpenAI GPT (gpt-3.5-turbo, gpt-4, gpt-4-turbo)
    - Anthropic Claude (claude-3-haiku, claude-3-sonnet)
    - 本地模型 (通过Ollama、vLLM等部署)
    """
    
    def __init__(self):
        """
        初始化聊天控制器
        
        配置说明:
        - 路由前缀: /api/v1/chat (遵循RESTful API设计规范)
        - 标签: chat (用于OpenAPI文档自动分组)  
        - 日志记录器: 记录所有请求和响应信息，便于调试和监控
        """
        self.router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
        self._logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """
        设置聊天相关的API路由 - Setup Chat API Routes
        
        配置的端点:
        1. POST /completions - 标准聊天完成接口
        2. POST /completions/stream - 流式聊天完成接口
        
        每个端点都包含:
        - 详细的中文文档说明
        - 请求参数验证
        - 错误处理和日志记录
        - 标准的响应格式
        """
        
        @self.router.post("/completions", 
                           summary="AI聊天完成接口",
                           description="发送消息给AI模型，获取智能回复",
                           response_description="AI模型的完整响应，包含回复内容和元数据")
        async def chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]:
            """
            AI聊天完成接口 - Chat Completion API
            
            功能描述:
            向指定的AI模型发送对话消息，获取智能回复响应。
            支持多轮对话上下文，自动选择最佳可用模型。
            
            请求参数:
            - model: 指定AI模型名称 (如: gemini-1.5-flash, gpt-3.5-turbo)
            - messages: 对话消息列表，包含角色和内容
            - temperature: 生成随机性控制 (0.0-2.0，默认0.7)
            - max_tokens: 最大输出token数 (默认1000)
            - provider: 指定AI提供商 (可选，自动选择)
            
            响应格式:
            - id: 响应唯一标识符
            - model: 实际使用的模型名称
            - choices: AI回复选择列表
            - usage: token使用统计
            - metadata: 响应元数据(响应时间、提供商等)
            
            支持的消息角色:
            - system: 系统指令，设定AI行为
            - user: 用户消息，用户的输入
            - assistant: AI助手回复，对话历史
            
            错误处理:
            - 400: 请求参数无效
            - 500: 服务器内部错误
            - 详细错误信息以中文返回
            """
            try:
                self._logger.info(f"Received chat completion request: model={request.model}, "
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
                
            except Exception as ai_error:
                self._logger.error(f"AI service error: {ai_error}")
                raise HTTPException(status_code=400, detail=str(ai_error))
            
            except Exception as e:
                self._logger.error(f"Unexpected error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
        
        @self.router.post("/completions/stream",
                           summary="AI流式聊天接口", 
                           description="发送消息给AI模型，获取实时流式回复",
                           response_description="Server-Sent Events格式的实时AI响应流")
        async def chat_completions_stream(request: ChatStreamRequest):
            """
            AI流式聊天完成接口 - Streaming Chat Completion API
            
            功能描述:
            向指定的AI模型发送对话消息，获取实时流式回复响应。
            适用于需要实时显示AI生成内容的场景，如在线聊天。
            
            请求参数:
            - model: 指定AI模型名称 (如: gemini-1.5-flash, gpt-3.5-turbo)
            - messages: 对话消息列表，包含角色和内容  
            - temperature: 生成随机性控制 (0.0-2.0，默认0.7)
            - max_tokens: 最大输出token数 (默认1000)
            - provider: 指定AI提供商 (可选，自动选择)
            
            响应格式 (Server-Sent Events):
            - Content-Type: text/plain; charset=utf-8
            - 数据格式: data: {JSON块}\n\n
            - 结束标记: data: [DONE]\n\n
            
            流式数据块格式:
            - id: 数据块唯一标识
            - choices: 包含增量文本内容
            - delta: 本次增量的具体内容
            - finish_reason: 完成原因 (stop, length等)
            
            使用场景:
            - 实时聊天界面
            - 逐字显示AI回复
            - 长文本生成的进度显示
            - 提升用户交互体验
            
            错误处理:
            - 流中包含错误信息
            - 详细错误信息以中文返回
            - 连接异常自动重试机制
            """
            try:
                self._logger.info(f"Received stream chat request: model={request.model}, "
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
                
            except Exception as ai_error:
                self._logger.error(f"AI service error in stream: {ai_error}")
                raise HTTPException(status_code=400, detail=str(ai_error))
            
            except Exception as e:
                self._logger.error(f"Unexpected error in stream completion: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
    
    def _validate_chat_request(self, request: ChatCompletionRequest):
        """
        验证聊天完成请求参数 - Validate Chat Completion Request
        
        验证内容:
        1. 消息列表不能为空
        2. 每条消息内容不能为空
        3. 消息角色必须有效 (system/user/assistant)
        4. 参数范围验证 (temperature, top_p, max_tokens)
        
        参数:
            request: 聊天完成请求对象
            
        异常:
            HTTPException: 参数验证失败时抛出400错误
        """
        if not request.messages:
            raise HTTPException(status_code=400, detail="消息列表不能为空")
        
        # 验证消息格式
        for i, message in enumerate(request.messages):
            if not message.content or not message.content.strip():
                raise HTTPException(status_code=400, detail=f"消息{i}内容不能为空")
            
            if message.role not in ['system', 'user', 'assistant']:
                raise HTTPException(status_code=400, detail=f"无效的消息角色: {message.role}")
        
        # 验证参数范围
        if request.temperature and not 0 <= request.temperature <= 2:
            raise HTTPException(status_code=400, detail="temperature参数必须在0-2之间")
        
        if request.top_p and not 0 <= request.top_p <= 1:
            raise HTTPException(status_code=400, detail="top_p参数必须在0-1之间")
        
        if request.max_tokens and request.max_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_tokens参数必须大于0")
    
    def _validate_stream_request(self, request: ChatStreamRequest):
        """
        验证流式聊天请求参数 - Validate Streaming Chat Request
        
        使用与标准聊天请求相同的验证逻辑，确保流式请求参数有效性。
        将流式请求转换为标准请求格式进行验证。
        
        参数:
            request: 流式聊天请求对象
            
        异常:
            HTTPException: 参数验证失败时抛出400错误
        """
        # 使用相同的验证逻辑
        chat_request = ChatCompletionRequest(
            model=request.model,
            provider=request.provider,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )
        self._validate_chat_request(chat_request)
    
    def _format_sse_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        格式化服务器推送事件数据块 - Format Server-Sent Events Chunk
        
        将AI模型响应数据转换为符合SSE规范的JSON字符串格式。
        确保中文字符正确编码，处理JSON序列化异常。
        
        参数:
            chunk: 待格式化的响应数据块
            
        返回:
            str: JSON格式的字符串，支持中文字符
            
        异常处理:
            序列化失败时返回标准错误格式
        """
        import json
        try:
            return json.dumps(chunk, ensure_ascii=False)
        except Exception as e:
            self._logger.error(f"Failed to format SSE chunk: {e}")
            return json.dumps({"error": {"message": "格式化响应失败", "type": "format_error"}})


# 创建控制器实例 - Controller Factory Function
def create_chat_controller() -> ChatController:
    """
    创建聊天控制器实例 - Create Chat Controller Instance
    
    工厂函数，用于创建和初始化聊天API控制器。
    遵循依赖注入模式，便于测试和维护。
    
    返回:
        ChatController: 已配置的聊天控制器实例
        
    使用方式:
        在main.py中注册路由:
        chat_controller = create_chat_controller()
        app.include_router(chat_controller.router)
    """
    return ChatController()