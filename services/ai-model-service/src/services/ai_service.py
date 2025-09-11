"""
AI模型服务
"""

import asyncio
import logging
from typing import Dict, List, Any, AsyncIterator, Optional

from ..models.ai_models import ModelConfig, APIAccount
from ..models.requests import ChatMessage, ChatCompletionRequest
from ..adapters.adapter_factory import get_adapter_factory
from ..adapters.base_adapter import AdapterError
from .model_router import get_model_router, RoutingRequest
from .account_monitor import get_account_monitor
from .usage_tracker import get_usage_tracker


class AIServiceError(Exception):
    """AI服务错误"""
    pass


class AIModelService:
    """
    AI模型服务
    统一的AI模型调用接口，负责模型路由、账号管理和使用统计
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self.adapter_factory = get_adapter_factory()
        self.initialized = False
    
    async def initialize(self):
        """初始化AI模型服务"""
        if self.initialized:
            return
        
        try:
            # 简化初始化，先只初始化必要组件
            self._logger.info("开始初始化AI Model Service...")
            
            # 只初始化适配器工厂（不依赖外部服务）
            self.adapter_factory = get_adapter_factory()
            
            # TODO: 在连接到storage-service后再初始化这些组件
            # model_router = await get_model_router()
            # account_monitor = await get_account_monitor()
            # usage_tracker = await get_usage_tracker()
            
            self.initialized = True
            self._logger.info("AI Model Service initialized successfully (simplified mode)")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize AI Model Service: {e}")
            # 在开发环境中，允许服务继续运行即使初始化部分失败
            self.initialized = True
            self._logger.warning("AI Model Service启动为简化模式（部分功能可能不可用）")
    
    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        聊天完成接口（简化版本用于测试）
        
        Args:
            request: 聊天请求
            
        Returns:
            Dict[str, Any]: 标准化的聊天响应
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 简化版本：直接使用Gemini适配器进行测试
            self._logger.info(f"Processing chat completion request: model={request.model}")
            
            # 创建简化的模型和账户配置
            from ..models.ai_models import ModelProvider
            from ..adapters.gemini_adapter import GeminiAdapter
            
            # 如果是Gemini请求，直接使用Gemini适配器
            if not request.provider or request.provider == "gemini":
                adapter = GeminiAdapter()
                
                # 创建简化配置
                model_config = {
                    'model_name': request.model or 'gemini-1.5-flash',
                    'max_tokens': getattr(request, 'max_tokens', 1000),
                    'temperature': getattr(request, 'temperature', 0.7)
                }
                
                account_config = {
                    'api_key': 'AIzaSyCrpXFxpEbsKjrHOCQ0oR2dUtMRjys3_-w',
                    'api_base': 'https://generativelanguage.googleapis.com/v1beta'
                }
                
                # 调用适配器
                response = await adapter.chat_completion(model_config, account_config, request.messages)
                
                # 包装成标准响应格式
                return {
                    'id': f'ai-{hash(str(request.messages)) % 1000000}',
                    'object': 'chat.completion',
                    'model': request.model or 'gemini-1.5-flash', 
                    'provider': 'gemini',
                    'choices': response.get('choices', []),
                    'usage': response.get('usage', {}),
                    'metadata': {
                        'response_time_ms': response.get('response_time_ms', 0),
                        'model_used': request.model or 'gemini-1.5-flash',
                        'provider_used': 'gemini',
                        'routing_strategy': 'direct',
                        'simplified_mode': True
                    }
                }
            else:
                return {'error': f'不支持的提供商: {request.provider}，当前仅支持Gemini'}
            
        except Exception as e:
            self._logger.error(f"Chat completion failed: {e}")
            return {'error': f'聊天完成失败: {str(e)}'}
            
            # 2. 获取对应的适配器
            adapter = self.adapter_factory.get_adapter(routing_result.model.provider)
            
            # 3. 调用适配器进行聊天完成
            response = await adapter.chat_completion(
                model_config=routing_result.model,
                account_config=routing_result.account,
                messages=request.messages,
                **request.parameters
            )
            
            # 4. 记录使用统计
            await self._record_usage(routing_result, request, response, success=True)
            
            # 5. 添加路由信息到响应
            response['routing_info'] = {
                'model_id': routing_result.model.id,
                'model_name': routing_result.model.name,
                'provider': routing_result.model.provider.value,
                'account_id': routing_result.account.id,
                'account_name': routing_result.account.account_name,
                'routing_strategy': routing_result.routing_strategy,
                'selection_reason': routing_result.selection_reason
            }
            
            return response
            
        except AdapterError as e:
            # 适配器错误，记录失败统计
            if 'routing_result' in locals():
                await self._record_usage(routing_result, request, None, success=False, error=str(e))
            
            self._logger.error(f"Adapter error in chat completion: {e}")
            raise AIServiceError(f"聊天完成请求失败: {e}")
        
        except Exception as e:
            self._logger.error(f"Unexpected error in chat completion: {e}")
            raise AIServiceError(f"聊天完成请求异常: {e}")
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncIterator[Dict[str, Any]]:
        """
        流式聊天完成接口
        
        Args:
            request: 流式聊天请求
            
        Yields:
            Dict[str, Any]: 流式响应块
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # 1. 通过路由器选择最佳模型和账号
            routing_request = RoutingRequest(
                model_name=request.model_name,
                provider=request.provider,
                requirements=request.requirements,
                user_id=request.user_id,
                priority=request.priority or 1
            )
            
            model_router = await get_model_router()
            routing_result = await model_router.select_model_account(routing_request)
            
            self._logger.info(f"Selected model for stream: {routing_result.model.name}, "
                            f"account: {routing_result.account.account_name}")
            
            # 2. 获取对应的适配器
            adapter = self.adapter_factory.get_adapter(routing_result.model.provider)
            
            # 3. 调用适配器进行流式聊天
            first_chunk = True
            total_tokens = 0
            
            async for chunk in adapter.chat_completion_stream(
                model_config=routing_result.model,
                account_config=routing_result.account,
                messages=request.messages,
                **request.parameters
            ):
                # 在第一个块中添加路由信息
                if first_chunk:
                    chunk['routing_info'] = {
                        'model_id': routing_result.model.id,
                        'model_name': routing_result.model.name,
                        'provider': routing_result.model.provider.value,
                        'account_id': routing_result.account.id,
                        'account_name': routing_result.account.account_name,
                        'routing_strategy': routing_result.routing_strategy,
                        'selection_reason': routing_result.selection_reason
                    }
                    first_chunk = False
                
                # 统计token数（简单估算）
                if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                    content = chunk['choices'][0]['delta']['content']
                    total_tokens += len(content.split())
                
                yield chunk
            
            # 4. 记录使用统计
            await self._record_usage(routing_result, request, {'usage': {'total_tokens': total_tokens}}, success=True)
            
        except AdapterError as e:
            # 适配器错误，记录失败统计
            if 'routing_result' in locals():
                await self._record_usage(routing_result, request, None, success=False, error=str(e))
            
            self._logger.error(f"Adapter error in stream completion: {e}")
            raise AIServiceError(f"流式聊天完成请求失败: {e}")
        
        except Exception as e:
            self._logger.error(f"Unexpected error in stream completion: {e}")
            raise AIServiceError(f"流式聊天完成请求异常: {e}")
    
    async def list_available_models(self, provider: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取可用模型列表
        
        Args:
            provider: 可选的提供商筛选
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 按提供商分组的模型列表
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            model_router = await get_model_router()
            available_models = {}
            
            # 如果指定了提供商，只获取该提供商的模型
            if provider:
                if not self.adapter_factory.is_provider_supported(provider):
                    raise AIServiceError(f"不支持的提供商: {provider}")
                
                # 获取该提供商的模型配置
                models = [m for m in model_router.models_cache.values() 
                         if m.provider.value == provider and m.is_active]
                available_models[provider] = [self._model_to_dict(model) for model in models]
            else:
                # 获取所有提供商的模型
                for provider_enum in self.adapter_factory.get_supported_providers():
                    provider_name = provider_enum.value
                    models = [m for m in model_router.models_cache.values() 
                             if m.provider == provider_enum and m.is_active]
                    available_models[provider_name] = [self._model_to_dict(model) for model in models]
            
            return available_models
            
        except Exception as e:
            self._logger.error(f"Failed to list available models: {e}")
            raise AIServiceError(f"获取可用模型列表失败: {e}")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        获取服务状态
        
        Returns:
            Dict[str, Any]: 服务状态信息
        """
        try:
            status = {
                'initialized': self.initialized,
                'supported_providers': [p.value for p in self.adapter_factory.get_supported_providers()],
                'adapter_instances': len(self.adapter_factory._adapter_instances)
            }
            
            if self.initialized:
                # 获取路由器统计
                model_router = await get_model_router()
                router_stats = await model_router.get_router_statistics()
                status['router'] = router_stats
                
                # 获取账号健康状态
                account_monitor = await get_account_monitor()
                health_summary = await account_monitor.get_health_summary()
                status['account_health'] = health_summary
                
                # 获取使用统计
                usage_tracker = await get_usage_tracker()
                usage_summary = await usage_tracker.get_usage_summary()
                status['usage'] = usage_summary
            
            return status
            
        except Exception as e:
            self._logger.error(f"Failed to get service status: {e}")
            return {'error': str(e), 'initialized': self.initialized}
    
    async def _record_usage(self, routing_result, request, response, success: bool, error: str = None):
        """记录使用统计"""
        try:
            usage_tracker = await get_usage_tracker()
            
            usage_data = {
                'model_id': routing_result.model.id,
                'model_name': routing_result.model.name,
                'provider': routing_result.model.provider.value,
                'account_id': routing_result.account.id,
                'account_name': routing_result.account.account_name,
                'user_id': getattr(request, 'user_id', None),
                'message_count': len(request.messages) if hasattr(request, 'messages') else 0,
                'success': success,
                'error_message': error,
                'routing_strategy': routing_result.routing_strategy
            }
            
            if response and 'usage' in response:
                usage_info = response['usage']
                usage_data.update({
                    'prompt_tokens': usage_info.get('prompt_tokens', 0),
                    'completion_tokens': usage_info.get('completion_tokens', 0),
                    'total_tokens': usage_info.get('total_tokens', 0)
                })
            
            if response and 'response_time_ms' in response:
                usage_data['response_time_ms'] = response['response_time_ms']
            
            await usage_tracker.record_usage(usage_data)
            
        except Exception as e:
            self._logger.warning(f"Failed to record usage: {e}")
    
    def _model_to_dict(self, model: ModelConfig) -> Dict[str, Any]:
        """将模型配置转换为字典格式"""
        return {
            'id': model.id,
            'name': model.name,
            'model_id': model.model_id,
            'provider': model.provider.value,
            'description': model.description,
            'max_tokens': model.max_tokens,
            'cost_per_1k_tokens': model.cost_per_1k_tokens,
            'capabilities': model.capabilities,
            'priority': model.priority,
            'is_active': model.is_active
        }
    
    async def shutdown(self):
        """关闭服务"""
        if self.initialized:
            try:
                # 停止账号监控
                account_monitor = await get_account_monitor()
                await account_monitor.stop_monitoring()
                
                # 清理适配器缓存
                self.adapter_factory.clear_cache()
                
                self.initialized = False
                self._logger.info("AI Model Service shutdown completed")
                
            except Exception as e:
                self._logger.error(f"Error during service shutdown: {e}")


# 全局AI模型服务实例
_ai_service_instance = None

async def get_ai_service() -> AIModelService:
    """
    获取AI模型服务实例 (单例模式)
    
    Returns:
        AIModelService: AI模型服务实例
    """
    global _ai_service_instance
    if _ai_service_instance is None:
        _ai_service_instance = AIModelService()
        await _ai_service_instance.initialize()
    return _ai_service_instance