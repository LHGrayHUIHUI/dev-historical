"""
AI服务单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.ai_service import AIModelService, AIServiceError
from src.models.requests import ChatRequest, ChatStreamRequest, ChatMessage
from src.models.ai_models import ModelProvider
from src.services.model_router import RoutingResult


class TestAIModelService:
    """AI模型服务测试"""
    
    @pytest.fixture
    async def service(self, mock_storage_client):
        """创建服务实例"""
        service = AIModelService()
        # 模拟初始化完成
        service.initialized = True
        return service
    
    @pytest.fixture
    def chat_request(self, mock_messages):
        """创建聊天请求"""
        return ChatRequest(
            model_name="Test GPT Model",
            provider="openai",
            messages=mock_messages,
            parameters={
                "temperature": 0.7,
                "max_tokens": 100
            },
            user_id="test-user"
        )
    
    @pytest.fixture
    def stream_request(self, mock_messages):
        """创建流式请求"""
        return ChatStreamRequest(
            model_name="Test GPT Model",
            provider="openai",
            messages=mock_messages,
            parameters={
                "temperature": 0.7,
                "max_tokens": 100
            },
            user_id="test-user"
        )
    
    @pytest.fixture
    def mock_routing_result(self, mock_model_config, mock_account_config):
        """模拟路由结果"""
        return RoutingResult(
            model=mock_model_config,
            account=mock_account_config,
            routing_strategy="priority",
            selection_reason="高健康评分"
        )
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, service, chat_request, mock_routing_result):
        """测试成功的聊天完成"""
        # 模拟路由器
        mock_router = AsyncMock()
        mock_router.select_model_account.return_value = mock_routing_result
        
        # 模拟适配器响应
        mock_adapter_response = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "model": "Test GPT Model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "你好！我是测试助手。"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            },
            "response_time_ms": 1200
        }
        
        # 模拟适配器
        mock_adapter = AsyncMock()
        mock_adapter.chat_completion.return_value = mock_adapter_response
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router), \
             patch.object(service.adapter_factory, 'get_adapter', return_value=mock_adapter), \
             patch.object(service, '_record_usage') as mock_record_usage:
            
            result = await service.chat_completion(chat_request)
        
        # 验证结果
        assert result["id"] == "chatcmpl-test123"
        assert result["choices"][0]["message"]["content"] == "你好！我是测试助手。"
        assert "routing_info" in result
        assert result["routing_info"]["model_name"] == "Test GPT Model"
        assert result["routing_info"]["routing_strategy"] == "priority"
        
        # 验证方法调用
        mock_router.select_model_account.assert_called_once()
        mock_adapter.chat_completion.assert_called_once_with(
            model_config=mock_routing_result.model,
            account_config=mock_routing_result.account,
            messages=chat_request.messages,
            temperature=0.7,
            max_tokens=100
        )
        mock_record_usage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_completion_routing_error(self, service, chat_request):
        """测试路由错误处理"""
        # 模拟路由器抛出异常
        mock_router = AsyncMock()
        mock_router.select_model_account.side_effect = Exception("没有可用的模型")
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router):
            with pytest.raises(AIServiceError, match="聊天完成请求异常"):
                await service.chat_completion(chat_request)
    
    @pytest.mark.asyncio
    async def test_chat_completion_adapter_error(self, service, chat_request, mock_routing_result):
        """测试适配器错误处理"""
        # 模拟路由器成功
        mock_router = AsyncMock()
        mock_router.select_model_account.return_value = mock_routing_result
        
        # 模拟适配器错误
        from src.adapters.base_adapter import AdapterError
        mock_adapter = AsyncMock()
        mock_adapter.chat_completion.side_effect = AdapterError("API调用失败")
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router), \
             patch.object(service.adapter_factory, 'get_adapter', return_value=mock_adapter), \
             patch.object(service, '_record_usage') as mock_record_usage:
            
            with pytest.raises(AIServiceError, match="聊天完成请求失败"):
                await service.chat_completion(chat_request)
            
            # 验证记录了失败的使用统计
            mock_record_usage.assert_called_once()
            call_args = mock_record_usage.call_args[1]
            assert call_args['success'] is False
            assert call_args['error'] == "API调用失败"
    
    @pytest.mark.asyncio
    async def test_chat_completion_stream_success(self, service, stream_request, mock_routing_result):
        """测试流式聊天完成"""
        # 模拟路由器
        mock_router = AsyncMock()
        mock_router.select_model_account.return_value = mock_routing_result
        
        # 模拟流式响应
        async def mock_stream():
            chunks = [
                {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "你好"}, "finish_reason": None}]
                },
                {
                    "id": "chatcmpl-test123", 
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "！"}, "finish_reason": None}]
                },
                {
                    "id": "chatcmpl-test123",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]
                }
            ]
            for chunk in chunks:
                yield chunk
        
        mock_adapter = AsyncMock()
        mock_adapter.chat_completion_stream.return_value = mock_stream()
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router), \
             patch.object(service.adapter_factory, 'get_adapter', return_value=mock_adapter), \
             patch.object(service, '_record_usage') as mock_record_usage:
            
            chunks = []
            async for chunk in service.chat_completion_stream(stream_request):
                chunks.append(chunk)
        
        # 验证流式响应
        assert len(chunks) == 3
        
        # 第一个块应该包含路由信息
        assert "routing_info" in chunks[0]
        assert chunks[0]["routing_info"]["model_name"] == "Test GPT Model"
        
        # 验证内容
        assert chunks[0]["choices"][0]["delta"]["content"] == "你好"
        assert chunks[1]["choices"][0]["delta"]["content"] == "！"
        assert chunks[2]["choices"][0]["finish_reason"] == "stop"
        
        # 验证记录了使用统计
        mock_record_usage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_available_models_all(self, service):
        """测试获取所有可用模型"""
        # 模拟路由器缓存
        mock_model_1 = MagicMock()
        mock_model_1.id = "model-1"
        mock_model_1.name = "OpenAI Model"
        mock_model_1.provider = ModelProvider.OPENAI
        mock_model_1.is_active = True
        
        mock_model_2 = MagicMock()
        mock_model_2.id = "model-2" 
        mock_model_2.name = "Claude Model"
        mock_model_2.provider = ModelProvider.CLAUDE
        mock_model_2.is_active = True
        
        mock_router = MagicMock()
        mock_router.models_cache = {
            "model-1": mock_model_1,
            "model-2": mock_model_2
        }
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router), \
             patch.object(service, '_model_to_dict', side_effect=lambda m: {"id": m.id, "name": m.name}):
            
            models = await service.list_available_models()
        
        # 验证结果
        assert "openai" in models
        assert "claude" in models
        assert len(models["openai"]) == 1
        assert len(models["claude"]) == 1
        assert models["openai"][0]["name"] == "OpenAI Model"
        assert models["claude"][0]["name"] == "Claude Model"
    
    @pytest.mark.asyncio
    async def test_list_available_models_provider_filter(self, service):
        """测试按提供商筛选模型"""
        mock_model = MagicMock()
        mock_model.id = "model-1"
        mock_model.name = "OpenAI Model"
        mock_model.provider = ModelProvider.OPENAI
        mock_model.is_active = True
        
        mock_router = MagicMock()
        mock_router.models_cache = {"model-1": mock_model}
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router), \
             patch.object(service, '_model_to_dict', return_value={"id": "model-1", "name": "OpenAI Model"}):
            
            models = await service.list_available_models(provider="openai")
        
        # 验证结果
        assert "openai" in models
        assert len(models["openai"]) == 1
        assert "claude" not in models
    
    @pytest.mark.asyncio
    async def test_list_available_models_unsupported_provider(self, service):
        """测试不支持的提供商"""
        with patch.object(service.adapter_factory, 'is_provider_supported', return_value=False):
            with pytest.raises(AIServiceError, match="不支持的提供商"):
                await service.list_available_models(provider="unsupported")
    
    @pytest.mark.asyncio
    async def test_get_service_status(self, service):
        """测试获取服务状态"""
        # 模拟各种依赖的状态
        mock_router = AsyncMock()
        mock_router.get_router_statistics.return_value = {"models_count": 5}
        
        mock_monitor = AsyncMock()
        mock_monitor.get_health_summary.return_value = {"total_accounts": 3}
        
        mock_tracker = AsyncMock() 
        mock_tracker.get_usage_summary.return_value = {"total_requests": 100}
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router), \
             patch('src.services.ai_service.get_account_monitor', return_value=mock_monitor), \
             patch('src.services.ai_service.get_usage_tracker', return_value=mock_tracker):
            
            status = await service.get_service_status()
        
        # 验证状态信息
        assert status["initialized"] is True
        assert "supported_providers" in status
        assert "router" in status
        assert "account_health" in status
        assert "usage" in status
        assert status["router"]["models_count"] == 5
        assert status["account_health"]["total_accounts"] == 3
        assert status["usage"]["total_requests"] == 100
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """测试服务初始化"""
        service = AIModelService()
        assert service.initialized is False
        
        # 模拟依赖初始化
        mock_router = AsyncMock()
        mock_monitor = AsyncMock()
        mock_tracker = AsyncMock()
        
        with patch('src.services.ai_service.get_model_router', return_value=mock_router), \
             patch('src.services.ai_service.get_account_monitor', return_value=mock_monitor), \
             patch('src.services.ai_service.get_usage_tracker', return_value=mock_tracker):
            
            await service.initialize()
        
        assert service.initialized is True
        mock_monitor.start_monitoring.assert_called_once()
        mock_tracker.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_shutdown(self, service):
        """测试服务关闭"""
        mock_monitor = AsyncMock()
        
        with patch('src.services.ai_service.get_account_monitor', return_value=mock_monitor):
            await service.shutdown()
        
        assert service.initialized is False
        mock_monitor.stop_monitoring.assert_called_once()
        service.adapter_factory.clear_cache.assert_called_once() if hasattr(service.adapter_factory, 'clear_cache') else None
    
    def test_model_to_dict(self, service, mock_model_config):
        """测试模型配置转换"""
        result = service._model_to_dict(mock_model_config)
        
        assert result["id"] == mock_model_config.id
        assert result["name"] == mock_model_config.name
        assert result["model_id"] == mock_model_config.model_id
        assert result["provider"] == mock_model_config.provider.value
        assert result["description"] == mock_model_config.description
        assert result["max_tokens"] == mock_model_config.max_tokens
        assert result["cost_per_1k_tokens"] == mock_model_config.cost_per_1k_tokens
        assert result["capabilities"] == mock_model_config.capabilities
        assert result["priority"] == mock_model_config.priority
        assert result["is_active"] == mock_model_config.is_active
    
    @pytest.mark.asyncio
    async def test_record_usage(self, service, mock_routing_result, chat_request):
        """测试使用统计记录"""
        mock_response = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "response_time_ms": 1200
        }
        
        mock_tracker = AsyncMock()
        
        with patch('src.services.ai_service.get_usage_tracker', return_value=mock_tracker):
            await service._record_usage(
                routing_result=mock_routing_result,
                request=chat_request,
                response=mock_response,
                success=True
            )
        
        # 验证调用参数
        mock_tracker.record_usage.assert_called_once()
        call_args = mock_tracker.record_usage.call_args[0][0]
        
        assert call_args["model_id"] == mock_routing_result.model.id
        assert call_args["model_name"] == mock_routing_result.model.name
        assert call_args["provider"] == mock_routing_result.model.provider.value
        assert call_args["account_id"] == mock_routing_result.account.id
        assert call_args["user_id"] == chat_request.user_id
        assert call_args["success"] is True
        assert call_args["prompt_tokens"] == 10
        assert call_args["completion_tokens"] == 5
        assert call_args["total_tokens"] == 15
        assert call_args["response_time_ms"] == 1200