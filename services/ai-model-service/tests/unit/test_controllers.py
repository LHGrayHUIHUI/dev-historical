"""
控制器单元测试
"""

import pytest
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.main import app
from src.controllers.chat_controller import ChatController
from src.controllers.models_controller import ModelsController
from src.controllers.status_controller import StatusController
from src.services.ai_service import AIServiceError


class TestChatController:
    """聊天控制器测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def controller(self):
        """创建控制器实例"""
        return ChatController()
    
    @pytest.fixture
    def valid_chat_request(self):
        """有效的聊天请求"""
        return {
            "model_name": "Test GPT Model",
            "provider": "openai",
            "messages": [
                {"role": "system", "content": "你是一个有用的助手"},
                {"role": "user", "content": "你好"}
            ],
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 100
            },
            "user_id": "test-user"
        }
    
    @pytest.mark.asyncio
    async def test_chat_completions_success(self, client, valid_chat_request):
        """测试成功的聊天完成"""
        mock_response = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "model": "Test GPT Model",
            "provider": "openai",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant", 
                        "content": "你好！我是AI助手。"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25
            },
            "routing_info": {
                "model_name": "Test GPT Model",
                "provider": "openai",
                "routing_strategy": "priority"
            }
        }
        
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.chat_completion.return_value = mock_response
            mock_get_service.return_value = mock_service
            
            response = client.post("/api/v1/chat/completions", json=valid_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-test123"
        assert data["choices"][0]["message"]["content"] == "你好！我是AI助手。"
        assert "routing_info" in data
    
    def test_chat_completions_invalid_request(self, client):
        """测试无效请求"""
        # 空消息列表
        invalid_request = {
            "model_name": "Test Model",
            "messages": [],
            "parameters": {}
        }
        
        response = client.post("/api/v1/chat/completions", json=invalid_request)
        assert response.status_code == 400
        assert "消息列表不能为空" in response.json()["detail"]
    
    def test_chat_completions_invalid_temperature(self, client, valid_chat_request):
        """测试无效的temperature参数"""
        valid_chat_request["parameters"]["temperature"] = 3.0  # 超出范围
        
        response = client.post("/api/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 400
        assert "temperature参数必须在0-2之间" in response.json()["detail"]
    
    def test_chat_completions_invalid_role(self, client, valid_chat_request):
        """测试无效的消息角色"""
        valid_chat_request["messages"][0]["role"] = "invalid_role"
        
        response = client.post("/api/v1/chat/completions", json=valid_chat_request)
        assert response.status_code == 400
        assert "无效的消息角色" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_chat_completions_service_error(self, client, valid_chat_request):
        """测试服务错误"""
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.chat_completion.side_effect = AIServiceError("模型服务不可用")
            mock_get_service.return_value = mock_service
            
            response = client.post("/api/v1/chat/completions", json=valid_chat_request)
        
        assert response.status_code == 400
        assert "模型服务不可用" in response.json()["detail"]
    
    def test_validate_chat_request(self, controller):
        """测试聊天请求验证"""
        from src.models.requests import ChatRequest, ChatMessage
        
        # 有效请求
        valid_request = ChatRequest(
            model_name="test",
            messages=[ChatMessage(role="user", content="hello")],
            parameters={"temperature": 0.7, "top_p": 0.9, "max_tokens": 100}
        )
        
        # 应该不抛出异常
        controller._validate_chat_request(valid_request)
        
        # 无效请求 - 空消息
        invalid_request = ChatRequest(
            model_name="test",
            messages=[],
            parameters={}
        )
        
        with pytest.raises(HTTPException, match="消息列表不能为空"):
            controller._validate_chat_request(invalid_request)
    
    def test_format_sse_chunk(self, controller):
        """测试SSE数据块格式化"""
        chunk = {
            "id": "test123",
            "choices": [{"delta": {"content": "hello"}}]
        }
        
        formatted = controller._format_sse_chunk(chunk)
        parsed = json.loads(formatted)
        
        assert parsed["id"] == "test123"
        assert parsed["choices"][0]["delta"]["content"] == "hello"
    
    def test_format_sse_chunk_error(self, controller):
        """测试SSE格式化错误处理"""
        # 创建一个无法JSON序列化的对象
        invalid_chunk = {"data": object()}
        
        formatted = controller._format_sse_chunk(invalid_chunk)
        parsed = json.loads(formatted)
        
        assert "error" in parsed
        assert parsed["error"]["type"] == "format_error"


class TestModelsController:
    """模型控制器测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_list_models_all(self, client):
        """测试获取所有模型"""
        mock_models = {
            "openai": [
                {
                    "id": "model-1",
                    "name": "GPT-3.5",
                    "provider": "openai",
                    "description": "OpenAI模型"
                }
            ],
            "claude": [
                {
                    "id": "model-2", 
                    "name": "Claude-3",
                    "provider": "claude",
                    "description": "Anthropic模型"
                }
            ]
        }
        
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_available_models.return_value = mock_models
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/models/")
        
        assert response.status_code == 200
        data = response.json()
        assert "openai" in data
        assert "claude" in data
        assert len(data["openai"]) == 1
        assert data["openai"][0]["name"] == "GPT-3.5"
    
    def test_list_models_with_provider_filter(self, client):
        """测试按提供商筛选模型"""
        mock_models = {
            "openai": [
                {
                    "id": "model-1",
                    "name": "GPT-3.5", 
                    "provider": "openai"
                }
            ]
        }
        
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_available_models.return_value = mock_models
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/models/?provider=openai")
        
        assert response.status_code == 200
        data = response.json()
        assert "openai" in data
        mock_service.list_available_models.assert_called_once_with(provider="openai")
    
    def test_list_models_invalid_provider(self, client):
        """测试无效提供商"""
        response = client.get("/api/v1/models/?provider=invalid")
        assert response.status_code == 400
        assert "不支持的提供商" in response.json()["detail"]
    
    def test_list_providers(self, client):
        """测试获取提供商列表"""
        from src.models.ai_models import ModelProvider
        
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.adapter_factory.get_supported_providers.return_value = [
                ModelProvider.OPENAI,
                ModelProvider.CLAUDE
            ]
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/models/providers")
        
        assert response.status_code == 200
        data = response.json()
        assert "supported_providers" in data
        assert "openai" in data["supported_providers"]
        assert "claude" in data["supported_providers"]
        assert "provider_details" in data
    
    def test_list_provider_models(self, client):
        """测试获取特定提供商的模型"""
        mock_models = {
            "openai": [
                {"id": "model-1", "name": "GPT-3.5"}
            ]
        }
        
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_available_models.return_value = mock_models
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/models/openai/models")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "GPT-3.5"
    
    def test_list_provider_models_not_found(self, client):
        """测试不存在的提供商模型"""
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_available_models.return_value = {}
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/models/nonexistent/models")
        
        assert response.status_code == 404
        assert "提供商 nonexistent 不存在" in response.json()["detail"]
    
    def test_get_model_capabilities(self, client):
        """测试获取模型能力说明"""
        response = client.get("/api/v1/models/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert "common_capabilities" in data
        assert "provider_capabilities" in data
        assert "parameter_explanations" in data
        
        # 检查关键内容
        assert "chat" in data["common_capabilities"]
        assert "openai" in data["provider_capabilities"]
        assert "temperature" in data["parameter_explanations"]


class TestStatusController:
    """状态控制器测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    def test_health_check_healthy(self, client):
        """测试健康状态检查 - 健康状态"""
        mock_status = {
            "initialized": True,
            "supported_providers": ["openai", "claude"]
        }
        
        with patch('src.services.ai_service.get_ai_service') as mock_get_service, \
             patch('src.services.model_router.get_model_router') as mock_get_router, \
             patch('src.services.usage_tracker.get_usage_tracker') as mock_get_tracker:
            
            mock_service = AsyncMock()
            mock_service.get_service_status.return_value = mock_status
            mock_get_service.return_value = mock_service
            
            mock_router = AsyncMock()
            mock_router.get_router_statistics.return_value = {"models_count": 5}
            mock_get_router.return_value = mock_router
            
            mock_tracker = AsyncMock()
            mock_tracker.is_running = True
            mock_tracker.batch_records = []
            mock_get_tracker.return_value = mock_tracker
            
            response = client.get("/api/v1/status/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai-model-service"
        assert "checks" in data
        assert data["checks"]["ai_service"]["status"] == "healthy"
    
    def test_health_check_unhealthy(self, client):
        """测试健康状态检查 - 不健康状态"""
        with patch('src.services.ai_service.get_ai_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_service_status.side_effect = Exception("服务异常")
            mock_get_service.return_value = mock_service
            
            response = client.get("/api/v1/status/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data or data["checks"]["ai_service"]["status"] == "unhealthy"
    
    def test_get_metrics(self, client):
        """测试获取服务指标"""
        mock_service_status = {"initialized": True}
        mock_usage_summary = {"total_requests": 100}
        mock_health_summary = {"total_accounts": 5}
        
        with patch('src.services.ai_service.get_ai_service') as mock_get_service, \
             patch('src.services.usage_tracker.get_usage_tracker') as mock_get_tracker, \
             patch('src.services.account_monitor.get_account_monitor') as mock_get_monitor:
            
            mock_service = AsyncMock()
            mock_service.get_service_status.return_value = mock_service_status
            mock_get_service.return_value = mock_service
            
            mock_tracker = AsyncMock() 
            mock_tracker.get_usage_summary.return_value = mock_usage_summary
            mock_get_tracker.return_value = mock_tracker
            
            mock_monitor = AsyncMock()
            mock_monitor.get_health_summary.return_value = mock_health_summary
            mock_get_monitor.return_value = mock_monitor
            
            response = client.get("/api/v1/status/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "ai-model-service"
        assert "metrics" in data
        assert data["metrics"]["ai_service"]["initialized"] is True
    
    def test_get_usage_stats_valid_period(self, client):
        """测试获取使用统计 - 有效周期"""
        mock_usage_data = {
            "period": "24h",
            "total_requests": 150,
            "success_rate": 0.96
        }
        
        with patch('src.services.usage_tracker.get_usage_tracker') as mock_get_tracker:
            mock_tracker = AsyncMock()
            mock_tracker.get_usage_summary.return_value = mock_usage_data
            mock_get_tracker.return_value = mock_tracker
            
            response = client.get("/api/v1/status/usage?period=24h&provider=openai")
        
        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "24h"
        assert data["total_requests"] == 150
        mock_tracker.get_usage_summary.assert_called_once_with(
            period="24h",
            provider="openai",
            model_id=None
        )
    
    def test_get_usage_stats_invalid_period(self, client):
        """测试获取使用统计 - 无效周期"""
        response = client.get("/api/v1/status/usage?period=invalid")
        assert response.status_code == 400
        assert "无效的统计周期" in response.json()["detail"]
    
    def test_get_performance_metrics(self, client):
        """测试获取性能指标"""
        mock_performance_data = {
            "period": "24h",
            "avg_response_time": 1200.5,
            "p95_response_time": 2000.0
        }
        
        with patch('src.services.usage_tracker.get_usage_tracker') as mock_get_tracker:
            mock_tracker = AsyncMock()
            mock_tracker.get_performance_metrics.return_value = mock_performance_data
            mock_get_tracker.return_value = mock_tracker
            
            response = client.get("/api/v1/status/performance?period=24h")
        
        assert response.status_code == 200
        data = response.json()
        assert data["avg_response_time"] == 1200.5
        assert data["p95_response_time"] == 2000.0
    
    def test_get_cost_analysis_valid_params(self, client):
        """测试获取成本分析 - 有效参数"""
        mock_cost_data = {
            "period": "30d",
            "group_by": "provider",
            "total_cost": 125.50
        }
        
        with patch('src.services.usage_tracker.get_usage_tracker') as mock_get_tracker:
            mock_tracker = AsyncMock()
            mock_tracker.get_cost_analysis.return_value = mock_cost_data
            mock_get_tracker.return_value = mock_tracker
            
            response = client.get("/api/v1/status/cost?period=30d&group_by=provider")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_cost"] == 125.50
        assert data["group_by"] == "provider"
    
    def test_get_cost_analysis_invalid_group_by(self, client):
        """测试获取成本分析 - 无效分组方式"""
        response = client.get("/api/v1/status/cost?group_by=invalid")
        assert response.status_code == 400
        assert "无效的分组方式" in response.json()["detail"]
    
    def test_get_account_status(self, client):
        """测试获取账号状态"""
        mock_health_data = {
            "total_accounts": 8,
            "average_health_score": 0.85,
            "status_distribution": {
                "active": 6,
                "error": 2
            }
        }
        
        with patch('src.services.account_monitor.get_account_monitor') as mock_get_monitor:
            mock_monitor = AsyncMock()
            mock_monitor.get_health_summary.return_value = mock_health_data
            mock_get_monitor.return_value = mock_monitor
            
            response = client.get("/api/v1/status/accounts")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_accounts"] == 8
        assert data["average_health_score"] == 0.85