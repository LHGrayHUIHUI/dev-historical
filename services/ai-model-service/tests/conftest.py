"""
测试配置和夹具
"""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from src.models.ai_models import ModelConfig, APIAccount, ModelProvider, AccountStatus
from src.models.requests import ChatMessage


@pytest.fixture(scope="session")
def event_loop():
    """创建异步事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_model_config():
    """模拟模型配置"""
    return ModelConfig(
        id="test-model-1",
        name="Test GPT Model",
        model_id="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        description="测试模型",
        api_endpoint="https://api.openai.com/v1",
        max_tokens=4096,
        cost_per_1k_tokens=0.002,
        capabilities={
            "chat": True,
            "streaming": True,
            "function_calling": False
        },
        priority=5,
        is_active=True
    )


@pytest.fixture
def mock_account_config():
    """模拟账号配置"""
    return APIAccount(
        id="test-account-1",
        account_name="Test OpenAI Account",
        provider=ModelProvider.OPENAI,
        api_key_encrypted="test-encrypted-key",
        endpoint_url=None,
        organization_id=None,
        status=AccountStatus.ACTIVE,
        quota_limit=1000000,
        quota_used=50000,
        health_score=0.95,
        error_count=0,
        last_used_at=None,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00"
    )


@pytest.fixture
def mock_messages():
    """模拟聊天消息"""
    return [
        ChatMessage(role="system", content="你是一个有用的助手"),
        ChatMessage(role="user", content="你好，请介绍一下你自己")
    ]


@pytest.fixture
def mock_openai_response():
    """模拟OpenAI API响应"""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "你好！我是一个AI助手，我可以帮助你回答问题、提供信息、协助解决问题等。有什么我可以为你做的吗？"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": 50
        }
    }


@pytest.fixture
def mock_storage_client():
    """模拟存储客户端"""
    client = AsyncMock()
    
    # 模拟获取模型配置
    client.get_ai_models.return_value = [
        {
            "id": "test-model-1",
            "name": "Test GPT Model",
            "model_id": "gpt-3.5-turbo",
            "provider": "openai",
            "description": "测试模型",
            "api_endpoint": "https://api.openai.com/v1",
            "max_tokens": 4096,
            "cost_per_1k_tokens": 0.002,
            "capabilities": {"chat": True, "streaming": True},
            "priority": 5,
            "is_active": True
        }
    ]
    
    # 模拟获取账号配置
    client.get_api_accounts.return_value = [
        {
            "id": "test-account-1",
            "account_name": "Test OpenAI Account",
            "provider": "openai",
            "api_key_encrypted": "test-encrypted-key",
            "status": "active",
            "quota_limit": 1000000,
            "quota_used": 50000,
            "health_score": 0.95,
            "error_count": 0
        }
    ]
    
    # 模拟统计数据
    client.get_usage_statistics.return_value = {
        "data": {
            "total_requests": 100,
            "total_success": 95,
            "total_errors": 5,
            "success_rate": 0.95,
            "avg_response_time": 1200,
            "total_tokens": 5000,
            "total_cost": 10.0
        }
    }
    
    # 模拟路由策略
    client.get_routing_strategies.return_value = [
        {
            "name": "default",
            "strategy_type": "priority",
            "config": {}
        }
    ]
    
    return client


@pytest.fixture
def mock_httpx_client():
    """模拟HTTP客户端"""
    client = AsyncMock()
    
    # 模拟成功响应
    response = MagicMock()
    response.json.return_value = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "测试响应内容"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }
    response.raise_for_status = MagicMock()
    
    client.post.return_value = response
    
    return client


@pytest.fixture
def mock_redis_client():
    """模拟Redis客户端"""
    client = AsyncMock()
    client.ping.return_value = True
    client.get.return_value = None
    client.set.return_value = True
    client.incr.return_value = 1
    client.expire.return_value = True
    client.keys.return_value = []
    client.delete.return_value = 1
    
    return client


@pytest.fixture
async def mock_ai_service():
    """模拟AI服务"""
    service = AsyncMock()
    service.initialized = True
    
    # 模拟聊天完成
    service.chat_completion.return_value = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Test GPT Model",
        "provider": "openai",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "测试响应"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        },
        "response_time_ms": 1200,
        "routing_info": {
            "model_id": "test-model-1",
            "model_name": "Test GPT Model",
            "provider": "openai",
            "account_id": "test-account-1",
            "routing_strategy": "priority"
        }
    }
    
    # 模拟可用模型列表
    service.list_available_models.return_value = {
        "openai": [
            {
                "id": "test-model-1",
                "name": "Test GPT Model",
                "model_id": "gpt-3.5-turbo",
                "provider": "openai",
                "description": "测试模型"
            }
        ]
    }
    
    # 模拟服务状态
    service.get_service_status.return_value = {
        "initialized": True,
        "supported_providers": ["openai", "claude"],
        "adapter_instances": 1
    }
    
    return service


class AsyncContextManager:
    """异步上下文管理器助手"""
    
    def __init__(self, async_value):
        self.async_value = async_value
    
    async def __aenter__(self):
        return await self.async_value() if asyncio.iscoroutinefunction(self.async_value) else self.async_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """创建异步上下文管理器"""
    return AsyncContextManager