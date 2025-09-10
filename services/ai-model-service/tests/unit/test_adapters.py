"""
适配器单元测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.claude_adapter import ClaudeAdapter
from src.adapters.base_adapter import AdapterError
from src.models.ai_models import ModelProvider


class TestOpenAIAdapter:
    """OpenAI适配器测试"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return OpenAIAdapter()
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, adapter, mock_model_config, mock_account_config, mock_messages):
        """测试成功的聊天完成"""
        # 模拟OpenAI客户端
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-test123"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].index = 0
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "测试响应"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch.object(adapter, '_get_client', return_value=mock_client):
            result = await adapter.chat_completion(
                model_config=mock_model_config,
                account_config=mock_account_config,
                messages=mock_messages,
                temperature=0.7,
                max_tokens=100
            )
        
        # 验证结果
        assert result['id'] == "chatcmpl-test123"
        assert result['object'] == 'chat.completion'
        assert result['model'] == mock_model_config.name
        assert result['provider'] == 'openai'
        assert len(result['choices']) == 1
        assert result['choices'][0]['message']['content'] == "测试响应"
        assert result['usage']['prompt_tokens'] == 10
        assert result['usage']['completion_tokens'] == 5
        assert result['usage']['total_tokens'] == 15
        
        # 验证客户端调用
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args['model'] == mock_model_config.model_id
        assert call_args['temperature'] == 0.7
        assert call_args['max_tokens'] == 100
        assert call_args['stream'] is False
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_invalid_messages(self, adapter, mock_model_config, mock_account_config):
        """测试无效消息的处理"""
        # 空消息列表
        with pytest.raises(AdapterError, match="Messages cannot be empty"):
            await adapter.chat_completion(
                model_config=mock_model_config,
                account_config=mock_account_config,
                messages=[],
            )
    
    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, adapter, mock_model_config, mock_account_config, mock_messages):
        """测试API错误处理"""
        import openai
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(),
            body=None
        )
        
        with patch.object(adapter, '_get_client', return_value=mock_client):
            with pytest.raises(AdapterError, match="请求频率限制"):
                await adapter.chat_completion(
                    model_config=mock_model_config,
                    account_config=mock_account_config,
                    messages=mock_messages
                )
    
    @pytest.mark.asyncio
    async def test_chat_completion_stream(self, adapter, mock_model_config, mock_account_config, mock_messages):
        """测试流式聊天完成"""
        # 模拟流式响应
        async def mock_stream_chunks():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="你好"), finish_reason=None)]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="！"), finish_reason=None)]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=""), finish_reason="stop")])
            ]
            for chunk in chunks:
                yield chunk
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_stream_chunks()
        
        with patch.object(adapter, '_get_client', return_value=mock_client):
            chunks = []
            async for chunk in adapter.chat_completion_stream(
                model_config=mock_model_config,
                account_config=mock_account_config,
                messages=mock_messages
            ):
                chunks.append(chunk)
        
        # 验证流式响应
        assert len(chunks) == 3
        assert chunks[0]['choices'][0]['delta']['content'] == "你好"
        assert chunks[1]['choices'][0]['delta']['content'] == "！"
        assert chunks[2]['choices'][0]['finish_reason'] == "stop"
    
    @pytest.mark.asyncio
    async def test_list_models(self, adapter, mock_account_config):
        """测试获取模型列表"""
        # 模拟模型列表响应
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-3.5-turbo"
        mock_model1.created = 1700000000
        
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-4"
        mock_model2.created = 1700000000
        
        mock_response = MagicMock()
        mock_response.data = [mock_model1, mock_model2]
        
        mock_client = AsyncMock()
        mock_client.models.list.return_value = mock_response
        
        with patch.object(adapter, '_get_client', return_value=mock_client):
            models = await adapter.list_models(mock_account_config)
        
        # 验证模型列表
        assert len(models) == 2
        assert models[0]['id'] == "gpt-3.5-turbo"
        assert models[0]['provider'] == "openai"
        assert models[1]['id'] == "gpt-4"
        assert models[1]['provider'] == "openai"
    
    def test_convert_messages_to_openai(self, adapter, mock_messages):
        """测试消息格式转换"""
        openai_messages = adapter._convert_messages_to_openai(mock_messages)
        
        assert len(openai_messages) == 2
        assert openai_messages[0]['role'] == 'system'
        assert openai_messages[0]['content'] == '你是一个有用的助手'
        assert openai_messages[1]['role'] == 'user'
        assert openai_messages[1]['content'] == '你好，请介绍一下你自己'


class TestClaudeAdapter:
    """Claude适配器测试"""
    
    @pytest.fixture
    def adapter(self):
        """创建适配器实例"""
        return ClaudeAdapter()
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, adapter, mock_model_config, mock_account_config, mock_messages):
        """测试成功的聊天完成"""
        # 修改模型配置为Claude
        mock_model_config.provider = ModelProvider.CLAUDE
        mock_model_config.model_id = "claude-3-sonnet"
        
        # 模拟Claude API响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "msg_test123",
            "type": "message", 
            "role": "assistant",
            "content": [{"type": "text", "text": "你好！我是Claude，一个AI助手。"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 20
            }
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        with patch.object(adapter, '_get_client', return_value=mock_client):
            result = await adapter.chat_completion(
                model_config=mock_model_config,
                account_config=mock_account_config,
                messages=mock_messages,
                temperature=0.7
            )
        
        # 验证结果
        assert result['id'] == "msg_test123"
        assert result['provider'] == 'claude'
        assert len(result['choices']) == 1
        assert result['choices'][0]['message']['content'] == "你好！我是Claude，一个AI助手。"
        assert result['usage']['prompt_tokens'] == 15
        assert result['usage']['completion_tokens'] == 20
        
        # 验证API调用
        mock_client.post.assert_called_once_with('/v1/messages', json=pytest.any)
        call_args = mock_client.post.call_args[1]['json']
        assert call_args['model'] == "claude-3-sonnet"
        assert call_args['temperature'] == 0.7
        assert call_args['stream'] is False
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message(self, adapter, mock_model_config, mock_account_config):
        """测试系统消息处理"""
        mock_model_config.provider = ModelProvider.CLAUDE
        
        messages = [
            MagicMock(role="system", content="你是一个专业的助手"),
            MagicMock(role="user", content="你好")
        ]
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "msg_test123",
            "content": [{"type": "text", "text": "你好"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        with patch.object(adapter, '_get_client', return_value=mock_client):
            await adapter.chat_completion(
                model_config=mock_model_config,
                account_config=mock_account_config,
                messages=messages
            )
        
        # 验证系统消息被正确处理
        call_args = mock_client.post.call_args[1]['json']
        assert 'system' in call_args
        assert call_args['system'] == "你是一个专业的助手"
        # messages中不应该包含system消息
        assert all(msg['role'] != 'system' for msg in call_args['messages'])
    
    @pytest.mark.asyncio
    async def test_chat_completion_http_error(self, adapter, mock_model_config, mock_account_config, mock_messages):
        """测试HTTP错误处理"""
        mock_model_config.provider = ModelProvider.CLAUDE
        
        mock_client = AsyncMock()
        error_response = MagicMock()
        error_response.status_code = 401
        error_response.json.return_value = {
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        }
        
        http_error = httpx.HTTPStatusError(
            message="401 Client Error",
            request=MagicMock(),
            response=error_response
        )
        mock_client.post.side_effect = http_error
        
        with patch.object(adapter, '_get_client', return_value=mock_client):
            with pytest.raises(AdapterError, match="认证失败"):
                await adapter.chat_completion(
                    model_config=mock_model_config,
                    account_config=mock_account_config,
                    messages=mock_messages
                )
    
    def test_convert_messages_to_claude(self, adapter, mock_messages):
        """测试消息格式转换"""
        claude_messages = adapter._convert_messages_to_claude(mock_messages)
        
        assert len(claude_messages) == 2
        assert claude_messages[0]['role'] == 'system'
        assert claude_messages[0]['content'] == '你是一个有用的助手'
        assert claude_messages[1]['role'] == 'user'
        assert claude_messages[1]['content'] == '你好，请介绍一下你自己'


class TestAdapterFactory:
    """适配器工厂测试"""
    
    @pytest.fixture
    def factory(self):
        """创建工厂实例"""
        from src.adapters.adapter_factory import AdapterFactory
        return AdapterFactory()
    
    def test_get_adapter_openai(self, factory):
        """测试获取OpenAI适配器"""
        adapter = factory.get_adapter(ModelProvider.OPENAI)
        assert isinstance(adapter, OpenAIAdapter)
    
    def test_get_adapter_claude(self, factory):
        """测试获取Claude适配器"""
        adapter = factory.get_adapter(ModelProvider.CLAUDE)
        assert isinstance(adapter, ClaudeAdapter)
    
    def test_get_adapter_singleton(self, factory):
        """测试单例模式"""
        adapter1 = factory.get_adapter(ModelProvider.OPENAI)
        adapter2 = factory.get_adapter(ModelProvider.OPENAI)
        assert adapter1 is adapter2
    
    def test_get_supported_providers(self, factory):
        """测试获取支持的提供商"""
        providers = factory.get_supported_providers()
        assert ModelProvider.OPENAI in providers
        assert ModelProvider.CLAUDE in providers
        assert ModelProvider.BAIDU in providers
        assert len(providers) == 6  # 6个支持的提供商
    
    def test_is_provider_supported(self, factory):
        """测试提供商支持检查"""
        assert factory.is_provider_supported(ModelProvider.OPENAI)
        assert factory.is_provider_supported(ModelProvider.CLAUDE)
    
    def test_clear_cache(self, factory):
        """测试清理缓存"""
        # 先获取一个适配器，确保缓存中有数据
        factory.get_adapter(ModelProvider.OPENAI)
        assert len(factory._adapter_instances) == 1
        
        # 清理缓存
        factory.clear_cache()
        assert len(factory._adapter_instances) == 0