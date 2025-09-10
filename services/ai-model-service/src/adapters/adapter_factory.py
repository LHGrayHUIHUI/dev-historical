"""
AI平台适配器工厂
"""

import logging
from typing import Dict, Type

from ..models.ai_models import ModelProvider
from .base_adapter import BaseAdapter
from .openai_adapter import OpenAIAdapter
from .claude_adapter import ClaudeAdapter
from .baidu_adapter import BaiduAdapter
from .alibaba_adapter import AlibabaAdapter
from .tencent_adapter import TencentAdapter
from .zhipu_adapter import ZhipuAdapter
from .gemini_adapter import GeminiAdapter


class AdapterFactory:
    """
    AI平台适配器工厂
    负责根据提供商创建对应的适配器实例
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        
        # 适配器类映射表
        self._adapter_classes: Dict[ModelProvider, Type[BaseAdapter]] = {
            ModelProvider.OPENAI: OpenAIAdapter,
            ModelProvider.CLAUDE: ClaudeAdapter,
            ModelProvider.BAIDU: BaiduAdapter,
            ModelProvider.ALIBABA: AlibabaAdapter,
            ModelProvider.TENCENT: TencentAdapter,
            ModelProvider.ZHIPU: ZhipuAdapter,
            ModelProvider.GEMINI: GeminiAdapter
        }
        
        # 适配器实例缓存
        self._adapter_instances: Dict[ModelProvider, BaseAdapter] = {}
    
    def get_adapter(self, provider: ModelProvider) -> BaseAdapter:
        """
        获取指定提供商的适配器实例
        
        Args:
            provider: AI提供商
            
        Returns:
            BaseAdapter: 对应的适配器实例
            
        Raises:
            ValueError: 当不支持的提供商时
        """
        if provider not in self._adapter_classes:
            supported_providers = list(self._adapter_classes.keys())
            raise ValueError(f"不支持的AI提供商: {provider}, 支持的提供商: {supported_providers}")
        
        # 使用单例模式，每个提供商只创建一个实例
        if provider not in self._adapter_instances:
            adapter_class = self._adapter_classes[provider]
            self._adapter_instances[provider] = adapter_class()
            self._logger.info(f"Created adapter instance for provider: {provider.value}")
        
        return self._adapter_instances[provider]
    
    def get_supported_providers(self) -> list[ModelProvider]:
        """
        获取所有支持的AI提供商列表
        
        Returns:
            List[ModelProvider]: 支持的提供商列表
        """
        return list(self._adapter_classes.keys())
    
    def is_provider_supported(self, provider: ModelProvider) -> bool:
        """
        检查是否支持指定的AI提供商
        
        Args:
            provider: AI提供商
            
        Returns:
            bool: 是否支持
        """
        return provider in self._adapter_classes
    
    def clear_cache(self):
        """清理适配器实例缓存"""
        self._adapter_instances.clear()
        self._logger.info("Adapter instance cache cleared")


# 全局适配器工厂实例
_adapter_factory_instance = None

def get_adapter_factory() -> AdapterFactory:
    """
    获取适配器工厂实例 (单例模式)
    
    Returns:
        AdapterFactory: 适配器工厂实例
    """
    global _adapter_factory_instance
    if _adapter_factory_instance is None:
        _adapter_factory_instance = AdapterFactory()
    return _adapter_factory_instance