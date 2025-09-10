"""
AI平台适配器包
"""

from .base_adapter import BaseAdapter, AdapterError
from .openai_adapter import OpenAIAdapter
from .claude_adapter import ClaudeAdapter
from .baidu_adapter import BaiduAdapter
from .alibaba_adapter import AlibabaAdapter
from .tencent_adapter import TencentAdapter
from .zhipu_adapter import ZhipuAdapter
from .gemini_adapter import GeminiAdapter

__all__ = [
    "BaseAdapter",
    "AdapterError",
    "OpenAIAdapter", 
    "ClaudeAdapter",
    "BaiduAdapter",
    "AlibabaAdapter", 
    "TencentAdapter",
    "ZhipuAdapter",
    "GeminiAdapter"
]