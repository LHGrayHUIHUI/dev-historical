"""
平台适配器模块

提供各个社交媒体平台的统一API接口适配器
实现平台特定的API调用和数据转换逻辑
"""

from .base_adapter import BasePlatformAdapter
from .weibo_adapter import WeiboAdapter
from .wechat_adapter import WeChatAdapter
from .douyin_adapter import DouyinAdapter
from .toutiao_adapter import ToutiaoAdapter
from .baijiahao_adapter import BaijiahaoAdapter

__all__ = [
    'BasePlatformAdapter',
    'WeiboAdapter', 
    'WeChatAdapter',
    'DouyinAdapter',
    'ToutiaoAdapter',
    'BaijiahaoAdapter'
]