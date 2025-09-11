"""
平台适配器模块

提供各个社交媒体平台的统一发布接口
实现平台特定的认证、发布和数据获取逻辑
"""

from .base_adapter import PlatformAdapter
from .weibo_adapter import WeiboAdapter  
from .wechat_adapter import WechatAdapter
from .douyin_adapter import DouyinAdapter
from .toutiao_adapter import ToutiaoAdapter
from .baijiahao_adapter import BaijiahaoAdapter

__all__ = [
    "PlatformAdapter",
    "WeiboAdapter",
    "WechatAdapter", 
    "DouyinAdapter",
    "ToutiaoAdapter",
    "BaijiahaoAdapter"
]