"""
API控制器模块

提供所有HTTP路由和控制器逻辑
处理请求验证、业务逻辑调用和响应格式化
"""

from .moderation_controller import router as moderation_router
from .admin_controller import router as admin_router
from .health_controller import router as health_router

__all__ = [
    "moderation_router",
    "admin_router", 
    "health_router"
]