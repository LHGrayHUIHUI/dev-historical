"""
控制器模块初始化

导出所有API控制器
"""

from .publishing_controller import router as publishing_router
from .platforms_controller import router as platforms_router
from .accounts_controller import router as accounts_router
from .health_controller import router as health_router

__all__ = [
    "publishing_router",
    "platforms_router", 
    "accounts_router",
    "health_router"
]