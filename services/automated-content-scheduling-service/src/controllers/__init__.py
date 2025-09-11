"""
API控制器包初始化
导出所有控制器路由以便主应用使用
"""
from .scheduling_controller import router as scheduling_router
from .analytics_controller import router as analytics_router
from .system_controller import router as system_router

# 导出所有路由器
__all__ = [
    "scheduling_router",
    "analytics_router", 
    "system_router"
]