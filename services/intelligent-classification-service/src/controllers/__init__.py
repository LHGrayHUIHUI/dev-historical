"""
智能分类服务控制器模块
提供所有API路由和控制器
"""

from .project_controller import router as project_router
from .model_controller import router as model_router
from .classification_controller import router as classification_router
from .data_controller import router as data_router

__all__ = [
    'project_router',
    'model_router', 
    'classification_router',
    'data_router'
]