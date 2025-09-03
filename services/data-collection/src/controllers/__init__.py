"""
API控制器包

包含所有REST API端点的控制器
"""

from .data_controller import router as data_router

__all__ = ["data_router"]