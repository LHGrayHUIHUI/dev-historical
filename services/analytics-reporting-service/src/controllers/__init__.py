"""
分析报告服务控制器模块

包含所有FastAPI路由控制器，负责处理HTTP请求并调用相应的服务。
提供RESTful API接口用于数据分析、报告生成、模板管理等功能。
"""

from .analytics_controller import analytics_router
from .reports_controller import reports_router
from .templates_controller import templates_router
from .data_sources_controller import data_sources_router
from .alerts_controller import alerts_router
from .system_controller import system_router

__all__ = [
    "analytics_router",
    "reports_router", 
    "templates_router",
    "data_sources_router",
    "alerts_router",
    "system_router"
]