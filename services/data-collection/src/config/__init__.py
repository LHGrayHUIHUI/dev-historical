"""
配置管理包

提供应用程序配置和设置管理
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]