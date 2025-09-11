"""
API控制器模块

提供多平台账号管理服务的RESTful API接口
包含账号管理、OAuth认证、权限控制等功能的控制器
"""

from .account_controller import router as account_router
from .oauth_controller import router as oauth_router
from .sync_controller import router as sync_router
from .permission_controller import router as permission_router
from .system_controller import router as system_router

__all__ = [
    'account_router',
    'oauth_router', 
    'sync_router',
    'permission_controller',
    'system_router'
]