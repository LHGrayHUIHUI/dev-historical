"""
多平台账号管理服务主应用

FastAPI应用程序入口，配置中间件、路由、数据库连接等
提供多平台社交媒体账号的统一管理API服务
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config.settings import settings
from .models.database import create_all_tables, close_database_connection
from .controllers import (
    account_router, oauth_router, sync_router, 
    permission_router, system_router
)
from .utils.exceptions import AccountManagementError, create_error_response

# 配置日志
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时执行
    logger.info("多平台账号管理服务启动中...")
    
    try:
        # 创建数据库表
        await create_all_tables()
        logger.info("数据库表创建完成")
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("多平台账号管理服务关闭中...")
    
    try:
        # 关闭数据库连接
        await close_database_connection()
        logger.info("数据库连接已关闭")
        
    except Exception as e:
        logger.error(f"应用关闭时出错: {e}")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.app_name,
    description="""
    多平台账号管理服务 - 统一管理多个社交媒体平台的账号

    ## 功能特性
    
    * **多平台支持**: 支持微博、微信、抖音、头条、百家号等主流平台
    * **OAuth认证**: 完整的OAuth 2.0认证流程
    * **账号管理**: 账号的添加、更新、删除、查询
    * **数据同步**: 支持账号信息和内容的定时同步
    * **权限控制**: 细粒度的账号访问权限管理
    * **安全加密**: 敏感数据采用AES-256加密存储
    * **API监控**: 完整的API调用统计和监控
    
    ## 支持的平台
    
    * 🐦 新浪微博 (Weibo)
    * 💬 微信公众号 (WeChat)
    * 🎵 抖音 (Douyin) 
    * 📰 今日头条 (Toutiao)
    * 📝 百家号 (Baijiahao)
    
    ## 认证方式
    
    所有API接口都需要通过OAuth 2.0认证获取访问令牌。
    """,
    version="1.0.0",
    contact={
        "name": "历史文本项目团队",
        "email": "support@historical-text.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "账号管理",
            "description": "社交媒体账号的CRUD操作和统计查询"
        },
        {
            "name": "OAuth认证", 
            "description": "OAuth 2.0认证流程和令牌管理"
        },
        {
            "name": "数据同步",
            "description": "账号数据的同步和更新操作"
        },
        {
            "name": "权限管理",
            "description": "账号访问权限的授权和控制"
        },
        {
            "name": "系统监控",
            "description": "服务状态监控和API统计"
        }
    ],
    docs_url="/docs" if settings.debug else None,  # 生产环境可关闭文档
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 配置受信任主机中间件
if settings.allowed_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )


# 全局异常处理器
@app.exception_handler(AccountManagementError)
async def account_management_exception_handler(request, exc: AccountManagementError):
    """处理账号管理相关异常"""
    logger.error(f"账号管理异常: {exc.message}")
    
    from .utils.exceptions import get_http_status_code
    status_code = get_http_status_code(exc)
    
    return JSONResponse(
        status_code=status_code,
        content=create_error_response(exc)
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """处理HTTP异常"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "error": "HTTPException",
                "message": str(exc.detail),
                "error_code": f"HTTP_{exc.status_code}",
                "details": {}
            },
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """处理通用异常"""
    logger.error(f"未处理异常: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "error": "InternalServerError",
                "message": "服务器内部错误，请稍后重试",
                "error_code": "INTERNAL_ERROR",
                "details": {"exception_type": type(exc).__name__}
            },
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "status_code": 500
        }
    )


# 注册路由
app.include_router(account_router)
app.include_router(oauth_router)
app.include_router(sync_router)
app.include_router(permission_router)
app.include_router(system_router)


# 根路径处理
@app.get(
    "/",
    tags=["系统信息"],
    summary="服务基本信息",
    description="获取多平台账号管理服务的基本信息"
)
async def root():
    """服务根路径"""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "description": "多平台账号管理服务 - 统一管理多个社交媒体平台的账号",
        "status": "running",
        "supported_platforms": ["weibo", "wechat", "douyin", "toutiao", "baijiahao"],
        "documentation": "/docs" if settings.debug else "已禁用",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


# 健康检查端点
@app.get(
    "/health",
    tags=["系统监控"],
    summary="健康检查",
    description="检查服务的健康状态"
)
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# 就绪检查端点（Kubernetes探针）
@app.get(
    "/ready",
    tags=["系统监控"], 
    summary="就绪检查",
    description="检查服务是否准备就绪接收请求"
)
async def readiness_check():
    """就绪检查"""
    # 这里可以添加数据库连接检查等逻辑
    try:
        # 简单的就绪检查
        return {
            "status": "ready",
            "service": settings.app_name,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "database": "connected",  # 实际项目中应该检查真实连接
            "redis": "connected"      # 实际项目中应该检查真实连接
        }
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务暂时不可用"
        )


# 应用程序入口点
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=settings.debug
    )