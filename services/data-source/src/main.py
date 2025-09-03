"""
数据源服务主应用
FastAPI应用入口，集成所有API路由和中间件
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from loguru import logger
import signal
from datetime import datetime

# 导入配置和依赖
from .config.settings import get_settings
from .database.database import init_database, close_database, get_database_manager
from .crawler.crawler_manager import get_crawler_manager
from .proxy.proxy_manager import get_proxy_manager

# 导入API路由
from .api.crawler import router as crawler_router
from .api.content import router as content_router
from .api.proxy import router as proxy_router

# 获取配置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("🚀 数据源服务启动中...")
    
    try:
        # 初始化数据库连接
        logger.info("📊 初始化数据库连接...")
        await init_database()
        
        # 初始化爬虫管理器
        logger.info("🕷️ 初始化爬虫管理器...")
        crawler_manager = await get_crawler_manager()
        await crawler_manager.initialize()
        
        # 初始化代理管理器
        logger.info("🌐 初始化代理管理器...")
        proxy_manager = await get_proxy_manager()
        await proxy_manager.initialize()
        
        # 注册信号处理器
        def signal_handler(signum, frame):
            logger.info(f"收到退出信号 {signum}")
            asyncio.create_task(graceful_shutdown())
        
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        
        logger.success("✅ 数据源服务启动完成")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    
    finally:
        # 关闭时执行
        logger.info("🛑 数据源服务关闭中...")
        await graceful_shutdown()


async def graceful_shutdown():
    """优雅关闭"""
    try:
        # 停止爬虫管理器
        logger.info("停止爬虫管理器...")
        crawler_manager = await get_crawler_manager()
        await crawler_manager.cleanup()
        
        # 关闭数据库连接
        logger.info("关闭数据库连接...")
        await close_database()
        
        logger.success("✅ 服务已优雅关闭")
        
    except Exception as e:
        logger.error(f"❌ 关闭服务时发生错误: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="历史文本项目 - 数据源服务",
    description="""
    ## 数据源服务 API
    
    提供多平台内容爬取、代理管理和内容处理功能的微服务。
    
    ### 主要功能
    - 🕷️ **爬虫管理**: 支持今日头条、百家号、小红书等平台
    - 📄 **内容管理**: 手动添加、批量导入、查询和更新内容
    - 🌐 **代理管理**: 代理池管理、自动测试和轮换
    - 📊 **监控统计**: 实时状态监控和数据统计
    
    ### 技术栈
    - **框架**: FastAPI + Python 3.9+
    - **数据库**: MongoDB + Redis
    - **爬虫**: Scrapy + Selenium
    - **代理**: 多供应商代理池
    
    ### 使用说明
    1. 使用 `/crawlers/` 接口管理爬虫任务
    2. 使用 `/content/` 接口管理内容数据
    3. 使用 `/proxy/` 接口管理代理设置
    4. 查看 `/health` 检查服务状态
    """,
    version="1.0.0",
    contact={
        "name": "历史文本项目团队",
        "email": "support@historical-text.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": f"http://localhost:{settings.service.port}",
            "description": "开发环境"
        },
        {
            "url": "https://api.historical-text.com",
            "description": "生产环境"
        }
    ],
    docs_url=settings.service.docs_url,
    redoc_url="/redoc",
    openapi_url=settings.service.openapi_url,
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.service.cors_origins,
    allow_credentials=True,
    allow_methods=settings.service.cors_methods,
    allow_headers=["*"],
)

# 配置信任主机中间件（生产环境）
if settings.service.environment.value == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.historical-text.com", "localhost"]
    )


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"全局异常: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "服务器内部错误",
                "details": str(exc) if settings.is_development() else None,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        }
    )


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = datetime.now()
    
    # 记录请求信息
    logger.info(f"📥 {request.method} {request.url} - {request.client.host if request.client else 'unknown'}")
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = (datetime.now() - start_time).total_seconds()
    
    # 记录响应信息
    logger.info(f"📤 {request.method} {request.url} - {response.status_code} ({process_time:.3f}s)")
    
    # 添加响应头
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Service-Name"] = settings.service.service_name
    response.headers["X-Service-Version"] = settings.service.service_version
    
    return response


# 注册API路由
app.include_router(crawler_router, prefix=settings.service.api_prefix)
app.include_router(content_router, prefix=settings.service.api_prefix)
app.include_router(proxy_router, prefix=settings.service.api_prefix)


# 根路径
@app.get("/", tags=["系统"])
async def root():
    """服务根路径"""
    return {
        "success": True,
        "data": {
            "service": settings.service.service_name,
            "version": settings.service.service_version,
            "environment": settings.service.environment,
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "docs_url": settings.service.docs_url,
            "api_prefix": settings.service.api_prefix
        },
        "message": "数据源服务运行中"
    }


# 健康检查
@app.get("/health", tags=["系统"])
async def health_check():
    """服务健康检查"""
    try:
        # 检查数据库连接
        db_manager = await get_database_manager()
        db_health = await db_manager.health_check()
        
        # 检查爬虫管理器
        crawler_manager = await get_crawler_manager()
        crawler_stats = await crawler_manager.get_statistics()
        
        # 检查代理管理器
        proxy_manager = await get_proxy_manager()
        proxy_stats = proxy_manager.get_proxy_statistics()
        
        # 综合健康状态
        overall_status = "healthy"
        issues = []
        
        # 检查数据库健康状态
        if db_health["mongodb"]["status"] != "connected":
            overall_status = "unhealthy"
            issues.append("MongoDB连接异常")
        
        if db_health["redis"]["status"] != "connected":
            overall_status = "unhealthy"
            issues.append("Redis连接异常")
        
        # 检查代理状态
        if proxy_stats["active_proxies"] == 0:
            overall_status = "degraded"
            issues.append("没有可用代理")
        
        return {
            "success": True,
            "data": {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "service": {
                    "name": settings.service.service_name,
                    "version": settings.service.service_version,
                    "environment": settings.service.environment
                },
                "components": {
                    "database": db_health,
                    "crawler": {
                        "total_tasks": crawler_stats["total_tasks"],
                        "running_tasks": crawler_stats["running_tasks"],
                        "success_rate": crawler_stats.get("overall_success_rate", 0)
                    },
                    "proxy": {
                        "total_proxies": proxy_stats["total_proxies"],
                        "active_proxies": proxy_stats["active_proxies"],
                        "success_rate": proxy_stats["average_success_rate"]
                    }
                },
                "issues": issues
            },
            "message": f"服务状态: {overall_status}"
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "success": False,
            "data": {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            "message": "健康检查失败"
        }


# 服务信息
@app.get("/info", tags=["系统"])
async def service_info():
    """获取服务详细信息"""
    return {
        "success": True,
        "data": {
            "service": {
                "name": settings.service.service_name,
                "version": settings.service.service_version,
                "environment": settings.service.environment,
                "host": settings.service.host,
                "port": settings.service.port
            },
            "features": {
                "crawler": {
                    "supported_platforms": ["toutiao", "baijiahao", "xiaohongshu"],
                    "max_concurrent_crawlers": settings.crawler.max_concurrent_crawlers,
                    "proxy_enabled": settings.crawler.enable_proxy
                },
                "content": {
                    "manual_upload": True,
                    "batch_import": True,
                    "file_formats": ["csv", "json"],
                    "search_enabled": True
                },
                "proxy": {
                    "auto_rotation": True,
                    "quality_detection": True,
                    "providers": list(settings.proxy.proxy_providers.keys())
                }
            },
            "api": {
                "prefix": settings.service.api_prefix,
                "docs": settings.service.docs_url,
                "openapi": settings.service.openapi_url
            }
        },
        "message": "服务信息获取成功"
    }


# 自定义OpenAPI文档
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # 添加自定义标签
    openapi_schema["tags"] = [
        {
            "name": "系统",
            "description": "系统级接口，包括健康检查、服务信息等"
        },
        {
            "name": "爬虫管理", 
            "description": "爬虫任务的创建、启动、停止和监控"
        },
        {
            "name": "内容管理",
            "description": "内容的添加、查询、更新和删除操作"
        },
        {
            "name": "代理管理",
            "description": "代理的获取、测试、统计和管理"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    log_format = settings.logging.log_format
    log_level = settings.logging.log_level.value
    
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    if settings.logging.log_file:
        logger.add(
            settings.logging.log_file,
            format=log_format,
            level=log_level,
            rotation=settings.logging.log_rotation,
            retention=settings.logging.log_retention
        )
    
    # 启动服务
    logger.info("🌟 启动数据源服务...")
    
    uvicorn.run(
        "src.main:app",
        host=settings.service.host,
        port=settings.service.port,
        workers=settings.service.workers,
        reload=settings.is_development(),
        log_config=None,  # 使用自定义日志配置
        access_log=False   # 关闭uvicorn自带访问日志，使用中间件记录
    )