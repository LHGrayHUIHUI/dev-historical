"""
文本发布服务主程序

FastAPI应用程序入口
配置中间件、路由和生命周期事件
"""

import logging
import logging.config
from contextual import ContextualAdapter
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config.settings import settings, LOGGING_CONFIG
from .models.database import create_engine_and_session, create_all_tables, close_database_connection
from .services.redis_service import redis_service
from .controllers import (
    publishing_router, platforms_router, 
    accounts_router, health_router
)

# 配置日志
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="多平台内容发布服务API",
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    请求日志中间件
    
    记录所有API请求的基本信息
    """
    start_time = datetime.utcnow()
    
    # 记录请求
    logger.info(f"请求开始: {request.method} {request.url}")
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    # 记录响应
    logger.info(
        f"请求完成: {request.method} {request.url} - "
        f"状态码: {response.status_code} - 耗时: {process_time:.3f}s"
    )
    
    # 添加处理时间头
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    
    统一处理未捕获的异常
    """
    logger.error(f"未处理异常: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# 注册路由
app.include_router(health_router)
app.include_router(publishing_router)
app.include_router(platforms_router)
app.include_router(accounts_router)


@app.get("/")
async def root():
    """
    根路径接口
    
    返回服务基本信息
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs" if settings.debug else "文档已禁用"
    }


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件
    
    初始化数据库连接和Redis连接
    """
    try:
        logger.info(f"正在启动 {settings.app_name} v{settings.app_version}")
        
        # 初始化数据库
        create_engine_and_session()
        await create_all_tables()
        logger.info("数据库初始化完成")
        
        # 初始化Redis连接
        await redis_service.connect()
        logger.info("Redis连接初始化完成")
        
        logger.info("服务启动完成")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭事件
    
    清理资源和关闭连接
    """
    try:
        logger.info("正在关闭服务...")
        
        # 关闭Redis连接
        await redis_service.disconnect()
        logger.info("Redis连接已关闭")
        
        # 关闭数据库连接
        await close_database_connection()
        logger.info("数据库连接已关闭")
        
        logger.info("服务关闭完成")
        
    except Exception as e:
        logger.error(f"服务关闭异常: {e}")


# 初始化数据的端点（仅在开发模式下可用）
if settings.debug:
    @app.post("/dev/init-data")
    async def init_development_data():
        """
        初始化开发数据
        
        创建示例平台和账号数据
        """
        try:
            from .models.database import DatabaseManager
            from .models.publishing_models import PublishingPlatform, PublishingAccount
            from .config.settings import PlatformConfig
            
            async with DatabaseManager() as db:
                # 创建平台数据
                for platform_name, config in PlatformConfig.PLATFORM_CONFIGS.items():
                    # 检查平台是否已存在
                    existing = await db.execute(
                        select(PublishingPlatform).where(
                            PublishingPlatform.platform_name == platform_name
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue
                    
                    platform = PublishingPlatform(
                        platform_name=platform_name,
                        platform_type="social_media",
                        display_name=config["display_name"],
                        api_endpoint=config["api_base_url"],
                        auth_type=config["auth_type"],
                        rate_limit_per_hour=config["rate_limit_per_hour"],
                        is_active=True
                    )
                    db.add(platform)
                
                await db.commit()
                logger.info("开发数据初始化完成")
                
                return {
                    "success": True,
                    "message": "开发数据初始化完成",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"初始化开发数据失败: {e}")
            return {
                "success": False,
                "message": f"初始化失败: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }


def create_app() -> FastAPI:
    """
    创建应用工厂函数
    
    Returns:
        FastAPI: 配置好的应用实例
    """
    return app


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=settings.debug,
        log_level=settings.log_level.lower()
    )