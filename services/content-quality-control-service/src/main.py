"""
内容质量控制服务主应用程序

基于FastAPI的内容质量控制微服务，提供多维度质量检测、
合规性审核和智能工作流管理功能。
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime

from loguru import logger

from .config.settings import settings
from .controllers import quality_controller, compliance_controller, review_controller
from .clients.storage_client import storage_client
from .models.quality_models import ErrorResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    # 启动时的初始化
    logger.info(f"🚀 启动 {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")
    logger.info(f"🌍 运行环境: {settings.SERVICE_ENVIRONMENT}")
    logger.info(f"🔧 调试模式: {settings.DEBUG}")
    
    # 检查外部服务连接
    try:
        health_result = await storage_client.health_check()
        if health_result.get("status") == "healthy":
            logger.info("✅ Storage Service 连接正常")
        else:
            logger.warning("⚠️ Storage Service 连接异常")
    except Exception as e:
        logger.error(f"❌ Storage Service 连接失败: {e}")
    
    # 初始化完成
    logger.info("✅ 服务初始化完成")
    
    yield
    
    # 关闭时的清理
    logger.info("🔄 正在关闭服务...")
    
    # 关闭外部客户端连接
    try:
        await storage_client.close()
        logger.info("✅ 外部服务连接已关闭")
    except Exception as e:
        logger.error(f"❌ 关闭外部服务连接失败: {e}")
    
    logger.info("✅ 服务关闭完成")

# 创建FastAPI应用实例
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.SERVICE_VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan
)

# ==================== 中间件配置 ====================

# CORS中间件
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 请求处理时间中间件
@app.middleware("http")
async def process_time_middleware(request: Request, call_next):
    """记录请求处理时间"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # 记录慢请求
    if process_time > 5.0:  # 超过5秒的请求
        logger.warning(f"慢请求: {request.method} {request.url} 耗时 {process_time:.2f}s")
    
    return response

# 请求日志中间件
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """记录API请求日志"""
    start_time = time.time()
    
    # 记录请求信息
    logger.info(f"📥 {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 记录响应信息
        logger.info(
            f"📤 {request.method} {request.url} -> "
            f"{response.status_code} ({process_time:.3f}s)"
        )
        
        return response
    
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"💥 {request.method} {request.url} -> "
            f"ERROR ({process_time:.3f}s): {str(e)}"
        )
        raise

# ==================== 异常处理 ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """参数错误处理"""
    logger.warning(f"参数错误: {str(exc)}")
    
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            success=False,
            message=str(exc),
            error_code="INVALID_PARAMETER",
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"服务异常: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            message="内部服务错误",
            error_code="INTERNAL_SERVER_ERROR",
            error_details={"exception_type": type(exc).__name__} if settings.DEBUG else None,
            timestamp=datetime.now()
        ).dict()
    )

# ==================== 路由注册 ====================

# 基础健康检查
@app.get("/health", tags=["健康检查"])
async def health_check():
    """基础健康检查"""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "timestamp": datetime.now().isoformat()
    }

# 详细健康检查
@app.get("/health/detailed", tags=["健康检查"])
async def detailed_health_check():
    """详细健康检查"""
    health_status = {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "environment": settings.SERVICE_ENVIRONMENT,
        "timestamp": datetime.now().isoformat(),
        "dependencies": {}
    }
    
    # 检查Storage Service
    try:
        storage_health = await storage_client.health_check()
        health_status["dependencies"]["storage_service"] = {
            "status": "healthy" if storage_health.get("status") != "unhealthy" else "unhealthy",
            "url": settings.STORAGE_SERVICE_URL
        }
    except Exception as e:
        health_status["dependencies"]["storage_service"] = {
            "status": "unhealthy",
            "error": str(e),
            "url": settings.STORAGE_SERVICE_URL
        }
        health_status["status"] = "degraded"
    
    return health_status

# 服务信息
@app.get("/info", tags=["服务信息"])
async def service_info():
    """获取服务信息"""
    return {
        "name": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "description": settings.API_DESCRIPTION,
        "environment": settings.SERVICE_ENVIRONMENT,
        "debug": settings.DEBUG,
        "config": {
            "max_content_length": settings.MAX_CONTENT_LENGTH,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "auto_approval_threshold": settings.AUTO_APPROVAL_THRESHOLD,
            "parallel_detection_enabled": settings.PARALLEL_DETECTION_ENABLED,
            "auto_fix_enabled": settings.AUTO_FIX_ENABLED
        },
        "endpoints": {
            "docs": f"{settings.API_V1_PREFIX}/docs",
            "redoc": f"{settings.API_V1_PREFIX}/redoc",
            "health": "/health",
            "quality_check": f"{settings.API_V1_PREFIX}/quality/check",
            "compliance_check": f"{settings.API_V1_PREFIX}/compliance/check",
            "create_review_task": f"{settings.API_V1_PREFIX}/review/tasks"
        }
    }

# 注册业务路由
app.include_router(
    quality_controller.router,
    prefix=settings.API_V1_PREFIX,
    tags=["质量检测"]
)

app.include_router(
    compliance_controller.router,
    prefix=settings.API_V1_PREFIX,
    tags=["合规检测"]
)

app.include_router(
    review_controller.router,
    prefix=settings.API_V1_PREFIX,
    tags=["审核工作流"]
)

# ==================== 根路径 ====================

@app.get("/", tags=["根路径"])
async def root():
    """根路径，返回服务基本信息"""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "message": "内容质量控制服务运行正常",
        "docs": f"{settings.API_V1_PREFIX}/docs",
        "health": "/health"
    }

# ==================== 启动函数 ====================

def create_app() -> FastAPI:
    """创建应用实例的工厂函数"""
    return app

if __name__ == "__main__":
    import uvicorn
    
    # 启动服务
    logger.info(f"🚀 启动 {settings.SERVICE_NAME}")
    logger.info(f"🌐 服务地址: http://{settings.SERVICE_HOST}:{settings.SERVICE_PORT}")
    logger.info(f"📚 API文档: http://{settings.SERVICE_HOST}:{settings.SERVICE_PORT}{settings.API_V1_PREFIX}/docs")
    
    uvicorn.run(
        "src.main:app",
        host=settings.SERVICE_HOST,
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG
    )