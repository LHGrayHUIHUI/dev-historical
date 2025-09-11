"""
Content Merger Service 主应用入口

多内容智能合并服务，提供基于AI的历史文本内容合并功能。
支持5种合并策略：时间线整合、主题归并、层次组织、逻辑关系构建、补充扩展。
"""

import logging
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
from typing import Dict, Any

from .config.settings import settings
from .controllers.merger_controller import router as merger_router
from .clients.storage_client import storage_client
from .clients.ai_service_client import ai_service_client

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.service.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    try:
        logger.info("Content Merger Service starting up...")
        
        # 启动时初始化
        logger.info("Initializing external service connections...")
        
        # 测试外部服务连接
        storage_healthy = await storage_client.health_check()
        ai_healthy = await ai_service_client.health_check()
        
        if not storage_healthy:
            logger.warning("Storage service not available, continuing with degraded mode")
        
        if not ai_healthy:
            logger.warning("AI service not available, continuing with degraded mode")
        
        logger.info(f"Content Merger Service started successfully on port {settings.service.port}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Content Merger Service: {str(e)}")
        raise
    finally:
        # 关闭时清理
        logger.info("Content Merger Service shutting down...")
        
        # 关闭客户端连接
        try:
            await storage_client._close_session()
            await ai_service_client._close_session()
            logger.info("External service connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="Content Merger Service",
    description="多内容智能合并服务 - 基于AI的历史文本内容合并平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 请求处理时间中间件
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc) if settings.service.debug else "Internal server error"
        }
    )

# 注册路由
app.include_router(
    merger_router,
    prefix="/api/v1/merger",
    tags=["内容合并"]
)

# 健康检查端点
@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    try:
        # 检查外部服务连接
        storage_healthy = await storage_client.health_check()
        ai_healthy = await ai_service_client.health_check()
        
        status = "healthy"
        if not storage_healthy or not ai_healthy:
            status = "degraded"
        
        return {
            "status": status,
            "service": "content-merger-service",
            "version": "1.0.0",
            "timestamp": time.time(),
            "dependencies": {
                "storage_service": storage_healthy,
                "ai_service": ai_healthy
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "content-merger-service",
                "error": str(e)
            }
        )

@app.get("/ready", tags=["系统"])
async def readiness_check():
    """就绪检查 - Kubernetes就绪探针"""
    try:
        # 验证关键依赖是否可用
        storage_healthy = await storage_client.health_check()
        
        if storage_healthy:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/info", tags=["系统"])
async def service_info():
    """服务信息"""
    return {
        "service": "content-merger-service",
        "version": "1.0.0",
        "description": "多内容智能合并服务",
        "features": [
            "时间线整合合并",
            "主题归并合并", 
            "层次组织合并",
            "逻辑关系构建合并",
            "补充扩展合并"
        ],
        "port": settings.service.port,
        "environment": settings.service.environment,
        "ai_integration": True,
        "storage_integration": True
    }

@app.get("/", tags=["系统"])
async def root():
    """根路径"""
    return {
        "message": "Content Merger Service",
        "description": "多内容智能合并服务API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # 直接运行时使用uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=settings.service.debug,
        log_level=settings.service.log_level.lower()
    )