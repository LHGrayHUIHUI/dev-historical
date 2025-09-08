"""
图像处理服务主应用
FastAPI应用配置和启动
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from .config.settings import settings
from .controllers.image_controller import router as image_router


# 全局应用状态管理
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("图像处理服务启动中...")
    
    # 启动时初始化
    try:
        # 创建临时目录
        temp_dir = Path(settings.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建上传目录
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"图像处理服务已启动，监听端口: {settings.port}")
        logger.info(f"临时目录: {temp_dir}")
        logger.info(f"上传目录: {upload_dir}")
        
        yield
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise
    finally:
        # 关闭时清理
        logger.info("图像处理服务正在关闭...")


# 创建FastAPI应用实例
app = FastAPI(
    title="图像处理服务",
    description="历史文档图像处理和优化服务",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# 注册路由
app.include_router(image_router, prefix="/api/v1")


# 健康检查端点
@app.get("/health")
async def health_check():
    """服务健康检查"""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "image-processing-service",
            "version": "1.0.0",
            "port": settings.port
        }
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes就绪探针"""
    try:
        # 检查临时目录是否可写
        temp_dir = Path(settings.temp_dir)
        if not temp_dir.exists() or not temp_dir.is_dir():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "临时目录不可用"}
            )
        
        return JSONResponse(
            content={
                "status": "ready",
                "service": "image-processing-service"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready", 
                "reason": str(e)
            }
        )


@app.get("/info")
async def service_info():
    """服务信息"""
    return JSONResponse(
        content={
            "service": "image-processing-service",
            "version": "1.0.0",
            "port": settings.port,
            "supported_formats": settings.supported_image_formats,
            "max_file_size": settings.max_file_size,
            "processing_engines": [
                "opencv", "pillow", "scikit_image", "torch"
            ],
            "processing_types": [
                "enhance", "denoise", "deskew", "resize", 
                "format_convert", "quality_assess", "auto_enhance"
            ]
        }
    )


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "服务内部错误",
            "detail": str(exc) if settings.debug else None
        }
    )


if __name__ == "__main__":
    # 直接运行服务
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )