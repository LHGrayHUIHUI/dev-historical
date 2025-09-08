"""
OCR服务主应用程序入口

无状态OCR文本识别微服务，专注于图像文本识别算法。
数据存储通过storage-service完成。

主要功能：
- 单图像OCR识别
- 批量图像处理
- 多OCR引擎支持
- 异步任务处理
- 无状态设计

Author: OCR开发团队
Created: 2025-01-15
Version: 2.0.0 (无状态架构)
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

# 导入配置
from .config.settings import get_settings

# 导入控制器
from .controllers.ocr_controller import include_router

# 导入服务
from .services.ocr_service import get_ocr_service

# 导入工具模块
from .utils.logger import setup_logging
from .utils.middleware import (
    LoggingMiddleware,
    RequestIDMiddleware,
    TimingMiddleware
)

# 获取配置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    处理应用启动和关闭时的初始化和清理工作。
    """
    # 启动时的初始化
    try:
        logging.info("OCR服务启动中...")
        
        # 初始化OCR服务（加载模型）
        ocr_service = await get_ocr_service()
        await ocr_service.initialize()
        logging.info("OCR引擎初始化完成")
        
        # 创建临时目录
        temp_dir = Path(settings.ocr.TEMP_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"临时目录创建: {temp_dir}")
        
        # 启动后台清理任务
        cleanup_task = asyncio.create_task(_start_cleanup_tasks())
        
        logging.info(f"OCR服务启动完成，端口: {settings.API_PORT}")
        
        yield  # 应用运行期间
        
    finally:
        # 关闭时的清理
        logging.info("OCR服务关闭中...")
        
        # 清理OCR服务资源
        if 'ocr_service' in locals():
            await ocr_service.cleanup()
            logging.info("OCR引擎资源清理完成")
        
        # 取消后台任务
        if 'cleanup_task' in locals():
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        logging.info("OCR服务关闭完成")


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例
    
    配置所有中间件、路由、异常处理等。
    
    Returns:
        配置完成的FastAPI应用实例
    """
    
    # 创建FastAPI应用
    app = FastAPI(
        title="OCR文本识别服务",
        description="""
        ## 无状态OCR识别微服务

        专门为古籍文献设计的高精度OCR文本识别服务，支持：

        - **多引擎支持**: PaddleOCR、Tesseract、EasyOCR
        - **高精度识别**: 针对古代汉字和繁体字优化
        - **异步处理**: 支持大批量文档的异步处理
        - **智能预处理**: 自动图像增强和倾斜校正
        - **文本后处理**: 繁简转换、标点规范化
        - **无状态设计**: 支持水平扩展，数据通过storage-service管理
        
        ### 快速开始
        
        1. 上传图像文件到 `/api/v1/ocr/recognize` 端点
        2. 获取识别结果或任务ID
        3. 如果是异步模式，通过 `/api/v1/ocr/task/{task_id}` 查询结果
        
        ### 支持格式
        
        - 图像格式：JPG、PNG、BMP、TIFF、WebP
        - 最大文件大小：50MB
        - 批量处理：最多20个文件
        
        ### 架构特点
        
        - **无状态服务**: 不直接连接数据库，通过storage-service进行数据管理
        - **计算专用**: 专注于OCR算法计算，不处理业务逻辑
        - **易扩展**: 支持Kubernetes水平扩展
        """,
        version=settings.SERVICE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # 配置CORS（从环境变量获取允许的域名）
    cors_origins = ["*"]  # 开发环境允许所有域名
    if settings.ENVIRONMENT == "production":
        # 生产环境需要配置具体域名
        cors_origins = ["https://your-domain.com"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # 添加Gzip压缩
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 添加自定义中间件
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # 注册OCR路由
    include_router(app)
    
    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理器"""
        logging.error(f"未处理的异常: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "message": "内部服务器错误" if settings.is_production else str(exc),
                "details": None if settings.is_production else {
                    "type": type(exc).__name__,
                    "args": exc.args
                }
            }
        )
    
    # 健康检查端点
    @app.get("/health", tags=["系统"])
    async def health_check():
        """健康检查端点"""
        try:
            ocr_service = await get_ocr_service()
            health_status = await ocr_service.health_check()
            
            return {
                "status": health_status.get("status", "unknown"),
                "service": settings.SERVICE_NAME,
                "version": settings.SERVICE_VERSION,
                "environment": settings.ENVIRONMENT,
                "ocr_engines": health_status.get("engines", {}),
                "available_engines": health_status.get("available_engines", [])
            }
        except Exception as e:
            logging.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "service": settings.SERVICE_NAME,
                "version": settings.SERVICE_VERSION,
                "environment": settings.ENVIRONMENT,
                "error": str(e)
            }
    
    # 准备就绪检查（Kubernetes readiness probe）
    @app.get("/ready", tags=["系统"])
    async def readiness_check():
        """准备就绪检查端点"""
        try:
            ocr_service = await get_ocr_service()
            if ocr_service._initialized:
                return {"status": "ready"}
            else:
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready", "message": "OCR服务未完全初始化"}
                )
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "error": str(e)}
            )
    
    # 服务信息端点
    @app.get("/info", tags=["系统"])
    async def service_info():
        """获取服务信息"""
        try:
            ocr_service = await get_ocr_service()
            available_engines = await ocr_service.get_available_engines()
        except Exception:
            available_engines = []
            
        return {
            "name": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "description": "OCR文本识别服务（无状态架构）",
            "environment": settings.ENVIRONMENT,
            "architecture": "stateless",
            "storage_service_url": settings.services.STORAGE_SERVICE_URL,
            "available_engines": available_engines,
            "default_engine": settings.ocr.DEFAULT_ENGINE,
            "supported_formats": settings.ocr.SUPPORTED_IMAGE_FORMATS,
            "max_file_size": settings.ocr.MAX_FILE_SIZE,
            "max_batch_size": settings.ocr.MAX_BATCH_SIZE,
            "max_concurrent_tasks": settings.ocr.MAX_CONCURRENT_TASKS
        }
    
    # 自定义OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # 添加自定义标签描述
        openapi_schema["tags"] = [
            {
                "name": "OCR识别",
                "description": "图像文字识别相关接口，支持同步和异步模式"
            },
            {
                "name": "系统",
                "description": "系统健康检查和服务信息"
            }
        ]
        
        # 添加服务器信息
        openapi_schema["servers"] = [
            {
                "url": f"http://localhost:{settings.API_PORT}",
                "description": "开发环境"
            }
        ]
        
        # 添加安全定义（如果需要的话）
        openapi_schema["components"] = {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app


async def _start_cleanup_tasks():
    """
    启动后台清理任务
    
    定期清理临时文件等。
    """
    import time
    
    while True:
        try:
            # 清理临时文件
            temp_dir = Path(settings.ocr.TEMP_DIR)
            if temp_dir.exists():
                current_time = time.time()
                max_age = settings.ocr.TEMP_FILE_MAX_AGE
                
                cleaned_count = 0
                for file_path in temp_dir.rglob("*"):
                    if file_path.is_file():
                        # 检查文件年龄
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age:
                            try:
                                file_path.unlink()
                                cleaned_count += 1
                            except Exception as e:
                                logging.warning(f"清理文件失败 {file_path}: {str(e)}")
                
                if cleaned_count > 0:
                    logging.info(f"清理了 {cleaned_count} 个临时文件")
            
            # 等待下次清理
            await asyncio.sleep(settings.ocr.TEMP_FILE_CLEANUP_INTERVAL)
            
        except asyncio.CancelledError:
            logging.info("清理任务被取消")
            break
        except Exception as e:
            logging.error(f"清理任务异常: {str(e)}")
            await asyncio.sleep(60)  # 出错后等待1分钟再重试


# 创建应用实例
app = create_app()


def main():
    """
    主函数，用于命令行启动
    """
    # 设置日志
    setup_logging(
        level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        json_format=settings.ENABLE_JSON_LOGS
    )
    
    logging.info(f"启动OCR服务 - 环境: {settings.ENVIRONMENT}")
    logging.info(f"服务架构: 无状态（数据通过storage-service管理）")
    logging.info(f"API地址: http://{settings.API_HOST}:{settings.API_PORT}")
    logging.info(f"API文档: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logging.info(f"Storage服务: {settings.services.STORAGE_SERVICE_URL}")
    
    # 启动服务
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.WORKERS if settings.is_production else 1,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.is_development,
        reload_dirs=["src"] if settings.is_development else None,
        access_log=True,
        loop="asyncio"
    )


if __name__ == "__main__":
    main()