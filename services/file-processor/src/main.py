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
# 导入API路由
from .api.process import router as process_router

# 获取配置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("🚀 文件处理服务启动中...")
    
    try:
        # TODO: 初始化文件处理器
        logger.info("📤 初始化文件处理器...")
        # await init_file_processors()
        logger.info("✅ 文件处理器初始化完成")
        
        
        # 注册信号处理器
        def signal_handler(signum, frame):
            logger.info(f"收到退出信号 {signum}")
            asyncio.create_task(graceful_shutdown())
        
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        
        logger.success("✅ 文件处理服务启动完成")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        raise
    
    finally:
        # 关闭时执行
        logger.info("🛑 文件处理服务关闭中...")
        await graceful_shutdown()


async def graceful_shutdown():
    """优雅关闭"""
    try:
        # TODO: 清理文件处理资源
        logger.info("清理文件处理资源...")
        # await cleanup_file_processors()
        
        logger.success("✅ 服务已优雅关闭")
        
    except Exception as e:
        logger.error(f"❌ 关闭服务时发生错误: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="历史文本项目 - 文件处理服务",
    description="""
    ## 📄 文件处理服务 API
    
    **专注于各种格式文件处理和文本提取的纯处理微服务**
    
    ### 🎯 核心职责
    - **📄 多格式文件处理**: PDF、Word、图片OCR、HTML等格式文件处理
    - **🔤 文本内容提取**: 从各种文件格式中提取纯文本内容
    - **🛡️ 文件安全检测**: 文件格式验证、病毒扫描、安全检查
    - **⚡ 异步处理**: 支持大文件的异步处理和状态跟踪
    - **📊 批量处理**: 支持多文件并发处理
    
    ### 🏗️ 服务定位
    - **✅ 纯文件处理服务**: 不涉及数据存储，专注文件处理算法
    - **✅ 无状态设计**: 所有处理结果通过API返回给调用方
    - **✅ 高性能处理**: 优化的文件处理算法和并发处理能力
    - **❌ 无数据库依赖**: 符合纯处理服务定位
    - **❌ 无数据持久化**: 所有数据存储由storage-service负责
    
    ### 🔧 技术栈
    - **框架**: FastAPI + Python 3.11+
    - **处理引擎**: PyPDF2, python-docx, Pillow, Tesseract OCR
    - **格式支持**: PDF, Word, Excel, 图片(JPG/PNG/GIF), HTML
    - **架构**: 无数据库依赖，纯处理逻辑
    
    ### 🔄 与其他服务协作
    - **调用方**: storage-service (统一存储服务)
    - **数据存储**: 由 storage-service 负责所有数据持久化
    - **调用方式**: 接收文件，返回处理结果，不保存数据
    - **服务边界**: 专注文件处理，不涉及存储管理
    
    ### 📚 使用说明
    1. 使用 `/api/v1/process/pdf` 处理PDF文件
    2. 使用 `/api/v1/process/image-ocr` 进行图片OCR识别
    3. 使用 `/api/v1/process/document` 处理通用文档
    4. 使用 `/api/v1/process/batch` 批量处理文件
    5. 使用 `/api/v1/process/status/{task_id}` 查询异步处理状态
    6. 访问 `/health` 检查服务状态
    7. 访问 `/docs` 查看完整API文档
    
    ### 🚫 不包含的功能
    - ❌ 数据库连接 (MongoDB, PostgreSQL, Redis)
    - ❌ 数据持久化存储
    - ❌ 业务逻辑处理
    - ❌ 内容管理功能
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
            "url": "http://file-processor:8000",
            "description": "Docker环境"
        },
        {
            "url": "https://api.historical-text.com",
            "description": "生产环境"
        }
    ],
    docs_url=settings.service.docs_url,
    redoc_url="/redoc",
    openapi_url=settings.service.openapi_url,
    openapi_tags=[
        {
            "name": "文件处理",
            "description": "📄 PDF、Word、图片OCR等文件处理功能"
        },
        {
            "name": "系统",
            "description": "🔧 健康检查、服务信息等系统级接口"
        }
    ],
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
app.include_router(process_router, prefix=settings.service.api_prefix)


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
        "message": "文件处理服务运行中"
    }


# 健康检查
@app.get("/health", 
         tags=["系统"],
         summary="健康检查",
         description="检查文件处理服务的健康状态和可用处理器",
         response_description="服务健康状态信息",
         responses={
             200: {
                 "description": "服务健康",
                 "content": {
                     "application/json": {
                         "example": {
                             "success": True,
                             "data": {
                                 "status": "healthy",
                                 "components": {
                                     "processors": {
                                         "status": "ready",
                                         "available_processors": ["pdf", "word", "image", "html"]
                                     }
                                 }
                             }
                         }
                     }
                 }
             },
             500: {"description": "服务不健康"}
         })
async def health_check():
    """服务健康检查
    
    检查文件处理服务的健康状态，包括：
    - 服务基本状态
    - 可用的文件处理器
    - 处理器就绪状态
    
    Returns:
        dict: 包含服务健康状态的响应
    """
    try:
        # 文件处理服务健康检查（无数据库依赖）
        overall_status = "healthy"
        issues = []
        
        # TODO: 检查文件处理器状态
        # processor_status = await check_processors_health()
        processor_status = {"status": "ready", "available_processors": ["pdf", "word", "image", "html"]}
        
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
                    "processors": processor_status
                },
                "issues": issues
            },
            "message": f"文件处理服务状态: {overall_status}"
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
@app.get("/info", 
         tags=["系统"],
         summary="获取服务信息",
         description="获取文件处理服务的详细配置和功能信息",
         response_description="服务配置和功能详情")
async def service_info():
    """获取服务详细信息
    
    返回文件处理服务的完整配置信息，包括：
    - 服务基本信息
    - 支持的文件格式
    - 处理功能特性
    - API接口信息
    
    Returns:
        dict: 包含服务详细信息的响应
    """
    return {
        "success": True,
        "data": {
            "service": {
                "name": settings.service.service_name,
                "version": settings.service.service_version,
                "environment": settings.service.environment,
                "host": settings.service.host,
                "port": settings.service.port,
                "type": "file-processor",
                "description": "纯文件处理服务，专注文件处理算法"
            },
            "capabilities": {
                "document_formats": [
                    {"extension": "pdf", "description": "PDF文档", "features": ["文本提取", "元数据提取"]},
                    {"extension": "docx", "description": "Word文档", "features": ["文本提取"]},
                    {"extension": "html", "description": "HTML文档", "features": ["文本提取", "结构化解析"]}
                ],
                "image_formats": [
                    {"extension": "jpg", "description": "JPEG图片", "features": ["OCR文字识别"]},
                    {"extension": "png", "description": "PNG图片", "features": ["OCR文字识别"]},
                    {"extension": "gif", "description": "GIF图片", "features": ["OCR文字识别"]}
                ],
                "processing_features": {
                    "async_processing": True,
                    "batch_processing": True,
                    "status_tracking": True,
                    "error_handling": True,
                    "file_validation": True
                }
            },
            "architecture": {
                "database_dependencies": False,
                "stateless_design": True,
                "storage_service_integration": True,
                "microservice_type": "pure_processing"
            },
            "api": {
                "prefix": settings.service.api_prefix,
                "docs": settings.service.docs_url,
                "openapi": settings.service.openapi_url,
                "endpoints": {
                    "pdf_processing": "/api/v1/process/pdf",
                    "image_ocr": "/api/v1/process/image-ocr",
                    "document_processing": "/api/v1/process/document",
                    "batch_processing": "/api/v1/process/batch",
                    "status_check": "/api/v1/process/status/{task_id}"
                }
            }
        },
        "message": "文件处理服务信息获取成功"
    }


# 自定义OpenAPI文档
def custom_openapi():
    """自定义OpenAPI文档配置"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # 添加自定义信息
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # 更新标签描述
    openapi_schema["tags"] = [
        {
            "name": "系统",
            "description": "🔧 系统级接口，包括健康检查、服务信息、就绪状态等"
        },
        {
            "name": "文件处理",
            "description": "📄 文件处理接口，支持PDF、Word、图片OCR、HTML等格式"
        }
    ]
    
    # 添加安全定义（如果需要）
    openapi_schema["components"] = openapi_schema.get("components", {})
    
    # 添加示例
    openapi_schema["info"]["x-examples"] = {
        "pdf_processing": "POST /api/v1/process/pdf",
        "image_ocr": "POST /api/v1/process/image-ocr",
        "health_check": "GET /health"
    }
    
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