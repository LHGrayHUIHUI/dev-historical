"""
数据采集服务主应用程序

FastAPI应用程序的入口点
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from .config import get_settings
from .controllers import data_router
from .controllers.content_controller import router as content_router
from .services.data_collection_service import DataCollectionService
from .services.message_queue_service import RabbitMQClient
from .utils.database import close_database, init_database
from .utils.storage import init_storage_client
from .workers.text_extraction_worker import TextExtractionWorker

# 全局变量
worker_task = None
message_queue_client = None
data_service = None


def setup_logging():
    """设置结构化日志"""
    settings = get_settings()
    
    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.json_logs else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 配置标准库日志
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper())
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理
    
    Args:
        app: FastAPI应用实例
    """
    global worker_task, message_queue_client, data_service
    
    settings = get_settings()
    logger = structlog.get_logger(__name__)
    
    # 启动时初始化
    logger.info("正在启动数据采集服务", version=settings.service_version)
    
    try:
        # 1. 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")
        
        # 2. 初始化存储服务
        init_storage_client()
        logger.info("存储服务初始化完成")
        
        # 3. 初始化数据采集服务
        data_service = DataCollectionService()
        
        # 4. 初始化消息队列
        message_queue_client = RabbitMQClient(settings.rabbitmq_url)
        await message_queue_client.connect()
        logger.info("消息队列连接成功")
        
        # 5. 设置数据服务的消息队列
        data_service.set_message_queue(message_queue_client)
        
        # 6. 启动文本提取工作器
        if settings.service_environment != 'testing':  # 测试环境不启动工作器
            worker = TextExtractionWorker(message_queue_client)
            worker_task = asyncio.create_task(worker.start())
            logger.info("文本提取工作器启动成功")
        
        # 将服务实例存储到app state中
        app.state.data_service = data_service
        app.state.message_queue = message_queue_client
        
        logger.info("数据采集服务启动完成")
        
    except Exception as e:
        logger.error("服务启动失败", error=str(e))
        raise
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭数据采集服务")
    
    try:
        # 1. 停止工作器
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            logger.info("文本提取工作器已停止")
        
        # 2. 关闭消息队列连接
        if message_queue_client:
            await message_queue_client.close()
            logger.info("消息队列连接已关闭")
        
        # 3. 关闭数据库连接
        await close_database()
        logger.info("数据库连接已关闭")
        
        logger.info("数据采集服务关闭完成")
        
    except Exception as e:
        logger.error("服务关闭时发生错误", error=str(e))


# 设置日志
setup_logging()
logger = structlog.get_logger(__name__)

# 获取配置
settings = get_settings()

# 创建FastAPI应用
app = FastAPI(
    title="历史文本项目 - 统一存储服务",
    description="""
    ## 🗄️ 统一存储服务 API
    
    **负责所有数据存储、内容管理、文件存储和业务逻辑处理的核心微服务**
    
    ### 🎯 核心定位
    - **统一存储服务**: 管理所有数据库和存储系统
    - **业务逻辑中心**: 负责所有业务规则和数据处理
    - **服务协调者**: 调用file-processor处理文件，统一管理数据流
    
    ### ✅ 核心功能
    - **🗄️ 统一数据库管理**: MongoDB + PostgreSQL + Redis + MinIO 完整存储栈
    - **📄 内容管理系统**: 历史文本内容的CRUD、搜索、分类、统计
    - **📁 文件存储管理**: MinIO对象存储的完整管理和API
    - **🔄 服务协调**: 调用file-processor处理文件，整合处理结果
    - **📊 业务分析**: 数据统计、搜索、报表和可视化支持
    - **⚡ 批量处理**: 支持大批量数据导入和处理
    
    ### 🗄️ 管理的存储系统
    - **MongoDB**: 历史文本内容、业务数据、用户数据
    - **PostgreSQL**: 文件元数据、处理记录、关系数据
    - **Redis**: 缓存、会话、任务队列、统计数据
    - **MinIO**: 文件对象存储（图片、视频、文档）
    - **RabbitMQ**: 消息队列、异步任务处理
    
    ### 🔧 技术架构
    - **框架**: FastAPI + Python 3.11+ + SQLAlchemy 2.0
    - **数据库**: MongoDB + PostgreSQL + Redis
    - **存储**: MinIO (S3兼容对象存储)
    - **消息队列**: RabbitMQ
    - **监控**: Prometheus + Grafana
    - **日志**: Structured logging + ELK Stack
    
    ### 🔄 服务协作
    - **调用服务**: file-processor (文件处理)
    - **被调用方**: Vue3前端、其他微服务
    - **外部依赖**: 所有数据库和存储系统
    
    ### 📚 API分组
    1. **内容管理API**: `/api/v1/content/` - 创建、查询、更新、删除内容
    2. **文件管理API**: `/api/v1/files/` - 文件上传、下载、管理
    3. **搜索统计API**: `/api/v1/content/search/`, `/api/v1/content/stats/`
    4. **系统接口**: `/health`, `/ready` - 健康检查和就绪状态
    """,
    version=settings.service_version,
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
            "url": f"http://localhost:{settings.service_port}",
            "description": "开发环境"
        },
        {
            "url": "http://storage-service:8000",
            "description": "Docker环境"
        },
        {
            "url": "https://api.historical-text.com",
            "description": "生产环境"
        }
    ],
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    openapi_tags=[
        {
            "name": "内容管理",
            "description": "📄 历史文本内容的创建、查询、更新、删除和搜索功能"
        },
        {
            "name": "文件存储",
            "description": "📁 文件上传、下载、MinIO对象存储管理功能"
        },
        {
            "name": "数据统计",
            "description": "📊 内容统计、搜索分析、业务数据报表功能"
        },
        {
            "name": "健康检查",
            "description": "🏥 服务健康状态、数据库连接、存储服务检查"
        }
    ]
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts
)


# 异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器
    
    Args:
        request: 请求对象
        exc: 验证异常
        
    Returns:
        错误响应
    """
    logger.warning(
        "请求验证失败",
        url=str(request.url),
        method=request.method,
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "error_message": "请求参数验证失败",
            "details": exc.errors(),
            "timestamp": structlog.processors.TimeStamper()(None, None, {})["timestamp"]
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器
    
    Args:
        request: 请求对象
        exc: 异常
        
    Returns:
        错误响应
    """
    logger.error(
        "未处理的异常",
        url=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "error_message": "服务器内部错误",
            "message": "请稍后重试或联系管理员",
            "timestamp": structlog.processors.TimeStamper()(None, None, {})["timestamp"]
        }
    )


# 健康检查端点
@app.get("/health", 
         tags=["健康检查"],
         summary="健康检查",
         description="检查统一存储服务的基本健康状态",
         response_description="服务健康状态信息",
         responses={
             200: {
                 "description": "服务健康",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "healthy",
                             "service": "storage-service",
                             "version": "1.0.0",
                             "checks": {
                                 "database": "healthy",
                                 "storage": "healthy",
                                 "message_queue": "healthy"
                             }
                         }
                     }
                 }
             }
         })
async def health_check():
    """健康检查端点
    
    检查统一存储服务的基本健康状态，包括：
    - 服务基本信息
    - 版本号和时间戳
    
    Returns:
        dict: 包含服务健康状态的响应
    """
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.service_version,
        "timestamp": structlog.processors.TimeStamper()(None, None, {})["timestamp"]
    }


@app.get("/ready", 
         tags=["健康检查"],
         summary="就绪检查",
         description="检查统一存储服务是否已准备好接受请求，包括所有依赖服务的连通性",
         response_description="服务就绪状态和所有依赖检查结果",
         responses={
             200: {
                 "description": "服务就绪",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "ready",
                             "service": "storage-service",
                             "checks": {
                                 "database": "healthy",
                                 "storage": "healthy",
                                 "message_queue": "healthy"
                             }
                         }
                     }
                 }
             },
             503: {
                 "description": "服务未就绪",
                 "content": {
                     "application/json": {
                         "example": {
                             "status": "not_ready",
                             "checks": {
                                 "database": "unhealthy",
                                 "storage": "healthy",
                                 "message_queue": "healthy"
                             }
                         }
                     }
                 }
             }
         })
async def readiness_check():
    """就绪检查端点
    
    全面检查统一存储服务的就绪状态，包括：
    - 数据库连接状态（MongoDB, PostgreSQL, Redis）
    - 对象存储连接状态（MinIO）
    - 消息队列连接状态（RabbitMQ）
    
    Returns:
        dict: 包含详细检查结果的响应
    """
    try:
        from .utils.database import check_database_connection
        from .utils.storage import check_storage_connection
        
        # 检查数据库连接
        db_healthy = await check_database_connection()
        
        # 检查存储服务
        storage_healthy = await check_storage_connection()
        
        # 检查消息队列
        mq_healthy = (
            message_queue_client is not None 
            and message_queue_client.is_connected
        )
        
        if db_healthy and storage_healthy and mq_healthy:
            return {
                "status": "ready",
                "service": settings.service_name,
                "version": settings.service_version,
                "checks": {
                    "database": "healthy",
                    "storage": "healthy",
                    "message_queue": "healthy"
                },
                "timestamp": structlog.processors.TimeStamper()(None, None, {})["timestamp"]
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "service": settings.service_name,
                    "checks": {
                        "database": "healthy" if db_healthy else "unhealthy",
                        "storage": "healthy" if storage_healthy else "unhealthy",
                        "message_queue": "healthy" if mq_healthy else "unhealthy"
                    },
                    "timestamp": structlog.processors.TimeStamper()(None, None, {})["timestamp"]
                }
            )
            
    except Exception as e:
        logger.error("就绪检查失败", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": structlog.processors.TimeStamper()(None, None, {})["timestamp"]
            }
        )


# 注册路由
app.include_router(data_router)  # 文件处理相关路由
app.include_router(content_router, prefix="/api/v1")  # 内容管理路由

# 导入并注册AI模型配置路由
from .controllers.ai_models_controller import create_ai_models_controller
ai_models_controller = create_ai_models_controller()
app.include_router(ai_models_controller.router)  # AI模型配置路由

# 添加Prometheus指标端点
if settings.metrics_enabled:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


# 添加启动事件日志
@app.on_event("startup")
async def startup_event():
    """启动事件处理"""
    logger.info(
        "数据采集服务已启动",
        host=settings.service_host,
        port=settings.service_port,
        environment=settings.service_environment,
        debug=settings.debug
    )


if __name__ == "__main__":
    # 直接运行时使用uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
        server_header=False,
        date_header=False
    )