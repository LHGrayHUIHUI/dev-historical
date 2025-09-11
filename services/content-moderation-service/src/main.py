"""
内容审核服务主应用程序

基于FastAPI的内容审核微服务
提供智能内容审核、违规检测、人工复审等功能
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
import uvicorn
import time

from .config.settings import settings, LOGGING_CONFIG
from .models.database import create_engine_and_session, create_all_tables, close_database_connection
from .controllers import moderation_router, admin_router, health_router

# 配置日志
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    处理应用启动和关闭时的资源管理
    """
    # 启动时执行
    logger.info("=== 内容审核服务启动中 ===")
    
    try:
        # 初始化数据库连接
        create_engine_and_session()
        logger.info("数据库连接初始化完成")
        
        # 创建数据表
        await create_all_tables()
        logger.info("数据库表创建完成")
        
        # 初始化其他组件
        await initialize_services()
        
        logger.info("=== 内容审核服务启动完成 ===")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("=== 内容审核服务关闭中 ===")
    
    try:
        # 清理数据库连接
        await close_database_connection()
        logger.info("数据库连接已关闭")
        
        # 清理其他资源
        await cleanup_services()
        
        logger.info("=== 内容审核服务关闭完成 ===")
        
    except Exception as e:
        logger.error(f"服务关闭失败: {e}")


async def initialize_services():
    """
    初始化各种服务组件
    """
    try:
        # 初始化分析器
        logger.info("初始化内容分析器...")
        
        # 这里可以预加载模型、检查依赖等
        # 例如：
        # - 加载AI模型
        # - 初始化敏感词库
        # - 连接外部服务
        
        logger.info("服务组件初始化完成")
        
    except Exception as e:
        logger.error(f"服务组件初始化失败: {e}")
        raise


async def cleanup_services():
    """
    清理服务资源
    """
    try:
        logger.info("清理服务资源...")
        # 这里可以释放资源、关闭连接等
        logger.info("服务资源清理完成")
        
    except Exception as e:
        logger.error(f"服务资源清理失败: {e}")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.app_name,
    description="智能内容审核服务，提供多媒体内容的自动审核和人工复审功能",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# 添加中间件

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 受信任主机中间件
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)


# 请求/响应中间件
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """
    请求处理中间件
    
    记录请求日志、处理时间统计等
    """
    start_time = time.time()
    
    # 记录请求信息
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    
    logger.info(f"收到请求: {method} {url} from {client_ip}")
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息
        logger.info(
            f"请求完成: {method} {url} "
            f"状态码: {response.status_code} "
            f"处理时间: {process_time:.3f}s"
        )
        
        # 添加处理时间头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"请求异常: {method} {url} "
            f"错误: {str(e)} "
            f"处理时间: {process_time:.3f}s"
        )
        raise


# 全局异常处理

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP异常处理器
    """
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": time.time()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    请求验证异常处理器
    """
    logger.warning(f"请求验证失败: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "请求参数验证失败",
            "error_code": "VALIDATION_ERROR",
            "error_details": exc.errors(),
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    通用异常处理器
    """
    logger.error(f"未处理异常: {type(exc).__name__}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "error_code": "INTERNAL_SERVER_ERROR",
            "timestamp": time.time()
        }
    )


# 注册路由
app.include_router(health_router)        # 健康检查
app.include_router(moderation_router)    # 内容审核
app.include_router(admin_router)         # 管理员功能


# 根路径
@app.get("/", summary="服务根路径")
async def root():
    """
    服务根路径，返回基本信息
    """
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "timestamp": time.time(),
        "docs_url": "/docs" if settings.debug else None,
        "health_check": "/api/v1/health"
    }


# 服务启动函数
def create_app() -> FastAPI:
    """
    创建并配置FastAPI应用
    
    Returns:
        FastAPI: 配置好的应用实例
    """
    return app


# 直接运行服务
if __name__ == "__main__":
    logger.info("直接启动内容审核服务...")
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=LOGGING_CONFIG,
        access_log=True,
        server_header=False,
        date_header=False
    )