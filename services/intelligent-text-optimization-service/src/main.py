"""
智能文本优化服务主入口 - Main Application Entry

FastAPI应用程序的主入口点，负责应用初始化、中间件配置、
路由注册和服务启动

核心功能:
1. FastAPI应用程序初始化
2. 中间件和CORS配置
3. 路由注册
4. 异常处理
5. 应用生命周期管理
"""

import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config.settings import get_settings, get_service_info
from .controllers.optimization_controller import OptimizationController
from .models.optimization_models import ApiResponse


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/service.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    处理启动和关闭事件
    """
    # 启动事件
    logger.info("智能文本优化服务启动中...")
    
    try:
        # 初始化服务组件
        settings = get_settings()
        
        # 验证配置
        await validate_configuration(settings)
        
        # 初始化外部依赖
        await initialize_dependencies()
        
        logger.info("服务初始化完成")
        
        yield
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise
    
    finally:
        # 关闭事件
        logger.info("智能文本优化服务关闭中...")
        
        try:
            # 清理资源
            await cleanup_resources()
            logger.info("服务关闭完成")
        except Exception as e:
            logger.error(f"服务关闭异常: {e}")


async def validate_configuration(settings):
    """验证配置"""
    logger.info("验证服务配置...")
    
    # 验证必要的配置项
    required_configs = [
        'storage_service_url',
        'ai_model_service_url',
        'redis_url'
    ]
    
    for config in required_configs:
        if not getattr(settings, config):
            raise ValueError(f"缺少必要配置: {config}")
    
    logger.info("配置验证通过")


async def initialize_dependencies():
    """初始化外部依赖"""
    logger.info("初始化外部依赖...")
    
    # 这里可以添加依赖初始化逻辑
    # 比如数据库连接、缓存连接等
    
    logger.info("外部依赖初始化完成")


async def cleanup_resources():
    """清理资源"""
    logger.info("清理服务资源...")
    
    # 这里可以添加资源清理逻辑
    # 比如关闭数据库连接、清理缓存等
    
    logger.info("资源清理完成")


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例
    
    Returns:
        配置好的FastAPI应用
    """
    settings = get_settings()
    service_info = get_service_info()
    
    # 创建FastAPI应用
    app = FastAPI(
        title="智能文本优化服务",
        description="基于AI的历史文本智能优化服务，支持文本润色、扩展、风格转换和现代化改写",
        version=service_info["service_version"],
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # 配置中间件
    configure_middleware(app, settings)
    
    # 注册路由
    register_routes(app)
    
    # 配置异常处理
    configure_exception_handlers(app)
    
    return app


def configure_middleware(app: FastAPI, settings):
    """配置中间件"""
    
    # CORS中间件
    if settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
    
    # Gzip压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 自定义请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        # 记录请求信息
        logger.info(f"请求开始: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # 计算处理时间
            process_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(
                f"请求完成: {request.method} {request.url.path} "
                f"- 状态码: {response.status_code} "
                f"- 耗时: {process_time:.3f}s"
            )
            
            # 添加响应头
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"请求异常: {request.method} {request.url.path} "
                f"- 错误: {str(e)} "
                f"- 耗时: {process_time:.3f}s"
            )
            raise


def register_routes(app: FastAPI):
    """注册路由"""
    
    # 创建控制器实例
    optimization_controller = OptimizationController()
    
    # 注册API路由
    app.include_router(optimization_controller.get_router())
    
    # 根路径
    @app.get("/", tags=["基础"])
    async def root():
        """服务根路径"""
        service_info = get_service_info()
        return ApiResponse.success_response(
            data=service_info,
            message="智能文本优化服务运行中"
        )
    
    # 健康检查
    @app.get("/health", tags=["健康检查"])
    async def health():
        """简单健康检查"""
        return {
            "status": "healthy",
            "service": "intelligent-text-optimization-service",
            "timestamp": asyncio.get_event_loop().time()
        }
    
    # 服务信息
    @app.get("/info", tags=["基础"])
    async def info():
        """获取服务信息"""
        return ApiResponse.success_response(
            data=get_service_info(),
            message="服务信息"
        )


def configure_exception_handlers(app: FastAPI):
    """配置异常处理器"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP异常处理"""
        logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiResponse.error_response(
                message=exc.detail,
                error_details={
                    "status_code": exc.status_code,
                    "path": request.url.path,
                    "method": request.method
                }
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """请求验证异常处理"""
        logger.warning(f"请求验证异常: {exc}")
        
        return JSONResponse(
            status_code=422,
            content=ApiResponse.error_response(
                message="请求参数验证失败",
                error_details={
                    "validation_errors": exc.errors(),
                    "path": request.url.path,
                    "method": request.method
                }
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理"""
        logger.error(f"服务异常: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=ApiResponse.error_response(
                message="服务器内部错误",
                error_details={
                    "error_type": type(exc).__name__,
                    "path": request.url.path,
                    "method": request.method
                }
            ).dict()
        )


# 创建应用实例
app = create_app()


def main():
    """主函数"""
    settings = get_settings()
    
    logger.info(f"启动智能文本优化服务...")
    logger.info(f"服务配置: {settings.service_host}:{settings.service_port}")
    logger.info(f"环境: {settings.service_environment}")
    logger.info(f"调试模式: {settings.debug}")
    
    try:
        uvicorn.run(
            "src.main:app",
            host=settings.service_host,
            port=settings.service_port,
            log_level="info" if settings.debug else "warning",
            reload=settings.debug,
            workers=1 if settings.debug else 4,
            access_log=settings.debug
        )
    except KeyboardInterrupt:
        logger.info("服务被手动停止")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()