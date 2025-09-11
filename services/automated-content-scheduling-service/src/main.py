"""
自动内容调度服务主应用入口
基于FastAPI的高性能异步Web服务
"""
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# 导入配置和组件
from .config.settings import get_settings
from .models import init_database, close_database
from .controllers import scheduling_router, analytics_router, system_router
from .utils.exceptions import SchedulingServiceError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/scheduling-service.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# 获取配置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("自动内容调度服务正在启动...")
    
    try:
        # 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")
        
        # 初始化ML优化服务（如果需要）
        # await optimization_service.initialize()
        # logger.info("ML优化服务初始化完成")
        
        logger.info(f"自动内容调度服务启动成功 - 版本: {settings.version}")
        
        yield
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise
    finally:
        # 关闭时执行清理
        logger.info("自动内容调度服务正在关闭...")
        
        try:
            # 关闭数据库连接
            await close_database()
            logger.info("数据库连接已关闭")
            
        except Exception as e:
            logger.error(f"服务关闭时出错: {e}")
        
        logger.info("自动内容调度服务已关闭")


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    
    app = FastAPI(
        title=settings.app_name,
        description="智能自动内容调度服务，提供多平台内容发布的智能调度、优化和管理功能",
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None
    )
    
    # 配置中间件
    setup_middleware(app)
    
    # 注册路由
    register_routes(app)
    
    # 注册异常处理器
    register_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI):
    """配置应用中间件"""
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.security.allowed_methods,
        allow_headers=["*"],
    )
    
    # 可信主机中间件
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]  # 生产环境应该配置具体主机
    )
    
    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """记录请求日志"""
        start_time = request.state.start_time = logger.info(f"开始处理请求: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            
            # 记录成功响应
            process_time = request.state.get('process_time', 0)
            logger.info(
                f"请求完成: {request.method} {request.url} - "
                f"状态码: {response.status_code} - 耗时: {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # 记录异常
            logger.error(f"请求处理异常: {request.method} {request.url} - 错误: {e}")
            raise


def register_routes(app: FastAPI):
    """注册API路由"""
    
    # 注册各个模块的路由
    app.include_router(scheduling_router, prefix="")
    app.include_router(analytics_router, prefix="")
    app.include_router(system_router, prefix="")
    
    # 根路径
    @app.get("/", tags=["根路径"])
    async def root():
        """服务根路径信息"""
        return {
            "service": settings.app_name,
            "version": settings.version,
            "environment": settings.environment.value,
            "status": "running",
            "docs_url": "/docs" if settings.debug else "disabled",
            "health_check": "/api/v1/system/health"
        }
    
    # 简单的健康检查端点（用于负载均衡器）
    @app.get("/health", tags=["健康检查"])
    async def simple_health_check():
        """简单健康检查"""
        return {"status": "healthy", "service": "automated-content-scheduling"}
    
    # 就绪检查端点
    @app.get("/ready", tags=["健康检查"])
    async def simple_ready_check():
        """简单就绪检查"""
        return {"status": "ready", "service": "automated-content-scheduling"}


def register_exception_handlers(app: FastAPI):
    """注册异常处理器"""
    
    @app.exception_handler(SchedulingServiceError)
    async def scheduling_error_handler(request: Request, exc: SchedulingServiceError):
        """处理调度服务相关异常"""
        logger.error(f"调度服务异常: {exc.message} - 详情: {exc.details}")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "调度服务错误",
                "message": exc.message,
                "details": exc.details,
                "timestamp": logger.info("当前时间").isoformat(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理通用异常"""
        logger.error(f"未处理的异常: {exc}", exc_info=True)
        
        if settings.debug:
            # 开发环境返回详细错误信息
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "内部服务器错误",
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "timestamp": logger.info("当前时间").isoformat(),
                    "path": str(request.url)
                }
            )
        else:
            # 生产环境返回通用错误信息
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "内部服务器错误",
                    "message": "服务暂时不可用，请稍后重试",
                    "timestamp": logger.info("当前时间").isoformat()
                }
            )


# 创建应用实例
app = create_app()


def main():
    """主函数 - 直接运行服务"""
    logger.info(f"启动自动内容调度服务 - 环境: {settings.environment.value}")
    
    # 运行配置
    run_config = {
        "app": "src.main:app",
        "host": settings.host,
        "port": settings.port,
        "reload": settings.debug and settings.environment.value == "development",
        "log_level": settings.monitoring.log_level.lower(),
        "access_log": True,
        "server_header": False,  # 隐藏服务器标识
        "date_header": True
    }
    
    # 开发环境使用自动重载
    if settings.debug:
        run_config.update({
            "reload": True,
            "reload_dirs": ["src"],
            "reload_excludes": ["*.pyc", "__pycache__", "logs", "models", "test-results"]
        })
    
    # 生产环境使用多进程
    if settings.environment.value == "production":
        import multiprocessing
        run_config.update({
            "workers": multiprocessing.cpu_count(),
            "worker_class": "uvicorn.workers.UvicornWorker"
        })
    
    # 启动服务
    uvicorn.run(**run_config)


if __name__ == "__main__":
    main()