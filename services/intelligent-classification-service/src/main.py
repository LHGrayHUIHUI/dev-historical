"""
智能分类服务主应用
无状态智能分类微服务，专注于机器学习文本分类
通过storage-service管理所有数据持久化
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import uvicorn
from contextlib import asynccontextmanager

from .config.settings import settings
from .controllers import (
    project_router,
    model_router,
    classification_router,
    data_router
)
from .clients.storage_client import StorageServiceClient


# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 智能分类服务启动中...")
    
    # 启动时初始化
    try:
        # 测试storage service连接
        storage_client = StorageServiceClient()
        health = await storage_client.health_check()
        logger.info(f"Storage service连接状态: {health.get('status', 'unknown')}")
        
        logger.info("✅ 智能分类服务启动完成")
    except Exception as e:
        logger.error(f"❌ 启动时检查失败: {e}")
    
    yield
    
    # 关闭时清理
    logger.info("🛑 智能分类服务关闭中...")
    logger.info("✅ 智能分类服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="智能分类服务",
    description="""
    历史文本智能分类微服务
    
    ## 主要功能
    
    * **项目管理** - 创建和管理分类项目
    * **模型训练** - 训练和管理机器学习模型  
    * **文档分类** - 单个和批量文档分类
    * **训练数据** - 管理训练数据集
    
    ## 技术特点
    
    * 🚀 **无状态架构** - 通过storage-service统一管理数据
    * 🤖 **多算法支持** - SVM、RandomForest、XGBoost、BERT等
    * 🌏 **中文优化** - 针对历史中文文档优化
    * 📊 **性能监控** - 完整的模型性能跟踪
    
    ## API版本
    
    当前版本: v1.0.0
    """,
    version=settings.service_version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    lifespan=lifespan
)


# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# 请求处理中间件
@app.middleware("http")
async def request_processing_middleware(request: Request, call_next):
    """请求处理中间件：添加请求跟踪和性能监控"""
    start_time = time.time()
    
    # 记录请求信息
    logger.info(f"📥 {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Service-Name"] = settings.service_name
        response.headers["X-Service-Version"] = settings.service_version
        
        logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - 处理失败: {e} - {process_time:.3f}s")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "服务内部错误",
                "error": str(e),
                "timestamp": time.time()
            }
        )


# 注册路由
app.include_router(project_router)
app.include_router(model_router)
app.include_router(classification_router)
app.include_router(data_router)


# 根路径
@app.get("/")
async def root():
    """服务根路径"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "environment": settings.environment,
        "status": "运行中",
        "description": "历史文本智能分类微服务",
        "endpoints": {
            "docs": f"{settings.api_prefix}/docs",
            "health": "/health",
            "projects": f"{settings.api_prefix}/projects",
            "models": f"{settings.api_prefix}/models",
            "classify": f"{settings.api_prefix}/classify",
            "data": f"{settings.api_prefix}/data"
        }
    }


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查storage service连接
        storage_client = StorageServiceClient()
        storage_health = await storage_client.health_check()
        
        is_healthy = storage_health.get('status') == 'healthy'
        
        return {
            "service": settings.service_name,
            "version": settings.service_version,
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": time.time(),
            "dependencies": {
                "storage_service": storage_health.get('status', 'unknown')
            },
            "system_info": {
                "environment": settings.environment,
                "debug": settings.debug,
                "port": settings.api_port
            }
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "service": settings.service_name,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


# 就绪检查（Kubernetes）
@app.get("/ready")
async def readiness_check():
    """就绪检查端点"""
    try:
        # 检查关键依赖
        storage_client = StorageServiceClient()
        await storage_client.health_check()
        
        return {
            "status": "ready",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": time.time()
            }
        )


# Prometheus指标端点
@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    # 这里可以实现Prometheus指标收集
    # 当前返回基本指标
    return {
        "intelligent_classification_service_info": {
            "version": settings.service_version,
            "environment": settings.environment
        },
        "intelligent_classification_service_up": 1,
        "intelligent_classification_service_start_time": time.time()
    }


# 服务信息
@app.get("/info")
async def service_info():
    """服务详细信息"""
    return {
        "service": {
            "name": settings.service_name,
            "version": settings.service_version,
            "environment": settings.environment,
            "debug": settings.debug
        },
        "api": {
            "prefix": settings.api_prefix,
            "host": settings.api_host,
            "port": settings.api_port
        },
        "features": {
            "classification_types": list(settings.predefined_labels.keys()),
            "model_types": list(settings.ml_models.keys()),
            "feature_extractors": list(settings.feature_extraction.keys()),
            "max_text_length": settings.max_text_length,
            "max_batch_size": settings.max_batch_size
        },
        "dependencies": {
            "storage_service": settings.storage_service_url,
            "nlp_service": settings.nlp_service_url,
            "knowledge_graph_service": settings.knowledge_graph_service_url
        },
        "performance": {
            "thresholds": settings.performance_thresholds,
            "max_concurrent_tasks": settings.max_concurrent_tasks,
            "classification_timeout": settings.classification_timeout
        }
    }


# 异常处理器
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404错误处理"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "请求的资源不存在",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """500错误处理"""
    logger.error(f"内部服务器错误: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务内部错误",
            "timestamp": time.time()
        }
    )


# 主函数
def main():
    """主函数：启动服务"""
    logger.info(f"🚀 启动智能分类服务...")
    logger.info(f"📍 环境: {settings.environment}")
    logger.info(f"🌐 地址: {settings.api_host}:{settings.api_port}")
    logger.info(f"📚 文档: http://{settings.api_host}:{settings.api_port}{settings.api_prefix}/docs")
    
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()