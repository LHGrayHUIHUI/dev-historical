"""
Content Quality Assessment Service 主应用入口

内容质量评估服务，提供智能化的多维度内容质量评估功能。
支持质量评估、趋势分析、基准管理和批量处理等功能。
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
from .controllers.assessment_controller import router as assessment_router
from .services.assessment_engine import assessment_engine
from .services.trend_analyzer import trend_analyzer
from .services.benchmark_manager import benchmark_manager
from .clients.storage_client import storage_client

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
        logger.info("Content Quality Assessment Service starting up...")
        
        # 启动时初始化
        logger.info("Initializing core services...")
        
        # 初始化评估引擎
        await assessment_engine.initialize()
        logger.info("Assessment engine initialized")
        
        # 测试外部服务连接
        storage_healthy = await storage_client.health_check()
        if not storage_healthy:
            logger.warning("Storage service not available, continuing with degraded mode")
        
        # 初始化默认基准
        await _initialize_default_benchmarks()
        
        logger.info(f"Content Quality Assessment Service started successfully on port {settings.service.port}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Content Quality Assessment Service: {str(e)}")
        raise
    finally:
        # 关闭时清理
        logger.info("Content Quality Assessment Service shutting down...")
        
        # 清理评估引擎
        try:
            await assessment_engine.cleanup()
            logger.info("Assessment engine cleanup completed")
        except Exception as e:
            logger.error(f"Error cleaning up assessment engine: {str(e)}")
        
        # 关闭客户端连接
        try:
            await storage_client._close_session()
            logger.info("External service connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")

async def _initialize_default_benchmarks():
    """初始化默认基准"""
    try:
        from .models.assessment_models import ContentType
        
        # 为每种内容类型创建默认基准（如果不存在）
        for content_type in ContentType:
            existing_benchmark = await benchmark_manager.get_default_benchmark(content_type)
            if existing_benchmark:
                logger.debug(f"Default benchmark exists for {content_type.value}")
            else:
                logger.info(f"Created default benchmark for {content_type.value}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize default benchmarks: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="Content Quality Assessment Service",
    description="内容质量评估系统 - 智能化多维度内容质量评估平台",
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
    allow_credentials=settings.api.cors_allow_credentials,
    allow_methods=settings.api.cors_allow_methods,
    allow_headers=settings.api.cors_allow_headers,
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

# 请求限流中间件
if settings.security.enable_rate_limiting:
    from collections import defaultdict
    from datetime import datetime, timedelta
    
    request_counts = defaultdict(list)
    
    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        client_ip = request.client.host
        now = datetime.now()
        
        # 清理过期记录
        cutoff_time = now - timedelta(minutes=1)
        request_counts[client_ip] = [
            req_time for req_time in request_counts[client_ip] 
            if req_time > cutoff_time
        ]
        
        # 检查频率限制
        if len(request_counts[client_ip]) >= settings.security.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "请求频率过高，请稍后再试",
                    "error": "Rate limit exceeded"
                }
            )
        
        # 记录当前请求
        request_counts[client_ip].append(now)
        
        response = await call_next(request)
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

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error": f"HTTP {exc.status_code}"
        }
    )

# 注册路由
app.include_router(
    assessment_router,
    tags=["质量评估"]
)

# 根路径
@app.get("/", tags=["系统"])
async def root():
    """根路径"""
    return {
        "message": "Content Quality Assessment Service",
        "description": "内容质量评估服务API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/quality/health",
        "features": [
            "多维度质量评估",
            "质量趋势分析", 
            "基准管理和对比",
            "批量评估处理",
            "智能改进建议"
        ]
    }

# 健康检查端点（简化版，完整版在controller中）
@app.get("/health", tags=["系统"])
async def simple_health_check():
    """简化健康检查"""
    try:
        return {
            "status": "healthy",
            "service": "content-quality-assessment-service",
            "version": "1.0.0",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "content-quality-assessment-service",
                "error": str(e)
            }
        )

@app.get("/ready", tags=["系统"])
async def simple_readiness_check():
    """简化就绪检查"""
    try:
        # 验证关键组件是否就绪
        engine_ready = assessment_engine.nlp is not None
        
        if engine_ready:
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
        "service": settings.service.name,
        "version": settings.service.version,
        "description": settings.service.description,
        "port": settings.service.port,
        "environment": settings.service.environment,
        "features": {
            "assessment_dimensions": settings.assessment_engine.enabled_dimensions,
            "ai_integration": True,
            "storage_integration": True,
            "cache_enabled": settings.assessment_engine.cache_assessment_results,
            "trend_analysis": True,
            "benchmark_management": True,
            "batch_processing": True
        },
        "limits": {
            "max_content_length": settings.assessment_engine.max_content_length,
            "max_batch_size": settings.assessment_engine.max_batch_size,
            "assessment_timeout": settings.assessment_engine.assessment_timeout
        },
        "capabilities": [
            "readability_assessment",
            "accuracy_analysis", 
            "completeness_evaluation",
            "coherence_checking",
            "relevance_scoring",
            "trend_prediction",
            "benchmark_comparison",
            "improvement_suggestions"
        ]
    }

# 性能监控端点
@app.get("/metrics", tags=["监控"])
async def get_metrics():
    """获取性能指标（Prometheus格式）"""
    try:
        # 这里可以集成Prometheus客户端库
        # 暂时返回简单的指标信息
        return {
            "service_info": {
                "name": settings.service.name,
                "version": settings.service.version,
                "uptime_seconds": time.time()  # 简化处理
            },
            "assessment_engine": {
                "enabled_dimensions": len(settings.assessment_engine.enabled_dimensions),
                "cache_enabled": settings.assessment_engine.cache_assessment_results,
                "max_content_length": settings.assessment_engine.max_content_length
            },
            "dependencies": {
                "storage_service": await storage_client.health_check(),
                "nlp_model_loaded": assessment_engine.nlp is not None,
                "redis_connected": assessment_engine.redis is not None if assessment_engine.cache_enabled else True
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Metrics collection failed"}
        )

if __name__ == "__main__":
    # 直接运行时使用uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=settings.service.debug,
        log_level=settings.service.log_level.lower()
    )