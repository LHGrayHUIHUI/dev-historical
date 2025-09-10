"""
AI模型服务主应用
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import get_settings
from .controllers.chat_controller import create_chat_controller
from .controllers.models_controller import create_models_controller
from .controllers.status_controller import create_status_controller
from .services.ai_service import get_ai_service


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ai-model-service.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时的初始化
    logger.info("Starting AI Model Service...")
    
    try:
        # 初始化AI服务
        ai_service = await get_ai_service()
        logger.info("AI Model Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Model Service: {e}")
        raise
    
    finally:
        # 关闭时的清理
        logger.info("Shutting down AI Model Service...")
        try:
            ai_service = await get_ai_service()
            await ai_service.shutdown()
            logger.info("AI Model Service shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="AI模型服务",
    description="统一的AI模型调用和管理服务，支持多平台AI模型智能路由",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 获取配置
settings = get_settings()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "内部服务器错误",
                "type": "internal_error"
            }
        }
    )


# 404处理
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404错误处理器"""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": "请求的资源不存在",
                "type": "not_found"
            }
        }
    )


# 根路径
@app.get("/")
async def root() -> Dict[str, Any]:
    """服务根路径"""
    return {
        "service": "AI模型服务",
        "version": "1.0.0",
        "description": "统一的AI模型调用和管理服务",
        "docs": "/docs",
        "health": "/api/v1/status/health",
        "supported_endpoints": {
            "chat": "/api/v1/chat/completions",
            "chat_stream": "/api/v1/chat/completions/stream", 
            "models": "/api/v1/models/",
            "providers": "/api/v1/models/providers",
            "status": "/api/v1/status/health",
            "metrics": "/api/v1/status/metrics"
        }
    }


# 服务信息
@app.get("/info")
async def service_info() -> Dict[str, Any]:
    """获取服务详细信息"""
    try:
        ai_service = await get_ai_service()
        service_status = await ai_service.get_service_status()
        
        return {
            "service": {
                "name": "AI模型服务",
                "version": "1.0.0",
                "port": settings.service_port,
                "environment": "development",  # 可以从环境变量读取
                "features": [
                    "多平台AI模型支持",
                    "智能模型路由",
                    "负载均衡",
                    "账号池管理",
                    "健康监控",
                    "使用统计",
                    "成本分析",
                    "流式响应"
                ]
            },
            "status": service_status,
            "configuration": {
                "storage_service_url": settings.storage_service_url,
                "redis_enabled": bool(settings.redis_url),
                "health_check_interval": settings.health_check_interval,
                "cache_ttl_models": settings.cache_ttl_models,
                "cache_ttl_accounts": settings.cache_ttl_accounts
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting service info: {e}")
        return {
            "service": {
                "name": "AI模型服务", 
                "version": "1.0.0",
                "status": "error"
            },
            "error": str(e)
        }


# 注册路由
chat_controller = create_chat_controller()
models_controller = create_models_controller()
status_controller = create_status_controller()

app.include_router(chat_controller.router)
app.include_router(models_controller.router)
app.include_router(status_controller.router)


# 开发模式下的调试端点
if settings.debug:
    @app.get("/debug/config")
    async def debug_config():
        """调试配置信息（仅开发模式）"""
        return {
            "settings": {
                "service_port": settings.service_port,
                "storage_service_url": settings.storage_service_url,
                "redis_url": settings.redis_url and "已配置" or "未配置",
                "debug": settings.debug,
                "log_level": settings.log_level,
                "health_check_interval": settings.health_check_interval,
                "quota_alert_threshold": settings.quota_alert_threshold,
                "cache_ttl_models": settings.cache_ttl_models,
                "cache_ttl_accounts": settings.cache_ttl_accounts,
                "default_routing_strategy": settings.default_routing_strategy,
                "cache_prefix": settings.cache_prefix
            }
        }


if __name__ == "__main__":
    import uvicorn
    
    # 创建logs目录
    import os
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Starting AI Model Service on port {settings.service_port}")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )