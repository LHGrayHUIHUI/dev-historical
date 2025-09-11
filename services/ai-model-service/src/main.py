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
from .controllers.models_management_controller import create_models_management_controller
from .services.ai_service_simplified import get_ai_service


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


# 创建FastAPI应用 - Create FastAPI Application
app = FastAPI(
    title="AI模型统一服务 - Unified AI Model Service",
    description="""
## 🤖 AI模型统一调用和管理服务

### 核心功能 Core Features
- **多AI提供商支持** - 支持Google Gemini、OpenAI GPT、Anthropic Claude等主流AI模型
- **智能模型路由** - 自动选择最佳可用模型，支持负载均衡和故障转移
- **统一API接口** - 兼容OpenAI ChatGPT API格式，简化集成和迁移
- **动态模型管理** - 运行时添加、更新、删除AI模型配置
- **流式响应支持** - 实时对话体验，支持Server-Sent Events
- **连接测试验证** - 自动验证模型可用性和性能监控

### 支持的AI提供商 Supported Providers
- 🔥 **Google Gemini** - gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash
- 🚀 **OpenAI GPT** - gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o
- 💎 **Anthropic Claude** - claude-3-haiku, claude-3-sonnet, claude-3-opus
- 🏠 **本地模型** - 支持Ollama、vLLM、FastChat等本地部署方案

### API端点概览 API Endpoints
- **聊天完成** `POST /api/v1/chat/completions` - 标准AI对话接口
- **流式聊天** `POST /api/v1/chat/completions/stream` - 实时对话流
- **模型列表** `GET /api/v1/models/` - 获取可用模型
- **模型管理** `/api/v1/models/management/` - CRUD模型配置
- **健康检查** `GET /health` - 服务状态监控
- **系统信息** `GET /info` - 详细服务信息

### 技术特性 Technical Features
- ⚡ **高性能异步** - 基于FastAPI和asyncio的异步架构
- 🔒 **安全认证** - JWT token支持和API密钥加密存储
- 📊 **监控统计** - 详细的使用统计和性能指标
- 🔄 **故障恢复** - 自动重试和故障转移机制
- 🌐 **跨域支持** - 完整的CORS配置
- 📝 **中文文档** - 完整的中文API文档和注释

### 部署说明 Deployment
支持Docker容器化部署，配置简单，开箱即用。
详细配置请参考项目文档和环境变量说明。
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "AI模型服务团队",
        "email": "ai-service@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
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


# 根路径 - Root Endpoint  
@app.get("/", 
         summary="服务根目录",
         description="获取AI模型服务的基本信息和API端点概览",
         response_description="服务基本信息和支持的端点列表")
async def root() -> Dict[str, Any]:
    """
    AI模型服务根目录 - AI Model Service Root Endpoint
    
    功能描述:
    返回服务的基本信息、版本号和所有可用API端点的概览。
    这是了解服务功能的入口端点。
    
    响应内容:
    - service: 服务名称
    - version: 当前版本号
    - description: 服务功能描述
    - docs: API文档地址
    - health: 健康检查端点
    - supported_endpoints: 所有可用API端点列表
    
    使用场景:
    - API探索和发现
    - 服务状态确认
    - 前端集成参考
    - 系统集成指南
    """
    return {
        "service": "AI模型服务",
        "version": "1.0.0",
        "description": "统一的AI模型调用和管理服务",
        "docs": "/docs",
        "health": "/health",
        "supported_endpoints": {
            "chat": "/api/v1/chat/completions",
            "chat_stream": "/api/v1/chat/completions/stream", 
            "models": "/api/v1/models/",
            "providers": "/api/v1/models/providers",
            "status": "/api/v1/status/health",
            "metrics": "/api/v1/status/metrics"
        }
    }


# 简单健康检查端点（Docker使用）- Health Check Endpoint for Docker
@app.get("/health",
         summary="服务健康检查",
         description="检查AI模型服务的运行状态和健康度",
         response_description="服务健康状态信息")
async def health_check() -> Dict[str, Any]:
    """
    服务健康检查端点 - Service Health Check Endpoint
    
    功能描述:
    检查AI模型服务的基本运行状态，包括服务可用性和AI模型连接状态。
    主要用于Docker容器健康检查和负载均衡器探测。
    
    响应内容 (健康):
    - status: "healthy" - 服务状态正常
    - service: "ai-model-service" - 服务标识
    - timestamp: 检查时间戳
    - mode: "simplified"/"full" - 服务运行模式
    
    响应内容 (不健康):
    - status: "unhealthy" - 服务状态异常
    - service: "ai-model-service" - 服务标识
    - error: 错误详细信息
    
    使用场景:
    - Docker容器健康检查
    - Kubernetes liveness probe
    - 负载均衡器健康探测
    - 监控系统状态收集
    
    技术说明:
    - 轻量级检查，响应快速
    - 包含AI服务初始化状态
    - 支持简化模式和完整模式
    """
    try:
        ai_service = await get_ai_service()
        health_data = await ai_service.health_check()
        
        return {
            "status": "healthy",
            "service": "ai-model-service",
            "timestamp": health_data.get("timestamp"),
            "mode": "simplified" if health_data.get("simplified_mode") else "full"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "ai-model-service", 
            "error": str(e)
        }


# 服务信息 - Service Information Endpoint
@app.get("/info",
         summary="获取服务详细信息",
         description="查询AI模型服务的完整配置和状态信息",
         response_description="服务详细信息包括配置、状态和功能特性")
async def service_info() -> Dict[str, Any]:
    """
    获取服务详细信息端点 - Service Information Endpoint
    
    功能描述:
    返回AI模型服务的完整信息，包括服务配置、运行状态、
    功能特性、配置参数等详细信息。
    
    响应内容:
    - service: 服务基本信息
      - name: 服务名称
      - version: 版本号
      - port: 服务端口
      - environment: 运行环境
      - features: 功能特性列表
    - status: 服务运行状态
    - configuration: 配置信息
      - storage_service_url: 存储服务URL
      - redis_enabled: Redis缓存状态
      - health_check_interval: 健康检查间隔
      - cache_ttl_*: 缓存过期时间配置
    
    功能特性包含:
    - 多平台AI模型支持
    - 智能模型路由
    - 负载均衡
    - 账号池管理
    - 健康监控
    - 使用统计
    - 成本分析
    - 流式响应
    
    使用场景:
    - 系统管理和监控
    - 故障排查和诊断
    - 配置验证和确认
    - 系统集成和对接
    - 性能分析和优化
    """
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
models_management_controller = create_models_management_controller()

app.include_router(chat_controller.router)
app.include_router(models_controller.router)
app.include_router(status_controller.router)
app.include_router(models_management_controller.router)


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