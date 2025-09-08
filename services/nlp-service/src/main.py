"""
NLP服务主应用入口
无状态NLP文本处理微服务
数据存储通过storage-service完成
"""

import asyncio
import time
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys

from .config.settings import settings
from .controllers.nlp_controller import router as nlp_router
from .services.nlp_service import NLPService
from .clients.storage_client import storage_client
from .schemas.nlp_schemas import BaseResponse, ServiceInfo, HealthCheckResponse

# 全局变量
nlp_service_instance: NLPService = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global nlp_service_instance
    
    logger.info("NLP服务启动中...")
    
    try:
        # 初始化NLP服务
        nlp_service_instance = NLPService()
        await nlp_service_instance.initialize_models()
        logger.info("NLP模型初始化完成")
        
        # 测试storage-service连接
        try:
            await storage_client.health_check()
            logger.info("Storage-service连接正常")
        except Exception as e:
            logger.warning(f"Storage-service连接失败: {str(e)}")
        
        logger.info("NLP服务启动完成")
        yield
        
    except Exception as e:
        logger.error(f"NLP服务启动失败: {str(e)}")
        raise
    
    finally:
        # 清理资源
        logger.info("NLP服务关闭中...")
        
        if nlp_service_instance:
            try:
                await nlp_service_instance.cleanup()
            except Exception as e:
                logger.error(f"NLP服务清理失败: {str(e)}")
        
        logger.info("NLP服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="NLP文本处理服务",
    description="无状态NLP文本处理微服务，支持分词、词性标注、命名实体识别、情感分析、关键词提取、文本摘要等功能",
    version=settings.service_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(nlp_router)


# ============ 健康检查和服务信息 ============

@app.get("/health", response_model=BaseResponse)
async def health_check():
    """基础健康检查"""
    return BaseResponse(
        success=True,
        message="NLP服务运行正常"
    )


@app.get("/api/v1/health/detailed", response_model=HealthCheckResponse)
async def detailed_health_check():
    """详细健康检查"""
    global nlp_service_instance
    
    # 检查NLP服务状态
    nlp_status = "healthy" if nlp_service_instance and nlp_service_instance.is_initialized else "unhealthy"
    
    # 检查storage-service连接
    storage_status = "unknown"
    try:
        await storage_client.health_check()
        storage_status = "healthy"
    except:
        storage_status = "unhealthy"
    
    # 获取可用引擎
    available_engines = []
    processing_capabilities = []
    
    if nlp_service_instance and nlp_service_instance.is_initialized:
        try:
            engines = nlp_service_instance.get_available_engines()
            available_engines = [engine.name for engine in engines]
        except:
            available_engines = ["jieba"]  # 默认总是可用的
        
        # 处理能力
        if settings.enable_segmentation:
            processing_capabilities.append("segmentation")
        if settings.enable_pos_tagging:
            processing_capabilities.append("pos_tagging")
        if settings.enable_ner:
            processing_capabilities.append("ner")
        if settings.enable_sentiment_analysis:
            processing_capabilities.append("sentiment_analysis")
        if settings.enable_keyword_extraction:
            processing_capabilities.append("keyword_extraction")
        if settings.enable_text_summarization:
            processing_capabilities.append("text_summarization")
        if settings.enable_text_similarity:
            processing_capabilities.append("text_similarity")
    
    service_info = ServiceInfo(
        service_name=settings.service_name,
        version=settings.service_version,
        status=nlp_status,
        uptime=time.time() - start_time,
        available_engines=available_engines,
        processing_capabilities=processing_capabilities
    )
    
    dependencies = {
        "storage-service": storage_status
    }
    
    return HealthCheckResponse(
        success=nlp_status == "healthy",
        message=f"NLP服务状态: {nlp_status}",
        service_info=service_info,
        dependencies=dependencies
    )


@app.get("/ready", response_model=BaseResponse)
async def readiness_check():
    """Kubernetes就绪探针"""
    global nlp_service_instance
    
    if not nlp_service_instance or not nlp_service_instance.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="NLP服务尚未完全初始化"
        )
    
    return BaseResponse(
        success=True,
        message="NLP服务已就绪"
    )


@app.get("/info", response_model=ServiceInfo)
async def service_info():
    """获取服务基本信息"""
    global nlp_service_instance
    
    # 获取可用引擎
    available_engines = []
    processing_capabilities = []
    
    if nlp_service_instance and nlp_service_instance.is_initialized:
        try:
            engines = nlp_service_instance.get_available_engines()
            available_engines = [engine.name for engine in engines]
        except:
            available_engines = ["jieba"]
        
        # 处理能力
        from .config.settings import get_feature_config
        features = get_feature_config()
        processing_capabilities = [k for k, v in features.items() if v]
    
    return ServiceInfo(
        service_name=settings.service_name,
        version=settings.service_version,
        status="running",
        uptime=time.time() - start_time,
        available_engines=available_engines,
        processing_capabilities=processing_capabilities
    )


# ============ 错误处理 ============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未捕获的异常: {str(exc)}")
    
    error_detail = str(exc) if settings.debug_show_error_details else "内部服务器错误"
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务处理异常",
            "error_details": error_detail if settings.debug else None
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}"
        }
    )


# ============ 根路径 ============

@app.get("/", response_model=BaseResponse)
async def root():
    """根路径欢迎信息"""
    return BaseResponse(
        success=True,
        message=f"欢迎使用NLP文本处理服务 v{settings.service_version}！访问 /docs 查看API文档"
    )


# ============ 启动配置 ============

def configure_logging():
    """配置日志"""
    # 移除默认的日志处理器
    logger.remove()
    
    # 添加控制台日志
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=settings.log_format,
        colorize=True
    )
    
    # 添加文件日志（如果配置了）
    if settings.log_file:
        logger.add(
            settings.log_file,
            level=settings.log_level,
            format=settings.log_format,
            rotation=f"{settings.log_max_size} bytes",
            retention=settings.log_backup_count,
            encoding="utf-8"
        )
    
    logger.info(f"日志配置完成，级别: {settings.log_level}")


if __name__ == "__main__":
    # 配置日志
    configure_logging()
    
    # 启动服务
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug_reload_on_change and settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_config=None  # 使用我们自己的日志配置
    )