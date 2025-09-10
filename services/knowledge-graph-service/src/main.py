"""
知识图谱构建服务主应用
基于FastAPI的无状态微服务架构
专注于历史文本的知识图谱构建和智能查询
"""

import asyncio
import sys
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger

# 添加src路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from controllers.knowledge_graph_controller import router as kg_router
from services.knowledge_graph_service import KnowledgeGraphService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时初始化资源，关闭时清理资源
    """
    # 启动时的初始化
    logger.info("知识图谱服务启动中...")
    
    try:
        # 初始化知识图谱服务
        kg_service = KnowledgeGraphService()
        await kg_service.initialize()
        
        # 将服务实例存储到应用状态
        app.state.kg_service = kg_service
        
        logger.info("知识图谱服务初始化完成")
        logger.info(f"服务版本: {settings.service_version}")
        logger.info(f"运行环境: {settings.environment}")
        logger.info(f"API端口: {settings.api_port}")
        logger.info(f"Storage服务: {settings.storage_service_url}")
        
        yield
        
    finally:
        # 关闭时的清理
        logger.info("知识图谱服务关闭中...")
        
        if hasattr(app.state, 'kg_service'):
            await app.state.kg_service.cleanup()
        
        logger.info("知识图谱服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="历史文本知识图谱构建服务",
    description="""
    专业的历史文本知识图谱构建和查询服务
    
    ## 主要功能
    
    * **实体抽取**: 支持spaCy、BERT、jieba等多种方法的命名实体识别
    * **关系抽取**: 基于规则和机器学习的实体关系抽取
    * **图谱构建**: 完整的知识图谱构建和优化
    * **智能查询**: 多种方式的图谱查询和分析
    * **概念挖掘**: 基于主题模型的概念发现
    * **批量处理**: 大规模文档的并行处理
    
    ## 技术特点
    
    * 无状态架构，所有数据通过storage-service管理
    * 支持中英文双语处理
    * 支持多种NLP模型和算法
    * 完整的API文档和错误处理
    * 异步处理和后台任务支持
    
    ## 联系方式
    
    * 项目地址: [Historical Text Project](https://github.com/yourorg/historical-text-project)
    * API版本: v1
    * 服务端口: 8006
    """,
    version=settings.service_version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 配置受信任主机中间件
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # 在生产环境中应该配置具体的主机列表
)

# 注册路由
app.include_router(kg_router, prefix=settings.api_prefix)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    请求日志中间件
    记录所有HTTP请求的基本信息和处理时间
    """
    import time
    
    start_time = time.time()
    
    # 记录请求开始
    logger.info(
        f"请求开始: {request.method} {request.url.path} "
        f"来源IP: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录请求完成
        logger.info(
            f"请求完成: {request.method} {request.url.path} "
            f"状态码: {response.status_code} "
            f"处理时间: {process_time:.4f}s"
        )
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        # 记录请求错误
        process_time = time.time() - start_time
        logger.error(
            f"请求异常: {request.method} {request.url.path} "
            f"错误: {str(e)} "
            f"处理时间: {process_time:.4f}s"
        )
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    HTTP异常处理器
    统一处理HTTP异常并返回标准格式的错误响应
    """
    logger.error(
        f"HTTP异常: {request.method} {request.url.path} "
        f"状态码: {exc.status_code} "
        f"详情: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None,
            "error_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    通用异常处理器
    处理所有未被捕获的异常
    """
    logger.error(
        f"未处理异常: {request.method} {request.url.path} "
        f"异常类型: {type(exc).__name__} "
        f"异常信息: {str(exc)}"
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务内部错误",
            "data": None,
            "error_code": 500,
            "path": str(request.url.path),
            "method": request.method,
            "error_type": type(exc).__name__
        }
    )


@app.get("/", include_in_schema=False)
async def root():
    """
    根路径处理
    返回服务基本信息
    """
    return {
        "service": "knowledge-graph-service",
        "version": settings.service_version,
        "status": "running",
        "environment": settings.environment,
        "api_docs": f"{settings.api_prefix}/docs",
        "health_check": f"{settings.api_prefix}/knowledge-graph/health"
    }


@app.get("/health", include_in_schema=False)
async def health_check():
    """
    基础健康检查端点
    用于容器和负载均衡器的健康检查
    """
    return {
        "status": "healthy",
        "service": "knowledge-graph-service",
        "version": settings.service_version,
        "timestamp": "2024-01-01T00:00:00Z"  # 实际应该使用当前时间
    }


@app.get("/ready", include_in_schema=False)
async def readiness_check():
    """
    就绪检查端点
    检查服务是否准备好接收请求
    """
    try:
        # 检查关键依赖是否可用
        if hasattr(app.state, 'kg_service'):
            # 简单的依赖检查
            await app.state.kg_service.storage_client.health_check()
            
            return {
                "status": "ready",
                "service": "knowledge-graph-service",
                "version": settings.service_version,
                "dependencies": "healthy"
            }
        else:
            raise Exception("知识图谱服务未初始化")
            
    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "knowledge-graph-service",
                "error": str(e)
            }
        )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """
    Prometheus指标端点
    返回服务的监控指标
    """
    # 这里应该返回Prometheus格式的指标
    # 实际实现需要集成prometheus_client库
    return {
        "note": "指标端点需要集成prometheus_client库",
        "service": "knowledge-graph-service",
        "status": "placeholder"
    }


def configure_logging():
    """
    配置日志系统
    使用loguru进行结构化日志记录
    """
    # 移除默认的日志处理器
    logger.remove()
    
    # 添加控制台日志处理器
    logger.add(
        sys.stdout,
        format=settings.log_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件日志处理器
    if settings.log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
        
        logger.add(
            settings.log_file,
            format=settings.log_format,
            level=settings.log_level,
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
    
    # 设置uvicorn使用loguru
    import logging
    logging.getLogger("uvicorn").handlers = []
    logging.getLogger("uvicorn.access").handlers = []


if __name__ == "__main__":
    """
    应用启动入口
    支持开发和生产环境的不同配置
    """
    # 配置日志
    configure_logging()
    
    # 根据环境选择不同的启动配置
    if settings.environment == "development":
        # 开发环境配置
        uvicorn.run(
            "main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,
            reload_dirs=["src"],
            access_log=True,
            log_level=settings.log_level.lower()
        )
    else:
        # 生产环境配置
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            workers=settings.workers,
            access_log=True,
            log_level=settings.log_level.lower(),
            loop="uvloop",  # 使用高性能事件循环
            http="httptools"  # 使用高性能HTTP解析器
        )