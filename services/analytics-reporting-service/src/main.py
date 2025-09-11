"""
分析报告服务主应用程序

基于FastAPI的高性能异步Web服务，提供：
- 数据分析API
- 报告生成服务
- 实时数据监控
- 多格式数据导出
- 智能洞察分析
"""

import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .config.settings import settings
from .models import init_database, close_databases, check_database_health
from .controllers import analytics_router
from .services.analytics_service import AnalyticsService
from .services.data_processor import DataProcessor
from .services.report_generator import ReportGenerator

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/analytics-service.log') if not settings.is_development else logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 全局服务实例
analytics_service = None
data_processor = None
report_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global analytics_service, data_processor, report_generator
    
    try:
        logger.info("正在启动分析报告服务...")
        
        # 初始化数据库连接
        await init_database()
        logger.info("数据库连接初始化完成")
        
        # 初始化核心服务
        analytics_service = AnalyticsService()
        await analytics_service.initialize()
        
        data_processor = DataProcessor()
        await data_processor.initialize()
        
        report_generator = ReportGenerator()
        await report_generator.initialize()
        
        logger.info("核心服务初始化完成")
        
        # 检查数据库健康状态
        health_status = await check_database_health()
        logger.info(f"数据库健康状态: {health_status}")
        
        yield  # 应用运行期间
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise
    
    finally:
        # 清理资源
        logger.info("正在关闭分析报告服务...")
        
        try:
            if analytics_service:
                # 如果服务有清理方法，调用清理
                pass
            
            if data_processor:
                # 清理数据处理器资源
                pass
            
            if report_generator:
                # 清理报告生成器资源
                pass
            
            # 关闭数据库连接
            await close_databases()
            logger.info("数据库连接已关闭")
            
        except Exception as e:
            logger.error(f"服务关闭时出错: {e}")
        
        logger.info("分析报告服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="历史文本项目 - 数据分析与报告服务",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# 自定义异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未捕获的异常: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误" if settings.is_production else str(exc),
            "code": 500,
            "path": str(request.url.path)
        }
    )


# 健康检查端点
@app.get("/health", tags=["系统"])
async def health_check():
    """基础健康检查"""
    return {"status": "healthy", "service": "analytics-reporting-service"}


@app.get("/ready", tags=["系统"])
async def readiness_check():
    """就绪状态检查"""
    try:
        # 检查数据库连接
        health_status = await check_database_health()
        
        all_healthy = all(health_status.values())
        
        return {
            "status": "ready" if all_healthy else "not_ready",
            "databases": health_status,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e)
            }
        )


@app.get("/info", tags=["系统"])
async def service_info():
    """服务信息"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "python_version": sys.version,
        "uptime": asyncio.get_event_loop().time()
    }


# 注册路由
app.include_router(
    analytics_router,
    prefix=settings.api_v1_prefix,
    tags=["数据分析"]
)

# 简化的其他路由器（实际项目中会包含完整实现）
@app.get(f"{settings.api_v1_prefix}/reports", tags=["报告管理"])
async def get_reports():
    """获取报告列表 - 简化实现"""
    return {"message": "报告管理功能开发中"}

@app.get(f"{settings.api_v1_prefix}/templates", tags=["模板管理"])
async def get_templates():
    """获取模板列表 - 简化实现"""
    return {"message": "模板管理功能开发中"}


# 自定义OpenAPI文档
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        # 分析报告服务 API

        ## 功能特性

        - **数据分析**: 内容表现、平台对比、趋势分析、用户行为分析
        - **异常检测**: 基于机器学习的智能异常检测
        - **报告生成**: 支持PDF、Excel、JSON多种格式
        - **实时监控**: 实时数据指标和质量监控
        - **数据导出**: 灵活的数据导出功能

        ## 数据库支持

        - **PostgreSQL**: 关系数据存储
        - **InfluxDB**: 时序数据存储  
        - **ClickHouse**: OLAP分析数据库
        - **Redis**: 缓存和实时数据

        ## 技术栈

        - **FastAPI**: 高性能异步Web框架
        - **pandas/NumPy**: 数据处理和分析
        - **scikit-learn**: 机器学习算法
        - **matplotlib/plotly**: 数据可视化
        - **reportlab**: PDF报告生成
        """,
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "/static/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# 中间件：请求日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录HTTP请求"""
    start_time = asyncio.get_event_loop().time()
    
    # 记录请求
    logger.info(f"请求开始: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # 计算处理时间
        process_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(
            f"请求完成: {request.method} {request.url.path} - "
            f"状态码: {response.status_code} - 耗时: {process_time:.3f}s"
        )
        
        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Service-Version"] = settings.app_version
        
        return response
        
    except Exception as e:
        process_time = asyncio.get_event_loop().time() - start_time
        logger.error(
            f"请求错误: {request.method} {request.url.path} - "
            f"错误: {str(e)} - 耗时: {process_time:.3f}s"
        )
        raise


# 启动函数
def start_server():
    """启动服务器"""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.is_development,
        workers=1 if settings.is_development else 4,
        access_log=settings.is_development,
        use_colors=True,
        server_header=False,
        date_header=False
    )


if __name__ == "__main__":
    start_server()