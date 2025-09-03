"""
监控指标中间件模块

此模块提供FastAPI应用的Prometheus监控指标收集功能，
包括HTTP请求计数、响应时间统计、活跃连接监控等核心指标。

主要功能：
- HTTP请求指标收集（请求数、响应时间、状态码分布）
- 活跃连接数监控
- 业务指标收集（文件上传、文本处理等）
- 自定义指标支持

Author: 开发团队
Created: 2025-09-03
Version: 1.0.0
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
from typing import Dict, Any, Optional
import structlog
import asyncio
import os

# 获取结构化日志记录器
logger = structlog.get_logger()

class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus指标收集中间件
    
    负责收集HTTP请求的各种监控指标，包括请求计数、
    响应时间分布、错误率统计等，为监控系统提供数据支持。
    
    收集的指标包括：
    - http_requests_total: HTTP请求总数（按方法、路径、状态码分类）
    - http_request_duration_seconds: HTTP请求持续时间直方图
    - active_connections: 当前活跃连接数
    - application_info: 应用信息（版本、环境等）
    
    Attributes:
        service_name: 服务名称，用于标识不同的微服务
        http_requests_total: HTTP请求总数计数器
        http_request_duration: HTTP请求持续时间直方图
        active_connections: 活跃连接数计量器
        application_info: 应用信息指标
    """
    
    def __init__(self, app: ASGIApp, service_name: str = "unknown-service"):
        """初始化监控中间件
        
        Args:
            app: ASGI应用实例
            service_name: 服务名称标识
        """
        super().__init__(app)
        self.service_name = service_name
        
        # HTTP请求总数计数器
        self.http_requests_total = Counter(
            name='http_requests_total',
            documentation='HTTP请求总数',
            labelnames=['method', 'endpoint', 'status', 'service']
        )
        
        # HTTP请求持续时间直方图
        # 使用适合Web应用的响应时间分桶
        self.http_request_duration = Histogram(
            name='http_request_duration_seconds',
            documentation='HTTP请求持续时间（秒）',
            labelnames=['method', 'endpoint', 'status', 'service'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # 活跃连接数计量器
        self.active_connections = Gauge(
            name='active_connections',
            documentation='当前活跃HTTP连接数',
            labelnames=['service']
        )
        
        # 应用信息指标
        self.application_info = Info(
            name='application_info',
            documentation='应用版本和环境信息'
        )
        
        # 设置应用信息
        self._set_application_info()
        
        logger.info(
            "Prometheus监控中间件初始化完成",
            service_name=self.service_name
        )
    
    def _set_application_info(self):
        """设置应用信息指标"""
        try:
            app_info = {
                'service_name': self.service_name,
                'version': os.getenv('SERVICE_VERSION', '1.0.0'),
                'environment': os.getenv('ENVIRONMENT', 'development'),
                'python_version': os.getenv('PYTHON_VERSION', '3.11'),
                'build_time': os.getenv('BUILD_TIME', 'unknown')
            }
            self.application_info.info(app_info)
            
            logger.debug(
                "应用信息指标设置完成",
                app_info=app_info
            )
        except Exception as e:
            logger.error(
                "设置应用信息指标失败",
                error=str(e),
                service_name=self.service_name
            )
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """处理HTTP请求并收集监控指标
        
        在请求处理前后收集各种监控指标，包括请求计数、
        响应时间、状态码等信息。
        
        Args:
            request: HTTP请求对象
            call_next: 下一个中间件处理函数
            
        Returns:
            HTTP响应对象，包含原始响应和监控相关的响应头
        """
        # 记录请求开始时间
        start_time = time.time()
        
        # 提取请求信息
        method = request.method
        path = request.url.path
        
        # 简化路径，将动态参数替换为占位符以减少指标基数
        endpoint = self._normalize_endpoint(path)
        
        # 增加活跃连接计数
        self.active_connections.labels(service=self.service_name).inc()
        
        logger.debug(
            "开始处理HTTP请求",
            method=method,
            endpoint=endpoint,
            client_ip=request.client.host if request.client else "unknown",
            service=self.service_name
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            duration = time.time() - start_time
            
            # 获取响应状态码
            status_code = str(response.status_code)
            
            # 记录指标
            self._record_request_metrics(
                method=method,
                endpoint=endpoint,
                status=status_code,
                duration=duration
            )
            
            # 添加监控相关的响应头
            response.headers["X-Service-Name"] = self.service_name
            response.headers["X-Response-Time"] = f"{duration:.6f}s"
            
            logger.info(
                "HTTP请求处理完成",
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=f"{duration:.3f}s",
                service=self.service_name
            )
            
            return response
            
        except Exception as e:
            # 计算处理时间（即使发生异常）
            duration = time.time() - start_time
            
            # 记录错误指标
            self._record_request_metrics(
                method=method,
                endpoint=endpoint,
                status="500",  # 异常情况下默认为500
                duration=duration
            )
            
            logger.error(
                "HTTP请求处理异常",
                method=method,
                endpoint=endpoint,
                duration=f"{duration:.3f}s",
                error=str(e),
                error_type=type(e).__name__,
                service=self.service_name
            )
            
            # 重新抛出异常
            raise
            
        finally:
            # 减少活跃连接计数
            self.active_connections.labels(service=self.service_name).dec()
    
    def _normalize_endpoint(self, path: str) -> str:
        """标准化API端点路径
        
        将包含动态参数的路径标准化为固定模式，
        避免因为路径参数导致监控指标基数过大。
        
        Args:
            path: 原始请求路径
            
        Returns:
            标准化后的端点路径
            
        Example:
            /api/v1/documents/123 -> /api/v1/documents/{id}
            /users/456/profile -> /users/{user_id}/profile
        """
        # 常见的动态参数模式替换
        import re
        
        # UUID模式替换
        path = re.sub(
            r'/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
            '/{uuid}',
            path
        )
        
        # 数字ID模式替换
        path = re.sub(r'/\d+(?=/|$)', '/{id}', path)
        
        # 文件名模式替换（保留扩展名信息）
        path = re.sub(r'/[^/]+\.(pdf|docx?|txt|jpg|jpeg|png|gif)(?=/|$)', '/file.{ext}', path)
        
        return path
    
    def _record_request_metrics(
        self, 
        method: str, 
        endpoint: str, 
        status: str, 
        duration: float
    ):
        """记录HTTP请求监控指标
        
        Args:
            method: HTTP方法
            endpoint: 标准化后的端点路径
            status: HTTP状态码
            duration: 请求处理时间（秒）
        """
        try:
            # 记录请求计数
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status,
                service=self.service_name
            ).inc()
            
            # 记录响应时间分布
            self.http_request_duration.labels(
                method=method,
                endpoint=endpoint,
                status=status,
                service=self.service_name
            ).observe(duration)
            
            logger.debug(
                "HTTP请求指标记录完成",
                method=method,
                endpoint=endpoint,
                status=status,
                duration=duration,
                service=self.service_name
            )
            
        except Exception as e:
            logger.error(
                "记录HTTP请求指标失败",
                error=str(e),
                method=method,
                endpoint=endpoint,
                status=status,
                service=self.service_name
            )


class BusinessMetricsCollector:
    """业务指标收集器
    
    用于收集特定于历史文本处理业务的监控指标，
    包括文件处理、文本分析、OCR识别等业务操作的监控数据。
    
    业务指标包括：
    - file_uploads_total: 文件上传总数
    - text_processing_duration: 文本处理时间
    - ocr_operations_total: OCR操作总数
    - file_virus_scan_total: 文件病毒扫描总数
    - text_processing_queue_size: 文本处理队列大小
    """
    
    def __init__(self, service_name: str):
        """初始化业务指标收集器
        
        Args:
            service_name: 服务名称标识
        """
        self.service_name = service_name
        
        # 文件上传指标
        self.file_uploads_total = Counter(
            name='file_uploads_total',
            documentation='文件上传总数',
            labelnames=['status', 'file_type', 'service']
        )
        
        # 文本处理时间指标
        self.text_processing_duration = Histogram(
            name='text_processing_duration_seconds',
            documentation='文本处理持续时间（秒）',
            labelnames=['operation', 'status', 'service'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
        )
        
        # OCR操作指标
        self.ocr_operations_total = Counter(
            name='ocr_operations_total',
            documentation='OCR操作总数',
            labelnames=['status', 'language', 'service']
        )
        
        # 文件病毒扫描指标
        self.file_virus_scan_total = Counter(
            name='file_virus_scan_total',
            documentation='文件病毒扫描总数',
            labelnames=['result', 'service']
        )
        
        # 文本处理队列大小指标
        self.text_processing_queue_size = Gauge(
            name='text_processing_queue_size',
            documentation='文本处理队列当前大小',
            labelnames=['queue_type', 'service']
        )
        
        # 认证相关指标
        self.auth_login_attempts_total = Counter(
            name='auth_login_attempts_total',
            documentation='用户登录尝试总数',
            labelnames=['status', 'method', 'service']
        )
        
        logger.info(
            "业务指标收集器初始化完成",
            service_name=self.service_name
        )
    
    def record_file_upload(self, status: str, file_type: str):
        """记录文件上传指标
        
        Args:
            status: 上传状态 (success/error/rejected)
            file_type: 文件类型 (pdf/docx/jpg等)
        """
        try:
            self.file_uploads_total.labels(
                status=status,
                file_type=file_type,
                service=self.service_name
            ).inc()
            
            logger.debug(
                "文件上传指标记录完成",
                status=status,
                file_type=file_type,
                service=self.service_name
            )
        except Exception as e:
            logger.error("记录文件上传指标失败", error=str(e))
    
    def record_text_processing(self, operation: str, status: str, duration: float):
        """记录文本处理指标
        
        Args:
            operation: 操作类型 (extract/analyze/ocr)
            status: 处理状态 (success/error/timeout)
            duration: 处理时间（秒）
        """
        try:
            self.text_processing_duration.labels(
                operation=operation,
                status=status,
                service=self.service_name
            ).observe(duration)
            
            logger.debug(
                "文本处理指标记录完成",
                operation=operation,
                status=status,
                duration=duration,
                service=self.service_name
            )
        except Exception as e:
            logger.error("记录文本处理指标失败", error=str(e))
    
    def record_ocr_operation(self, status: str, language: str = "unknown"):
        """记录OCR操作指标
        
        Args:
            status: OCR状态 (success/error/low_confidence)
            language: 识别语言 (zh/en/auto)
        """
        try:
            self.ocr_operations_total.labels(
                status=status,
                language=language,
                service=self.service_name
            ).inc()
            
            logger.debug(
                "OCR操作指标记录完成",
                status=status,
                language=language,
                service=self.service_name
            )
        except Exception as e:
            logger.error("记录OCR操作指标失败", error=str(e))
    
    def record_virus_scan(self, result: str):
        """记录病毒扫描指标
        
        Args:
            result: 扫描结果 (clean/infected/error)
        """
        try:
            self.file_virus_scan_total.labels(
                result=result,
                service=self.service_name
            ).inc()
            
            logger.debug(
                "病毒扫描指标记录完成",
                result=result,
                service=self.service_name
            )
        except Exception as e:
            logger.error("记录病毒扫描指标失败", error=str(e))
    
    def update_queue_size(self, queue_type: str, size: int):
        """更新队列大小指标
        
        Args:
            queue_type: 队列类型 (ocr/text_analysis/file_processing)
            size: 当前队列大小
        """
        try:
            self.text_processing_queue_size.labels(
                queue_type=queue_type,
                service=self.service_name
            ).set(size)
            
            logger.debug(
                "队列大小指标更新完成",
                queue_type=queue_type,
                size=size,
                service=self.service_name
            )
        except Exception as e:
            logger.error("更新队列大小指标失败", error=str(e))
    
    def record_auth_attempt(self, status: str, method: str = "password"):
        """记录认证尝试指标
        
        Args:
            status: 认证状态 (success/failed/blocked)
            method: 认证方法 (password/oauth/jwt)
        """
        try:
            self.auth_login_attempts_total.labels(
                status=status,
                method=method,
                service=self.service_name
            ).inc()
            
            logger.debug(
                "认证尝试指标记录完成",
                status=status,
                method=method,
                service=self.service_name
            )
        except Exception as e:
            logger.error("记录认证尝试指标失败", error=str(e))


# 全局业务指标收集器实例
business_metrics: Optional[BusinessMetricsCollector] = None

def get_business_metrics(service_name: str = "unknown-service") -> BusinessMetricsCollector:
    """获取业务指标收集器实例
    
    使用单例模式确保整个应用只有一个指标收集器实例。
    
    Args:
        service_name: 服务名称
        
    Returns:
        业务指标收集器实例
    """
    global business_metrics
    if business_metrics is None:
        business_metrics = BusinessMetricsCollector(service_name)
    return business_metrics

def generate_metrics() -> str:
    """生成Prometheus格式的监控指标
    
    Returns:
        Prometheus格式的指标数据字符串
    """
    try:
        metrics_data = generate_latest()
        logger.debug("监控指标生成完成", metrics_size=len(metrics_data))
        return metrics_data
    except Exception as e:
        logger.error("生成监控指标失败", error=str(e))
        return ""