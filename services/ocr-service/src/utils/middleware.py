"""
FastAPI中间件

提供请求处理、日志记录、性能监控等中间件功能。

主要功能：
- 请求ID生成和追踪
- 请求/响应日志记录  
- 请求处理时间统计
- 错误处理和监控
- CORS和安全头处理

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import time
import uuid
import logging
from typing import Callable, Any
import threading

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp


# 线程本地存储，用于跨中间件共享请求上下文
request_local = threading.local()

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    请求ID中间件
    
    为每个请求生成唯一的ID，用于日志追踪和问题定位。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求，生成和传递请求ID
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理器
            
        Returns:
            响应对象
        """
        # 从请求头获取或生成请求ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # 将请求ID存储到请求状态和线程本地存储
        request.state.request_id = request_id
        request_local.request_id = request_id
        
        # 处理请求
        response = await call_next(request)
        
        # 在响应头中返回请求ID
        response.headers["X-Request-ID"] = request_id
        
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    请求处理时间中间件
    
    记录每个请求的处理时间，用于性能监控。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求，记录处理时间
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理器
            
        Returns:
            响应对象
        """
        start_time = time.time()
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 在响应头中添加处理时间
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录性能日志
        if hasattr(request.state, 'request_id'):
            logger.info(
                f"请求处理完成",
                extra={
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "path": str(request.url.path),
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    
    记录所有HTTP请求的详细信息。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求，记录请求日志
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理器
            
        Returns:
            响应对象
        """
        # 获取客户端信息
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        
        # 记录请求开始
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.info(
            f"请求开始",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "query_params": str(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent
            }
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 记录请求成功
            logger.info(
                f"请求成功",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers)
                }
            )
            
            return response
            
        except Exception as e:
            # 记录请求异常
            logger.error(
                f"请求异常: {str(e)}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "error_type": type(e).__name__
                }
            )
            
            # 返回统一的错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_code": "INTERNAL_ERROR", 
                    "message": "内部服务器错误",
                    "request_id": request_id
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """
        获取真实的客户端IP地址
        
        Args:
            request: FastAPI请求对象
            
        Returns:
            客户端IP地址
        """
        # 检查反向代理头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # 回退到直接连接IP
        if request.client:
            return request.client.host
        
        return "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全头中间件
    
    为响应添加标准的安全HTTP头。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求，添加安全头
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理器
            
        Returns:
            响应对象
        """
        response = await call_next(request)
        
        # 添加安全头
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    简单的内存限流中间件
    
    基于客户端IP的请求频率限制。
    """
    
    def __init__(self, app: ASGIApp, max_requests: int = 100, window_seconds: int = 60):
        """
        初始化限流中间件
        
        Args:
            app: ASGI应用
            max_requests: 时间窗口内最大请求数
            window_seconds: 时间窗口长度（秒）
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {ip: [timestamp1, timestamp2, ...]}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求，执行限流检查
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理器
            
        Returns:
            响应对象或限流错误响应
        """
        # 获取客户端IP
        client_ip = self._get_client_ip(request)
        
        # 检查限流
        if self._is_rate_limited(client_ip):
            logger.warning(
                f"请求被限流",
                extra={
                    "client_ip": client_ip,
                    "path": str(request.url.path)
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error_code": "RATE_LIMITED",
                    "message": "请求过于频繁，请稍后再试"
                },
                headers={
                    "Retry-After": str(self.window_seconds)
                }
            )
        
        # 记录请求
        self._record_request(client_ip)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """
        检查是否超过限流阈值
        
        Args:
            client_ip: 客户端IP
            
        Returns:
            是否被限流
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        if client_ip not in self.requests:
            return False
        
        # 清理过期的请求记录
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > window_start
        ]
        
        # 检查是否超过限制
        return len(self.requests[client_ip]) >= self.max_requests
    
    def _record_request(self, client_ip: str):
        """
        记录请求时间戳
        
        Args:
            client_ip: 客户端IP
        """
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        self.requests[client_ip].append(now)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    指标收集中间件
    
    收集HTTP请求的各种指标用于监控。
    """
    
    def __init__(self, app: ASGIApp):
        """
        初始化指标中间件
        
        Args:
            app: ASGI应用
        """
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "total_processing_time": 0.0,
            "status_codes": {},
            "endpoints": {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        处理请求，收集指标
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件或路由处理器
            
        Returns:
            响应对象
        """
        start_time = time.time()
        
        # 更新请求计数
        self.metrics["total_requests"] += 1
        
        try:
            response = await call_next(request)
            
            # 记录处理时间
            process_time = time.time() - start_time
            self.metrics["total_processing_time"] += process_time
            
            # 记录状态码
            status_code = response.status_code
            self.metrics["status_codes"][status_code] = \
                self.metrics["status_codes"].get(status_code, 0) + 1
            
            # 记录端点访问
            endpoint = f"{request.method} {request.url.path}"
            if endpoint not in self.metrics["endpoints"]:
                self.metrics["endpoints"][endpoint] = {
                    "count": 0,
                    "total_time": 0.0,
                    "errors": 0
                }
            
            endpoint_metrics = self.metrics["endpoints"][endpoint]
            endpoint_metrics["count"] += 1
            endpoint_metrics["total_time"] += process_time
            
            return response
            
        except Exception as e:
            # 记录错误
            self.metrics["total_errors"] += 1
            
            endpoint = f"{request.method} {request.url.path}"
            if endpoint in self.metrics["endpoints"]:
                self.metrics["endpoints"][endpoint]["errors"] += 1
            
            raise
    
    def get_metrics(self) -> dict:
        """
        获取收集的指标
        
        Returns:
            指标字典
        """
        # 计算平均响应时间
        avg_response_time = (
            self.metrics["total_processing_time"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )
        
        # 计算错误率
        error_rate = (
            self.metrics["total_errors"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "avg_response_time": avg_response_time,
            "error_rate": error_rate
        }


# 全局指标中间件实例
metrics_middleware_instance = None


def get_metrics() -> dict:
    """
    获取全局指标
    
    Returns:
        指标字典
    """
    global metrics_middleware_instance
    if metrics_middleware_instance:
        return metrics_middleware_instance.get_metrics()
    return {}


def init_metrics_middleware(app: Any) -> MetricsMiddleware:
    """
    初始化全局指标中间件
    
    Args:
        app: FastAPI应用实例
        
    Returns:
        指标中间件实例
    """
    global metrics_middleware_instance
    metrics_middleware_instance = MetricsMiddleware(app)
    return metrics_middleware_instance