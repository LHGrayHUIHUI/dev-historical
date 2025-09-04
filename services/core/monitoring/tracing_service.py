"""
链路追踪服务模块

此模块提供分布式链路追踪功能，支持Jaeger和OpenTelemetry，
用于跟踪请求在微服务之间的调用链路，帮助定位性能瓶颈和故障。

主要功能：
- 链路追踪初始化和配置
- Span创建和管理
- 链路上下文传播
- Jaeger后端集成
- 性能指标收集

Author: 开发团队
Created: 2025-09-04
Version: 1.0.0
"""

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry import propagate
import structlog
import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time

# 获取结构化日志记录器
logger = structlog.get_logger()

class TracingService:
    """链路追踪服务类
    
    负责初始化和配置OpenTelemetry链路追踪，
    提供统一的链路追踪接口和管理功能。
    
    Attributes:
        service_name: 服务名称，用于标识追踪来源
        jaeger_endpoint: Jaeger收集器端点
        tracer: OpenTelemetry追踪器实例
        is_enabled: 是否启用链路追踪
    """
    
    def __init__(
        self,
        service_name: str,
        jaeger_endpoint: Optional[str] = None,
        sampling_rate: float = 0.1,
        enable_db_instrumentation: bool = True
    ):
        """初始化链路追踪服务
        
        Args:
            service_name: 服务名称
            jaeger_endpoint: Jaeger收集器端点URL
            sampling_rate: 采样率（0.0-1.0）
            enable_db_instrumentation: 是否启用数据库自动插桩
        """
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint or os.getenv(
            'JAEGER_ENDPOINT', 'http://localhost:14268/api/traces'
        )
        self.sampling_rate = sampling_rate
        self.enable_db_instrumentation = enable_db_instrumentation
        self.is_enabled = False
        self.tracer = None
        
        # 初始化追踪服务
        self._initialize_tracing()
        
        logger.info(
            "链路追踪服务初始化完成",
            service_name=self.service_name,
            jaeger_endpoint=self.jaeger_endpoint,
            sampling_rate=self.sampling_rate,
            is_enabled=self.is_enabled
        )
    
    def _initialize_tracing(self):
        """初始化OpenTelemetry链路追踪配置"""
        try:
            # 创建资源标识
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                "service.version": os.getenv('SERVICE_VERSION', '1.0.0'),
                "service.environment": os.getenv('ENVIRONMENT', 'development'),
                "service.instance.id": os.getenv('HOSTNAME', 'unknown')
            })
            
            # 配置追踪提供者
            trace.set_tracer_provider(TracerProvider(resource=resource))
            tracer_provider = trace.get_tracer_provider()
            
            # 配置Jaeger导出器
            jaeger_exporter = JaegerExporter(
                endpoint=self.jaeger_endpoint,
                timeout=30
            )
            
            # 添加批量Span处理器
            span_processor = BatchSpanProcessor(
                jaeger_exporter,
                max_queue_size=2048,
                schedule_delay_millis=5000,
                max_export_batch_size=512
            )
            tracer_provider.add_span_processor(span_processor)
            
            # 配置传播器
            propagate.set_global_textmap(B3MultiFormat())
            
            # 获取追踪器
            self.tracer = trace.get_tracer(__name__, version="1.0.0")
            
            # 自动插桩HTTP请求
            RequestsInstrumentor().instrument()
            
            # 数据库插桩（可选）
            if self.enable_db_instrumentation:
                try:
                    PymongoInstrumentor().instrument()
                    RedisInstrumentor().instrument()
                    logger.debug("数据库自动插桩已启用")
                except Exception as e:
                    logger.warning("数据库插桩失败", error=str(e))
            
            self.is_enabled = True
            logger.info("链路追踪初始化成功")
            
        except Exception as e:
            logger.error(
                "链路追踪初始化失败",
                error=str(e),
                error_type=type(e).__name__
            )
            self.is_enabled = False
    
    def instrument_fastapi(self, app):
        """为FastAPI应用添加自动链路追踪
        
        Args:
            app: FastAPI应用实例
        """
        if not self.is_enabled:
            logger.warning("链路追踪未启用，跳过FastAPI插桩")
            return
            
        try:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI自动链路追踪已启用")
        except Exception as e:
            logger.error("FastAPI插桩失败", error=str(e))
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ):
        """创建操作追踪上下文管理器
        
        Args:
            operation_name: 操作名称
            attributes: 追踪属性字典
            record_exception: 是否记录异常
            
        Yields:
            Span: OpenTelemetry Span对象
        """
        if not self.is_enabled or not self.tracer:
            # 如果追踪未启用，返回空上下文
            yield None
            return
            
        with self.tracer.start_as_current_span(operation_name) as span:
            try:
                # 添加基础属性
                span.set_attribute("service.name", self.service_name)
                span.set_attribute("operation.start_time", time.time())
                
                # 添加自定义属性
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                logger.debug(
                    "开始追踪操作",
                    operation_name=operation_name,
                    trace_id=format(span.get_span_context().trace_id, '032x'),
                    span_id=format(span.get_span_context().span_id, '016x')
                )
                
                yield span
                
            except Exception as e:
                if record_exception:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                logger.error(
                    "追踪操作异常",
                    operation_name=operation_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
            
            finally:
                span.set_attribute("operation.end_time", time.time())
    
    def create_child_span(
        self,
        operation_name: str,
        parent_context=None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """创建子Span
        
        Args:
            operation_name: 操作名称
            parent_context: 父上下文
            attributes: 追踪属性
            
        Returns:
            Span: 子Span对象
        """
        if not self.is_enabled or not self.tracer:
            return None
            
        span = self.tracer.start_span(
            operation_name,
            context=parent_context
        )
        
        # 添加属性
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        return span
    
    def add_event(self, span, name: str, attributes: Optional[Dict[str, Any]] = None):
        """向Span添加事件
        
        Args:
            span: Span对象
            name: 事件名称
            attributes: 事件属性
        """
        if span and self.is_enabled:
            span.add_event(name, attributes or {})
    
    def set_span_attributes(self, span, attributes: Dict[str, Any]):
        """设置Span属性
        
        Args:
            span: Span对象
            attributes: 属性字典
        """
        if span and self.is_enabled:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
    
    def get_current_trace_id(self) -> Optional[str]:
        """获取当前追踪ID
        
        Returns:
            str: 当前追踪ID的十六进制字符串表示
        """
        if not self.is_enabled:
            return None
            
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.get_span_context().is_valid:
                return format(current_span.get_span_context().trace_id, '032x')
        except Exception as e:
            logger.error("获取追踪ID失败", error=str(e))
        
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """获取当前SpanID
        
        Returns:
            str: 当前SpanID的十六进制字符串表示
        """
        if not self.is_enabled:
            return None
            
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.get_span_context().is_valid:
                return format(current_span.get_span_context().span_id, '016x')
        except Exception as e:
            logger.error("获取SpanID失败", error=str(e))
        
        return None
    
    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """将追踪上下文注入到HTTP头中
        
        Args:
            headers: HTTP头字典
            
        Returns:
            Dict[str, str]: 包含追踪上下文的HTTP头
        """
        if not self.is_enabled:
            return headers
            
        try:
            # 注入追踪上下文
            propagate.inject(headers)
            logger.debug("追踪上下文注入完成", headers_count=len(headers))
        except Exception as e:
            logger.error("追踪上下文注入失败", error=str(e))
        
        return headers
    
    def extract_context(self, headers: Dict[str, str]):
        """从HTTP头中提取追踪上下文
        
        Args:
            headers: HTTP头字典
            
        Returns:
            追踪上下文对象
        """
        if not self.is_enabled:
            return None
            
        try:
            context = propagate.extract(headers)
            logger.debug("追踪上下文提取完成")
            return context
        except Exception as e:
            logger.error("追踪上下文提取失败", error=str(e))
            return None
    
    def shutdown(self):
        """关闭链路追踪服务，清理资源"""
        try:
            if self.is_enabled:
                # 关闭追踪提供者
                tracer_provider = trace.get_tracer_provider()
                if hasattr(tracer_provider, 'shutdown'):
                    tracer_provider.shutdown()
                
                self.is_enabled = False
                logger.info("链路追踪服务已关闭")
        except Exception as e:
            logger.error("关闭链路追踪服务失败", error=str(e))


# 全局追踪服务实例
_tracing_service: Optional[TracingService] = None

def get_tracing_service(
    service_name: str = "unknown-service",
    jaeger_endpoint: Optional[str] = None,
    sampling_rate: float = 0.1
) -> TracingService:
    """获取全局链路追踪服务实例
    
    Args:
        service_name: 服务名称
        jaeger_endpoint: Jaeger端点
        sampling_rate: 采样率
        
    Returns:
        TracingService: 链路追踪服务实例
    """
    global _tracing_service
    if _tracing_service is None:
        _tracing_service = TracingService(
            service_name=service_name,
            jaeger_endpoint=jaeger_endpoint,
            sampling_rate=sampling_rate
        )
    return _tracing_service

def initialize_tracing_for_service(
    service_name: str,
    app=None,
    jaeger_endpoint: Optional[str] = None,
    sampling_rate: float = 0.1
) -> TracingService:
    """为服务初始化链路追踪
    
    Args:
        service_name: 服务名称
        app: FastAPI应用实例（可选）
        jaeger_endpoint: Jaeger端点
        sampling_rate: 采样率
        
    Returns:
        TracingService: 配置好的链路追踪服务实例
    """
    tracing_service = get_tracing_service(
        service_name=service_name,
        jaeger_endpoint=jaeger_endpoint,
        sampling_rate=sampling_rate
    )
    
    # 如果提供了FastAPI应用，则自动插桩
    if app:
        tracing_service.instrument_fastapi(app)
    
    return tracing_service