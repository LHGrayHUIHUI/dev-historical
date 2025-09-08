"""
日志配置工具

提供统一的日志配置和管理功能，支持结构化日志、
日志轮转、不同环境的日志级别等。

主要功能：
- 统一日志格式
- 结构化JSON日志
- 日志文件轮转
- 不同环境的日志配置
- 请求ID追踪

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import logging
import logging.handlers
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path
import datetime


class JSONFormatter(logging.Formatter):
    """
    JSON格式日志处理器
    
    将日志记录格式化为结构化的JSON格式，
    便于日志聚合和分析工具处理。
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为JSON
        
        Args:
            record: 日志记录对象
            
        Returns:
            JSON格式的日志字符串
        """
        # 基础日志字段
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry, ensure_ascii=False)


class RequestIDFilter(logging.Filter):
    """
    请求ID过滤器
    
    为日志记录添加请求ID，用于追踪单个请求
    在整个处理流程中的日志。
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        为日志记录添加请求ID
        
        Args:
            record: 日志记录对象
            
        Returns:
            是否记录此日志
        """
        # 尝试从上下文获取请求ID
        request_id = getattr(record, 'request_id', None)
        if not request_id:
            # 如果没有请求ID，可以从当前上下文或线程本地存储获取
            import threading
            current_thread = threading.current_thread()
            request_id = getattr(current_thread, 'request_id', 'no-request-id')
        
        record.request_id = request_id
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False,
    include_request_id: bool = True
) -> None:
    """
    设置应用日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        max_size: 日志文件最大大小
        backup_count: 日志文件备份数量
        json_format: 是否使用JSON格式
        include_request_id: 是否包含请求ID
    """
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 设置日志格式
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # 添加请求ID过滤器
    if include_request_id:
        console_handler.addFilter(RequestIDFilter())
    
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        
        if include_request_id:
            file_handler.addFilter(RequestIDFilter())
        
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    logging.info(f"日志系统初始化完成 - 级别: {level}")


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    增强的日志适配器
    
    提供额外的上下文信息和便捷方法。
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        """
        初始化日志适配器
        
        Args:
            logger: 基础日志记录器
            extra: 额外的上下文信息
        """
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        处理日志消息和参数
        
        Args:
            msg: 日志消息
            kwargs: 关键字参数
            
        Returns:
            处理后的消息和参数
        """
        # 合并额外信息
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra.copy()
        
        return msg, kwargs
    
    def log_with_context(
        self, 
        level: int, 
        msg: str, 
        context: Dict[str, Any],
        *args, 
        **kwargs
    ):
        """
        记录带上下文的日志
        
        Args:
            level: 日志级别
            msg: 日志消息
            context: 上下文信息
            *args: 位置参数
            **kwargs: 关键字参数
        """
        extra_data = self.extra.copy()
        extra_data.update(context)
        
        kwargs['extra'] = kwargs.get('extra', {})
        kwargs['extra']['extra_data'] = extra_data
        
        self.log(level, msg, *args, **kwargs)


def create_service_logger(service_name: str, **context) -> LoggerAdapter:
    """
    创建服务专用日志记录器
    
    Args:
        service_name: 服务名称
        **context: 额外上下文信息
        
    Returns:
        增强的日志适配器
    """
    logger = get_logger(service_name)
    
    extra_context = {
        "service": service_name,
        **context
    }
    
    return LoggerAdapter(logger, extra_context)


# 便捷的日志记录函数

def log_function_call(func_name: str, args: tuple, kwargs: Dict[str, Any]):
    """
    记录函数调用日志
    
    Args:
        func_name: 函数名称
        args: 位置参数
        kwargs: 关键字参数
    """
    logger = get_logger("function_calls")
    logger.debug(f"调用函数 {func_name}，参数: args={args}, kwargs={kwargs}")


def log_performance(operation: str, duration: float, **metrics):
    """
    记录性能指标日志
    
    Args:
        operation: 操作名称
        duration: 持续时间（秒）
        **metrics: 其他性能指标
    """
    logger = get_logger("performance")
    
    context = {
        "operation": operation,
        "duration_seconds": duration,
        **metrics
    }
    
    logger.info(f"性能指标 - {operation}: {duration:.3f}秒", extra={"extra_data": context})


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    logger_name: str = "errors"
):
    """
    记录带上下文的错误日志
    
    Args:
        error: 异常对象
        context: 错误上下文
        logger_name: 日志记录器名称
    """
    logger = get_logger(logger_name)
    
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **context
    }
    
    logger.error(
        f"发生错误: {str(error)}", 
        exc_info=True,
        extra={"extra_data": error_context}
    )


# 装饰器

def log_exceptions(logger_name: str = "exceptions"):
    """
    异常日志装饰器
    
    自动记录函数异常日志。
    
    Args:
        logger_name: 日志记录器名称
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error_with_context(
                    e,
                    {
                        "function": func.__name__,
                        "args": str(args)[:200],  # 限制参数日志长度
                        "kwargs": str(kwargs)[:200]
                    },
                    logger_name
                )
                raise
        return wrapper
    return decorator


def log_timing(operation_name: str = None):
    """
    函数执行时间日志装饰器
    
    Args:
        operation_name: 操作名称
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                log_performance(op_name, duration)
                
                return result
            except Exception:
                duration = time.time() - start_time
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                log_performance(f"{op_name}_failed", duration)
                raise
        
        return wrapper
    return decorator