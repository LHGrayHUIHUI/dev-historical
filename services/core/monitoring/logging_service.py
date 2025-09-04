"""
日志管理服务模块

此模块提供统一的日志管理功能，包括结构化日志配置、
日志收集、ElasticSearch集成、日志查询和分析等功能。

主要功能：
- 结构化日志配置
- 日志格式标准化
- ElasticSearch日志存储
- 日志查询和过滤
- 日志聚合分析
- 日志轮转和清理

Author: 开发团队
Created: 2025-09-04
Version: 1.0.0
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level, StackInfoRenderer
import logging
import logging.handlers
from pathlib import Path
import aiohttp
from elasticsearch import AsyncElasticsearch
import uuid

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogConfig:
    """日志配置类
    
    定义日志的输出格式、存储位置、轮转策略等配置信息。
    
    Attributes:
        service_name: 服务名称
        log_level: 日志级别
        log_format: 日志格式（json/text）
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        log_dir: 日志文件目录
        max_file_size: 单个日志文件最大大小（MB）
        backup_count: 保留的历史日志文件数量
        elasticsearch_enabled: 是否启用ElasticSearch
        elasticsearch_url: ElasticSearch服务地址
        elasticsearch_index: ElasticSearch索引前缀
    """
    service_name: str = "unknown-service"
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"  # json 或 text
    console_output: bool = True
    file_output: bool = True
    log_dir: str = "logs"
    max_file_size: int = 100  # MB
    backup_count: int = 10
    elasticsearch_enabled: bool = False
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "application-logs"
    include_trace_info: bool = True
    
    def __post_init__(self):
        """后初始化处理"""
        # 确保日志目录存在
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class LogEntry:
    """日志条目类
    
    表示一条标准化的日志记录。
    
    Attributes:
        timestamp: 时间戳
        level: 日志级别
        message: 日志消息
        service_name: 服务名称
        logger_name: 记录器名称
        module: 模块名称
        function: 函数名称
        line_number: 行号
        trace_id: 追踪ID
        span_id: SpanID
        user_id: 用户ID
        request_id: 请求ID
        extra: 额外字段
    """
    timestamp: datetime
    level: str
    message: str
    service_name: str
    logger_name: str = ""
    module: str = ""
    function: str = ""
    line_number: Optional[int] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "service_name": self.service_name,
            "logger_name": self.logger_name,
            "module": self.module,
            "function": self.function,
            "line_number": self.line_number,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            **self.extra
        }

class ElasticsearchHandler(logging.Handler):
    """ElasticSearch日志处理器
    
    将日志记录异步写入ElasticSearch。
    """
    
    def __init__(self, 
                 elasticsearch_url: str,
                 index_prefix: str,
                 service_name: str,
                 batch_size: int = 100,
                 flush_interval: int = 5):
        """初始化ElasticSearch处理器
        
        Args:
            elasticsearch_url: ElasticSearch服务地址
            index_prefix: 索引前缀
            service_name: 服务名称
            batch_size: 批量写入大小
            flush_interval: 刷新间隔（秒）
        """
        super().__init__()
        self.elasticsearch_url = elasticsearch_url
        self.index_prefix = index_prefix
        self.service_name = service_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # 日志缓冲区
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()
        
        # ElasticSearch客户端
        self._es_client = None
        self._flush_task = None
        self._running = False
        
        # 启动刷新任务
        asyncio.create_task(self._start_flush_task())
    
    async def _start_flush_task(self):
        """启动刷新任务"""
        try:
            self._es_client = AsyncElasticsearch([self.elasticsearch_url])
            self._running = True
            self._flush_task = asyncio.create_task(self._flush_loop())
        except Exception as e:
            structlog.get_logger().error(
                "ElasticSearch连接失败",
                error=str(e),
                url=self.elasticsearch_url
            )
    
    async def _flush_loop(self):
        """刷新循环"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                structlog.get_logger().error(
                    "日志刷新异常",
                    error=str(e)
                )
    
    def emit(self, record: logging.LogRecord):
        """发送日志记录
        
        Args:
            record: 日志记录对象
        """
        try:
            # 格式化日志记录
            log_entry = self._format_record(record)
            
            # 添加到缓冲区
            asyncio.create_task(self._add_to_buffer(log_entry))
            
        except Exception:
            self.handleError(record)
    
    def _format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """格式化日志记录
        
        Args:
            record: 日志记录对象
            
        Returns:
            Dict[str, Any]: 格式化后的日志字典
        """
        # 基础字段
        log_entry = {
            "@timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": self.service_name,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.format(record)
        
        # 添加额外字段
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        return log_entry
    
    async def _add_to_buffer(self, log_entry: Dict[str, Any]):
        """添加到缓冲区
        
        Args:
            log_entry: 日志条目
        """
        async with self._buffer_lock:
            self._buffer.append(log_entry)
            
            # 如果缓冲区满了，立即刷新
            if len(self._buffer) >= self.batch_size:
                await self._flush_buffer()
    
    async def _flush_buffer(self):
        """刷新缓冲区到ElasticSearch"""
        if not self._es_client or not self._buffer:
            return
        
        async with self._buffer_lock:
            if not self._buffer:
                return
            
            # 构建批量操作
            actions = []
            current_date = datetime.utcnow().strftime('%Y-%m-%d')
            index_name = f"{self.index_prefix}-{current_date}"
            
            for log_entry in self._buffer:
                action = {
                    "_index": index_name,
                    "_source": log_entry
                }
                actions.append(action)
            
            try:
                # 批量写入
                if actions:
                    response = await self._es_client.bulk(
                        body={"index": {}},
                        index=index_name,
                        body=actions
                    )
                    
                    if response.get('errors'):
                        structlog.get_logger().warning(
                            "部分日志写入ES失败",
                            errors=response.get('errors')
                        )
                
                # 清空缓冲区
                self._buffer.clear()
                
            except Exception as e:
                structlog.get_logger().error(
                    "日志批量写入ES失败",
                    error=str(e),
                    buffer_size=len(self._buffer)
                )
    
    async def close(self):
        """关闭处理器"""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # 最后刷新一次
        await self._flush_buffer()
        
        # 关闭ES客户端
        if self._es_client:
            await self._es_client.close()

class LoggingService:
    """日志管理服务类
    
    提供统一的日志配置、管理和查询功能。
    
    Attributes:
        config: 日志配置
        logger: 结构化日志记录器
        es_handler: ElasticSearch处理器
    """
    
    def __init__(self, config: LogConfig):
        """初始化日志服务
        
        Args:
            config: 日志配置
        """
        self.config = config
        self.es_handler: Optional[ElasticsearchHandler] = None
        
        # 配置日志系统
        self._configure_logging()
        
        # 获取日志记录器
        self.logger = structlog.get_logger()
        
        structlog.get_logger().info(
            "日志管理服务初始化完成",
            service_name=config.service_name,
            log_level=config.log_level.value,
            elasticsearch_enabled=config.elasticsearch_enabled
        )
    
    def _configure_logging(self):
        """配置日志系统"""
        # 配置structlog处理器
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TimeStamper(fmt="iso"),
            StackInfoRenderer() if self.config.include_trace_info else lambda _, __, ___: {},
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # 根据格式选择渲染器
        if self.config.log_format == "json":
            processors.append(JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        
        # 配置structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # 配置标准库logging
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 控制台处理器
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.log_level.value))
            
            if self.config.log_format == "json":
                formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                )
            else:
                formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 文件处理器
        if self.config.file_output:
            log_file = os.path.join(self.config.log_dir, f"{self.config.service_name}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size * 1024 * 1024,  # 转换为字节
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, self.config.log_level.value))
            
            if self.config.log_format == "json":
                formatter = logging.Formatter('%(message)s')
            else:
                formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # ElasticSearch处理器
        if self.config.elasticsearch_enabled:
            self.es_handler = ElasticsearchHandler(
                elasticsearch_url=self.config.elasticsearch_url,
                index_prefix=self.config.elasticsearch_index,
                service_name=self.config.service_name
            )
            self.es_handler.setLevel(getattr(logging, self.config.log_level.value))
            root_logger.addHandler(self.es_handler)
    
    def bind_context(self, **kwargs) -> structlog.BoundLogger:
        """绑定日志上下文
        
        Args:
            **kwargs: 上下文键值对
            
        Returns:
            structlog.BoundLogger: 绑定上下文的日志记录器
        """
        return self.logger.bind(**kwargs)
    
    def bind_request_context(self, 
                           request_id: Optional[str] = None,
                           trace_id: Optional[str] = None,
                           span_id: Optional[str] = None,
                           user_id: Optional[str] = None) -> structlog.BoundLogger:
        """绑定请求上下文
        
        Args:
            request_id: 请求ID
            trace_id: 追踪ID
            span_id: SpanID
            user_id: 用户ID
            
        Returns:
            structlog.BoundLogger: 绑定请求上下文的日志记录器
        """
        context = {}
        
        if request_id:
            context['request_id'] = request_id
        if trace_id:
            context['trace_id'] = trace_id
        if span_id:
            context['span_id'] = span_id
        if user_id:
            context['user_id'] = user_id
        
        return self.logger.bind(**context)
    
    async def search_logs(self,
                         query: str = "*",
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         level: Optional[LogLevel] = None,
                         size: int = 100,
                         sort_order: str = "desc") -> List[Dict[str, Any]]:
        """搜索日志
        
        Args:
            query: 查询语句
            start_time: 开始时间
            end_time: 结束时间
            level: 日志级别
            size: 返回数量
            sort_order: 排序顺序（asc/desc）
            
        Returns:
            List[Dict[str, Any]]: 日志列表
        """
        if not self.config.elasticsearch_enabled or not self.es_handler:
            raise ValueError("ElasticSearch未启用")
        
        try:
            es_client = AsyncElasticsearch([self.config.elasticsearch_url])
            
            # 构建查询体
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"query_string": {"query": query}}
                        ],
                        "filter": []
                    }
                },
                "sort": [
                    {"@timestamp": {"order": sort_order}}
                ],
                "size": size
            }
            
            # 添加时间范围过滤
            if start_time or end_time:
                time_range = {}
                if start_time:
                    time_range["gte"] = start_time.isoformat()
                if end_time:
                    time_range["lte"] = end_time.isoformat()
                
                search_body["query"]["bool"]["filter"].append({
                    "range": {"@timestamp": time_range}
                })
            
            # 添加日志级别过滤
            if level:
                search_body["query"]["bool"]["filter"].append({
                    "term": {"level": level.value}
                })
            
            # 执行搜索
            index_pattern = f"{self.config.elasticsearch_index}-*"
            response = await es_client.search(
                index=index_pattern,
                body=search_body
            )
            
            # 提取结果
            logs = []
            for hit in response['hits']['hits']:
                log_entry = hit['_source']
                log_entry['_id'] = hit['_id']
                log_entry['_score'] = hit['_score']
                logs.append(log_entry)
            
            await es_client.close()
            return logs
            
        except Exception as e:
            self.logger.error(
                "日志搜索失败",
                error=str(e),
                error_type=type(e).__name__
            )
            return []
    
    async def get_log_statistics(self,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取日志统计信息
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.config.elasticsearch_enabled:
            return {}
        
        try:
            es_client = AsyncElasticsearch([self.config.elasticsearch_url])
            
            # 构建聚合查询
            agg_body = {
                "size": 0,
                "aggs": {
                    "levels": {
                        "terms": {"field": "level"}
                    },
                    "services": {
                        "terms": {"field": "service"}
                    },
                    "timeline": {
                        "date_histogram": {
                            "field": "@timestamp",
                            "interval": "1h"
                        }
                    }
                }
            }
            
            # 添加时间范围过滤
            if start_time or end_time:
                time_range = {}
                if start_time:
                    time_range["gte"] = start_time.isoformat()
                if end_time:
                    time_range["lte"] = end_time.isoformat()
                
                agg_body["query"] = {
                    "range": {"@timestamp": time_range}
                }
            
            # 执行聚合查询
            index_pattern = f"{self.config.elasticsearch_index}-*"
            response = await es_client.search(
                index=index_pattern,
                body=agg_body
            )
            
            # 提取统计结果
            stats = {
                "total_logs": response['hits']['total']['value'],
                "levels": {},
                "services": {},
                "timeline": []
            }
            
            # 日志级别统计
            for bucket in response['aggregations']['levels']['buckets']:
                stats['levels'][bucket['key']] = bucket['doc_count']
            
            # 服务统计
            for bucket in response['aggregations']['services']['buckets']:
                stats['services'][bucket['key']] = bucket['doc_count']
            
            # 时间线统计
            for bucket in response['aggregations']['timeline']['buckets']:
                stats['timeline'].append({
                    'timestamp': bucket['key_as_string'],
                    'count': bucket['doc_count']
                })
            
            await es_client.close()
            return stats
            
        except Exception as e:
            self.logger.error(
                "获取日志统计失败",
                error=str(e)
            )
            return {}
    
    async def cleanup_old_logs(self, retention_days: int = 30):
        """清理过期日志
        
        Args:
            retention_days: 日志保留天数
        """
        if not self.config.elasticsearch_enabled:
            return
        
        try:
            es_client = AsyncElasticsearch([self.config.elasticsearch_url])
            
            # 计算过期时间
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
            
            # 获取所有索引
            indices_response = await es_client.indices.get(
                index=f"{self.config.elasticsearch_index}-*"
            )
            
            # 删除过期索引
            deleted_indices = []
            for index_name in indices_response.keys():
                # 提取日期部分
                date_part = index_name.split('-')[-1]
                try:
                    index_date = datetime.strptime(date_part, '%Y-%m-%d')
                    if index_date < cutoff_date:
                        await es_client.indices.delete(index=index_name)
                        deleted_indices.append(index_name)
                except ValueError:
                    # 日期格式不匹配，跳过
                    continue
            
            await es_client.close()
            
            self.logger.info(
                "日志清理完成",
                retention_days=retention_days,
                deleted_indices=deleted_indices
            )
            
        except Exception as e:
            self.logger.error(
                "日志清理失败",
                error=str(e)
            )
    
    async def close(self):
        """关闭日志服务"""
        if self.es_handler:
            await self.es_handler.close()
        
        self.logger.info("日志管理服务已关闭")

# 全局日志服务实例
_logging_service: Optional[LoggingService] = None

def get_logging_service(config: Optional[LogConfig] = None) -> LoggingService:
    """获取全局日志服务实例
    
    Args:
        config: 日志配置（首次调用时需要）
        
    Returns:
        LoggingService: 日志服务实例
    """
    global _logging_service
    if _logging_service is None:
        if config is None:
            config = LogConfig()
        _logging_service = LoggingService(config)
    return _logging_service

def create_default_log_config(
    service_name: str,
    log_level: LogLevel = LogLevel.INFO,
    elasticsearch_enabled: bool = False
) -> LogConfig:
    """创建默认日志配置
    
    Args:
        service_name: 服务名称
        log_level: 日志级别
        elasticsearch_enabled: 是否启用ElasticSearch
        
    Returns:
        LogConfig: 日志配置对象
    """
    return LogConfig(
        service_name=service_name,
        log_level=log_level,
        elasticsearch_enabled=elasticsearch_enabled,
        elasticsearch_url=os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200'),
        elasticsearch_index=f"{service_name}-logs"
    )