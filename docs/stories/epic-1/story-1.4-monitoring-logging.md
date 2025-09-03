# Story 1.4: 系统监控与日志管理

## 基本信息
- **Story ID**: 1.4
- **Epic**: Epic 1 - 微服务基础架构和数据采集
- **标题**: 系统监控与日志管理
- **优先级**: 中
- **状态**: 待开发
- **预估工期**: 4-5天

## 用户故事
**作为** 运维工程师  
**我希望** 有完善的系统监控与日志管理功能  
**以便** 实时了解系统运行状态，快速定位和解决问题，确保系统稳定运行

## 需求描述
建立全面的监控和日志管理系统，包括应用性能监控、基础设施监控、日志收集分析、告警通知等功能，为微服务架构提供可观测性支持。

## 技术实现

### 核心技术栈
- **监控系统**: 
  - Prometheus 2.47+ (指标收集)
  - Grafana 10.2+ (可视化)
  - AlertManager 0.26+ (告警管理)
- **日志系统**: 
  - Elasticsearch 8.11+ (日志存储)
  - Logstash 8.11+ (日志处理)
  - Kibana 8.11+ (日志分析)
  - Filebeat 8.11+ (日志收集)
- **链路追踪**: Jaeger 1.51+
- **应用监控**: 
  - Node.js: prom-client, winston
  - Python: prometheus_client, structlog
- **基础设施监控**: 
  - Node Exporter (服务器监控)
  - cAdvisor (容器监控)
  - Blackbox Exporter (黑盒监控)

### 监控架构设计

#### Prometheus配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # 应用服务监控
  - job_name: 'auth-service'
    static_configs:
      - targets: ['auth-service:3001']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'data-collection-service'
    static_configs:
      - targets: ['data-collection-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  # 基础设施监控
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
      
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
      
  # 数据库监控
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
```

#### 告警规则配置
```yaml
# alert_rules.yml
groups:
  - name: application_alerts
    rules:
      # 服务可用性告警
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务 {{ $labels.job }} 不可用"
          description: "服务 {{ $labels.job }} 已经停止响应超过1分钟"
          
      # 高错误率告警
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "服务 {{ $labels.job }} 错误率过高"
          description: "服务 {{ $labels.job }} 5分钟内错误率超过10%"
          
      # 响应时间告警
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "服务 {{ $labels.job }} 响应时间过长"
          description: "服务 {{ $labels.job }} 95%分位响应时间超过1秒"
          
  - name: infrastructure_alerts
    rules:
      # CPU使用率告警
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "服务器 {{ $labels.instance }} CPU使用率过高"
          description: "服务器 {{ $labels.instance }} CPU使用率超过80%"
          
      # 内存使用率告警
      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "服务器 {{ $labels.instance }} 内存使用率过高"
          description: "服务器 {{ $labels.instance }} 内存使用率超过85%"
          
      # 磁盘空间告警
      - alert: LowDiskSpace
        expr: (1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100 > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "服务器 {{ $labels.instance }} 磁盘空间不足"
          description: "服务器 {{ $labels.instance }} 磁盘使用率超过90%"
```

### 应用监控集成

#### Node.js服务监控
```typescript
// src/middleware/metrics.middleware.ts
import { Injectable, NestMiddleware } from '@nestjs/common'
import { Request, Response, NextFunction } from 'express'
import * as promClient from 'prom-client'

@Injectable()
export class MetricsMiddleware implements NestMiddleware {
  private readonly httpRequestsTotal: promClient.Counter<string>
  private readonly httpRequestDuration: promClient.Histogram<string>
  private readonly activeConnections: promClient.Gauge<string>
  
  constructor() {
    // 注册默认指标
    promClient.collectDefaultMetrics({ prefix: 'auth_service_' })
    
    // HTTP请求总数
    this.httpRequestsTotal = new promClient.Counter({
      name: 'http_requests_total',
      help: 'Total number of HTTP requests',
      labelNames: ['method', 'route', 'status']
    })
    
    // HTTP请求持续时间
    this.httpRequestDuration = new promClient.Histogram({
      name: 'http_request_duration_seconds',
      help: 'Duration of HTTP requests in seconds',
      labelNames: ['method', 'route', 'status'],
      buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
    })
    
    // 活跃连接数
    this.activeConnections = new promClient.Gauge({
      name: 'active_connections',
      help: 'Number of active connections'
    })
  }
  
  /**
   * 处理HTTP请求监控
   * @param req 请求对象
   * @param res 响应对象
   * @param next 下一个中间件
   */
  use(req: Request, res: Response, next: NextFunction) {
    const startTime = Date.now()
    
    // 增加活跃连接数
    this.activeConnections.inc()
    
    // 监听响应结束事件
    res.on('finish', () => {
      const duration = (Date.now() - startTime) / 1000
      const route = req.route?.path || req.path
      
      // 记录请求总数
      this.httpRequestsTotal.inc({
        method: req.method,
        route: route,
        status: res.statusCode.toString()
      })
      
      // 记录请求持续时间
      this.httpRequestDuration.observe(
        {
          method: req.method,
          route: route,
          status: res.statusCode.toString()
        },
        duration
      )
      
      // 减少活跃连接数
      this.activeConnections.dec()
    })
    
    next()
  }
}

// src/controllers/metrics.controller.ts
@Controller('metrics')
export class MetricsController {
  /**
   * 获取Prometheus指标
   * @returns Prometheus格式的指标数据
   */
  @Get()
  async getMetrics(): Promise<string> {
    return promClient.register.metrics()
  }
  
  /**
   * 健康检查端点
   * @returns 健康状态
   */
  @Get('/health')
  async healthCheck(): Promise<object> {
    return {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      version: process.env.npm_package_version
    }
  }
}
```

#### Python服务监控
```python
# src/middleware/metrics_middleware.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Request, Response
import time
from typing import Callable

class MetricsMiddleware:
    def __init__(self):
        # HTTP请求总数
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        # HTTP请求持续时间
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'Duration of HTTP requests in seconds',
            ['method', 'endpoint', 'status'],
            buckets=[0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10]
        )
        
        # 活跃请求数
        self.active_requests = Gauge(
            'active_requests',
            'Number of active requests'
        )
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        处理HTTP请求监控
        
        Args:
            request: FastAPI请求对象
            call_next: 下一个中间件
            
        Returns:
            响应对象
        """
        start_time = time.time()
        
        # 增加活跃请求数
        self.active_requests.inc()
        
        try:
            # 执行请求
            response = await call_next(request)
            
            # 计算请求持续时间
            duration = time.time() - start_time
            
            # 记录指标
            self.http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            self.http_request_duration.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).observe(duration)
            
            return response
            
        finally:
            # 减少活跃请求数
            self.active_requests.dec()

# src/routers/metrics.py
from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import psutil
import time

router = APIRouter()

@router.get('/metrics')
async def get_metrics():
    """
    获取Prometheus指标
    
    Returns:
        Prometheus格式的指标数据
    """
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get('/health')
async def health_check():
    """
    健康检查端点
    
    Returns:
        健康状态信息
    """
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime': time.time() - psutil.boot_time(),
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent
        },
        'cpu_percent': psutil.cpu_percent(),
        'disk_usage': psutil.disk_usage('/').percent
    }
```

### 日志管理系统

#### 结构化日志配置
```typescript
// src/config/logger.config.ts
import { WinstonModule } from 'nest-winston'
import * as winston from 'winston'
import 'winston-elasticsearch'

export const loggerConfig = WinstonModule.createLogger({
  transports: [
    // 控制台输出
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.colorize(),
        winston.format.printf(({ timestamp, level, message, context, trace }) => {
          return `${timestamp} [${context}] ${level}: ${message}${trace ? `\n${trace}` : ''}`
        })
      )
    }),
    
    // 文件输出
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      )
    }),
    
    new winston.transports.File({
      filename: 'logs/combined.log',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      )
    }),
    
    // Elasticsearch输出
    new winston.transports.Elasticsearch({
      level: 'info',
      clientOpts: {
        node: process.env.ELASTICSEARCH_URL || 'http://localhost:9200'
      },
      index: 'auth-service-logs',
      transformer: (logData: any) => {
        return {
          '@timestamp': new Date().toISOString(),
          service: 'auth-service',
          level: logData.level,
          message: logData.message,
          context: logData.meta?.context,
          trace_id: logData.meta?.trace_id,
          user_id: logData.meta?.user_id,
          request_id: logData.meta?.request_id
        }
      }
    })
  ]
})
```

#### Python结构化日志
```python
# src/config/logging_config.py
import structlog
import logging.config
from pythonjsonlogger import jsonlogger

def configure_logging():
    """
    配置结构化日志
    """
    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # 配置标准库logging
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": jsonlogger.JsonFormatter,
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "json",
                "filename": "logs/data-collection.log"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": True
            }
        }
    })

# src/middleware/logging_middleware.py
import structlog
from fastapi import Request
import uuid
import time

logger = structlog.get_logger()

async def logging_middleware(request: Request, call_next):
    """
    日志记录中间件
    
    Args:
        request: FastAPI请求对象
        call_next: 下一个中间件
        
    Returns:
        响应对象
    """
    # 生成请求ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # 绑定请求上下文
    logger_with_context = logger.bind(
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get('user-agent'),
        client_ip=request.client.host
    )
    
    # 记录请求开始
    logger_with_context.info("Request started")
    
    try:
        # 执行请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录请求完成
        logger_with_context.info(
            "Request completed",
            status_code=response.status_code,
            process_time=process_time
        )
        
        return response
        
    except Exception as e:
        # 记录请求错误
        logger_with_context.error(
            "Request failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### Grafana仪表板配置

#### 应用监控仪表板
```json
{
  "dashboard": {
    "title": "历史文本处理系统 - 应用监控",
    "panels": [
      {
        "title": "服务可用性",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "{{ job }}"
          }
        ]
      },
      {
        "title": "请求速率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{ job }} - {{ method }}"
          }
        ]
      },
      {
        "title": "响应时间分布",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(http_request_duration_seconds_bucket[5m])",
            "legendFormat": "{{ le }}"
          }
        ]
      },
      {
        "title": "错误率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "{{ job }}"
          }
        ]
      }
    ]
  }
}
```

## API设计

#### 监控服务控制器
```typescript
// src/controllers/monitoring.controller.ts
import { Controller, Get, Query, Param } from '@nestjs/common'
import { ApiTags, ApiOperation, ApiQuery, ApiResponse } from '@nestjs/swagger'
import { MonitoringService } from '../services/monitoring.service'
import { MetricsQueryDto, AlertsQueryDto } from '../dto/monitoring.dto'

@ApiTags('监控管理')
@Controller('api/v1/monitoring')
export class MonitoringController {
  constructor(private readonly monitoringService: MonitoringService) {}

  /**
   * 获取服务健康状态
   * @returns 服务健康状态信息
   */
  @Get('health')
  @ApiOperation({ summary: '获取服务健康状态' })
  @ApiResponse({ status: 200, description: '健康状态信息' })
  async getHealthStatus() {
    return await this.monitoringService.getHealthStatus()
  }

  /**
   * 获取系统指标
   * @param query 查询参数
   * @returns 系统指标数据
   */
  @Get('metrics')
  @ApiOperation({ summary: '获取系统指标' })
  @ApiQuery({ name: 'service', required: false, description: '服务名称' })
  @ApiQuery({ name: 'timeRange', required: false, description: '时间范围' })
  @ApiQuery({ name: 'metric', required: false, description: '指标名称' })
  async getMetrics(@Query() query: MetricsQueryDto) {
    return await this.monitoringService.getMetrics(query)
  }

  /**
   * 获取告警信息
   * @param query 查询参数
   * @returns 告警信息列表
   */
  @Get('alerts')
  @ApiOperation({ summary: '获取告警信息' })
  @ApiQuery({ name: 'status', required: false, description: '告警状态' })
  @ApiQuery({ name: 'severity', required: false, description: '告警级别' })
  @ApiQuery({ name: 'service', required: false, description: '服务名称' })
  async getAlerts(@Query() query: AlertsQueryDto) {
    return await this.monitoringService.getAlerts(query)
  }

  /**
   * 获取服务拓扑
   * @returns 服务依赖关系图
   */
  @Get('topology')
  @ApiOperation({ summary: '获取服务拓扑' })
  async getServiceTopology() {
    return await this.monitoringService.getServiceTopology()
  }

  /**
   * 获取链路追踪信息
   * @param traceId 追踪ID
   * @returns 链路追踪详情
   */
  @Get('traces/:traceId')
  @ApiOperation({ summary: '获取链路追踪信息' })
  async getTraceDetails(@Param('traceId') traceId: string) {
    return await this.monitoringService.getTraceDetails(traceId)
  }

  /**
   * 获取日志查询结果
   * @param query 查询参数
   * @returns 日志数据
   */
  @Get('logs')
  @ApiOperation({ summary: '查询日志' })
  @ApiQuery({ name: 'query', required: true, description: '查询语句' })
  @ApiQuery({ name: 'timeRange', required: false, description: '时间范围' })
  @ApiQuery({ name: 'size', required: false, description: '返回数量' })
  async searchLogs(@Query() query: any) {
    return await this.monitoringService.searchLogs(query)
  }
}

// src/services/monitoring.service.ts
import { Injectable } from '@nestjs/common'
import { PrometheusService } from './prometheus.service'
import { ElasticsearchService } from './elasticsearch.service'
import { JaegerService } from './jaeger.service'
import { AlertManagerService } from './alertmanager.service'

@Injectable()
export class MonitoringService {
  constructor(
    private readonly prometheusService: PrometheusService,
    private readonly elasticsearchService: ElasticsearchService,
    private readonly jaegerService: JaegerService,
    private readonly alertManagerService: AlertManagerService
  ) {}

  /**
   * 获取服务健康状态
   * @returns 健康状态信息
   */
  async getHealthStatus() {
    const services = await this.prometheusService.queryServiceStatus()
    const overallStatus = services.every(s => s.status === 'healthy') ? 'healthy' : 'unhealthy'
    
    return {
      services,
      overall_status: overallStatus,
      timestamp: new Date().toISOString(),
      summary: {
        total: services.length,
        healthy: services.filter(s => s.status === 'healthy').length,
        unhealthy: services.filter(s => s.status !== 'healthy').length
      }
    }
  }

  /**
   * 获取系统指标
   * @param query 查询参数
   * @returns 指标数据
   */
  async getMetrics(query: any) {
    const timeRange = query.timeRange || '1h'
    const service = query.service
    const metric = query.metric
    
    const metrics = await this.prometheusService.queryMetrics({
      service,
      metric,
      timeRange,
      step: '1m'
    })
    
    return {
      query: query,
      timeRange,
      metrics,
      timestamp: new Date().toISOString()
    }
  }

  /**
   * 获取告警信息
   * @param query 查询参数
   * @returns 告警列表
   */
  async getAlerts(query: any) {
    const alerts = await this.alertManagerService.getAlerts(query)
    
    return {
      alerts,
      total: alerts.length,
      summary: {
        critical: alerts.filter(a => a.severity === 'critical').length,
        warning: alerts.filter(a => a.severity === 'warning').length,
        info: alerts.filter(a => a.severity === 'info').length
      },
      timestamp: new Date().toISOString()
    }
  }

  /**
   * 获取服务拓扑
   * @returns 服务依赖关系
   */
  async getServiceTopology() {
    const services = await this.jaegerService.getServices()
    const dependencies = await this.jaegerService.getDependencies()
    
    return {
      services,
      dependencies,
      graph: this.buildTopologyGraph(services, dependencies)
    }
  }

  /**
   * 获取链路追踪详情
   * @param traceId 追踪ID
   * @returns 追踪详情
   */
  async getTraceDetails(traceId: string) {
    const trace = await this.jaegerService.getTrace(traceId)
    
    return {
      traceId,
      trace,
      summary: {
        duration: trace.duration,
        spans: trace.spans.length,
        services: [...new Set(trace.spans.map(s => s.process.serviceName))],
        errors: trace.spans.filter(s => s.tags.some(t => t.key === 'error' && t.value)).length
      }
    }
  }

  /**
   * 搜索日志
   * @param query 查询参数
   * @returns 日志数据
   */
  async searchLogs(query: any) {
    const logs = await this.elasticsearchService.search({
      index: 'application-logs-*',
      body: {
        query: {
          bool: {
            must: [
              {
                query_string: {
                  query: query.query || '*'
                }
              },
              {
                range: {
                  '@timestamp': {
                    gte: query.from || 'now-1h',
                    lte: query.to || 'now'
                  }
                }
              }
            ]
          }
        },
        sort: [{ '@timestamp': { order: 'desc' } }],
        size: query.size || 100
      }
    })
    
    return {
      logs: logs.body.hits.hits.map(hit => hit._source),
      total: logs.body.hits.total.value,
      took: logs.body.took,
      query: query
    }
  }

  /**
   * 构建拓扑图
   * @param services 服务列表
   * @param dependencies 依赖关系
   * @returns 拓扑图数据
   */
  private buildTopologyGraph(services: any[], dependencies: any[]) {
    const nodes = services.map(service => ({
      id: service.name,
      label: service.name,
      type: 'service',
      metadata: service
    }))
    
    const edges = dependencies.map(dep => ({
      source: dep.parent,
      target: dep.child,
      weight: dep.callCount,
      metadata: dep
    }))
    
    return { nodes, edges }
  }
}

// src/dto/monitoring.dto.ts
import { IsOptional, IsString, IsEnum, IsNumber } from 'class-validator'
import { ApiPropertyOptional } from '@nestjs/swagger'

export class MetricsQueryDto {
  @ApiPropertyOptional({ description: '服务名称' })
  @IsOptional()
  @IsString()
  service?: string

  @ApiPropertyOptional({ description: '时间范围', example: '1h' })
  @IsOptional()
  @IsString()
  timeRange?: string

  @ApiPropertyOptional({ description: '指标名称' })
  @IsOptional()
  @IsString()
  metric?: string

  @ApiPropertyOptional({ description: '查询步长', example: '1m' })
  @IsOptional()
  @IsString()
  step?: string
}

export class AlertsQueryDto {
  @ApiPropertyOptional({ description: '告警状态', enum: ['active', 'resolved', 'suppressed'] })
  @IsOptional()
  @IsEnum(['active', 'resolved', 'suppressed'])
  status?: string

  @ApiPropertyOptional({ description: '告警级别', enum: ['critical', 'warning', 'info'] })
  @IsOptional()
  @IsEnum(['critical', 'warning', 'info'])
  severity?: string

  @ApiPropertyOptional({ description: '服务名称' })
  @IsOptional()
  @IsString()
  service?: string

  @ApiPropertyOptional({ description: '页面大小' })
  @IsOptional()
  @IsNumber()
  limit?: number

  @ApiPropertyOptional({ description: '页面偏移' })
  @IsOptional()
  @IsNumber()
  offset?: number
}
```

#### 分布式追踪集成
```typescript
// src/services/jaeger.service.ts
import { Injectable } from '@nestjs/common'
import { ConfigService } from '@nestjs/config'
import axios from 'axios'

@Injectable()
export class JaegerService {
  private readonly jaegerUrl: string
  
  constructor(private readonly configService: ConfigService) {
    this.jaegerUrl = this.configService.get<string>('JAEGER_QUERY_URL') || 'http://localhost:16686'
  }

  /**
   * 获取所有服务列表
   * @returns 服务列表
   */
  async getServices(): Promise<any[]> {
    try {
      const response = await axios.get(`${this.jaegerUrl}/api/services`)
      return response.data.data || []
    } catch (error) {
      console.error('Failed to fetch services from Jaeger:', error)
      return []
    }
  }

  /**
   * 获取服务依赖关系
   * @param endTs 结束时间戳
   * @param lookback 回溯时间（毫秒）
   * @returns 依赖关系列表
   */
  async getDependencies(endTs?: number, lookback: number = 86400000): Promise<any[]> {
    try {
      const params = {
        endTs: endTs || Date.now() * 1000,
        lookback
      }
      
      const response = await axios.get(`${this.jaegerUrl}/api/dependencies`, { params })
      return response.data.data || []
    } catch (error) {
      console.error('Failed to fetch dependencies from Jaeger:', error)
      return []
    }
  }

  /**
   * 获取链路追踪详情
   * @param traceId 追踪ID
   * @returns 追踪详情
   */
  async getTrace(traceId: string): Promise<any> {
    try {
      const response = await axios.get(`${this.jaegerUrl}/api/traces/${traceId}`)
      return response.data.data[0] || null
    } catch (error) {
      console.error(`Failed to fetch trace ${traceId} from Jaeger:`, error)
      return null
    }
  }

  /**
   * 搜索链路追踪
   * @param params 搜索参数
   * @returns 追踪列表
   */
  async searchTraces(params: {
    service?: string
    operation?: string
    tags?: string
    start?: number
    end?: number
    limit?: number
  }): Promise<any[]> {
    try {
      const searchParams = {
        service: params.service,
        operation: params.operation,
        tags: params.tags,
        start: params.start || (Date.now() - 3600000) * 1000, // 1小时前
        end: params.end || Date.now() * 1000,
        limit: params.limit || 20
      }
      
      const response = await axios.get(`${this.jaegerUrl}/api/traces`, { params: searchParams })
      return response.data.data || []
    } catch (error) {
      console.error('Failed to search traces from Jaeger:', error)
      return []
    }
  }
}

// src/services/prometheus.service.ts
import { Injectable } from '@nestjs/common'
import { ConfigService } from '@nestjs/config'
import axios from 'axios'

@Injectable()
export class PrometheusService {
  private readonly prometheusUrl: string
  
  constructor(private readonly configService: ConfigService) {
    this.prometheusUrl = this.configService.get<string>('PROMETHEUS_URL') || 'http://localhost:9090'
  }

  /**
   * 查询Prometheus指标
   * @param params 查询参数
   * @returns 指标数据
   */
  async queryMetrics(params: {
    service?: string
    metric?: string
    timeRange?: string
    step?: string
  }): Promise<any> {
    try {
      let query = params.metric || 'up'
      
      if (params.service) {
        query += `{job="${params.service}"}`
      }
      
      const endTime = Math.floor(Date.now() / 1000)
      const startTime = endTime - this.parseTimeRange(params.timeRange || '1h')
      
      const response = await axios.get(`${this.prometheusUrl}/api/v1/query_range`, {
        params: {
          query,
          start: startTime,
          end: endTime,
          step: params.step || '1m'
        }
      })
      
      return response.data.data
    } catch (error) {
      console.error('Failed to query metrics from Prometheus:', error)
      return { result: [] }
    }
  }

  /**
   * 查询服务状态
   * @returns 服务状态列表
   */
  async queryServiceStatus(): Promise<any[]> {
    try {
      const response = await axios.get(`${this.prometheusUrl}/api/v1/query`, {
        params: {
          query: 'up'
        }
      })
      
      const results = response.data.data.result || []
      
      return results.map(result => ({
        name: result.metric.job || result.metric.instance,
        status: result.value[1] === '1' ? 'healthy' : 'unhealthy',
        instance: result.metric.instance,
        job: result.metric.job,
        last_check: new Date().toISOString(),
        uptime: this.calculateUptime(result.metric.instance)
      }))
    } catch (error) {
      console.error('Failed to query service status from Prometheus:', error)
      return []
    }
  }

  /**
   * 解析时间范围
   * @param timeRange 时间范围字符串（如 '1h', '30m', '1d'）
   * @returns 秒数
   */
  private parseTimeRange(timeRange: string): number {
    const match = timeRange.match(/(\d+)([smhd])/)
    if (!match) return 3600 // 默认1小时
    
    const value = parseInt(match[1])
    const unit = match[2]
    
    switch (unit) {
      case 's': return value
      case 'm': return value * 60
      case 'h': return value * 3600
      case 'd': return value * 86400
      default: return 3600
    }
  }

  /**
   * 计算服务运行时间
   * @param instance 实例名称
   * @returns 运行时间（秒）
   */
  private async calculateUptime(instance: string): Promise<number> {
    try {
      const response = await axios.get(`${this.prometheusUrl}/api/v1/query`, {
        params: {
          query: `time() - process_start_time_seconds{instance="${instance}"}`
        }
      })
      
      const result = response.data.data.result[0]
      return result ? parseFloat(result.value[1]) : 0
    } catch (error) {
      return 0
    }
  }
}

// src/services/alertmanager.service.ts
import { Injectable } from '@nestjs/common'
import { ConfigService } from '@nestjs/config'
import axios from 'axios'

@Injectable()
export class AlertManagerService {
  private readonly alertManagerUrl: string
  
  constructor(private readonly configService: ConfigService) {
    this.alertManagerUrl = this.configService.get<string>('ALERTMANAGER_URL') || 'http://localhost:9093'
  }

  /**
   * 获取告警列表
   * @param params 查询参数
   * @returns 告警列表
   */
  async getAlerts(params: {
    status?: string
    severity?: string
    service?: string
    limit?: number
    offset?: number
  }): Promise<any[]> {
    try {
      const response = await axios.get(`${this.alertManagerUrl}/api/v1/alerts`)
      let alerts = response.data.data || []
      
      // 过滤告警
      if (params.status) {
        alerts = alerts.filter(alert => alert.status.state === params.status)
      }
      
      if (params.severity) {
        alerts = alerts.filter(alert => alert.labels.severity === params.severity)
      }
      
      if (params.service) {
        alerts = alerts.filter(alert => 
          alert.labels.job === params.service || 
          alert.labels.service === params.service
        )
      }
      
      // 分页
      const offset = params.offset || 0
      const limit = params.limit || 50
      
      return alerts.slice(offset, offset + limit).map(alert => ({
        id: alert.fingerprint,
        name: alert.labels.alertname,
        severity: alert.labels.severity,
        status: alert.status.state,
        started_at: alert.startsAt,
        ended_at: alert.endsAt,
        description: alert.annotations.description || alert.annotations.summary,
        labels: alert.labels,
        annotations: alert.annotations,
        generator_url: alert.generatorURL
      }))
    } catch (error) {
      console.error('Failed to fetch alerts from AlertManager:', error)
      return []
    }
  }

  /**
   * 静默告警
   * @param alertId 告警ID
   * @param duration 静默时长（秒）
   * @param comment 静默原因
   * @returns 操作结果
   */
  async silenceAlert(alertId: string, duration: number, comment: string): Promise<boolean> {
    try {
      const endTime = new Date(Date.now() + duration * 1000).toISOString()
      
      const silenceData = {
        matchers: [
          {
            name: 'fingerprint',
            value: alertId,
            isRegex: false
          }
        ],
        startsAt: new Date().toISOString(),
        endsAt: endTime,
        comment: comment,
        createdBy: 'monitoring-service'
      }
      
      await axios.post(`${this.alertManagerUrl}/api/v1/silences`, silenceData)
      return true
    } catch (error) {
      console.error('Failed to silence alert:', error)
      return false
    }
  }
}
```

### 监控配置管理
```typescript
// src/config/monitoring.config.ts
import { registerAs } from '@nestjs/config'

export default registerAs('monitoring', () => ({
  // Prometheus配置
  prometheus: {
    url: process.env.PROMETHEUS_URL || 'http://localhost:9090',
    scrapeInterval: process.env.PROMETHEUS_SCRAPE_INTERVAL || '15s',
    evaluationInterval: process.env.PROMETHEUS_EVALUATION_INTERVAL || '15s'
  },
  
  // Grafana配置
  grafana: {
    url: process.env.GRAFANA_URL || 'http://localhost:3000',
    adminUser: process.env.GRAFANA_ADMIN_USER || 'admin',
    adminPassword: process.env.GRAFANA_ADMIN_PASSWORD || 'admin'
  },
  
  // AlertManager配置
  alertManager: {
    url: process.env.ALERTMANAGER_URL || 'http://localhost:9093',
    webhookUrl: process.env.ALERTMANAGER_WEBHOOK_URL,
    slackWebhook: process.env.SLACK_WEBHOOK_URL,
    emailConfig: {
      smtp: process.env.SMTP_SERVER,
      from: process.env.ALERT_EMAIL_FROM,
      to: process.env.ALERT_EMAIL_TO
    }
  },
  
  // Jaeger配置
  jaeger: {
    queryUrl: process.env.JAEGER_QUERY_URL || 'http://localhost:16686',
    collectorUrl: process.env.JAEGER_COLLECTOR_URL || 'http://localhost:14268',
    samplingRate: parseFloat(process.env.JAEGER_SAMPLING_RATE || '0.1')
  },
  
  // Elasticsearch配置
  elasticsearch: {
    url: process.env.ELASTICSEARCH_URL || 'http://localhost:9200',
    username: process.env.ELASTICSEARCH_USERNAME,
    password: process.env.ELASTICSEARCH_PASSWORD,
    indexPrefix: process.env.ELASTICSEARCH_INDEX_PREFIX || 'application-logs',
    maxRetries: parseInt(process.env.ELASTICSEARCH_MAX_RETRIES || '3'),
    requestTimeout: parseInt(process.env.ELASTICSEARCH_REQUEST_TIMEOUT || '30000')
  },
  
  // 指标配置
  metrics: {
    defaultLabels: {
      service: process.env.SERVICE_NAME || 'monitoring-service',
      version: process.env.SERVICE_VERSION || '1.0.0',
      environment: process.env.NODE_ENV || 'development'
    },
    collectInterval: parseInt(process.env.METRICS_COLLECT_INTERVAL || '5000'),
    retentionPeriod: process.env.METRICS_RETENTION_PERIOD || '30d'
  },
  
  // 告警配置
  alerts: {
    defaultSeverity: process.env.DEFAULT_ALERT_SEVERITY || 'warning',
    escalationTimeout: parseInt(process.env.ALERT_ESCALATION_TIMEOUT || '300'),
    maxAlerts: parseInt(process.env.MAX_ALERTS_PER_SERVICE || '100'),
    groupWait: process.env.ALERT_GROUP_WAIT || '10s',
    groupInterval: process.env.ALERT_GROUP_INTERVAL || '10s',
    repeatInterval: process.env.ALERT_REPEAT_INTERVAL || '1h'
  }
}))

// src/modules/monitoring.module.ts
import { Module } from '@nestjs/common'
import { ConfigModule } from '@nestjs/config'
import { MonitoringController } from '../controllers/monitoring.controller'
import { MonitoringService } from '../services/monitoring.service'
import { PrometheusService } from '../services/prometheus.service'
import { JaegerService } from '../services/jaeger.service'
import { AlertManagerService } from '../services/alertmanager.service'
import { ElasticsearchService } from '../services/elasticsearch.service'
import monitoringConfig from '../config/monitoring.config'

@Module({
  imports: [
    ConfigModule.forFeature(monitoringConfig)
  ],
  controllers: [MonitoringController],
  providers: [
    MonitoringService,
    PrometheusService,
    JaegerService,
    AlertManagerService,
    ElasticsearchService
  ],
  exports: [
    MonitoringService,
    PrometheusService,
    JaegerService,
    AlertManagerService,
    ElasticsearchService
  ]
})
export class MonitoringModule {}
```

## 验收标准

### 功能验收
- [ ] Prometheus指标收集正常
- [ ] Grafana仪表板显示正确
- [ ] 告警规则配置有效
- [ ] 日志收集和查询功能完整
- [ ] 链路追踪数据准确
- [ ] 监控API响应正常
- [ ] 服务拓扑图生成正确
- [ ] 告警静默和管理功能正常
- [ ] 分布式追踪搜索功能完整

### 性能验收
- [ ] 指标收集延迟 < 15秒
- [ ] 仪表板加载时间 < 3秒
- [ ] 日志查询响应时间 < 2秒
- [ ] 监控系统资源占用 < 10%
- [ ] 链路追踪查询响应时间 < 1秒
- [ ] 告警处理延迟 < 30秒

### 可用性验收
- [ ] 监控系统可用性 > 99.9%
- [ ] 告警通知及时性 < 1分钟
- [ ] 数据保留期 > 30天
- [ ] 仪表板自动刷新正常
- [ ] 监控服务自动恢复能力
- [ ] 多渠道告警通知支持

## 业务价值
- 提供系统运行状态的实时可见性
- 快速发现和定位系统问题
- 支持容量规划和性能优化
- 提高系统可靠性和稳定性

## 依赖关系
- **前置条件**: Story 1.1 (微服务架构)
- **后续依赖**: 所有其他服务的监控集成

## 风险与缓解
- **风险**: 监控数据量过大影响性能
- **缓解**: 合理设置采集频率和数据保留策略
- **风险**: 告警风暴影响运维效率
- **缓解**: 设置告警聚合和抑制规则

## 开发任务分解
1. Prometheus和Grafana部署配置 (1天)
2. 应用监控指标集成 (1天)
3. 告警规则配置和测试 (1天)
4. ELK日志系统搭建 (1天)
5. 仪表板开发和API接口 (1天)