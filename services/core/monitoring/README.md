# 监控服务模块 (Monitoring Service Module)

## 🏆 项目状态
**Epic 1 Story 1.4 已完成** ✅ (2025-09-04) - 系统监控与日志管理服务已完成开发，包含完整的Prometheus + Grafana + Jaeger + ELK Stack监控栈，通过集成测试验证。

## 概述

监控服务模块为历史文本处理项目提供完整的可观测性支持，包括指标收集、链路追踪、告警管理和日志聚合等功能。该模块实现了基于Prometheus + Grafana + Jaeger + ELK Stack的全栈监控解决方案。

## 模块结构

```
services/core/monitoring/
├── __init__.py                 # 模块初始化，导出所有监控组件
├── metrics_middleware.py       # Prometheus指标中间件
├── monitoring_controller.py    # 监控API控制器
├── tracing_service.py          # OpenTelemetry链路追踪服务
├── alert_service.py           # 告警规则管理和通知服务
├── logging_service.py         # 统一日志收集和搜索服务
├── monitoring_service.py      # 监控服务主入口和编排
└── README.md                  # 本文档
```

## 核心组件

### 📊 指标收集 (metrics_middleware.py)
- **功能**: Prometheus指标收集和HTTP请求监控
- **特性**: 
  - HTTP请求总数、响应时间、错误率统计
  - 业务指标收集 (文件上传、文本处理、OCR操作等)
  - 队列大小监控和病毒扫描统计
- **使用**: FastAPI中间件自动集成

### 🔍 链路追踪 (tracing_service.py) 
- **功能**: 基于OpenTelemetry的分布式链路追踪
- **特性**:
  - Jaeger后端集成和B3传播格式支持
  - FastAPI、HTTP请求、数据库操作自动插桩
  - 上下文管理器和手动Span管理
  - 可配置采样率和批量处理
- **使用**: 服务初始化时自动启用

### 🚨 告警管理 (alert_service.py)
- **功能**: 完整的告警规则管理和多渠道通知系统
- **特性**:
  - Prometheus查询表达式规则评估
  - 邮件(SMTP)和Slack Webhook通知
  - 告警生命周期管理(创建/更新/解决/静默)
  - 默认告警规则 (服务健康、基础设施、安全)
- **使用**: 后台异步告警监控

### 📝 日志管理 (logging_service.py)
- **功能**: 统一日志收集、存储和搜索服务  
- **特性**:
  - 基于structlog的结构化JSON日志
  - ElasticSearch异步批量写入和索引管理
  - 全文搜索、时间范围、级别过滤
  - 日志统计分析和自动清理
- **使用**: 应用日志统一接入点

### 🎛️ 监控控制器 (monitoring_controller.py)
- **功能**: RESTful API提供监控数据查询和管理
- **特性**:
  - 健康检查、系统信息、指标查询
  - 告警管理和日志搜索API
  - Swagger文档自动生成
- **使用**: HTTP API访问监控功能

### 🏗️ 主服务 (monitoring_service.py)  
- **功能**: 监控服务主入口和生命周期管理
- **特性**:
  - 统一初始化所有监控组件
  - FastAPI应用创建和中间件集成
  - 后台任务管理和优雅关闭
  - 配置管理和环境适配
- **使用**: 独立服务或作为模块集成

## 快速开始

### 1. 基本使用

```python
# 创建监控服务实例
from services.core.monitoring.monitoring_service import create_monitoring_service

# 使用默认配置
monitoring_service = create_monitoring_service()

# 初始化监控组件
await monitoring_service.initialize()

# 启动监控服务 (独立服务模式)
await monitoring_service.start()
```

### 2. 集成到现有FastAPI应用

```python
from services.core.monitoring import PrometheusMetricsMiddleware, get_business_metrics
from services.core.monitoring.tracing_service import initialize_tracing_for_service

app = FastAPI(title="My Service")

# 添加监控中间件
metrics_middleware = PrometheusMetricsMiddleware(app, service_name="my-service")
app.add_middleware(metrics_middleware.__class__, middleware=metrics_middleware)

# 启用链路追踪
tracing_service = initialize_tracing_for_service("my-service")
tracing_service.instrument_fastapi(app)

# 使用业务指标
business_metrics = get_business_metrics("my-service")
business_metrics.record_text_processing("analyze", "success", 2.5)
```

### 3. 日志记录

```python
from services.core.monitoring.logging_service import get_logging_service, create_default_log_config

# 创建日志配置
log_config = create_default_log_config("my-service")
logging_service = get_logging_service(log_config)

# 使用结构化日志
logger = logging_service.bind_context(component="my-component")
logger.info("处理请求完成", user_id="123", processing_time=2.5)

# 搜索日志
logs = await logging_service.search_logs("error", size=100)
```

### 4. 告警配置

```python
from services.core.monitoring.alert_service import get_alert_manager, AlertRule, AlertSeverity

alert_manager = get_alert_manager()

# 添加自定义告警规则
custom_rule = AlertRule(
    name="high_response_time",
    query="http_request_duration_seconds > 2.0",
    condition="> 0",
    duration=300,
    severity=AlertSeverity.WARNING,
    summary="响应时间过长",
    description="API响应时间超过2秒"
)

alert_manager.add_rule(custom_rule)
await alert_manager.start_monitoring()
```

## 配置选项

### 环境变量配置

```bash
# 服务配置
SERVICE_NAME=monitoring-service
SERVICE_VERSION=1.0.0
SERVICE_HOST=0.0.0.0  
SERVICE_PORT=8004

# 链路追踪配置
TRACING_ENABLED=true
JAEGER_ENDPOINT=http://localhost:14268/api/traces
JAEGER_SAMPLING_RATE=0.1

# 告警配置
EMAIL_ALERTS_ENABLED=false
SLACK_ALERTS_ENABLED=false
SMTP_SERVER=localhost
ALERT_EMAILS=admin@example.com

# 日志配置
LOG_LEVEL=INFO
ELASTICSEARCH_ENABLED=true
ELASTICSEARCH_URL=http://localhost:9200
```

### 程序化配置

```python
config = {
    "service": {
        "name": "my-monitoring-service",
        "host": "0.0.0.0",
        "port": 8004,
        "debug": False
    },
    "metrics": {
        "enabled": True,
        "prometheus_enabled": True,
        "collect_interval": 15
    },
    "tracing": {
        "enabled": True,
        "jaeger_endpoint": "http://jaeger:14268/api/traces",
        "sampling_rate": 0.1
    },
    "alerting": {
        "enabled": True,
        "check_interval": 60,
        "email_enabled": False,
        "slack_enabled": False
    },
    "logging": {
        "enabled": True,
        "level": "INFO",
        "elasticsearch_enabled": True
    }
}

monitoring_service = create_monitoring_service("my-service", config)
```

## API文档

启动监控服务后，访问以下端点查看完整API文档:

- **Swagger UI**: http://localhost:8004/docs
- **ReDoc**: http://localhost:8004/redoc  
- **OpenAPI Schema**: http://localhost:8004/openapi.json

### 主要API端点

```
GET  /                          # 服务状态
GET  /api/v1/monitoring/health  # 健康检查
GET  /api/v1/monitoring/system  # 系统信息
GET  /api/v1/monitoring/metrics # Prometheus指标
GET  /api/v1/monitoring/alerts  # 活跃告警
POST /api/v1/monitoring/alerts/{id}/silence  # 静默告警
GET  /api/v1/monitoring/logs/search  # 搜索日志
```

## 部署

### Docker部署

```bash
# 启动完整监控栈
docker-compose -f docker-compose.monitoring.yml up -d

# 访问监控服务
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)  
# Jaeger: http://localhost:16686
# Kibana: http://localhost:5601
```

### Kubernetes部署

参考 `infrastructure/kubernetes/monitoring/` 目录下的部署文件。

## 测试

```bash
# 运行单元测试
python -m pytest tests/unit/monitoring/ -v

# 运行集成测试  
python -m pytest tests/integration/monitoring/ -v

# 运行特定测试
python -m pytest tests/unit/monitoring/test_monitoring_service.py::TestMonitoringService::test_monitoring_service_initialization -v
```

## 故障排除

### 常见问题

1. **模块导入失败**: 检查OpenTelemetry依赖包是否完整安装
2. **Jaeger连接失败**: 确认Jaeger服务正在运行且端点正确
3. **ElasticSearch写入失败**: 检查ES服务状态和网络连通性
4. **告警不触发**: 验证Prometheus查询表达式和告警规则配置

### 日志调试

```bash
# 查看监控服务日志
docker logs monitoring-service

# 查看特定组件日志
tail -f logs/monitoring-service.log | grep ERROR
```

## 性能考虑

- **指标收集**: 默认15秒间隔，可根据需要调整
- **链路追踪**: 默认10%采样率，生产环境建议1-5%
- **日志批处理**: 默认批量大小100条，高负载环境可增加
- **ElasticSearch**: 建议分配至少2GB内存用于日志索引

## 版本历史

- **v1.0.0** (2025-09-04): 初始版本，完整监控栈实现
  - Prometheus + Grafana指标监控
  - Jaeger链路追踪集成  
  - AlertManager告警管理
  - ELK Stack日志系统
  - 完整API接口和文档

## 贡献

监控服务模块作为历史文本处理项目的核心基础设施组件，欢迎提交问题报告和改进建议。

## 许可证

本项目使用与主项目相同的许可证。