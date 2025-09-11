# 分析报告服务 (Analytics Reporting Service)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于AI的智能数据分析和报告生成服务，为历史文本项目提供全面的数据洞察、可视化分析和多格式报告导出功能。

## 🚀 核心特性

### 智能数据分析
- **多维度分析**: 内容表现、平台对比、趋势分析、用户行为分析
- **机器学习**: 基于scikit-learn的异常检测和预测分析
- **实时监控**: 实时数据收集和指标监控
- **智能洞察**: 自动生成数据洞察和业务建议

### 多格式报告生成
- **PDF报告**: 专业的PDF格式分析报告
- **Excel导出**: 支持图表的Excel数据导出
- **JSON数据**: 结构化JSON格式数据导出
- **可视化图表**: matplotlib、plotly多种图表库支持

### 企业级架构
- **多数据库**: PostgreSQL + InfluxDB + ClickHouse + Redis
- **异步处理**: 基于Celery的分布式任务队列
- **高可用**: 支持水平扩展和负载均衡
- **API优先**: RESTful API设计，支持微服务架构

## 🏗️ 技术架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│     Vue 3       │    │   API Gateway    │    │  分析服务集群        │
│   前端界面       │◄──►│  (Kong/Nginx)    │◄──►│FastAPI + ML + Celery│
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          │
                       ┌──────────────────────────────────┼──────────────────┐
                       │                                  │                  │
              ┌────────▼────────┐              ┌────────▼────────┐  ┌───────▼──────┐
              │   PostgreSQL    │              │   InfluxDB      │  │  ClickHouse  │
              │                 │              │                 │  │              │
              │ 分析任务、模板   │              │ 时序指标数据     │  │ OLAP分析数据 │
              │ 报告、告警配置   │              │ 实时监控数据     │  │ 聚合统计数据 │
              └─────────────────┘              └─────────────────┘  └──────────────┘
                       │
              ┌────────▼────────┐
              │     Redis       │
              │                 │
              │ 缓存、队列、     │
              │ 会话、实时数据   │
              └─────────────────┘
```

## 📦 安装部署

### 环境要求

- **Python**: 3.11+
- **PostgreSQL**: 15+
- **InfluxDB**: 2.7+
- **ClickHouse**: 23.8+
- **Redis**: 7+
- **Docker**: 20.10+ (推荐)
- **Docker Compose**: 2.0+

### 快速启动 (Docker)

1. **克隆项目**
```bash
git clone <repository-url>
cd services/analytics-reporting-service
```

2. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，配置必要的环境变量
```

3. **启动开发环境**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

4. **访问服务**
- **API文档**: http://localhost:8099/docs
- **服务健康**: http://localhost:8099/health
- **Redis管理**: http://localhost:8084 (admin/admin)
- **InfluxDB界面**: http://localhost:8086
- **Grafana监控**: http://localhost:3001 (admin/admin123)

### 本地开发

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置数据库**
```bash
# 启动数据库服务
docker-compose -f docker-compose.dev.yml up postgres-analytics influxdb-analytics clickhouse-analytics redis-analytics -d

# 运行数据库初始化
python -c "
import asyncio
from src.models import init_database
asyncio.run(init_database())
"
```

3. **启动服务**
```bash
# 启动主服务
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8099

# 启动Celery Worker (新终端)
celery -A src.scheduler.celery_app worker -Q analytics,reports -l info

# 启动Celery Beat (新终端)
celery -A src.scheduler.celery_app beat -l info
```

## 🔧 配置说明

### 核心配置

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `ENVIRONMENT` | 运行环境 | development |
| `DEBUG` | 调试模式 | true |
| `HOST` | 服务监听地址 | 0.0.0.0 |
| `PORT` | 服务端口 | 8099 |

### 数据库配置

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `DB_POSTGRES_URL` | PostgreSQL连接URL | postgresql+asyncpg://postgres:password@localhost:5439/historical_text_analytics |
| `DB_INFLUXDB_URL` | InfluxDB连接URL | http://localhost:8086 |
| `DB_CLICKHOUSE_HOST` | ClickHouse主机地址 | localhost |
| `DB_REDIS_URL` | Redis连接URL | redis://localhost:6383/6 |

### 机器学习配置

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `ML_MODEL_CACHE_DIR` | 模型缓存目录 | ./models |
| `ML_ANOMALY_DETECTION_THRESHOLD` | 异常检测阈值 | 0.95 |
| `ML_FORECAST_DAYS` | 预测天数 | 30 |
| `ML_MIN_TRAINING_SAMPLES` | 最小训练样本数 | 100 |

## 📖 API 使用指南

### 创建分析任务

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8099/api/v1/analytics/tasks",
        params={"user_id": "user123"},
        json={
            "title": "内容表现分析",
            "task_type": "content_performance",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-31T23:59:59Z",
            "priority": 8
        }
    )
    
    result = response.json()
    print(f"任务ID: {result['data']['task_id']}")
```

### 获取分析结果

```python
# 内容表现分析
performance = await client.get(
    "http://localhost:8099/api/v1/analytics/content-performance",
    params={
        "user_id": "user123",
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-31T23:59:59Z"
    }
)

# 平台对比分析
comparison = await client.get(
    "http://localhost:8099/api/v1/analytics/platform-comparison",
    params={"user_id": "user123"}
)

# 趋势分析
trends = await client.get(
    "http://localhost:8099/api/v1/analytics/trends",
    params={
        "user_id": "user123",
        "metric_names": "views,likes,comments",
        "time_period": "daily"
    }
)
```

### 生成报告

```python
# 生成PDF报告
report = await client.post(
    "http://localhost:8099/api/v1/reports/generate",
    params={"user_id": "user123"},
    json={
        "analysis_task_id": "task-uuid-here",
        "export_format": "pdf",
        "title": "月度内容分析报告"
    }
)

# 下载报告
report_id = report.json()["data"]["report_id"]
download = await client.get(f"http://localhost:8099/api/v1/reports/{report_id}/download")
```

## 🧪 测试

### 运行测试

```bash
# 安装测试依赖
pip install -r requirements.txt

# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term

# 运行特定测试模块
pytest tests/unit/test_analytics_service.py -v

# 运行集成测试
pytest tests/integration/ -v
```

### 测试数据库

```bash
# 启动测试数据库
docker-compose --profile test up postgres-test influxdb-test -d

# 设置测试环境变量
export ENVIRONMENT=testing
export DB_POSTGRES_URL=postgresql+asyncpg://test_user:test_password@localhost:5440/test_analytics
```

## 📊 监控和运维

### 健康检查

```bash
# 基础健康检查
curl http://localhost:8099/health

# 详细系统状态
curl http://localhost:8099/ready

# 服务信息
curl http://localhost:8099/info
```

### 性能监控

- **Grafana仪表板**: http://localhost:3001 - 数据可视化和监控
- **InfluxDB界面**: http://localhost:8086 - 时序数据管理
- **Redis Commander**: http://localhost:8084 - Redis数据管理
- **API文档**: http://localhost:8099/docs - 交互式API文档

### 日志管理

```bash
# 查看服务日志
docker-compose -f docker-compose.dev.yml logs -f analytics-reporting-service

# 查看Celery Worker日志
docker-compose -f docker-compose.dev.yml logs -f celery-worker-analytics

# 查看数据库日志
docker-compose -f docker-compose.dev.yml logs -f postgres-analytics
```

## 🔐 安全考虑

### 数据保护
- **敏感数据加密**: 数据库连接信息和API密钥采用环境变量管理
- **输入验证**: 严格的API参数验证和SQL注入防护
- **访问控制**: 基于用户ID的细粒度权限管理

### API安全
- **CORS控制**: 可配置的跨域访问策略
- **错误处理**: 生产环境不暴露敏感错误信息
- **日志审计**: 完整的API访问日志记录

## 🚀 生产部署

### Docker部署

```bash
# 构建生产镜像
docker build --target production -t analytics-reporting-service:latest .

# 使用生产配置启动
docker-compose -f docker-compose.yml up -d
```

### 环境变量配置

```bash
# 生产环境必须设置的环境变量
export ENVIRONMENT=production
export DEBUG=false
export SECRET_KEY="your-256-bit-secret-key"

# 数据库连接（生产环境）
export DB_POSTGRES_URL="postgresql+asyncpg://user:password@prod-db:5432/analytics"
export DB_INFLUXDB_TOKEN="production-influxdb-token"
export DB_REDIS_URL="redis://prod-redis:6379/6"
```

## 🤝 开发贡献

### 开发流程

1. Fork项目并创建功能分支
2. 编写代码和测试用例
3. 确保所有测试通过
4. 遵循代码规范和提交规范
5. 提交Pull Request

### 代码规范

```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 代码检查
flake8 src/ tests/
mypy src/

# 运行所有质量检查
black src/ && isort src/ && flake8 src/ && mypy src/ && pytest
```

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 🆘 支持和反馈

- **问题反馈**: [GitHub Issues](https://github.com/your-org/analytics-reporting-service/issues)
- **功能请求**: [Feature Requests](https://github.com/your-org/analytics-reporting-service/discussions)
- **技术支持**: analytics-support@yourdomain.com

## 📚 相关文档

- [API文档](http://localhost:8099/docs) - 完整的REST API文档
- [架构设计](docs/architecture.md) - 系统架构详细说明
- [数据模型](docs/data-models.md) - 数据库模型设计文档
- [部署指南](docs/deployment.md) - 生产环境部署指导
- [开发指南](docs/development.md) - 本地开发环境搭建

## 📈 路线图

### v1.1 (计划中)
- [ ] 更多机器学习算法支持 (深度学习、时序预测)
- [ ] 实时数据推送和WebSocket支持
- [ ] 高级可视化组件和自定义图表
- [ ] 报告调度和自动发送功能

### v1.2 (未来)
- [ ] 多租户支持和数据隔离
- [ ] 高级权限管理和数据脱敏
- [ ] 国际化和多语言支持
- [ ] 云原生部署和自动扩缩容

---

**版本**: 1.0.0  
**更新时间**: 2024-01-11  
**维护者**: 历史文本优化项目团队