# 自动内容调度服务 (Automated Content Scheduling Service)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于AI的智能内容调度服务，为多平台社交媒体内容发布提供自动化调度、优化和管理功能。

## 🚀 核心特性

### 智能调度引擎
- **机器学习优化**: 基于RandomForest的发布时间智能优化
- **多策略调度**: 支持立即执行、最优时间、用户偏好、负载均衡等策略
- **冲突检测**: 自动检测时间重叠、资源冲突、平台限制等问题
- **循环任务**: 支持复杂的RRULE循环规则

### 多平台集成
- **5大平台支持**: 新浪微博、微信公众号、抖音、今日头条、百家号
- **统一API**: 与多平台账号管理服务和内容发布服务无缝集成
- **平台特定配置**: 针对不同平台的个性化配置和限制

### 性能分析与优化
- **实时性能监控**: 参与度、触达率、转化率等关键指标跟踪
- **用户行为分析**: 基于历史数据的个性化发布时间推荐
- **A/B测试支持**: 对比分析不同调度策略的效果

### 高可用架构
- **异步处理**: 基于Celery的分布式任务队列
- **水平扩展**: 支持多Worker节点部署
- **容错机制**: 完整的重试、降级和恢复策略

## 🏗️ 技术架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│     Vue 3       │    │   API Gateway    │    │  调度服务集群        │
│   前端界面       │◄──►│  (Kong/Nginx)    │◄──►│   FastAPI + ML      │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          │
                       ┌──────────────────────────────────┼──────────────────────┐
                       │                                  │                      │
              ┌────────▼────────┐              ┌────────▼────────┐    ┌─────────▼─────────┐
              │   PostgreSQL    │              │   Redis集群      │    │   外部服务集成     │
              │                 │              │                 │    │                   │
              │ 调度任务、分析   │              │ 缓存、队列、     │    │ 账号管理、内容发布  │
              │ 冲突、优化日志   │              │ 会话管理        │    │ 数据存储服务      │
              └─────────────────┘              └─────────────────┘    └───────────────────┘
                       │
              ┌────────▼────────┐
              │  Celery集群     │
              │                 │
              │ 调度、发布、     │
              │ 优化Worker      │
              └─────────────────┘
```

## 📦 安装部署

### 环境要求

- **Python**: 3.11+
- **PostgreSQL**: 15+
- **Redis**: 7+
- **Docker**: 20.10+ (推荐)
- **Docker Compose**: 2.0+

### 快速启动 (Docker)

1. **克隆项目**
```bash
git clone <repository-url>
cd automated-content-scheduling-service
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
- **API文档**: http://localhost:8095/docs
- **Flower监控**: http://localhost:5555
- **Redis管理**: http://localhost:8081 (admin/admin)

### 本地开发

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置数据库**
```bash
# 启动PostgreSQL和Redis
docker-compose -f docker-compose.dev.yml up postgres-scheduling redis-scheduling -d

# 运行数据库迁移
python -c "
import asyncio
from src.models import init_database
asyncio.run(init_database())
"
```

3. **启动服务**
```bash
# 启动主服务
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8095

# 启动Celery Worker (新终端)
celery -A src.scheduler.celery_app worker -Q scheduling,publishing,optimization -l info

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
| `PORT` | 服务端口 | 8095 |

### 数据库配置

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `DATABASE_URL` | PostgreSQL连接URL | postgresql+asyncpg://postgres:password@localhost:5436/historical_text_scheduling |
| `REDIS_URL` | Redis连接URL | redis://localhost:6382/3 |

### ML优化配置

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `ML_MODEL_TYPE` | 机器学习模型类型 | RandomForestRegressor |
| `ML_FEATURE_WINDOW_DAYS` | 特征时间窗口(天) | 30 |
| `ML_MIN_TRAINING_SAMPLES` | 最小训练样本数 | 100 |

### 外部服务配置

```bash
# 账号管理服务
ACCOUNT_MANAGEMENT_SERVICE_URL=http://localhost:8091

# 内容发布服务  
CONTENT_PUBLISHING_SERVICE_URL=http://localhost:8094

# 数据存储服务
STORAGE_SERVICE_URL=http://localhost:8002
```

## 📖 API 使用指南

### 创建调度任务

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8095/api/v1/scheduling/tasks",
        params={"user_id": 12345},
        json={
            "title": "重要产品发布公告",
            "content_id": "content_123",
            "content_body": "我们很高兴宣布新产品的正式发布...",
            "target_platforms": ["weibo", "wechat", "douyin"],
            "preferred_time": "2024-01-15T10:00:00Z",
            "task_type": "single",
            "priority": 8,
            "optimization_enabled": True,
            "strategy": "optimal_time"
        }
    )
    
    result = response.json()
    print(f"任务ID: {result['data']['task_id']}")
    print(f"优化后时间: {result['data']['scheduled_time']}")
```

### 获取分析报告

```python
# 获取用户仪表板
dashboard = await client.get(
    "http://localhost:8095/api/v1/analytics/dashboard",
    params={"user_id": 12345}
)

# 获取性能指标
performance = await client.get(
    "http://localhost:8095/api/v1/analytics/performance",
    params={
        "user_id": 12345,
        "platforms": "weibo,wechat",
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-31T23:59:59Z"
    }
)

# 获取冲突分析
conflicts = await client.get(
    "http://localhost:8095/api/v1/analytics/conflicts",
    params={"user_id": 12345, "days": 30}
)
```

### 批量操作

```python
# 批量创建任务
batch_tasks = [
    {
        "title": f"任务 {i}",
        "content_id": f"content_{i}",
        "content_body": f"内容 {i}",
        "target_platforms": ["weibo"],
        "priority": 5,
        "optimization_enabled": True
    }
    for i in range(1, 6)
]

response = await client.post(
    "http://localhost:8095/api/v1/scheduling/tasks/batch",
    params={"user_id": 12345},
    json=batch_tasks
)
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
pytest tests/unit/test_scheduling_service.py -v

# 运行集成测试
pytest tests/integration/ -v
```

### 测试数据库

```bash
# 启动测试数据库
docker-compose --profile test up postgres-test redis-test -d

# 设置测试环境变量
export ENVIRONMENT=testing
export DATABASE_URL=postgresql+asyncpg://test_user:test_password@localhost:5437/test_scheduling
```

## 📊 监控和运维

### 健康检查

```bash
# 基础健康检查
curl http://localhost:8095/health

# 详细系统状态
curl http://localhost:8095/api/v1/system/health

# 系统指标 (Prometheus格式)
curl http://localhost:8095/api/v1/system/metrics
```

### 性能监控

- **Flower界面**: http://localhost:5555 - Celery任务监控
- **Redis Commander**: http://localhost:8081 - Redis数据管理
- **API文档**: http://localhost:8095/docs - 交互式API文档

### 日志管理

```bash
# 查看服务日志
docker-compose -f docker-compose.dev.yml logs -f automated-content-scheduling-service

# 查看Worker日志
docker-compose -f docker-compose.dev.yml logs -f celery-worker-scheduling

# 获取系统日志
curl "http://localhost:8095/api/v1/system/logs?level=INFO&lines=100"
```

## 🔐 安全考虑

### 数据保护
- **敏感数据加密**: OAuth令牌和关键配置采用AES-256加密
- **输入验证**: 严格的API参数验证和SQL注入防护
- **访问控制**: 基于用户ID的细粒度权限管理

### API安全
- **速率限制**: 防止API滥用，默认1000请求/小时
- **CORS控制**: 可配置的跨域访问策略
- **错误处理**: 生产环境不暴露敏感错误信息

### 部署安全
```bash
# 生产环境必须设置安全密钥
export SECRET_KEY="your-256-bit-secret-key"

# 限制数据库访问
export DATABASE_URL="postgresql+asyncpg://limited_user:strong_password@db:5432/scheduling_db"

# 配置防火墙规则
# 仅允许必要端口对外访问
```

## 🚀 生产部署

### Docker部署

```bash
# 构建生产镜像
docker build -t scheduling-service:latest .

# 使用生产配置启动
docker-compose -f docker-compose.yml up -d
```

### Kubernetes部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: automated-content-scheduling
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scheduling-service
  template:
    metadata:
      labels:
        app: scheduling-service
    spec:
      containers:
      - name: scheduling-service
        image: scheduling-service:latest
        ports:
        - containerPort: 8095
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8095
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8095
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 负载均衡配置 (Nginx)

```nginx
upstream scheduling_service {
    server 127.0.0.1:8095 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8096 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    location /scheduling/ {
        proxy_pass http://scheduling_service/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # 缓存设置
        proxy_cache_bypass $http_upgrade;
        proxy_cache_valid 200 5m;
    }
}
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

### 提交规范

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 其他修改

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 🆘 支持和反馈

- **问题反馈**: [GitHub Issues](https://github.com/your-org/automated-content-scheduling-service/issues)
- **功能请求**: [Feature Requests](https://github.com/your-org/automated-content-scheduling-service/discussions)
- **安全漏洞**: security@yourdomain.com

## 📚 相关文档

- [API文档](http://localhost:8095/docs) - 完整的REST API文档
- [架构设计](docs/architecture.md) - 系统架构详细说明
- [部署指南](docs/deployment.md) - 生产环境部署指导
- [开发指南](docs/development.md) - 本地开发环境搭建
- [故障排查](docs/troubleshooting.md) - 常见问题解决方案

## 📈 路线图

### v2.0 (计划中)
- [ ] 更多社交媒体平台支持 (Instagram, LinkedIn, Twitter)
- [ ] 实时数据推送和WebSocket支持
- [ ] 高级A/B测试功能
- [ ] 可视化调度日历界面

### v2.1 (未来)
- [ ] AI内容生成集成
- [ ] 多语言内容支持
- [ ] 高级用户行为预测
- [ ] 企业级权限管理系统

---

**版本**: 1.0.0  
**更新时间**: 2024-01-11  
**维护者**: 历史文本优化项目团队