# 多平台账号管理服务 (Multi-Platform Account Management Service)

统一管理多个社交媒体平台账号的微服务，支持OAuth认证、数据同步、权限控制等功能。

## 🚀 功能特性

### 核心功能
- **多平台支持**: 支持微博、微信、抖音、头条、百家号等主流平台
- **OAuth认证**: 完整的OAuth 2.0认证流程，安全可靠
- **账号管理**: 账号的添加、更新、删除、查询和统计
- **数据同步**: 支持账号信息、统计数据、发布内容的定时同步
- **权限控制**: 细粒度的账号访问权限管理
- **安全加密**: 敏感数据采用AES-256加密存储

### 技术特性
- **异步处理**: 基于FastAPI的高性能异步Web框架
- **数据库**: PostgreSQL关系数据库，支持事务和ACID特性
- **缓存**: Redis缓存提升响应速度和降低数据库压力
- **容器化**: 完整的Docker和docker-compose配置
- **API文档**: 自动生成的OpenAPI/Swagger文档
- **监控**: 内置健康检查、性能指标和API统计

## 🏗️ 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   前端应用      │    │   API网关        │    │  多平台账号管理服务  │
│                 │◄──►│                  │◄──►│                     │
│  Vue3/React     │    │  Kong/Nginx      │    │     FastAPI         │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          │
                       ┌──────────────────────────────────┼──────────────────────────┐
                       │                                  │                          │
              ┌────────▼────────┐              ┌────────▼────────┐      ┌─────────▼─────────┐
              │   PostgreSQL    │              │      Redis      │      │  社交媒体平台API  │
              │                 │              │                 │      │                   │
              │  账号、权限、    │              │  缓存、会话、    │      │ 微博、微信、抖音、 │
              │  同步日志等      │              │  队列状态等      │      │ 头条、百家号等    │
              └─────────────────┘              └─────────────────┘      └───────────────────┘
```

## 📦 安装部署

### 环境要求

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (可选)

### 快速启动 (Docker)

1. **克隆项目**
```bash
git clone <repository-url>
cd multi-platform-account-management-service
```

2. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，配置数据库连接、API密钥等
```

3. **启动服务 (开发环境)**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

4. **启动服务 (生产环境)**
```bash
docker-compose up -d
```

5. **访问服务**
- API文档: http://localhost:8091/docs
- ReDoc文档: http://localhost:8091/redoc
- 健康检查: http://localhost:8091/health

### 本地开发

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置数据库**
```bash
# 启动PostgreSQL和Redis
docker-compose -f docker-compose.dev.yml up postgres redis -d

# 运行数据库迁移
python -m alembic upgrade head
```

3. **启动开发服务器**
```bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8091
```

## 🔧 配置说明

### 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `APP_NAME` | 应用名称 | Multi-Platform Account Management Service |
| `DEBUG` | 调试模式 | false |
| `HOST` | 服务监听地址 | 0.0.0.0 |
| `PORT` | 服务端口 | 8091 |
| `DATABASE_URL` | PostgreSQL连接URL | postgresql://postgres:password@localhost:5433/historical_text_accounts |
| `REDIS_URL` | Redis连接URL | redis://localhost:6379/0 |
| `ENCRYPTION_KEY` | 数据加密密钥 | 32字符长度密钥 |
| `OAUTH_CALLBACK_BASE_URL` | OAuth回调基础URL | http://localhost:8091/api/v1/oauth/callback |

### 平台配置

在 `src/config/settings.py` 中配置各个社交媒体平台的OAuth信息：

```python
WEIBO_CONFIG = PlatformConfig(
    client_id="your_weibo_app_key",
    client_secret="your_weibo_app_secret",
    authorize_url="https://api.weibo.com/oauth2/authorize",
    token_url="https://api.weibo.com/oauth2/access_token",
    scope="read,write"
)
```

## 📖 API 使用指南

### 认证流程

1. **获取授权URL**
```bash
GET /api/v1/oauth/authorize/weibo?user_id=123
```

2. **用户授权后处理回调**
```bash
POST /api/v1/oauth/callback/weibo
{
    "code": "authorization_code",
    "state": "state_code"
}
```

3. **添加账号**
```bash
POST /api/v1/accounts/?user_id=123
{
    "platform_name": "weibo",
    "auth_code": "authorization_code"
}
```

### 账号管理

```bash
# 获取账号列表
GET /api/v1/accounts/?user_id=123&page=1&size=20

# 获取账号详情
GET /api/v1/accounts/456?user_id=123

# 更新账号信息
PUT /api/v1/accounts/456?user_id=123
{
    "display_name": "新的显示名称",
    "bio": "更新的个人简介"
}

# 删除账号
DELETE /api/v1/accounts/456?user_id=123
```

### 数据同步

```bash
# 同步单个账号
POST /api/v1/sync/account/456
{
    "account_id": 456,
    "sync_types": ["profile", "stats", "posts"],
    "force": false
}

# 批量同步
POST /api/v1/sync/batch
{
    "account_ids": [456, 789],
    "sync_types": ["profile", "stats"]
}

# 获取同步状态
GET /api/v1/sync/account/456/status
```

## 🧪 测试

### 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio pytest-cov

# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term

# 运行特定测试文件
pytest tests/test_account_service.py -v
```

### 测试数据库

测试使用独立的测试数据库，配置如下：

```bash
# 启动测试数据库
docker-compose --profile with-test-db up postgres-test -d

# 设置测试环境变量
export TEST_DATABASE_URL=postgresql://test_user:test_password@localhost:5435/historical_text_accounts_test
```

## 📊 监控和运维

### 健康检查

```bash
# 基础健康检查
GET /health

# 就绪检查 (Kubernetes探针)
GET /ready

# 系统状态详情
GET /api/v1/system/status
```

### 性能监控

```bash
# 获取性能指标 (Prometheus兼容)
GET /api/v1/system/metrics

# API使用统计
GET /api/v1/system/api-stats?days=7

# 同步任务统计
GET /api/v1/sync/statistics?days=7
```

### 日志管理

```bash
# 获取系统日志
GET /api/v1/system/logs?level=INFO&lines=100

# Docker日志
docker-compose logs -f multi-platform-account-management-service
```

## 🔐 安全考虑

### 数据保护
- **加密存储**: OAuth令牌和敏感信息使用AES-256加密
- **传输安全**: 支持HTTPS和TLS加密传输
- **访问控制**: 基于用户和权限的细粒度访问控制
- **审计日志**: 记录所有重要操作的审计日志

### API安全
- **速率限制**: 防止API滥用和DDoS攻击
- **输入验证**: 严格的输入参数验证和清理
- **CORS控制**: 配置合适的跨域访问策略
- **错误处理**: 不暴露敏感的错误信息

## 🚀 部署到生产环境

### Docker部署

```bash
# 构建生产镜像
docker build -t account-management-service:latest .

# 使用生产配置启动
docker-compose -f docker-compose.yml up -d
```

### Kubernetes部署

```yaml
# 见 k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: account-management-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: account-management-service
  template:
    metadata:
      labels:
        app: account-management-service
    spec:
      containers:
      - name: account-management-service
        image: account-management-service:latest
        ports:
        - containerPort: 8091
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

### 反向代理配置 (Nginx)

```nginx
upstream account_management {
    server 127.0.0.1:8091;
    server 127.0.0.1:8092 backup;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    location /account-management/ {
        proxy_pass http://account_management/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🤝 贡献指南

### 开发流程

1. Fork项目并创建功能分支
2. 编写代码和测试
3. 确保所有测试通过
4. 提交Pull Request

### 代码规范

- 遵循PEP 8 Python编码规范
- 使用类型注解 (Type Hints)
- 编写详细的文档字符串
- 单元测试覆盖率不低于80%

### 提交规范

- feat: 新功能
- fix: Bug修复
- docs: 文档更新
- style: 代码格式化
- refactor: 重构
- test: 测试相关
- chore: 其他修改

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 🆘 支持和反馈

- **问题反馈**: 请提交 [GitHub Issue](https://github.com/yourorg/multi-platform-account-management-service/issues)
- **功能请求**: 请提交 [Feature Request](https://github.com/yourorg/multi-platform-account-management-service/issues/new?template=feature_request.md)
- **安全漏洞**: 请发送邮件至 security@yourdomain.com

## 📚 相关文档

- [API文档](http://localhost:8091/docs) - 在线API文档
- [架构设计](docs/architecture.md) - 系统架构设计文档
- [部署指南](docs/deployment.md) - 详细部署指南
- [平台接入](docs/platform-integration.md) - 新平台接入指南
- [故障排查](docs/troubleshooting.md) - 常见问题解决方案