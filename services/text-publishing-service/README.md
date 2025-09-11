# 文本发布服务

多平台内容发布服务，支持微博、微信、抖音、今日头条、百家号等主流社交媒体平台的统一发布管理。

## 🚀 功能特性

- **多平台支持**: 支持5个主流发布平台的统一接口
- **智能调度**: 支持立即发布和定时发布
- **负载均衡**: 智能账号选择和配额管理
- **实时监控**: 任务状态实时跟踪和进度显示
- **重试机制**: 自动重试失败的发布任务
- **统计分析**: 详细的发布数据统计和分析

## 📋 支持的平台

| 平台 | 状态 | 内容类型 | 限制 |
|------|------|----------|------|
| 微博 | ✅ 完整支持 | 文本+图片 | 140字符 |
| 微信公众号 | ✅ 基础支持 | 图文消息 | 20000字符 |
| 抖音 | ✅ 基础支持 | 短视频+文本 | 2200字符 |
| 今日头条 | ✅ 基础支持 | 文章+图片 | 5000字符 |
| 百家号 | ✅ 基础支持 | 文章+视频 | 8000字符 |

## 🏗️ 架构设计

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vue3 前端     │    │   FastAPI 后端  │    │   平台适配器     │
│                 │    │                 │    │                 │
│ • 发布管理      │◄──►│ • 任务管理      │◄──►│ • 微博适配器     │
│ • 状态监控      │    │ • 账号管理      │    │ • 微信适配器     │
│ • 统计分析      │    │ • 统计分析      │    │ • 抖音适配器     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲
                                │
                    ┌─────────────────┐
                    │   数据存储      │
                    │                 │
                    │ • PostgreSQL    │
                    │ • Redis         │
                    │ • RabbitMQ      │
                    └─────────────────┘
```

## 🛠️ 技术栈

### 后端
- **框架**: FastAPI 0.104+
- **语言**: Python 3.11+
- **数据库**: PostgreSQL 15+
- **缓存**: Redis 7+
- **任务队列**: Celery + RabbitMQ
- **ORM**: SQLAlchemy 2.0 (异步)

### 前端
- **框架**: Vue 3 + TypeScript
- **状态管理**: Pinia
- **UI组件**: Element Plus
- **构建工具**: Vite

### 基础设施
- **容器化**: Docker + Docker Compose
- **编排**: Kubernetes
- **监控**: Prometheus + Grafana
- **日志**: ELK Stack

## 📦 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd Historical Text Project/services/text-publishing-service

# 复制环境变量配置
cp .env.example .env

# 编辑配置文件，设置数据库连接等
vim .env
```

### 2. 使用Docker Compose启动（推荐）

```bash
# 启动所有服务
docker-compose -f ../../docker-compose.dev.yml up -d

# 查看服务状态
docker-compose -f ../../docker-compose.dev.yml ps

# 查看日志
docker-compose -f ../../docker-compose.dev.yml logs -f text-publishing-service
```

### 3. 本地开发模式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动数据库和Redis（使用Docker）
docker-compose -f ../../docker-compose.dev.yml up -d postgres redis rabbitmq

# 运行数据库迁移
alembic upgrade head

# 启动服务
python -m src.main
```

### 4. 初始化数据

```bash
# 访问开发数据初始化接口（仅开发模式）
curl -X POST http://localhost:8080/dev/init-data
```

## 🔧 配置说明

### 环境变量

| 变量 | 说明 | 默认值 | 必填 |
|------|------|--------|------|
| `DATABASE_URL` | PostgreSQL连接字符串 | - | ✅ |
| `REDIS_URL` | Redis连接字符串 | - | ✅ |
| `WEIBO_API_KEY` | 微博API密钥 | - | 可选 |
| `WECHAT_APP_ID` | 微信APP ID | - | 可选 |
| `DEBUG` | 调试模式 | `false` | ❌ |

### 平台API配置

每个平台都需要相应的API凭据：

```bash
# 微博配置
WEIBO_API_KEY=your_api_key
WEIBO_API_SECRET=your_api_secret

# 微信配置
WECHAT_APP_ID=your_app_id
WECHAT_APP_SECRET=your_app_secret
```

## 📚 API文档

服务启动后，可以通过以下地址访问API文档：

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### 主要接口

#### 创建发布任务
```http
POST /api/v1/publish
Content-Type: application/json

{
    "content": "发布内容",
    "platforms": ["weibo", "douyin"],
    "title": "标题",
    "scheduled_at": "2024-01-01T12:00:00Z"
}
```

#### 查询任务状态
```http
GET /api/v1/publish/tasks/{task_uuid}/status
```

#### 获取任务列表
```http
GET /api/v1/publish/tasks?status=published&page=1&page_size=20
```

#### 获取统计数据
```http
GET /api/v1/publish/statistics?start_date=2024-01-01&end_date=2024-01-31
```

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_api.py -v

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term
```

### 测试覆盖率

目标：80%以上的代码覆盖率

```bash
# 查看覆盖率报告
open htmlcov/index.html
```

## 🚀 部署

### Docker部署

```bash
# 构建镜像
docker build -t text-publishing-service .

# 运行容器
docker run -d \
  --name text-publishing-service \
  -p 8080:8080 \
  --env-file .env \
  text-publishing-service
```

### Kubernetes部署

```bash
# 部署到K8s集群
kubectl apply -f k8s/
```

## 📊 监控

### 健康检查

- **基础健康**: `GET /health/`
- **就绪检查**: `GET /health/ready`
- **Prometheus指标**: `GET /health/metrics`

### 日志

服务使用结构化JSON日志：

```bash
# 查看服务日志
docker logs text-publishing-service

# 实时监控日志
docker logs -f text-publishing-service
```

## 🔒 安全

### API认证

服务使用JWT Bearer Token认证：

```http
Authorization: Bearer <your-jwt-token>
```

### 数据加密

- 敏感的账号凭据在数据库中加密存储
- 所有API通信建议使用HTTPS
- 支持CORS跨域访问控制

## 🤝 开发指南

### 项目结构

```
src/
├── config/          # 配置管理
├── models/          # 数据模型
├── services/        # 业务逻辑服务
├── controllers/     # API控制器
├── adapters/        # 平台适配器
├── utils/           # 工具函数
└── main.py         # 应用入口

tests/              # 测试文件
migrations/         # 数据库迁移
docs/              # 文档
```

### 添加新平台

1. 创建平台适配器：`src/adapters/new_platform_adapter.py`
2. 继承`PlatformAdapter`基类
3. 实现认证和发布方法
4. 在配置中添加平台信息
5. 编写测试用例

### 代码规范

```bash
# 格式化代码
black src/ tests/
isort src/ tests/

# 代码检查
flake8 src/ tests/
mypy src/
```

## 🐛 问题排查

### 常见问题

1. **数据库连接失败**
   ```bash
   # 检查数据库服务状态
   docker-compose ps postgres
   # 检查连接字符串配置
   echo $DATABASE_URL
   ```

2. **Redis连接失败**
   ```bash
   # 检查Redis服务
   docker-compose ps redis
   # 测试连接
   redis-cli -u $REDIS_URL ping
   ```

3. **平台API认证失败**
   - 检查API密钥配置
   - 确认账号权限和配额

### 调试模式

```bash
# 启用调试模式
export DEBUG=true
python -m src.main
```

## 📝 更新日志

### v1.0.0 (2024-01-01)
- ✨ 初始版本发布
- ✅ 支持5个主流发布平台
- 🚀 多平台统一发布接口
- 📊 实时状态监控和统计

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建feature分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 提交Pull Request

## 📞 支持

如果您有任何问题，请通过以下方式联系：

- 提交Issue：[GitHub Issues](https://github.com/your-repo/issues)
- 邮件：support@yourcompany.com

---

**📈 为历史文本优化项目提供强大的多平台发布能力！**