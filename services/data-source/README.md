# 数据源服务 (Data Source Service)

历史文本项目的数据获取微服务，负责从多个平台爬取内容、代理管理和数据处理。

## 🚀 快速开始

### 环境要求

- Python 3.9+
- MongoDB 5.0+
- Redis 7.0+
- Docker & Docker Compose (可选)

### 本地开发

1. **克隆代码并安装依赖**
   ```bash
   cd services/data-source
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，配置数据库连接等信息
   ```

3. **启动数据库服务**
   ```bash
   # 使用Docker Compose启动依赖服务
   docker-compose up -d mongo redis consul
   ```

4. **运行服务**
   ```bash
   python -m src.main
   ```

5. **访问API文档**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker部署

```bash
# 构建镜像
docker build -t data-source-service .

# 使用Docker Compose运行完整服务栈
docker-compose up -d

# 检查服务状态
docker-compose ps
```

### Kubernetes部署

```bash
# 创建命名空间和配置
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# 部署服务
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 检查部署状态
kubectl get pods -n data-source
```

## 📚 API 接口

### 爬虫管理
- `POST /api/v1/crawlers/` - 创建爬虫任务
- `POST /api/v1/crawlers/{task_id}/start` - 启动任务
- `POST /api/v1/crawlers/{task_id}/stop` - 停止任务
- `GET /api/v1/crawlers/{task_id}/status` - 获取任务状态
- `GET /api/v1/crawlers/` - 获取任务列表
- `GET /api/v1/crawlers/statistics` - 获取统计信息

### 内容管理
- `POST /api/v1/content/` - 手动添加内容
- `POST /api/v1/content/batch` - 批量添加内容
- `POST /api/v1/content/upload` - 文件导入内容
- `GET /api/v1/content/` - 获取内容列表
- `GET /api/v1/content/{content_id}` - 获取内容详情
- `PUT /api/v1/content/{content_id}` - 更新内容
- `DELETE /api/v1/content/{content_id}` - 删除内容
- `GET /api/v1/content/statistics/overview` - 获取内容统计

### 代理管理
- `GET /api/v1/proxy/` - 获取代理列表
- `GET /api/v1/proxy/active` - 获取可用代理
- `GET /api/v1/proxy/best` - 获取最佳代理
- `POST /api/v1/proxy/test` - 测试代理
- `POST /api/v1/proxy/refresh` - 刷新代理列表
- `GET /api/v1/proxy/statistics` - 获取代理统计

### 系统接口
- `GET /` - 服务信息
- `GET /health` - 健康检查
- `GET /info` - 详细信息

## 🏗️ 架构设计

### 核心组件

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   Web Interface │    │   Monitoring    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   Data Source Service     │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │  Crawler Manager    │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  Proxy Manager      │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  Content Processor  │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  Database Manager   │  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │        Data Layer         │
                    │  ┌─────────┐ ┌─────────┐  │
                    │  │ MongoDB │ │  Redis  │  │
                    │  └─────────┘ └─────────┘  │
                    └───────────────────────────┘
```

### 支持平台

- **今日头条** - 新闻资讯爬取
- **百家号** - 自媒体内容爬取  
- **小红书** - 生活方式内容爬取
- **手动添加** - 支持单个/批量/文件导入
- **扩展接口** - 可插拔的数据源扩展

### 反封禁策略

- **IP代理池** - 支持免费和付费代理
- **User-Agent轮换** - 模拟不同浏览器
- **请求频率控制** - 智能延迟策略
- **验证码识别** - 自动处理验证码
- **账号池管理** - 多账号轮换机制

## 🛠️ 开发指南

### 项目结构

```
services/data-source/
├── src/
│   ├── api/                 # API路由层
│   │   ├── crawler.py      # 爬虫管理接口
│   │   ├── content.py      # 内容管理接口
│   │   └── proxy.py        # 代理管理接口
│   ├── crawler/            # 爬虫模块
│   │   └── crawler_manager.py
│   ├── proxy/              # 代理模块
│   │   └── proxy_manager.py
│   ├── models/             # 数据模型
│   │   └── content.py
│   ├── database/           # 数据库模块
│   │   └── database.py
│   ├── config/             # 配置模块
│   │   └── settings.py
│   └── main.py             # 应用入口
├── tests/                  # 测试代码
│   ├── unit/              # 单元测试
│   └── integration/       # 集成测试
├── k8s/                   # Kubernetes配置
├── docker/                # Docker配置文件
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 添加新平台爬虫

1. **创建爬虫类**
   ```python
   class NewPlatformCrawler(BaseCrawler):
       async def crawl(self):
           # 实现具体爬取逻辑
           pass
   ```

2. **注册爬虫**
   ```python
   # 在CrawlerManager中添加
   self.crawler_classes[ContentSource.NEW_PLATFORM] = NewPlatformCrawler
   ```

3. **更新数据模型**
   ```python
   # 在ContentSource枚举中添加
   NEW_PLATFORM = "new_platform"
   ```

### 配置管理

使用Pydantic Settings进行配置管理，支持：
- 环境变量覆盖
- 嵌套配置结构
- 类型验证
- 默认值设置

### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_crawler_manager.py

# 运行集成测试
pytest tests/integration/

# 生成测试覆盖率报告
pytest --cov=src --cov-report=html
```

## 📊 监控运维

### 健康检查

```bash
# 基础健康检查
curl http://localhost:8000/health

# 详细服务信息
curl http://localhost:8000/info
```

### 日志管理

服务使用结构化日志，支持：
- 多级别日志 (DEBUG/INFO/WARNING/ERROR)
- JSON格式输出 (生产环境)
- 日志轮转和保留策略
- ELK Stack集成

### 指标监控

- **Prometheus指标** - 通过 `/metrics` 端点暴露
- **自定义指标** - 爬虫成功率、代理可用性等
- **Grafana看板** - 可视化监控面板

### 服务发现

- **Consul集成** - 自动服务注册和发现
- **健康检查** - 定期检查服务状态
- **负载均衡** - 支持多实例部署

## 🔧 配置说明

### 环境变量配置

主要配置项说明：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `SERVICE_ENVIRONMENT` | 运行环境 | `development` |
| `SERVICE_PORT` | 服务端口 | `8000` |
| `CRAWLER_MAX_CONCURRENT_CRAWLERS` | 最大并发爬虫数 | `10` |
| `CRAWLER_ENABLE_PROXY` | 是否启用代理 | `false` |
| `DB_MONGODB_URL` | MongoDB连接URL | `mongodb://localhost:27017` |
| `DB_REDIS_URL` | Redis连接URL | `redis://localhost:6379` |
| `LOG_LOG_LEVEL` | 日志级别 | `INFO` |

完整配置参考 `.env.example` 文件。

### 生产环境配置

生产环境建议配置：
- 启用代理池 (`CRAWLER_ENABLE_PROXY=true`)
- 增加并发数 (`CRAWLER_MAX_CONCURRENT_CRAWLERS=20`)
- 使用JSON日志 (`LOG_JSON_LOGS=true`)
- 启用监控 (`MONITOR_ENABLE_METRICS=true`)
- 配置安全密钥 (`SERVICE_SECRET_KEY`)

## 🚨 故障排除

### 常见问题

1. **MongoDB连接失败**
   - 检查MongoDB服务状态
   - 确认连接URL和认证信息
   - 检查网络连接

2. **代理获取失败**
   - 检查代理源可用性
   - 验证代理配置
   - 查看代理测试日志

3. **爬虫任务卡住**
   - 检查目标网站状态
   - 查看网络连接
   - 检查反封禁策略

4. **内存使用过高**
   - 调整并发爬虫数量
   - 检查数据处理逻辑
   - 监控MongoDB连接池

### 日志分析

```bash
# 查看服务日志
docker-compose logs data-source-service

# 过滤错误日志
docker-compose logs data-source-service | grep ERROR

# 实时监控日志
docker-compose logs -f data-source-service
```

### 性能优化

1. **数据库优化**
   - 创建合适的索引
   - 使用连接池
   - 定期清理过期数据

2. **爬虫优化**
   - 调整请求间隔
   - 使用高质量代理
   - 实现增量爬取

3. **缓存策略**
   - 使用Redis缓存热点数据
   - 实现内容去重缓存
   - 缓存代理测试结果

## 📝 更新日志

### v1.0.0 (2024-01-XX)
- ✅ 完成多平台爬虫支持
- ✅ 实现代理管理系统
- ✅ 添加内容管理接口
- ✅ 支持Docker和Kubernetes部署
- ✅ 完成监控和日志系统
- ✅ 添加完整的测试覆盖

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../../LICENSE) 文件。

## 🤝 贡献

欢迎提交 Pull Request 或创建 Issue！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request