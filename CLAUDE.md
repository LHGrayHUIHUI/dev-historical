# CLAUDE.md

本文件为Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概览

历史文本优化项目是一个基于AI技术的智能文本处理平台，专注于历史文献的数字化、优化和智能分析。项目采用现代化的微服务架构，通过Vue3统一管理界面实现可视化的内容管理和发布系统。

### 技术架构
- **微服务**: FastAPI + Python 3.11+ 后端服务
- **前端**: Vue 3 + TypeScript + Vite (规划中)
- **存储**: MongoDB + PostgreSQL + Redis + MinIO
- **基础设施**: Docker + Kubernetes + Kong API网关
- **监控**: Prometheus + Grafana + ELK栈 + Jaeger

## 核心命令

### 开发环境

#### Docker 快速启动 (推荐)
```bash
# 启动开发环境
docker-compose -f docker-compose.dev.yml up -d

# 启动生产环境  
docker-compose -f docker-compose.production.yml up -d

# 启动监控栈
docker-compose -f docker-compose.monitoring.yml up -d

# 检查服务状态
docker-compose -f docker-compose.dev.yml ps

# 查看日志
docker-compose -f docker-compose.dev.yml logs -f
```

#### 独立服务启动
```bash
# 文件处理服务 (端口 8001)
cd services/file-processor
pip install -r requirements.txt
python -m src.main

# 存储服务 (端口 8002)
cd services/storage-service
pip install -r requirements.txt  
python -m src.main
```

### 测试

#### 运行测试
```bash
# 运行所有测试并生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term

# 运行特定测试文件
pytest tests/unit/test_monitoring.py -v
pytest tests/integration/ -v

# 运行带特定标记的测试
pytest -m "unit"
pytest -m "integration"
```

#### 测试结构
- `tests/unit/` - 单个组件的单元测试
- `tests/integration/` - 服务交互的集成测试
- `test-results/` - 按时间戳组织的测试输出
- 覆盖率报告生成在 `htmlcov/`

### 代码质量

#### 代码检查和格式化
```bash
# 格式化代码
black src/ tests/
isort src/ tests/

# 代码检查
flake8 src/ tests/
mypy src/

# 所有质量检查
black src/ && isort src/ && flake8 src/ && mypy src/
```

#### 提交规范
项目遵循 Conventional Commits 规范：
- `feat(epic-1)`: Epic 1 相关功能
- `feat(epic-2)`: Epic 2 相关功能  
- `fix(service)`: 微服务 bug 修复
- `docs`: 文档更新

### Docker 操作

#### 构建镜像
```bash
# 构建优化的生产镜像
./scripts/build-optimized-images.sh

# 构建单个服务
docker build -f services/file-processor/Dockerfile -t file-processor-service services/file-processor
docker build -f services/storage-service/Dockerfile -t storage-service-service services/storage-service
```

#### Kubernetes 部署
```bash
# 创建命名空间
kubectl create namespace historical-text

# 部署基础设施
kubectl apply -f infrastructure/kubernetes/

# 部署服务
kubectl apply -f infrastructure/kubernetes/services/
```

## 架构深度解析

### 核心服务架构

项目遵循清晰的微服务分离原则：

#### file-processor (端口 8001)
- **纯文件处理服务** - 无数据库依赖
- **无状态设计** - 所有结果通过API返回
- **支持格式**: PDF、Word、图片(OCR)、HTML、纯文本
- **核心功能**: 文本提取、文件验证、病毒扫描
- **服务定位**: 专业化文件处理算法

#### storage-service (端口 8002) 
- **统一存储服务** - 管理所有数据库和存储系统
- **业务逻辑中心** - 所有CRUD操作和数据管理  
- **数据库**: MongoDB + PostgreSQL + Redis + MinIO + RabbitMQ
- **核心功能**: 内容管理、文件存储、批量处理
- **服务定位**: 所有数据操作的统一入口

#### 服务交互模式
```
前端/API客户端 → storage-service (8002) → file-processor (8001)
                           ↓
                   所有存储系统
                   (MongoDB, PostgreSQL, Redis, MinIO, RabbitMQ)
```

### 数据库策略

#### MongoDB (端口 27018)
- 历史文本内容和业务数据
- 非结构化内容的文档存储
- 集合: `contents`, `datasets`, `processing_logs`

#### PostgreSQL (端口 5433)  
- 文件元数据和处理记录
- 关系数据和审计日志
- 表: `files`, `processing_jobs`, `user_sessions`

#### Redis (端口 6380)
- 缓存层和会话管理
- 任务队列和实时数据
- 键: `cache:*`, `session:*`, `queue:*`

#### MinIO (端口 9001/9002)
- 所有文件类型的对象存储
- S3兼容的文件操作API
- 存储桶: `historical-text-files`, `processed-content`

### 配置管理

#### 环境文件结构
```bash
# 开发环境
.env                      # 本地开发设置
.env.docker              # Docker特定配置
services/*/requirements.txt  # 服务特定依赖

# 生产环境  
docker-compose.production.yml  # 生产容器配置
infrastructure/kubernetes/     # K8s部署配置
```

#### 核心配置区域
- **数据库连接**: 每个服务有专用连接配置
- **文件处理限制**: 最大文件大小、允许格式、处理超时  
- **监控设置**: Prometheus指标、Grafana仪表板、告警规则
- **安全设置**: JWT令牌、API速率限制、CORS策略

### BMAD框架集成

项目集成了BMAD v4.42.1框架，包含10个专业代理：

#### 可用代理 (使用 `/BMad:agents:` 前缀)
- `dev` - 全栈开发工程师，负责实现  
- `architect` - 系统架构设计
- `pm` - 产品经理，负责规划
- `qa` - 质量保证和测试
- `analyst` - 业务需求分析
- `po` - 产品负责人，负责策略
- `sm` - 敏捷教练，负责项目管理  
- `ux-expert` - 用户体验设计
- `bmad-master` - 整体协调
- `bmad-orchestrator` - 工作流编排

#### 代理工作流
1. **Epic规划**: PM代理按Epic优先级规划开发
2. **Story分析**: Analyst代理详细分析业务需求  
3. **架构设计**: Architect定义API和数据模型
4. **开发实现**: Dev代理实现微服务
5. **质量保证**: QA代理运行集成测试
6. **部署验证**: 生产前在测试环境验证

## 开发工作流

### 基于Epic的开发

#### 当前状态 (Epic 1 完成 ✅)
- **Epic 1**: 微服务基础设施和数据获取 (100% 完成)
  - Story 1.1: 微服务架构 ✅
  - Story 1.2: 数据获取服务 ✅  
  - Story 1.3: 数据采集与存储 ✅
  - Story 1.4: 系统监控与日志 ✅

#### 下一步优先级  
- **Epic 2**: 数据处理和智能分类 (0% 完成)
- **Epic 3**: AI大模型服务和内容优化 (0% 完成)  
- **Epic 4**: 发布管理和Vue3统一界面 (0% 完成)

### 服务开发模式

#### FastAPI 微服务结构
```python
# 标准服务组织结构
services/{service-name}/
├── src/
│   ├── main.py              # FastAPI 应用入口
│   ├── config/settings.py   # Pydantic 配置  
│   ├── models/              # SQLAlchemy/Pydantic 模型
│   ├── controllers/         # FastAPI 路由处理器
│   ├── services/            # 业务逻辑层
│   ├── repositories/        # 数据访问层
│   └── utils/               # 辅助工具
├── tests/
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试  
├── Dockerfile             # 容器定义
├── requirements.txt       # Python 依赖
└── README.md             # 服务文档
```

#### API设计约定
- **REST模式**: 标准HTTP方法和状态码
- **响应格式**: 统一的JSON结构和 `BaseResponse`
- **错误处理**: 结构化异常和正确的HTTP代码
- **文档**: `/docs` 路径下的自动OpenAPI/Swagger文档
- **版本控制**: API路径使用 `/api/v1/` 前缀

### 测试策略

#### 测试组织  
- **单元测试**: 测试单个函数和类
- **集成测试**: 测试服务交互和数据库操作
- **测试数据**: 使用工厂和夹具确保测试数据一致性
- **覆盖率目标**: 要求80%以上的代码覆盖率

#### 测试执行模式
```python
# conftest.py中的Pytest配置
# pytest-asyncio的异步测试支持
# 测试隔离的数据库夹具
# 适当模拟外部依赖
```

### 监控和可观测性

#### 指标收集 (Prometheus)
- **HTTP请求**: 响应时间、状态码、请求速率
- **业务指标**: 文件处理计数、用户操作、系统健康状态
- **基础设施**: 每个服务的CPU、内存、磁盘使用情况
- **数据库指标**: 连接池、查询性能

#### 日志策略  
- **结构化日志**: JSON格式便于解析
- **日志级别**: 开发用DEBUG，生产用INFO/WARN/ERROR
- **集中收集**: ELK栈聚合所有服务日志
- **关联ID**: 跨服务边界跟踪请求

#### 服务监控端点
```http
GET /health       # 基本健康检查
GET /ready        # K8s就绪探针
GET /metrics      # Prometheus指标端点  
GET /info         # 服务版本和构建信息
```

## 文件组织要求

### 测试结果管理
- **位置**: 将所有测试输出存储在 `test-results/` 中，按时间戳组织
- **结构**: 每次测试运行使用 `test-results/YYYY-MM-DD-HHMMSS/` 
- **内容**: 覆盖率报告、测试日志、性能指标

### 变更文档  
- **位置**: 使用时间戳条目更新 `changelogs/` 
- **格式**: 遵循语义版本控制和常规变更日志格式
- **范围**: 记录所有重大变更和架构决策

### 文档更新
- **仪表板**: 与 `DEVELOPMENT_DASHBOARD.md` 同步所有变更
- **README**: 保持根目录 `README.md` 与项目状态同步
- **服务文档**: API变更时更新各服务的README文件

## 常用模式

### 数据库连接管理
```python
# 具有适当生命周期的异步数据库会话
# 连接池配置
# 基于环境的连接字符串
# 优雅关闭处理
```

### 错误处理
```python  
# 自定义异常层次结构
# 结构化错误响应
# 正确的HTTP状态码
# 集中错误日志记录
```

### 配置加载
```python
# Pydantic Settings 用于类型安全配置
# 环境变量验证  
# 开发与生产配置
# 密钥管理实践
```

### 异步处理
```python
# 用于后台任务的RabbitMQ消息队列
# Celery风格的任务管理
# 长期运行操作的进度跟踪
# 全面使用适当的async/await模式
```

此代码库代表了一个生产就绪的微服务平台，具有全面的测试、监控和部署基础设施。架构优先考虑关注点的清晰分离、可扩展性和可维护性。