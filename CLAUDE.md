# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

历史文本优化项目是一个基于AI技术的智能文本处理平台，专注于历史文献的数字化、优化和智能分析。项目采用微服务架构，通过Vue3统一管理界面实现可视化的内容管理和发布系统。

## 快速开始指南

### 新Claude实例开发流程
1. **理解项目状态**: 当前Epic 1进展75%，已完成3个Story，正在开发Story 1.4
2. **启动开发环境**: `docker-compose -f docker-compose.dev.yml up -d`
3. **验证服务**: 访问 http://localhost:8001/docs (数据源服务) 和 http://localhost:8003/docs (数据采集服务)
4. **运行测试**: `python -m pytest tests/unit/ -v` 确保环境正常
5. **开始开发**: 使用 `/BMad:agents:dev` 激活开发代理，执行 `*develop-story` 开发当前Story

### 关键开发原则
- **文档优先**: 每个功能都有详细的Story文档，先读文档再编码
- **测试驱动**: 编写代码的同时编写测试，确保功能可靠性
- **中文注释**: 所有代码注释必须使用中文，注释密度>30%
- **微服务架构**: 每个服务独立开发、测试、部署

## 架构概览

这是一个基于文档驱动开发的微服务项目，当前已从设计阶段进入实现阶段，已有3个核心服务投入生产：

### 微服务架构
- **数据源服务**: 多平台内容获取（今日头条、百家号、小红书等）
- **数据存储服务**: MongoDB + PostgreSQL + Redis 数据存储
- **数据处理服务**: 智能数据分类和相似度检测
- **AI模型服务**: 大模型服务集群（ChatGLM3、Qwen、Baichuan2等）
- **文本优化服务**: 历史文本格式重新组织和优化
- **内容发布服务**: 多平台内容发布（微博、微信、抖音等）
- **客户消息服务**: 个性化客户沟通和消息推送

### 前端架构
- Vue 3.3+ + TypeScript 5.0+ + Vite 4.0+
- 状态管理: Pinia
- UI组件: Element Plus + Tailwind CSS
- 图表: ECharts + Chart.js + D3.js

### 后端技术栈
- 微服务框架: FastAPI + Python 3.11
- 数据库: MongoDB 6.0+ + PostgreSQL + Redis 7.0+
- 消息队列: Apache Kafka 3.0+ + RabbitMQ + Celery
- 容器化: Docker + Kubernetes + GPU Operator

## 文档结构与组织规范

### 文档存放规则

#### 1. 项目更新日志
**存放位置**: `changelogs/YYYY-MM-DD/`
- 按日期组织存放，一个日期一个文件夹
- 命名格式：`changelogs/2025-09-04/architecture-refactor.md`
- 禁止直接在项目根目录下创建更新文档

#### 2. 测试数据和日志
**存放位置**: `test-results/`
```
test-results/
├── YYYY-MM-DD/        # 按日期存放测试结果
│   ├── unit/          # 单元测试结果
│   ├── integration/   # 集成测试结果  
│   └── performance/   # 性能测试结果
├── data/              # 测试数据文件
│   ├── sample_content.json    # 示例内容数据
│   ├── mock_data/     # 模拟数据集
│   └── fixtures/      # 测试夹具
├── logs/              # 测试日志
│   ├── error.log      # 错误日志
│   ├── debug.log      # 调试日志
│   └── performance.log # 性能日志
└── reports/           # 测试报告
    ├── coverage/      # 覆盖率报告
    ├── html/          # HTML格式报告
    └── junit/         # JUnit格式报告
```

**重要规则**:
- 所有测试相关文件必须存放在 `test-results/` 目录下
- 禁止在项目根目录下直接创建测试文件
- 测试数据文件统一存放在 `test-results/data/` 下
- 临时测试文件也必须存放在此目录结构中

#### 3. 项目模块文档
**存放位置**: `modules/{module-name}/docs/`
- 每个模块的使用介绍必须存放在相应模块的docs文件夹下
- 禁止在项目根目录下直接放置模块使用文档
- 示例：`modules/data-source/docs/README.md`

### 核心文档位置
- **Epic和用户故事**: `docs/epics/epics.md` 和 `docs/user-stories.md`
- **架构文档**: `docs/architecture/` - 包含13个架构设计文档
- **详细用户故事**: `docs/stories/` - 按Epic组织的实现文档
- **前端文档**: `docs/frontend/` - 前端架构和组件设计

### 关键架构文档
- `docs/architecture/01-system-overview.md` - 系统整体架构
- `docs/architecture/02-microservices-architecture.md` - 微服务架构设计
- `docs/architecture/03-data-architecture.md` - 数据架构设计
- `docs/architecture/05-api-design.md` - API设计规范

## BMAD核心配置

项目使用BMAD (Business Model Automation and Documentation) 框架进行产品管理：

### 配置文件
- `.bmad-core/core-config.yaml` - 项目核心配置
- PRD文件位置: `docs/prd.md` (支持分片存储)
- 架构文档位置: `docs/architecture/` (支持分片存储)
- 用户故事位置: `docs/stories/`

### BMAD专业代理团队
项目集成了BMAD v4.42.1框架，提供完整的专业代理团队：

#### 核心开发代理
- **产品经理** (`/BMad:agents:pm`) - PRD创建、Epic规划、用户故事管理
- **开发工程师** (`/BMad:agents:dev`) - 代码实现、调试、重构、测试执行
- **架构师** (`/BMad:agents:architect`) - 系统架构设计和技术选型
- **QA工程师** (`/BMad:agents:qa`) - 质量保证、测试策略、代码审查

#### 业务与设计代理  
- **业务分析师** (`/BMad:agents:analyst`) - 需求分析、业务流程优化
- **产品负责人** (`/BMad:agents:po`) - 产品策略、路线图规划
- **UX专家** (`/BMad:agents:ux-expert`) - 用户体验设计、界面优化
- **项目经理** (`/BMad:agents:sm`) - 项目管理、进度跟踪

#### 管理与编排代理
- **主控代理** (`/BMad:agents:bmad-master`) - 整体协调和项目管理
- **编排代理** (`/BMad:agents:bmad-orchestrator`) - 工作流编排和任务分配

## 开发规范

### Epic开发优先级
1. **Epic 1**: 微服务基础设施和数据获取 (4-6周)
2. **Epic 2**: 数据处理和智能分类微服务 (6-8周)  
3. **Epic 3**: AI大模型服务和内容文本优化 (4-6周)
4. **Epic 4**: 发布管理和Vue3统一界面 (6-8周)

### 代码规范
- **Python**: 遵循PEP 8，使用Black和isort格式化，pytest单元测试
- **Vue3**: TypeScript覆盖率>90%，组件复用率>80%，使用ESLint和Prettier
- **提交规范**: 使用Conventional Commits，按Epic分类（如：`feat(epic-1): 新功能`）

### 文档维护规范
- **日志更新**: 每次重大变更后必须在 `changelogs/` 下创建日期文件夹和更新文档
- **测试数据**: 所有测试相关文件必须放在 `test-results/` 目录下
- **模块文档**: 每个模块的使用说明必须放在对应模块的 `docs/` 文件夹下
- **根目录清洁**: 禁止在项目根目录下直接创建文档文件（除README.md和CLAUDE.md外）

### 注释要求
- 所有注释必须使用中文
- 方法、类等添加全面注释（至少30%的注释密度）
- 函数级注释和完整文档字符串

## 环境配置

### 基础要求
- Python 3.11+
- Node.js 18+
- Docker 20.10+ & Docker Compose
- Kubernetes 1.25+ (生产环境)

### 数据库
- MongoDB 6.0+
- PostgreSQL 14+
- Redis 7.0+

### AI/GPU环境
- NVIDIA Driver 470+
- CUDA 11.8+
- Docker GPU支持
- Kubernetes GPU Operator

## 项目状态

### 当前状态
- ✅ 完整的架构设计和技术选型
- ✅ 4个Epic详细规划和19个用户故事定义
- ✅ 完整的项目文档体系
- ✅ 用户故事详细技术实现文档
- ✅ BMAD v4.42.1框架集成，配置10个专业代理
- ✅ Claude Code开发环境配置完成
- ✅ **Story 1.1完成**: 微服务基础架构搭建
- ✅ **Story 1.2完成**: 数据获取服务开发
- ✅ **Story 1.3完成**: 数据采集存储服务开发
- ✅ **Story 1.4完成**: 系统监控与日志管理
- ✅ **Docker Hub发布**: 两个核心微服务镜像已上传
- ✅ **微服务架构分离**: data-source与data-collection业务功能清晰分离 🆕
- 📋 Vue3前端应用待开发

### 工作方式
这是一个**文档驱动开发**的项目，目前已从设计阶段进入实现阶段：

**Epic 1 进展** - 微服务基础设施和数据获取（75%完成）：

**Story 1.1已完成** - 微服务基础架构搭建：
- 完整的Docker Compose和Kubernetes部署配置
- 基于Consul的服务注册发现系统
- Kong API网关配置（JWT、限流、CORS、监控）
- 分布式配置管理系统（基于Consul KV）
- 标准健康检查系统（支持K8s探针）
- 完整的单元测试和集成测试套件

**Story 1.2已完成** - 数据获取服务开发：
- 多平台爬虫支持（今日头条、百家号、小红书等）
- 智能代理管理系统（90个免费代理源）
- 反封禁策略和频率控制
- 完整RESTful API和Swagger文档
- Docker镜像: lhgray/historical-projects:data-source-latest (562MB)

**Story 1.3已完成** - 数据采集存储服务开发：
- 多格式文本提取（PDF、Word、图片OCR、HTML等）
- 多数据库架构（PostgreSQL + MongoDB + MinIO）
- 异步处理框架（RabbitMQ消息队列）
- 安全检测系统（病毒扫描、重复检测）
- Docker镜像: lhgray/historical-projects:data-collection-latest (748MB)

**开发规范**：
1. 阅读相关的Epic和用户故事文档
2. 理解微服务架构设计和API规范
3. 遵循已定义的技术栈和代码规范
4. 使用BMAD代理进行专业开发协作

### BMAD工作流集成
- 可使用 `/BMad:agents:pm` 激活产品管理代理
- 使用相关命令创建和管理项目文档

## 常用开发命令

### 环境管理
```bash
# 启动开发环境（本地开发）
docker-compose -f docker-compose.dev.yml up -d

# 启动生产环境测试
docker-compose -f docker-compose.production.yml up -d

# 查看服务状态
docker-compose -f docker-compose.dev.yml ps

# 查看服务日志
docker-compose -f docker-compose.dev.yml logs -f [service_name]

# 停止环境
docker-compose -f docker-compose.dev.yml down --volumes
```

### 测试命令
```bash
# 运行单元测试
python -m pytest tests/unit/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 运行特定测试文件
python -m pytest tests/unit/test_models.py -v

# 运行带覆盖率的测试
python -m pytest tests/unit/ --cov=src --cov-report=html

# 执行集成测试脚本
python tests/integration_test_runner.py

# 运行优化镜像测试
./scripts/test-optimized-images.sh
```

### 代码质量检查
```bash
# Python代码格式化
cd services/data-source && black src/
cd services/data-collection && black src/

# Python导入排序
cd services/data-source && isort src/
cd services/data-collection && isort src/

# 代码格式化检查
cd services/data-source && black --check src/
cd services/data-collection && black --check src/
```

### Docker操作
```bash
# 构建优化镜像
./scripts/build-optimized-images.sh

# 验证Docker Hub镜像
./scripts/verify-docker-hub.sh

# 手动构建数据源服务
cd services/data-source && docker build -t historical-text-data-source .

# 手动构建数据采集服务  
cd services/data-collection && docker build -t historical-text-data-collection .
```

### 服务开发
```bash
# 启动数据源服务（开发模式）
cd services/data-source && python -m src.main

# 启动数据采集服务（开发模式）
cd services/data-collection && python -m src.main

# 安装Python依赖
cd services/data-source && pip install -r requirements.txt
cd services/data-collection && pip install -r requirements.txt
```

### BMAD代理命令
```bash
# 核心开发代理
/BMad:agents:pm               # 激活产品管理代理 (John)
/BMad:agents:dev              # 激活开发工程师代理 (James) 
/BMad:agents:architect        # 激活架构师代理
/BMad:agents:qa               # 激活QA工程师代理

# 开发代理 (Dev) 命令
*develop-story               # 执行用户故事开发
*run-tests                   # 运行代码检查和测试
*review-qa                   # 执行QA修复任务
*explain                     # 解释实现过程和原理
```

### 文档和代码结构
```bash
# 项目文档结构
docs/architecture/           # 架构文档（13个文档）
docs/stories/               # 用户故事实现文档
docs/prd.md                 # 产品需求文档
CHANGELOG.md                # 项目更新历史
DEVELOPMENT_DASHBOARD.md    # 开发进度面板

# 核心代码结构
services/core/              # 核心基础设施服务
services/data-source/       # 数据源服务（已完成）
services/data-collection/   # 数据采集服务（已完成）
infrastructure/             # K8s和Docker配置
tests/                      # 测试套件
```

### 已实现的基础架构

#### 核心服务位置
- `services/core/registry/` - 服务注册与发现
- `services/core/config/` - 分布式配置管理  
- `services/core/health/` - 健康检查系统

#### 基础设施配置
- `docker-compose.yml` - 本地开发环境
- `infrastructure/kubernetes/` - K8s生产部署
- `infrastructure/docker/` - 服务配置文件

#### 测试套件
- `tests/unit/` - 单元测试
- `tests/integration/` - 集成测试

## 关键架构理解

### 微服务架构概览
项目采用事件驱动的微服务架构，当前已实现：

**已完成的服务** (Epic 1 - 75%完成)：
- **数据源服务** (`services/data-source/`) - 多平台内容爬取和代理管理
- **数据采集服务** (`services/data-collection/`) - 文件处理、存储和内容提取
- **基础设施服务** (`services/core/`) - 服务注册、配置管理、健康检查

**服务间通信**：
- API Gateway: Kong (配置在 `infrastructure/kong/`)
- 消息队列: RabbitMQ (异步任务处理)
- 服务发现: Consul (服务注册与发现)
- 配置管理: Consul KV (分布式配置)

**数据存储层**：
- PostgreSQL: 结构化数据存储 
- MongoDB: 非结构化内容存储
- Redis: 缓存和会话管理
- MinIO: 对象存储服务

### 开发模式说明
1. **文档驱动开发**: 所有功能先有详细技术文档（`docs/stories/`）
2. **测试驱动开发**: 每个服务都有完整的单元和集成测试
3. **容器化优先**: 所有服务都有Docker镜像，支持K8s部署

### 当前开发状态
- ✅ **Story 1.1**: 微服务基础架构 (完成)
- ✅ **Story 1.2**: 数据获取服务 (完成，Docker镜像: lhgray/historical-projects:data-source-latest)  
- ✅ **Story 1.3**: 数据采集存储服务 (完成，Docker镜像: lhgray/historical-projects:data-collection-latest)
- 🚧 **Story 1.4**: 系统监控与日志管理 (开发中)

## 开发工作流要求

### 代码开发流程
1. **阅读Story文档**: 查看 `docs/stories/epic-*/story-*.md` 了解具体需求
2. **理解服务架构**: 参考 `docs/architecture/02-microservices-architecture.md`
3. **遵循代码规范**: Python使用Black+isort，至少30%中文注释密度
4. **编写测试**: 单元测试覆盖率>80%，集成测试验证端到端功能
5. **Docker化**: 每个新服务必须提供Dockerfile和health check

### 提交要求
- 提交信息格式: `feat(epic-1): 新功能描述` 或 `fix(epic-2): 修复问题描述`
- 所有Python代码必须有完整的中文文档字符串
- 每次提交后运行代码格式化: `black src/ && isort src/`

### 测试策略
- **单元测试**: `tests/unit/` - 测试单个函数和类
- **集成测试**: `tests/integration/` - 测试服务间交互  
- **端到端测试**: `tests/integration_test_runner.py` - 完整流程验证

---

## 📋 最新文档更新记录 (2025-09-03)

### 📝 本次更新的文档
- **Story 1.3文档**: 状态从"待开发" → "✅ 已完成"，添加完成总结
- **README.md**: 更新项目状态，添加Docker镜像信息
- **DEVELOPMENT_DASHBOARD.md**: 更新项目进度、Epic完成度、最新成就
- **CHANGELOG.md**: 添加Story 1.3完成记录和Docker镜像优化总结
- **CLAUDE.md**: 更新项目状态，记录Epic 1进展

### 🎯 文档同步重点
- **状态同步**: 所有文档的项目状态已与实际开发进度同步
- **Docker信息**: 所有相关文档已添加Docker Hub镜像信息
- **进度更新**: Epic 1进度从50% → 75%，Story完成从2/19 → 3/19
- **成就记录**: 详细记录了Docker镜像优化和依赖精简成果

### 📊 文档更新统计
- **更新文档数**: 5个核心文档
- **新增内容**: Story 1.3完成总结、Docker镜像优化记录
- **状态同步**: 项目整体进度从40% → 45%
- **镜像信息**: 添加两个Docker Hub镜像的详细信息

### 🔄 文档一致性保障
- 所有文档的Story 1.3状态已统一为"已完成"
- Docker镜像信息在各文档中保持一致
- Epic 1完成度在所有文档中都更新为75%
- 项目总体进度指标已同步更新

### 📝 文档维护建议
- **定期同步**: 每完成一个Story后及时更新所有相关文档
- **版本管理**: 重要里程碑完成后记录到CHANGELOG.md
- **状态追踪**: DEVELOPMENT_DASHBOARD.md应实时反映开发状态
- **Docker记录**: 所有镜像发布都应记录在相关文档中

---

*CLAUDE.md文档最后更新时间: 2025-09-04*