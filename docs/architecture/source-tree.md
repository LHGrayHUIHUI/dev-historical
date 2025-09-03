# 源码结构文档

## 1. 概述

本文档详细描述了历史文本优化项目的完整源码组织结构。项目采用微服务架构，按功能模块进行组织，遵循领域驱动设计（DDD）原则，确保代码的可维护性、可扩展性和可测试性。

### 设计原则
- **领域驱动设计**: 按业务领域组织代码结构
- **分层架构**: 清晰的分层，职责分离
- **模块化**: 高内聚、低耦合的模块设计
- **可测试性**: 便于单元测试和集成测试
- **可维护性**: 清晰的命名约定和文档结构

## 2. 项目根目录结构

```
Historical Text Project/
├── .bmad-core/                 # BMAD框架核心配置
│   ├── agents/                 # 代理配置文件
│   ├── checklists/            # 检查清单模板
│   ├── tasks/                 # 任务定义文件
│   ├── workflows/             # 工作流配置
│   └── core-config.yaml       # 核心配置文件
├── .claude/                   # Claude Code配置
│   └── commands/              # 自定义命令
├── docs/                      # 项目文档
│   ├── architecture/          # 架构文档
│   ├── epics/                # Epic规划文档
│   ├── frontend/             # 前端文档
│   └── stories/              # 用户故事文档
├── frontend/                  # 前端应用（未来创建）
├── services/                  # 微服务代码
│   ├── core/                 # 核心基础服务
│   ├── data-source/          # 数据源服务
│   ├── data-collection/      # 数据采集服务
│   ├── ocr-service/          # OCR识别服务（计划）
│   ├── ai-analysis/          # AI分析服务（计划）
│   ├── search-service/       # 搜索服务（计划）
│   └── notification/         # 通知服务（计划）
├── infrastructure/           # 基础设施配置
│   ├── kubernetes/          # K8s部署文件
│   ├── docker/             # Docker配置
│   ├── kong/               # API网关配置
│   └── monitoring/         # 监控配置
├── tests/                   # 测试代码
│   ├── unit/               # 单元测试
│   ├── integration/        # 集成测试
│   └── e2e/               # 端到端测试
├── scripts/                # 工具脚本
├── docker-compose.dev.yml  # 开发环境配置
├── docker-compose.production.yml  # 生产环境配置
├── requirements.txt        # 项目依赖
├── CLAUDE.md              # Claude Code指导文档
├── README.md              # 项目说明文档
└── CHANGELOG.md           # 变更日志
```

## 3. 微服务源码结构

### 3.1 标准微服务结构模板

每个微服务都遵循统一的目录结构，以数据源服务为例：

```
services/data-source/
├── src/                          # 源代码目录
│   ├── __init__.py              # 包初始化文件
│   ├── main.py                  # 应用入口点
│   ├── config/                  # 配置管理
│   │   ├── __init__.py
│   │   ├── settings.py          # 配置类定义
│   │   └── logging.py           # 日志配置
│   ├── models/                  # 数据模型
│   │   ├── __init__.py
│   │   ├── base.py             # 基础模型类
│   │   ├── data_source.py      # 数据源模型
│   │   └── scraped_content.py  # 爬取内容模型
│   ├── schemas/                 # 数据模式验证
│   │   ├── __init__.py
│   │   ├── request.py          # 请求模式
│   │   ├── response.py         # 响应模式
│   │   └── validation.py       # 验证规则
│   ├── controllers/             # 控制器层
│   │   ├── __init__.py
│   │   ├── scraper_controller.py  # 爬虫控制器
│   │   └── health_controller.py   # 健康检查控制器
│   ├── services/                # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── scraper_service.py  # 爬虫业务逻辑
│   │   ├── proxy_service.py    # 代理管理服务
│   │   └── content_service.py  # 内容处理服务
│   ├── repositories/            # 数据访问层
│   │   ├── __init__.py
│   │   ├── base_repository.py  # 基础仓库类
│   │   └── content_repository.py # 内容仓库
│   ├── utils/                   # 工具类
│   │   ├── __init__.py
│   │   ├── http_client.py      # HTTP客户端
│   │   ├── parser.py           # 内容解析器
│   │   └── validator.py        # 验证工具
│   ├── middleware/              # 中间件
│   │   ├── __init__.py
│   │   ├── auth_middleware.py  # 认证中间件
│   │   ├── cors_middleware.py  # 跨域中间件
│   │   └── logging_middleware.py # 日志中间件
│   ├── exceptions/              # 自定义异常
│   │   ├── __init__.py
│   │   ├── base_exceptions.py  # 基础异常类
│   │   └── scraper_exceptions.py # 爬虫异常
│   └── external/                # 外部服务接口
│       ├── __init__.py
│       ├── proxy_client.py     # 代理客户端
│       └── platform_parsers/   # 平台解析器
│           ├── __init__.py
│           ├── toutiao_parser.py
│           ├── baijiahao_parser.py
│           └── xiaohongshu_parser.py
├── migrations/                  # 数据库迁移文件
│   ├── env.py                  # Alembic环境配置
│   ├── script.py.mako          # 迁移脚本模板
│   └── versions/               # 迁移版本文件
├── tests/                      # 测试代码
│   ├── __init__.py
│   ├── unit/                   # 单元测试
│   │   ├── test_services/
│   │   ├── test_controllers/
│   │   └── test_utils/
│   ├── integration/            # 集成测试
│   │   ├── test_api/
│   │   └── test_database/
│   └── fixtures/               # 测试数据
├── docker/                     # Docker相关文件
│   ├── Dockerfile.dev         # 开发环境Docker文件
│   └── entrypoint.sh          # 容器启动脚本
├── logs/                       # 日志文件目录
├── Dockerfile                  # 生产环境Docker文件
├── Dockerfile.optimized        # 优化版Docker文件
├── requirements.txt            # 项目依赖
├── alembic.ini                # Alembic配置文件
├── .env.example               # 环境变量示例
└── README.md                  # 服务说明文档
```

### 3.2 核心基础服务结构

核心基础服务提供通用的基础设施功能：

```
services/core/
├── config/                     # 分布式配置管理
│   ├── __init__.py
│   └── config_manager.py       # 配置管理器
├── gateway/                    # API网关服务（保留）
├── health/                     # 健康检查服务
│   ├── __init__.py
│   └── health_checker.py       # 健康检查器
├── registry/                   # 服务注册与发现
│   ├── __init__.py
│   └── service_registry.py     # 服务注册器
├── auth/                       # 认证授权服务（计划）
│   ├── __init__.py
│   ├── jwt_manager.py          # JWT管理器
│   ├── permission_checker.py   # 权限检查器
│   └── user_service.py         # 用户服务
├── cache/                      # 缓存管理服务
│   ├── __init__.py
│   ├── redis_manager.py        # Redis管理器
│   └── cache_decorator.py      # 缓存装饰器
└── monitoring/                 # 监控和指标收集
    ├── __init__.py
    ├── metrics_collector.py    # 指标收集器
    └── tracing.py             # 分布式追踪
```

### 3.3 数据采集服务详细结构

```
services/data-collection/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # 配置管理
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── data_controller.py   # 数据控制器
│   ├── models/                  # 数据模型层
│   │   ├── __init__.py
│   │   ├── base.py             # 基础模型
│   │   ├── data_source.py      # 数据源模型
│   │   ├── dataset.py          # 数据集模型
│   │   └── text_content.py     # 文本内容模型
│   ├── processors/              # 文件处理器
│   │   ├── __init__.py
│   │   ├── base.py             # 基础处理器
│   │   ├── pdf_extractor.py    # PDF提取器
│   │   ├── word_extractor.py   # Word提取器
│   │   ├── image_extractor.py  # 图像提取器
│   │   ├── html_extractor.py   # HTML提取器
│   │   └── text_extractor.py   # 纯文本提取器
│   ├── schemas/                 # 数据验证模式
│   │   ├── __init__.py
│   │   └── data_schemas.py     # 数据模式定义
│   ├── services/                # 业务服务层
│   │   ├── __init__.py
│   │   ├── data_service.py     # 数据处理服务
│   │   ├── file_service.py     # 文件管理服务
│   │   ├── storage_service.py  # 存储服务
│   │   └── validation_service.py # 验证服务
│   ├── utils/                   # 工具类
│   │   ├── __init__.py
│   │   ├── file_utils.py       # 文件工具
│   │   ├── text_utils.py       # 文本处理工具
│   │   └── security_utils.py   # 安全工具
│   └── main.py                 # 应用入口
├── migrations/                  # 数据库迁移
├── docker/                     # Docker配置
├── logs/                       # 日志目录
└── requirements.txt            # 依赖配置
```

## 4. 前端源码结构（规划）

### 4.1 Vue3前端应用结构

```
frontend/
├── public/                     # 静态资源
│   ├── index.html             # HTML模板
│   ├── favicon.ico            # 网站图标
│   └── icons/                 # 图标资源
├── src/                       # 源代码
│   ├── main.ts               # 应用入口
│   ├── App.vue               # 根组件
│   ├── components/           # 组件库
│   │   ├── common/          # 通用组件
│   │   │   ├── BaseButton.vue
│   │   │   ├── BaseModal.vue
│   │   │   ├── BaseTable.vue
│   │   │   └── LoadingSpinner.vue
│   │   ├── forms/           # 表单组件
│   │   │   ├── DocumentUpload.vue
│   │   │   ├── SearchForm.vue
│   │   │   └── UserProfile.vue
│   │   ├── layout/          # 布局组件
│   │   │   ├── AppHeader.vue
│   │   │   ├── AppSidebar.vue
│   │   │   ├── AppFooter.vue
│   │   │   └── Breadcrumb.vue
│   │   └── business/        # 业务组件
│   │       ├── DocumentCard.vue
│   │       ├── ProcessingStatus.vue
│   │       ├── TextViewer.vue
│   │       └── AnalysisResults.vue
│   ├── views/               # 页面组件
│   │   ├── Home.vue         # 首页
│   │   ├── Dashboard.vue    # 仪表板
│   │   ├── documents/       # 文档管理页面
│   │   │   ├── DocumentList.vue
│   │   │   ├── DocumentDetail.vue
│   │   │   ├── DocumentUpload.vue
│   │   │   └── DocumentSearch.vue
│   │   ├── analysis/        # 分析页面
│   │   │   ├── AnalysisView.vue
│   │   │   └── ResultsView.vue
│   │   ├── admin/           # 管理页面
│   │   │   ├── UserManagement.vue
│   │   │   ├── SystemSettings.vue
│   │   │   └── SystemMonitor.vue
│   │   └── auth/            # 认证页面
│   │       ├── Login.vue
│   │       ├── Register.vue
│   │       └── Profile.vue
│   ├── router/              # 路由配置
│   │   ├── index.ts         # 主路由文件
│   │   ├── modules/         # 路由模块
│   │   │   ├── documents.ts
│   │   │   ├── analysis.ts
│   │   │   ├── admin.ts
│   │   │   └── auth.ts
│   │   └── guards/          # 路由守卫
│   │       ├── auth.guard.ts
│   │       └── permission.guard.ts
│   ├── stores/              # 状态管理 (Pinia)
│   │   ├── index.ts         # Store入口
│   │   ├── auth.store.ts    # 认证状态
│   │   ├── user.store.ts    # 用户状态
│   │   ├── document.store.ts # 文档状态
│   │   ├── analysis.store.ts # 分析状态
│   │   └── app.store.ts     # 应用全局状态
│   ├── api/                 # API接口
│   │   ├── index.ts         # API入口
│   │   ├── client.ts        # HTTP客户端配置
│   │   ├── auth.api.ts      # 认证API
│   │   ├── user.api.ts      # 用户API
│   │   ├── document.api.ts  # 文档API
│   │   ├── analysis.api.ts  # 分析API
│   │   └── types/           # TypeScript类型定义
│   │       ├── auth.types.ts
│   │       ├── user.types.ts
│   │       ├── document.types.ts
│   │       └── analysis.types.ts
│   ├── composables/         # 组合式函数
│   │   ├── useAuth.ts       # 认证相关
│   │   ├── useDocument.ts   # 文档相关
│   │   ├── useAnalysis.ts   # 分析相关
│   │   ├── useApi.ts        # API调用
│   │   ├── usePermission.ts # 权限检查
│   │   └── useWebSocket.ts  # WebSocket连接
│   ├── utils/               # 工具函数
│   │   ├── index.ts         # 工具入口
│   │   ├── format.ts        # 格式化工具
│   │   ├── validate.ts      # 验证工具
│   │   ├── date.ts          # 日期工具
│   │   ├── file.ts          # 文件工具
│   │   └── storage.ts       # 存储工具
│   ├── styles/              # 样式文件
│   │   ├── main.scss        # 主样式文件
│   │   ├── variables.scss   # 变量定义
│   │   ├── mixins.scss      # 混入样式
│   │   ├── reset.scss       # 重置样式
│   │   └── themes/          # 主题样式
│   │       ├── light.scss
│   │       └── dark.scss
│   ├── assets/              # 资源文件
│   │   ├── images/          # 图片资源
│   │   ├── icons/           # 图标资源
│   │   └── fonts/           # 字体资源
│   ├── locales/             # 国际化文件
│   │   ├── index.ts         # i18n配置
│   │   ├── zh-CN.json       # 中文语言包
│   │   └── en-US.json       # 英文语言包
│   └── plugins/             # 插件配置
│       ├── element-plus.ts  # Element Plus配置
│       ├── axios.ts         # Axios配置
│       └── charts.ts        # 图表库配置
├── tests/                   # 测试文件
│   ├── unit/               # 单元测试
│   │   ├── components/
│   │   ├── stores/
│   │   └── utils/
│   ├── e2e/                # 端到端测试
│   │   ├── specs/
│   │   └── fixtures/
│   └── __mocks__/          # 模拟数据
├── .env                    # 环境变量
├── .env.development        # 开发环境变量
├── .env.production         # 生产环境变量
├── vite.config.ts          # Vite配置
├── tsconfig.json           # TypeScript配置
├── package.json            # 包管理配置
├── tailwind.config.js      # Tailwind CSS配置
├── eslint.config.js        # ESLint配置
├── prettier.config.js      # Prettier配置
└── README.md               # 前端说明文档
```

## 5. 基础设施代码结构

### 5.1 Kubernetes配置结构

```
infrastructure/kubernetes/
├── namespaces/                 # 命名空间定义
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── services/                  # 微服务部署配置
│   ├── data-source/
│   │   ├── deployment.yaml    # 部署配置
│   │   ├── service.yaml       # 服务定义
│   │   ├── configmap.yaml     # 配置映射
│   │   ├── secret.yaml        # 密钥配置
│   │   └── hpa.yaml           # 水平扩展配置
│   ├── data-collection/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── pvc.yaml           # 持久卷声明
│   └── core/
│       ├── config-service/
│       ├── health-service/
│       └── registry-service/
├── databases/                 # 数据库部署配置
│   ├── postgresql/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   ├── secret.yaml
│   │   └── pvc.yaml
│   ├── mongodb/
│   │   ├── statefulset.yaml
│   │   ├── service.yaml
│   │   └── pvc.yaml
│   ├── redis/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── elasticsearch/
│       ├── statefulset.yaml
│       ├── service.yaml
│       └── pvc.yaml
├── middleware/               # 中间件配置
│   ├── kong/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   ├── rabbitmq/
│   │   ├── statefulset.yaml
│   │   └── service.yaml
│   └── kafka/
│       ├── zookeeper.yaml
│       ├── kafka.yaml
│       └── service.yaml
├── monitoring/               # 监控配置
│   ├── prometheus/
│   │   ├── deployment.yaml
│   │   ├── configmap.yaml
│   │   └── service.yaml
│   ├── grafana/
│   │   ├── deployment.yaml
│   │   ├── configmap.yaml
│   │   └── service.yaml
│   └── jaeger/
│       ├── deployment.yaml
│       └── service.yaml
├── storage/                  # 存储配置
│   ├── minio/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── pvc.yaml
│   └── storage-classes/
│       ├── fast-ssd.yaml
│       └── standard.yaml
├── ingress/                  # 入口配置
│   ├── nginx-ingress.yaml
│   ├── tls-certificates.yaml
│   └── rate-limiting.yaml
└── rbac/                     # 权限配置
    ├── service-accounts.yaml
    ├── cluster-roles.yaml
    └── role-bindings.yaml
```

### 5.2 Docker配置结构

```
infrastructure/docker/
├── base-images/               # 基础镜像定义
│   ├── python-base.dockerfile
│   ├── node-base.dockerfile
│   └── nginx-base.dockerfile
├── development/               # 开发环境配置
│   ├── docker-compose.yml    # 开发环境编排
│   ├── .env.dev              # 开发环境变量
│   └── volumes/              # 数据卷映射
├── production/                # 生产环境配置
│   ├── docker-compose.yml    # 生产环境编排
│   ├── .env.prod             # 生产环境变量
│   └── secrets/              # 密钥文件
├── nginx/                    # Nginx配置
│   ├── nginx.conf            # 主配置文件
│   ├── sites/                # 站点配置
│   └── ssl/                  # SSL证书
└── scripts/                  # Docker脚本
    ├── build-all.sh          # 批量构建脚本
    ├── push-images.sh        # 推送镜像脚本
    └── health-check.sh       # 健康检查脚本
```

## 6. 测试代码结构

### 6.1 测试目录组织

```
tests/
├── conftest.py                # pytest配置文件
├── __init__.py
├── unit/                      # 单元测试
│   ├── __init__.py
│   ├── services/              # 服务层测试
│   │   ├── test_scraper_service.py
│   │   ├── test_document_service.py
│   │   ├── test_ocr_service.py
│   │   └── test_ai_service.py
│   ├── controllers/           # 控制器测试
│   │   ├── test_data_controller.py
│   │   ├── test_auth_controller.py
│   │   └── test_health_controller.py
│   ├── models/               # 模型测试
│   │   ├── test_user_model.py
│   │   ├── test_document_model.py
│   │   └── test_data_source_model.py
│   ├── utils/                # 工具类测试
│   │   ├── test_validators.py
│   │   ├── test_parsers.py
│   │   └── test_security_utils.py
│   └── repositories/         # 数据访问层测试
│       ├── test_user_repository.py
│       └── test_document_repository.py
├── integration/              # 集成测试
│   ├── __init__.py
│   ├── api/                  # API集成测试
│   │   ├── test_auth_api.py
│   │   ├── test_document_api.py
│   │   ├── test_scraper_api.py
│   │   └── test_analysis_api.py
│   ├── database/             # 数据库集成测试
│   │   ├── test_postgresql.py
│   │   ├── test_mongodb.py
│   │   └── test_redis.py
│   ├── external/             # 外部服务集成测试
│   │   ├── test_proxy_service.py
│   │   └── test_storage_service.py
│   └── workflows/            # 工作流测试
│       ├── test_document_processing.py
│       └── test_user_registration.py
├── e2e/                      # 端到端测试
│   ├── __init__.py
│   ├── specs/                # 测试规范
│   │   ├── test_user_journey.py
│   │   ├── test_document_workflow.py
│   │   └── test_admin_functions.py
│   ├── pages/                # 页面对象模型
│   │   ├── base_page.py
│   │   ├── login_page.py
│   │   ├── dashboard_page.py
│   │   └── document_page.py
│   ├── fixtures/             # 测试数据
│   │   ├── users.json
│   │   ├── documents.json
│   │   └── test_files/
│   └── screenshots/          # 测试截图
├── performance/              # 性能测试
│   ├── load_tests/           # 负载测试
│   │   ├── test_api_load.py
│   │   └── test_database_load.py
│   ├── stress_tests/         # 压力测试
│   │   └── test_concurrent_users.py
│   └── benchmarks/           # 基准测试
│       ├── test_ocr_performance.py
│       └── test_ai_inference.py
├── security/                 # 安全测试
│   ├── test_authentication.py
│   ├── test_authorization.py
│   ├── test_input_validation.py
│   └── test_sql_injection.py
├── fixtures/                 # 共享测试数据
│   ├── user_fixtures.py
│   ├── document_fixtures.py
│   └── api_fixtures.py
├── helpers/                  # 测试助手
│   ├── database_helper.py
│   ├── api_helper.py
│   └── mock_helper.py
└── reports/                  # 测试报告
    ├── coverage/             # 覆盖率报告
    ├── junit/                # JUnit格式报告
    └── html/                 # HTML格式报告
```

### 6.2 测试配置文件

```python
# tests/conftest.py - pytest全局配置
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# 测试数据库URL
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test_db"

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建测试事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """创建测试数据库引擎"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # 创建测试表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # 清理测试表
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """创建测试数据库会话"""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def test_client() -> TestClient:
    """创建测试客户端"""
    from src.main import app
    return TestClient(app)

@pytest.fixture
def sample_user_data():
    """示例用户数据"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }

@pytest.fixture
def sample_document_data():
    """示例文档数据"""
    return {
        "title": "测试文档",
        "content": "这是测试文档的内容",
        "file_type": "text/plain",
        "file_size": 1024
    }

# 测试标记定义
pytest_plugins = ["pytest_asyncio"]

# 测试标记
def pytest_configure(config):
    """配置测试标记"""
    config.addinivalue_line("markers", "unit: 标记为单元测试")
    config.addinivalue_line("markers", "integration: 标记为集成测试")
    config.addinivalue_line("markers", "e2e: 标记为端到端测试")
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "external: 标记为需要外部依赖的测试")
```

## 7. 文档结构

### 7.1 项目文档组织

```
docs/
├── architecture/              # 架构文档
│   ├── 01-system-overview.md  # 系统概览
│   ├── 02-microservices-architecture.md  # 微服务架构
│   ├── 03-data-architecture.md  # 数据架构
│   ├── 04-deployment-architecture.md  # 部署架构
│   ├── 05-api-design.md       # API设计
│   ├── 06-performance-scaling.md  # 性能与扩展
│   ├── 07-security-design.md  # 安全设计
│   ├── 08-monitoring-operations.md  # 监控运维
│   ├── 09-development-testing.md  # 开发测试
│   ├── 09-message-queue.md    # 消息队列
│   ├── 10-third-party-integration.md  # 第三方集成
│   ├── 11-performance-optimization.md  # 性能优化
│   ├── 12-scalability-design.md  # 可扩展性设计
│   ├── 13-disaster-recovery.md  # 灾难恢复
│   ├── coding-standards.md    # 编码规范
│   ├── tech-stack.md          # 技术栈说明
│   ├── source-tree.md         # 源码结构
│   └── README.md              # 架构文档索引
├── epics/                     # Epic规划文档
│   └── epics.md               # Epic总览
├── stories/                   # 用户故事文档
│   ├── epic-1/                # Epic 1相关故事
│   │   ├── story-1.1-microservice-infrastructure.md
│   │   ├── story-1.2-auth-service.md
│   │   ├── story-1.3-data-collection-service.md
│   │   └── story-1.4-monitoring-logging.md
│   ├── epic-2/                # Epic 2相关故事
│   │   ├── story-2.1-ocr-service.md
│   │   ├── story-2.2-nlp-service.md
│   │   ├── story-2.3-image-processing-service.md
│   │   ├── story-2.4-knowledge-graph-service.md
│   │   └── story-2.5-intelligent-classification-service.md
│   ├── epic-3/                # Epic 3相关故事
│   │   ├── story-3.1-search-engine-service.md
│   │   ├── story-3.2-recommendation-system.md
│   │   ├── story-3.3-data-visualization-service.md
│   │   ├── story-3.4-report-generation-service.md
│   │   └── story-3.5-user-behavior-analytics.md
│   ├── epic-4/                # Epic 4相关故事
│   │   ├── story-4.1-text-publishing-service.md
│   │   ├── story-4.2-content-moderation-service.md
│   │   ├── story-4.3-multi-platform-account-management.md
│   │   ├── story-4.4-automated-content-scheduling.md
│   │   └── story-4.5-analytics-reporting.md
│   ├── user-story-template.md  # 故事模板
│   └── README.md              # 故事文档索引
├── frontend/                  # 前端文档
│   ├── frontend-overview.md   # 前端概览
│   ├── frontend-development.md  # 前端开发指南
│   ├── frontend-development-guide.md  # 开发详细指南
│   ├── frontend-pages.md      # 页面设计
│   ├── frontend-components.md  # 组件设计
│   ├── frontend-api.md        # API接口文档
│   ├── brownfield-enhancement-prd.md  # 增强PRD
│   └── README.md              # 前端文档索引
├── api/                       # API文档
│   ├── openapi.yaml          # OpenAPI规范文件
│   ├── authentication.md     # 认证API
│   ├── users.md              # 用户API
│   ├── documents.md          # 文档API
│   ├── analysis.md           # 分析API
│   └── search.md             # 搜索API
├── deployment/               # 部署文档
│   ├── local-development.md  # 本地开发环境
│   ├── kubernetes-deployment.md  # K8s部署指南
│   ├── docker-deployment.md  # Docker部署指南
│   ├── monitoring-setup.md   # 监控配置
│   └── troubleshooting.md    # 故障排除
├── user-guides/              # 用户指南
│   ├── getting-started.md    # 快速开始
│   ├── user-manual.md        # 用户手册
│   ├── admin-guide.md        # 管理员指南
│   └── faq.md                # 常见问题
├── development/              # 开发文档
│   ├── coding-guidelines.md  # 编码指南
│   ├── testing-strategy.md   # 测试策略
│   ├── code-review.md        # 代码审查
│   └── contribution-guide.md  # 贡献指南
├── operations/               # 运维文档
│   ├── monitoring.md         # 监控指南
│   ├── logging.md            # 日志管理
│   ├── backup-recovery.md    # 备份恢复
│   └── security-ops.md       # 安全运维
├── images/                   # 文档图片
│   ├── architecture/         # 架构图
│   ├── workflows/            # 流程图
│   └── screenshots/          # 截图
├── prd.md                    # 产品需求文档
├── user-stories.md           # 用户故事汇总
└── README.md                 # 文档入口
```

### 7.2 代码注释规范

每个源文件都应该包含完整的中文注释：

```python
# src/services/document_service.py
"""
文档处理服务模块

此模块提供文档的完整生命周期管理功能，包括文档上传、
存储、处理、分析和查询等核心业务逻辑。

主要功能：
- 文档上传和验证
- 多格式文档解析（PDF、Word、图片等）
- OCR文字识别和提取
- AI智能分析和标签生成
- 全文搜索索引构建
- 文档版本管理

Author: 开发团队
Created: 2025-01-15
Modified: 2025-09-03
Version: 1.2.0
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.document import Document, DocumentStatus
from ..schemas.document import DocumentCreate, DocumentUpdate
from ..repositories.document_repository import DocumentRepository
from ..utils.file_processor import FileProcessor
from ..external.ocr_client import OCRClient
from ..external.ai_analyzer import AIAnalyzer
from ..exceptions.document_exceptions import (
    DocumentNotFoundError,
    InvalidFileFormatError,
    ProcessingTimeoutError
)

import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """文档业务逻辑服务类
    
    负责处理文档相关的所有业务逻辑，包括文档的创建、
    更新、删除、查询以及文档内容的处理和分析。
    
    该类采用依赖注入模式，通过构造函数注入所需的
    外部依赖，便于单元测试和模块解耦。
    
    Attributes:
        _repository: 文档数据访问仓库
        _file_processor: 文件处理器
        _ocr_client: OCR服务客户端
        _ai_analyzer: AI分析服务客户端
        _session: 数据库会话
    
    Example:
        >>> service = DocumentService(session, repository)
        >>> document = await service.create_document(document_data)
        >>> result = await service.process_document(document.id)
    """
    
    def __init__(
        self,
        session: AsyncSession,
        repository: DocumentRepository,
        file_processor: FileProcessor,
        ocr_client: OCRClient,
        ai_analyzer: AIAnalyzer
    ):
        """初始化文档服务
        
        Args:
            session: 数据库异步会话实例
            repository: 文档数据访问仓库实例
            file_processor: 文件处理器实例
            ocr_client: OCR服务客户端实例
            ai_analyzer: AI分析服务客户端实例
        """
        self._session = session
        self._repository = repository
        self._file_processor = file_processor
        self._ocr_client = ocr_client
        self._ai_analyzer = ai_analyzer
        
        logger.info("文档服务初始化完成")
    
    async def create_document(
        self, 
        document_data: DocumentCreate,
        file_content: bytes,
        user_id: str
    ) -> Document:
        """创建新文档
        
        执行文档创建的完整流程，包括数据验证、文件存储、
        数据库记录创建和异步处理任务的启动。
        
        处理流程：
        1. 验证输入参数和文件格式
        2. 生成唯一的文档ID
        3. 存储文件到对象存储系统
        4. 创建数据库记录
        5. 启动异步处理任务
        6. 返回文档基本信息
        
        Args:
            document_data: 文档创建数据，包含标题、描述等基本信息
            file_content: 文件二进制内容
            user_id: 创建文档的用户ID
            
        Returns:
            创建成功的文档对象，包含ID、状态、创建时间等信息
            
        Raises:
            InvalidFileFormatError: 文件格式不支持时抛出
            FileTooLargeError: 文件过大时抛出
            DatabaseError: 数据库操作失败时抛出
            StorageError: 文件存储失败时抛出
            
        Example:
            >>> document_data = DocumentCreate(
            ...     title="测试文档",
            ...     description="这是一个测试文档"
            ... )
            >>> with open("test.pdf", "rb") as f:
            ...     content = f.read()
            >>> document = await service.create_document(
            ...     document_data, content, "user-123"
            ... )
            >>> print(f"文档创建成功: {document.id}")
        """
        logger.info(f"开始创建文档: {document_data.title}, 用户: {user_id}")
        
        try:
            # 第一步：验证文件格式和大小
            await self._validate_file(file_content, document_data.file_name)
            logger.debug("文件验证通过")
            
            # 第二步：存储文件到对象存储
            file_path = await self._file_processor.store_file(
                file_content, 
                document_data.file_name,
                user_id
            )
            logger.debug(f"文件存储成功: {file_path}")
            
            # 第三步：创建文档数据库记录
            document = await self._repository.create({
                **document_data.dict(),
                "file_path": file_path,
                "file_size": len(file_content),
                "status": DocumentStatus.UPLOADED,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            logger.info(f"文档创建成功: {document.id}")
            
            # 第四步：启动异步处理任务
            await self._start_processing_task(document.id)
            logger.debug(f"异步处理任务已启动: {document.id}")
            
            return document
            
        except Exception as e:
            logger.error(f"文档创建失败: {document_data.title}, 错误: {e}")
            # 回滚操作：清理已存储的文件
            if 'file_path' in locals():
                await self._file_processor.delete_file(file_path)
            raise
    
    async def process_document(self, document_id: str) -> Dict[str, Any]:
        """处理文档内容
        
        执行文档的完整处理流程，包括OCR识别、AI分析、
        内容提取和索引构建等步骤。此方法通常由异步
        任务队列调用。
        
        处理步骤：
        1. 获取文档信息和文件
        2. 根据文件类型选择处理策略
        3. 执行OCR文字识别（图片和扫描PDF）
        4. 进行AI内容分析和标签提取
        5. 构建全文搜索索引
        6. 更新文档状态和结果
        7. 发送处理完成通知
        
        Args:
            document_id: 要处理的文档ID
            
        Returns:
            处理结果字典，包含以下字段：
            - success: 处理是否成功
            - extracted_text: 提取的文本内容
            - ocr_confidence: OCR识别置信度
            - ai_tags: AI生成的标签列表
            - processing_time: 处理耗时（秒）
            - error_message: 错误信息（如果失败）
            
        Raises:
            DocumentNotFoundError: 文档不存在时抛出
            ProcessingTimeoutError: 处理超时时抛出
            OCRServiceError: OCR服务异常时抛出
            AIAnalysisError: AI分析服务异常时抛出
            
        Example:
            >>> result = await service.process_document("doc-123")
            >>> if result["success"]:
            ...     print(f"处理成功，提取文本长度: {len(result['extracted_text'])}")
            ...     print(f"AI标签: {result['ai_tags']}")
            >>> else:
            ...     print(f"处理失败: {result['error_message']}")
        """
        start_time = datetime.utcnow()
        logger.info(f"开始处理文档: {document_id}")
        
        try:
            # 第一步：获取文档信息
            document = await self._repository.get_by_id(document_id)
            if not document:
                raise DocumentNotFoundError(f"文档不存在: {document_id}")
            
            # 更新状态为处理中
            await self._repository.update_status(document_id, DocumentStatus.PROCESSING)
            logger.debug(f"文档状态已更新为处理中: {document_id}")
            
            # 第二步：读取文件内容
            file_content = await self._file_processor.read_file(document.file_path)
            logger.debug(f"文件读取成功，大小: {len(file_content)} bytes")
            
            # 第三步：根据文件类型执行不同的处理逻辑
            extracted_text = ""
            ocr_confidence = 0.0
            
            if document.mime_type.startswith('image/'):
                # 图片文件：直接进行OCR识别
                ocr_result = await self._ocr_client.recognize_image(file_content)
                extracted_text = ocr_result.text
                ocr_confidence = ocr_result.confidence
                logger.debug(f"图片OCR识别完成，置信度: {ocr_confidence}")
                
            elif document.mime_type == 'application/pdf':
                # PDF文件：先尝试文本提取，失败则OCR
                try:
                    extracted_text = await self._file_processor.extract_pdf_text(file_content)
                    ocr_confidence = 1.0  # 直接文本提取，置信度为100%
                    logger.debug("PDF文本直接提取成功")
                except Exception:
                    # 可能是扫描版PDF，使用OCR
                    ocr_result = await self._ocr_client.recognize_pdf(file_content)
                    extracted_text = ocr_result.text
                    ocr_confidence = ocr_result.confidence
                    logger.debug(f"PDF OCR识别完成，置信度: {ocr_confidence}")
                    
            else:
                # 其他格式文档（Word、TXT等）
                extracted_text = await self._file_processor.extract_text(
                    file_content, document.mime_type
                )
                ocr_confidence = 1.0
                logger.debug("文档文本提取成功")
            
            # 第四步：AI内容分析
            ai_result = await self._ai_analyzer.analyze_text(extracted_text)
            ai_tags = ai_result.get('tags', [])
            sentiment = ai_result.get('sentiment', 'neutral')
            keywords = ai_result.get('keywords', [])
            logger.debug(f"AI分析完成，生成 {len(ai_tags)} 个标签")
            
            # 第五步：构建搜索索引（异步执行，不阻塞主流程）
            asyncio.create_task(
                self._build_search_index(document_id, extracted_text, ai_tags)
            )
            
            # 第六步：更新文档处理结果
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            await self._repository.update_processing_result(document_id, {
                "status": DocumentStatus.COMPLETED,
                "extracted_text": extracted_text,
                "ocr_confidence": ocr_confidence,
                "ai_tags": ai_tags,
                "sentiment": sentiment,
                "keywords": keywords,
                "processing_time": processing_time,
                "updated_at": datetime.utcnow()
            })
            
            logger.info(f"文档处理完成: {document_id}, 耗时: {processing_time:.2f}秒")
            
            # 构建返回结果
            result = {
                "success": True,
                "extracted_text": extracted_text,
                "ocr_confidence": ocr_confidence,
                "ai_tags": ai_tags,
                "sentiment": sentiment,
                "keywords": keywords,
                "processing_time": processing_time
            }
            
            # 第七步：发送处理完成通知（异步执行）
            asyncio.create_task(
                self._send_completion_notification(document_id, result)
            )
            
            return result
            
        except Exception as e:
            # 处理失败，更新状态并记录错误
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            error_message = str(e)
            
            await self._repository.update_status(
                document_id, 
                DocumentStatus.FAILED,
                error_message=error_message
            )
            
            logger.error(f"文档处理失败: {document_id}, 错误: {error_message}, 耗时: {processing_time:.2f}秒")
            
            return {
                "success": False,
                "error_message": error_message,
                "processing_time": processing_time
            }
    
    async def _validate_file(self, file_content: bytes, filename: str) -> None:
        """验证文件格式和大小
        
        Args:
            file_content: 文件内容
            filename: 文件名
            
        Raises:
            InvalidFileFormatError: 文件格式不支持
            FileTooLargeError: 文件过大
        """
        # 文件大小检查（50MB限制）
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_size:
            raise FileTooLargeError(f"文件过大: {len(file_content)} bytes，限制: {max_size} bytes")
        
        # 文件格式检查
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.jpg', '.jpeg', '.png', '.tiff'}
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise InvalidFileFormatError(f"不支持的文件格式: {file_extension}")
        
        # 文件内容魔数检查（防止扩展名伪装）
        mime_type = self._file_processor.detect_mime_type(file_content)
        allowed_mime_types = {
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'image/jpeg',
            'image/png',
            'image/tiff'
        }
        
        if mime_type not in allowed_mime_types:
            raise InvalidFileFormatError(f"文件内容与扩展名不匹配: {mime_type}")
        
        logger.debug(f"文件验证通过: {filename}, 类型: {mime_type}, 大小: {len(file_content)} bytes")
```

## 8. 配置文件结构

### 8.1 环境配置文件

```bash
# .env.example - 环境变量示例
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/historical_text
MONGO_URL=mongodb://localhost:27017/historical_text
REDIS_URL=redis://localhost:6379/0
ELASTICSEARCH_URL=http://localhost:9200

# 安全配置
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# 文件存储配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=historical-text-files

# 外部服务配置
OCR_SERVICE_URL=http://localhost:8001
AI_SERVICE_URL=http://localhost:8002

# 应用配置
APP_NAME=Historical Text Processing Platform
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# 微服务配置
SERVICE_REGISTRY_URL=http://consul:8500
API_GATEWAY_URL=http://kong:8000

# 监控配置
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
```

### 8.2 应用配置结构

```python
# config/settings.py - 应用配置
from pydantic import BaseSettings, Field
from typing import List, Optional
from pathlib import Path

class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    app_name: str = Field("Historical Text Processing Platform", description="应用名称")
    app_version: str = Field("1.0.0", description="应用版本")
    debug: bool = Field(False, description="调试模式")
    log_level: str = Field("INFO", description="日志级别")
    
    # 服务配置
    host: str = Field("0.0.0.0", description="监听主机")
    port: int = Field(8000, description="监听端口")
    reload: bool = Field(False, description="热重载")
    
    # 数据库配置
    database_url: str = Field(..., description="PostgreSQL连接URL")
    mongo_url: str = Field(..., description="MongoDB连接URL")
    redis_url: str = Field(..., description="Redis连接URL")
    elasticsearch_url: str = Field(..., description="Elasticsearch连接URL")
    
    # 安全配置
    secret_key: str = Field(..., description="JWT密钥")
    jwt_algorithm: str = Field("HS256", description="JWT算法")
    jwt_expire_minutes: int = Field(30, description="JWT过期时间（分钟）")
    
    # 文件存储配置
    minio_endpoint: str = Field(..., description="MinIO端点")
    minio_access_key: str = Field(..., description="MinIO访问密钥")
    minio_secret_key: str = Field(..., description="MinIO秘密密钥")
    minio_bucket: str = Field("historical-text-files", description="MinIO存储桶")
    
    # 外部服务配置
    ocr_service_url: str = Field(..., description="OCR服务URL")
    ai_service_url: str = Field(..., description="AI服务URL")
    
    # 微服务配置
    service_registry_url: str = Field(..., description="服务注册中心URL")
    api_gateway_url: str = Field(..., description="API网关URL")
    
    # 文件处理配置
    max_file_size: int = Field(50 * 1024 * 1024, description="最大文件大小（字节）")
    allowed_file_types: List[str] = Field(
        ["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png", "tiff"],
        description="允许的文件类型"
    )
    
    # 处理配置
    max_concurrent_tasks: int = Field(10, description="最大并发任务数")
    task_timeout: int = Field(300, description="任务超时时间（秒）")
    
    # 监控配置
    prometheus_multiproc_dir: Optional[str] = Field(None, description="Prometheus多进程目录")
    jaeger_agent_host: str = Field("localhost", description="Jaeger Agent主机")
    jaeger_agent_port: int = Field(6831, description="Jaeger Agent端口")
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        case_sensitive = False

# 全局配置实例
settings = Settings()
```

## 9. 总结

本文档详细描述了历史文本优化项目的完整源码结构，包括：

1. **项目整体结构**: 清晰的模块化组织，便于开发和维护
2. **微服务架构**: 标准化的服务结构模板，确保一致性
3. **分层设计**: Controller-Service-Repository分层架构
4. **测试组织**: 完整的测试代码结构和配置
5. **文档系统**: 系统化的文档组织和维护策略
6. **配置管理**: 灵活的环境配置和应用配置

### 关键设计原则
- **领域驱动**: 按业务领域组织代码
- **分层架构**: 清晰的职责分离
- **模块化**: 高内聚、低耦合
- **可测试性**: 便于单元测试和集成测试
- **可维护性**: 清晰的命名和完善的文档

### 开发建议
1. **遵循结构规范**: 严格按照定义的目录结构组织代码
2. **保持一致性**: 所有微服务使用相同的结构模板
3. **完善注释**: 确保30%以上的中文注释密度
4. **测试覆盖**: 保持80%以上的测试覆盖率
5. **文档同步**: 及时更新相关文档

通过遵循本文档定义的源码结构规范，可以确保项目的长期可维护性和团队协作效率。

---

**文档版本**: v1.0  
**最后更新**: 2025-09-03  
**负责人**: 架构师团队  
**审核人**: 技术总监