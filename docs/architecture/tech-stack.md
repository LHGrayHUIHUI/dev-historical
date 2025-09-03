# 技术栈文档

## 1. 概述

本文档详细描述了历史文本优化项目的完整技术栈选择和架构设计。项目采用现代化的微服务架构，结合云原生技术和AI/ML能力，构建高性能、高可用的智能文本处理平台。

### 技术选型原则
- **云原生**: 容器化部署，支持Kubernetes编排
- **微服务化**: 服务独立部署，松耦合架构
- **高性能**: 异步处理，缓存优化，数据库调优
- **可扩展性**: 支持水平扩展和弹性伸缩
- **安全性**: 多层安全防护，数据加密传输和存储
- **可观测性**: 完整的监控、日志、链路追踪体系

## 2. 整体架构图

```mermaid
graph TB
    subgraph "前端层 Frontend Layer"
        WEB[Vue3 Web应用]
        MOBILE[React Native移动端]
        ADMIN[管理后台]
    end
    
    subgraph "网关层 Gateway Layer"
        CDN[CDN边缘节点]
        LB[负载均衡器]
        KONG[Kong API网关]
    end
    
    subgraph "微服务层 Microservices Layer"
        AUTH[认证服务<br/>FastAPI + JWT]
        USER[用户服务<br/>FastAPI + PostgreSQL]
        DOC[文档服务<br/>FastAPI + MongoDB]
        OCR[OCR服务<br/>Python + TesseractOCR]
        AI[AI分析服务<br/>Python + Transformers]
        SEARCH[搜索服务<br/>FastAPI + Elasticsearch]
        NOTIFY[通知服务<br/>FastAPI + RabbitMQ]
    end
    
    subgraph "数据层 Data Layer"
        PG[(PostgreSQL 15+<br/>用户、权限、配置)]
        MONGO[(MongoDB 6.0+<br/>文档、内容)]
        REDIS[(Redis 7.0+<br/>缓存、会话)]
        ES[(Elasticsearch 8+<br/>全文搜索)]
        MINIO[(MinIO<br/>文件存储)]
    end
    
    subgraph "基础设施层 Infrastructure Layer"
        K8S[Kubernetes 1.25+]
        CONSUL[Consul服务发现]
        PROMETHEUS[Prometheus监控]
        GRAFANA[Grafana可视化]
        ELK[ELK日志栈]
    end
    
    WEB --> CDN
    MOBILE --> CDN
    ADMIN --> CDN
    CDN --> LB
    LB --> KONG
    
    KONG --> AUTH
    KONG --> USER
    KONG --> DOC
    KONG --> OCR
    KONG --> AI
    KONG --> SEARCH
    KONG --> NOTIFY
    
    AUTH --> PG
    AUTH --> REDIS
    USER --> PG
    USER --> REDIS
    DOC --> MONGO
    DOC --> MINIO
    OCR --> MINIO
    AI --> REDIS
    SEARCH --> ES
    NOTIFY --> REDIS
    
    K8S --> AUTH
    K8S --> USER
    K8S --> DOC
    K8S --> OCR
    K8S --> AI
    K8S --> SEARCH
    K8S --> NOTIFY
```

## 3. 前端技术栈

### 3.1 核心框架和库

| 技术 | 版本 | 用途 | 选择理由 |
|------|------|------|----------|
| **Vue.js** | 3.3+ | 前端框架 | 渐进式框架，生态丰富，开发效率高 |
| **TypeScript** | 5.0+ | 类型系统 | 强类型支持，提高代码质量和维护性 |
| **Vite** | 4.0+ | 构建工具 | 快速热更新，现代化构建工具 |
| **Pinia** | 2.1+ | 状态管理 | Vue3官方推荐，简洁的API |
| **Vue Router** | 4.2+ | 路由管理 | Vue官方路由器，支持动态路由 |
| **Element Plus** | 2.4+ | UI组件库 | 企业级UI组件，文档完善 |

### 3.2 样式和UI增强

```typescript
// 样式技术栈配置
const styleConfig = {
  // CSS预处理器
  preprocessor: 'Sass/SCSS',
  
  // 原子化CSS框架
  utility: 'Tailwind CSS 3.x',
  
  // 图标库
  icons: [
    'Element Plus Icons',
    'Heroicons',
    'Lucide Vue'
  ],
  
  // 动画库
  animation: [
    'Vue Transition',
    'Framer Motion',
    'GSAP (按需使用)'
  ]
}
```

### 3.3 工具和开发增强

```json
{
  "开发工具": {
    "代码格式化": "Prettier 3.x",
    "代码检查": "ESLint 8.x + Vue ESLint Plugin",
    "类型检查": "Vue TSC",
    "测试框架": "Vitest + Vue Test Utils",
    "E2E测试": "Playwright",
    "组件文档": "Storybook 7.x"
  },
  "性能优化": {
    "打包分析": "Bundle Analyzer",
    "图片优化": "Vite ImageOptimize Plugin",
    "代码分割": "Dynamic Imports",
    "懒加载": "Vue3 Suspense + Async Components"
  },
  "开发体验": {
    "热更新": "Vite HMR",
    "自动导入": "Auto Import Plugin",
    "组件自动注册": "Components Auto Import",
    "API自动生成": "OpenAPI Generator"
  }
}
```

### 3.4 前端架构特性

```typescript
// 项目结构和特性
interface FrontendArchitecture {
  // 组件设计模式
  componentPattern: {
    atomic: '原子化设计原则',
    composition: 'Composition API模式',
    composables: '可复用逻辑封装',
    renderless: '无渲染组件模式'
  }
  
  // 状态管理策略
  stateManagement: {
    global: 'Pinia全局状态',
    local: 'Vue3 Reactive本地状态',
    server: 'Vue Query服务端状态',
    form: 'VeeValidate表单状态'
  }
  
  // 路由策略
  routing: {
    lazy: '路由懒加载',
    guard: '路由守卫认证',
    meta: '路由元信息管理',
    nested: '嵌套路由结构'
  }
  
  // 国际化支持
  i18n: {
    framework: 'Vue I18n 9.x',
    loading: '异步语言包加载',
    fallback: '回退语言机制',
    format: 'ICU消息格式'
  }
}
```

## 4. 后端技术栈

### 4.1 微服务框架

| 服务名称 | 框架 | 数据库 | 主要功能 |
|----------|------|--------|----------|
| **认证服务** | FastAPI 0.104+ | PostgreSQL + Redis | JWT认证、权限管理、单点登录 |
| **用户服务** | FastAPI 0.104+ | PostgreSQL + Redis | 用户信息、角色权限、用户画像 |
| **文档服务** | FastAPI 0.104+ | MongoDB + MinIO | 文档CRUD、版本管理、元数据 |
| **OCR服务** | FastAPI 0.104+ | MinIO + PostgreSQL | 图像识别、文字提取、结果缓存 |
| **AI分析服务** | FastAPI 0.104+ | Redis + PostgreSQL | NLP分析、实体识别、内容分类 |
| **搜索服务** | FastAPI 0.104+ | Elasticsearch + Redis | 全文搜索、语义检索、搜索推荐 |
| **数据采集服务** | FastAPI 0.104+ | MongoDB + PostgreSQL | 爬虫调度、数据清洗、质量控制 |

### 4.2 Python 核心依赖

```python
# 核心Web框架栈
web_stack = {
    'FastAPI': '0.104.1',          # 现代化Web框架
    'uvicorn[standard]': '0.24.0',  # ASGI服务器
    'pydantic': '2.5.1',           # 数据验证和序列化
    'pydantic-settings': '2.1.0',  # 配置管理
}

# 异步处理栈
async_stack = {
    'asyncio': 'Python内置',        # 异步IO
    'aiohttp': '3.9.1',           # 异步HTTP客户端
    'aiofiles': '23.2.1',         # 异步文件操作
    'asyncpg': '0.29.0',          # PostgreSQL异步驱动
    'motor': '3.3.2',             # MongoDB异步驱动
    'aioredis': '2.0.1',          # Redis异步客户端
    'aio-pika': '9.3.1',          # RabbitMQ异步客户端
}

# 数据库和ORM
database_stack = {
    'SQLAlchemy': '2.0.23',        # ORM框架
    'alembic': '1.13.0',          # 数据库迁移
    'asyncpg': '0.29.0',          # PostgreSQL驱动
    'psycopg2-binary': '2.9.9',   # PostgreSQL同步驱动
    'pymongo': '4.6.0',           # MongoDB驱动
    'redis[hiredis]': '5.0.1',    # Redis驱动
}

# 认证和安全
security_stack = {
    'python-jose[cryptography]': '3.3.0',  # JWT处理
    'passlib[bcrypt]': '1.7.4',           # 密码哈希
    'python-multipart': '0.0.6',          # 文件上传支持
    'cryptography': '41.0.7',             # 加密库
}

# AI和机器学习
ai_ml_stack = {
    'transformers': '4.35.2',      # Hugging Face Transformers
    'torch': '2.1.1',            # PyTorch (可选GPU版本)
    'numpy': '1.25.2',           # 数值计算
    'pandas': '2.1.3',           # 数据处理
    'scikit-learn': '1.3.2',     # 机器学习
    'langdetect': '1.0.9',       # 语言检测
}

# 文档处理
document_stack = {
    'PyPDF2': '3.0.1',           # PDF处理
    'pdfplumber': '0.10.3',      # PDF解析增强
    'python-docx': '1.1.0',      # Word文档处理
    'openpyxl': '3.1.2',         # Excel处理
    'Pillow': '10.1.0',          # 图像处理
    'python-magic': '0.4.27',    # 文件类型检测
    'beautifulsoup4': '4.12.2',  # HTML解析
    'lxml': '4.9.3',             # XML/HTML解析器
}

# 爬虫和数据采集
scraping_stack = {
    'scrapy': '2.11.0',          # 爬虫框架
    'selenium': '4.16.0',        # 浏览器自动化
    'requests': '2.31.0',        # HTTP库
    'httpx': '0.25.2',           # 现代HTTP客户端
    'aiohttp': '3.9.1',          # 异步HTTP
    'beautifulsoup4': '4.12.2',  # HTML解析
    'aiohttp-socks': '0.8.4',    # 代理支持
}

# 监控和日志
monitoring_stack = {
    'prometheus-client': '0.19.0',  # Prometheus指标
    'structlog': '23.2.0',         # 结构化日志
    'loguru': '0.7.2',             # 日志库
    'psutil': '5.9.6',             # 系统监控
    'python-json-logger': '2.0.7', # JSON日志格式
}

# 测试框架
testing_stack = {
    'pytest': '7.4.3',           # 测试框架
    'pytest-asyncio': '0.21.1',  # 异步测试支持
    'pytest-cov': '4.1.0',       # 覆盖率测试
    'pytest-mock': '3.12.0',     # Mock支持
    'factory-boy': '3.3.0',      # 测试数据工厂
    'httpx': '0.25.2',           # 测试HTTP客户端
}

# 开发工具
dev_tools = {
    'black': '23.11.0',          # 代码格式化
    'isort': '5.12.0',           # 导入排序
    'flake8': '6.1.0',           # 代码检查
    'mypy': '1.7.1',             # 类型检查
    'bandit': '1.7.5',          # 安全检查
    'pre-commit': '3.6.0',      # Git钩子
}
```

### 4.3 服务架构特性

```python
# 微服务架构特性配置
from typing import Dict, List, Any

class MicroserviceConfig:
    """微服务配置"""
    
    # 服务发现和注册
    service_discovery = {
        'consul': {
            'host': 'consul',
            'port': 8500,
            'health_check_interval': '10s',
            'deregister_critical_after': '1m'
        }
    }
    
    # API网关配置
    api_gateway = {
        'kong': {
            'admin_url': 'http://kong:8001',
            'proxy_url': 'http://kong:8000',
            'plugins': [
                'jwt',           # JWT认证
                'rate-limiting', # 限流
                'cors',          # 跨域
                'prometheus',    # 监控
                'request-logging' # 请求日志
            ]
        }
    }
    
    # 负载均衡策略
    load_balancing = {
        'algorithm': 'round-robin',  # 轮询算法
        'health_check': True,        # 健康检查
        'sticky_session': False,     # 不使用粘性会话
        'timeout': 30               # 超时时间
    }
    
    # 熔断器配置
    circuit_breaker = {
        'failure_threshold': 5,      # 失败阈值
        'recovery_timeout': 60,      # 恢复超时
        'expected_exception': 'ConnectionError'
    }
    
    # 重试策略
    retry_config = {
        'max_attempts': 3,           # 最大重试次数
        'backoff_factor': 2,         # 退避因子
        'jitter': True               # 随机延迟
    }
    
    # 缓存策略
    cache_config = {
        'default_ttl': 3600,         # 默认过期时间
        'max_memory': '2gb',         # 最大内存使用
        'eviction_policy': 'lru'     # 淘汰策略
    }
```

## 5. 数据存储技术栈

### 5.1 数据库选型

| 数据库 | 版本 | 主要用途 | 数据类型 | 性能特点 |
|--------|------|----------|----------|----------|
| **PostgreSQL** | 15+ | 用户、权限、配置、关系数据 | 结构化数据 | ACID事务，复杂查询，JSON支持 |
| **MongoDB** | 6.0+ | 文档内容、元数据、非结构化数据 | 文档数据 | 水平扩展，灵活schema，聚合查询 |
| **Redis** | 7.0+ | 缓存、会话、消息队列 | 键值数据 | 内存存储，高并发，丰富数据类型 |
| **Elasticsearch** | 8+ | 全文搜索、日志分析、聚合查询 | 搜索数据 | 分布式搜索，实时分析，RESTful API |
| **MinIO** | RELEASE.2023-11+ | 文件存储、图片、文档 | 对象存储 | S3兼容，分布式，高可用 |

### 5.2 数据架构设计

```mermaid
graph TB
    subgraph "应用层 Application Layer"
        AUTH_APP[认证服务]
        USER_APP[用户服务]
        DOC_APP[文档服务]
        SEARCH_APP[搜索服务]
    end
    
    subgraph "缓存层 Cache Layer"
        REDIS_AUTH[认证缓存]
        REDIS_USER[用户缓存]
        REDIS_DOC[文档缓存]
        REDIS_SEARCH[搜索缓存]
    end
    
    subgraph "数据库层 Database Layer"
        PG_USERS[(PostgreSQL<br/>用户表)]
        PG_AUTH[(PostgreSQL<br/>权限表)]
        MONGO_DOCS[(MongoDB<br/>文档集合)]
        ES_INDEX[(Elasticsearch<br/>搜索索引)]
    end
    
    subgraph "存储层 Storage Layer"
        MINIO_FILES[(MinIO<br/>文件存储)]
        MINIO_BACKUP[(MinIO<br/>备份存储)]
    end
    
    AUTH_APP --> REDIS_AUTH
    AUTH_APP --> PG_AUTH
    
    USER_APP --> REDIS_USER
    USER_APP --> PG_USERS
    
    DOC_APP --> REDIS_DOC
    DOC_APP --> MONGO_DOCS
    DOC_APP --> MINIO_FILES
    
    SEARCH_APP --> REDIS_SEARCH
    SEARCH_APP --> ES_INDEX
    
    MONGO_DOCS -.-> MINIO_BACKUP
    PG_USERS -.-> MINIO_BACKUP
```

### 5.3 数据库配置优化

```yaml
# PostgreSQL配置优化
postgresql_config:
  # 连接配置
  max_connections: 200
  shared_buffers: '4GB'
  effective_cache_size: '12GB'
  
  # 性能优化
  work_mem: '256MB'
  maintenance_work_mem: '1GB'
  checkpoint_completion_target: 0.9
  wal_buffers: '64MB'
  
  # 日志配置
  log_statement: 'mod'
  log_min_duration_statement: 1000
  log_checkpoints: 'on'
  
  # 备份配置
  wal_level: 'replica'
  max_wal_senders: 3
  archive_mode: 'on'

# MongoDB配置优化
mongodb_config:
  # 存储引擎
  storage:
    engine: 'wiredTiger'
    wiredTiger:
      engineConfig:
        cacheSizeGB: 8
        journalCompressor: 'snappy'
  
  # 复制集配置
  replication:
    replSetName: 'historical-text-rs'
    
  # 分片配置
  sharding:
    clusterRole: 'shardsvr'
    
  # 索引配置
  operationProfiling:
    slowOpThresholdMs: 100
    mode: 'slowOp'

# Redis配置优化
redis_config:
  # 内存配置
  maxmemory: '8gb'
  maxmemory-policy: 'allkeys-lru'
  
  # 持久化配置
  save: '900 1 300 10 60 10000'
  appendonly: 'yes'
  appendfsync: 'everysec'
  
  # 集群配置
  cluster-enabled: 'yes'
  cluster-config-file: 'nodes.conf'
  cluster-node-timeout: 15000
  
  # 性能优化
  tcp-keepalive: 60
  timeout: 300
  tcp-backlog: 511

# Elasticsearch配置优化
elasticsearch_config:
  # 集群配置
  cluster.name: 'historical-text-cluster'
  node.roles: ['master', 'data', 'ingest']
  
  # 内存配置
  bootstrap.memory_lock: true
  
  # 网络配置
  network.host: '0.0.0.0'
  http.port: 9200
  transport.port: 9300
  
  # 索引配置
  index:
    number_of_shards: 3
    number_of_replicas: 1
    refresh_interval: '30s'
    
  # 搜索配置
  search:
    max_buckets: 65536
    allow_expensive_queries: true
```

## 6. 消息队列和异步处理

### 6.1 消息系统架构

```mermaid
graph LR
    subgraph "生产者 Producers"
        WEB[Web应用]
        API[API服务]
        CRON[定时任务]
    end
    
    subgraph "消息中间件 Message Middleware"
        RABBITMQ[RabbitMQ 3.12+]
        KAFKA[Apache Kafka 3.5+]
    end
    
    subgraph "消费者 Consumers"
        OCR_WORKER[OCR处理器]
        AI_WORKER[AI分析器]
        NOTIFY_WORKER[通知发送器]
        EXPORT_WORKER[导出处理器]
    end
    
    subgraph "任务调度 Task Scheduling"
        CELERY[Celery 5.3+]
        BEAT[Celery Beat]
        FLOWER[Flower监控]
    end
    
    WEB --> RABBITMQ
    API --> RABBITMQ
    CRON --> KAFKA
    
    RABBITMQ --> OCR_WORKER
    RABBITMQ --> AI_WORKER
    RABBITMQ --> NOTIFY_WORKER
    KAFKA --> EXPORT_WORKER
    
    CELERY --> RABBITMQ
    BEAT --> CELERY
    FLOWER --> CELERY
```

### 6.2 消息队列技术选型

| 技术 | 版本 | 使用场景 | 特点 |
|------|------|----------|------|
| **RabbitMQ** | 3.12+ | 任务队列、服务间通信 | 可靠性高、路由灵活、管理界面友好 |
| **Apache Kafka** | 3.5+ | 日志收集、数据流处理 | 高吞吐量、持久化、分区并行 |
| **Redis Stream** | 7.0+ | 轻量级消息、缓存失效通知 | 内存速度、简单易用、持久化可选 |
| **Celery** | 5.3+ | 异步任务处理、定时任务 | Python生态、分布式、监控完善 |

### 6.3 异步任务配置

```python
# Celery配置
from celery import Celery
from kombu import Queue, Exchange

class CeleryConfig:
    """Celery配置类"""
    
    # Broker配置
    broker_url = 'amqp://guest:guest@rabbitmq:5672//'
    result_backend = 'redis://redis:6379/0'
    
    # 任务序列化
    task_serializer = 'json'
    accept_content = ['json']
    result_serializer = 'json'
    timezone = 'Asia/Shanghai'
    enable_utc = True
    
    # 任务路由
    task_routes = {
        'tasks.ocr.*': {'queue': 'ocr_queue'},
        'tasks.ai.*': {'queue': 'ai_queue'},
        'tasks.notify.*': {'queue': 'notify_queue'},
        'tasks.export.*': {'queue': 'export_queue'},
    }
    
    # 队列配置
    task_queues = [
        Queue('ocr_queue', Exchange('ocr'), routing_key='ocr',
              queue_arguments={'x-max-priority': 10}),
        Queue('ai_queue', Exchange('ai'), routing_key='ai',
              queue_arguments={'x-max-priority': 5}),
        Queue('notify_queue', Exchange('notify'), routing_key='notify',
              queue_arguments={'x-max-priority': 3}),
        Queue('export_queue', Exchange('export'), routing_key='export',
              queue_arguments={'x-max-priority': 1}),
    ]
    
    # 工作进程配置
    worker_prefetch_multiplier = 1
    worker_max_tasks_per_child = 1000
    worker_disable_rate_limits = False
    
    # 任务执行配置
    task_acks_late = True
    task_reject_on_worker_lost = True
    task_soft_time_limit = 300  # 5分钟软限制
    task_time_limit = 360       # 6分钟硬限制
    
    # 定时任务配置
    beat_schedule = {
        'cleanup-expired-tokens': {
            'task': 'tasks.auth.cleanup_expired_tokens',
            'schedule': 3600.0,  # 每小时执行
        },
        'generate-daily-report': {
            'task': 'tasks.report.generate_daily_report',
            'schedule': crontab(hour=2, minute=0),  # 每天凌晨2点
        },
        'backup-database': {
            'task': 'tasks.maintenance.backup_database',
            'schedule': crontab(hour=3, minute=0, day_of_week=1),  # 每周一凌晨3点
        }
    }

# 任务定义示例
from celery import Task
from typing import Dict, Any
import traceback

class BaseTask(Task):
    """基础任务类"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败回调"""
        logger.error(f"任务失败: {task_id}, 异常: {exc}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        
        # 发送失败通知
        self.send_failure_notification(task_id, exc, args, kwargs)
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """任务重试回调"""
        logger.warning(f"任务重试: {task_id}, 异常: {exc}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """任务成功回调"""
        logger.info(f"任务完成: {task_id}")

@celery.task(base=BaseTask, bind=True, max_retries=3)
def process_document_async(self, document_id: str) -> Dict[str, Any]:
    """异步处理文档任务
    
    Args:
        document_id: 文档ID
        
    Returns:
        处理结果字典
    """
    try:
        logger.info(f"开始异步处理文档: {document_id}")
        
        # 更新文档状态为处理中
        update_document_status(document_id, "processing")
        
        # 执行OCR识别
        ocr_result = perform_ocr(document_id)
        
        # 执行AI分析
        ai_result = perform_ai_analysis(ocr_result)
        
        # 建立搜索索引
        create_search_index(document_id, ai_result)
        
        # 更新文档状态为完成
        update_document_status(document_id, "completed")
        
        result = {
            "document_id": document_id,
            "status": "completed",
            "ocr_confidence": ocr_result.get("confidence"),
            "ai_tags": ai_result.get("tags"),
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"文档处理完成: {document_id}")
        return result
        
    except Exception as exc:
        logger.error(f"文档处理失败: {document_id}, 错误: {exc}")
        
        # 更新文档状态为失败
        update_document_status(document_id, "failed")
        
        # 指数退避重试
        raise self.retry(
            exc=exc,
            countdown=60 * (2 ** self.request.retries),
            max_retries=3
        )
```

## 7. AI/ML技术栈

### 7.1 AI模型和框架

| 技术 | 版本 | 用途 | 模型类型 |
|------|------|------|----------|
| **Transformers** | 4.35+ | NLP任务、文本分析 | BERT、RoBERTa、GPT系列 |
| **PyTorch** | 2.1+ | 深度学习训练和推理 | 神经网络、Transformer |
| **TensorFlow** | 2.14+ | 机器学习模型部署 | CNN、RNN、LSTM |
| **scikit-learn** | 1.3+ | 传统机器学习 | 分类、聚类、回归 |
| **spaCy** | 3.7+ | NLP流水线处理 | 命名实体识别、词性标注 |
| **OpenCV** | 4.8+ | 图像处理和OCR | 图像预处理、特征提取 |

### 7.2 AI服务架构

```python
# AI模型管理配置
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """模型类型枚举"""
    NER = "named_entity_recognition"      # 命名实体识别
    CLASSIFICATION = "text_classification" # 文本分类
    SENTIMENT = "sentiment_analysis"       # 情感分析
    SUMMARIZATION = "text_summarization"   # 文本摘要
    OCR = "optical_character_recognition"  # 光学字符识别
    LANGUAGE_DETECTION = "language_detection" # 语言检测

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    type: ModelType
    model_path: str
    version: str
    device: str = "cpu"  # cpu or cuda
    batch_size: int = 32
    max_length: int = 512
    confidence_threshold: float = 0.8

class AIModelManager:
    """AI模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.models: Dict[str, Any] = {}
        self.model_configs = {
            # 中文命名实体识别
            "chinese_ner": ModelConfig(
                name="chinese_ner",
                type=ModelType.NER,
                model_path="hfl/chinese-roberta-wwm-ext",
                version="1.0.0",
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            
            # 文本分类
            "text_classifier": ModelConfig(
                name="text_classifier",
                type=ModelType.CLASSIFICATION,
                model_path="bert-base-chinese",
                version="1.0.0",
                batch_size=16
            ),
            
            # 情感分析
            "sentiment_analyzer": ModelConfig(
                name="sentiment_analyzer",
                type=ModelType.SENTIMENT,
                model_path="chinese-roberta-wwm-ext-sentiment",
                version="1.0.0"
            ),
            
            # 语言检测
            "language_detector": ModelConfig(
                name="language_detector",
                type=ModelType.LANGUAGE_DETECTION,
                model_path="custom/langdetect",
                version="1.0.0"
            )
        }
    
    async def load_model(self, model_name: str) -> bool:
        """加载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            加载成功返回True
        """
        if model_name in self.models:
            logger.info(f"模型已加载: {model_name}")
            return True
        
        if model_name not in self.model_configs:
            logger.error(f"未知模型: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        
        try:
            logger.info(f"开始加载模型: {model_name}")
            
            if config.type == ModelType.NER:
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                tokenizer = AutoTokenizer.from_pretrained(config.model_path)
                model = AutoModelForTokenClassification.from_pretrained(config.model_path)
                
                self.models[model_name] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "config": config
                }
            
            elif config.type == ModelType.CLASSIFICATION:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer = AutoTokenizer.from_pretrained(config.model_path)
                model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
                
                self.models[model_name] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "config": config
                }
            
            # 更多模型类型...
            
            logger.info(f"模型加载成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {model_name}, 错误: {e}")
            return False
    
    async def predict(
        self, 
        model_name: str, 
        text: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """模型预测
        
        Args:
            model_name: 模型名称
            text: 输入文本
            options: 可选参数
            
        Returns:
            预测结果
        """
        if model_name not in self.models:
            await self.load_model(model_name)
        
        if model_name not in self.models:
            raise ModelNotLoadedError(f"模型未加载: {model_name}")
        
        model_info = self.models[model_name]
        config = model_info["config"]
        
        try:
            if config.type == ModelType.NER:
                return await self._predict_ner(model_info, text, options)
            elif config.type == ModelType.CLASSIFICATION:
                return await self._predict_classification(model_info, text, options)
            elif config.type == ModelType.SENTIMENT:
                return await self._predict_sentiment(model_info, text, options)
            else:
                raise UnsupportedModelTypeError(f"不支持的模型类型: {config.type}")
                
        except Exception as e:
            logger.error(f"模型预测失败: {model_name}, 错误: {e}")
            raise ModelPredictionError(f"预测失败: {str(e)}")
    
    async def _predict_ner(
        self, 
        model_info: Dict[str, Any], 
        text: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """命名实体识别预测"""
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        config = model_info["config"]
        
        # 分词和编码
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=config.max_length,
            truncation=True,
            padding=True
        )
        
        # 模型预测
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 解析结果
        entities = self._parse_ner_results(
            text, 
            inputs, 
            predictions, 
            tokenizer,
            config.confidence_threshold
        )
        
        return {
            "text": text,
            "entities": entities,
            "model": config.name,
            "confidence_threshold": config.confidence_threshold
        }

# OCR服务配置
class OCRConfig:
    """OCR配置"""
    
    # Tesseract配置
    tesseract_config = {
        'cmd': 'tesseract',
        'lang': 'chi_sim+eng',  # 中文简体+英文
        'oem': 1,               # OCR引擎模式
        'psm': 6,               # 页面分割模式
        'config': '--dpi 300'   # DPI设置
    }
    
    # 图像预处理配置
    image_preprocessing = {
        'resize': True,
        'denoise': True,
        'deskew': True,
        'enhance_contrast': True,
        'remove_background': False
    }
    
    # 后处理配置
    text_postprocessing = {
        'remove_noise': True,
        'correct_spacing': True,
        'normalize_punctuation': True,
        'confidence_filter': 0.6
    }

@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float
    bounding_boxes: List[Dict[str, Any]]
    language: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_boxes": self.bounding_boxes,
            "language": self.language,
            "processing_time": self.processing_time
        }
```

### 7.3 GPU加速配置

```yaml
# GPU计算环境配置
gpu_environment:
  # NVIDIA运行时
  nvidia_docker: true
  cuda_version: "11.8"
  cudnn_version: "8.6"
  
  # PyTorch GPU版本
  pytorch_cuda: "cu118"
  
  # 资源限制
  gpu_memory_fraction: 0.8
  mixed_precision: true
  
  # Kubernetes GPU配置
  kubernetes_gpu:
    resource_limits:
      nvidia.com/gpu: 1
    node_selector:
      accelerator: nvidia-tesla-v100
    
# 模型优化配置
model_optimization:
  # 量化配置
  quantization:
    enabled: true
    method: "dynamic"  # static, dynamic, qat
    
  # 剪枝配置
  pruning:
    enabled: false
    sparsity: 0.5
    
  # 知识蒸馏
  distillation:
    enabled: false
    teacher_model: "bert-large"
    student_model: "bert-base"
    
  # ONNX优化
  onnx_export:
    enabled: true
    opset_version: 11
    dynamic_axes: true
```

## 8. 容器化和编排

### 8.1 Docker容器化

```dockerfile
# 多阶段构建Dockerfile示例
# 阶段1: 基础Python环境
FROM python:3.11-slim as python-base

# 设置Python环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 阶段2: 依赖安装
FROM python-base as dependencies

# 创建工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 阶段3: 应用构建
FROM dependencies as application

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app

# 复制应用代码
COPY --chown=app:app . /app

# 切换到应用用户
USER app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Kubernetes部署配置

```yaml
# 服务部署清单
apiVersion: apps/v1
kind: Deployment
metadata:
  name: historical-text-api
  labels:
    app: historical-text-api
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: historical-text-api
  template:
    metadata:
      labels:
        app: historical-text-api
        version: v1.0.0
    spec:
      containers:
      - name: api
        image: lhgray/historical-projects:api-latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: log-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: api-config
      - name: log-volume
        emptyDir: {}

---
# 服务暴露
apiVersion: v1
kind: Service
metadata:
  name: historical-text-api-service
spec:
  selector:
    app: historical-text-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# 水平扩展配置
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: historical-text-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: historical-text-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# Ingress配置
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: historical-text-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.historical-text.com
    secretName: historical-text-tls
  rules:
  - host: api.historical-text.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: historical-text-api-service
            port:
              number: 80
```

### 8.3 Helm Chart配置

```yaml
# values.yaml - Helm配置文件
global:
  imageRegistry: "lhgray"
  imageTag: "latest"
  pullPolicy: "IfNotPresent"

# 应用配置
app:
  name: "historical-text"
  version: "1.0.0"
  environment: "production"

# 副本配置
replicaCount: 3
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# 资源配置
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"

# 数据库配置
postgresql:
  enabled: true
  auth:
    postgresPassword: "postgres123"
    database: "historical_text"
  primary:
    persistence:
      size: "100Gi"
      storageClass: "fast-ssd"

mongodb:
  enabled: true
  auth:
    enabled: true
    rootPassword: "mongo123"
  persistence:
    size: "200Gi"
    storageClass: "fast-ssd"

redis:
  enabled: true
  auth:
    enabled: true
    password: "redis123"
  master:
    persistence:
      size: "50Gi"

# 监控配置
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: "30s"
  grafanaDashboard:
    enabled: true

# 网络配置
service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
  - host: api.historical-text.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: historical-text-tls
    hosts:
    - api.historical-text.com
```

## 9. 监控和可观测性

### 9.1 监控技术栈

| 组件 | 版本 | 用途 | 特点 |
|------|------|------|------|
| **Prometheus** | 2.47+ | 指标收集和存储 | 时序数据库、PromQL查询、告警规则 |
| **Grafana** | 10.2+ | 可视化面板 | 丰富图表、告警通知、团队协作 |
| **Jaeger** | 1.50+ | 分布式链路追踪 | 请求跟踪、性能分析、依赖关系 |
| **ELK Stack** | 8.10+ | 日志收集和分析 | 日志聚合、全文搜索、实时分析 |
| **AlertManager** | 0.26+ | 告警管理 | 告警聚合、路由分发、静默管理 |

### 9.2 监控架构

```mermaid
graph TB
    subgraph "应用层 Application Layer"
        APP1[认证服务]
        APP2[文档服务]
        APP3[AI服务]
        APP4[搜索服务]
    end
    
    subgraph "指标收集 Metrics Collection"
        PROMETHEUS[Prometheus]
        NODE_EXPORTER[Node Exporter]
        CADVISOR[cAdvisor]
    end
    
    subgraph "日志收集 Log Collection"
        FILEBEAT[Filebeat]
        LOGSTASH[Logstash]
        ELASTICSEARCH[Elasticsearch]
    end
    
    subgraph "链路追踪 Tracing"
        JAEGER_AGENT[Jaeger Agent]
        JAEGER_COLLECTOR[Jaeger Collector]
        JAEGER_QUERY[Jaeger Query]
    end
    
    subgraph "可视化和告警 Visualization & Alerting"
        GRAFANA[Grafana]
        KIBANA[Kibana]
        ALERTMANAGER[AlertManager]
    end
    
    APP1 --> PROMETHEUS
    APP2 --> PROMETHEUS
    APP3 --> PROMETHEUS
    APP4 --> PROMETHEUS
    
    APP1 --> FILEBEAT
    APP2 --> FILEBEAT
    APP3 --> FILEBEAT
    APP4 --> FILEBEAT
    
    APP1 --> JAEGER_AGENT
    APP2 --> JAEGER_AGENT
    APP3 --> JAEGER_AGENT
    APP4 --> JAEGER_AGENT
    
    NODE_EXPORTER --> PROMETHEUS
    CADVISOR --> PROMETHEUS
    
    FILEBEAT --> LOGSTASH
    LOGSTASH --> ELASTICSEARCH
    
    JAEGER_AGENT --> JAEGER_COLLECTOR
    JAEGER_COLLECTOR --> JAEGER_QUERY
    
    PROMETHEUS --> GRAFANA
    PROMETHEUS --> ALERTMANAGER
    ELASTICSEARCH --> KIBANA
    JAEGER_QUERY --> GRAFANA
```

### 9.3 监控配置

```yaml
# Prometheus配置
prometheus_config:
  global:
    scrape_interval: 15s
    evaluation_interval: 15s
  
  rule_files:
    - "alert_rules/*.yml"
  
  alerting:
    alertmanagers:
    - static_configs:
      - targets:
        - alertmanager:9093
  
  scrape_configs:
  - job_name: 'historical-text-api'
    static_configs:
    - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
  
  - job_name: 'node-exporter'
    static_configs:
    - targets: ['node-exporter:9100']
  
  - job_name: 'cadvisor'
    static_configs:
    - targets: ['cadvisor:8080']

# 告警规则
alert_rules:
  groups:
  - name: api_alerts
    rules:
    - alert: HighCPUUsage
      expr: cpu_usage_percent > 80
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "CPU使用率过高"
        description: "{{ $labels.instance }} CPU使用率已超过80%，当前值: {{ $value }}%"
    
    - alert: HighMemoryUsage
      expr: memory_usage_percent > 85
      for: 3m
      labels:
        severity: critical
      annotations:
        summary: "内存使用率过高"
        description: "{{ $labels.instance }} 内存使用率已超过85%，当前值: {{ $value }}%"
    
    - alert: ServiceDown
      expr: up == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "服务不可用"
        description: "{{ $labels.instance }} 服务已下线超过1分钟"
    
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "错误率过高"
        description: "5xx错误率超过10%，当前值: {{ $value }}"

# Grafana仪表板配置
grafana_dashboards:
  - name: "API性能监控"
    panels:
    - title: "请求QPS"
      type: "graph"
      query: "rate(http_requests_total[5m])"
    
    - title: "响应时间"
      type: "graph"  
      query: "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
    
    - title: "错误率"
      type: "singlestat"
      query: "rate(http_requests_total{status=~\"5..\"}[5m])"
    
    - title: "CPU使用率"
      type: "graph"
      query: "cpu_usage_percent"
    
    - title: "内存使用率"
      type: "graph"
      query: "memory_usage_percent"

# ELK配置
elk_config:
  elasticsearch:
    cluster.name: "historical-text-logs"
    node.name: "es-node-1"
    network.host: "0.0.0.0"
    discovery.type: "single-node"
    
  logstash:
    input:
      beats:
        port: 5044
    filter:
      grok:
        match: { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
      date:
        match: [ "timestamp", "ISO8601" ]
    output:
      elasticsearch:
        hosts: ["elasticsearch:9200"]
        index: "historical-text-logs-%{+YYYY.MM.dd}"
  
  kibana:
    server.host: "0.0.0.0"
    elasticsearch.hosts: ["http://elasticsearch:9200"]
```

## 10. 安全架构

### 10.1 安全技术栈

| 组件 | 版本 | 用途 | 安全特性 |
|------|------|------|----------|
| **OAuth 2.0 + JWT** | - | 认证授权 | 无状态令牌、细粒度权限 |
| **Let's Encrypt** | - | SSL证书 | 自动化HTTPS、证书轮换 |
| **HashiCorp Vault** | 1.15+ | 密钥管理 | 动态密钥、审计日志 |
| **OWASP ZAP** | 2.14+ | 安全扫描 | 漏洞检测、安全测试 |
| **Falco** | 0.36+ | 运行时安全 | 异常行为检测、入侵检测 |

### 10.2 安全配置

```yaml
# 安全配置
security_config:
  # 认证配置
  authentication:
    jwt:
      algorithm: "RS256"
      access_token_expire: 900   # 15分钟
      refresh_token_expire: 86400 # 24小时
      issuer: "historical-text-api"
    
    oauth2:
      providers:
        - name: "github"
          client_id: "${GITHUB_CLIENT_ID}"
          client_secret: "${GITHUB_CLIENT_SECRET}"
        - name: "google"
          client_id: "${GOOGLE_CLIENT_ID}"
          client_secret: "${GOOGLE_CLIENT_SECRET}"
  
  # 访问控制
  access_control:
    rbac:
      enabled: true
      roles:
        - name: "admin"
          permissions: ["*"]
        - name: "user"
          permissions: ["read:documents", "write:own_documents"]
        - name: "viewer"
          permissions: ["read:public_documents"]
    
    rate_limiting:
      global: "1000/hour"
      per_user: "100/minute"
      per_ip: "200/minute"
  
  # 数据加密
  encryption:
    at_rest:
      algorithm: "AES-256-GCM"
      key_rotation: "90d"
    in_transit:
      tls_version: "1.3"
      cipher_suites:
        - "TLS_AES_256_GCM_SHA384"
        - "TLS_CHACHA20_POLY1305_SHA256"
  
  # 审计配置
  audit:
    enabled: true
    events:
      - "authentication"
      - "authorization"
      - "data_access"
      - "configuration_changes"
    retention: "2y"
    
  # 安全头配置
  security_headers:
    content_security_policy: "default-src 'self'"
    x_frame_options: "DENY"
    x_content_type_options: "nosniff"
    strict_transport_security: "max-age=31536000; includeSubDomains"
```

## 11. 开发和部署工具

### 11.1 CI/CD工具链

| 工具 | 版本 | 用途 | 特点 |
|------|------|------|------|
| **GitLab CI/CD** | 16.5+ | 持续集成部署 | Git集成、流水线、自动部署 |
| **Docker** | 24.0+ | 容器化 | 轻量级虚拟化、环境一致性 |
| **Kubernetes** | 1.28+ | 容器编排 | 服务发现、负载均衡、滚动更新 |
| **Helm** | 3.13+ | 包管理 | 应用打包、版本管理、模板化 |
| **ArgoCD** | 2.8+ | GitOps部署 | 声明式部署、自动同步 |

### 11.2 开发工具

```yaml
# 开发工具配置
development_tools:
  # 代码质量
  code_quality:
    python:
      formatter: "black==23.11.0"
      import_sorter: "isort==5.12.0"
      linter: "flake8==6.1.0"
      type_checker: "mypy==1.7.1"
      security_scanner: "bandit==1.7.5"
    
    frontend:
      linter: "eslint@8.55.0"
      formatter: "prettier@3.1.0"
      type_checker: "typescript@5.3.0"
  
  # 测试工具
  testing:
    unit_test: "pytest==7.4.3"
    integration_test: "pytest-integration==0.2.3"
    e2e_test: "playwright==1.40.0"
    load_test: "locust==2.17.0"
    api_test: "httpx==0.25.2"
  
  # 文档工具
  documentation:
    api_docs: "OpenAPI 3.0 + Swagger UI"
    code_docs: "Sphinx==7.2.6"
    architecture_docs: "Mermaid + Markdown"
  
  # 数据库工具
  database:
    migration: "Alembic==1.13.0"
    seeding: "Factory Boy==3.3.0"
    monitoring: "pg_stat_statements"
  
  # 性能工具
  performance:
    profiling: "py-spy==0.3.14"
    memory_analysis: "memory-profiler==0.61.0"
    load_testing: "Artillery + K6"

# 环境管理
environment_management:
  development:
    orchestration: "Docker Compose"
    hot_reload: true
    debug_mode: true
    log_level: "DEBUG"
  
  testing:
    orchestration: "Kubernetes (Minikube)"
    test_data: "Fixtures + Factories"
    coverage_threshold: 80
    
  staging:
    orchestration: "Kubernetes"
    data_sync: "Production subset"
    monitoring: "Limited"
    
  production:
    orchestration: "Kubernetes"
    high_availability: true
    monitoring: "Full stack"
    backup_strategy: "3-2-1 rule"
```

## 12. 性能优化技术

### 12.1 缓存策略

```python
# 多层缓存架构
cache_architecture = {
    # L1: 应用内存缓存
    'l1_cache': {
        'type': 'LRU Cache',
        'size': '256MB',
        'ttl': '5m',
        'use_cases': ['热点数据', '计算结果']
    },
    
    # L2: Redis缓存
    'l2_cache': {
        'type': 'Redis Cluster',
        'size': '8GB',
        'ttl': '1h-24h',
        'use_cases': ['会话数据', '频繁查询', 'API响应']
    },
    
    # L3: CDN缓存
    'l3_cache': {
        'type': 'CloudFlare CDN',
        'size': 'Unlimited',
        'ttl': '7d-30d',
        'use_cases': ['静态资源', '公共API响应']
    }
}
```

### 12.2 数据库优化

```sql
-- 索引优化策略
-- 用户表索引
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_users_status ON users(status) WHERE status != 'deleted';
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at DESC);

-- 文档表复合索引
CREATE INDEX CONCURRENTLY idx_documents_user_status ON documents(user_id, status);
CREATE INDEX CONCURRENTLY idx_documents_search ON documents USING GIN(to_tsvector('chinese', title || ' ' || content));

-- 分区表设计
CREATE TABLE document_logs (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- 按月分区
CREATE TABLE document_logs_2024_01 PARTITION OF document_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 连接池配置
connection_pool_config = {
    'min_connections': 5,
    'max_connections': 20,
    'connection_timeout': 30,
    'idle_timeout': 300,
    'max_lifetime': 3600
}
```

### 12.3 异步和并发优化

```python
# 异步优化配置
async_optimization = {
    'event_loop_policy': 'uvloop',  # 高性能事件循环
    'connection_pooling': True,      # 连接池复用
    'batch_processing': True,        # 批量处理
    'streaming_response': True,      # 流式响应
    'worker_processes': 'auto',      # 自动检测CPU核心数
    'max_requests': 10000,          # 每个worker处理的最大请求数
    'max_requests_jitter': 1000,    # 请求数随机抖动
    'graceful_timeout': 120,        # 优雅关闭超时
    'keep_alive': 2                 # HTTP keep-alive 超时
}
```

---

**文档版本**: v1.0  
**最后更新**: 2025-09-03  
**负责人**: 架构师团队  
**审核人**: 技术总监