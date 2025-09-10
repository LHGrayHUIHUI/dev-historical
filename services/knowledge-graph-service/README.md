# 知识图谱构建服务 (Knowledge Graph Service)

历史文本项目的知识图谱构建和查询微服务，专门用于从历史文献中抽取实体、关系并构建结构化的知识图谱。

## 📋 目录

- [服务概览](#服务概览)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [API文档](#api文档)
- [配置说明](#配置说明)
- [部署指南](#部署指南)
- [开发指南](#开发指南)
- [性能优化](#性能优化)
- [监控运维](#监控运维)
- [常见问题](#常见问题)

## 🎯 服务概览

### 核心功能

- **实体抽取**: 支持spaCy、BERT、jieba等多种方法的命名实体识别
- **关系抽取**: 基于规则和机器学习的实体关系发现
- **图谱构建**: 完整的知识图谱构建、优化和质量评估
- **智能查询**: 多种方式的图谱查询和路径发现
- **概念挖掘**: 基于主题模型的概念和主题发现
- **批量处理**: 大规模文档的并行处理和后台任务

### 服务特点

- ✅ **无状态架构**: 所有数据通过storage-service管理，服务可水平扩展
- ✅ **多语言支持**: 专门优化的中英文NLP处理能力
- ✅ **算法多样**: 集成多种NLP模型和图算法
- ✅ **异步处理**: 支持后台任务和批量处理
- ✅ **云原生**: 完整的Docker和Kubernetes支持

### 端口信息

- **开发环境**: http://localhost:8006
- **API文档**: http://localhost:8006/api/v1/docs
- **健康检查**: http://localhost:8006/health

## 🏗️ 技术架构

### 核心技术栈

```yaml
运行时:
  - Python 3.11+
  - FastAPI 0.104+
  - uvicorn (ASGI服务器)

NLP处理:
  - spaCy 3.7+ (中英文NER)
  - transformers 4.35+ (BERT模型)
  - jieba 0.42+ (中文分词)
  - sentence-transformers (句子向量)

图算法:
  - NetworkX 3.2+ (图构建和分析)
  - python-igraph (高性能图算法)
  - community (社区发现)

主题模型:
  - gensim 4.3+ (LDA主题模型)
  - wordcloud (概念可视化)

数据处理:
  - pandas (结构化数据处理)
  - scikit-learn (机器学习)
  - numpy, scipy (数值计算)

HTTP客户端:
  - httpx (异步HTTP客户端)
  - requests (同步HTTP客户端)
```

### 服务架构

```
┌─────────────────────────────────────────────────────┐
│                 Knowledge Graph Service              │
├─────────────────────────────────────────────────────┤
│  Controllers (FastAPI Routes)                       │
│  ├─ /entities/extract     ├─ /concepts/mine        │
│  ├─ /relations/extract    ├─ /batch/process        │
│  ├─ /graph/construct      ├─ /projects/{id}/stats  │
│  └─ /graph/query         └─ /health                │
├─────────────────────────────────────────────────────┤
│  Services (Business Logic)                          │
│  ├─ Entity Extraction    ├─ Concept Mining         │
│  ├─ Relation Extraction  ├─ Graph Construction     │
│  └─ Graph Querying      └─ Batch Processing       │
├─────────────────────────────────────────────────────┤
│  NLP Models & Algorithms                            │
│  ├─ spaCy (zh_core_web_sm)                         │
│  ├─ BERT (bert-base-chinese)                       │
│  ├─ jieba + pseg                                   │
│  ├─ LDA Topic Model                                │
│  └─ NetworkX Graph Algorithms                      │
├─────────────────────────────────────────────────────┤
│  Storage Client (HTTP)                              │
│  └─ Storage Service (8002) ─┐                      │
└─────────────────────────────────┼───────────────────┘
                                  │
                  ┌───────────────▼──────────────┐
                  │     Storage Service          │
                  │  MongoDB + PostgreSQL        │
                  │  + Redis + MinIO             │
                  └──────────────────────────────┘
```

## 🚀 快速开始

### 前置依赖

确保以下服务已运行：
- Storage Service (端口 8002)
- MongoDB (端口 27018)  
- PostgreSQL (端口 5433)
- Redis (端口 6380)

### Docker 启动 (推荐)

```bash
# 1. 克隆项目
cd services/knowledge-graph-service

# 2. 构建镜像
docker build -t knowledge-graph-service .

# 3. 使用docker-compose启动
docker-compose up -d

# 4. 检查服务状态
curl http://localhost:8006/health
```

### 本地开发启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载spaCy中文模型
python -m spacy download zh_core_web_sm

# 3. 设置环境变量
export STORAGE_SERVICE_URL="http://localhost:8002"
export PYTHONPATH="src"

# 4. 启动服务
python -m src.main

# 5. 访问API文档
open http://localhost:8006/api/v1/docs
```

### 快速测试

```bash
# 实体抽取测试
curl -X POST "http://localhost:8006/api/v1/knowledge-graph/entities/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "唐太宗李世民于贞观年间统治唐朝，建立了贞观之治。",
    "method": "spacy",
    "language": "zh"
  }'

# 健康检查
curl http://localhost:8006/health
```

## 📚 API文档

### 核心API端点

#### 1. 实体抽取 API

```http
POST /api/v1/knowledge-graph/entities/extract
Content-Type: application/json

{
  "text": "要处理的文本内容",
  "method": "spacy|bert|jieba|hybrid",
  "language": "zh|en",
  "confidence_threshold": 0.75
}
```

**响应示例:**
```json
{
  "success": true,
  "message": "实体抽取成功",
  "data": {
    "entities": [
      {
        "name": "唐太宗",
        "entity_type": "PERSON",
        "start_pos": 0,
        "end_pos": 3,
        "confidence_score": 0.95,
        "context": "唐太宗李世民于贞观年间"
      }
    ],
    "total_entities": 1,
    "extraction_method": "spacy",
    "processing_time": 0.25
  }
}
```

#### 2. 关系抽取 API

```http
POST /api/v1/knowledge-graph/relations/extract
Content-Type: application/json

{
  "text": "要处理的文本内容",
  "entities": ["实体1", "实体2"],  // 可选，预提供的实体
  "confidence_threshold": 0.7,
  "max_distance": 100,
  "language": "zh"
}
```

#### 3. 图谱构建 API

```http
POST /api/v1/knowledge-graph/graph/construct
Content-Type: application/json

{
  "project_id": "project-123",
  "entities": [...],           // 实体列表
  "relations": [...],          // 关系列表
  "optimize_graph": true,
  "remove_duplicates": true,
  "calculate_centrality": true
}
```

#### 4. 图谱查询 API

```http
POST /api/v1/knowledge-graph/graph/query
Content-Type: application/json

{
  "project_id": "project-123",
  "query_type": "entity|relation|path|neighbors",
  "query_params": {
    "entity_name": "唐太宗",
    "relation_type": "统治"
  },
  "limit": 100,
  "offset": 0
}
```

#### 5. 概念挖掘 API

```http
POST /api/v1/knowledge-graph/concepts/mine
Content-Type: application/json

{
  "documents": ["文档1", "文档2", ...],
  "num_topics": 10,
  "min_frequency": 3,
  "language": "zh"
}
```

#### 6. 批量处理 API

```http
POST /api/v1/knowledge-graph/batch/process
Content-Type: application/json

{
  "documents": ["文档1", "文档2", ...],
  "method": "hybrid",
  "project_id": "project-123"
}
```

### 查询参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|---------|
| `method` | string | 抽取方法: `spacy`, `bert`, `jieba`, `hybrid` | `hybrid` |
| `language` | string | 语言: `zh`, `en` | `zh` |
| `confidence_threshold` | float | 置信度阈值 (0.0-1.0) | `0.75` |
| `max_distance` | int | 关系抽取最大距离(字符数) | `100` |
| `limit` | int | 查询结果限制 | `100` |
| `offset` | int | 查询结果偏移 | `0` |

### 错误处理

所有API都返回统一的错误格式：

```json
{
  "success": false,
  "message": "错误描述信息",
  "data": null,
  "error_code": 400,
  "path": "/api/v1/knowledge-graph/entities/extract",
  "method": "POST"
}
```

## ⚙️ 配置说明

### 环境变量配置

```bash
# 基础服务配置
SERVICE_NAME="knowledge-graph-service"
API_HOST="0.0.0.0"
API_PORT="8006"
ENVIRONMENT="development"
DEBUG="true"

# Storage Service配置
STORAGE_SERVICE_URL="http://localhost:8002"
STORAGE_SERVICE_TIMEOUT="180"
STORAGE_SERVICE_RETRIES="3"

# 知识图谱配置
MAX_TEXT_LENGTH="10000"
MAX_BATCH_SIZE="50"
GRAPH_CONSTRUCTION_TIMEOUT="600"
MAX_CONCURRENT_TASKS="3"

# NLP模型配置
SPACY_MODEL_ZH="zh_core_web_sm"
SPACY_MODEL_EN="en_core_web_sm"
BERT_MODEL_NAME="bert-base-chinese"
SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L6-v2"

# 实体识别配置
ENTITY_CONFIDENCE_THRESHOLD="0.75"
ENTITY_SIMILARITY_THRESHOLD="0.85"
MAX_ENTITY_LENGTH="50"
MIN_ENTITY_LENGTH="2"

# 关系抽取配置
RELATION_CONFIDENCE_THRESHOLD="0.70"
MAX_RELATION_DISTANCE="100"

# 图谱构建配置
GRAPH_MAX_NODES="10000"
GRAPH_MAX_EDGES="50000"
GRAPH_CLUSTERING_THRESHOLD="0.3"

# 概念挖掘配置
TOPIC_MODEL_NUM_TOPICS="20"
TOPIC_MODEL_PASSES="10"
MIN_CONCEPT_FREQUENCY="3"

# 日志配置
LOG_LEVEL="INFO"
LOG_FILE="logs/knowledge_graph_service.log"
```

### 支持的实体类型

```python
PERSON        # 人物 (人名)
LOCATION      # 地点 (地名)  
ORGANIZATION  # 组织 (机构名)
EVENT         # 事件
TIME          # 时间
CONCEPT       # 概念
OBJECT        # 物品
WORK          # 作品
```

### 支持的关系类型

```python
出生于    # BORN_IN
死于      # DIED_IN
任职于    # WORKED_AT
位于      # LOCATED_IN
创建      # FOUNDED
影响      # INFLUENCED
参与      # PARTICIPATED_IN
属于      # BELONGS_TO
统治      # RULED
继承      # INHERITED
师从      # LEARNED_FROM
包含      # CONTAINS
```

## 🚢 部署指南

### Docker部署

```bash
# 1. 构建镜像
docker build -t knowledge-graph-service .

# 2. 运行容器
docker run -d \
  --name knowledge-graph-service \
  -p 8006:8006 \
  -e STORAGE_SERVICE_URL="http://storage-service:8002" \
  -e ENVIRONMENT="production" \
  -v $(pwd)/logs:/app/logs \
  knowledge-graph-service:latest

# 3. 查看日志
docker logs -f knowledge-graph-service
```

### Kubernetes部署

```bash
# 1. 应用配置
kubectl apply -f k8s-deployment.yaml

# 2. 检查部署状态
kubectl get pods -n historical-text -l app=knowledge-graph-service

# 3. 检查服务
kubectl get svc -n historical-text knowledge-graph-service

# 4. 查看日志
kubectl logs -f deployment/knowledge-graph-service -n historical-text

# 5. 端口转发(测试用)
kubectl port-forward svc/knowledge-graph-service 8006:8006 -n historical-text
```

### 生产环境部署清单

- [ ] **依赖服务**: 确保storage-service正常运行
- [ ] **资源配置**: 至少2GB内存，2核CPU
- [ ] **存储配置**: 配置持久化卷用于模型和日志
- [ ] **网络配置**: 配置Ingress或LoadBalancer
- [ ] **监控配置**: 配置Prometheus指标收集
- [ ] **日志配置**: 配置集中日志收集
- [ ] **备份配置**: 配置模型文件备份
- [ ] **扩缩容**: 配置HPA自动扩缩容
- [ ] **安全配置**: 配置网络策略和安全上下文

## 🔧 开发指南

### 开发环境设置

```bash
# 1. 克隆代码
git clone <repository-url>
cd services/knowledge-graph-service

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装开发依赖
pip install pytest pytest-asyncio pytest-cov black isort flake8

# 5. 下载模型
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm

# 6. 运行测试
pytest tests/ -v
```

### 项目结构

```
services/knowledge-graph-service/
├── src/                          # 源代码
│   ├── main.py                   # FastAPI应用入口
│   ├── config/
│   │   └── settings.py           # 配置管理
│   ├── controllers/
│   │   └── knowledge_graph_controller.py  # API控制器
│   ├── services/
│   │   └── knowledge_graph_service.py     # 核心业务逻辑
│   ├── clients/
│   │   └── storage_client.py     # Storage服务客户端
│   └── schemas/
│       └── knowledge_graph_schemas.py     # Pydantic模型
├── tests/                        # 测试代码
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   └── conftest.py              # pytest配置
├── logs/                         # 日志文件
├── temp/                         # 临时文件
├── cache/                        # 缓存文件
├── requirements.txt              # Python依赖
├── Dockerfile                    # Docker镜像
├── docker-compose.yml           # Docker Compose配置
├── k8s-deployment.yaml          # Kubernetes部署配置
└── README.md                    # 项目文档
```

### 代码风格

项目使用以下代码风格工具：

```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 代码检查
flake8 src/ tests/
mypy src/

# 运行所有检查
black src/ && isort src/ && flake8 src/ && mypy src/
```

### 添加新功能

1. **添加API端点**: 在 `controllers/knowledge_graph_controller.py` 中添加新的路由
2. **添加业务逻辑**: 在 `services/knowledge_graph_service.py` 中实现核心逻辑
3. **添加数据模型**: 在 `schemas/knowledge_graph_schemas.py` 中定义Pydantic模型
4. **编写测试**: 在 `tests/` 目录下添加对应的测试
5. **更新文档**: 更新本README和API文档

### 测试指南

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_entity_extraction.py -v

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term

# 运行集成测试(需要外部服务)
pytest tests/integration/ -v
```

## ⚡ 性能优化

### 推荐配置

#### 开发环境
- CPU: 2核
- 内存: 2GB
- 存储: 10GB SSD

#### 生产环境
- CPU: 4核+
- 内存: 4GB+
- 存储: 50GB+ SSD
- 网络: 1Gbps+

### 性能调优参数

```python
# 并发处理
MAX_CONCURRENT_TASKS = 3        # 知识图谱构建并发数
MAX_WORKERS = 4                 # 并行处理工作进程数

# 批量处理
MAX_BATCH_SIZE = 50            # 最大批量处理文档数
CHUNK_SIZE = 1000              # 数据块大小

# 缓存配置
ENABLE_CACHE = True            # 启用本地缓存
CACHE_MAX_SIZE = 1000          # 最大缓存条目数
CACHE_TTL = 3600              # 缓存过期时间(秒)

# 图谱限制
GRAPH_MAX_NODES = 10000       # 单个图谱最大节点数
GRAPH_MAX_EDGES = 50000       # 单个图谱最大边数
```

### 性能监控指标

- **处理延迟**: 各API的平均响应时间
- **吞吐量**: 每秒处理的请求数和文档数
- **资源使用**: CPU、内存、磁盘使用率
- **错误率**: API错误率和超时率
- **模型效率**: NLP模型的推理时间

## 📊 监控运维

### 健康检查端点

```bash
# 基础健康检查
GET /health
{
  "service": "knowledge-graph-service",
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00Z"
}

# 就绪检查
GET /ready
# 检查依赖服务连接状态

# 详细健康检查
GET /api/v1/knowledge-graph/health
# 返回详细的服务状态和依赖检查
```

### 日志配置

日志文件位置: `logs/knowledge_graph_service.log`

日志级别:
- `DEBUG`: 详细调试信息
- `INFO`: 一般操作信息  
- `WARNING`: 警告信息
- `ERROR`: 错误信息

### Prometheus指标

服务暴露以下监控指标 (计划中):

```
# HTTP请求指标
http_requests_total{method, endpoint, status_code}
http_request_duration_seconds{method, endpoint}

# 业务指标  
entities_extracted_total{method, language}
relations_extracted_total{language}
graphs_constructed_total
batch_processing_duration_seconds

# 资源指标
knowledge_graph_cache_size
knowledge_graph_active_tasks
knowledge_graph_model_load_time_seconds
```

### 告警规则 (建议)

```yaml
# 高错误率告警
- alert: KnowledgeGraphHighErrorRate
  expr: rate(http_requests_total{job="knowledge-graph-service",status_code!~"2.."}[5m]) > 0.1
  for: 5m
  annotations:
    summary: "知识图谱服务错误率过高"

# 响应时间告警  
- alert: KnowledgeGraphSlowResponse
  expr: histogram_quantile(0.95, http_request_duration_seconds{job="knowledge-graph-service"}) > 30
  for: 5m
  annotations:
    summary: "知识图谱服务响应过慢"

# 服务不可用告警
- alert: KnowledgeGraphDown
  expr: up{job="knowledge-graph-service"} == 0
  for: 1m
  annotations:
    summary: "知识图谱服务不可用"
```

## ❓ 常见问题

### Q: 服务启动时下载模型很慢怎么办？
A: 可以预先下载模型到本地：
```bash
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```
或者使用已包含模型的Docker镜像。

### Q: 实体抽取准确率不高怎么办？
A: 
1. 尝试使用混合方法 `method=hybrid`
2. 调整置信度阈值 `confidence_threshold`
3. 根据领域特点选择合适的模型

### Q: 图谱构建失败怎么办？
A: 检查以下几点：
1. Storage Service是否正常运行
2. 实体和关系数据是否有效
3. 检查内存使用是否超限
4. 查看详细错误日志

### Q: 批量处理任务卡住怎么办？
A: 
1. 检查任务状态: `GET /api/v1/knowledge-graph/batch/status/{task_id}`
2. 查看服务日志了解具体错误
3. 适当减少批量大小 `MAX_BATCH_SIZE`

### Q: 服务内存使用过高怎么办？
A:
1. 检查是否有内存泄漏
2. 调整缓存大小配置
3. 减少并发任务数 `MAX_CONCURRENT_TASKS`
4. 增加服务器内存配置

### Q: 如何自定义实体类型和关系类型？
A: 修改 `config/settings.py` 中的以下配置：
```python
supported_entity_types = ["PERSON", "LOCATION", ...]
supported_relation_types = ["出生于", "位于", ...]
```

### Q: 如何集成自定义NLP模型？
A: 在 `services/knowledge_graph_service.py` 中扩展抽取方法，添加新的模型加载和推理逻辑。

---

## 📞 技术支持

如有问题或建议，请联系：
- 项目团队: historical-text-team@example.com
- 技术文档: [项目Wiki](https://github.com/yourorg/historical-text-project/wiki)
- 问题反馈: [GitHub Issues](https://github.com/yourorg/historical-text-project/issues)

---

*最后更新: 2024-01-01*