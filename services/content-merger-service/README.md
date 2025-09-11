# Content Merger Service

多内容智能合并服务 - 基于AI的历史文本内容合并平台

## 服务概述

Content Merger Service是历史文本优化项目的核心AI服务之一，专门负责多个内容源的智能合并。该服务提供5种不同的合并策略，能够根据内容特点自动选择最优的合并方式，并通过AI大模型进行智能内容生成和优化。

### 核心功能

- **多策略合并**: 支持5种专业合并策略
- **AI驱动**: 集成AI大模型进行智能内容生成
- **质量评估**: 多维度内容质量评估体系
- **异步处理**: 支持后台任务处理和批量合并
- **关系分析**: 智能内容关系分析和相似度计算

## 合并策略

### 1. 时间线整合 (Timeline)
- **适用场景**: 历史事件、时间序列内容
- **特点**: 按时间顺序整合，保持时间逻辑
- **AI温度**: 0.2 (保守，确保时间准确性)

### 2. 主题归并 (Topic)
- **适用场景**: 相关主题内容、专题文章
- **特点**: 按主题分类合并，突出主题完整性
- **AI温度**: 0.3 (平衡创新与准确性)

### 3. 层次组织 (Hierarchy)
- **适用场景**: 结构化内容、分级信息
- **特点**: 按重要性层次组织，突出核心信息
- **AI温度**: 0.2 (保守，确保逻辑性)

### 4. 逻辑关系 (Logic)
- **适用场景**: 论证性内容、分析文章
- **特点**: 构建逻辑关系，强调因果推理
- **AI温度**: 0.1 (最保守，确保逻辑严密)

### 5. 补充扩展 (Supplement)
- **适用场景**: 内容补充、知识扩展
- **特点**: 用相关内容补充主要内容
- **AI温度**: 0.4 (更具创新性)

## 技术架构

### 服务配置
- **端口**: 8011
- **框架**: FastAPI + Python 3.11
- **数据库**: Redis (缓存层，数据库5)
- **外部依赖**: AI Model Service (8008), Storage Service (8002)

### 核心组件

#### 1. 内容合并引擎 (`ContentMergerEngine`)
```python
# 主合并引擎，协调各种合并策略
async def merge_contents(
    source_contents: List[str],
    strategy: MergeStrategy,
    mode: MergeMode,
    config: MergeConfig
) -> MergeResult
```

#### 2. 内容分析器 (`ContentAnalyzer`)
```python
# 内容特征提取和分析
async def analyze_content(content: str) -> ContentAnalysis:
    # 主题提取、实体识别、时间信息分析、关键点提取
```

#### 3. 质量评估器 (`QualityAssessor`)
```python
# 多维度质量评估
async def assess_quality(
    original_contents: List[str],
    merged_content: str,
    strategy: MergeStrategy
) -> QualityMetrics
```

### 质量评估维度

1. **一致性 (Consistency)**: 内容逻辑一致性
2. **完整性 (Completeness)**: 信息保留完整度
3. **流畅性 (Fluency)**: 语言表达流畅度
4. **原创性 (Originality)**: 避免简单复制
5. **事实准确性 (Factual Accuracy)**: 历史事实准确性

## API接口

### 基础合并接口

#### 创建合并任务
```http
POST /api/v1/merger/create
Content-Type: application/json

{
  "source_contents": ["内容1", "内容2", "内容3"],
  "strategy": "timeline",
  "mode": "comprehensive",
  "target_length": 2000,
  "target_style": "academic",
  "instructions": "保持学术风格"
}
```

#### 查询任务状态
```http
GET /api/v1/merger/tasks/{task_id}/status
```

#### 获取合并结果
```http
GET /api/v1/merger/tasks/{task_id}/result
```

### 高级功能接口

#### 批量合并
```http
POST /api/v1/merger/batch
Content-Type: application/json

{
  "merge_requests": [...],
  "parallel": true
}
```

#### 内容关系分析
```http
POST /api/v1/merger/analyze-relationships
Content-Type: application/json

{
  "contents": ["内容1", "内容2"]
}
```

#### 合并预览
```http
POST /api/v1/merger/preview
Content-Type: application/json

{
  "source_contents": [...],
  "strategy": "topic",
  "preview_length": 500
}
```

## 部署配置

### Docker部署
```bash
# 构建镜像
docker build -t content-merger-service .

# 运行容器
docker run -d \
  --name content-merger \
  -p 8011:8011 \
  -e ENVIRONMENT=development \
  content-merger-service
```

### Docker Compose
```yaml
content-merger-service:
  build: ./services/content-merger-service
  ports:
    - "8011:8011"
  environment:
    - ENVIRONMENT=development
    - STORAGE_SERVICE_URL=http://storage-service:8002
    - AI_MODEL_SERVICE_URL=http://ai-model-service:8008
  depends_on:
    - storage-service
    - ai-model-service
    - redis
```

### 环境变量

#### 服务配置
- `ENVIRONMENT`: 运行环境 (development/production)
- `HOST`: 服务主机 (默认: 0.0.0.0)
- `PORT`: 服务端口 (默认: 8011)
- `DEBUG`: 调试模式 (默认: false)

#### 外部服务
- `STORAGE_SERVICE_URL`: 存储服务地址
- `AI_MODEL_SERVICE_URL`: AI模型服务地址
- `REDIS_URL`: Redis连接地址

#### AI模型配置
- `DEFAULT_MODEL`: 默认AI模型
- `DEFAULT_TEMPERATURE`: 默认温度参数
- `DEFAULT_MAX_TOKENS`: 默认最大token数

## 监控和运维

### 健康检查
```bash
# 基础健康检查
curl http://localhost:8011/health

# 就绪检查 (Kubernetes)
curl http://localhost:8011/ready

# 服务信息
curl http://localhost:8011/info
```

### 日志配置
- **开发环境**: DEBUG级别，详细日志
- **生产环境**: INFO级别，结构化JSON日志
- **错误追踪**: 包含请求ID的分布式追踪

### 性能监控
- **指标收集**: Prometheus metrics at `/metrics`
- **请求追踪**: 每个请求包含处理时间
- **缓存监控**: Redis缓存命中率和性能

## 开发指南

### 本地开发
```bash
# 安装依赖
pip install -r requirements.txt

# 下载NLP模型
python -m spacy download zh_core_web_sm

# 启动服务
python -m src.main
```

### 测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/
pytest tests/integration/

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 代码结构
```
src/
├── main.py                 # 应用入口
├── config/
│   └── settings.py         # 配置管理
├── models/
│   └── merger_models.py    # 数据模型
├── controllers/
│   └── merger_controller.py # API控制器
├── services/
│   ├── content_merger_engine.py    # 合并引擎
│   ├── content_analyzer.py         # 内容分析器
│   └── quality_assessor.py         # 质量评估器
└── clients/
    ├── storage_client.py            # 存储服务客户端
    └── ai_service_client.py         # AI服务客户端
```

## 故障排除

### 常见问题

#### 1. AI服务连接失败
```bash
# 检查AI服务状态
curl http://ai-model-service:8008/health

# 检查网络连接
ping ai-model-service
```

#### 2. 存储服务连接失败
```bash
# 检查存储服务状态
curl http://storage-service:8002/health

# 检查数据库连接
redis-cli -h redis ping
```

#### 3. 内存使用过高
- 检查NLP模型加载
- 优化批量处理大小
- 调整缓存配置

#### 4. 合并质量差
- 调整AI模型参数
- 优化提示语模板
- 检查输入内容质量

### 性能优化

1. **缓存优化**: 利用Redis缓存分析结果
2. **并行处理**: 使用异步处理提高并发
3. **模型优化**: 选择合适的AI模型和参数
4. **内存管理**: 优化大文本处理和NLP模型加载

## 版本历史

- **v1.0.0**: 初始版本，支持5种合并策略和完整API
- 支持多策略智能合并
- 集成AI大模型服务
- 完整的质量评估体系
- 异步任务处理
- 容器化部署支持