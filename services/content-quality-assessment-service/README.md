# Content Quality Assessment Service

内容质量评估服务 - 智能化多维度内容质量评估平台

## 服务概述

Content Quality Assessment Service是历史文本优化项目的核心AI服务之一，专门负责智能化的内容质量评估。该服务提供全面的多维度质量分析，包括可读性、准确性、完整性、连贯性、相关性等维度的评估，并通过AI大模型提供智能改进建议。

### 核心功能

- **多维度质量评估**: 支持5个核心质量维度的专业评估
- **AI驱动分析**: 集成AI大模型进行智能分析和建议生成
- **质量趋势分析**: 历史数据趋势分析和预测
- **基准管理**: 质量基准设置、对比和合规性检查
- **批量处理**: 支持大规模内容的并行评估
- **智能仪表板**: 综合质量管控和可视化分析

## 评估维度

### 1. 可读性 (Readability)
- **评估内容**: 句子结构、词汇复杂度、语言风格
- **分析方法**: spaCy NLP分析 + 中文可读性算法
- **适用场景**: 教育内容、公共文档、用户手册

### 2. 准确性 (Accuracy)
- **评估内容**: 事实一致性、语法正确性、专业术语使用
- **分析方法**: AI辅助事实检查 + 语法分析
- **适用场景**: 学术论文、历史文档、技术文档

### 3. 完整性 (Completeness)
- **评估内容**: 结构完整性、信息完整性、逻辑完整性
- **分析方法**: 结构分析 + AI内容完整性检查
- **适用场景**: 报告文档、说明书、学术研究

### 4. 连贯性 (Coherence)
- **评估内容**: 逻辑流程、段落衔接、论证一致性
- **分析方法**: 文本相似度计算 + AI连贯性分析
- **适用场景**: 叙述文本、论证文章、长篇内容

### 5. 相关性 (Relevance)
- **评估内容**: 主题相关性、受众适配性、实用价值
- **分析方法**: 主题建模 + AI相关性评估
- **适用场景**: 营销内容、教育材料、专业文档

## 技术架构

### 服务配置
- **端口**: 8012
- **框架**: FastAPI + Python 3.11
- **数据库**: Redis (缓存层，数据库6)
- **外部依赖**: AI Model Service (8008), Storage Service (8002)

### 核心组件

#### 1. 质量评估引擎 (`ContentQualityAssessmentEngine`)
```python
# 主评估引擎，支持多维度并行评估
async def assess_quality(
    request: QualityAssessmentRequest
) -> QualityAssessmentResult
```

#### 2. 趋势分析器 (`QualityTrendAnalyzer`)
```python
# 历史趋势分析和预测
async def analyze_quality_trend(
    content_id: str,
    start_date: datetime,
    end_date: datetime
) -> QualityTrendAnalysis
```

#### 3. 基准管理器 (`QualityBenchmarkManager`)
```python
# 质量基准管理和对比分析
async def compare_with_benchmark(
    assessment_result: QualityAssessmentResult,
    benchmark_id: str
) -> BenchmarkComparison
```

#### 4. 外部服务集成
- **Storage Client**: 数据存储和历史查询
- **AI Service Client**: AI模型调用和智能分析

### NLP处理能力

#### 中文文本处理
- **分词工具**: jieba 0.42.1
- **语言模型**: spaCy zh_core_web_sm
- **特征提取**: TF-IDF、词性标注、实体识别
- **相似度计算**: 余弦相似度、文本向量化

#### AI模型集成
- **文本分析**: 基于Transformer的语义理解
- **质量评估**: 多模型融合的质量打分
- **建议生成**: 基于GPT的改进建议生成

## API接口

### 基础评估接口

#### 单次质量评估
```http
POST /api/v1/quality/assess
Content-Type: application/json

{
  "content": "待评估的文本内容...",
  "content_type": "historical_document",
  "content_id": "content_123",
  "target_audience": "学术研究者",
  "custom_weights": {
    "readability": 0.2,
    "accuracy": 0.3,
    "completeness": 0.2,
    "coherence": 0.2,
    "relevance": 0.1
  }
}
```

#### 批量质量评估
```http
POST /api/v1/quality/batch-assess
Content-Type: application/json

{
  "requests": [...],
  "parallel_processing": true,
  "max_concurrent": 10,
  "timeout_minutes": 30
}
```

#### 获取评估结果
```http
GET /api/v1/quality/assessment/{assessment_id}
```

### 趋势分析接口

#### 质量趋势分析
```http
POST /api/v1/quality/trend-analysis
Content-Type: application/json

{
  "content_id": "content_123",
  "start_date": "2024-01-01T00:00:00",
  "end_date": "2024-12-31T23:59:59"
}
```

#### 历史趋势查询
```http
GET /api/v1/quality/content/{content_id}/trend-analyses?limit=10&offset=0
```

### 基准管理接口

#### 创建质量基准
```http
POST /api/v1/quality/benchmarks
Content-Type: application/json

{
  "name": "历史文档标准基准",
  "content_type": "historical_document",
  "target_audience": "学术研究者",
  "dimension_standards": {
    "readability": 75.0,
    "accuracy": 85.0,
    "completeness": 80.0,
    "coherence": 80.0,
    "relevance": 85.0
  },
  "overall_standard": 80.0
}
```

#### 基准对比分析
```http
POST /api/v1/quality/compare-benchmark
Content-Type: application/json

{
  "assessment_id": "qa_123456789",
  "benchmark_id": "benchmark_789"
}
```

### 仪表板接口

#### 质量仪表板
```http
GET /api/v1/quality/dashboard/{content_id}?period_days=30
```

#### 质量统计
```http
GET /api/v1/quality/statistics?content_type=historical_document&start_date=2024-01-01
```

### 系统接口

#### 健康检查
```http
GET /api/v1/quality/health
```

#### 服务信息
```http
GET /api/v1/quality/info
```

## 部署配置

### Docker部署
```bash
# 构建镜像
docker build -t content-quality-assessment-service .

# 运行容器
docker run -d \
  --name quality-assessment \
  -p 8012:8012 \
  -e ENVIRONMENT=development \
  content-quality-assessment-service
```

### Docker Compose
```yaml
content-quality-assessment-service:
  build: ./services/content-quality-assessment-service
  ports:
    - "8012:8012"
  environment:
    - ENVIRONMENT=development
    - STORAGE_SERVICE_URL=http://storage-service:8002
    - AI_MODEL_SERVICE_URL=http://ai-model-service:8008
    - REDIS_URL=redis://redis:6379/6
  depends_on:
    - storage-service
    - ai-model-service
    - redis
```

### 环境变量

#### 服务配置
- `ENVIRONMENT`: 运行环境 (development/production)
- `HOST`: 服务主机 (默认: 0.0.0.0)
- `PORT`: 服务端口 (默认: 8012)
- `DEBUG`: 调试模式 (默认: false)

#### 外部服务
- `STORAGE_SERVICE_URL`: 存储服务地址
- `AI_MODEL_SERVICE_URL`: AI模型服务地址
- `REDIS_URL`: Redis连接地址

#### 评估引擎配置
- `ENABLED_DIMENSIONS`: 启用的评估维度
- `MAX_CONTENT_LENGTH`: 最大内容长度
- `ASSESSMENT_TIMEOUT`: 评估超时时间
- `CACHE_ASSESSMENT_RESULTS`: 是否缓存评估结果

#### NLP模型配置
- `SPACY_MODEL`: spaCy模型名称
- `MODEL_CACHE_DIR`: 模型缓存目录
- `NLP_MAX_WORKERS`: NLP处理线程数

## 质量评估标准

### 评分标准

#### 评分范围
- **评分区间**: 0-100分
- **等级划分**: A(90-100), B(80-89), C(70-79), D(60-69), F(0-59)

#### 维度权重
不同内容类型的默认权重配置：

**历史文档**:
- 准确性: 30%, 完整性: 25%, 连贯性: 20%, 相关性: 15%, 可读性: 10%

**学术论文**:
- 准确性: 35%, 连贯性: 25%, 完整性: 20%, 相关性: 15%, 可读性: 5%

**教育内容**:
- 可读性: 30%, 准确性: 25%, 相关性: 20%, 完整性: 15%, 连贯性: 10%

### 质量基准

#### 默认基准
系统为每种内容类型提供默认质量基准：

| 内容类型 | 可读性 | 准确性 | 完整性 | 连贯性 | 相关性 | 综合标准 |
|----------|--------|--------|--------|--------|--------|----------|
| 历史文档 | 75 | 85 | 80 | 80 | 85 | 80 |
| 学术论文 | 70 | 90 | 85 | 85 | 90 | 85 |
| 教育内容 | 85 | 85 | 80 | 85 | 80 | 82 |

#### 自定义基准
- 支持创建最多100个自定义基准
- 基准自动验证和一致性检查
- 基准使用统计和性能分析

## 性能指标

### 目标性能
- **单次评估**: < 5秒 (1000字以内)
- **批量处理**: 支持50+并发评估
- **趋势分析**: < 3秒 (30天数据)
- **基准对比**: < 1秒

### 资源配置
- **内存**: 1-2GB (含NLP模型)
- **CPU**: 2-4核
- **存储**: 临时文件和模型缓存

### 缓存策略
- **评估结果**: 24小时TTL
- **趋势分析**: 1小时TTL
- **基准数据**: 长期缓存
- **NLP处理**: 内存缓存

## 监控和运维

### 健康检查
```bash
# 基础健康检查
curl http://localhost:8012/health

# 详细健康检查
curl http://localhost:8012/api/v1/quality/health

# 就绪检查 (Kubernetes)
curl http://localhost:8012/ready
```

### 性能监控
```bash
# 性能指标
curl http://localhost:8012/metrics

# 服务信息
curl http://localhost:8012/info
```

### 日志配置
- **开发环境**: DEBUG级别，详细日志
- **生产环境**: INFO级别，结构化JSON日志
- **错误追踪**: 包含请求ID的分布式追踪

### 监控指标
- **评估请求**: 成功率、响应时间、并发数
- **质量分数**: 平均分、分布统计、趋势变化
- **系统资源**: CPU、内存、磁盘使用率
- **依赖服务**: 外部服务可用性和响应时间

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
├── main.py                     # 应用入口
├── config/
│   └── settings.py             # 配置管理
├── models/
│   └── assessment_models.py    # 数据模型
├── controllers/
│   └── assessment_controller.py # API控制器
├── services/
│   ├── assessment_engine.py    # 评估引擎
│   ├── trend_analyzer.py       # 趋势分析器
│   └── benchmark_manager.py    # 基准管理器
└── clients/
    ├── storage_client.py       # 存储服务客户端
    └── ai_service_client.py    # AI服务客户端
```

## 故障排除

### 常见问题

#### 1. NLP模型加载失败
```bash
# 重新下载spaCy模型
python -m spacy download zh_core_web_sm

# 检查模型文件
python -c "import spacy; nlp = spacy.load('zh_core_web_sm'); print('模型加载成功')"
```

#### 2. 评估超时
- 检查内容长度是否超过限制
- 调整ASSESSMENT_TIMEOUT配置
- 优化NLP处理参数

#### 3. Redis连接失败
```bash
# 检查Redis服务
redis-cli -h redis ping

# 检查连接配置
echo $REDIS_URL
```

#### 4. AI服务调用失败
```bash
# 检查AI服务状态
curl http://ai-model-service:8008/health

# 检查网络连接
ping ai-model-service
```

### 性能优化

1. **缓存优化**: 启用Redis缓存，合理设置TTL
2. **并发控制**: 调整max_concurrent_assessments配置
3. **模型优化**: 使用更轻量的NLP模型
4. **批量处理**: 利用并行评估提高吞吐量

### 错误码说明

- **400**: 请求参数错误
- **404**: 资源未找到  
- **408**: 评估超时
- **429**: 请求频率过高
- **500**: 内部服务错误
- **503**: 服务不可用

## 版本历史

- **v1.0.0**: 初始版本，支持5维度质量评估和完整功能
- 多维度智能质量评估
- AI驱动的分析和建议
- 质量趋势分析和预测
- 基准管理和对比
- 批量处理和并发评估
- 缓存优化和性能监控