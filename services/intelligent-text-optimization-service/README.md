# 智能文本优化服务 (Intelligent Text Optimization Service)

## 概述

智能文本优化服务是历史文本项目的核心组件，专门负责对历史文献进行AI驱动的智能优化处理。支持文本润色、内容扩展、风格转换和现代化改写等多种优化模式。

## 核心功能

### 1. 文本优化类型
- **文本润色 (Polish)**: 改善语言表达流畅性和准确性
- **内容扩展 (Expand)**: 基于历史背景增加相关细节描述  
- **风格转换 (Style Convert)**: 转换为不同历史时期的文体风格
- **现代化改写 (Modernize)**: 将古文转换为现代汉语表达

### 2. 优化模式
- **历史文档格式 (Historical Format)**: 按史书体例重新组织
- **学术规范 (Academic)**: 遵循现代学术写作规范
- **文学性 (Literary)**: 增强文本的文学表达力
- **简化表达 (Simplified)**: 简化为通俗易懂的表达

### 3. 质量评估体系
- **多维度评估**: 可读性、学术性、历史准确性、语言质量、结构质量
- **客观指标**: BLEU、ROUGE等量化评估指标
- **改进建议**: 自动生成优化建议和改进方向

### 4. 批量处理能力
- **异步处理**: 支持大规模文档批量优化
- **进度监控**: 实时跟踪处理进度和状态
- **错误处理**: 失败重试机制和错误统计

### 5. 版本管理
- **多版本生成**: 为每个优化任务生成多个版本
- **版本对比**: 提供版本差异分析和对比
- **版本选择**: 智能推荐和手动选择功能

## 技术架构

### 服务架构
```
┌─────────────────────────────────────────┐
│              API接口层                   │
│    (FastAPI + REST API)                │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│              业务逻辑层                   │
│  ┌─────────┬─────────┬─────────┬───────┐  │
│  │优化引擎  │质量评估  │策略管理  │批量处理│  │
│  └─────────┴─────────┴─────────┴───────┘  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│              外部服务层                   │
│  ┌─────────┬─────────────┬─────────────┐  │
│  │AI模型   │Storage      │Redis缓存    │  │
│  │服务     │Service      │             │  │
│  └─────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────┘
```

### 核心组件

#### 文本优化引擎 (TextOptimizationEngine)
- **文本分析**: 使用jieba、spaCy等NLP工具进行文本特征分析
- **AI模型调用**: 集成多种AI模型执行文本优化
- **结果处理**: 处理AI响应并生成结构化结果

#### 质量评估器 (QualityAssessor)  
- **可读性分析**: 评估文本的可读性和理解难度
- **学术质量**: 评估用词规范性和学术严谨性
- **历史准确性**: 检查历史信息的保持情况
- **综合评分**: 根据优化类型计算加权综合分数

#### 策略管理器 (OptimizationStrategyManager)
- **策略加载**: 从配置中加载优化策略
- **智能选择**: 基于文本特征自动选择最佳策略  
- **性能统计**: 跟踪策略使用效果和成功率

#### 批量处理管理器 (BatchOptimizationManager)
- **任务调度**: 管理批量优化任务的生命周期
- **并发控制**: 控制并发任务数量和资源使用
- **进度监控**: 实时跟踪处理进度和状态

## API接口

### 基础API

#### 单文档优化
```http
POST /api/v1/optimization/optimize
Content-Type: application/json

{
    "content": "朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州...",
    "optimization_type": "polish",
    "optimization_mode": "historical_format", 
    "parameters": {
        "target_length": null,
        "preserve_entities": true,
        "quality_threshold": 85.0,
        "custom_instructions": "请保持史书体例的庄重感"
    },
    "generate_versions": 3
}
```

#### 批量优化
```http
POST /api/v1/optimization/batch

{
    "job_name": "史书文档批量优化",
    "document_ids": ["doc1", "doc2", "doc3"],
    "optimization_config": {
        "optimization_type": "polish",
        "optimization_mode": "academic",
        "parameters": {
            "quality_threshold": 80.0
        }
    },
    "parallel_processing": true,
    "max_concurrent_tasks": 5
}
```

#### 任务状态查询
```http
GET /api/v1/optimization/batch/{job_id}/status
```

#### 版本管理
```http
GET /api/v1/optimization/tasks/{task_id}/versions
GET /api/v1/optimization/compare?version1=v1&version2=v2
POST /api/v1/optimization/tasks/{task_id}/select-version
```

### 健康检查
```http
GET /health
GET /api/v1/optimization/health
```

## 环境配置

### 必需环境变量
```bash
# 服务配置
SERVICE_NAME=intelligent-text-optimization-service
SERVICE_VERSION=1.0.0
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8009
SERVICE_ENVIRONMENT=production

# 外部服务配置
STORAGE_SERVICE_URL=http://storage-service:8000
AI_MODEL_SERVICE_URL=http://ai-model-service:8000
REDIS_URL=redis://redis:6379/2

# 优化配置  
MAX_CONTENT_LENGTH=100000
MAX_BATCH_SIZE=1000
OPTIMIZATION_TIMEOUT=180
QUALITY_ASSESSMENT_ENABLED=true
MIN_QUALITY_SCORE=70.0

# AI模型配置
DEFAULT_OPTIMIZATION_MODE=historical_format
MAX_VERSIONS_PER_TASK=5
ENABLE_PARALLEL_OPTIMIZATION=true
```

### 可选环境变量
```bash
# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT={time} | {level} | {message}

# 性能配置
RATE_LIMIT_PER_MINUTE=60
CONCURRENT_OPTIMIZATION_LIMIT=10
REDIS_DEFAULT_TTL=3600

# NLP模型配置
JIEBA_DICT_PATH=/path/to/custom/dict.txt
SPACY_MODEL=zh_core_web_sm
MODEL_CACHE_DIR=/tmp/models

# 开发配置
DEBUG=false
ENABLE_CORS=true
TEST_MODE=false
MOCK_AI_RESPONSES=false
```

## 部署指南

### Docker部署
```bash
# 构建镜像
docker build -t intelligent-text-optimization-service .

# 运行容器
docker run -d \
  --name text-optimization \
  -p 8009:8009 \
  -e STORAGE_SERVICE_URL=http://storage-service:8000 \
  -e AI_MODEL_SERVICE_URL=http://ai-model-service:8000 \
  -e REDIS_URL=redis://redis:6379/2 \
  intelligent-text-optimization-service
```

### Docker Compose集成
```yaml
intelligent-text-optimization-service:
  image: intelligent-text-optimization-service:latest
  container_name: text-optimization-service
  ports:
    - "8009:8009"
  environment:
    - SERVICE_ENVIRONMENT=production
    - STORAGE_SERVICE_URL=http://storage-service:8000
    - AI_MODEL_SERVICE_URL=http://ai-model-service:8000
    - REDIS_URL=redis://redis:6379/2
  depends_on:
    - storage-service
    - ai-model-service
    - redis
  restart: unless-stopped
```

### 开发环境启动
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m src.main

# 或使用uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8009 --reload
```

## 测试

### 运行测试
```bash
# 安装测试依赖
pip install pytest pytest-asyncio pytest-mock

# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_text_optimization_integration.py -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term
```

### 集成测试
```bash
# 启动依赖服务
docker-compose up -d storage-service ai-model-service redis

# 运行集成测试
pytest tests/test_text_optimization_integration.py::TestIntegrationFlow -v
```

## 性能指标

### 性能目标
- **单文档优化**: < 3秒
- **批量任务启动**: < 1秒  
- **质量评估**: < 0.5秒
- **API响应时间**: < 500ms

### 处理能力
- **支持文档长度**: 最大100,000字符
- **并发处理**: > 50个任务
- **批量处理**: 支持1,000+文档
- **日处理能力**: > 10,000文档

### 质量指标
- **平均质量提升**: > 15分
- **历史准确性**: > 95%
- **语言流畅度**: > 90%
- **系统可用性**: > 99.5%

## 监控和日志

### 健康检查端点
- `GET /health` - 基础健康检查
- `GET /api/v1/optimization/health` - 详细健康状态
- `GET /api/v1/optimization/statistics` - 性能统计

### 日志配置
```python
# 日志格式
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO", 
    "service": "intelligent-text-optimization-service",
    "message": "文本优化完成",
    "context": {
        "task_id": "task-uuid",
        "optimization_type": "polish",
        "quality_score": 88.5,
        "processing_time_ms": 2300
    }
}
```

### 监控指标
- 优化任务成功率
- 平均处理时间
- 质量分数分布
- API响应时间
- 错误率和异常统计

## 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 检查日志
docker logs text-optimization-service

# 检查配置
echo $STORAGE_SERVICE_URL
echo $AI_MODEL_SERVICE_URL

# 验证依赖服务
curl http://storage-service:8000/health
curl http://ai-model-service:8000/health
```

#### 2. 优化质量差
```bash  
# 检查AI模型状态
curl http://ai-model-service:8000/api/v1/models

# 验证策略配置
curl http://localhost:8009/api/v1/optimization/strategies

# 调整质量阈值
export MIN_QUALITY_SCORE=60.0
```

#### 3. 批量处理慢
```bash
# 增加并发任务数
export CONCURRENT_OPTIMIZATION_LIMIT=20

# 检查Redis连接
redis-cli -u $REDIS_URL ping

# 监控资源使用
docker stats text-optimization-service
```

#### 4. 内存占用高
```bash
# 减少并发任务
export MAX_CONCURRENT_TASKS=3

# 清理模型缓存
rm -rf /tmp/models/*

# 重启服务
docker restart text-optimization-service
```

## 开发指南

### 代码结构
```
src/
├── config/          # 配置管理
├── models/          # 数据模型
├── controllers/     # API控制器
├── services/        # 业务逻辑服务
├── clients/         # 外部服务客户端
└── utils/           # 工具函数

tests/
├── unit/            # 单元测试
├── integration/     # 集成测试
└── fixtures/        # 测试数据
```

### 贡献指南
1. Fork项目仓库
2. 创建功能分支
3. 编写测试用例
4. 确保代码质量
5. 提交Pull Request

### 代码规范
- 使用Black格式化代码
- 使用isort整理导入
- 使用mypy进行类型检查
- 维持80%以上测试覆盖率

## 更新日志

### v1.0.0 (2024-01-15)
- 初始版本发布
- 支持4种优化类型和4种优化模式
- 实现多维度质量评估体系
- 提供批量处理和版本管理功能
- 集成AI模型服务和存储服务

## 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 联系方式

- 项目地址: https://github.com/your-org/historical-text-project
- 问题反馈: https://github.com/your-org/historical-text-project/issues
- 技术文档: https://docs.your-org.com/historical-text-project