# 内容质量控制服务 (Content Quality Control Service)

## 概述

内容质量控制服务是历史文本项目的核心组件，专门负责对内容进行多维度质量检测、合规性审核和智能工作流管理。

## 核心功能

### 1. 多维度质量检测
- **语法检测**: 检测语法错误、句法结构、标点使用等
- **逻辑分析**: 分析内容的逻辑一致性、时间逻辑、因果关系
- **格式检查**: 验证文档格式、段落结构、标题规范
- **事实验证**: 检查历史事实、数字合理性等
- **学术标准**: 评估学术写作规范性、用词准确性

### 2. 合规性审核
- **敏感词检测**: 基于可配置词库的敏感词汇检测
- **政策合规**: 检查内容是否符合发布政策
- **版权检查**: 评估版权风险，检测引用规范
- **学术诚信**: 检测抄袭、重复内容等学术诚信问题

### 3. 智能审核工作流
- **自动化审核**: 基于质量分数和风险评分的自动审核
- **人工审核分配**: 智能任务分配和优先级管理
- **多级审核流程**: 支持多步骤的审核工作流
- **进度跟踪**: 实时监控审核进度和状态

### 4. 质量改进建议
- **自动修复**: 对可修复问题提供自动修复方案
- **改进建议**: 针对性的质量提升建议
- **版本管理**: 支持多版本内容对比和选择

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
│  │质量检测  │合规检测  │工作流   │批量处理│  │
│  │引擎     │引擎     │管理    │管理器  │  │
│  └─────────┴─────────┴─────────┴───────┘  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│              外部服务层                   │
│  ┌─────────────┬─────────────┬─────────┐  │
│  │Storage      │Redis缓存    │NLP工具  │  │
│  │Service      │             │库       │  │
│  └─────────────┴─────────────┴─────────┘  │
└─────────────────────────────────────────┘
```

### 核心组件

#### 质量检测引擎 (QualityDetectionEngine)
- **多检测器并行**: 语法、逻辑、格式、事实、学术检测器
- **智能评分**: 基于权重的综合质量评分算法
- **自动修复**: 可修复问题的自动修复建议生成

#### 合规检测引擎 (ComplianceEngine)
- **敏感词检测**: 基于词库的模式匹配和上下文分析
- **政策合规**: 多种合规规则的并行检测
- **风险评估**: 综合风险评分和状态判断

#### 审核工作流管理器 (ReviewWorkflowManager)
- **智能任务创建**: 基于检测结果的自动任务创建
- **工作流选择**: 根据内容特征选择合适的审核流程
- **任务分配**: 多种分配策略的智能任务分配

## API接口

### 基础API

#### 质量检测
```http
POST /api/v1/quality/check
Content-Type: application/json

{
    "content": "待检测的文本内容",
    "content_type": "historical_text",
    "check_options": {
        "grammar_check": true,
        "logic_check": true,
        "format_check": true,
        "factual_check": true,
        "academic_check": true
    },
    "auto_fix": true
}
```

#### 合规检测
```http
POST /api/v1/compliance/check
Content-Type: application/json

{
    "content": "待检测的文本内容",
    "check_types": ["sensitive_words", "policy", "copyright", "academic_integrity"],
    "strict_mode": false
}
```

#### 创建审核任务
```http
POST /api/v1/review/tasks
Content-Type: application/json

{
    "content_id": "content_uuid",
    "quality_result": {...},
    "compliance_result": {...},
    "priority": "high"
}
```

#### 综合质量检测
```http
POST /api/v1/review/comprehensive-check
Content-Type: application/json

{
    "content": "待检测的文本内容",
    "content_id": "content_uuid",
    "auto_create_task": true
}
```

### 健康检查
```http
GET /health                    # 基础健康检查
GET /health/detailed          # 详细健康检查
GET /info                     # 服务信息
```

## 环境配置

### 必需环境变量
```bash
# 服务配置
SERVICE_NAME=content-quality-control-service
SERVICE_VERSION=1.0.0
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8010
SERVICE_ENVIRONMENT=production

# 外部服务配置
STORAGE_SERVICE_URL=http://storage-service:8000
STORAGE_SERVICE_TIMEOUT=30
STORAGE_SERVICE_RETRIES=3

# Redis配置
REDIS_URL=redis://redis:6379/4
REDIS_KEY_PREFIX=quality_control
REDIS_DEFAULT_TTL=3600

# 质量检测配置
MAX_CONTENT_LENGTH=100000
QUALITY_CHECK_TIMEOUT=30
AUTO_APPROVAL_THRESHOLD=90.0
HUMAN_REVIEW_THRESHOLD=70.0

# 安全配置
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
JWT_ALGORITHM=HS256
```

### 可选环境变量
```bash
# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT="{time} | {level} | {message}"

# 性能配置
MAX_BATCH_SIZE=100
CONCURRENT_CHECK_LIMIT=10
PARALLEL_DETECTION_ENABLED=true

# NLP模型配置
SPACY_MODEL=zh_core_web_sm
MODEL_CACHE_DIR=/tmp/models

# 开发配置
DEBUG=false
ENABLE_CORS=true
TEST_MODE=false
```

## 部署指南

### Docker部署
```bash
# 构建镜像
docker build -t content-quality-control-service .

# 运行容器
docker run -d \
  --name quality-control \
  -p 8010:8010 \
  -e STORAGE_SERVICE_URL=http://storage-service:8000 \
  -e REDIS_URL=redis://redis:6379/4 \
  content-quality-control-service
```

### Docker Compose集成
```yaml
content-quality-control-service:
  image: content-quality-control-service:latest
  container_name: content-quality-control
  ports:
    - "8010:8010"
  environment:
    - SERVICE_ENVIRONMENT=production
    - STORAGE_SERVICE_URL=http://storage-service:8000
    - REDIS_URL=redis://redis:6379/4
  depends_on:
    - storage-service
    - redis
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8010/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### 开发环境启动
```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m src.main

# 或使用uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8010 --reload
```

## 测试

### 运行测试
```bash
# 安装测试依赖
pip install pytest pytest-asyncio pytest-mock

# 运行所有测试
pytest

# 运行特定测试
pytest tests/unit/test_quality_detection_engine.py -v
pytest tests/integration/test_content_quality_integration.py -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term
```

### 集成测试
```bash
# 启动依赖服务
docker-compose up -d storage-service redis

# 运行集成测试
pytest tests/integration/ -v
```

## 性能指标

### 性能目标
- **质量检测**: < 3秒
- **合规检测**: < 2秒
- **综合检测**: < 5秒
- **API响应**: < 500ms

### 处理能力
- **并发检测**: > 50个任务
- **批量处理**: 支持100+文档
- **日处理能力**: > 5,000文档

### 质量指标
- **检测准确率**: > 90%
- **假阳性率**: < 5%
- **假阴性率**: < 3%
- **系统可用性**: > 99.5%

## 监控和日志

### 健康检查端点
- `GET /health` - 基础健康检查
- `GET /health/detailed` - 详细健康状态
- `GET /info` - 服务信息

### 日志配置
```python
# 结构化日志格式
{
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "service": "content-quality-control-service",
    "message": "质量检测完成",
    "context": {
        "check_id": "check-uuid",
        "overall_score": 85.5,
        "processing_time_ms": 2300,
        "issues_count": 3
    }
}
```

### 监控指标
- 检测任务成功率
- 平均处理时间
- 质量分数分布
- API响应时间
- 错误率和异常统计

## 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 检查日志
docker logs content-quality-control

# 检查配置
echo $STORAGE_SERVICE_URL
echo $REDIS_URL

# 验证依赖服务
curl http://storage-service:8000/health
```

#### 2. 检测性能慢
```bash
# 检查并发限制
export CONCURRENT_CHECK_LIMIT=20

# 检查Redis连接
redis-cli -u $REDIS_URL ping

# 监控资源使用
docker stats content-quality-control
```

#### 3. 检测准确率低
```bash
# 检查NLP模型
python -c "import spacy; nlp = spacy.load('zh_core_web_sm'); print('OK')"

# 更新敏感词库
curl -X POST http://localhost:8010/api/v1/compliance/sensitive-words \
  -H "Content-Type: application/json" \
  -d '{"word": "新敏感词", "category": "test", "severity_level": 5}'

# 调整检测阈值
export AUTO_APPROVAL_THRESHOLD=85.0
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
- 实现多维度质量检测功能
- 实现合规性审核功能
- 实现智能审核工作流
- 集成storage-service
- 提供完整的REST API

## 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 联系方式

- 项目地址: https://github.com/your-org/historical-text-project
- 问题反馈: https://github.com/your-org/historical-text-project/issues
- 技术文档: https://docs.your-org.com/historical-text-project