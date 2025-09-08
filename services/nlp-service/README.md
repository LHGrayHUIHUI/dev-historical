# NLP文本处理服务

无状态NLP文本处理微服务，专为历史文献设计的高精度自然语言处理服务。专注于语言分析算法，数据存储通过storage-service完成。

## 功能特性

### 🎯 核心功能

- **多引擎支持**: 集成spaCy、jieba、HanLP、Transformers等主流NLP框架
- **全面分析**: 分词、词性标注、命名实体识别、情感分析、关键词提取、文本摘要
- **古汉语支持**: 针对古代汉语、繁体字、异体字专门优化
- **异步处理**: 支持大批量文本的异步NLP处理
- **智能预处理**: 自动繁简转换、文本清理、格式标准化
- **相似度计算**: 基于句子嵌入的语义相似度分析

### 🚀 技术特性

- **无状态架构**: 不直接连接数据库，通过storage-service进行数据管理
- **现代架构**: 基于FastAPI + Python 3.11构建
- **高性能**: 异步I/O，支持并发处理和模型缓存
- **水平扩展**: 无状态设计，支持Kubernetes水平扩展
- **云原生**: 完整Docker支持，微服务架构
- **专业分工**: 专注NLP计算，不处理业务逻辑

## 快速开始

### 环境要求

- Python 3.11+
- Storage Service (端口 8002) - 用于数据存储
- Docker & Docker Compose (推荐)

### Docker快速部署（推荐）

```bash
# 克隆代码
git clone <repository-url>
cd services/nlp-service

# 复制环境配置
cp .env.example .env

# 编辑环境变量，配置storage-service地址
vim .env

# 启动NLP服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f nlp-service
```

服务启动后访问：
- API文档: http://localhost:8004/docs
- 健康检查: http://localhost:8004/health
- 服务信息: http://localhost:8004/info

### 本地开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 下载spaCy中文模型（可选）
python -m spacy download zh_core_web_sm

# 配置环境变量
cp .env.example .env
# 编辑.env文件，配置storage-service地址等

# 启动开发服务器
python -m src.main
```

**注意**: 本地开发需要确保storage-service已启动并可访问。

## API使用示例

### 文本分词（同步模式）

```bash
curl -X POST "http://localhost:8004/api/v1/nlp/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "中华人民共和国成立于1949年10月1日",
    "processing_type": "segmentation",
    "language": "zh",
    "engine": "jieba",
    "async_mode": false
  }'
```

### 情感分析

```bash
curl -X POST "http://localhost:8004/api/v1/nlp/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这本历史书写得非常精彩，让人爱不释手！",
    "processing_type": "sentiment",
    "language": "zh",
    "engine": "transformers"
  }'
```

### 关键词提取

```bash
curl -X POST "http://localhost:8004/api/v1/nlp/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "历史文献数字化是保护和传承文化遗产的重要手段...",
    "processing_type": "keywords",
    "config": {
      "method": "textrank",
      "top_k": 10
    }
  }'
```

### 命名实体识别

```bash
curl -X POST "http://localhost:8004/api/v1/nlp/ner" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "明朝永乐年间，郑和下西洋到达了马六甲和斯里兰卡",
    "processing_type": "ner",
    "language": "zh",
    "engine": "spacy"
  }'
```

### 文本摘要

```bash
curl -X POST "http://localhost:8004/api/v1/nlp/summary" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "长篇历史文献内容...",
    "processing_type": "summary",
    "config": {
      "method": "extractive",
      "max_sentences": 3,
      "compression_ratio": 0.3
    }
  }'
```

### 文本相似度计算

```bash
curl -X POST "http://localhost:8004/api/v1/nlp/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "古代中国的科技成就",
    "text2": "中华古代科学技术发展",
    "method": "sentence_transformer",
    "language": "zh"
  }'
```

### 批量处理

```bash
curl -X POST "http://localhost:8004/api/v1/nlp/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "第一段历史文本",
      "第二段历史文本",
      "第三段历史文本"
    ],
    "processing_type": "segmentation",
    "language": "zh",
    "engine": "jieba"
  }'
```

### 查询任务状态

```bash
curl -X GET "http://localhost:8004/api/v1/nlp/tasks/{task_id}"
```

### 获取可用引擎

```bash
curl -X GET "http://localhost:8004/api/v1/nlp/engines"
```

## 项目结构

```
services/nlp-service/
├── src/                        # 源代码目录
│   ├── config/                 # 配置管理
│   │   └── settings.py         # 应用配置（无状态）
│   ├── controllers/            # API控制器
│   │   └── nlp_controller.py   # NLP处理接口
│   ├── clients/                # 外部服务客户端
│   │   └── storage_client.py   # Storage服务客户端
│   ├── services/               # 业务逻辑层
│   │   └── nlp_service.py      # NLP服务类（纯计算）
│   ├── schemas/                # Pydantic模型
│   │   └── nlp_schemas.py      # NLP相关模型
│   └── main.py                 # 应用入口点
├── tests/                      # 测试代码
├── temp/                       # 临时文件（仅此目录）
├── models/                     # NLP模型缓存
├── dictionaries/               # 自定义词典
├── docker-compose.yml          # Docker编排
├── Dockerfile                  # Docker镜像
├── requirements.txt            # Python依赖（精简版）
├── .env.example               # 环境变量示例
└── README.md                  # 项目文档
```

### 架构特点

- **无数据库层**: 移除了直接数据库连接，通过HTTP与storage-service通信
- **服务客户端**: 新增 `clients/` 目录，处理与其他微服务的通信
- **纯计算服务**: `services/` 专注NLP算法，不处理数据持久化
- **精简配置**: 配置文件仅包含NLP引擎和服务通信设置

## 配置说明

### 环境变量

主要配置项（完整列表见`.env.example`）：

```bash
# 服务配置
NLP_SERVICE_NAME=nlp-service
NLP_SERVICE_VERSION=1.0.0
NLP_SERVICE_ENVIRONMENT=development
NLP_SERVICE_API_HOST=0.0.0.0
NLP_SERVICE_API_PORT=8004

# Storage Service配置（必需）
NLP_SERVICE_STORAGE_SERVICE_URL=http://localhost:8002
NLP_SERVICE_STORAGE_SERVICE_TIMEOUT=60
NLP_SERVICE_STORAGE_SERVICE_RETRIES=3

# NLP引擎配置
NLP_SERVICE_DEFAULT_NLP_ENGINE=spacy
NLP_SERVICE_DEFAULT_LANGUAGE=zh
NLP_SERVICE_MAX_TEXT_LENGTH=1000000  # 1MB
NLP_SERVICE_MAX_BATCH_SIZE=50

# 临时文件配置
NLP_SERVICE_TEMP_DIR=/tmp/nlp-service
```

### NLP引擎配置

#### spaCy（推荐用于词性标注和NER）
```python
{
    "model": "zh_core_web_sm",
    "disable": [],  # 可禁用不需要的组件
    "exclude": []   # 可排除特定功能
}
```

#### jieba（推荐用于中文分词）
```python
{
    "enable_parallel": True,
    "parallel_workers": 4,
    "hmm": True
}
```

#### Transformers（推荐用于情感分析）
```python
{
    "sentiment_model": "uer/roberta-base-finetuned-chinanews-chinese",
    "device": -1  # -1 for CPU, 0 for GPU
}
```

#### Sentence Transformers（推荐用于相似度计算）
```python
{
    "sentence_model": "shibing624/text2vec-base-chinese"
}
```

## 支持功能

### 文本处理功能
- **分词 (Segmentation)**: jieba、spaCy、HanLP
- **词性标注 (POS Tagging)**: jieba、spaCy
- **命名实体识别 (NER)**: spaCy、规则匹配
- **情感分析 (Sentiment Analysis)**: Transformers、词典匹配
- **关键词提取 (Keyword Extraction)**: TextRank、TF-IDF、词频统计
- **文本摘要 (Text Summarization)**: 抽取式摘要
- **文本相似度 (Text Similarity)**: 句子嵌入、TF-IDF

### 支持语言
- **简体中文 (zh)**: 主要支持语言
- **English (en)**: 部分功能支持
- **古汉语 (zh-classical)**: 基础支持

### 输入格式
- **文本长度**: 最大1MB（可配置）
- **批量限制**: 50个文本（可配置）
- **字符编码**: UTF-8

### 输出格式
- **JSON响应**: 结构化API响应
- **详细信息**: 词汇位置、置信度、统计信息
- **元数据**: 处理时间、使用引擎、配置参数

## 性能优化

### 模型缓存
- 本地模型文件缓存
- 结果内存缓存（LRU策略）
- 预加载常用模型

### 并发处理
- 异步I/O处理
- 信号量控制并发数
- 批量处理优化

### 文本预处理
- 繁简转换（OpenCC）
- 文本清理和标准化
- 智能语言检测

## 监控与运维

### 健康检查
```bash
# 基础健康检查
curl http://localhost:8004/health

# 详细组件状态
curl http://localhost:8004/api/v1/health/detailed
```

### 性能指标
- 请求响应时间
- 处理成功率
- 错误率统计
- 资源使用情况
- 各引擎使用统计

### 日志管理
```bash
# 查看服务日志
docker-compose logs -f nlp-service

# 查看错误日志
docker-compose logs -f nlp-service | grep ERROR
```

## 开发指南

### 代码规范
- Python代码风格：Black + isort
- 类型注解：完整的类型提示
- 文档字符串：Google风格
- 注释密度：30%+中文注释

### 测试
```bash
# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 添加新的NLP功能

1. **扩展Schema**: 在 `schemas/nlp_schemas.py` 中定义新的请求/响应模型
2. **实现算法**: 在 `services/nlp_service.py` 中添加处理方法
3. **添加接口**: 在 `controllers/nlp_controller.py` 中创建API端点
4. **更新配置**: 在 `config/settings.py` 中添加相关配置
5. **编写测试**: 添加对应的单元测试和集成测试

### 提交规范
```bash
git commit -m "feat(nlp): 添加新的文本分类功能"
git commit -m "fix(api): 修复批量处理内存泄漏问题"
git commit -m "docs: 更新API文档"
```

## 部署指南

### Docker部署
```bash
# 构建镜像
docker build -t nlp-service:latest .

# 运行容器
docker run -d \
  --name nlp-service \
  -p 8004:8004 \
  -e NLP_SERVICE_STORAGE_SERVICE_URL=http://your-storage-service:8002 \
  nlp-service:latest
```

### Kubernetes部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlp-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlp-service
  template:
    metadata:
      labels:
        app: nlp-service
    spec:
      containers:
      - name: nlp-service
        image: nlp-service:latest
        ports:
        - containerPort: 8004
        env:
        - name: NLP_SERVICE_STORAGE_SERVICE_URL
          value: "http://storage-service:8002"
        resources:
          requests:
            memory: "3Gi"
            cpu: "2"
          limits:
            memory: "6Gi"
            cpu: "4"
```

## 故障排除

### 常见问题

1. **NLP模型初始化失败**
   - 检查模型文件是否下载完整
   - 验证网络连接（模型可能需要在线下载）
   - 检查磁盘空间是否充足

2. **Storage服务连接失败**
   - 检查storage-service服务状态
   - 验证服务URL配置
   - 确保网络连通性

3. **处理超时**
   - 调整任务超时时间配置
   - 检查文本长度是否超出限制
   - 考虑使用异步模式处理大文本

4. **内存不足**
   - 增加容器内存限制
   - 减少并发处理任务数
   - 清理模型缓存

5. **处理准确率低**
   - 尝试不同的NLP引擎
   - 调整模型参数
   - 检查文本预处理设置

### 调试模式
```bash
# 启用详细日志
export NLP_SERVICE_LOG_LEVEL=DEBUG

# 启用错误详情
export NLP_SERVICE_DEBUG_SHOW_ERROR_DETAILS=true

# 查看服务健康状态
curl http://localhost:8004/health

# 查看可用NLP引擎
curl http://localhost:8004/api/v1/nlp/engines
```

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新历史。

## 支持

- 📧 邮件: support@example.com
- 🐛 问题报告: [GitHub Issues](https://github.com/your-org/nlp-service/issues)
- 📚 文档: [项目Wiki](https://github.com/your-org/nlp-service/wiki)
- 💬 讨论: [GitHub Discussions](https://github.com/your-org/nlp-service/discussions)