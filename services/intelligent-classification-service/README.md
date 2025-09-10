# 智能分类服务 (Intelligent Classification Service)

历史文本智能分类微服务，专注于机器学习文档分类算法，支持多种传统和深度学习模型的训练与预测。

## 🚀 服务概述

智能分类服务是历史文本优化项目的核心组件之一，提供无状态的文档分类能力。服务采用现代微服务架构，通过storage-service统一管理数据持久化，专注于算法和模型的实现。

### 主要特性

- ✨ **多算法支持** - SVM、RandomForest、XGBoost、LightGBM、BERT等
- 🌏 **中文优化** - 专门针对古代中文历史文献优化的文本处理
- 📊 **性能监控** - 完整的模型性能跟踪和MLflow实验管理
- 🔄 **批量处理** - 高效的批量文档分类处理
- 📈 **A/B测试** - 支持多模型对比和性能评估
- 🛡️ **无状态架构** - 所有数据通过storage-service管理，易于扩展

### 技术栈

- **框架**: FastAPI + Python 3.11+
- **机器学习**: scikit-learn, XGBoost, LightGBM
- **深度学习**: PyTorch, Transformers, Sentence-Transformers
- **中文NLP**: jieba, spaCy, HanLP
- **实验跟踪**: MLflow
- **容器化**: Docker + Kubernetes

## 📋 目录结构

```
intelligent-classification-service/
├── src/                          # 源代码目录
│   ├── main.py                   # FastAPI应用入口
│   ├── config/                   # 配置管理
│   │   └── settings.py           # 服务配置
│   ├── controllers/              # API控制器
│   │   ├── project_controller.py # 项目管理API
│   │   ├── model_controller.py   # 模型管理API
│   │   ├── classification_controller.py  # 分类API
│   │   └── data_controller.py    # 训练数据API
│   ├── services/                 # 业务逻辑层
│   │   ├── model_trainer.py      # 模型训练服务
│   │   └── classification_service.py  # 分类预测服务
│   ├── utils/                    # 工具模块
│   │   ├── text_preprocessing.py # 文本预处理
│   │   └── feature_extraction.py # 特征提取
│   ├── clients/                  # 外部服务客户端
│   │   └── storage_client.py     # Storage服务客户端
│   └── schemas/                  # 数据模型定义
│       └── classification_schemas.py  # Pydantic模型
├── requirements.txt              # Python依赖
├── Dockerfile                    # Docker构建文件
├── docker-compose.yml           # Docker Compose配置
├── k8s-deployment.yaml          # Kubernetes部署配置
├── .env.example                 # 环境变量示例
└── README.md                    # 本文档
```

## 🛠️ 快速开始

### 环境要求

- Python 3.11+
- Docker (可选)
- 8GB+ 内存（用于深度学习模型）

### 本地开发

1. **克隆代码**
   ```bash
   cd services/intelligent-classification-service
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑.env文件，配置storage-service等依赖服务地址
   ```

4. **启动服务**
   ```bash
   python -m src.main
   ```

5. **查看API文档**
   ```
   http://localhost:8007/api/v1/docs
   ```

### Docker部署

1. **构建镜像**
   ```bash
   docker build -t intelligent-classification-service:latest .
   ```

2. **启动服务**
   ```bash
   docker-compose up -d
   ```

3. **查看服务状态**
   ```bash
   docker-compose ps
   docker-compose logs -f intelligent-classification-service
   ```

### Kubernetes部署

1. **部署到Kubernetes**
   ```bash
   kubectl apply -f k8s-deployment.yaml
   ```

2. **检查部署状态**
   ```bash
   kubectl get pods -n historical-text -l app=intelligent-classification-service
   kubectl get svc -n historical-text intelligent-classification-service
   ```

## 📖 API使用指南

### 基础概念

- **项目(Project)**: 分类项目，包含分类类型、标签和配置
- **训练数据(Training Data)**: 用于模型训练的标注数据
- **模型(Model)**: 训练好的分类模型
- **分类任务(Classification Task)**: 文档分类请求和结果

### 主要API端点

#### 项目管理

```bash
# 创建分类项目
POST /api/v1/projects

# 获取项目详情
GET /api/v1/projects/{project_id}

# 更新项目配置
PUT /api/v1/projects/{project_id}

# 获取项目列表
GET /api/v1/projects?classification_type=topic&limit=10

# 获取支持的分类类型
GET /api/v1/projects/supported/types
```

#### 训练数据管理

```bash
# 添加训练数据
POST /api/v1/data/training-data

# 批量添加训练数据
POST /api/v1/data/training-data/batch

# 上传训练数据文件
POST /api/v1/data/training-data/upload

# 获取训练数据
GET /api/v1/data/training-data/{project_id}?limit=100&offset=0

# 验证数据质量
POST /api/v1/data/training-data/{project_id}/validate
```

#### 模型训练

```bash
# 训练模型
POST /api/v1/models/train

# 获取模型信息
GET /api/v1/models/{model_id}

# 获取项目的所有模型
GET /api/v1/models/project/{project_id}

# 激活模型
POST /api/v1/models/{model_id}/activate

# 获取模型性能
GET /api/v1/models/{model_id}/performance

# 模型基准测试
POST /api/v1/models/{model_id}/benchmark
```

#### 文档分类

```bash
# 单文档分类
POST /api/v1/classify/single

# 批量文档分类
POST /api/v1/classify/batch

# 异步批量分类
POST /api/v1/classify/async-batch

# 带详细解释的分类
POST /api/v1/classify/predict-with-explanation

# 比较多个模型
POST /api/v1/classify/compare-models

# 获取分类历史
GET /api/v1/classify/history/{project_id}
```

### 使用示例

#### 1. 创建分类项目

```python
import requests

# 创建主题分类项目
project_data = {
    "name": "古代文献主题分类",
    "description": "对古代历史文献进行主题分类",
    "classification_type": "topic",
    "language": "zh",
    "custom_labels": ["政治", "军事", "经济", "文化", "社会"]
}

response = requests.post(
    "http://localhost:8007/api/v1/projects",
    json=project_data
)
project = response.json()["data"]
project_id = project["id"]
```

#### 2. 添加训练数据

```python
# 单条训练数据
training_data = {
    "project_id": project_id,
    "text_content": "汉武帝时期，国力强盛，多次出征匈奴，扩张版图。",
    "true_label": "政治",
    "label_confidence": 1.0,
    "data_source": "史记"
}

response = requests.post(
    "http://localhost:8007/api/v1/data/training-data",
    json=training_data
)

# 批量训练数据
batch_data = {
    "project_id": project_id,
    "training_data": [
        {
            "text_content": "唐朝诗歌繁荣，李白、杜甫等诗人名垂青史。",
            "true_label": "文化"
        },
        {
            "text_content": "宋朝商业发达，海上丝绸之路贸易繁荣。",
            "true_label": "经济"
        }
    ]
}

response = requests.post(
    "http://localhost:8007/api/v1/data/training-data/batch",
    json=batch_data
)
```

#### 3. 训练模型

```python
# 训练随机森林模型
training_request = {
    "project_id": project_id,
    "model_type": "random_forest",
    "feature_extractor": "tfidf",
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 10
    },
    "training_config": {
        "test_size": 0.2,
        "cv_folds": 5
    }
}

response = requests.post(
    "http://localhost:8007/api/v1/models/train",
    json=training_request
)
model_info = response.json()["data"]
model_id = model_info["model_id"]
```

#### 4. 文档分类

```python
# 单文档分类
classification_request = {
    "project_id": project_id,
    "text_content": "清朝康熙年间，实行闭关锁国政策，限制对外贸易。",
    "return_probabilities": True,
    "return_explanation": True
}

response = requests.post(
    "http://localhost:8007/api/v1/classify/single",
    json=classification_request
)
result = response.json()["data"]

print(f"预测标签: {result['predicted_label']}")
print(f"置信度: {result['confidence_score']:.3f}")
print(f"解释: {result['explanation']}")

# 批量文档分类
batch_request = {
    "project_id": project_id,
    "documents": [
        {"text_content": "秦始皇统一六国，建立中央集权制度。"},
        {"text_content": "唐代佛教兴盛，寺院众多，僧侣地位较高。"},
        {"text_content": "明代手工业发达，景德镇瓷器享誉世界。"}
    ],
    "return_probabilities": True
}

response = requests.post(
    "http://localhost:8007/api/v1/classify/batch",
    json=batch_request
)
batch_result = response.json()["data"]

for result in batch_result["results"]:
    print(f"文档: {result['document_id']} -> {result['predicted_label']}")
```

## 🔧 配置说明

### 环境变量配置

服务通过环境变量进行配置，主要配置项包括：

#### 基础服务配置
- `SERVICE_NAME`: 服务名称
- `SERVICE_VERSION`: 服务版本
- `ENVIRONMENT`: 运行环境 (development/production)
- `DEBUG`: 调试模式开关

#### API配置
- `API_HOST`: 服务绑定地址
- `API_PORT`: 服务端口 (默认8007)
- `API_PREFIX`: API路径前缀

#### 依赖服务配置
- `STORAGE_SERVICE_URL`: Storage服务地址
- `NLP_SERVICE_URL`: NLP服务地址 (可选)
- `KNOWLEDGE_GRAPH_SERVICE_URL`: 知识图谱服务地址 (可选)

#### 智能分类配置
- `MAX_TEXT_LENGTH`: 最大文本长度
- `MAX_BATCH_SIZE`: 最大批量处理大小
- `CLASSIFICATION_TIMEOUT`: 分类超时时间

### 预定义分类类型

服务支持以下预定义分类类型：

- **topic** (主题分类): 政治、军事、经济、文化、社会、科技、宗教、教育
- **era** (时代分类): 先秦、秦汉、魏晋南北朝、隋唐、宋元、明清、近代、现代
- **document_type** (文档类型): 史书、文集、诗词、奏疏、碑刻、档案、日记、书信
- **importance** (重要性评级): 极高、高、中、低
- **sentiment** (情感分析): 正面、负面、中性
- **genre** (体裁分类): 纪传体、编年体、纪事本末体、政书、杂史

### 支持的算法

#### 传统机器学习
- **SVM**: 支持向量机，适合小样本分类
- **RandomForest**: 随机森林，平衡性能与解释性
- **XGBoost**: 梯度提升树，高性能集成算法
- **LightGBM**: 轻量级梯度提升，训练速度快

#### 特征提取方法
- **TF-IDF**: 词频-逆文档频率，经典文本特征
- **Word2Vec**: 词向量模型，捕获语义关系
- **FastText**: 子词级别词向量，处理OOV问题
- **BERT**: 双向编码器，深度语义理解
- **Sentence-Transformers**: 句子级别语义向量

## 📊 性能监控

### MLflow实验跟踪

服务集成MLflow进行实验管理：

```bash
# 查看MLflow UI
http://localhost:8007/mlflow
```

### Prometheus指标

服务提供Prometheus指标：

```bash
# 指标端点
GET /metrics
```

主要指标包括：
- 请求处理时间
- 分类准确率
- 模型训练时间
- 系统资源使用

### 健康检查

```bash
# 健康检查
GET /health

# 就绪检查
GET /ready

# 服务信息
GET /info
```

## 🧪 测试指南

### 运行测试

```bash
# 安装测试依赖
pip install -r requirements.txt

# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 生成测试覆盖率报告
pytest --cov=src --cov-report=html
```

### 测试数据

测试使用的示例历史文本数据位于 `tests/data/` 目录，包含：
- 不同时代的历史文献片段
- 多种文体的文本样本
- 预标注的分类数据

## 🚨 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 检查storage-service连接
   curl http://localhost:8002/health
   
   # 检查端口占用
   lsof -i :8007
   ```

2. **模型训练失败**
   ```bash
   # 检查训练数据质量
   POST /api/v1/data/training-data/{project_id}/validate
   
   # 查看训练日志
   docker-compose logs intelligent-classification-service
   ```

3. **分类性能差**
   - 检查训练数据质量和平衡性
   - 尝试不同的特征提取方法
   - 调整模型超参数
   - 增加训练数据量

4. **内存不足**
   ```bash
   # 调整Docker内存限制
   # 或减小批量处理大小
   MAX_BATCH_SIZE=50
   ```

### 日志分析

```bash
# 查看实时日志
docker-compose logs -f intelligent-classification-service

# 查看错误日志
docker-compose logs intelligent-classification-service | grep ERROR

# 查看特定时间段日志
docker-compose logs --since="2024-01-01T00:00:00" intelligent-classification-service
```

## 🤝 开发指南

### 贡献代码

1. Fork项目仓库
2. 创建特性分支: `git checkout -b feature/new-algorithm`
3. 提交更改: `git commit -am 'Add new algorithm'`
4. 推送分支: `git push origin feature/new-algorithm`
5. 提交Pull Request

### 代码规范

- 使用Black进行代码格式化
- 使用isort整理import语句
- 使用flake8进行代码检查
- 使用mypy进行类型检查
- 注释覆盖率应达到30%以上

### 添加新算法

1. 在 `utils/feature_extraction.py` 中添加新的特征提取器
2. 在 `services/model_trainer.py` 中添加新的模型类型
3. 更新 `config/settings.py` 中的模型配置
4. 添加相应的测试用例
5. 更新API文档

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 支持与反馈

- 📧 Email: support@historical-text.com
- 🐛 Bug报告: [GitHub Issues](https://github.com/project/issues)
- 📖 文档: [项目Wiki](https://github.com/project/wiki)
- 💬 讨论: [GitHub Discussions](https://github.com/project/discussions)

---

**智能分类服务** - 历史文本优化项目核心组件
版本: v1.0.0 | 更新时间: 2024-01-01