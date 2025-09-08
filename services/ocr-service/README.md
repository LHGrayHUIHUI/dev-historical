# OCR文本识别服务

无状态OCR文本识别微服务，专为历史文献设计的高精度文字识别服务。专注于图像文本识别算法，数据存储通过storage-service完成。

## 功能特性

### 🎯 核心功能

- **多引擎支持**: 集成PaddleOCR、Tesseract、EasyOCR等主流OCR引擎
- **高精度识别**: 针对古代汉字、繁体字、异体字专门优化
- **异步处理**: 支持大批量文档的异步识别处理
- **智能预处理**: 自动图像增强、去噪、倾斜校正
- **文本后处理**: 繁简转换、标点规范化、错误纠正

### 🚀 技术特性

- **无状态架构**: 不直接连接数据库，通过storage-service进行数据管理
- **现代架构**: 基于FastAPI + Python 3.11构建
- **高性能**: 异步I/O，支持并发处理
- **水平扩展**: 无状态设计，支持Kubernetes水平扩展
- **云原生**: 完整Docker支持，微服务架构
- **专业分工**: 专注OCR计算，不处理业务逻辑

## 快速开始

### 环境要求

- Python 3.11+
- Storage Service (端口 8002) - 用于数据存储
- Docker & Docker Compose (推荐)

### Docker快速部署（推荐）

```bash
# 克隆代码
git clone <repository-url>
cd services/ocr-service

# 复制环境配置
cp .env.example .env

# 编辑环境变量，配置storage-service地址
vim .env

# 启动OCR服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f ocr-service
```

服务启动后访问：
- API文档: http://localhost:8003/docs
- 健康检查: http://localhost:8003/health
- 服务信息: http://localhost:8003/info

### 本地开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，配置storage-service地址等

# 启动开发服务器
python -m src.main
```

**注意**: 本地开发需要确保storage-service已启动并可访问。

## API使用示例

### 单图像识别（同步模式）

```bash
curl -X POST "http://localhost:8003/api/v1/ocr/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  -F "engine=paddleocr" \
  -F "confidence_threshold=0.8" \
  -F "async_mode=false"
```

### 单图像识别（异步模式）

```bash
curl -X POST "http://localhost:8003/api/v1/ocr/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  -F "engine=paddleocr" \
  -F "confidence_threshold=0.8" \
  -F "async_mode=true"
```

### 批量图像识别

```bash
curl -X POST "http://localhost:8003/api/v1/ocr/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "engine=paddleocr"
```

### 查询任务状态

```bash
curl -X GET "http://localhost:8003/api/v1/ocr/task/{task_id}"
```

### 获取可用引擎

```bash
curl -X GET "http://localhost:8003/api/v1/ocr/engines"
```

## 项目结构

```
services/ocr-service/
├── src/                        # 源代码目录
│   ├── config/                 # 配置管理
│   │   └── settings.py         # 应用配置（无状态）
│   ├── controllers/            # API控制器
│   │   └── ocr_controller.py   # OCR识别接口
│   ├── clients/                # 外部服务客户端
│   │   └── storage_client.py   # Storage服务客户端
│   ├── services/               # 业务逻辑层
│   │   └── ocr_service.py      # OCR服务类（纯计算）
│   ├── schemas/                # Pydantic模型
│   │   └── ocr_schemas.py      # OCR相关模型
│   ├── utils/                  # 工具模块
│   │   ├── image_processor.py  # 图像预处理
│   │   ├── text_processor.py   # 文本后处理
│   │   ├── logger.py           # 日志工具
│   │   └── middleware.py       # FastAPI中间件
│   └── main.py                 # 应用入口点
├── tests/                      # 测试代码
├── temp/                       # 临时文件（仅此目录）
├── docker-compose.yml          # Docker编排
├── Dockerfile                  # Docker镜像
├── requirements.txt            # Python依赖（精简版）
├── .env.example               # 环境变量示例
└── README.md                  # 项目文档
```

### 架构特点

- **无数据库层**: 移除了 `database/`、`models/`、`repositories/` 目录
- **服务客户端**: 新增 `clients/` 目录，通过HTTP与storage-service通信
- **纯计算服务**: `services/` 专注OCR算法，不处理数据持久化
- **精简配置**: 配置文件仅包含OCR引擎和服务通信设置

## 配置说明

### 环境变量

主要配置项（完整列表见`.env.example`）：

```bash
# 服务配置
OCR_ENVIRONMENT=development
OCR_API_HOST=0.0.0.0
OCR_API_PORT=8003

# Storage Service配置（必需）
OCR_SERVICE_STORAGE_SERVICE_URL=http://localhost:8002
OCR_SERVICE_STORAGE_SERVICE_TIMEOUT=30
OCR_SERVICE_STORAGE_SERVICE_RETRIES=3

# OCR引擎配置
OCR_DEFAULT_ENGINE=paddleocr
OCR_DEFAULT_CONFIDENCE_THRESHOLD=0.8
OCR_MAX_FILE_SIZE=52428800  # 50MB
OCR_MAX_BATCH_SIZE=20

# 临时文件配置
OCR_TEMP_DIR=/tmp/ocr-service
```

### OCR引擎配置

#### PaddleOCR（推荐）
```python
{
    "use_angle_cls": True,
    "lang": "ch",
    "use_gpu": True,
    "det_thresh": 0.3,
    "rec_thresh": 0.7
}
```

#### Tesseract
```python
{
    "lang": "chi_sim+eng",
    "oem": 3,
    "psm": 6,
    "config": "--dpi 300"
}
```

#### EasyOCR
```python
{
    "lang_list": ["ch_sim", "en"],
    "gpu": True,
    "detail": 1
}
```

## 支持格式

### 输入格式
- **图像格式**: JPG, JPEG, PNG, BMP, TIFF, WebP
- **最大文件大小**: 50MB（可配置）
- **批量限制**: 20个文件（可配置）

### 输出格式
- **文本内容**: 完整识别文本
- **置信度**: 每个文本块的置信度分数
- **边界框**: 文字在图像中的位置坐标
- **元数据**: 处理时间、引擎信息、语言检测等

## 性能优化

### 图像预处理
- 自适应去噪算法
- 智能对比度增强
- 自动倾斜校正
- 多种二值化算法

### 文本后处理
- 繁简转换（OpenCC）
- 标点符号规范化
- OCR错误纠正
- 异体字处理

### 系统优化
- 异步I/O处理
- 连接池管理
- 结果缓存
- 负载均衡

## 监控与运维

### 健康检查
```bash
# 基础健康检查
curl http://localhost:8000/health

# 详细组件状态
curl http://localhost:8000/api/v1/health/detailed
```

### 性能指标
- 请求响应时间
- 识别成功率
- 错误率统计
- 资源使用情况

### 日志管理
```bash
# 查看服务日志
docker-compose logs -f ocr-service

# 查看错误日志
docker-compose logs -f ocr-service | grep ERROR
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

### 提交规范
```bash
git commit -m "feat(ocr): 添加新的OCR引擎支持"
git commit -m "fix(api): 修复批量处理内存泄漏问题"
git commit -m "docs: 更新API文档"
```

## 部署指南

### Docker部署
```bash
# 构建镜像
docker build -t ocr-service:latest .

# 运行容器
docker run -d \
  --name ocr-service \
  -p 8003:8003 \
  -e OCR_SERVICE_STORAGE_SERVICE_URL=http://your-storage-service:8002 \
  ocr-service:latest
```

### Kubernetes部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ocr-service
  template:
    metadata:
      labels:
        app: ocr-service
    spec:
      containers:
      - name: ocr-service
        image: ocr-service:latest
        ports:
        - containerPort: 8003
        env:
        - name: OCR_SERVICE_STORAGE_SERVICE_URL
          value: "http://storage-service:8002"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 故障排除

### 常见问题

1. **OCR引擎初始化失败**
   - 检查模型文件是否下载完整
   - 验证GPU环境配置（如果使用GPU）

2. **Storage服务连接失败**
   - 检查storage-service服务状态
   - 验证服务URL配置
   - 确保网络连通性

3. **文件上传失败**
   - 检查文件大小限制
   - 验证文件格式支持

4. **识别准确率低**
   - 调整置信度阈值
   - 启用图像预处理
   - 尝试不同OCR引擎

5. **异步任务处理异常**
   - 检查storage-service任务管理功能
   - 验证任务状态更新

### 调试模式
```bash
# 启用详细日志
export OCR_LOG_LEVEL=DEBUG

# 启用错误详情
export OCR_DEBUG_SHOW_ERROR_DETAILS=true

# 查看服务健康状态
curl http://localhost:8003/health

# 查看可用OCR引擎
curl http://localhost:8003/api/v1/ocr/engines
```

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新历史。

## 支持

- 📧 邮件: support@example.com
- 🐛 问题报告: [GitHub Issues](https://github.com/your-org/ocr-service/issues)
- 📚 文档: [项目Wiki](https://github.com/your-org/ocr-service/wiki)
- 💬 讨论: [GitHub Discussions](https://github.com/your-org/ocr-service/discussions)