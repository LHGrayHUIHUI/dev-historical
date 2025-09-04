# 文件处理服务 (file-processor)

## 🏆 项目状态
**Epic 1 已完成** ✅ (2025-09-04) - 作为Story 1.3数据采集存储服务的核心组件，文件处理服务已完成开发并通过生产验证。

## 📋 服务概述

**file-processor** 是历史文本项目中的纯文件处理微服务，专注于各种格式文件的处理和文本提取。

### 🎯 核心定位
- **纯文件处理服务**: 不涉及数据存储，专注文件处理算法
- **无状态设计**: 所有处理结果通过API返回给调用方
- **高性能处理**: 优化的文件处理算法和并发处理能力

### ✅ 核心功能
- 📄 **多格式文件处理**: PDF、Word、图片OCR、HTML等格式文件处理
- 🔤 **文本内容提取**: 从各种文件格式中提取纯文本内容
- 🛡️ **文件安全检测**: 文件格式验证、病毒扫描、安全检查
- ⚡ **异步处理**: 支持大文件的异步处理和状态跟踪
- 📊 **批量处理**: 支持多文件并发处理

### 🚫 不包含的功能
- ❌ 数据库连接 (MongoDB, PostgreSQL, Redis)
- ❌ 数据持久化存储
- ❌ 业务逻辑处理
- ❌ 内容管理功能

## 🏗️ 技术架构

### 技术栈
- **框架**: FastAPI + Python 3.11+
- **处理引擎**: PyPDF2, python-docx, Pillow, Tesseract OCR
- **格式支持**: PDF, Word, Excel, 图片(JPG/PNG/GIF), HTML
- **架构**: 无数据库依赖，纯处理逻辑

### 依赖关系
- **无数据库依赖**: 符合纯处理服务定位
- **调用方**: storage-service (统一存储服务)
- **外部依赖**: Tesseract OCR, ImageMagick (可选)

## 🚀 API 文档

### 核心处理接口

#### 1. PDF文件处理
```http
POST /api/v1/process/pdf
Content-Type: multipart/form-data

参数:
- file: PDF文件 (required)
- extract_text: 是否提取文本 (default: true)
- extract_metadata: 是否提取元数据 (default: true)

响应:
{
  "success": true,
  "task_id": "uuid-string",
  "status": "completed",
  "result": {
    "filename": "document.pdf",
    "text_content": "提取的文本内容...",
    "metadata": {
      "pages": 10,
      "title": "文档标题",
      "author": "作者名称"
    }
  }
}
```

#### 2. 图片OCR处理
```http
POST /api/v1/process/image-ocr
Content-Type: multipart/form-data

参数:
- file: 图片文件 (required)
- language: OCR语言 (default: "chi_sim+eng")
- enhance_image: 是否增强图片 (default: true)

响应:
{
  "success": true,
  "task_id": "uuid-string", 
  "status": "completed",
  "result": {
    "filename": "image.jpg",
    "recognized_text": "识别出的文字内容...",
    "confidence": 0.95
  }
}
```

#### 3. 通用文档处理
```http
POST /api/v1/process/document
Content-Type: multipart/form-data

支持格式: PDF, DOC, DOCX, HTML, TXT
```

#### 4. 批量文件处理
```http
POST /api/v1/process/batch
Content-Type: multipart/form-data

参数:
- files[]: 多个文件 (最多10个)
- extract_text: 是否提取文本 (default: true)

响应: ProcessResponse数组
```

#### 5. 处理状态查询
```http
GET /api/v1/process/status/{task_id}

响应:
{
  "task_id": "uuid-string",
  "status": "processing|completed|failed",
  "progress": 75.0,
  "result": {...},
  "created_at": "2025-09-04T17:00:00Z"
}
```

#### 6. 支持格式查询
```http
GET /api/v1/process/supported-formats

响应:
{
  "document_formats": [
    {"extension": "pdf", "description": "PDF文档", "features": ["文本提取", "元数据提取"]},
    {"extension": "docx", "description": "Word文档", "features": ["文本提取"]}
  ],
  "image_formats": [
    {"extension": "jpg", "description": "JPEG图片", "features": ["OCR文字识别"]}
  ]
}
```

### 系统接口

#### 健康检查
```http
GET /health

响应:
{
  "success": true,
  "data": {
    "status": "healthy",
    "components": {
      "processors": {
        "status": "ready",
        "available_processors": ["pdf", "word", "image", "html"]
      }
    }
  }
}
```

#### 服务信息
```http
GET /info

响应: 服务详细信息
```

## 🔄 与其他服务协作

### 调用方式
```
storage-service → file-processor
    ↓
file-processor (处理文件)
    ↓
返回处理结果 → storage-service (存储数据)
```

### 典型调用场景
1. **用户上传文件**: storage-service接收文件，调用file-processor处理
2. **批量文件处理**: storage-service发送多个文件给file-processor
3. **异步处理**: file-processor处理大文件，返回任务ID，storage-service轮询状态

## 🛠️ 开发和部署

### 本地开发
```bash
cd services/file-processor

# 安装依赖
pip install -r requirements.txt

# 启动服务 (开发模式)
python -m src.main

# 访问API文档
http://localhost:8001/docs
```

### Docker部署
```bash
# 构建镜像
docker build -t file-processor .

# 运行容器
docker run -p 8001:8000 \
  -e SERVICE_NAME=file-processor \
  -e LOG_LEVEL=INFO \
  file-processor
```

### 环境变量
```bash
# 服务配置
SERVICE_NAME=file-processor
SERVICE_VERSION=1.0.0
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000

# 日志配置  
LOG_LEVEL=INFO
LOG_FORMAT={time} | {level} | {message}

# 处理配置
MAX_FILE_SIZE_MB=50
SUPPORTED_LANGUAGES=chi_sim+eng
PROCESSING_TIMEOUT=300
```

## 📊 性能和监控

### 处理能力
- **PDF文档**: ~2-5秒/文档 (取决于页数)
- **图片OCR**: ~3-8秒/图片 (取决于分辨率)
- **Word文档**: ~1-3秒/文档
- **并发处理**: 支持10个并发任务

### 监控指标
- 处理请求数量和成功率
- 平均处理时间和性能指标
- 错误率和异常类型统计
- 支持格式的处理分布

### 日志记录
- 所有处理请求和结果
- 错误详情和异常堆栈
- 性能指标和处理时间
- 文件格式和大小统计

## 🔧 故障排除

### 常见问题

1. **OCR识别不准确**
   - 检查图片清晰度和分辨率
   - 尝试不同的language参数
   - 启用enhance_image图片增强

2. **PDF处理失败**
   - 确认PDF文件未损坏
   - 检查是否为扫描版PDF (需要OCR)
   - 查看错误日志了解具体原因

3. **处理超时**
   - 检查文件大小是否超限
   - 调整PROCESSING_TIMEOUT配置
   - 考虑使用异步处理模式

### 错误代码
- `400`: 不支持的文件格式
- `413`: 文件大小超过限制  
- `422`: 请求参数验证失败
- `500`: 内部处理错误

## 📝 更新日志

### v1.0.0 (2025-09-04)
- ✅ 微服务架构重构完成
- ✅ 移除所有数据库依赖
- ✅ 新增专业文件处理API
- ✅ 支持PDF、Word、图片OCR、HTML等格式
- ✅ 实现批量处理和异步任务管理
- ✅ 完整的API文档和健康检查

## 📞 联系信息

- **维护团队**: 历史文本项目团队
- **技术支持**: support@historical-text.com
- **API文档**: http://localhost:8001/docs
- **健康检查**: http://localhost:8001/health

---

**注意**: 此服务专注于文件处理，所有数据存储和业务逻辑由 storage-service 负责。