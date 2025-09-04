# 统一存储服务 (storage-service)

## 📋 服务概述

**storage-service** 是历史文本项目的核心统一存储微服务，负责所有数据存储、内容管理、文件存储和业务逻辑处理。

### 🎯 核心定位
- **统一存储服务**: 管理所有数据库和存储系统
- **业务逻辑中心**: 负责所有业务规则和数据处理
- **服务协调者**: 调用file-processor处理文件，统一管理数据流

### ✅ 核心功能
- 🗄️ **统一数据库管理**: MongoDB + PostgreSQL + Redis + MinIO 完整存储栈
- 📄 **内容管理系统**: 历史文本内容的CRUD、搜索、分类、统计
- 📁 **文件存储管理**: MinIO对象存储的完整管理和API
- 🔄 **服务协调**: 调用file-processor处理文件，整合处理结果
- 📊 **业务分析**: 数据统计、搜索、报表和可视化支持
- ⚡ **批量处理**: 支持大批量数据导入和处理

### 🗄️ 管理的存储系统
- **MongoDB**: 历史文本内容、业务数据、用户数据
- **PostgreSQL**: 文件元数据、处理记录、关系数据
- **Redis**: 缓存、会话、任务队列、统计数据
- **MinIO**: 文件对象存储（图片、视频、文档）
- **RabbitMQ**: 消息队列、异步任务处理

## 🏗️ 技术架构

### 技术栈
- **框架**: FastAPI + Python 3.11+ + SQLAlchemy 2.0
- **数据库**: MongoDB + PostgreSQL + Redis
- **存储**: MinIO (S3兼容对象存储)
- **消息队列**: RabbitMQ
- **监控**: Prometheus + Grafana
- **日志**: Structured logging + ELK Stack

### 依赖关系
- **调用服务**: file-processor (文件处理)
- **被调用方**: Vue3前端、其他微服务
- **外部依赖**: 所有数据库和存储系统

## 🚀 API 文档

### 内容管理 API

#### 1. 创建内容 (纯数据)
```http
POST /api/v1/content/
Content-Type: application/json

请求体:
{
  "title": "历史文献标题",
  "content": "文档内容...",
  "source": "manual",
  "author": "作者名称",
  "images": ["http://localhost:9001/historical-images/file1.jpg"],
  "videos": ["http://localhost:9001/historical-videos/video1.mp4"],
  "keywords": ["历史", "文献"],
  "tags": ["重要", "古代"],
  "category": "历史文档"
}

响应:
{
  "id": "uuid-string",
  "title": "历史文献标题",
  "content": "文档内容...",
  "images": ["http://localhost:9001/historical-images/file1.jpg"],
  "videos": ["http://localhost:9001/historical-videos/video1.mp4"],
  "status": "active",
  "created_at": "2025-09-04T17:00:00Z",
  "updated_at": "2025-09-04T17:00:00Z"
}
```

#### 2. 创建内容 + 文件处理
```http
POST /api/v1/content/with-files
Content-Type: multipart/form-data

参数:
- title: 内容标题 (required)
- content: 内容正文 (required)
- source: 来源 (default: "manual")
- author: 作者
- category: 分类
- keywords: 关键词 (逗号分隔)
- tags: 标签 (逗号分隔)
- image_files[]: 图片文件数组
- video_files[]: 视频文件数组

处理流程:
1. 接收文件和内容数据
2. 调用file-processor处理文件
3. 存储文件到MinIO
4. 创建内容记录关联文件URL
5. 返回完整的内容记录

响应:
{
  "id": "uuid-string",
  "title": "内容标题",
  "content": "内容正文",
  "images": ["http://localhost:9001/historical-images/20250904/uuid_file1.jpg"],
  "videos": ["http://localhost:9001/historical-videos/20250904/uuid_video1.mp4"],
  "summary": "包含2个图片和1个视频的内容",
  "status": "active"
}
```

#### 3. 查询内容列表
```http
GET /api/v1/content/?skip=0&limit=20&source=manual&category=历史文档

响应: ContentResponse数组
```

#### 4. 获取内容详情
```http
GET /api/v1/content/{content_id}

响应: ContentResponse对象
```

#### 5. 更新内容
```http
PUT /api/v1/content/{content_id}
Content-Type: application/json

请求体: ContentCreate对象
响应: 更新后的ContentResponse对象
```

#### 6. 删除内容
```http
DELETE /api/v1/content/{content_id}

响应:
{
  "success": true,
  "message": "内容删除成功",
  "deleted_id": "uuid-string"
}
```

#### 7. 搜索内容
```http
GET /api/v1/content/search/?q=历史文献&skip=0&limit=20

响应:
{
  "success": true,
  "data": {
    "query": "历史文献",
    "total": 150,
    "results": [...],
    "pagination": {
      "skip": 0,
      "limit": 20,
      "has_more": true
    }
  }
}
```

#### 8. 内容统计
```http
GET /api/v1/content/stats/

响应:
{
  "success": true,
  "data": {
    "total_content": 500,
    "source_distribution": {"manual": 300, "import": 200},
    "category_distribution": {"历史文档": 150, "古籍": 100},
    "media_stats": {
      "total_images": 800,
      "total_videos": 200,
      "content_with_media": 250
    }
  }
}
```

### 文件管理 API (原有功能保持)

#### 9. 文件上传
```http
POST /api/v1/files/upload
Content-Type: multipart/form-data

参数:
- file: 文件 (required)
- dataset_id: 数据集ID
- metadata: 文件元数据 (JSON)

响应:
{
  "success": true,
  "file_id": "uuid-string",
  "filename": "document.pdf",
  "size": 1024000,
  "url": "http://localhost:9001/historical-files/20250904/uuid_document.pdf",
  "content_type": "application/pdf"
}
```

#### 10. 批量文件上传
```http
POST /api/v1/files/batch-upload
Content-Type: multipart/form-data

参数:
- files[]: 多个文件 (required)
- dataset_id: 数据集ID

响应: 上传结果数组
```

### 系统接口

#### 健康检查
```http
GET /health

响应:
{
  "status": "healthy",
  "service": "storage-service",
  "version": "1.0.0",
  "checks": {
    "database": "healthy",
    "storage": "healthy", 
    "message_queue": "healthy"
  }
}
```

#### 就绪检查
```http
GET /ready

响应: 详细的服务就绪状态
```

## 🔄 与其他服务协作

### 调用file-processor的典型流程

```python
# storage-service内部调用file-processor
async def process_uploaded_file(file: UploadFile):
    # 1. 先存储文件到MinIO
    file_url = await store_file_to_minio(file)
    
    # 2. 调用file-processor处理文件
    async with httpx.AsyncClient() as client:
        files = {"file": (file.filename, file.file, file.content_type)}
        response = await client.post(
            "http://file-processor:8000/api/v1/process/document",
            files=files
        )
        processing_result = response.json()
    
    # 3. 将处理结果和文件URL一起存储到数据库
    content_data = {
        "file_url": file_url,
        "extracted_text": processing_result["result"]["text_content"],
        "metadata": processing_result["result"]["metadata"]
    }
    
    return await save_to_database(content_data)
```

### 服务调用链
```
Vue3前端 → storage-service → file-processor
              ↓                    ↓
         存储所有数据    ←    返回处理结果
              ↓
         返回完整结果 → Vue3前端
```

## 🛠️ 开发和部署

### 本地开发
```bash
cd services/storage-service

# 安装依赖
pip install -r requirements.txt

# 启动依赖服务
docker-compose up -d mongodb postgresql redis minio rabbitmq

# 启动服务 (开发模式)
python -m src.main

# 访问API文档
http://localhost:8002/docs
```

### Docker完整部署
```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs storage-service
```

### 环境变量
```bash
# 服务配置
SERVICE_NAME=storage-service
SERVICE_VERSION=1.0.0
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000

# 数据库配置
MONGODB_URL=mongodb://mongodb:27017/historical_text
POSTGRESQL_URL=postgresql://user:password@postgresql:5432/historical_text
REDIS_URL=redis://redis:6379/0

# MinIO配置
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=testuser
MINIO_SECRET_KEY=testpass123
MINIO_BUCKET_NAME=historical-bucket

# RabbitMQ配置
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

# 外部服务
FILE_PROCESSOR_URL=http://file-processor:8000

# 监控配置
METRICS_ENABLED=true
```

## 📊 数据库模式

### MongoDB集合
```javascript
// content集合 - 历史文本内容
{
  _id: ObjectId("..."),
  title: "历史文献标题",
  content: "文档正文内容...",
  source: "manual|import|api",
  author: "作者名称",
  images: ["http://localhost:9001/..."],
  videos: ["http://localhost:9001/..."],
  keywords: ["历史", "文献"],
  tags: ["重要", "古代"],
  category: "历史文档",
  status: "active|draft|archived",
  created_at: ISODate("2025-09-04T17:00:00Z"),
  updated_at: ISODate("2025-09-04T17:00:00Z")
}
```

### PostgreSQL表
```sql
-- 文件元数据表
CREATE TABLE file_metadata (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_name VARCHAR(255),
    content_type VARCHAR(100),
    size_bytes BIGINT,
    file_hash VARCHAR(64),
    storage_path TEXT,
    storage_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 文本提取记录表
CREATE TABLE text_extraction (
    id UUID PRIMARY KEY,
    file_id UUID REFERENCES file_metadata(id),
    extracted_text TEXT,
    extraction_method VARCHAR(50),
    confidence_score DECIMAL(3,2),
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Redis键空间
```
# 缓存
cache:content:{id} -> ContentResponse JSON
cache:search:{query_hash} -> 搜索结果 JSON

# 会话
session:{session_id} -> 用户会话数据

# 统计
stats:daily:{date} -> 每日统计数据
stats:content_count -> 内容总数量
stats:file_count -> 文件总数量

# 任务队列
queue:file_processing -> 文件处理任务队列
queue:content_indexing -> 内容索引任务队列
```

## 📊 性能和监控

### 处理能力
- **内容管理**: ~1000 req/s (CRUD操作)
- **文件上传**: ~50 files/s (依赖文件大小)
- **搜索查询**: ~500 req/s (带缓存)
- **批量导入**: ~100 content/s

### 监控指标
- HTTP请求量、响应时间、错误率
- 数据库连接池状态和查询性能
- 文件存储使用量和传输速度
- 缓存命中率和内存使用
- 队列长度和处理延迟

### 告警规则
- 响应时间 > 2秒
- 错误率 > 5%
- 数据库连接数 > 80%
- 存储使用率 > 90%
- 队列积压 > 100个任务

## 🔧 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库服务状态
   - 验证连接字符串和凭据
   - 查看网络连通性

2. **文件上传失败**
   - 检查MinIO服务状态
   - 验证存储桶权限
   - 确认文件大小限制

3. **文件处理超时**
   - 检查file-processor服务状态
   - 调整处理超时配置
   - 查看处理队列状态

4. **搜索性能差**
   - 检查MongoDB索引状态
   - 优化搜索查询
   - 增加Redis缓存

### 错误代码
- `400`: 请求参数错误
- `404`: 资源不存在
- `422`: 数据验证失败
- `500`: 服务内部错误
- `503`: 依赖服务不可用

## 📝 更新日志

### v1.0.0 (2025-09-04)
- ✅ 微服务架构重构完成
- ✅ 整合所有数据库和存储管理
- ✅ 新增完整的内容管理API
- ✅ 实现与file-processor的服务协作
- ✅ 支持多媒体内容创建和管理
- ✅ 完整的搜索、统计、分析功能
- ✅ 生产级监控、日志、健康检查

### 架构变更
- **服务重命名**: data-collection → storage-service
- **职责扩展**: 从纯文件处理 → 统一存储管理
- **API整合**: 文件存储API + 内容管理API
- **依赖优化**: 调用file-processor而非重复实现

## 📞 联系信息

- **维护团队**: 历史文本项目团队
- **技术支持**: support@historical-text.com
- **API文档**: http://localhost:8002/docs
- **健康检查**: http://localhost:8002/health
- **监控面板**: http://localhost:3000 (Grafana)

---

**重要**: 此服务是项目的核心数据服务，负责所有存储和业务逻辑。前端应用和其他服务都通过此服务访问数据。