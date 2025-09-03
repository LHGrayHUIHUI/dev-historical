# 数据采集与存储服务 (Data Collection Service)

## 服务概述

数据采集与存储服务是历史文本项目的核心微服务之一，负责文件上传、文本提取和数据存储管理。该服务支持多种文件格式的智能文本提取，并提供完整的数据集管理功能。

## 核心功能

### 📁 多格式文件处理
- **PDF文档**: 使用pdfplumber和PyPDF2进行文本提取，支持OCR回退
- **Word文档**: 支持.docx和.doc格式，提取段落、表格和页眉页脚
- **图像文件**: 集成Tesseract OCR，支持中英文文本识别
- **HTML文档**: 智能解析网页结构，提取纯文本内容
- **纯文本**: 自动编码检测，支持多种字符编码

### 🗄️ 多数据库架构
- **PostgreSQL**: 存储结构化数据（数据集、文本内容）
- **MinIO**: 对象存储，存储原始文件
- **Redis**: 缓存和会话存储
- **RabbitMQ**: 异步任务队列

### 🔄 异步处理框架
- 基于RabbitMQ的消息队列
- 后台工作器进行文本提取
- 实时处理状态更新
- 失败重试机制

### 🛡️ 安全检测
- ClamAV病毒扫描
- 文件类型验证
- 文件大小限制
- SHA256哈希去重

## 技术架构

### 核心技术栈
- **后端框架**: FastAPI 0.104+ (Python 3.11)
- **数据库ORM**: SQLAlchemy 2.0 (异步)
- **数据验证**: Pydantic 2.5+
- **异步处理**: asyncio + aiofiles
- **文件处理**: 
  - PyPDF2, pdfplumber (PDF)
  - python-docx (Word)
  - Pillow + pytesseract (OCR)
  - BeautifulSoup4 (HTML)
- **监控**: Prometheus + structlog

### 项目结构

```
services/data-collection/
├── src/
│   ├── controllers/          # API控制器
│   ├── services/            # 业务服务层
│   ├── processors/          # 文本提取器
│   ├── workers/            # 后台工作器
│   ├── models/             # 数据模型
│   ├── schemas/            # API数据模式
│   ├── config/             # 配置管理
│   └── utils/              # 工具函数
├── migrations/             # 数据库迁移
├── tests/                  # 测试套件
├── requirements.txt        # Python依赖
├── Dockerfile             # 容器配置
└── README.md
```

## API接口

### 文件上传
```http
POST /api/v1/data/upload
Content-Type: multipart/form-data

file: (binary)
source_id: uuid
metadata: json_string
```

### 批量上传
```http
POST /api/v1/data/upload/batch
Content-Type: multipart/form-data

files[]: (binary array)
source_id: uuid
metadata: json_string
```

### 数据集管理
```http
GET /api/v1/data/datasets                    # 获取数据集列表
GET /api/v1/data/datasets/{id}               # 获取数据集详情
PUT /api/v1/data/datasets/{id}               # 更新数据集
DELETE /api/v1/data/datasets/{id}            # 删除数据集
GET /api/v1/data/datasets/{id}/processing-status  # 获取处理状态
POST /api/v1/data/datasets/{id}/reprocess    # 重新处理
```

### 健康检查
```http
GET /health                                  # 健康检查
GET /ready                                   # 就绪检查
GET /api/v1/data/info                       # 服务信息
```

## 快速开始

### 环境要求
- Python 3.11+
- PostgreSQL 14+
- Redis 7.0+
- RabbitMQ 3.12+
- MinIO (S3兼容存储)
- Tesseract OCR (可选)
- ClamAV (可选)

### 本地开发

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑.env文件，配置数据库连接等
   ```

3. **运行数据库迁移**
   ```bash
   alembic upgrade head
   ```

4. **启动服务**
   ```bash
   python -m src.main
   ```

### Docker部署

1. **构建镜像**
   ```bash
   docker build -t data-collection-service .
   ```

2. **运行服务**
   ```bash
   docker run -d \
     --name data-collection \
     -p 8002:8002 \
     -e DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db \
     -e REDIS_URL=redis://host:6379/0 \
     -e RABBITMQ_URL=amqp://user:pass@host:5672/ \
     -e MINIO_ENDPOINT=host:9000 \
     data-collection-service
   ```

### Docker Compose

使用项目根目录的docker-compose配置：

```bash
# 启动开发环境
docker-compose -f docker-compose.yml up data-collection-service

# 启动生产环境
docker-compose -f docker-compose.production.yml up data-collection-service
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| SERVICE_PORT | 服务端口 | 8002 |
| DATABASE_URL | PostgreSQL连接URL | 必填 |
| MONGODB_URL | MongoDB连接URL | 必填 |
| REDIS_URL | Redis连接URL | 必填 |
| RABBITMQ_URL | RabbitMQ连接URL | 必填 |
| MINIO_ENDPOINT | MinIO端点 | 必填 |
| MINIO_ACCESS_KEY | MinIO访问密钥 | 必填 |
| MINIO_SECRET_KEY | MinIO秘密密钥 | 必填 |
| MAX_FILE_SIZE | 最大文件大小(字节) | 104857600 |
| MAX_BATCH_SIZE | 批量上传最大文件数 | 50 |
| VIRUS_SCAN_ENABLED | 是否启用病毒扫描 | true |
| OCR_ENABLED | 是否启用OCR | true |

### 支持的文件类型

- `application/pdf` - PDF文档
- `application/msword` - Word文档 (.doc)
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document` - Word文档 (.docx)
- `text/plain` - 纯文本
- `text/html` - HTML文档
- `image/jpeg`, `image/png`, `image/tiff` - 图像文件(OCR)

## 监控和日志

### 指标监控
服务集成Prometheus指标：
- 文件上传计数器
- 处理时间直方图
- 活跃上传数量
- 提取成功/失败计数器

访问 `/metrics` 端点获取指标数据。

### 结构化日志
使用structlog进行结构化日志记录：
```python
logger.info("文件上传成功", 
    filename="document.pdf",
    dataset_id="uuid-here",
    processing_time=1.23
)
```

## 测试

### 运行测试
```bash
# 单元测试
pytest tests/unit/

# 集成测试
pytest tests/integration/

# 所有测试
pytest tests/ -v

# 覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 测试覆盖
- 数据模型测试
- 文本提取器测试
- API接口测试
- 业务服务测试

## 部署检查清单

### 生产环境部署前检查

- [ ] 数据库连接配置正确
- [ ] MinIO存储桶已创建
- [ ] RabbitMQ队列已声明
- [ ] 文件权限设置正确
- [ ] 环境变量已设置
- [ ] 健康检查端点正常
- [ ] 监控指标可访问
- [ ] 日志输出正常
- [ ] 安全扫描通过

### 性能优化建议

1. **数据库优化**
   - 为常用查询创建索引
   - 配置连接池大小
   - 启用查询缓存

2. **文件处理优化**
   - 调整并发处理数量
   - 配置合适的内存限制
   - 使用SSD存储临时文件

3. **监控告警**
   - 设置处理时间告警
   - 监控队列积压情况
   - 跟踪错误率指标

## 故障排除

### 常见问题

1. **文件上传失败**
   - 检查文件大小限制
   - 验证文件类型支持
   - 确认存储空间充足

2. **文本提取失败**
   - 检查依赖库安装
   - 验证文件格式完整性
   - 查看工作器日志

3. **数据库连接错误**
   - 验证连接参数
   - 检查网络连通性
   - 确认数据库服务状态

4. **队列处理停滞**
   - 检查RabbitMQ状态
   - 重启工作器进程
   - 清理死信队列

## 开发指南

### 添加新的文本提取器

1. 继承`TextExtractor`基类
2. 实现`extract`和`supports_file_type`方法
3. 在`DataCollectionService`中注册
4. 添加相应的测试

示例：
```python
class CustomExtractor(TextExtractor):
    SUPPORTED_TYPES = {'application/custom'}
    
    def supports_file_type(self, file_type: str) -> bool:
        return file_type in self.SUPPORTED_TYPES
    
    async def extract(self, file_path: str, **kwargs):
        # 实现提取逻辑
        pass
```

### 贡献代码

1. Fork项目仓库
2. 创建特性分支
3. 编写代码和测试
4. 提交Pull Request

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](../../LICENSE)文件。