# 🌐 服务端口和API接口映射表

## 📋 服务端口总览

| 服务类型 | 服务名称 | 端口 | 协议 | 状态 | API文档 | 健康检查 |
|---------|---------|------|------|------|---------|----------|
| **核心服务** | Storage Service | 8002 | HTTP | ✅ 运行 | `/docs` | `/health` |
| **计算服务** | File Processor | 8001 | HTTP | ✅ 运行 | `/docs` | `/health` |
| **计算服务** | OCR Service | 8003 | HTTP | ✅ 运行 | `/docs` | `/health` |
| **计算服务** | NLP Service | 8004 | HTTP | ✅ 运行 | `/docs` | `/health` |
| **计算服务** | Image Processing | 8005 | HTTP | ✅ 运行 | `/docs` | `/health` |
| **数据库** | MongoDB | 27018 | TCP | ✅ 运行 | - | - |
| **数据库** | PostgreSQL | 5433 | TCP | ✅ 运行 | - | - |
| **缓存** | Redis | 6380 | TCP | ✅ 运行 | - | - |
| **存储** | MinIO | 9001/9002 | HTTP | ✅ 运行 | `/minio` | - |
| **队列** | RabbitMQ | 5672 | AMQP | ✅ 运行 | `:15672` | - |
| **监控** | Prometheus | 9090 | HTTP | ✅ 运行 | - | - |
| **监控** | Grafana | 3000 | HTTP | ✅ 运行 | - | - |
| **追踪** | Jaeger | 16686 | HTTP | ✅ 运行 | - | - |

---

## 🔌 核心API接口映射

### 1. Storage Service (:8002) - 统一数据管理中心

#### 基础接口
```
GET    /health                    # 健康检查
GET    /ready                     # 就绪检查  
GET    /info                      # 服务信息
GET    /docs                      # API文档
```

#### 文件管理接口
```
POST   /api/v1/files/upload       # 文件上传
GET    /api/v1/files/{file_id}    # 获取文件
DELETE /api/v1/files/{file_id}    # 删除文件
GET    /api/v1/files/list         # 文件列表
POST   /api/v1/files/process      # 文件处理请求
```

#### 内容管理接口
```
POST   /api/v1/contents           # 创建内容
GET    /api/v1/contents/{id}      # 获取内容  
PUT    /api/v1/contents/{id}      # 更新内容
DELETE /api/v1/contents/{id}      # 删除内容
GET    /api/v1/contents/search    # 内容搜索
```

#### 数据集管理接口
```
POST   /api/v1/datasets           # 创建数据集
GET    /api/v1/datasets/{id}      # 获取数据集
PUT    /api/v1/datasets/{id}      # 更新数据集
DELETE /api/v1/datasets/{id}      # 删除数据集
GET    /api/v1/datasets/list      # 数据集列表
```

#### OCR管理接口
```
POST   /api/v1/ocr/tasks          # 创建OCR任务
GET    /api/v1/ocr/tasks/{id}     # 获取任务状态
PUT    /api/v1/ocr/tasks/{id}     # 更新任务状态  
GET    /api/v1/ocr/results/{id}   # 获取OCR结果
```

#### NLP管理接口
```
POST   /api/v1/nlp/tasks          # 创建NLP任务
GET    /api/v1/nlp/tasks/{id}     # 获取任务状态
PUT    /api/v1/nlp/tasks/{id}     # 更新任务状态
GET    /api/v1/nlp/results/{id}   # 获取NLP结果
```

#### 图像处理管理接口
```
POST   /api/v1/image-processing/tasks     # 创建图像处理任务
GET    /api/v1/image-processing/tasks/{id} # 获取任务状态
PUT    /api/v1/image-processing/tasks/{id} # 更新任务状态  
GET    /api/v1/image-processing/results/{id} # 获取处理结果
```

---

### 2. File Processor (:8001) - 文件处理服务

#### 文件处理接口
```
POST   /api/v1/process/extract    # 文本提取
POST   /api/v1/process/validate   # 文件验证
POST   /api/v1/process/scan       # 病毒扫描
GET    /api/v1/formats/supported  # 支持格式列表
```

#### 批量处理接口
```
POST   /api/v1/batch/process      # 批量文件处理
GET    /api/v1/batch/{batch_id}   # 批量任务状态
DELETE /api/v1/batch/{batch_id}   # 取消批量任务
```

---

### 3. OCR Service (:8003) - 文字识别服务

#### OCR处理接口
```
POST   /api/v1/ocr/recognize      # 图像文字识别
POST   /api/v1/ocr/recognize-async # 异步识别
GET    /api/v1/ocr/tasks/{task_id}/status # 任务状态
GET    /api/v1/ocr/tasks/{task_id}/result # 识别结果
```

#### 批量OCR接口
```
POST   /api/v1/ocr/batch-recognize # 批量识别
GET    /api/v1/ocr/batch/{batch_id} # 批量任务状态
DELETE /api/v1/ocr/batch/{batch_id} # 取消批量任务
```

#### 引擎管理接口
```
GET    /api/v1/ocr/engines        # 可用引擎列表
GET    /api/v1/ocr/engines/{engine}/info # 引擎信息
POST   /api/v1/ocr/engines/test   # 引擎测试
```

---

### 4. NLP Service (:8004) - 自然语言处理服务

#### NLP分析接口
```
POST   /api/v1/nlp/analyze        # 综合文本分析
POST   /api/v1/nlp/tokenize       # 分词处理
POST   /api/v1/nlp/pos-tag        # 词性标注
POST   /api/v1/nlp/ner            # 命名实体识别
POST   /api/v1/nlp/sentiment      # 情感分析
POST   /api/v1/nlp/keywords       # 关键词提取
POST   /api/v1/nlp/summarize      # 文本摘要
POST   /api/v1/nlp/similarity     # 相似度计算
```

#### 批量NLP接口
```
POST   /api/v1/nlp/batch-analyze  # 批量分析
GET    /api/v1/nlp/batch/{batch_id} # 批量任务状态
DELETE /api/v1/nlp/batch/{batch_id} # 取消批量任务
```

#### 任务管理接口
```
GET    /api/v1/nlp/tasks          # 任务列表
GET    /api/v1/nlp/tasks/{task_id} # 任务详情
DELETE /api/v1/nlp/tasks/{task_id} # 删除任务
```

---

### 5. Image Processing Service (:8005) - 图像处理服务

#### 图像处理接口
```
POST   /api/v1/images/process     # 图像处理
POST   /api/v1/images/process-async # 异步处理
POST   /api/v1/images/enhance     # 图像增强
POST   /api/v1/images/denoise     # 去噪处理
POST   /api/v1/images/deskew      # 倾斜校正
POST   /api/v1/images/resize      # 尺寸调整
POST   /api/v1/images/convert     # 格式转换
POST   /api/v1/images/assess-quality # 质量评估
```

#### 批量处理接口
```
POST   /api/v1/images/batch-process # 批量处理
GET    /api/v1/images/batch/{batch_id} # 批量任务状态
DELETE /api/v1/images/batch/{batch_id} # 取消批量任务
```

#### 任务管理接口
```
GET    /api/v1/tasks              # 任务列表
GET    /api/v1/tasks/{task_id}    # 任务状态
PUT    /api/v1/tasks/{task_id}    # 更新任务
DELETE /api/v1/tasks/{task_id}    # 删除任务
POST   /api/v1/tasks/{task_id}/retry # 重试任务
```

#### 引擎和统计接口
```
GET    /api/v1/engines            # 处理引擎列表
GET    /api/v1/statistics         # 处理统计信息
```

---

## 🔄 服务间调用流程

### 典型文件处理流程

```
1. 客户端 → Storage Service (:8002)
   POST /api/v1/files/upload
   
2. Storage Service → File Processor (:8001)  
   POST /api/v1/process/extract
   
3. File Processor → Storage Service (:8002)
   返回提取的文本内容
   
4. Storage Service → Database
   保存文件元数据和内容
   
5. Storage Service → 客户端
   返回处理结果
```

### OCR识别流程

```
1. 客户端 → Storage Service (:8002)
   POST /api/v1/ocr/tasks
   
2. Storage Service → OCR Service (:8003)
   POST /api/v1/ocr/recognize-async
   
3. OCR Service → Storage Service (:8002)
   定期更新任务状态和结果
   
4. 客户端 → Storage Service (:8002)  
   GET /api/v1/ocr/results/{id}
```

---

## 🛠️ 开发和测试工具

### API测试命令

#### 健康检查
```bash
# 检查所有服务健康状态
curl http://localhost:8001/health  # File Processor
curl http://localhost:8002/health  # Storage Service  
curl http://localhost:8003/health  # OCR Service
curl http://localhost:8004/health  # NLP Service
curl http://localhost:8005/health  # Image Processing
```

#### 文件上传测试
```bash
# 上传测试文件
curl -X POST "http://localhost:8002/api/v1/files/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.pdf"
```

#### OCR测试
```bash
# OCR识别测试
curl -X POST "http://localhost:8002/api/v1/ocr/tasks" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.jpg", "engine": "tesseract"}'
```

#### NLP分析测试
```bash
# NLP分析测试
curl -X POST "http://localhost:8002/api/v1/nlp/tasks" \
  -H "Content-Type: application/json" \
  -d '{"text": "这是一段中文测试文本", "analysis_types": ["tokenize", "ner", "sentiment"]}'
```

#### 图像处理测试
```bash
# 图像处理测试  
curl -X POST "http://localhost:8002/api/v1/image-processing/tasks" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.jpg", "processing_type": "enhance"}'
```

---

## 📊 监控和日志接口

### Prometheus指标端点
```
http://localhost:8001/metrics  # File Processor 指标
http://localhost:8002/metrics  # Storage Service 指标  
http://localhost:8003/metrics  # OCR Service 指标
http://localhost:8004/metrics  # NLP Service 指标
http://localhost:8005/metrics  # Image Processing 指标
```

### 日志查看
```bash
# Docker容器日志
docker logs storage-service
docker logs file-processor  
docker logs ocr-service
docker logs nlp-service
docker logs image-processing-service

# 实时日志跟踪
docker logs -f storage-service
```

---

## 🔐 认证和权限

### JWT Token获取
```bash
# 获取访问令牌 (假设的认证端点)
curl -X POST "http://localhost:8002/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

### 带认证的API调用
```bash
# 使用JWT Token调用API
curl -X GET "http://localhost:8002/api/v1/contents/list" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

*API接口文档版本: v2.3*  
*最后更新: 2025-09-08*  
*文档维护: Historical Text Project Team*