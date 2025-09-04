# 微服务重命名与架构重构方案

**日期**: 2025-09-04  
**类型**: 重大架构调整  
**影响级别**: 架构重设计  

## 🎯 用户反馈分析

### 用户正确的理解
> "data-source只做文件的处理，最终的存储都是交由data-collection来完成的"

这个理解完全正确！我之前的分离方案还不够彻底，存在以下问题：
1. **data-source仍然保留了MongoDB/Redis连接** - 违背了单一职责原则
2. **服务命名混乱** - data-source/data-collection名称不能清晰表达职责
3. **存储职责分散** - 应该有统一的存储服务管理所有数据库

## 🔄 重新设计的架构

### 新的服务命名和职责

#### 📝 file-processor (原data-source)
**职责**: 纯文件处理服务
```yaml
核心功能:
  ✅ 文件格式处理 (PDF, Word, 图片OCR, HTML等)
  ✅ 文本内容提取
  ✅ 文件安全检测和验证
  ✅ 多格式转换
  ✅ 异步处理工作流

技术特点:
  - 无数据库连接
  - 专注文件处理算法
  - 通过API调用storage-service保存结果
  - 轻量级、高性能的处理服务

API示例:
  POST /api/v1/process/pdf          # PDF文件处理
  POST /api/v1/process/image-ocr    # 图片OCR处理
  POST /api/v1/process/document     # 通用文档处理
  GET  /api/v1/process/status/{id}  # 处理状态查询
```

#### 🗄️ storage-service (原data-collection)  
**职责**: 统一存储和数据管理服务
```yaml
核心功能:
  ✅ 统一数据库管理 (MongoDB + PostgreSQL + Redis + MinIO)
  ✅ 内容数据CRUD (历史文本、元数据)
  ✅ 文件存储管理 (MinIO对象存储)
  ✅ 数据搜索和统计
  ✅ 批量数据操作
  ✅ 业务逻辑和数据验证

技术架构:
  - MongoDB: 内容数据、业务数据
  - PostgreSQL: 文件元数据、处理记录
  - Redis: 缓存、会话、队列
  - MinIO: 对象文件存储
  - RabbitMQ: 消息队列

API示例:
  POST /api/v1/content/             # 创建内容记录
  GET  /api/v1/content/             # 查询内容
  POST /api/v1/files/upload         # 文件上传存储
  GET  /api/v1/content/search       # 内容搜索
  GET  /api/v1/stats/               # 数据统计
```

## 🔧 服务间协作流程

### 标准处理流程
```mermaid
graph TD
    A[Vue3前端] --> B[上传文件到storage-service]
    B --> C[storage-service调用file-processor]
    C --> D[file-processor处理文件返回结果]
    D --> E[storage-service保存所有数据]
    E --> F[返回完整结果给前端]
    
    subgraph "清晰分工"
        G[file-processor: 只处理文件]
        H[storage-service: 管理所有存储]
    end
```

### API调用链
```javascript
// 前端上传文件和内容
const response = await fetch('/api/storage/content/create', {
    method: 'POST',
    body: formData  // 包含文件 + 内容数据
});

// storage-service内部流程:
// 1. 接收文件，存储到MinIO
// 2. 调用file-processor处理文件
// 3. 将提取的文本和元数据存储到数据库
// 4. 返回完整的内容记录给前端
```

## 📋 重构执行计划

### 第1阶段: 服务重命名
```bash
# 重命名目录和服务
mv services/data-source/ services/file-processor/
mv services/data-collection/ services/storage-service/

# 更新配置文件
- 更新Docker Compose配置
- 更新Kubernetes配置  
- 更新服务注册配置
```

### 第2阶段: file-processor重构
```yaml
需要移除:
  - 所有数据库连接代码 (MongoDB, Redis)
  - 数据存储相关代码
  - 内容管理API
  
需要保留:
  - 文件处理器 (PDF, Word, OCR等)
  - 文件验证和安全检测
  - 处理状态管理 (内存中)
  
需要新增:
  - 调用storage-service的API客户端
  - 处理结果回调机制
```

### 第3阶段: storage-service增强
```yaml
需要整合:
  - file-processor的数据库连接
  - 完整的内容管理API
  - 统一的存储管理
  
需要新增:
  - 调用file-processor的API
  - 文件处理工作流编排
  - 统一的业务逻辑层
```

### 第4阶段: API重新设计
```yaml
storage-service API:
  # 内容管理
  POST /api/v1/content/create       # 创建内容(含文件处理)
  GET  /api/v1/content/             # 查询内容
  PUT  /api/v1/content/{id}         # 更新内容
  
  # 文件管理  
  POST /api/v1/files/upload         # 文件上传
  GET  /api/v1/files/{id}           # 文件信息
  
  # 数据统计
  GET  /api/v1/stats/content        # 内容统计
  GET  /api/v1/stats/files          # 文件统计

file-processor API:
  # 文件处理
  POST /api/v1/process/pdf          # PDF处理
  POST /api/v1/process/image        # 图片OCR
  POST /api/v1/process/document     # 通用文档
  GET  /api/v1/process/{task_id}    # 处理状态
```

## 🎯 重构后的优势

### ✅ 架构更清晰
- **file-processor**: 专注文件处理算法，无存储依赖
- **storage-service**: 统一管理所有数据存储和业务逻辑
- **职责边界明确**: 处理和存储完全分离

### ✅ 开发更简单
- 前端只需要调用storage-service的API
- 服务间调用关系简单清晰
- 每个服务的测试和部署更独立

### ✅ 扩展性更好
- file-processor可以独立扩展处理能力
- storage-service可以独立优化存储性能
- 可以轻松添加新的文件处理器

### ✅ 运维更容易
- 存储服务集中管理，备份恢复更简单
- 处理服务无状态，可以随意扩缩容
- 监控和日志更集中

## 📊 配置文件更新

### Docker Compose更新
```yaml
services:
  file-processor:
    build: ./services/file-processor
    ports:
      - "8001:8000"
    environment:
      - STORAGE_SERVICE_URL=http://storage-service:8000
    
  storage-service:
    build: ./services/storage-service  
    ports:
      - "8002:8000"
    depends_on:
      - mongodb
      - postgresql
      - redis
      - minio
```

### 环境变量更新
```yaml
file-processor:
  STORAGE_SERVICE_URL: "http://storage-service:8000"
  LOG_LEVEL: "INFO"
  
storage-service:
  MONGODB_URL: "mongodb://mongodb:27017"
  POSTGRESQL_URL: "postgresql://user:pass@postgresql:5432/db"
  REDIS_URL: "redis://redis:6379"
  MINIO_ENDPOINT: "minio:9000"
```

## 🔄 迁移策略

### 数据迁移
```bash
# file-processor服务清理
1. 移除数据库连接配置
2. 移除数据模型定义
3. 移除存储相关API
4. 保留文件处理核心逻辑

# storage-service服务整合  
1. 整合所有数据库连接
2. 整合内容管理API
3. 添加文件处理调用逻辑
4. 统一业务逻辑层
```

### API兼容性
```yaml
过渡期策略:
  - 保持原有API端点的兼容性
  - 逐步迁移到新的API设计
  - 提供API迁移文档和工具
```

## 📝 文档更新清单

### 需要更新的文档
- [ ] README.md - 更新服务架构说明
- [ ] CLAUDE.md - 更新开发指南
- [ ] docker-compose.yml - 更新服务配置
- [ ] 架构文档 - 重新设计微服务架构图
- [ ] API文档 - 更新API端点和调用方式
- [ ] 部署文档 - 更新Kubernetes配置

### 需要创建的文档
- [ ] 服务重命名和迁移指南
- [ ] 新架构的开发文档
- [ ] API迁移指南
- [ ] 服务调用示例

---

## 🎉 总结

用户的反馈非常准确，指出了当前架构的核心问题：
1. **存储职责应该统一** - 所有数据库连接都应该在storage-service中
2. **文件处理应该纯粹** - file-processor只做处理，不做存储
3. **服务命名应该直观** - 服务名称应该清楚表达其职责

这次重构将建立更清晰、更合理的微服务架构，符合单一职责原则和微服务最佳实践。

**执行建议**: 按阶段进行重构，确保每个阶段都有充分的测试和验证。