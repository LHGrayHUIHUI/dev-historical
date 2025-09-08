# 🏗️ 历史文本项目系统架构图

## 📊 完整系统架构概览

```mermaid
graph TB
    %% 前端层
    subgraph "Frontend Layer"
        VUE[Vue 3 Frontend<br/>管理界面]
        API[API Gateway<br/>Kong/Nginx]
    end
    
    %% 微服务层
    subgraph "Microservices Layer"
        STORAGE[Storage Service<br/>:8002<br/>统一数据管理]
        FILE[File Processor<br/>:8001<br/>文件处理]
        OCR[OCR Service<br/>:8003<br/>文字识别]
        NLP[NLP Service<br/>:8004<br/>文本分析]
        IMG[Image Processing<br/>:8005<br/>图像处理]
    end
    
    %% 存储层
    subgraph "Storage Layer"
        MONGO[(MongoDB<br/>:27018<br/>文档数据)]
        PG[(PostgreSQL<br/>:5433<br/>结构化数据)]
        REDIS[(Redis<br/>:6380<br/>缓存/队列)]
        MINIO[(MinIO<br/>:9001/9002<br/>文件存储)]
        RABBIT[(RabbitMQ<br/>:5672<br/>消息队列)]
    end
    
    %% 监控层
    subgraph "Monitoring Layer"
        PROM[Prometheus<br/>:9090<br/>指标收集]
        GRAF[Grafana<br/>:3000<br/>监控面板]
        JAEGER[Jaeger<br/>:16686<br/>链路追踪]
        ELK[ELK Stack<br/>日志管理]
    end
    
    %% 连接关系
    VUE --> API
    API --> STORAGE
    
    %% 服务间通信
    STORAGE <--> FILE
    STORAGE <--> OCR  
    STORAGE <--> NLP
    STORAGE <--> IMG
    
    %% 存储连接 (只有Storage Service连接数据库)
    STORAGE --> MONGO
    STORAGE --> PG
    STORAGE --> REDIS
    STORAGE --> MINIO
    STORAGE --> RABBIT
    
    %% 监控连接
    STORAGE -.-> PROM
    FILE -.-> PROM
    OCR -.-> PROM
    NLP -.-> PROM
    IMG -.-> PROM
    
    PROM --> GRAF
    STORAGE -.-> JAEGER
    FILE -.-> JAEGER
    OCR -.-> JAEGER
    NLP -.-> JAEGER
    IMG -.-> JAEGER
    
    %% 样式
    classDef frontend fill:#e1f5fe
    classDef service fill:#f3e5f5
    classDef storage fill:#e8f5e8
    classDef monitor fill:#fff3e0
    
    class VUE,API frontend
    class STORAGE,FILE,OCR,NLP,IMG service
    class MONGO,PG,REDIS,MINIO,RABBIT storage
    class PROM,GRAF,JAEGER,ELK monitor
```

---

## 🔄 数据流向详细图

### 1. 文件处理流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Storage as Storage Service<br/>(:8002)
    participant File as File Processor<br/>(:8001)
    participant DB as 数据库集群
    
    Client->>Storage: 1. 上传文件请求
    Storage->>File: 2. 调用文件处理API
    File->>File: 3. 文件格式验证
    File->>File: 4. 病毒扫描检查
    File->>File: 5. 文本内容提取
    File-->>Storage: 6. 返回处理结果
    Storage->>DB: 7. 保存文件元数据
    Storage->>DB: 8. 存储文本内容
    Storage-->>Client: 9. 返回处理完成
```

### 2. OCR文字识别流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Storage as Storage Service<br/>(:8002)
    participant OCR as OCR Service<br/>(:8003)
    participant Redis as Redis缓存
    participant DB as 数据库集群
    
    Client->>Storage: 1. OCR识别请求
    Storage->>Redis: 2. 检查缓存
    alt 缓存命中
        Redis-->>Storage: 3a. 返回缓存结果
    else 缓存未命中
        Storage->>OCR: 3b. 调用OCR识别
        OCR->>OCR: 4. 图像预处理
        OCR->>OCR: 5. 文字识别 (Tesseract/PaddleOCR)
        OCR->>OCR: 6. 结果后处理
        OCR-->>Storage: 7. 返回识别结果
        Storage->>Redis: 8. 缓存结果
    end
    Storage->>DB: 9. 保存OCR结果
    Storage-->>Client: 10. 返回识别文本
```

### 3. NLP文本分析流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Storage as Storage Service<br/>(:8002)
    participant NLP as NLP Service<br/>(:8004)
    participant DB as 数据库集群
    
    Client->>Storage: 1. NLP分析请求
    Storage->>NLP: 2. 调用NLP分析API
    
    par 并行处理多个NLP任务
        NLP->>NLP: 3a. 分词处理
        NLP->>NLP: 3b. 词性标注
        NLP->>NLP: 3c. 命名实体识别
        NLP->>NLP: 3d. 情感分析
        NLP->>NLP: 3e. 关键词提取
        NLP->>NLP: 3f. 文本摘要
    end
    
    NLP-->>Storage: 4. 返回分析结果
    Storage->>DB: 5. 保存NLP分析数据
    Storage-->>Client: 6. 返回分析结果
```

### 4. 图像处理流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant Storage as Storage Service<br/>(:8002)
    participant IMG as Image Processing<br/>(:8005)
    participant MinIO as MinIO存储
    participant DB as 数据库集群
    
    Client->>Storage: 1. 图像处理请求
    Storage->>IMG: 2. 调用图像处理API
    IMG->>IMG: 3. 图像质量评估
    
    par 并行图像处理
        IMG->>IMG: 4a. 图像增强
        IMG->>IMG: 4b. 去噪处理
        IMG->>IMG: 4c. 倾斜校正
        IMG->>IMG: 4d. 尺寸调整
    end
    
    IMG-->>Storage: 5. 返回处理后图像
    Storage->>MinIO: 6. 存储处理后图像
    Storage->>DB: 7. 保存处理记录
    Storage-->>Client: 8. 返回处理结果
```

---

## 🌐 服务间调用关系详图

```mermaid
graph LR
    %% 服务定义
    Client[客户端应用]
    
    subgraph "核心服务架构"
        Storage[Storage Service<br/>:8002<br/>🔄 有状态服务]
        
        subgraph "无状态计算服务"
            File[File Processor<br/>:8001]
            OCR[OCR Service<br/>:8003] 
            NLP[NLP Service<br/>:8004]
            IMG[Image Processing<br/>:8005]
        end
    end
    
    subgraph "数据存储集群"
        MongoDB[(MongoDB<br/>文档存储)]
        PostgreSQL[(PostgreSQL<br/>关系数据)]
        Redis[(Redis<br/>缓存)]
        MinIO[(MinIO<br/>文件存储)]
        RabbitMQ[(RabbitMQ<br/>消息队列)]
    end
    
    %% API调用关系
    Client -->|HTTP REST API| Storage
    
    %% 服务间HTTP调用
    Storage -.->|HTTP Client| File
    Storage -.->|HTTP Client| OCR
    Storage -.->|HTTP Client| NLP
    Storage -.->|HTTP Client| IMG
    
    %% 数据库连接 (只有Storage Service)
    Storage -->|数据CRUD| MongoDB
    Storage -->|元数据管理| PostgreSQL
    Storage -->|缓存/队列| Redis
    Storage -->|文件上传/下载| MinIO
    Storage -->|异步任务| RabbitMQ
    
    %% 服务状态标识
    classDef stateless fill:#e3f2fd,stroke:#1976d2
    classDef stateful fill:#f3e5f5,stroke:#7b1fa2
    classDef storage fill:#e8f5e8,stroke:#388e3c
    classDef client fill:#fff3e0,stroke:#f57c00
    
    class File,OCR,NLP,IMG stateless
    class Storage stateful
    class MongoDB,PostgreSQL,Redis,MinIO,RabbitMQ storage
    class Client client
```

---

## 📋 服务详细信息表

### 🚀 微服务端口分配

| 服务名称 | 端口 | 状态 | 架构类型 | 主要功能 | 依赖服务 |
|---------|------|------|---------|---------|----------|
| **Storage Service** | 8002 | ✅ 运行 | 有状态 | 统一数据管理中心 | MongoDB, PostgreSQL, Redis, MinIO, RabbitMQ |
| **File Processor** | 8001 | ✅ 运行 | 无状态 | 文件处理和内容提取 | Storage Service |
| **OCR Service** | 8003 | ✅ 运行 | 无状态 | 光学字符识别 | Storage Service |
| **NLP Service** | 8004 | ✅ 运行 | 无状态 | 自然语言处理 | Storage Service |
| **Image Processing** | 8005 | ✅ 运行 | 无状态 | 图像处理和优化 | Storage Service |

### 💾 数据存储层

| 存储类型 | 端口 | 主要用途 | 数据类型 |
|----------|------|---------|----------|
| **MongoDB** | 27018 | 文档和内容存储 | 历史文本、处理结果、业务数据 |
| **PostgreSQL** | 5433 | 关系数据管理 | 文件元数据、用户信息、审计日志 |
| **Redis** | 6380 | 缓存和队列 | 会话缓存、任务队列、临时数据 |
| **MinIO** | 9001/9002 | 对象存储 | 原始文件、处理后文件、备份 |
| **RabbitMQ** | 5672 | 消息队列 | 异步任务、事件通知 |

### 📊 监控和日志

| 组件 | 端口 | 功能 | 监控对象 |
|------|------|------|----------|
| **Prometheus** | 9090 | 指标收集 | 所有微服务的性能指标 |
| **Grafana** | 3000 | 可视化面板 | 系统监控仪表板 |
| **Jaeger** | 16686 | 分布式追踪 | 服务间调用链路 |
| **ELK Stack** | 5601 | 日志管理 | 集中日志分析和搜索 |

---

## 🔒 安全和通信协议

### 🌐 服务间通信

1. **HTTP REST API**
   - 所有服务间采用HTTP/HTTPS协议
   - JSON格式数据交换
   - 标准RESTful接口设计

2. **身份验证**
   - JWT Token认证机制
   - 服务间API密钥验证
   - 请求速率限制

3. **数据加密**
   - HTTPS传输加密
   - 敏感数据数据库加密
   - 文件存储加密

### 📈 性能优化策略

1. **缓存策略**
   - Redis多层缓存
   - OCR结果缓存
   - 静态资源CDN缓存

2. **负载均衡**
   - 无状态服务水平扩展
   - Kubernetes Pod自动缩放
   - 数据库连接池优化

3. **异步处理**
   - RabbitMQ消息队列
   - 长时间任务异步执行
   - 批量处理优化

---

## 🚀 扩展性设计

### 水平扩展能力

```mermaid
graph TB
    subgraph "负载均衡层"
        LB[Load Balancer<br/>Kong/Nginx]
    end
    
    subgraph "无状态服务集群"
        F1[File Processor #1]
        F2[File Processor #2]
        F3[File Processor #3]
        
        O1[OCR Service #1]
        O2[OCR Service #2]
        
        N1[NLP Service #1]
        N2[NLP Service #2]
        
        I1[Image Processing #1]
        I2[Image Processing #2]
    end
    
    subgraph "有状态服务"
        S1[Storage Service]
        S2[Storage Service Replica]
    end
    
    subgraph "数据库集群"
        DB[(Database Cluster)]
    end
    
    LB --> F1 & F2 & F3
    LB --> O1 & O2
    LB --> N1 & N2
    LB --> I1 & I2
    LB --> S1
    
    S1 <--> S2
    S1 --> DB
    S2 --> DB
    
    F1 & F2 & F3 -.-> S1
    O1 & O2 -.-> S1
    N1 & N2 -.-> S1
    I1 & I2 -.-> S1
```

---

*文档创建时间: 2025-09-08*  
*系统架构版本: v2.3*  
*维护者: Historical Text Project Team*