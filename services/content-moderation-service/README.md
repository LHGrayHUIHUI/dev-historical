# 内容审核服务 (Content Moderation Service)

基于AI和机器学习的智能内容审核微服务，支持文本、图像、视频、音频等多媒体内容的自动审核和人工复审。

## 🚀 功能特性

### 核心功能
- **多媒体内容审核**: 支持文本、图像、视频、音频内容的智能分析
- **实时审核**: 异步处理，支持实时和批量审核模式
- **智能分类**: 基于AI模型进行内容分类和风险评估
- **人工复审**: 支持审核员介入和申诉处理流程
- **规则管理**: 灵活的审核规则配置和敏感词库管理

### 技术特性
- **高性能**: 基于FastAPI和异步处理的高性能架构
- **可扩展**: 微服务架构，支持水平扩展
- **AI集成**: 集成多种AI模型和算法
- **监控完善**: 完整的健康检查和指标监控
- **数据安全**: 支持内容去重和隐私保护

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    内容审核服务                                │
├─────────────────────────────────────────────────────────────┤
│  API层                                                      │
│  ├── 审核任务管理 (moderation_controller.py)                │
│  ├── 管理员功能 (admin_controller.py)                        │
│  └── 健康检查 (health_controller.py)                         │
├─────────────────────────────────────────────────────────────┤
│  服务层                                                      │
│  └── 内容审核服务 (moderation_service.py)                    │
├─────────────────────────────────────────────────────────────┤
│  分析器层                                                     │
│  ├── 文本分析器 (TextAnalyzer)                               │
│  ├── 图像分析器 (ImageAnalyzer)                              │
│  ├── 视频分析器 (VideoAnalyzer)                              │
│  └── 音频分析器 (AudioAnalyzer)                              │
├─────────────────────────────────────────────────────────────┤
│  数据层                                                      │
│  ├── PostgreSQL (任务数据)                                   │
│  ├── Redis (缓存&队列)                                       │
│  └── MinIO (文件存储)                                        │
└─────────────────────────────────────────────────────────────┘
```

## 📦 目录结构

```
content-moderation-service/
├── src/
│   ├── analyzers/           # 内容分析器
│   │   ├── base_analyzer.py    # 基础分析器
│   │   ├── text_analyzer.py    # 文本分析器
│   │   ├── image_analyzer.py   # 图像分析器
│   │   ├── video_analyzer.py   # 视频分析器
│   │   └── audio_analyzer.py   # 音频分析器
│   ├── config/              # 配置管理
│   │   └── settings.py         # 应用配置
│   ├── controllers/         # API控制器
│   │   ├── moderation_controller.py  # 审核API
│   │   ├── admin_controller.py       # 管理API
│   │   └── health_controller.py      # 健康检查
│   ├── models/              # 数据模型
│   │   ├── database.py         # 数据库配置
│   │   ├── moderation_models.py # SQLAlchemy模型
│   │   └── schemas.py          # Pydantic模式
│   ├── services/            # 业务逻辑
│   │   └── moderation_service.py # 核心审核服务
│   └── main.py              # 应用入口
├── tests/                   # 测试代码
├── requirements.txt         # 依赖列表
├── Dockerfile              # Docker配置
└── README.md               # 说明文档
```

## 🛠️ 安装与部署

### 环境要求

- Python 3.11+
- PostgreSQL 13+
- Redis 6+
- Docker & Docker Compose (可选)

### 本地开发

1. **克隆代码**
```bash
cd services/content-moderation-service
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等
```

4. **初始化数据库**
```bash
# 确保PostgreSQL服务运行中
python -c "
from src.models.database import create_all_tables
import asyncio
asyncio.run(create_all_tables())
"
```

5. **启动服务**
```bash
python -m src.main
```

服务将在 http://localhost:8090 启动

### Docker 部署

1. **构建镜像**
```bash
docker build -t content-moderation-service .
```

2. **运行容器**
```bash
docker run -d \
  --name content-moderation \
  -p 8090:8090 \
  -e DATABASE_URL="postgresql://user:pass@host:port/db" \
  -e REDIS_URL="redis://host:port/db" \
  content-moderation-service
```

### 使用 Docker Compose

```bash
# 在项目根目录
docker-compose -f docker-compose.dev.yml up content-moderation-service
```

## 🔧 配置说明

### 主要配置项

```python
# 数据库配置
DATABASE_URL = "postgresql://postgres:password@localhost:5433/historical_text_moderation"
REDIS_URL = "redis://localhost:6380/10"

# 服务配置
APP_NAME = "Content Moderation Service"
DEBUG = False
HOST = "0.0.0.0"
PORT = 8090

# 文件处理限制
MAX_FILE_SIZE = 200MB
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif"]
SUPPORTED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/mov"]
SUPPORTED_AUDIO_TYPES = ["audio/mp3", "audio/wav", "audio/aac"]

# 审核阈值
TEXT_CONFIDENCE_THRESHOLD = 0.8
IMAGE_CONFIDENCE_THRESHOLD = 0.7
VIDEO_CONFIDENCE_THRESHOLD = 0.75
AUDIO_CONFIDENCE_THRESHOLD = 0.6
```

## 📚 API 文档

### 核心审核接口

#### 创建审核任务
```http
POST /api/v1/moderation/tasks
Content-Type: application/json

{
  "content_id": "unique-content-id",
  "content_type": "text",
  "content_text": "待审核的文本内容",
  "source_platform": "platform-name",
  "user_id": "user-uuid"
}
```

#### 获取审核结果
```http
GET /api/v1/moderation/tasks/{task_id}/result
```

#### 快速内容分析
```http
POST /api/v1/moderation/analyze/quick
Content-Type: application/json

{
  "content": "待分析的内容",
  "content_type": "text",
  "quick_mode": true
}
```

#### 批量审核
```http
POST /api/v1/moderation/tasks/batch
Content-Type: application/json

{
  "tasks": [
    {
      "content_id": "id1",
      "content_type": "text",
      "content_text": "文本1"
    },
    {
      "content_id": "id2", 
      "content_type": "image",
      "content_url": "http://example.com/image.jpg"
    }
  ]
}
```

### 管理员接口

#### 规则管理
```http
POST /api/v1/admin/rules
GET /api/v1/admin/rules
PUT /api/v1/admin/rules/{rule_id}
DELETE /api/v1/admin/rules/{rule_id}
```

#### 敏感词管理
```http
POST /api/v1/admin/sensitive-words
GET /api/v1/admin/sensitive-words
```

#### 统计信息
```http
GET /api/v1/admin/stats?days=7
```

### 健康检查

```http
GET /api/v1/health              # 基础健康检查
GET /api/v1/health/detailed     # 详细健康状态
GET /api/v1/ready              # Kubernetes就绪探针
GET /api/v1/live               # Kubernetes存活探针
GET /api/v1/metrics            # Prometheus指标
```

## 🧪 测试

### 运行单元测试
```bash
pytest tests/ -v
```

### 运行特定测试
```bash
pytest tests/test_text_analyzer.py -v
```

### 测试覆盖率
```bash
pytest --cov=src --cov-report=html
```

## 📊 监控与运维

### 健康检查

服务提供多个健康检查端点：

- `/api/v1/health` - 基础健康检查
- `/api/v1/health/detailed` - 详细组件状态
- `/api/v1/ready` - Kubernetes就绪探针
- `/api/v1/live` - Kubernetes存活探针

### Prometheus指标

服务暴露以下关键指标：

- `content_moderation_uptime_seconds` - 服务运行时间
- `content_moderation_memory_usage_percent` - 内存使用率
- `content_moderation_cpu_usage_percent` - CPU使用率
- `content_moderation_tasks_total` - 处理任务总数
- `content_moderation_violations_total` - 检测违规总数

### 日志管理

服务使用结构化日志记录：

```python
# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    }
}
```

## 🔒 安全考虑

### 数据安全
- 内容哈希去重，避免重复处理
- 敏感信息脱敏处理
- 文件大小和类型限制

### 访问控制
- API速率限制
- 请求验证和授权
- CORS和受信任主机配置

### 隐私保护
- 审核结果数据加密存储
- 定期数据清理策略
- 遵循数据保护法规

## 🚨 故障处理

### 常见问题

1. **数据库连接失败**
```bash
# 检查数据库配置和连接
python -c "from src.models.database import get_database_url; print(get_database_url())"
```

2. **AI模型加载失败**
```bash
# 检查模型文件和依赖
ls -la models/
pip list | grep -E "(tensorflow|torch|opencv)"
```

3. **内存不足**
```bash
# 监控内存使用情况
curl http://localhost:8090/api/v1/health/detailed
```

### 性能优化

1. **分析器优化**
   - 调整置信度阈值
   - 启用GPU加速
   - 使用模型量化

2. **数据库优化**
   - 索引优化
   - 连接池调优
   - 查询优化

3. **缓存策略**
   - Redis缓存配置
   - 内容去重缓存
   - 结果缓存策略

## 📈 扩展开发

### 添加新的分析器

1. 继承 `BaseAnalyzer` 类
2. 实现必要的抽象方法
3. 注册到服务中

```python
from src.analyzers.base_analyzer import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def __init__(self, config=None):
        super().__init__(config)
    
    async def analyze(self, content, metadata=None):
        # 实现分析逻辑
        pass
    
    def get_supported_types(self):
        return ["custom/type"]
```

### 添加新的审核规则

通过管理员API添加规则：

```json
{
  "name": "自定义规则",
  "rule_type": "keyword",
  "content_types": ["text"],
  "rule_config": {
    "keywords": ["关键词1", "关键词2"],
    "case_sensitive": false
  },
  "severity": "high",
  "action": "block"
}
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目地址: [GitHub Repository](https://github.com/your-org/historical-text-project)
- 问题反馈: [Issues](https://github.com/your-org/historical-text-project/issues)
- 文档地址: [Documentation](https://docs.your-domain.com)

## 🗺️ 发展路线

### 近期计划
- [ ] 支持更多AI模型集成
- [ ] 添加实时流处理能力
- [ ] 优化批量处理性能
- [ ] 增强监控和告警功能

### 长期规划
- [ ] 多语言内容支持
- [ ] 联邦学习和隐私计算
- [ ] 边缘计算部署支持
- [ ] 图形化管理界面

---

**注意**: 本服务是历史文本优化项目的一部分，专注于提供企业级的内容审核解决方案。