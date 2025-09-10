# AI模型服务 (AI Model Service)

统一的AI模型调用和管理服务，提供多平台AI模型的智能路由、负载均衡、账号池管理和使用统计功能。

## 🚀 功能特性

### 核心功能
- **多平台支持**: 支持OpenAI、Claude、百度文心一言、阿里云通义千问、腾讯混元、智谱AI等主流平台
- **智能路由**: 基于健康状态、成本、性能等指标智能选择最佳模型和账号
- **负载均衡**: 支持轮询、权重、健康评分等多种负载均衡策略  
- **账号池管理**: 多账号轮换、健康监控、配额管理
- **流式支持**: 完整支持流式对话接口
- **监控告警**: 实时监控账号状态、API可用性和性能指标

### 高级功能
- **使用统计**: 详细的调用统计、成本分析和性能指标
- **故障转移**: 自动检测故障账号并切换到备用账号
- **缓存优化**: Redis缓存提升路由决策性能
- **配额管理**: 智能配额分配和使用预警
- **安全加密**: API密钥加密存储，安全管理凭证

## 📋 技术架构

### 服务架构
```
                    ┌─────────────────┐
                    │   FastAPI App   │
                    └─────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼────┐  ┌──────▼──────┐ ┌──────▼──────┐
    │Chat API    │  │Models API   │ │Status API   │
    │Controller  │  │Controller   │ │Controller   │
    └───────┬────┘  └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                    ┌───────▼────────┐
                    │  AI Service    │
                    │  (核心业务层)    │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐ ┌───────▼───────┐ ┌────────▼────────┐
│Model Router    │ │Account Monitor│ │Usage Tracker    │
│(智能路由)       │ │(健康监控)      │ │(使用统计)        │
└───────┬────────┘ └───────────────┘ └─────────────────┘
        │
        └─────────────────────────────────────────────┐
                                                      │
┌─────────────────────────────────────────────────────▼─────────┐
│                    Adapter Factory                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ OpenAI   │ │  Claude  │ │  Baidu   │ │ Alibaba/Tencent  │ │
│  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ /Zhipu Adapters  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 数据流向
```
请求 → 路由选择 → 适配器调用 → 平台API → 响应处理 → 统计记录
```

## 🛠️ 环境配置

### 环境变量
```bash
# 服务配置
SERVICE_PORT=8007
DEBUG=True
LOG_LEVEL=INFO

# 存储服务URL (通过storage-service访问数据库)
STORAGE_SERVICE_URL=http://localhost:8002

# Redis配置 (可选，用于缓存和负载均衡)
REDIS_URL=redis://localhost:6380

# 监控配置
HEALTH_CHECK_INTERVAL=300
QUOTA_ALERT_THRESHOLD=0.8

# 缓存配置
CACHE_TTL_MODELS=3600
CACHE_TTL_ACCOUNTS=1800
CACHE_PREFIX=ai_model_service:

# 路由策略
DEFAULT_ROUTING_STRATEGY=priority
```

### 依赖要求
- Python 3.11+
- FastAPI 0.104+
- Redis (可选)
- Storage Service (端口8002)

## 🚀 快速启动

### 1. 开发环境
```bash
# 克隆项目
cd services/ai-model-service

# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m src.main
```

### 2. Docker环境
```bash
# 构建镜像
docker build -t ai-model-service .

# 运行容器
docker run -p 8007:8007 -e STORAGE_SERVICE_URL=http://host.docker.internal:8002 ai-model-service
```

### 3. 微服务集成
```bash
# 使用docker-compose启动完整环境
docker-compose -f docker-compose.dev.yml up ai-model-service
```

## 📊 API文档

服务启动后访问：
- **Swagger文档**: http://localhost:8007/docs
- **ReDoc文档**: http://localhost:8007/redoc

### 核心接口

#### 1. 聊天完成
```http
POST /api/v1/chat/completions
Content-Type: application/json

{
  "model_name": "GPT-3.5",
  "provider": "openai",
  "messages": [
    {"role": "system", "content": "你是一个有用的助手"},
    {"role": "user", "content": "你好"}
  ],
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "user_id": "user123"
}
```

#### 2. 流式聊天
```http
POST /api/v1/chat/completions/stream
Content-Type: application/json

{
  "model_name": "GPT-4",
  "messages": [
    {"role": "user", "content": "写一首关于AI的诗"}
  ],
  "parameters": {"temperature": 0.8}
}
```

#### 3. 获取可用模型
```http
GET /api/v1/models/
GET /api/v1/models/?provider=openai
GET /api/v1/models/openai/models
```

#### 4. 服务状态监控
```http
GET /api/v1/status/health
GET /api/v1/status/metrics  
GET /api/v1/status/usage?period=24h
GET /api/v1/status/performance
GET /api/v1/status/cost?group_by=provider
```

## 🎯 智能路由策略

### 支持的路由策略

1. **优先级路由** (priority)
   - 按模型优先级选择
   - 适合有明确偏好的场景

2. **成本优化路由** (cost_based)  
   - 优先选择成本较低的模型
   - 适合成本敏感的应用

3. **轮询路由** (round_robin)
   - 公平分配请求到各个模型
   - 适合负载均衡场景

4. **权重路由** (weighted)
   - 按模型权重随机选择
   - 支持灵活的流量分配

5. **健康评分路由** (health_based)
   - 基于账号健康状态选择
   - 确保服务可用性

### 路由决策因素
- **账号健康评分**: 成功率、响应时间、错误率
- **配额使用情况**: 剩余配额、使用率
- **模型能力匹配**: 功能需求、性能要求  
- **成本考虑**: 按token成本、预算限制

## 📈 监控和统计

### 健康监控指标
- **成功率**: API调用成功率
- **响应时间**: 平均响应时间、P95/P99
- **错误率**: 各种错误类型统计
- **配额使用**: 配额使用率、剩余配额

### 使用统计
- **调用量统计**: 按时间、模型、用户维度
- **成本分析**: 按提供商、模型的成本分布
- **性能指标**: 吞吐量、延迟分布
- **错误分析**: 错误类型、频率分析

### 告警机制
- **配额告警**: 配额使用超过阈值时告警
- **健康告警**: 账号健康评分低于阈值时告警
- **错误告警**: 错误率超过阈值时告警
- **可用性告警**: 服务不可用时告警

## 🔧 配置管理

### 模型配置
通过Storage Service管理模型配置:
```json
{
  "id": "gpt-3.5-turbo",
  "name": "GPT-3.5 Turbo",
  "provider": "openai", 
  "model_id": "gpt-3.5-turbo",
  "max_tokens": 4096,
  "cost_per_1k_tokens": 0.002,
  "priority": 5,
  "is_active": true,
  "capabilities": {
    "chat": true,
    "streaming": true,
    "function_calling": true
  }
}
```

### 账号配置
```json
{
  "id": "openai-account-1",
  "account_name": "OpenAI主账号",
  "provider": "openai",
  "api_key_encrypted": "encrypted_key_here",
  "quota_limit": 1000000,
  "quota_used": 45000,
  "status": "active",
  "health_score": 0.95
}
```

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit/ -v

# 运行带覆盖率的测试
pytest --cov=src --cov-report=html --cov-report=term

# 运行特定测试
pytest tests/unit/test_adapters.py::TestOpenAIAdapter::test_chat_completion_success -v
```

### 测试结构
```
tests/
├── conftest.py              # 测试配置和夹具
├── unit/                    # 单元测试
│   ├── test_adapters.py     # 适配器测试
│   ├── test_ai_service.py   # AI服务测试
│   ├── test_controllers.py  # 控制器测试
│   └── test_models.py       # 数据模型测试
└── integration/             # 集成测试
    ├── test_api_endpoints.py # API端点测试
    └── test_service_integration.py
```

## 🐳 容器化部署

### 构建优化镜像
```bash
# 多阶段构建，优化镜像大小
docker build -f Dockerfile -t ai-model-service:latest .

# 查看镜像大小
docker images ai-model-service:latest
```

### 生产部署
```bash
# 使用生产配置启动
docker-compose -f docker-compose.production.yml up ai-model-service

# 或使用Kubernetes部署
kubectl apply -f k8s/ai-model-service-deployment.yaml
```

### 健康检查
容器内置健康检查：
```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8007/api/v1/status/health || exit 1
```

## 🔒 安全考虑

### API密钥安全
- **加密存储**: 所有API密钥加密后存储在数据库中
- **权限隔离**: 每个账号使用独立的API密钥
- **定期轮换**: 支持API密钥的定期更新

### 访问控制  
- **认证授权**: 支持JWT令牌认证
- **请求限制**: 支持按用户/IP的请求频率限制
- **敏感信息**: 错误信息中自动过滤敏感数据

### 网络安全
- **HTTPS支持**: 生产环境强制HTTPS
- **CORS配置**: 合理的跨域资源共享配置
- **防火墙**: 只暴露必要的服务端口

## 📝 开发指南

### 添加新的AI平台适配器

1. **创建适配器类**
```python
# src/adapters/new_platform_adapter.py
from .base_adapter import BaseAdapter

class NewPlatformAdapter(BaseAdapter):
    async def chat_completion(self, model_config, account_config, messages, **kwargs):
        # 实现聊天完成逻辑
        pass
    
    async def chat_completion_stream(self, model_config, account_config, messages, **kwargs):
        # 实现流式聊天逻辑  
        pass
```

2. **注册适配器**
```python
# src/adapters/adapter_factory.py
from .new_platform_adapter import NewPlatformAdapter

self._adapter_classes[ModelProvider.NEW_PLATFORM] = NewPlatformAdapter
```

3. **添加测试**
```python
# tests/unit/test_adapters.py
class TestNewPlatformAdapter:
    # 添加适配器测试用例
    pass
```

### 扩展路由策略

1. **在model_router.py中添加新策略**
```python
elif strategy_type == 'new_strategy':
    selected = await self._new_strategy_select(models)
    return selected, 'new_strategy'
```

2. **实现策略逻辑**
```python
async def _new_strategy_select(self, models):
    # 实现新的选择逻辑
    return selected_model
```

## 🤝 贡献指南

1. Fork项目并创建功能分支
2. 编写代码并添加测试用例
3. 确保测试通过: `pytest`
4. 检查代码风格: `black src/ && isort src/ && flake8 src/`
5. 提交PR并描述变更内容

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🆘 故障排除

### 常见问题

1. **服务启动失败**
   - 检查环境变量配置是否正确
   - 确认Storage Service是否可访问
   - 查看日志文件了解详细错误

2. **API调用失败**
   - 验证AI平台API密钥是否正确
   - 检查网络连接是否正常
   - 查看账号配额是否充足

3. **性能问题**
   - 启用Redis缓存加速路由决策
   - 调整健康检查间隔
   - 优化数据库查询性能

4. **内存使用过高**
   - 检查批量统计记录是否正常刷新
   - 调整缓存TTL时间
   - 监控适配器连接池

### 调试技巧
- 启用DEBUG模式查看详细日志
- 使用 `/debug/config` 端点查看配置
- 通过 `/api/v1/status/metrics` 监控关键指标

---

**联系方式**: 如有问题请在GitHub Issues中提出或联系开发团队。