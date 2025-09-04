# 数据源服务模块

## 📋 模块概述

数据源服务是历史文本项目的核心模块，专注于历史文本内容的管理、存储和处理。经过架构重构，该模块已简化为纯内容管理服务，移除了复杂的爬虫和代理功能。

## 🎯 核心功能

### 内容管理
- **手动输入**：单个内容的手动添加和编辑
- **批量导入**：支持JSON和CSV文件的批量内容导入
- **内容搜索**：多维度的内容搜索和过滤
- **质量管理**：自动内容质量评估和统计分析

### API接口
- RESTful API设计
- Swagger文档自动生成
- 统一的响应格式
- 完整的错误处理

## 📁 模块结构

```
modules/data-source/
├── docs/                     # 模块文档
│   ├── README.md            # 模块说明
│   ├── api-guide.md         # API使用指南
│   ├── configuration.md     # 配置说明
│   └── examples/            # 使用示例
├── src/                     # 源代码
│   ├── api/                # API接口
│   ├── config/             # 配置管理
│   ├── database/           # 数据库操作
│   ├── models/             # 数据模型
│   └── main.py             # 应用入口
├── tests/                  # 测试文件
└── requirements.txt        # 依赖文件
```

## 🚀 快速开始

### 环境要求
- Python 3.9+
- MongoDB 4.4+
- Redis 6.2+

### 安装依赖
```bash
cd modules/data-source/
pip install -r requirements.txt
```

### 启动服务
```bash
python -m src.main
```

### 验证安装
```bash
# 健康检查
curl http://localhost:8001/health

# API文档
open http://localhost:8001/docs
```

## 📊 配置说明

### 环境变量
```bash
# 数据库配置
DB_MONGODB_URL=mongodb://localhost:27017
DB_MONGODB_DB_NAME=historical_text_data
DB_REDIS_URL=redis://localhost:6379

# 服务配置
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8001
SERVICE_ENVIRONMENT=development

# 内容管理配置
CONTENT_MAX_BATCH_SIZE=100
CONTENT_MAX_FILE_SIZE_MB=50
CONTENT_AUTO_DEDUPLICATE=true
```

### 配置文件
详细配置说明请参考：[configuration.md](./configuration.md)

## 🔧 API使用

### 基础接口
```bash
# 添加单个内容
POST /api/v1/content/

# 批量添加内容
POST /api/v1/content/batch

# 文件导入
POST /api/v1/content/upload

# 获取内容列表
GET /api/v1/content/

# 获取统计信息
GET /api/v1/content/statistics/overview
```

详细API文档请参考：[api-guide.md](./api-guide.md)

## 📈 性能指标

### 基准性能
- **启动时间**：< 6秒
- **内存使用**：< 350MB
- **API响应时间**：< 200ms
- **并发处理**：100+ 请求/秒

### 扩展能力
- **内容存储**：支持百万级内容
- **批量导入**：最大100条/批次
- **文件大小**：最大50MB/文件

## 🧪 测试

### 运行测试
```bash
# 单元测试
python -m pytest tests/unit/ -v

# 集成测试
python -m pytest tests/integration/ -v

# 覆盖率测试
python -m pytest tests/ --cov=src --cov-report=html
```

### 测试数据
测试相关文件存放在 `test-results/` 目录下：
- 测试日志
- 测试数据
- 测试报告

## 📚 相关文档

- [API使用指南](./api-guide.md)
- [配置说明](./configuration.md)
- [部署指南](./deployment.md)
- [故障排除](./troubleshooting.md)

## 🔄 版本历史

### v2.0.0 (2025-09-04)
- 🔥 **重大重构**：移除爬虫和代理模块
- ✨ **新增**：增强的内容管理功能
- ✨ **新增**：文件批量导入支持
- 🐛 **修复**：性能优化和稳定性改进

### v1.3.0 (2025-09-03)
- ✨ 数据采集存储服务集成
- 🔧 完整的RESTful API
- 📊 统计分析功能

### v1.2.0 (2025-09-02)
- ✨ 多平台爬虫支持
- 🔧 代理管理系统
- 🐛 性能优化

## 🆘 技术支持

### 常见问题
请查看：[troubleshooting.md](./troubleshooting.md)

### 获取帮助
- 查看API文档：http://localhost:8001/docs
- 提交Issue到项目仓库
- 联系开发团队

---

*模块文档版本：v2.0*  
*最后更新：2025-09-04*  
*维护团队：数据源服务开发组*