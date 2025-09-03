# 前端开发文档索引

## 概述

本文档是历史文本项目前端开发的主索引文档。为了更好的管理和维护，原始的长文档已被拆分为多个专门的子文档，每个文档专注于特定的开发领域。

## 文档结构

### 📋 [前端开发概览](./frontend-overview.md)
**项目概述、技术栈和架构设计**

- 项目目标和核心功能
- 技术栈选择 (Vue 3.3+, TypeScript 5.0+, Vite 4.0+)
- 项目架构和目录结构
- 核心设计原则和模式
- 快速开始指南

### 🎨 [前端页面设计](./frontend-pages.md)
**8个核心页面的详细设计规范**

- Dashboard - 数据概览和监控面板
- Data Source Management - 数据源管理
- Content Management - 内容管理和AI优化
- Publish Management - 发布平台管理
- Customer Management - 客户管理
- System Settings - 系统设置
- AI Text Optimization Monitoring - AI优化监控
- System Monitoring & Operations - 系统监控运维

### 🧩 [前端组件规范](./frontend-components.md)
**组件设计规范和实现示例**

- 基础组件 (Button, DataTable, Form)
- 业务组件 (ContentCard, StatusMonitor, AITaskCard)
- 图表组件 (MetricsChart, RealtimeChart, HeatmapChart)
- Composable函数规范
- 组件使用指南

### 🔌 [前端API接口设计](./frontend-api.md)
**API接口、状态管理和路由配置**

- HTTP客户端配置 (Axios拦截器)
- API接口设计 (用户认证、内容管理、数据源管理)
- 状态管理设计 (Pinia stores)
- 路由设计和权限控制
- 错误处理机制

### 🛠️ [前端开发规范指南](./frontend-development-guide.md)
**开发规范、测试策略和部署配置**

- 代码规范 (TypeScript, Vue, CSS/SCSS)
- 开发工作流和Git规范
- 测试策略 (单元测试、集成测试、E2E测试)
- 性能优化策略
- 错误处理规范
- 部署配置 (Docker, Nginx, CI/CD)
- 开发工具配置
- 最佳实践总结

## 快速导航

### 🚀 新手入门
1. 阅读 [前端开发概览](./frontend-overview.md) 了解项目整体架构
2. 查看 [前端开发规范指南](./frontend-development-guide.md) 中的开发工具配置
3. 参考 [前端页面设计](./frontend-pages.md) 了解页面结构

### 🔧 开发阶段
1. 查阅 [前端组件规范](./frontend-components.md) 了解可用组件
2. 参考 [前端API接口设计](./frontend-api.md) 进行数据交互
3. 遵循 [前端开发规范指南](./frontend-development-guide.md) 中的编码规范

### 🧪 测试部署
1. 按照 [前端开发规范指南](./frontend-development-guide.md) 中的测试策略进行测试
2. 使用文档中的部署配置进行部署

## 技术栈概览

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue | 3.3+ | 前端框架 |
| TypeScript | 5.0+ | 类型系统 |
| Vite | 4.0+ | 构建工具 |
| Pinia | 2.1+ | 状态管理 |
| Element Plus | 2.3+ | UI组件库 |
| Tailwind CSS | 3.3+ | CSS框架 |
| Chart.js/ECharts | - | 图表库 |
| Axios | 1.4+ | HTTP客户端 |
| Vue Router | 4.2+ | 路由管理 |

## 项目特色

- 🎯 **AI驱动**: 集成AI文本优化功能
- 📊 **数据可视化**: 丰富的图表和监控面板
- 🔄 **实时更新**: WebSocket实时数据推送
- 📱 **响应式设计**: 支持多设备访问
- 🛡️ **类型安全**: 全面的TypeScript支持
- ⚡ **高性能**: 代码分割和懒加载优化
- 🧪 **测试完备**: 完整的测试覆盖
- 🚀 **自动化部署**: CI/CD流水线

## 贡献指南

1. **代码规范**: 严格遵循 [前端开发规范指南](./frontend-development-guide.md)
2. **提交规范**: 使用约定式提交格式
3. **测试要求**: 新功能必须包含相应测试
4. **文档更新**: 重要变更需要更新相关文档
5. **代码审查**: 所有代码变更需要经过审查

## 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 技术支持: tech-support@example.com
- 💬 团队讨论: #frontend-dev 频道
- 📝 问题反馈: 项目Issue页面

---

**最后更新**: 2024年1月
**文档版本**: v2.0
**维护团队**: 前端开发组