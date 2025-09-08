# 测试结果管理

此目录用于存储所有测试输出，按时间戳进行组织。

## 目录结构

```
test-results/
├── YYYY-MM-DD-HHMMSS/     # 每次测试运行的时间戳目录
│   ├── coverage/           # 覆盖率报告
│   ├── test-logs/         # 测试日志
│   ├── performance/       # 性能指标
│   └── screenshots/       # E2E测试截图 (如适用)
└── README.md              # 本说明文件
```

## 使用指南

### 运行测试时
所有测试命令应自动创建带时间戳的目录：
```bash
# 示例测试运行
pytest --cov=src --cov-report=html:test-results/$(date +"%Y-%m-%d-%H%M%S")/coverage
```

### 清理旧结果
建议定期清理30天以上的测试结果：
```bash
find test-results/ -name "20*" -type d -mtime +30 -exec rm -rf {} \;
```

## 注意事项

- 每次重大测试运行都应创建新的时间戳目录
- 保留重要里程碑的测试结果用于历史对比
- CI/CD流水线应自动管理测试结果的生成和清理