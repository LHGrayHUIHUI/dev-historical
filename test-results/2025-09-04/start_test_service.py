#!/usr/bin/env python3
"""
测试服务启动脚本 - 用于测试环境
"""

import uvicorn
import sys
import os
from pathlib import Path

# 添加服务模块路径
service_path = Path(__file__).parent.parent.parent / "services" / "data-source"
os.chdir(str(service_path))
sys.path.insert(0, str(service_path))

# 设置测试环境变量
os.environ["SERVICE_PORT"] = "8004"
os.environ["MONITOR_METRICS_PORT"] = "8005"
os.environ["SERVICE_ENVIRONMENT"] = "testing"
os.environ["DB_MONGODB_URL"] = "mongodb://testuser:testpass123@localhost:27018/historical_text_test"
os.environ["DB_REDIS_URL"] = "redis://localhost:6379/2"

if __name__ == "__main__":
    print("🚀 启动测试服务...")
    print("📍 服务端口: 8004")
    print("📍 指标端口: 8005") 
    print("📍 测试环境配置已加载")
    
    try:
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8004,
            reload=False,
            log_level="info",
            access_log=False
        )
    except KeyboardInterrupt:
        print("👋 测试服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        sys.exit(1)