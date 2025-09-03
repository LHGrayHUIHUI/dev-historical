#!/usr/bin/env python3
"""
快速集成测试 - 验证Epic 1基本功能
"""

import sys
import time
from pathlib import Path

def test_service_imports():
    """测试服务模块导入"""
    print("🔍 测试服务模块导入...")
    
    try:
        # 测试核心基础设施服务导入
        sys.path.append(str(Path.cwd() / "services" / "core"))
        from registry.service_registry import ServiceRegistry
        from config.config_manager import ConfigManager
        from health.health_checker import HealthChecker
        print("  ✅ 核心基础设施服务模块导入成功")
        
        # 测试数据源服务导入
        sys.path.append(str(Path.cwd() / "services" / "data-source" / "src"))
        from crawler.crawler_manager import CrawlerManager
        from models.data_source import DataSource
        print("  ✅ 数据源服务模块导入成功")
        
        # 测试数据采集服务导入
        sys.path.append(str(Path.cwd() / "services" / "data-collection" / "src"))
        from models.dataset import Dataset
        from services.data_collection_service import DataCollectionService
        print("  ✅ 数据采集服务模块导入成功")
        
        # 测试监控服务导入
        sys.path.append(str(Path.cwd() / "services" / "core" / "monitoring"))
        from metrics_middleware import PrometheusMetricsMiddleware
        from monitoring_controller import router
        print("  ✅ 监控服务模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def test_configuration_loading():
    """测试配置加载"""
    print("\n🔍 测试配置系统...")
    
    try:
        # 测试数据源服务配置
        sys.path.append(str(Path.cwd() / "services" / "data-source" / "src"))
        from config.settings import get_settings as get_ds_settings
        ds_settings = get_ds_settings()
        print(f"  ✅ 数据源服务配置加载成功 (环境: {ds_settings.environment})")
        
        # 测试数据采集服务配置  
        sys.path.append(str(Path.cwd() / "services" / "data-collection" / "src"))
        from config.settings import get_settings as get_dc_settings
        dc_settings = get_dc_settings()
        print(f"  ✅ 数据采集服务配置加载成功 (环境: {dc_settings.service_environment})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False

def test_model_functionality():
    """测试数据模型功能"""
    print("\n🔍 测试数据模型...")
    
    try:
        from uuid import uuid4
        
        # 测试数据源模型
        sys.path.append(str(Path.cwd() / "services" / "data-source" / "src"))
        from models.data_source import DataSource as DSDataSource
        
        ds = DSDataSource(
            name="测试数据源",
            type="api",
            description="集成测试用数据源",
            config={"test": True},
            created_by=uuid4(),
            status="active"
        )
        assert ds.is_active == True
        assert ds.get_config_value("test") == True
        print("  ✅ 数据源模型功能正常")
        
        # 测试数据集模型
        sys.path.append(str(Path.cwd() / "services" / "data-collection" / "src"))
        from models.dataset import Dataset
        
        dataset = Dataset(
            name="测试数据集",
            source_id=uuid4(),
            file_path="/test/path",
            file_size=1024,
            file_type="text/plain",
            file_hash="test123",
            created_by=uuid4(),
            processing_status="pending"
        )
        assert dataset.processing_status == "pending"
        assert not dataset.is_processed
        print("  ✅ 数据集模型功能正常")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据模型测试失败: {e}")
        return False

def main():
    """运行快速集成测试"""
    print("🚀 开始运行Epic 1快速集成测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(test_service_imports())
    test_results.append(test_configuration_loading()) 
    test_results.append(test_model_functionality())
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    passed = sum(test_results)
    total = len(test_results)
    failed = total - passed
    
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  📈 成功率: {passed/total*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 Epic 1基本集成功能验证通过！")
        return 0
    else:
        print("\n⚠️  部分集成测试失败，请检查相关模块")
        return 1

if __name__ == "__main__":
    exit(main())