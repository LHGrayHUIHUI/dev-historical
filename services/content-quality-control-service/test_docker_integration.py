#!/usr/bin/env python3
"""
内容质量控制服务 Docker 集成测试

这个脚本用于验证服务在 Docker 环境中是否能正常启动和运行。
"""

import time
import requests
import sys

def wait_for_service(url, max_attempts=30, delay=2):
    """等待服务启动"""
    print(f"等待服务启动: {url}")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ 服务已启动 (尝试 {attempt + 1}/{max_attempts})")
                return True
        except requests.exceptions.RequestException as e:
            print(f"⏳ 尝试 {attempt + 1}/{max_attempts} - 服务未就绪: {e}")
        
        time.sleep(delay)
    
    print(f"❌ 服务在 {max_attempts * delay} 秒后仍未就绪")
    return False

def test_basic_endpoints():
    """测试基础端点"""
    base_url = "http://localhost:8010"
    
    endpoints = [
        ("/health", "健康检查"),
        ("/info", "服务信息"),
        ("/docs", "API文档")
    ]
    
    print("\n🔍 测试基础端点...")
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {description} ({endpoint}): {response.status_code}")
            else:
                print(f"⚠️ {description} ({endpoint}): {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {description} ({endpoint}): 连接失败 - {e}")

def test_quality_check():
    """测试质量检测端点"""
    base_url = "http://localhost:8010"
    
    print("\n🧪 测试质量检测端点...")
    
    test_data = {
        "content": "朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州。",
        "content_type": "historical_text",
        "check_options": {
            "grammar_check": True,
            "logic_check": True,
            "format_check": True,
            "factual_check": True,
            "academic_check": True
        },
        "auto_fix": False
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/quality/check",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 质量检测成功")
            print(f"   总分: {result.get('data', {}).get('overall_score', 'N/A')}")
            print(f"   处理时间: {result.get('data', {}).get('processing_time_ms', 'N/A')}ms")
        else:
            print(f"⚠️ 质量检测响应状态码: {response.status_code}")
            print(f"   响应内容: {response.text[:200]}...")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 质量检测请求失败: {e}")

def main():
    """主测试函数"""
    print("🚀 内容质量控制服务 Docker 集成测试")
    print("=" * 50)
    
    # 等待服务启动
    if not wait_for_service("http://localhost:8010/health"):
        print("❌ 服务启动失败，退出测试")
        sys.exit(1)
    
    # 测试基础端点
    test_basic_endpoints()
    
    # 测试功能端点
    test_quality_check()
    
    print("\n✨ 集成测试完成")

if __name__ == "__main__":
    main()