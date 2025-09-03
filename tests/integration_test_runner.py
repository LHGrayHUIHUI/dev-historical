#!/usr/bin/env python3
"""
历史文本项目 - 集成测试运行器

测试微服务间的集成和通信
"""

import asyncio
import json
import sys
import time
from typing import Dict, List

import requests


class IntegrationTestRunner:
    """集成测试运行器"""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # 服务端点配置
        self.services = {
            'data-source': 'http://data-source-service:8000',
            'data-collection': 'http://data-collection-service:8002'
        }
    
    def log_info(self, message: str):
        """输出信息日志"""
        print(f"[INFO] {message}")
    
    def log_success(self, message: str):
        """输出成功日志"""
        print(f"[SUCCESS] ✅ {message}")
        self.test_results['passed'] += 1
    
    def log_error(self, message: str):
        """输出错误日志"""
        print(f"[ERROR] ❌ {message}")
        self.test_results['failed'] += 1
        self.test_results['errors'].append(message)
    
    def wait_for_service(self, service_name: str, url: str, timeout: int = 120) -> bool:
        """等待服务启动"""
        self.log_info(f"等待 {service_name} 服务启动...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    self.log_success(f"{service_name} 服务已启动")
                    return True
            except Exception as e:
                time.sleep(5)
        
        self.log_error(f"{service_name} 服务启动超时")
        return False
    
    def test_health_endpoints(self):
        """测试健康检查端点"""
        self.log_info("开始测试健康检查端点...")
        
        for service_name, base_url in self.services.items():
            try:
                # 测试健康检查
                response = requests.get(f"{base_url}/health", timeout=10)
                
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        self.log_success(f"{service_name} 健康检查通过")
                    else:
                        self.log_error(f"{service_name} 健康检查失败: {health_data}")
                else:
                    self.log_error(f"{service_name} 健康检查返回状态码: {response.status_code}")
                    
            except Exception as e:
                self.log_error(f"{service_name} 健康检查异常: {str(e)}")
    
    def test_service_info(self):
        """测试服务信息端点"""
        self.log_info("开始测试服务信息端点...")
        
        endpoints = {
            'data-source': '/api/v1/crawler/info',
            'data-collection': '/api/v1/data/info'
        }
        
        for service_name, endpoint in endpoints.items():
            try:
                base_url = self.services[service_name]
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    info_data = response.json()
                    if 'service_name' in info_data:
                        self.log_success(f"{service_name} 服务信息获取成功")
                    else:
                        self.log_error(f"{service_name} 服务信息格式错误")
                else:
                    self.log_error(f"{service_name} 服务信息获取失败: {response.status_code}")
                    
            except Exception as e:
                self.log_error(f"{service_name} 服务信息获取异常: {str(e)}")
    
    def test_basic_api_endpoints(self):
        """测试基础API端点"""
        self.log_info("开始测试基础API端点...")
        
        # 测试数据源服务API
        try:
            # 获取代理列表
            response = requests.get(
                f"{self.services['data-source']}/api/v1/crawler/proxies",
                timeout=15
            )
            
            if response.status_code == 200:
                self.log_success("数据源服务代理API响应正常")
            else:
                self.log_error(f"数据源服务代理API失败: {response.status_code}")
                
        except Exception as e:
            self.log_error(f"数据源服务API测试异常: {str(e)}")
    
    def test_cross_service_communication(self):
        """测试跨服务通信"""
        self.log_info("开始测试跨服务通信...")
        
        # 这里可以添加更复杂的集成测试
        # 例如：数据源服务提供数据，数据采集服务处理数据
        
        self.log_info("跨服务通信测试需要进一步实现")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 历史文本项目 - 微服务集成测试")
        print("=" * 50)
        
        # 等待所有服务启动
        all_services_ready = True
        for service_name, base_url in self.services.items():
            if not self.wait_for_service(service_name, base_url):
                all_services_ready = False
        
        if not all_services_ready:
            self.log_error("部分服务未能启动，跳过集成测试")
            return False
        
        # 运行测试套件
        self.test_health_endpoints()
        self.test_service_info()
        self.test_basic_api_endpoints()
        self.test_cross_service_communication()
        
        # 输出测试结果
        print("\n" + "=" * 50)
        print("📊 集成测试结果汇总:")
        print(f"   通过: {self.test_results['passed']}")
        print(f"   失败: {self.test_results['failed']}")
        
        if self.test_results['errors']:
            print("\n❌ 失败的测试:")
            for error in self.test_results['errors']:
                print(f"   • {error}")
        
        success_rate = (
            self.test_results['passed'] / 
            (self.test_results['passed'] + self.test_results['failed'])
            if (self.test_results['passed'] + self.test_results['failed']) > 0 
            else 0
        ) * 100
        
        print(f"\n成功率: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 集成测试基本通过！")
            return True
        else:
            print("😞 集成测试需要改进")
            return False


if __name__ == "__main__":
    runner = IntegrationTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)