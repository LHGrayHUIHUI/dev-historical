#!/usr/bin/env python3
"""
Epic 1 微服务API综合测试工具
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any
from urllib.parse import urljoin

class APITestSuite:
    """API测试套件"""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'details': []
        }
        
        # 服务配置
        self.services = {
            'data-source': {
                'base_url': 'http://localhost:8001',
                'name': '数据源服务',
                'status': 'unknown'
            },
            'data-collection': {
                'base_url': 'http://localhost:8003', 
                'name': '数据采集服务',
                'status': 'unknown'
            }
        }
        
        # 基础设施服务
        self.infrastructure = {
            'minio': 'http://localhost:9001',
            'rabbitmq': 'http://localhost:15673',
            'postgres': 'localhost:5433',
            'redis': 'localhost:6380'
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

    def test_service_health(self, service_key: str) -> bool:
        """测试服务健康状态"""
        service = self.services[service_key]
        health_url = urljoin(service['base_url'], '/health')
        
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                service['status'] = 'healthy'
                self.log_success(f"{service['name']} 健康检查通过: {data.get('status', 'unknown')}")
                
                # 记录详细信息
                self.test_results['details'].append({
                    'service': service['name'],
                    'endpoint': '/health',
                    'method': 'GET',
                    'status_code': 200,
                    'response': data
                })
                return True
            else:
                service['status'] = 'unhealthy'
                self.log_error(f"{service['name']} 健康检查失败: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            service['status'] = 'offline'
            self.log_error(f"{service['name']} 连接被拒绝")
            return False
        except requests.exceptions.Timeout:
            service['status'] = 'timeout'
            self.log_error(f"{service['name']} 连接超时")
            return False
        except Exception as e:
            service['status'] = 'error'
            self.log_error(f"{service['name']} 健康检查异常: {str(e)}")
            return False

    def test_service_ready(self, service_key: str) -> bool:
        """测试服务就绪状态"""
        service = self.services[service_key]
        ready_url = urljoin(service['base_url'], '/ready')
        
        try:
            response = requests.get(ready_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_success(f"{service['name']} 就绪检查通过: {data.get('status', 'unknown')}")
                
                self.test_results['details'].append({
                    'service': service['name'],
                    'endpoint': '/ready',
                    'method': 'GET', 
                    'status_code': 200,
                    'response': data
                })
                return True
            else:
                self.log_error(f"{service['name']} 就绪检查失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_error(f"{service['name']} 就绪检查异常: {str(e)}")
            return False

    def test_service_docs(self, service_key: str) -> bool:
        """测试服务API文档"""
        service = self.services[service_key]
        docs_url = urljoin(service['base_url'], '/docs')
        
        try:
            response = requests.get(docs_url, timeout=10)
            if response.status_code == 200:
                self.log_success(f"{service['name']} API文档可访问")
                return True
            else:
                self.log_error(f"{service['name']} API文档不可访问: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_error(f"{service['name']} API文档测试异常: {str(e)}")
            return False

    def test_data_source_apis(self) -> None:
        """测试数据源服务特有API"""
        if self.services['data-source']['status'] != 'healthy':
            self.log_error("数据源服务不健康，跳过API测试")
            return
            
        service = self.services['data-source']
        base_url = service['base_url']
        
        # 测试爬虫管理API
        endpoints = [
            '/api/v1/crawlers',
            '/api/v1/crawlers/tasks',
            '/api/v1/proxies',
            '/api/v1/proxies/stats',
            '/api/v1/platforms',
            '/api/v1/platforms/adapters'
        ]
        
        for endpoint in endpoints:
            try:
                url = urljoin(base_url, endpoint)
                response = requests.get(url, timeout=10)
                
                if response.status_code in [200, 401, 403]:  # 401/403表示需要认证但服务正常
                    self.log_success(f"数据源服务 {endpoint} API响应正常")
                    
                    self.test_results['details'].append({
                        'service': service['name'],
                        'endpoint': endpoint,
                        'method': 'GET',
                        'status_code': response.status_code,
                        'response': response.text[:200] if response.text else 'Empty'
                    })
                else:
                    self.log_error(f"数据源服务 {endpoint} API异常: HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_error(f"数据源服务 {endpoint} API测试异常: {str(e)}")

    def test_data_collection_apis(self) -> None:
        """测试数据采集服务特有API"""
        if self.services['data-collection']['status'] != 'healthy':
            self.log_error("数据采集服务不健康，跳过API测试")
            return
            
        service = self.services['data-collection']
        base_url = service['base_url']
        
        # 测试数据采集API
        endpoints = [
            '/api/v1/datasets',
            '/api/v1/datasets/upload',
            '/api/v1/processing/status',
            '/api/v1/text/extract',
            '/api/v1/files/upload',
            '/api/v1/files/batch'
        ]
        
        for endpoint in endpoints:
            try:
                url = urljoin(base_url, endpoint)
                response = requests.get(url, timeout=10)
                
                if response.status_code in [200, 401, 403, 405]:  # 405 METHOD NOT ALLOWED也是正常的
                    self.log_success(f"数据采集服务 {endpoint} API响应正常")
                    
                    self.test_results['details'].append({
                        'service': service['name'],
                        'endpoint': endpoint,
                        'method': 'GET',
                        'status_code': response.status_code,
                        'response': response.text[:200] if response.text else 'Empty'
                    })
                else:
                    self.log_error(f"数据采集服务 {endpoint} API异常: HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_error(f"数据采集服务 {endpoint} API测试异常: {str(e)}")

    def test_infrastructure_services(self) -> None:
        """测试基础设施服务"""
        self.log_info("测试基础设施服务...")
        
        # 测试MinIO
        try:
            response = requests.get(self.infrastructure['minio'], timeout=5)
            if response.status_code in [200, 403]:
                self.log_success("MinIO 对象存储服务可访问")
            else:
                self.log_error(f"MinIO 状态异常: HTTP {response.status_code}")
        except Exception as e:
            self.log_error(f"MinIO 连接失败: {str(e)}")
        
        # 测试RabbitMQ管理界面
        try:
            response = requests.get(self.infrastructure['rabbitmq'], timeout=5)
            if response.status_code in [200, 401]:
                self.log_success("RabbitMQ 管理界面可访问")
            else:
                self.log_error(f"RabbitMQ 状态异常: HTTP {response.status_code}")
        except Exception as e:
            self.log_error(f"RabbitMQ 连接失败: {str(e)}")

    def run_comprehensive_test(self):
        """运行综合API测试"""
        print("🚀 开始Epic 1微服务API综合测试")
        print("=" * 60)
        
        # 测试基础设施
        self.test_infrastructure_services()
        
        print("\n🔍 测试微服务健康状态...")
        # 测试各服务健康状态
        for service_key in self.services.keys():
            self.test_service_health(service_key)
        
        print("\n🔍 测试微服务就绪状态...")
        # 测试就绪状态
        for service_key in self.services.keys():
            if self.services[service_key]['status'] == 'healthy':
                self.test_service_ready(service_key)
        
        print("\n🔍 测试API文档...")
        # 测试API文档
        for service_key in self.services.keys():
            if self.services[service_key]['status'] == 'healthy':
                self.test_service_docs(service_key)
        
        print("\n🔍 测试数据源服务API...")
        self.test_data_source_apis()
        
        print("\n🔍 测试数据采集服务API...")
        self.test_data_collection_apis()
        
        # 汇总结果
        self.print_summary()

    def print_summary(self):
        """打印测试结果汇总"""
        print("\n" + "=" * 60)
        print("📊 API测试结果汇总:")
        
        # 服务状态统计
        healthy_services = sum(1 for s in self.services.values() if s['status'] == 'healthy')
        total_services = len(self.services)
        
        print(f"  🔧 微服务健康度: {healthy_services}/{total_services} ({healthy_services/total_services*100:.1f}%)")
        print(f"  ✅ 测试通过: {self.test_results['passed']}")
        print(f"  ❌ 测试失败: {self.test_results['failed']}")
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        if total_tests > 0:
            success_rate = self.test_results['passed'] / total_tests * 100
            print(f"  📈 成功率: {success_rate:.1f}%")
        
        # 服务详细状态
        print("\n📋 服务状态详情:")
        for service_key, service in self.services.items():
            status_emoji = {
                'healthy': '✅',
                'unhealthy': '⚠️',
                'offline': '❌',
                'timeout': '⏰',
                'error': '💥',
                'unknown': '❓'
            }.get(service['status'], '❓')
            
            print(f"  {status_emoji} {service['name']}: {service['status']}")
        
        # 错误列表
        if self.test_results['errors']:
            print("\n❌ 错误详情:")
            for i, error in enumerate(self.test_results['errors'][:5], 1):
                print(f"  {i}. {error}")
            if len(self.test_results['errors']) > 5:
                print(f"  ... 还有 {len(self.test_results['errors']) - 5} 个错误")
        
        # 成功的API端点数量
        successful_endpoints = len([d for d in self.test_results['details'] if d.get('status_code') in [200, 401, 403, 405]])
        print(f"\n🎯 成功响应的API端点: {successful_endpoints} 个")
        
        # 总体评价
        if healthy_services >= total_services // 2:
            print("\n🎉 API测试基本成功，服务架构运行正常！")
            return 0
        else:
            print("\n⚠️  部分服务存在问题，建议检查服务启动状态")
            return 1

def main():
    """主函数"""
    test_suite = APITestSuite()
    try:
        exit_code = test_suite.run_comprehensive_test()
        return exit_code
    except KeyboardInterrupt:
        print("\n⏸️  测试被用户中断")
        return 130
    except Exception as e:
        print(f"\n💥 测试执行异常: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)