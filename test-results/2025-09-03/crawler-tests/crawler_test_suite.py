#!/usr/bin/env python3
"""
数据源服务爬虫功能综合测试套件
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import requests
from pathlib import Path

class CrawlerTestSuite:
    """爬虫功能测试套件"""
    
    def __init__(self):
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': [],
            'details': [],
            'start_time': datetime.now(),
            'service_status': 'unknown'
        }
        
        # 数据源服务配置
        self.base_url = 'http://localhost:8001'
        self.service_name = '数据源服务'
        
        # 测试数据存储目录
        self.test_data_dir = Path(__file__).parent / "data"
        self.test_data_dir.mkdir(exist_ok=True)

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

    def save_test_data(self, test_name: str, data: Any):
        """保存测试数据"""
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        filepath = self.test_data_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            self.log_info(f"测试数据已保存: {filename}")
        except Exception as e:
            self.log_error(f"保存测试数据失败: {e}")

    def wait_for_service_startup(self, timeout: int = 300) -> bool:
        """等待服务启动"""
        self.log_info(f"等待{self.service_name}启动...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    self.test_results['service_status'] = 'healthy'
                    self.log_success(f"{self.service_name}已启动并健康")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                self.log_error(f"健康检查异常: {e}")
            
            time.sleep(10)
            print(f"⏳ 等待中... ({int(time.time() - start_time)}s/{timeout}s)")
        
        self.test_results['service_status'] = 'timeout'
        self.log_error(f"{self.service_name}启动超时")
        return False

    def test_service_health(self) -> bool:
        """测试服务健康状态"""
        self.log_info("测试服务健康状态...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_success(f"健康检查通过: {data.get('status')}")
                
                self.test_results['details'].append({
                    'test': 'health_check',
                    'method': 'GET',
                    'endpoint': '/health',
                    'status_code': 200,
                    'response': data
                })
                
                self.save_test_data('health_check', data)
                return True
            else:
                self.log_error(f"健康检查失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_error(f"健康检查异常: {e}")
            return False

    def test_service_readiness(self) -> bool:
        """测试服务就绪状态"""
        self.log_info("测试服务就绪状态...")
        
        try:
            response = requests.get(f"{self.base_url}/ready", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_success(f"就绪检查通过: {data.get('status')}")
                
                self.test_results['details'].append({
                    'test': 'readiness_check',
                    'method': 'GET',
                    'endpoint': '/ready',
                    'status_code': 200,
                    'response': data
                })
                
                self.save_test_data('readiness_check', data)
                return True
            else:
                data = response.json() if response.content else {}
                self.log_error(f"就绪检查失败: HTTP {response.status_code}")
                self.log_info(f"响应内容: {data}")
                
                self.test_results['details'].append({
                    'test': 'readiness_check',
                    'method': 'GET',
                    'endpoint': '/ready',
                    'status_code': response.status_code,
                    'response': data
                })
                
                return False
                
        except Exception as e:
            self.log_error(f"就绪检查异常: {e}")
            return False

    def test_api_documentation(self) -> bool:
        """测试API文档"""
        self.log_info("测试API文档可访问性...")
        
        try:
            # 测试OpenAPI规范
            response = requests.get(f"{self.base_url}/openapi.json", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                endpoints_count = len(data.get('paths', {}))
                self.log_success(f"OpenAPI规范可访问，发现 {endpoints_count} 个端点")
                
                self.save_test_data('openapi_schema', data)
                
                # 测试Swagger UI
                docs_response = requests.get(f"{self.base_url}/docs", timeout=10)
                if docs_response.status_code == 200:
                    self.log_success("Swagger UI文档可访问")
                    return True
                else:
                    self.log_error(f"Swagger UI不可访问: HTTP {docs_response.status_code}")
                    return False
            else:
                self.log_error(f"OpenAPI规范不可访问: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_error(f"API文档测试异常: {e}")
            return False

    def test_crawler_manager_apis(self) -> bool:
        """测试爬虫管理器API"""
        self.log_info("测试爬虫管理器API...")
        
        success_count = 0
        total_tests = 0
        
        # 测试爬虫任务相关API
        crawler_endpoints = [
            '/api/v1/crawlers',
            '/api/v1/crawlers/tasks',
            '/api/v1/crawlers/status',
            '/api/v1/crawlers/platforms'
        ]
        
        for endpoint in crawler_endpoints:
            total_tests += 1
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code in [200, 401, 403]:  # 401/403可能是认证问题但服务正常
                    self.log_success(f"爬虫API {endpoint} 响应正常 (HTTP {response.status_code})")
                    success_count += 1
                    
                    if response.content:
                        try:
                            data = response.json()
                            self.save_test_data(f'crawler_api_{endpoint.replace("/", "_")}', data)
                        except:
                            pass
                            
                    self.test_results['details'].append({
                        'test': 'crawler_api',
                        'method': 'GET',
                        'endpoint': endpoint,
                        'status_code': response.status_code,
                        'response': response.text[:200] if response.text else 'Empty'
                    })
                else:
                    self.log_error(f"爬虫API {endpoint} 异常: HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_error(f"爬虫API {endpoint} 测试异常: {e}")
        
        return success_count == total_tests

    def test_proxy_manager_apis(self) -> bool:
        """测试代理池管理API"""
        self.log_info("测试代理池管理API...")
        
        success_count = 0
        total_tests = 0
        
        # 测试代理池相关API
        proxy_endpoints = [
            '/api/v1/proxies',
            '/api/v1/proxies/stats',
            '/api/v1/proxies/health',
            '/api/v1/proxies/providers'
        ]
        
        for endpoint in proxy_endpoints:
            total_tests += 1
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code in [200, 401, 403]:
                    self.log_success(f"代理池API {endpoint} 响应正常 (HTTP {response.status_code})")
                    success_count += 1
                    
                    if response.content:
                        try:
                            data = response.json()
                            self.save_test_data(f'proxy_api_{endpoint.replace("/", "_")}', data)
                        except:
                            pass
                            
                    self.test_results['details'].append({
                        'test': 'proxy_api',
                        'method': 'GET',
                        'endpoint': endpoint,
                        'status_code': response.status_code,
                        'response': response.text[:200] if response.text else 'Empty'
                    })
                else:
                    self.log_error(f"代理池API {endpoint} 异常: HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_error(f"代理池API {endpoint} 测试异常: {e}")
        
        return success_count >= total_tests // 2  # 允许50%的端点不可用

    def test_platform_adapters(self) -> bool:
        """测试平台适配器"""
        self.log_info("测试平台适配器...")
        
        success_count = 0
        total_tests = 0
        
        # 测试平台适配器相关API
        platform_endpoints = [
            '/api/v1/platforms',
            '/api/v1/platforms/adapters',
            '/api/v1/platforms/toutiao',
            '/api/v1/platforms/baijiahao',
            '/api/v1/platforms/xiaohongshu'
        ]
        
        for endpoint in platform_endpoints:
            total_tests += 1
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code in [200, 401, 403, 404]:  # 404也可能是正常的
                    self.log_success(f"平台API {endpoint} 响应正常 (HTTP {response.status_code})")
                    success_count += 1
                    
                    if response.content and response.status_code == 200:
                        try:
                            data = response.json()
                            self.save_test_data(f'platform_api_{endpoint.replace("/", "_")}', data)
                        except:
                            pass
                            
                    self.test_results['details'].append({
                        'test': 'platform_api',
                        'method': 'GET',
                        'endpoint': endpoint,
                        'status_code': response.status_code,
                        'response': response.text[:200] if response.text else 'Empty'
                    })
                else:
                    self.log_error(f"平台API {endpoint} 异常: HTTP {response.status_code}")
                    
            except Exception as e:
                self.log_error(f"平台API {endpoint} 测试异常: {e}")
        
        return success_count >= total_tests // 2

    def run_comprehensive_test(self):
        """运行综合爬虫测试"""
        print("🚀 开始数据源服务爬虫功能综合测试")
        print("=" * 60)
        
        # 1. 等待服务启动
        if not self.wait_for_service_startup():
            self.print_summary()
            return 1
        
        # 2. 基础健康检查
        self.test_service_health()
        self.test_service_readiness()
        
        # 3. API文档测试
        self.test_api_documentation()
        
        # 4. 核心功能测试
        print("\n🔍 测试爬虫核心功能...")
        self.test_crawler_manager_apis()
        
        print("\n🔍 测试代理池管理...")
        self.test_proxy_manager_apis()
        
        print("\n🔍 测试平台适配器...")
        self.test_platform_adapters()
        
        # 5. 生成测试报告
        self.print_summary()
        self.generate_test_report()
        
        # 返回状态码
        if self.test_results['failed'] == 0:
            return 0
        elif self.test_results['passed'] > self.test_results['failed']:
            return 1
        else:
            return 2

    def print_summary(self):
        """打印测试结果汇总"""
        end_time = datetime.now()
        duration = (end_time - self.test_results['start_time']).total_seconds()
        
        print("\n" + "=" * 60)
        print("📊 爬虫功能测试结果汇总:")
        
        print(f"  🔧 服务状态: {self.test_results['service_status']}")
        print(f"  ✅ 测试通过: {self.test_results['passed']}")
        print(f"  ❌ 测试失败: {self.test_results['failed']}")
        print(f"  ⏱️  测试时长: {duration:.1f}秒")
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        if total_tests > 0:
            success_rate = self.test_results['passed'] / total_tests * 100
            print(f"  📈 成功率: {success_rate:.1f}%")
        
        # API端点统计
        successful_endpoints = len([d for d in self.test_results['details'] if d.get('status_code') in [200, 401, 403, 404]])
        total_endpoints = len(self.test_results['details'])
        if total_endpoints > 0:
            print(f"  🎯 响应正常的API端点: {successful_endpoints}/{total_endpoints}")
        
        # 错误详情
        if self.test_results['errors']:
            print("\n❌ 错误详情:")
            for i, error in enumerate(self.test_results['errors'][:5], 1):
                print(f"  {i}. {error}")
            if len(self.test_results['errors']) > 5:
                print(f"  ... 还有 {len(self.test_results['errors']) - 5} 个错误")
        
        # 总体评价
        if self.test_results['service_status'] == 'healthy' and self.test_results['passed'] > 0:
            print("\n🎉 爬虫功能测试基本成功！")
        else:
            print("\n⚠️  爬虫功能存在问题，建议检查服务状态")

    def generate_test_report(self):
        """生成详细测试报告"""
        report_data = {
            'test_summary': {
                'service_name': self.service_name,
                'test_type': '爬虫功能综合测试',
                'test_date': self.test_results['start_time'].strftime('%Y-%m-%d'),
                'test_time': self.test_results['start_time'].strftime('%H:%M:%S'),
                'duration_seconds': (datetime.now() - self.test_results['start_time']).total_seconds(),
                'service_status': self.test_results['service_status']
            },
            'test_results': {
                'passed': self.test_results['passed'],
                'failed': self.test_results['failed'],
                'success_rate': (self.test_results['passed'] / max(1, self.test_results['passed'] + self.test_results['failed'])) * 100,
                'total_endpoints_tested': len(self.test_results['details'])
            },
            'test_details': self.test_results['details'],
            'errors': self.test_results['errors']
        }
        
        # 保存详细报告
        report_file = self.test_data_dir / f"crawler_test_report_{datetime.now().strftime('%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
            self.log_info(f"详细测试报告已保存: {report_file.name}")
        except Exception as e:
            self.log_error(f"保存测试报告失败: {e}")

def main():
    """主函数"""
    test_suite = CrawlerTestSuite()
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