#!/usr/bin/env python3
"""
Epic 3 综合集成测试脚本
测试AI模型服务和内容优化服务的功能
"""

import asyncio
import json
import aiohttp
import time
from datetime import datetime
from typing import Dict, List, Any


class Epic3IntegrationTest:
    """Epic 3集成测试类"""
    
    def __init__(self):
        self.base_urls = {
            'ai_model': 'http://localhost:8008',
            'storage': 'http://localhost:8002',
            'quality_control': 'http://localhost:8010',
            'text_optimization': 'http://localhost:8009',
            'content_merger': 'http://localhost:8011',
            'quality_assessment': 'http://localhost:8012'
        }
        self.test_results = []
        
    async def health_check_all_services(self) -> Dict[str, Any]:
        """检查所有服务的健康状态"""
        print("🔍 检查所有Epic 3服务健康状态...")
        
        async with aiohttp.ClientSession() as session:
            health_results = {}
            
            for service_name, base_url in self.base_urls.items():
                try:
                    async with session.get(f"{base_url}/health", timeout=10) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            health_results[service_name] = {
                                'status': 'healthy',
                                'response': health_data
                            }
                            print(f"  ✅ {service_name}: 健康")
                        else:
                            health_results[service_name] = {
                                'status': 'unhealthy',
                                'http_status': response.status
                            }
                            print(f"  ❌ {service_name}: 不健康 (HTTP {response.status})")
                except Exception as e:
                    health_results[service_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    print(f"  💥 {service_name}: 连接错误 - {e}")
                    
        return health_results
    
    async def test_ai_model_service(self) -> Dict[str, Any]:
        """测试AI模型服务功能"""
        print("\n🤖 测试AI模型服务...")
        
        test_result = {
            'service': 'ai_model_service',
            'tests': [],
            'overall_status': 'pass'
        }
        
        async with aiohttp.ClientSession() as session:
            # 测试1: 聊天完成功能
            try:
                chat_payload = {
                    "messages": [
                        {"role": "user", "content": "请对以下历史文本进行简化：史记者，司马迁所著，纪传体通史也。"}
                    ],
                    "model": "gemini-1.5-flash",
                    "max_tokens": 150,
                    "temperature": 0.7
                }
                
                async with session.post(
                    f"{self.base_urls['ai_model']}/api/v1/chat/completions",
                    json=chat_payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        chat_result = await response.json()
                        test_result['tests'].append({
                            'name': 'chat_completion',
                            'status': 'pass',
                            'response': chat_result,
                            'details': f"成功生成回复，模型: {chat_result.get('model', 'unknown')}"
                        })
                        print("  ✅ 聊天完成功能测试通过")
                    else:
                        test_result['tests'].append({
                            'name': 'chat_completion',
                            'status': 'fail',
                            'http_status': response.status,
                            'details': f"HTTP错误: {response.status}"
                        })
                        test_result['overall_status'] = 'fail'
                        print(f"  ❌ 聊天完成功能测试失败: HTTP {response.status}")
                        
            except Exception as e:
                test_result['tests'].append({
                    'name': 'chat_completion',
                    'status': 'error',
                    'error': str(e)
                })
                test_result['overall_status'] = 'fail'
                print(f"  💥 聊天完成功能测试异常: {e}")
            
            # 测试2: 模型提供商列表
            try:
                async with session.get(
                    f"{self.base_urls['ai_model']}/api/v1/models/providers",
                    timeout=10
                ) as response:
                    if response.status == 200:
                        providers_result = await response.json()
                        test_result['tests'].append({
                            'name': 'providers_list',
                            'status': 'pass',
                            'response': providers_result,
                            'details': f"支持 {len(providers_result.get('providers', {}))} 个提供商"
                        })
                        print("  ✅ 模型提供商列表测试通过")
                    else:
                        test_result['tests'].append({
                            'name': 'providers_list',
                            'status': 'fail',
                            'http_status': response.status
                        })
                        test_result['overall_status'] = 'fail'
                        print(f"  ❌ 模型提供商列表测试失败: HTTP {response.status}")
                        
            except Exception as e:
                test_result['tests'].append({
                    'name': 'providers_list',
                    'status': 'error',
                    'error': str(e)
                })
                test_result['overall_status'] = 'fail'
                print(f"  💥 模型提供商列表测试异常: {e}")
        
        return test_result
    
    async def test_quality_control_service(self) -> Dict[str, Any]:
        """测试内容质量控制服务"""
        print("\n🔍 测试内容质量控制服务...")
        
        test_result = {
            'service': 'quality_control_service',
            'tests': [],
            'overall_status': 'pass'
        }
        
        async with aiohttp.ClientSession() as session:
            # 测试质量检测功能
            try:
                quality_payload = {
                    "content": "史记是中国历史上第一部纪传体通史，作者司马迁。全书共一百三十卷，记述了从传说中的黄帝到汉武帝约三千年的历史。",
                    "content_type": "historical_text",
                    "metadata": {
                        "title": "史记介绍",
                        "author": "测试"
                    }
                }
                
                async with session.post(
                    f"{self.base_urls['quality_control']}/api/v1/quality/check",
                    json=quality_payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        quality_result = await response.json()
                        test_result['tests'].append({
                            'name': 'quality_check',
                            'status': 'pass',
                            'response': quality_result,
                            'details': f"质量分数: {quality_result.get('data', {}).get('overall_score', 'unknown')}"
                        })
                        print("  ✅ 质量检测功能测试通过")
                    else:
                        test_result['tests'].append({
                            'name': 'quality_check',
                            'status': 'fail',
                            'http_status': response.status
                        })
                        test_result['overall_status'] = 'fail'
                        print(f"  ❌ 质量检测功能测试失败: HTTP {response.status}")
                        
            except Exception as e:
                test_result['tests'].append({
                    'name': 'quality_check',
                    'status': 'error',
                    'error': str(e)
                })
                test_result['overall_status'] = 'fail'
                print(f"  💥 质量检测功能测试异常: {e}")
            
            # 测试合规性检查功能
            try:
                compliance_payload = {
                    "content": "这是一个历史文献内容的合规性测试。",
                    "content_type": "historical_text",
                    "check_types": ["sensitive_words", "content_security"]
                }
                
                async with session.post(
                    f"{self.base_urls['quality_control']}/api/v1/compliance/check",
                    json=compliance_payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        compliance_result = await response.json()
                        test_result['tests'].append({
                            'name': 'compliance_check',
                            'status': 'pass',
                            'response': compliance_result,
                            'details': f"合规状态: {compliance_result.get('data', {}).get('compliance_status', 'unknown')}"
                        })
                        print("  ✅ 合规性检查功能测试通过")
                    else:
                        test_result['tests'].append({
                            'name': 'compliance_check',
                            'status': 'fail',
                            'http_status': response.status
                        })
                        test_result['overall_status'] = 'fail'
                        print(f"  ❌ 合规性检查功能测试失败: HTTP {response.status}")
                        
            except Exception as e:
                test_result['tests'].append({
                    'name': 'compliance_check',
                    'status': 'error',
                    'error': str(e)
                })
                test_result['overall_status'] = 'fail'
                print(f"  💥 合规性检查功能测试异常: {e}")
        
        return test_result
    
    async def test_storage_service_ai_models(self) -> Dict[str, Any]:
        """测试存储服务的AI模型管理功能"""
        print("\n💾 测试存储服务AI模型管理...")
        
        test_result = {
            'service': 'storage_service_ai_models',
            'tests': [],
            'overall_status': 'pass'
        }
        
        async with aiohttp.ClientSession() as session:
            # 测试获取AI模型配置列表
            try:
                async with session.get(
                    f"{self.base_urls['storage']}/api/v1/ai-models/configs",
                    timeout=10
                ) as response:
                    if response.status == 200:
                        configs_result = await response.json()
                        test_result['tests'].append({
                            'name': 'get_ai_model_configs',
                            'status': 'pass',
                            'response': configs_result,
                            'details': f"找到 {configs_result.get('total', 0)} 个AI模型配置"
                        })
                        print("  ✅ AI模型配置列表获取测试通过")
                    else:
                        test_result['tests'].append({
                            'name': 'get_ai_model_configs',
                            'status': 'fail',
                            'http_status': response.status
                        })
                        test_result['overall_status'] = 'fail'
                        print(f"  ❌ AI模型配置列表获取测试失败: HTTP {response.status}")
                        
            except Exception as e:
                test_result['tests'].append({
                    'name': 'get_ai_model_configs',
                    'status': 'error',
                    'error': str(e)
                })
                test_result['overall_status'] = 'fail'
                print(f"  💥 AI模型配置列表获取测试异常: {e}")
        
        return test_result
    
    async def check_service_availability(self, service_name: str, url: str) -> bool:
        """检查单个服务是否可用"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合测试"""
        print("🚀 开始Epic 3综合集成测试")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 健康检查
        health_results = await self.health_check_all_services()
        
        # 2. 核心服务测试
        core_tests = []
        
        # AI模型服务测试
        if health_results.get('ai_model', {}).get('status') == 'healthy':
            ai_test_result = await self.test_ai_model_service()
            core_tests.append(ai_test_result)
        else:
            print("⚠️ AI模型服务不健康，跳过相关测试")
        
        # 质量控制服务测试
        if health_results.get('quality_control', {}).get('status') == 'healthy':
            quality_test_result = await self.test_quality_control_service()
            core_tests.append(quality_test_result)
        else:
            print("⚠️ 质量控制服务不健康，跳过相关测试")
        
        # 存储服务AI模型功能测试
        if health_results.get('storage', {}).get('status') == 'healthy':
            storage_test_result = await self.test_storage_service_ai_models()
            core_tests.append(storage_test_result)
        else:
            print("⚠️ 存储服务不健康，跳过相关测试")
        
        # 计算测试结果
        total_time = time.time() - start_time
        
        # 统计测试结果
        total_tests = sum(len(test['tests']) for test in core_tests)
        passed_tests = sum(
            len([t for t in test['tests'] if t['status'] == 'pass']) 
            for test in core_tests
        )
        failed_tests = total_tests - passed_tests
        
        # 确定整体状态
        overall_status = 'pass' if failed_tests == 0 else 'fail'
        
        # 生成综合报告
        comprehensive_result = {
            'test_suite': 'Epic 3 Comprehensive Integration Test',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(total_time, 2),
            'overall_status': overall_status,
            'summary': {
                'total_services_tested': len([t for t in core_tests]),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': round((passed_tests / total_tests * 100) if total_tests > 0 else 0, 2)
            },
            'health_check_results': health_results,
            'detailed_test_results': core_tests
        }
        
        # 打印总结
        print("\n" + "=" * 60)
        print("📊 Epic 3测试总结:")
        print(f"  总测试数量: {total_tests}")
        print(f"  通过测试: {passed_tests}")
        print(f"  失败测试: {failed_tests}")
        print(f"  成功率: {comprehensive_result['summary']['success_rate']:.1f}%")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  整体状态: {'✅ 通过' if overall_status == 'pass' else '❌ 失败'}")
        print("=" * 60)
        
        return comprehensive_result
    
    async def save_test_report(self, results: Dict[str, Any]):
        """保存测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic3_integration_test_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📝 测试报告已保存: {filename}")


async def main():
    """主函数"""
    tester = Epic3IntegrationTest()
    
    try:
        # 运行综合测试
        results = await tester.run_comprehensive_test()
        
        # 保存测试报告
        await tester.save_test_report(results)
        
        # 根据测试结果设置退出代码
        if results['overall_status'] == 'pass':
            print("\n🎉 所有测试通过！")
            exit(0)
        else:
            print("\n⚠️ 部分测试失败，请检查详细报告。")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        exit(130)
    except Exception as e:
        print(f"\n💥 测试过程中发生异常: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())