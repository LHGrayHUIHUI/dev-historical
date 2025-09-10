"""
SS-INT-003: Redis缓存策略集成测试
优先级: P0 - 性能优化系统
"""

import asyncio
import json
import aiohttp
import redis
from datetime import datetime
from typing import Dict, List, Any


class RedisCacheIntegrationTester:
    """Redis缓存集成测试器"""
    
    def __init__(self):
        self.base_url = "http://localhost:8002"
        self.redis_client = None
        self.test_results = []
        
    async def log_test(self, name: str, status: str, details: Dict = None, error: str = None):
        """记录测试结果"""
        result = {
            "test_name": name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "error": error
        }
        self.test_results.append(result)
        
        print(f"{'✅' if status == 'PASSED' else '❌'} {name}: {status}")
        if error:
            print(f"   错误: {error}")
        if details and status == "PASSED":
            print(f"   详情: {details}")
    
    async def test_redis_connection(self):
        """测试Redis连接
        
        测试场景: SS-INT-003-001
        验证点: Redis服务连接和基本操作
        """
        try:
            # 连接到Redis (Docker环境)
            self.redis_client = redis.Redis(host='localhost', port=6380, db=0, decode_responses=True)
            
            # 测试ping
            ping_result = self.redis_client.ping()
            
            # 测试基本操作
            test_key = "integration_test_key"
            test_value = "integration_test_value"
            
            # SET操作
            set_result = self.redis_client.set(test_key, test_value)
            
            # GET操作
            get_result = self.redis_client.get(test_key)
            
            # 验证结果
            if ping_result and set_result and get_result == test_value:
                details = {
                    "ping_success": True,
                    "set_operation": "success",
                    "get_operation": "success",
                    "data_consistency": get_result == test_value
                }
                
                # 清理测试键
                self.redis_client.delete(test_key)
                
                await self.log_test("Redis连接测试", "PASSED", details)
                return True
            else:
                await self.log_test("Redis连接测试", "FAILED", 
                                  error=f"基本操作失败: ping={ping_result}, set={set_result}, get={get_result}")
                return False
                
        except Exception as e:
            await self.log_test("Redis连接测试", "FAILED", error=str(e))
            return False
    
    async def test_cache_performance(self):
        """测试缓存性能
        
        测试场景: SS-INT-003-002
        验证点: Redis读写性能和响应时间
        """
        try:
            if not self.redis_client:
                await self.log_test("缓存性能测试", "FAILED", error="Redis连接未建立")
                return False
            
            # 性能测试参数
            test_operations = 100
            large_data = "x" * 1000  # 1KB数据
            
            # 写入性能测试
            write_start = asyncio.get_event_loop().time()
            for i in range(test_operations):
                key = f"perf_test_{i}"
                value = f"{large_data}_{i}"
                self.redis_client.set(key, value)
            write_end = asyncio.get_event_loop().time()
            
            write_time = write_end - write_start
            
            # 读取性能测试
            read_start = asyncio.get_event_loop().time()
            for i in range(test_operations):
                key = f"perf_test_{i}"
                self.redis_client.get(key)
            read_end = asyncio.get_event_loop().time()
            
            read_time = read_end - read_start
            
            # 清理测试数据
            keys_to_delete = [f"perf_test_{i}" for i in range(test_operations)]
            self.redis_client.delete(*keys_to_delete)
            
            details = {
                "test_operations": test_operations,
                "data_size_per_operation": "1KB",
                "write_time_total": round(write_time, 3),
                "write_time_avg": round(write_time / test_operations * 1000, 3),  # ms per operation
                "read_time_total": round(read_time, 3),
                "read_time_avg": round(read_time / test_operations * 1000, 3),  # ms per operation
                "performance_acceptable": write_time < 5 and read_time < 5
            }
            
            if write_time < 5 and read_time < 5:  # 5秒内完成100次操作
                await self.log_test("缓存性能测试", "PASSED", details)
                return True
            else:
                await self.log_test("缓存性能测试", "FAILED", 
                                  details, 
                                  error=f"性能不达标: 写入{write_time:.2f}s, 读取{read_time:.2f}s")
                return False
                
        except Exception as e:
            await self.log_test("缓存性能测试", "FAILED", error=str(e))
            return False
    
    async def test_cache_expiration(self):
        """测试缓存过期机制
        
        测试场景: SS-INT-003-003
        验证点: TTL设置和过期处理
        """
        try:
            if not self.redis_client:
                await self.log_test("缓存过期测试", "FAILED", error="Redis连接未建立")
                return False
            
            # 测试TTL设置
            expire_key = "expire_test_key"
            expire_value = "expire_test_value"
            ttl_seconds = 2
            
            # 设置带过期时间的键值
            self.redis_client.set(expire_key, expire_value, ex=ttl_seconds)
            
            # 立即检查
            immediate_value = self.redis_client.get(expire_key)
            immediate_ttl = self.redis_client.ttl(expire_key)
            
            # 等待过期
            await asyncio.sleep(ttl_seconds + 1)
            
            # 过期后检查
            expired_value = self.redis_client.get(expire_key)
            expired_ttl = self.redis_client.ttl(expire_key)
            
            details = {
                "immediate_value_correct": immediate_value == expire_value,
                "initial_ttl": immediate_ttl,
                "expired_value_none": expired_value is None,
                "final_ttl": expired_ttl,
                "expiration_working": immediate_value == expire_value and expired_value is None
            }
            
            if immediate_value == expire_value and expired_value is None:
                await self.log_test("缓存过期测试", "PASSED", details)
                return True
            else:
                await self.log_test("缓存过期测试", "FAILED", 
                                  details,
                                  error="过期机制未正常工作")
                return False
                
        except Exception as e:
            await self.log_test("缓存过期测试", "FAILED", error=str(e))
            return False
    
    async def test_cache_data_types(self):
        """测试缓存数据类型支持
        
        测试场景: SS-INT-003-004
        验证点: 不同数据类型的缓存支持
        """
        try:
            if not self.redis_client:
                await self.log_test("缓存数据类型测试", "FAILED", error="Redis连接未建立")
                return False
            
            test_data = {
                "string_data": "简单字符串数据",
                "json_data": json.dumps({"name": "测试", "value": 123, "list": [1, 2, 3]}),
                "chinese_data": "包含中文字符的数据：你好世界！",
                "special_chars": "特殊字符：@#$%^&*()_+-=[]{}|;:,.<>?",
                "number_as_string": "12345"
            }
            
            # 存储不同类型的数据
            storage_results = {}
            for key, value in test_data.items():
                cache_key = f"datatype_test_{key}"
                storage_results[key] = self.redis_client.set(cache_key, value)
            
            # 读取并验证数据
            retrieval_results = {}
            for key, original_value in test_data.items():
                cache_key = f"datatype_test_{key}"
                retrieved_value = self.redis_client.get(cache_key)
                retrieval_results[key] = retrieved_value == original_value
            
            # 清理测试数据
            keys_to_delete = [f"datatype_test_{key}" for key in test_data.keys()]
            self.redis_client.delete(*keys_to_delete)
            
            all_stored = all(storage_results.values())
            all_retrieved = all(retrieval_results.values())
            
            details = {
                "test_data_types": list(test_data.keys()),
                "storage_results": storage_results,
                "retrieval_accuracy": retrieval_results,
                "all_types_supported": all_stored and all_retrieved
            }
            
            if all_stored and all_retrieved:
                await self.log_test("缓存数据类型测试", "PASSED", details)
                return True
            else:
                await self.log_test("缓存数据类型测试", "FAILED", 
                                  details,
                                  error=f"数据类型支持不完整: 存储={all_stored}, 读取={all_retrieved}")
                return False
                
        except Exception as e:
            await self.log_test("缓存数据类型测试", "FAILED", error=str(e))
            return False
    
    async def test_cache_memory_usage(self):
        """测试缓存内存使用
        
        测试场景: SS-INT-003-005
        验证点: 内存使用情况和限制
        """
        try:
            if not self.redis_client:
                await self.log_test("缓存内存测试", "FAILED", error="Redis连接未建立")
                return False
            
            # 获取初始内存信息
            initial_info = self.redis_client.info('memory')
            initial_memory = initial_info.get('used_memory', 0)
            
            # 创建一些测试数据
            large_data = "x" * 10000  # 10KB 数据
            num_keys = 50
            
            for i in range(num_keys):
                key = f"memory_test_{i}"
                self.redis_client.set(key, f"{large_data}_{i}")
            
            # 获取使用后的内存信息
            after_info = self.redis_client.info('memory')
            after_memory = after_info.get('used_memory', 0)
            
            memory_increase = after_memory - initial_memory
            
            # 清理测试数据
            keys_to_delete = [f"memory_test_{i}" for i in range(num_keys)]
            deleted_count = self.redis_client.delete(*keys_to_delete)
            
            # 获取清理后的内存信息
            final_info = self.redis_client.info('memory')
            final_memory = final_info.get('used_memory', 0)
            
            details = {
                "initial_memory_bytes": initial_memory,
                "after_storage_memory_bytes": after_memory,
                "final_memory_bytes": final_memory,
                "memory_increase_bytes": memory_increase,
                "memory_increase_kb": round(memory_increase / 1024, 2),
                "keys_created": num_keys,
                "keys_deleted": deleted_count,
                "memory_cleanup_working": final_memory < after_memory
            }
            
            if deleted_count == num_keys and memory_increase > 0:
                await self.log_test("缓存内存测试", "PASSED", details)
                return True
            else:
                await self.log_test("缓存内存测试", "FAILED", 
                                  details,
                                  error=f"内存管理异常: 删除了{deleted_count}/{num_keys}个键")
                return False
                
        except Exception as e:
            await self.log_test("缓存内存测试", "FAILED", error=str(e))
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.redis_client:
            self.redis_client.close()
    
    async def run_all_tests(self):
        """运行所有Redis缓存集成测试"""
        print("⚡ 开始执行Redis缓存集成测试...")
        
        try:
            await self.test_redis_connection()
            await self.test_cache_performance()
            await self.test_cache_expiration()
            await self.test_cache_data_types()
            await self.test_cache_memory_usage()
        finally:
            self.cleanup()
        
        # 生成测试摘要
        passed_tests = len([t for t in self.test_results if t["status"] == "PASSED"])
        total_tests = len(self.test_results)
        
        print(f"\n📊 Redis缓存集成测试摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {total_tests - passed_tests}")
        print(f"   成功率: {round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0}%")
        
        return self.test_results


async def main():
    tester = RedisCacheIntegrationTester()
    results = await tester.run_all_tests()
    
    # 保存测试结果
    with open("/Users/yjlh/Documents/code/Historical Text Project/test-results/redis_integration_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_type": "redis_integration",
            "execution_time": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())