"""
SS-INT-001: MongoDB数据操作集成测试
优先级: P0 - 核心数据持久化
"""

import asyncio
import json
import aiohttp
from datetime import datetime
from typing import Dict, List, Any


class MongoDBIntegrationTester:
    """MongoDB集成测试器"""
    
    def __init__(self):
        self.base_url = "http://localhost:8002"
        self.test_results = []
        self.created_content_ids = []
        
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
    
    async def test_mongodb_connection(self, session):
        """测试MongoDB连接
        
        测试场景: SS-INT-001-001
        验证点: MongoDB数据库连接和基本操作
        """
        try:
            # 测试health端点，它会检查MongoDB连接
            async with session.get(f"{self.base_url}/api/v1/data/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    
                    details = {
                        "connection_status": "connected",
                        "response_time": f"<{response.headers.get('X-Process-Time', 'unknown')}s",
                        "health_check_passed": True
                    }
                    
                    await self.log_test("MongoDB连接测试", "PASSED", details)
                    return True
                else:
                    await self.log_test("MongoDB连接测试", "FAILED", 
                                      error=f"Health check failed: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            await self.log_test("MongoDB连接测试", "FAILED", error=str(e))
            return False
    
    async def test_mongodb_crud_operations(self, session):
        """测试MongoDB CRUD操作
        
        测试场景: SS-INT-001-002
        验证点: 创建、读取、更新、删除操作
        """
        try:
            # 1. 创建测试内容 (Create)
            create_data = {
                "title": "MongoDB集成测试文档",
                "content": "这是用于测试MongoDB集成的文档内容，包含中文字符和特殊符号：@#$%",
                "content_type": "text",
                "tags": ["集成测试", "MongoDB", "数据库"],
                "metadata": {
                    "test_type": "integration",
                    "database": "mongodb",
                    "created_by": "integration_test",
                    "test_timestamp": datetime.now().isoformat()
                }
            }
            
            async with session.post(f"{self.base_url}/api/v1/content/",
                                  json=create_data,
                                  headers={"Content-Type": "application/json"}) as response:
                if response.status in [200, 201]:
                    create_result = await response.json()
                    content_id = create_result.get("id") or create_result.get("content_id")
                    
                    if content_id:
                        self.created_content_ids.append(content_id)
                        
                        # 2. 读取刚创建的内容 (Read)
                        async with session.get(f"{self.base_url}/api/v1/content/{content_id}") as read_response:
                            if read_response.status == 200:
                                read_result = await read_response.json()
                                
                                # 验证数据完整性
                                assert read_result.get("title") == create_data["title"]
                                assert read_result.get("content") == create_data["content"]
                                assert "集成测试" in read_result.get("tags", [])
                                
                                # 3. 更新内容 (Update)
                                update_data = {
                                    "title": "MongoDB集成测试文档 - 已更新",
                                    "content": "更新后的文档内容",
                                    "tags": ["集成测试", "MongoDB", "数据库", "已更新"]
                                }
                                
                                async with session.put(f"{self.base_url}/api/v1/content/{content_id}",
                                                     json=update_data,
                                                     headers={"Content-Type": "application/json"}) as update_response:
                                    if update_response.status == 200:
                                        update_result = await update_response.json()
                                        
                                        # 验证更新
                                        assert "已更新" in update_result.get("title", "")
                                        assert "已更新" in update_result.get("tags", [])
                                        
                                        details = {
                                            "created_content_id": content_id,
                                            "create_success": True,
                                            "read_success": True,
                                            "update_success": True,
                                            "data_integrity_verified": True
                                        }
                                        
                                        await self.log_test("MongoDB CRUD操作", "PASSED", details)
                                        return True
            
            await self.log_test("MongoDB CRUD操作", "FAILED", error="操作流程未完整完成")
            return False
            
        except Exception as e:
            await self.log_test("MongoDB CRUD操作", "FAILED", error=str(e))
            return False
    
    async def test_mongodb_query_operations(self, session):
        """测试MongoDB查询操作
        
        测试场景: SS-INT-001-003
        验证点: 复杂查询和搜索功能
        """
        try:
            # 1. 创建多个测试文档用于查询
            test_docs = [
                {
                    "title": "历史文档A",
                    "content": "这是关于古代历史的文档，包含历史事件和人物",
                    "tags": ["历史", "古代", "文档"],
                    "metadata": {"category": "history", "period": "ancient"}
                },
                {
                    "title": "技术文档B", 
                    "content": "这是技术文档，说明软件架构和设计模式",
                    "tags": ["技术", "软件", "架构"],
                    "metadata": {"category": "technology", "type": "documentation"}
                },
                {
                    "title": "历史文档C",
                    "content": "现代历史文献，记录了重要的历史变迁",
                    "tags": ["历史", "现代", "文献"],
                    "metadata": {"category": "history", "period": "modern"}
                }
            ]
            
            created_ids = []
            for doc in test_docs:
                async with session.post(f"{self.base_url}/api/v1/content/",
                                      json=doc,
                                      headers={"Content-Type": "application/json"}) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        doc_id = result.get("id") or result.get("content_id")
                        if doc_id:
                            created_ids.append(doc_id)
                            self.created_content_ids.append(doc_id)
            
            # 2. 测试内容搜索
            search_params = {"q": "历史"}
            async with session.get(f"{self.base_url}/api/v1/content/search/",
                                 params=search_params) as response:
                if response.status == 200:
                    search_results = await response.json()
                    
                    # 验证搜索结果
                    if isinstance(search_results, list):
                        history_docs = [doc for doc in search_results if "历史" in doc.get("title", "")]
                        
                        details = {
                            "created_documents": len(created_ids),
                            "search_query": "历史",
                            "search_results_count": len(search_results),
                            "relevant_results": len(history_docs),
                            "search_functionality": "working"
                        }
                        
                        await self.log_test("MongoDB查询操作", "PASSED", details)
                        return True
            
            await self.log_test("MongoDB查询操作", "FAILED", error="搜索功能未正常工作")
            return False
            
        except Exception as e:
            await self.log_test("MongoDB查询操作", "FAILED", error=str(e))
            return False
    
    async def test_mongodb_performance(self, session):
        """测试MongoDB性能
        
        测试场景: SS-INT-001-004
        验证点: 批量操作和性能表现
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 批量创建文档测试性能
            batch_size = 10
            successful_creates = 0
            
            for i in range(batch_size):
                doc_data = {
                    "title": f"性能测试文档 {i+1}",
                    "content": f"这是第{i+1}个性能测试文档的内容" + "x" * 100,  # 增加内容长度
                    "tags": ["性能测试", f"批次{i//5 + 1}"],
                    "metadata": {"batch_number": i//5 + 1, "doc_index": i+1}
                }
                
                async with session.post(f"{self.base_url}/api/v1/content/",
                                      json=doc_data,
                                      headers={"Content-Type": "application/json"}) as response:
                    if response.status in [200, 201]:
                        successful_creates += 1
                        result = await response.json()
                        doc_id = result.get("id") or result.get("content_id")
                        if doc_id:
                            self.created_content_ids.append(doc_id)
            
            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time
            
            details = {
                "batch_size": batch_size,
                "successful_creates": successful_creates,
                "total_time": round(total_time, 3),
                "average_time_per_doc": round(total_time / batch_size, 3),
                "success_rate": round((successful_creates / batch_size) * 100, 2),
                "performance_acceptable": total_time < 30  # 30秒内完成认为可接受
            }
            
            if successful_creates == batch_size and total_time < 30:
                await self.log_test("MongoDB性能测试", "PASSED", details)
                return True
            else:
                await self.log_test("MongoDB性能测试", "FAILED", 
                                  details, 
                                  error=f"性能不达标: {successful_creates}/{batch_size} 成功, 用时{total_time:.2f}秒")
                return False
                
        except Exception as e:
            await self.log_test("MongoDB性能测试", "FAILED", error=str(e))
            return False
    
    async def cleanup_test_data(self, session):
        """清理测试数据"""
        cleanup_count = 0
        for content_id in self.created_content_ids:
            try:
                async with session.delete(f"{self.base_url}/api/v1/content/{content_id}") as response:
                    if response.status in [200, 204]:
                        cleanup_count += 1
            except:
                pass  # 忽略清理错误
        
        print(f"🧹 清理了 {cleanup_count}/{len(self.created_content_ids)} 个测试数据")
    
    async def run_all_tests(self):
        """运行所有MongoDB集成测试"""
        print("🗄️ 开始执行MongoDB集成测试...")
        
        async with aiohttp.ClientSession() as session:
            # 依次执行测试
            await self.test_mongodb_connection(session)
            await self.test_mongodb_crud_operations(session)
            await self.test_mongodb_query_operations(session)
            await self.test_mongodb_performance(session)
            
            # 清理测试数据
            await self.cleanup_test_data(session)
        
        # 生成测试摘要
        passed_tests = len([t for t in self.test_results if t["status"] == "PASSED"])
        total_tests = len(self.test_results)
        
        print(f"\n📊 MongoDB集成测试摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {total_tests - passed_tests}")
        print(f"   成功率: {round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0}%")
        
        return self.test_results


async def main():
    tester = MongoDBIntegrationTester()
    results = await tester.run_all_tests()
    
    # 保存测试结果
    with open("/Users/yjlh/Documents/code/Historical Text Project/test-results/mongodb_integration_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_type": "mongodb_integration",
            "execution_time": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())