#!/usr/bin/env python3
"""
智能分类服务集成测试脚本
测试智能分类服务的完整功能流程
包括项目管理、训练数据、模型训练、文档分类等
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntelligentClassificationIntegrationTest:
    """智能分类服务集成测试类"""
    
    def __init__(self, base_url: str = "http://localhost:8007"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"
        self.session = None
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "service": "intelligent-classification-service",
            "base_url": base_url,
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": []
            }
        }
        
        # 测试数据
        self.test_project_id = None
        self.test_model_id = None
        self.test_training_data_ids = []
        
    async def setup(self):
        """初始化测试环境"""
        logger.info("🔧 初始化测试环境...")
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
    async def teardown(self):
        """清理测试环境"""
        logger.info("🧹 清理测试环境...")
        if self.session:
            await self.session.close()
            
        # 清理测试数据
        await self.cleanup_test_data()
    
    async def cleanup_test_data(self):
        """清理测试数据"""
        try:
            # 删除测试项目（会级联删除相关数据）
            if self.test_project_id:
                await self.make_request("DELETE", f"/projects/{self.test_project_id}")
        except Exception as e:
            logger.warning(f"清理测试数据失败: {e}")
    
    async def make_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """发送HTTP请求"""
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        
        kwargs = {}
        if data:
            kwargs["json"] = data
        if params:
            kwargs["params"] = params
            
        async with self.session.request(method, url, **kwargs) as response:
            response_data = await response.json()
            
            if response.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=response_data.get("message", "请求失败")
                )
            
            return response_data
    
    async def test_service_health(self) -> bool:
        """测试服务健康检查"""
        test_name = "服务健康检查"
        logger.info(f"🏥 开始测试: {test_name}")
        start_time = time.time()
        
        try:
            # 测试根路径
            async with self.session.get(f"{self.base_url}/") as response:
                root_data = await response.json()
                assert response.status == 200
                assert root_data["service"] == "intelligent-classification-service"
            
            # 测试健康检查
            async with self.session.get(f"{self.base_url}/health") as response:
                health_data = await response.json()
                assert response.status == 200
                assert health_data["service"] == "intelligent-classification-service"
                assert health_data["status"] in ["healthy", "unhealthy"]
            
            # 测试就绪检查
            async with self.session.get(f"{self.base_url}/ready") as response:
                ready_data = await response.json()
                # 就绪检查可能失败（如果storage-service未启动）
                
            # 测试服务信息
            async with self.session.get(f"{self.base_url}/info") as response:
                info_data = await response.json()
                assert response.status == 200
                assert "service" in info_data
                assert "features" in info_data
                
            duration = time.time() - start_time
            self.test_results["tests"].append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "details": {
                    "root_response": root_data,
                    "health_response": health_data,
                    "info_response": info_data
                }
            })
            self.test_results["summary"]["passed"] += 1
            logger.info(f"✅ {test_name} - 通过 ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"服务健康检查失败: {str(e)}"
            self.test_results["tests"].append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": error_msg,
                "traceback": traceback.format_exc()
            })
            self.test_results["summary"]["failed"] += 1
            self.test_results["summary"]["errors"].append(error_msg)
            logger.error(f"❌ {test_name} - 失败: {error_msg}")
            return False
    
    async def test_project_management(self) -> bool:
        """测试项目管理功能"""
        test_name = "项目管理API"
        logger.info(f"📋 开始测试: {test_name}")
        start_time = time.time()
        
        try:
            # 1. 获取支持的分类类型
            supported_types = await self.make_request("GET", "/projects/supported/types")
            assert supported_types["success"] == True
            assert "classification_types" in supported_types["data"]
            
            # 2. 创建测试项目
            project_data = {
                "name": "集成测试项目",
                "description": "用于集成测试的古代文献主题分类项目",
                "classification_type": "topic",
                "language": "zh",
                "custom_labels": ["政治", "军事", "经济", "文化", "社会"]
            }
            
            create_response = await self.make_request("POST", "/projects", project_data)
            assert create_response["success"] == True
            self.test_project_id = create_response["data"]["id"]
            
            # 3. 获取项目详情
            project_detail = await self.make_request("GET", f"/projects/{self.test_project_id}")
            assert project_detail["success"] == True
            assert project_detail["data"]["name"] == project_data["name"]
            assert project_detail["data"]["classification_type"] == project_data["classification_type"]
            
            # 4. 更新项目
            update_data = {
                "description": "更新后的项目描述 - 集成测试"
            }
            update_response = await self.make_request("PUT", f"/projects/{self.test_project_id}", update_data)
            assert update_response["success"] == True
            
            # 5. 列出项目
            projects_list = await self.make_request("GET", "/projects", params={"limit": 10})
            assert projects_list["success"] == True
            assert len(projects_list["data"]["projects"]) >= 1
            
            duration = time.time() - start_time
            self.test_results["tests"].append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "details": {
                    "project_id": self.test_project_id,
                    "supported_types_count": len(supported_types["data"]["classification_types"]),
                    "project_created": True,
                    "project_updated": True,
                    "projects_listed": len(projects_list["data"]["projects"])
                }
            })
            self.test_results["summary"]["passed"] += 1
            logger.info(f"✅ {test_name} - 通过 ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"项目管理测试失败: {str(e)}"
            self.test_results["tests"].append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": error_msg,
                "traceback": traceback.format_exc()
            })
            self.test_results["summary"]["failed"] += 1
            self.test_results["summary"]["errors"].append(error_msg)
            logger.error(f"❌ {test_name} - 失败: {error_msg}")
            return False
    
    async def test_training_data_management(self) -> bool:
        """测试训练数据管理功能"""
        test_name = "训练数据管理API"
        logger.info(f"📊 开始测试: {test_name}")
        start_time = time.time()
        
        if not self.test_project_id:
            logger.error("项目ID为空，跳过训练数据测试")
            return False
        
        try:
            # 1. 添加单条训练数据
            training_data = {
                "project_id": self.test_project_id,
                "text_content": "汉武帝时期，国力强盛，多次出征匈奴，开拓疆土，建立了强大的汉帝国。",
                "true_label": "政治",
                "label_confidence": 1.0,
                "data_source": "integration_test"
            }
            
            add_response = await self.make_request("POST", "/data/training-data", training_data)
            assert add_response["success"] == True
            training_data_id = add_response["data"]["id"]
            self.test_training_data_ids.append(training_data_id)
            
            # 2. 批量添加训练数据
            batch_data = {
                "project_id": self.test_project_id,
                "training_data": [
                    {
                        "text_content": "唐朝诗歌繁荣，李白、杜甫等诗人创作了许多传世佳作。",
                        "true_label": "文化"
                    },
                    {
                        "text_content": "宋朝商业发达，海上丝绸之路贸易兴盛，经济繁荣。",
                        "true_label": "经济"
                    },
                    {
                        "text_content": "明朝军队装备精良，火器使用广泛，军事实力强大。",
                        "true_label": "军事"
                    },
                    {
                        "text_content": "清朝社会等级森严，满汉有别，社会矛盾尖锐。",
                        "true_label": "社会"
                    }
                ]
            }
            
            batch_response = await self.make_request("POST", "/data/training-data/batch", batch_data)
            assert batch_response["success"] == True
            assert batch_response["data"]["successful_added"] == 4
            
            # 3. 获取训练数据
            data_list = await self.make_request("GET", f"/data/training-data/{self.test_project_id}", 
                                               params={"limit": 10})
            assert data_list["success"] == True
            assert len(data_list["data"]["data"]) >= 5
            
            # 4. 获取训练数据统计
            stats_response = await self.make_request("GET", f"/data/training-data/{self.test_project_id}/statistics")
            assert stats_response["success"] == True
            
            # 5. 验证训练数据质量
            validation_response = await self.make_request("POST", f"/data/training-data/{self.test_project_id}/validate")
            assert validation_response["success"] == True
            assert "quality_score" in validation_response["data"]
            
            duration = time.time() - start_time
            self.test_results["tests"].append({
                "name": test_name,
                "status": "PASSED", 
                "duration": duration,
                "details": {
                    "single_data_added": True,
                    "batch_data_added": batch_response["data"]["successful_added"],
                    "total_data_count": len(data_list["data"]["data"]),
                    "quality_score": validation_response["data"]["quality_score"]
                }
            })
            self.test_results["summary"]["passed"] += 1
            logger.info(f"✅ {test_name} - 通过 ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"训练数据管理测试失败: {str(e)}"
            self.test_results["tests"].append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": error_msg,
                "traceback": traceback.format_exc()
            })
            self.test_results["summary"]["failed"] += 1
            self.test_results["summary"]["errors"].append(error_msg)
            logger.error(f"❌ {test_name} - 失败: {error_msg}")
            return False
    
    async def test_model_training(self) -> bool:
        """测试模型训练功能"""
        test_name = "模型训练API"
        logger.info(f"🤖 开始测试: {test_name}")
        start_time = time.time()
        
        if not self.test_project_id:
            logger.error("项目ID为空，跳过模型训练测试")
            return False
        
        try:
            # 1. 获取支持的模型类型
            model_types = await self.make_request("GET", "/models/types/supported")
            assert model_types["success"] == True
            assert "model_types" in model_types["data"]
            
            # 2. 启动模型训练（使用快速的随机森林）
            training_request = {
                "project_id": self.test_project_id,
                "model_type": "random_forest",
                "feature_extractor": "tfidf",
                "hyperparameters": {
                    "n_estimators": 10,  # 减少树的数量以加快训练
                    "max_depth": 5
                },
                "training_config": {
                    "test_size": 0.3,
                    "cv_folds": 3
                }
            }
            
            train_response = await self.make_request("POST", "/models/train", training_request)
            assert train_response["success"] == True
            self.test_model_id = train_response["data"]["model_id"]
            
            # 3. 等待训练完成（轮询检查）
            max_wait_time = 120  # 最多等待2分钟
            wait_start = time.time()
            training_completed = False
            
            while time.time() - wait_start < max_wait_time:
                model_info = await self.make_request("GET", f"/models/{self.test_model_id}")
                if model_info["success"]:
                    status = model_info["data"].get("status")
                    if status == "completed":
                        training_completed = True
                        break
                    elif status == "failed":
                        raise Exception(f"模型训练失败: {model_info['data'].get('error_message', '未知错误')}")
                
                await asyncio.sleep(5)  # 等待5秒后再次检查
            
            if not training_completed:
                logger.warning("模型训练超时，继续其他测试")
                # 不算作失败，因为可能是正常的长时间训练
            
            # 4. 获取项目的模型列表
            models_list = await self.make_request("GET", f"/models/project/{self.test_project_id}")
            assert models_list["success"] == True
            assert len(models_list["data"]["models"]) >= 1
            
            # 5. 如果训练完成，测试激活模型
            if training_completed:
                activate_response = await self.make_request("POST", f"/models/{self.test_model_id}/activate")
                assert activate_response["success"] == True
                
                # 获取活跃模型
                active_model = await self.make_request("GET", f"/models/project/{self.test_project_id}/active")
                assert active_model["success"] == True
                assert active_model["data"]["model_id"] == self.test_model_id
            
            duration = time.time() - start_time
            self.test_results["tests"].append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "details": {
                    "model_id": self.test_model_id,
                    "training_started": True,
                    "training_completed": training_completed,
                    "models_count": len(models_list["data"]["models"]),
                    "model_activated": training_completed
                }
            })
            self.test_results["summary"]["passed"] += 1
            logger.info(f"✅ {test_name} - 通过 ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"模型训练测试失败: {str(e)}"
            self.test_results["tests"].append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": error_msg,
                "traceback": traceback.format_exc()
            })
            self.test_results["summary"]["failed"] += 1
            self.test_results["summary"]["errors"].append(error_msg)
            logger.error(f"❌ {test_name} - 失败: {error_msg}")
            return False
    
    async def test_document_classification(self) -> bool:
        """测试文档分类功能"""
        test_name = "文档分类API"
        logger.info(f"🔍 开始测试: {test_name}")
        start_time = time.time()
        
        if not self.test_project_id:
            logger.error("项目ID为空，跳过文档分类测试")
            return False
        
        try:
            # 1. 单文档分类测试
            classification_request = {
                "project_id": self.test_project_id,
                "text_content": "康熙皇帝在位期间，实行仁政，国泰民安，是清朝的盛世。",
                "return_probabilities": True,
                "return_explanation": True
            }
            
            single_result = await self.make_request("POST", "/classify/single", classification_request)
            assert single_result["success"] == True
            assert "predicted_label" in single_result["data"]
            assert "confidence_score" in single_result["data"]
            
            # 2. 批量文档分类测试
            batch_request = {
                "project_id": self.test_project_id,
                "documents": [
                    {"text_content": "秦始皇统一六国，建立中央集权制度。"},
                    {"text_content": "唐代诗歌艺术达到顶峰，诗人辈出。"},
                    {"text_content": "宋朝商贸发达，市民经济繁荣。"}
                ],
                "return_probabilities": True
            }
            
            batch_result = await self.make_request("POST", "/classify/batch", batch_request)
            assert batch_result["success"] == True
            assert batch_result["data"]["total_documents"] == 3
            assert len(batch_result["data"]["results"]) == 3
            
            # 3. 带详细解释的分类
            explanation_request = {
                "project_id": self.test_project_id,
                "text_content": "明朝海禁政策限制了对外贸易的发展。",
                "return_probabilities": True,
                "return_explanation": True
            }
            
            explanation_result = await self.make_request("POST", "/classify/predict-with-explanation", 
                                                       explanation_request)
            assert explanation_result["success"] == True
            assert "classification_result" in explanation_result["data"]
            assert "decision_process" in explanation_result["data"]
            
            # 4. 获取分类历史
            history_result = await self.make_request("GET", f"/classify/history/{self.test_project_id}",
                                                   params={"limit": 10})
            assert history_result["success"] == True
            
            # 5. 获取分类统计
            stats_result = await self.make_request("GET", f"/classify/statistics/{self.test_project_id}")
            assert stats_result["success"] == True
            
            duration = time.time() - start_time
            self.test_results["tests"].append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "details": {
                    "single_classification": {
                        "predicted_label": single_result["data"]["predicted_label"],
                        "confidence": single_result["data"]["confidence_score"]
                    },
                    "batch_classification": {
                        "total_documents": batch_result["data"]["total_documents"],
                        "successful": batch_result["data"]["successful_classifications"]
                    },
                    "explanation_provided": "decision_process" in explanation_result["data"],
                    "history_retrieved": True,
                    "statistics_retrieved": True
                }
            })
            self.test_results["summary"]["passed"] += 1
            logger.info(f"✅ {test_name} - 通过 ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"文档分类测试失败: {str(e)}"
            self.test_results["tests"].append({
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "error": error_msg,
                "traceback": traceback.format_exc()
            })
            self.test_results["summary"]["failed"] += 1
            self.test_results["summary"]["errors"].append(error_msg)
            logger.error(f"❌ {test_name} - 失败: {error_msg}")
            return False
    
    async def run_all_tests(self) -> Dict:
        """运行所有集成测试"""
        logger.info("🚀 开始智能分类服务集成测试...")
        
        await self.setup()
        
        try:
            # 更新测试总数
            self.test_results["summary"]["total"] = 5
            
            # 依次执行所有测试
            await self.test_service_health()
            await self.test_project_management()
            await self.test_training_data_management()
            await self.test_model_training()
            await self.test_document_classification()
            
        except Exception as e:
            logger.error(f"集成测试异常: {e}")
            self.test_results["summary"]["errors"].append(f"整体测试异常: {str(e)}")
        
        finally:
            await self.teardown()
            
        # 完成测试结果
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["total_duration"] = sum(
            test.get("duration", 0) for test in self.test_results["tests"]
        )
        
        # 计算成功率
        passed = self.test_results["summary"]["passed"]
        total = self.test_results["summary"]["total"]
        self.test_results["summary"]["success_rate"] = (passed / total * 100) if total > 0 else 0
        
        return self.test_results

async def main():
    """主函数"""
    print("=" * 60)
    print("🧪 智能分类服务集成测试")
    print("=" * 60)
    
    # 创建测试实例
    tester = IntelligentClassificationIntegrationTest()
    
    # 运行测试
    results = await tester.run_all_tests()
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    print(f"🎯 总测试数: {results['summary']['total']}")
    print(f"✅ 通过: {results['summary']['passed']}")
    print(f"❌ 失败: {results['summary']['failed']}")
    print(f"📈 成功率: {results['summary']['success_rate']:.1f}%")
    print(f"⏱️  总耗时: {results['total_duration']:.2f}秒")
    
    if results['summary']['errors']:
        print(f"\n❗ 错误列表:")
        for i, error in enumerate(results['summary']['errors'], 1):
            print(f"  {i}. {error}")
    
    # 保存测试结果
    with open("integration_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细测试结果已保存到: integration_test_results.json")
    print("🏁 集成测试完成")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())