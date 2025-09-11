"""
内容质量控制服务集成测试

测试整个服务的端到端功能，包括API接口、服务集成、
工作流程等的完整性和正确性。
"""

import pytest
import asyncio
import httpx
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app
from src.models.quality_models import (
    QualityCheckRequest, ComplianceCheckRequest,
    ReviewTaskCreateRequest, ReviewDecisionRequest,
    ReviewDecision
)

class TestContentQualityIntegration:
    """内容质量控制服务集成测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_storage_service(self):
        """模拟Storage Service响应"""
        with patch('src.clients.storage_client.StorageServiceClient._make_request') as mock:
            # 模拟健康检查响应
            mock.return_value = {"status": "healthy"}
            yield mock
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
    
    def test_detailed_health_check(self, client, mock_storage_service):
        """测试详细健康检查端点"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "dependencies" in data
        assert "storage_service" in data["dependencies"]
    
    def test_service_info(self, client):
        """测试服务信息端点"""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "config" in data
        assert "endpoints" in data
    
    def test_root_endpoint(self, client):
        """测试根路径端点"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert "service" in data

class TestQualityCheckIntegration:
    """质量检测集成测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_storage_responses(self):
        """模拟Storage Service的各种响应"""
        with patch('src.clients.storage_client.StorageServiceClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # 模拟保存质量检测结果
            mock_client.save_quality_check_result.return_value = {
                "success": True,
                "data": {"saved": True}
            }
            
            # 模拟获取质量规则
            mock_client.get_quality_rules.return_value = {
                "data": []
            }
            
            # 模拟健康检查
            mock_client.health_check.return_value = {
                "status": "healthy"
            }
            
            yield mock_client
    
    def test_quality_check_api(self, client, mock_storage_responses):
        """测试质量检测API"""
        # 准备测试数据
        test_content = "朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州。"
        
        request_data = {
            "content": test_content,
            "content_type": "historical_text",
            "check_options": {
                "grammar_check": True,
                "logic_check": True,
                "format_check": True,
                "factual_check": True,
                "academic_check": True
            },
            "auto_fix": True
        }
        
        response = client.post("/api/v1/quality/check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        
        quality_data = data["data"]
        assert "check_id" in quality_data
        assert "overall_score" in quality_data
        assert "status" in quality_data
        assert "quality_analysis" in quality_data
        assert "issues" in quality_data
        assert "suggestions" in quality_data
        assert "processing_time_ms" in quality_data
        
        # 验证分数范围
        assert 0 <= quality_data["overall_score"] <= 100
    
    def test_quality_check_invalid_content(self, client):
        """测试无效内容的质量检测"""
        # 测试空内容
        request_data = {
            "content": "",
            "content_type": "general"
        }
        
        response = client.post("/api/v1/quality/check", json=request_data)
        assert response.status_code == 422  # 验证错误
    
    def test_quality_check_long_content(self, client):
        """测试过长内容的质量检测"""
        # 创建超长内容
        long_content = "测试" * 50000  # 100000字符
        
        request_data = {
            "content": long_content,
            "content_type": "general"
        }
        
        response = client.post("/api/v1/quality/check", json=request_data)
        assert response.status_code == 400  # 参数错误
    
    def test_batch_quality_check_api(self, client, mock_storage_responses):
        """测试批量质量检测API"""
        request_data = {
            "content_ids": ["content1", "content2", "content3"],
            "check_options": {
                "grammar_check": True,
                "logic_check": True,
                "format_check": True,
                "factual_check": True,
                "academic_check": True
            },
            "parallel_processing": True,
            "max_concurrent_tasks": 3
        }
        
        response = client.post("/api/v1/quality/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        
        batch_data = data["data"]
        assert "batch_id" in batch_data
        assert "total_items" in batch_data
        assert "completed_items" in batch_data
        assert "success_rate" in batch_data
        assert batch_data["total_items"] == 3

class TestComplianceCheckIntegration:
    """合规检测集成测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_storage_responses(self):
        """模拟Storage Service的各种响应"""
        with patch('src.clients.storage_client.StorageServiceClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # 模拟敏感词列表
            mock_client.get_sensitive_words.return_value = {
                "data": [
                    {
                        "word": "测试敏感词",
                        "category": "test",
                        "severity_level": 5,
                        "replacement_suggestion": "替代词"
                    }
                ]
            }
            
            # 模拟保存合规检测结果
            mock_client.save_compliance_check_result.return_value = {
                "success": True,
                "data": {"saved": True}
            }
            
            # 模拟健康检查
            mock_client.health_check.return_value = {
                "status": "healthy"
            }
            
            yield mock_client
    
    def test_compliance_check_api(self, client, mock_storage_responses):
        """测试合规检测API"""
        request_data = {
            "content": "这是一个包含测试敏感词的内容。",
            "check_types": ["sensitive_words", "policy", "copyright", "academic_integrity"],
            "strict_mode": False
        }
        
        response = client.post("/api/v1/compliance/check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        
        compliance_data = data["data"]
        assert "check_id" in compliance_data
        assert "compliance_status" in compliance_data
        assert "risk_score" in compliance_data
        assert "violations" in compliance_data
        assert "policy_compliance" in compliance_data
        assert "recommendations" in compliance_data
        
        # 验证风险分数范围
        assert 0 <= compliance_data["risk_score"] <= 10
    
    def test_compliance_check_clean_content(self, client, mock_storage_responses):
        """测试干净内容的合规检测"""
        request_data = {
            "content": "这是一个完全正常的内容，没有任何违规信息。",
            "check_types": ["sensitive_words", "policy"]
        }
        
        response = client.post("/api/v1/compliance/check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        compliance_data = data["data"]
        
        # 干净内容应该有较低的风险分数
        assert compliance_data["risk_score"] <= 3
    
    def test_get_sensitive_words_api(self, client, mock_storage_responses):
        """测试获取敏感词列表API"""
        response = client.get("/api/v1/compliance/sensitive-words")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_add_sensitive_word_api(self, client, mock_storage_responses):
        """测试添加敏感词API"""
        mock_storage_responses.add_sensitive_word.return_value = {
            "success": True,
            "data": {"id": "new_word_id"}
        }
        
        word_data = {
            "word": "新敏感词",
            "category": "test",
            "severity_level": 5,
            "replacement_suggestion": "替代词"
        }
        
        response = client.post("/api/v1/compliance/sensitive-words", json=word_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
    
    def test_batch_compliance_check_api(self, client, mock_storage_responses):
        """测试批量合规检测API"""
        request_data = {
            "contents": [
                "第一个测试内容",
                "第二个测试内容包含测试敏感词",
                "第三个正常内容"
            ],
            "check_types": ["sensitive_words", "policy"],
            "strict_mode": False
        }
        
        response = client.post("/api/v1/compliance/batch-check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        
        batch_data = data["data"]
        assert batch_data["total_items"] == 3
        assert "results" in batch_data
        assert len(batch_data["results"]) == 3

class TestReviewWorkflowIntegration:
    """审核工作流集成测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_storage_responses(self):
        """模拟Storage Service的各种响应"""
        with patch('src.clients.storage_client.StorageServiceClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # 模拟创建审核任务
            mock_client.create_review_task.return_value = {
                "task_id": "test_task_id"
            }
            
            # 模拟获取工作流
            mock_client.get_active_workflows.return_value = {
                "data": [
                    {
                        "id": "workflow_1",
                        "workflow_name": "standard_review",
                        "auto_approval_threshold": 85.0,
                        "auto_approval_risk_threshold": 3
                    }
                ]
            }
            
            # 模拟获取审核任务
            mock_client.get_review_task.return_value = {
                "data": {
                    "task_id": "test_task_id",
                    "content_id": "test_content",
                    "task_status": "pending",
                    "priority_score": 5
                }
            }
            
            # 模拟获取任务列表
            mock_client.get_review_tasks.return_value = {
                "data": {
                    "tasks": [
                        {
                            "task_id": "task1",
                            "content_id": "content1",
                            "priority_score": 8,
                            "task_status": "pending"
                        }
                    ],
                    "pagination": {
                        "total": 1,
                        "page": 1,
                        "per_page": 20
                    }
                }
            }
            
            # 模拟健康检查
            mock_client.health_check.return_value = {
                "status": "healthy"
            }
            
            yield mock_client
    
    def test_create_review_task_api(self, client, mock_storage_responses):
        """测试创建审核任务API"""
        request_data = {
            "content_id": "test_content_123",
            "quality_result": {
                "overall_score": 75.5,
                "status": "needs_review",
                "issues": []
            },
            "compliance_result": {
                "compliance_status": "warning",
                "risk_score": 4,
                "violations": []
            }
        }
        
        response = client.post("/api/v1/review/tasks", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        
        task_data = data["data"]
        assert "task_id" in task_data
        assert "status" in task_data
        assert "priority_score" in task_data
    
    def test_get_review_task_api(self, client, mock_storage_responses):
        """测试获取审核任务详情API"""
        response = client.get("/api/v1/review/tasks/test_task_id")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
    
    def test_get_review_tasks_list_api(self, client, mock_storage_responses):
        """测试获取审核任务列表API"""
        response = client.get("/api/v1/review/tasks?status=pending&page=1&per_page=20")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert "pagination" in data
    
    def test_submit_review_decision_api(self, client, mock_storage_responses):
        """测试提交审核决策API"""
        mock_storage_responses.update_review_task.return_value = {"success": True}
        mock_storage_responses.submit_review_decision.return_value = {"success": True}
        
        decision_data = {
            "decision": "approve",
            "notes": "内容质量良好，通过审核",
            "required_changes": [],
            "review_time_minutes": 15
        }
        
        response = client.post("/api/v1/review/tasks/test_task_id/decision", json=decision_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
    
    def test_get_pending_tasks_api(self, client, mock_storage_responses):
        """测试获取待审核任务API"""
        response = client.get("/api/v1/review/tasks/pending?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
    
    def test_comprehensive_quality_check_api(self, client, mock_storage_responses):
        """测试综合质量检测API"""
        # 模拟保存检测结果
        mock_storage_responses.save_quality_check_result.return_value = {"success": True}
        mock_storage_responses.save_compliance_check_result.return_value = {"success": True}
        
        # 模拟获取敏感词
        mock_storage_responses.get_sensitive_words.return_value = {"data": []}
        
        request_data = {
            "content": "朱元璋是明朝的开国皇帝，建立了明朝政权。",
            "content_id": "test_content_456",
            "auto_create_task": True
        }
        
        response = client.post("/api/v1/review/comprehensive-check", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        
        result_data = data["data"]
        assert "quality_check" in result_data
        assert "compliance_check" in result_data
        assert "review_task" in result_data

class TestEndToEndWorkflow:
    """端到端工作流测试类"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_storage_responses(self):
        """模拟完整的Storage Service响应"""
        with patch('src.clients.storage_client.StorageServiceClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # 模拟所有必要的响应
            mock_client.get_sensitive_words.return_value = {"data": []}
            mock_client.save_quality_check_result.return_value = {"success": True}
            mock_client.save_compliance_check_result.return_value = {"success": True}
            mock_client.create_review_task.return_value = {"task_id": "e2e_task_id"}
            mock_client.get_active_workflows.return_value = {
                "data": [
                    {
                        "id": "standard_workflow",
                        "workflow_name": "standard_review",
                        "auto_approval_threshold": 90.0,
                        "auto_approval_risk_threshold": 2
                    }
                ]
            }
            mock_client.update_review_task.return_value = {"success": True}
            mock_client.submit_review_decision.return_value = {"success": True}
            mock_client.health_check.return_value = {"status": "healthy"}
            
            yield mock_client
    
    def test_complete_workflow(self, client, mock_storage_responses):
        """测试完整的内容质量控制工作流"""
        # 步骤1: 进行综合质量检测
        content = "朱元璋，明朝开国皇帝，1328年生于濠州。其政治才能卓越，军事策略高明。"
        
        check_response = client.post("/api/v1/review/comprehensive-check", json={
            "content": content,
            "content_id": "e2e_content_123",
            "auto_create_task": True
        })
        
        assert check_response.status_code == 200
        check_data = check_response.json()["data"]
        
        # 验证质量检测结果
        quality_result = check_data["quality_check"]
        assert "overall_score" in quality_result
        assert "status" in quality_result
        
        # 验证合规检测结果
        compliance_result = check_data["compliance_check"]
        assert "compliance_status" in compliance_result
        assert "risk_score" in compliance_result
        
        # 验证审核任务创建
        task_result = check_data["review_task"]
        assert "task_id" in task_result
        task_id = task_result["task_id"]
        
        # 步骤2: 模拟审核决策
        decision_response = client.post(f"/api/v1/review/tasks/{task_id}/decision", json={
            "decision": "approve_with_changes",
            "notes": "内容整体良好，建议优化部分表述",
            "required_changes": [
                {
                    "position": 10,
                    "description": "建议调整语法",
                    "suggestion": "更规范的表达"
                }
            ],
            "review_time_minutes": 20
        })
        
        assert decision_response.status_code == 200
        decision_data = decision_response.json()["data"]
        
        # 验证决策处理结果
        assert "new_status" in decision_data
        assert "decision" in decision_data
        
        # 步骤3: 验证整个流程的数据一致性
        assert quality_result["overall_score"] >= 0
        assert compliance_result["risk_score"] >= 0
        assert task_result["priority_score"] >= 1
    
    def test_auto_approval_workflow(self, client, mock_storage_responses):
        """测试自动审核通过工作流"""
        # 使用高质量内容测试自动审核
        high_quality_content = "朱元璋是明朝开国皇帝，其治国理念和政策对后世产生了深远影响。"
        
        response = client.post("/api/v1/review/comprehensive-check", json={
            "content": high_quality_content,
            "content_id": "auto_approve_content",
            "auto_create_task": True
        })
        
        assert response.status_code == 200
        data = response.json()["data"]
        
        # 验证可能的自动审核
        task_result = data["review_task"]
        # 根据配置，高质量低风险内容可能会被自动审核通过
        assert "task_id" in task_result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])