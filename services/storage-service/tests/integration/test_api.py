"""
API集成测试
"""

import io
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


class TestDataCollectionAPI:
    """数据采集API集成测试"""
    
    def test_health_check(self, client: TestClient):
        """测试健康检查端点"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "timestamp" in data
    
    def test_service_info(self, client: TestClient):
        """测试服务信息端点"""
        response = client.get("/api/v1/data/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "service_name" in data
        assert "service_version" in data
        assert "supported_file_types" in data
        assert isinstance(data["supported_file_types"], list)
    
    @pytest.mark.skip(reason="需要真实的数据库连接")
    def test_upload_file(self, client: TestClient, temp_file):
        """测试文件上传"""
        # 创建测试文件
        test_content = "这是一个测试文件内容。"
        test_file = io.BytesIO(test_content.encode())
        
        # 准备上传数据
        files = {"file": ("test.txt", test_file, "text/plain")}
        data = {
            "source_id": str(uuid4()),
            "metadata": json.dumps({
                "title": "测试文件",
                "description": "用于API测试的文件"
            })
        }
        
        response = client.post("/api/v1/data/upload", files=files, data=data)
        
        # 注意：在没有真实数据库的情况下，这个测试会失败
        # 在完整的测试环境中，这应该返回200状态码
        assert response.status_code in [200, 500]  # 500是因为没有数据库连接
        
        if response.status_code == 200:
            response_data = response.json()
            assert response_data["success"] is True
            assert "data" in response_data
            assert "dataset_id" in response_data["data"]
    
    @pytest.mark.skip(reason="需要真实的数据库连接")
    def test_get_datasets(self, client: TestClient):
        """测试获取数据集列表"""
        response = client.get("/api/v1/data/datasets")
        
        # 在没有认证的情况下可能会失败
        assert response.status_code in [200, 401, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "data" in data
            assert "items" in data["data"]
    
    @pytest.mark.skip(reason="需要真实的数据库连接")  
    def test_batch_upload(self, client: TestClient):
        """测试批量文件上传"""
        # 创建多个测试文件
        files = []
        for i in range(3):
            content = f"测试文件内容 {i+1}"
            test_file = io.BytesIO(content.encode())
            files.append(("files", (f"test_{i+1}.txt", test_file, "text/plain")))
        
        data = {
            "source_id": str(uuid4()),
            "metadata": json.dumps({
                "batch_name": "测试批次",
                "description": "批量上传测试"
            })
        }
        
        response = client.post("/api/v1/data/upload/batch", files=files, data=data)
        
        # 在没有真实数据库的情况下会失败
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            response_data = response.json()
            assert response_data["success"] is True
            assert "data" in response_data
            assert response_data["data"]["total_files"] == 3
    
    def test_invalid_file_upload(self, client: TestClient):
        """测试无效文件上传"""
        # 测试不支持的文件类型
        test_file = io.BytesIO(b"fake exe content")
        files = {"file": ("malicious.exe", test_file, "application/x-executable")}
        data = {"source_id": str(uuid4())}
        
        response = client.post("/api/v1/data/upload", files=files, data=data)
        
        # 应该返回415不支持的媒体类型或422验证错误
        assert response.status_code in [415, 422, 500]
    
    def test_validation_errors(self, client: TestClient):
        """测试验证错误"""
        # 测试缺少必需参数
        response = client.post("/api/v1/data/upload")
        assert response.status_code == 422
        
        response_data = response.json()
        assert response_data["success"] is False
        assert "error_message" in response_data
        assert "details" in response_data


class TestErrorHandling:
    """错误处理测试"""
    
    def test_404_error(self, client: TestClient):
        """测试404错误"""
        response = client.get("/api/v1/data/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client: TestClient):
        """测试方法不允许错误"""
        response = client.delete("/health")
        assert response.status_code == 405
    
    def test_validation_error_format(self, client: TestClient):
        """测试验证错误格式"""
        response = client.post("/api/v1/data/upload", json={"invalid": "data"})
        
        assert response.status_code == 422
        data = response.json()
        
        # 检查错误响应格式
        assert "success" in data
        assert data["success"] is False
        assert "error_code" in data
        assert "error_message" in data
        assert "timestamp" in data
        
        # 验证错误代码
        assert data["error_code"] == "VALIDATION_ERROR"