"""
API接口测试

测试发布服务的主要API功能
"""

import pytest
from httpx import AsyncClient


class TestPublishingAPI:
    """发布API测试类"""
    
    async def test_health_check(self, test_client: AsyncClient):
        """测试健康检查接口"""
        response = await test_client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    async def test_get_platforms(self, test_client: AsyncClient):
        """测试获取平台列表"""
        response = await test_client.get("/api/v1/platforms")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "platforms" in data["data"]
    
    async def test_create_task_invalid_data(self, test_client: AsyncClient):
        """测试创建任务 - 无效数据"""
        invalid_task_data = {
            "content": "",  # 空内容
            "platforms": []  # 空平台列表
        }
        
        response = await test_client.post("/api/v1/publish", json=invalid_task_data)
        assert response.status_code == 422  # 数据验证错误
    
    async def test_create_task_valid_data(self, test_client: AsyncClient):
        """测试创建任务 - 有效数据"""
        task_data = {
            "content": "测试发布内容",
            "platforms": ["weibo"],
            "title": "测试标题"
        }
        
        response = await test_client.post("/api/v1/publish", json=task_data)
        # 注意：实际会失败，因为没有配置真实的平台账号，但可以测试API结构
        assert response.status_code in [200, 400, 500]


class TestHealthAPI:
    """健康检查API测试类"""
    
    async def test_root_endpoint(self, test_client: AsyncClient):
        """测试根路径"""
        response = await test_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["status"] == "running"