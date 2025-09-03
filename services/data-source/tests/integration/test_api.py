"""
API集成测试
测试完整的API接口功能
"""

import pytest
import json
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch


class TestCrawlerAPI:
    """爬虫API测试"""
    
    @pytest.mark.asyncio
    async def test_create_crawler_task(self, async_client, sample_crawler_config, assert_response_structure):
        """测试创建爬虫任务"""
        response = await async_client.post("/api/v1/crawlers/", json=sample_crawler_config)
        
        assert response.status_code == 200
        data = response.json()
        
        assert_response_structure(data, ["task_id", "platform", "keywords", "status"])
        assert data["data"]["platform"] == sample_crawler_config["platform"]
        assert data["data"]["keywords"] == sample_crawler_config["keywords"]
        assert data["data"]["status"] == "idle"
    
    @pytest.mark.asyncio
    async def test_create_invalid_crawler_task(self, async_client):
        """测试创建无效爬虫任务"""
        invalid_config = {
            "platform": "invalid_platform",  # 无效平台
            "keywords": [],  # 空关键词列表
            "max_pages": -1  # 无效页数
        }
        
        response = await async_client.post("/api/v1/crawlers/", json=invalid_config)
        assert response.status_code == 422  # 验证错误
    
    @pytest.mark.asyncio
    async def test_start_crawler_task(self, async_client, sample_crawler_config, mock_crawler_manager):
        """测试启动爬虫任务"""
        # 模拟成功启动
        mock_crawler_manager.start_task.return_value = True
        
        # 首先创建任务
        create_response = await async_client.post("/api/v1/crawlers/", json=sample_crawler_config)
        task_id = create_response.json()["data"]["task_id"]
        
        # 启动任务
        response = await async_client.post(f"/api/v1/crawlers/{task_id}/start")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == task_id
    
    @pytest.mark.asyncio
    async def test_start_nonexistent_task(self, async_client, mock_crawler_manager):
        """测试启动不存在的任务"""
        mock_crawler_manager.start_task.return_value = False
        mock_crawler_manager.get_task_status.return_value = None
        
        response = await async_client.post("/api/v1/crawlers/fake-task-id/start")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, async_client, sample_crawler_config, mock_crawler_manager):
        """测试获取任务状态"""
        # 创建模拟任务
        from src.crawler.crawler_manager import CrawlerTask, CrawlerStatus, CrawlerConfig
        from src.models.content import ContentSource
        from datetime import datetime
        
        mock_task = CrawlerTask(
            task_id="test-task-123",
            config=CrawlerConfig(
                platform=ContentSource.TOUTIAO,
                keywords=["历史", "文化"],
                max_pages=5
            ),
            status=CrawlerStatus.RUNNING,
            created_at=datetime.now()
        )
        mock_task.progress = 50.0
        mock_task.success_items = 10
        mock_task.failed_items = 2
        
        mock_crawler_manager.get_task_status.return_value = mock_task
        
        response = await async_client.get("/api/v1/crawlers/test-task-123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "test-task-123"
        assert data["data"]["status"] == "running"
        assert data["data"]["progress"] == 50.0
        assert data["data"]["success_items"] == 10
        assert data["data"]["failed_items"] == 2
    
    @pytest.mark.asyncio
    async def test_get_all_tasks(self, async_client, mock_crawler_manager):
        """测试获取所有任务"""
        # 模拟返回空列表
        mock_crawler_manager.get_all_tasks.return_value = []
        
        response = await async_client.get("/api/v1/crawlers/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "items" in data["data"]
        assert "total" in data["data"]
        assert "page" in data["data"]
        assert "size" in data["data"]
    
    @pytest.mark.asyncio
    async def test_get_crawler_statistics(self, async_client, mock_crawler_manager):
        """测试获取爬虫统计"""
        mock_stats = {
            "total_tasks": 10,
            "running_tasks": 2,
            "finished_tasks": 7,
            "error_tasks": 1,
            "total_success_items": 150,
            "total_failed_items": 20,
            "overall_success_rate": 88.2
        }
        mock_crawler_manager.get_statistics.return_value = mock_stats
        
        response = await async_client.get("/api/v1/crawlers/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == mock_stats


class TestContentAPI:
    """内容API测试"""
    
    @pytest.mark.asyncio
    async def test_create_content(self, async_client, sample_content_data, mock_db_manager):
        """测试创建内容"""
        # 模拟数据库操作
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None  # 不存在重复内容
        mock_collection.insert_one.return_value = AsyncMock(inserted_id="mock_id")
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.post("/api/v1/content/", json=sample_content_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "id" in data["data"]
        assert data["data"]["title"] == sample_content_data["title"]
        assert data["data"]["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_create_duplicate_content(self, async_client, sample_content_data, mock_db_manager):
        """测试创建重复内容"""
        # 模拟数据库找到重复内容
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = {"id": "existing-id"}
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.post("/api/v1/content/", json=sample_content_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "existing_id" in data["data"]
    
    @pytest.mark.asyncio
    async def test_batch_create_content(self, async_client, mock_data_generator, mock_db_manager):
        """测试批量创建内容"""
        contents = mock_data_generator.generate_content_batch(5)
        batch_data = {
            "contents": contents,
            "batch_name": "测试批次",
            "auto_deduplicate": True
        }
        
        # 模拟数据库操作
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None  # 无重复
        mock_collection.insert_one.return_value = AsyncMock(inserted_id="mock_id")
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.post("/api/v1/content/batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_count"] == 5
        assert data["data"]["success_count"] > 0
    
    @pytest.mark.asyncio
    async def test_get_content_list(self, async_client, mock_db_manager):
        """测试获取内容列表"""
        # 模拟数据库查询结果
        mock_collection = AsyncMock()
        mock_collection.count_documents.return_value = 0
        
        # 模拟游标
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = AsyncMock(return_value=iter([]))
        mock_collection.find.return_value.sort.return_value.skip.return_value.limit.return_value = mock_cursor
        
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.get("/api/v1/content/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "items" in data["data"]
        assert "total" in data["data"]
    
    @pytest.mark.asyncio
    async def test_get_content_detail(self, async_client, mock_db_manager):
        """测试获取内容详情"""
        # 模拟找到内容
        mock_content = {
            "id": "test-content-123",
            "title": "测试内容",
            "content": "这是测试内容的正文",
            "source": "manual",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00"
        }
        
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = mock_content
        mock_collection.update_one.return_value = AsyncMock()
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.get("/api/v1/content/test-content-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == "test-content-123"
        assert data["data"]["title"] == "测试内容"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_content(self, async_client, mock_db_manager):
        """测试获取不存在的内容"""
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.get("/api/v1/content/nonexistent-id")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_update_content(self, async_client, mock_db_manager):
        """测试更新内容"""
        update_data = {
            "title": "更新后的标题",
            "category": "新分类"
        }
        
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = {"id": "test-id"}  # 存在
        mock_collection.update_one.return_value = AsyncMock(modified_count=1)
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.put("/api/v1/content/test-id", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == "test-id"
    
    @pytest.mark.asyncio
    async def test_delete_content(self, async_client, mock_db_manager):
        """测试删除内容"""
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = {"id": "test-id", "title": "测试内容"}
        mock_collection.delete_one.return_value = AsyncMock(deleted_count=1)
        mock_db_manager.get_mongodb_collection.return_value = mock_collection
        
        response = await async_client.delete("/api/v1/content/test-id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == "test-id"


class TestProxyAPI:
    """代理API测试"""
    
    @pytest.mark.asyncio
    async def test_get_proxy_list(self, async_client, mock_proxy_manager):
        """测试获取代理列表"""
        mock_proxy_manager.proxies = {}
        
        response = await async_client.get("/api/v1/proxy/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "items" in data["data"]
        assert "total" in data["data"]
    
    @pytest.mark.asyncio
    async def test_get_active_proxies(self, async_client, mock_proxy_manager):
        """测试获取可用代理"""
        mock_proxy_manager.active_proxies = []
        mock_proxy_manager.proxies = {}
        
        response = await async_client.get("/api/v1/proxy/active")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] == 0
    
    @pytest.mark.asyncio
    async def test_get_best_proxy(self, async_client, mock_proxy_manager):
        """测试获取最佳代理"""
        mock_proxy_manager.get_proxy.return_value = None  # 无可用代理
        
        response = await async_client.get("/api/v1/proxy/best")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["data"] is None
    
    @pytest.mark.asyncio
    async def test_test_proxy(self, async_client, sample_proxy_data, mock_proxy_manager):
        """测试代理测试接口"""
        # 模拟代理测试成功
        with patch('src.proxy.proxy_manager.ProxyManager.test_proxy') as mock_test:
            mock_test.return_value = True
            
            response = await async_client.post("/api/v1/proxy/test", json=sample_proxy_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["test_result"] is True
    
    @pytest.mark.asyncio
    async def test_refresh_proxy_list(self, async_client, mock_proxy_manager):
        """测试刷新代理列表"""
        # 模拟刷新前后的数量
        mock_proxy_manager.proxies = {"old_proxy": "data"}
        mock_proxy_manager.refresh_proxies = AsyncMock()
        mock_proxy_manager.active_proxies = []
        
        response = await async_client.post("/api/v1/proxy/refresh")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_proxy_manager.refresh_proxies.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_proxy_statistics(self, async_client, mock_proxy_manager):
        """测试获取代理统计"""
        mock_stats = {
            "total_proxies": 10,
            "active_proxies": 7,
            "banned_proxies": 2,
            "average_success_rate": 85.5
        }
        mock_proxy_manager.get_proxy_statistics.return_value = mock_stats
        
        response = await async_client.get("/api/v1/proxy/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == mock_stats


class TestSystemAPI:
    """系统API测试"""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client):
        """测试根端点"""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "service" in data["data"]
        assert "version" in data["data"]
        assert "status" in data["data"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, async_client, mock_db_manager, mock_crawler_manager, mock_proxy_manager):
        """测试健康检查"""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]
        assert "components" in data["data"]
        assert "database" in data["data"]["components"]
        assert "crawler" in data["data"]["components"]
        assert "proxy" in data["data"]["components"]
    
    @pytest.mark.asyncio
    async def test_service_info(self, async_client):
        """测试服务信息"""
        response = await async_client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "service" in data["data"]
        assert "features" in data["data"]
        assert "api" in data["data"]


class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.mark.asyncio
    async def test_404_error(self, async_client):
        """测试404错误"""
        response = await async_client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_validation_error(self, async_client):
        """测试验证错误"""
        invalid_data = {
            "invalid_field": "invalid_value"
        }
        
        response = await async_client.post("/api/v1/crawlers/", json=invalid_data)
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_internal_server_error(self, async_client, mock_crawler_manager):
        """测试内部服务器错误"""
        # 模拟异常
        mock_crawler_manager.get_statistics.side_effect = Exception("测试异常")
        
        response = await async_client.get("/api/v1/crawlers/statistics")
        assert response.status_code == 500
        
        data = response.json()
        assert data["success"] is False
        assert "error" in data