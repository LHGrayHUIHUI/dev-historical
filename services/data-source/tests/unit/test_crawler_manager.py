"""
爬虫管理器单元测试
"""

import pytest
import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.crawler.crawler_manager import (
    CrawlerManager,
    CrawlerConfig,
    CrawlerTask,
    CrawlerStatus,
    TaskPriority,
    ToutiaoCrawler
)
from src.models.content import ContentSource


@pytest.fixture
def crawler_config():
    """测试爬虫配置"""
    return CrawlerConfig(
        platform=ContentSource.TOUTIAO,
        keywords=["历史", "文化"],
        max_pages=5,
        interval=1.0,
        priority=TaskPriority.NORMAL
    )


@pytest.fixture
def crawler_manager():
    """测试爬虫管理器"""
    manager = CrawlerManager()
    # 模拟数据库操作，避免真实数据库连接
    manager._save_task_to_db = AsyncMock()
    manager._load_tasks_from_db = AsyncMock(return_value=[])
    manager._update_task_in_db = AsyncMock()
    manager._delete_task_from_db = AsyncMock()
    return manager


class TestCrawlerManager:
    """爬虫管理器测试类"""
    
    @pytest.mark.asyncio
    async def test_create_task(self, crawler_manager, crawler_config):
        """测试创建爬虫任务"""
        task_id = await crawler_manager.create_task(crawler_config)
        
        assert task_id is not None
        assert task_id in crawler_manager.tasks
        
        task = crawler_manager.tasks[task_id]
        assert task.config.platform == ContentSource.TOUTIAO
        assert task.config.keywords == ["历史", "文化"]
        assert task.status == CrawlerStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_create_multiple_tasks(self, crawler_manager, crawler_config):
        """测试创建多个任务"""
        task_ids = []
        for i in range(3):
            config = CrawlerConfig(
                platform=ContentSource.TOUTIAO,
                keywords=[f"关键词{i}"],
                max_pages=3
            )
            task_id = await crawler_manager.create_task(config)
            task_ids.append(task_id)
        
        assert len(task_ids) == 3
        assert len(set(task_ids)) == 3  # 确保ID唯一
        assert len(crawler_manager.tasks) == 3
    
    @pytest.mark.asyncio
    async def test_start_task_success(self, crawler_manager, crawler_config):
        """测试成功启动任务"""
        # 创建任务
        task_id = await crawler_manager.create_task(crawler_config)
        
        # 模拟HTTP会话
        crawler_manager.session = AsyncMock()
        
        # 启动任务
        with patch('asyncio.create_task') as mock_create_task:
            success = await crawler_manager.start_task(task_id)
            
            assert success is True
            assert task_id in crawler_manager.crawlers
            mock_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_nonexistent_task(self, crawler_manager):
        """测试启动不存在的任务"""
        fake_task_id = str(uuid.uuid4())
        success = await crawler_manager.start_task(fake_task_id)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_start_task_wrong_status(self, crawler_manager, crawler_config):
        """测试启动错误状态的任务"""
        task_id = await crawler_manager.create_task(crawler_config)
        
        # 修改任务状态
        crawler_manager.tasks[task_id].status = CrawlerStatus.RUNNING
        
        success = await crawler_manager.start_task(task_id)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_stop_task(self, crawler_manager, crawler_config):
        """测试停止任务"""
        task_id = await crawler_manager.create_task(crawler_config)
        
        # 设置任务状态为运行中
        task = crawler_manager.tasks[task_id]
        task.status = CrawlerStatus.RUNNING
        
        # 创建模拟爬虫
        mock_crawler = AsyncMock()
        crawler_manager.crawlers[task_id] = mock_crawler
        
        success = await crawler_manager.stop_task(task_id)
        
        assert success is True
        assert task.status == CrawlerStatus.STOPPED
        mock_crawler.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pause_and_resume_task(self, crawler_manager, crawler_config):
        """测试暂停和恢复任务"""
        task_id = await crawler_manager.create_task(crawler_config)
        
        # 设置任务状态为运行中
        task = crawler_manager.tasks[task_id]
        task.status = CrawlerStatus.RUNNING
        
        # 创建模拟爬虫
        mock_crawler = AsyncMock()
        crawler_manager.crawlers[task_id] = mock_crawler
        
        # 测试暂停
        success = await crawler_manager.pause_task(task_id)
        assert success is True
        mock_crawler.pause.assert_called_once()
        
        # 测试恢复
        success = await crawler_manager.resume_task(task_id)
        assert success is True
        mock_crawler.resume.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, crawler_manager, crawler_config):
        """测试获取任务状态"""
        task_id = await crawler_manager.create_task(crawler_config)
        
        task = crawler_manager.get_task_status(task_id)
        assert task is not None
        assert task.task_id == task_id
        assert task.config.platform == ContentSource.TOUTIAO
        
        # 测试获取不存在任务的状态
        fake_task_id = str(uuid.uuid4())
        task = crawler_manager.get_task_status(fake_task_id)
        assert task is None
    
    @pytest.mark.asyncio
    async def test_get_all_tasks(self, crawler_manager):
        """测试获取所有任务"""
        # 创建多个任务
        configs = [
            CrawlerConfig(platform=ContentSource.TOUTIAO, keywords=["历史"]),
            CrawlerConfig(platform=ContentSource.BAIJIAHAO, keywords=["文化"]),
            CrawlerConfig(platform=ContentSource.XIAOHONGSHU, keywords=["艺术"])
        ]
        
        for config in configs:
            await crawler_manager.create_task(config)
        
        all_tasks = crawler_manager.get_all_tasks()
        assert len(all_tasks) == 3
        
        platforms = [task.config.platform for task in all_tasks]
        assert ContentSource.TOUTIAO in platforms
        assert ContentSource.BAIJIAHAO in platforms
        assert ContentSource.XIAOHONGSHU in platforms
    
    @pytest.mark.asyncio
    async def test_get_running_tasks(self, crawler_manager, crawler_config):
        """测试获取运行中的任务"""
        # 创建多个任务
        task_ids = []
        for i in range(3):
            task_id = await crawler_manager.create_task(crawler_config)
            task_ids.append(task_id)
        
        # 设置其中两个为运行状态
        crawler_manager.tasks[task_ids[0]].status = CrawlerStatus.RUNNING
        crawler_manager.tasks[task_ids[1]].status = CrawlerStatus.RUNNING
        crawler_manager.tasks[task_ids[2]].status = CrawlerStatus.IDLE
        
        running_tasks = crawler_manager.get_running_tasks()
        assert len(running_tasks) == 2
        
        for task in running_tasks:
            assert task.status == CrawlerStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, crawler_manager):
        """测试获取统计信息"""
        # 创建测试任务并设置不同状态
        configs = [CrawlerConfig(platform=ContentSource.TOUTIAO, keywords=["test"]) for _ in range(5)]
        task_ids = []
        
        for config in configs:
            task_id = await crawler_manager.create_task(config)
            task_ids.append(task_id)
        
        # 设置不同的任务状态和统计数据
        crawler_manager.tasks[task_ids[0]].status = CrawlerStatus.RUNNING
        crawler_manager.tasks[task_ids[0]].success_items = 10
        crawler_manager.tasks[task_ids[0]].failed_items = 2
        
        crawler_manager.tasks[task_ids[1]].status = CrawlerStatus.FINISHED
        crawler_manager.tasks[task_ids[1]].success_items = 15
        crawler_manager.tasks[task_ids[1]].failed_items = 1
        
        crawler_manager.tasks[task_ids[2]].status = CrawlerStatus.ERROR
        crawler_manager.tasks[task_ids[2]].failed_items = 5
        
        stats = await crawler_manager.get_statistics()
        
        assert stats["total_tasks"] == 5
        assert stats["running_tasks"] == 1
        assert stats["finished_tasks"] == 1
        assert stats["error_tasks"] == 1
        assert stats["total_success_items"] == 25
        assert stats["total_failed_items"] == 8
        assert stats["overall_success_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, crawler_manager):
        """测试并发任务限制"""
        # 设置较小的并发限制进行测试
        original_limit = crawler_manager.settings.crawler.max_concurrent_crawlers
        crawler_manager.settings.crawler.max_concurrent_crawlers = 2
        
        try:
            # 创建多个任务
            task_ids = []
            for i in range(5):
                config = CrawlerConfig(platform=ContentSource.TOUTIAO, keywords=[f"test{i}"])
                task_id = await crawler_manager.create_task(config)
                task_ids.append(task_id)
                
                # 设置为运行状态以模拟已启动的任务
                if i < 2:
                    crawler_manager.tasks[task_id].status = CrawlerStatus.RUNNING
            
            # 尝试启动第三个任务（应该失败）
            crawler_manager.session = AsyncMock()
            success = await crawler_manager.start_task(task_ids[2])
            assert success is False
            
        finally:
            # 恢复原始限制
            crawler_manager.settings.crawler.max_concurrent_crawlers = original_limit


class TestCrawlerTask:
    """爬虫任务测试类"""
    
    def test_crawler_task_properties(self):
        """测试爬虫任务属性计算"""
        config = CrawlerConfig(
            platform=ContentSource.TOUTIAO,
            keywords=["test"]
        )
        
        task = CrawlerTask(
            task_id="test-task-123",
            config=config,
            status=CrawlerStatus.RUNNING,
            created_at=datetime.now()
        )
        
        # 测试初始状态
        assert task.success_rate == 0.0
        assert task.duration is None  # started_at未设置
        
        # 设置统计数据
        task.success_items = 8
        task.failed_items = 2
        
        # 测试成功率计算
        assert task.success_rate == 80.0
        
        # 测试持续时间计算
        task.started_at = datetime.now()
        duration = task.duration
        assert duration is not None
        assert duration.total_seconds() >= 0


class TestToutiaoCrawler:
    """今日头条爬虫测试类"""
    
    @pytest.mark.asyncio
    async def test_crawler_initialization(self):
        """测试爬虫初始化"""
        config = CrawlerConfig(
            platform=ContentSource.TOUTIAO,
            keywords=["历史", "文化"],
            max_pages=3,
            interval=1.0
        )
        
        task = CrawlerTask(
            task_id="test-crawler-task",
            config=config,
            status=CrawlerStatus.IDLE,
            created_at=datetime.now()
        )
        
        session = AsyncMock()
        crawler = ToutiaoCrawler(task, session)
        
        assert crawler.task == task
        assert crawler.session == session
        assert crawler.is_running is False
        assert crawler.should_stop is False
    
    @pytest.mark.asyncio
    async def test_crawler_stop_signal(self):
        """测试爬虫停止信号"""
        config = CrawlerConfig(platform=ContentSource.TOUTIAO, keywords=["test"])
        task = CrawlerTask("test", config, CrawlerStatus.IDLE, datetime.now())
        
        session = AsyncMock()
        crawler = ToutiaoCrawler(task, session)
        
        await crawler.stop()
        assert crawler.should_stop is True
    
    @pytest.mark.asyncio
    async def test_crawler_pause_resume(self):
        """测试爬虫暂停和恢复"""
        config = CrawlerConfig(platform=ContentSource.TOUTIAO, keywords=["test"])
        task = CrawlerTask("test", config, CrawlerStatus.RUNNING, datetime.now())
        
        session = AsyncMock()
        crawler = ToutiaoCrawler(task, session)
        
        # 测试暂停
        await crawler.pause()
        assert task.status == CrawlerStatus.PAUSED
        
        # 测试恢复
        await crawler.resume()
        assert task.status == CrawlerStatus.RUNNING
    
    def test_crawler_update_progress(self):
        """测试爬虫进度更新"""
        config = CrawlerConfig(platform=ContentSource.TOUTIAO, keywords=["test"])
        task = CrawlerTask("test", config, CrawlerStatus.RUNNING, datetime.now())
        
        session = AsyncMock()
        crawler = ToutiaoCrawler(task, session)
        
        # 更新进度
        crawler.update_progress(5, 10)
        assert task.progress == 50.0
        assert task.total_items == 10
        
        # 测试除零保护
        crawler.update_progress(0, 0)
        assert task.progress == 0.0