"""
模型测试

测试数据库模型的基本功能
"""

import pytest
from datetime import datetime, timedelta

from src.models.publishing_models import (
    PublishingPlatform, PublishingAccount, PublishingTask
)


class TestPublishingModels:
    """发布模型测试类"""
    
    async def test_create_platform(self, test_db):
        """测试创建平台"""
        platform = PublishingPlatform(
            platform_name="test_weibo",
            platform_type="social_media",
            display_name="测试微博",
            api_endpoint="https://api.weibo.com/2",
            auth_type="oauth2",
            rate_limit_per_hour=100,
            is_active=True
        )
        
        test_db.add(platform)
        await test_db.commit()
        
        assert platform.id is not None
        assert platform.platform_name == "test_weibo"
        assert platform.is_active is True
    
    async def test_create_account(self, test_db):
        """测试创建账号"""
        # 先创建平台
        platform = PublishingPlatform(
            platform_name="test_platform",
            platform_type="social_media",
            display_name="测试平台",
            is_active=True
        )
        test_db.add(platform)
        await test_db.flush()
        
        # 创建账号
        account = PublishingAccount(
            platform_id=platform.id,
            account_name="test_account",
            account_identifier="test@example.com",
            auth_credentials={"access_token": "test_token"},
            daily_quota=50,
            used_quota=0
        )
        
        test_db.add(account)
        await test_db.commit()
        
        assert account.id is not None
        assert account.account_name == "test_account"
        assert account.is_available is True
        assert account.quota_remaining == 50
    
    async def test_create_task(self, test_db):
        """测试创建任务"""
        # 先创建平台和账号
        platform = PublishingPlatform(
            platform_name="test_platform",
            platform_type="social_media",
            display_name="测试平台",
            is_active=True
        )
        test_db.add(platform)
        await test_db.flush()
        
        account = PublishingAccount(
            platform_id=platform.id,
            account_name="test_account",
            daily_quota=50
        )
        test_db.add(account)
        await test_db.flush()
        
        # 创建任务
        task = PublishingTask(
            platform_id=platform.id,
            account_id=account.id,
            title="测试标题",
            content="测试内容",
            status="pending",
            scheduled_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        test_db.add(task)
        await test_db.commit()
        
        assert task.id is not None
        assert task.task_uuid is not None
        assert task.is_pending is True
        assert task.can_cancel is True