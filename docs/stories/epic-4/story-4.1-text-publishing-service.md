# Story 4.1: 文本发送服务

## 用户故事概述

**用户角色**: 内容发布员  
**需求描述**: 我需要统一的文本发送服务，以便将优化后的内容发布到多个平台  
**业务价值**: 实现多平台统一发布，提高发布效率

## 核心技术栈

### 后端技术栈
- **服务框架**: FastAPI + Python 3.11
- **任务队列**: Celery + Redis
- **调度器**: APScheduler + Cron
- **重试机制**: Tenacity + 指数退避
- **监控**: Prometheus + Grafana
- **日志**: ELK Stack (Elasticsearch + Logstash + Kibana)
- **配置管理**: Pydantic + YAML配置
- **数据库**: PostgreSQL (发布记录) + Redis (缓存)
- **消息队列**: RabbitMQ + Kafka
- **容器化**: Docker + Kubernetes

### 前端技术栈
- **框架**: Vue 3 + TypeScript
- **状态管理**: Pinia
- **UI组件**: Element Plus
- **HTTP客户端**: Axios
- **图表**: ECharts
- **构建工具**: Vite

## 数据模型设计

### PostgreSQL 数据表

```sql
-- 发布平台配置表
CREATE TABLE publishing_platforms (
    id SERIAL PRIMARY KEY,
    platform_name VARCHAR(50) NOT NULL UNIQUE,
    platform_type VARCHAR(20) NOT NULL, -- 'social_media', 'blog', 'news'
    api_endpoint VARCHAR(255),
    auth_type VARCHAR(20), -- 'oauth', 'api_key', 'cookie'
    config_schema JSONB,
    is_active BOOLEAN DEFAULT true,
    rate_limit_per_hour INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 发布账号表
CREATE TABLE publishing_accounts (
    id SERIAL PRIMARY KEY,
    platform_id INTEGER REFERENCES publishing_platforms(id),
    account_name VARCHAR(100) NOT NULL,
    account_identifier VARCHAR(255), -- 用户名或ID
    auth_credentials JSONB, -- 加密存储的认证信息
    account_status VARCHAR(20) DEFAULT 'active', -- 'active', 'suspended', 'expired'
    daily_quota INTEGER DEFAULT 50,
    used_quota INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 发布任务表
CREATE TABLE publishing_tasks (
    id SERIAL PRIMARY KEY,
    task_uuid UUID UNIQUE DEFAULT gen_random_uuid(),
    content_id INTEGER, -- 关联内容ID
    platform_id INTEGER REFERENCES publishing_platforms(id),
    account_id INTEGER REFERENCES publishing_accounts(id),
    title VARCHAR(500),
    content TEXT NOT NULL,
    media_urls JSONB, -- 图片、视频URL数组
    publish_config JSONB, -- 发布配置（标签、分类等）
    scheduled_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'published', 'failed', 'cancelled'
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    platform_post_id VARCHAR(255), -- 平台返回的帖子ID
    published_url VARCHAR(500),
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 发布统计表
CREATE TABLE publishing_stats (
    id SERIAL PRIMARY KEY,
    platform_id INTEGER REFERENCES publishing_platforms(id),
    account_id INTEGER REFERENCES publishing_accounts(id),
    date DATE NOT NULL,
    total_posts INTEGER DEFAULT 0,
    successful_posts INTEGER DEFAULT 0,
    failed_posts INTEGER DEFAULT 0,
    engagement_metrics JSONB, -- 点赞、评论、分享等数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(platform_id, account_id, date)
);

-- 创建索引
CREATE INDEX idx_publishing_tasks_status ON publishing_tasks(status);
CREATE INDEX idx_publishing_tasks_scheduled ON publishing_tasks(scheduled_at);
CREATE INDEX idx_publishing_tasks_platform ON publishing_tasks(platform_id);
CREATE INDEX idx_publishing_accounts_platform ON publishing_accounts(platform_id);
CREATE INDEX idx_publishing_stats_date ON publishing_stats(date);
```

### Redis 缓存结构

```python
# 账号使用配额缓存
account_quota:{account_id} = {
    "daily_limit": 50,
    "used_today": 23,
    "reset_time": "2024-01-01T00:00:00Z"
}

# 平台限流缓存
platform_rate_limit:{platform_id} = {
    "requests_per_hour": 100,
    "current_requests": 45,
    "window_start": "2024-01-01T10:00:00Z"
}

# 发布任务缓存
task_status:{task_uuid} = {
    "status": "processing",
    "progress": 75,
    "message": "正在上传媒体文件"
}

# 平台健康状态缓存
platform_health:{platform_id} = {
    "status": "healthy",
    "last_check": "2024-01-01T10:30:00Z",
    "response_time": 250,
    "error_rate": 0.02
}
```

## 服务架构设计

### 核心服务类

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from celery import Celery
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import aioredis
import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential

class PublishingService:
    """
    文本发送服务核心类
    负责管理多平台内容发布的统一接口
    """
    
    def __init__(self, db_pool, redis_client, celery_app):
        self.db_pool = db_pool
        self.redis = redis_client
        self.celery = celery_app
        self.platform_adapters = {}
        self._load_platform_adapters()
    
    def _load_platform_adapters(self):
        """
        加载各平台适配器
        """
        from .adapters import (
            WeiboAdapter, WechatAdapter, DouyinAdapter,
            ToutiaoAdapter, BaijiahaoAdapter
        )
        
        self.platform_adapters = {
            'weibo': WeiboAdapter(),
            'wechat': WechatAdapter(),
            'douyin': DouyinAdapter(),
            'toutiao': ToutiaoAdapter(),
            'baijiahao': BaijiahaoAdapter()
        }
    
    async def create_publishing_task(
        self, 
        content: str,
        platforms: List[str],
        title: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        scheduled_at: Optional[datetime] = None,
        publish_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        创建发布任务
        
        Args:
            content: 发布内容
            platforms: 目标平台列表
            title: 标题（可选）
            media_urls: 媒体文件URL列表
            scheduled_at: 定时发布时间
            publish_config: 发布配置
            
        Returns:
            List[str]: 任务UUID列表
        """
        task_uuids = []
        
        async with self.db_pool.acquire() as conn:
            for platform in platforms:
                # 选择可用账号
                account = await self._select_available_account(conn, platform)
                if not account:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"No available account for platform: {platform}"
                    )
                
                # 创建发布任务
                task_uuid = await self._create_task_record(
                    conn, account, content, title, media_urls, 
                    scheduled_at, publish_config
                )
                task_uuids.append(task_uuid)
                
                # 提交到任务队列
                if scheduled_at and scheduled_at > datetime.utcnow():
                    # 定时任务
                    self.celery.send_task(
                        'publish_content',
                        args=[task_uuid],
                        eta=scheduled_at
                    )
                else:
                    # 立即执行
                    self.celery.send_task(
                        'publish_content',
                        args=[task_uuid]
                    )
        
        return task_uuids
    
    async def _select_available_account(self, conn, platform: str):
        """
        选择可用的发布账号
        
        Args:
            conn: 数据库连接
            platform: 平台名称
            
        Returns:
            dict: 账号信息
        """
        query = """
        SELECT pa.*, pp.rate_limit_per_hour
        FROM publishing_accounts pa
        JOIN publishing_platforms pp ON pa.platform_id = pp.id
        WHERE pp.platform_name = $1 
        AND pa.account_status = 'active'
        AND pa.used_quota < pa.daily_quota
        ORDER BY pa.used_quota ASC, pa.last_used_at ASC NULLS FIRST
        LIMIT 1
        """
        
        account = await conn.fetchrow(query, platform)
        
        if account:
            # 检查Redis中的实时配额
            quota_key = f"account_quota:{account['id']}"
            quota_data = await self.redis.hgetall(quota_key)
            
            if quota_data:
                used_today = int(quota_data.get('used_today', 0))
                if used_today >= account['daily_quota']:
                    return None
        
        return account
    
    async def _create_task_record(
        self, conn, account, content, title, media_urls, 
        scheduled_at, publish_config
    ) -> str:
        """
        创建任务记录
        """
        query = """
        INSERT INTO publishing_tasks (
            platform_id, account_id, title, content, media_urls,
            publish_config, scheduled_at, status
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING task_uuid
        """
        
        import json
        task_uuid = await conn.fetchval(
            query,
            account['platform_id'],
            account['id'],
            title,
            content,
            json.dumps(media_urls) if media_urls else None,
            json.dumps(publish_config) if publish_config else None,
            scheduled_at,
            'pending'
        )
        
        return str(task_uuid)
    
    async def get_task_status(self, task_uuid: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_uuid: 任务UUID
            
        Returns:
            Dict[str, Any]: 任务状态信息
        """
        # 先从Redis缓存获取实时状态
        cache_key = f"task_status:{task_uuid}"
        cached_status = await self.redis.hgetall(cache_key)
        
        if cached_status:
            return {
                'task_uuid': task_uuid,
                'status': cached_status.get('status'),
                'progress': int(cached_status.get('progress', 0)),
                'message': cached_status.get('message'),
                'updated_at': cached_status.get('updated_at')
            }
        
        # 从数据库获取
        async with self.db_pool.acquire() as conn:
            query = """
            SELECT pt.*, pp.platform_name, pa.account_name
            FROM publishing_tasks pt
            JOIN publishing_platforms pp ON pt.platform_id = pp.id
            JOIN publishing_accounts pa ON pt.account_id = pa.id
            WHERE pt.task_uuid = $1
            """
            
            task = await conn.fetchrow(query, task_uuid)
            
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            return {
                'task_uuid': task_uuid,
                'platform': task['platform_name'],
                'account': task['account_name'],
                'status': task['status'],
                'title': task['title'],
                'scheduled_at': task['scheduled_at'],
                'published_at': task['published_at'],
                'published_url': task['published_url'],
                'error_message': task['error_message'],
                'retry_count': task['retry_count'],
                'created_at': task['created_at'],
                'updated_at': task['updated_at']
            }
    
    async def cancel_task(self, task_uuid: str) -> bool:
        """
        取消发布任务
        
        Args:
            task_uuid: 任务UUID
            
        Returns:
            bool: 是否成功取消
        """
        async with self.db_pool.acquire() as conn:
            query = """
            UPDATE publishing_tasks 
            SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
            WHERE task_uuid = $1 AND status IN ('pending', 'processing')
            RETURNING id
            """
            
            result = await conn.fetchval(query, task_uuid)
            
            if result:
                # 从Redis删除缓存
                await self.redis.delete(f"task_status:{task_uuid}")
                
                # 撤销Celery任务
                self.celery.control.revoke(task_uuid, terminate=True)
                
                return True
            
            return False
    
    async def get_publishing_statistics(
        self, 
        start_date: datetime,
        end_date: datetime,
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取发布统计数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            platform: 平台名称（可选）
            
        Returns:
            Dict[str, Any]: 统计数据
        """
        async with self.db_pool.acquire() as conn:
            base_query = """
            SELECT 
                pp.platform_name,
                COUNT(*) as total_tasks,
                COUNT(CASE WHEN pt.status = 'published' THEN 1 END) as successful_tasks,
                COUNT(CASE WHEN pt.status = 'failed' THEN 1 END) as failed_tasks,
                AVG(CASE WHEN pt.published_at IS NOT NULL THEN 
                    EXTRACT(EPOCH FROM (pt.published_at - pt.created_at)) END) as avg_publish_time
            FROM publishing_tasks pt
            JOIN publishing_platforms pp ON pt.platform_id = pp.id
            WHERE pt.created_at BETWEEN $1 AND $2
            """
            
            params = [start_date, end_date]
            
            if platform:
                base_query += " AND pp.platform_name = $3"
                params.append(platform)
            
            base_query += " GROUP BY pp.platform_name ORDER BY total_tasks DESC"
            
            stats = await conn.fetch(base_query, *params)
            
            return {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'platforms': [
                    {
                        'platform': row['platform_name'],
                        'total_tasks': row['total_tasks'],
                        'successful_tasks': row['successful_tasks'],
                        'failed_tasks': row['failed_tasks'],
                        'success_rate': round(row['successful_tasks'] / row['total_tasks'] * 100, 2) if row['total_tasks'] > 0 else 0,
                        'avg_publish_time': round(row['avg_publish_time'], 2) if row['avg_publish_time'] else 0
                    }
                    for row in stats
                ]
            }

class PlatformAdapter:
    """
    平台适配器基类
    定义各平台发布接口的统一规范
    """
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        平台认证
        
        Args:
            credentials: 认证凭据
            
        Returns:
            bool: 认证是否成功
        """
        raise NotImplementedError
    
    async def publish_content(
        self, 
        content: str,
        title: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        发布内容
        
        Args:
            content: 内容文本
            title: 标题
            media_urls: 媒体文件URL列表
            config: 发布配置
            
        Returns:
            Dict[str, Any]: 发布结果
        """
        raise NotImplementedError
    
    async def get_post_metrics(self, post_id: str) -> Dict[str, Any]:
        """
        获取帖子数据指标
        
        Args:
            post_id: 帖子ID
            
        Returns:
            Dict[str, Any]: 指标数据
        """
        raise NotImplementedError
    
    async def delete_post(self, post_id: str) -> bool:
        """
        删除帖子
        
        Args:
            post_id: 帖子ID
            
        Returns:
            bool: 是否成功删除
        """
        raise NotImplementedError

class TaskProcessor:
    """
    任务处理器
    负责执行具体的发布任务
    """
    
    def __init__(self, publishing_service: PublishingService):
        self.publishing_service = publishing_service
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def process_publishing_task(self, task_uuid: str):
        """
        处理发布任务
        
        Args:
            task_uuid: 任务UUID
        """
        try:
            # 更新任务状态为处理中
            await self._update_task_status(task_uuid, 'processing', 0, '开始处理发布任务')
            
            # 获取任务详情
            task_info = await self._get_task_info(task_uuid)
            
            # 获取平台适配器
            adapter = self.publishing_service.platform_adapters.get(
                task_info['platform_name']
            )
            
            if not adapter:
                raise Exception(f"Unsupported platform: {task_info['platform_name']}")
            
            # 认证
            await self._update_task_status(task_uuid, 'processing', 25, '正在进行平台认证')
            
            auth_success = await adapter.authenticate(task_info['auth_credentials'])
            if not auth_success:
                raise Exception("Platform authentication failed")
            
            # 上传媒体文件（如果有）
            if task_info['media_urls']:
                await self._update_task_status(task_uuid, 'processing', 50, '正在上传媒体文件')
                # 处理媒体文件上传逻辑
                pass
            
            # 发布内容
            await self._update_task_status(task_uuid, 'processing', 75, '正在发布内容')
            
            publish_result = await adapter.publish_content(
                content=task_info['content'],
                title=task_info['title'],
                media_urls=task_info['media_urls'],
                config=task_info['publish_config']
            )
            
            # 更新任务状态为成功
            await self._update_task_status(
                task_uuid, 'published', 100, '发布成功',
                platform_post_id=publish_result.get('post_id'),
                published_url=publish_result.get('url')
            )
            
            # 更新账号使用配额
            await self._update_account_quota(task_info['account_id'])
            
        except Exception as e:
            # 更新任务状态为失败
            await self._update_task_status(
                task_uuid, 'failed', 0, f'发布失败: {str(e)}'
            )
            raise
    
    async def _update_task_status(
        self, 
        task_uuid: str, 
        status: str, 
        progress: int, 
        message: str,
        platform_post_id: Optional[str] = None,
        published_url: Optional[str] = None
    ):
        """
        更新任务状态
        """
        # 更新Redis缓存
        cache_key = f"task_status:{task_uuid}"
        await self.publishing_service.redis.hset(cache_key, mapping={
            'status': status,
            'progress': progress,
            'message': message,
            'updated_at': datetime.utcnow().isoformat()
        })
        await self.publishing_service.redis.expire(cache_key, 3600)  # 1小时过期
        
        # 更新数据库
        async with self.publishing_service.db_pool.acquire() as conn:
            update_fields = [
                "status = $2",
                "updated_at = CURRENT_TIMESTAMP"
            ]
            params = [task_uuid, status]
            
            if status == 'failed':
                update_fields.append("error_message = $3")
                update_fields.append("retry_count = retry_count + 1")
                params.append(message)
            elif status == 'published':
                update_fields.extend([
                    "published_at = CURRENT_TIMESTAMP",
                    "platform_post_id = $3",
                    "published_url = $4"
                ])
                params.extend([platform_post_id, published_url])
            
            query = f"""
            UPDATE publishing_tasks 
            SET {', '.join(update_fields)}
            WHERE task_uuid = $1
            """
            
            await conn.execute(query, *params)
    
    async def _get_task_info(self, task_uuid: str) -> Dict[str, Any]:
        """
        获取任务信息
        """
        async with self.publishing_service.db_pool.acquire() as conn:
            query = """
            SELECT 
                pt.*,
                pp.platform_name,
                pa.auth_credentials,
                pa.account_name
            FROM publishing_tasks pt
            JOIN publishing_platforms pp ON pt.platform_id = pp.id
            JOIN publishing_accounts pa ON pt.account_id = pa.id
            WHERE pt.task_uuid = $1
            """
            
            task = await conn.fetchrow(query, task_uuid)
            
            if not task:
                raise Exception(f"Task not found: {task_uuid}")
            
            import json
            return {
                'task_uuid': task_uuid,
                'platform_name': task['platform_name'],
                'account_id': task['account_id'],
                'account_name': task['account_name'],
                'title': task['title'],
                'content': task['content'],
                'media_urls': json.loads(task['media_urls']) if task['media_urls'] else None,
                'publish_config': json.loads(task['publish_config']) if task['publish_config'] else None,
                'auth_credentials': json.loads(task['auth_credentials']) if task['auth_credentials'] else None
            }
    
    async def _update_account_quota(self, account_id: int):
        """
        更新账号使用配额
        """
        # 更新Redis配额缓存
        quota_key = f"account_quota:{account_id}"
        await self.publishing_service.redis.hincrby(quota_key, 'used_today', 1)
        
        # 更新数据库
        async with self.publishing_service.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE publishing_accounts SET used_quota = used_quota + 1, last_used_at = CURRENT_TIMESTAMP WHERE id = $1",
                account_id
            )
```

## API设计

### 核心API接口

```yaml
openapi: 3.0.0
info:
  title: 文本发送服务 API
  version: 1.0.0
  description: 多平台内容发布服务接口

paths:
  /api/v1/publish:
    post:
      summary: 创建发布任务
      tags: [Publishing]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [content, platforms]
              properties:
                content:
                  type: string
                  description: 发布内容
                  maxLength: 10000
                platforms:
                  type: array
                  items:
                    type: string
                    enum: [weibo, wechat, douyin, toutiao, baijiahao]
                  description: 目标平台列表
                title:
                  type: string
                  description: 标题
                  maxLength: 500
                media_urls:
                  type: array
                  items:
                    type: string
                    format: uri
                  description: 媒体文件URL列表
                scheduled_at:
                  type: string
                  format: date-time
                  description: 定时发布时间
                publish_config:
                  type: object
                  description: 发布配置
                  properties:
                    tags:
                      type: array
                      items:
                        type: string
                    category:
                      type: string
                    visibility:
                      type: string
                      enum: [public, private, friends]
      responses:
        '200':
          description: 任务创建成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  task_uuids:
                    type: array
                    items:
                      type: string
                      format: uuid
                  message:
                    type: string
        '400':
          description: 请求参数错误
        '500':
          description: 服务器内部错误

  /api/v1/tasks/{task_uuid}/status:
    get:
      summary: 获取任务状态
      tags: [Publishing]
      parameters:
        - name: task_uuid
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: 任务状态信息
          content:
            application/json:
              schema:
                type: object
                properties:
                  task_uuid:
                    type: string
                    format: uuid
                  platform:
                    type: string
                  account:
                    type: string
                  status:
                    type: string
                    enum: [pending, processing, published, failed, cancelled]
                  progress:
                    type: integer
                    minimum: 0
                    maximum: 100
                  message:
                    type: string
                  published_url:
                    type: string
                    format: uri
                  error_message:
                    type: string
                  created_at:
                    type: string
                    format: date-time
                  updated_at:
                    type: string
                    format: date-time
        '404':
          description: 任务不存在

  /api/v1/tasks/{task_uuid}/cancel:
    post:
      summary: 取消发布任务
      tags: [Publishing]
      parameters:
        - name: task_uuid
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: 取消成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  message:
                    type: string
        '400':
          description: 任务无法取消
        '404':
          description: 任务不存在

  /api/v1/tasks:
    get:
      summary: 获取任务列表
      tags: [Publishing]
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, processing, published, failed, cancelled]
        - name: platform
          in: query
          schema:
            type: string
        - name: start_date
          in: query
          schema:
            type: string
            format: date
        - name: end_date
          in: query
          schema:
            type: string
            format: date
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: page_size
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: 任务列表
          content:
            application/json:
              schema:
                type: object
                properties:
                  tasks:
                    type: array
                    items:
                      $ref: '#/components/schemas/PublishingTask'
                  pagination:
                    type: object
                    properties:
                      page:
                        type: integer
                      page_size:
                        type: integer
                      total:
                        type: integer
                      pages:
                        type: integer

  /api/v1/platforms:
    get:
      summary: 获取支持的平台列表
      tags: [Platforms]
      responses:
        '200':
          description: 平台列表
          content:
            application/json:
              schema:
                type: object
                properties:
                  platforms:
                    type: array
                    items:
                      type: object
                      properties:
                        name:
                          type: string
                        display_name:
                          type: string
                        type:
                          type: string
                        is_active:
                          type: boolean
                        rate_limit:
                          type: integer
                        supported_features:
                          type: array
                          items:
                            type: string

  /api/v1/accounts:
    get:
      summary: 获取发布账号列表
      tags: [Accounts]
      parameters:
        - name: platform
          in: query
          schema:
            type: string
      responses:
        '200':
          description: 账号列表
          content:
            application/json:
              schema:
                type: object
                properties:
                  accounts:
                    type: array
                    items:
                      type: object
                      properties:
                        id:
                          type: integer
                        platform:
                          type: string
                        account_name:
                          type: string
                        status:
                          type: string
                        daily_quota:
                          type: integer
                        used_quota:
                          type: integer
                        last_used_at:
                          type: string
                          format: date-time
    
    post:
      summary: 添加发布账号
      tags: [Accounts]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [platform, account_name, auth_credentials]
              properties:
                platform:
                  type: string
                account_name:
                  type: string
                account_identifier:
                  type: string
                auth_credentials:
                  type: object
                daily_quota:
                  type: integer
                  default: 50
      responses:
        '201':
          description: 账号添加成功
        '400':
          description: 请求参数错误

  /api/v1/statistics:
    get:
      summary: 获取发布统计数据
      tags: [Statistics]
      parameters:
        - name: start_date
          in: query
          required: true
          schema:
            type: string
            format: date
        - name: end_date
          in: query
          required: true
          schema:
            type: string
            format: date
        - name: platform
          in: query
          schema:
            type: string
        - name: group_by
          in: query
          schema:
            type: string
            enum: [day, week, month]
            default: day
      responses:
        '200':
          description: 统计数据
          content:
            application/json:
              schema:
                type: object
                properties:
                  period:
                    type: object
                    properties:
                      start_date:
                        type: string
                        format: date
                      end_date:
                        type: string
                        format: date
                  platforms:
                    type: array
                    items:
                      type: object
                      properties:
                        platform:
                          type: string
                        total_tasks:
                          type: integer
                        successful_tasks:
                          type: integer
                        failed_tasks:
                          type: integer
                        success_rate:
                          type: number
                        avg_publish_time:
                          type: number

components:
  schemas:
    PublishingTask:
      type: object
      properties:
        task_uuid:
          type: string
          format: uuid
        platform:
          type: string
        account:
          type: string
        title:
          type: string
        content:
          type: string
        status:
          type: string
        scheduled_at:
          type: string
          format: date-time
        published_at:
          type: string
          format: date-time
        published_url:
          type: string
          format: uri
        error_message:
          type: string
        retry_count:
          type: integer
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
```

## Vue3 前端组件

### 1. 发布管理主页面

```vue
<template>
  <div class="publishing-management">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>内容发布管理</span>
          <el-button type="primary" @click="showPublishDialog = true">
            <el-icon><Plus /></el-icon>
            新建发布任务
          </el-button>
        </div>
      </template>
      
      <!-- 统计卡片 -->
      <div class="stats-cards">
        <el-row :gutter="20">
          <el-col :span="6">
            <el-card class="stat-card">
              <div class="stat-content">
                <div class="stat-value">{{ stats.total_tasks }}</div>
                <div class="stat-label">总任务数</div>
              </div>
              <el-icon class="stat-icon"><Document /></el-icon>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat-card">
              <div class="stat-content">
                <div class="stat-value">{{ stats.successful_tasks }}</div>
                <div class="stat-label">成功发布</div>
              </div>
              <el-icon class="stat-icon success"><Check /></el-icon>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat-card">
              <div class="stat-content">
                <div class="stat-value">{{ stats.failed_tasks }}</div>
                <div class="stat-label">发布失败</div>
              </div>
              <el-icon class="stat-icon error"><Close /></el-icon>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card class="stat-card">
              <div class="stat-content">
                <div class="stat-value">{{ stats.success_rate }}%</div>
                <div class="stat-label">成功率</div>
              </div>
              <el-icon class="stat-icon"><TrendCharts /></el-icon>
            </el-card>
          </el-col>
        </el-row>
      </div>
      
      <!-- 筛选条件 -->
      <div class="filter-section">
        <el-form :model="filters" inline>
          <el-form-item label="状态">
            <el-select v-model="filters.status" placeholder="选择状态" clearable>
              <el-option label="待处理" value="pending" />
              <el-option label="处理中" value="processing" />
              <el-option label="已发布" value="published" />
              <el-option label="失败" value="failed" />
              <el-option label="已取消" value="cancelled" />
            </el-select>
          </el-form-item>
          <el-form-item label="平台">
            <el-select v-model="filters.platform" placeholder="选择平台" clearable>
              <el-option 
                v-for="platform in platforms" 
                :key="platform.name"
                :label="platform.display_name" 
                :value="platform.name" 
              />
            </el-select>
          </el-form-item>
          <el-form-item label="日期范围">
            <el-date-picker
              v-model="filters.dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              format="YYYY-MM-DD"
              value-format="YYYY-MM-DD"
            />
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="loadTasks">查询</el-button>
            <el-button @click="resetFilters">重置</el-button>
          </el-form-item>
        </el-form>
      </div>
      
      <!-- 任务列表 -->
      <el-table :data="tasks" v-loading="loading" stripe>
        <el-table-column prop="task_uuid" label="任务ID" width="120">
          <template #default="{ row }">
            <el-tooltip :content="row.task_uuid" placement="top">
              <span class="task-uuid">{{ row.task_uuid.substring(0, 8) }}...</span>
            </el-tooltip>
          </template>
        </el-table-column>
        <el-table-column prop="platform" label="平台" width="100">
          <template #default="{ row }">
            <el-tag :type="getPlatformTagType(row.platform)">{{ getPlatformDisplayName(row.platform) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="title" label="标题" width="200" show-overflow-tooltip />
        <el-table-column prop="content" label="内容" width="300" show-overflow-tooltip />
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusTagType(row.status)">{{ getStatusDisplayName(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="scheduled_at" label="计划时间" width="160">
          <template #default="{ row }">
            <span v-if="row.scheduled_at">{{ formatDateTime(row.scheduled_at) }}</span>
            <span v-else class="text-muted">立即发布</span>
          </template>
        </el-table-column>
        <el-table-column prop="published_at" label="发布时间" width="160">
          <template #default="{ row }">
            <span v-if="row.published_at">{{ formatDateTime(row.published_at) }}</span>
            <span v-else class="text-muted">-</span>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="160">
          <template #default="{ row }">
            {{ formatDateTime(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button 
              size="small" 
              @click="viewTaskDetail(row)"
            >
              详情
            </el-button>
            <el-button 
              v-if="row.published_url"
              size="small" 
              type="success"
              @click="openPublishedUrl(row.published_url)"
            >
              查看
            </el-button>
            <el-button 
              v-if="canCancelTask(row.status)"
              size="small" 
              type="danger"
              @click="cancelTask(row.task_uuid)"
            >
              取消
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <!-- 分页 -->
      <div class="pagination-wrapper">
        <el-pagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="pagination.total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="loadTasks"
          @current-change="loadTasks"
        />
      </div>
    </el-card>
    
    <!-- 新建发布任务对话框 -->
    <PublishDialog 
      v-model="showPublishDialog" 
      :platforms="platforms"
      @success="handlePublishSuccess"
    />
    
    <!-- 任务详情对话框 -->
    <TaskDetailDialog 
      v-model="showDetailDialog" 
      :task="selectedTask"
      @refresh="loadTasks"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Document, Check, Close, TrendCharts } from '@element-plus/icons-vue'
import { publishingApi } from '@/api/publishing'
import PublishDialog from './components/PublishDialog.vue'
import TaskDetailDialog from './components/TaskDetailDialog.vue'

// 响应式数据
const loading = ref(false)
const showPublishDialog = ref(false)
const showDetailDialog = ref(false)
const selectedTask = ref(null)

const tasks = ref([])
const platforms = ref([])
const stats = ref({
  total_tasks: 0,
  successful_tasks: 0,
  failed_tasks: 0,
  success_rate: 0
})

const filters = reactive({
  status: '',
  platform: '',
  dateRange: []
})

const pagination = reactive({
  page: 1,
  pageSize: 20,
  total: 0
})

/**
 * 加载任务列表
 */
const loadTasks = async () => {
  loading.value = true
  try {
    const params = {
      page: pagination.page,
      page_size: pagination.pageSize,
      status: filters.status || undefined,
      platform: filters.platform || undefined,
      start_date: filters.dateRange[0] || undefined,
      end_date: filters.dateRange[1] || undefined
    }
    
    const response = await publishingApi.getTasks(params)
    tasks.value = response.data.tasks
    pagination.total = response.data.pagination.total
  } catch (error) {
    console.error('加载任务列表失败:', error)
    ElMessage.error('加载任务列表失败')
  } finally {
    loading.value = false
  }
}

/**
 * 加载平台列表
 */
const loadPlatforms = async () => {
  try {
    const response = await publishingApi.getPlatforms()
    platforms.value = response.data.platforms
  } catch (error) {
    console.error('加载平台列表失败:', error)
  }
}

/**
 * 加载统计数据
 */
const loadStats = async () => {
  try {
    const endDate = new Date()
    const startDate = new Date()
    startDate.setDate(startDate.getDate() - 30) // 最近30天
    
    const response = await publishingApi.getStatistics({
      start_date: startDate.toISOString().split('T')[0],
      end_date: endDate.toISOString().split('T')[0]
    })
    
    // 计算总体统计
    const platforms = response.data.platforms
    stats.value = {
      total_tasks: platforms.reduce((sum, p) => sum + p.total_tasks, 0),
      successful_tasks: platforms.reduce((sum, p) => sum + p.successful_tasks, 0),
      failed_tasks: platforms.reduce((sum, p) => sum + p.failed_tasks, 0),
      success_rate: platforms.length > 0 
        ? Math.round(platforms.reduce((sum, p) => sum + p.success_rate, 0) / platforms.length)
        : 0
    }
  } catch (error) {
    console.error('加载统计数据失败:', error)
  }
}

/**
 * 重置筛选条件
 */
const resetFilters = () => {
  filters.status = ''
  filters.platform = ''
  filters.dateRange = []
  pagination.page = 1
  loadTasks()
}

/**
 * 查看任务详情
 */
const viewTaskDetail = (task) => {
  selectedTask.value = task
  showDetailDialog.value = true
}

/**
 * 取消任务
 */
const cancelTask = async (taskUuid: string) => {
  try {
    await ElMessageBox.confirm(
      '确定要取消这个发布任务吗？',
      '确认取消',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await publishingApi.cancelTask(taskUuid)
    ElMessage.success('任务已取消')
    loadTasks()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('取消任务失败:', error)
      ElMessage.error('取消任务失败')
    }
  }
}

/**
 * 打开发布链接
 */
const openPublishedUrl = (url: string) => {
  window.open(url, '_blank')
}

/**
 * 处理发布成功
 */
const handlePublishSuccess = () => {
  loadTasks()
  loadStats()
}

/**
 * 获取平台标签类型
 */
const getPlatformTagType = (platform: string) => {
  const typeMap = {
    'weibo': 'danger',
    'wechat': 'success',
    'douyin': '',
    'toutiao': 'warning',
    'baijiahao': 'info'
  }
  return typeMap[platform] || ''
}

/**
 * 获取平台显示名称
 */
const getPlatformDisplayName = (platform: string) => {
  const nameMap = {
    'weibo': '微博',
    'wechat': '微信',
    'douyin': '抖音',
    'toutiao': '头条',
    'baijiahao': '百家号'
  }
  return nameMap[platform] || platform
}

/**
 * 获取状态标签类型
 */
const getStatusTagType = (status: string) => {
  const typeMap = {
    'pending': 'info',
    'processing': 'warning',
    'published': 'success',
    'failed': 'danger',
    'cancelled': ''
  }
  return typeMap[status] || ''
}

/**
 * 获取状态显示名称
 */
const getStatusDisplayName = (status: string) => {
  const nameMap = {
    'pending': '待处理',
    'processing': '处理中',
    'published': '已发布',
    'failed': '失败',
    'cancelled': '已取消'
  }
  return nameMap[status] || status
}

/**
 * 判断是否可以取消任务
 */
const canCancelTask = (status: string) => {
  return ['pending', 'processing'].includes(status)
}

/**
 * 格式化日期时间
 */
const formatDateTime = (dateTime: string) => {
  return new Date(dateTime).toLocaleString('zh-CN')
}

// 生命周期
onMounted(() => {
  loadTasks()
  loadPlatforms()
  loadStats()
})
</script>

<style scoped>
.publishing-management {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stats-cards {
  margin-bottom: 20px;
}

.stat-card {
  position: relative;
  overflow: hidden;
}

.stat-content {
  padding: 20px;
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #909399;
}

.stat-icon {
  position: absolute;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 40px;
  color: #e4e7ed;
}

.stat-icon.success {
  color: #67c23a;
}

.stat-icon.error {
  color: #f56c6c;
}

.filter-section {
  margin-bottom: 20px;
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.task-uuid {
  font-family: monospace;
  font-size: 12px;
  color: #606266;
}

.text-muted {
  color: #c0c4cc;
}

.pagination-wrapper {
  margin-top: 20px;
  text-align: right;
}
</style>
```

## 验收标准

### 功能验收标准

1. **发布任务创建**
   - ✅ 支持创建单平台和多平台发布任务
   - ✅ 支持文本内容和媒体文件发布
   - ✅ 支持立即发布和定时发布
   - ✅ 支持发布配置（标签、分类、可见性）
   - ✅ 表单验证完整，错误提示清晰

2. **任务状态管理**
   - ✅ 实时显示任务执行状态和进度
   - ✅ 支持任务取消功能
   - ✅ 支持任务重试机制
   - ✅ 详细的错误信息记录和展示

3. **平台适配**
   - ✅ 支持微博、微信、抖音、头条、百家号等主流平台
   - ✅ 统一的平台接口适配
   - ✅ 平台特定的发布配置支持
   - ✅ 平台认证和授权管理

4. **账号管理**
   - ✅ 支持多账号管理
   - ✅ 账号状态监控
   - ✅ 发布配额管理
   - ✅ 账号负载均衡

5. **统计分析**
   - ✅ 发布成功率统计
   - ✅ 平台发布数据分析
   - ✅ 发布时间分析
   - ✅ 错误统计和分析

### 性能验收标准

1. **响应时间**
   - 任务创建响应时间 < 2秒
   - 状态查询响应时间 < 1秒
   - 任务列表加载时间 < 3秒
   - 统计数据查询时间 < 5秒

2. **并发处理**
   - 支持同时处理100个发布任务
   - 支持1000个并发用户访问
   - 任务队列处理能力 > 500任务/分钟

3. **可用性**
   - 系统可用性 > 99.5%
   - 平台适配器故障不影响其他平台
   - 支持服务降级和熔断

4. **扩展性**
   - 支持水平扩展
   - 支持新平台快速接入
   - 支持配置热更新

### 安全验收标准

1. **认证授权**
   - 所有API接口需要身份认证
   - 基于角色的访问控制
   - 平台账号凭据加密存储
   - 支持OAuth2.0认证流程

2. **数据安全**
   - 敏感数据加密传输
   - 数据库连接加密
   - 定期清理过期认证信息
   - 审计日志记录

3. **接口安全**
   - API限流保护
   - 输入参数验证
   - SQL注入防护
   - XSS攻击防护

## 业务价值

### 直接价值

1. **效率提升**
   - 减少90%的手动发布时间
   - 支持批量发布，提高内容分发效率
   - 自动化重试机制，减少人工干预

2. **成本降低**
   - 减少人力成本60%
   - 降低发布错误率80%
   - 统一管理降低维护成本

3. **覆盖扩大**
   - 支持多平台同步发布
   - 提高内容触达率
   - 扩大品牌影响力

### 间接价值

1. **数据洞察**
   - 发布效果数据分析
   - 平台表现对比
   - 优化发布策略

2. **风险控制**
   - 统一的内容审核
   - 发布状态监控
   - 异常情况预警

3. **业务扩展**
   - 支持新平台快速接入
   - 为内容营销提供基础设施
   - 支持个性化发布策略

## 依赖关系

### 技术依赖

1. **基础设施**
   - PostgreSQL 数据库
   - Redis 缓存服务
   - RabbitMQ 消息队列
   - Docker 容器化环境

2. **外部服务**
   - 各平台开放API
   - 文件存储服务
   - 监控告警服务
   - 日志收集服务

3. **内部服务**
   - 用户认证服务
   - 文件上传服务
   - 内容管理服务
   - 通知服务

### 业务依赖

1. **平台账号**
   - 各平台开发者账号
   - API调用权限
   - 发布账号授权

2. **内容来源**
   - 内容管理系统
   - 媒体资源库
   - 内容审核流程

3. **运营支持**
   - 发布策略制定
   - 平台规则了解
   - 异常处理流程

### 环境依赖

1. **开发环境**
   - Python 3.11+
   - Node.js 18+
   - Docker & Docker Compose
   - Git版本控制

2. **测试环境**
   - 测试数据库
   - 模拟平台API
   - 性能测试工具
   - 自动化测试框架

3. **生产环境**
   - Kubernetes集群
   - 负载均衡器
   - 监控系统
   - 备份系统

## 风险评估

### 技术风险

1. **平台API变更** (高风险)
   - **风险描述**: 第三方平台API接口变更或限制
   - **影响程度**: 可能导致特定平台发布功能失效
   - **缓解措施**: 
     - 建立API版本管理机制
     - 实现适配器模式，便于快速调整
     - 与平台方建立沟通渠道
     - 实现降级方案

2. **并发性能瓶颈** (中风险)
   - **风险描述**: 高并发场景下系统性能下降
   - **影响程度**: 影响用户体验和发布效率
   - **缓解措施**:
     - 实现任务队列和负载均衡
     - 数据库连接池优化
     - 缓存策略优化
     - 性能监控和预警

3. **数据一致性** (中风险)
   - **风险描述**: 分布式环境下数据同步问题
   - **影响程度**: 可能导致任务状态不一致
   - **缓解措施**:
     - 实现分布式锁机制
     - 事务管理优化
     - 数据校验和修复机制
     - 定期数据一致性检查

### 业务风险

1. **平台政策变化** (高风险)
   - **风险描述**: 平台发布政策或规则变更
   - **影响程度**: 可能导致账号被封或发布受限
   - **缓解措施**:
     - 密切关注平台政策动态
     - 实现内容合规检查
     - 建立多账号备份机制
     - 制定应急响应预案

2. **内容合规风险** (中风险)
   - **风险描述**: 发布内容违反平台规定
   - **影响程度**: 账号被限制或内容被删除
   - **缓解措施**:
     - 集成内容审核机制
     - 建立敏感词过滤
     - 人工审核流程
     - 发布前预检查

3. **账号安全风险** (中风险)
   - **风险描述**: 账号被盗用或异常登录
   - **影响程度**: 可能导致恶意发布或数据泄露
   - **缓解措施**:
     - 强化账号认证机制
     - 异常行为监控
     - 定期密码更新
     - 访问权限控制

### 运营风险

1. **人员依赖** (低风险)
   - **风险描述**: 关键人员离职影响系统维护
   - **影响程度**: 可能影响系统稳定性和功能迭代
   - **缓解措施**:
     - 完善技术文档
     - 知识转移和培训
     - 代码规范化
     - 建立技术团队

2. **服务依赖** (中风险)
   - **风险描述**: 依赖的外部服务不稳定
   - **影响程度**: 可能影响系统整体可用性
   - **缓解措施**:
     - 服务降级机制
     - 多服务商备选方案
     - 服务监控和告警
     - SLA协议保障

## 开发任务分解

### 后端开发任务

#### 阶段一：基础架构 (预计 2 周)

1. **数据库设计与实现** (3天)
   - 设计数据库表结构
   - 创建数据库迁移脚本
   - 实现数据模型类
   - 创建索引和约束

2. **核心服务框架** (4天)
   - 搭建FastAPI项目结构
   - 配置数据库连接池
   - 集成Redis缓存
   - 配置Celery任务队列
   - 实现基础中间件

3. **平台适配器基类** (3天)
   - 设计适配器接口
   - 实现基础适配器类
   - 创建认证机制
   - 实现错误处理

#### 阶段二：核心功能 (预计 3 周)

1. **发布服务实现** (5天)
   - 实现PublishingService类
   - 任务创建和管理
   - 账号选择算法
   - 状态更新机制

2. **任务处理器** (4天)
   - 实现TaskProcessor类
   - 异步任务执行
   - 重试机制
   - 错误处理和日志

3. **平台适配器实现** (6天)
   - 微博适配器 (1天)
   - 微信适配器 (1天)
   - 抖音适配器 (1天)
   - 头条适配器 (1天)
   - 百家号适配器 (1天)
   - 适配器测试 (1天)

#### 阶段三：API接口 (预计 1.5 周)

1. **发布管理API** (3天)
   - 创建发布任务接口
   - 任务状态查询接口
   - 任务取消接口
   - 任务列表接口

2. **平台管理API** (2天)
   - 平台列表接口
   - 账号管理接口
   - 统计数据接口

3. **API文档和测试** (2天)
   - OpenAPI文档生成
   - 接口单元测试
   - 集成测试

#### 阶段四：监控和优化 (预计 1 周)

1. **监控集成** (2天)
   - Prometheus指标收集
   - 日志结构化
   - 健康检查接口

2. **性能优化** (2天)
   - 数据库查询优化
   - 缓存策略优化
   - 并发处理优化

3. **安全加固** (1天)
   - 输入验证加强
   - 权限控制完善
   - 敏感数据加密

### 前端开发任务

#### 阶段一：项目搭建 (预计 0.5 周)

1. **项目初始化** (1天)
   - Vue3 + TypeScript项目搭建
   - 配置构建工具和代码规范
   - 集成Element Plus

2. **基础配置** (1天)
   - 路由配置
   - 状态管理配置
   - API客户端配置
   - 样式主题配置

3. **公共组件** (1天)
   - 布局组件
   - 通用工具函数
   - 类型定义

#### 阶段二：核心页面 (预计 2 周)

1. **发布管理主页面** (4天)
   - 任务列表展示
   - 筛选和搜索功能
   - 统计卡片组件
   - 分页组件

2. **发布任务创建** (3天)
   - 发布表单设计
   - 文件上传组件
   - 表单验证
   - 平台选择组件

3. **任务详情页面** (3天)
   - 任务信息展示
   - 实时状态更新
   - 操作按钮
   - 错误信息展示

#### 阶段三：功能完善 (预计 1 周)

1. **交互优化** (2天)
   - 加载状态处理
   - 错误提示优化
   - 用户体验改进

2. **响应式适配** (1天)
   - 移动端适配
   - 不同屏幕尺寸优化

3. **性能优化** (1天)
   - 组件懒加载
   - 图片优化
   - 打包优化

4. **测试和调试** (1天)
   - 单元测试
   - 端到端测试
   - 浏览器兼容性测试

### 测试任务

#### 单元测试 (预计 1 周)

1. **后端单元测试** (3天)
   - 服务类测试
   - API接口测试
   - 数据库操作测试
   - 适配器测试

2. **前端单元测试** (2天)
   - 组件测试
   - 工具函数测试
   - 状态管理测试

#### 集成测试 (预计 1 周)

1. **API集成测试** (2天)
   - 接口联调测试
   - 数据流测试
   - 错误场景测试

2. **端到端测试** (2天)
   - 用户流程测试
   - 跨浏览器测试
   - 性能测试

3. **平台集成测试** (1天)
   - 真实平台发布测试
   - 认证流程测试
   - 异常处理测试

### 部署任务

#### 环境搭建 (预计 0.5 周)

1. **Docker化** (1天)
   - 编写Dockerfile
   - Docker Compose配置
   - 环境变量配置

2. **CI/CD配置** (1天)
   - 构建流水线
   - 自动化测试
   - 部署脚本

3. **监控配置** (1天)
   - 日志收集配置
   - 指标监控配置
   - 告警规则配置

## 时间估算

### 总体时间安排

- **后端开发**: 7.5 周
- **前端开发**: 3.5 周
- **测试**: 2 周
- **部署**: 0.5 周
- **总计**: 8 周 (考虑并行开发)

### 人力资源需求

- **后端开发工程师**: 2人
- **前端开发工程师**: 1人
- **测试工程师**: 1人
- **DevOps工程师**: 0.5人
- **产品经理**: 0.5人

### 关键里程碑

1. **第2周末**: 基础架构完成
2. **第4周末**: 核心功能完成
3. **第6周末**: API和前端页面完成
4. **第7周末**: 测试完成
5. **第8周末**: 部署上线

## 成功指标

### 技术指标

1. **功能完整性**: 100%需求实现
2. **代码覆盖率**: >80%
3. **API响应时间**: <2秒
4. **系统可用性**: >99.5%
5. **并发处理能力**: >100任务/分钟

### 业务指标

1. **发布成功率**: >95%
2. **用户满意度**: >4.5/5
3. **发布效率提升**: >90%
4. **错误率降低**: >80%
5. **平台覆盖率**: 100%主流平台

### 运营指标

1. **系统稳定性**: 无重大故障
2. **响应时间**: 问题解决<4小时
3. **文档完整性**: 100%功能有文档
4. **团队技能**: 100%成员掌握核心技术
5. **知识传承**: 完整的技术文档和培训材料

### 2. 发布任务创建对话框

```vue
<template>
  <el-dialog 
    v-model="visible" 
    title="新建发布任务" 
    width="800px"
    :before-close="handleClose"
  >
    <el-form 
      ref="formRef" 
      :model="form" 
      :rules="rules" 
      label-width="100px"
    >
      <el-form-item label="标题" prop="title">
        <el-input 
          v-model="form.title" 
          placeholder="请输入标题（可选）"
          maxlength="500"
          show-word-limit
        />
      </el-form-item>
      
      <el-form-item label="内容" prop="content">
        <el-input 
          v-model="form.content" 
          type="textarea" 
          placeholder="请输入发布内容"
          :rows="8"
          maxlength="10000"
          show-word-limit
        />
      </el-form-item>
      
      <el-form-item label="目标平台" prop="platforms">
        <el-checkbox-group v-model="form.platforms">
          <el-checkbox 
            v-for="platform in platforms" 
            :key="platform.name"
            :label="platform.name"
            :disabled="!platform.is_active"
          >
            {{ platform.display_name }}
            <el-tag v-if="!platform.is_active" size="small" type="info">不可用</el-tag>
          </el-checkbox>
        </el-checkbox-group>
      </el-form-item>
      
      <el-form-item label="媒体文件">
        <el-upload
          v-model:file-list="fileList"
          :action="uploadUrl"
          :headers="uploadHeaders"
          :on-success="handleUploadSuccess"
          :on-error="handleUploadError"
          :before-upload="beforeUpload"
          multiple
          list-type="picture-card"
          accept="image/*,video/*"
        >
          <el-icon><Plus /></el-icon>
        </el-upload>
        <div class="upload-tip">
          支持图片和视频文件，单个文件不超过50MB
        </div>
      </el-form-item>
      
      <el-form-item label="发布时间">
        <el-radio-group v-model="publishType">
          <el-radio label="immediate">立即发布</el-radio>
          <el-radio label="scheduled">定时发布</el-radio>
        </el-radio-group>
        <el-date-picker
          v-if="publishType === 'scheduled'"
          v-model="form.scheduled_at"
          type="datetime"
          placeholder="选择发布时间"
          format="YYYY-MM-DD HH:mm:ss"
          value-format="YYYY-MM-DDTHH:mm:ss"
          :disabled-date="disabledDate"
          :disabled-hours="disabledHours"
          style="margin-left: 20px; width: 200px;"
        />
      </el-form-item>
      
      <el-form-item label="发布配置">
        <el-collapse>
          <el-collapse-item title="高级配置" name="advanced">
            <el-form-item label="标签">
              <el-tag
                v-for="tag in form.publish_config.tags"
                :key="tag"
                closable
                @close="removeTag(tag)"
                style="margin-right: 8px;"
              >
                {{ tag }}
              </el-tag>
              <el-input
                v-if="inputVisible"
                ref="inputRef"
                v-model="inputValue"
                size="small"
                style="width: 100px;"
                @keyup.enter="handleInputConfirm"
                @blur="handleInputConfirm"
              />
              <el-button v-else size="small" @click="showInput">+ 添加标签</el-button>
            </el-form-item>
            
            <el-form-item label="分类">
              <el-select v-model="form.publish_config.category" placeholder="选择分类" clearable>
                <el-option label="科技" value="tech" />
                <el-option label="娱乐" value="entertainment" />
                <el-option label="体育" value="sports" />
                <el-option label="财经" value="finance" />
                <el-option label="教育" value="education" />
                <el-option label="生活" value="lifestyle" />
              </el-select>
            </el-form-item>
            
            <el-form-item label="可见性">
              <el-radio-group v-model="form.publish_config.visibility">
                <el-radio label="public">公开</el-radio>
                <el-radio label="friends">仅好友</el-radio>
                <el-radio label="private">私密</el-radio>
              </el-radio-group>
            </el-form-item>
          </el-collapse-item>
        </el-collapse>
      </el-form-item>
    </el-form>
    
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="handleClose">取消</el-button>
        <el-button type="primary" @click="handleSubmit" :loading="submitting">
          {{ publishType === 'immediate' ? '立即发布' : '创建定时任务' }}
        </el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, computed, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { publishingApi } from '@/api/publishing'

interface Props {
  modelValue: boolean
  platforms: Array<any>
}

interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'success'): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

// 响应式数据
const formRef = ref()
const inputRef = ref()
const submitting = ref(false)
const publishType = ref('immediate')
const inputVisible = ref(false)
const inputValue = ref('')
const fileList = ref([])

const form = reactive({
  title: '',
  content: '',
  platforms: [],
  scheduled_at: '',
  media_urls: [],
  publish_config: {
    tags: [],
    category: '',
    visibility: 'public'
  }
})

// 表单验证规则
const rules = {
  content: [
    { required: true, message: '请输入发布内容', trigger: 'blur' },
    { min: 1, max: 10000, message: '内容长度在 1 到 10000 个字符', trigger: 'blur' }
  ],
  platforms: [
    { required: true, message: '请选择至少一个发布平台', trigger: 'change' }
  ]
}

// 计算属性
const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const uploadUrl = computed(() => {
  return '/api/v1/upload/media'
})

const uploadHeaders = computed(() => {
  return {
    'Authorization': `Bearer ${localStorage.getItem('token')}`
  }
})

/**
 * 禁用过去的日期
 */
const disabledDate = (time: Date) => {
  return time.getTime() < Date.now() - 8.64e7 // 禁用昨天之前的日期
}

/**
 * 禁用过去的小时
 */
const disabledHours = () => {
  const hours = []
  const now = new Date()
  const currentHour = now.getHours()
  
  for (let i = 0; i < currentHour; i++) {
    hours.push(i)
  }
  
  return hours
}

/**
 * 显示标签输入框
 */
const showInput = () => {
  inputVisible.value = true
  nextTick(() => {
    inputRef.value?.focus()
  })
}

/**
 * 确认添加标签
 */
const handleInputConfirm = () => {
  if (inputValue.value && !form.publish_config.tags.includes(inputValue.value)) {
    form.publish_config.tags.push(inputValue.value)
  }
  inputVisible.value = false
  inputValue.value = ''
}

/**
 * 移除标签
 */
const removeTag = (tag: string) => {
  const index = form.publish_config.tags.indexOf(tag)
  if (index > -1) {
    form.publish_config.tags.splice(index, 1)
  }
}

/**
 * 上传前检查
 */
const beforeUpload = (file: File) => {
  const isValidType = file.type.startsWith('image/') || file.type.startsWith('video/')
  const isLt50M = file.size / 1024 / 1024 < 50
  
  if (!isValidType) {
    ElMessage.error('只能上传图片或视频文件!')
    return false
  }
  if (!isLt50M) {
    ElMessage.error('文件大小不能超过 50MB!')
    return false
  }
  return true
}

/**
 * 上传成功处理
 */
const handleUploadSuccess = (response: any) => {
  if (response.success) {
    form.media_urls.push(response.data.url)
    ElMessage.success('文件上传成功')
  } else {
    ElMessage.error('文件上传失败')
  }
}

/**
 * 上传失败处理
 */
const handleUploadError = () => {
  ElMessage.error('文件上传失败')
}

/**
 * 提交表单
 */
const handleSubmit = async () => {
  try {
    await formRef.value?.validate()
    
    submitting.value = true
    
    const submitData = {
      title: form.title || undefined,
      content: form.content,
      platforms: form.platforms,
      media_urls: form.media_urls.length > 0 ? form.media_urls : undefined,
      scheduled_at: publishType.value === 'scheduled' ? form.scheduled_at : undefined,
      publish_config: {
        tags: form.publish_config.tags.length > 0 ? form.publish_config.tags : undefined,
        category: form.publish_config.category || undefined,
        visibility: form.publish_config.visibility
      }
    }
    
    const response = await publishingApi.createPublishTask(submitData)
    
    ElMessage.success(`成功创建 ${response.data.task_uuids.length} 个发布任务`)
    emit('success')
    handleClose()
    
  } catch (error) {
    console.error('创建发布任务失败:', error)
    ElMessage.error('创建发布任务失败')
  } finally {
    submitting.value = false
  }
}

/**
 * 关闭对话框
 */
const handleClose = () => {
  // 重置表单
  formRef.value?.resetFields()
  form.title = ''
  form.content = ''
  form.platforms = []
  form.scheduled_at = ''
  form.media_urls = []
  form.publish_config = {
    tags: [],
    category: '',
    visibility: 'public'
  }
  publishType.value = 'immediate'
  fileList.value = []
  
  visible.value = false
}
</script>

<style scoped>
.upload-tip {
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
}

.dialog-footer {
  text-align: right;
}
</style>
```

### 3. 任务详情对话框

```vue
<template>
  <el-dialog 
    v-model="visible" 
    title="任务详情" 
    width="700px"
    :before-close="handleClose"
  >
    <div v-if="task" class="task-detail">
      <!-- 基本信息 -->
      <el-descriptions title="基本信息" :column="2" border>
        <el-descriptions-item label="任务ID">
          <el-tag type="info">{{ task.task_uuid }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="getStatusTagType(task.status)">{{ getStatusDisplayName(task.status) }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="平台">
          <el-tag :type="getPlatformTagType(task.platform)">{{ getPlatformDisplayName(task.platform) }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="账号">
          {{ task.account }}
        </el-descriptions-item>
        <el-descriptions-item label="创建时间">
          {{ formatDateTime(task.created_at) }}
        </el-descriptions-item>
        <el-descriptions-item label="更新时间">
          {{ formatDateTime(task.updated_at) }}
        </el-descriptions-item>
        <el-descriptions-item label="计划时间" v-if="task.scheduled_at">
          {{ formatDateTime(task.scheduled_at) }}
        </el-descriptions-item>
        <el-descriptions-item label="发布时间" v-if="task.published_at">
          {{ formatDateTime(task.published_at) }}
        </el-descriptions-item>
      </el-descriptions>
      
      <!-- 内容信息 -->
      <el-descriptions title="内容信息" :column="1" border style="margin-top: 20px;">
        <el-descriptions-item label="标题" v-if="task.title">
          {{ task.title }}
        </el-descriptions-item>
        <el-descriptions-item label="内容">
          <div class="content-preview">{{ task.content }}</div>
        </el-descriptions-item>
      </el-descriptions>
      
      <!-- 发布结果 -->
      <el-descriptions 
        v-if="task.status === 'published' || task.status === 'failed'" 
        title="发布结果" 
        :column="1" 
        border 
        style="margin-top: 20px;"
      >
        <el-descriptions-item label="发布链接" v-if="task.published_url">
          <el-link :href="task.published_url" target="_blank" type="primary">
            {{ task.published_url }}
          </el-link>
        </el-descriptions-item>
        <el-descriptions-item label="错误信息" v-if="task.error_message">
          <el-alert :title="task.error_message" type="error" :closable="false" />
        </el-descriptions-item>
        <el-descriptions-item label="重试次数" v-if="task.retry_count > 0">
          {{ task.retry_count }}
        </el-descriptions-item>
      </el-descriptions>
      
      <!-- 实时状态 -->
      <div v-if="task.status === 'processing'" class="real-time-status">
        <h4>实时状态</h4>
        <el-progress 
          :percentage="realTimeStatus.progress" 
          :status="realTimeStatus.progress === 100 ? 'success' : undefined"
        />
        <p class="status-message">{{ realTimeStatus.message }}</p>
        <p class="last-update">最后更新: {{ formatDateTime(realTimeStatus.updated_at) }}</p>
      </div>
    </div>
    
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="handleClose">关闭</el-button>
        <el-button 
          v-if="task && task.published_url" 
          type="success" 
          @click="openPublishedUrl"
        >
          查看发布内容
        </el-button>
        <el-button 
          v-if="task && canCancelTask(task.status)" 
          type="danger" 
          @click="cancelTask"
        >
          取消任务
        </el-button>
        <el-button type="primary" @click="refreshStatus">
          刷新状态
        </el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { publishingApi } from '@/api/publishing'

interface Props {
  modelValue: boolean
  task: any
}

interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'refresh'): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

// 响应式数据
const realTimeStatus = ref({
  progress: 0,
  message: '',
  updated_at: ''
})

let statusTimer: NodeJS.Timeout | null = null

// 计算属性
const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

/**
 * 刷新任务状态
 */
const refreshStatus = async () => {
  if (!props.task?.task_uuid) return
  
  try {
    const response = await publishingApi.getTaskStatus(props.task.task_uuid)
    
    // 更新任务信息
    Object.assign(props.task, response.data)
    
    // 更新实时状态
    if (response.data.status === 'processing') {
      realTimeStatus.value = {
        progress: response.data.progress || 0,
        message: response.data.message || '',
        updated_at: response.data.updated_at || ''
      }
    }
    
    emit('refresh')
  } catch (error) {
    console.error('刷新状态失败:', error)
    ElMessage.error('刷新状态失败')
  }
}

/**
 * 取消任务
 */
const cancelTask = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要取消这个发布任务吗？',
      '确认取消',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await publishingApi.cancelTask(props.task.task_uuid)
    ElMessage.success('任务已取消')
    
    // 刷新状态
    await refreshStatus()
    
  } catch (error) {
    if (error !== 'cancel') {
      console.error('取消任务失败:', error)
      ElMessage.error('取消任务失败')
    }
  }
}

/**
 * 打开发布链接
 */
const openPublishedUrl = () => {
  if (props.task?.published_url) {
    window.open(props.task.published_url, '_blank')
  }
}

/**
 * 关闭对话框
 */
const handleClose = () => {
  // 清除定时器
  if (statusTimer) {
    clearInterval(statusTimer)
    statusTimer = null
  }
  
  visible.value = false
}

/**
 * 获取平台标签类型
 */
const getPlatformTagType = (platform: string) => {
  const typeMap = {
    'weibo': 'danger',
    'wechat': 'success',
    'douyin': '',
    'toutiao': 'warning',
    'baijiahao': 'info'
  }
  return typeMap[platform] || ''
}

/**
 * 获取平台显示名称
 */
const getPlatformDisplayName = (platform: string) => {
  const nameMap = {
    'weibo': '微博',
    'wechat': '微信',
    'douyin': '抖音',
    'toutiao': '头条',
    'baijiahao': '百家号'
  }
  return nameMap[platform] || platform
}

/**
 * 获取状态标签类型
 */
const getStatusTagType = (status: string) => {
  const typeMap = {
    'pending': 'info',
    'processing': 'warning',
    'published': 'success',
    'failed': 'danger',
    'cancelled': ''
  }
  return typeMap[status] || ''
}

/**
 * 获取状态显示名称
 */
const getStatusDisplayName = (status: string) => {
  const nameMap = {
    'pending': '待处理',
    'processing': '处理中',
    'published': '已发布',
    'failed': '失败',
    'cancelled': '已取消'
  }
  return nameMap[status] || status
}

/**
 * 判断是否可以取消任务
 */
const canCancelTask = (status: string) => {
  return ['pending', 'processing'].includes(status)
}

/**
 * 格式化日期时间
 */
const formatDateTime = (dateTime: string) => {
  return new Date(dateTime).toLocaleString('zh-CN')
}

// 监听对话框显示状态
watch(visible, (newVal) => {
  if (newVal && props.task) {
    // 立即刷新一次状态
    refreshStatus()
    
    // 如果任务正在处理中，启动定时刷新
    if (props.task.status === 'processing') {
      statusTimer = setInterval(() => {
        refreshStatus()
      }, 3000) // 每3秒刷新一次
    }
  } else {
    // 清除定时器
    if (statusTimer) {
      clearInterval(statusTimer)
      statusTimer = null
    }
  }
})

// 组件卸载时清除定时器
onUnmounted(() => {
  if (statusTimer) {
    clearInterval(statusTimer)
  }
})
</script>

<style scoped>
.task-detail {
  max-height: 600px;
  overflow-y: auto;
}

.content-preview {
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  background-color: #f5f7fa;
  padding: 12px;
  border-radius: 4px;
  font-size: 14px;
  line-height: 1.5;
}

.real-time-status {
  margin-top: 20px;
  padding: 16px;
  background-color: #f0f9ff;
  border-radius: 4px;
  border-left: 4px solid #409eff;
}

.real-time-status h4 {
  margin: 0 0 12px 0;
  color: #303133;
}

.status-message {
  margin: 8px 0;
  color: #606266;
  font-size: 14px;
}

.last-update {
  margin: 4px 0 0 0;
  color: #909399;
  font-size: 12px;
}

.dialog-footer {
  text-align: right;
}
</style>
```