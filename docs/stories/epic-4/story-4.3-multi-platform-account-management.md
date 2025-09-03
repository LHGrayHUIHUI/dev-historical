# Story 4.3: 多平台账号管理服务

## 用户故事描述

**作为** 内容运营人员  
**我希望** 能够统一管理多个社交媒体平台的账号信息  
**以便** 高效地进行跨平台内容发布和账号运营

## 功能概述

多平台账号管理服务是一个集中化的账号管理系统，支持主流社交媒体平台的账号接入、认证、管理和监控。该服务提供统一的账号管理界面，支持批量操作、权限控制、账号状态监控等功能，为内容发布和运营提供基础支撑。

## 核心技术栈

### 后端技术栈

- **Web框架**: FastAPI 0.104+
- **异步处理**: Celery 5.3+, Redis 7.0+
- **任务调度**: APScheduler 3.10+
- **HTTP客户端**: httpx 0.25+, aiohttp 3.9+
- **OAuth认证**: authlib 1.2+, python-jose 3.3+
- **加密解密**: cryptography 41.0+, PyJWT 2.8+
- **数据库**: PostgreSQL 15+, SQLAlchemy 2.0+
- **缓存**: Redis 7.0+, redis-py 5.0+
- **消息队列**: RabbitMQ 3.12+, Celery 5.3+
- **监控**: Prometheus 0.17+, Grafana 10.0+
- **日志**: structlog 23.1+, ELK Stack 8.0+
- **容器化**: Docker 24.0+, Kubernetes 1.28+

### 前端技术栈

- **框架**: Vue 3.3+, TypeScript 5.0+
- **状态管理**: Pinia 2.1+
- **路由**: Vue Router 4.2+
- **UI组件**: Element Plus 2.4+
- **组合式API**: @vue/composition-api 1.7+
- **HTTP客户端**: Axios 1.5+
- **日期处理**: Day.js 1.11+
- **工具库**: Lodash 4.17+
- **图表**: ECharts 5.4+, Chart.js 4.4+
- **构建工具**: Vite 4.4+
- **代码规范**: ESLint 8.50+, Prettier 3.0+
- **测试**: Vitest 0.34+

## 数据模型设计

### PostgreSQL 数据模型

```sql
-- 平台配置表
CREATE TABLE platforms (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200) NOT NULL,
    platform_type VARCHAR(50) NOT NULL, -- social_media, blog, news, etc.
    api_base_url VARCHAR(500),
    oauth_config JSONB,
    rate_limits JSONB,
    features JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 账号表
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    platform_id INTEGER REFERENCES platforms(id),
    user_id INTEGER NOT NULL,
    account_name VARCHAR(200) NOT NULL,
    account_id VARCHAR(200),
    display_name VARCHAR(200),
    avatar_url VARCHAR(500),
    bio TEXT,
    follower_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,
    post_count INTEGER DEFAULT 0,
    verification_status VARCHAR(50) DEFAULT 'unverified',
    account_type VARCHAR(50) DEFAULT 'personal', -- personal, business, creator
    status VARCHAR(50) DEFAULT 'active', -- active, suspended, expired, error
    last_sync_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(platform_id, account_id)
);

-- 账号认证信息表
CREATE TABLE account_credentials (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE,
    access_token TEXT,
    refresh_token TEXT,
    token_type VARCHAR(50) DEFAULT 'Bearer',
    expires_at TIMESTAMP,
    scope TEXT,
    encrypted_data JSONB, -- 加密存储的敏感信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 账号权限表
CREATE TABLE account_permissions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL,
    permission_type VARCHAR(100) NOT NULL, -- read, write, admin, publish
    granted_by INTEGER,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    UNIQUE(account_id, user_id, permission_type)
);

-- 账号同步日志表
CREATE TABLE account_sync_logs (
    id SERIAL PRIMARY KEY,
    account_id INTEGER REFERENCES accounts(id),
    sync_type VARCHAR(50) NOT NULL, -- profile, stats, posts, followers
    status VARCHAR(50) NOT NULL, -- success, failed, partial
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    error_message TEXT,
    sync_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 平台API使用统计表
CREATE TABLE api_usage_stats (
    id SERIAL PRIMARY KEY,
    platform_id INTEGER REFERENCES platforms(id),
    account_id INTEGER REFERENCES accounts(id),
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    request_date DATE NOT NULL,
    request_hour INTEGER NOT NULL,
    request_count INTEGER DEFAULT 1,
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(platform_id, account_id, endpoint, method, request_date, request_hour)
);

-- 创建索引
CREATE INDEX idx_accounts_platform_user ON accounts(platform_id, user_id);
CREATE INDEX idx_accounts_status ON accounts(status);
CREATE INDEX idx_account_credentials_expires ON account_credentials(expires_at);
CREATE INDEX idx_account_permissions_user ON account_permissions(user_id, is_active);
CREATE INDEX idx_sync_logs_account_time ON account_sync_logs(account_id, created_at);
CREATE INDEX idx_api_stats_platform_date ON api_usage_stats(platform_id, request_date);
```

### Redis 数据模型

```python
# 账号缓存
account_cache_key = "account:{account_id}"
account_data = {
    "id": 123,
    "platform_name": "weibo",
    "account_name": "@example_user",
    "status": "active",
    "last_sync": "2024-01-15T10:30:00Z",
    "stats": {
        "followers": 10000,
        "following": 500,
        "posts": 1200
    }
}

# 平台配置缓存
platform_config_key = "platform:config:{platform_name}"
platform_config = {
    "api_base_url": "https://api.weibo.com",
    "rate_limits": {
        "requests_per_hour": 1000,
        "requests_per_day": 10000
    },
    "oauth_config": {
        "client_id": "encrypted_client_id",
        "scopes": ["read", "write", "publish"]
    }
}

# API速率限制
rate_limit_key = "rate_limit:{platform}:{account_id}:{endpoint}"
rate_limit_data = {
    "requests_made": 45,
    "limit": 100,
    "reset_time": 1705320000,
    "window_start": 1705316400
}

# 账号同步状态
sync_status_key = "sync_status:{account_id}"
sync_status = {
    "is_syncing": True,
    "sync_type": "profile",
    "started_at": "2024-01-15T10:30:00Z",
    "progress": 65,
    "last_error": None
}

# 用户账号列表缓存
user_accounts_key = "user_accounts:{user_id}"
user_accounts = [
    {
        "account_id": 123,
        "platform": "weibo",
        "account_name": "@example_user",
        "status": "active",
        "permissions": ["read", "write", "publish"]
    }
]
```

## 服务架构设计

### 核心服务类

```python
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet
import httpx
import asyncio
from authlib.integrations.httpx_client import AsyncOAuth2Client

class AccountManagementService:
    """
    多平台账号管理服务
    负责账号的添加、认证、管理和同步
    """
    
    def __init__(self, db: Session, redis_client, encryption_key: str):
        self.db = db
        self.redis = redis_client
        self.cipher = Fernet(encryption_key.encode())
        self.platform_adapters = {}
        self._init_platform_adapters()
    
    def _init_platform_adapters(self):
        """初始化平台适配器"""
        from .adapters import (
            WeiboAdapter, WeChatAdapter, DouyinAdapter,
            ToutiaoAdapter, BaijiahaoAdapter
        )
        
        self.platform_adapters = {
            'weibo': WeiboAdapter(),
            'wechat': WeChatAdapter(),
            'douyin': DouyinAdapter(),
            'toutiao': ToutiaoAdapter(),
            'baijiahao': BaijiahaoAdapter()
        }
    
    async def add_account(self, user_id: int, platform_name: str, 
                         auth_code: str) -> Dict[str, Any]:
        """
        添加新账号
        
        Args:
            user_id: 用户ID
            platform_name: 平台名称
            auth_code: 授权码
            
        Returns:
            账号信息字典
        """
        try:
            # 获取平台配置
            platform = await self._get_platform_config(platform_name)
            if not platform:
                raise ValueError(f"不支持的平台: {platform_name}")
            
            # 获取平台适配器
            adapter = self.platform_adapters.get(platform_name)
            if not adapter:
                raise ValueError(f"平台适配器未找到: {platform_name}")
            
            # 通过授权码获取访问令牌
            token_data = await adapter.exchange_code_for_token(
                auth_code, platform['oauth_config']
            )
            
            # 获取账号信息
            account_info = await adapter.get_account_info(token_data['access_token'])
            
            # 检查账号是否已存在
            existing_account = await self._get_account_by_platform_id(
                platform['id'], account_info['id']
            )
            
            if existing_account:
                # 更新现有账号的令牌
                await self._update_account_credentials(
                    existing_account['id'], token_data
                )
                account_id = existing_account['id']
            else:
                # 创建新账号
                account_id = await self._create_account(
                    user_id, platform['id'], account_info
                )
                
                # 保存认证信息
                await self._save_account_credentials(account_id, token_data)
                
                # 设置默认权限
                await self._set_account_permissions(
                    account_id, user_id, ['read', 'write', 'publish']
                )
            
            # 同步账号数据
            await self.sync_account_data(account_id)
            
            # 缓存账号信息
            await self._cache_account_info(account_id)
            
            return {
                'account_id': account_id,
                'platform': platform_name,
                'account_name': account_info.get('username'),
                'display_name': account_info.get('display_name'),
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"添加账号失败: {str(e)}")
            raise
    
    async def get_user_accounts(self, user_id: int, 
                               platform_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取用户的所有账号
        
        Args:
            user_id: 用户ID
            platform_filter: 平台过滤器
            
        Returns:
            账号列表
        """
        try:
            # 尝试从缓存获取
            cache_key = f"user_accounts:{user_id}"
            cached_accounts = await self.redis.get(cache_key)
            
            if cached_accounts:
                accounts = json.loads(cached_accounts)
                if platform_filter:
                    accounts = [acc for acc in accounts if acc['platform'] == platform_filter]
                return accounts
            
            # 从数据库查询
            query = """
                SELECT a.id, a.account_name, a.display_name, a.avatar_url,
                       a.follower_count, a.following_count, a.post_count,
                       a.status, a.last_sync_at, p.name as platform_name,
                       p.display_name as platform_display_name
                FROM accounts a
                JOIN platforms p ON a.platform_id = p.id
                JOIN account_permissions ap ON a.id = ap.account_id
                WHERE ap.user_id = %s AND ap.is_active = true
            """
            
            params = [user_id]
            if platform_filter:
                query += " AND p.name = %s"
                params.append(platform_filter)
            
            query += " ORDER BY a.created_at DESC"
            
            result = await self.db.execute(query, params)
            accounts = []
            
            for row in result.fetchall():
                account = {
                    'id': row.id,
                    'platform': row.platform_name,
                    'platform_display_name': row.platform_display_name,
                    'account_name': row.account_name,
                    'display_name': row.display_name,
                    'avatar_url': row.avatar_url,
                    'stats': {
                        'followers': row.follower_count,
                        'following': row.following_count,
                        'posts': row.post_count
                    },
                    'status': row.status,
                    'last_sync': row.last_sync_at.isoformat() if row.last_sync_at else None
                }
                accounts.append(account)
            
            # 缓存结果
            await self.redis.setex(
                cache_key, 300, json.dumps(accounts, default=str)
            )
            
            return accounts
            
        except Exception as e:
            logger.error(f"获取用户账号失败: {str(e)}")
            raise
    
    async def sync_account_data(self, account_id: int, 
                               sync_types: List[str] = None) -> Dict[str, Any]:
        """
        同步账号数据
        
        Args:
            account_id: 账号ID
            sync_types: 同步类型列表 ['profile', 'stats', 'posts']
            
        Returns:
            同步结果
        """
        if sync_types is None:
            sync_types = ['profile', 'stats']
        
        try:
            # 检查是否正在同步
            sync_status_key = f"sync_status:{account_id}"
            is_syncing = await self.redis.get(sync_status_key)
            
            if is_syncing:
                return {'status': 'already_syncing', 'message': '账号正在同步中'}
            
            # 设置同步状态
            await self.redis.setex(sync_status_key, 3600, json.dumps({
                'is_syncing': True,
                'started_at': datetime.utcnow().isoformat(),
                'sync_types': sync_types
            }))
            
            # 获取账号信息
            account = await self._get_account_by_id(account_id)
            if not account:
                raise ValueError(f"账号不存在: {account_id}")
            
            # 获取访问令牌
            credentials = await self._get_account_credentials(account_id)
            if not credentials or await self._is_token_expired(credentials):
                # 尝试刷新令牌
                if credentials and credentials.get('refresh_token'):
                    await self._refresh_access_token(account_id)
                    credentials = await self._get_account_credentials(account_id)
                else:
                    raise ValueError("访问令牌已过期，需要重新授权")
            
            # 获取平台适配器
            platform_name = account['platform_name']
            adapter = self.platform_adapters.get(platform_name)
            
            sync_results = {}
            
            # 同步账号资料
            if 'profile' in sync_types:
                try:
                    profile_data = await adapter.get_account_info(
                        credentials['access_token']
                    )
                    await self._update_account_profile(account_id, profile_data)
                    sync_results['profile'] = 'success'
                except Exception as e:
                    sync_results['profile'] = f'failed: {str(e)}'
            
            # 同步统计数据
            if 'stats' in sync_types:
                try:
                    stats_data = await adapter.get_account_stats(
                        credentials['access_token']
                    )
                    await self._update_account_stats(account_id, stats_data)
                    sync_results['stats'] = 'success'
                except Exception as e:
                    sync_results['stats'] = f'failed: {str(e)}'
            
            # 同步帖子数据
            if 'posts' in sync_types:
                try:
                    posts_data = await adapter.get_recent_posts(
                        credentials['access_token'], limit=50
                    )
                    await self._update_account_posts(account_id, posts_data)
                    sync_results['posts'] = 'success'
                except Exception as e:
                    sync_results['posts'] = f'failed: {str(e)}'
            
            # 更新同步时间
            await self._update_account_sync_time(account_id)
            
            # 记录同步日志
            await self._log_sync_result(account_id, sync_types, sync_results)
            
            # 清除同步状态
            await self.redis.delete(sync_status_key)
            
            # 更新缓存
            await self._cache_account_info(account_id)
            
            return {
                'status': 'completed',
                'account_id': account_id,
                'sync_results': sync_results,
                'synced_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            # 清除同步状态
            await self.redis.delete(sync_status_key)
            logger.error(f"同步账号数据失败: {str(e)}")
            raise
    
    async def remove_account(self, account_id: int, user_id: int) -> bool:
        """
        移除账号
        
        Args:
            account_id: 账号ID
            user_id: 用户ID
            
        Returns:
            是否成功移除
        """
        try:
            # 检查权限
            has_permission = await self._check_account_permission(
                account_id, user_id, 'admin'
            )
            
            if not has_permission:
                raise PermissionError("没有权限移除此账号")
            
            # 撤销平台授权
            account = await self._get_account_by_id(account_id)
            credentials = await self._get_account_credentials(account_id)
            
            if credentials and account:
                platform_name = account['platform_name']
                adapter = self.platform_adapters.get(platform_name)
                
                if adapter:
                    try:
                        await adapter.revoke_token(credentials['access_token'])
                    except Exception as e:
                        logger.warning(f"撤销令牌失败: {str(e)}")
            
            # 删除数据库记录
            await self.db.execute(
                "DELETE FROM account_credentials WHERE account_id = %s",
                [account_id]
            )
            
            await self.db.execute(
                "DELETE FROM account_permissions WHERE account_id = %s",
                [account_id]
            )
            
            await self.db.execute(
                "UPDATE accounts SET status = 'removed' WHERE id = %s",
                [account_id]
            )
            
            await self.db.commit()
            
            # 清除缓存
            await self._clear_account_cache(account_id, user_id)
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"移除账号失败: {str(e)}")
            raise
    
    async def get_account_status(self, account_id: int) -> Dict[str, Any]:
        """
        获取账号状态
        
        Args:
            account_id: 账号ID
            
        Returns:
            账号状态信息
        """
        try:
            # 获取基本信息
            account = await self._get_account_by_id(account_id)
            if not account:
                raise ValueError(f"账号不存在: {account_id}")
            
            # 获取同步状态
            sync_status_key = f"sync_status:{account_id}"
            sync_status = await self.redis.get(sync_status_key)
            
            if sync_status:
                sync_info = json.loads(sync_status)
            else:
                sync_info = {'is_syncing': False}
            
            # 获取令牌状态
            credentials = await self._get_account_credentials(account_id)
            token_status = 'valid'
            
            if not credentials:
                token_status = 'missing'
            elif await self._is_token_expired(credentials):
                token_status = 'expired'
            
            # 获取最近的同步日志
            recent_sync = await self._get_recent_sync_log(account_id)
            
            return {
                'account_id': account_id,
                'platform': account['platform_name'],
                'account_name': account['account_name'],
                'status': account['status'],
                'token_status': token_status,
                'sync_status': sync_info,
                'last_sync': account.get('last_sync_at'),
                'recent_sync_log': recent_sync
            }
            
        except Exception as e:
            logger.error(f"获取账号状态失败: {str(e)}")
            raise

class PlatformAdapter:
    """
    平台适配器基类
    定义各平台需要实现的接口
    """
    
    async def exchange_code_for_token(self, auth_code: str, 
                                     oauth_config: Dict[str, Any]) -> Dict[str, Any]:
        """交换授权码获取访问令牌"""
        raise NotImplementedError
    
    async def refresh_access_token(self, refresh_token: str, 
                                  oauth_config: Dict[str, Any]) -> Dict[str, Any]:
        """刷新访问令牌"""
        raise NotImplementedError
    
    async def get_account_info(self, access_token: str) -> Dict[str, Any]:
        """获取账号信息"""
        raise NotImplementedError
    
    async def get_account_stats(self, access_token: str) -> Dict[str, Any]:
        """获取账号统计数据"""
        raise NotImplementedError
    
    async def get_recent_posts(self, access_token: str, 
                              limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的帖子"""
        raise NotImplementedError
    
    async def revoke_token(self, access_token: str) -> bool:
        """撤销访问令牌"""
        raise NotImplementedError

class WeiboAdapter(PlatformAdapter):
    """
    微博平台适配器
    """
    
    def __init__(self):
        self.api_base_url = "https://api.weibo.com"
        self.client = httpx.AsyncClient()
    
    async def exchange_code_for_token(self, auth_code: str, 
                                     oauth_config: Dict[str, Any]) -> Dict[str, Any]:
        """微博OAuth2令牌交换"""
        try:
            token_url = f"{self.api_base_url}/oauth2/access_token"
            
            data = {
                'client_id': oauth_config['client_id'],
                'client_secret': oauth_config['client_secret'],
                'grant_type': 'authorization_code',
                'code': auth_code,
                'redirect_uri': oauth_config['redirect_uri']
            }
            
            response = await self.client.post(token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            
            return {
                'access_token': token_data['access_token'],
                'refresh_token': token_data.get('refresh_token'),
                'expires_in': token_data.get('expires_in', 3600),
                'scope': token_data.get('scope'),
                'token_type': 'Bearer'
            }
            
        except Exception as e:
            logger.error(f"微博令牌交换失败: {str(e)}")
            raise
    
    async def get_account_info(self, access_token: str) -> Dict[str, Any]:
        """获取微博账号信息"""
        try:
            url = f"{self.api_base_url}/2/users/show.json"
            
            headers = {'Authorization': f'Bearer {access_token}'}
            params = {'access_token': access_token}
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            user_data = response.json()
            
            return {
                'id': str(user_data['id']),
                'username': user_data['screen_name'],
                'display_name': user_data['name'],
                'avatar_url': user_data.get('profile_image_url'),
                'bio': user_data.get('description'),
                'follower_count': user_data.get('followers_count', 0),
                'following_count': user_data.get('friends_count', 0),
                'post_count': user_data.get('statuses_count', 0),
                'verification_status': 'verified' if user_data.get('verified') else 'unverified'
            }
            
        except Exception as e:
            logger.error(f"获取微博账号信息失败: {str(e)}")
            raise

class RateLimitManager:
    """
    API速率限制管理器
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(self, platform: str, account_id: int, 
                              endpoint: str) -> Dict[str, Any]:
        """
        检查API速率限制
        
        Args:
            platform: 平台名称
            account_id: 账号ID
            endpoint: API端点
            
        Returns:
            速率限制状态
        """
        try:
            rate_limit_key = f"rate_limit:{platform}:{account_id}:{endpoint}"
            
            # 获取当前限制状态
            limit_data = await self.redis.get(rate_limit_key)
            
            if limit_data:
                limit_info = json.loads(limit_data)
                
                # 检查是否超出限制
                if limit_info['requests_made'] >= limit_info['limit']:
                    reset_time = datetime.fromtimestamp(limit_info['reset_time'])
                    
                    if datetime.utcnow() < reset_time:
                        return {
                            'allowed': False,
                            'limit': limit_info['limit'],
                            'remaining': 0,
                            'reset_time': reset_time.isoformat(),
                            'retry_after': (reset_time - datetime.utcnow()).total_seconds()
                        }
                    else:
                        # 重置计数器
                        await self._reset_rate_limit(rate_limit_key, platform, endpoint)
                        limit_info = await self._get_default_limit(platform, endpoint)
                
                return {
                    'allowed': True,
                    'limit': limit_info['limit'],
                    'remaining': limit_info['limit'] - limit_info['requests_made'],
                    'reset_time': datetime.fromtimestamp(limit_info['reset_time']).isoformat()
                }
            else:
                # 初始化速率限制
                limit_info = await self._get_default_limit(platform, endpoint)
                await self.redis.setex(
                    rate_limit_key, 3600, json.dumps(limit_info)
                )
                
                return {
                    'allowed': True,
                    'limit': limit_info['limit'],
                    'remaining': limit_info['limit'],
                    'reset_time': datetime.fromtimestamp(limit_info['reset_time']).isoformat()
                }
                
        except Exception as e:
            logger.error(f"检查速率限制失败: {str(e)}")
            # 默认允许请求
            return {'allowed': True, 'limit': 100, 'remaining': 100}
    
    async def record_api_request(self, platform: str, account_id: int, 
                                endpoint: str, status_code: int, 
                                response_time: int) -> None:
        """
        记录API请求
        
        Args:
            platform: 平台名称
            account_id: 账号ID
            endpoint: API端点
            status_code: 响应状态码
            response_time: 响应时间(毫秒)
        """
        try:
            # 更新速率限制计数
            rate_limit_key = f"rate_limit:{platform}:{account_id}:{endpoint}"
            
            limit_data = await self.redis.get(rate_limit_key)
            if limit_data:
                limit_info = json.loads(limit_data)
                limit_info['requests_made'] += 1
                
                if status_code >= 400:
                    limit_info['error_count'] = limit_info.get('error_count', 0) + 1
                
                await self.redis.setex(
                    rate_limit_key, 3600, json.dumps(limit_info)
                )
            
            # 记录API使用统计
            await self._record_api_stats(
                platform, account_id, endpoint, status_code, response_time
            )
            
        except Exception as e:
            logger.error(f"记录API请求失败: {str(e)}")
```

## API设计

### 1. 账号管理API

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/accounts", tags=["账号管理"])

class AddAccountRequest(BaseModel):
    platform: str
    auth_code: str
    redirect_uri: Optional[str] = None

class AccountResponse(BaseModel):
    id: int
    platform: str
    platform_display_name: str
    account_name: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    stats: Dict[str, int]
    status: str
    last_sync: Optional[str]
    permissions: List[str]

@router.post("/add", response_model=Dict[str, Any])
async def add_account(
    request: AddAccountRequest,
    current_user: User = Depends(get_current_user),
    account_service: AccountManagementService = Depends(get_account_service)
):
    """
    添加新账号
    
    - **platform**: 平台名称 (weibo, wechat, douyin, toutiao, baijiahao)
    - **auth_code**: OAuth授权码
    - **redirect_uri**: 重定向URI (可选)
    """
    try:
        result = await account_service.add_account(
            user_id=current_user.id,
            platform_name=request.platform,
            auth_code=request.auth_code
        )
        
        return {
            "success": True,
            "message": "账号添加成功",
            "data": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="添加账号失败")

@router.get("/list", response_model=List[AccountResponse])
async def get_accounts(
    platform: Optional[str] = Query(None, description="平台过滤器"),
    status: Optional[str] = Query(None, description="状态过滤器"),
    current_user: User = Depends(get_current_user),
    account_service: AccountManagementService = Depends(get_account_service)
):
    """
    获取用户账号列表
    
    - **platform**: 平台过滤器 (可选)
    - **status**: 状态过滤器 (可选)
    """
    try:
        accounts = await account_service.get_user_accounts(
            user_id=current_user.id,
            platform_filter=platform
        )
        
        if status:
            accounts = [acc for acc in accounts if acc['status'] == status]
        
        return accounts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="获取账号列表失败")

@router.post("/{account_id}/sync")
async def sync_account(
    account_id: int,
    sync_types: List[str] = Query(["profile", "stats"], description="同步类型"),
    current_user: User = Depends(get_current_user),
    account_service: AccountManagementService = Depends(get_account_service)
):
    """
    同步账号数据
    
    - **account_id**: 账号ID
    - **sync_types**: 同步类型列表 (profile, stats, posts)
    """
    try:
        # 检查权限
        has_permission = await account_service._check_account_permission(
            account_id, current_user.id, 'read'
        )
        
        if not has_permission:
            raise HTTPException(status_code=403, detail="没有权限访问此账号")
        
        result = await account_service.sync_account_data(
            account_id=account_id,
            sync_types=sync_types
        )
        
        return {
            "success": True,
            "message": "同步完成",
            "data": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="同步失败")

@router.get("/{account_id}/status")
async def get_account_status(
    account_id: int,
    current_user: User = Depends(get_current_user),
    account_service: AccountManagementService = Depends(get_account_service)
):
    """
    获取账号状态
    
    - **account_id**: 账号ID
    """
    try:
        # 检查权限
        has_permission = await account_service._check_account_permission(
            account_id, current_user.id, 'read'
        )
        
        if not has_permission:
            raise HTTPException(status_code=403, detail="没有权限访问此账号")
        
        status = await account_service.get_account_status(account_id)
        
        return {
            "success": True,
            "data": status
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="获取状态失败")

@router.delete("/{account_id}")
async def remove_account(
    account_id: int,
    current_user: User = Depends(get_current_user),
    account_service: AccountManagementService = Depends(get_account_service)
):
    """
    移除账号
    
    - **account_id**: 账号ID
    """
    try:
        success = await account_service.remove_account(
            account_id=account_id,
            user_id=current_user.id
        )
        
        if success:
            return {
                "success": True,
                "message": "账号移除成功"
            }
        else:
            raise HTTPException(status_code=400, detail="移除账号失败")
            
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="移除账号失败")
```

### 2. 平台管理API

```python
@router.get("/platforms", response_model=List[Dict[str, Any]])
async def get_platforms(
    active_only: bool = Query(True, description="仅显示活跃平台"),
    platform_service: PlatformService = Depends(get_platform_service)
):
    """
    获取支持的平台列表
    
    - **active_only**: 仅显示活跃平台
    """
    try:
        platforms = await platform_service.get_platforms(active_only=active_only)
        return platforms
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="获取平台列表失败")

@router.get("/platforms/{platform_name}/oauth-url")
async def get_oauth_url(
    platform_name: str,
    redirect_uri: str = Query(..., description="重定向URI"),
    state: Optional[str] = Query(None, description="状态参数"),
    platform_service: PlatformService = Depends(get_platform_service)
):
    """
    获取OAuth授权URL
    
    - **platform_name**: 平台名称
    - **redirect_uri**: 重定向URI
    - **state**: 状态参数 (可选)
    """
    try:
        oauth_url = await platform_service.get_oauth_url(
            platform_name=platform_name,
            redirect_uri=redirect_uri,
            state=state
        )
        
        return {
            "success": True,
            "data": {
                "oauth_url": oauth_url,
                "platform": platform_name
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="获取授权URL失败")
```

### 3. 权限管理API

```python
class PermissionRequest(BaseModel):
    user_id: int
    permissions: List[str]
    expires_at: Optional[str] = None

@router.post("/{account_id}/permissions")
async def grant_permissions(
    account_id: int,
    request: PermissionRequest,
    current_user: User = Depends(get_current_user),
    account_service: AccountManagementService = Depends(get_account_service)
):
    """
    授予账号权限
    
    - **account_id**: 账号ID
    - **user_id**: 用户ID
    - **permissions**: 权限列表 (read, write, publish, admin)
    - **expires_at**: 过期时间 (可选)
    """
    try:
        # 检查是否有管理权限
        has_admin = await account_service._check_account_permission(
            account_id, current_user.id, 'admin'
        )
        
        if not has_admin:
            raise HTTPException(status_code=403, detail="没有管理权限")
        
        success = await account_service._set_account_permissions(
            account_id=account_id,
            user_id=request.user_id,
            permissions=request.permissions,
            expires_at=request.expires_at
        )
        
        return {
            "success": True,
            "message": "权限授予成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="授予权限失败")

@router.get("/{account_id}/permissions")
async def get_account_permissions(
    account_id: int,
    current_user: User = Depends(get_current_user),
    account_service: AccountManagementService = Depends(get_account_service)
):
    """
    获取账号权限列表
    
    - **account_id**: 账号ID
    """
    try:
        # 检查权限
        has_permission = await account_service._check_account_permission(
            account_id, current_user.id, 'read'
        )
        
        if not has_permission:
            raise HTTPException(status_code=403, detail="没有权限访问此账号")
        
        permissions = await account_service._get_account_permissions(account_id)
        
        return {
            "success": True,
            "data": permissions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="获取权限失败")
```

## Vue3 前端组件

### 1. 账号管理主页面

```vue
<template>
  <div class="account-management">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">账号管理</h1>
        <p class="page-description">管理您的社交媒体账号，统一进行内容发布和运营</p>
      </div>
      <div class="header-actions">
        <el-button type="primary" @click="showAddAccountDialog = true">
          <el-icon><Plus /></el-icon>
          添加账号
        </el-button>
        <el-button @click="refreshAccounts">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>
    </div>
    
    <!-- 统计卡片 -->
    <div class="stats-cards">
      <el-row :gutter="20">
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-icon total">
                <el-icon><User /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ stats.totalAccounts }}</div>
                <div class="stat-label">总账号数</div>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-icon active">
                <el-icon><CircleCheck /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ stats.activeAccounts }}</div>
                <div class="stat-label">活跃账号</div>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-icon platforms">
                <el-icon><Grid /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ stats.platformCount }}</div>
                <div class="stat-label">接入平台</div>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-icon syncing">
                <el-icon><Loading /></el-icon>
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ stats.syncingAccounts }}</div>
                <div class="stat-label">同步中</div>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 筛选和搜索 -->
    <div class="filter-section">
      <el-card>
        <el-row :gutter="20" align="middle">
          <el-col :span="6">
            <el-select
              v-model="filters.platform"
              placeholder="选择平台"
              clearable
              @change="handleFilterChange"
            >
              <el-option label="全部平台" value="" />
              <el-option
                v-for="platform in platforms"
                :key="platform.name"
                :label="platform.display_name"
                :value="platform.name"
              />
            </el-select>
          </el-col>
          <el-col :span="6">
            <el-select
              v-model="filters.status"
              placeholder="选择状态"
              clearable
              @change="handleFilterChange"
            >
              <el-option label="全部状态" value="" />
              <el-option label="活跃" value="active" />
              <el-option label="已暂停" value="suspended" />
              <el-option label="已过期" value="expired" />
              <el-option label="错误" value="error" />
            </el-select>
          </el-col>
          <el-col :span="8">
            <el-input
              v-model="filters.search"
              placeholder="搜索账号名称或显示名称"
              clearable
              @input="handleSearchChange"
            >
              <template #prefix>
                <el-icon><Search /></el-icon>
              </template>
            </el-input>
          </el-col>
          <el-col :span="4">
            <el-button-group>
              <el-button
                :type="viewMode === 'card' ? 'primary' : ''"
                @click="viewMode = 'card'"
              >
                <el-icon><Grid /></el-icon>
              </el-button>
              <el-button
                :type="viewMode === 'list' ? 'primary' : ''"
                @click="viewMode = 'list'"
              >
                <el-icon><List /></el-icon>
              </el-button>
            </el-button-group>
          </el-col>
        </el-row>
      </el-card>
    </div>
    
    <!-- 账号列表 - 卡片视图 -->
    <div v-if="viewMode === 'card'" class="accounts-grid">
      <el-row :gutter="20">
        <el-col
          v-for="account in filteredAccounts"
          :key="account.id"
          :span="8"
          class="account-col"
        >
          <el-card class="account-card" :class="getAccountCardClass(account)">
            <div class="account-header">
              <div class="account-avatar">
                <el-avatar
                  :src="account.avatar_url"
                  :size="50"
                  :icon="UserFilled"
                />
                <div class="platform-badge">
                  <img
                    :src="getPlatformIcon(account.platform)"
                    :alt="account.platform"
                    class="platform-icon"
                  />
                </div>
              </div>
              <div class="account-info">
                <div class="account-name">{{ account.display_name || account.account_name }}</div>
                <div class="account-handle">{{ account.account_name }}</div>
                <div class="account-platform">{{ account.platform_display_name }}</div>
              </div>
              <div class="account-status">
                <el-tag :type="getStatusColor(account.status)" size="small">
                  {{ getStatusLabel(account.status) }}
                </el-tag>
              </div>
            </div>
            
            <div class="account-stats">
              <div class="stat-item">
                <div class="stat-value">{{ formatNumber(account.stats.followers) }}</div>
                <div class="stat-label">粉丝</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">{{ formatNumber(account.stats.following) }}</div>
                <div class="stat-label">关注</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">{{ formatNumber(account.stats.posts) }}</div>
                <div class="stat-label">帖子</div>
              </div>
            </div>
            
            <div class="account-footer">
              <div class="last-sync">
                <el-icon><Clock /></el-icon>
                <span>{{ getLastSyncText(account.last_sync) }}</span>
              </div>
              <div class="account-actions">
                <el-button-group size="small">
                  <el-button @click="syncAccount(account.id)" :loading="account.syncing">
                    <el-icon><Refresh /></el-icon>
                  </el-button>
                  <el-button @click="viewAccountDetails(account)">
                    <el-icon><View /></el-icon>
                  </el-button>
                  <el-button @click="showAccountSettings(account)">
                    <el-icon><Setting /></el-icon>
                  </el-button>
                </el-button-group>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 账号列表 - 列表视图 -->
    <div v-else class="accounts-table">
      <el-card>
        <el-table
          :data="filteredAccounts"
          style="width: 100%"
          @selection-change="handleSelectionChange"
        >
          <el-table-column type="selection" width="55" />
          
          <el-table-column label="账号" min-width="200">
            <template #default="{ row }">
              <div class="account-cell">
                <div class="account-avatar-small">
                  <el-avatar
                    :src="row.avatar_url"
                    :size="40"
                    :icon="UserFilled"
                  />
                  <img
                    :src="getPlatformIcon(row.platform)"
                    :alt="row.platform"
                    class="platform-icon-small"
                  />
                </div>
                <div class="account-info-small">
                  <div class="account-name-small">{{ row.display_name || row.account_name }}</div>
                  <div class="account-handle-small">{{ row.account_name }}</div>
                </div>
              </div>
            </template>
          </el-table-column>
          
          <el-table-column label="平台" width="120">
            <template #default="{ row }">
              <el-tag size="small">{{ row.platform_display_name }}</el-tag>
            </template>
          </el-table-column>
          
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <el-tag :type="getStatusColor(row.status)" size="small">
                {{ getStatusLabel(row.status) }}
              </el-tag>
            </template>
          </el-table-column>
          
          <el-table-column label="粉丝数" width="100" sortable>
            <template #default="{ row }">
              {{ formatNumber(row.stats.followers) }}
            </template>
          </el-table-column>
          
          <el-table-column label="关注数" width="100" sortable>
            <template #default="{ row }">
              {{ formatNumber(row.stats.following) }}
            </template>
          </el-table-column>
          
          <el-table-column label="帖子数" width="100" sortable>
            <template #default="{ row }">
              {{ formatNumber(row.stats.posts) }}
            </template>
          </el-table-column>
          
          <el-table-column label="最后同步" width="150">
            <template #default="{ row }">
              {{ getLastSyncText(row.last_sync) }}
            </template>
          </el-table-column>
          
          <el-table-column label="操作" width="200" fixed="right">
            <template #default="{ row }">
              <el-button-group size="small">
                <el-button @click="syncAccount(row.id)" :loading="row.syncing">
                  <el-icon><Refresh /></el-icon>
                  同步
                </el-button>
                <el-button @click="viewAccountDetails(row)">
                  <el-icon><View /></el-icon>
                  详情
                </el-button>
                <el-button @click="showAccountSettings(row)">
                  <el-icon><Setting /></el-icon>
                  设置
                </el-button>
              </el-button-group>
            </template>
          </el-table-column>
        </el-table>
        
        <!-- 批量操作 -->
        <div v-if="selectedAccounts.length > 0" class="batch-actions">
          <el-alert
            :title="`已选择 ${selectedAccounts.length} 个账号`"
            type="info"
            show-icon
            :closable="false"
          >
            <template #default>
              <el-button-group size="small">
                <el-button @click="batchSyncAccounts">
                  <el-icon><Refresh /></el-icon>
                  批量同步
                </el-button>
                <el-button @click="batchRemoveAccounts" type="danger">
                  <el-icon><Delete /></el-icon>
                  批量移除
                </el-button>
              </el-button-group>
            </template>
          </el-alert>
        </div>
      </el-card>
    </div>
    
    <!-- 添加账号对话框 -->
    <AddAccountDialog
      v-model="showAddAccountDialog"
      @account-added="handleAccountAdded"
    />
    
    <!-- 账号详情对话框 -->
    <AccountDetailsDialog
      v-model="showDetailsDialog"
      :account="selectedAccount"
      @account-updated="handleAccountUpdated"
    />
    
    <!-- 账号设置对话框 -->
    <AccountSettingsDialog
      v-model="showSettingsDialog"
      :account="selectedAccount"
      @account-updated="handleAccountUpdated"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Plus, Refresh, User, CircleCheck, Grid, List, Search,
  Clock, View, Setting, Delete, Loading, UserFilled
} from '@element-plus/icons-vue'
import { accountApi } from '@/api/account'
import { formatNumber, formatDate } from '@/utils/format'
import AddAccountDialog from './components/AddAccountDialog.vue'
import AccountDetailsDialog from './components/AccountDetailsDialog.vue'
import AccountSettingsDialog from './components/AccountSettingsDialog.vue'

// 响应式数据
const accounts = ref<Account[]>([])
const platforms = ref<Platform[]>([])
const loading = ref(false)
const viewMode = ref<'card' | 'list'>('card')
const selectedAccounts = ref<Account[]>([])

// 对话框状态
const showAddAccountDialog = ref(false)
const showDetailsDialog = ref(false)
const showSettingsDialog = ref(false)
const selectedAccount = ref<Account | null>(null)

// 筛选条件
const filters = reactive({
  platform: '',
  status: '',
  search: ''
})

// 统计数据
const stats = computed(() => {
  const totalAccounts = accounts.value.length
  const activeAccounts = accounts.value.filter(acc => acc.status === 'active').length
  const platformCount = new Set(accounts.value.map(acc => acc.platform)).size
  const syncingAccounts = accounts.value.filter(acc => acc.syncing).length
  
  return {
    totalAccounts,
    activeAccounts,
    platformCount,
    syncingAccounts
  }
})

// 过滤后的账号列表
const filteredAccounts = computed(() => {
  let result = accounts.value
  
  if (filters.platform) {
    result = result.filter(acc => acc.platform === filters.platform)
  }
  
  if (filters.status) {
    result = result.filter(acc => acc.status === filters.status)
  }
  
  if (filters.search) {
    const searchTerm = filters.search.toLowerCase()
    result = result.filter(acc => 
      acc.account_name.toLowerCase().includes(searchTerm) ||
      (acc.display_name && acc.display_name.toLowerCase().includes(searchTerm))
    )
  }
  
  return result
})

// 生命周期钩子
onMounted(() => {
  loadAccounts()
  loadPlatforms()
})

// 方法定义
const loadAccounts = async () => {
  try {
    loading.value = true
    const response = await accountApi.getAccounts()
    accounts.value = response.data.map(acc => ({ ...acc, syncing: false }))
  } catch (error) {
    ElMessage.error('加载账号列表失败')
  } finally {
    loading.value = false
  }
}

const loadPlatforms = async () => {
  try {
    const response = await accountApi.getPlatforms()
    platforms.value = response.data
  } catch (error) {
    console.error('加载平台列表失败:', error)
  }
}

const refreshAccounts = () => {
  loadAccounts()
}

const handleFilterChange = () => {
  // 筛选条件变化时的处理逻辑
}

const handleSearchChange = () => {
  // 搜索条件变化时的处理逻辑
}

const syncAccount = async (accountId: number) => {
  try {
    const account = accounts.value.find(acc => acc.id === accountId)
    if (account) {
      account.syncing = true
    }
    
    await accountApi.syncAccount(accountId, ['profile', 'stats'])
    ElMessage.success('同步完成')
    
    // 重新加载账号信息
    await loadAccounts()
  } catch (error) {
    ElMessage.error('同步失败')
  } finally {
    const account = accounts.value.find(acc => acc.id === accountId)
    if (account) {
      account.syncing = false
    }
  }
}

const viewAccountDetails = (account: Account) => {
  selectedAccount.value = account
  showDetailsDialog.value = true
}

const showAccountSettings = (account: Account) => {
  selectedAccount.value = account
  showSettingsDialog.value = true
}

const handleSelectionChange = (selection: Account[]) => {
  selectedAccounts.value = selection
}

const batchSyncAccounts = async () => {
  try {
    const promises = selectedAccounts.value.map(acc => {
      acc.syncing = true
      return accountApi.syncAccount(acc.id, ['profile', 'stats'])
    })
    
    await Promise.all(promises)
    ElMessage.success(`成功同步 ${selectedAccounts.value.length} 个账号`)
    
    await loadAccounts()
  } catch (error) {
    ElMessage.error('批量同步失败')
  }
}

const batchRemoveAccounts = async () => {
  try {
    await ElMessageBox.confirm(
      `确定要移除选中的 ${selectedAccounts.value.length} 个账号吗？`,
      '确认移除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    const promises = selectedAccounts.value.map(acc => 
      accountApi.removeAccount(acc.id)
    )
    
    await Promise.all(promises)
    ElMessage.success(`成功移除 ${selectedAccounts.value.length} 个账号`)
    
    await loadAccounts()
    selectedAccounts.value = []
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('批量移除失败')
    }
  }
}

const handleAccountAdded = () => {
  showAddAccountDialog.value = false
  loadAccounts()
  ElMessage.success('账号添加成功')
}

const handleAccountUpdated = () => {
  loadAccounts()
  ElMessage.success('账号更新成功')
}

// 工具方法
const getAccountCardClass = (account: Account) => {
  return {
    'account-card--active': account.status === 'active',
    'account-card--suspended': account.status === 'suspended',
    'account-card--error': account.status === 'error',
    'account-card--syncing': account.syncing
  }
}

const getStatusColor = (status: string) => {
  const colorMap = {
    active: 'success',
    suspended: 'warning',
    expired: 'info',
    error: 'danger'
  }
  return colorMap[status] || 'info'
}

const getStatusLabel = (status: string) => {
  const labelMap = {
    active: '活跃',
    suspended: '已暂停',
    expired: '已过期',
    error: '错误'
  }
  return labelMap[status] || status
}

const getPlatformIcon = (platform: string) => {
  const iconMap = {
    weibo: '/icons/weibo.svg',
    wechat: '/icons/wechat.svg',
    douyin: '/icons/douyin.svg',
    toutiao: '/icons/toutiao.svg',
    baijiahao: '/icons/baijiahao.svg'
  }
  return iconMap[platform] || '/icons/default.svg'
}

const getLastSyncText = (lastSync: string | null) => {
  if (!lastSync) return '从未同步'
  
  const syncDate = new Date(lastSync)
  const now = new Date()
  const diffMs = now.getTime() - syncDate.getTime()
  const diffMins = Math.floor(diffMs / (1000 * 60))
  
  if (diffMins < 1) return '刚刚'
  if (diffMins < 60) return `${diffMins}分钟前`
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}小时前`
  return formatDate(syncDate)
}

// 类型定义
interface Account {
  id: number
  platform: string
  platform_display_name: string
  account_name: string
  display_name?: string
  avatar_url?: string
  stats: {
    followers: number
    following: number
    posts: number
  }
  status: string
  last_sync: string | null
  syncing?: boolean
}

interface Platform {
  name: string
  display_name: string
  platform_type: string
  is_active: boolean
}
</script>

<style scoped>
.account-management {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
}

.header-content {
  flex: 1;
}

.page-title {
  font-size: 28px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.page-description {
  font-size: 16px;
  color: #6b7280;
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.stats-cards {
  margin-bottom: 24px;
}

.stat-card {
  border: none;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.stat-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.stat-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
}

.stat-icon.total {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stat-icon.active {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.stat-icon.platforms {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.stat-icon.syncing {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1;
}

.stat-label {
  font-size: 14px;
  color: #6b7280;
  margin-top: 4px;
}

.filter-section {
  margin-bottom: 24px;
}

.accounts-grid {
  margin-bottom: 24px;
}

.account-col {
  margin-bottom: 20px;
}

.account-card {
  border: none;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  overflow: hidden;
}

.account-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: translateY(-2px);
}

.account-card--active {
  border-left: 4px solid #10b981;
}

.account-card--suspended {
  border-left: 4px solid #f59e0b;
}

.account-card--error {
  border-left: 4px solid #ef4444;
}

.account-card--syncing {
  border-left: 4px solid #3b82f6;
}

.account-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 16px;
}

.account-avatar {
  position: relative;
}

.platform-badge {
  position: absolute;
  bottom: -4px;
  right: -4px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: white;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.platform-icon {
  width: 16px;
  height: 16px;
}

.account-info {
  flex: 1;
  min-width: 0;
}

.account-name {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.account-handle {
  font-size: 14px;
  color: #6b7280;
  margin-bottom: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.account-platform {
  font-size: 12px;
  color: #9ca3af;
}

.account-stats {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
  padding: 12px;
  background: #f9fafb;
  border-radius: 8px;
}

.stat-item {
  text-align: center;
}

.stat-item .stat-value {
  font-size: 18px;
  font-weight: 600;
  color: #1f2937;
  line-height: 1;
}

.stat-item .stat-label {
  font-size: 12px;
  color: #6b7280;
  margin-top: 2px;
}

.account-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.last-sync {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #6b7280;
}

.account-cell {
  display: flex;
  align-items: center;
  gap: 12px;
}

.account-avatar-small {
  position: relative;
}

.platform-icon-small {
  position: absolute;
  bottom: -2px;
  right: -2px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: white;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.account-info-small {
  flex: 1;
  min-width: 0;
}

.account-name-small {
  font-size: 14px;
  font-weight: 500;
  color: #1f2937;
  margin-bottom: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.account-handle-small {
  font-size: 12px;
  color: #6b7280;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.batch-actions {
  margin-top: 16px;
}

.accounts-table {
  margin-bottom: 24px;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    gap: 16px;
  }
  
  .header-actions {
    width: 100%;
    justify-content: flex-start;
  }
  
  .stats-cards .el-col {
    margin-bottom: 12px;
  }
  
  .account-col {
    span: 24;
  }
}
</style>
```

### 2. 添加账号对话框组件

```vue
<template>
  <el-dialog
    v-model="visible"
    title="添加账号"
    width="600px"
    :close-on-click-modal="false"
    @close="handleClose"
  >
    <div class="add-account-dialog">
      <!-- 步骤指示器 -->
      <el-steps :active="currentStep" align-center class="steps">
        <el-step title="选择平台" />
        <el-step title="授权登录" />
        <el-step title="完成添加" />
      </el-steps>
      
      <!-- 步骤1: 选择平台 -->
      <div v-if="currentStep === 0" class="step-content">
        <h3>选择要添加的平台</h3>
        <div class="platform-grid">
          <div
            v-for="platform in availablePlatforms"
            :key="platform.name"
            class="platform-item"
            :class="{ 'platform-item--selected': selectedPlatform === platform.name }"
            @click="selectPlatform(platform.name)"
          >
            <img
              :src="getPlatformIcon(platform.name)"
              :alt="platform.display_name"
              class="platform-logo"
            />
            <div class="platform-name">{{ platform.display_name }}</div>
            <div class="platform-desc">{{ platform.description }}</div>
          </div>
        </div>
      </div>
      
      <!-- 步骤2: 授权登录 -->
      <div v-if="currentStep === 1" class="step-content">
        <div class="auth-content">
          <div class="auth-header">
            <img
              :src="getPlatformIcon(selectedPlatform)"
              :alt="selectedPlatform"
              class="auth-platform-icon"
            />
            <h3>授权 {{ getPlatformName(selectedPlatform) }}</h3>
            <p>请点击下方按钮前往 {{ getPlatformName(selectedPlatform) }} 进行授权</p>
          </div>
          
          <div class="auth-actions">
            <el-button
              type="primary"
              size="large"
              @click="startOAuth"
              :loading="authLoading"
            >
              前往授权
            </el-button>
          </div>
          
          <div v-if="authCode" class="auth-code-section">
            <el-alert
              title="授权成功"
              type="success"
              description="已获取到授权码，正在添加账号..."
              show-icon
              :closable="false"
            />
          </div>
        </div>
      </div>
      
      <!-- 步骤3: 完成添加 -->
      <div v-if="currentStep === 2" class="step-content">
        <div class="success-content">
          <el-result
            icon="success"
            title="账号添加成功"
            :sub-title="`${addedAccount?.platform_display_name} 账号 ${addedAccount?.account_name} 已成功添加`"
          >
            <template #extra>
              <el-button type="primary" @click="handleFinish">
                完成
              </el-button>
              <el-button @click="addAnother">
                继续添加
              </el-button>
            </template>
          </el-result>
        </div>
      </div>
    </div>
    
    <template #footer>
      <div class="dialog-footer">
        <el-button @click="handleClose">取消</el-button>
        <el-button
          v-if="currentStep === 0"
          type="primary"
          :disabled="!selectedPlatform"
          @click="nextStep"
        >
          下一步
        </el-button>
        <el-button
          v-if="currentStep === 1"
          @click="prevStep"
        >
          上一步
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { accountApi } from '@/api/account'

// Props
interface Props {
  modelValue: boolean
}

const props = defineProps<Props>()

// Emits
interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'account-added', account: any): void
}

const emit = defineEmits<Emits>()

// 响应式数据
const currentStep = ref(0)
const selectedPlatform = ref('')
const authCode = ref('')
const authLoading = ref(false)
const addedAccount = ref(null)

const availablePlatforms = ref([
  {
    name: 'weibo',
    display_name: '微博',
    description: '新浪微博社交平台'
  },
  {
    name: 'wechat',
    display_name: '微信公众号',
    description: '微信公众号平台'
  },
  {
    name: 'douyin',
    display_name: '抖音',
    description: '抖音短视频平台'
  },
  {
    name: 'toutiao',
    display_name: '今日头条',
    description: '今日头条资讯平台'
  },
  {
    name: 'baijiahao',
    display_name: '百家号',
    description: '百度百家号平台'
  }
])

// 计算属性
const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

// 监听器
watch(visible, (newVal) => {
  if (newVal) {
    resetDialog()
  }
})

// 方法
const resetDialog = () => {
  currentStep.value = 0
  selectedPlatform.value = ''
  authCode.value = ''
  authLoading.value = false
  addedAccount.value = null
}

const selectPlatform = (platform: string) => {
  selectedPlatform.value = platform
}

const nextStep = () => {
  if (currentStep.value < 2) {
    currentStep.value++
  }
}

const prevStep = () => {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

const startOAuth = async () => {
  try {
    authLoading.value = true
    
    // 获取OAuth授权URL
    const response = await accountApi.getOAuthUrl(selectedPlatform.value)
    const oauthUrl = response.data.oauth_url
    
    // 打开新窗口进行授权
    const authWindow = window.open(
      oauthUrl,
      'oauth_auth',
      'width=600,height=700,scrollbars=yes,resizable=yes'
    )
    
    // 监听授权回调
    const checkAuth = setInterval(() => {
      try {
        if (authWindow?.closed) {
          clearInterval(checkAuth)
          authLoading.value = false
          return
        }
        
        // 检查URL变化获取授权码
        const url = authWindow?.location.href
        if (url && url.includes('code=')) {
          const urlParams = new URLSearchParams(url.split('?')[1])
          const code = urlParams.get('code')
          
          if (code) {
            authCode.value = code
            authWindow?.close()
            clearInterval(checkAuth)
            addAccount(code)
          }
        }
      } catch (error) {
        // 跨域错误，继续检查
      }
    }, 1000)
    
  } catch (error) {
    ElMessage.error('获取授权URL失败')
    authLoading.value = false
  }
}

const addAccount = async (code: string) => {
  try {
    const response = await accountApi.addAccount({
      platform: selectedPlatform.value,
      auth_code: code
    })
    
    addedAccount.value = response.data
    currentStep.value = 2
    authLoading.value = false
    
  } catch (error) {
    ElMessage.error('添加账号失败')
    authLoading.value = false
  }
}

const handleFinish = () => {
  emit('account-added', addedAccount.value)
  visible.value = false
}

const addAnother = () => {
  resetDialog()
}

const handleClose = () => {
  visible.value = false
}

const getPlatformIcon = (platform: string) => {
  const iconMap = {
    weibo: '/icons/weibo.svg',
    wechat: '/icons/wechat.svg',
    douyin: '/icons/douyin.svg',
    toutiao: '/icons/toutiao.svg',
    baijiahao: '/icons/baijiahao.svg'
  }
  return iconMap[platform] || '/icons/default.svg'
}

const getPlatformName = (platform: string) => {
  const nameMap = {
    weibo: '微博',
    wechat: '微信公众号',
    douyin: '抖音',
    toutiao: '今日头条',
    baijiahao: '百家号'
  }
  return nameMap[platform] || platform
}
</script>

<style scoped>
.add-account-dialog {
  padding: 20px 0;
}

.steps {
  margin-bottom: 40px;
}

.step-content {
  min-height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.step-content h3 {
  margin-bottom: 24px;
  color: #1f2937;
  font-size: 18px;
}

.platform-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  width: 100%;
  max-width: 500px;
}

.platform-item {
  padding: 20px;
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.platform-item:hover {
  border-color: #3b82f6;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
}

.platform-item--selected {
  border-color: #3b82f6;
  background: #eff6ff;
}

.platform-logo {
  width: 48px;
  height: 48px;
  margin-bottom: 12px;
}

.platform-name {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 4px;
}

.platform-desc {
  font-size: 12px;
  color: #6b7280;
}

.auth-content {
  text-align: center;
  width: 100%;
}

.auth-header {
  margin-bottom: 32px;
}

.auth-platform-icon {
  width: 64px;
  height: 64px;
  margin-bottom: 16px;
}

.auth-header h3 {
  margin-bottom: 8px;
}

.auth-header p {
  color: #6b7280;
  margin: 0;
}

.auth-actions {
  margin-bottom: 24px;
}

.auth-code-section {
  max-width: 400px;
  margin: 0 auto;
}

.success-content {
  width: 100%;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}
</style>
```

## 验收标准

### 功能验收标准

1. **账号添加功能**
   - 支持主流社交媒体平台的OAuth2授权流程
   - 能够正确获取和存储账号基本信息
   - 支持授权失败的错误处理和重试机制
   - 能够检测和处理重复账号添加

2. **账号管理功能**
   - 提供账号列表的多种视图模式（卡片/列表）
   - 支持按平台、状态等条件筛选账号
   - 支持账号搜索功能
   - 提供账号批量操作功能

3. **账号同步功能**
   - 能够定期同步账号基本信息和统计数据
   - 支持手动触发同步操作
   - 提供同步进度和状态显示
   - 支持同步失败的错误处理

4. **权限管理功能**
   - 支持多用户账号权限管理
   - 提供细粒度的权限控制（读取、写入、发布、管理）
   - 支持权限的授予、撤销和过期管理

### 性能验收标准

1. **响应时间要求**
   - 账号列表加载时间 < 2秒
   - 账号同步操作响应时间 < 5秒
   - OAuth授权流程完成时间 < 30秒
   - API请求平均响应时间 < 500ms

2. **并发处理能力**
   - 支持100个并发用户同时使用
   - 支持同时管理1000个账号
   - 支持每小时10000次API调用

3. **系统稳定性**
   - 系统可用性 ≥ 99.5%
   - 平均故障恢复时间 < 5分钟
   - 数据一致性保证 100%

### 安全验收标准

1. **数据安全**
   - 访问令牌采用AES-256加密存储
   - 敏感数据传输使用HTTPS协议
   - 实施数据访问审计日志

2. **访问控制**
   - 实施基于角色的访问控制（RBAC）
   - 支持API访问频率限制
   - 提供账号操作权限验证

3. **合规要求**
   - 遵循各平台的API使用条款
   - 实施用户数据隐私保护
   - 支持数据删除和导出功能

## 商业价值

### 直接价值

1. **运营效率提升**
   - 统一账号管理减少50%的账号维护时间
   - 自动化同步减少80%的手动数据更新工作
   - 批量操作提升60%的账号管理效率

2. **成本节约**
   - 减少人工账号管理成本约30%
   - 降低账号信息不一致导致的运营错误
   - 提高内容发布的准确性和及时性

### 间接价值

1. **业务扩展支持**
   - 为多平台内容发布提供基础支撑
   - 支持更大规模的社交媒体运营
   - 提升品牌在多平台的一致性表现

2. **数据价值挖掘**
   - 收集多平台账号数据用于分析
   - 支持跨平台运营策略制定
   - 为用户行为分析提供数据基础

## 依赖关系

### 技术依赖

1. **外部服务依赖**
   - 各社交媒体平台的OAuth2 API
   - Redis缓存服务
   - PostgreSQL数据库
   - RabbitMQ消息队列

2. **内部系统依赖**
   - 用户认证服务
   - 权限管理服务
   - 日志监控服务
   - 配置管理服务

### 业务依赖

1. **平台合作关系**
   - 需要获得各平台的开发者资质
   - 需要申请相应的API访问权限
   - 需要遵循平台的使用条款和限制

2. **运营流程依赖**
   - 需要建立账号管理的标准流程
   - 需要制定权限分配的管理制度
   - 需要建立账号安全的监控机制

### 环境依赖

1. **开发环境**
   - Python 3.9+ 开发环境
   - Node.js 16+ 前端开发环境
   - Docker容器化环境
   - Kubernetes集群环境

2. **生产环境**
   - 高可用的数据库集群
   - 分布式缓存集群
   - 负载均衡和反向代理
   - 监控和日志收集系统

## 风险评估

### 技术风险

1. **API变更风险**
   - **风险描述**: 第三方平台API接口变更或废弃
   - **影响程度**: 高
   - **缓解措施**: 建立API版本管理机制，及时跟踪平台更新

2. **性能瓶颈风险**
   - **风险描述**: 大量账号同步导致系统性能下降
   - **影响程度**: 中
   - **缓解措施**: 实施分批同步和限流机制

3. **数据一致性风险**
   - **风险描述**: 多平台数据同步可能出现不一致
   - **影响程度**: 中
   - **缓解措施**: 建立数据校验和修复机制

### 业务风险

1. **平台政策风险**
   - **风险描述**: 平台政策变化影响API使用
   - **影响程度**: 高
   - **缓解措施**: 建立多平台备选方案，分散风险

2. **合规风险**
   - **风险描述**: 数据处理不符合隐私保护法规
   - **影响程度**: 高
   - **缓解措施**: 严格遵循数据保护法规，建立合规审查机制

### 运营风险

1. **账号安全风险**
   - **风险描述**: 账号凭证泄露或被恶意使用
   - **影响程度**: 高
   - **缓解措施**: 实施多层安全防护和访问监控

2. **服务可用性风险**
   - **风险描述**: 系统故障影响账号管理功能
   - **影响程度**: 中
   - **缓解措施**: 建立高可用架构和快速恢复机制

## 开发任务分解

### 后端开发任务

1. **核心服务开发** (预计15个工作日)
   - 账号管理服务核心逻辑实现
   - 平台适配器接口设计和实现
   - OAuth2认证流程实现
   - 账号数据同步机制实现

2. **数据库设计** (预计5个工作日)
   - 数据库表结构设计
   - 索引优化和性能调优
   - 数据迁移脚本编写
   - 数据备份和恢复策略

3. **API接口开发** (预计10个工作日)
   - RESTful API接口实现
   - API文档编写
   - 接口测试和调试
   - 错误处理和异常管理

4. **安全机制实现** (预计8个工作日)
   - 数据加密和解密实现
   - 访问权限控制实现
   - API限流和防护实现
   - 安全审计日志实现

### 前端开发任务

1. **主界面开发** (预计12个工作日)
   - 账号管理主页面实现
   - 账号列表和筛选功能
   - 统计数据展示组件
   - 响应式布局适配

2. **对话框组件开发** (预计10个工作日)
   - 添加账号对话框实现
   - 账号详情对话框实现
   - 账号设置对话框实现
   - 权限管理对话框实现

3. **交互功能实现** (预计8个工作日)
   - 账号同步功能实现
   - 批量操作功能实现
   - 实时状态更新实现
   - 错误提示和处理实现

### 测试任务

1. **单元测试** (预计8个工作日)
   - 后端服务单元测试
   - 前端组件单元测试
   - 测试覆盖率达到80%以上

2. **集成测试** (预计6个工作日)
   - API接口集成测试
   - 前后端集成测试
   - 第三方平台集成测试

3. **性能测试** (预计4个工作日)
   - 负载测试和压力测试
   - 数据库性能测试
   - 接口响应时间测试

### 部署任务

1. **环境准备** (预计3个工作日)
   - 开发环境搭建
   - 测试环境配置
   - 生产环境准备

2. **部署实施** (预计2个工作日)
   - 应用部署和配置
   - 数据库初始化
   - 监控和日志配置

## 人力资源需求

### 开发团队配置

1. **后端开发工程师**: 2人
   - 负责核心服务和API开发
   - 要求有Python/FastAPI开发经验
   - 要求有OAuth2和第三方API集成经验

2. **前端开发工程师**: 2人
   - 负责Vue3前端界面开发
   - 要求有Vue3/TypeScript开发经验
   - 要求有Element Plus组件库使用经验

3. **测试工程师**: 1人
   - 负责功能测试和性能测试
   - 要求有API测试和自动化测试经验

4. **DevOps工程师**: 1人
   - 负责部署和运维工作
   - 要求有Docker/Kubernetes经验

### 项目关键里程碑

1. **需求确认完成**: 第1周
2. **技术方案评审**: 第2周
3. **核心功能开发完成**: 第6周
4. **功能测试完成**: 第8周
5. **性能优化完成**: 第9周
6. **生产环境部署**: 第10周

### 成功指标

#### 技术指标

1. **功能完整性**: 100%的需求功能实现
2. **代码质量**: 代码覆盖率 ≥ 80%
3. **性能指标**: API响应时间 < 500ms
4. **稳定性指标**: 系统可用性 ≥ 99.5%

#### 业务指标

1. **用户采用率**: 90%的目标用户开始使用
2. **操作效率**: 账号管理效率提升 ≥ 50%
3. **错误率**: 账号操作错误率 < 1%
4. **用户满意度**: 用户满意度评分 ≥ 4.5/5.0

#### 运营指标

1. **系统负载**: 支持100并发用户
2. **数据准确性**: 账号数据同步准确率 ≥ 99%
3. **响应时间**: 用户操作响应时间 < 2秒
4. **故障恢复**: 平均故障恢复时间 < 5分钟