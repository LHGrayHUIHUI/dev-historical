"""
账号管理核心服务

提供账号添加、认证、管理和同步的核心功能
支持多平台OAuth认证和令牌管理
"""

import json
import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_, func
from cryptography.fernet import Fernet
import redis.asyncio as redis
import httpx

from ..models.account_models import Platform, Account, AccountCredential, AccountPermission, AccountSyncLog
from ..config.settings import settings, PlatformConfig
from .oauth_service import OAuthService
from .encryption_service import EncryptionService

logger = logging.getLogger(__name__)


class AccountManagementService:
    """
    多平台账号管理服务
    
    负责账号的添加、认证、管理和同步
    提供统一的账号管理接口
    """
    
    def __init__(self, db: AsyncSession, redis_client: redis.Redis):
        """
        初始化账号管理服务
        
        Args:
            db: 数据库会话
            redis_client: Redis客户端
        """
        self.db = db
        self.redis = redis_client
        self.oauth_service = OAuthService(redis_client)
        self.encryption_service = EncryptionService(settings.encryption_key)
        
        logger.info("账号管理服务初始化完成")
    
    async def get_platforms(self, active_only: bool = True) -> List[Platform]:
        """
        获取支持的平台列表
        
        Args:
            active_only: 是否只返回激活的平台
            
        Returns:
            List[Platform]: 平台列表
        """
        try:
            query = select(Platform)
            if active_only:
                query = query.where(Platform.is_active == True)
            
            query = query.order_by(Platform.name)
            
            result = await self.db.execute(query)
            platforms = result.scalars().all()
            
            logger.info(f"获取到{len(platforms)}个平台")
            return platforms
            
        except Exception as e:
            logger.error(f"获取平台列表失败: {e}")
            raise
    
    async def get_platform_by_name(self, name: str) -> Optional[Platform]:
        """
        根据名称获取平台信息
        
        Args:
            name: 平台名称
            
        Returns:
            Optional[Platform]: 平台信息
        """
        try:
            query = select(Platform).where(
                and_(Platform.name == name, Platform.is_active == True)
            )
            
            result = await self.db.execute(query)
            platform = result.scalar_one_or_none()
            
            return platform
            
        except Exception as e:
            logger.error(f"获取平台信息失败: {e}")
            raise
    
    async def create_platform(self, platform_data: Dict[str, Any]) -> Platform:
        """
        创建新平台配置
        
        Args:
            platform_data: 平台配置数据
            
        Returns:
            Platform: 创建的平台对象
        """
        try:
            platform = Platform(
                name=platform_data['name'],
                display_name=platform_data['display_name'],
                platform_type=platform_data['platform_type'],
                api_base_url=platform_data.get('api_base_url'),
                oauth_config=platform_data['oauth_config'],
                rate_limits=platform_data.get('rate_limits', {}),
                features=platform_data.get('features', {}),
                is_active=platform_data.get('is_active', True)
            )
            
            self.db.add(platform)
            await self.db.commit()
            await self.db.refresh(platform)
            
            logger.info(f"创建平台成功: {platform.name}")
            return platform
            
        except Exception as e:
            logger.error(f"创建平台失败: {e}")
            await self.db.rollback()
            raise
    
    async def add_account(self, user_id: int, platform_name: str, auth_code: str, 
                         redirect_uri: str = None) -> Dict[str, Any]:
        """
        添加新账号
        
        Args:
            user_id: 用户ID
            platform_name: 平台名称
            auth_code: 授权码
            redirect_uri: 重定向URI
            
        Returns:
            Dict[str, Any]: 账号信息
        """
        try:
            # 获取平台配置
            platform = await self.get_platform_by_name(platform_name)
            if not platform:
                raise ValueError(f"不支持的平台: {platform_name}")
            
            # 通过OAuth服务交换访问令牌
            token_data = await self.oauth_service.exchange_code_for_token(
                platform_name, auth_code, redirect_uri
            )
            
            # 获取账号信息
            account_info = await self._get_account_info_from_platform(
                platform_name, token_data['access_token']
            )
            
            # 检查账号是否已存在
            existing_account = await self._get_account_by_platform_id(
                platform.id, account_info['id']
            )
            
            if existing_account:
                # 更新现有账号的令牌
                await self._update_account_credentials(
                    existing_account.id, token_data
                )
                account_id = existing_account.id
                account = existing_account
            else:
                # 创建新账号
                account = Account(
                    platform_id=platform.id,
                    user_id=user_id,
                    account_name=account_info.get('username', account_info.get('name', '')),
                    account_id=str(account_info['id']),
                    display_name=account_info.get('display_name', account_info.get('name')),
                    avatar_url=account_info.get('avatar_url'),
                    bio=account_info.get('bio', account_info.get('description')),
                    follower_count=account_info.get('followers_count', 0),
                    following_count=account_info.get('following_count', 0),
                    post_count=account_info.get('statuses_count', 0),
                    verification_status='verified' if account_info.get('verified') else 'unverified',
                    status='active'
                )
                
                self.db.add(account)
                await self.db.commit()
                await self.db.refresh(account)
                account_id = account.id
                
                # 创建认证信息
                await self._create_account_credentials(account_id, token_data)
                
                # 给用户分配管理权限
                await self._grant_permission(account_id, user_id, 'admin')
            
            # 缓存账号信息
            await self._cache_account_info(account)
            
            logger.info(f"账号添加成功: {platform_name} - {account.account_name}")
            
            return {
                'account_id': account_id,
                'platform_name': platform_name,
                'account_name': account.account_name,
                'display_name': account.display_name,
                'status': account.status
            }
            
        except Exception as e:
            logger.error(f"添加账号失败: {e}")
            await self.db.rollback()
            raise
    
    async def get_user_accounts(self, user_id: int, platform_name: str = None, 
                              include_stats: bool = True) -> List[Dict[str, Any]]:
        """
        获取用户的账号列表
        
        Args:
            user_id: 用户ID
            platform_name: 平台名称过滤
            include_stats: 是否包含统计信息
            
        Returns:
            List[Dict[str, Any]]: 账号列表
        """
        try:
            # 从缓存获取
            cache_key = f"user_accounts:{user_id}:{platform_name or 'all'}"
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # 从数据库查询
            query = select(Account).join(Platform).where(
                or_(
                    Account.user_id == user_id,
                    Account.id.in_(
                        select(AccountPermission.account_id).where(
                            and_(
                                AccountPermission.user_id == user_id,
                                AccountPermission.is_active == True
                            )
                        )
                    )
                )
            )
            
            if platform_name:
                query = query.where(Platform.name == platform_name)
            
            result = await self.db.execute(query)
            accounts = result.scalars().all()
            
            account_list = []
            for account in accounts:
                account_data = {
                    'account_id': account.id,
                    'platform_name': account.platform.name,
                    'platform_display_name': account.platform.display_name,
                    'account_name': account.account_name,
                    'display_name': account.display_name,
                    'avatar_url': account.avatar_url,
                    'status': account.status,
                    'verification_status': account.verification_status,
                    'account_type': account.account_type,
                    'last_sync_at': account.last_sync_at.isoformat() if account.last_sync_at else None
                }
                
                if include_stats:
                    account_data.update({
                        'follower_count': account.follower_count,
                        'following_count': account.following_count,
                        'post_count': account.post_count
                    })
                
                # 获取用户权限
                permissions = await self._get_user_permissions(account.id, user_id)
                account_data['permissions'] = permissions
                
                account_list.append(account_data)
            
            # 缓存结果
            await self.redis.setex(cache_key, 300, json.dumps(account_list, default=str))
            
            logger.info(f"获取到用户{user_id}的{len(account_list)}个账号")
            return account_list
            
        except Exception as e:
            logger.error(f"获取用户账号失败: {e}")
            raise
    
    async def sync_account_data(self, account_id: int, sync_types: List[str] = None, 
                               force: bool = False) -> Dict[str, Any]:
        """
        同步账号数据
        
        Args:
            account_id: 账号ID
            sync_types: 同步类型列表
            force: 是否强制同步
            
        Returns:
            Dict[str, Any]: 同步结果
        """
        if sync_types is None:
            sync_types = ['profile', 'stats']
        
        try:
            # 检查是否正在同步
            sync_status_key = f"sync_status:{account_id}"
            if not force:
                is_syncing = await self.redis.get(sync_status_key)
                if is_syncing:
                    return {'status': 'already_syncing', 'message': '账号正在同步中'}
            
            # 设置同步状态
            sync_data = {
                'is_syncing': True,
                'started_at': datetime.utcnow().isoformat(),
                'sync_types': sync_types,
                'progress': 0
            }
            await self.redis.setex(sync_status_key, 3600, json.dumps(sync_data))
            
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
            
            # 记录同步开始
            sync_log = AccountSyncLog(
                account_id=account_id,
                sync_type=','.join(sync_types),
                status='in_progress',
                started_at=datetime.utcnow()
            )
            self.db.add(sync_log)
            await self.db.commit()
            
            sync_results = {}
            total_steps = len(sync_types)
            current_step = 0
            
            # 更新进度
            async def update_progress(step_name: str):
                nonlocal current_step
                current_step += 1
                progress = int((current_step / total_steps) * 100)
                sync_data['progress'] = progress
                await self.redis.setex(sync_status_key, 3600, json.dumps(sync_data))
            
            # 同步账号资料
            if 'profile' in sync_types:
                try:
                    profile_data = await self._get_account_info_from_platform(
                        account.platform.name, credentials['access_token']
                    )
                    await self._update_account_profile(account_id, profile_data)
                    sync_results['profile'] = 'success'
                    await update_progress('profile')
                except Exception as e:
                    sync_results['profile'] = f'failed: {str(e)}'
                    logger.error(f"同步账号资料失败: {e}")
            
            # 同步统计数据
            if 'stats' in sync_types:
                try:
                    stats_data = await self._get_account_stats_from_platform(
                        account.platform.name, credentials['access_token']
                    )
                    await self._update_account_stats(account_id, stats_data)
                    sync_results['stats'] = 'success'
                    await update_progress('stats')
                except Exception as e:
                    sync_results['stats'] = f'failed: {str(e)}'
                    logger.error(f"同步统计数据失败: {e}")
            
            # 更新同步时间
            await self._update_account_sync_time(account_id)
            
            # 更新同步日志
            sync_log.completed_at = datetime.utcnow()
            sync_log.status = 'success' if all(result == 'success' for result in sync_results.values()) else 'partial'
            sync_log.sync_data = sync_results
            await self.db.commit()
            
            # 清除同步状态
            await self.redis.delete(sync_status_key)
            
            # 更新账号缓存
            updated_account = await self._get_account_by_id(account_id)
            await self._cache_account_info(updated_account)
            
            logger.info(f"账号同步完成: {account_id} - {sync_results}")
            
            return {
                'status': 'completed',
                'account_id': account_id,
                'sync_results': sync_results,
                'sync_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            # 记录同步失败
            if 'sync_log' in locals():
                sync_log.completed_at = datetime.utcnow()
                sync_log.status = 'failed'
                sync_log.error_message = str(e)
                await self.db.commit()
            
            # 清除同步状态
            await self.redis.delete(sync_status_key)
            
            logger.error(f"账号同步失败: {e}")
            raise
    
    async def batch_sync_accounts(self, account_ids: List[int], sync_types: List[str] = None) -> Dict[str, Any]:
        """
        批量同步账号
        
        Args:
            account_ids: 账号ID列表
            sync_types: 同步类型
            
        Returns:
            Dict[str, Any]: 批量同步结果
        """
        if sync_types is None:
            sync_types = ['profile', 'stats']
        
        try:
            batch_id = f"batch_sync_{int(datetime.utcnow().timestamp())}"
            total_accounts = len(account_ids)
            completed_accounts = 0
            failed_accounts = 0
            results = []
            
            logger.info(f"开始批量同步{total_accounts}个账号")
            
            # 限制并发数
            semaphore = asyncio.Semaphore(settings.max_concurrent_syncs)
            
            async def sync_single_account(account_id: int) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        result = await self.sync_account_data(
                            account_id, sync_types, force=False
                        )
                        return {'account_id': account_id, 'status': 'success', 'result': result}
                    except Exception as e:
                        return {'account_id': account_id, 'status': 'failed', 'error': str(e)}
            
            # 并发执行同步任务
            tasks = [sync_single_account(account_id) for account_id in account_ids]
            results = await asyncio.gather(*tasks)
            
            # 统计结果
            for result in results:
                if result['status'] == 'success':
                    completed_accounts += 1
                else:
                    failed_accounts += 1
            
            logger.info(f"批量同步完成: 成功{completed_accounts}个，失败{failed_accounts}个")
            
            return {
                'batch_id': batch_id,
                'total_accounts': total_accounts,
                'completed_accounts': completed_accounts,
                'failed_accounts': failed_accounts,
                'results': results,
                'completion_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"批量同步失败: {e}")
            raise
    
    async def delete_account(self, account_id: int, user_id: int) -> bool:
        """
        删除账号
        
        Args:
            account_id: 账号ID
            user_id: 用户ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            # 检查权限
            has_permission = await self._check_user_permission(account_id, user_id, 'admin')
            if not has_permission:
                raise ValueError("没有权限删除此账号")
            
            # 获取账号信息
            account = await self._get_account_by_id(account_id)
            if not account:
                raise ValueError(f"账号不存在: {account_id}")
            
            # 删除账号（级联删除相关数据）
            await self.db.execute(
                delete(Account).where(Account.id == account_id)
            )
            await self.db.commit()
            
            # 清除缓存
            await self._clear_account_cache(account_id)
            
            logger.info(f"账号删除成功: {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除账号失败: {e}")
            await self.db.rollback()
            raise
    
    # 私有辅助方法
    
    async def _get_account_by_id(self, account_id: int) -> Optional[Account]:
        """根据ID获取账号"""
        query = select(Account).where(Account.id == account_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_account_by_platform_id(self, platform_id: int, platform_account_id: str) -> Optional[Account]:
        """根据平台ID和平台账号ID获取账号"""
        query = select(Account).where(
            and_(
                Account.platform_id == platform_id,
                Account.account_id == platform_account_id
            )
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_account_credentials(self, account_id: int) -> Optional[Dict[str, Any]]:
        """获取账号认证信息"""
        query = select(AccountCredential).where(AccountCredential.account_id == account_id)
        result = await self.db.execute(query)
        credential = result.scalar_one_or_none()
        
        if credential:
            # 解密敏感数据
            decrypted_data = {}
            if credential.encrypted_data:
                try:
                    decrypted_data = self.encryption_service.decrypt_data(credential.encrypted_data)
                except Exception as e:
                    logger.error(f"解密认证数据失败: {e}")
            
            return {
                'access_token': credential.access_token,
                'refresh_token': credential.refresh_token,
                'token_type': credential.token_type,
                'expires_at': credential.expires_at,
                'scope': credential.scope,
                **decrypted_data
            }
        
        return None
    
    async def _create_account_credentials(self, account_id: int, token_data: Dict[str, Any]):
        """创建账号认证信息"""
        # 加密敏感数据
        sensitive_data = {k: v for k, v in token_data.items() 
                         if k not in ['access_token', 'refresh_token', 'token_type', 'expires_in', 'scope']}
        
        encrypted_data = None
        if sensitive_data:
            encrypted_data = self.encryption_service.encrypt_data(sensitive_data)
        
        expires_at = None
        if 'expires_in' in token_data:
            expires_at = datetime.utcnow() + timedelta(seconds=token_data['expires_in'])
        
        credential = AccountCredential(
            account_id=account_id,
            access_token=token_data.get('access_token'),
            refresh_token=token_data.get('refresh_token'),
            token_type=token_data.get('token_type', 'Bearer'),
            expires_at=expires_at,
            scope=token_data.get('scope'),
            encrypted_data=encrypted_data
        )
        
        self.db.add(credential)
    
    async def _update_account_credentials(self, account_id: int, token_data: Dict[str, Any]):
        """更新账号认证信息"""
        expires_at = None
        if 'expires_in' in token_data:
            expires_at = datetime.utcnow() + timedelta(seconds=token_data['expires_in'])
        
        await self.db.execute(
            update(AccountCredential)
            .where(AccountCredential.account_id == account_id)
            .values(
                access_token=token_data.get('access_token'),
                refresh_token=token_data.get('refresh_token'),
                token_type=token_data.get('token_type', 'Bearer'),
                expires_at=expires_at,
                scope=token_data.get('scope'),
                updated_at=func.now()
            )
        )
    
    async def _is_token_expired(self, credentials: Dict[str, Any]) -> bool:
        """检查令牌是否过期"""
        if not credentials.get('expires_at'):
            return False
        
        expires_at = credentials['expires_at']
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        
        # 提前5分钟过期
        return expires_at < datetime.utcnow() + timedelta(minutes=5)
    
    async def _refresh_access_token(self, account_id: int):
        """刷新访问令牌"""
        # 这里需要实现令牌刷新逻辑
        # 不同平台的刷新方式可能不同
        pass
    
    async def _get_account_info_from_platform(self, platform_name: str, access_token: str) -> Dict[str, Any]:
        """从平台获取账号信息"""
        # 这里需要调用各平台的API获取账号信息
        # 返回标准化的账号信息格式
        return {
            'id': 'mock_account_id',
            'username': 'mock_user',
            'name': 'Mock User',
            'display_name': 'Mock Display Name',
            'avatar_url': 'https://example.com/avatar.jpg',
            'bio': 'Mock bio',
            'followers_count': 1000,
            'following_count': 100,
            'statuses_count': 50,
            'verified': False
        }
    
    async def _get_account_stats_from_platform(self, platform_name: str, access_token: str) -> Dict[str, Any]:
        """从平台获取账号统计数据"""
        # 这里需要调用各平台的API获取统计数据
        return {
            'followers_count': 1000,
            'following_count': 100,
            'statuses_count': 50
        }
    
    async def _update_account_profile(self, account_id: int, profile_data: Dict[str, Any]):
        """更新账号资料"""
        await self.db.execute(
            update(Account)
            .where(Account.id == account_id)
            .values(
                display_name=profile_data.get('display_name'),
                avatar_url=profile_data.get('avatar_url'),
                bio=profile_data.get('bio'),
                verification_status='verified' if profile_data.get('verified') else 'unverified',
                updated_at=func.now()
            )
        )
    
    async def _update_account_stats(self, account_id: int, stats_data: Dict[str, Any]):
        """更新账号统计数据"""
        await self.db.execute(
            update(Account)
            .where(Account.id == account_id)
            .values(
                follower_count=stats_data.get('followers_count', 0),
                following_count=stats_data.get('following_count', 0),
                post_count=stats_data.get('statuses_count', 0),
                updated_at=func.now()
            )
        )
    
    async def _update_account_sync_time(self, account_id: int):
        """更新账号同步时间"""
        await self.db.execute(
            update(Account)
            .where(Account.id == account_id)
            .values(last_sync_at=func.now())
        )
    
    async def _grant_permission(self, account_id: int, user_id: int, permission_type: str, 
                              expires_at: datetime = None):
        """授予权限"""
        permission = AccountPermission(
            account_id=account_id,
            user_id=user_id,
            permission_type=permission_type,
            expires_at=expires_at,
            is_active=True
        )
        self.db.add(permission)
    
    async def _get_user_permissions(self, account_id: int, user_id: int) -> List[str]:
        """获取用户权限"""
        query = select(AccountPermission.permission_type).where(
            and_(
                AccountPermission.account_id == account_id,
                AccountPermission.user_id == user_id,
                AccountPermission.is_active == True
            )
        )
        
        result = await self.db.execute(query)
        permissions = result.scalars().all()
        
        return list(permissions)
    
    async def _check_user_permission(self, account_id: int, user_id: int, permission_type: str) -> bool:
        """检查用户权限"""
        permissions = await self._get_user_permissions(account_id, user_id)
        return permission_type in permissions or 'admin' in permissions
    
    async def _cache_account_info(self, account: Account):
        """缓存账号信息"""
        cache_key = f"account:{account.id}"
        account_data = {
            'id': account.id,
            'platform_name': account.platform.name,
            'account_name': account.account_name,
            'display_name': account.display_name,
            'status': account.status,
            'last_sync': account.last_sync_at.isoformat() if account.last_sync_at else None
        }
        
        await self.redis.setex(cache_key, 3600, json.dumps(account_data, default=str))
    
    async def _clear_account_cache(self, account_id: int):
        """清除账号缓存"""
        cache_keys = [
            f"account:{account_id}",
            f"user_accounts:*"  # 清除相关用户账号列表缓存
        ]
        
        for pattern in cache_keys:
            if '*' in pattern:
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
            else:
                await self.redis.delete(pattern)