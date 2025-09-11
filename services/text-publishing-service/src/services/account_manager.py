"""
账号管理器

负责发布账号的选择、配额管理和状态维护
提供智能负载均衡和故障转移功能
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc, update
from fastapi import HTTPException

from ..models import PublishingPlatform, PublishingAccount
from ..models.schemas import AccountStatus
from .redis_service import RedisService
from ..config.settings import settings

logger = logging.getLogger(__name__)


class AccountManager:
    """
    账号管理器类
    
    负责账号选择、配额管理、状态监控等功能
    """
    
    def __init__(self, db_session: AsyncSession, redis_service: RedisService):
        self.db = db_session
        self.redis = redis_service
    
    async def select_available_account(
        self, 
        platform_name: str,
        exclude_accounts: Optional[List[int]] = None
    ) -> Optional[PublishingAccount]:
        """
        选择可用的发布账号
        
        使用智能负载均衡算法，优先选择使用次数少的活跃账号
        
        Args:
            platform_name: 平台名称
            exclude_accounts: 要排除的账号ID列表
            
        Returns:
            PublishingAccount: 可用账号，如果没有则返回None
        """
        try:
            # 构建查询条件
            conditions = [
                PublishingPlatform.platform_name == platform_name,
                PublishingPlatform.is_active == True,
                PublishingAccount.account_status == AccountStatus.ACTIVE,
                PublishingAccount.used_quota < PublishingAccount.daily_quota
            ]
            
            if exclude_accounts:
                conditions.append(~PublishingAccount.id.in_(exclude_accounts))
            
            # 查询可用账号，按使用次数和最后使用时间排序
            stmt = select(PublishingAccount).join(
                PublishingPlatform,
                PublishingAccount.platform_id == PublishingPlatform.id
            ).where(
                and_(*conditions)
            ).order_by(
                asc(PublishingAccount.used_quota),
                asc(PublishingAccount.last_used_at).nulls_first()
            ).limit(5)  # 获取前5个候选账号
            
            result = await self.db.execute(stmt)
            candidates = result.scalars().all()
            
            if not candidates:
                logger.warning(f"平台 {platform_name} 无可用账号")
                return None
            
            # 使用Redis检查实时配额
            for account in candidates:
                if await self._check_account_realtime_quota(account):
                    logger.info(f"选择账号 {account.account_name} (ID: {account.id})")
                    return account
            
            logger.warning(f"平台 {platform_name} 所有账号配额已用完")
            return None
            
        except Exception as e:
            logger.error(f"选择账号失败: {e}")
            return None
    
    async def _check_account_realtime_quota(self, account: PublishingAccount) -> bool:
        """
        检查账号实时配额
        
        Args:
            account: 账号实例
            
        Returns:
            bool: 是否有可用配额
        """
        try:
            # 从Redis获取实时配额信息
            quota_info = await self.redis.get_account_quota(account.id)
            
            if not quota_info:
                # 如果Redis中没有信息，初始化
                await self.redis.set_account_quota(
                    account.id,
                    account.daily_quota,
                    account.used_quota
                )
                quota_info = {
                    'daily_limit': account.daily_quota,
                    'used_today': account.used_quota
                }
            
            # 检查是否超出配额
            used_today = quota_info.get('used_today', 0)
            daily_limit = quota_info.get('daily_limit', account.daily_quota)
            
            # 检查过期时间
            if account.expires_at and account.expires_at <= datetime.utcnow():
                logger.warning(f"账号 {account.account_name} 已过期")
                await self._deactivate_expired_account(account)
                return False
            
            return used_today < daily_limit
            
        except Exception as e:
            logger.error(f"检查账号配额失败: {e}")
            return False
    
    async def _deactivate_expired_account(self, account: PublishingAccount):
        """
        停用过期账号
        
        Args:
            account: 账号实例
        """
        try:
            account.account_status = AccountStatus.EXPIRED
            await self.db.commit()
            logger.info(f"账号 {account.account_name} 已设为过期状态")
        except Exception as e:
            logger.error(f"停用过期账号失败: {e}")
    
    async def update_account_usage(
        self, 
        account_id: int, 
        increment: int = 1,
        success: bool = True
    ):
        """
        更新账号使用次数
        
        Args:
            account_id: 账号ID
            increment: 增加次数
            success: 是否成功
        """
        try:
            # 更新Redis计数
            await self.redis.increment_account_usage(account_id, increment)
            
            # 更新数据库
            stmt = update(PublishingAccount).where(
                PublishingAccount.id == account_id
            ).values(
                used_quota=PublishingAccount.used_quota + increment,
                last_used_at=func.now()
            )
            
            await self.db.execute(stmt)
            
            logger.debug(f"账号 {account_id} 使用次数已更新: +{increment}")
            
        except Exception as e:
            logger.error(f"更新账号使用次数失败: {e}")
    
    async def get_account_list(
        self,
        platform: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        获取账号列表
        
        Args:
            platform: 平台过滤
            status: 状态过滤
            page: 页码
            page_size: 每页数量
            
        Returns:
            Dict[str, Any]: 账号列表和分页信息
        """
        try:
            # 构建查询条件
            conditions = []
            
            if platform:
                conditions.append(PublishingPlatform.platform_name == platform)
            
            if status:
                conditions.append(PublishingAccount.account_status == status)
            
            where_clause = and_(*conditions) if conditions else None
            
            # 查询总数
            count_stmt = select(func.count(PublishingAccount.id)).select_from(
                PublishingAccount.__table__.join(
                    PublishingPlatform.__table__,
                    PublishingAccount.platform_id == PublishingPlatform.id
                )
            )
            if where_clause is not None:
                count_stmt = count_stmt.where(where_clause)
            
            total_result = await self.db.execute(count_stmt)
            total = total_result.scalar()
            
            # 计算分页
            offset = (page - 1) * page_size
            pages = (total + page_size - 1) // page_size
            
            # 查询数据
            stmt = select(
                PublishingAccount,
                PublishingPlatform.platform_name
            ).join(
                PublishingPlatform,
                PublishingAccount.platform_id == PublishingPlatform.id
            )
            
            if where_clause is not None:
                stmt = stmt.where(where_clause)
            
            stmt = stmt.order_by(
                desc(PublishingAccount.updated_at)
            ).offset(offset).limit(page_size)
            
            result = await self.db.execute(stmt)
            rows = result.all()
            
            # 构建账号列表
            accounts = []
            for account, platform_name in rows:
                # 获取实时配额信息
                quota_info = await self.redis.get_account_quota(account.id)
                used_today = quota_info.get('used_today', account.used_quota) if quota_info else account.used_quota
                
                account_dict = {
                    'id': account.id,
                    'platform': platform_name,
                    'account_name': account.account_name,
                    'account_identifier': account.account_identifier,
                    'account_status': account.account_status,
                    'daily_quota': account.daily_quota,
                    'used_quota': used_today,
                    'quota_remaining': max(0, account.daily_quota - used_today),
                    'last_used_at': account.last_used_at.isoformat() if account.last_used_at else None,
                    'expires_at': account.expires_at.isoformat() if account.expires_at else None,
                    'created_at': account.created_at.isoformat(),
                    'updated_at': account.updated_at.isoformat(),
                    'is_available': account.is_available and used_today < account.daily_quota
                }
                accounts.append(account_dict)
            
            return {
                'accounts': accounts,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total': total,
                    'pages': pages
                }
            }
            
        except Exception as e:
            logger.error(f"获取账号列表失败: {e}")
            raise HTTPException(status_code=500, detail="获取账号列表失败")
    
    async def create_account(self, account_data: Dict[str, Any]) -> int:
        """
        创建发布账号
        
        Args:
            account_data: 账号数据
            
        Returns:
            int: 账号ID
        """
        try:
            # 验证平台存在
            platform_stmt = select(PublishingPlatform).where(
                PublishingPlatform.id == account_data['platform_id']
            )
            platform_result = await self.db.execute(platform_stmt)
            platform = platform_result.scalar_one_or_none()
            
            if not platform:
                raise HTTPException(status_code=400, detail="平台不存在")
            
            # 检查账号是否已存在
            if account_data.get('account_identifier'):
                existing_stmt = select(PublishingAccount).where(
                    and_(
                        PublishingAccount.platform_id == account_data['platform_id'],
                        PublishingAccount.account_identifier == account_data['account_identifier']
                    )
                )
                existing_result = await self.db.execute(existing_stmt)
                existing_account = existing_result.scalar_one_or_none()
                
                if existing_account:
                    raise HTTPException(status_code=400, detail="该账号已存在")
            
            # 创建账号
            account = PublishingAccount(
                platform_id=account_data['platform_id'],
                account_name=account_data['account_name'],
                account_identifier=account_data.get('account_identifier'),
                auth_credentials=account_data.get('auth_credentials', {}),
                daily_quota=account_data.get('daily_quota', 50),
                expires_at=account_data.get('expires_at')
            )
            
            self.db.add(account)
            await self.db.flush()
            
            # 初始化Redis配额
            await self.redis.set_account_quota(
                account.id,
                account.daily_quota,
                0
            )
            
            await self.db.commit()
            logger.info(f"成功创建账号: {account.account_name}")
            
            return account.id
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"创建账号失败: {e}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="创建账号失败")
    
    async def update_account(
        self, 
        account_id: int, 
        update_data: Dict[str, Any]
    ) -> bool:
        """
        更新账号信息
        
        Args:
            account_id: 账号ID
            update_data: 更新数据
            
        Returns:
            bool: 是否更新成功
        """
        try:
            # 查询账号
            stmt = select(PublishingAccount).where(
                PublishingAccount.id == account_id
            )
            result = await self.db.execute(stmt)
            account = result.scalar_one_or_none()
            
            if not account:
                raise HTTPException(status_code=404, detail="账号不存在")
            
            # 更新字段
            update_fields = {}
            if 'account_name' in update_data:
                update_fields['account_name'] = update_data['account_name']
            if 'account_identifier' in update_data:
                update_fields['account_identifier'] = update_data['account_identifier']
            if 'auth_credentials' in update_data:
                update_fields['auth_credentials'] = update_data['auth_credentials']
            if 'account_status' in update_data:
                update_fields['account_status'] = update_data['account_status']
            if 'daily_quota' in update_data:
                update_fields['daily_quota'] = update_data['daily_quota']
            if 'expires_at' in update_data:
                update_fields['expires_at'] = update_data['expires_at']
            
            if update_fields:
                update_fields['updated_at'] = func.now()
                
                update_stmt = update(PublishingAccount).where(
                    PublishingAccount.id == account_id
                ).values(**update_fields)
                
                await self.db.execute(update_stmt)
                
                # 更新Redis配额信息
                if 'daily_quota' in update_data:
                    await self.redis.set_account_quota(
                        account_id,
                        update_data['daily_quota'],
                        account.used_quota
                    )
                
                await self.db.commit()
                logger.info(f"账号 {account_id} 更新成功")
                return True
            
            return False
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"更新账号失败: {e}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="更新账号失败")
    
    async def delete_account(self, account_id: int) -> bool:
        """
        删除账号
        
        Args:
            account_id: 账号ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            # 检查账号是否存在
            stmt = select(PublishingAccount).where(
                PublishingAccount.id == account_id
            )
            result = await self.db.execute(stmt)
            account = result.scalar_one_or_none()
            
            if not account:
                raise HTTPException(status_code=404, detail="账号不存在")
            
            # 检查是否有未完成的任务
            from ..models import PublishingTask
            task_stmt = select(func.count(PublishingTask.id)).where(
                and_(
                    PublishingTask.account_id == account_id,
                    PublishingTask.status.in_(['pending', 'processing'])
                )
            )
            task_result = await self.db.execute(task_stmt)
            pending_tasks = task_result.scalar()
            
            if pending_tasks > 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"账号有 {pending_tasks} 个未完成任务，无法删除"
                )
            
            # 删除账号
            await self.db.delete(account)
            
            # 清除Redis数据
            quota_key = f"account_quota:{account_id}"
            await self.redis.delete_cache(quota_key)
            
            await self.db.commit()
            logger.info(f"账号 {account_id} 已删除")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"删除账号失败: {e}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="删除账号失败")
    
    async def reset_daily_quotas(self):
        """
        重置所有账号的每日配额
        
        通常在每日凌晨执行
        """
        try:
            # 重置数据库中的使用次数
            reset_stmt = update(PublishingAccount).values(
                used_quota=0,
                updated_at=func.now()
            )
            result = await self.db.execute(reset_stmt)
            affected_rows = result.rowcount
            
            # 清除Redis中的配额缓存
            # 获取所有账号ID
            account_stmt = select(PublishingAccount.id)
            account_result = await self.db.execute(account_stmt)
            account_ids = [row.id for row in account_result.fetchall()]
            
            # 批量删除Redis缓存
            if account_ids:
                quota_keys = [f"account_quota:{aid}" for aid in account_ids]
                await self.redis.delete_multiple(quota_keys)
            
            await self.db.commit()
            logger.info(f"已重置 {affected_rows} 个账号的每日配额")
            
        except Exception as e:
            logger.error(f"重置每日配额失败: {e}")
            await self.db.rollback()
    
    async def get_account_health_status(self) -> Dict[str, Any]:
        """
        获取账号健康状态统计
        
        Returns:
            Dict[str, Any]: 健康状态统计
        """
        try:
            # 统计各状态账号数量
            status_stmt = select(
                PublishingAccount.account_status,
                func.count(PublishingAccount.id).label('count')
            ).group_by(PublishingAccount.account_status)
            
            status_result = await self.db.execute(status_stmt)
            status_stats = {row.account_status: row.count for row in status_result.fetchall()}
            
            # 统计即将过期的账号
            tomorrow = datetime.utcnow() + timedelta(days=1)
            week_later = datetime.utcnow() + timedelta(days=7)
            
            expiring_stmt = select(
                func.count(PublishingAccount.id).label('expiring_soon'),
                func.count(
                    func.case((PublishingAccount.expires_at <= tomorrow, 1))
                ).label('expiring_tomorrow')
            ).where(
                and_(
                    PublishingAccount.expires_at.is_not(None),
                    PublishingAccount.expires_at <= week_later
                )
            )
            
            expiring_result = await self.db.execute(expiring_stmt)
            expiring_stats = expiring_result.first()
            
            # 统计配额使用情况
            quota_stmt = select(
                func.count(PublishingAccount.id).label('total_accounts'),
                func.count(
                    func.case((PublishingAccount.used_quota >= PublishingAccount.daily_quota, 1))
                ).label('quota_exhausted'),
                func.avg(
                    func.cast(PublishingAccount.used_quota, 'float') / 
                    func.case((PublishingAccount.daily_quota > 0, PublishingAccount.daily_quota), else_=1) * 100
                ).label('avg_quota_usage')
            ).where(PublishingAccount.account_status == AccountStatus.ACTIVE)
            
            quota_result = await self.db.execute(quota_stmt)
            quota_stats = quota_result.first()
            
            return {
                'status_distribution': status_stats,
                'expiring_accounts': {
                    'expiring_within_week': expiring_stats.expiring_soon or 0,
                    'expiring_tomorrow': expiring_stats.expiring_tomorrow or 0
                },
                'quota_usage': {
                    'total_active_accounts': quota_stats.total_accounts or 0,
                    'quota_exhausted_accounts': quota_stats.quota_exhausted or 0,
                    'average_quota_usage': round(quota_stats.avg_quota_usage or 0, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"获取账号健康状态失败: {e}")
            return {}