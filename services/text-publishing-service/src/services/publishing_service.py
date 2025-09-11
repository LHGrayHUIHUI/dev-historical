"""
文本发送服务核心类

负责管理多平台内容发布的统一接口
提供任务创建、状态管理、账号选择等核心功能
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from fastapi import HTTPException

from ..models import (
    PublishingPlatform, PublishingAccount, PublishingTask, PublishingStats
)
from ..models.schemas import (
    TaskCreateSchema, TaskSchema, TaskStatus, AccountStatus
)
from .redis_service import RedisService
from .account_manager import AccountManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class PublishingService:
    """
    文本发送服务核心类
    
    负责管理多平台内容发布的统一接口
    提供任务创建、状态查询、取消等功能
    """
    
    def __init__(self, db_session: AsyncSession, redis_service: RedisService):
        self.db = db_session
        self.redis = redis_service
        self.account_manager = AccountManager(db_session, redis_service)
        self.platform_adapters = {}
        self._load_platform_adapters()
    
    def _load_platform_adapters(self):
        """
        加载各平台适配器
        
        动态加载所有平台适配器实例
        """
        try:
            # 延迟导入避免循环依赖
            from ..adapters import (
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
            
            logger.info(f"已加载 {len(self.platform_adapters)} 个平台适配器")
            
        except ImportError as e:
            logger.warning(f"部分平台适配器加载失败: {e}")
            self.platform_adapters = {}
    
    async def create_publishing_task(
        self,
        task_data: TaskCreateSchema,
        user_id: Optional[int] = None
    ) -> List[str]:
        """
        创建发布任务
        
        Args:
            task_data: 任务创建数据
            user_id: 用户ID
            
        Returns:
            List[str]: 任务UUID列表
        """
        task_uuids = []
        
        try:
            # 验证平台支持
            await self._validate_platforms(task_data.platforms)
            
            # 为每个平台创建任务
            for platform_name in task_data.platforms:
                # 获取平台信息
                platform = await self._get_platform_by_name(platform_name)
                if not platform:
                    logger.warning(f"平台不存在: {platform_name}")
                    continue
                
                # 检查平台限流
                if not await self.redis.check_platform_rate_limit(
                    platform.id, platform.rate_limit_per_hour
                ):
                    raise HTTPException(
                        status_code=429,
                        detail=f"平台 {platform_name} 达到限流限制"
                    )
                
                # 选择可用账号
                account = await self.account_manager.select_available_account(platform_name)
                if not account:
                    logger.warning(f"平台 {platform_name} 无可用账号")
                    continue
                
                # 创建任务记录
                task_uuid = await self._create_task_record(
                    platform, account, task_data, user_id
                )
                
                if task_uuid:
                    task_uuids.append(task_uuid)
                    
                    # 增加平台使用计数
                    await self.redis.increment_platform_usage(platform.id)
                    
                    # 提交到任务队列
                    await self._submit_task_to_queue(task_uuid, task_data.scheduled_at)
            
            if not task_uuids:
                raise HTTPException(
                    status_code=400,
                    detail="无法为任何平台创建任务，请检查平台状态和账号配置"
                )
            
            logger.info(f"成功创建 {len(task_uuids)} 个发布任务")
            return task_uuids
            
        except Exception as e:
            logger.error(f"创建发布任务失败: {e}")
            raise
    
    async def _validate_platforms(self, platforms: List[str]):
        """
        验证平台列表
        
        Args:
            platforms: 平台名称列表
        """
        if len(platforms) > settings.max_platforms_per_task:
            raise HTTPException(
                status_code=400,
                detail=f"单次最多支持 {settings.max_platforms_per_task} 个平台"
            )
        
        # 查询平台是否存在且激活
        stmt = select(PublishingPlatform).where(
            and_(
                PublishingPlatform.platform_name.in_(platforms),
                PublishingPlatform.is_active == True
            )
        )
        result = await self.db.execute(stmt)
        active_platforms = result.scalars().all()
        
        active_platform_names = {p.platform_name for p in active_platforms}
        unsupported_platforms = set(platforms) - active_platform_names
        
        if unsupported_platforms:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的平台: {', '.join(unsupported_platforms)}"
            )
    
    async def _get_platform_by_name(self, platform_name: str) -> Optional[PublishingPlatform]:
        """
        根据名称获取平台信息
        
        Args:
            platform_name: 平台名称
            
        Returns:
            PublishingPlatform: 平台信息
        """
        stmt = select(PublishingPlatform).where(
            PublishingPlatform.platform_name == platform_name
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _create_task_record(
        self,
        platform: PublishingPlatform,
        account: PublishingAccount,
        task_data: TaskCreateSchema,
        user_id: Optional[int] = None
    ) -> Optional[str]:
        """
        创建任务记录
        
        Args:
            platform: 平台信息
            account: 账号信息
            task_data: 任务数据
            user_id: 用户ID
            
        Returns:
            str: 任务UUID
        """
        try:
            # 创建任务实例
            task = PublishingTask(
                content_id=task_data.content_id,
                platform_id=platform.id,
                account_id=account.id,
                title=task_data.title,
                content=task_data.content,
                media_urls=task_data.media_urls,
                publish_config=task_data.publish_config.dict() if task_data.publish_config else None,
                scheduled_at=task_data.scheduled_at,
                status=TaskStatus.PENDING
            )
            
            # 保存到数据库
            self.db.add(task)
            await self.db.flush()  # 获取ID但不提交
            
            # 初始化Redis状态缓存
            await self.redis.set_task_status(
                str(task.task_uuid),
                TaskStatus.PENDING,
                0,
                "任务已创建，等待处理"
            )
            
            logger.info(f"创建任务记录成功: {task.task_uuid}")
            return str(task.task_uuid)
            
        except Exception as e:
            logger.error(f"创建任务记录失败: {e}")
            return None
    
    async def _submit_task_to_queue(
        self, 
        task_uuid: str, 
        scheduled_at: Optional[datetime] = None
    ):
        """
        提交任务到队列
        
        Args:
            task_uuid: 任务UUID
            scheduled_at: 计划执行时间
        """
        try:
            # 这里应该集成Celery任务队列
            # 由于当前环境可能没有Celery，先记录日志
            if scheduled_at and scheduled_at > datetime.utcnow():
                logger.info(f"任务 {task_uuid} 已安排在 {scheduled_at} 执行")
                # celery_app.send_task('publish_content', args=[task_uuid], eta=scheduled_at)
            else:
                logger.info(f"任务 {task_uuid} 已提交立即执行")
                # celery_app.send_task('publish_content', args=[task_uuid])
                
        except Exception as e:
            logger.error(f"提交任务到队列失败: {e}")
    
    async def get_task_status(self, task_uuid: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_uuid: 任务UUID
            
        Returns:
            Dict[str, Any]: 任务状态信息
        """
        try:
            # 先从Redis获取实时状态
            cached_status = await self.redis.get_task_status(task_uuid)
            if cached_status:
                return cached_status
            
            # 从数据库获取
            stmt = select(
                PublishingTask,
                PublishingPlatform.platform_name,
                PublishingAccount.account_name
            ).join(
                PublishingPlatform, PublishingTask.platform_id == PublishingPlatform.id
            ).join(
                PublishingAccount, PublishingTask.account_id == PublishingAccount.id  
            ).where(
                PublishingTask.task_uuid == UUID(task_uuid)
            )
            
            result = await self.db.execute(stmt)
            row = result.first()
            
            if not row:
                raise HTTPException(status_code=404, detail="任务不存在")
            
            task, platform_name, account_name = row
            
            return {
                'task_uuid': task_uuid,
                'platform': platform_name,
                'account': account_name,
                'status': task.status,
                'title': task.title,
                'content': task.content,
                'scheduled_at': task.scheduled_at.isoformat() if task.scheduled_at else None,
                'published_at': task.published_at.isoformat() if task.published_at else None,
                'published_url': task.published_url,
                'error_message': task.error_message,
                'retry_count': task.retry_count,
                'max_retries': task.max_retries,
                'created_at': task.created_at.isoformat(),
                'updated_at': task.updated_at.isoformat()
            }
            
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的任务UUID格式")
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            raise HTTPException(status_code=500, detail="获取任务状态失败")
    
    async def cancel_task(self, task_uuid: str) -> bool:
        """
        取消发布任务
        
        Args:
            task_uuid: 任务UUID
            
        Returns:
            bool: 是否成功取消
        """
        try:
            # 更新数据库状态
            stmt = select(PublishingTask).where(
                PublishingTask.task_uuid == UUID(task_uuid)
            )
            result = await self.db.execute(stmt)
            task = result.scalar_one_or_none()
            
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            
            if not task.can_cancel:
                raise HTTPException(status_code=400, detail="任务无法取消")
            
            # 更新状态
            task.status = TaskStatus.CANCELLED
            task.updated_at = func.now()
            
            # 更新Redis缓存
            await self.redis.set_task_status(
                task_uuid,
                TaskStatus.CANCELLED,
                0,
                "任务已取消"
            )
            
            # 撤销Celery任务
            # celery_app.control.revoke(task_uuid, terminate=True)
            
            await self.db.commit()
            logger.info(f"任务 {task_uuid} 已取消")
            return True
            
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的任务UUID格式")
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False
    
    async def get_task_list(
        self,
        status: Optional[str] = None,
        platform: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        获取任务列表
        
        Args:
            status: 状态过滤
            platform: 平台过滤
            start_date: 开始日期
            end_date: 结束日期
            page: 页码
            page_size: 每页数量
            
        Returns:
            Dict[str, Any]: 任务列表和分页信息
        """
        try:
            # 构建查询条件
            conditions = []
            
            if status:
                conditions.append(PublishingTask.status == status)
            
            if platform:
                conditions.append(PublishingPlatform.platform_name == platform)
            
            if start_date:
                conditions.append(PublishingTask.created_at >= start_date)
            
            if end_date:
                conditions.append(PublishingTask.created_at <= end_date)
            
            where_clause = and_(*conditions) if conditions else None
            
            # 查询总数
            count_stmt = select(func.count(PublishingTask.id)).select_from(
                PublishingTask.__table__.join(
                    PublishingPlatform.__table__,
                    PublishingTask.platform_id == PublishingPlatform.id
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
                PublishingTask,
                PublishingPlatform.platform_name,
                PublishingAccount.account_name
            ).join(
                PublishingPlatform, PublishingTask.platform_id == PublishingPlatform.id
            ).join(
                PublishingAccount, PublishingTask.account_id == PublishingAccount.id
            )
            
            if where_clause is not None:
                stmt = stmt.where(where_clause)
            
            stmt = stmt.order_by(desc(PublishingTask.created_at)).offset(offset).limit(page_size)
            
            result = await self.db.execute(stmt)
            rows = result.all()
            
            # 构建任务列表
            tasks = []
            for task, platform_name, account_name in rows:
                task_dict = {
                    'id': task.id,
                    'task_uuid': str(task.task_uuid),
                    'platform': platform_name,
                    'account': account_name,
                    'title': task.title,
                    'content': task.content[:100] + '...' if len(task.content) > 100 else task.content,
                    'status': task.status,
                    'scheduled_at': task.scheduled_at.isoformat() if task.scheduled_at else None,
                    'published_at': task.published_at.isoformat() if task.published_at else None,
                    'published_url': task.published_url,
                    'error_message': task.error_message,
                    'retry_count': task.retry_count,
                    'created_at': task.created_at.isoformat(),
                    'updated_at': task.updated_at.isoformat()
                }
                tasks.append(task_dict)
            
            return {
                'tasks': tasks,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total': total,
                    'pages': pages
                }
            }
            
        except Exception as e:
            logger.error(f"获取任务列表失败: {e}")
            raise HTTPException(status_code=500, detail="获取任务列表失败")
    
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
        try:
            # 构建查询条件
            conditions = [
                PublishingTask.created_at >= start_date,
                PublishingTask.created_at <= end_date
            ]
            
            if platform:
                conditions.append(PublishingPlatform.platform_name == platform)
            
            # 统计查询
            stmt = select(
                PublishingPlatform.platform_name,
                func.count(PublishingTask.id).label('total_tasks'),
                func.count(
                    func.case((PublishingTask.status == TaskStatus.PUBLISHED, 1))
                ).label('successful_tasks'),
                func.count(
                    func.case((PublishingTask.status == TaskStatus.FAILED, 1))
                ).label('failed_tasks'),
                func.avg(
                    func.case((
                        PublishingTask.published_at.is_not(None),
                        func.extract('epoch', 
                            PublishingTask.published_at - PublishingTask.created_at
                        )
                    ))
                ).label('avg_publish_time')
            ).select_from(
                PublishingTask.__table__.join(
                    PublishingPlatform.__table__,
                    PublishingTask.platform_id == PublishingPlatform.id
                )
            ).where(
                and_(*conditions)
            ).group_by(
                PublishingPlatform.platform_name
            ).order_by(
                desc('total_tasks')
            )
            
            result = await self.db.execute(stmt)
            stats = result.all()
            
            # 构建结果
            platform_stats = []
            for row in stats:
                success_rate = 0
                if row.total_tasks > 0:
                    success_rate = round((row.successful_tasks / row.total_tasks) * 100, 2)
                
                platform_stats.append({
                    'platform': row.platform_name,
                    'total_tasks': row.total_tasks,
                    'successful_tasks': row.successful_tasks,
                    'failed_tasks': row.failed_tasks,
                    'success_rate': success_rate,
                    'avg_publish_time': round(row.avg_publish_time or 0, 2)
                })
            
            return {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'platforms': platform_stats
            }
            
        except Exception as e:
            logger.error(f"获取统计数据失败: {e}")
            raise HTTPException(status_code=500, detail="获取统计数据失败")