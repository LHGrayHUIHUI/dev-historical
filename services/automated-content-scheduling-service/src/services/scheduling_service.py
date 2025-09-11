"""
核心内容调度服务
提供智能调度算法、冲突检测、任务管理等核心功能
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update
from sqlalchemy.orm import selectinload

from ..models import (
    SchedulingTask, SchedulingConflict, TaskExecutionLog, OptimizationLog,
    TaskStatus, TaskType, ConflictType, ConflictSeverity,
    get_db_session
)
from ..config.settings import get_settings
from .optimization_service import OptimizationService
from .conflict_detection_service import ConflictDetectionService
from .platform_integration_service import PlatformIntegrationService

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SchedulingRequest:
    """调度请求数据类"""
    user_id: int
    content_id: str
    title: str
    content_body: str
    target_platforms: List[str]
    preferred_time: Optional[datetime] = None
    task_type: TaskType = TaskType.SINGLE
    recurrence_rule: Optional[str] = None
    priority: int = 5
    optimization_enabled: bool = True
    platform_configs: Optional[Dict[str, Any]] = None
    content_metadata: Optional[Dict[str, Any]] = None


@dataclass
class SchedulingResult:
    """调度结果数据类"""
    task_id: UUID
    scheduled_time: datetime
    conflicts: List[Dict[str, Any]]
    optimization_applied: bool
    optimization_score: Optional[float] = None
    warnings: List[str] = None
    platform_specific_times: Optional[Dict[str, datetime]] = None


class SchedulingStrategy(str, Enum):
    """调度策略枚举"""
    IMMEDIATE = "immediate"           # 立即执行
    OPTIMAL_TIME = "optimal_time"     # 最优时间
    USER_PREFERRED = "user_preferred" # 用户偏好
    LOAD_BALANCED = "load_balanced"   # 负载均衡
    CONFLICT_AVOIDANCE = "conflict_avoidance"  # 冲突避免


class ContentSchedulingService:
    """内容调度核心服务类"""
    
    def __init__(self):
        self.settings = get_settings()
        self.optimization_service = OptimizationService()
        self.conflict_service = ConflictDetectionService()
        self.platform_service = PlatformIntegrationService()
        
    async def schedule_content(
        self,
        request: SchedulingRequest,
        strategy: SchedulingStrategy = SchedulingStrategy.OPTIMAL_TIME
    ) -> SchedulingResult:
        """
        调度内容发布任务
        
        Args:
            request: 调度请求
            strategy: 调度策略
            
        Returns:
            SchedulingResult: 调度结果
        """
        try:
            logger.info(f"开始调度内容任务，用户ID: {request.user_id}, 策略: {strategy}")
            
            async with get_db_session() as session:
                # 1. 创建初始任务
                task = await self._create_initial_task(session, request)
                
                # 2. 根据策略确定调度时间
                scheduled_time = await self._determine_scheduled_time(
                    session, request, strategy
                )
                
                # 3. 应用智能优化（如果启用）
                optimization_applied = False
                optimization_score = None
                
                if request.optimization_enabled:
                    optimized_result = await self.optimization_service.optimize_scheduling_time(
                        user_id=request.user_id,
                        platforms=request.target_platforms,
                        content_metadata=request.content_metadata or {},
                        preferred_time=scheduled_time
                    )
                    
                    if optimized_result.optimized_time != scheduled_time:
                        scheduled_time = optimized_result.optimized_time
                        optimization_applied = True
                        optimization_score = optimized_result.optimization_score
                        
                        # 记录优化日志
                        await self._log_optimization(
                            session, task.id, request.preferred_time or scheduled_time,
                            optimized_result
                        )
                
                # 4. 检测和处理冲突
                conflicts = await self._detect_and_handle_conflicts(
                    session, task, scheduled_time
                )
                
                # 5. 更新任务信息
                await self._update_task_scheduling(
                    session, task, scheduled_time, optimization_applied,
                    optimization_score
                )
                
                # 6. 生成平台特定时间（如果需要）
                platform_times = await self._generate_platform_specific_times(
                    request.target_platforms, scheduled_time, conflicts
                )
                
                await session.commit()
                
                result = SchedulingResult(
                    task_id=task.id,
                    scheduled_time=scheduled_time,
                    conflicts=[conflict.to_dict() for conflict in conflicts],
                    optimization_applied=optimization_applied,
                    optimization_score=optimization_score,
                    platform_specific_times=platform_times,
                    warnings=self._generate_warnings(conflicts, optimization_applied)
                )
                
                logger.info(f"内容调度完成，任务ID: {task.id}, 调度时间: {scheduled_time}")
                return result
                
        except Exception as e:
            logger.error(f"内容调度失败: {e}")
            raise
    
    async def _create_initial_task(
        self, 
        session: AsyncSession, 
        request: SchedulingRequest
    ) -> SchedulingTask:
        """创建初始调度任务"""
        task = SchedulingTask(
            user_id=request.user_id,
            title=request.title,
            content_id=request.content_id,
            content_body=request.content_body,
            content_metadata=request.content_metadata,
            task_type=request.task_type,
            target_platforms=request.target_platforms,
            platform_configs=request.platform_configs,
            recurrence_rule=request.recurrence_rule,
            priority=request.priority,
            optimization_enabled=request.optimization_enabled,
            scheduled_time=request.preferred_time or datetime.utcnow(),
            status=TaskStatus.PENDING
        )
        
        session.add(task)
        await session.flush()  # 获取task.id
        return task
    
    async def _determine_scheduled_time(
        self,
        session: AsyncSession,
        request: SchedulingRequest,
        strategy: SchedulingStrategy
    ) -> datetime:
        """根据策略确定调度时间"""
        
        if strategy == SchedulingStrategy.IMMEDIATE:
            return datetime.utcnow()
        
        if strategy == SchedulingStrategy.USER_PREFERRED and request.preferred_time:
            return request.preferred_time
        
        if strategy == SchedulingStrategy.OPTIMAL_TIME:
            # 使用ML模型预测最优时间
            optimal_time = await self.optimization_service.predict_optimal_time(
                user_id=request.user_id,
                platforms=request.target_platforms,
                content_metadata=request.content_metadata or {}
            )
            return optimal_time
        
        if strategy == SchedulingStrategy.LOAD_BALANCED:
            # 基于系统负载选择时间
            return await self._find_load_balanced_time(session, request)
        
        if strategy == SchedulingStrategy.CONFLICT_AVOIDANCE:
            # 避免冲突的时间选择
            return await self._find_conflict_free_time(session, request)
        
        # 默认返回用户偏好时间或当前时间
        return request.preferred_time or datetime.utcnow()
    
    async def _find_load_balanced_time(
        self,
        session: AsyncSession,
        request: SchedulingRequest
    ) -> datetime:
        """寻找负载均衡的调度时间"""
        
        # 查询未来24小时的任务分布
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)
        
        # 按小时统计任务数量
        hour_counts = {}
        
        result = await session.execute(
            select(
                func.extract('hour', SchedulingTask.scheduled_time).label('hour'),
                func.count(SchedulingTask.id).label('count')
            ).where(
                and_(
                    SchedulingTask.scheduled_time >= start_time,
                    SchedulingTask.scheduled_time <= end_time,
                    SchedulingTask.status.in_([TaskStatus.PENDING, TaskStatus.SCHEDULED])
                )
            ).group_by(
                func.extract('hour', SchedulingTask.scheduled_time)
            )
        )
        
        for hour, count in result:
            hour_counts[int(hour)] = count
        
        # 找到任务数最少的小时
        current_hour = start_time.hour
        min_count = float('inf')
        best_hour = current_hour
        
        for i in range(24):
            hour = (current_hour + i) % 24
            count = hour_counts.get(hour, 0)
            if count < min_count:
                min_count = count
                best_hour = hour
        
        # 计算最终调度时间
        if best_hour >= current_hour:
            target_time = start_time.replace(
                hour=best_hour, minute=0, second=0, microsecond=0
            )
        else:
            target_time = (start_time + timedelta(days=1)).replace(
                hour=best_hour, minute=0, second=0, microsecond=0
            )
        
        return target_time
    
    async def _find_conflict_free_time(
        self,
        session: AsyncSession,
        request: SchedulingRequest
    ) -> datetime:
        """寻找无冲突的调度时间"""
        
        base_time = request.preferred_time or datetime.utcnow()
        
        # 检查前后24小时内的时间窗口
        for offset_minutes in range(0, 24 * 60, 30):  # 每30分钟检查一次
            candidate_time = base_time + timedelta(minutes=offset_minutes)
            
            # 检查该时间是否有冲突
            conflicts = await self.conflict_service.detect_conflicts(
                session, request.user_id, request.target_platforms, candidate_time
            )
            
            if not conflicts:
                return candidate_time
        
        # 如果找不到无冲突时间，返回用户偏好时间
        logger.warning(f"无法找到无冲突的调度时间，用户ID: {request.user_id}")
        return base_time
    
    async def _detect_and_handle_conflicts(
        self,
        session: AsyncSession,
        task: SchedulingTask,
        scheduled_time: datetime
    ) -> List[SchedulingConflict]:
        """检测并处理调度冲突"""
        
        conflicts = await self.conflict_service.detect_conflicts(
            session, task.user_id, task.target_platforms, scheduled_time, task.id
        )
        
        # 为每个冲突创建记录
        conflict_records = []
        for conflict_data in conflicts:
            conflict = SchedulingConflict(
                task_id=task.id,
                conflict_type=ConflictType(conflict_data['type']),
                severity=ConflictSeverity(conflict_data['severity']),
                description=conflict_data['description'],
                conflicted_task_id=conflict_data.get('conflicted_task_id'),
                conflicted_resource=conflict_data.get('conflicted_resource'),
                platform_name=conflict_data.get('platform_name'),
                conflict_details=conflict_data.get('details', {}),
                suggested_resolution=conflict_data.get('resolution', {})
            )
            
            session.add(conflict)
            conflict_records.append(conflict)
        
        return conflict_records
    
    async def _log_optimization(
        self,
        session: AsyncSession,
        task_id: UUID,
        original_time: datetime,
        optimization_result
    ):
        """记录优化过程"""
        
        optimization_log = OptimizationLog(
            task_id=task_id,
            optimization_type="ml_time_optimization",
            original_scheduled_time=original_time,
            optimized_scheduled_time=optimization_result.optimized_time,
            optimization_score=optimization_result.optimization_score,
            predicted_engagement=optimization_result.predicted_metrics.get('engagement_rate'),
            predicted_reach=optimization_result.predicted_metrics.get('reach'),
            confidence_score=optimization_result.confidence_score,
            optimization_factors=optimization_result.optimization_factors,
            model_version=optimization_result.model_version,
            model_params=optimization_result.model_params
        )
        
        session.add(optimization_log)
    
    async def _update_task_scheduling(
        self,
        session: AsyncSession,
        task: SchedulingTask,
        scheduled_time: datetime,
        optimization_applied: bool,
        optimization_score: Optional[float]
    ):
        """更新任务调度信息"""
        
        task.scheduled_time = scheduled_time
        task.status = TaskStatus.SCHEDULED
        
        if optimization_applied:
            task.optimization_status = task.optimization_status.OPTIMIZED
            task.optimization_score = optimization_score
        
        await session.flush()
    
    async def _generate_platform_specific_times(
        self,
        platforms: List[str],
        base_time: datetime,
        conflicts: List[SchedulingConflict]
    ) -> Dict[str, datetime]:
        """生成平台特定的发布时间"""
        
        platform_times = {}
        
        # 如果有平台特定的冲突，需要调整时间
        for platform in platforms:
            platform_time = base_time
            
            # 检查该平台是否有特定冲突需要调整时间
            for conflict in conflicts:
                if (conflict.platform_name == platform and 
                    conflict.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]):
                    
                    # 根据冲突建议调整时间
                    if conflict.suggested_resolution:
                        suggested_time = conflict.suggested_resolution.get('suggested_time')
                        if suggested_time:
                            platform_time = datetime.fromisoformat(suggested_time)
                            break
            
            platform_times[platform] = platform_time
        
        return platform_times
    
    def _generate_warnings(
        self,
        conflicts: List[SchedulingConflict],
        optimization_applied: bool
    ) -> List[str]:
        """生成警告信息"""
        
        warnings = []
        
        # 冲突相关警告
        high_severity_conflicts = [
            c for c in conflicts 
            if c.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]
        ]
        
        if high_severity_conflicts:
            warnings.append(f"检测到 {len(high_severity_conflicts)} 个高优先级冲突")
        
        # 优化相关警告
        if not optimization_applied:
            warnings.append("智能优化未被应用，可能影响发布效果")
        
        return warnings
    
    async def reschedule_task(
        self,
        task_id: UUID,
        new_scheduled_time: datetime,
        reason: str = "用户手动重新调度"
    ) -> SchedulingResult:
        """重新调度任务"""
        
        try:
            async with get_db_session() as session:
                # 获取任务
                result = await session.execute(
                    select(SchedulingTask)
                    .options(selectinload(SchedulingTask.conflicts))
                    .where(SchedulingTask.id == task_id)
                )
                task = result.scalar_one_or_none()
                
                if not task:
                    raise ValueError(f"未找到任务: {task_id}")
                
                if task.status not in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
                    raise ValueError(f"任务状态不允许重新调度: {task.status}")
                
                # 记录原始时间
                original_time = task.scheduled_time
                
                # 清除现有冲突
                await session.execute(
                    update(SchedulingConflict)
                    .where(SchedulingConflict.task_id == task_id)
                    .values(is_resolved=True, resolution_method=f"重新调度: {reason}")
                )
                
                # 检测新时间的冲突
                conflicts = await self._detect_and_handle_conflicts(
                    session, task, new_scheduled_time
                )
                
                # 更新任务
                task.scheduled_time = new_scheduled_time
                task.original_scheduled_time = original_time
                
                await session.commit()
                
                return SchedulingResult(
                    task_id=task.id,
                    scheduled_time=new_scheduled_time,
                    conflicts=[conflict.to_dict() for conflict in conflicts],
                    optimization_applied=False,
                    warnings=self._generate_warnings(conflicts, False)
                )
                
        except Exception as e:
            logger.error(f"重新调度任务失败: {e}")
            raise
    
    async def cancel_task(self, task_id: UUID, reason: str = "用户取消") -> bool:
        """取消调度任务"""
        
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(SchedulingTask).where(SchedulingTask.id == task_id)
                )
                task = result.scalar_one_or_none()
                
                if not task:
                    return False
                
                if task.status == TaskStatus.RUNNING:
                    raise ValueError("正在执行的任务无法取消")
                
                task.status = TaskStatus.CANCELLED
                
                # 记录取消日志
                log_entry = TaskExecutionLog(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                    error_message=f"任务被取消: {reason}",
                    execution_start_time=datetime.utcnow(),
                    execution_end_time=datetime.utcnow(),
                    execution_duration=0
                )
                
                session.add(log_entry)
                await session.commit()
                
                logger.info(f"任务已取消: {task_id}, 原因: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            raise
    
    async def get_user_scheduled_tasks(
        self,
        user_id: int,
        status_filter: Optional[List[TaskStatus]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取用户的调度任务列表"""
        
        try:
            async with get_db_session() as session:
                query = select(SchedulingTask).where(SchedulingTask.user_id == user_id)
                
                if status_filter:
                    query = query.where(SchedulingTask.status.in_(status_filter))
                
                query = query.order_by(SchedulingTask.scheduled_time.desc())
                query = query.offset(offset).limit(limit)
                
                result = await session.execute(query)
                tasks = result.scalars().all()
                
                return [self._task_to_dict(task) for task in tasks]
                
        except Exception as e:
            logger.error(f"获取用户任务列表失败: {e}")
            raise
    
    def _task_to_dict(self, task: SchedulingTask) -> Dict[str, Any]:
        """将任务对象转换为字典"""
        return {
            "id": str(task.id),
            "title": task.title,
            "content_id": task.content_id,
            "task_type": task.task_type.value,
            "status": task.status.value,
            "scheduled_time": task.scheduled_time.isoformat(),
            "target_platforms": task.target_platforms,
            "priority": task.priority,
            "optimization_enabled": task.optimization_enabled,
            "optimization_score": task.optimization_score,
            "created_time": task.created_time.isoformat(),
            "updated_time": task.updated_time.isoformat()
        }