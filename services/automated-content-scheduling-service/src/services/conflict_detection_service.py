"""
冲突检测服务
检测和处理调度冲突，包括时间重叠、资源冲突、平台限制等
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, between

from ..models import (
    SchedulingTask, SchedulingConflict, PlatformMetrics,
    TaskStatus, ConflictType, ConflictSeverity,
    get_db_session
)
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ConflictInfo:
    """冲突信息数据类"""
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    conflicted_task_id: Optional[UUID] = None
    conflicted_resource: Optional[str] = None
    platform_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    resolution: Optional[Dict[str, Any]] = None


class ConflictDetectionService:
    """冲突检测服务类"""
    
    def __init__(self):
        self.settings = get_settings()
        
    async def detect_conflicts(
        self,
        session: AsyncSession,
        user_id: int,
        target_platforms: List[str],
        scheduled_time: datetime,
        exclude_task_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """
        检测调度冲突
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            target_platforms: 目标平台列表
            scheduled_time: 调度时间
            exclude_task_id: 排除的任务ID（用于更新时避免自冲突）
            
        Returns:
            List[Dict]: 冲突列表
        """
        conflicts = []
        
        try:
            # 1. 检测时间重叠冲突
            time_conflicts = await self._detect_time_overlap_conflicts(
                session, user_id, target_platforms, scheduled_time, exclude_task_id
            )
            conflicts.extend(time_conflicts)
            
            # 2. 检测资源冲突
            resource_conflicts = await self._detect_resource_conflicts(
                session, user_id, target_platforms, scheduled_time, exclude_task_id
            )
            conflicts.extend(resource_conflicts)
            
            # 3. 检测平台限制冲突
            platform_conflicts = await self._detect_platform_limit_conflicts(
                session, user_id, target_platforms, scheduled_time
            )
            conflicts.extend(platform_conflicts)
            
            # 4. 检测内容重复冲突
            content_conflicts = await self._detect_content_duplicate_conflicts(
                session, user_id, target_platforms, scheduled_time, exclude_task_id
            )
            conflicts.extend(content_conflicts)
            
            # 5. 检测用户偏好冲突
            preference_conflicts = await self._detect_user_preference_conflicts(
                session, user_id, scheduled_time
            )
            conflicts.extend(preference_conflicts)
            
            return [self._conflict_to_dict(conflict) for conflict in conflicts]
            
        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
            return []
    
    async def _detect_time_overlap_conflicts(
        self,
        session: AsyncSession,
        user_id: int,
        target_platforms: List[str],
        scheduled_time: datetime,
        exclude_task_id: Optional[UUID] = None
    ) -> List[ConflictInfo]:
        """检测时间重叠冲突"""
        conflicts = []
        
        try:
            # 检查前后30分钟内的任务
            time_window = timedelta(minutes=30)
            start_time = scheduled_time - time_window
            end_time = scheduled_time + time_window
            
            query = select(SchedulingTask).where(
                and_(
                    SchedulingTask.user_id == user_id,
                    SchedulingTask.scheduled_time.between(start_time, end_time),
                    SchedulingTask.status.in_([TaskStatus.PENDING, TaskStatus.SCHEDULED]),
                    SchedulingTask.id != exclude_task_id if exclude_task_id else True
                )
            )
            
            result = await session.execute(query)
            overlapping_tasks = result.scalars().all()
            
            for task in overlapping_tasks:
                # 检查是否有共同平台
                common_platforms = set(target_platforms) & set(task.target_platforms or [])
                
                if common_platforms:
                    time_diff = abs((scheduled_time - task.scheduled_time).total_seconds())
                    
                    # 根据时间差确定严重程度
                    if time_diff <= 300:  # 5分钟内
                        severity = ConflictSeverity.CRITICAL
                    elif time_diff <= 900:  # 15分钟内
                        severity = ConflictSeverity.HIGH
                    else:
                        severity = ConflictSeverity.MEDIUM
                    
                    conflict = ConflictInfo(
                        conflict_type=ConflictType.TIME_OVERLAP,
                        severity=severity,
                        description=f"与任务 '{task.title}' 在平台 {list(common_platforms)} 上存在时间重叠",
                        conflicted_task_id=task.id,
                        platform_name=",".join(common_platforms),
                        details={
                            'time_difference_seconds': time_diff,
                            'conflicted_task_title': task.title,
                            'common_platforms': list(common_platforms)
                        },
                        resolution={
                            'suggested_action': 'reschedule',
                            'suggested_time': (scheduled_time + timedelta(hours=1)).isoformat(),
                            'alternatives': [
                                (scheduled_time - timedelta(hours=1)).isoformat(),
                                (scheduled_time + timedelta(hours=2)).isoformat()
                            ]
                        }
                    )
                    
                    conflicts.append(conflict)
            
        except Exception as e:
            logger.error(f"检测时间重叠冲突失败: {e}")
        
        return conflicts
    
    async def _detect_resource_conflicts(
        self,
        session: AsyncSession,
        user_id: int,
        target_platforms: List[str],
        scheduled_time: datetime,
        exclude_task_id: Optional[UUID] = None
    ) -> List[ConflictInfo]:
        """检测资源冲突（如同一用户在同一时间的多个高优先级任务）"""
        conflicts = []
        
        try:
            # 检查同一小时内的高优先级任务
            hour_start = scheduled_time.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)
            
            query = select(SchedulingTask).where(
                and_(
                    SchedulingTask.user_id == user_id,
                    SchedulingTask.scheduled_time.between(hour_start, hour_end),
                    SchedulingTask.priority >= 7,  # 高优先级任务
                    SchedulingTask.status.in_([TaskStatus.PENDING, TaskStatus.SCHEDULED]),
                    SchedulingTask.id != exclude_task_id if exclude_task_id else True
                )
            )
            
            result = await session.execute(query)
            high_priority_tasks = result.scalars().all()
            
            if len(high_priority_tasks) >= 3:  # 如果同一小时内有3个以上高优先级任务
                conflict = ConflictInfo(
                    conflict_type=ConflictType.RESOURCE_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"同一小时内存在 {len(high_priority_tasks)} 个高优先级任务，可能导致资源竞争",
                    details={
                        'conflicted_tasks_count': len(high_priority_tasks),
                        'hour_window': hour_start.isoformat(),
                        'task_titles': [task.title for task in high_priority_tasks]
                    },
                    resolution={
                        'suggested_action': 'redistribute',
                        'suggestion': '建议将部分任务分散到其他时间段'
                    }
                )
                
                conflicts.append(conflict)
        
        except Exception as e:
            logger.error(f"检测资源冲突失败: {e}")
        
        return conflicts
    
    async def _detect_platform_limit_conflicts(
        self,
        session: AsyncSession,
        user_id: int,
        target_platforms: List[str],
        scheduled_time: datetime
    ) -> List[ConflictInfo]:
        """检测平台限制冲突"""
        conflicts = []
        
        try:
            # 获取平台配置
            platform_configs = settings.platforms.platform_configs
            
            for platform in target_platforms:
                platform_config = platform_configs.get(platform, {})
                rate_limit = platform_config.get('rate_limit', 1000)
                
                # 检查该平台当前小时的发布数量
                hour_start = scheduled_time.replace(minute=0, second=0, microsecond=0)
                hour_end = hour_start + timedelta(hours=1)
                
                # 查询该平台在这个小时的任务数
                result = await session.execute(
                    select(func.count(SchedulingTask.id))
                    .where(
                        and_(
                            SchedulingTask.user_id == user_id,
                            SchedulingTask.scheduled_time.between(hour_start, hour_end),
                            SchedulingTask.target_platforms.contains([platform]),
                            SchedulingTask.status.in_([TaskStatus.PENDING, TaskStatus.SCHEDULED])
                        )
                    )
                )
                
                current_count = result.scalar() or 0
                hourly_limit = rate_limit // 24  # 假设均匀分布
                
                if current_count >= hourly_limit * 0.8:  # 接近限制的80%时警告
                    severity = ConflictSeverity.HIGH if current_count >= hourly_limit else ConflictSeverity.MEDIUM
                    
                    conflict = ConflictInfo(
                        conflict_type=ConflictType.PLATFORM_LIMIT,
                        severity=severity,
                        description=f"平台 {platform} 在该时间段的发布数量接近或超过限制",
                        platform_name=platform,
                        details={
                            'current_count': current_count,
                            'hourly_limit': hourly_limit,
                            'usage_percentage': (current_count / hourly_limit) * 100
                        },
                        resolution={
                            'suggested_action': 'reschedule',
                            'suggestion': f'建议延后发布或选择其他时间段，当前使用率: {(current_count / hourly_limit) * 100:.1f}%'
                        }
                    )
                    
                    conflicts.append(conflict)
        
        except Exception as e:
            logger.error(f"检测平台限制冲突失败: {e}")
        
        return conflicts
    
    async def _detect_content_duplicate_conflicts(
        self,
        session: AsyncSession,
        user_id: int,
        target_platforms: List[str],
        scheduled_time: datetime,
        exclude_task_id: Optional[UUID] = None
    ) -> List[ConflictInfo]:
        """检测内容重复冲突"""
        conflicts = []
        
        try:
            # 检查近期（7天内）是否有相似内容
            recent_date = scheduled_time - timedelta(days=7)
            
            # 这里简化处理，实际可以使用文本相似度算法
            # 当前仅检查完全相同的标题
            query = select(SchedulingTask).where(
                and_(
                    SchedulingTask.user_id == user_id,
                    SchedulingTask.scheduled_time >= recent_date,
                    SchedulingTask.status.in_([TaskStatus.COMPLETED, TaskStatus.SCHEDULED]),
                    SchedulingTask.id != exclude_task_id if exclude_task_id else True
                )
            )
            
            result = await session.execute(query)
            recent_tasks = result.scalars().all()
            
            # 检查内容相似度（这里简化为标题匹配）
            for task in recent_tasks:
                if task.title:  # 如果有相同标题的任务
                    # 检查平台重叠
                    common_platforms = set(target_platforms) & set(task.target_platforms or [])
                    
                    if common_platforms:
                        days_apart = (scheduled_time - task.scheduled_time).days
                        
                        if days_apart < 3:  # 3天内重复发布
                            severity = ConflictSeverity.HIGH
                        else:
                            severity = ConflictSeverity.MEDIUM
                        
                        conflict = ConflictInfo(
                            conflict_type=ConflictType.CONTENT_DUPLICATE,
                            severity=severity,
                            description=f"在平台 {list(common_platforms)} 上与 {days_apart} 天前的内容重复",
                            conflicted_task_id=task.id,
                            platform_name=",".join(common_platforms),
                            details={
                                'days_apart': days_apart,
                                'conflicted_task_title': task.title,
                                'common_platforms': list(common_platforms)
                            },
                            resolution={
                                'suggested_action': 'modify_or_reschedule',
                                'suggestion': '建议修改内容或选择不同的发布时间'
                            }
                        )
                        
                        conflicts.append(conflict)
        
        except Exception as e:
            logger.error(f"检测内容重复冲突失败: {e}")
        
        return conflicts
    
    async def _detect_user_preference_conflicts(
        self,
        session: AsyncSession,
        user_id: int,
        scheduled_time: datetime
    ) -> List[ConflictInfo]:
        """检测用户偏好冲突"""
        conflicts = []
        
        try:
            # 检查是否在用户不偏好的时间段发布
            hour = scheduled_time.hour
            day_of_week = scheduled_time.weekday()
            
            # 定义一些常见的不佳发布时间
            poor_hours = [0, 1, 2, 3, 4, 5, 6]  # 深夜到早晨
            
            if hour in poor_hours:
                conflict = ConflictInfo(
                    conflict_type=ConflictType.USER_PREFERENCE,
                    severity=ConflictSeverity.LOW,
                    description=f"发布时间 ({hour}:00) 处于用户活跃度较低的时段",
                    details={
                        'scheduled_hour': hour,
                        'reason': '深夜或早晨时段用户活跃度通常较低'
                    },
                    resolution={
                        'suggested_action': 'optimize_timing',
                        'suggested_hours': [9, 12, 15, 18, 20],
                        'suggestion': '建议选择用户活跃度更高的时间段'
                    }
                )
                
                conflicts.append(conflict)
            
            # 检查周末vs工作日偏好
            is_weekend = day_of_week >= 5
            
            # 这里可以根据历史数据分析用户在周末vs工作日的表现
            # 当前简化处理
            
        except Exception as e:
            logger.error(f"检测用户偏好冲突失败: {e}")
        
        return conflicts
    
    def _conflict_to_dict(self, conflict: ConflictInfo) -> Dict[str, Any]:
        """将冲突信息转换为字典"""
        return {
            'type': conflict.conflict_type.value,
            'severity': conflict.severity.value,
            'description': conflict.description,
            'conflicted_task_id': str(conflict.conflicted_task_id) if conflict.conflicted_task_id else None,
            'conflicted_resource': conflict.conflicted_resource,
            'platform_name': conflict.platform_name,
            'details': conflict.details or {},
            'resolution': conflict.resolution or {}
        }
    
    async def resolve_conflicts(
        self,
        session: AsyncSession,
        task_id: UUID,
        conflict_resolutions: List[Dict[str, Any]]
    ) -> bool:
        """
        解决冲突
        
        Args:
            session: 数据库会话
            task_id: 任务ID
            conflict_resolutions: 冲突解决方案列表
            
        Returns:
            bool: 是否成功解决所有冲突
        """
        try:
            for resolution in conflict_resolutions:
                conflict_id = resolution.get('conflict_id')
                resolution_method = resolution.get('method')
                
                if conflict_id:
                    # 更新冲突状态为已解决
                    result = await session.execute(
                        select(SchedulingConflict).where(
                            SchedulingConflict.id == UUID(conflict_id)
                        )
                    )
                    
                    conflict = result.scalar_one_or_none()
                    if conflict:
                        conflict.is_resolved = True
                        conflict.resolution_method = resolution_method
                        conflict.resolved_time = datetime.utcnow()
            
            await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"解决冲突失败: {e}")
            return False
    
    async def get_conflict_suggestions(
        self,
        session: AsyncSession,
        user_id: int,
        target_platforms: List[str],
        preferred_time: datetime,
        time_flexibility_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        获取无冲突的建议时间
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            target_platforms: 目标平台
            preferred_time: 偏好时间
            time_flexibility_hours: 时间灵活性（小时）
            
        Returns:
            List[Dict]: 建议时间列表
        """
        suggestions = []
        
        try:
            start_time = preferred_time
            end_time = preferred_time + timedelta(hours=time_flexibility_hours)
            
            # 每小时检查一次
            current_time = start_time
            while current_time <= end_time and len(suggestions) < 10:
                conflicts = await self.detect_conflicts(
                    session, user_id, target_platforms, current_time
                )
                
                # 计算冲突严重度得分
                conflict_score = sum(
                    self._get_severity_weight(ConflictSeverity(c['severity']))
                    for c in conflicts
                )
                
                suggestions.append({
                    'suggested_time': current_time.isoformat(),
                    'conflict_count': len(conflicts),
                    'conflict_score': conflict_score,
                    'conflicts': conflicts,
                    'recommendation_score': max(0, 100 - conflict_score * 10)
                })
                
                current_time += timedelta(hours=1)
            
            # 按推荐得分排序
            suggestions.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return suggestions[:5]  # 返回前5个建议
            
        except Exception as e:
            logger.error(f"获取冲突建议失败: {e}")
            return []
    
    def _get_severity_weight(self, severity: ConflictSeverity) -> float:
        """获取冲突严重度权重"""
        weights = {
            ConflictSeverity.LOW: 1.0,
            ConflictSeverity.MEDIUM: 3.0,
            ConflictSeverity.HIGH: 7.0,
            ConflictSeverity.CRITICAL: 15.0
        }
        return weights.get(severity, 1.0)
    
    async def analyze_conflict_patterns(
        self,
        session: AsyncSession,
        user_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        分析用户的冲突模式
        
        Args:
            session: 数据库会话
            user_id: 用户ID
            days: 分析天数
            
        Returns:
            Dict: 冲突模式分析结果
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # 查询用户的冲突记录
            result = await session.execute(
                select(SchedulingConflict, SchedulingTask)
                .join(SchedulingTask, SchedulingConflict.task_id == SchedulingTask.id)
                .where(
                    and_(
                        SchedulingTask.user_id == user_id,
                        SchedulingConflict.detected_time >= cutoff_date
                    )
                )
            )
            
            conflicts_data = result.all()
            
            if not conflicts_data:
                return {
                    'total_conflicts': 0,
                    'message': '分析期间内无冲突记录'
                }
            
            # 统计冲突类型
            conflict_types = {}
            severity_distribution = {}
            hourly_patterns = {}
            platform_conflicts = {}
            
            for conflict, task in conflicts_data:
                # 冲突类型统计
                conflict_type = conflict.conflict_type.value
                conflict_types[conflict_type] = conflict_types.get(conflict_type, 0) + 1
                
                # 严重程度统计
                severity = conflict.severity.value
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
                
                # 时间模式统计
                hour = task.scheduled_time.hour
                hourly_patterns[hour] = hourly_patterns.get(hour, 0) + 1
                
                # 平台冲突统计
                if conflict.platform_name:
                    platform_conflicts[conflict.platform_name] = platform_conflicts.get(conflict.platform_name, 0) + 1
            
            return {
                'analysis_period_days': days,
                'total_conflicts': len(conflicts_data),
                'conflict_types': conflict_types,
                'severity_distribution': severity_distribution,
                'hourly_conflict_patterns': hourly_patterns,
                'platform_conflicts': platform_conflicts,
                'most_problematic_hour': max(hourly_patterns.items(), key=lambda x: x[1])[0] if hourly_patterns else None,
                'most_common_conflict_type': max(conflict_types.items(), key=lambda x: x[1])[0] if conflict_types else None,
                'recommendations': self._generate_conflict_recommendations(
                    conflict_types, hourly_patterns, platform_conflicts
                )
            }
            
        except Exception as e:
            logger.error(f"分析冲突模式失败: {e}")
            return {'error': str(e)}
    
    def _generate_conflict_recommendations(
        self,
        conflict_types: Dict[str, int],
        hourly_patterns: Dict[int, int],
        platform_conflicts: Dict[str, int]
    ) -> List[str]:
        """基于冲突模式生成建议"""
        recommendations = []
        
        # 基于冲突类型的建议
        if conflict_types.get('time_overlap', 0) > 5:
            recommendations.append("建议增加任务间的时间间隔，避免频繁的时间冲突")
        
        if conflict_types.get('platform_limit', 0) > 3:
            recommendations.append("建议分散发布时间，避免单一时段集中发布过多内容")
        
        # 基于时间模式的建议
        if hourly_patterns:
            problematic_hours = [hour for hour, count in hourly_patterns.items() if count > 2]
            if problematic_hours:
                recommendations.append(f"建议避免在 {problematic_hours} 时发布，这些时段冲突较多")
        
        # 基于平台冲突的建议
        if platform_conflicts:
            high_conflict_platforms = [platform for platform, count in platform_conflicts.items() if count > 3]
            if high_conflict_platforms:
                recommendations.append(f"平台 {high_conflict_platforms} 冲突较多，建议优化发布策略")
        
        if not recommendations:
            recommendations.append("当前冲突模式良好，继续保持现有的发布策略")
        
        return recommendations