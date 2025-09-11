"""
数据模型包初始化
导出所有数据模型以便其他模块使用
"""
from .database import Base, get_db, get_db_session, db_manager, init_database, close_database

# 导入所有模型以确保它们被SQLAlchemy注册
from .scheduling_models import (
    SchedulingTask,
    SchedulingTemplate,
    SchedulingConflict,
    TaskExecutionLog,
    OptimizationLog,
    SchedulingQueue,
    # 枚举类型
    TaskStatus,
    TaskType,
    ConflictType,
    ConflictSeverity,
    OptimizationStatus
)

from .analytics_models import (
    PlatformMetrics,
    ContentPerformance,
    UserBehaviorPattern,
    MLModelMetrics,
    SystemMetrics
)

# 导出所有模型和相关类
__all__ = [
    # 数据库相关
    "Base",
    "get_db",
    "get_db_session",
    "db_manager",
    "init_database", 
    "close_database",
    
    # 调度模型
    "SchedulingTask",
    "SchedulingTemplate",
    "SchedulingConflict",
    "TaskExecutionLog",
    "OptimizationLog",
    "SchedulingQueue",
    
    # 分析模型
    "PlatformMetrics",
    "ContentPerformance", 
    "UserBehaviorPattern",
    "MLModelMetrics",
    "SystemMetrics",
    
    # 枚举类型
    "TaskStatus",
    "TaskType",
    "ConflictType",
    "ConflictSeverity",
    "OptimizationStatus",
]