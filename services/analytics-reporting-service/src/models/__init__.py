"""
分析报告服务数据模型模块

包含所有数据库模型、Pydantic模型和数据传输对象的初始化。
支持PostgreSQL关系数据模型、InfluxDB时序数据模型和
Redis缓存数据结构。
"""

from .database import get_db, init_database
from .analytics_models import (
    AnalysisTask,
    AnalysisTaskStatus,
    AnalysisTaskType,
    ReportTemplate,
    GeneratedReport,
    DataSource,
    AlertRule,
    AlertSeverity
)
from .schemas import (
    # 分析任务相关
    AnalysisTaskCreate,
    AnalysisTaskUpdate,
    AnalysisTaskResponse,
    AnalysisTaskList,
    
    # 报告相关
    ReportTemplateCreate,
    ReportTemplateUpdate,
    ReportTemplateResponse,
    GeneratedReportResponse,
    ReportExportRequest,
    
    # 数据源相关
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceResponse,
    
    # 告警相关
    AlertRuleCreate,
    AlertRuleUpdate,
    AlertRuleResponse,
    AlertTrigger,
    
    # 分析结果相关
    AnalyticsMetrics,
    ContentPerformance,
    PlatformComparison,
    TrendAnalysis,
    UserBehaviorInsights,
    
    # 时序数据相关
    TimeSeriesPoint,
    MetricQuery,
    AggregationResult,
    
    # 可视化相关
    ChartConfig,
    DashboardConfig,
    VisualizationRequest,
    
    # 导出相关
    ExportFormat,
    ExportRequest,
    ExportResponse,
    
    # 公共响应
    BaseResponse,
    PaginationParams,
    PaginatedResponse
)

__all__ = [
    # 数据库
    "get_db",
    "init_database",
    
    # 数据库模型
    "AnalysisTask",
    "AnalysisTaskStatus", 
    "AnalysisTaskType",
    "ReportTemplate",
    "GeneratedReport",
    "DataSource",
    "AlertRule",
    "AlertSeverity",
    
    # Pydantic模式
    "AnalysisTaskCreate",
    "AnalysisTaskUpdate",
    "AnalysisTaskResponse",
    "AnalysisTaskList",
    "ReportTemplateCreate",
    "ReportTemplateUpdate", 
    "ReportTemplateResponse",
    "GeneratedReportResponse",
    "ReportExportRequest",
    "DataSourceCreate",
    "DataSourceUpdate",
    "DataSourceResponse",
    "AlertRuleCreate",
    "AlertRuleUpdate",
    "AlertRuleResponse",
    "AlertTrigger",
    "AnalyticsMetrics",
    "ContentPerformance",
    "PlatformComparison",
    "TrendAnalysis",
    "UserBehaviorInsights",
    "TimeSeriesPoint",
    "MetricQuery",
    "AggregationResult",
    "ChartConfig",
    "DashboardConfig",
    "VisualizationRequest",
    "ExportFormat",
    "ExportRequest",
    "ExportResponse",
    "BaseResponse",
    "PaginationParams",
    "PaginatedResponse"
]