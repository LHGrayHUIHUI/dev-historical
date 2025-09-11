"""
分析报告服务Pydantic模式定义

包含所有API请求/响应模式、数据传输对象和验证规则。
提供类型安全的数据序列化和反序列化功能。
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from pydantic import BaseModel, Field, validator

from .analytics_models import (
    AnalysisTaskType, 
    AnalysisTaskStatus,
    AlertSeverity
)


# ===== 基础响应模式 =====

class BaseResponse(BaseModel):
    """基础API响应模式"""
    success: bool = Field(default=True, description="请求是否成功")
    message: str = Field(default="操作成功", description="响应消息")
    code: int = Field(default=200, description="业务状态码")


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


class PaginatedResponse(BaseResponse):
    """分页响应模式"""
    total: int = Field(description="总记录数")
    page: int = Field(description="当前页码") 
    page_size: int = Field(description="每页大小")
    total_pages: int = Field(description="总页数")


# ===== 分析任务相关模式 =====

class AnalysisTaskCreate(BaseModel):
    """创建分析任务请求"""
    title: str = Field(..., min_length=1, max_length=255, description="任务标题")
    description: Optional[str] = Field(None, description="任务描述")
    task_type: AnalysisTaskType = Field(..., description="任务类型")
    
    # 配置信息
    config: Optional[Dict[str, Any]] = Field(default={}, description="任务配置")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="分析参数")
    filters: Optional[Dict[str, Any]] = Field(default={}, description="过滤条件")
    
    # 时间范围
    start_date: Optional[datetime] = Field(None, description="分析开始时间")
    end_date: Optional[datetime] = Field(None, description="分析结束时间")
    
    # 执行配置
    scheduled_at: Optional[datetime] = Field(None, description="计划执行时间")
    priority: int = Field(default=5, ge=1, le=10, description="优先级")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """验证时间范围"""
        if v and values.get('start_date') and v <= values['start_date']:
            raise ValueError('结束时间必须晚于开始时间')
        return v


class AnalysisTaskUpdate(BaseModel):
    """更新分析任务请求"""
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="任务标题")
    description: Optional[str] = Field(None, description="任务描述")
    status: Optional[AnalysisTaskStatus] = Field(None, description="任务状态")
    
    config: Optional[Dict[str, Any]] = Field(None, description="任务配置")
    parameters: Optional[Dict[str, Any]] = Field(None, description="分析参数")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    
    scheduled_at: Optional[datetime] = Field(None, description="计划执行时间")
    priority: Optional[int] = Field(None, ge=1, le=10, description="优先级")


class AnalysisTaskResponse(BaseModel):
    """分析任务响应"""
    id: uuid.UUID = Field(description="任务ID")
    title: str = Field(description="任务标题")
    description: Optional[str] = Field(description="任务描述")
    task_type: AnalysisTaskType = Field(description="任务类型")
    status: AnalysisTaskStatus = Field(description="任务状态")
    
    user_id: str = Field(description="用户ID")
    created_by: Optional[str] = Field(description="创建者")
    
    progress: int = Field(description="执行进度(0-100)")
    priority: int = Field(description="优先级")
    
    start_date: Optional[datetime] = Field(description="分析开始时间")
    end_date: Optional[datetime] = Field(description="分析结束时间")
    
    scheduled_at: Optional[datetime] = Field(description="计划执行时间")
    started_at: Optional[datetime] = Field(description="实际开始时间")
    completed_at: Optional[datetime] = Field(description="完成时间")
    
    result_data: Optional[Dict[str, Any]] = Field(description="分析结果")
    error_message: Optional[str] = Field(description="错误信息")
    
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    
    class Config:
        from_attributes = True


class AnalysisTaskList(PaginatedResponse):
    """分析任务列表响应"""
    data: List[AnalysisTaskResponse] = Field(description="任务列表")


# ===== 报告模板相关模式 =====

class ReportTemplateCreate(BaseModel):
    """创建报告模板请求"""
    name: str = Field(..., min_length=1, max_length=255, description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    category: Optional[str] = Field(None, description="模板分类")
    
    template_config: Dict[str, Any] = Field(..., description="模板配置")
    chart_configs: Optional[Dict[str, Any]] = Field(default={}, description="图表配置")
    layout_config: Optional[Dict[str, Any]] = Field(default={}, description="布局配置")
    
    data_sources: Optional[Dict[str, Any]] = Field(default={}, description="数据源配置")
    default_filters: Optional[Dict[str, Any]] = Field(default={}, description="默认过滤器")
    
    theme: str = Field(default="default", description="主题")
    custom_styles: Optional[Dict[str, Any]] = Field(default={}, description="自定义样式")
    
    is_public: bool = Field(default=False, description="是否公共模板")


class ReportTemplateUpdate(BaseModel):
    """更新报告模板请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    category: Optional[str] = Field(None, description="模板分类")
    
    template_config: Optional[Dict[str, Any]] = Field(None, description="模板配置")
    chart_configs: Optional[Dict[str, Any]] = Field(None, description="图表配置")
    layout_config: Optional[Dict[str, Any]] = Field(None, description="布局配置")
    
    data_sources: Optional[Dict[str, Any]] = Field(None, description="数据源配置")
    default_filters: Optional[Dict[str, Any]] = Field(None, description="默认过滤器")
    
    theme: Optional[str] = Field(None, description="主题")
    custom_styles: Optional[Dict[str, Any]] = Field(None, description="自定义样式")
    
    is_public: Optional[bool] = Field(None, description="是否公共模板")


class ReportTemplateResponse(BaseModel):
    """报告模板响应"""
    id: uuid.UUID = Field(description="模板ID")
    name: str = Field(description="模板名称")
    description: Optional[str] = Field(description="模板描述")
    category: Optional[str] = Field(description="模板分类")
    
    user_id: Optional[str] = Field(description="所属用户ID")
    is_public: bool = Field(description="是否公共模板")
    
    template_config: Dict[str, Any] = Field(description="模板配置")
    chart_configs: Optional[Dict[str, Any]] = Field(description="图表配置")
    layout_config: Optional[Dict[str, Any]] = Field(description="布局配置")
    
    theme: str = Field(description="主题")
    usage_count: int = Field(description="使用次数")
    
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    
    class Config:
        from_attributes = True


# ===== 生成报告相关模式 =====

class ReportExportRequest(BaseModel):
    """报告导出请求"""
    analysis_task_id: uuid.UUID = Field(description="分析任务ID")
    template_id: Optional[uuid.UUID] = Field(None, description="模板ID")
    title: str = Field(..., min_length=1, max_length=255, description="报告标题")
    description: Optional[str] = Field(None, description="报告描述")
    format: str = Field(default="pdf", description="导出格式")


class GeneratedReportResponse(BaseModel):
    """生成报告响应"""
    id: uuid.UUID = Field(description="报告ID")
    title: str = Field(description="报告标题")
    description: Optional[str] = Field(description="报告描述")
    
    user_id: str = Field(description="用户ID")
    
    analysis_task_id: Optional[uuid.UUID] = Field(description="关联任务ID")
    template_id: Optional[uuid.UUID] = Field(description="使用模板ID")
    
    report_period_start: Optional[datetime] = Field(description="报告期间开始")
    report_period_end: Optional[datetime] = Field(description="报告期间结束")
    
    file_path: Optional[str] = Field(description="文件路径")
    file_size: Optional[int] = Field(description="文件大小")
    file_format: Optional[str] = Field(description="文件格式")
    
    generation_status: str = Field(description="生成状态")
    is_shared: bool = Field(description="是否已分享")
    
    view_count: int = Field(description="查看次数")
    download_count: int = Field(description="下载次数")
    
    generated_at: datetime = Field(description="生成时间")
    expires_at: Optional[datetime] = Field(description="过期时间")
    
    class Config:
        from_attributes = True


# ===== 数据源相关模式 =====

class DataSourceCreate(BaseModel):
    """创建数据源请求"""
    name: str = Field(..., min_length=1, max_length=255, description="数据源名称")
    description: Optional[str] = Field(None, description="数据源描述")
    source_type: str = Field(..., description="数据源类型")
    
    connection_config: Dict[str, Any] = Field(..., description="连接配置")
    credentials: Optional[Dict[str, Any]] = Field(default={}, description="认证信息")
    
    schema_config: Optional[Dict[str, Any]] = Field(default={}, description="数据结构配置")
    refresh_interval: int = Field(default=3600, ge=60, description="刷新间隔(秒)")


class DataSourceUpdate(BaseModel):
    """更新数据源请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="数据源名称")
    description: Optional[str] = Field(None, description="数据源描述")
    
    connection_config: Optional[Dict[str, Any]] = Field(None, description="连接配置")
    credentials: Optional[Dict[str, Any]] = Field(None, description="认证信息")
    
    schema_config: Optional[Dict[str, Any]] = Field(None, description="数据结构配置")
    refresh_interval: Optional[int] = Field(None, ge=60, description="刷新间隔(秒)")
    
    is_active: Optional[bool] = Field(None, description="是否启用")


class DataSourceResponse(BaseModel):
    """数据源响应"""
    id: uuid.UUID = Field(description="数据源ID")
    name: str = Field(description="数据源名称")
    description: Optional[str] = Field(description="数据源描述")
    source_type: str = Field(description="数据源类型")
    
    user_id: str = Field(description="用户ID")
    
    is_active: bool = Field(description="是否启用")
    last_sync_at: Optional[datetime] = Field(description="最后同步时间")
    sync_status: str = Field(description="同步状态")
    
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    
    class Config:
        from_attributes = True


# ===== 告警相关模式 =====

class AlertRuleCreate(BaseModel):
    """创建告警规则请求"""
    name: str = Field(..., min_length=1, max_length=255, description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    
    metric_name: str = Field(..., description="监控指标名称")
    condition: str = Field(..., description="触发条件")
    threshold_value: float = Field(..., description="阈值")
    
    evaluation_window: int = Field(default=300, ge=60, description="评估时间窗口(秒)")
    evaluation_frequency: int = Field(default=60, ge=30, description="评估频率(秒)")
    
    severity: AlertSeverity = Field(..., description="严重程度")
    notification_channels: Optional[Dict[str, Any]] = Field(default={}, description="通知渠道配置")
    
    filters: Optional[Dict[str, Any]] = Field(default={}, description="过滤条件")


class AlertRuleUpdate(BaseModel):
    """更新告警规则请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    
    condition: Optional[str] = Field(None, description="触发条件")
    threshold_value: Optional[float] = Field(None, description="阈值")
    
    evaluation_window: Optional[int] = Field(None, ge=60, description="评估时间窗口(秒)")
    evaluation_frequency: Optional[int] = Field(None, ge=30, description="评估频率(秒)")
    
    severity: Optional[AlertSeverity] = Field(None, description="严重程度")
    notification_channels: Optional[Dict[str, Any]] = Field(None, description="通知渠道配置")
    
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    is_active: Optional[bool] = Field(None, description="是否启用")


class AlertRuleResponse(BaseModel):
    """告警规则响应"""
    id: uuid.UUID = Field(description="规则ID")
    name: str = Field(description="规则名称")
    description: Optional[str] = Field(description="规则描述")
    
    metric_name: str = Field(description="监控指标名称")
    condition: str = Field(description="触发条件")
    threshold_value: float = Field(description="阈值")
    
    evaluation_window: int = Field(description="评估时间窗口(秒)")
    evaluation_frequency: int = Field(description="评估频率(秒)")
    
    severity: AlertSeverity = Field(description="严重程度")
    
    is_active: bool = Field(description="是否启用")
    last_triggered_at: Optional[datetime] = Field(description="最后触发时间")
    trigger_count: int = Field(description="触发次数")
    
    user_id: str = Field(description="用户ID")
    
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    
    class Config:
        from_attributes = True


class AlertTrigger(BaseModel):
    """告警触发信息"""
    rule_id: uuid.UUID = Field(description="规则ID")
    rule_name: str = Field(description="规则名称")
    severity: AlertSeverity = Field(description="严重程度")
    
    actual_value: float = Field(description="实际值")
    threshold_value: float = Field(description="阈值")
    
    triggered_at: datetime = Field(description="触发时间")
    
    message: str = Field(description="告警消息")


# ===== 分析结果相关模式 =====

class AnalyticsMetrics(BaseModel):
    """分析指标数据"""
    total_views: int = Field(description="总浏览量")
    total_likes: int = Field(description="总点赞数")
    total_comments: int = Field(description="总评论数")
    total_shares: int = Field(description="总分享数")
    
    avg_engagement_rate: float = Field(description="平均参与度")
    avg_click_through_rate: float = Field(description="平均点击率")
    avg_conversion_rate: float = Field(description="平均转化率")
    
    growth_rate: float = Field(description="增长率")
    
    period_start: datetime = Field(description="统计开始时间")
    period_end: datetime = Field(description="统计结束时间")


class ContentPerformance(BaseModel):
    """内容表现分析"""
    content_id: str = Field(description="内容ID")
    title: str = Field(description="内容标题")
    platform: str = Field(description="平台")
    
    views: int = Field(description="浏览量")
    likes: int = Field(description="点赞数")
    comments: int = Field(description="评论数")
    shares: int = Field(description="分享数")
    
    engagement_rate: float = Field(description="参与度")
    click_through_rate: float = Field(description="点击率")
    conversion_rate: float = Field(description="转化率")
    
    performance_score: float = Field(description="表现评分")
    
    publish_time: datetime = Field(description="发布时间")


class PlatformComparison(BaseModel):
    """平台对比分析"""
    platform: str = Field(description="平台名称")
    
    total_content: int = Field(description="内容总数")
    total_views: int = Field(description="总浏览量")
    total_engagement: int = Field(description="总互动数")
    
    avg_engagement_rate: float = Field(description="平均参与度")
    avg_reach: float = Field(description="平均触达率")
    
    top_content_id: str = Field(description="最佳内容ID")
    
    metrics_comparison: Dict[str, float] = Field(description="指标对比")


class TrendAnalysis(BaseModel):
    """趋势分析"""
    metric_name: str = Field(description="指标名称")
    time_period: str = Field(description="时间周期")
    
    trend_direction: str = Field(description="趋势方向")
    growth_rate: float = Field(description="增长率")
    
    data_points: List[Dict[str, Union[datetime, float]]] = Field(description="数据点")
    
    forecast: Optional[List[Dict[str, Union[datetime, float]]]] = Field(
        None, description="预测数据"
    )
    
    seasonality: Optional[Dict[str, Any]] = Field(None, description="季节性分析")
    
    anomalies: List[Dict[str, Any]] = Field(default=[], description="异常点")


class UserBehaviorInsights(BaseModel):
    """用户行为洞察"""
    active_users: int = Field(description="活跃用户数")
    new_users: int = Field(description="新用户数")
    returning_users: int = Field(description="回访用户数")
    
    avg_session_duration: float = Field(description="平均会话时长")
    avg_pages_per_session: float = Field(description="平均访问页面数")
    
    top_content: List[str] = Field(description="热门内容ID列表")
    popular_times: List[int] = Field(description="热门时段(小时)")
    
    user_segments: Dict[str, int] = Field(description="用户分群")
    behavior_patterns: Dict[str, Any] = Field(description="行为模式")


# ===== 时序数据相关模式 =====

class TimeSeriesPoint(BaseModel):
    """时序数据点"""
    timestamp: datetime = Field(description="时间戳")
    value: float = Field(description="值")
    tags: Optional[Dict[str, str]] = Field(default={}, description="标签")


class MetricQuery(BaseModel):
    """指标查询请求"""
    metric_name: str = Field(description="指标名称")
    start_time: datetime = Field(description="开始时间")
    end_time: datetime = Field(description="结束时间")
    
    filters: Optional[Dict[str, Any]] = Field(default={}, description="过滤条件")
    aggregation: str = Field(default="mean", description="聚合方式")
    group_by: Optional[List[str]] = Field(default=[], description="分组字段")
    
    interval: Optional[str] = Field(None, description="时间间隔")


class AggregationResult(BaseModel):
    """聚合结果"""
    metric_name: str = Field(description="指标名称")
    aggregation_type: str = Field(description="聚合类型")
    
    result: Union[float, Dict[str, float]] = Field(description="聚合结果")
    count: int = Field(description="数据点数量")
    
    time_range: Dict[str, datetime] = Field(description="时间范围")
    tags: Optional[Dict[str, str]] = Field(description="标签")


# ===== 可视化相关模式 =====

class ChartConfig(BaseModel):
    """图表配置"""
    chart_type: str = Field(description="图表类型")
    title: str = Field(description="图表标题")
    
    x_axis: Dict[str, Any] = Field(description="X轴配置")
    y_axis: Dict[str, Any] = Field(description="Y轴配置")
    
    series: List[Dict[str, Any]] = Field(description="数据系列配置")
    
    colors: Optional[List[str]] = Field(None, description="颜色配置")
    theme: str = Field(default="default", description="主题")
    
    options: Optional[Dict[str, Any]] = Field(default={}, description="其他选项")


class DashboardConfig(BaseModel):
    """仪表板配置"""
    title: str = Field(description="仪表板标题")
    description: Optional[str] = Field(None, description="描述")
    
    layout: Dict[str, Any] = Field(description="布局配置")
    charts: List[ChartConfig] = Field(description="图表配置列表")
    
    filters: Optional[Dict[str, Any]] = Field(default={}, description="全局过滤器")
    refresh_interval: int = Field(default=300, description="刷新间隔(秒)")


class VisualizationRequest(BaseModel):
    """可视化请求"""
    chart_configs: List[ChartConfig] = Field(description="图表配置")
    data_query: MetricQuery = Field(description="数据查询")
    
    export_format: Optional[str] = Field(None, description="导出格式")


# ===== 导出相关模式 =====

class ExportFormat(str, Enum):
    """导出格式枚举"""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    PNG = "png"
    SVG = "svg"


class ExportRequest(BaseModel):
    """导出请求"""
    data_type: str = Field(description="数据类型")
    format: ExportFormat = Field(description="导出格式")
    
    query_params: Optional[Dict[str, Any]] = Field(default={}, description="查询参数")
    export_options: Optional[Dict[str, Any]] = Field(default={}, description="导出选项")
    
    include_charts: bool = Field(default=True, description="是否包含图表")
    include_raw_data: bool = Field(default=False, description="是否包含原始数据")


class ExportResponse(BaseModel):
    """导出响应"""
    task_id: uuid.UUID = Field(description="导出任务ID")
    status: str = Field(description="导出状态")
    
    file_url: Optional[str] = Field(None, description="文件下载URL")
    file_size: Optional[int] = Field(None, description="文件大小")
    
    created_at: datetime = Field(description="创建时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")