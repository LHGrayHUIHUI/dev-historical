"""
质量控制数据模型

定义内容质量检测、合规性审核、审核工作流等核心数据结构
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

# ==================== 枚举类型定义 ====================

class IssueType(str, Enum):
    """问题类型枚举"""
    GRAMMAR_ERROR = "grammar_error"
    LOGIC_INCONSISTENCY = "logic_inconsistency"
    FORMAT_VIOLATION = "format_violation"
    FACTUAL_ERROR = "factual_error"
    ACADEMIC_STANDARD = "academic_standard"
    STYLE_ISSUE = "style_issue"
    STRUCTURE_PROBLEM = "structure_problem"

class IssueSeverity(str, Enum):
    """问题严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(str, Enum):
    """合规状态枚举"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    PENDING = "pending"

class ViolationType(str, Enum):
    """违规类型枚举"""
    SENSITIVE_WORD = "sensitive_word"
    POLICY_VIOLATION = "policy_violation"
    COPYRIGHT_VIOLATION = "copyright_violation"
    ACADEMIC_INTEGRITY = "academic_integrity"
    FORMAT_VIOLATION = "format_violation"

class ReviewStatus(str, Enum):
    """审核状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ESCALATED = "escalated"

class ReviewDecision(str, Enum):
    """审核决策枚举"""
    APPROVE = "approve"
    REJECT = "reject"
    APPROVE_WITH_CHANGES = "approve_with_changes"
    ESCALATE = "escalate"
    REQUEST_REVISION = "request_revision"

class TaskPriority(str, Enum):
    """任务优先级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

# ==================== 质量检测模型 ====================

class QualityIssue(BaseModel):
    """质量问题数据模型"""
    issue_type: IssueType = Field(..., description="问题类型")
    severity: IssueSeverity = Field(..., description="严重程度")
    position: int = Field(..., description="问题位置（字符索引）")
    length: Optional[int] = Field(None, description="问题文本长度")
    description: str = Field(..., description="问题描述")
    suggestion: str = Field(..., description="修改建议")
    auto_fixable: bool = Field(default=False, description="是否可自动修复")
    confidence: float = Field(..., ge=0.0, le=1.0, description="检测置信度")
    context: Optional[str] = Field(None, description="问题上下文")
    rule_id: Optional[str] = Field(None, description="触发的规则ID")
    
    @validator("position")
    def validate_position(cls, v):
        if v < 0:
            raise ValueError("位置不能为负数")
        return v

class QualityMetrics(BaseModel):
    """质量指标数据模型"""
    grammar_score: float = Field(..., ge=0.0, le=100.0, description="语法得分")
    logic_score: float = Field(..., ge=0.0, le=100.0, description="逻辑得分")
    format_score: float = Field(..., ge=0.0, le=100.0, description="格式得分")
    factual_score: float = Field(..., ge=0.0, le=100.0, description="事实得分")
    academic_score: float = Field(..., ge=0.0, le=100.0, description="学术得分")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="总体得分")
    readability_index: Optional[float] = Field(None, description="可读性指数")
    complexity_score: Optional[float] = Field(None, description="复杂度得分")

class QualityCheckRequest(BaseModel):
    """质量检测请求模型"""
    content: str = Field(..., min_length=1, description="待检测内容")
    content_type: str = Field(default="general", description="内容类型")
    check_options: Dict[str, bool] = Field(default_factory=lambda: {
        "grammar_check": True,
        "logic_check": True,
        "format_check": True,
        "factual_check": True,
        "academic_check": True
    }, description="检测选项")
    auto_fix: bool = Field(default=True, description="是否启用自动修复")
    language: str = Field(default="zh", description="内容语言")
    
    @validator("content")
    def validate_content_length(cls, v, values):
        # 这里可以通过设置获取最大长度限制
        max_length = 100000  # 可以从配置中获取
        if len(v) > max_length:
            raise ValueError(f"内容长度不能超过{max_length}字符")
        return v

class QualityCheckResult(BaseModel):
    """质量检测结果模型"""
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="检测ID")
    content_id: Optional[str] = Field(None, description="内容ID")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="总体质量分数")
    status: str = Field(..., description="检测状态")
    metrics: QualityMetrics = Field(..., description="详细指标")
    issues: List[QualityIssue] = Field(default_factory=list, description="检测到的问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")
    auto_fixes: List[Dict[str, Any]] = Field(default_factory=list, description="自动修复方案")
    processing_time_ms: int = Field(..., description="处理时间（毫秒）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

# ==================== 合规检测模型 ====================

class ComplianceViolation(BaseModel):
    """合规违规数据模型"""
    violation_type: ViolationType = Field(..., description="违规类型")
    severity: int = Field(..., ge=1, le=10, description="严重程度(1-10)")
    position: int = Field(..., description="违规位置")
    length: Optional[int] = Field(None, description="违规文本长度")
    content: str = Field(..., description="违规内容")
    description: str = Field(..., description="违规描述")
    rule_id: Optional[str] = Field(None, description="违规规则ID")
    category: Optional[str] = Field(None, description="违规分类")
    action: str = Field(..., description="建议动作")
    suggestion: str = Field(..., description="修改建议")
    confidence: float = Field(..., ge=0.0, le=1.0, description="检测置信度")

class ComplianceCheckRequest(BaseModel):
    """合规检测请求模型"""
    content: str = Field(..., min_length=1, description="待检测内容")
    check_types: List[str] = Field(default_factory=lambda: [
        "sensitive_words", "policy", "copyright", "academic_integrity"
    ], description="检测类型列表")
    strict_mode: bool = Field(default=False, description="严格模式")
    
class ComplianceCheckResult(BaseModel):
    """合规检测结果模型"""
    check_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="检测ID")
    content_id: Optional[str] = Field(None, description="内容ID")
    compliance_status: ComplianceStatus = Field(..., description="合规状态")
    risk_score: int = Field(..., ge=0, le=10, description="风险评分")
    violations: List[ComplianceViolation] = Field(default_factory=list, description="违规详情")
    policy_compliance: Dict[str, str] = Field(default_factory=dict, description="政策合规状态")
    recommendations: List[str] = Field(default_factory=list, description="整改建议")
    processing_time_ms: int = Field(..., description="处理时间（毫秒）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

# ==================== 审核工作流模型 ====================

class WorkflowStep(BaseModel):
    """工作流步骤模型"""
    step: int = Field(..., ge=1, description="步骤序号")
    name: str = Field(..., description="步骤名称")
    description: Optional[str] = Field(None, description="步骤描述")
    assignee_type: str = Field(..., description="执行者类型")
    auto_assignee: Optional[str] = Field(None, description="自动分配的执行者")
    estimated_time: int = Field(..., description="预估时间（分钟）")
    required_skills: List[str] = Field(default_factory=list, description="所需技能")
    escalation_threshold: Optional[int] = Field(None, description="升级阈值（分钟）")

class ReviewTask(BaseModel):
    """审核任务模型"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务ID")
    content_id: str = Field(..., description="内容ID")
    workflow_id: Optional[str] = Field(None, description="工作流ID")
    current_step: int = Field(default=1, description="当前步骤")
    assigned_reviewer: Optional[str] = Field(None, description="分配的审核员")
    task_status: ReviewStatus = Field(default=ReviewStatus.PENDING, description="任务状态")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="优先级")
    priority_score: int = Field(default=5, ge=1, le=10, description="优先级评分")
    estimated_review_time: int = Field(..., description="预估审核时间（分钟）")
    actual_review_time: Optional[int] = Field(None, description="实际审核时间（分钟）")
    quality_summary: Optional[Dict[str, Any]] = Field(None, description="质量检测摘要")
    compliance_summary: Optional[Dict[str, Any]] = Field(None, description="合规检测摘要")
    review_notes: Optional[str] = Field(None, description="审核备注")
    decision_reason: Optional[str] = Field(None, description="决策原因")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    assigned_at: Optional[datetime] = Field(None, description="分配时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")

class ReviewDecisionRequest(BaseModel):
    """审核决策请求模型"""
    decision: ReviewDecision = Field(..., description="审核决策")
    notes: str = Field(..., min_length=1, description="审核备注")
    required_changes: List[Dict[str, Any]] = Field(default_factory=list, description="要求的修改")
    review_time_minutes: int = Field(..., ge=0, description="审核用时（分钟）")
    next_reviewer: Optional[str] = Field(None, description="下一步审核员")

class ReviewTaskCreateRequest(BaseModel):
    """创建审核任务请求模型"""
    content_id: str = Field(..., description="内容ID")
    quality_result: Optional[Dict[str, Any]] = Field(None, description="质量检测结果")
    compliance_result: Optional[Dict[str, Any]] = Field(None, description="合规检测结果")
    priority: Optional[TaskPriority] = Field(None, description="指定优先级")
    assigned_reviewer: Optional[str] = Field(None, description="指定审核员")
    workflow_type: Optional[str] = Field(None, description="工作流类型")

# ==================== 审核历史模型 ====================

class ReviewHistoryEntry(BaseModel):
    """审核历史条目模型"""
    reviewer_id: str = Field(..., description="审核员ID")
    step: int = Field(..., description="审核步骤")
    action: ReviewDecision = Field(..., description="审核动作")
    notes: str = Field(..., description="审核备注")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    time_spent_minutes: int = Field(..., description="用时（分钟）")
    changes_requested: List[Dict[str, Any]] = Field(default_factory=list, description="要求的修改")

class ContentReviewRecord(BaseModel):
    """内容审核记录模型"""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="记录ID")
    content_id: str = Field(..., description="内容ID")
    original_content: str = Field(..., description="原始内容")
    processed_content: Optional[str] = Field(None, description="处理后内容")
    quality_analysis: Optional[QualityCheckResult] = Field(None, description="质量分析结果")
    compliance_analysis: Optional[ComplianceCheckResult] = Field(None, description="合规分析结果")
    review_history: List[ReviewHistoryEntry] = Field(default_factory=list, description="审核历史")
    final_decision: Optional[Dict[str, Any]] = Field(None, description="最终决策")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="统计指标")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

# ==================== 批量处理模型 ====================

class BatchQualityCheckRequest(BaseModel):
    """批量质量检测请求模型"""
    content_ids: List[str] = Field(..., min_items=1, max_items=100, description="内容ID列表")
    check_options: Dict[str, bool] = Field(default_factory=lambda: {
        "grammar_check": True,
        "logic_check": True,
        "format_check": True,
        "factual_check": True,
        "academic_check": True
    }, description="检测选项")
    parallel_processing: bool = Field(default=True, description="并行处理")
    max_concurrent_tasks: int = Field(default=5, ge=1, le=20, description="最大并发任务数")

class BatchProcessingResult(BaseModel):
    """批量处理结果模型"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="批次ID")
    total_items: int = Field(..., description="总条目数")
    completed_items: int = Field(default=0, description="已完成条目数")
    failed_items: int = Field(default=0, description="失败条目数")
    success_rate: float = Field(default=0.0, description="成功率")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="处理结果")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="错误信息")
    processing_time_ms: int = Field(..., description="总处理时间（毫秒）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

# ==================== 统计报告模型 ====================

class QualityStatistics(BaseModel):
    """质量统计模型"""
    total_checks: int = Field(default=0, description="总检测次数")
    average_score: float = Field(default=0.0, description="平均质量分数")
    pass_rate: float = Field(default=0.0, description="通过率")
    common_issues: List[Dict[str, Any]] = Field(default_factory=list, description="常见问题")
    improvement_trends: Dict[str, float] = Field(default_factory=dict, description="改进趋势")

class ComplianceStatistics(BaseModel):
    """合规统计模型"""
    total_checks: int = Field(default=0, description="总检测次数")
    compliance_rate: float = Field(default=0.0, description="合规率")
    violation_distribution: Dict[str, int] = Field(default_factory=dict, description="违规分布")
    risk_level_distribution: Dict[str, int] = Field(default_factory=dict, description="风险等级分布")

class ReviewStatistics(BaseModel):
    """审核统计模型"""
    total_reviews: int = Field(default=0, description="总审核次数")
    auto_approved: int = Field(default=0, description="自动通过数")
    human_reviewed: int = Field(default=0, description="人工审核数")
    rejected: int = Field(default=0, description="拒绝数")
    average_review_time: float = Field(default=0.0, description="平均审核时间")
    reviewer_performance: List[Dict[str, Any]] = Field(default_factory=list, description="审核员表现")

# ==================== 通用响应模型 ====================

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(default="", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")

class DataResponse(BaseResponse):
    """数据响应模型"""
    data: Union[Dict[str, Any], List[Any], None] = Field(None, description="响应数据")

class PaginatedResponse(BaseResponse):
    """分页响应模型"""
    data: List[Any] = Field(default_factory=list, description="数据列表")
    pagination: Dict[str, Any] = Field(default_factory=dict, description="分页信息")

class ErrorResponse(BaseResponse):
    """错误响应模型"""
    error_code: str = Field(..., description="错误代码")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    success: bool = Field(default=False, description="操作失败")