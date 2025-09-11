"""
审核工作流管理器

负责管理内容审核的完整工作流程，包括任务创建、分配、
进度跟踪、决策处理和流程优化。
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from loguru import logger

from ..models.quality_models import (
    ReviewTask, ReviewTaskCreateRequest, ReviewDecisionRequest,
    ReviewHistoryEntry, QualityCheckResult, ComplianceCheckResult,
    ReviewStatus, ReviewDecision, TaskPriority, WorkflowStep
)
from ..config.settings import settings
from ..clients.storage_client import StorageServiceClient

class TaskAssignmentStrategy(Enum):
    """任务分配策略枚举"""
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    WORKLOAD = "workload"
    EXPERTISE = "expertise"

class ReviewWorkflowManager:
    """
    审核工作流管理器主类
    
    负责管理整个审核流程的生命周期
    """
    
    def __init__(self, storage_client: StorageServiceClient):
        """
        初始化审核工作流管理器
        
        Args:
            storage_client: 存储服务客户端
        """
        self.storage_client = storage_client
        self.assignment_strategy = TaskAssignmentStrategy(settings.TASK_ASSIGNMENT_STRATEGY)
        
        # 工作流缓存
        self._workflows_cache = {}
        self._cache_expiry = None
        
        # 审核员工作负载统计
        self._reviewer_workload = {}
        
        logger.info("审核工作流管理器初始化完成")
    
    async def create_review_task(self, request: ReviewTaskCreateRequest) -> Dict[str, Any]:
        """
        创建审核任务
        
        Args:
            request: 创建任务请求
            
        Returns:
            创建结果，包含任务ID和工作流信息
        """
        logger.info(f"创建审核任务: content_id={request.content_id}")
        
        # 计算任务优先级
        priority_score = await self._calculate_priority(
            request.quality_result,
            request.compliance_result
        )
        
        # 选择合适的工作流
        workflow = await self._select_workflow(
            request.quality_result,
            request.compliance_result,
            request.workflow_type
        )
        
        # 预估审核时间
        estimated_time = await self._estimate_review_time(
            request.quality_result,
            request.compliance_result,
            workflow
        )
        
        # 创建任务数据
        task_data = {
            "content_id": request.content_id,
            "workflow_id": workflow.get("id"),
            "priority_score": priority_score,
            "estimated_review_time": estimated_time,
            "quality_summary": self._create_quality_summary(request.quality_result),
            "compliance_summary": self._create_compliance_summary(request.compliance_result)
        }
        
        # 如果指定了审核员，添加到任务数据
        if request.assigned_reviewer:
            task_data["assigned_reviewer"] = request.assigned_reviewer
        
        # 通过storage-service创建任务
        result = await self.storage_client.create_review_task(task_data)
        task_id = result.get("task_id")
        
        if not task_id:
            raise RuntimeError("创建审核任务失败")
        
        # 检查是否符合自动审核条件
        auto_approval_result = await self._check_auto_approval(
            request.quality_result,
            request.compliance_result,
            workflow
        )
        
        if auto_approval_result["can_auto_approve"]:
            # 自动审核通过
            await self._auto_approve_task(task_id, auto_approval_result["reason"])
            logger.info(f"任务{task_id}自动审核通过")
        else:
            # 分配给人工审核
            await self._assign_to_reviewer(task_id, priority_score, request.assigned_reviewer)
            logger.info(f"任务{task_id}已分配给人工审核")
        
        # 准备返回结果
        workflow_steps = workflow.get("workflow_steps", [])
        
        return {
            "task_id": task_id,
            "status": "auto_approved" if auto_approval_result["can_auto_approve"] else "pending",
            "priority_score": priority_score,
            "estimated_review_time": estimated_time,
            "workflow_steps": workflow_steps,
            "auto_approval": auto_approval_result["can_auto_approve"],
            "assigned_reviewer": task_data.get("assigned_reviewer")
        }
    
    async def process_review_decision(self, 
                                    task_id: str, 
                                    decision_request: ReviewDecisionRequest) -> Dict[str, Any]:
        """
        处理审核决策
        
        Args:
            task_id: 任务ID
            decision_request: 决策请求
            
        Returns:
            处理结果
        """
        logger.info(f"处理审核决策: task_id={task_id}, decision={decision_request.decision}")
        
        # 获取当前任务信息
        task_result = await self.storage_client.get_review_task(task_id)
        current_task = task_result.get("data", {})
        
        if not current_task:
            raise ValueError(f"任务{task_id}不存在")
        
        # 验证任务状态
        if current_task.get("task_status") not in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]:
            raise ValueError(f"任务{task_id}状态不允许处理决策")
        
        # 获取工作流信息
        workflow_id = current_task.get("workflow_id")
        workflow = await self._get_workflow_by_id(workflow_id) if workflow_id else {}
        
        # 处理不同的决策类型
        next_steps = []
        new_status = ReviewStatus.COMPLETED
        
        if decision_request.decision == ReviewDecision.APPROVE:
            new_status = ReviewStatus.COMPLETED
            next_steps.append("内容审核通过，可以发布")
            
        elif decision_request.decision == ReviewDecision.REJECT:
            new_status = ReviewStatus.REJECTED
            next_steps.append("内容被拒绝，不允许发布")
            
        elif decision_request.decision == ReviewDecision.APPROVE_WITH_CHANGES:
            new_status = ReviewStatus.COMPLETED
            next_steps.append("内容通过但需要修改")
            next_steps.append("作者需要根据建议进行修改")
            
        elif decision_request.decision == ReviewDecision.REQUEST_REVISION:
            new_status = ReviewStatus.PENDING
            next_steps.append("要求作者修改后重新提交")
            
        elif decision_request.decision == ReviewDecision.ESCALATE:
            # 升级到下一级审核
            next_reviewer = await self._get_next_level_reviewer(current_task, workflow)
            if next_reviewer:
                new_status = ReviewStatus.PENDING
                next_steps.append(f"升级到高级审核员: {next_reviewer}")
                # 重新分配任务
                await self._reassign_task(task_id, next_reviewer)
            else:
                new_status = ReviewStatus.ESCALATED
                next_steps.append("已升级到最高级别审核")
        
        # 更新任务状态
        update_data = {
            "task_status": new_status.value,
            "review_notes": decision_request.notes,
            "decision_reason": f"{decision_request.decision.value}: {decision_request.notes}",
            "actual_review_time": decision_request.review_time_minutes,
            "completed_at": datetime.now().isoformat() if new_status in [ReviewStatus.COMPLETED, ReviewStatus.REJECTED] else None
        }
        
        if decision_request.next_reviewer:
            update_data["assigned_reviewer"] = decision_request.next_reviewer
        
        # 提交决策到storage-service
        decision_data = {
            "decision": decision_request.decision.value,
            "notes": decision_request.notes,
            "required_changes": decision_request.required_changes,
            "review_time_minutes": decision_request.review_time_minutes
        }
        
        # 同时更新任务和提交决策
        await asyncio.gather(
            self.storage_client.update_review_task(task_id, update_data),
            self.storage_client.submit_review_decision(task_id, decision_data)
        )
        
        # 更新审核员工作负载统计
        reviewer_id = current_task.get("assigned_reviewer")
        if reviewer_id:
            await self._update_reviewer_workload(reviewer_id, -1)
        
        logger.info(f"审核决策处理完成: task_id={task_id}, new_status={new_status}")
        
        return {
            "task_id": task_id,
            "new_status": new_status.value,
            "decision": decision_request.decision.value,
            "next_steps": next_steps,
            "processing_time": decision_request.review_time_minutes
        }
    
    async def get_pending_tasks(self, 
                               reviewer_id: Optional[str] = None,
                               priority: Optional[str] = None,
                               limit: int = 20) -> Dict[str, Any]:
        """
        获取待审核任务列表
        
        Args:
            reviewer_id: 审核员ID
            priority: 优先级过滤
            limit: 返回数量限制
            
        Returns:
            任务列表
        """
        params = {
            "status": ReviewStatus.PENDING.value,
            "per_page": limit
        }
        
        if reviewer_id:
            params["assigned_to"] = reviewer_id
        if priority:
            params["priority"] = priority
        
        result = await self.storage_client.get_review_tasks(**params)
        
        # 按优先级排序
        tasks = result.get("data", {}).get("tasks", [])
        sorted_tasks = sorted(tasks, key=lambda x: x.get("priority_score", 5), reverse=True)
        
        return {
            "tasks": sorted_tasks,
            "total": len(sorted_tasks),
            "pending_count": len(sorted_tasks)
        }
    
    async def get_task_statistics(self, 
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取任务统计信息
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            统计信息
        """
        # 通过storage-service获取统计数据
        stats_result = await self.storage_client.get_review_statistics(start_date, end_date)
        
        return stats_result.get("data", {})
    
    # ==================== 私有方法 ====================
    
    async def _calculate_priority(self, 
                                quality_result: Optional[Dict[str, Any]],
                                compliance_result: Optional[Dict[str, Any]]) -> int:
        """计算任务优先级"""
        priority = 5  # 基础优先级
        
        # 质量检测结果影响优先级
        if quality_result:
            overall_score = quality_result.get("overall_score", 80)
            if overall_score < 60:
                priority += 4
            elif overall_score < 75:
                priority += 2
            
            # 严重问题增加优先级
            issues = quality_result.get("issues", [])
            critical_issues = [issue for issue in issues if issue.get("severity") == "critical"]
            priority += len(critical_issues)
        
        # 合规检测结果影响优先级
        if compliance_result:
            risk_score = compliance_result.get("risk_score", 0)
            if risk_score >= 8:
                priority += 5
            elif risk_score >= 5:
                priority += 3
            elif risk_score >= 3:
                priority += 1
        
        return min(10, max(1, priority))
    
    async def _select_workflow(self, 
                             quality_result: Optional[Dict[str, Any]],
                             compliance_result: Optional[Dict[str, Any]],
                             workflow_type: Optional[str] = None) -> Dict[str, Any]:
        """选择合适的审核工作流"""
        # 获取可用工作流
        workflows = await self._get_available_workflows()
        
        if not workflows:
            # 返回默认工作流
            return self._get_default_workflow()
        
        # 如果指定了工作流类型
        if workflow_type:
            for workflow in workflows:
                if workflow.get("workflow_name") == workflow_type:
                    return workflow
        
        # 根据内容特征选择工作流
        if compliance_result and compliance_result.get("risk_score", 0) >= 7:
            # 高风险内容使用严格审核流程
            strict_workflows = [w for w in workflows if "strict" in w.get("workflow_name", "").lower()]
            if strict_workflows:
                return strict_workflows[0]
        
        # 默认使用第一个工作流
        return workflows[0] if workflows else self._get_default_workflow()
    
    async def _estimate_review_time(self, 
                                  quality_result: Optional[Dict[str, Any]],
                                  compliance_result: Optional[Dict[str, Any]],
                                  workflow: Dict[str, Any]) -> int:
        """预估审核时间（分钟）"""
        base_time = workflow.get("estimated_time", 30)
        
        # 根据质量问题调整时间
        if quality_result:
            issues_count = len(quality_result.get("issues", []))
            base_time += issues_count * 2
        
        # 根据合规风险调整时间
        if compliance_result:
            risk_score = compliance_result.get("risk_score", 0)
            base_time += risk_score * 3
            
            violations_count = len(compliance_result.get("violations", []))
            base_time += violations_count * 2
        
        return min(120, max(15, base_time))  # 限制在15-120分钟之间
    
    def _create_quality_summary(self, quality_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """创建质量检测摘要"""
        if not quality_result:
            return {}
        
        return {
            "overall_score": quality_result.get("overall_score", 0),
            "issues_count": len(quality_result.get("issues", [])),
            "critical_issues": len([
                issue for issue in quality_result.get("issues", [])
                if issue.get("severity") == "critical"
            ]),
            "status": quality_result.get("status", "unknown")
        }
    
    def _create_compliance_summary(self, compliance_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """创建合规检测摘要"""
        if not compliance_result:
            return {}
        
        return {
            "compliance_status": compliance_result.get("compliance_status", "unknown"),
            "risk_score": compliance_result.get("risk_score", 0),
            "violations_count": len(compliance_result.get("violations", [])),
            "high_risk_violations": len([
                violation for violation in compliance_result.get("violations", [])
                if violation.get("severity", 0) >= 7
            ])
        }
    
    async def _check_auto_approval(self, 
                                 quality_result: Optional[Dict[str, Any]],
                                 compliance_result: Optional[Dict[str, Any]],
                                 workflow: Dict[str, Any]) -> Dict[str, Any]:
        """检查是否可以自动审核通过"""
        can_auto_approve = False
        reason = ""
        
        # 获取自动审核阈值
        auto_approval_threshold = workflow.get("auto_approval_threshold", settings.AUTO_APPROVAL_THRESHOLD)
        auto_approval_risk_threshold = workflow.get("auto_approval_risk_threshold", settings.AUTO_APPROVAL_RISK_THRESHOLD)
        
        # 检查质量分数
        quality_score = quality_result.get("overall_score", 0) if quality_result else 80
        risk_score = compliance_result.get("risk_score", 0) if compliance_result else 0
        
        # 检查是否有严重问题
        has_critical_issues = False
        if quality_result:
            critical_issues = [
                issue for issue in quality_result.get("issues", [])
                if issue.get("severity") == "critical"
            ]
            has_critical_issues = len(critical_issues) > 0
        
        # 检查是否有高风险违规
        has_high_risk_violations = False
        if compliance_result:
            high_risk_violations = [
                violation for violation in compliance_result.get("violations", [])
                if violation.get("severity", 0) >= 8
            ]
            has_high_risk_violations = len(high_risk_violations) > 0
        
        # 决定是否自动通过
        if (quality_score >= auto_approval_threshold and 
            risk_score <= auto_approval_risk_threshold and
            not has_critical_issues and
            not has_high_risk_violations):
            can_auto_approve = True
            reason = f"质量分数{quality_score}≥{auto_approval_threshold}，风险评分{risk_score}≤{auto_approval_risk_threshold}，无严重问题"
        else:
            reasons = []
            if quality_score < auto_approval_threshold:
                reasons.append(f"质量分数{quality_score}<{auto_approval_threshold}")
            if risk_score > auto_approval_risk_threshold:
                reasons.append(f"风险评分{risk_score}>{auto_approval_risk_threshold}")
            if has_critical_issues:
                reasons.append("存在严重质量问题")
            if has_high_risk_violations:
                reasons.append("存在高风险违规")
            reason = "；".join(reasons)
        
        return {
            "can_auto_approve": can_auto_approve,
            "reason": reason,
            "quality_score": quality_score,
            "risk_score": risk_score
        }
    
    async def _auto_approve_task(self, task_id: str, reason: str):
        """自动审核通过任务"""
        update_data = {
            "task_status": ReviewStatus.COMPLETED.value,
            "review_notes": f"自动审核通过: {reason}",
            "decision_reason": "auto_approval",
            "actual_review_time": 0,
            "completed_at": datetime.now().isoformat()
        }
        
        await self.storage_client.update_review_task(task_id, update_data)
    
    async def _assign_to_reviewer(self, 
                                task_id: str, 
                                priority_score: int,
                                specified_reviewer: Optional[str] = None):
        """分配任务给审核员"""
        if specified_reviewer:
            reviewer_id = specified_reviewer
        else:
            reviewer_id = await self._select_reviewer(priority_score)
        
        if reviewer_id:
            update_data = {
                "assigned_reviewer": reviewer_id,
                "task_status": ReviewStatus.IN_PROGRESS.value,
                "assigned_at": datetime.now().isoformat()
            }
            
            await self.storage_client.update_review_task(task_id, update_data)
            await self._update_reviewer_workload(reviewer_id, 1)
        
    async def _select_reviewer(self, priority_score: int) -> Optional[str]:
        """选择合适的审核员"""
        # 这里应该实现审核员选择逻辑
        # 可以根据工作负载、专业领域、在线状态等因素选择
        
        # 简化实现：返回默认审核员
        return "default_reviewer"
    
    async def _get_available_workflows(self) -> List[Dict[str, Any]]:
        """获取可用的工作流"""
        current_time = time.time()
        
        # 检查缓存
        if (self._cache_expiry is None or 
            current_time > self._cache_expiry or 
            not self._workflows_cache):
            
            try:
                result = await self.storage_client.get_active_workflows()
                self._workflows_cache = result.get("data", [])
                self._cache_expiry = current_time + 600  # 缓存10分钟
            except Exception as e:
                logger.error(f"获取工作流失败: {e}")
                return [self._get_default_workflow()]
        
        return self._workflows_cache
    
    def _get_default_workflow(self) -> Dict[str, Any]:
        """获取默认工作流"""
        return {
            "id": "default_workflow",
            "workflow_name": "standard_review",
            "estimated_time": 30,
            "auto_approval_threshold": settings.AUTO_APPROVAL_THRESHOLD,
            "auto_approval_risk_threshold": settings.AUTO_APPROVAL_RISK_THRESHOLD,
            "workflow_steps": [
                {
                    "step": 1,
                    "name": "初步审核",
                    "assignee_type": "reviewer",
                    "estimated_time": 20
                },
                {
                    "step": 2,
                    "name": "终审",
                    "assignee_type": "senior_reviewer", 
                    "estimated_time": 10
                }
            ]
        }
    
    async def _get_workflow_by_id(self, workflow_id: str) -> Dict[str, Any]:
        """根据ID获取工作流"""
        workflows = await self._get_available_workflows()
        for workflow in workflows:
            if workflow.get("id") == workflow_id:
                return workflow
        return self._get_default_workflow()
    
    async def _get_next_level_reviewer(self, current_task: Dict, workflow: Dict) -> Optional[str]:
        """获取下一级审核员"""
        # 简化实现：返回高级审核员
        return "senior_reviewer"
    
    async def _reassign_task(self, task_id: str, new_reviewer: str):
        """重新分配任务"""
        update_data = {
            "assigned_reviewer": new_reviewer,
            "task_status": ReviewStatus.IN_PROGRESS.value,
            "assigned_at": datetime.now().isoformat()
        }
        
        await self.storage_client.update_review_task(task_id, update_data)
    
    async def _update_reviewer_workload(self, reviewer_id: str, change: int):
        """更新审核员工作负载"""
        current_load = self._reviewer_workload.get(reviewer_id, 0)
        self._reviewer_workload[reviewer_id] = max(0, current_load + change)