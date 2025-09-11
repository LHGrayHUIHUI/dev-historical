"""
审核工作流控制器

提供审核工作流相关的API端点，包括创建审核任务、处理审核决策、
查询任务状态、获取统计信息等功能。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional

from loguru import logger

from ..models.quality_models import (
    ReviewTaskCreateRequest, ReviewDecisionRequest, DataResponse, 
    PaginatedResponse, TaskPriority, ReviewStatus
)
from ..services.review_workflow_manager import ReviewWorkflowManager
from ..clients.storage_client import StorageServiceClient, get_storage_client
from ..config.settings import settings

# 创建路由器
router = APIRouter(prefix="/review", tags=["审核工作流"])

@router.post("/tasks",
             response_model=DataResponse,
             summary="创建审核任务",
             description="根据质量和合规检测结果创建审核任务")
async def create_review_task(
    request: ReviewTaskCreateRequest,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    创建审核任务
    
    根据内容的质量检测和合规检测结果，创建相应的审核任务，
    并自动分配给合适的审核员或进行自动审核。
    """
    try:
        logger.info(f"创建审核任务: content_id={request.content_id}")
        
        # 创建审核工作流管理器
        workflow_manager = ReviewWorkflowManager(storage_client)
        
        # 创建审核任务
        result = await workflow_manager.create_review_task(request)
        
        return DataResponse(
            success=True,
            message="审核任务创建成功",
            data=result
        )
        
    except ValueError as e:
        logger.error(f"创建审核任务参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建审核任务失败: {e}")
        raise HTTPException(status_code=500, detail="创建审核任务失败")

@router.get("/tasks/{task_id}",
            response_model=DataResponse,
            summary="获取审核任务详情",
            description="根据任务ID获取审核任务的详细信息")
async def get_review_task(
    task_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取审核任务详情
    
    返回指定任务的详细信息，包括任务状态、审核历史、
    质量检测结果等。
    """
    try:
        logger.info(f"获取审核任务详情: task_id={task_id}")
        
        # 通过storage-service获取任务详情
        result = await storage_client.get_review_task(task_id)
        
        if not result.get("data"):
            raise HTTPException(status_code=404, detail="审核任务不存在")
        
        return DataResponse(
            success=True,
            message="审核任务详情获取成功",
            data=result.get("data")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取审核任务详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取审核任务详情失败")

@router.get("/tasks",
            response_model=PaginatedResponse,
            summary="获取审核任务列表",
            description="获取审核任务列表，支持多种过滤条件")
async def get_review_tasks(
    status: Optional[str] = None,
    assigned_to: Optional[str] = None,
    priority: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> PaginatedResponse:
    """
    获取审核任务列表
    
    支持按状态、审核员、优先级等条件过滤，
    并提供分页功能。
    """
    try:
        logger.info(f"获取审核任务列表: status={status}, assigned_to={assigned_to}")
        
        # 验证参数
        if per_page > 100:
            per_page = 100
        
        # 通过storage-service获取任务列表
        result = await storage_client.get_review_tasks(
            status=status,
            assigned_to=assigned_to,
            priority=priority,
            page=page,
            per_page=per_page
        )
        
        tasks_data = result.get("data", {})
        
        return PaginatedResponse(
            success=True,
            message="审核任务列表获取成功",
            data=tasks_data.get("tasks", []),
            pagination=tasks_data.get("pagination", {})
        )
        
    except Exception as e:
        logger.error(f"获取审核任务列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取审核任务列表失败")

@router.post("/tasks/{task_id}/decision",
             response_model=DataResponse,
             summary="提交审核决策",
             description="审核员提交对任务的审核决策")
async def submit_review_decision(
    task_id: str,
    decision_request: ReviewDecisionRequest,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    提交审核决策
    
    审核员提交审核决策，包括通过、拒绝、要求修改等。
    系统会根据决策类型进行相应的后续处理。
    """
    try:
        logger.info(f"提交审核决策: task_id={task_id}, decision={decision_request.decision}")
        
        # 创建审核工作流管理器
        workflow_manager = ReviewWorkflowManager(storage_client)
        
        # 处理审核决策
        result = await workflow_manager.process_review_decision(task_id, decision_request)
        
        return DataResponse(
            success=True,
            message="审核决策提交成功",
            data=result
        )
        
    except ValueError as e:
        logger.error(f"审核决策参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"提交审核决策失败: {e}")
        raise HTTPException(status_code=500, detail="提交审核决策失败")

@router.get("/tasks/pending",
            response_model=DataResponse,
            summary="获取待审核任务",
            description="获取当前待审核的任务列表")
async def get_pending_tasks(
    reviewer_id: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 20,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取待审核任务
    
    返回当前状态为待审核的任务列表，可按审核员和优先级过滤。
    """
    try:
        logger.info(f"获取待审核任务: reviewer_id={reviewer_id}, priority={priority}")
        
        # 创建审核工作流管理器
        workflow_manager = ReviewWorkflowManager(storage_client)
        
        # 获取待审核任务
        result = await workflow_manager.get_pending_tasks(reviewer_id, priority, limit)
        
        return DataResponse(
            success=True,
            message="待审核任务获取成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"获取待审核任务失败: {e}")
        raise HTTPException(status_code=500, detail="获取待审核任务失败")

@router.get("/workflows",
            response_model=DataResponse,
            summary="获取审核工作流",
            description="获取可用的审核工作流列表")
async def get_review_workflows(
    content_type: Optional[str] = None,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取审核工作流
    
    返回当前激活的审核工作流列表，可按内容类型过滤。
    """
    try:
        logger.info(f"获取审核工作流: content_type={content_type}")
        
        # 通过storage-service获取工作流列表
        result = await storage_client.get_review_workflows(content_type)
        
        return DataResponse(
            success=True,
            message="审核工作流获取成功",
            data=result.get("data", [])
        )
        
    except Exception as e:
        logger.error(f"获取审核工作流失败: {e}")
        raise HTTPException(status_code=500, detail="获取审核工作流失败")

@router.get("/statistics",
            response_model=DataResponse,
            summary="审核统计",
            description="获取审核相关的统计数据")
async def get_review_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取审核统计数据
    
    返回指定时间范围内的审核统计信息，包括审核效率、
    通过率、审核员表现等。
    """
    try:
        logger.info(f"获取审核统计数据: {start_date} - {end_date}")
        
        # 创建审核工作流管理器
        workflow_manager = ReviewWorkflowManager(storage_client)
        
        # 获取统计数据
        result = await workflow_manager.get_task_statistics(start_date, end_date)
        
        return DataResponse(
            success=True,
            message="审核统计数据获取成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"获取审核统计数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取审核统计数据失败")

@router.put("/tasks/{task_id}/assign",
            response_model=DataResponse,
            summary="分配审核任务",
            description="将审核任务分配给指定审核员")
async def assign_review_task(
    task_id: str,
    reviewer_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    分配审核任务
    
    将指定的审核任务分配给指定的审核员。
    """
    try:
        logger.info(f"分配审核任务: task_id={task_id}, reviewer_id={reviewer_id}")
        
        # 更新任务分配
        update_data = {
            "assigned_reviewer": reviewer_id,
            "task_status": ReviewStatus.IN_PROGRESS.value,
            "assigned_at": datetime.now().isoformat()
        }
        
        result = await storage_client.update_review_task(task_id, update_data)
        
        return DataResponse(
            success=True,
            message="审核任务分配成功",
            data=result.get("data", {})
        )
        
    except Exception as e:
        logger.error(f"分配审核任务失败: {e}")
        raise HTTPException(status_code=500, detail="分配审核任务失败")

@router.get("/tasks/{task_id}/history",
            response_model=DataResponse,
            summary="获取审核历史",
            description="获取指定任务的审核历史记录")
async def get_review_history(
    task_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取审核历史
    
    返回指定任务的完整审核历史记录。
    """
    try:
        logger.info(f"获取审核历史: task_id={task_id}")
        
        # 通过storage-service获取审核记录
        result = await storage_client.get_review_record(task_id)
        
        if not result.get("data"):
            raise HTTPException(status_code=404, detail="审核记录不存在")
        
        review_record = result.get("data", {})
        history = review_record.get("review_history", [])
        
        return DataResponse(
            success=True,
            message="审核历史获取成功",
            data={
                "task_id": task_id,
                "review_history": history,
                "total_steps": len(history)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取审核历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取审核历史失败")

@router.post("/comprehensive-check",
             response_model=DataResponse,
             summary="综合质量检测",
             description="对内容进行质量检测、合规检测并创建审核任务")
async def comprehensive_quality_check(
    content: str,
    content_id: Optional[str] = None,
    auto_create_task: bool = True,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    综合质量检测
    
    对内容进行完整的质量检测和合规检测，
    并根据结果自动创建审核任务。
    """
    try:
        logger.info(f"开始综合质量检测，内容长度: {len(content)}")
        
        # 导入必要的模块
        from ..services.quality_detection_engine import QualityDetectionEngine
        from ..services.compliance_engine import ComplianceEngine
        from ..services.review_workflow_manager import ReviewWorkflowManager
        from ..models.quality_models import QualityCheckRequest, ComplianceCheckRequest
        
        # 创建检测引擎
        quality_engine = QualityDetectionEngine(storage_client)
        compliance_engine = ComplianceEngine(storage_client)
        
        # 并行执行质量检测和合规检测
        quality_request = QualityCheckRequest(content=content)
        compliance_request = ComplianceCheckRequest(content=content)
        
        quality_result, compliance_result = await asyncio.gather(
            quality_engine.check_quality(quality_request),
            compliance_engine.check_compliance(compliance_request)
        )
        
        # 保存检测结果
        if settings.ENABLE_RESULT_CACHE:
            await asyncio.gather(
                storage_client.save_quality_check_result(quality_result),
                storage_client.save_compliance_check_result(compliance_result)
            )
        
        # 准备综合结果
        comprehensive_result = {
            "quality_check": {
                "check_id": quality_result.check_id,
                "overall_score": quality_result.overall_score,
                "status": quality_result.status,
                "issues_count": len(quality_result.issues),
                "processing_time_ms": quality_result.processing_time_ms
            },
            "compliance_check": {
                "check_id": compliance_result.check_id,
                "compliance_status": compliance_result.compliance_status.value,
                "risk_score": compliance_result.risk_score,
                "violations_count": len(compliance_result.violations),
                "processing_time_ms": compliance_result.processing_time_ms
            }
        }
        
        # 如果需要，创建审核任务
        if auto_create_task and content_id:
            workflow_manager = ReviewWorkflowManager(storage_client)
            
            task_request = ReviewTaskCreateRequest(
                content_id=content_id,
                quality_result=quality_result.dict(),
                compliance_result=compliance_result.dict()
            )
            
            task_result = await workflow_manager.create_review_task(task_request)
            comprehensive_result["review_task"] = task_result
        
        return DataResponse(
            success=True,
            message="综合质量检测完成",
            data=comprehensive_result
        )
        
    except ValueError as e:
        logger.error(f"综合质量检测参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"综合质量检测失败: {e}")
        raise HTTPException(status_code=500, detail="综合质量检测服务异常")