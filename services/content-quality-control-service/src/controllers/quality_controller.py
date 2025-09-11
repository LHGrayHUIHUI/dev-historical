"""
质量检测控制器

提供质量检测相关的API端点，包括单文档质量检测、
批量质量检测、质量历史查询等功能。
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import asyncio

from loguru import logger

from ..models.quality_models import (
    QualityCheckRequest, QualityCheckResult, BatchQualityCheckRequest,
    BatchProcessingResult, DataResponse, ErrorResponse
)
from ..services.quality_detection_engine import QualityDetectionEngine
from ..clients.storage_client import StorageServiceClient, get_storage_client
from ..config.settings import settings

# 创建路由器
router = APIRouter(prefix="/quality", tags=["质量检测"])

@router.post("/check", 
             response_model=DataResponse,
             summary="单文档质量检测",
             description="对单个文档进行全面的质量检测")
async def check_quality(
    request: QualityCheckRequest,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    单文档质量检测
    
    对提供的内容进行多维度质量检测，包括语法、逻辑、格式、
    事实准确性和学术标准等方面的分析。
    """
    try:
        logger.info(f"开始质量检测，内容长度: {len(request.content)}")
        
        # 创建质量检测引擎
        detection_engine = QualityDetectionEngine(storage_client)
        
        # 执行质量检测
        result = await detection_engine.check_quality(request)
        
        # 保存检测结果
        if settings.ENABLE_RESULT_CACHE:
            try:
                await storage_client.save_quality_check_result(result)
                logger.debug(f"质量检测结果已保存: {result.check_id}")
            except Exception as e:
                logger.warning(f"保存质量检测结果失败: {e}")
        
        # 准备响应数据
        response_data = {
            "check_id": result.check_id,
            "overall_score": result.overall_score,
            "status": result.status,
            "quality_analysis": {
                "grammar_score": result.metrics.grammar_score,
                "logic_score": result.metrics.logic_score,
                "format_score": result.metrics.format_score,
                "factual_score": result.metrics.factual_score,
                "academic_score": result.metrics.academic_score
            },
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity.value,
                    "position": issue.position,
                    "description": issue.description,
                    "suggestion": issue.suggestion,
                    "auto_fixable": issue.auto_fixable,
                    "confidence": issue.confidence
                }
                for issue in result.issues
            ],
            "suggestions": result.suggestions,
            "auto_fixes": result.auto_fixes,
            "processing_time_ms": result.processing_time_ms
        }
        
        return DataResponse(
            success=True,
            message="质量检测完成",
            data=response_data
        )
        
    except ValueError as e:
        logger.error(f"质量检测参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"质量检测失败: {e}")
        raise HTTPException(status_code=500, detail="质量检测服务异常")

@router.post("/batch",
             response_model=DataResponse,
             summary="批量质量检测",
             description="对多个文档进行批量质量检测")
async def batch_quality_check(
    request: BatchQualityCheckRequest,
    background_tasks: BackgroundTasks,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    批量质量检测
    
    对多个文档进行并行质量检测，支持异步处理和进度跟踪。
    """
    try:
        logger.info(f"开始批量质量检测，文档数量: {len(request.content_ids)}")
        
        # 验证批量大小
        if len(request.content_ids) > settings.MAX_BATCH_SIZE:
            raise ValueError(f"批量大小超过限制 ({settings.MAX_BATCH_SIZE})")
        
        # 创建批量处理任务
        batch_result = BatchProcessingResult(
            total_items=len(request.content_ids),
            results=[],
            errors=[]
        )
        
        if request.parallel_processing:
            # 并行处理
            await _process_batch_parallel(
                request, batch_result, storage_client, background_tasks
            )
        else:
            # 串行处理
            await _process_batch_sequential(
                request, batch_result, storage_client
            )
        
        return DataResponse(
            success=True,
            message=f"批量质量检测完成，成功处理{batch_result.completed_items}个文档",
            data={
                "batch_id": batch_result.batch_id,
                "total_items": batch_result.total_items,
                "completed_items": batch_result.completed_items,
                "failed_items": batch_result.failed_items,
                "success_rate": batch_result.success_rate,
                "processing_time_ms": batch_result.processing_time_ms,
                "results": batch_result.results,
                "errors": batch_result.errors
            }
        )
        
    except ValueError as e:
        logger.error(f"批量质量检测参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量质量检测失败: {e}")
        raise HTTPException(status_code=500, detail="批量质量检测服务异常")

@router.get("/history/{content_id}",
            response_model=DataResponse,
            summary="质量检测历史",
            description="获取指定内容的质量检测历史记录")
async def get_quality_history(
    content_id: str,
    limit: int = 10,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取质量检测历史
    
    返回指定内容的历史质量检测记录，包括分数变化趋势。
    """
    try:
        logger.info(f"获取质量检测历史: content_id={content_id}")
        
        # 通过storage-service获取历史记录
        history_result = await storage_client.get_quality_history(content_id, limit)
        
        return DataResponse(
            success=True,
            message="质量检测历史获取成功",
            data=history_result.get("data", {})
        )
        
    except Exception as e:
        logger.error(f"获取质量检测历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取质量检测历史失败")

@router.get("/result/{check_id}",
            response_model=DataResponse,
            summary="获取检测结果",
            description="根据检测ID获取质量检测结果")
async def get_quality_result(
    check_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取质量检测结果
    
    根据检测ID返回详细的质量检测结果。
    """
    try:
        logger.info(f"获取质量检测结果: check_id={check_id}")
        
        # 通过storage-service获取检测结果
        result = await storage_client.get_quality_check_result(check_id)
        
        if not result.get("data"):
            raise HTTPException(status_code=404, detail="检测结果不存在")
        
        return DataResponse(
            success=True,
            message="质量检测结果获取成功",
            data=result.get("data")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取质量检测结果失败: {e}")
        raise HTTPException(status_code=500, detail="获取质量检测结果失败")

@router.get("/statistics",
            response_model=DataResponse,
            summary="质量统计",
            description="获取质量检测统计数据")
async def get_quality_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取质量统计数据
    
    返回指定时间范围内的质量检测统计信息。
    """
    try:
        logger.info(f"获取质量统计数据: {start_date} - {end_date}")
        
        # 通过storage-service获取统计数据
        stats_result = await storage_client.get_quality_statistics(start_date, end_date)
        
        return DataResponse(
            success=True,
            message="质量统计数据获取成功",
            data=stats_result.get("data", {})
        )
        
    except Exception as e:
        logger.error(f"获取质量统计数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取质量统计数据失败")

# ==================== 内部辅助函数 ====================

async def _process_batch_parallel(
    request: BatchQualityCheckRequest,
    batch_result: BatchProcessingResult,
    storage_client: StorageServiceClient,
    background_tasks: BackgroundTasks
) -> None:
    """并行处理批量检测"""
    import time
    start_time = time.time()
    
    # 创建质量检测引擎
    detection_engine = QualityDetectionEngine(storage_client)
    
    # 创建并发任务
    semaphore = asyncio.Semaphore(request.max_concurrent_tasks)
    tasks = []
    
    for content_id in request.content_ids:
        task = _process_single_content_with_semaphore(
            semaphore, detection_engine, content_id, request.check_options
        )
        tasks.append(task)
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    for i, result in enumerate(results):
        content_id = request.content_ids[i]
        
        if isinstance(result, Exception):
            batch_result.failed_items += 1
            batch_result.errors.append({
                "content_id": content_id,
                "error": str(result)
            })
        else:
            batch_result.completed_items += 1
            batch_result.results.append({
                "content_id": content_id,
                "check_id": result.check_id,
                "overall_score": result.overall_score,
                "status": result.status
            })
    
    # 计算成功率和处理时间
    batch_result.success_rate = batch_result.completed_items / batch_result.total_items
    batch_result.processing_time_ms = int((time.time() - start_time) * 1000)

async def _process_batch_sequential(
    request: BatchQualityCheckRequest,
    batch_result: BatchProcessingResult,
    storage_client: StorageServiceClient
) -> None:
    """串行处理批量检测"""
    import time
    start_time = time.time()
    
    # 创建质量检测引擎
    detection_engine = QualityDetectionEngine(storage_client)
    
    for content_id in request.content_ids:
        try:
            # 这里需要获取实际内容，简化处理
            content = f"模拟内容 {content_id}"
            
            # 创建检测请求
            check_request = QualityCheckRequest(
                content=content,
                check_options=request.check_options
            )
            
            # 执行检测
            result = await detection_engine.check_quality(check_request)
            
            batch_result.completed_items += 1
            batch_result.results.append({
                "content_id": content_id,
                "check_id": result.check_id,
                "overall_score": result.overall_score,
                "status": result.status
            })
            
        except Exception as e:
            batch_result.failed_items += 1
            batch_result.errors.append({
                "content_id": content_id,
                "error": str(e)
            })
    
    # 计算成功率和处理时间
    batch_result.success_rate = batch_result.completed_items / batch_result.total_items if batch_result.total_items > 0 else 0
    batch_result.processing_time_ms = int((time.time() - start_time) * 1000)

async def _process_single_content_with_semaphore(
    semaphore: asyncio.Semaphore,
    detection_engine: QualityDetectionEngine,
    content_id: str,
    check_options: Dict[str, bool]
) -> QualityCheckResult:
    """使用信号量控制的单个内容处理"""
    async with semaphore:
        # 这里需要获取实际内容，简化处理
        content = f"模拟内容 {content_id}"
        
        # 创建检测请求
        request = QualityCheckRequest(
            content=content,
            check_options=check_options
        )
        
        # 执行检测
        return await detection_engine.check_quality(request)