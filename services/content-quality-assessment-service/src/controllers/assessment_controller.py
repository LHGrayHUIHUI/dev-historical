"""
质量评估控制器

提供质量评估的REST API接口，包括单次评估、批量评估、趋势分析、
基准管理等功能的HTTP端点。
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import time

from ..config.settings import settings
from ..models.assessment_models import (
    QualityAssessmentRequest, QualityAssessmentResult, BatchAssessmentRequest,
    BatchAssessmentResult, QualityTrendAnalysis, QualityBenchmark,
    BenchmarkComparison, QualityDashboardData, ContentType, QualityDimension,
    AssessmentResponse, BatchAssessmentResponse, TrendAnalysisResponse,
    BenchmarkResponse, ComparisonResponse, DashboardResponse, ListResponse,
    HealthStatus
)
from ..services.assessment_engine import assessment_engine
from ..services.trend_analyzer import trend_analyzer
from ..services.benchmark_manager import benchmark_manager
from ..clients.storage_client import storage_client

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/quality", tags=["质量评估"])

# ==================== 依赖注入 ====================

async def get_assessment_engine():
    """获取评估引擎实例"""
    return assessment_engine

async def get_trend_analyzer():
    """获取趋势分析器实例"""
    return trend_analyzer

async def get_benchmark_manager():
    """获取基准管理器实例"""
    return benchmark_manager

# ==================== 质量评估接口 ====================

@router.post("/assess", response_model=AssessmentResponse)
async def assess_content_quality(
    request: QualityAssessmentRequest,
    background_tasks: BackgroundTasks,
    engine: assessment_engine = Depends(get_assessment_engine)
) -> AssessmentResponse:
    """
    评估内容质量
    
    Args:
        request: 评估请求
        background_tasks: 后台任务
        engine: 评估引擎
        
    Returns:
        AssessmentResponse: 评估结果
    """
    try:
        start_time = time.time()
        
        # 验证请求
        if len(request.content) > settings.assessment_engine.max_content_length:
            raise HTTPException(
                status_code=400,
                detail=f"内容长度超过限制 ({settings.assessment_engine.max_content_length}字符)"
            )
        
        logger.info(f"Starting quality assessment for content {request.content_id}")
        
        # 执行评估
        result = await engine.assess_quality(request)
        
        # 记录评估时间
        processing_time = time.time() - start_time
        logger.info(f"Quality assessment completed for {request.content_id} in {processing_time:.2f}s")
        
        return AssessmentResponse(
            success=True,
            message="质量评估完成",
            data=result
        )
        
    except ValueError as e:
        logger.warning(f"Assessment validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")

@router.get("/assessment/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment_result(assessment_id: str):
    """获取评估结果"""
    try:
        response = await storage_client.get_assessment_result(assessment_id)
        
        if response.get('success') and response.get('data'):
            result = QualityAssessmentResult(**response['data'])
            return AssessmentResponse(
                success=True,
                message="获取评估结果成功",
                data=result
            )
        else:
            raise HTTPException(status_code=404, detail="评估结果未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get assessment result {assessment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取评估结果失败: {str(e)}")

@router.get("/content/{content_id}/assessments", response_model=ListResponse)
async def get_content_assessments(
    content_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None)
):
    """获取内容的历史评估记录"""
    try:
        response = await storage_client.get_content_assessments(
            content_id=content_id,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date
        )
        
        if response.get('success'):
            assessments = response.get('data', [])
            total = response.get('total', len(assessments))
            
            return ListResponse(
                success=True,
                message=f"获取到{len(assessments)}条评估记录",
                data=assessments,
                total=total,
                page=offset // limit + 1,
                page_size=limit,
                has_next=offset + limit < total
            )
        else:
            raise HTTPException(status_code=500, detail="获取评估记录失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get content assessments for {content_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取评估记录失败: {str(e)}")

@router.delete("/assessment/{assessment_id}")
async def delete_assessment_result(assessment_id: str):
    """删除评估结果"""
    try:
        response = await storage_client.delete_assessment_result(assessment_id)
        
        if response.get('success'):
            return {"success": True, "message": "评估结果已删除"}
        else:
            raise HTTPException(status_code=404, detail="评估结果未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete assessment result {assessment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除评估结果失败: {str(e)}")

# ==================== 批量评估接口 ====================

@router.post("/batch-assess", response_model=BatchAssessmentResponse)
async def batch_assess_content_quality(
    request: BatchAssessmentRequest,
    background_tasks: BackgroundTasks,
    engine: assessment_engine = Depends(get_assessment_engine)
) -> BatchAssessmentResponse:
    """批量评估内容质量"""
    try:
        if len(request.requests) > settings.assessment_engine.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"批量请求数量超过限制 ({settings.assessment_engine.max_batch_size})"
            )
        
        logger.info(f"Starting batch assessment for {len(request.requests)} contents")
        
        batch_start_time = datetime.now()
        results = []
        failed_requests = []
        
        if request.parallel_processing:
            # 并行处理
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def process_single_request(req):
                async with semaphore:
                    try:
                        return await engine.assess_quality(req)
                    except Exception as e:
                        failed_requests.append({
                            "content_id": req.content_id,
                            "error": str(e)
                        })
                        return None
            
            # 创建并行任务
            tasks = [process_single_request(req) for req in request.requests]
            
            # 执行并行评估，设置超时
            try:
                completed_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=request.timeout_minutes * 60
                )
                
                results = [r for r in completed_results if r is not None and not isinstance(r, Exception)]
                
            except asyncio.TimeoutError:
                logger.error(f"Batch assessment timeout after {request.timeout_minutes} minutes")
                raise HTTPException(status_code=408, detail="批量评估超时")
        
        else:
            # 串行处理
            for req in request.requests:
                try:
                    result = await engine.assess_quality(req)
                    results.append(result)
                except Exception as e:
                    failed_requests.append({
                        "content_id": req.content_id,
                        "error": str(e)
                    })
        
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        
        # 构建批量结果
        batch_result = BatchAssessmentResult(
            batch_id=request.batch_id,
            total_requests=len(request.requests),
            completed_count=len(results),
            failed_count=len(failed_requests),
            results=results,
            failed_requests=failed_requests,
            batch_start_time=batch_start_time,
            batch_end_time=batch_end_time,
            total_duration=batch_duration
        )
        
        logger.info(f"Batch assessment completed: {len(results)} successful, {len(failed_requests)} failed")
        
        return BatchAssessmentResponse(
            success=True,
            message=f"批量评估完成，成功{len(results)}个，失败{len(failed_requests)}个",
            data=batch_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量评估失败: {str(e)}")

# ==================== 趋势分析接口 ====================

@router.post("/trend-analysis", response_model=TrendAnalysisResponse)
async def analyze_quality_trend(
    content_id: str,
    start_date: datetime,
    end_date: datetime,
    analyzer: trend_analyzer = Depends(get_trend_analyzer)
):
    """分析质量趋势"""
    try:
        # 验证日期范围
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="开始日期必须早于结束日期")
        
        period_days = (end_date - start_date).days
        if period_days > settings.trend_analysis.analysis_window_days * 2:
            raise HTTPException(
                status_code=400,
                detail=f"分析周期不能超过{settings.trend_analysis.analysis_window_days * 2}天"
            )
        
        logger.info(f"Starting trend analysis for content {content_id}")
        
        analysis = await analyzer.analyze_quality_trend(content_id, start_date, end_date)
        
        return TrendAnalysisResponse(
            success=True,
            message="趋势分析完成",
            data=analysis
        )
        
    except ValueError as e:
        logger.warning(f"Trend analysis validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"趋势分析失败: {str(e)}")

@router.get("/trend-analysis/{analysis_id}", response_model=TrendAnalysisResponse)
async def get_trend_analysis(analysis_id: str):
    """获取趋势分析结果"""
    try:
        response = await storage_client.get_trend_analysis(analysis_id)
        
        if response.get('success') and response.get('data'):
            analysis = QualityTrendAnalysis(**response['data'])
            return TrendAnalysisResponse(
                success=True,
                message="获取趋势分析成功",
                data=analysis
            )
        else:
            raise HTTPException(status_code=404, detail="趋势分析结果未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trend analysis {analysis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取趋势分析失败: {str(e)}")

@router.get("/content/{content_id}/trend-analyses", response_model=ListResponse)
async def get_content_trend_analyses(
    content_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0)
):
    """获取内容的趋势分析历史"""
    try:
        response = await storage_client.get_content_trend_analyses(
            content_id=content_id,
            limit=limit,
            offset=offset
        )
        
        if response.get('success'):
            analyses = response.get('data', [])
            total = response.get('total', len(analyses))
            
            return ListResponse(
                success=True,
                message=f"获取到{len(analyses)}条趋势分析记录",
                data=analyses,
                total=total,
                page=offset // limit + 1,
                page_size=limit,
                has_next=offset + limit < total
            )
        else:
            raise HTTPException(status_code=500, detail="获取趋势分析记录失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get content trend analyses for {content_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取趋势分析记录失败: {str(e)}")

# ==================== 基准管理接口 ====================

@router.post("/benchmarks", response_model=BenchmarkResponse)
async def create_benchmark(
    benchmark: QualityBenchmark,
    manager: benchmark_manager = Depends(get_benchmark_manager)
):
    """创建质量基准"""
    try:
        # 检查自定义基准数量限制
        existing_benchmarks = await manager.list_benchmarks(
            content_type=benchmark.content_type,
            is_active=True
        )
        
        custom_benchmarks = [b for b in existing_benchmarks if not b.is_default]
        if len(custom_benchmarks) >= settings.benchmark.max_custom_benchmarks:
            raise HTTPException(
                status_code=400,
                detail=f"自定义基准数量已达上限 ({settings.benchmark.max_custom_benchmarks})"
            )
        
        benchmark_id = await manager.create_benchmark(benchmark)
        benchmark.benchmark_id = benchmark_id
        
        return BenchmarkResponse(
            success=True,
            message="质量基准创建成功",
            data=benchmark
        )
        
    except ValueError as e:
        logger.warning(f"Benchmark creation validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Benchmark creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建质量基准失败: {str(e)}")

@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkResponse)
async def get_benchmark(
    benchmark_id: str,
    manager: benchmark_manager = Depends(get_benchmark_manager)
):
    """获取质量基准"""
    try:
        benchmark = await manager.get_benchmark(benchmark_id)
        
        if benchmark:
            return BenchmarkResponse(
                success=True,
                message="获取质量基准成功",
                data=benchmark
            )
        else:
            raise HTTPException(status_code=404, detail="质量基准未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get benchmark {benchmark_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取质量基准失败: {str(e)}")

@router.put("/benchmarks/{benchmark_id}", response_model=BenchmarkResponse)
async def update_benchmark(
    benchmark_id: str,
    update_data: Dict[str, Any],
    manager: benchmark_manager = Depends(get_benchmark_manager)
):
    """更新质量基准"""
    try:
        success = await manager.update_benchmark(benchmark_id, update_data)
        
        if success:
            updated_benchmark = await manager.get_benchmark(benchmark_id)
            return BenchmarkResponse(
                success=True,
                message="质量基准更新成功",
                data=updated_benchmark
            )
        else:
            raise HTTPException(status_code=404, detail="质量基准未找到")
            
    except ValueError as e:
        logger.warning(f"Benchmark update validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新质量基准失败: {str(e)}")

@router.get("/benchmarks", response_model=ListResponse)
async def list_benchmarks(
    content_type: Optional[ContentType] = Query(default=None),
    is_active: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    manager: benchmark_manager = Depends(get_benchmark_manager)
):
    """列出质量基准"""
    try:
        benchmarks = await manager.list_benchmarks(
            content_type=content_type,
            is_active=is_active,
            limit=limit,
            offset=offset
        )
        
        return ListResponse(
            success=True,
            message=f"获取到{len(benchmarks)}个质量基准",
            data=[b.dict() for b in benchmarks],
            total=len(benchmarks),  # 简化处理
            page=offset // limit + 1,
            page_size=limit,
            has_next=len(benchmarks) == limit
        )
        
    except Exception as e:
        logger.error(f"Failed to list benchmarks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取质量基准列表失败: {str(e)}")

@router.delete("/benchmarks/{benchmark_id}")
async def delete_benchmark(
    benchmark_id: str,
    manager: benchmark_manager = Depends(get_benchmark_manager)
):
    """删除质量基准"""
    try:
        success = await manager.delete_benchmark(benchmark_id)
        
        if success:
            return {"success": True, "message": "质量基准已删除"}
        else:
            raise HTTPException(status_code=404, detail="质量基准未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete benchmark {benchmark_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除质量基准失败: {str(e)}")

@router.post("/compare-benchmark", response_model=ComparisonResponse)
async def compare_with_benchmark(
    assessment_id: str,
    benchmark_id: str,
    manager: benchmark_manager = Depends(get_benchmark_manager)
):
    """与基准对比"""
    try:
        # 获取评估结果
        response = await storage_client.get_assessment_result(assessment_id)
        
        if not response.get('success') or not response.get('data'):
            raise HTTPException(status_code=404, detail="评估结果未找到")
        
        assessment_result = QualityAssessmentResult(**response['data'])
        
        # 执行基准对比
        comparison = await manager.compare_with_benchmark(assessment_result, benchmark_id)
        
        return ComparisonResponse(
            success=True,
            message="基准对比完成",
            data=comparison
        )
        
    except ValueError as e:
        logger.warning(f"Benchmark comparison validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"基准对比失败: {str(e)}")

@router.get("/benchmarks/{benchmark_id}/performance")
async def get_benchmark_performance(
    benchmark_id: str,
    period_days: int = Query(default=30, ge=1, le=365),
    manager: benchmark_manager = Depends(get_benchmark_manager)
):
    """获取基准性能分析"""
    try:
        analysis = await manager.analyze_benchmark_performance(benchmark_id, period_days)
        
        if analysis:
            return {
                "success": True,
                "message": "基准性能分析完成",
                "data": analysis
            }
        else:
            raise HTTPException(status_code=404, detail="基准未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark performance analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"基准性能分析失败: {str(e)}")

# ==================== 仪表板接口 ====================

@router.get("/dashboard/{content_id}", response_model=DashboardResponse)
async def get_quality_dashboard(
    content_id: str,
    period_days: int = Query(default=30, ge=1, le=365)
):
    """获取质量仪表板数据"""
    try:
        response = await storage_client.get_quality_dashboard_data(content_id, period_days)
        
        if response.get('success') and response.get('data'):
            dashboard_data = QualityDashboardData(**response['data'])
            return DashboardResponse(
                success=True,
                message="获取仪表板数据成功",
                data=dashboard_data
            )
        else:
            raise HTTPException(status_code=404, detail="仪表板数据未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quality dashboard for {content_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取仪表板数据失败: {str(e)}")

# ==================== 统计接口 ====================

@router.get("/statistics")
async def get_quality_statistics(
    content_ids: Optional[str] = Query(default=None, description="逗号分隔的内容ID列表"),
    content_type: Optional[ContentType] = Query(default=None),
    start_date: Optional[datetime] = Query(default=None),
    end_date: Optional[datetime] = Query(default=None)
):
    """获取质量统计数据"""
    try:
        content_id_list = None
        if content_ids:
            content_id_list = [cid.strip() for cid in content_ids.split(',') if cid.strip()]
        
        content_type_str = content_type.value if content_type else None
        
        response = await storage_client.get_quality_statistics(
            content_ids=content_id_list,
            content_type=content_type_str,
            start_date=start_date,
            end_date=end_date
        )
        
        if response.get('success'):
            return {
                "success": True,
                "message": "获取质量统计成功",
                "data": response.get('data', {})
            }
        else:
            raise HTTPException(status_code=500, detail="获取质量统计失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quality statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取质量统计失败: {str(e)}")

@router.get("/summary")
async def get_assessment_summary(
    period_days: int = Query(default=7, ge=1, le=365)
):
    """获取评估摘要统计"""
    try:
        response = await storage_client.get_assessment_summary(period_days)
        
        if response.get('success'):
            return {
                "success": True,
                "message": "获取评估摘要成功",
                "data": response.get('data', {})
            }
        else:
            raise HTTPException(status_code=500, detail="获取评估摘要失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get assessment summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取评估摘要失败: {str(e)}")

# ==================== 系统接口 ====================

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """健康检查"""
    try:
        # 检查外部服务连接
        storage_healthy = await storage_client.health_check()
        
        dependencies = {
            "storage_service": storage_healthy,
            "assessment_engine": assessment_engine.nlp is not None,
            "redis_cache": assessment_engine.redis is not None if assessment_engine.cache_enabled else True
        }
        
        all_healthy = all(dependencies.values())
        status = "healthy" if all_healthy else "degraded"
        
        return HealthStatus(
            status=status,
            version="1.0.0",
            dependencies=dependencies,
            system_info={
                "service_name": settings.service.name,
                "environment": settings.service.environment,
                "enabled_dimensions": settings.assessment_engine.enabled_dimensions,
                "cache_enabled": settings.assessment_engine.cache_assessment_results
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthStatus(
            status="unhealthy",
            version="1.0.0",
            dependencies={},
            system_info={"error": str(e)}
        )

@router.get("/ready")
async def readiness_check():
    """就绪检查 - Kubernetes就绪探针"""
    try:
        # 验证关键依赖是否可用
        storage_healthy = await storage_client.health_check()
        engine_ready = assessment_engine.nlp is not None
        
        if storage_healthy and engine_ready:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/info")
async def service_info():
    """服务信息"""
    return {
        "service": settings.service.name,
        "version": settings.service.version,
        "description": settings.service.description,
        "features": [
            "多维度质量评估",
            "质量趋势分析",
            "基准管理和对比",
            "批量评估处理",
            "智能改进建议"
        ],
        "supported_dimensions": [dim.value for dim in QualityDimension],
        "supported_content_types": [ct.value for ct in ContentType],
        "port": settings.service.port,
        "environment": settings.service.environment,
        "ai_integration": True,
        "storage_integration": True,
        "cache_enabled": settings.assessment_engine.cache_assessment_results
    }