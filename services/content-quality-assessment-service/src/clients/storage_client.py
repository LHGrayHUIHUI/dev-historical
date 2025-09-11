"""
Storage Service 客户端

与storage-service通信的HTTP客户端，提供评估结果存储、历史数据查询、
基准管理等数据访问功能。
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime
import json

from ..config.settings import settings
from ..models.assessment_models import (
    QualityAssessmentResult, QualityBenchmark, QualityTrendAnalysis,
    BenchmarkComparison, QualityDashboardData
)

logger = logging.getLogger(__name__)

class StorageServiceClient:
    """Storage Service HTTP客户端"""
    
    def __init__(self):
        self.base_url = settings.external_services.storage_service_url
        self.timeout = settings.external_services.storage_service_timeout
        self.retries = settings.external_services.storage_service_retries
        self._session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._close_session()
    
    async def _create_session(self):
        """创建HTTP会话"""
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "content-quality-assessment-service/1.0.0"
                }
            )
    
    async def _close_session(self):
        """关闭HTTP会话"""
        if self._session:
            await self._session.aclose()
            self._session = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, method: str, endpoint: str, 
                          data: Optional[Dict] = None,
                          params: Optional[Dict] = None) -> Dict[str, Any]:
        """发送HTTP请求"""
        if not self._session:
            await self._create_session()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == "GET":
                response = await self._session.get(url, params=params)
            elif method.upper() == "POST":
                response = await self._session.post(url, json=data, params=params)
            elif method.upper() == "PUT":
                response = await self._session.put(url, json=data, params=params)
            elif method.upper() == "DELETE":
                response = await self._session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"Storage service request successful: {response.status_code}")
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Storage service HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Storage service request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Storage service unexpected error: {str(e)}")
            raise
    
    # ==================== 评估结果管理 ====================
    
    async def save_assessment_result(self, result: QualityAssessmentResult) -> Dict[str, Any]:
        """保存评估结果"""
        try:
            data = {
                "assessment_id": result.assessment_id,
                "content_id": result.content_id,
                "content_type": result.content_type,
                "overall_score": result.overall_score,
                "grade": result.grade,
                "metrics": [metric.dict() for metric in result.metrics],
                "strengths": result.strengths,
                "weaknesses": result.weaknesses,
                "recommendations": result.recommendations,
                "assessment_time": result.assessment_time.isoformat(),
                "processing_duration": result.processing_duration,
                "model_versions": result.model_versions,
                "status": result.status,
                "error_info": result.error_info.dict() if result.error_info else None,
                "processing_metrics": result.processing_metrics.dict() if result.processing_metrics else None
            }
            
            response = await self._make_request(
                "POST", 
                "/api/v1/quality-assessments",
                data=data
            )
            
            logger.info(f"Assessment result saved: {result.assessment_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to save assessment result: {str(e)}")
            raise
    
    async def get_assessment_result(self, assessment_id: str) -> Dict[str, Any]:
        """获取评估结果"""
        try:
            response = await self._make_request(
                "GET",
                f"/api/v1/quality-assessments/{assessment_id}"
            )
            
            logger.debug(f"Retrieved assessment result: {assessment_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get assessment result {assessment_id}: {str(e)}")
            raise
    
    async def get_content_assessments(self, 
                                    content_id: str,
                                    limit: int = 10,
                                    offset: int = 0,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取内容的历史评估记录"""
        try:
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            response = await self._make_request(
                "GET",
                f"/api/v1/content/{content_id}/quality-assessments",
                params=params
            )
            
            logger.debug(f"Retrieved {len(response.get('data', []))} assessments for content {content_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get content assessments for {content_id}: {str(e)}")
            raise
    
    async def delete_assessment_result(self, assessment_id: str) -> Dict[str, Any]:
        """删除评估结果"""
        try:
            response = await self._make_request(
                "DELETE",
                f"/api/v1/quality-assessments/{assessment_id}"
            )
            
            logger.info(f"Assessment result deleted: {assessment_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to delete assessment result {assessment_id}: {str(e)}")
            raise
    
    # ==================== 基准管理 ====================
    
    async def create_quality_benchmark(self, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """创建质量基准"""
        try:
            response = await self._make_request(
                "POST",
                "/api/v1/quality-benchmarks",
                data=benchmark
            )
            
            logger.info(f"Quality benchmark created: {benchmark.get('benchmark_id')}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to create quality benchmark: {str(e)}")
            raise
    
    async def get_quality_benchmark(self, benchmark_id: str) -> Dict[str, Any]:
        """获取质量基准"""
        try:
            response = await self._make_request(
                "GET",
                f"/api/v1/quality-benchmarks/{benchmark_id}"
            )
            
            logger.debug(f"Retrieved quality benchmark: {benchmark_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get quality benchmark {benchmark_id}: {str(e)}")
            raise
    
    async def update_quality_benchmark(self, 
                                     benchmark_id: str,
                                     benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新质量基准"""
        try:
            response = await self._make_request(
                "PUT",
                f"/api/v1/quality-benchmarks/{benchmark_id}",
                data=benchmark_data
            )
            
            logger.info(f"Quality benchmark updated: {benchmark_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to update quality benchmark {benchmark_id}: {str(e)}")
            raise
    
    async def list_quality_benchmarks(self,
                                    content_type: Optional[str] = None,
                                    is_active: Optional[bool] = None,
                                    limit: int = 50,
                                    offset: int = 0) -> Dict[str, Any]:
        """列出质量基准"""
        try:
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if content_type:
                params["content_type"] = content_type
            if is_active is not None:
                params["is_active"] = is_active
            
            response = await self._make_request(
                "GET",
                "/api/v1/quality-benchmarks",
                params=params
            )
            
            logger.debug(f"Retrieved {len(response.get('data', []))} quality benchmarks")
            return response
            
        except Exception as e:
            logger.error(f"Failed to list quality benchmarks: {str(e)}")
            raise
    
    async def delete_quality_benchmark(self, benchmark_id: str) -> Dict[str, Any]:
        """删除质量基准"""
        try:
            response = await self._make_request(
                "DELETE",
                f"/api/v1/quality-benchmarks/{benchmark_id}"
            )
            
            logger.info(f"Quality benchmark deleted: {benchmark_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to delete quality benchmark {benchmark_id}: {str(e)}")
            raise
    
    # ==================== 趋势分析数据 ====================
    
    async def save_trend_analysis(self, analysis: QualityTrendAnalysis) -> Dict[str, Any]:
        """保存趋势分析结果"""
        try:
            data = {
                "analysis_id": analysis.analysis_id,
                "content_id": analysis.content_id,
                "analysis_period_start": analysis.analysis_period[0].isoformat(),
                "analysis_period_end": analysis.analysis_period[1].isoformat(),
                "overall_trend": analysis.overall_trend.dict(),
                "dimension_trends": [trend.dict() for trend in analysis.dimension_trends],
                "improvement_suggestions": analysis.improvement_suggestions,
                "risk_alerts": analysis.risk_alerts,
                "trend_summary": analysis.trend_summary,
                "next_assessment_date": analysis.next_assessment_date.isoformat() if analysis.next_assessment_date else None,
                "predicted_performance": analysis.predicted_performance,
                "analysis_time": analysis.analysis_time.isoformat(),
                "min_data_points_met": analysis.min_data_points_met,
                "confidence_level": analysis.confidence_level
            }
            
            response = await self._make_request(
                "POST",
                "/api/v1/quality-trend-analyses",
                data=data
            )
            
            logger.info(f"Trend analysis saved: {analysis.analysis_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to save trend analysis: {str(e)}")
            raise
    
    async def get_trend_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """获取趋势分析结果"""
        try:
            response = await self._make_request(
                "GET",
                f"/api/v1/quality-trend-analyses/{analysis_id}"
            )
            
            logger.debug(f"Retrieved trend analysis: {analysis_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get trend analysis {analysis_id}: {str(e)}")
            raise
    
    async def get_content_trend_analyses(self,
                                       content_id: str,
                                       limit: int = 10,
                                       offset: int = 0) -> Dict[str, Any]:
        """获取内容的趋势分析历史"""
        try:
            params = {
                "limit": limit,
                "offset": offset
            }
            
            response = await self._make_request(
                "GET",
                f"/api/v1/content/{content_id}/quality-trend-analyses",
                params=params
            )
            
            logger.debug(f"Retrieved {len(response.get('data', []))} trend analyses for content {content_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get content trend analyses for {content_id}: {str(e)}")
            raise
    
    # ==================== 基准对比数据 ====================
    
    async def save_benchmark_comparison(self, comparison: BenchmarkComparison) -> Dict[str, Any]:
        """保存基准对比结果"""
        try:
            data = {
                "comparison_id": comparison.comparison_id,
                "content_id": comparison.content_id,
                "benchmark_id": comparison.benchmark_id,
                "assessment_result": comparison.assessment_result.dict(),
                "comparison_details": {
                    str(k): v for k, v in comparison.comparison_details.items()
                },
                "meets_standard": comparison.meets_standard,
                "compliance_score": comparison.compliance_score,
                "improvement_gaps": comparison.improvement_gaps,
                "priority_improvements": comparison.priority_improvements,
                "estimated_effort": comparison.estimated_effort,
                "comparison_time": comparison.comparison_time.isoformat(),
                "benchmark_version": comparison.benchmark_version
            }
            
            response = await self._make_request(
                "POST",
                "/api/v1/benchmark-comparisons",
                data=data
            )
            
            logger.info(f"Benchmark comparison saved: {comparison.comparison_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to save benchmark comparison: {str(e)}")
            raise
    
    async def get_benchmark_comparison(self, comparison_id: str) -> Dict[str, Any]:
        """获取基准对比结果"""
        try:
            response = await self._make_request(
                "GET",
                f"/api/v1/benchmark-comparisons/{comparison_id}"
            )
            
            logger.debug(f"Retrieved benchmark comparison: {comparison_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get benchmark comparison {comparison_id}: {str(e)}")
            raise
    
    # ==================== 仪表板数据 ====================
    
    async def get_quality_dashboard_data(self,
                                       content_id: str,
                                       period_days: int = 30) -> Dict[str, Any]:
        """获取质量仪表板数据"""
        try:
            params = {
                "period_days": period_days
            }
            
            response = await self._make_request(
                "GET",
                f"/api/v1/content/{content_id}/quality-dashboard",
                params=params
            )
            
            logger.debug(f"Retrieved quality dashboard data for content {content_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get quality dashboard data for {content_id}: {str(e)}")
            raise
    
    # ==================== 统计和分析 ====================
    
    async def get_quality_statistics(self,
                                   content_ids: Optional[List[str]] = None,
                                   content_type: Optional[str] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取质量统计数据"""
        try:
            params = {}
            
            if content_ids:
                params["content_ids"] = ",".join(content_ids)
            if content_type:
                params["content_type"] = content_type
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            response = await self._make_request(
                "GET",
                "/api/v1/quality-statistics",
                params=params
            )
            
            logger.debug("Retrieved quality statistics")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get quality statistics: {str(e)}")
            raise
    
    async def get_assessment_summary(self,
                                   period_days: int = 7) -> Dict[str, Any]:
        """获取评估摘要统计"""
        try:
            params = {
                "period_days": period_days
            }
            
            response = await self._make_request(
                "GET",
                "/api/v1/assessment-summary",
                params=params
            )
            
            logger.debug(f"Retrieved assessment summary for {period_days} days")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get assessment summary: {str(e)}")
            raise
    
    # ==================== 健康检查 ====================
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            response = await self._make_request("GET", "/health")
            return response.get("status") == "healthy"
            
        except Exception as e:
            logger.error(f"Storage service health check failed: {str(e)}")
            return False

# 全局客户端实例
storage_client = StorageServiceClient()