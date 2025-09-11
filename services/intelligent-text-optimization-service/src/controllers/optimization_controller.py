"""
优化控制器 - Optimization Controller

提供文本优化相关的REST API接口，包括单文档优化、
批量优化、版本管理等功能的HTTP接口

核心功能:
1. 单文档文本优化API
2. 批量文档优化API  
3. 优化任务状态查询
4. 版本管理和对比
5. 优化策略管理
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..config.settings import get_settings
from ..models.optimization_models import (
    OptimizationRequest, OptimizationResult, ApiResponse,
    BatchOptimizationRequest, BatchOptimizationStatus,
    VersionComparison, HealthStatus
)
from ..services.text_optimization_engine import TextOptimizationEngine, TextOptimizationError
from ..services.batch_optimization_manager import BatchOptimizationManager, BatchOptimizationError
from ..services.optimization_strategy_manager import OptimizationStrategyManager, StrategySelectionError
from ..clients.storage_service_client import StorageServiceClient, StorageServiceError
from ..clients.ai_model_service_client import AIModelServiceClient, AIModelServiceError


logger = logging.getLogger(__name__)


class OptimizationController:
    """
    优化控制器
    处理文本优化相关的HTTP请求
    """
    
    def __init__(self):
        """初始化优化控制器"""
        self.settings = get_settings()
        self.router = APIRouter(prefix="/api/v1/optimization", tags=["文本优化"])
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化依赖的服务组件
        self._init_dependencies()
        
        # 注册路由
        self._register_routes()
    
    def _init_dependencies(self):
        """初始化依赖组件"""
        # 初始化客户端
        self.storage_client = StorageServiceClient()
        self.ai_client = AIModelServiceClient()
        
        # 初始化服务组件
        self.strategy_manager = OptimizationStrategyManager(self.storage_client)
        self.optimization_engine = TextOptimizationEngine(self.ai_client, self.storage_client)
        self.batch_manager = BatchOptimizationManager(self.optimization_engine, self.storage_client)
    
    def _register_routes(self):
        """注册API路由"""
        # 单文档优化
        self.router.add_api_route(
            "/optimize", 
            self.optimize_text, 
            methods=["POST"],
            summary="单文档文本优化",
            description="对单个文档进行文本优化处理"
        )
        
        # 批量优化
        self.router.add_api_route(
            "/batch",
            self.create_batch_optimization,
            methods=["POST"],
            summary="创建批量优化任务",
            description="创建批量文档优化任务"
        )
        
        # 批量任务状态查询
        self.router.add_api_route(
            "/batch/{job_id}/status",
            self.get_batch_status,
            methods=["GET"],
            summary="查询批量任务状态",
            description="查询批量优化任务的执行状态和进度"
        )
        
        # 任务控制
        self.router.add_api_route(
            "/batch/{job_id}/pause",
            self.pause_batch_job,
            methods=["POST"],
            summary="暂停批量任务"
        )
        
        self.router.add_api_route(
            "/batch/{job_id}/resume",
            self.resume_batch_job,
            methods=["POST"],
            summary="恢复批量任务"
        )
        
        self.router.add_api_route(
            "/batch/{job_id}/cancel",
            self.cancel_batch_job,
            methods=["POST"],
            summary="取消批量任务"
        )
        
        # 版本管理
        self.router.add_api_route(
            "/tasks/{task_id}/versions",
            self.get_optimization_versions,
            methods=["GET"],
            summary="获取优化版本列表",
            description="获取指定任务的所有优化版本"
        )
        
        self.router.add_api_route(
            "/compare",
            self.compare_versions,
            methods=["GET"],
            summary="版本对比",
            description="对比两个优化版本的差异"
        )
        
        self.router.add_api_route(
            "/tasks/{task_id}/select-version",
            self.select_version,
            methods=["POST"],
            summary="选择优化版本",
            description="选择指定的优化版本作为最终结果"
        )
        
        # 策略管理
        self.router.add_api_route(
            "/strategies",
            self.get_optimization_strategies,
            methods=["GET"],
            summary="获取优化策略列表"
        )
        
        # 统计信息
        self.router.add_api_route(
            "/statistics",
            self.get_optimization_statistics,
            methods=["GET"],
            summary="获取优化统计信息"
        )
        
        # 健康检查
        self.router.add_api_route(
            "/health",
            self.health_check,
            methods=["GET"],
            summary="健康检查"
        )
    
    async def optimize_text(self, request: OptimizationRequest) -> ApiResponse:
        """
        单文档文本优化
        
        Args:
            request: 优化请求
            
        Returns:
            优化结果
        """
        try:
            self._logger.info(f"收到文本优化请求 (类型: {request.optimization_type.value}, 模式: {request.optimization_mode.value})")
            
            # 验证请求
            await self._validate_optimization_request(request)
            
            # 执行优化
            start_time = datetime.utcnow()
            result = await self.optimization_engine.optimize_text(request)
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # 构建响应数据
            response_data = {
                "task_id": result.task_id,
                "versions": [
                    {
                        "version_id": version.version_id,
                        "content": version.content,
                        "title": version.title,
                        "quality_score": version.quality_metrics.overall_score,
                        "metrics": {
                            "readability_score": version.quality_metrics.readability_score,
                            "academic_score": version.quality_metrics.academic_score,
                            "historical_accuracy": version.quality_metrics.historical_accuracy,
                            "language_quality": version.quality_metrics.language_quality,
                            "structure_score": version.quality_metrics.structure_score
                        },
                        "improvements": version.improvements,
                        "processing_time_ms": version.processing_time_ms,
                        "token_usage": {
                            "prompt_tokens": version.token_usage.prompt_tokens,
                            "completion_tokens": version.token_usage.completion_tokens,
                            "total_tokens": version.token_usage.total_tokens
                        },
                        "model_used": version.model_used,
                        "is_selected": version.is_selected,
                        "created_at": version.created_at.isoformat()
                    }
                    for version in result.versions
                ],
                "recommended_version": result.recommended_version,
                "statistics": {
                    "total_versions": result.total_versions,
                    "average_quality_score": result.average_quality_score,
                    "best_quality_score": result.best_quality_score,
                    "total_processing_time_ms": int(processing_time)
                }
            }
            
            self._logger.info(f"文本优化完成 (任务ID: {result.task_id}, 最高质量: {result.best_quality_score})")
            
            return ApiResponse.success_response(
                data=response_data,
                message="文本优化完成"
            )
            
        except TextOptimizationError as e:
            self._logger.error(f"文本优化失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文本优化失败: {str(e)}"
            )
        except Exception as e:
            self._logger.error(f"文本优化异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def create_batch_optimization(
        self, 
        request: BatchOptimizationRequest,
        background_tasks: BackgroundTasks
    ) -> ApiResponse:
        """
        创建批量优化任务
        
        Args:
            request: 批量优化请求
            background_tasks: 后台任务
            
        Returns:
            任务创建结果
        """
        try:
            self._logger.info(f"收到批量优化请求 ({request.job_name}, {len(request.document_ids)} 个文档)")
            
            # 验证请求
            await self._validate_batch_request(request)
            
            # 创建批量任务
            job_id = await self.batch_manager.create_batch_job(request)
            
            # 预估完成时间
            estimated_time_per_doc = 5000  # 5秒 (毫秒)
            total_estimated_ms = len(request.document_ids) * estimated_time_per_doc
            estimated_completion = datetime.utcnow().timestamp() + (total_estimated_ms / 1000)
            
            response_data = {
                "job_id": job_id,
                "job_name": request.job_name,
                "total_documents": len(request.document_ids),
                "estimated_completion_time": datetime.fromtimestamp(estimated_completion).isoformat(),
                "status": "pending"
            }
            
            self._logger.info(f"批量优化任务创建成功: {job_id}")
            
            return ApiResponse.success_response(
                data=response_data,
                message="批量优化任务创建成功"
            )
            
        except BatchOptimizationError as e:
            self._logger.error(f"批量优化任务创建失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"批量任务创建失败: {str(e)}"
            )
        except Exception as e:
            self._logger.error(f"批量优化任务创建异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def get_batch_status(self, job_id: str) -> ApiResponse:
        """
        查询批量任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态信息
        """
        try:
            status_info = await self.batch_manager.get_batch_job_status(job_id)
            
            response_data = {
                "job_id": status_info.job_id,
                "job_name": status_info.job_name,
                "status": status_info.status.value,
                "progress": {
                    "total_documents": status_info.total_documents,
                    "completed_documents": status_info.completed_documents,
                    "failed_documents": status_info.failed_documents,
                    "progress_percentage": status_info.progress_percentage
                },
                "timing": {
                    "started_at": status_info.started_at.isoformat() if status_info.started_at else None,
                    "completed_at": status_info.completed_at.isoformat() if status_info.completed_at else None,
                    "estimated_completion_time": status_info.estimated_completion_time.isoformat() if status_info.estimated_completion_time else None,
                    "estimated_remaining_time": status_info.estimated_remaining_time
                },
                "results": status_info.results,
                "error_summary": status_info.error_summary
            }
            
            return ApiResponse.success_response(data=response_data)
            
        except BatchOptimizationError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务不存在: {str(e)}"
            )
        except Exception as e:
            self._logger.error(f"查询批量任务状态异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def pause_batch_job(self, job_id: str) -> ApiResponse:
        """暂停批量任务"""
        try:
            success = await self.batch_manager.pause_batch_job(job_id)
            
            if success:
                return ApiResponse.success_response(message="任务已暂停")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="无法暂停任务"
                )
        except Exception as e:
            self._logger.error(f"暂停批量任务异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def resume_batch_job(self, job_id: str) -> ApiResponse:
        """恢复批量任务"""
        try:
            success = await self.batch_manager.resume_batch_job(job_id)
            
            if success:
                return ApiResponse.success_response(message="任务已恢复")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="无法恢复任务"
                )
        except Exception as e:
            self._logger.error(f"恢复批量任务异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def cancel_batch_job(self, job_id: str) -> ApiResponse:
        """取消批量任务"""
        try:
            success = await self.batch_manager.cancel_batch_job(job_id)
            
            if success:
                return ApiResponse.success_response(message="任务已取消")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="无法取消任务"
                )
        except Exception as e:
            self._logger.error(f"取消批量任务异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def get_optimization_versions(self, task_id: str) -> ApiResponse:
        """获取优化版本列表"""
        try:
            versions_data = await self.storage_client.get_optimization_versions(task_id)
            
            return ApiResponse.success_response(data=versions_data)
            
        except StorageServiceError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务不存在: {str(e)}"
            )
        except Exception as e:
            self._logger.error(f"获取优化版本异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def compare_versions(
        self, 
        version1: str, 
        version2: str
    ) -> ApiResponse:
        """版本对比"""
        try:
            # 从存储服务获取版本内容
            # 这里简化实现，实际应该调用专门的版本对比服务
            
            comparison_data = {
                "version1": version1,
                "version2": version2,
                "differences": [],  # 实际应该包含详细的差异信息
                "similarity_score": 0.0,
                "quality_comparison": {},
                "improvement_areas": [],
                "recommendation": "请人工验证对比结果"
            }
            
            return ApiResponse.success_response(data=comparison_data)
            
        except Exception as e:
            self._logger.error(f"版本对比异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def select_version(self, task_id: str, version_data: Dict[str, str]) -> ApiResponse:
        """选择优化版本"""
        try:
            version_id = version_data.get("version_id")
            if not version_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="缺少version_id参数"
                )
            
            result = await self.storage_client.select_optimization_version(task_id, version_id)
            
            return ApiResponse.success_response(
                data=result,
                message="版本选择成功"
            )
            
        except StorageServiceError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务或版本不存在: {str(e)}"
            )
        except Exception as e:
            self._logger.error(f"选择版本异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def get_optimization_strategies(
        self,
        active_only: bool = True,
        optimization_type: Optional[str] = None,
        optimization_mode: Optional[str] = None
    ) -> ApiResponse:
        """获取优化策略列表"""
        try:
            strategies = await self.strategy_manager.get_all_strategies()
            
            # 过滤策略
            filtered_strategies = []
            for strategy in strategies:
                if active_only and not strategy.is_active:
                    continue
                if optimization_type and strategy.optimization_type.value != optimization_type:
                    continue
                if optimization_mode and strategy.optimization_mode.value != optimization_mode:
                    continue
                
                filtered_strategies.append({
                    "strategy_id": strategy.strategy_id,
                    "name": strategy.name,
                    "description": strategy.description,
                    "optimization_type": strategy.optimization_type.value,
                    "optimization_mode": strategy.optimization_mode.value,
                    "usage_count": strategy.usage_count,
                    "success_rate": strategy.success_rate,
                    "avg_quality_improvement": strategy.avg_quality_improvement,
                    "is_default": strategy.is_default
                })
            
            return ApiResponse.success_response(data={"strategies": filtered_strategies})
            
        except Exception as e:
            self._logger.error(f"获取优化策略异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def get_optimization_statistics(self) -> ApiResponse:
        """获取优化统计信息"""
        try:
            # 获取批量管理器统计
            batch_stats = self.batch_manager.get_manager_statistics()
            
            # 构建响应数据
            statistics = {
                "batch_processing": batch_stats,
                "service_info": {
                    "service_name": self.settings.service_name,
                    "service_version": self.settings.service_version,
                    "uptime": "运行中",
                    "environment": self.settings.service_environment
                },
                "performance": {
                    "average_processing_time_ms": batch_stats.get("average_processing_time_ms", 0),
                    "total_processed_documents": batch_stats.get("total_processed_documents", 0)
                }
            }
            
            return ApiResponse.success_response(data=statistics)
            
        except Exception as e:
            self._logger.error(f"获取统计信息异常: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )
    
    async def health_check(self) -> HealthStatus:
        """健康检查"""
        try:
            # 检查依赖服务
            dependencies = {}
            
            # 检查Storage Service
            try:
                storage_healthy = await self.storage_client.health_check()
                dependencies["storage_service"] = storage_healthy
            except Exception:
                dependencies["storage_service"] = False
            
            # 检查AI Model Service
            try:
                ai_healthy = await self.ai_client.health_check()
                dependencies["ai_model_service"] = ai_healthy
            except Exception:
                dependencies["ai_model_service"] = False
            
            # 计算总体健康状态
            all_healthy = all(dependencies.values())
            service_status = "healthy" if all_healthy else "degraded"
            
            return HealthStatus(
                service_name=self.settings.service_name,
                service_version=self.settings.service_version,
                status=service_status,
                dependencies=dependencies,
                system_info={
                    "active_batch_jobs": len(self.batch_manager.active_jobs),
                    "total_processed_documents": self.batch_manager.total_processed_documents
                }
            )
            
        except Exception as e:
            self._logger.error(f"健康检查异常: {e}")
            return HealthStatus(
                service_name=self.settings.service_name,
                service_version=self.settings.service_version,
                status="unhealthy",
                dependencies={},
                system_info={"error": str(e)}
            )
    
    async def _validate_optimization_request(self, request: OptimizationRequest):
        """验证优化请求"""
        # 检查内容长度
        if len(request.content) > self.settings.max_content_length:
            raise ValueError(f"文本长度超出限制: {len(request.content)} > {self.settings.max_content_length}")
        
        # 检查生成版本数量
        if request.generate_versions > self.settings.max_versions_per_task:
            raise ValueError(f"版本数量超出限制: {request.generate_versions} > {self.settings.max_versions_per_task}")
        
        # 检查质量阈值
        if request.parameters.quality_threshold < self.settings.min_quality_score:
            raise ValueError(f"质量阈值过低: {request.parameters.quality_threshold} < {self.settings.min_quality_score}")
    
    async def _validate_batch_request(self, request: BatchOptimizationRequest):
        """验证批量请求"""
        # 检查文档数量
        if len(request.document_ids) > self.settings.max_batch_size:
            raise ValueError(f"批量大小超出限制: {len(request.document_ids)} > {self.settings.max_batch_size}")
        
        # 检查并发任务数
        if request.max_concurrent_tasks > self.settings.concurrent_optimization_limit:
            raise ValueError(f"并发任务数超出限制: {request.max_concurrent_tasks} > {self.settings.concurrent_optimization_limit}")
    
    def get_router(self) -> APIRouter:
        """获取路由器"""
        return self.router