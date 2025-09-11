"""
批量优化管理器 - Batch Optimization Manager

负责管理大规模文档的批量文本优化任务，支持异步处理、
进度监控、错误处理和结果统计

核心功能:
1. 批量任务创建和调度
2. 异步任务执行和监控
3. 进度跟踪和状态更新
4. 错误处理和重试机制
5. 结果统计和报告生成
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum

from ..config.settings import get_settings
from ..models.optimization_models import (
    OptimizationType, OptimizationMode, TaskStatus,
    BatchOptimizationRequest, BatchOptimizationStatus,
    OptimizationRequest, OptimizationParameters
)
from ..clients.storage_service_client import StorageServiceClient
from .text_optimization_engine import TextOptimizationEngine


logger = logging.getLogger(__name__)


class BatchJobStatus(str, Enum):
    """批量任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """批量任务内部表示"""
    job_id: str
    job_name: str
    user_id: str
    document_ids: List[str]
    optimization_config: Dict[str, Any]
    
    # 状态信息
    status: BatchJobStatus = BatchJobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 进度信息
    total_documents: int = 0
    completed_documents: int = 0
    failed_documents: int = 0
    
    # 配置信息
    parallel_processing: bool = True
    max_concurrent_tasks: int = 5
    priority: int = 1
    
    # 结果统计
    total_processing_time_ms: int = 0
    average_quality_score: float = 0.0
    successful_optimizations: int = 0
    error_summary: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.total_documents == 0:
            self.total_documents = len(self.document_ids)
    
    @property
    def progress_percentage(self) -> float:
        """计算进度百分比"""
        if self.total_documents == 0:
            return 100.0
        return (self.completed_documents + self.failed_documents) / self.total_documents * 100
    
    @property
    def estimated_remaining_time(self) -> Optional[str]:
        """估算剩余时间"""
        if self.completed_documents == 0 or self.status != BatchJobStatus.RUNNING:
            return None
        
        avg_time_per_doc = self.total_processing_time_ms / max(self.completed_documents, 1)
        remaining_docs = self.total_documents - self.completed_documents - self.failed_documents
        
        if remaining_docs <= 0:
            return "00:00:00"
        
        remaining_ms = avg_time_per_doc * remaining_docs
        remaining_seconds = int(remaining_ms / 1000)
        
        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        seconds = remaining_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass
class DocumentTask:
    """单个文档任务"""
    document_id: str
    job_id: str
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    processing_time_ms: int = 0
    quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


class BatchOptimizationError(Exception):
    """批量优化错误"""
    pass


class TaskQueue:
    """
    任务队列管理器
    负责任务的调度和执行
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """初始化任务队列"""
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.pending_queue: List[DocumentTask] = []
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def add_task(self, task: DocumentTask):
        """添加任务到队列"""
        self.pending_queue.append(task)
        self._logger.debug(f"任务添加到队列: {task.document_id}")
    
    async def process_queue(self, processor_func):
        """处理队列中的任务"""
        while self.pending_queue or self.running_tasks:
            # 启动新任务
            while (len(self.running_tasks) < self.max_concurrent_tasks and 
                   self.pending_queue):
                task = self.pending_queue.pop(0)
                
                # 创建异步任务
                async_task = asyncio.create_task(processor_func(task))
                self.running_tasks[task.document_id] = async_task
                
                self._logger.debug(f"启动任务处理: {task.document_id}")
            
            if not self.running_tasks:
                break
            
            # 等待任何一个任务完成
            done, pending = await asyncio.wait(
                self.running_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 清理已完成的任务
            completed_tasks = []
            for task_id, async_task in list(self.running_tasks.items()):
                if async_task in done:
                    completed_tasks.append(task_id)
                    del self.running_tasks[task_id]
            
            self._logger.debug(f"完成任务: {completed_tasks}")
            
            # 短暂等待避免过度占用CPU
            await asyncio.sleep(0.1)
    
    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态"""
        return {
            'pending': len(self.pending_queue),
            'running': len(self.running_tasks),
            'total': len(self.pending_queue) + len(self.running_tasks)
        }


class BatchOptimizationManager:
    """
    批量优化管理器主类
    统一管理批量优化任务的生命周期
    """
    
    def __init__(
        self, 
        optimization_engine: TextOptimizationEngine,
        storage_client: StorageServiceClient
    ):
        """
        初始化批量优化管理器
        
        Args:
            optimization_engine: 文本优化引擎
            storage_client: 存储服务客户端
        """
        self.settings = get_settings()
        self.optimization_engine = optimization_engine
        self.storage_client = storage_client
        
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 活动任务管理
        self.active_jobs: Dict[str, BatchJob] = {}
        self.task_queues: Dict[str, TaskQueue] = {}
        
        # 性能监控
        self.total_processed_documents = 0
        self.total_processing_time_ms = 0
        self.average_processing_time_ms = 0
    
    async def create_batch_job(self, request: BatchOptimizationRequest) -> str:
        """
        创建批量优化任务
        
        Args:
            request: 批量优化请求
            
        Returns:
            任务ID
        """
        try:
            job_id = str(uuid4())
            
            self._logger.info(f"创建批量优化任务: {job_id} ({request.job_name})")
            
            # 验证请求
            await self._validate_batch_request(request)
            
            # 创建批量任务
            batch_job = BatchJob(
                job_id=job_id,
                job_name=request.job_name,
                user_id=request.user_id,
                document_ids=request.document_ids.copy(),
                optimization_config=request.optimization_config,
                parallel_processing=request.parallel_processing,
                max_concurrent_tasks=min(request.max_concurrent_tasks, self.settings.max_batch_size),
                priority=request.priority
            )
            
            # 保存到存储服务
            job_data = await self._prepare_job_data_for_storage(batch_job, request)
            await self.storage_client.create_batch_optimization_job(job_data)
            
            # 添加到活动任务
            self.active_jobs[job_id] = batch_job
            
            # 创建任务队列
            task_queue = TaskQueue(max_concurrent_tasks=batch_job.max_concurrent_tasks)
            self.task_queues[job_id] = task_queue
            
            # 异步开始处理
            asyncio.create_task(self._process_batch_job(job_id))
            
            self._logger.info(f"批量优化任务创建成功: {job_id} ({len(request.document_ids)} 个文档)")
            
            return job_id
            
        except Exception as e:
            self._logger.error(f"创建批量优化任务失败: {e}")
            raise BatchOptimizationError(f"创建批量任务失败: {str(e)}")
    
    async def get_batch_job_status(self, job_id: str) -> BatchOptimizationStatus:
        """
        获取批量任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态信息
        """
        try:
            # 先检查活动任务
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                return await self._build_status_response(job)
            
            # 从存储服务获取
            job_data = await self.storage_client.get_batch_optimization_job(job_id)
            return BatchOptimizationStatus(**job_data['data'])
            
        except Exception as e:
            self._logger.error(f"获取批量任务状态失败 (job_id={job_id}): {e}")
            raise BatchOptimizationError(f"获取任务状态失败: {str(e)}")
    
    async def pause_batch_job(self, job_id: str) -> bool:
        """
        暂停批量任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功暂停
        """
        try:
            if job_id not in self.active_jobs:
                return False
            
            job = self.active_jobs[job_id]
            if job.status == BatchJobStatus.RUNNING:
                job.status = BatchJobStatus.PAUSED
                
                # 更新存储服务
                await self.storage_client.update_batch_job_progress(job_id, status='paused')
                
                self._logger.info(f"批量任务已暂停: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"暂停批量任务失败: {e}")
            return False
    
    async def resume_batch_job(self, job_id: str) -> bool:
        """
        恢复批量任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功恢复
        """
        try:
            if job_id not in self.active_jobs:
                return False
            
            job = self.active_jobs[job_id]
            if job.status == BatchJobStatus.PAUSED:
                job.status = BatchJobStatus.RUNNING
                
                # 更新存储服务
                await self.storage_client.update_batch_job_progress(job_id, status='running')
                
                self._logger.info(f"批量任务已恢复: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"恢复批量任务失败: {e}")
            return False
    
    async def cancel_batch_job(self, job_id: str) -> bool:
        """
        取消批量任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功取消
        """
        try:
            if job_id not in self.active_jobs:
                return False
            
            job = self.active_jobs[job_id]
            job.status = BatchJobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            
            # 清理资源
            if job_id in self.task_queues:
                del self.task_queues[job_id]
            
            # 更新存储服务
            await self.storage_client.update_batch_job_progress(job_id, status='cancelled')
            
            self._logger.info(f"批量任务已取消: {job_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"取消批量任务失败: {e}")
            return False
    
    async def _validate_batch_request(self, request: BatchOptimizationRequest):
        """验证批量请求"""
        # 检查文档数量限制
        if len(request.document_ids) > self.settings.max_batch_size:
            raise BatchOptimizationError(
                f"文档数量超出限制: {len(request.document_ids)} > {self.settings.max_batch_size}"
            )
        
        # 检查优化配置
        config = request.optimization_config
        required_fields = ['optimization_type', 'optimization_mode']
        
        for field in required_fields:
            if field not in config:
                raise BatchOptimizationError(f"缺少必需的配置字段: {field}")
        
        # 验证优化类型和模式
        try:
            OptimizationType(config['optimization_type'])
            OptimizationMode(config['optimization_mode'])
        except ValueError as e:
            raise BatchOptimizationError(f"无效的优化配置: {e}")
    
    async def _prepare_job_data_for_storage(self, job: BatchJob, request: BatchOptimizationRequest) -> Dict[str, Any]:
        """准备存储的任务数据"""
        return {
            'job_id': job.job_id,
            'user_id': job.user_id,
            'job_name': job.job_name,
            'source_document_ids': job.document_ids,
            'optimization_config': job.optimization_config,
            'total_documents': job.total_documents,
            'status': job.status.value,
            'parallel_processing': job.parallel_processing,
            'max_concurrent_tasks': job.max_concurrent_tasks,
            'priority': job.priority,
            'notification_email': request.notification_email,
            'webhook_url': request.webhook_url,
            'tags': request.tags
        }
    
    async def _process_batch_job(self, job_id: str):
        """处理批量任务"""
        try:
            job = self.active_jobs[job_id]
            task_queue = self.task_queues[job_id]
            
            self._logger.info(f"开始处理批量任务: {job_id}")
            
            # 更新任务状态
            job.status = BatchJobStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            # 创建文档任务
            document_tasks = []
            for doc_id in job.document_ids:
                task = DocumentTask(
                    document_id=doc_id,
                    job_id=job_id
                )
                document_tasks.append(task)
                await task_queue.add_task(task)
            
            # 处理任务队列
            await task_queue.process_queue(
                lambda task: self._process_document_task(task, job)
            )
            
            # 完成任务
            await self._complete_batch_job(job)
            
        except Exception as e:
            self._logger.error(f"批量任务处理失败 (job_id={job_id}): {e}")
            await self._fail_batch_job(job_id, str(e))
    
    async def _process_document_task(self, task: DocumentTask, job: BatchJob):
        """处理单个文档任务"""
        start_time = time.time()
        
        try:
            # 检查任务状态
            if job.status != BatchJobStatus.RUNNING:
                return
            
            task.status = TaskStatus.PROCESSING
            task.attempts += 1
            
            self._logger.debug(f"处理文档任务: {task.document_id} (尝试 {task.attempts}/{task.max_attempts})")
            
            # 获取文档内容
            document = await self.storage_client.get_document(task.document_id)
            content = document.get('content', '')
            
            if not content.strip():
                raise Exception("文档内容为空")
            
            # 构建优化请求
            optimization_request = await self._build_optimization_request(
                content, job.optimization_config, job.user_id
            )
            
            # 执行优化
            optimization_result = await self.optimization_engine.optimize_text(
                optimization_request
            )
            
            # 处理结果
            task.processing_time_ms = int((time.time() - start_time) * 1000)
            task.quality_score = optimization_result.best_quality_score
            task.status = TaskStatus.COMPLETED
            
            # 更新任务统计
            await self._update_job_progress(job, task, success=True)
            
            self._logger.debug(f"文档任务完成: {task.document_id} (质量分数: {task.quality_score})")
            
        except Exception as e:
            task.last_error = str(e)
            task.processing_time_ms = int((time.time() - start_time) * 1000)
            
            # 判断是否重试
            if task.attempts < task.max_attempts:
                # 重试
                task.status = TaskStatus.PENDING
                await asyncio.sleep(2 ** task.attempts)  # 指数退避
                await self._process_document_task(task, job)
            else:
                # 标记失败
                task.status = TaskStatus.FAILED
                await self._update_job_progress(job, task, success=False, error=str(e))
                
                self._logger.warning(f"文档任务失败: {task.document_id} - {e}")
    
    async def _build_optimization_request(
        self, 
        content: str, 
        config: Dict[str, Any], 
        user_id: str
    ) -> OptimizationRequest:
        """构建优化请求"""
        parameters = OptimizationParameters()
        
        # 从配置中设置参数
        if 'parameters' in config:
            param_config = config['parameters']
            parameters = OptimizationParameters(
                target_length=param_config.get('target_length'),
                quality_threshold=param_config.get('quality_threshold', 80.0),
                custom_instructions=param_config.get('custom_instructions'),
                temperature=param_config.get('temperature', 0.7),
                max_tokens=param_config.get('max_tokens')
            )
        
        return OptimizationRequest(
            content=content,
            optimization_type=OptimizationType(config['optimization_type']),
            optimization_mode=OptimizationMode(config['optimization_mode']),
            parameters=parameters,
            generate_versions=config.get('generate_versions', 1),
            user_id=user_id
        )
    
    async def _update_job_progress(
        self, 
        job: BatchJob, 
        task: DocumentTask, 
        success: bool, 
        error: Optional[str] = None
    ):
        """更新任务进度"""
        if success:
            job.completed_documents += 1
            job.successful_optimizations += 1
            job.total_processing_time_ms += task.processing_time_ms
            
            # 更新平均质量分数
            total_quality = (job.average_quality_score * (job.successful_optimizations - 1) + 
                           task.quality_score)
            job.average_quality_score = total_quality / job.successful_optimizations
            
        else:
            job.failed_documents += 1
            if error and error not in job.error_summary:
                job.error_summary.append(error)
        
        # 更新存储服务
        await self.storage_client.update_batch_job_progress(
            job_id=job.job_id,
            completed_increment=1 if success else 0,
            failed_increment=0 if success else 1,
            error_message=error
        )
        
        # 更新全局统计
        self.total_processed_documents += 1
        self.total_processing_time_ms += task.processing_time_ms
        if self.total_processed_documents > 0:
            self.average_processing_time_ms = (
                self.total_processing_time_ms // self.total_processed_documents
            )
    
    async def _complete_batch_job(self, job: BatchJob):
        """完成批量任务"""
        job.status = BatchJobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        
        # 更新存储服务
        await self.storage_client.update_batch_job_progress(
            job_id=job.job_id,
            status='completed'
        )
        
        # 发送通知 (如果配置了)
        await self._send_completion_notification(job)
        
        # 清理资源
        if job.job_id in self.task_queues:
            del self.task_queues[job.job_id]
        
        self._logger.info(
            f"批量任务完成: {job.job_id} "
            f"(成功: {job.successful_optimizations}, 失败: {job.failed_documents}, "
            f"平均质量: {job.average_quality_score:.1f})"
        )
    
    async def _fail_batch_job(self, job_id: str, error: str):
        """标记任务失败"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_summary.append(error)
            
            # 更新存储服务
            await self.storage_client.update_batch_job_progress(
                job_id=job_id,
                status='failed',
                error_message=error
            )
            
            # 清理资源
            if job_id in self.task_queues:
                del self.task_queues[job_id]
        
        self._logger.error(f"批量任务失败: {job_id} - {error}")
    
    async def _build_status_response(self, job: BatchJob) -> BatchOptimizationStatus:
        """构建状态响应"""
        # 获取队列状态
        queue_status = {}
        if job.job_id in self.task_queues:
            queue_status = self.task_queues[job.job_id].get_queue_status()
        
        # 计算预计完成时间
        estimated_completion = None
        if (job.status == BatchJobStatus.RUNNING and 
            job.started_at and job.completed_documents > 0):
            
            elapsed = datetime.utcnow() - job.started_at
            avg_time_per_doc = elapsed.total_seconds() / max(job.completed_documents, 1)
            remaining_docs = job.total_documents - job.completed_documents - job.failed_documents
            
            if remaining_docs > 0:
                remaining_seconds = avg_time_per_doc * remaining_docs
                estimated_completion = datetime.utcnow() + timedelta(seconds=remaining_seconds)
        
        # 构建结果统计
        results = {
            'successful_optimizations': job.successful_optimizations,
            'failed_documents': job.failed_documents,
            'average_quality_score': round(job.average_quality_score, 1),
            'total_processing_time_ms': job.total_processing_time_ms,
            'queue_status': queue_status
        }
        
        return BatchOptimizationStatus(
            job_id=job.job_id,
            job_name=job.job_name,
            status=TaskStatus(job.status.value),
            total_documents=job.total_documents,
            completed_documents=job.completed_documents,
            failed_documents=job.failed_documents,
            progress_percentage=job.progress_percentage,
            started_at=job.started_at,
            completed_at=job.completed_at,
            estimated_completion_time=estimated_completion,
            estimated_remaining_time=job.estimated_remaining_time,
            results=results,
            error_summary=job.error_summary,
            user_id=job.user_id,
            created_at=job.created_at
        )
    
    async def _send_completion_notification(self, job: BatchJob):
        """发送完成通知"""
        try:
            # 这里可以实现邮件通知、Webhook回调等
            # 目前只记录日志
            self._logger.info(f"批量任务完成通知: {job.job_id}")
            
            # TODO: 实现实际的通知机制
            # - 邮件通知
            # - Webhook回调
            # - WebSocket推送
            
        except Exception as e:
            self._logger.warning(f"发送完成通知失败: {e}")
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        return {
            'active_jobs': len(self.active_jobs),
            'total_processed_documents': self.total_processed_documents,
            'total_processing_time_ms': self.total_processing_time_ms,
            'average_processing_time_ms': self.average_processing_time_ms,
            'job_statuses': {
                status.value: sum(1 for job in self.active_jobs.values() if job.status == status)
                for status in BatchJobStatus
            }
        }