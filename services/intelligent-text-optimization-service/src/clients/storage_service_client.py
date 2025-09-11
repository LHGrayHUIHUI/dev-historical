"""
Storage Service客户端 - Storage Service Client

负责与Storage Service通信，管理文本优化相关的数据存储操作
包括优化任务、版本管理、策略配置等数据的CRUD操作

核心功能:
1. 优化任务数据管理
2. 优化版本存储和检索
3. 优化策略配置管理
4. 批量任务状态管理
5. 用户偏好设置管理
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import UUID

import httpx
from pydantic import BaseModel

from ..config.settings import get_settings
from ..models.optimization_models import (
    OptimizationType, OptimizationMode, TaskStatus,
    OptimizationResult, BatchOptimizationStatus,
    OptimizationStrategy
)


logger = logging.getLogger(__name__)


class StorageServiceError(Exception):
    """Storage Service客户端错误"""
    pass


class StorageServiceClient:
    """
    Storage Service HTTP客户端
    
    提供与Storage Service通信的接口，用于管理文本优化相关数据
    支持异步操作、错误处理和重试机制
    """
    
    def __init__(self):
        """初始化Storage Service客户端"""
        self.settings = get_settings()
        self.base_url = self.settings.storage_service_url.rstrip('/')
        self.timeout = self.settings.storage_service_timeout
        self.max_retries = self.settings.storage_service_retries
        
        # HTTP客户端配置
        self.client_config = {
            "timeout": httpx.Timeout(self.timeout),
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            params: 查询参数  
            retries: 重试次数
            
        Returns:
            响应数据
            
        Raises:
            StorageServiceError: 请求失败时抛出
        """
        if retries is None:
            retries = self.max_retries
        
        url = f"{self.base_url}{endpoint}"
        self._logger.debug(f"Making {method} request to {url}")
        
        for attempt in range(retries + 1):
            try:
                timeout = httpx.Timeout(self.timeout)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code in [200, 201]:
                        return response.json()
                    elif response.status_code == 404:
                        raise StorageServiceError(f"资源未找到: {endpoint}")
                    elif response.status_code == 422:
                        error_data = response.json()
                        raise StorageServiceError(f"请求验证失败: {error_data}")
                    else:
                        response.raise_for_status()
                        
            except httpx.ConnectError as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"Storage Service连接失败，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise StorageServiceError(f"无法连接到Storage Service: {e}")
            except httpx.TimeoutException as e:
                if attempt < retries:
                    wait_time = 2 ** attempt  
                    self._logger.warning(f"Storage Service请求超时，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise StorageServiceError(f"Storage Service请求超时: {e}")
            except Exception as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"Storage Service请求失败，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise StorageServiceError(f"Storage Service请求失败: {e}")
        
        raise StorageServiceError("达到最大重试次数")
    
    # === 优化任务管理 ===
    
    async def create_optimization_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建优化任务
        
        Args:
            task_data: 任务数据
            
        Returns:
            创建的任务信息
        """
        try:
            response = await self._make_request(
                "POST", "/api/v1/text-optimization/tasks", data=task_data
            )
            self._logger.info(f"创建优化任务成功: {response.get('task_id')}")
            return response
        except Exception as e:
            self._logger.error(f"创建优化任务失败: {e}")
            raise
    
    async def get_optimization_task(self, task_id: str) -> Dict[str, Any]:
        """
        获取优化任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务详情
        """
        try:
            response = await self._make_request("GET", f"/api/v1/text-optimization/tasks/{task_id}")
            return response
        except Exception as e:
            self._logger.error(f"获取优化任务失败 (task_id={task_id}): {e}")
            raise
    
    async def update_optimization_task(self, task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新优化任务
        
        Args:
            task_id: 任务ID
            updates: 更新数据
            
        Returns:
            更新后的任务信息
        """
        try:
            response = await self._make_request(
                "PUT", f"/api/v1/text-optimization/tasks/{task_id}", data=updates
            )
            self._logger.info(f"更新优化任务成功: {task_id}")
            return response
        except Exception as e:
            self._logger.error(f"更新优化任务失败 (task_id={task_id}): {e}")
            raise
    
    async def get_user_optimization_tasks(
        self, 
        user_id: str, 
        status: Optional[TaskStatus] = None,
        skip: int = 0,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        获取用户的优化任务列表
        
        Args:
            user_id: 用户ID
            status: 任务状态过滤
            skip: 跳过条数
            limit: 限制条数
            
        Returns:
            任务列表
        """
        try:
            params = {"user_id": user_id, "skip": skip, "limit": limit}
            if status:
                params["status"] = status.value
                
            response = await self._make_request(
                "GET", "/api/v1/text-optimization/tasks", params=params
            )
            return response
        except Exception as e:
            self._logger.error(f"获取用户优化任务失败 (user_id={user_id}): {e}")
            raise
    
    # === 优化版本管理 ===
    
    async def save_optimization_version(self, version_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存优化版本
        
        Args:
            version_data: 版本数据
            
        Returns:
            保存的版本信息
        """
        try:
            response = await self._make_request(
                "POST", "/api/v1/text-optimization/versions", data=version_data
            )
            self._logger.info(f"保存优化版本成功: {response.get('version_id')}")
            return response
        except Exception as e:
            self._logger.error(f"保存优化版本失败: {e}")
            raise
    
    async def get_optimization_versions(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务的所有版本
        
        Args:
            task_id: 任务ID
            
        Returns:
            版本列表
        """
        try:
            response = await self._make_request(
                "GET", f"/api/v1/text-optimization/tasks/{task_id}/versions"
            )
            return response
        except Exception as e:
            self._logger.error(f"获取优化版本失败 (task_id={task_id}): {e}")
            raise
    
    async def select_optimization_version(self, task_id: str, version_id: str) -> Dict[str, Any]:
        """
        选择优化版本
        
        Args:
            task_id: 任务ID
            version_id: 版本ID
            
        Returns:
            操作结果
        """
        try:
            data = {"version_id": version_id}
            response = await self._make_request(
                "POST", f"/api/v1/text-optimization/tasks/{task_id}/select-version", data=data
            )
            self._logger.info(f"选择优化版本成功: {version_id}")
            return response
        except Exception as e:
            self._logger.error(f"选择优化版本失败 (task_id={task_id}, version_id={version_id}): {e}")
            raise
    
    # === 优化策略管理 ===
    
    async def get_optimization_strategies(
        self, 
        active_only: bool = True,
        optimization_type: Optional[OptimizationType] = None,
        optimization_mode: Optional[OptimizationMode] = None
    ) -> Dict[str, Any]:
        """
        获取优化策略列表
        
        Args:
            active_only: 只获取激活的策略
            optimization_type: 优化类型过滤
            optimization_mode: 优化模式过滤
            
        Returns:
            策略列表
        """
        try:
            params = {"active_only": active_only}
            if optimization_type:
                params["optimization_type"] = optimization_type.value
            if optimization_mode:
                params["optimization_mode"] = optimization_mode.value
                
            response = await self._make_request(
                "GET", "/api/v1/text-optimization/strategies", params=params
            )
            return response
        except Exception as e:
            self._logger.error(f"获取优化策略失败: {e}")
            raise
    
    async def get_optimization_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        获取指定优化策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略详情
        """
        try:
            response = await self._make_request(
                "GET", f"/api/v1/text-optimization/strategies/{strategy_id}"
            )
            return response
        except Exception as e:
            self._logger.error(f"获取优化策略失败 (strategy_id={strategy_id}): {e}")
            raise
    
    async def update_strategy_statistics(self, strategy_id: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新策略统计信息
        
        Args:
            strategy_id: 策略ID
            stats: 统计数据
            
        Returns:
            更新结果
        """
        try:
            response = await self._make_request(
                "PUT", f"/api/v1/text-optimization/strategies/{strategy_id}/stats", data=stats
            )
            self._logger.debug(f"更新策略统计成功: {strategy_id}")
            return response
        except Exception as e:
            self._logger.error(f"更新策略统计失败 (strategy_id={strategy_id}): {e}")
            raise
    
    # === 批量任务管理 ===
    
    async def create_batch_optimization_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建批量优化任务
        
        Args:
            job_data: 任务数据
            
        Returns:
            创建的任务信息
        """
        try:
            response = await self._make_request(
                "POST", "/api/v1/text-optimization/batch-jobs", data=job_data
            )
            self._logger.info(f"创建批量优化任务成功: {response.get('job_id')}")
            return response
        except Exception as e:
            self._logger.error(f"创建批量优化任务失败: {e}")
            raise
    
    async def get_batch_optimization_job(self, job_id: str) -> Dict[str, Any]:
        """
        获取批量优化任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态信息
        """
        try:
            response = await self._make_request(
                "GET", f"/api/v1/text-optimization/batch-jobs/{job_id}"
            )
            return response
        except Exception as e:
            self._logger.error(f"获取批量优化任务失败 (job_id={job_id}): {e}")
            raise
    
    async def update_batch_job_progress(
        self,
        job_id: str,
        completed_increment: int = 0,
        failed_increment: int = 0,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        更新批量任务进度
        
        Args:
            job_id: 任务ID
            completed_increment: 完成任务增量
            failed_increment: 失败任务增量
            error_message: 错误信息
            
        Returns:
            更新结果
        """
        try:
            data = {
                "completed_increment": completed_increment,
                "failed_increment": failed_increment
            }
            if error_message:
                data["error_message"] = error_message
                
            response = await self._make_request(
                "PUT", f"/api/v1/text-optimization/batch-jobs/{job_id}/progress", data=data
            )
            return response
        except Exception as e:
            self._logger.error(f"更新批量任务进度失败 (job_id={job_id}): {e}")
            raise
    
    # === 用户偏好管理 ===
    
    async def get_user_optimization_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户优化偏好设置
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户偏好设置
        """
        try:
            response = await self._make_request(
                "GET", f"/api/v1/text-optimization/users/{user_id}/preferences"
            )
            return response
        except Exception as e:
            self._logger.error(f"获取用户优化偏好失败 (user_id={user_id}): {e}")
            raise
    
    async def update_user_optimization_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        更新用户优化偏好设置
        
        Args:
            user_id: 用户ID
            preferences: 偏好设置
            
        Returns:
            更新结果
        """
        try:
            response = await self._make_request(
                "PUT", f"/api/v1/text-optimization/users/{user_id}/preferences", data=preferences
            )
            self._logger.info(f"更新用户优化偏好成功: {user_id}")
            return response
        except Exception as e:
            self._logger.error(f"更新用户优化偏好失败 (user_id={user_id}): {e}")
            raise
    
    # === 文档管理 ===
    
    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        获取文档内容
        
        Args:
            document_id: 文档ID
            
        Returns:
            文档内容
        """
        try:
            response = await self._make_request("GET", f"/api/v1/documents/{document_id}")
            return response
        except Exception as e:
            self._logger.error(f"获取文档失败 (document_id={document_id}): {e}")
            raise
    
    async def get_documents_batch(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        批量获取文档内容
        
        Args:
            document_ids: 文档ID列表
            
        Returns:
            文档内容列表
        """
        try:
            data = {"document_ids": document_ids}
            response = await self._make_request("POST", "/api/v1/documents/batch", data=data)
            return response
        except Exception as e:
            self._logger.error(f"批量获取文档失败: {e}")
            raise
    
    # === 健康检查 ===
    
    async def health_check(self) -> bool:
        """
        检查Storage Service健康状态
        
        Returns:
            健康状态
        """
        try:
            await self._make_request("GET", "/health")
            return True
        except Exception as e:
            self._logger.warning(f"Storage Service健康检查失败: {e}")
            return False