"""
Storage Service客户端
图像处理服务与storage-service的HTTP通信客户端
处理所有数据存储操作
"""

import httpx
import asyncio
from typing import Dict, Any, List, Optional, Union
from uuid import UUID
import json
import time
from loguru import logger

from ..config.settings import settings


class StorageServiceClient:
    """Storage Service HTTP客户端"""
    
    def __init__(self):
        self.base_url = settings.storage_service_url.rstrip('/')
        self.timeout = settings.storage_service_timeout
        self.retries = settings.storage_service_retries
        
        # HTTP客户端配置
        self.client_config = {
            "timeout": httpx.Timeout(self.timeout),
            "limits": httpx.Limits(max_connections=20, max_keepalive_connections=5)
        }
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """发送HTTP请求到storage-service"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.retries):
            try:
                async with httpx.AsyncClient(**self.client_config) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=data if data else None,
                        params=params,
                        files=files
                    )
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP错误 {e.response.status_code}: {e.response.text}")
                if e.response.status_code < 500 or attempt == self.retries - 1:
                    raise
            except httpx.RequestError as e:
                logger.error(f"请求错误: {str(e)}")
                if attempt == self.retries - 1:
                    raise
            
            # 重试延迟
            if attempt < self.retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"请求失败，已重试 {self.retries} 次")
    
    # ============ 图像处理任务管理 ============
    
    async def create_image_processing_task(
        self,
        dataset_id: Optional[Union[str, UUID]],
        original_image_path: str,
        processing_type: str,
        processing_engine: str,
        config: Optional[Dict] = None,
        created_by: Optional[Union[str, UUID]] = None,
        priority: int = 5
    ) -> Dict[str, Any]:
        """创建图像处理任务"""
        data = {
            "dataset_id": str(dataset_id) if dataset_id else None,
            "original_image_path": original_image_path,
            "processing_type": processing_type,
            "processing_status": "pending",
            "processing_engine": processing_engine,
            "config": config or {},
            "priority": priority,
            "created_by": str(created_by) if created_by else None
        }
        
        return await self._make_request("POST", "/api/v1/image-processing/tasks", data=data)
    
    async def update_task_status(
        self,
        task_id: Union[str, UUID],
        status: str,
        progress: Optional[int] = None,
        processed_image_path: Optional[str] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新图像处理任务状态"""
        data = {
            "processing_status": status,
            "progress": progress,
            "processed_image_path": processed_image_path,
            "processing_time": processing_time,
            "error_message": error_message
        }
        
        if status == "processing":
            data["started_at"] = time.time()
        elif status in ["completed", "failed", "cancelled"]:
            data["completed_at"] = time.time()
        
        # 移除None值
        data = {k: v for k, v in data.items() if v is not None}
        
        return await self._make_request("PUT", f"/api/v1/image-processing/tasks/{task_id}", data=data)
    
    async def get_task(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取图像处理任务详情"""
        return await self._make_request("GET", f"/api/v1/image-processing/tasks/{task_id}")
    
    async def get_tasks(
        self,
        dataset_id: Optional[Union[str, UUID]] = None,
        processing_type: Optional[str] = None,
        status: Optional[str] = None,
        user_id: Optional[Union[str, UUID]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取图像处理任务列表"""
        params = {"limit": limit, "offset": offset}
        if dataset_id:
            params["dataset_id"] = str(dataset_id)
        if processing_type:
            params["processing_type"] = processing_type
        if status:
            params["status"] = status
        if user_id:
            params["user_id"] = str(user_id)
        
        return await self._make_request("GET", "/api/v1/image-processing/tasks", params=params)
    
    # ============ 图像处理结果存储 ============
    
    async def save_processing_result(
        self,
        task_id: Union[str, UUID],
        original_image_info: Dict[str, Any],
        processed_image_info: Dict[str, Any],
        processed_image_path: str,
        quality_before: Optional[Dict[str, Any]] = None,
        quality_after: Optional[Dict[str, Any]] = None,
        processing_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """保存图像处理结果"""
        data = {
            "task_id": str(task_id),
            "original_image_info": original_image_info,
            "processed_image_info": processed_image_info,
            "processed_image_path": processed_image_path,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "processing_metrics": processing_metrics or {}
        }
        
        return await self._make_request("POST", "/api/v1/image-processing/results", data=data)
    
    async def get_processing_result(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取图像处理结果"""
        return await self._make_request("GET", f"/api/v1/image-processing/results/{task_id}")
    
    # ============ 图像质量评估存储 ============
    
    async def save_quality_assessment(
        self,
        task_id: Union[str, UUID],
        image_path: str,
        assessment_type: str,  # "before" or "after"
        quality_metrics: Dict[str, float],
        assessment_method: str = "opencv"
    ) -> Dict[str, Any]:
        """保存图像质量评估结果"""
        data = {
            "task_id": str(task_id),
            "image_path": image_path,
            "assessment_type": assessment_type,
            "quality_metrics": quality_metrics,
            "assessment_method": assessment_method
        }
        
        return await self._make_request("POST", "/api/v1/image-processing/quality-assessments", data=data)
    
    async def get_quality_assessment(
        self,
        task_id: Union[str, UUID],
        assessment_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取图像质量评估结果"""
        params = {}
        if assessment_type:
            params["assessment_type"] = assessment_type
        
        return await self._make_request(
            "GET", 
            f"/api/v1/image-processing/quality-assessments/{task_id}",
            params=params
        )
    
    # ============ 批量处理管理 ============
    
    async def create_batch_processing_task(
        self,
        batch_id: str,
        image_paths: List[str],
        processing_type: str,
        processing_engine: str,
        config: Optional[Dict] = None,
        dataset_id: Optional[Union[str, UUID]] = None,
        created_by: Optional[Union[str, UUID]] = None,
        priority: int = 5
    ) -> Dict[str, Any]:
        """创建批量图像处理任务"""
        data = {
            "batch_id": batch_id,
            "image_paths": image_paths,
            "processing_type": processing_type,
            "processing_engine": processing_engine,
            "config": config or {},
            "dataset_id": str(dataset_id) if dataset_id else None,
            "created_by": str(created_by) if created_by else None,
            "priority": priority,
            "total_images": len(image_paths),
            "status": "pending"
        }
        
        return await self._make_request("POST", "/api/v1/image-processing/batch-tasks", data=data)
    
    async def update_batch_task_progress(
        self,
        batch_id: str,
        processed_count: int,
        failed_count: int,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新批量处理任务进度"""
        data = {
            "processed_count": processed_count,
            "failed_count": failed_count,
            "progress": int((processed_count + failed_count) * 100 / (processed_count + failed_count)) if (processed_count + failed_count) > 0 else 0
        }
        
        if status:
            data["status"] = status
            if status in ["completed", "failed", "cancelled"]:
                data["completed_at"] = time.time()
        
        return await self._make_request("PUT", f"/api/v1/image-processing/batch-tasks/{batch_id}", data=data)
    
    async def get_batch_task(self, batch_id: str) -> Dict[str, Any]:
        """获取批量处理任务详情"""
        return await self._make_request("GET", f"/api/v1/image-processing/batch-tasks/{batch_id}")
    
    # ============ 文件管理 ============
    
    async def upload_processed_image(
        self,
        image_data: bytes,
        filename: str,
        content_type: str = "image/jpeg",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """上传处理后的图像到存储服务"""
        files = {
            "file": (filename, image_data, content_type)
        }
        
        data = {}
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        return await self._make_request("POST", "/api/v1/files/upload", data=data, files=files)
    
    async def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """获取图像文件信息"""
        params = {"file_path": image_path}
        return await self._make_request("GET", "/api/v1/files/info", params=params)
    
    async def download_image(self, image_path: str) -> bytes:
        """从存储服务下载图像"""
        params = {"file_path": image_path}
        
        async with httpx.AsyncClient(**self.client_config) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/files/download",
                params=params
            )
            response.raise_for_status()
            return response.content
    
    async def delete_image(self, image_path: str) -> Dict[str, Any]:
        """删除存储的图像"""
        data = {"file_path": image_path}
        return await self._make_request("DELETE", "/api/v1/files/delete", data=data)
    
    # ============ 数据集操作 ============
    
    async def get_dataset(self, dataset_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取数据集信息"""
        return await self._make_request("GET", f"/api/v1/datasets/{dataset_id}")
    
    async def get_dataset_images(
        self,
        dataset_id: Union[str, UUID],
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取数据集中的图像列表"""
        params = {"limit": limit, "offset": offset}
        return await self._make_request("GET", f"/api/v1/datasets/{dataset_id}/images", params=params)
    
    async def update_dataset_processing_status(
        self,
        dataset_id: Union[str, UUID],
        image_processing_status: str,
        image_processing_progress: Optional[int] = None
    ) -> Dict[str, Any]:
        """更新数据集的图像处理状态"""
        data = {
            "image_processing_status": image_processing_status,
            "image_processing_progress": image_processing_progress
        }
        
        return await self._make_request("PUT", f"/api/v1/datasets/{dataset_id}/image-status", data=data)
    
    # ============ 处理配置管理 ============
    
    async def save_processing_config(
        self,
        name: str,
        processing_type: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        is_default: bool = False,
        created_by: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """保存处理配置模板"""
        data = {
            "name": name,
            "processing_type": processing_type,
            "config": config,
            "description": description,
            "is_default": is_default,
            "is_active": True,
            "created_by": str(created_by) if created_by else None
        }
        
        return await self._make_request("POST", "/api/v1/image-processing/configs", data=data)
    
    async def get_processing_configs(
        self,
        processing_type: Optional[str] = None,
        is_default: Optional[bool] = None
    ) -> Dict[str, Any]:
        """获取处理配置模板列表"""
        params = {}
        if processing_type:
            params["processing_type"] = processing_type
        if is_default is not None:
            params["is_default"] = is_default
        
        return await self._make_request("GET", "/api/v1/image-processing/configs", params=params)
    
    # ============ 统计和分析 ============
    
    async def get_processing_statistics(
        self,
        dataset_id: Optional[Union[str, UUID]] = None,
        processing_type: Optional[str] = None,
        engine: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取图像处理统计信息"""
        params = {}
        if dataset_id:
            params["dataset_id"] = str(dataset_id)
        if processing_type:
            params["processing_type"] = processing_type
        if engine:
            params["engine"] = engine
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        
        return await self._make_request("GET", "/api/v1/image-processing/statistics", params=params)
    
    async def health_check(self) -> Dict[str, Any]:
        """检查storage-service健康状态"""
        return await self._make_request("GET", "/health")


# 全局storage client实例
storage_client = StorageServiceClient()