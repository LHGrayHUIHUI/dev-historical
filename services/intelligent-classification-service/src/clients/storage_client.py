"""
Storage Service HTTP客户端
处理智能分类服务与storage-service之间的所有数据通信
遵循无状态架构原则，所有数据持久化通过storage-service完成
"""

import httpx
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
from ..config.settings import settings


class StorageServiceClient:
    """Storage Service HTTP客户端"""
    
    def __init__(self):
        self.base_url = settings.storage_service_url
        self.timeout = settings.storage_service_timeout
        self.max_retries = settings.storage_service_retries
        self.logger = logging.getLogger(__name__)
        
        # HTTP客户端配置
        self.client_config = {
            "timeout": httpx.Timeout(self.timeout),
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=100),
            "follow_redirects": True
        }
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """发送HTTP请求到storage-service"""
        url = f"{self.base_url}/api/v1/{endpoint.lstrip('/')}"
        
        # 准备请求头
        request_headers = {
            "Content-Type": "application/json",
            "User-Agent": "intelligent-classification-service/1.0.0"
        }
        if headers:
            request_headers.update(headers)
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(**self.client_config) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers=request_headers
                    )
                    
                    # 检查响应状态
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.TimeoutException:
                self.logger.warning(f"Storage service request timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))
                
            except httpx.HTTPStatusError as e:
                self.logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                if attempt == self.max_retries - 1 or e.response.status_code < 500:
                    raise
                await asyncio.sleep(1 * (attempt + 1))
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))
    
    # ============ 分类项目管理 ============
    
    async def create_classification_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建分类项目"""
        return await self._make_request("POST", "classification/projects", data=project_data)
    
    async def get_classification_project(self, project_id: str) -> Dict[str, Any]:
        """获取分类项目详情"""
        return await self._make_request("GET", f"classification/projects/{project_id}")
    
    async def update_classification_project(
        self, 
        project_id: str, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新分类项目"""
        return await self._make_request("PUT", f"classification/projects/{project_id}", data=update_data)
    
    async def delete_classification_project(self, project_id: str) -> Dict[str, Any]:
        """删除分类项目"""
        return await self._make_request("DELETE", f"classification/projects/{project_id}")
    
    async def list_classification_projects(
        self, 
        limit: int = 100, 
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """列出分类项目"""
        params = {"limit": limit, "offset": offset}
        if filters:
            params.update(filters)
        return await self._make_request("GET", "classification/projects", params=params)
    
    # ============ 训练数据管理 ============
    
    async def create_training_data(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加训练数据"""
        return await self._make_request("POST", "classification/training-data", data=training_data)
    
    async def create_training_data_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量添加训练数据"""
        return await self._make_request("POST", "classification/training-data/batch", data={"data": batch_data})
    
    async def get_training_data(
        self, 
        project_id: str,
        limit: int = 1000,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取项目的训练数据"""
        params = {"project_id": project_id, "limit": limit, "offset": offset}
        return await self._make_request("GET", "classification/training-data", params=params)
    
    async def update_training_data(
        self, 
        training_data_id: str, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新训练数据"""
        return await self._make_request("PUT", f"classification/training-data/{training_data_id}", data=update_data)
    
    async def delete_training_data(self, training_data_id: str) -> Dict[str, Any]:
        """删除训练数据"""
        return await self._make_request("DELETE", f"classification/training-data/{training_data_id}")
    
    async def get_training_data_statistics(self, project_id: str) -> Dict[str, Any]:
        """获取训练数据统计"""
        return await self._make_request("GET", f"classification/projects/{project_id}/training-stats")
    
    # ============ 分类模型管理 ============
    
    async def create_classification_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建分类模型记录"""
        return await self._make_request("POST", "classification/models", data=model_data)
    
    async def get_classification_model(self, model_id: str) -> Dict[str, Any]:
        """获取分类模型详情"""
        return await self._make_request("GET", f"classification/models/{model_id}")
    
    async def update_classification_model(
        self, 
        model_id: str, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新分类模型"""
        return await self._make_request("PUT", f"classification/models/{model_id}", data=update_data)
    
    async def delete_classification_model(self, model_id: str) -> Dict[str, Any]:
        """删除分类模型"""
        return await self._make_request("DELETE", f"classification/models/{model_id}")
    
    async def list_classification_models(
        self, 
        project_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """列出项目的分类模型"""
        params = {"project_id": project_id, "limit": limit, "offset": offset}
        return await self._make_request("GET", "classification/models", params=params)
    
    async def set_active_model(self, project_id: str, model_id: str) -> Dict[str, Any]:
        """设置活跃模型"""
        data = {"active_model_id": model_id}
        return await self._make_request("PUT", f"classification/projects/{project_id}/active-model", data=data)
    
    async def get_active_model(self, project_id: str) -> Dict[str, Any]:
        """获取项目的活跃模型"""
        return await self._make_request("GET", f"classification/projects/{project_id}/active-model")
    
    # ============ 分类任务管理 ============
    
    async def create_classification_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建分类任务"""
        return await self._make_request("POST", "classification/tasks", data=task_data)
    
    async def get_classification_task(self, task_id: str) -> Dict[str, Any]:
        """获取分类任务详情"""
        return await self._make_request("GET", f"classification/tasks/{task_id}")
    
    async def update_classification_task(
        self, 
        task_id: str, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新分类任务状态"""
        return await self._make_request("PUT", f"classification/tasks/{task_id}", data=update_data)
    
    async def list_classification_tasks(
        self, 
        project_id: str,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """列出分类任务"""
        params = {"project_id": project_id, "limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._make_request("GET", "classification/tasks", params=params)
    
    # ============ 分类结果管理 ============
    
    async def create_classification_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """保存分类结果"""
        return await self._make_request("POST", "classification/results", data=result_data)
    
    async def get_classification_result(self, result_id: str) -> Dict[str, Any]:
        """获取分类结果"""
        return await self._make_request("GET", f"classification/results/{result_id}")
    
    async def get_classification_results_by_task(self, task_id: str) -> Dict[str, Any]:
        """获取任务的分类结果"""
        params = {"task_id": task_id}
        return await self._make_request("GET", "classification/results", params=params)
    
    async def update_classification_result(
        self, 
        result_id: str, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新分类结果"""
        return await self._make_request("PUT", f"classification/results/{result_id}", data=update_data)
    
    async def list_classification_results(
        self, 
        project_id: str,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """列出分类结果"""
        params = {"project_id": project_id, "limit": limit, "offset": offset}
        if filters:
            params.update(filters)
        return await self._make_request("GET", "classification/results", params=params)
    
    # ============ 模型性能统计 ============
    
    async def get_model_performance_stats(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能统计"""
        return await self._make_request("GET", f"classification/models/{model_id}/performance")
    
    async def get_project_statistics(self, project_id: str) -> Dict[str, Any]:
        """获取项目统计信息"""
        return await self._make_request("GET", f"classification/projects/{project_id}/statistics")
    
    async def update_model_performance(
        self, 
        model_id: str, 
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新模型性能数据"""
        return await self._make_request("PUT", f"classification/models/{model_id}/performance", data=performance_data)
    
    # ============ 批量操作 ============
    
    async def create_batch_task(self, batch_task_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建批量任务"""
        return await self._make_request("POST", "classification/batch-tasks", data=batch_task_data)
    
    async def get_batch_task_status(self, batch_task_id: str) -> Dict[str, Any]:
        """获取批量任务状态"""
        return await self._make_request("GET", f"classification/batch-tasks/{batch_task_id}")
    
    async def update_batch_task_status(
        self, 
        batch_task_id: str, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新批量任务状态"""
        return await self._make_request("PUT", f"classification/batch-tasks/{batch_task_id}", data=update_data)
    
    # ============ 模型文件管理 ============
    
    async def upload_model_file(
        self, 
        model_id: str, 
        file_path: str, 
        file_content: bytes
    ) -> Dict[str, Any]:
        """上传模型文件"""
        # 这里简化处理，实际应该支持文件上传
        file_info = {
            "model_id": model_id,
            "file_path": file_path,
            "file_size": len(file_content),
            "upload_time": datetime.now().isoformat()
        }
        return await self._make_request("POST", "classification/model-files", data=file_info)
    
    async def download_model_file(self, model_id: str, file_path: str) -> bytes:
        """下载模型文件"""
        # 这里简化处理，实际应该返回文件内容
        response = await self._make_request("GET", f"classification/model-files/{model_id}")
        # 实际实现需要处理二进制文件下载
        return b""
    
    async def delete_model_file(self, model_id: str, file_path: str) -> Dict[str, Any]:
        """删除模型文件"""
        data = {"file_path": file_path}
        return await self._make_request("DELETE", f"classification/model-files/{model_id}", data=data)
    
    # ============ 数据导入导出 ============
    
    async def export_training_data(
        self, 
        project_id: str, 
        export_format: str = "json"
    ) -> Dict[str, Any]:
        """导出训练数据"""
        params = {"format": export_format}
        return await self._make_request("GET", f"classification/projects/{project_id}/export-data", params=params)
    
    async def import_training_data(
        self, 
        project_id: str, 
        import_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """导入训练数据"""
        data = {"project_id": project_id, "data": import_data}
        return await self._make_request("POST", "classification/import-data", data=data)
    
    # ============ 系统管理 ============
    
    async def health_check(self) -> Dict[str, Any]:
        """检查storage-service健康状态"""
        try:
            # 直接调用健康检查端点，不添加API前缀
            url = f"{self.base_url}/health"
            async with httpx.AsyncClient(**self.client_config) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error(f"Storage service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_system_info(self) -> Dict[str, Any]:
        """获取storage-service系统信息"""
        return await self._make_request("GET", "system/info")
    
    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, Any]:
        """清理旧数据"""
        data = {"days_old": days_old}
        return await self._make_request("POST", "system/cleanup", data=data)
    
    # ============ 日志和审计 ============
    
    async def log_operation(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录操作日志"""
        return await self._make_request("POST", "audit/operations", data=operation_data)
    
    async def get_operation_logs(
        self, 
        limit: int = 100, 
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取操作日志"""
        params = {"limit": limit, "offset": offset}
        if filters:
            params.update(filters)
        return await self._make_request("GET", "audit/operations", params=params)
    
    # ============ 辅助方法 ============
    
    async def close(self):
        """关闭客户端连接"""
        # httpx AsyncClient 会自动管理连接
        pass
    
    def get_storage_url(self, endpoint: str) -> str:
        """获取storage service的完整URL"""
        return f"{self.base_url}/api/v1/{endpoint.lstrip('/')}"