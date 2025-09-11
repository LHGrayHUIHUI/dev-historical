"""
Storage Service 客户端

该模块提供与storage-service的HTTP通信接口，
支持内容获取、合并任务管理、结果存储等操作。
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime

from ..config.settings import settings
from ..models.merger_models import (
    ContentItem, MergeResult, ContentAnalysis, 
    ContentRelationship, MergeError
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
                    "User-Agent": "content-merger-service/1.0.0"
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
                response = await self._session.put(url, json=data)
            elif method.upper() == "DELETE":
                response = await self._session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"Request successful: {response.status_code}")
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise MergeError(f"Storage service error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise MergeError(f"Failed to connect to storage service: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise MergeError(f"Storage service request failed: {str(e)}")
    
    # 内容相关操作
    
    async def get_content_by_id(self, content_id: str) -> Optional[ContentItem]:
        """根据ID获取内容"""
        try:
            response = await self._make_request("GET", f"/api/v1/contents/{content_id}")
            
            if response.get("success") and response.get("data"):
                content_data = response["data"]
                return ContentItem(
                    id=content_data["id"],
                    title=content_data["title"],
                    content=content_data["content"],
                    metadata=content_data.get("metadata", {}),
                    analysis=content_data.get("analysis", {})
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get content {content_id}: {str(e)}")
            raise MergeError(f"Failed to retrieve content: {str(e)}")
    
    async def get_contents_by_ids(self, content_ids: List[str]) -> List[ContentItem]:
        """根据ID列表批量获取内容"""
        try:
            # 并发获取所有内容
            tasks = [self.get_content_by_id(content_id) for content_id in content_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            contents = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to get content {content_ids[i]}: {str(result)}")
                    continue
                
                if result is not None:
                    contents.append(result)
                else:
                    logger.warning(f"Content {content_ids[i]} not found")
            
            if not contents:
                raise MergeError("No valid contents found")
            
            return contents
            
        except Exception as e:
            logger.error(f"Failed to get contents: {str(e)}")
            raise MergeError(f"Failed to retrieve contents: {str(e)}")
    
    async def save_content_analysis(self, content_id: str, 
                                  analysis: ContentAnalysis) -> bool:
        """保存内容分析结果"""
        try:
            data = {
                "content_id": content_id,
                "analysis_type": "comprehensive",
                "analysis_result": analysis.dict(),
                "analysis_time_ms": 0,
                "confidence_score": 0.95
            }
            
            response = await self._make_request("POST", "/api/v1/analysis/results", data)
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to save analysis for {content_id}: {str(e)}")
            return False
    
    # 合并任务相关操作
    
    async def create_merge_task(self, task_data: Dict[str, Any]) -> str:
        """创建合并任务"""
        try:
            response = await self._make_request("POST", "/api/v1/merge/tasks", task_data)
            
            if response.get("success") and response.get("data"):
                return response["data"]["task_id"]
            
            raise MergeError("Failed to create merge task")
            
        except Exception as e:
            logger.error(f"Failed to create merge task: {str(e)}")
            raise MergeError(f"Failed to create merge task: {str(e)}")
    
    async def update_merge_task_status(self, task_id: str, status: str,
                                     progress: Optional[int] = None,
                                     error_message: Optional[str] = None) -> bool:
        """更新合并任务状态"""
        try:
            data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if progress is not None:
                data["progress_percentage"] = progress
            
            if error_message:
                data["error_message"] = error_message
            
            response = await self._make_request("PUT", f"/api/v1/merge/tasks/{task_id}/status", data)
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to update task {task_id} status: {str(e)}")
            return False
    
    async def save_merge_result(self, task_id: str, result: MergeResult) -> bool:
        """保存合并结果"""
        try:
            data = {
                "task_id": task_id,
                "result": result.dict(),
                "saved_at": datetime.now().isoformat()
            }
            
            response = await self._make_request("POST", "/api/v1/merge/results", data)
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to save merge result for task {task_id}: {str(e)}")
            return False
    
    async def get_merge_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取合并任务状态"""
        try:
            response = await self._make_request("GET", f"/api/v1/merge/tasks/{task_id}/status")
            
            if response.get("success") and response.get("data"):
                return response["data"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task {task_id} status: {str(e)}")
            return None
    
    async def get_merge_result(self, task_id: str) -> Optional[MergeResult]:
        """获取合并结果"""
        try:
            response = await self._make_request("GET", f"/api/v1/merge/tasks/{task_id}/result")
            
            if response.get("success") and response.get("data"):
                result_data = response["data"]["result"]
                return MergeResult(**result_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get merge result for task {task_id}: {str(e)}")
            return None
    
    # 关系分析相关操作
    
    async def save_content_relationship(self, relationship: ContentRelationship) -> bool:
        """保存内容关系"""
        try:
            data = relationship.dict()
            data["detected_at"] = datetime.now().isoformat()
            
            response = await self._make_request("POST", "/api/v1/content/relationships", data)
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to save content relationship: {str(e)}")
            return False
    
    async def get_content_relationships(self, content_ids: List[str]) -> List[ContentRelationship]:
        """获取内容关系"""
        try:
            params = {"content_ids": ",".join(content_ids)}
            response = await self._make_request("GET", "/api/v1/content/relationships", params=params)
            
            relationships = []
            if response.get("success") and response.get("data"):
                for rel_data in response["data"]:
                    relationships.append(ContentRelationship(**rel_data))
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to get content relationships: {str(e)}")
            return []
    
    # 批量操作
    
    async def create_batch_merge_job(self, job_data: Dict[str, Any]) -> str:
        """创建批量合并任务"""
        try:
            response = await self._make_request("POST", "/api/v1/merge/batch", job_data)
            
            if response.get("success") and response.get("data"):
                return response["data"]["job_id"]
            
            raise MergeError("Failed to create batch merge job")
            
        except Exception as e:
            logger.error(f"Failed to create batch merge job: {str(e)}")
            raise MergeError(f"Failed to create batch merge job: {str(e)}")
    
    async def update_merge_job_progress(self, job_id: str, 
                                      completed: bool = False,
                                      failed: bool = False,
                                      error: Optional[str] = None) -> bool:
        """更新批量合并任务进度"""
        try:
            data = {
                "job_id": job_id,
                "updated_at": datetime.now().isoformat()
            }
            
            if completed:
                data["status"] = "completed"
            elif failed:
                data["status"] = "failed"
                if error:
                    data["error_message"] = error
            
            response = await self._make_request("PUT", f"/api/v1/merge/batch/{job_id}/progress", data)
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to update batch job {job_id} progress: {str(e)}")
            return False
    
    # 配置和偏好相关操作
    
    async def get_merge_strategies(self) -> List[Dict[str, Any]]:
        """获取合并策略配置"""
        try:
            response = await self._make_request("GET", "/api/v1/merge/strategies")
            
            if response.get("success") and response.get("data"):
                return response["data"]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get merge strategies: {str(e)}")
            return []
    
    async def get_user_merge_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户合并偏好"""
        try:
            response = await self._make_request("GET", f"/api/v1/users/{user_id}/merge-preferences")
            
            if response.get("success") and response.get("data"):
                return response["data"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user {user_id} merge preferences: {str(e)}")
            return None
    
    async def update_user_merge_preferences(self, user_id: str, 
                                          preferences: Dict[str, Any]) -> bool:
        """更新用户合并偏好"""
        try:
            data = {
                "user_id": user_id,
                "preferences": preferences,
                "updated_at": datetime.now().isoformat()
            }
            
            response = await self._make_request("PUT", f"/api/v1/users/{user_id}/merge-preferences", data)
            return response.get("success", False)
            
        except Exception as e:
            logger.error(f"Failed to update user {user_id} merge preferences: {str(e)}")
            return False
    
    # 健康检查
    
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