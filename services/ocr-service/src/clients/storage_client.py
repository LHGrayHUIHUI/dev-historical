"""
Storage服务客户端

无状态OCR服务与storage-service的通信接口。
所有数据存储和检索操作都通过此客户端完成。

Author: OCR开发团队
Created: 2025-01-15
Version: 2.0.0 (无状态架构)
"""

import asyncio
from typing import Dict, List, Optional, Any
import httpx
from pydantic import BaseModel
import json
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class OCRTaskRequest(BaseModel):
    """OCR任务请求模型"""
    file_path: str
    engine: str = "paddleocr"
    confidence_threshold: float = 0.8
    language_codes: str = "zh,en"
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    metadata: Optional[Dict[str, Any]] = None


class OCRResult(BaseModel):
    """OCR识别结果模型"""
    task_id: str
    file_path: str
    engine: str
    status: str  # pending, processing, completed, failed
    confidence_threshold: float
    text_content: Optional[str] = None
    bounding_boxes: Optional[List[Dict]] = None
    confidence_scores: Optional[List[float]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


def handle_http_errors(func):
    """HTTP错误处理装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP错误 {e.response.status_code}: {e.response.text}")
            raise StorageClientError(f"Storage服务请求失败: {e.response.status_code}")
        except httpx.ConnectError:
            logger.error("无法连接到Storage服务")
            raise StorageClientError("无法连接到Storage服务")
        except httpx.TimeoutException:
            logger.error("Storage服务请求超时")
            raise StorageClientError("Storage服务请求超时")
        except Exception as e:
            logger.error(f"Storage客户端未知错误: {str(e)}")
            raise StorageClientError(f"Storage客户端错误: {str(e)}")
    return wrapper


class StorageClientError(Exception):
    """Storage客户端异常"""
    pass


class StorageServiceClient:
    """Storage服务客户端
    
    负责OCR服务与storage-service之间的所有数据交互。
    包括任务管理、结果存储、文件操作等。
    """
    
    def __init__(self, base_url: str, timeout: int = 30, retries: int = 3):
        """初始化Storage客户端
        
        Args:
            base_url: Storage服务基础URL
            timeout: 请求超时时间（秒）
            retries: 重试次数
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retries = retries
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.client.aclose()
    
    @handle_http_errors
    async def create_ocr_task(self, task_request: OCRTaskRequest) -> str:
        """创建OCR任务
        
        Args:
            task_request: OCR任务请求数据
            
        Returns:
            创建的任务ID
        """
        response = await self.client.post(
            "/api/v1/ocr/tasks",
            json=task_request.dict()
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("data", {}).get("task_id")
        
        if not task_id:
            raise StorageClientError("创建任务失败：未返回任务ID")
        
        logger.info(f"成功创建OCR任务: {task_id}")
        return task_id
    
    @handle_http_errors
    async def get_ocr_result(self, task_id: str) -> Optional[OCRResult]:
        """获取OCR任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            OCR结果对象，如果不存在返回None
        """
        response = await self.client.get(f"/api/v1/ocr/tasks/{task_id}")
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        result = response.json()
        
        task_data = result.get("data")
        if not task_data:
            return None
        
        return OCRResult(**task_data)
    
    @handle_http_errors
    async def update_ocr_task_status(self, task_id: str, status: str, 
                                   error_message: Optional[str] = None) -> bool:
        """更新OCR任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态 (pending, processing, completed, failed)
            error_message: 错误消息（状态为failed时）
            
        Returns:
            更新是否成功
        """
        update_data = {"status": status}
        if error_message:
            update_data["error_message"] = error_message
        
        response = await self.client.patch(
            f"/api/v1/ocr/tasks/{task_id}/status",
            json=update_data
        )
        response.raise_for_status()
        
        logger.info(f"成功更新任务状态: {task_id} -> {status}")
        return True
    
    @handle_http_errors
    async def save_ocr_result(self, task_id: str, result_data: Dict[str, Any]) -> bool:
        """保存OCR识别结果
        
        Args:
            task_id: 任务ID
            result_data: 结果数据（包含text_content, bounding_boxes等）
            
        Returns:
            保存是否成功
        """
        response = await self.client.patch(
            f"/api/v1/ocr/tasks/{task_id}/result",
            json=result_data
        )
        response.raise_for_status()
        
        logger.info(f"成功保存OCR结果: {task_id}")
        return True
    
    @handle_http_errors
    async def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典，如果不存在返回None
        """
        response = await self.client.get(
            "/api/v1/files/info",
            params={"file_path": file_path}
        )
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        result = response.json()
        
        return result.get("data")
    
    @handle_http_errors
    async def download_file(self, file_path: str) -> bytes:
        """下载文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件字节内容
        """
        response = await self.client.get(
            "/api/v1/files/download",
            params={"file_path": file_path}
        )
        response.raise_for_status()
        
        return response.content
    
    @handle_http_errors
    async def upload_processed_file(self, file_path: str, content: bytes, 
                                  content_type: str = "application/octet-stream") -> str:
        """上传处理后的文件
        
        Args:
            file_path: 目标文件路径
            content: 文件内容
            content_type: 文件MIME类型
            
        Returns:
            上传后的文件路径
        """
        files = {"file": (file_path, content, content_type)}
        
        response = await self.client.post(
            "/api/v1/files/upload",
            files=files,
            params={"file_path": file_path}
        )
        response.raise_for_status()
        
        result = response.json()
        uploaded_path = result.get("data", {}).get("file_path")
        
        if not uploaded_path:
            raise StorageClientError("上传文件失败：未返回文件路径")
        
        logger.info(f"成功上传文件: {uploaded_path}")
        return uploaded_path
    
    @handle_http_errors
    async def list_pending_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取待处理任务列表
        
        Args:
            limit: 返回任务数量限制
            
        Returns:
            待处理任务列表
        """
        response = await self.client.get(
            "/api/v1/ocr/tasks",
            params={"status": "pending", "limit": limit}
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("data", [])
    
    @handle_http_errors
    async def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息
        
        Returns:
            任务统计数据
        """
        response = await self.client.get("/api/v1/ocr/tasks/statistics")
        response.raise_for_status()
        
        result = response.json()
        return result.get("data", {})
    
    async def health_check(self) -> bool:
        """检查Storage服务健康状态
        
        Returns:
            服务是否健康
        """
        try:
            response = await self.client.get("/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Storage服务健康检查失败: {str(e)}")
            return False


async def get_storage_client(base_url: str, timeout: int = 30, retries: int = 3) -> StorageServiceClient:
    """获取Storage客户端实例
    
    Args:
        base_url: Storage服务URL
        timeout: 请求超时时间
        retries: 重试次数
        
    Returns:
        Storage客户端实例
    """
    return StorageServiceClient(base_url, timeout, retries)