"""
Storage Service客户端
NLP服务与storage-service的HTTP通信客户端
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
    
    # ============ NLP任务管理 ============
    
    async def create_nlp_task(
        self,
        dataset_id: Union[str, UUID],
        text_content: str,
        processing_type: str,
        nlp_model: str,
        language: str = "zh",
        config: Optional[Dict] = None,
        created_by: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """创建NLP处理任务"""
        data = {
            "dataset_id": str(dataset_id),
            "text_content": text_content,
            "text_length": len(text_content),
            "processing_type": processing_type,
            "processing_status": "pending",
            "nlp_model": nlp_model,
            "language": language,
            "config": config or {},
            "created_by": str(created_by) if created_by else None
        }
        
        return await self._make_request("POST", "/api/v1/nlp/tasks", data=data)
    
    async def update_nlp_task_status(
        self,
        task_id: Union[str, UUID],
        status: str,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新NLP任务状态"""
        data = {
            "processing_status": status,
            "processing_time": processing_time,
            "error_message": error_message
        }
        
        if status == "processing":
            data["started_at"] = time.time()
        elif status in ["completed", "failed"]:
            data["completed_at"] = time.time()
        
        return await self._make_request("PUT", f"/api/v1/nlp/tasks/{task_id}", data=data)
    
    async def get_nlp_task(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取NLP任务详情"""
        return await self._make_request("GET", f"/api/v1/nlp/tasks/{task_id}")
    
    async def get_nlp_tasks(
        self,
        dataset_id: Optional[Union[str, UUID]] = None,
        processing_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取NLP任务列表"""
        params = {"limit": limit, "offset": offset}
        if dataset_id:
            params["dataset_id"] = str(dataset_id)
        if processing_type:
            params["processing_type"] = processing_type
        if status:
            params["status"] = status
        
        return await self._make_request("GET", "/api/v1/nlp/tasks", params=params)
    
    # ============ 分词结果存储 ============
    
    async def save_segmentation_result(
        self,
        task_id: Union[str, UUID],
        original_text: str,
        segmented_text: str,
        words: List[Dict],
        segmentation_method: str
    ) -> Dict[str, Any]:
        """保存分词结果"""
        data = {
            "task_id": str(task_id),
            "original_text": original_text,
            "segmented_text": segmented_text,
            "word_count": len(words),
            "unique_word_count": len(set(word["text"] for word in words)),
            "words": words,
            "segmentation_method": segmentation_method
        }
        
        return await self._make_request("POST", "/api/v1/nlp/segmentation", data=data)
    
    async def get_segmentation_result(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取分词结果"""
        return await self._make_request("GET", f"/api/v1/nlp/segmentation/{task_id}")
    
    # ============ 词性标注结果存储 ============
    
    async def save_pos_tagging_result(
        self,
        task_id: Union[str, UUID],
        words_with_pos: List[Dict],
        pos_distribution: Dict[str, int],
        tagging_method: str
    ) -> Dict[str, Any]:
        """保存词性标注结果"""
        data = {
            "task_id": str(task_id),
            "words_with_pos": words_with_pos,
            "pos_distribution": pos_distribution,
            "tagging_method": tagging_method
        }
        
        return await self._make_request("POST", "/api/v1/nlp/pos-tagging", data=data)
    
    async def get_pos_tagging_result(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取词性标注结果"""
        return await self._make_request("GET", f"/api/v1/nlp/pos-tagging/{task_id}")
    
    # ============ 命名实体识别结果存储 ============
    
    async def save_ner_result(
        self,
        task_id: Union[str, UUID],
        entities: List[Dict],
        entity_types: Dict[str, int],
        ner_model: str
    ) -> Dict[str, Any]:
        """保存命名实体识别结果"""
        data = {
            "task_id": str(task_id),
            "entities": entities,
            "entity_types": entity_types,
            "ner_model": ner_model
        }
        
        return await self._make_request("POST", "/api/v1/nlp/ner", data=data)
    
    async def get_ner_result(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取命名实体识别结果"""
        return await self._make_request("GET", f"/api/v1/nlp/ner/{task_id}")
    
    # ============ 情感分析结果存储 ============
    
    async def save_sentiment_result(
        self,
        task_id: Union[str, UUID],
        sentiment_label: str,
        sentiment_score: float,
        confidence: float,
        emotion_details: Optional[Dict] = None,
        sentiment_model: str = ""
    ) -> Dict[str, Any]:
        """保存情感分析结果"""
        data = {
            "task_id": str(task_id),
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "emotion_details": emotion_details or {},
            "sentiment_model": sentiment_model
        }
        
        return await self._make_request("POST", "/api/v1/nlp/sentiment", data=data)
    
    async def get_sentiment_result(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取情感分析结果"""
        return await self._make_request("GET", f"/api/v1/nlp/sentiment/{task_id}")
    
    # ============ 关键词提取结果存储 ============
    
    async def save_keyword_result(
        self,
        task_id: Union[str, UUID],
        keywords: List[Dict],
        extraction_method: str
    ) -> Dict[str, Any]:
        """保存关键词提取结果"""
        data = {
            "task_id": str(task_id),
            "keywords": keywords,
            "extraction_method": extraction_method,
            "keyword_count": len(keywords)
        }
        
        return await self._make_request("POST", "/api/v1/nlp/keywords", data=data)
    
    async def get_keyword_result(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取关键词提取结果"""
        return await self._make_request("GET", f"/api/v1/nlp/keywords/{task_id}")
    
    # ============ 文本摘要结果存储 ============
    
    async def save_summary_result(
        self,
        task_id: Union[str, UUID],
        original_length: int,
        summary_text: str,
        summary_method: str,
        summary_model: str = ""
    ) -> Dict[str, Any]:
        """保存文本摘要结果"""
        data = {
            "task_id": str(task_id),
            "original_length": original_length,
            "summary_text": summary_text,
            "summary_length": len(summary_text),
            "compression_ratio": len(summary_text) / original_length if original_length > 0 else 0,
            "summary_method": summary_method,
            "summary_model": summary_model
        }
        
        return await self._make_request("POST", "/api/v1/nlp/summary", data=data)
    
    async def get_summary_result(self, task_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取文本摘要结果"""
        return await self._make_request("GET", f"/api/v1/nlp/summary/{task_id}")
    
    # ============ 数据集操作 ============
    
    async def get_dataset(self, dataset_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取数据集信息"""
        return await self._make_request("GET", f"/api/v1/datasets/{dataset_id}")
    
    async def get_dataset_content(self, dataset_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取数据集内容"""
        return await self._make_request("GET", f"/api/v1/datasets/{dataset_id}/content")
    
    async def update_dataset_processing_status(
        self,
        dataset_id: Union[str, UUID],
        nlp_status: str,
        nlp_progress: Optional[int] = None
    ) -> Dict[str, Any]:
        """更新数据集的NLP处理状态"""
        data = {
            "nlp_processing_status": nlp_status,
            "nlp_processing_progress": nlp_progress
        }
        
        return await self._make_request("PUT", f"/api/v1/datasets/{dataset_id}/nlp-status", data=data)
    
    # ============ 统计和分析 ============
    
    async def get_nlp_statistics(
        self,
        dataset_id: Optional[Union[str, UUID]] = None,
        processing_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取NLP处理统计"""
        params = {}
        if dataset_id:
            params["dataset_id"] = str(dataset_id)
        if processing_type:
            params["processing_type"] = processing_type
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        
        return await self._make_request("GET", "/api/v1/nlp/statistics", params=params)
    
    async def health_check(self) -> Dict[str, Any]:
        """检查storage-service健康状态"""
        return await self._make_request("GET", "/health")


# 全局storage client实例
storage_client = StorageServiceClient()