"""
Storage Service客户端
知识图谱服务与storage-service的HTTP通信客户端
处理所有数据存储和图谱管理操作
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
    
    # ============ 知识图谱项目管理 ============
    
    async def create_knowledge_graph_project(
        self,
        name: str,
        description: Optional[str],
        domain: str,
        language: str = "zh",
        entity_types: List[str] = None,
        relation_types: List[str] = None,
        created_by: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """创建知识图谱项目"""
        data = {
            "name": name,
            "description": description,
            "domain": domain,
            "language": language,
            "entity_types": entity_types or [],
            "relation_types": relation_types or [],
            "created_by": str(created_by) if created_by else None,
            "service_type": "knowledge_graph",
            "status": "active"
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/projects", data=data)
    
    async def get_knowledge_graph_project(self, project_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取知识图谱项目详情"""
        return await self._make_request("GET", f"/api/v1/knowledge-graph/projects/{project_id}")
    
    async def list_knowledge_graph_projects(
        self,
        user_id: Optional[Union[str, UUID]] = None,
        domain: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取知识图谱项目列表"""
        params = {"limit": limit, "offset": offset}
        if user_id:
            params["user_id"] = str(user_id)
        if domain:
            params["domain"] = domain
        if status:
            params["status"] = status
        
        return await self._make_request("GET", "/api/v1/knowledge-graph/projects", params=params)
    
    async def update_knowledge_graph_project(
        self,
        project_id: Union[str, UUID],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新知识图谱项目"""
        return await self._make_request("PUT", f"/api/v1/knowledge-graph/projects/{project_id}", data=updates)
    
    async def delete_knowledge_graph_project(self, project_id: Union[str, UUID]) -> Dict[str, Any]:
        """删除知识图谱项目"""
        return await self._make_request("DELETE", f"/api/v1/knowledge-graph/projects/{project_id}")
    
    # ============ 实体管理 ============
    
    async def create_entity_extraction_task(
        self,
        project_id: Union[str, UUID],
        document_id: Optional[Union[str, UUID]],
        text_content: str,
        extraction_method: str,
        extraction_config: Optional[Dict] = None,
        created_by: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """创建实体抽取任务"""
        data = {
            "project_id": str(project_id),
            "document_id": str(document_id) if document_id else None,
            "text_content": text_content[:1000],  # 只保存前1000字符
            "extraction_method": extraction_method,
            "extraction_config": extraction_config or {},
            "status": "pending",
            "created_by": str(created_by) if created_by else None
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/entity-extraction-tasks", data=data)
    
    async def update_entity_extraction_task(
        self,
        task_id: Union[str, UUID],
        status: str,
        entities_found: Optional[int] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新实体抽取任务状态"""
        data = {
            "status": status,
            "entities_found": entities_found,
            "processing_time": processing_time,
            "error_message": error_message
        }
        
        if status == "processing":
            data["started_at"] = time.time()
        elif status in ["completed", "failed", "cancelled"]:
            data["completed_at"] = time.time()
        
        # 移除None值
        data = {k: v for k, v in data.items() if v is not None}
        
        return await self._make_request("PUT", f"/api/v1/knowledge-graph/entity-extraction-tasks/{task_id}", data=data)
    
    async def save_entity(
        self,
        project_id: Union[str, UUID],
        entity_data: Dict[str, Any],
        source_document_id: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """保存实体"""
        data = {
            "project_id": str(project_id),
            "name": entity_data["name"],
            "entity_type": entity_data["entity_type"],
            "aliases": entity_data.get("aliases", []),
            "description": entity_data.get("description"),
            "properties": entity_data.get("properties", {}),
            "confidence_score": entity_data.get("confidence_score", 0.8),
            "source_documents": [str(source_document_id)] if source_document_id else [],
            "mention_count": 1
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/entities", data=data)
    
    async def get_entity(self, entity_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取实体详情"""
        return await self._make_request("GET", f"/api/v1/knowledge-graph/entities/{entity_id}")
    
    async def list_entities(
        self,
        project_id: Union[str, UUID],
        entity_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取实体列表"""
        params = {
            "project_id": str(project_id),
            "limit": limit,
            "offset": offset
        }
        if entity_type:
            params["entity_type"] = entity_type
        if name_pattern:
            params["name_pattern"] = name_pattern
        
        return await self._make_request("GET", "/api/v1/knowledge-graph/entities", params=params)
    
    async def update_entity(
        self,
        entity_id: Union[str, UUID],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新实体"""
        return await self._make_request("PUT", f"/api/v1/knowledge-graph/entities/{entity_id}", data=updates)
    
    async def delete_entity(self, entity_id: Union[str, UUID]) -> Dict[str, Any]:
        """删除实体"""
        return await self._make_request("DELETE", f"/api/v1/knowledge-graph/entities/{entity_id}")
    
    # ============ 关系管理 ============
    
    async def create_relation_extraction_task(
        self,
        project_id: Union[str, UUID],
        document_id: Optional[Union[str, UUID]],
        text_content: str,
        extraction_method: str,
        extraction_config: Optional[Dict] = None,
        created_by: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """创建关系抽取任务"""
        data = {
            "project_id": str(project_id),
            "document_id": str(document_id) if document_id else None,
            "text_content": text_content[:1000],
            "extraction_method": extraction_method,
            "extraction_config": extraction_config or {},
            "status": "pending",
            "created_by": str(created_by) if created_by else None
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/relation-extraction-tasks", data=data)
    
    async def update_relation_extraction_task(
        self,
        task_id: Union[str, UUID],
        status: str,
        relations_found: Optional[int] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新关系抽取任务状态"""
        data = {
            "status": status,
            "relations_found": relations_found,
            "processing_time": processing_time,
            "error_message": error_message
        }
        
        if status == "processing":
            data["started_at"] = time.time()
        elif status in ["completed", "failed", "cancelled"]:
            data["completed_at"] = time.time()
        
        # 移除None值
        data = {k: v for k, v in data.items() if v is not None}
        
        return await self._make_request("PUT", f"/api/v1/knowledge-graph/relation-extraction-tasks/{task_id}", data=data)
    
    async def save_relation(
        self,
        project_id: Union[str, UUID],
        relation_data: Dict[str, Any],
        source_document_id: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """保存关系"""
        data = {
            "project_id": str(project_id),
            "subject_entity_id": relation_data["subject_entity_id"],
            "predicate": relation_data["predicate"],
            "object_entity_id": relation_data["object_entity_id"],
            "confidence_score": relation_data.get("confidence_score", 0.8),
            "context": relation_data.get("context", ""),
            "source_sentence": relation_data.get("source_sentence", ""),
            "source_document_id": str(source_document_id) if source_document_id else None,
            "properties": relation_data.get("properties", {})
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/relations", data=data)
    
    async def get_relation(self, relation_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取关系详情"""
        return await self._make_request("GET", f"/api/v1/knowledge-graph/relations/{relation_id}")
    
    async def list_relations(
        self,
        project_id: Union[str, UUID],
        subject_entity_id: Optional[Union[str, UUID]] = None,
        object_entity_id: Optional[Union[str, UUID]] = None,
        predicate: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取关系列表"""
        params = {
            "project_id": str(project_id),
            "limit": limit,
            "offset": offset
        }
        if subject_entity_id:
            params["subject_entity_id"] = str(subject_entity_id)
        if object_entity_id:
            params["object_entity_id"] = str(object_entity_id)
        if predicate:
            params["predicate"] = predicate
        
        return await self._make_request("GET", "/api/v1/knowledge-graph/relations", params=params)
    
    # ============ 图谱构建和查询 ============
    
    async def create_graph_construction_task(
        self,
        project_id: Union[str, UUID],
        construction_config: Dict[str, Any],
        created_by: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """创建图谱构建任务"""
        data = {
            "project_id": str(project_id),
            "construction_config": construction_config,
            "status": "pending",
            "created_by": str(created_by) if created_by else None
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/construction-tasks", data=data)
    
    async def update_graph_construction_task(
        self,
        task_id: Union[str, UUID],
        status: str,
        nodes_count: Optional[int] = None,
        edges_count: Optional[int] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新图谱构建任务状态"""
        data = {
            "status": status,
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "processing_time": processing_time,
            "error_message": error_message
        }
        
        # 移除None值
        data = {k: v for k, v in data.items() if v is not None}
        
        return await self._make_request("PUT", f"/api/v1/knowledge-graph/construction-tasks/{task_id}", data=data)
    
    async def query_knowledge_graph(
        self,
        project_id: Union[str, UUID],
        query_type: str,
        query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """查询知识图谱"""
        data = {
            "project_id": str(project_id),
            "query_type": query_type,
            "parameters": query_params
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/query", data=data)
    
    # ============ 概念挖掘 ============
    
    async def create_concept_mining_task(
        self,
        project_id: Union[str, UUID],
        corpus_documents: List[str],
        mining_method: str,
        mining_config: Dict[str, Any],
        created_by: Optional[Union[str, UUID]] = None
    ) -> Dict[str, Any]:
        """创建概念挖掘任务"""
        data = {
            "project_id": str(project_id),
            "corpus_documents": corpus_documents,
            "mining_method": mining_method,
            "mining_config": mining_config,
            "status": "pending",
            "created_by": str(created_by) if created_by else None
        }
        
        return await self._make_request("POST", "/api/v1/knowledge-graph/concept-mining-tasks", data=data)
    
    async def update_concept_mining_task(
        self,
        task_id: Union[str, UUID],
        status: str,
        concepts_found: Optional[int] = None,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新概念挖掘任务状态"""
        data = {
            "status": status,
            "concepts_found": concepts_found,
            "processing_time": processing_time,
            "error_message": error_message
        }
        
        # 移除None值
        data = {k: v for k, v in data.items() if v is not None}
        
        return await self._make_request("PUT", f"/api/v1/knowledge-graph/concept-mining-tasks/{task_id}", data=data)
    
    # ============ 统计和分析 ============
    
    async def get_knowledge_graph_statistics(
        self,
        project_id: Optional[Union[str, UUID]] = None,
        user_id: Optional[Union[str, UUID]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        params = {}
        if project_id:
            params["project_id"] = str(project_id)
        if user_id:
            params["user_id"] = str(user_id)
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        
        return await self._make_request("GET", "/api/v1/knowledge-graph/statistics", params=params)
    
    # ============ 文档管理 ============
    
    async def get_document(self, document_id: Union[str, UUID]) -> Dict[str, Any]:
        """获取文档信息"""
        return await self._make_request("GET", f"/api/v1/documents/{document_id}")
    
    async def list_documents(
        self,
        dataset_id: Optional[Union[str, UUID]] = None,
        user_id: Optional[Union[str, UUID]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取文档列表"""
        params = {"limit": limit, "offset": offset}
        if dataset_id:
            params["dataset_id"] = str(dataset_id)
        if user_id:
            params["user_id"] = str(user_id)
        
        return await self._make_request("GET", "/api/v1/documents", params=params)
    
    # ============ 缓存管理 ============
    
    async def cache_get(self, key: str) -> Dict[str, Any]:
        """从缓存获取数据"""
        params = {"key": key}
        return await self._make_request("GET", "/api/v1/cache", params=params)
    
    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """设置缓存数据"""
        data = {
            "key": key,
            "value": value,
            "ttl": ttl or 3600
        }
        return await self._make_request("POST", "/api/v1/cache", data=data)
    
    async def cache_delete(self, key: str) -> Dict[str, Any]:
        """删除缓存数据"""
        data = {"key": key}
        return await self._make_request("DELETE", "/api/v1/cache", data=data)
    
    # ============ 健康检查 ============
    
    async def health_check(self) -> Dict[str, Any]:
        """检查storage-service健康状态"""
        return await self._make_request("GET", "/health")


# 全局storage client实例
storage_client = StorageServiceClient()