"""
知识图谱构建API控制器
提供实体抽取、关系抽取、图谱构建和查询的REST API接口
基于无状态架构，所有数据通过storage-service持久化
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import logging

from ..schemas.knowledge_graph_schemas import (
    # 请求响应模型
    BaseResponse,
    EntityExtractionRequest, EntityExtractionResponse,
    RelationExtractionRequest, RelationExtractionResponse,
    GraphConstructionRequest, GraphConstructionResponse,
    GraphQueryRequest, GraphQueryResponse,
    ConceptMiningRequest, ConceptMiningResponse,
    
    # 数据模型
    Entity, Relation, KnowledgeGraph,
    ProcessingStatus, TaskStatus, GraphStatistics,
    
    # 枚举
    EntityType, RelationType, ExtractionMethod, GraphFormat
)
from ..services.knowledge_graph_service import KnowledgeGraphService
from ..config.settings import settings

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/knowledge-graph", tags=["Knowledge Graph"])

# 初始化知识图谱服务
kg_service = KnowledgeGraphService()


@router.post("/entities/extract", response_model=BaseResponse[EntityExtractionResponse])
async def extract_entities(request: EntityExtractionRequest) -> BaseResponse[EntityExtractionResponse]:
    """
    从文本中抽取命名实体
    
    支持多种抽取方法：
    - spacy: 基于spaCy的中文NER模型
    - bert: 基于BERT的实体识别
    - jieba: 基于jieba的中文分词和实体识别
    - hybrid: 混合多种方法的结果
    """
    try:
        logger.info(f"开始实体抽取任务，方法: {request.method}, 文本长度: {len(request.text)}")
        
        # 验证文本长度
        if len(request.text) > settings.max_text_length:
            raise HTTPException(
                status_code=400, 
                detail=f"文本长度超出限制，最大允许{settings.max_text_length}字符"
            )
        
        # 执行实体抽取
        entities = await kg_service.extract_entities(
            text=request.text,
            method=request.method,
            confidence_threshold=request.confidence_threshold,
            language=request.language or settings.default_language
        )
        
        # 构建响应
        response = EntityExtractionResponse(
            entities=entities,
            total_entities=len(entities),
            extraction_method=request.method,
            language=request.language or settings.default_language,
            confidence_threshold=request.confidence_threshold,
            processing_time=0.0  # 实际处理时间由service层计算
        )
        
        logger.info(f"实体抽取完成，共提取{len(entities)}个实体")
        
        return BaseResponse(
            success=True,
            message="实体抽取成功",
            data=response
        )
        
    except Exception as e:
        logger.error(f"实体抽取失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"实体抽取失败: {str(e)}",
            data=None
        )


@router.post("/relations/extract", response_model=BaseResponse[RelationExtractionResponse])
async def extract_relations(request: RelationExtractionRequest) -> BaseResponse[RelationExtractionResponse]:
    """
    从文本中抽取实体间的关系
    
    支持基于规则和机器学习的关系抽取方法
    可与实体抽取结果结合使用
    """
    try:
        logger.info(f"开始关系抽取任务，文本长度: {len(request.text)}")
        
        # 验证文本长度
        if len(request.text) > settings.max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"文本长度超出限制，最大允许{settings.max_text_length}字符"
            )
        
        # 执行关系抽取
        relations = await kg_service.extract_relations(
            text=request.text,
            entities=request.entities,
            confidence_threshold=request.confidence_threshold,
            max_distance=request.max_distance,
            language=request.language or settings.default_language
        )
        
        # 构建响应
        response = RelationExtractionResponse(
            relations=relations,
            total_relations=len(relations),
            input_entities=len(request.entities) if request.entities else 0,
            confidence_threshold=request.confidence_threshold,
            max_distance=request.max_distance,
            language=request.language or settings.default_language,
            processing_time=0.0
        )
        
        logger.info(f"关系抽取完成，共抽取{len(relations)}个关系")
        
        return BaseResponse(
            success=True,
            message="关系抽取成功",
            data=response
        )
        
    except Exception as e:
        logger.error(f"关系抽取失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"关系抽取失败: {str(e)}",
            data=None
        )


@router.post("/graph/construct", response_model=BaseResponse[GraphConstructionResponse])
async def construct_graph(request: GraphConstructionRequest) -> BaseResponse[GraphConstructionResponse]:
    """
    构建知识图谱
    
    基于提供的实体和关系构建完整的知识图谱
    支持图谱优化、去重和质量评估
    """
    try:
        logger.info(f"开始图谱构建任务，项目ID: {request.project_id}")
        
        # 验证输入数据
        if not request.entities and not request.relations:
            raise HTTPException(
                status_code=400,
                detail="必须提供实体或关系数据"
            )
        
        # 执行图谱构建
        result = await kg_service.construct_graph(
            project_id=request.project_id,
            entities=request.entities,
            relations=request.relations,
            optimize_graph=request.optimize_graph,
            remove_duplicates=request.remove_duplicates,
            calculate_centrality=request.calculate_centrality
        )
        
        # 构建响应
        response = GraphConstructionResponse(
            graph_id=result["graph_id"],
            project_id=request.project_id,
            nodes_count=result["nodes_count"],
            edges_count=result["edges_count"],
            graph_statistics=result.get("statistics"),
            centrality_metrics=result.get("centrality_metrics"),
            quality_score=result.get("quality_score"),
            optimization_applied=request.optimize_graph,
            duplicates_removed=request.remove_duplicates,
            processing_time=result.get("processing_time", 0.0)
        )
        
        logger.info(f"图谱构建完成，图谱ID: {result['graph_id']}")
        
        return BaseResponse(
            success=True,
            message="知识图谱构建成功",
            data=response
        )
        
    except Exception as e:
        logger.error(f"图谱构建失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"图谱构建失败: {str(e)}",
            data=None
        )


@router.post("/graph/query", response_model=BaseResponse[GraphQueryResponse])
async def query_graph(request: GraphQueryRequest) -> BaseResponse[GraphQueryResponse]:
    """
    查询知识图谱
    
    支持多种查询方式：
    - 实体查询：根据实体名称或类型查询
    - 关系查询：根据关系类型查询
    - 路径查询：查找实体间的路径
    - 邻居查询：查找实体的邻居节点
    """
    try:
        logger.info(f"开始图谱查询，项目ID: {request.project_id}, 查询类型: {request.query_type}")
        
        # 执行图谱查询
        result = await kg_service.query_graph(
            project_id=request.project_id,
            query_type=request.query_type,
            query_params=request.query_params,
            limit=request.limit,
            offset=request.offset
        )
        
        # 构建响应
        response = GraphQueryResponse(
            project_id=request.project_id,
            query_type=request.query_type,
            results=result.get("results", []),
            total_results=result.get("total_results", 0),
            query_time=result.get("query_time", 0.0),
            limit=request.limit,
            offset=request.offset
        )
        
        logger.info(f"图谱查询完成，返回{len(response.results)}条结果")
        
        return BaseResponse(
            success=True,
            message="图谱查询成功",
            data=response
        )
        
    except Exception as e:
        logger.error(f"图谱查询失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"图谱查询失败: {str(e)}",
            data=None
        )


@router.post("/concepts/mine", response_model=BaseResponse[ConceptMiningResponse])
async def mine_concepts(request: ConceptMiningRequest) -> BaseResponse[ConceptMiningResponse]:
    """
    从文本集合中挖掘概念主题
    
    使用LDA主题模型和词嵌入技术
    识别历史文本中的重要概念和主题
    """
    try:
        logger.info(f"开始概念挖掘任务，文档数量: {len(request.documents)}")
        
        # 验证输入数据
        if not request.documents:
            raise HTTPException(
                status_code=400,
                detail="必须提供至少一个文档"
            )
        
        if len(request.documents) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"文档数量超出限制，最大允许{settings.max_batch_size}个"
            )
        
        # 执行概念挖掘
        result = await kg_service.mine_concepts(
            documents=request.documents,
            num_topics=request.num_topics,
            min_frequency=request.min_frequency,
            language=request.language or settings.default_language
        )
        
        # 构建响应
        response = ConceptMiningResponse(
            concepts=result.get("concepts", []),
            topics=result.get("topics", []),
            concept_relations=result.get("concept_relations", []),
            total_concepts=len(result.get("concepts", [])),
            total_topics=len(result.get("topics", [])),
            num_documents=len(request.documents),
            language=request.language or settings.default_language,
            processing_time=result.get("processing_time", 0.0)
        )
        
        logger.info(f"概念挖掘完成，发现{response.total_concepts}个概念")
        
        return BaseResponse(
            success=True,
            message="概念挖掘成功",
            data=response
        )
        
    except Exception as e:
        logger.error(f"概念挖掘失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"概念挖掘失败: {str(e)}",
            data=None
        )


@router.get("/projects/{project_id}/statistics", response_model=BaseResponse[GraphStatistics])
async def get_project_statistics(project_id: str) -> BaseResponse[GraphStatistics]:
    """
    获取项目的知识图谱统计信息
    
    包括实体数量、关系数量、图谱连通性等统计指标
    """
    try:
        logger.info(f"获取项目统计信息，项目ID: {project_id}")
        
        # 获取统计信息
        statistics = await kg_service.get_project_statistics(project_id)
        
        if not statistics:
            raise HTTPException(
                status_code=404,
                detail="项目不存在或无统计信息"
            )
        
        return BaseResponse(
            success=True,
            message="获取统计信息成功",
            data=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"获取统计信息失败: {str(e)}",
            data=None
        )


@router.post("/batch/process", response_model=BaseResponse[Dict[str, Any]])
async def batch_process(
    request: List[str],
    background_tasks: BackgroundTasks,
    method: ExtractionMethod = ExtractionMethod.hybrid,
    project_id: Optional[str] = None
) -> BaseResponse[Dict[str, Any]]:
    """
    批量处理文档
    
    支持大批量文档的实体抽取、关系抽取和图谱构建
    使用后台任务处理，返回任务ID供查询进度
    """
    try:
        logger.info(f"开始批量处理任务，文档数量: {len(request)}")
        
        # 验证输入
        if not request:
            raise HTTPException(
                status_code=400,
                detail="必须提供至少一个文档"
            )
        
        if len(request) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"文档数量超出限制，最大允许{settings.max_batch_size}个"
            )
        
        # 生成任务ID
        task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 添加后台任务
        background_tasks.add_task(
            _process_batch_documents,
            task_id=task_id,
            documents=request,
            method=method,
            project_id=project_id
        )
        
        response_data = {
            "task_id": task_id,
            "status": "started",
            "total_documents": len(request),
            "method": method.value,
            "project_id": project_id
        }
        
        logger.info(f"批量处理任务已启动，任务ID: {task_id}")
        
        return BaseResponse(
            success=True,
            message="批量处理任务已启动",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"批量处理启动失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"批量处理启动失败: {str(e)}",
            data=None
        )


@router.get("/batch/status/{task_id}", response_model=BaseResponse[TaskStatus])
async def get_batch_status(task_id: str) -> BaseResponse[TaskStatus]:
    """
    查询批量处理任务状态
    
    返回任务的当前状态、进度和结果信息
    """
    try:
        logger.info(f"查询批量任务状态，任务ID: {task_id}")
        
        # 获取任务状态（通过storage-service）
        status = await kg_service.get_batch_status(task_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail="任务不存在"
            )
        
        return BaseResponse(
            success=True,
            message="获取任务状态成功",
            data=status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务状态失败: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"查询任务状态失败: {str(e)}",
            data=None
        )


async def _process_batch_documents(
    task_id: str, 
    documents: List[str], 
    method: ExtractionMethod,
    project_id: Optional[str] = None
) -> None:
    """
    后台批量处理文档的任务函数
    """
    try:
        logger.info(f"开始执行批量处理任务: {task_id}")
        
        # 执行批量处理
        await kg_service.batch_extract(
            documents=documents,
            method=method,
            project_id=project_id,
            task_id=task_id
        )
        
        logger.info(f"批量处理任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"批量处理任务失败 {task_id}: {str(e)}")
        # 这里应该更新任务状态为失败
        # 实际实现中需要通过storage-service记录错误状态


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    服务健康检查
    
    检查服务状态、依赖服务连接和系统资源
    """
    try:
        # 检查storage-service连接
        storage_health = await kg_service.storage_client.health_check()
        
        # 检查NLP模型状态
        models_status = await kg_service.check_models_status()
        
        return {
            "service": "knowledge-graph-service",
            "status": "healthy",
            "version": settings.service_version,
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "storage_service": storage_health,
                "nlp_models": models_status
            }
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "service": "knowledge-graph-service",
            "status": "unhealthy",
            "version": settings.service_version,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }