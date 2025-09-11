"""
内容合并API控制器

该模块提供内容合并服务的REST API接口，包括合并任务创建、
状态查询、结果获取、关系分析等功能。
"""

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from ..models.merger_models import (
    MergeRequest, BatchMergeRequest, RelationshipAnalysisRequest,
    MergePreviewRequest, MergeTaskResponse, MergeResultResponse,
    RelationshipAnalysisResponse, MergePreviewResponse, TaskProgressResponse,
    BatchJobResponse, BaseResponse, TaskStatus, MergeError
)
from ..services.content_merger_engine import ContentMergerEngine
from ..services.content_analyzer import ContentAnalyzer
from ..services.quality_assessor import QualityAssessor
from ..clients.storage_client import StorageServiceClient
from ..clients.ai_service_client import AIServiceClient
from ..config.settings import settings

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/merge", tags=["content-merger"])

# 全局服务实例
storage_client = StorageServiceClient()
ai_client = AIServiceClient()
content_analyzer = ContentAnalyzer()
quality_assessor = QualityAssessor()
merger_engine = ContentMergerEngine(ai_client, content_analyzer, quality_assessor)

# 任务状态存储（实际应用中应使用Redis或数据库）
task_registry: Dict[str, Dict[str, Any]] = {}

@router.post("/create", response_model=MergeTaskResponse)
async def create_merge_task(
    request: MergeRequest,
    background_tasks: BackgroundTasks
) -> MergeTaskResponse:
    """
    创建内容合并任务
    
    Args:
        request: 合并请求
        background_tasks: 后台任务管理器
        
    Returns:
        合并任务响应
    """
    try:
        logger.info(f"Creating merge task with strategy: {request.strategy}, mode: {request.mode}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 验证源内容存在性
        async with storage_client:
            source_contents = await storage_client.get_contents_by_ids(request.source_content_ids)
        
        if len(source_contents) != len(request.source_content_ids):
            raise HTTPException(
                status_code=404,
                detail="部分源内容不存在"
            )
        
        # 估算处理时间
        total_length = sum(len(content.content) for content in source_contents)
        estimated_time = max(3, total_length // 1000)  # 简化的时间估算
        
        # 创建任务记录
        task_data = {
            "task_id": task_id,
            "user_id": request.user_id,
            "source_content_ids": request.source_content_ids,
            "strategy": request.strategy.value,
            "mode": request.mode.value,
            "config": request.config.dict(),
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "estimated_time_minutes": estimated_time
        }
        
        # 保存到storage-service
        async with storage_client:
            await storage_client.create_merge_task(task_data)
        
        # 注册任务到本地缓存
        task_registry[task_id] = {
            "status": TaskStatus.PENDING,
            "progress": 0,
            "created_at": datetime.now(),
            "source_contents": source_contents,
            "request": request
        }
        
        # 启动后台合并任务
        background_tasks.add_task(process_merge_task, task_id)
        
        # 分析源内容特征
        source_analysis = await analyze_source_contents(source_contents)
        
        return MergeTaskResponse(
            success=True,
            message="合并任务创建成功",
            data={
                "task_id": task_id,
                "status": TaskStatus.PENDING.value,
                "estimated_time_minutes": estimated_time,
                "source_analysis": source_analysis
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create merge task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"创建合并任务失败: {str(e)}"
        )

@router.get("/tasks/{task_id}/status", response_model=TaskProgressResponse)
async def get_task_status(task_id: str) -> TaskProgressResponse:
    """
    获取合并任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务进度响应
    """
    try:
        # 首先从本地缓存获取
        if task_id in task_registry:
            task_info = task_registry[task_id]
            
            return TaskProgressResponse(
                success=True,
                data={
                    "task_id": task_id,
                    "status": task_info["status"].value,
                    "progress_percentage": task_info["progress"],
                    "current_step": task_info.get("current_step"),
                    "estimated_remaining_time": task_info.get("estimated_remaining_time"),
                    "steps_completed": task_info.get("steps_completed", []),
                    "error_message": task_info.get("error_message")
                }
            )
        
        # 从storage-service获取
        async with storage_client:
            task_status = await storage_client.get_merge_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=404,
                detail="任务不存在"
            )
        
        return TaskProgressResponse(
            success=True,
            data=task_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取任务状态失败: {str(e)}"
        )

@router.get("/tasks/{task_id}/result", response_model=MergeResultResponse)
async def get_merge_result(task_id: str) -> MergeResultResponse:
    """
    获取合并结果
    
    Args:
        task_id: 任务ID
        
    Returns:
        合并结果响应
    """
    try:
        # 检查任务状态
        if task_id in task_registry:
            task_info = task_registry[task_id]
            if task_info["status"] != TaskStatus.COMPLETED:
                raise HTTPException(
                    status_code=400,
                    detail=f"任务尚未完成，当前状态: {task_info['status'].value}"
                )
        
        # 从storage-service获取结果
        async with storage_client:
            result = await storage_client.get_merge_result(task_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="合并结果不存在"
            )
        
        return MergeResultResponse(
            success=True,
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get merge result: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取合并结果失败: {str(e)}"
        )

@router.post("/batch", response_model=BatchJobResponse)
async def create_batch_merge(
    request: BatchMergeRequest,
    background_tasks: BackgroundTasks
) -> BatchJobResponse:
    """
    创建批量合并任务
    
    Args:
        request: 批量合并请求
        background_tasks: 后台任务管理器
        
    Returns:
        批量任务响应
    """
    try:
        logger.info(f"Creating batch merge job with {len(request.content_groups)} groups")
        
        # 生成任务ID
        job_id = str(uuid.uuid4())
        
        # 创建批量任务
        job_data = {
            "job_id": job_id,
            "user_id": request.user_id,
            "content_groups": request.content_groups,
            "merge_config": request.merge_config.dict(),
            "job_name": request.job_name,
            "total_groups": len(request.content_groups),
            "created_at": datetime.now().isoformat()
        }
        
        async with storage_client:
            await storage_client.create_batch_merge_job(job_data)
        
        # 估算完成时间
        total_contents = sum(len(group) for group in request.content_groups)
        estimated_completion_minutes = max(5, total_contents * 2)
        
        # 启动批量处理任务
        background_tasks.add_task(process_batch_merge_job, job_id, request)
        
        return BatchJobResponse(
            success=True,
            message="批量合并任务创建成功",
            data={
                "job_id": job_id,
                "total_groups": len(request.content_groups),
                "estimated_completion_time": f"{estimated_completion_minutes} minutes"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create batch merge job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"创建批量合并任务失败: {str(e)}"
        )

@router.post("/analyze-relationships", response_model=RelationshipAnalysisResponse)
async def analyze_content_relationships(
    request: RelationshipAnalysisRequest
) -> RelationshipAnalysisResponse:
    """
    分析内容关系
    
    Args:
        request: 关系分析请求
        
    Returns:
        关系分析响应
    """
    try:
        logger.info(f"Analyzing relationships for {len(request.content_ids)} contents")
        
        # 获取内容
        async with storage_client:
            contents = await storage_client.get_contents_by_ids(request.content_ids)
        
        if len(contents) != len(request.content_ids):
            raise HTTPException(
                status_code=404,
                detail="部分内容不存在"
            )
        
        # 分析内容特征
        analyses = []
        for content in contents:
            analysis = await content_analyzer.analyze_content(content)
            analyses.append(analysis)
        
        # 计算相似度矩阵
        similarity_matrix = []
        for i in range(len(contents)):
            row = []
            for j in range(len(contents)):
                if i == j:
                    row.append(1.0)
                else:
                    similarity = await content_analyzer.analyze_content_similarity(
                        contents[i].content, contents[j].content
                    )
                    row.append(similarity)
            similarity_matrix.append(row)
        
        # 分析时间顺序
        temporal_order = []
        for i, analysis in enumerate(analyses):
            temporal_info = analysis.get('temporal_info', {})
            time_score = 0.0
            
            # 从时间信息计算分数
            dates = temporal_info.get('specific_dates', [])
            if dates:
                try:
                    years = [int(d) for d in dates if d.isdigit()]
                    if years:
                        time_score = float(min(years))
                except:
                    pass
            
            temporal_order.append({
                "content_id": contents[i].id,
                "time_score": time_score
            })
        
        # 分析主题重叠
        topic_overlaps = []
        topic_content_map = {}
        
        for i, analysis in enumerate(analyses):
            topics = analysis.get('topics', [])
            for topic in topics:
                topic_name = topic.get('topic', '')
                if topic_name:
                    if topic_name not in topic_content_map:
                        topic_content_map[topic_name] = []
                    topic_content_map[topic_name].append({
                        'content_id': contents[i].id,
                        'relevance': topic.get('relevance', 1.0)
                    })
        
        for topic_name, topic_contents in topic_content_map.items():
            if len(topic_contents) > 1:
                overlap_score = sum(tc['relevance'] for tc in topic_contents) / len(topic_contents)
                topic_overlaps.append({
                    "topic": topic_name,
                    "contents": [tc['content_id'] for tc in topic_contents],
                    "overlap_score": overlap_score
                })
        
        # 分析实体连接
        entity_connections = []
        entity_content_map = {}
        
        for i, analysis in enumerate(analyses):
            entities = analysis.get('entities', [])
            for entity in entities:
                entity_name = entity.get('name', '')
                if entity_name:
                    if entity_name not in entity_content_map:
                        entity_content_map[entity_name] = []
                    entity_content_map[entity_name].append({
                        'content_id': contents[i].id,
                        'importance': entity.get('importance', 1.0)
                    })
        
        for entity_name, entity_contents in entity_content_map.items():
            if len(entity_contents) > 1:
                connection_strength = sum(ec['importance'] for ec in entity_contents) / len(entity_contents)
                entity_connections.append({
                    "entity": entity_name,
                    "contents": [ec['content_id'] for ec in entity_contents],
                    "connection_strength": connection_strength
                })
        
        # 生成合并推荐
        merge_recommendations = await generate_merge_recommendations(
            similarity_matrix, temporal_order, topic_overlaps, entity_connections
        )
        
        relationships = {
            "similarity_matrix": {
                "content_ids": request.content_ids,
                "matrix": similarity_matrix
            },
            "temporal_order": temporal_order,
            "topic_overlaps": topic_overlaps,
            "entity_connections": entity_connections
        }
        
        return RelationshipAnalysisResponse(
            success=True,
            data={
                "relationships": relationships,
                "merge_recommendations": merge_recommendations
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze relationships: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"关系分析失败: {str(e)}"
        )

@router.post("/preview", response_model=MergePreviewResponse)
async def preview_merge(request: MergePreviewRequest) -> MergePreviewResponse:
    """
    预览合并结果
    
    Args:
        request: 合并预览请求
        
    Returns:
        合并预览响应
    """
    try:
        logger.info(f"Generating merge preview for {len(request.content_ids)} contents")
        
        # 获取内容
        async with storage_client:
            contents = await storage_client.get_contents_by_ids(request.content_ids)
        
        if len(contents) != len(request.content_ids):
            raise HTTPException(
                status_code=404,
                detail="部分内容不存在"
            )
        
        # 生成预览标题
        preview_title = await generate_preview_title(contents, request.strategy)
        
        # 生成预览章节
        preview_sections = []
        
        if request.strategy.value == "timeline":
            # 时间线预览
            sections = await generate_timeline_preview_sections(contents, request.preview_sections)
        elif request.strategy.value == "topic":
            # 主题预览
            sections = await generate_topic_preview_sections(contents, request.preview_sections)
        else:
            # 默认预览
            sections = await generate_default_preview_sections(contents, request.preview_sections)
        
        preview_sections = sections
        
        # 估算质量和处理时间
        total_length = sum(len(content.content) for content in contents)
        estimated_quality = min(90.0, 70.0 + len(contents) * 5)  # 简化估算
        estimated_processing_time = f"{max(3, total_length // 1000)}-{max(5, total_length // 500)}分钟"
        
        preview_data = {
            "title": preview_title,
            "sections": preview_sections,
            "estimated_quality": estimated_quality,
            "estimated_processing_time": estimated_processing_time
        }
        
        return MergePreviewResponse(
            success=True,
            data={"preview": preview_data}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate preview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"生成预览失败: {str(e)}"
        )

# 辅助函数

async def analyze_source_contents(contents: List[Any]) -> Dict[str, Any]:
    """分析源内容特征"""
    total_length = sum(len(content.content) for content in contents)
    
    # 估算重叠度
    if len(contents) > 1:
        overlap_count = 0
        total_comparisons = 0
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                similarity = await content_analyzer.analyze_content_similarity(
                    contents[i].content, contents[j].content
                )
                if similarity > 0.3:
                    overlap_count += 1
                total_comparisons += 1
        
        estimated_overlap = (overlap_count / total_comparisons * 100) if total_comparisons > 0 else 0
    else:
        estimated_overlap = 0
    
    # 计算复杂度分数
    complexity_scores = []
    for content in contents:
        analysis = await content_analyzer.analyze_content(content)
        complexity_scores.append(analysis.get('complexity_score', 5.0))
    
    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 5.0
    
    return {
        "total_content_length": total_length,
        "estimated_overlap": round(estimated_overlap, 1),
        "complexity_score": round(avg_complexity, 1)
    }

async def generate_merge_recommendations(similarity_matrix: List[List[float]],
                                       temporal_order: List[Dict],
                                       topic_overlaps: List[Dict],
                                       entity_connections: List[Dict]) -> List[Dict]:
    """生成合并推荐"""
    recommendations = []
    
    # 基于时间信息推荐时间线策略
    has_temporal_info = any(item['time_score'] > 0 for item in temporal_order)
    if has_temporal_info:
        recommendations.append({
            "strategy": "timeline",
            "confidence": 0.88,
            "reason": "内容具有明确的时间顺序"
        })
    
    # 基于主题重叠推荐主题策略
    if topic_overlaps:
        avg_overlap = sum(topic['overlap_score'] for topic in topic_overlaps) / len(topic_overlaps)
        if avg_overlap > 0.6:
            recommendations.append({
                "strategy": "topic",
                "confidence": 0.75,
                "reason": "存在较强的主题关联性"
            })
    
    # 基于实体连接推荐逻辑策略
    if entity_connections:
        strong_connections = [ec for ec in entity_connections if ec['connection_strength'] > 0.7]
        if strong_connections:
            recommendations.append({
                "strategy": "logic",
                "confidence": 0.65,
                "reason": "实体间存在较强的逻辑关联"
            })
    
    # 按置信度排序
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    return recommendations[:3]  # 返回前3个推荐

async def generate_preview_title(contents: List[Any], strategy: Any) -> str:
    """生成预览标题"""
    if len(contents) == 1:
        return contents[0].title
    
    # 提取关键词生成标题
    all_titles = [content.title for content in contents]
    
    if strategy.value == "timeline":
        return "历史发展脉络综述"
    elif strategy.value == "topic":
        return "主题内容综合分析"
    else:
        return f"多内容合并综述（{len(contents)}篇）"

async def generate_timeline_preview_sections(contents: List[Any], 
                                           section_count: int) -> List[Dict]:
    """生成时间线预览章节"""
    sections = []
    
    # 按时间分组（简化实现）
    time_groups = ["早期", "中期", "后期"]
    
    for i, period in enumerate(time_groups[:section_count]):
        section = {
            "title": f"{period}发展阶段",
            "preview_content": f"在{period}，历史事件按时间顺序展开...",
            "estimated_length": 800 + i * 200
        }
        sections.append(section)
    
    return sections

async def generate_topic_preview_sections(contents: List[Any], 
                                        section_count: int) -> List[Dict]:
    """生成主题预览章节"""
    sections = []
    
    # 主题分类（简化实现）
    topics = ["政治发展", "经济变化", "文化演进", "社会变迁"]
    
    for i, topic in enumerate(topics[:section_count]):
        section = {
            "title": topic,
            "preview_content": f"关于{topic}的综合分析，整合相关内容...",
            "estimated_length": 1000 + i * 300
        }
        sections.append(section)
    
    return sections

async def generate_default_preview_sections(contents: List[Any], 
                                          section_count: int) -> List[Dict]:
    """生成默认预览章节"""
    sections = []
    
    section_titles = ["概述", "详细分析", "总结"]
    
    for i, title in enumerate(section_titles[:section_count]):
        section = {
            "title": title,
            "preview_content": f"{title}部分将整合多个内容的相关信息...",
            "estimated_length": 600 + i * 400
        }
        sections.append(section)
    
    return sections

# 后台任务处理函数

async def process_merge_task(task_id: str):
    """处理合并任务"""
    try:
        logger.info(f"Processing merge task: {task_id}")
        
        if task_id not in task_registry:
            logger.error(f"Task {task_id} not found in registry")
            return
        
        task_info = task_registry[task_id]
        source_contents = task_info["source_contents"]
        request = task_info["request"]
        
        # 更新任务状态
        await update_task_status(task_id, TaskStatus.ANALYZING, 10, "开始分析内容")
        
        # 执行合并
        await update_task_status(task_id, TaskStatus.MERGING, 50, "执行内容合并")
        
        result = await merger_engine.merge_contents(request)
        
        # 保存结果
        await update_task_status(task_id, TaskStatus.MERGING, 90, "保存合并结果")
        
        async with storage_client:
            await storage_client.save_merge_result(task_id, result)
            await storage_client.update_merge_task_status(task_id, TaskStatus.COMPLETED.value, 100)
        
        # 更新本地状态
        task_registry[task_id]["status"] = TaskStatus.COMPLETED
        task_registry[task_id]["progress"] = 100
        task_registry[task_id]["result"] = result
        
        logger.info(f"Merge task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Merge task {task_id} failed: {str(e)}")
        
        # 更新错误状态
        await update_task_status(task_id, TaskStatus.FAILED, 0, None, str(e))
        
        async with storage_client:
            await storage_client.update_merge_task_status(
                task_id, TaskStatus.FAILED.value, error_message=str(e)
            )

async def update_task_status(task_id: str, status: TaskStatus, 
                           progress: int, current_step: Optional[str] = None,
                           error_message: Optional[str] = None):
    """更新任务状态"""
    if task_id in task_registry:
        task_registry[task_id]["status"] = status
        task_registry[task_id]["progress"] = progress
        
        if current_step:
            task_registry[task_id]["current_step"] = current_step
        
        if error_message:
            task_registry[task_id]["error_message"] = error_message

async def process_batch_merge_job(job_id: str, request: BatchMergeRequest):
    """处理批量合并任务"""
    try:
        logger.info(f"Processing batch merge job: {job_id}")
        
        for i, content_group in enumerate(request.content_groups):
            logger.info(f"Processing group {i+1}/{len(request.content_groups)}")
            
            # 为每个组创建单独的合并请求
            group_request = MergeRequest(
                source_content_ids=content_group,
                strategy=request.merge_config.get('strategy', 'topic'),
                mode=request.merge_config.get('mode', 'comprehensive'),
                config=request.merge_config,
                user_id=request.user_id
            )
            
            # 获取内容并执行合并
            async with storage_client:
                contents = await storage_client.get_contents_by_ids(content_group)
            
            # 使用临时请求对象进行合并
            temp_request = MergeRequest(
                source_contents=contents,
                strategy=group_request.strategy,
                mode=group_request.mode,
                config=group_request.config
            )
            
            result = await merger_engine.merge_contents(temp_request)
            
            # 保存组结果
            async with storage_client:
                await storage_client.save_merge_result(f"{job_id}_group_{i}", result)
        
        # 更新批量任务状态
        async with storage_client:
            await storage_client.update_merge_job_progress(job_id, completed=True)
        
        logger.info(f"Batch merge job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Batch merge job {job_id} failed: {str(e)}")
        
        async with storage_client:
            await storage_client.update_merge_job_progress(job_id, failed=True, error=str(e))