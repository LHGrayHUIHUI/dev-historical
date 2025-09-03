"""
爬虫管理API接口
提供爬虫任务的创建、管理和监控接口
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from ..crawler.crawler_manager import (
    get_crawler_manager,
    CrawlerManager,
    CrawlerConfig,
    CrawlerTask,
    TaskPriority,
    CrawlerStatus
)
from ..models.content import ContentSource
from ..config.settings import get_settings

# 创建路由器
router = APIRouter(prefix="/crawlers", tags=["爬虫管理"])


class CreateCrawlerRequest(BaseModel):
    """创建爬虫任务请求模型"""
    platform: ContentSource = Field(..., description="平台类型")
    keywords: List[str] = Field(..., description="关键词列表", min_items=1, max_items=20)
    max_pages: int = Field(default=10, description="最大抓取页数", ge=1, le=100)
    interval: float = Field(default=5.0, description="请求间隔(秒)", ge=0.1, le=60.0)
    timeout: int = Field(default=300, description="超时时间(秒)", ge=30, le=3600)
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="任务优先级")
    proxy_enabled: bool = Field(default=True, description="是否启用代理")
    retry_attempts: int = Field(default=3, description="重试次数", ge=1, le=10)
    custom_settings: Optional[Dict[str, Any]] = Field(default=None, description="自定义配置")
    
    class Config:
        json_schema_extra = {
            "example": {
                "platform": "toutiao",
                "keywords": ["历史", "文化"],
                "max_pages": 10,
                "interval": 5.0,
                "timeout": 300,
                "priority": 5,
                "proxy_enabled": True,
                "retry_attempts": 3,
                "custom_settings": {}
            }
        }


class CrawlerTaskResponse(BaseModel):
    """爬虫任务响应模型"""
    task_id: str
    platform: ContentSource
    keywords: List[str]
    status: CrawlerStatus
    progress: float
    total_items: int
    success_items: int
    failed_items: int
    success_rate: float
    current_page: int
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    
    @classmethod
    def from_crawler_task(cls, task: CrawlerTask) -> "CrawlerTaskResponse":
        """从CrawlerTask对象创建响应模型"""
        duration_seconds = None
        if task.duration:
            duration_seconds = task.duration.total_seconds()
        
        return cls(
            task_id=task.task_id,
            platform=task.config.platform,
            keywords=task.config.keywords,
            status=task.status,
            progress=task.progress,
            total_items=task.total_items,
            success_items=task.success_items,
            failed_items=task.failed_items,
            success_rate=task.success_rate,
            current_page=task.current_page,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            duration_seconds=duration_seconds,
            error_message=task.error_message
        )


class CrawlerStatisticsResponse(BaseModel):
    """爬虫统计响应模型"""
    total_tasks: int
    running_tasks: int
    finished_tasks: int
    error_tasks: int
    total_success_items: int
    total_failed_items: int
    overall_success_rate: float


@router.post("/", response_model=Dict[str, Any], summary="创建爬虫任务")
async def create_crawler_task(
    request: CreateCrawlerRequest,
    background_tasks: BackgroundTasks,
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    创建新的爬虫任务
    
    - **platform**: 平台类型 (toutiao, baijiahao, xiaohongshu)
    - **keywords**: 关键词列表，用于内容抓取
    - **max_pages**: 最大抓取页数
    - **interval**: 请求间隔时间，避免过于频繁请求
    - **priority**: 任务优先级，影响执行顺序
    """
    try:
        # 创建爬虫配置
        config = CrawlerConfig(
            platform=request.platform,
            keywords=request.keywords,
            max_pages=request.max_pages,
            interval=request.interval,
            timeout=request.timeout,
            priority=request.priority,
            proxy_enabled=request.proxy_enabled,
            retry_attempts=request.retry_attempts,
            custom_settings=request.custom_settings or {}
        )
        
        # 创建任务
        task_id = await crawler_manager.create_task(config)
        
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "platform": request.platform,
                "keywords": request.keywords,
                "status": CrawlerStatus.IDLE
            },
            "message": "爬虫任务创建成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建爬虫任务失败: {str(e)}")


@router.post("/{task_id}/start", response_model=Dict[str, Any], summary="启动爬虫任务")
async def start_crawler_task(
    task_id: str = Path(..., description="任务ID"),
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    启动指定的爬虫任务
    
    - **task_id**: 爬虫任务的唯一标识符
    """
    try:
        success = await crawler_manager.start_task(task_id)
        
        if not success:
            task = crawler_manager.get_task_status(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="爬虫任务不存在")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"无法启动任务，当前状态: {task.status}"
                )
        
        return {
            "success": True,
            "message": "爬虫任务启动成功",
            "data": {"task_id": task_id}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动爬虫任务失败: {str(e)}")


@router.post("/{task_id}/stop", response_model=Dict[str, Any], summary="停止爬虫任务")
async def stop_crawler_task(
    task_id: str = Path(..., description="任务ID"),
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    停止指定的爬虫任务
    """
    try:
        success = await crawler_manager.stop_task(task_id)
        
        if not success:
            task = crawler_manager.get_task_status(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="爬虫任务不存在")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"无法停止任务，当前状态: {task.status}"
                )
        
        return {
            "success": True,
            "message": "爬虫任务停止成功",
            "data": {"task_id": task_id}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止爬虫任务失败: {str(e)}")


@router.post("/{task_id}/pause", response_model=Dict[str, Any], summary="暂停爬虫任务")
async def pause_crawler_task(
    task_id: str = Path(..., description="任务ID"),
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    暂停指定的爬虫任务
    """
    try:
        success = await crawler_manager.pause_task(task_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="无法暂停任务")
        
        return {
            "success": True,
            "message": "爬虫任务暂停成功",
            "data": {"task_id": task_id}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"暂停爬虫任务失败: {str(e)}")


@router.post("/{task_id}/resume", response_model=Dict[str, Any], summary="恢复爬虫任务")
async def resume_crawler_task(
    task_id: str = Path(..., description="任务ID"),
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    恢复暂停的爬虫任务
    """
    try:
        success = await crawler_manager.resume_task(task_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="无法恢复任务")
        
        return {
            "success": True,
            "message": "爬虫任务恢复成功",
            "data": {"task_id": task_id}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复爬虫任务失败: {str(e)}")


@router.get("/{task_id}/status", response_model=Dict[str, Any], summary="获取任务状态")
async def get_crawler_task_status(
    task_id: str = Path(..., description="任务ID"),
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    获取爬虫任务的详细状态信息
    """
    try:
        task = crawler_manager.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="爬虫任务不存在")
        
        task_response = CrawlerTaskResponse.from_crawler_task(task)
        
        return {
            "success": True,
            "data": task_response.dict(),
            "message": "获取任务状态成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.get("/", response_model=Dict[str, Any], summary="获取所有任务")
async def get_all_crawler_tasks(
    status: Optional[CrawlerStatus] = Query(None, description="按状态过滤"),
    platform: Optional[ContentSource] = Query(None, description="按平台过滤"),
    page: int = Query(1, description="页码", ge=1),
    size: int = Query(20, description="每页数量", ge=1, le=100),
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    获取所有爬虫任务列表，支持分页和过滤
    """
    try:
        all_tasks = crawler_manager.get_all_tasks()
        
        # 过滤任务
        filtered_tasks = all_tasks
        if status:
            filtered_tasks = [task for task in filtered_tasks if task.status == status]
        if platform:
            filtered_tasks = [task for task in filtered_tasks if task.config.platform == platform]
        
        # 排序（按创建时间倒序）
        filtered_tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        # 分页
        total = len(filtered_tasks)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        page_tasks = filtered_tasks[start_idx:end_idx]
        
        # 转换为响应模型
        task_responses = [CrawlerTaskResponse.from_crawler_task(task) for task in page_tasks]
        
        return {
            "success": True,
            "data": {
                "items": [task.dict() for task in task_responses],
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            },
            "message": "获取任务列表成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.get("/running", response_model=Dict[str, Any], summary="获取运行中任务")
async def get_running_crawler_tasks(
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    获取当前运行中的所有爬虫任务
    """
    try:
        running_tasks = crawler_manager.get_running_tasks()
        task_responses = [CrawlerTaskResponse.from_crawler_task(task) for task in running_tasks]
        
        return {
            "success": True,
            "data": {
                "items": [task.dict() for task in task_responses],
                "count": len(running_tasks)
            },
            "message": "获取运行中任务成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取运行中任务失败: {str(e)}")


@router.get("/statistics", response_model=Dict[str, Any], summary="获取统计信息")
async def get_crawler_statistics(
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    获取爬虫系统的统计信息
    """
    try:
        stats = await crawler_manager.get_statistics()
        
        return {
            "success": True,
            "data": stats,
            "message": "获取统计信息成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.delete("/{task_id}", response_model=Dict[str, Any], summary="删除任务")
async def delete_crawler_task(
    task_id: str = Path(..., description="任务ID"),
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    删除指定的爬虫任务（仅限已完成或已停止的任务）
    """
    try:
        task = crawler_manager.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="爬虫任务不存在")
        
        # 只允许删除已完成、已停止或错误状态的任务
        if task.status in [CrawlerStatus.RUNNING, CrawlerStatus.PAUSED]:
            raise HTTPException(
                status_code=400,
                detail="无法删除运行中或暂停的任务，请先停止任务"
            )
        
        # 从内存中删除任务
        if task_id in crawler_manager.tasks:
            del crawler_manager.tasks[task_id]
        
        # TODO: 从数据库中删除任务记录
        
        return {
            "success": True,
            "message": "任务删除成功",
            "data": {"task_id": task_id}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


@router.post("/stop-all", response_model=Dict[str, Any], summary="停止所有任务")
async def stop_all_crawler_tasks(
    crawler_manager: CrawlerManager = Depends(get_crawler_manager)
):
    """
    紧急停止所有正在运行的爬虫任务
    """
    try:
        await crawler_manager.stop_all_crawlers()
        
        return {
            "success": True,
            "message": "所有爬虫任务已停止"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止所有任务失败: {str(e)}")