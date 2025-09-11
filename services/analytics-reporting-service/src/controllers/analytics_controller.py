"""
数据分析控制器

提供数据分析相关的API接口：
- 内容表现分析
- 平台对比分析  
- 趋势分析
- 用户行为分析
- 异常检测
- 智能洞察
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models import (
    get_db, BaseResponse, PaginationParams, PaginatedResponse,
    AnalysisTaskCreate, AnalysisTaskUpdate, AnalysisTaskResponse, AnalysisTaskList,
    AnalyticsMetrics, ContentPerformance, PlatformComparison,
    TrendAnalysis, UserBehaviorInsights, TimeSeriesPoint,
    MetricQuery, AggregationResult, VisualizationRequest
)
from ..services.analytics_service import AnalyticsService
from ..services.data_processor import DataProcessor
from ..config.settings import settings

logger = logging.getLogger(__name__)

# 创建路由器
analytics_router = APIRouter(
    prefix="/analytics",
    tags=["数据分析"],
    responses={
        404: {"description": "资源未找到"},
        500: {"description": "服务器内部错误"}
    }
)

# 全局服务实例
analytics_service = AnalyticsService()
data_processor = DataProcessor()


@analytics_router.on_event("startup")
async def startup():
    """启动时初始化服务"""
    await analytics_service.initialize()
    await data_processor.initialize()


# ===== 分析任务管理 =====

@analytics_router.post(
    "/tasks",
    response_model=BaseResponse,
    summary="创建分析任务",
    description="创建新的数据分析任务"
)
async def create_analysis_task(
    task_data: AnalysisTaskCreate,
    user_id: str = Query(..., description="用户ID"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """创建分析任务"""
    try:
        logger.info(f"创建分析任务 - 用户: {user_id}, 类型: {task_data.task_type}")
        
        async with get_db() as db:
            from ..models import AnalysisTask, AnalysisTaskStatus
            
            # 创建任务记录
            task = AnalysisTask(
                title=task_data.title,
                description=task_data.description,
                task_type=task_data.task_type,
                status=AnalysisTaskStatus.PENDING,
                user_id=user_id,
                config=task_data.config,
                parameters=task_data.parameters,
                filters=task_data.filters,
                start_date=task_data.start_date,
                end_date=task_data.end_date,
                scheduled_at=task_data.scheduled_at,
                priority=task_data.priority
            )
            
            db.add(task)
            await db.commit()
            await db.refresh(task)
            
            # 后台执行分析任务
            background_tasks.add_task(execute_analysis_task, task.id, user_id)
            
            return BaseResponse(
                message=f"分析任务创建成功，任务ID: {task.id}",
                data={"task_id": str(task.id), "status": task.status.value}
            )
    
    except Exception as e:
        logger.error(f"创建分析任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建分析任务失败: {str(e)}")


@analytics_router.get(
    "/tasks",
    response_model=AnalysisTaskList,
    summary="获取分析任务列表",
    description="获取用户的分析任务列表"
)
async def get_analysis_tasks(
    user_id: str = Query(..., description="用户ID"),
    pagination: PaginationParams = Depends(),
    task_type: Optional[str] = Query(None, description="任务类型过滤"),
    status: Optional[str] = Query(None, description="状态过滤")
):
    """获取分析任务列表"""
    try:
        async with get_db() as db:
            from sqlalchemy import text
            
            # 构建查询条件
            where_conditions = ["user_id = :user_id"]
            params = {"user_id": user_id}
            
            if task_type:
                where_conditions.append("task_type = :task_type")
                params["task_type"] = task_type
                
            if status:
                where_conditions.append("status = :status")
                params["status"] = status
            
            # 获取总数
            count_query = f"""
                SELECT COUNT(*) FROM analysis_tasks 
                WHERE {' AND '.join(where_conditions)} AND deleted_at IS NULL
            """
            total_result = await db.execute(text(count_query), params)
            total = total_result.scalar()
            
            # 获取分页数据
            offset = (pagination.page - 1) * pagination.page_size
            data_query = f"""
                SELECT * FROM analysis_tasks 
                WHERE {' AND '.join(where_conditions)} AND deleted_at IS NULL
                ORDER BY created_at DESC 
                LIMIT :limit OFFSET :offset
            """
            params.update({"limit": pagination.page_size, "offset": offset})
            
            result = await db.execute(text(data_query), params)
            tasks = result.fetchall()
            
            # 转换为响应格式
            task_list = []
            for task in tasks:
                task_response = AnalysisTaskResponse.from_orm(task)
                task_list.append(task_response)
            
            total_pages = (total + pagination.page_size - 1) // pagination.page_size
            
            return AnalysisTaskList(
                success=True,
                message="获取任务列表成功",
                data=task_list,
                total=total,
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=total_pages
            )
    
    except Exception as e:
        logger.error(f"获取分析任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@analytics_router.get(
    "/tasks/{task_id}",
    response_model=BaseResponse,
    summary="获取分析任务详情",
    description="获取指定分析任务的详细信息"
)
async def get_analysis_task(
    task_id: UUID,
    user_id: str = Query(..., description="用户ID")
):
    """获取分析任务详情"""
    try:
        async with get_db() as db:
            from sqlalchemy import text
            
            query = """
                SELECT * FROM analysis_tasks 
                WHERE id = :task_id AND user_id = :user_id AND deleted_at IS NULL
            """
            result = await db.execute(text(query), {"task_id": str(task_id), "user_id": user_id})
            task = result.fetchone()
            
            if not task:
                raise HTTPException(status_code=404, detail="分析任务不存在")
            
            task_response = AnalysisTaskResponse.from_orm(task)
            
            return BaseResponse(
                message="获取任务详情成功",
                data=task_response.dict()
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取分析任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


@analytics_router.put(
    "/tasks/{task_id}",
    response_model=BaseResponse,
    summary="更新分析任务",
    description="更新指定分析任务的信息"
)
async def update_analysis_task(
    task_id: UUID,
    task_update: AnalysisTaskUpdate,
    user_id: str = Query(..., description="用户ID")
):
    """更新分析任务"""
    try:
        async with get_db() as db:
            from sqlalchemy import text
            
            # 检查任务是否存在
            check_query = """
                SELECT id FROM analysis_tasks 
                WHERE id = :task_id AND user_id = :user_id AND deleted_at IS NULL
            """
            result = await db.execute(text(check_query), {"task_id": str(task_id), "user_id": user_id})
            if not result.fetchone():
                raise HTTPException(status_code=404, detail="分析任务不存在")
            
            # 构建更新字段
            update_fields = []
            params = {"task_id": str(task_id), "user_id": user_id}
            
            if task_update.title is not None:
                update_fields.append("title = :title")
                params["title"] = task_update.title
                
            if task_update.description is not None:
                update_fields.append("description = :description")
                params["description"] = task_update.description
                
            if task_update.status is not None:
                update_fields.append("status = :status")
                params["status"] = task_update.status.value
                
            if task_update.config is not None:
                update_fields.append("config = :config")
                params["config"] = task_update.config
                
            if task_update.parameters is not None:
                update_fields.append("parameters = :parameters")
                params["parameters"] = task_update.parameters
                
            if task_update.filters is not None:
                update_fields.append("filters = :filters")
                params["filters"] = task_update.filters
                
            if task_update.scheduled_at is not None:
                update_fields.append("scheduled_at = :scheduled_at")
                params["scheduled_at"] = task_update.scheduled_at
                
            if task_update.priority is not None:
                update_fields.append("priority = :priority")
                params["priority"] = task_update.priority
            
            # 添加更新时间
            update_fields.append("updated_at = NOW()")
            
            if not update_fields:
                return BaseResponse(message="没有需要更新的字段")
            
            # 执行更新
            update_query = f"""
                UPDATE analysis_tasks 
                SET {', '.join(update_fields)}
                WHERE id = :task_id AND user_id = :user_id
            """
            await db.execute(text(update_query), params)
            await db.commit()
            
            return BaseResponse(message="分析任务更新成功")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新分析任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新任务失败: {str(e)}")


@analytics_router.delete(
    "/tasks/{task_id}",
    response_model=BaseResponse,
    summary="删除分析任务",
    description="软删除指定的分析任务"
)
async def delete_analysis_task(
    task_id: UUID,
    user_id: str = Query(..., description="用户ID")
):
    """删除分析任务"""
    try:
        async with get_db() as db:
            from sqlalchemy import text
            
            # 软删除任务
            delete_query = """
                UPDATE analysis_tasks 
                SET deleted_at = NOW(), updated_at = NOW()
                WHERE id = :task_id AND user_id = :user_id AND deleted_at IS NULL
            """
            result = await db.execute(text(delete_query), {"task_id": str(task_id), "user_id": user_id})
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="分析任务不存在")
            
            await db.commit()
            
            return BaseResponse(message="分析任务删除成功")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除分析任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


# ===== 数据分析接口 =====

@analytics_router.get(
    "/content-performance",
    response_model=BaseResponse,
    summary="内容表现分析",
    description="分析内容的表现指标，包括浏览量、互动数等"
)
async def analyze_content_performance(
    user_id: str = Query(..., description="用户ID"),
    content_ids: Optional[str] = Query(None, description="内容ID列表，逗号分隔"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    platforms: Optional[str] = Query(None, description="平台列表，逗号分隔")
):
    """内容表现分析"""
    try:
        logger.info(f"执行内容表现分析 - 用户: {user_id}")
        
        # 解析参数
        content_id_list = content_ids.split(',') if content_ids else None
        platform_list = platforms.split(',') if platforms else None
        
        # 执行分析
        performance_data = await analytics_service.analyze_content_performance(
            user_id=user_id,
            content_ids=content_id_list,
            start_date=start_date,
            end_date=end_date,
            platforms=platform_list
        )
        
        return BaseResponse(
            message=f"内容表现分析完成，共分析 {len(performance_data)} 个内容",
            data=[perf.dict() for perf in performance_data]
        )
    
    except Exception as e:
        logger.error(f"内容表现分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"内容表现分析失败: {str(e)}")


@analytics_router.get(
    "/platform-comparison",
    response_model=BaseResponse,
    summary="平台对比分析",
    description="对比分析不同平台的表现指标"
)
async def analyze_platform_comparison(
    user_id: str = Query(..., description="用户ID"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    metrics: Optional[str] = Query(None, description="对比指标，逗号分隔")
):
    """平台对比分析"""
    try:
        logger.info(f"执行平台对比分析 - 用户: {user_id}")
        
        # 解析指标参数
        metric_list = metrics.split(',') if metrics else None
        
        # 执行分析
        comparison_data = await analytics_service.analyze_platform_comparison(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            metrics=metric_list
        )
        
        return BaseResponse(
            message=f"平台对比分析完成，共分析 {len(comparison_data)} 个平台",
            data=[comp.dict() for comp in comparison_data]
        )
    
    except Exception as e:
        logger.error(f"平台对比分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"平台对比分析失败: {str(e)}")


@analytics_router.get(
    "/trends",
    response_model=BaseResponse,
    summary="趋势分析",
    description="分析数据指标的变化趋势和预测"
)
async def analyze_trends(
    user_id: str = Query(..., description="用户ID"),
    metric_names: str = Query(..., description="指标名称列表，逗号分隔"),
    time_period: str = Query("daily", description="时间周期：daily, weekly, monthly"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    include_forecast: bool = Query(True, description="是否包含预测")
):
    """趋势分析"""
    try:
        logger.info(f"执行趋势分析 - 用户: {user_id}, 指标: {metric_names}")
        
        # 解析指标名称
        metric_name_list = metric_names.split(',')
        
        # 验证时间周期
        if time_period not in ['daily', 'weekly', 'monthly']:
            raise HTTPException(status_code=400, detail="时间周期必须是 daily, weekly 或 monthly")
        
        # 执行分析
        trend_data = await analytics_service.analyze_trends(
            user_id=user_id,
            metric_names=metric_name_list,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date,
            include_forecast=include_forecast
        )
        
        return BaseResponse(
            message=f"趋势分析完成，共分析 {len(trend_data)} 个指标",
            data=[trend.dict() for trend in trend_data]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"趋势分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"趋势分析失败: {str(e)}")


@analytics_router.get(
    "/user-behavior",
    response_model=BaseResponse,
    summary="用户行为分析",
    description="分析用户行为模式和洞察"
)
async def analyze_user_behavior(
    user_id: str = Query(..., description="用户ID"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期")
):
    """用户行为分析"""
    try:
        logger.info(f"执行用户行为分析 - 用户: {user_id}")
        
        # 执行分析
        behavior_insights = await analytics_service.analyze_user_behavior(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return BaseResponse(
            message="用户行为分析完成",
            data=behavior_insights.dict()
        )
    
    except Exception as e:
        logger.error(f"用户行为分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"用户行为分析失败: {str(e)}")


@analytics_router.get(
    "/anomalies",
    response_model=BaseResponse,
    summary="异常检测",
    description="检测数据中的异常点"
)
async def detect_anomalies(
    user_id: str = Query(..., description="用户ID"),
    metric_name: str = Query(..., description="指标名称"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    threshold: Optional[float] = Query(None, description="异常阈值")
):
    """异常检测"""
    try:
        logger.info(f"执行异常检测 - 用户: {user_id}, 指标: {metric_name}")
        
        # 执行异常检测
        anomalies = await analytics_service.detect_anomalies(
            user_id=user_id,
            metric_name=metric_name,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold
        )
        
        return BaseResponse(
            message=f"异常检测完成，发现 {len(anomalies)} 个异常点",
            data=anomalies
        )
    
    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"异常检测失败: {str(e)}")


# ===== 数据收集和处理 =====

@analytics_router.post(
    "/data-collection",
    response_model=BaseResponse,
    summary="触发数据收集",
    description="手动触发平台数据收集"
)
async def trigger_data_collection(
    user_id: str = Query(..., description="用户ID"),
    platforms: str = Query(..., description="平台列表，逗号分隔"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """触发数据收集"""
    try:
        logger.info(f"触发数据收集 - 用户: {user_id}, 平台: {platforms}")
        
        platform_list = platforms.split(',')
        
        # 后台执行数据收集
        background_tasks.add_task(
            collect_platform_data_task,
            user_id,
            platform_list,
            start_date,
            end_date
        )
        
        return BaseResponse(
            message=f"数据收集任务已启动，收集平台: {', '.join(platform_list)}"
        )
    
    except Exception as e:
        logger.error(f"触发数据收集失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发数据收集失败: {str(e)}")


@analytics_router.post(
    "/data-aggregation",
    response_model=BaseResponse,
    summary="触发数据聚合",
    description="手动触发数据聚合处理"
)
async def trigger_data_aggregation(
    user_id: str = Query(..., description="用户ID"),
    time_period: str = Query("daily", description="聚合周期"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """触发数据聚合"""
    try:
        logger.info(f"触发数据聚合 - 用户: {user_id}, 周期: {time_period}")
        
        # 验证时间周期
        if time_period not in ['hourly', 'daily', 'weekly', 'monthly']:
            raise HTTPException(status_code=400, detail="时间周期必须是 hourly, daily, weekly 或 monthly")
        
        # 后台执行数据聚合
        background_tasks.add_task(
            aggregate_data_task,
            user_id,
            time_period,
            start_date,
            end_date
        )
        
        return BaseResponse(
            message=f"数据聚合任务已启动，聚合周期: {time_period}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"触发数据聚合失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发数据聚合失败: {str(e)}")


# ===== 后台任务函数 =====

async def execute_analysis_task(task_id: UUID, user_id: str):
    """后台执行分析任务"""
    try:
        logger.info(f"开始执行分析任务: {task_id}")
        
        # 更新任务状态为运行中
        async with get_db() as db:
            from sqlalchemy import text
            
            update_query = """
                UPDATE analysis_tasks 
                SET status = 'running', started_at = NOW(), updated_at = NOW()
                WHERE id = :task_id
            """
            await db.execute(text(update_query), {"task_id": str(task_id)})
            await db.commit()
        
        # 获取任务详情
        async with get_db() as db:
            query = "SELECT * FROM analysis_tasks WHERE id = :task_id"
            result = await db.execute(text(query), {"task_id": str(task_id)})
            task = result.fetchone()
        
        if not task:
            logger.error(f"分析任务不存在: {task_id}")
            return
        
        # 根据任务类型执行相应的分析
        result_data = {}
        
        if task.task_type == 'content_performance':
            performance_data = await analytics_service.analyze_content_performance(
                user_id=user_id,
                start_date=task.start_date,
                end_date=task.end_date
            )
            result_data['content_performance'] = [perf.dict() for perf in performance_data]
            
        elif task.task_type == 'platform_comparison':
            comparison_data = await analytics_service.analyze_platform_comparison(
                user_id=user_id,
                start_date=task.start_date,
                end_date=task.end_date
            )
            result_data['platform_data'] = [comp.dict() for comp in comparison_data]
            
        elif task.task_type == 'trend_analysis':
            # 从任务参数中获取指标名称
            metric_names = task.parameters.get('metric_names', ['views', 'likes']) if task.parameters else ['views', 'likes']
            trend_data = await analytics_service.analyze_trends(
                user_id=user_id,
                metric_names=metric_names,
                start_date=task.start_date,
                end_date=task.end_date
            )
            result_data['trend_data'] = [trend.dict() for trend in trend_data]
            
        elif task.task_type == 'user_behavior':
            behavior_data = await analytics_service.analyze_user_behavior(
                user_id=user_id,
                start_date=task.start_date,
                end_date=task.end_date
            )
            result_data['user_behavior'] = behavior_data.dict()
        
        # 更新任务状态为完成
        async with get_db() as db:
            complete_query = """
                UPDATE analysis_tasks 
                SET status = 'completed', completed_at = NOW(), 
                    result_data = :result_data, progress = 100, updated_at = NOW()
                WHERE id = :task_id
            """
            await db.execute(text(complete_query), {
                "task_id": str(task_id),
                "result_data": result_data
            })
            await db.commit()
        
        logger.info(f"分析任务执行完成: {task_id}")
        
    except Exception as e:
        logger.error(f"执行分析任务失败: {task_id}, 错误: {e}")
        
        # 更新任务状态为失败
        async with get_db() as db:
            fail_query = """
                UPDATE analysis_tasks 
                SET status = 'failed', error_message = :error_message, updated_at = NOW()
                WHERE id = :task_id
            """
            await db.execute(text(fail_query), {
                "task_id": str(task_id),
                "error_message": str(e)
            })
            await db.commit()


async def collect_platform_data_task(
    user_id: str,
    platforms: List[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime]
):
    """后台数据收集任务"""
    try:
        logger.info(f"开始收集平台数据 - 用户: {user_id}, 平台: {platforms}")
        
        collection_stats = await data_processor.collect_platform_data(
            user_id=user_id,
            platforms=platforms,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"平台数据收集完成 - 统计: {collection_stats}")
        
    except Exception as e:
        logger.error(f"平台数据收集失败: {e}")


async def aggregate_data_task(
    user_id: str,
    time_period: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime]
):
    """后台数据聚合任务"""
    try:
        logger.info(f"开始数据聚合 - 用户: {user_id}, 周期: {time_period}")
        
        aggregation_stats = await data_processor.aggregate_data(
            user_id=user_id,
            time_period=time_period,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"数据聚合完成 - 统计: {aggregation_stats}")
        
    except Exception as e:
        logger.error(f"数据聚合失败: {e}")


# ===== 数据质量监控 =====

@analytics_router.get(
    "/data-quality",
    response_model=BaseResponse,
    summary="数据质量报告",
    description="获取数据质量监控报告"
)
async def get_data_quality_report(
    user_id: str = Query(..., description="用户ID")
):
    """获取数据质量报告"""
    try:
        logger.info(f"获取数据质量报告 - 用户: {user_id}")
        
        # 执行数据质量监控
        quality_report = await data_processor.monitor_data_quality(user_id)
        
        return BaseResponse(
            message="数据质量报告生成成功",
            data=quality_report
        )
    
    except Exception as e:
        logger.error(f"获取数据质量报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据质量报告失败: {str(e)}")


# ===== 实时指标接口 =====

@analytics_router.get(
    "/realtime-metrics",
    response_model=BaseResponse,
    summary="实时指标",
    description="获取实时数据指标"
)
async def get_realtime_metrics(
    user_id: str = Query(..., description="用户ID"),
    metric_names: Optional[str] = Query(None, description="指标名称列表，逗号分隔")
):
    """获取实时指标"""
    try:
        logger.info(f"获取实时指标 - 用户: {user_id}")
        
        from ..models import get_redis
        import json
        
        redis_client = await get_redis()
        
        # 获取指定的或所有实时指标
        if metric_names:
            metric_list = metric_names.split(',')
        else:
            # 获取所有实时指标的键
            keys = await redis_client.keys("real_time:*")
            metric_list = [key.split(':', 1)[1] for key in keys]
        
        realtime_data = {}
        for metric_name in metric_list:
            cache_key = f"real_time:{metric_name}"
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                realtime_data[metric_name] = json.loads(cached_data)
        
        return BaseResponse(
            message=f"获取实时指标成功，共 {len(realtime_data)} 个指标",
            data=realtime_data
        )
    
    except Exception as e:
        logger.error(f"获取实时指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取实时指标失败: {str(e)}")


# ===== 数据导出接口 =====

@analytics_router.get(
    "/export",
    response_model=BaseResponse,
    summary="数据导出",
    description="导出分析数据"
)
async def export_analytics_data(
    user_id: str = Query(..., description="用户ID"),
    data_type: str = Query(..., description="数据类型"),
    format: str = Query("csv", description="导出格式: csv, json, excel"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期")
):
    """导出分析数据"""
    try:
        logger.info(f"导出分析数据 - 用户: {user_id}, 类型: {data_type}, 格式: {format}")
        
        # 验证格式
        if format not in ['csv', 'json', 'excel']:
            raise HTTPException(status_code=400, detail="导出格式必须是 csv, json 或 excel")
        
        # 根据数据类型获取数据
        if data_type == 'content_performance':
            data = await analytics_service.analyze_content_performance(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            export_data = [item.dict() for item in data]
            
        elif data_type == 'platform_comparison':
            data = await analytics_service.analyze_platform_comparison(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            export_data = [item.dict() for item in data]
            
        else:
            raise HTTPException(status_code=400, detail="不支持的数据类型")
        
        # 生成导出文件
        import tempfile
        import pandas as pd
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            df = pd.DataFrame(export_data)
            filename = f"{data_type}_{timestamp}.csv"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
        elif format == 'excel':
            df = pd.DataFrame(export_data)
            filename = f"{data_type}_{timestamp}.xlsx"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            df.to_excel(filepath, index=False)
            
        elif format == 'json':
            filename = f"{data_type}_{timestamp}.json"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        return BaseResponse(
            message=f"数据导出成功，格式: {format}",
            data={
                "filename": filename,
                "filepath": filepath,
                "record_count": len(export_data)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出分析数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"导出数据失败: {str(e)}")