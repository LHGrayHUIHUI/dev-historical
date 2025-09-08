"""
图像处理控制器
FastAPI路由处理器，处理图像处理相关的HTTP请求
无状态架构，数据存储通过storage-service完成
"""

import asyncio
import time
import uuid
import tempfile
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, File, UploadFile, Form, status
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from ..schemas.image_schemas import *
from ..services.image_processing_service import ImageProcessingService
from ..clients.storage_client import StorageServiceClient, storage_client
from ..config.settings import settings


# 创建路由器
router = APIRouter(prefix="/api/v1/image-processing", tags=["图像处理"])

# 图像处理服务实例（全局单例）
image_service: Optional[ImageProcessingService] = None


async def get_image_service() -> ImageProcessingService:
    """获取图像处理服务实例"""
    global image_service
    if image_service is None:
        image_service = ImageProcessingService()
        await image_service.initialize()
    return image_service


async def get_storage_client() -> StorageServiceClient:
    """获取storage客户端实例"""
    return storage_client


# ============ 图像处理接口 ============

@router.post("/process", response_model=ImageProcessingResponse)
async def process_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="待处理的图像文件"),
    processing_type: ProcessingType = Form(..., description="处理类型"),
    config: str = Form(default="{}", description="处理配置JSON字符串"),
    engine: Optional[ProcessingEngine] = Form(default=None, description="指定处理引擎"),
    async_mode: bool = Form(default=False, description="是否异步处理"),
    dataset_id: Optional[str] = Form(default=None, description="关联数据集ID"),
    priority: int = Form(default=5, description="任务优先级"),
    service: ImageProcessingService = Depends(get_image_service),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """单图像处理接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 验证文件类型
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 验证文件大小
        if image.size and image.size > settings.max_image_size:
            max_size_mb = settings.max_image_size / (1024 * 1024)
            raise HTTPException(status_code=400, detail=f"文件大小不能超过{max_size_mb:.0f}MB")
        
        # 解析配置
        try:
            import json
            processing_config = ProcessingConfig.parse_obj(json.loads(config))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="配置格式错误")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"配置验证失败: {str(e)}")
        
        # 读取图像数据
        image_data = await image.read()
        
        # 如果是异步模式，创建任务记录并后台处理
        if async_mode:
            await storage.create_image_processing_task(
                dataset_id=dataset_id,
                original_image_path=image.filename or "uploaded_image",
                processing_type=processing_type.value,
                processing_engine=engine.value if engine else settings.default_processing_engine,
                config=processing_config.dict() if processing_config else {},
                priority=priority
            )
            
            # 后台处理
            background_tasks.add_task(
                _process_image_async,
                task_id, image_data, image.filename, processing_type,
                processing_config, engine, dataset_id, service, storage
            )
            
            return ImageProcessingResponse(
                success=True,
                message="图像处理任务已创建，请通过task_id查询结果",
                task_id=task_id,
                status="processing"
            )
        
        # 同步处理
        result = await _process_image_sync(
            image_data, processing_type, processing_config, 
            engine, service
        )
        
        processing_time = time.time() - start_time
        
        # 保存结果到storage-service（如果有dataset_id）
        if dataset_id:
            task_record = await storage.create_image_processing_task(
                dataset_id=dataset_id,
                original_image_path=image.filename or "uploaded_image",
                processing_type=processing_type.value,
                processing_engine=engine.value if engine else settings.default_processing_engine,
                config=processing_config.dict() if processing_config else {}
            )
            
            task_id = task_record.get("task_id", task_id)
            
            # 保存处理结果
            await storage.save_processing_result(
                task_id=task_id,
                original_image_info=result["original_info"],
                processed_image_info=result["processed_info"],
                processed_image_path="processed_" + (image.filename or "image"),
                quality_before=result.get("quality_before"),
                quality_after=result.get("quality_after"),
                processing_metrics=result.get("metrics", {})
            )
            
            await storage.update_task_status(
                task_id=task_id,
                status="completed",
                processing_time=processing_time
            )
        
        return ImageProcessingResponse(
            success=True,
            message="图像处理完成",
            task_id=task_id,
            status="completed",
            result=ProcessingResult(
                task_id=task_id,
                processing_type=processing_type,
                engine=engine.value if engine else settings.default_processing_engine,
                original_image_info=ImageInfo.parse_obj(result["original_info"]),
                processed_image_info=ImageInfo.parse_obj(result["processed_info"]),
                original_image_path=image.filename or "uploaded_image",
                processed_image_path="processed_" + (image.filename or "image"),
                processing_time=processing_time,
                quality_before=QualityMetrics.parse_obj(result["quality_before"]) if result.get("quality_before") else None,
                quality_after=QualityMetrics.parse_obj(result["quality_after"]) if result.get("quality_after") else None,
                config_used=processing_config.dict() if processing_config else {}
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像处理失败: {str(e)}")
        
        # 更新任务状态为失败
        if async_mode and dataset_id:
            try:
                await storage.update_task_status(
                    task_id=task_id,
                    status="failed",
                    error_message=str(e)
                )
            except:
                pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像处理失败: {str(e)}"
        )


@router.post("/batch", response_model=BatchProcessingResponse)
async def batch_process_images(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    service: ImageProcessingService = Depends(get_image_service),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """批量图像处理接口"""
    
    batch_id = str(uuid.uuid4())
    
    try:
        # 验证图像数量
        if len(request.image_paths) > settings.max_batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"单次批量处理不能超过{settings.max_batch_size}张图像"
            )
        
        # 创建批量任务记录
        await storage.create_batch_processing_task(
            batch_id=batch_id,
            image_paths=request.image_paths,
            processing_type=request.processing_type.value,
            processing_engine=request.engine.value if request.engine else settings.default_processing_engine,
            config=request.config.dict() if request.config else {},
            dataset_id=request.dataset_id,
            priority=request.priority
        )
        
        # 后台处理
        background_tasks.add_task(
            _batch_process_images_async,
            batch_id, request, service, storage
        )
        
        return BatchProcessingResponse(
            success=True,
            message=f"批量图像处理任务已创建，共{len(request.image_paths)}张图像",
            batch_id=batch_id,
            status="processing",
            total_images=len(request.image_paths)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量处理失败: {str(e)}"
        )


@router.post("/quality-assessment", response_model=QualityAssessmentResponse)
async def assess_image_quality(
    image: UploadFile = File(..., description="待评估的图像文件"),
    reference_image: Optional[UploadFile] = File(None, description="参考图像文件"),
    service: ImageProcessingService = Depends(get_image_service)
):
    """图像质量评估接口"""
    
    try:
        # 验证文件类型
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        start_time = time.time()
        
        # 读取图像数据
        image_data = await image.read()
        image_array = service.load_image(image_data)
        
        reference_array = None
        if reference_image:
            if not reference_image.content_type or not reference_image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="参考图像文件类型不支持")
            reference_data = await reference_image.read()
            reference_array = service.load_image(reference_data)
        
        # 评估图像质量
        quality_metrics = await service.assess_image_quality(image_array, reference_array)
        
        assessment_time = time.time() - start_time
        
        # 生成改进建议
        recommendations = _generate_quality_recommendations(quality_metrics)
        
        return QualityAssessmentResponse(
            success=True,
            message="图像质量评估完成",
            quality_metrics=quality_metrics,
            assessment_time=assessment_time,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像质量评估失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像质量评估失败: {str(e)}"
        )


# ============ 任务管理接口 ============

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """获取处理任务状态"""
    
    try:
        task_data = await storage.get_task(task_id)
        
        return TaskStatusResponse(
            success=True,
            message="任务信息获取成功",
            task=ImageProcessingTask.parse_obj(task_data)
        )
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {task_id}"
        )


@router.get("/tasks", response_model=TaskListResponse)
async def get_task_list(
    dataset_id: Optional[str] = Query(None, description="数据集ID"),
    processing_type: Optional[ProcessingType] = Query(None, description="处理类型"),
    status: Optional[ProcessingStatus] = Query(None, description="任务状态"),
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="页大小"),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """获取处理任务列表"""
    
    try:
        offset = (page - 1) * size
        
        tasks_data = await storage.get_tasks(
            dataset_id=dataset_id,
            processing_type=processing_type.value if processing_type else None,
            status=status.value if status else None,
            limit=size,
            offset=offset
        )
        
        tasks = [ImageProcessingTask.parse_obj(task) for task in tasks_data.get('tasks', [])]
        
        return TaskListResponse(
            success=True,
            message="任务列表获取成功",
            tasks=tasks,
            total=tasks_data.get('total', 0),
            page=page,
            size=size
        )
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务列表失败: {str(e)}"
        )


@router.get("/batch/{batch_id}", response_model=BatchProcessingResponse)
async def get_batch_status(
    batch_id: str,
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """获取批量处理任务状态"""
    
    try:
        batch_data = await storage.get_batch_task(batch_id)
        
        return BatchProcessingResponse(
            success=True,
            message="批量任务信息获取成功",
            batch_id=batch_id,
            status=batch_data.get("status", "unknown"),
            total_images=batch_data.get("total_images", 0),
            result=BatchProcessingResult.parse_obj(batch_data) if batch_data.get("status") == "completed" else None
        )
        
    except Exception as e:
        logger.error(f"获取批量任务状态失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批量任务不存在: {batch_id}"
        )


# ============ 引擎信息接口 ============

@router.get("/engines", response_model=EnginesResponse)
async def get_available_engines(
    service: ImageProcessingService = Depends(get_image_service)
):
    """获取可用的图像处理引擎列表"""
    
    try:
        engines = service.get_available_engines()
        
        return EnginesResponse(
            success=True,
            message="引擎列表获取成功",
            engines=engines
        )
        
    except Exception as e:
        logger.error(f"获取引擎列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取引擎列表失败: {str(e)}"
        )


# ============ 统计接口 ============

@router.get("/statistics", response_model=StatisticsResponse)
async def get_processing_statistics(
    dataset_id: Optional[str] = Query(None, description="数据集ID"),
    processing_type: Optional[ProcessingType] = Query(None, description="处理类型"),
    engine: Optional[str] = Query(None, description="处理引擎"),
    date_from: Optional[str] = Query(None, description="开始日期"),
    date_to: Optional[str] = Query(None, description="结束日期"),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """获取图像处理统计信息"""
    
    try:
        stats_data = await storage.get_processing_statistics(
            dataset_id=dataset_id,
            processing_type=processing_type.value if processing_type else None,
            engine=engine,
            date_from=date_from,
            date_to=date_to
        )
        
        return StatisticsResponse(
            success=True,
            message="统计信息获取成功",
            statistics=ProcessingStatistics.parse_obj(stats_data)
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


# ============ 工具接口 ============

@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """删除处理任务"""
    
    try:
        await storage.update_task_status(task_id, "cancelled")
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "任务删除成功"}
        )
        
    except Exception as e:
        logger.error(f"删除任务失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除任务失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/retry", response_model=ImageProcessingResponse)
async def retry_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    service: ImageProcessingService = Depends(get_image_service),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """重试失败的处理任务"""
    
    try:
        # 获取原始任务信息
        task_data = await storage.get_task(task_id)
        
        if task_data.get("status") != "failed":
            raise HTTPException(status_code=400, detail="只能重试失败的任务")
        
        # 重新提交任务
        await storage.update_task_status(task_id, "pending")
        
        # 后台重新处理
        background_tasks.add_task(
            _retry_task_async,
            task_id, task_data, service, storage
        )
        
        return ImageProcessingResponse(
            success=True,
            message="任务重试已提交",
            task_id=task_id,
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重试任务失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重试任务失败: {str(e)}"
        )


# ============ 辅助函数 ============

async def _process_image_sync(
    image_data: bytes,
    processing_type: ProcessingType,
    config: Optional[ProcessingConfig],
    engine: Optional[ProcessingEngine],
    service: ImageProcessingService
) -> Dict[str, Any]:
    """同步处理图像"""
    
    # 加载图像
    image = service.load_image(image_data)
    original_info = service.get_image_info(image).dict()
    
    # 处理前质量评估
    quality_before = None
    if settings.enable_quality_assessment:
        quality_before = (await service.assess_image_quality(image)).dict()
    
    # 执行处理
    if processing_type == ProcessingType.ENHANCE and config and config.enhance:
        processed_image = await service.enhance_image(image, config.enhance)
    elif processing_type == ProcessingType.DENOISE and config and config.denoise:
        processed_image = await service.denoise_image(image, config.denoise)
    elif processing_type == ProcessingType.DESKEW and config and config.deskew:
        processed_image, angle = await service.deskew_image(image, config.deskew)
    elif processing_type == ProcessingType.RESIZE and config and config.resize:
        processed_image = await service.resize_image(image, config.resize)
    elif processing_type == ProcessingType.FORMAT_CONVERT and config and config.format_convert:
        processed_image = await service.convert_format(image, config.format_convert)
    elif processing_type == ProcessingType.AUTO_ENHANCE:
        processed_image = await service.auto_enhance_image(image)
    elif processing_type == ProcessingType.QUALITY_ASSESSMENT:
        quality_metrics = await service.assess_image_quality(image)
        return {
            "original_info": original_info,
            "processed_info": original_info,
            "quality_before": quality_metrics.dict(),
            "quality_after": None
        }
    else:
        raise ValueError(f"不支持的处理类型: {processing_type}")
    
    processed_info = service.get_image_info(processed_image).dict()
    
    # 处理后质量评估
    quality_after = None
    if settings.enable_quality_assessment:
        quality_after = (await service.assess_image_quality(processed_image)).dict()
    
    return {
        "original_info": original_info,
        "processed_info": processed_info,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "processed_image": processed_image
    }


async def _process_image_async(
    task_id: str,
    image_data: bytes,
    filename: str,
    processing_type: ProcessingType,
    config: Optional[ProcessingConfig],
    engine: Optional[ProcessingEngine],
    dataset_id: Optional[str],
    service: ImageProcessingService,
    storage: StorageServiceClient
):
    """异步处理图像"""
    
    try:
        await storage.update_task_status(task_id, "processing")
        
        start_time = time.time()
        result = await _process_image_sync(image_data, processing_type, config, engine, service)
        processing_time = time.time() - start_time
        
        # 保存处理结果
        await storage.save_processing_result(
            task_id=task_id,
            original_image_info=result["original_info"],
            processed_image_info=result["processed_info"],
            processed_image_path="processed_" + filename,
            quality_before=result.get("quality_before"),
            quality_after=result.get("quality_after")
        )
        
        await storage.update_task_status(
            task_id=task_id,
            status="completed",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"异步图像处理失败: {str(e)}")
        await storage.update_task_status(
            task_id=task_id,
            status="failed",
            error_message=str(e)
        )


async def _batch_process_images_async(
    batch_id: str,
    request: BatchProcessingRequest,
    service: ImageProcessingService,
    storage: StorageServiceClient
):
    """异步批量处理图像"""
    
    try:
        processed_count = 0
        failed_count = 0
        results = []
        
        # 处理每张图像
        for image_path in request.image_paths:
            try:
                # 下载图像
                image_data = await storage.download_image(image_path)
                
                # 处理图像
                result = await _process_image_sync(
                    image_data, request.processing_type, request.config, request.engine, service
                )
                
                processed_count += 1
                results.append({
                    "image_path": image_path,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"批量处理图像失败 {image_path}: {str(e)}")
                failed_count += 1
                results.append({
                    "image_path": image_path,
                    "success": False,
                    "error": str(e)
                })
            
            # 更新进度
            await storage.update_batch_task_progress(
                batch_id=batch_id,
                processed_count=processed_count,
                failed_count=failed_count
            )
        
        # 完成批量任务
        await storage.update_batch_task_progress(
            batch_id=batch_id,
            processed_count=processed_count,
            failed_count=failed_count,
            status="completed"
        )
        
    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")
        await storage.update_batch_task_progress(
            batch_id=batch_id,
            processed_count=0,
            failed_count=len(request.image_paths),
            status="failed"
        )


async def _retry_task_async(
    task_id: str,
    task_data: Dict[str, Any],
    service: ImageProcessingService,
    storage: StorageServiceClient
):
    """异步重试任务"""
    
    try:
        # 重新处理
        processing_type = ProcessingType(task_data["processing_type"])
        config = ProcessingConfig.parse_obj(task_data.get("config", {}))
        
        # 下载原始图像
        image_data = await storage.download_image(task_data["original_image_path"])
        
        # 处理图像
        result = await _process_image_sync(image_data, processing_type, config, None, service)
        
        # 更新任务状态
        await storage.update_task_status(task_id, "completed")
        
    except Exception as e:
        logger.error(f"重试任务失败: {str(e)}")
        await storage.update_task_status(
            task_id=task_id,
            status="failed",
            error_message=f"重试失败: {str(e)}"
        )


def _generate_quality_recommendations(quality_metrics: QualityMetrics) -> List[str]:
    """根据质量评估生成改进建议"""
    recommendations = []
    
    if quality_metrics.brightness_score < 0.3:
        recommendations.append("图像亮度偏低，建议增强亮度")
    elif quality_metrics.brightness_score > 0.8:
        recommendations.append("图像亮度偏高，建议降低亮度")
    
    if quality_metrics.contrast_score < 0.2:
        recommendations.append("图像对比度偏低，建议增强对比度或使用CLAHE")
    
    if quality_metrics.sharpness_score < 0.5:
        recommendations.append("图像清晰度偏低，建议应用锐化处理")
    
    if quality_metrics.noise_level > 0.3:
        recommendations.append("图像噪声较高，建议使用去噪处理")
    
    if quality_metrics.blur_level > 0.4:
        recommendations.append("图像存在模糊，建议使用锐化或超分辨率处理")
    
    if abs(quality_metrics.skew_angle) > 1.0:
        recommendations.append("图像存在倾斜，建议进行倾斜校正")
    
    if quality_metrics.overall_quality < 0.6:
        recommendations.append("建议使用自动增强功能提升整体质量")
    
    if not recommendations:
        recommendations.append("图像质量良好，无需额外处理")
    
    return recommendations