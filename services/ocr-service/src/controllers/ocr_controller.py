"""
OCR服务API控制器

无状态OCR服务的FastAPI路由控制器。
专注于OCR计算任务，数据存储通过storage-service完成。

Author: OCR开发团队  
Created: 2025-01-15
Version: 2.0.0 (无状态架构)
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import asyncio
import logging
import uuid
from pathlib import Path

from ..services.ocr_service import get_ocr_service, OCRService, OCREngineError, ImageProcessingError
from ..clients.storage_client import (
    StorageServiceClient, OCRTaskRequest, StorageClientError,
    get_storage_client
)
from ..schemas.ocr_schemas import (
    OCRRecognizeRequest, OCRRecognizeResponse,
    OCRBatchRequest, OCRBatchResponse,
    OCRTaskStatusResponse, OCREnginesResponse,
    BaseResponse
)
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# 创建路由器
router = APIRouter(prefix="/ocr", tags=["OCR识别"])


async def get_storage_service() -> StorageServiceClient:
    """依赖注入：获取Storage服务客户端"""
    return get_storage_client(
        base_url=settings.services.STORAGE_SERVICE_URL,
        timeout=settings.services.STORAGE_SERVICE_TIMEOUT,
        retries=settings.services.STORAGE_SERVICE_RETRIES
    )


@router.post("/recognize", response_model=OCRRecognizeResponse, summary="单图像OCR识别")
async def recognize_image(
    file: UploadFile = File(..., description="待识别的图像文件"),
    engine: str = Form("paddleocr", description="OCR引擎类型"),
    confidence_threshold: float = Form(0.8, description="置信度阈值"),
    language_codes: str = Form("zh,en", description="语言代码"),
    enable_preprocessing: bool = Form(True, description="启用预处理"),
    enable_postprocessing: bool = Form(True, description="启用后处理"),
    async_mode: bool = Form(False, description="异步模式"),
    ocr_service: OCRService = Depends(get_ocr_service),
    storage_client: StorageServiceClient = Depends(get_storage_service)
) -> OCRRecognizeResponse:
    """
    单个图像OCR文字识别
    
    支持多种OCR引擎和配置选项，可选择同步或异步处理模式。
    异步模式下会创建任务并返回任务ID，需要通过查询接口获取结果。
    """
    try:
        # 验证文件格式
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图像文件")
        
        # 验证文件大小
        if file.size and file.size > settings.ocr.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小不能超过{settings.ocr.MAX_FILE_SIZE // 1024 // 1024}MB"
            )
        
        # 验证引擎支持
        available_engines = await ocr_service.get_available_engines()
        if engine not in available_engines:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的OCR引擎: {engine}, 可用引擎: {available_engines}"
            )
        
        # 读取文件内容
        file_content = await file.read()
        
        if async_mode:
            # 异步模式：创建任务并通过storage-service存储
            async with storage_client:
                # 创建OCR任务请求
                task_request = OCRTaskRequest(
                    file_path=f"ocr-temp/{uuid.uuid4()}/{file.filename}",
                    engine=engine,
                    confidence_threshold=confidence_threshold,
                    language_codes=language_codes,
                    enable_preprocessing=enable_preprocessing,
                    enable_postprocessing=enable_postprocessing,
                    metadata={
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "file_size": len(file_content)
                    }
                )
                
                # 创建任务
                task_id = await storage_client.create_ocr_task(task_request)
                
                # 上传文件到storage-service
                await storage_client.upload_processed_file(
                    file_path=task_request.file_path,
                    content=file_content,
                    content_type=file.content_type
                )
                
                # 更新任务状态为处理中
                await storage_client.update_ocr_task_status(task_id, "processing")
            
            # 后台处理任务
            asyncio.create_task(_process_async_task(task_id, file_content, engine, 
                                                  confidence_threshold, language_codes,
                                                  enable_preprocessing, enable_postprocessing))
            
            return OCRRecognizeResponse(
                success=True,
                message="任务已创建，异步处理中",
                data={
                    "task_id": task_id,
                    "status": "processing",
                    "async_mode": True
                }
            )
        else:
            # 同步模式：直接处理
            result = await ocr_service.recognize_image(
                image_content=file_content,
                engine=engine,
                confidence_threshold=confidence_threshold,
                language_codes=language_codes,
                enable_preprocessing=enable_preprocessing,
                enable_postprocessing=enable_postprocessing,
                metadata={
                    "filename": file.filename,
                    "content_type": file.content_type
                }
            )
            
            if result["success"]:
                return OCRRecognizeResponse(
                    success=True,
                    message="OCR识别完成",
                    data={
                        "text_content": result["text_content"],
                        "confidence": result["confidence"],
                        "word_count": result["word_count"],
                        "char_count": result["char_count"],
                        "processing_time": result["processing_time"],
                        "language_detected": result["language_detected"],
                        "bounding_boxes": result["bounding_boxes"],
                        "text_blocks": result["text_blocks"],
                        "metadata": result["metadata"],
                        "async_mode": False
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"OCR识别失败: {result['error_message']}"
                )
                
    except HTTPException:
        raise
    except StorageClientError as e:
        logger.error(f"Storage服务错误: {e}")
        raise HTTPException(status_code=503, detail=f"Storage服务不可用: {str(e)}")
    except (OCREngineError, ImageProcessingError) as e:
        logger.error(f"OCR处理错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"OCR识别未知错误: {e}")
        raise HTTPException(status_code=500, detail="OCR识别服务内部错误")


async def _process_async_task(
    task_id: str, 
    file_content: bytes,
    engine: str,
    confidence_threshold: float,
    language_codes: str,
    enable_preprocessing: bool,
    enable_postprocessing: bool
):
    """后台处理异步OCR任务"""
    try:
        ocr_service = await get_ocr_service()
        
        # 执行OCR识别
        result = await ocr_service.recognize_image(
            image_content=file_content,
            engine=engine,
            confidence_threshold=confidence_threshold,
            language_codes=language_codes,
            enable_preprocessing=enable_preprocessing,
            enable_postprocessing=enable_postprocessing,
            metadata={"task_id": task_id}
        )
        
        # 保存结果到storage-service
        async with get_storage_client(
            base_url=settings.services.STORAGE_SERVICE_URL,
            timeout=settings.services.STORAGE_SERVICE_TIMEOUT
        ) as storage_client:
            if result["success"]:
                # 保存识别结果
                result_data = {
                    "text_content": result["text_content"],
                    "confidence": result["confidence"],
                    "bounding_boxes": result.get("bounding_boxes", []),
                    "text_blocks": result.get("text_blocks", []),
                    "language_detected": result.get("language_detected"),
                    "word_count": result["word_count"],
                    "char_count": result["char_count"],
                    "processing_time": result["processing_time"],
                    "metadata": result["metadata"]
                }
                await storage_client.save_ocr_result(task_id, result_data)
                await storage_client.update_ocr_task_status(task_id, "completed")
            else:
                # 保存错误信息
                await storage_client.update_ocr_task_status(
                    task_id, "failed", result.get("error_message")
                )
                
    except Exception as e:
        logger.error(f"异步任务处理失败: {task_id}, 错误: {e}")
        try:
            async with get_storage_client(
                base_url=settings.services.STORAGE_SERVICE_URL,
                timeout=settings.services.STORAGE_SERVICE_TIMEOUT
            ) as storage_client:
                await storage_client.update_ocr_task_status(task_id, "failed", str(e))
        except Exception as storage_error:
            logger.error(f"更新任务状态失败: {storage_error}")


@router.post("/batch", response_model=OCRBatchResponse, summary="批量OCR识别")
async def batch_recognize(
    files: List[UploadFile] = File(..., description="待识别的图像文件列表"),
    engine: str = Form("paddleocr", description="OCR引擎类型"),
    confidence_threshold: float = Form(0.8, description="置信度阈值"),
    language_codes: str = Form("zh,en", description="语言代码"),
    enable_preprocessing: bool = Form(True, description="启用预处理"),
    enable_postprocessing: bool = Form(True, description="启用后处理"),
    ocr_service: OCRService = Depends(get_ocr_service)
) -> OCRBatchResponse:
    """
    批量图像OCR文字识别
    
    支持多个图像文件的并发处理，返回每个文件的识别结果。
    """
    try:
        # 验证文件数量
        if len(files) > settings.ocr.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"批量文件数量不能超过{settings.ocr.MAX_BATCH_SIZE}"
            )
        
        # 验证引擎支持
        available_engines = await ocr_service.get_available_engines()
        if engine not in available_engines:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的OCR引擎: {engine}, 可用引擎: {available_engines}"
            )
        
        # 准备图像数据
        images_data = []
        for i, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"文件{file.filename}不是图像格式"
                )
            
            if file.size and file.size > settings.ocr.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件{file.filename}大小超出限制"
                )
            
            file_content = await file.read()
            images_data.append({
                "image_content": file_content,
                "image_id": f"batch_{i}_{file.filename}",
                "filename": file.filename,
                "content_type": file.content_type
            })
        
        # 批量处理
        results = await ocr_service.process_batch_images(
            images_data=images_data,
            engine=engine,
            confidence_threshold=confidence_threshold,
            language_codes=language_codes,
            enable_preprocessing=enable_preprocessing,
            enable_postprocessing=enable_postprocessing
        )
        
        # 统计结果
        successful_count = sum(1 for r in results if r.get("success"))
        failed_count = len(results) - successful_count
        
        return OCRBatchResponse(
            success=True,
            message=f"批量识别完成，成功: {successful_count}, 失败: {failed_count}",
            data={
                "total_files": len(files),
                "successful_count": successful_count,
                "failed_count": failed_count,
                "results": results
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量OCR识别错误: {e}")
        raise HTTPException(status_code=500, detail="批量OCR识别服务内部错误")


@router.get("/task/{task_id}", response_model=OCRTaskStatusResponse, summary="查询OCR任务状态")
async def get_task_status(
    task_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_service)
) -> OCRTaskStatusResponse:
    """
    查询OCR任务的处理状态和结果
    
    用于异步模式下查询任务进度和获取最终结果。
    """
    try:
        async with storage_client:
            # 获取任务结果
            result = await storage_client.get_ocr_result(task_id)
            
            if not result:
                raise HTTPException(status_code=404, detail="任务不存在")
            
            return OCRTaskStatusResponse(
                success=True,
                message="任务状态查询成功",
                data={
                    "task_id": result.task_id,
                    "status": result.status,
                    "text_content": result.text_content,
                    "confidence_scores": result.confidence_scores,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message,
                    "created_at": result.created_at,
                    "completed_at": result.completed_at
                }
            )
            
    except HTTPException:
        raise
    except StorageClientError as e:
        logger.error(f"Storage服务错误: {e}")
        raise HTTPException(status_code=503, detail=f"Storage服务不可用: {str(e)}")
    except Exception as e:
        logger.error(f"查询任务状态错误: {e}")
        raise HTTPException(status_code=500, detail="查询任务状态服务内部错误")


@router.get("/engines", response_model=OCREnginesResponse, summary="获取可用OCR引擎")
async def get_available_engines(
    ocr_service: OCRService = Depends(get_ocr_service)
) -> OCREnginesResponse:
    """
    获取当前可用的OCR引擎列表
    
    返回已初始化并可正常使用的OCR引擎信息。
    """
    try:
        available_engines = await ocr_service.get_available_engines()
        
        # 获取每个引擎的详细配置
        engine_details = {}
        for engine in available_engines:
            engine_details[engine] = settings.get_engine_config(engine)
        
        return OCREnginesResponse(
            success=True,
            message="获取可用引擎成功",
            data={
                "available_engines": available_engines,
                "default_engine": settings.ocr.DEFAULT_ENGINE,
                "engine_configs": engine_details,
                "total_count": len(available_engines)
            }
        )
        
    except Exception as e:
        logger.error(f"获取可用引擎错误: {e}")
        raise HTTPException(status_code=500, detail="获取可用引擎服务内部错误")


@router.get("/health", response_model=BaseResponse, summary="OCR服务健康检查")
async def health_check(
    ocr_service: OCRService = Depends(get_ocr_service)
) -> BaseResponse:
    """
    OCR服务健康状态检查
    
    检查OCR引擎状态、服务可用性等信息。
    """
    try:
        health_status = await ocr_service.health_check()
        
        return BaseResponse(
            success=True,
            message="健康检查完成",
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"健康检查错误: {e}")
        return BaseResponse(
            success=False,
            message=f"健康检查失败: {str(e)}",
            data={"status": "unhealthy", "error": str(e)}
        )


@router.get("/statistics", response_model=BaseResponse, summary="OCR服务统计信息")
async def get_statistics(
    storage_client: StorageServiceClient = Depends(get_storage_service)
) -> BaseResponse:
    """
    获取OCR服务的统计信息
    
    通过storage-service获取任务统计数据。
    """
    try:
        async with storage_client:
            statistics = await storage_client.get_task_statistics()
            
            return BaseResponse(
                success=True,
                message="统计信息获取成功",
                data=statistics
            )
            
    except StorageClientError as e:
        logger.error(f"Storage服务错误: {e}")
        raise HTTPException(status_code=503, detail=f"Storage服务不可用: {str(e)}")
    except Exception as e:
        logger.error(f"获取统计信息错误: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息服务内部错误")


# 添加路由到应用
def include_router(app):
    """将OCR路由添加到FastAPI应用"""
    app.include_router(router)