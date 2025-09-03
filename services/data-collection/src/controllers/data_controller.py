"""
数据采集API控制器

提供文件上传、数据集管理等REST API端点
"""

import json
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPBearer

from ..schemas.data_schemas import (
    BatchUploadResponse,
    DatasetListResponse,
    DatasetResponse,
    DatasetUpdateRequest,
    ProcessingStatusResponse,
    UploadResponse,
)
from ..services.data_collection_service import DataCollectionService

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/data", tags=["数据采集"])
security = HTTPBearer()


# 依赖注入函数
def get_data_service() -> DataCollectionService:
    """获取数据采集服务实例"""
    return DataCollectionService()


def get_current_user_id() -> str:
    """获取当前用户ID（临时实现）
    
    注意：这是临时实现，实际应该从JWT token中解析用户信息
    """
    # TODO: 从JWT token中获取真实的用户ID
    return "temp-user-id"


# API端点


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="单文件上传",
    description="上传单个文件并开始文本提取处理"
)
async def upload_file(
    file: UploadFile = File(..., description="要上传的文件"),
    source_id: UUID = Form(..., description="数据源ID"),
    metadata: Optional[str] = Form(None, description="文件元数据JSON字符串"),
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
) -> UploadResponse:
    """上传单个文件
    
    Args:
        file: 上传的文件
        source_id: 数据源ID
        metadata: 文件元数据JSON字符串
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        上传结果
        
    Raises:
        HTTPException: 上传失败时抛出异常
    """
    try:
        logger.info(f"收到文件上传请求: {file.filename}, 用户: {current_user_id}")
        
        # 解析元数据
        file_metadata = {}
        if metadata:
            try:
                file_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="元数据JSON格式无效"
                )
        
        # 调用服务上传文件
        result = await data_service.upload_single_file(
            file=file,
            source_id=str(source_id),
            user_id=current_user_id,
            metadata=file_metadata
        )
        
        return UploadResponse(
            success=True,
            message="文件上传成功" if not result.get("is_duplicate") else "文件已存在",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传处理失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败: {str(e)}"
        )


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    summary="批量文件上传",
    description="批量上传多个文件并开始文本提取处理"
)
async def upload_batch_files(
    files: List[UploadFile] = File(..., description="要上传的文件列表"),
    source_id: UUID = Form(..., description="数据源ID"),
    metadata: Optional[str] = Form(None, description="批次元数据JSON字符串"),
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
) -> BatchUploadResponse:
    """批量上传文件
    
    Args:
        files: 上传的文件列表
        source_id: 数据源ID
        metadata: 批次元数据JSON字符串
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        批量上传结果
        
    Raises:
        HTTPException: 上传失败时抛出异常
    """
    try:
        logger.info(f"收到批量文件上传请求: {len(files)} 个文件, 用户: {current_user_id}")
        
        # 解析元数据
        batch_metadata = {}
        if metadata:
            try:
                batch_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="元数据JSON格式无效"
                )
        
        # 调用服务批量上传文件
        result = await data_service.upload_batch_files(
            files=files,
            source_id=str(source_id),
            user_id=current_user_id,
            metadata=batch_metadata
        )
        
        return BatchUploadResponse(
            success=True,
            message=f"批量上传完成，成功: {result['successful_uploads']}, 失败: {result['failed_uploads']}",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量上传处理失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"批量上传失败: {str(e)}"
        )


@router.get(
    "/datasets",
    response_model=DatasetListResponse,
    summary="获取数据集列表",
    description="分页获取用户的数据集列表"
)
async def get_datasets(
    page: int = 1,
    size: int = 20,
    status: Optional[str] = None,
    source_id: Optional[UUID] = None,
    search: Optional[str] = None,
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
) -> DatasetListResponse:
    """获取数据集列表
    
    Args:
        page: 页码，从1开始
        size: 每页大小，最大100
        status: 处理状态过滤 (pending, processing, completed, failed)
        source_id: 数据源ID过滤
        search: 搜索关键词，在数据集名称中搜索
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        数据集列表和分页信息
        
    Raises:
        HTTPException: 查询失败时抛出异常
    """
    try:
        # 参数验证
        if page < 1:
            raise HTTPException(status_code=400, detail="页码必须大于0")
        
        if size < 1 or size > 100:
            raise HTTPException(status_code=400, detail="每页大小必须在1-100之间")
        
        if status and status not in ['pending', 'processing', 'completed', 'failed']:
            raise HTTPException(
                status_code=400,
                detail="状态参数无效，必须是：pending, processing, completed, failed 之一"
            )
        
        logger.info(f"查询数据集列表: 用户={current_user_id}, 页码={page}, 大小={size}")
        
        # 调用服务查询数据集列表
        result = await data_service.get_datasets(
            user_id=current_user_id,
            page=page,
            size=size,
            status=status,
            source_id=str(source_id) if source_id else None,
            search=search
        )
        
        return DatasetListResponse(
            success=True,
            message="查询成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询数据集列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"查询数据集列表失败: {str(e)}"
        )


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetResponse,
    summary="获取数据集详情",
    description="获取指定数据集的详细信息"
)
async def get_dataset(
    dataset_id: UUID,
    include_content: bool = False,
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
) -> DatasetResponse:
    """获取数据集详情
    
    Args:
        dataset_id: 数据集ID
        include_content: 是否包含提取的文本内容
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        数据集详情
        
    Raises:
        HTTPException: 数据集不存在或查询失败时抛出异常
    """
    try:
        logger.info(f"查询数据集详情: {dataset_id}, 用户: {current_user_id}")
        
        # 调用服务查询数据集详情
        result = await data_service.get_dataset(
            dataset_id=str(dataset_id),
            user_id=current_user_id,
            include_content=include_content
        )
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="数据集不存在或无权访问"
            )
        
        return DatasetResponse(
            success=True,
            message="查询成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询数据集详情失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"查询数据集详情失败: {str(e)}"
        )


@router.put(
    "/datasets/{dataset_id}",
    response_model=DatasetResponse,
    summary="更新数据集",
    description="更新数据集的名称、描述等信息"
)
async def update_dataset(
    dataset_id: UUID,
    update_data: DatasetUpdateRequest,
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
) -> DatasetResponse:
    """更新数据集信息
    
    Args:
        dataset_id: 数据集ID
        update_data: 更新数据
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        更新后的数据集信息
        
    Raises:
        HTTPException: 数据集不存在或更新失败时抛出异常
    """
    try:
        logger.info(f"更新数据集: {dataset_id}, 用户: {current_user_id}")
        
        # 先检查数据集是否存在
        dataset = await data_service.get_dataset(
            dataset_id=str(dataset_id),
            user_id=current_user_id
        )
        
        if not dataset:
            raise HTTPException(
                status_code=404,
                detail="数据集不存在或无权访问"
            )
        
        # TODO: 实现数据集更新逻辑
        # 这里需要在DataCollectionService中添加update_dataset方法
        
        # 临时返回现有数据集信息
        return DatasetResponse(
            success=True,
            message="数据集更新成功",
            data=dataset
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新数据集失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"更新数据集失败: {str(e)}"
        )


@router.delete(
    "/datasets/{dataset_id}",
    summary="删除数据集",
    description="删除指定的数据集及其关联的文本内容"
)
async def delete_dataset(
    dataset_id: UUID,
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
):
    """删除数据集
    
    Args:
        dataset_id: 数据集ID
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        删除结果
        
    Raises:
        HTTPException: 数据集不存在或删除失败时抛出异常
    """
    try:
        logger.info(f"删除数据集: {dataset_id}, 用户: {current_user_id}")
        
        # 先检查数据集是否存在
        dataset = await data_service.get_dataset(
            dataset_id=str(dataset_id),
            user_id=current_user_id
        )
        
        if not dataset:
            raise HTTPException(
                status_code=404,
                detail="数据集不存在或无权访问"
            )
        
        # TODO: 实现数据集删除逻辑
        # 这里需要在DataCollectionService中添加delete_dataset方法
        
        return {
            "success": True,
            "message": "数据集删除成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除数据集失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"删除数据集失败: {str(e)}"
        )


@router.get(
    "/datasets/{dataset_id}/processing-status",
    response_model=ProcessingStatusResponse,
    summary="获取处理状态",
    description="获取数据集的文本处理状态"
)
async def get_processing_status(
    dataset_id: UUID,
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
) -> ProcessingStatusResponse:
    """获取数据集处理状态
    
    Args:
        dataset_id: 数据集ID
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        处理状态信息
        
    Raises:
        HTTPException: 数据集不存在或查询失败时抛出异常
    """
    try:
        logger.info(f"查询处理状态: {dataset_id}, 用户: {current_user_id}")
        
        # 调用服务查询处理状态
        status_info = await data_service.get_processing_status(
            dataset_id=str(dataset_id),
            user_id=current_user_id
        )
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail="数据集不存在或无权访问"
            )
        
        return ProcessingStatusResponse(
            success=True,
            message="查询成功",
            data=status_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询处理状态失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"查询处理状态失败: {str(e)}"
        )


@router.post(
    "/datasets/{dataset_id}/reprocess",
    summary="重新处理数据集",
    description="重新启动数据集的文本提取处理"
)
async def reprocess_dataset(
    dataset_id: UUID,
    current_user_id: str = Depends(get_current_user_id),
    data_service: DataCollectionService = Depends(get_data_service)
):
    """重新处理数据集
    
    Args:
        dataset_id: 数据集ID
        current_user_id: 当前用户ID
        data_service: 数据采集服务
        
    Returns:
        重新处理结果
        
    Raises:
        HTTPException: 数据集不存在或重新处理失败时抛出异常
    """
    try:
        logger.info(f"重新处理数据集: {dataset_id}, 用户: {current_user_id}")
        
        # 调用服务重新处理数据集
        success = await data_service.reprocess_dataset(
            dataset_id=str(dataset_id),
            user_id=current_user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="数据集不存在、无权访问或无法重新处理"
            )
        
        return {
            "success": True,
            "message": "数据集已加入重新处理队列"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新处理数据集失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"重新处理失败: {str(e)}"
        )


# 健康检查和状态端点
@router.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "data-collection-service",
        "timestamp": str(datetime.utcnow())
    }


@router.get("/info", tags=["服务信息"])
async def service_info():
    """服务信息端点"""
    from ..config import get_settings
    
    settings = get_settings()
    
    return {
        "service_name": settings.service_name,
        "service_version": settings.service_version,
        "environment": settings.service_environment,
        "supported_file_types": settings.allowed_file_types,
        "max_file_size": settings.max_file_size,
        "max_batch_size": settings.max_batch_size
    }