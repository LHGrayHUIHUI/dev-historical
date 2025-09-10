"""
智能分类服务项目管理控制器
处理分类项目的创建、更新、查询和删除
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..clients.storage_client import StorageServiceClient
from ..schemas.classification_schemas import (
    BaseResponse, ClassificationProjectCreate, ClassificationProject,
    ProjectStatistics
)
from ..config.settings import settings

# 创建路由器
router = APIRouter(prefix="/api/v1/projects", tags=["项目管理"])
logger = logging.getLogger(__name__)


# 依赖注入
async def get_storage_client() -> StorageServiceClient:
    """获取存储服务客户端"""
    return StorageServiceClient()


@router.post("/", response_model=BaseResponse)
async def create_classification_project(
    project_data: ClassificationProjectCreate,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """创建分类项目
    
    创建新的文档分类项目，包括项目配置和分类标签设定
    """
    try:
        logger.info(f"创建分类项目: {project_data.name}")
        
        # 验证分类类型
        if not settings.is_supported_classification_type(project_data.classification_type):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的分类类型: {project_data.classification_type}"
            )
        
        # 准备项目数据
        project_dict = project_data.dict()
        
        # 如果没有提供自定义标签，使用预定义标签
        if not project_dict.get('custom_labels'):
            project_dict['class_labels'] = settings.get_classification_labels(
                project_data.classification_type
            )
        else:
            project_dict['class_labels'] = project_dict.pop('custom_labels')
        
        # 设置默认配置
        if not project_dict.get('model_config'):
            project_dict['model_config'] = {
                'default_model_type': 'random_forest',
                'default_feature_extractor': 'tfidf',
                'enable_model_comparison': True,
                'auto_model_selection': True
            }
        
        if not project_dict.get('training_config'):
            project_dict['training_config'] = settings.training_config
        
        if not project_dict.get('feature_config'):
            project_dict['feature_config'] = {
                'tfidf': settings.get_feature_config('tfidf'),
                'word2vec': settings.get_feature_config('word2vec')
            }
        
        # 添加项目元数据
        project_dict.update({
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'status': 'active'
        })
        
        # 调用storage service创建项目
        result = await storage_client.create_classification_project(project_dict)
        
        logger.info(f"分类项目创建成功，项目ID: {result.get('id', 'unknown')}")
        
        return BaseResponse(
            success=True,
            message="分类项目创建成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建分类项目失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建分类项目失败: {str(e)}")


@router.get("/{project_id}", response_model=BaseResponse)
async def get_classification_project(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """获取分类项目详情
    
    获取指定项目的详细信息，包括配置和统计数据
    """
    try:
        logger.info(f"获取分类项目详情: {project_id}")
        
        # 获取项目信息
        project = await storage_client.get_classification_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail=f"项目 {project_id} 不存在")
        
        # 获取项目统计信息
        try:
            statistics = await storage_client.get_project_statistics(project_id)
            project['statistics'] = statistics
        except Exception as e:
            logger.warning(f"获取项目统计失败: {e}")
            project['statistics'] = None
        
        return BaseResponse(
            success=True,
            message="获取项目详情成功",
            data=project
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取项目详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取项目详情失败: {str(e)}")


@router.put("/{project_id}", response_model=BaseResponse)
async def update_classification_project(
    project_id: str,
    update_data: Dict[str, Any],
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """更新分类项目
    
    更新项目配置、标签或其他设置
    """
    try:
        logger.info(f"更新分类项目: {project_id}")
        
        # 添加更新时间戳
        update_data['updated_at'] = datetime.now().isoformat()
        
        # 验证更新数据中的分类类型
        if 'classification_type' in update_data:
            if not settings.is_supported_classification_type(update_data['classification_type']):
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的分类类型: {update_data['classification_type']}"
                )
        
        # 调用storage service更新项目
        result = await storage_client.update_classification_project(project_id, update_data)
        
        logger.info(f"分类项目更新成功: {project_id}")
        
        return BaseResponse(
            success=True,
            message="分类项目更新成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新分类项目失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新分类项目失败: {str(e)}")


@router.delete("/{project_id}", response_model=BaseResponse)
async def delete_classification_project(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """删除分类项目
    
    删除项目及其相关的所有数据（训练数据、模型等）
    """
    try:
        logger.info(f"删除分类项目: {project_id}")
        
        # 调用storage service删除项目
        result = await storage_client.delete_classification_project(project_id)
        
        logger.info(f"分类项目删除成功: {project_id}")
        
        return BaseResponse(
            success=True,
            message="分类项目删除成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"删除分类项目失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除分类项目失败: {str(e)}")


@router.get("/", response_model=BaseResponse)
async def list_classification_projects(
    limit: int = 100,
    offset: int = 0,
    classification_type: Optional[str] = None,
    status: Optional[str] = None,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """列出分类项目
    
    获取分类项目列表，支持分页和过滤
    """
    try:
        logger.info(f"列出分类项目，limit: {limit}, offset: {offset}")
        
        # 构建过滤条件
        filters = {}
        if classification_type:
            if not settings.is_supported_classification_type(classification_type):
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的分类类型: {classification_type}"
                )
            filters['classification_type'] = classification_type
        
        if status:
            filters['status'] = status
        
        # 调用storage service获取项目列表
        result = await storage_client.list_classification_projects(
            limit=limit,
            offset=offset,
            filters=filters if filters else None
        )
        
        return BaseResponse(
            success=True,
            message="获取项目列表成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取项目列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取项目列表失败: {str(e)}")


@router.get("/{project_id}/statistics", response_model=BaseResponse)
async def get_project_statistics(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """获取项目统计信息
    
    获取项目的详细统计数据，包括训练数据、模型性能等
    """
    try:
        logger.info(f"获取项目统计信息: {project_id}")
        
        # 获取项目统计
        statistics = await storage_client.get_project_statistics(project_id)
        
        if not statistics:
            raise HTTPException(status_code=404, detail=f"项目 {project_id} 的统计信息不存在")
        
        return BaseResponse(
            success=True,
            message="获取项目统计信息成功",
            data=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取项目统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取项目统计信息失败: {str(e)}")


@router.get("/supported/types", response_model=BaseResponse)
async def get_supported_classification_types():
    """获取支持的分类类型
    
    返回系统支持的所有分类类型及其预定义标签
    """
    try:
        supported_types = {
            'classification_types': list(settings.predefined_labels.keys()),
            'predefined_labels': settings.predefined_labels,
            'model_types': list(settings.ml_models.keys()),
            'feature_extractors': list(settings.feature_extraction.keys())
        }
        
        return BaseResponse(
            success=True,
            message="获取支持的分类类型成功",
            data=supported_types
        )
        
    except Exception as e:
        logger.error(f"获取支持的分类类型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取支持的分类类型失败: {str(e)}")


@router.post("/{project_id}/archive", response_model=BaseResponse)
async def archive_project(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """归档项目
    
    将项目状态设置为已归档，停止活动但保留数据
    """
    try:
        logger.info(f"归档项目: {project_id}")
        
        update_data = {
            'status': 'archived',
            'archived_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        result = await storage_client.update_classification_project(project_id, update_data)
        
        logger.info(f"项目归档成功: {project_id}")
        
        return BaseResponse(
            success=True,
            message="项目归档成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"项目归档失败: {e}")
        raise HTTPException(status_code=500, detail=f"项目归档失败: {str(e)}")


@router.post("/{project_id}/activate", response_model=BaseResponse)
async def activate_project(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """激活项目
    
    将归档或暂停的项目重新激活
    """
    try:
        logger.info(f"激活项目: {project_id}")
        
        update_data = {
            'status': 'active',
            'activated_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        result = await storage_client.update_classification_project(project_id, update_data)
        
        logger.info(f"项目激活成功: {project_id}")
        
        return BaseResponse(
            success=True,
            message="项目激活成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"项目激活失败: {e}")
        raise HTTPException(status_code=500, detail=f"项目激活失败: {str(e)}")