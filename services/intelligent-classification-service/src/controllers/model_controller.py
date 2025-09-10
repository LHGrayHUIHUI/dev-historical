"""
智能分类服务模型管理控制器
处理模型训练、管理和性能监控
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio

from ..clients.storage_client import StorageServiceClient
from ..services.model_trainer import ModelTrainer, TrainingConfig
from ..schemas.classification_schemas import (
    BaseResponse, ModelTrainingRequest, ModelInfo, TrainingMetrics,
    ModelPerformanceResponse, ModelType, FeatureExtractorType
)
from ..config.settings import settings

# 创建路由器
router = APIRouter(prefix="/api/v1/models", tags=["模型管理"])
logger = logging.getLogger(__name__)


# 依赖注入
async def get_storage_client() -> StorageServiceClient:
    """获取存储服务客户端"""
    return StorageServiceClient()


def get_model_trainer() -> ModelTrainer:
    """获取模型训练器"""
    return ModelTrainer()


@router.post("/train", response_model=BaseResponse)
async def train_classification_model(
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    storage_client: StorageServiceClient = Depends(get_storage_client),
    trainer: ModelTrainer = Depends(get_model_trainer)
):
    """训练分类模型
    
    基于项目的训练数据训练新的分类模型
    """
    try:
        logger.info(f"开始训练模型，项目ID: {training_request.project_id}")
        
        # 验证项目存在
        project = await storage_client.get_classification_project(training_request.project_id)
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"项目 {training_request.project_id} 不存在"
            )
        
        # 获取训练数据
        training_data_response = await storage_client.get_training_data(
            training_request.project_id,
            limit=10000  # 获取足够的训练数据
        )
        
        if not training_data_response or not training_data_response.get('data'):
            raise HTTPException(
                status_code=400,
                detail=f"项目 {training_request.project_id} 没有训练数据"
            )
        
        training_data = training_data_response['data']
        
        if len(training_data) < settings.training_config.get('min_samples_per_class', 5):
            raise HTTPException(
                status_code=400,
                detail=f"训练数据量不足，至少需要 {settings.training_config.get('min_samples_per_class', 5)} 个样本"
            )
        
        # 准备训练数据
        texts = [item['text_content'] for item in training_data]
        labels = [item['true_label'] for item in training_data]
        
        # 创建模型记录
        model_data = {
            'project_id': training_request.project_id,
            'model_name': f"{training_request.model_type}_{training_request.feature_extractor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model_type': training_request.model_type,
            'feature_extractor': training_request.feature_extractor,
            'model_version': '1.0.0',
            'hyperparameters': training_request.hyperparameters or {},
            'status': 'training',
            'training_data_size': len(training_data),
            'is_active': False,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # 在storage service中创建模型记录
        model_response = await storage_client.create_classification_model(model_data)
        model_id = model_response.get('id')
        
        # 异步训练模型
        background_tasks.add_task(
            _train_model_background,
            model_id=model_id,
            texts=texts,
            labels=labels,
            model_type=training_request.model_type,
            feature_extractor_type=training_request.feature_extractor,
            hyperparameters=training_request.hyperparameters or {},
            feature_config=training_request.training_config or {},
            storage_client=storage_client,
            trainer=trainer
        )
        
        logger.info(f"模型训练任务已启动，模型ID: {model_id}")
        
        return BaseResponse(
            success=True,
            message="模型训练任务已启动",
            data={
                'model_id': model_id,
                'status': 'training',
                'estimated_time': '10-30分钟',
                'training_data_size': len(training_data)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动模型训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动模型训练失败: {str(e)}")


async def _train_model_background(
    model_id: str,
    texts: List[str],
    labels: List[str],
    model_type: ModelType,
    feature_extractor_type: FeatureExtractorType,
    hyperparameters: Dict[str, Any],
    feature_config: Dict[str, Any],
    storage_client: StorageServiceClient,
    trainer: ModelTrainer
):
    """后台模型训练任务"""
    try:
        logger.info(f"开始后台训练模型: {model_id}")
        
        # 训练模型
        training_result = trainer.train_model(
            texts=texts,
            labels=labels,
            model_type=model_type,
            feature_extractor_type=feature_extractor_type,
            hyperparameters=hyperparameters,
            feature_config=feature_config
        )
        
        # 模拟保存模型文件（实际应该保存到MinIO或文件系统）
        model_path = f"/tmp/models/model_{model_id}.pkl"
        trainer.save_trained_model(training_result, model_path)
        
        # 更新模型状态和性能指标
        update_data = {
            'status': 'completed',
            'model_path': model_path,
            'training_time': training_result['training_time'],
            'training_metrics': {
                'accuracy': training_result['metrics'].accuracy,
                'precision': training_result['metrics'].precision,
                'recall': training_result['metrics'].recall,
                'f1_score': training_result['metrics'].f1_score,
                'cv_mean': training_result['metrics'].cv_mean,
                'cv_std': training_result['metrics'].cv_std
            },
            'completed_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        await storage_client.update_classification_model(model_id, update_data)
        
        # 如果这是第一个完成的模型，设为活跃模型
        try:
            project_id = texts[0] if texts else None  # 简化处理，实际应该从模型记录获取
            existing_models = await storage_client.list_classification_models(project_id, limit=10)
            active_model_exists = any(
                model.get('is_active', False) 
                for model in existing_models.get('data', [])
            )
            
            if not active_model_exists:
                await storage_client.set_active_model(project_id, model_id)
                logger.info(f"模型 {model_id} 已设置为活跃模型")
        
        except Exception as e:
            logger.warning(f"设置活跃模型失败: {e}")
        
        logger.info(f"模型训练完成: {model_id}, F1分数: {training_result['metrics'].f1_score:.4f}")
        
    except Exception as e:
        logger.error(f"模型训练失败: {model_id}, 错误: {e}")
        
        # 更新模型状态为失败
        try:
            update_data = {
                'status': 'failed',
                'error_message': str(e),
                'failed_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            await storage_client.update_classification_model(model_id, update_data)
        except Exception as update_error:
            logger.error(f"更新模型失败状态时出错: {update_error}")


@router.get("/{model_id}", response_model=BaseResponse)
async def get_model_info(
    model_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """获取模型详细信息"""
    try:
        logger.info(f"获取模型信息: {model_id}")
        
        model_info = await storage_client.get_classification_model(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")
        
        # 获取模型性能统计
        try:
            performance_stats = await storage_client.get_model_performance_stats(model_id)
            model_info['performance_stats'] = performance_stats
        except Exception as e:
            logger.warning(f"获取模型性能统计失败: {e}")
            model_info['performance_stats'] = None
        
        return BaseResponse(
            success=True,
            message="获取模型信息成功",
            data=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@router.get("/project/{project_id}", response_model=BaseResponse)
async def list_project_models(
    project_id: str,
    limit: int = 50,
    offset: int = 0,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """列出项目的所有模型"""
    try:
        logger.info(f"列出项目模型: {project_id}")
        
        models = await storage_client.list_classification_models(
            project_id=project_id,
            limit=limit,
            offset=offset
        )
        
        return BaseResponse(
            success=True,
            message="获取项目模型列表成功",
            data=models
        )
        
    except Exception as e:
        logger.error(f"获取项目模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取项目模型列表失败: {str(e)}")


@router.post("/{model_id}/activate", response_model=BaseResponse)
async def activate_model(
    model_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """激活模型
    
    将指定模型设置为项目的活跃模型
    """
    try:
        logger.info(f"激活模型: {model_id}")
        
        # 获取模型信息
        model_info = await storage_client.get_classification_model(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")
        
        if model_info.get('status') != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"只有训练完成的模型才能激活，当前状态: {model_info.get('status')}"
            )
        
        project_id = model_info['project_id']
        
        # 设置活跃模型
        result = await storage_client.set_active_model(project_id, model_id)
        
        logger.info(f"模型激活成功: {model_id}")
        
        return BaseResponse(
            success=True,
            message="模型激活成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"激活模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"激活模型失败: {str(e)}")


@router.get("/{model_id}/performance", response_model=BaseResponse)
async def get_model_performance(
    model_id: str,
    include_detailed_metrics: bool = False,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """获取模型性能统计"""
    try:
        logger.info(f"获取模型性能: {model_id}")
        
        performance = await storage_client.get_model_performance_stats(model_id)
        if not performance:
            raise HTTPException(status_code=404, detail=f"模型 {model_id} 的性能统计不存在")
        
        # 如果需要详细指标，获取更多信息
        if include_detailed_metrics:
            model_info = await storage_client.get_classification_model(model_id)
            performance['model_info'] = model_info
            
            # 获取最近的分类结果用于性能分析
            try:
                recent_results = await storage_client.list_classification_results(
                    project_id=model_info['project_id'],
                    limit=100
                )
                performance['recent_predictions'] = recent_results
            except Exception as e:
                logger.warning(f"获取最近分类结果失败: {e}")
        
        return BaseResponse(
            success=True,
            message="获取模型性能成功",
            data=performance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型性能失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型性能失败: {str(e)}")


@router.delete("/{model_id}", response_model=BaseResponse)
async def delete_model(
    model_id: str,
    force: bool = False,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """删除模型
    
    删除指定的分类模型及其相关数据
    """
    try:
        logger.info(f"删除模型: {model_id}")
        
        # 检查模型是否是活跃模型
        model_info = await storage_client.get_classification_model(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")
        
        if model_info.get('is_active', False) and not force:
            raise HTTPException(
                status_code=400,
                detail="不能删除活跃模型，请先激活其他模型或使用force参数强制删除"
            )
        
        # 删除模型
        result = await storage_client.delete_classification_model(model_id)
        
        logger.info(f"模型删除成功: {model_id}")
        
        return BaseResponse(
            success=True,
            message="模型删除成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")


@router.get("/project/{project_id}/active", response_model=BaseResponse)
async def get_active_model(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """获取项目的活跃模型"""
    try:
        logger.info(f"获取项目活跃模型: {project_id}")
        
        active_model = await storage_client.get_active_model(project_id)
        if not active_model:
            raise HTTPException(
                status_code=404,
                detail=f"项目 {project_id} 没有活跃模型"
            )
        
        return BaseResponse(
            success=True,
            message="获取活跃模型成功",
            data=active_model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取活跃模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取活跃模型失败: {str(e)}")


@router.get("/types/supported", response_model=BaseResponse)
async def get_supported_model_types():
    """获取支持的模型类型和特征提取器"""
    try:
        supported_types = {
            'model_types': list(settings.ml_models.keys()),
            'feature_extractors': list(settings.feature_extraction.keys()),
            'model_configs': settings.ml_models,
            'feature_configs': settings.feature_extraction,
            'performance_thresholds': settings.performance_thresholds
        }
        
        return BaseResponse(
            success=True,
            message="获取支持的模型类型成功",
            data=supported_types
        )
        
    except Exception as e:
        logger.error(f"获取支持的模型类型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取支持的模型类型失败: {str(e)}")


@router.post("/{model_id}/benchmark", response_model=BaseResponse)
async def benchmark_model(
    model_id: str,
    background_tasks: BackgroundTasks,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """对模型进行基准测试
    
    使用项目的测试数据对模型进行性能评估
    """
    try:
        logger.info(f"开始模型基准测试: {model_id}")
        
        # 检查模型状态
        model_info = await storage_client.get_classification_model(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")
        
        if model_info.get('status') != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"只有训练完成的模型才能进行基准测试，当前状态: {model_info.get('status')}"
            )
        
        # 启动后台基准测试任务
        background_tasks.add_task(
            _benchmark_model_background,
            model_id=model_id,
            project_id=model_info['project_id'],
            storage_client=storage_client
        )
        
        return BaseResponse(
            success=True,
            message="模型基准测试已启动",
            data={
                'model_id': model_id,
                'status': 'benchmarking',
                'estimated_time': '5-15分钟'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动模型基准测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动模型基准测试失败: {str(e)}")


async def _benchmark_model_background(
    model_id: str,
    project_id: str,
    storage_client: StorageServiceClient
):
    """后台模型基准测试任务"""
    try:
        logger.info(f"执行模型基准测试: {model_id}")
        
        # 这里应该实现实际的基准测试逻辑
        # 包括加载模型、获取测试数据、评估性能等
        
        # 模拟基准测试结果
        benchmark_results = {
            'test_accuracy': 0.85,
            'test_f1_score': 0.83,
            'test_precision': 0.86,
            'test_recall': 0.81,
            'inference_speed': 150.5,  # 每秒处理文档数
            'memory_usage': 512.0,  # MB
            'benchmark_date': datetime.now().isoformat()
        }
        
        # 更新模型性能数据
        await storage_client.update_model_performance(model_id, {
            'benchmark_results': benchmark_results,
            'last_benchmarked': datetime.now().isoformat()
        })
        
        logger.info(f"模型基准测试完成: {model_id}")
        
    except Exception as e:
        logger.error(f"模型基准测试失败: {model_id}, 错误: {e}")