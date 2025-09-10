"""
智能分类服务分类预测控制器
处理文档分类请求和批量分类任务
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ..services.classification_service import classification_service, ClassificationService
from ..schemas.classification_schemas import (
    BaseResponse, ClassificationRequest, ClassificationResult,
    BatchClassificationRequest, BatchClassificationResult
)

# 创建路由器
router = APIRouter(prefix="/api/v1/classify", tags=["文档分类"])
logger = logging.getLogger(__name__)


# 依赖注入
def get_classification_service() -> ClassificationService:
    """获取分类服务实例"""
    return classification_service


@router.post("/single", response_model=BaseResponse)
async def classify_single_document(
    request: ClassificationRequest,
    service: ClassificationService = Depends(get_classification_service)
):
    """单文档分类
    
    对单个文档进行智能分类，返回预测标签和置信度
    """
    try:
        logger.info(f"单文档分类请求 - 项目: {request.project_id}")
        
        # 验证请求数据
        if not request.text_content or not request.text_content.strip():
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        if len(request.text_content) > 10000:
            raise HTTPException(status_code=400, detail="文本长度超过限制（最大10000字符）")
        
        # 执行分类
        result = await service.classify_single_document(request)
        
        logger.info(f"单文档分类完成 - 任务: {result.task_id}, 预测: {result.predicted_label}")
        
        return BaseResponse(
            success=True,
            message="文档分类成功",
            data=result.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"单文档分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"单文档分类失败: {str(e)}")


@router.post("/batch", response_model=BaseResponse)
async def classify_batch_documents(
    request: BatchClassificationRequest,
    service: ClassificationService = Depends(get_classification_service)
):
    """批量文档分类
    
    对多个文档进行批量分类处理
    """
    try:
        logger.info(f"批量文档分类请求 - 项目: {request.project_id}, 文档数: {len(request.documents)}")
        
        # 验证请求数据
        if not request.documents:
            raise HTTPException(status_code=400, detail="文档列表不能为空")
        
        if len(request.documents) > 100:
            raise HTTPException(status_code=400, detail="批量大小超过限制（最大100个文档）")
        
        # 验证每个文档
        for i, doc in enumerate(request.documents):
            if not doc.get('text_content') or not doc.get('text_content').strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"第{i+1}个文档的文本内容不能为空"
                )
            
            if len(doc.get('text_content', '')) > 10000:
                raise HTTPException(
                    status_code=400,
                    detail=f"第{i+1}个文档的文本长度超过限制（最大10000字符）"
                )
        
        # 执行批量分类
        result = await service.classify_batch_documents(request)
        
        logger.info(f"批量分类完成 - 任务: {result.batch_task_id}, 成功: {result.successful_classifications}")
        
        return BaseResponse(
            success=True,
            message="批量文档分类成功",
            data=result.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量文档分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量文档分类失败: {str(e)}")


@router.post("/async-batch", response_model=BaseResponse)
async def classify_batch_documents_async(
    request: BatchClassificationRequest,
    background_tasks: BackgroundTasks,
    service: ClassificationService = Depends(get_classification_service)
):
    """异步批量文档分类
    
    对大量文档进行异步批量分类，立即返回任务ID
    """
    try:
        logger.info(f"异步批量分类请求 - 项目: {request.project_id}, 文档数: {len(request.documents)}")
        
        # 验证请求数据
        if not request.documents:
            raise HTTPException(status_code=400, detail="文档列表不能为空")
        
        if len(request.documents) > 1000:
            raise HTTPException(status_code=400, detail="异步批量大小超过限制（最大1000个文档）")
        
        # 创建异步任务ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # 添加后台任务
        background_tasks.add_task(
            _process_async_batch_classification,
            task_id=task_id,
            request=request,
            service=service
        )
        
        logger.info(f"异步批量分类任务已创建: {task_id}")
        
        return BaseResponse(
            success=True,
            message="异步批量分类任务已创建",
            data={
                'task_id': task_id,
                'status': 'processing',
                'document_count': len(request.documents),
                'estimated_time': f"{len(request.documents) // 10 + 1}分钟"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建异步批量分类任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建异步批量分类任务失败: {str(e)}")


async def _process_async_batch_classification(
    task_id: str,
    request: BatchClassificationRequest,
    service: ClassificationService
):
    """处理异步批量分类任务"""
    try:
        logger.info(f"开始处理异步批量分类任务: {task_id}")
        
        # 执行批量分类
        result = await service.classify_batch_documents(request)
        
        # 这里可以将结果保存到数据库或缓存中
        # 实际实现中应该通过storage service保存结果
        logger.info(f"异步批量分类任务完成: {task_id}, 成功: {result.successful_classifications}")
        
    except Exception as e:
        logger.error(f"异步批量分类任务失败: {task_id}, 错误: {e}")


@router.get("/task/{task_id}", response_model=BaseResponse)
async def get_async_task_status(
    task_id: str,
    service: ClassificationService = Depends(get_classification_service)
):
    """获取异步任务状态
    
    查询异步批量分类任务的执行状态和结果
    """
    try:
        logger.info(f"查询异步任务状态: {task_id}")
        
        # 这里应该从storage service查询任务状态
        # 简化实现，返回模拟状态
        task_status = {
            'task_id': task_id,
            'status': 'completed',  # pending, processing, completed, failed
            'progress': 100,
            'created_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
            'result': {
                'total_documents': 50,
                'successful_classifications': 48,
                'failed_classifications': 2
            }
        }
        
        return BaseResponse(
            success=True,
            message="获取任务状态成功",
            data=task_status
        )
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.post("/predict-with-explanation", response_model=BaseResponse)
async def classify_with_detailed_explanation(
    request: ClassificationRequest,
    service: ClassificationService = Depends(get_classification_service)
):
    """带详细解释的文档分类
    
    执行分类并提供详细的分类原因解释
    """
    try:
        logger.info(f"带解释的文档分类请求 - 项目: {request.project_id}")
        
        # 强制启用解释和特征重要性
        request.return_explanation = True
        request.return_probabilities = True
        
        # 执行分类
        result = await service.classify_single_document(request)
        
        # 添加更详细的解释信息
        detailed_explanation = {
            'classification_result': result.dict(),
            'decision_process': {
                'step1': '文本预处理和特征提取',
                'step2': '模型预测计算',
                'step3': '置信度评估',
                'step4': '结果解释生成'
            },
            'model_confidence': {
                'level': 'high' if result.confidence_score > 0.8 else 'medium' if result.confidence_score > 0.6 else 'low',
                'score': result.confidence_score,
                'threshold': 0.6
            }
        }
        
        logger.info(f"带解释的分类完成 - 任务: {result.task_id}")
        
        return BaseResponse(
            success=True,
            message="带解释的文档分类成功",
            data=detailed_explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"带解释的文档分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"带解释的文档分类失败: {str(e)}")


@router.post("/compare-models", response_model=BaseResponse)
async def compare_model_predictions(
    project_id: str,
    text_content: str,
    model_ids: List[str],
    service: ClassificationService = Depends(get_classification_service)
):
    """比较多个模型的预测结果
    
    使用项目中的多个模型对同一文档进行分类，比较结果
    """
    try:
        logger.info(f"模型预测比较 - 项目: {project_id}, 模型数: {len(model_ids)}")
        
        if not model_ids:
            raise HTTPException(status_code=400, detail="必须指定至少一个模型ID")
        
        if len(model_ids) > 5:
            raise HTTPException(status_code=400, detail="最多比较5个模型")
        
        if not text_content or not text_content.strip():
            raise HTTPException(status_code=400, detail="文本内容不能为空")
        
        comparison_results = []
        
        # 对每个模型执行分类
        for model_id in model_ids:
            try:
                request = ClassificationRequest(
                    project_id=project_id,
                    text_content=text_content,
                    model_id=model_id,
                    return_probabilities=True,
                    return_explanation=False  # 比较时关闭解释以提升性能
                )
                
                result = await service.classify_single_document(request)
                comparison_results.append({
                    'model_id': model_id,
                    'predicted_label': result.predicted_label,
                    'confidence_score': result.confidence_score,
                    'probability_distribution': result.probability_distribution,
                    'processing_time': result.processing_time
                })
                
            except Exception as e:
                logger.warning(f"模型 {model_id} 预测失败: {e}")
                comparison_results.append({
                    'model_id': model_id,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # 分析比较结果
        successful_results = [r for r in comparison_results if 'error' not in r]
        
        if not successful_results:
            raise HTTPException(status_code=500, detail="所有模型预测均失败")
        
        # 计算一致性统计
        predictions = [r['predicted_label'] for r in successful_results]
        unique_predictions = list(set(predictions))
        
        consensus_analysis = {
            'total_models': len(model_ids),
            'successful_models': len(successful_results),
            'failed_models': len(model_ids) - len(successful_results),
            'unique_predictions': len(unique_predictions),
            'predictions': unique_predictions,
            'consensus': len(unique_predictions) == 1,
            'majority_prediction': max(set(predictions), key=predictions.count) if predictions else None
        }
        
        response_data = {
            'model_results': comparison_results,
            'consensus_analysis': consensus_analysis,
            'text_content': text_content[:200] + '...' if len(text_content) > 200 else text_content
        }
        
        logger.info(f"模型预测比较完成 - 一致性: {consensus_analysis['consensus']}")
        
        return BaseResponse(
            success=True,
            message="模型预测比较完成",
            data=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型预测比较失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型预测比较失败: {str(e)}")


@router.get("/history/{project_id}", response_model=BaseResponse)
async def get_classification_history(
    project_id: str,
    limit: int = 100,
    offset: int = 0,
    label_filter: Optional[str] = None,
    min_confidence: Optional[float] = None,
    service: ClassificationService = Depends(get_classification_service)
):
    """获取分类历史记录
    
    查询项目的分类历史，支持过滤和分页
    """
    try:
        logger.info(f"获取分类历史 - 项目: {project_id}")
        
        # 构建过滤条件
        filters = {}
        if label_filter:
            filters['predicted_label'] = label_filter
        if min_confidence is not None:
            filters['min_confidence'] = min_confidence
        
        # 获取分类历史（这里应该调用storage service）
        # 简化实现，返回模拟数据
        history_data = {
            'total_records': 150,
            'current_page': offset // limit + 1,
            'total_pages': (150 + limit - 1) // limit,
            'records': [
                {
                    'task_id': f'task_{i}',
                    'document_id': f'doc_{i}',
                    'predicted_label': '文化',
                    'confidence_score': 0.85 + (i % 10) * 0.01,
                    'created_at': datetime.now().isoformat(),
                    'model_id': 'model_123'
                }
                for i in range(min(limit, 150 - offset))
            ]
        }
        
        return BaseResponse(
            success=True,
            message="获取分类历史成功",
            data=history_data
        )
        
    except Exception as e:
        logger.error(f"获取分类历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取分类历史失败: {str(e)}")


@router.get("/statistics/{project_id}", response_model=BaseResponse)
async def get_classification_statistics(
    project_id: str,
    service: ClassificationService = Depends(get_classification_service)
):
    """获取分类统计信息
    
    获取项目的分类统计数据和性能指标
    """
    try:
        logger.info(f"获取分类统计 - 项目: {project_id}")
        
        statistics = await service.get_project_statistics(project_id)
        
        return BaseResponse(
            success=True,
            message="获取分类统计成功",
            data=statistics
        )
        
    except Exception as e:
        logger.error(f"获取分类统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取分类统计失败: {str(e)}")