"""
智能分类服务训练数据管理控制器
处理训练数据的添加、更新、查询和统计
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Dict, List, Any, Optional
import logging
import json
import csv
import io
from datetime import datetime

from ..clients.storage_client import StorageServiceClient
from ..schemas.classification_schemas import (
    BaseResponse, TrainingDataCreate, TrainingDataBatch, TrainingData
)
from ..utils.text_preprocessing import ChineseTextPreprocessor

# 创建路由器
router = APIRouter(prefix="/api/v1/data", tags=["训练数据管理"])
logger = logging.getLogger(__name__)


# 依赖注入
async def get_storage_client() -> StorageServiceClient:
    """获取存储服务客户端"""
    return StorageServiceClient()


def get_text_preprocessor() -> ChineseTextPreprocessor:
    """获取文本预处理器"""
    return ChineseTextPreprocessor()


@router.post("/training-data", response_model=BaseResponse)
async def add_training_data(
    training_data: TrainingDataCreate,
    storage_client: StorageServiceClient = Depends(get_storage_client),
    preprocessor: ChineseTextPreprocessor = Depends(get_text_preprocessor)
):
    """添加单条训练数据
    
    为项目添加新的训练数据样本
    """
    try:
        logger.info(f"添加训练数据 - 项目: {training_data.project_id}")
        
        # 验证项目存在
        project = await storage_client.get_classification_project(training_data.project_id)
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"项目 {training_data.project_id} 不存在"
            )
        
        # 验证标签是否在项目支持的标签列表中
        project_labels = project.get('class_labels', [])
        if project_labels and training_data.true_label not in project_labels:
            raise HTTPException(
                status_code=400,
                detail=f"标签 '{training_data.true_label}' 不在项目支持的标签列表中: {project_labels}"
            )
        
        # 文本预处理和特征提取
        try:
            processed_tokens = preprocessor.preprocess(training_data.text_content, return_tokens=True)
            text_stats = preprocessor.get_text_statistics(training_data.text_content)
        except Exception as e:
            logger.warning(f"文本预处理失败: {e}")
            processed_tokens = []
            text_stats = {}
        
        # 准备训练数据
        training_data_dict = training_data.dict()
        training_data_dict.update({
            'text_features': {
                'token_count': len(processed_tokens),
                'unique_tokens': len(set(processed_tokens)),
                'text_statistics': text_stats
            },
            'created_at': datetime.now().isoformat()
        })
        
        # 调用storage service添加训练数据
        result = await storage_client.create_training_data(training_data_dict)
        
        logger.info(f"训练数据添加成功，ID: {result.get('id', 'unknown')}")
        
        return BaseResponse(
            success=True,
            message="训练数据添加成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加训练数据失败: {str(e)}")


@router.post("/training-data/batch", response_model=BaseResponse)
async def add_training_data_batch(
    batch_data: TrainingDataBatch,
    storage_client: StorageServiceClient = Depends(get_storage_client),
    preprocessor: ChineseTextPreprocessor = Depends(get_text_preprocessor)
):
    """批量添加训练数据
    
    批量为项目添加多条训练数据
    """
    try:
        logger.info(f"批量添加训练数据 - 项目: {batch_data.project_id}, 数量: {len(batch_data.training_data)}")
        
        if len(batch_data.training_data) > 1000:
            raise HTTPException(
                status_code=400,
                detail="批量大小超过限制（最大1000条）"
            )
        
        # 验证项目存在
        project = await storage_client.get_classification_project(batch_data.project_id)
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"项目 {batch_data.project_id} 不存在"
            )
        
        project_labels = project.get('class_labels', [])
        
        # 处理每条训练数据
        processed_data = []
        failed_count = 0
        
        for i, data in enumerate(batch_data.training_data):
            try:
                # 验证必需字段
                if not data.get('text_content') or not data.get('true_label'):
                    logger.warning(f"第{i+1}条数据缺少必需字段")
                    failed_count += 1
                    continue
                
                # 验证标签
                if project_labels and data['true_label'] not in project_labels:
                    logger.warning(f"第{i+1}条数据标签无效: {data['true_label']}")
                    failed_count += 1
                    continue
                
                # 文本预处理
                try:
                    processed_tokens = preprocessor.preprocess(data['text_content'], return_tokens=True)
                    text_stats = preprocessor.get_text_statistics(data['text_content'])
                except Exception as e:
                    logger.warning(f"第{i+1}条数据预处理失败: {e}")
                    processed_tokens = []
                    text_stats = {}
                
                # 添加处理后的数据
                processed_item = {
                    'project_id': batch_data.project_id,
                    'text_content': data['text_content'],
                    'true_label': data['true_label'],
                    'document_id': data.get('document_id'),
                    'label_confidence': data.get('label_confidence', 1.0),
                    'data_source': data.get('data_source', 'batch_import'),
                    'text_features': {
                        'token_count': len(processed_tokens),
                        'unique_tokens': len(set(processed_tokens)),
                        'text_statistics': text_stats
                    },
                    'created_at': datetime.now().isoformat()
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"处理第{i+1}条数据失败: {e}")
                failed_count += 1
                continue
        
        if not processed_data:
            raise HTTPException(
                status_code=400,
                detail="没有有效的训练数据可以添加"
            )
        
        # 批量添加到storage service
        result = await storage_client.create_training_data_batch(processed_data)
        
        success_count = len(processed_data)
        logger.info(f"批量添加训练数据完成 - 成功: {success_count}, 失败: {failed_count}")
        
        return BaseResponse(
            success=True,
            message=f"批量添加训练数据完成，成功: {success_count}条，失败: {failed_count}条",
            data={
                'total_submitted': len(batch_data.training_data),
                'successful_added': success_count,
                'failed_count': failed_count,
                'batch_result': result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量添加训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量添加训练数据失败: {str(e)}")


@router.post("/training-data/upload", response_model=BaseResponse)
async def upload_training_data_file(
    project_id: str,
    file: UploadFile = File(...),
    storage_client: StorageServiceClient = Depends(get_storage_client),
    preprocessor: ChineseTextPreprocessor = Depends(get_text_preprocessor)
):
    """上传训练数据文件
    
    支持CSV、JSON格式的训练数据文件上传
    """
    try:
        logger.info(f"上传训练数据文件 - 项目: {project_id}, 文件: {file.filename}")
        
        # 验证文件类型
        if not file.filename:
            raise HTTPException(status_code=400, detail="未提供文件名")
        
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['csv', 'json']:
            raise HTTPException(
                status_code=400,
                detail="不支持的文件格式，仅支持CSV和JSON文件"
            )
        
        # 验证项目存在
        project = await storage_client.get_classification_project(project_id)
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"项目 {project_id} 不存在"
            )
        
        project_labels = project.get('class_labels', [])
        
        # 读取文件内容
        content = await file.read()
        
        # 解析文件内容
        training_data = []
        
        if file_extension == 'csv':
            # 解析CSV文件
            try:
                csv_content = content.decode('utf-8')
                csv_reader = csv.DictReader(io.StringIO(csv_content))
                
                for row in csv_reader:
                    if 'text_content' in row and 'true_label' in row:
                        training_data.append({
                            'text_content': row['text_content'].strip(),
                            'true_label': row['true_label'].strip(),
                            'document_id': row.get('document_id'),
                            'label_confidence': float(row.get('label_confidence', 1.0)),
                            'data_source': 'csv_upload'
                        })
                        
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV文件解析失败: {str(e)}"
                )
        
        elif file_extension == 'json':
            # 解析JSON文件
            try:
                json_content = content.decode('utf-8')
                data = json.loads(json_content)
                
                if isinstance(data, list):
                    training_data = data
                elif isinstance(data, dict) and 'training_data' in data:
                    training_data = data['training_data']
                else:
                    raise ValueError("JSON格式不正确，期望数组或包含training_data字段的对象")
                
                # 验证JSON数据格式
                for item in training_data:
                    if not isinstance(item, dict) or 'text_content' not in item or 'true_label' not in item:
                        raise ValueError("JSON数据格式不正确，每个项目需包含text_content和true_label字段")
                    
                    item.setdefault('data_source', 'json_upload')
                        
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"JSON文件解析失败: {str(e)}"
                )
        
        if not training_data:
            raise HTTPException(
                status_code=400,
                detail="文件中未找到有效的训练数据"
            )
        
        # 使用批量添加接口
        batch_request = TrainingDataBatch(
            project_id=project_id,
            training_data=training_data
        )
        
        return await add_training_data_batch(batch_request, storage_client, preprocessor)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传训练数据文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传训练数据文件失败: {str(e)}")


@router.get("/training-data/{project_id}", response_model=BaseResponse)
async def get_training_data(
    project_id: str,
    limit: int = 100,
    offset: int = 0,
    label_filter: Optional[str] = None,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """获取项目的训练数据
    
    分页获取项目的训练数据，支持标签过滤
    """
    try:
        logger.info(f"获取训练数据 - 项目: {project_id}, limit: {limit}, offset: {offset}")
        
        # 调用storage service获取训练数据
        result = await storage_client.get_training_data(
            project_id=project_id,
            limit=limit,
            offset=offset
        )
        
        # 如果有标签过滤，在结果中过滤
        if label_filter and result.get('data'):
            filtered_data = [
                item for item in result['data']
                if item.get('true_label') == label_filter
            ]
            result['data'] = filtered_data
            result['filtered_count'] = len(filtered_data)
        
        return BaseResponse(
            success=True,
            message="获取训练数据成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"获取训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练数据失败: {str(e)}")


@router.put("/training-data/{training_data_id}", response_model=BaseResponse)
async def update_training_data(
    training_data_id: str,
    update_data: Dict[str, Any],
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """更新训练数据
    
    更新指定的训练数据条目
    """
    try:
        logger.info(f"更新训练数据: {training_data_id}")
        
        # 添加更新时间戳
        update_data['updated_at'] = datetime.now().isoformat()
        
        # 如果更新了文本内容，重新进行预处理
        if 'text_content' in update_data:
            try:
                preprocessor = ChineseTextPreprocessor()
                processed_tokens = preprocessor.preprocess(update_data['text_content'], return_tokens=True)
                text_stats = preprocessor.get_text_statistics(update_data['text_content'])
                
                update_data['text_features'] = {
                    'token_count': len(processed_tokens),
                    'unique_tokens': len(set(processed_tokens)),
                    'text_statistics': text_stats
                }
            except Exception as e:
                logger.warning(f"文本预处理失败: {e}")
        
        # 调用storage service更新数据
        result = await storage_client.update_training_data(training_data_id, update_data)
        
        logger.info(f"训练数据更新成功: {training_data_id}")
        
        return BaseResponse(
            success=True,
            message="训练数据更新成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"更新训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新训练数据失败: {str(e)}")


@router.delete("/training-data/{training_data_id}", response_model=BaseResponse)
async def delete_training_data(
    training_data_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """删除训练数据
    
    删除指定的训练数据条目
    """
    try:
        logger.info(f"删除训练数据: {training_data_id}")
        
        # 调用storage service删除数据
        result = await storage_client.delete_training_data(training_data_id)
        
        logger.info(f"训练数据删除成功: {training_data_id}")
        
        return BaseResponse(
            success=True,
            message="训练数据删除成功",
            data=result
        )
        
    except Exception as e:
        logger.error(f"删除训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除训练数据失败: {str(e)}")


@router.get("/training-data/{project_id}/statistics", response_model=BaseResponse)
async def get_training_data_statistics(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """获取训练数据统计信息
    
    获取项目训练数据的统计分析
    """
    try:
        logger.info(f"获取训练数据统计 - 项目: {project_id}")
        
        # 调用storage service获取统计信息
        statistics = await storage_client.get_training_data_statistics(project_id)
        
        return BaseResponse(
            success=True,
            message="获取训练数据统计成功",
            data=statistics
        )
        
    except Exception as e:
        logger.error(f"获取训练数据统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练数据统计失败: {str(e)}")


@router.post("/training-data/{project_id}/export", response_model=BaseResponse)
async def export_training_data(
    project_id: str,
    export_format: str = "json",
    include_features: bool = False,
    storage_client: StorageServiceClient = Depends(get_storage_client)
):
    """导出训练数据
    
    导出项目的训练数据为指定格式
    """
    try:
        logger.info(f"导出训练数据 - 项目: {project_id}, 格式: {export_format}")
        
        if export_format not in ['json', 'csv']:
            raise HTTPException(
                status_code=400,
                detail="不支持的导出格式，仅支持json和csv"
            )
        
        # 调用storage service导出数据
        export_result = await storage_client.export_training_data(
            project_id=project_id,
            export_format=export_format
        )
        
        # 处理导出结果
        result = {
            'project_id': project_id,
            'export_format': export_format,
            'export_time': datetime.now().isoformat(),
            'include_features': include_features,
            'data_url': export_result.get('download_url'),  # 应该返回下载链接
            'record_count': export_result.get('record_count', 0)
        }
        
        logger.info(f"训练数据导出完成 - 项目: {project_id}")
        
        return BaseResponse(
            success=True,
            message="训练数据导出成功",
            data=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"导出训练数据失败: {str(e)}")


@router.post("/training-data/{project_id}/validate", response_model=BaseResponse)
async def validate_training_data(
    project_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client),
    preprocessor: ChineseTextPreprocessor = Depends(get_text_preprocessor)
):
    """验证训练数据质量
    
    检查项目训练数据的质量和一致性
    """
    try:
        logger.info(f"验证训练数据质量 - 项目: {project_id}")
        
        # 获取所有训练数据
        training_data_response = await storage_client.get_training_data(
            project_id=project_id,
            limit=10000
        )
        
        if not training_data_response or not training_data_response.get('data'):
            raise HTTPException(
                status_code=404,
                detail=f"项目 {project_id} 没有训练数据"
            )
        
        training_data = training_data_response['data']
        
        # 进行质量检查
        validation_results = {
            'total_records': len(training_data),
            'label_distribution': {},
            'text_length_stats': {
                'min_length': float('inf'),
                'max_length': 0,
                'avg_length': 0,
                'median_length': 0
            },
            'quality_issues': {
                'empty_texts': 0,
                'duplicate_texts': 0,
                'very_short_texts': 0,
                'very_long_texts': 0,
                'low_confidence_labels': 0
            },
            'recommendations': []
        }
        
        text_lengths = []
        text_contents = []
        
        for item in training_data:
            text_content = item.get('text_content', '')
            true_label = item.get('true_label', '')
            label_confidence = item.get('label_confidence', 1.0)
            
            # 统计标签分布
            validation_results['label_distribution'][true_label] = \
                validation_results['label_distribution'].get(true_label, 0) + 1
            
            # 统计文本长度
            text_length = len(text_content)
            text_lengths.append(text_length)
            text_contents.append(text_content)
            
            # 检查质量问题
            if not text_content.strip():
                validation_results['quality_issues']['empty_texts'] += 1
            elif text_length < 10:
                validation_results['quality_issues']['very_short_texts'] += 1
            elif text_length > 5000:
                validation_results['quality_issues']['very_long_texts'] += 1
            
            if label_confidence < 0.8:
                validation_results['quality_issues']['low_confidence_labels'] += 1
        
        # 检查重复文本
        unique_texts = set(text_contents)
        validation_results['quality_issues']['duplicate_texts'] = len(text_contents) - len(unique_texts)
        
        # 计算文本长度统计
        if text_lengths:
            text_lengths.sort()
            validation_results['text_length_stats'] = {
                'min_length': min(text_lengths),
                'max_length': max(text_lengths),
                'avg_length': sum(text_lengths) / len(text_lengths),
                'median_length': text_lengths[len(text_lengths) // 2]
            }
        
        # 生成建议
        recommendations = []
        
        if validation_results['quality_issues']['empty_texts'] > 0:
            recommendations.append("发现空文本，建议清理或补充内容")
        
        if validation_results['quality_issues']['duplicate_texts'] > 0:
            recommendations.append("发现重复文本，建议去重以避免过拟合")
        
        if validation_results['quality_issues']['very_short_texts'] > len(training_data) * 0.1:
            recommendations.append("短文本比例较高，可能影响分类效果")
        
        # 检查标签平衡性
        label_counts = list(validation_results['label_distribution'].values())
        if label_counts:
            max_count = max(label_counts)
            min_count = min(label_counts)
            if max_count > min_count * 5:
                recommendations.append("标签分布不平衡，建议增加少数类样本或使用采样技术")
        
        if len(validation_results['label_distribution']) < 2:
            recommendations.append("标签类别过少，无法进行有效分类")
        elif len(validation_results['label_distribution']) > 20:
            recommendations.append("标签类别过多，可能需要合并相似标签")
        
        validation_results['recommendations'] = recommendations
        
        # 计算数据质量分数（0-1）
        total_issues = sum(validation_results['quality_issues'].values())
        quality_score = max(0, 1 - (total_issues / len(training_data)))
        validation_results['quality_score'] = quality_score
        
        logger.info(f"训练数据验证完成 - 项目: {project_id}, 质量分数: {quality_score:.3f}")
        
        return BaseResponse(
            success=True,
            message="训练数据验证完成",
            data=validation_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证训练数据失败: {str(e)}")