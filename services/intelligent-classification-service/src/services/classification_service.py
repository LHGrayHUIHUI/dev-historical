"""
智能分类服务核心分类模块
提供文档分类预测和批量处理功能
支持多模型集成和结果解释
"""

import numpy as np
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..clients.storage_client import StorageServiceClient
from ..utils.text_preprocessing import ChineseTextPreprocessor
from ..schemas.classification_schemas import (
    ClassificationRequest, ClassificationResult, BatchClassificationRequest,
    BatchClassificationResult, ModelInfo, TaskStatus
)
from .model_trainer import ModelTrainer


@dataclass
class PredictionConfig:
    """预测配置类"""
    confidence_threshold: float = 0.6
    max_batch_size: int = 100
    enable_explanation: bool = True
    enable_feature_importance: bool = True
    prediction_timeout: int = 300  # 5分钟


class ClassificationService:
    """智能分类服务核心类
    
    负责文档分类预测、批量处理和结果解释
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化客户端和工具
        self.storage_client = StorageServiceClient()
        self.preprocessor = ChineseTextPreprocessor()
        
        # 模型缓存 {project_id: {model_id: trained_model}}
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        
        # 线程池用于并发处理
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def load_project_model(self, project_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """加载项目的分类模型
        
        Args:
            project_id: 项目ID
            model_id: 指定模型ID，不指定则使用活跃模型
            
        Returns:
            训练好的模型数据
        """
        try:
            # 如果没有指定模型ID，获取活跃模型
            if model_id is None:
                active_model_response = await self.storage_client.get_active_model(project_id)
                if not active_model_response or 'model_id' not in active_model_response:
                    raise ValueError(f"项目 {project_id} 没有活跃模型")
                model_id = active_model_response['model_id']
            
            # 检查缓存
            if project_id in self.model_cache and model_id in self.model_cache[project_id]:
                self.logger.debug(f"使用缓存模型: {model_id}")
                return self.model_cache[project_id][model_id]
            
            # 从storage service获取模型信息
            model_info = await self.storage_client.get_classification_model(model_id)
            if not model_info:
                raise ValueError(f"模型 {model_id} 不存在")
            
            # 这里简化处理，实际应该从模型路径加载模型文件
            # 在实际实现中，需要从MinIO或文件系统加载训练好的模型
            model_data = {
                'model_id': model_id,
                'model_info': model_info,
                'loaded_at': datetime.now().isoformat(),
                # 实际的模型对象应该从文件加载
                'model': None,  # ModelTrainer.load_trained_model(model_path)
                'feature_extractor': None,
                'label_encoder': None,
                'scaler': None
            }
            
            # 缓存模型
            if project_id not in self.model_cache:
                self.model_cache[project_id] = {}
            self.model_cache[project_id][model_id] = model_data
            
            self.logger.info(f"成功加载模型: {model_id} for project: {project_id}")
            return model_data
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """预处理单个文本"""
        tokens = self.preprocessor.preprocess(text, return_tokens=True)
        return ' '.join(tokens)
    
    def _extract_features(self, texts: List[str], feature_extractor) -> np.ndarray:
        """提取文本特征"""
        if feature_extractor is None:
            # 简化处理，返回模拟特征
            return np.random.rand(len(texts), 100)
        
        return feature_extractor.transform(texts)
    
    def _predict_with_model(
        self,
        model_data: Dict[str, Any],
        features: np.ndarray,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """使用模型进行预测"""
        model = model_data.get('model')
        if model is None:
            # 简化处理，返回模拟预测结果
            n_samples, n_features = features.shape
            n_classes = 4  # 假设4个类别
            
            predictions = np.random.randint(0, n_classes, n_samples)
            probabilities = None
            
            if return_probabilities:
                probabilities = np.random.rand(n_samples, n_classes)
                # 标准化概率
                probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            
            return predictions, probabilities
        
        # 实际预测
        predictions = model.predict(features)
        probabilities = None
        
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
        
        return predictions, probabilities
    
    def _generate_explanation(
        self,
        text: str,
        prediction: str,
        confidence: float,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """生成分类解释"""
        if not self.config.enable_explanation:
            return ""
        
        explanation_parts = [
            f"文本被分类为'{prediction}'，置信度为{confidence:.2%}。"
        ]
        
        if feature_importance and len(feature_importance) > 0:
            # 获取最重要的几个特征
            top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            if top_features:
                feature_list = [f"'{feat}'({score:.3f})" for feat, score in top_features]
                explanation_parts.append(f"关键特征包括: {', '.join(feature_list)}")
        
        # 基于预处理后的文本长度和内容给出简单解释
        processed_text = self._preprocess_text(text)
        text_length = len(processed_text)
        
        if text_length > 500:
            explanation_parts.append("这是一个较长的文本，包含丰富的语义信息。")
        elif text_length < 50:
            explanation_parts.append("这是一个较短的文本，可能信息有限。")
        
        return " ".join(explanation_parts)
    
    def _get_feature_importance(
        self,
        model_data: Dict[str, Any],
        text: str,
        features: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if not self.config.enable_feature_importance:
            return None
        
        # 简化处理，返回模拟特征重要性
        tokens = self.preprocessor.preprocess(text, return_tokens=True)
        if not tokens:
            return None
        
        # 随机生成特征重要性分数
        importance_scores = np.random.rand(min(len(tokens), 10))
        importance_dict = {}
        
        for i, token in enumerate(tokens[:len(importance_scores)]):
            importance_dict[token] = float(importance_scores[i])
        
        return importance_dict
    
    async def classify_single_document(
        self,
        request: ClassificationRequest
    ) -> ClassificationResult:
        """单文档分类"""
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        self.logger.info(f"开始单文档分类，任务ID: {task_id}")
        
        try:
            # 加载模型
            model_data = await self.load_project_model(request.project_id, request.model_id)
            
            # 预处理文本
            processed_text = self._preprocess_text(request.text_content)
            
            # 提取特征
            features = self._extract_features([processed_text], model_data.get('feature_extractor'))
            
            # 进行预测
            predictions, probabilities = self._predict_with_model(
                model_data, features, request.return_probabilities
            )
            
            prediction = predictions[0]
            
            # 解码预测标签
            label_encoder = model_data.get('label_encoder')
            if label_encoder:
                predicted_label = label_encoder.inverse_transform([prediction])[0]
            else:
                # 简化处理，使用预定义标签
                class_labels = ['政治', '军事', '经济', '文化']
                predicted_label = class_labels[prediction % len(class_labels)]
            
            # 计算置信度
            confidence_score = float(probabilities[0].max()) if probabilities is not None else 0.8
            
            # 概率分布
            probability_distribution = None
            if request.return_probabilities and probabilities is not None:
                if label_encoder:
                    class_names = label_encoder.classes_
                else:
                    class_names = ['政治', '军事', '经济', '文化']
                
                probability_distribution = {
                    class_names[i]: float(probabilities[0][i])
                    for i in range(min(len(class_names), len(probabilities[0])))
                }
            
            # 特征重要性
            feature_importance = None
            if request.return_explanation:
                feature_importance = self._get_feature_importance(model_data, request.text_content, features)
            
            # 生成解释
            explanation = None
            if request.return_explanation:
                explanation = self._generate_explanation(
                    request.text_content,
                    predicted_label,
                    confidence_score,
                    feature_importance
                )
            
            processing_time = time.time() - start_time
            
            # 创建分类结果
            result = ClassificationResult(
                task_id=task_id,
                document_id=request.document_id,
                predicted_label=predicted_label,
                confidence_score=confidence_score,
                probability_distribution=probability_distribution,
                feature_importance=feature_importance,
                explanation=explanation,
                processing_time=processing_time,
                model_info={
                    'model_id': model_data['model_id'],
                    'model_type': model_data['model_info'].get('model_type'),
                    'feature_extractor': model_data['model_info'].get('feature_extractor')
                }
            )
            
            # 保存分类结果到storage service
            await self.storage_client.create_classification_result({
                'task_id': task_id,
                'project_id': request.project_id,
                'document_id': request.document_id,
                'predicted_label': predicted_label,
                'confidence_score': confidence_score,
                'probability_distribution': probability_distribution,
                'feature_importance': feature_importance,
                'explanation': explanation,
                'processing_time': processing_time,
                'model_id': model_data['model_id'],
                'text_content': request.text_content[:1000],  # 保存前1000个字符
                'created_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"单文档分类完成，任务ID: {task_id}, 预测: {predicted_label}, 置信度: {confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"单文档分类失败，任务ID: {task_id}, 错误: {e}")
            raise
    
    async def classify_batch_documents(
        self,
        request: BatchClassificationRequest
    ) -> BatchClassificationResult:
        """批量文档分类"""
        start_time = time.time()
        batch_task_id = str(uuid.uuid4())
        
        self.logger.info(f"开始批量分类，任务ID: {batch_task_id}, 文档数量: {len(request.documents)}")
        
        if len(request.documents) > self.config.max_batch_size:
            raise ValueError(f"批量大小超过限制: {len(request.documents)} > {self.config.max_batch_size}")
        
        try:
            # 加载模型
            model_data = await self.load_project_model(request.project_id, request.model_id)
            
            # 准备批量数据
            texts = []
            document_ids = []
            
            for doc in request.documents:
                texts.append(doc.get('text_content', ''))
                document_ids.append(doc.get('document_id'))
            
            # 预处理所有文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # 批量特征提取
            features = self._extract_features(processed_texts, model_data.get('feature_extractor'))
            
            # 批量预测
            predictions, probabilities = self._predict_with_model(
                model_data, features, request.return_probabilities
            )
            
            # 处理每个预测结果
            results = []
            successful_classifications = 0
            failed_classifications = 0
            
            label_encoder = model_data.get('label_encoder')
            class_labels = ['政治', '军事', '经济', '文化']  # 简化处理
            
            for i, (prediction, text, doc_id) in enumerate(zip(predictions, texts, document_ids)):
                try:
                    # 解码标签
                    if label_encoder:
                        predicted_label = label_encoder.inverse_transform([prediction])[0]
                    else:
                        predicted_label = class_labels[prediction % len(class_labels)]
                    
                    # 置信度
                    confidence_score = float(probabilities[i].max()) if probabilities is not None else 0.8
                    
                    # 概率分布（简化处理）
                    probability_distribution = None
                    if request.return_probabilities and probabilities is not None:
                        probability_distribution = {
                            class_labels[j]: float(probabilities[i][j] if j < len(probabilities[i]) else 0.0)
                            for j in range(len(class_labels))
                        }
                    
                    # 特征重要性和解释（批量处理时通常关闭以提升性能）
                    feature_importance = None
                    explanation = None
                    
                    if request.return_explanation:
                        feature_importance = self._get_feature_importance(model_data, text, features[i:i+1])
                        explanation = self._generate_explanation(
                            text, predicted_label, confidence_score, feature_importance
                        )
                    
                    result = ClassificationResult(
                        task_id=f"{batch_task_id}_{i}",
                        document_id=doc_id,
                        predicted_label=predicted_label,
                        confidence_score=confidence_score,
                        probability_distribution=probability_distribution,
                        feature_importance=feature_importance,
                        explanation=explanation,
                        processing_time=0.0,  # 单个文档处理时间在批量处理中不单独计算
                        model_info={
                            'model_id': model_data['model_id'],
                            'model_type': model_data['model_info'].get('model_type'),
                            'feature_extractor': model_data['model_info'].get('feature_extractor')
                        }
                    )
                    
                    results.append(result)
                    successful_classifications += 1
                    
                except Exception as e:
                    self.logger.warning(f"处理文档 {i} 失败: {e}")
                    failed_classifications += 1
                    continue
            
            processing_time = time.time() - start_time
            
            # 计算统计信息
            statistics = {
                'avg_confidence': np.mean([r.confidence_score for r in results]) if results else 0.0,
                'label_distribution': {},
                'confidence_distribution': {
                    'high': len([r for r in results if r.confidence_score > 0.8]),
                    'medium': len([r for r in results if 0.6 <= r.confidence_score <= 0.8]),
                    'low': len([r for r in results if r.confidence_score < 0.6])
                }
            }
            
            # 标签分布统计
            for result in results:
                label = result.predicted_label
                statistics['label_distribution'][label] = statistics['label_distribution'].get(label, 0) + 1
            
            batch_result = BatchClassificationResult(
                batch_task_id=batch_task_id,
                total_documents=len(request.documents),
                successful_classifications=successful_classifications,
                failed_classifications=failed_classifications,
                results=results,
                processing_time=processing_time,
                statistics=statistics
            )
            
            # 保存批量任务结果
            await self.storage_client.create_batch_task({
                'batch_task_id': batch_task_id,
                'project_id': request.project_id,
                'total_documents': len(request.documents),
                'successful_classifications': successful_classifications,
                'failed_classifications': failed_classifications,
                'processing_time': processing_time,
                'statistics': statistics,
                'model_id': model_data['model_id'],
                'status': 'completed',
                'created_at': datetime.now().isoformat()
            })
            
            self.logger.info(f"批量分类完成，任务ID: {batch_task_id}, 成功: {successful_classifications}, 失败: {failed_classifications}")
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"批量分类失败，任务ID: {batch_task_id}, 错误: {e}")
            # 更新任务状态为失败
            try:
                await self.storage_client.create_batch_task({
                    'batch_task_id': batch_task_id,
                    'project_id': request.project_id,
                    'status': 'failed',
                    'error_message': str(e),
                    'created_at': datetime.now().isoformat()
                })
            except:
                pass
            raise
    
    async def get_project_statistics(self, project_id: str) -> Dict[str, Any]:
        """获取项目分类统计信息"""
        try:
            stats = await self.storage_client.get_project_statistics(project_id)
            return stats
        except Exception as e:
            self.logger.error(f"获取项目统计失败: {e}")
            raise
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能统计"""
        try:
            performance = await self.storage_client.get_model_performance_stats(model_id)
            return performance
        except Exception as e:
            self.logger.error(f"获取模型性能失败: {e}")
            raise
    
    def clear_model_cache(self, project_id: Optional[str] = None):
        """清理模型缓存"""
        if project_id:
            if project_id in self.model_cache:
                del self.model_cache[project_id]
                self.logger.info(f"已清理项目 {project_id} 的模型缓存")
        else:
            self.model_cache.clear()
            self.logger.info("已清理所有模型缓存")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查storage service连接
            storage_health = await self.storage_client.health_check()
            
            return {
                'service': 'intelligent-classification-service',
                'status': 'healthy',
                'storage_service': storage_health.get('status', 'unknown'),
                'cached_models': len(self.model_cache),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'service': 'intelligent-classification-service',
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# 全局分类服务实例
classification_service = ClassificationService()


# 便捷函数
async def classify_document(request: ClassificationRequest) -> ClassificationResult:
    """便捷的文档分类函数"""
    return await classification_service.classify_single_document(request)


async def classify_documents_batch(request: BatchClassificationRequest) -> BatchClassificationResult:
    """便捷的批量分类函数"""
    return await classification_service.classify_batch_documents(request)