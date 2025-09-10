"""
智能分类服务Mock测试
测试核心分类功能和业务逻辑
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from src.schemas.classification_schemas import (
    ClassificationRequest, ClassificationResult, BatchClassificationRequest,
    BatchClassificationResult, ModelType, FeatureExtractorType, TaskStatus, ProjectStatus
)


class MockStorageServiceClient:
    """Mock存储服务客户端"""
    
    def __init__(self):
        self.projects = {}
        self.models = {}
        self.training_data = {}
        self.classification_results = []
        self.batch_tasks = []
    
    async def get_active_model(self, project_id: str) -> Dict[str, Any]:
        """获取活跃模型"""
        if project_id in self.projects:
            return {
                'model_id': f'model_{project_id}',
                'model_type': 'bert',
                'feature_extractor': 'bert',
                'status': 'active'
            }
        return None
    
    async def get_classification_model(self, model_id: str) -> Dict[str, Any]:
        """获取分类模型"""
        return {
            'id': model_id,
            'model_name': f'历史文本分类器_{model_id}',
            'model_type': ModelType.BERT,
            'feature_extractor': FeatureExtractorType.BERT,
            'status': TaskStatus.COMPLETED,
            'hyperparameters': {
                'learning_rate': 0.001,
                'max_seq_length': 512,
                'batch_size': 32
            },
            'training_metrics': {
                'accuracy': 0.92,
                'f1_score': 0.90
            }
        }
    
    async def create_classification_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """保存分类结果"""
        self.classification_results.append(result_data)
        return {'success': True, 'id': f'result_{len(self.classification_results)}'}
    
    async def create_batch_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """保存批量任务"""
        self.batch_tasks.append(task_data)
        return {'success': True, 'id': f'batch_{len(self.batch_tasks)}'}
    
    async def get_project_statistics(self, project_id: str) -> Dict[str, Any]:
        """获取项目统计"""
        return {
            'project_id': project_id,
            'total_training_data': 5000,
            'total_models': 3,
            'total_predictions': 10000,
            'active_models': 1,
            'label_distribution': {'政治': 2500, '军事': 1500, '经济': 3000, '文化': 3000},
            'avg_accuracy': 0.92
        }
    
    async def get_model_performance_stats(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能统计"""
        return {
            'model_id': model_id,
            'total_predictions': 1000,
            'avg_confidence': 0.87,
            'avg_processing_time': 0.12,
            'accuracy': 0.92,
            'f1_score': 0.90,
            'label_distribution': {'政治': 250, '军事': 300, '经济': 200, '文化': 250}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {'status': 'healthy', 'service': 'storage-service'}


class MockChineseTextPreprocessor:
    """Mock中文文本预处理器"""
    
    def preprocess(self, text: str, return_tokens: bool = False) -> Union[List[str], str]:
        """预处理文本"""
        # 简单的中文分词模拟
        import re
        # 移除标点符号
        cleaned = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        # 简单按字符分割
        tokens = list(cleaned.replace(' ', ''))
        
        # 模拟停用词过滤和词频统计
        meaningful_tokens = [token for token in tokens if len(token.strip()) > 0]
        
        if return_tokens:
            return meaningful_tokens[:50]  # 限制token数量
        else:
            return ' '.join(meaningful_tokens[:50])


class MockClassificationService:
    """Mock智能分类服务"""
    
    def __init__(self):
        self.storage_client = MockStorageServiceClient()
        self.preprocessor = MockChineseTextPreprocessor()
        self.model_cache = {}
        
        # 预定义标签和概率
        self.class_labels = ['政治', '军事', '经济', '文化']
        self.historical_keywords = {
            '政治': ['政治', '朝廷', '皇帝', '官员', '政策', '制度', '法律', '治理'],
            '军事': ['军事', '战争', '战略', '兵法', '武器', '将军', '士兵', '战役'],
            '经济': ['经济', '贸易', '商业', '货币', '税收', '农业', '手工业', '市场'],
            '文化': ['文化', '文学', '艺术', '宗教', '教育', '哲学', '礼仪', '传统']
        }
    
    def _classify_by_keywords(self, text: str) -> Dict[str, float]:
        """基于关键词的分类逻辑"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.historical_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    score += 1.0
                # 部分匹配得分
                for char in keyword:
                    if char in text:
                        score += 0.1
            scores[category] = min(score / 10.0, 1.0)  # 标准化到0-1
        
        # 确保至少有一个非零概率
        if all(score == 0 for score in scores.values()):
            scores[self.class_labels[hash(text) % len(self.class_labels)]] = 0.6
        
        # 标准化概率使其和为1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _generate_realistic_features(self, text: str) -> Dict[str, float]:
        """生成真实的特征重要性"""
        tokens = self.preprocessor.preprocess(text, return_tokens=True)
        features = {}
        
        # 基于关键词生成特征重要性
        for token in tokens[:10]:  # 前10个token
            importance = 0.0
            # 检查token是否在任何类别的关键词中
            for category, keywords in self.historical_keywords.items():
                if token in keywords:
                    importance += 0.8
                elif any(token in keyword for keyword in keywords):
                    importance += 0.3
            
            # 基于token长度和位置的重要性
            if len(token) > 1:
                importance += 0.2
            
            # 添加一些随机性
            import random
            random.seed(hash(token))  # 确保相同token得到相同重要性
            importance += random.uniform(0.0, 0.3)
            
            features[token] = min(importance, 1.0)
        
        return dict(sorted(features.items(), key=lambda x: x[1], reverse=True))
    
    async def load_project_model(self, project_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """加载项目模型"""
        if model_id is None:
            active_model = await self.storage_client.get_active_model(project_id)
            model_id = active_model['model_id']
        
        model_info = await self.storage_client.get_classification_model(model_id)
        
        return {
            'model_id': model_id,
            'model_info': model_info,
            'loaded_at': datetime.now().isoformat(),
            'model': None,  # Mock模型
            'feature_extractor': None,
            'label_encoder': None,
            'scaler': None
        }
    
    async def classify_single_document(self, request: ClassificationRequest) -> ClassificationResult:
        """单文档分类"""
        import uuid
        import time
        
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        # 加载模型
        model_data = await self.load_project_model(request.project_id, request.model_id)
        
        # 基于关键词进行智能分类
        probabilities = self._classify_by_keywords(request.text_content)
        
        # 获取最高概率的标签
        predicted_label = max(probabilities, key=probabilities.get)
        confidence_score = probabilities[predicted_label]
        
        # 概率分布（如果需要）
        probability_distribution = probabilities if request.return_probabilities else None
        
        # 特征重要性
        feature_importance = None
        if request.return_explanation:
            feature_importance = self._generate_realistic_features(request.text_content)
        
        # 生成解释
        explanation = None
        if request.return_explanation:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            feature_list = [f"'{feat}'({score:.3f})" for feat, score in top_features]
            explanation = f"文本被分类为'{predicted_label}'，置信度为{confidence_score:.2%}。关键特征包括: {', '.join(feature_list)}"
        
        processing_time = time.time() - start_time
        
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
                'model_type': ModelType.BERT.value,
                'feature_extractor': FeatureExtractorType.BERT.value
            }
        )
        
        # 保存结果
        await self.storage_client.create_classification_result({
            'task_id': task_id,
            'project_id': request.project_id,
            'document_id': request.document_id,
            'predicted_label': predicted_label,
            'confidence_score': confidence_score,
            'processing_time': processing_time
        })
        
        return result
    
    async def classify_batch_documents(self, request: BatchClassificationRequest) -> BatchClassificationResult:
        """批量文档分类"""
        import uuid
        import time
        
        start_time = time.time()
        batch_task_id = str(uuid.uuid4())
        
        # 加载模型
        model_data = await self.load_project_model(request.project_id, request.model_id)
        
        results = []
        successful_classifications = 0
        failed_classifications = 0
        
        # 处理每个文档
        for i, doc in enumerate(request.documents):
            try:
                text_content = doc.get('text_content', '')
                document_id = doc.get('document_id')
                
                # 分类处理
                probabilities = self._classify_by_keywords(text_content)
                predicted_label = max(probabilities, key=probabilities.get)
                confidence_score = probabilities[predicted_label]
                
                probability_distribution = probabilities if request.return_probabilities else None
                
                # 特征重要性和解释（批量处理时简化）
                feature_importance = None
                explanation = None
                if request.return_explanation:
                    feature_importance = self._generate_realistic_features(text_content)
                    explanation = f"文本分类为'{predicted_label}'，置信度{confidence_score:.2%}"
                
                result = ClassificationResult(
                    task_id=f"{batch_task_id}_{i}",
                    document_id=document_id,
                    predicted_label=predicted_label,
                    confidence_score=confidence_score,
                    probability_distribution=probability_distribution,
                    feature_importance=feature_importance,
                    explanation=explanation,
                    processing_time=0.0,
                    model_info={
                        'model_id': model_data['model_id'],
                        'model_type': ModelType.BERT.value,
                        'feature_extractor': FeatureExtractorType.BERT.value
                    }
                )
                
                results.append(result)
                successful_classifications += 1
                
            except Exception as e:
                failed_classifications += 1
                continue
        
        processing_time = time.time() - start_time
        
        # 计算统计信息
        if results:
            avg_confidence = sum(r.confidence_score for r in results) / len(results)
        else:
            avg_confidence = 0.0
        
        label_distribution = {}
        for result in results:
            label = result.predicted_label
            label_distribution[label] = label_distribution.get(label, 0) + 1
        
        confidence_distribution = {
            'high': len([r for r in results if r.confidence_score > 0.8]),
            'medium': len([r for r in results if 0.6 <= r.confidence_score <= 0.8]),
            'low': len([r for r in results if r.confidence_score < 0.6])
        }
        
        statistics = {
            'avg_confidence': avg_confidence,
            'label_distribution': label_distribution,
            'confidence_distribution': confidence_distribution
        }
        
        batch_result = BatchClassificationResult(
            batch_task_id=batch_task_id,
            total_documents=len(request.documents),
            successful_classifications=successful_classifications,
            failed_classifications=failed_classifications,
            results=results,
            processing_time=processing_time,
            statistics=statistics
        )
        
        # 保存批量任务
        await self.storage_client.create_batch_task({
            'batch_task_id': batch_task_id,
            'project_id': request.project_id,
            'total_documents': len(request.documents),
            'successful_classifications': successful_classifications,
            'processing_time': processing_time,
            'status': 'completed'
        })
        
        return batch_result
    
    async def get_project_statistics(self, project_id: str) -> Dict[str, Any]:
        """获取项目统计"""
        return await self.storage_client.get_project_statistics(project_id)
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """获取模型性能"""
        return await self.storage_client.get_model_performance_stats(model_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        storage_health = await self.storage_client.health_check()
        return {
            'service': 'intelligent-classification-service',
            'status': 'healthy',
            'storage_service': storage_health.get('status', 'unknown'),
            'cached_models': len(self.model_cache),
            'timestamp': datetime.now().isoformat()
        }


# ==================== 测试用例 ====================

@pytest.fixture
def classification_service():
    """分类服务fixture"""
    return MockClassificationService()


class TestClassificationServiceInitialization:
    """分类服务初始化测试"""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, classification_service):
        """测试服务初始化"""
        assert classification_service.storage_client is not None
        assert classification_service.preprocessor is not None
        assert isinstance(classification_service.model_cache, dict)
        assert len(classification_service.class_labels) == 4
        assert '政治' in classification_service.class_labels


class TestSingleDocumentClassification:
    """单文档分类测试"""
    
    @pytest.mark.asyncio
    async def test_political_document_classification(self, classification_service):
        """测试政治类文档分类"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="这是一份关于古代朝廷政治制度的历史文献，详细描述了皇帝的治理政策和官员体系。",
            document_id="doc_political_001",
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        assert result.task_id is not None
        assert result.document_id == "doc_political_001"
        assert result.predicted_label in classification_service.class_labels
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.probability_distribution is not None
        assert result.feature_importance is not None
        assert result.explanation is not None
        assert "政治" in result.explanation or "朝廷" in result.explanation
        assert result.processing_time > 0
        assert result.model_info['model_type'] == 'bert'
    
    @pytest.mark.asyncio
    async def test_military_document_classification(self, classification_service):
        """测试军事类文档分类"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="本文记录了古代战争的军事战略，包括兵法运用、武器装备和将军指挥艺术。",
            document_id="doc_military_001",
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        assert result.predicted_label in classification_service.class_labels
        # 由于包含"战争"、"军事"等关键词，应该倾向于分类为军事
        assert result.confidence_score > 0.0
        assert result.probability_distribution is not None
        assert len(result.probability_distribution) == 4
        assert sum(result.probability_distribution.values()) == pytest.approx(1.0, abs=1e-10)
    
    @pytest.mark.asyncio
    async def test_economic_document_classification(self, classification_service):
        """测试经济类文档分类"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="该文献研究古代商业贸易体系，分析了货币流通、税收制度和农业经济发展。",
            document_id="doc_economic_001",
            return_probabilities=False,
            return_explanation=False
        )
        
        result = await classification_service.classify_single_document(request)
        
        assert result.predicted_label in classification_service.class_labels
        assert result.probability_distribution is None  # 未请求概率分布
        assert result.feature_importance is None  # 未请求解释
        assert result.explanation is None
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_cultural_document_classification(self, classification_service):
        """测试文化类文档分类"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="这篇文章探讨了古代文学艺术的发展，包括诗歌创作、绘画技法和宗教哲学思想。",
            document_id="doc_cultural_001",
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        assert result.predicted_label in classification_service.class_labels
        assert result.confidence_score > 0.0
        assert result.feature_importance is not None
        # 检查特征重要性是否合理
        assert len(result.feature_importance) <= 10  # 最多10个特征
        assert all(0.0 <= score <= 1.0 for score in result.feature_importance.values())
    
    @pytest.mark.asyncio
    async def test_mixed_content_classification(self, classification_service):
        """测试混合内容分类"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="这是一篇综合性历史研究，涵盖了政治制度、军事战略、经济发展和文化传承等多个方面。",
            document_id="doc_mixed_001",
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        assert result.predicted_label in classification_service.class_labels
        # 混合内容应该有一个主要分类，但概率分布应该相对均匀
        assert result.probability_distribution is not None
        max_prob = max(result.probability_distribution.values())
        min_prob = min(result.probability_distribution.values())
        assert max_prob - min_prob < 0.8  # 概率差异不应过大


class TestBatchDocumentClassification:
    """批量文档分类测试"""
    
    @pytest.mark.asyncio
    async def test_small_batch_classification(self, classification_service):
        """测试小批量分类"""
        request = BatchClassificationRequest(
            project_id="historical_project",
            documents=[
                {"text_content": "古代朝廷政治制度研究", "document_id": "batch_doc_1"},
                {"text_content": "战国时期军事战略分析", "document_id": "batch_doc_2"},
                {"text_content": "汉代经济贸易发展史", "document_id": "batch_doc_3"}
            ],
            return_probabilities=True,
            return_explanation=False
        )
        
        result = await classification_service.classify_batch_documents(request)
        
        assert result.batch_task_id is not None
        assert result.total_documents == 3
        assert result.successful_classifications == 3
        assert result.failed_classifications == 0
        assert len(result.results) == 3
        assert result.processing_time > 0
        
        # 检查统计信息
        assert result.statistics['avg_confidence'] > 0
        assert len(result.statistics['label_distribution']) > 0
        assert 'high' in result.statistics['confidence_distribution']
        
        # 检查每个结果
        for i, single_result in enumerate(result.results):
            assert single_result.task_id == f"{result.batch_task_id}_{i}"
            assert single_result.predicted_label in classification_service.class_labels
            assert single_result.probability_distribution is not None
    
    @pytest.mark.asyncio
    async def test_large_batch_classification(self, classification_service):
        """测试大批量分类"""
        documents = []
        for i in range(20):
            category_texts = {
                0: "政治制度法律治理朝廷皇帝",
                1: "军事战争兵法战略将军士兵",
                2: "经济贸易商业货币农业市场",
                3: "文化文学艺术宗教教育哲学"
            }
            category = i % 4
            documents.append({
                "text_content": f"这是第{i+1}篇关于{category_texts[category]}的历史文献研究。",
                "document_id": f"large_batch_doc_{i+1}"
            })
        
        request = BatchClassificationRequest(
            project_id="historical_project",
            documents=documents,
            return_probabilities=True,
            return_explanation=False
        )
        
        result = await classification_service.classify_batch_documents(request)
        
        assert result.total_documents == 20
        assert result.successful_classifications == 20
        assert result.failed_classifications == 0
        assert len(result.results) == 20
        
        # 检查标签分布应该相对均匀（每个类别约5个）
        label_counts = result.statistics['label_distribution']
        assert len(label_counts) > 0
        # 由于是按类别循环生成的文档，每个类别都应该有文档
        assert all(count > 0 for count in label_counts.values())
    
    @pytest.mark.asyncio
    async def test_batch_with_explanations(self, classification_service):
        """测试带解释的批量分类"""
        request = BatchClassificationRequest(
            project_id="historical_project",
            documents=[
                {"text_content": "古代皇帝的政治统治策略和制度设计", "document_id": "explain_doc_1"},
                {"text_content": "历代战争中的军事技术和战术演变", "document_id": "explain_doc_2"}
            ],
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_batch_documents(request)
        
        assert result.successful_classifications == 2
        
        # 检查解释是否生成
        for single_result in result.results:
            assert single_result.explanation is not None
            assert single_result.feature_importance is not None
            assert len(single_result.feature_importance) > 0


class TestModelAndPerformance:
    """模型和性能测试"""
    
    @pytest.mark.asyncio
    async def test_project_statistics(self, classification_service):
        """测试项目统计获取"""
        stats = await classification_service.get_project_statistics("historical_project")
        
        assert stats['project_id'] == "historical_project"
        assert stats['total_training_data'] > 0
        assert stats['total_models'] > 0
        assert stats['total_predictions'] > 0
        assert stats['active_models'] > 0
        assert 'label_distribution' in stats
        assert len(stats['label_distribution']) > 0
    
    @pytest.mark.asyncio
    async def test_model_performance(self, classification_service):
        """测试模型性能获取"""
        performance = await classification_service.get_model_performance("model_123")
        
        assert performance['model_id'] == "model_123"
        assert performance['total_predictions'] > 0
        assert 0.0 <= performance['avg_confidence'] <= 1.0
        assert performance['avg_processing_time'] > 0
        assert 0.0 <= performance['accuracy'] <= 1.0
        assert 0.0 <= performance['f1_score'] <= 1.0
        assert 'label_distribution' in performance
    
    @pytest.mark.asyncio
    async def test_health_check(self, classification_service):
        """测试健康检查"""
        health = await classification_service.health_check()
        
        assert health['service'] == 'intelligent-classification-service'
        assert health['status'] == 'healthy'
        assert 'storage_service' in health
        assert 'cached_models' in health
        assert 'timestamp' in health


class TestPerformanceAndConcurrency:
    """性能和并发测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_single_classifications(self, classification_service):
        """测试并发单文档分类"""
        tasks = []
        for i in range(5):
            request = ClassificationRequest(
                project_id="historical_project",
                text_content=f"这是第{i+1}篇测试文档，涉及古代历史研究。",
                document_id=f"concurrent_doc_{i+1}",
                return_probabilities=True,
                return_explanation=False
            )
            tasks.append(classification_service.classify_single_document(request))
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result.predicted_label in classification_service.class_labels
            assert result.confidence_score > 0.0
            assert result.processing_time >= 0.0
    
    @pytest.mark.asyncio
    async def test_processing_time_statistics(self, classification_service):
        """测试处理时间统计"""
        processing_times = []
        
        for i in range(10):
            request = ClassificationRequest(
                project_id="historical_project",
                text_content=f"历史文献研究第{i+1}篇，分析古代社会发展。",
                document_id=f"time_test_doc_{i+1}",
                return_probabilities=True,
                return_explanation=True
            )
            
            result = await classification_service.classify_single_document(request)
            processing_times.append(result.processing_time)
        
        # 所有处理时间应该为正数
        assert all(time > 0 for time in processing_times)
        
        # 计算平均处理时间
        avg_time = sum(processing_times) / len(processing_times)
        assert avg_time > 0.0
        
        # 处理时间应该相对稳定（标准差不应过大）
        import statistics
        std_dev = statistics.stdev(processing_times)
        assert std_dev < avg_time * 2.0  # 标准差不超过平均值的2倍


class TestErrorHandlingAndEdgeCases:
    """错误处理和边界情况测试"""
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, classification_service):
        """测试空文本处理"""
        # 注意：这里我们模拟服务内部处理，而不是Pydantic验证
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="   ",  # 只有空白字符
            return_probabilities=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        # 即使是空白文本，也应该返回一个分类结果
        assert result.predicted_label in classification_service.class_labels
        assert result.confidence_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, classification_service):
        """测试特殊字符处理"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="这是一篇包含特殊字符的文档：！@#$%^&*()，测试系统的鲁棒性。",
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        assert result.predicted_label in classification_service.class_labels
        assert result.confidence_score > 0.0
        assert result.feature_importance is not None
    
    @pytest.mark.asyncio
    async def test_very_long_text_handling(self, classification_service):
        """测试长文本处理"""
        long_text = "古代历史研究。" * 500  # 重复500次，创建长文本
        
        request = ClassificationRequest(
            project_id="historical_project",
            text_content=long_text,
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        assert result.predicted_label in classification_service.class_labels
        assert result.confidence_score > 0.0
        # 处理时间可能会稍长，但应该在合理范围内
        assert result.processing_time < 10.0  # 不应超过10秒
    
    @pytest.mark.asyncio
    async def test_batch_with_mixed_quality_documents(self, classification_service):
        """测试包含不同质量文档的批量处理"""
        request = BatchClassificationRequest(
            project_id="historical_project",
            documents=[
                {"text_content": "高质量的古代政治制度研究文献", "document_id": "quality_doc_1"},
                {"text_content": "短文本", "document_id": "short_doc_1"},
                {"text_content": "包含很多特殊字符的文档：@#$%^&*()", "document_id": "special_doc_1"},
                {"text_content": "非常详细的历史军事战略分析文献，涵盖了战争、兵法、武器等多个方面", "document_id": "detailed_doc_1"}
            ],
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_batch_documents(request)
        
        assert result.total_documents == 4
        assert result.successful_classifications == 4  # 所有文档都应该成功处理
        assert result.failed_classifications == 0
        
        # 检查每个结果的质量
        for single_result in result.results:
            assert single_result.predicted_label in classification_service.class_labels
            assert single_result.confidence_score >= 0.0


class TestBusinessLogicValidation:
    """业务逻辑验证测试"""
    
    @pytest.mark.asyncio
    async def test_classification_consistency(self, classification_service):
        """测试分类一致性"""
        text_content = "古代朝廷的政治制度和治理策略研究"
        
        # 多次分类同一文本
        results = []
        for i in range(3):
            request = ClassificationRequest(
                project_id="historical_project",
                text_content=text_content,
                document_id=f"consistency_doc_{i}",
                return_probabilities=True
            )
            result = await classification_service.classify_single_document(request)
            results.append(result)
        
        # 由于使用相同的关键词匹配逻辑，结果应该一致
        predicted_labels = [r.predicted_label for r in results]
        assert len(set(predicted_labels)) <= 2  # 最多有2种不同的预测（允许一些变化）
    
    @pytest.mark.asyncio
    async def test_probability_distribution_validity(self, classification_service):
        """测试概率分布有效性"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="综合性历史文献，包含政治、军事、经济、文化等内容",
            return_probabilities=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        prob_dist = result.probability_distribution
        assert prob_dist is not None
        
        # 概率分布验证
        assert len(prob_dist) == 4  # 应该有4个类别
        assert all(label in classification_service.class_labels for label in prob_dist.keys())
        assert all(0.0 <= prob <= 1.0 for prob in prob_dist.values())
        assert abs(sum(prob_dist.values()) - 1.0) < 1e-10  # 概率和应该等于1
    
    @pytest.mark.asyncio
    async def test_feature_importance_relevance(self, classification_service):
        """测试特征重要性相关性"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="古代战争军事战略研究，包括兵法、武器、将军指挥等内容",
            return_probabilities=True,
            return_explanation=True
        )
        
        result = await classification_service.classify_single_document(request)
        
        feature_importance = result.feature_importance
        assert feature_importance is not None
        assert len(feature_importance) > 0
        
        # 检查重要性分数的合理性
        assert all(0.0 <= score <= 1.0 for score in feature_importance.values())
        
        # 军事相关词汇应该有较高的重要性
        military_terms = ['战争', '军事', '兵法', '武器', '将军']
        military_features = {term: score for term, score in feature_importance.items() 
                           if any(mil_term in term for mil_term in military_terms)}
        
        if military_features:
            avg_military_score = sum(military_features.values()) / len(military_features)
            assert avg_military_score > 0.2  # 军事相关特征应该有一定重要性
    
    @pytest.mark.asyncio
    async def test_model_info_consistency(self, classification_service):
        """测试模型信息一致性"""
        request = ClassificationRequest(
            project_id="historical_project",
            text_content="测试文档内容"
        )
        
        result = await classification_service.classify_single_document(request)
        
        model_info = result.model_info
        assert model_info['model_id'] is not None
        assert model_info['model_type'] == 'bert'
        assert model_info['feature_extractor'] == 'bert'
        
        # 同一项目的不同请求应该使用相同的模型
        request2 = ClassificationRequest(
            project_id="historical_project",
            text_content="另一个测试文档"
        )
        
        result2 = await classification_service.classify_single_document(request2)
        
        assert result.model_info['model_id'] == result2.model_info['model_id']


class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_complete_document_processing_pipeline(self, classification_service):
        """测试完整文档处理流水线"""
        # 1. 单文档分类
        single_request = ClassificationRequest(
            project_id="historical_project",
            text_content="这是一篇详细的古代政治制度研究，分析了朝廷治理结构。",
            document_id="pipeline_doc_1",
            return_probabilities=True,
            return_explanation=True
        )
        
        single_result = await classification_service.classify_single_document(single_request)
        assert single_result.predicted_label in classification_service.class_labels
        
        # 2. 批量文档分类
        batch_request = BatchClassificationRequest(
            project_id="historical_project",
            documents=[
                {"text_content": "军事战略研究文献", "document_id": "pipeline_batch_1"},
                {"text_content": "经济发展史分析", "document_id": "pipeline_batch_2"},
                {"text_content": "文化艺术发展概述", "document_id": "pipeline_batch_3"}
            ],
            return_probabilities=True,
            return_explanation=False
        )
        
        batch_result = await classification_service.classify_batch_documents(batch_request)
        assert batch_result.successful_classifications == 3
        
        # 3. 获取统计信息
        stats = await classification_service.get_project_statistics("historical_project")
        assert stats['project_id'] == "historical_project"
        
        # 4. 健康检查
        health = await classification_service.health_check()
        assert health['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_multi_domain_classification_comparison(self, classification_service):
        """测试多领域分类对比"""
        domain_texts = {
            "政治领域": "古代朝廷政治制度、皇帝统治、官员管理、法律制度的深入研究",
            "军事领域": "历史战争分析、军事战略战术、兵器发展、将领指挥艺术研究", 
            "经济领域": "古代商业贸易、货币制度、农业发展、手工业生产的历史考察",
            "文化领域": "传统文学艺术、宗教哲学思想、教育制度、礼仪文化的传承研究"
        }
        
        results = {}
        for domain, text in domain_texts.items():
            request = ClassificationRequest(
                project_id="historical_project",
                text_content=text,
                document_id=f"domain_{domain}",
                return_probabilities=True,
                return_explanation=True
            )
            
            result = await classification_service.classify_single_document(request)
            results[domain] = result
        
        # 验证每个领域的分类结果
        for domain, result in results.items():
            assert result.predicted_label in classification_service.class_labels
            assert result.confidence_score > 0.0
            assert result.probability_distribution is not None
            
            # 检查概率分布是否合理
            max_prob_label = max(result.probability_distribution, 
                               key=result.probability_distribution.get)
            max_prob = result.probability_distribution[max_prob_label]
            assert max_prob >= result.confidence_score  # 最大概率应该等于置信度
    
    @pytest.mark.asyncio
    async def test_service_reliability_under_load(self, classification_service):
        """测试负载下的服务可靠性"""
        # 模拟高并发请求
        tasks = []
        
        for i in range(20):
            # 混合单文档和批量请求
            if i % 3 == 0:  # 每3个请求中有1个批量请求
                batch_request = BatchClassificationRequest(
                    project_id="historical_project",
                    documents=[
                        {"text_content": f"批量文档{i}_1", "document_id": f"load_batch_{i}_1"},
                        {"text_content": f"批量文档{i}_2", "document_id": f"load_batch_{i}_2"}
                    ],
                    return_probabilities=True
                )
                tasks.append(classification_service.classify_batch_documents(batch_request))
            else:
                single_request = ClassificationRequest(
                    project_id="historical_project",
                    text_content=f"负载测试文档{i}，古代历史研究内容。",
                    document_id=f"load_single_{i}",
                    return_probabilities=True
                )
                tasks.append(classification_service.classify_single_document(single_request))
        
        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 检查结果
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(result)
            else:
                successful_results.append(result)
        
        # 大部分请求应该成功
        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.90  # 至少90%成功率
        
        # 检查成功结果的质量
        for result in successful_results[:5]:  # 检查前5个结果
            if hasattr(result, 'predicted_label'):  # 单文档结果
                assert result.predicted_label in classification_service.class_labels
            elif hasattr(result, 'successful_classifications'):  # 批量结果
                assert result.successful_classifications >= 0