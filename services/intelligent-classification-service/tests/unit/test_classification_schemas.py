"""
智能分类服务数据模型测试
测试所有Pydantic模型和数据验证规则
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any

from src.schemas.classification_schemas import (
    # 枚举类型
    ClassificationType, ModelType, FeatureExtractorType, TaskStatus, ProjectStatus,
    
    # 基础响应模型
    BaseResponse,
    
    # 项目管理相关模型
    ClassificationProjectCreate, ClassificationProject,
    
    # 训练数据相关模型
    TrainingDataCreate, TrainingDataBatch, TrainingData,
    
    # 模型训练相关模型
    ModelTrainingRequest, ModelInfo, TrainingMetrics, ModelTrainingResponse,
    
    # 文档分类相关模型
    ClassificationRequest, ClassificationResult, 
    BatchClassificationRequest, BatchClassificationResult,
    
    # 模型性能相关模型
    ModelPerformanceRequest, UsageStatistics, ModelPerformanceResponse,
    
    # 任务管理相关模型
    TaskInfo,
    
    # 项目统计相关模型
    ProjectStatistics,
    
    # 系统配置相关模型
    SystemConfig,
    
    # 健康检查相关模型
    HealthCheck,
    
    # 模型导出和导入
    ModelExportRequest, ModelImportRequest,
    
    # A/B测试相关模型
    ABTestRequest, ABTestResult
)


class TestEnumTypes:
    """枚举类型测试"""
    
    def test_classification_type_enum(self):
        """测试分类类型枚举"""
        assert ClassificationType.TOPIC == "topic"
        assert ClassificationType.ERA == "era"
        assert ClassificationType.DOCUMENT_TYPE == "document_type"
        assert ClassificationType.IMPORTANCE == "importance"
        assert ClassificationType.SENTIMENT == "sentiment"
        assert ClassificationType.GENRE == "genre"
        
        # 测试枚举成员数量
        assert len(ClassificationType) == 6
    
    def test_model_type_enum(self):
        """测试机器学习模型类型枚举"""
        assert ModelType.SVM == "svm"
        assert ModelType.RANDOM_FOREST == "random_forest"
        assert ModelType.XGBOOST == "xgboost"
        assert ModelType.LIGHTGBM == "lightgbm"
        assert ModelType.BERT == "bert"
        assert ModelType.ROBERTA == "roberta"
        
        # 测试所有模型类型都可访问
        all_models = [ModelType.SVM, ModelType.RANDOM_FOREST, ModelType.XGBOOST, 
                      ModelType.LIGHTGBM, ModelType.BERT, ModelType.ROBERTA]
        assert len(all_models) == 6
    
    def test_feature_extractor_type_enum(self):
        """测试特征提取器类型枚举"""
        assert FeatureExtractorType.TFIDF == "tfidf"
        assert FeatureExtractorType.WORD2VEC == "word2vec"
        assert FeatureExtractorType.FASTTEXT == "fasttext"
        assert FeatureExtractorType.BERT == "bert"
        assert FeatureExtractorType.SENTENCE_TRANSFORMER == "sentence_transformer"
    
    def test_task_status_enum(self):
        """测试任务状态枚举"""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.PROCESSING == "processing"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"
    
    def test_project_status_enum(self):
        """测试项目状态枚举"""
        assert ProjectStatus.ACTIVE == "active"
        assert ProjectStatus.TRAINING == "training"
        assert ProjectStatus.COMPLETED == "completed"
        assert ProjectStatus.ARCHIVED == "archived"


class TestBaseResponseModel:
    """基础响应模型测试"""
    
    def test_base_response_creation(self):
        """测试基础响应模型创建"""
        response = BaseResponse(
            success=True,
            message="操作成功",
            data={"result": "test"}
        )
        
        assert response.success is True
        assert response.message == "操作成功"
        assert response.data == {"result": "test"}
        assert isinstance(response.timestamp, datetime)
    
    def test_base_response_minimal(self):
        """测试最小化基础响应"""
        response = BaseResponse(
            success=False,
            message="操作失败"
        )
        
        assert response.success is False
        assert response.message == "操作失败"
        assert response.data is None
        assert isinstance(response.timestamp, datetime)


class TestProjectModels:
    """项目相关模型测试"""
    
    def test_classification_project_create(self):
        """测试创建分类项目请求"""
        project_create = ClassificationProjectCreate(
            name="历史文本分类",
            description="对历史文本进行主题分类",
            classification_type=ClassificationType.TOPIC,
            domain="历史文献",
            language="zh",
            custom_labels=["政治", "军事", "经济", "文化"],
            ml_model_config={"max_features": 10000},
            ml_training_config={"epochs": 10},
            ml_feature_config={"ngram_range": [1, 3]}
        )
        
        assert project_create.name == "历史文本分类"
        assert project_create.description == "对历史文本进行主题分类"
        assert project_create.classification_type == ClassificationType.TOPIC
        assert project_create.domain == "历史文献"
        assert project_create.language == "zh"
        assert project_create.custom_labels == ["政治", "军事", "经济", "文化"]
    
    def test_classification_project_minimal(self):
        """测试最小化项目创建"""
        project_create = ClassificationProjectCreate(
            name="简单分类",
            classification_type=ClassificationType.SENTIMENT
        )
        
        assert project_create.name == "简单分类"
        assert project_create.classification_type == ClassificationType.SENTIMENT
        assert project_create.description is None
        assert project_create.domain is None
        assert project_create.language == "zh"
        assert project_create.custom_labels is None
    
    def test_classification_project_name_validation(self):
        """测试项目名称验证"""
        # 空名称应该失败
        with pytest.raises(ValueError):
            ClassificationProjectCreate(
                name="",
                classification_type=ClassificationType.TOPIC
            )
        
        # 过长名称应该失败
        with pytest.raises(ValueError):
            ClassificationProjectCreate(
                name="x" * 201,  # 超过200字符限制
                classification_type=ClassificationType.TOPIC
            )
    
    def test_classification_project_full(self):
        """测试完整分类项目模型"""
        project = ClassificationProject(
            id="project_123",
            name="历史文本分类",
            description="历史文献的智能分类系统",
            classification_type=ClassificationType.TOPIC,
            domain="历史文献",
            language="zh",
            status=ProjectStatus.ACTIVE,
            class_labels=["政治", "军事", "经济", "文化"],
            ml_model_config={"model_type": "bert", "max_seq_length": 512},
            ml_training_config={"batch_size": 32, "epochs": 10},
            ml_feature_config={"vocab_size": 50000},
            created_by="user_123",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert project.id == "project_123"
        assert project.name == "历史文本分类"
        assert project.status == ProjectStatus.ACTIVE
        assert len(project.class_labels) == 4
        assert "政治" in project.class_labels


class TestTrainingDataModels:
    """训练数据相关模型测试"""
    
    def test_training_data_create(self):
        """测试训练数据创建"""
        training_data = TrainingDataCreate(
            project_id="project_123",
            text_content="这是一篇关于古代政治制度的文献",
            true_label="政治",
            document_id="doc_456",
            label_confidence=0.9,
            data_source="expert_annotation"
        )
        
        assert training_data.project_id == "project_123"
        assert training_data.text_content == "这是一篇关于古代政治制度的文献"
        assert training_data.true_label == "政治"
        assert training_data.document_id == "doc_456"
        assert training_data.label_confidence == 0.9
        assert training_data.data_source == "expert_annotation"
    
    def test_training_data_minimal(self):
        """测试最小化训练数据"""
        training_data = TrainingDataCreate(
            project_id="project_123",
            text_content="简单文本",
            true_label="文化"
        )
        
        assert training_data.project_id == "project_123"
        assert training_data.text_content == "简单文本"
        assert training_data.true_label == "文化"
        assert training_data.document_id is None
        assert training_data.label_confidence == 1.0  # 默认值
        assert training_data.data_source == "manual"  # 默认值
    
    def test_training_data_validation(self):
        """测试训练数据验证"""
        # 文本长度验证
        with pytest.raises(ValueError):
            TrainingDataCreate(
                project_id="project_123",
                text_content="",  # 空文本
                true_label="文化"
            )
        
        # 置信度范围验证
        with pytest.raises(ValueError):
            TrainingDataCreate(
                project_id="project_123",
                text_content="测试文本",
                true_label="文化",
                label_confidence=1.5  # 超出范围
            )
        
        with pytest.raises(ValueError):
            TrainingDataCreate(
                project_id="project_123",
                text_content="测试文本",
                true_label="文化",
                label_confidence=-0.1  # 低于范围
            )
    
    def test_training_data_batch(self):
        """测试批量训练数据"""
        batch_data = TrainingDataBatch(
            project_id="project_123",
            training_data=[
                {"text_content": "政治文献1", "true_label": "政治"},
                {"text_content": "军事文献1", "true_label": "军事"},
                {"text_content": "经济文献1", "true_label": "经济"}
            ]
        )
        
        assert batch_data.project_id == "project_123"
        assert len(batch_data.training_data) == 3
        assert batch_data.training_data[0]["true_label"] == "政治"
    
    def test_training_data_batch_validation(self):
        """测试批量数据验证"""
        # 空列表验证
        with pytest.raises(ValueError):
            TrainingDataBatch(
                project_id="project_123",
                training_data=[]  # 空列表
            )
    
    def test_training_data_full_model(self):
        """测试完整训练数据模型"""
        training_data = TrainingData(
            id="training_123",
            project_id="project_123",
            document_id="doc_456",
            text_content="这是一篇关于古代军事战略的文献，详细描述了各种战术运用。",
            true_label="军事",
            label_confidence=0.95,
            data_source="expert_annotation",
            text_features={"word_count": 25, "char_count": 50},
            metadata={"annotator": "expert_1", "domain": "ancient_military"},
            created_at=datetime.now()
        )
        
        assert training_data.id == "training_123"
        assert training_data.true_label == "军事"
        assert training_data.text_features["word_count"] == 25
        assert training_data.metadata["annotator"] == "expert_1"


class TestModelTrainingModels:
    """模型训练相关模型测试"""
    
    def test_model_training_request(self):
        """测试模型训练请求"""
        training_request = ModelTrainingRequest(
            project_id="project_123",
            model_type=ModelType.BERT,
            feature_extractor=FeatureExtractorType.BERT,
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            training_config={"epochs": 10, "early_stopping": True}
        )
        
        assert training_request.project_id == "project_123"
        assert training_request.model_type == ModelType.BERT
        assert training_request.feature_extractor == FeatureExtractorType.BERT
        assert training_request.hyperparameters["learning_rate"] == 0.001
    
    def test_model_info(self):
        """测试模型信息"""
        model_info = ModelInfo(
            id="model_123",
            project_id="project_123",
            model_name="历史文本BERT分类器",
            model_type=ModelType.BERT,
            feature_extractor=FeatureExtractorType.BERT,
            model_version="v1.0",
            model_path="/models/bert_classifier_v1.pkl",
            hyperparameters={"learning_rate": 0.001, "max_seq_length": 512},
            status=TaskStatus.COMPLETED,
            is_active=True,
            training_data_size=1000,
            training_time=3600.0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert model_info.id == "model_123"
        assert model_info.model_name == "历史文本BERT分类器"
        assert model_info.model_type == ModelType.BERT
        assert model_info.status == TaskStatus.COMPLETED
        assert model_info.is_active is True
        assert model_info.training_data_size == 1000
    
    def test_training_metrics(self):
        """测试训练指标"""
        metrics = TrainingMetrics(
            accuracy=0.92,
            precision=0.91,
            recall=0.90,
            f1_score=0.905,
            cv_mean=0.918,
            cv_std=0.025,
            confusion_matrix=[[85, 5, 3, 2], [4, 88, 2, 1], [2, 3, 89, 1], [1, 2, 1, 91]],
            classification_report={
                "政治": {"precision": 0.92, "recall": 0.89, "f1-score": 0.905},
                "军事": {"precision": 0.90, "recall": 0.93, "f1-score": 0.915},
                "经济": {"precision": 0.94, "recall": 0.91, "f1-score": 0.925},
                "文化": {"precision": 0.96, "recall": 0.92, "f1-score": 0.940}
            }
        )
        
        assert metrics.accuracy == 0.92
        assert metrics.f1_score == 0.905
        assert len(metrics.confusion_matrix) == 4
        assert len(metrics.confusion_matrix[0]) == 4
        assert "政治" in metrics.classification_report
    
    def test_model_training_response(self):
        """测试模型训练响应"""
        training_metrics = TrainingMetrics(
            accuracy=0.90, precision=0.89, recall=0.88, f1_score=0.885,
            cv_mean=0.892, cv_std=0.018,
            confusion_matrix=[[90, 10], [5, 95]],
            classification_report={"class1": {"precision": 0.9}}
        )
        
        response = ModelTrainingResponse(
            model_id="model_123",
            project_id="project_123",
            training_metrics=training_metrics,
            training_time=1800.0,
            model_path="/models/model_123.pkl",
            status=TaskStatus.COMPLETED
        )
        
        assert response.model_id == "model_123"
        assert response.training_time == 1800.0
        assert response.status == TaskStatus.COMPLETED
        assert response.training_metrics.accuracy == 0.90


class TestClassificationModels:
    """分类相关模型测试"""
    
    def test_classification_request(self):
        """测试分类请求"""
        request = ClassificationRequest(
            project_id="project_123",
            text_content="这是一篇关于古代经济制度的历史文献",
            document_id="doc_789",
            model_id="model_123",
            return_probabilities=True,
            return_explanation=True
        )
        
        assert request.project_id == "project_123"
        assert "古代经济制度" in request.text_content
        assert request.document_id == "doc_789"
        assert request.model_id == "model_123"
        assert request.return_probabilities is True
        assert request.return_explanation is True
    
    def test_classification_request_minimal(self):
        """测试最小化分类请求"""
        request = ClassificationRequest(
            project_id="project_123",
            text_content="简单文本"
        )
        
        assert request.project_id == "project_123"
        assert request.text_content == "简单文本"
        assert request.document_id is None
        assert request.model_id is None
        assert request.return_probabilities is True  # 默认值
        assert request.return_explanation is True  # 默认值
    
    def test_classification_request_validation(self):
        """测试分类请求验证"""
        # 空文本验证
        with pytest.raises(ValueError):
            ClassificationRequest(
                project_id="project_123",
                text_content=""
            )
        
        # 文本长度限制
        with pytest.raises(ValueError):
            ClassificationRequest(
                project_id="project_123",
                text_content="x" * 10001  # 超过10000字符限制
            )
    
    def test_classification_result(self):
        """测试分类结果"""
        result = ClassificationResult(
            task_id="task_123",
            document_id="doc_789",
            predicted_label="经济",
            confidence_score=0.87,
            probability_distribution={
                "政治": 0.05,
                "军事": 0.08,
                "经济": 0.87,
                "文化": 0.00
            },
            feature_importance={
                "经济": 0.25,
                "制度": 0.20,
                "古代": 0.15,
                "历史": 0.12,
                "文献": 0.08
            },
            explanation="文本被分类为'经济'，置信度为87%。关键特征包括: '经济'、'制度'等词汇。",
            processing_time=0.15,
            model_info={
                "model_id": "model_123",
                "model_type": "bert",
                "feature_extractor": "bert"
            }
        )
        
        assert result.task_id == "task_123"
        assert result.predicted_label == "经济"
        assert result.confidence_score == 0.87
        assert result.probability_distribution["经济"] == 0.87
        assert "经济" in result.feature_importance
        assert "87%" in result.explanation
        assert result.processing_time == 0.15
        assert result.model_info["model_type"] == "bert"
    
    def test_batch_classification_request(self):
        """测试批量分类请求"""
        batch_request = BatchClassificationRequest(
            project_id="project_123",
            documents=[
                {"text_content": "政治文献1", "document_id": "doc_1"},
                {"text_content": "军事文献2", "document_id": "doc_2"},
                {"text_content": "经济文献3", "document_id": "doc_3"}
            ],
            model_id="model_123",
            return_probabilities=True,
            return_explanation=False
        )
        
        assert batch_request.project_id == "project_123"
        assert len(batch_request.documents) == 3
        assert batch_request.documents[0]["text_content"] == "政治文献1"
        assert batch_request.model_id == "model_123"
        assert batch_request.return_probabilities is True
        assert batch_request.return_explanation is False
    
    def test_batch_classification_request_validation(self):
        """测试批量分类请求验证"""
        # 空文档列表
        with pytest.raises(ValueError):
            BatchClassificationRequest(
                project_id="project_123",
                documents=[]
            )
        
        # 超过最大批量大小
        with pytest.raises(ValueError):
            BatchClassificationRequest(
                project_id="project_123",
                documents=[{"text_content": f"文档{i}"} for i in range(101)]  # 超过100个限制
            )
    
    def test_batch_classification_result(self):
        """测试批量分类结果"""
        # 创建示例分类结果
        result1 = ClassificationResult(
            task_id="batch_123_0", predicted_label="政治", confidence_score=0.85,
            processing_time=0.1, model_info={"model_id": "model_123"}
        )
        result2 = ClassificationResult(
            task_id="batch_123_1", predicted_label="军事", confidence_score=0.92,
            processing_time=0.12, model_info={"model_id": "model_123"}
        )
        
        batch_result = BatchClassificationResult(
            batch_task_id="batch_123",
            total_documents=2,
            successful_classifications=2,
            failed_classifications=0,
            results=[result1, result2],
            processing_time=0.25,
            statistics={
                "avg_confidence": 0.885,
                "label_distribution": {"政治": 1, "军事": 1},
                "confidence_distribution": {"high": 2, "medium": 0, "low": 0}
            }
        )
        
        assert batch_result.batch_task_id == "batch_123"
        assert batch_result.total_documents == 2
        assert batch_result.successful_classifications == 2
        assert batch_result.failed_classifications == 0
        assert len(batch_result.results) == 2
        assert batch_result.statistics["avg_confidence"] == 0.885


class TestPerformanceModels:
    """性能相关模型测试"""
    
    def test_model_performance_request(self):
        """测试模型性能查询请求"""
        request = ModelPerformanceRequest(
            model_id="model_123",
            include_usage_stats=True,
            include_detailed_metrics=True
        )
        
        assert request.model_id == "model_123"
        assert request.include_usage_stats is True
        assert request.include_detailed_metrics is True
    
    def test_usage_statistics(self):
        """测试使用统计"""
        stats = UsageStatistics(
            total_predictions=1000,
            avg_confidence=0.85,
            avg_processing_time=0.12,
            confidence_distribution={"high": 600, "medium": 300, "low": 100},
            label_distribution={"政治": 250, "军事": 300, "经济": 200, "文化": 250},
            daily_usage={"2025-09-01": 100, "2025-09-02": 150, "2025-09-03": 200},
            error_rate=0.02
        )
        
        assert stats.total_predictions == 1000
        assert stats.avg_confidence == 0.85
        assert stats.confidence_distribution["high"] == 600
        assert stats.label_distribution["军事"] == 300
        assert stats.daily_usage["2025-09-03"] == 200
        assert stats.error_rate == 0.02


class TestTaskManagementModels:
    """任务管理相关模型测试"""
    
    def test_task_info(self):
        """测试任务信息"""
        task = TaskInfo(
            id="task_123",
            project_id="project_123",
            task_type="classification",
            status=TaskStatus.PROCESSING,
            progress=0.65,
            started_at=datetime.now(),
            processing_time=120.0,
            result={"predictions": 100, "success_rate": 0.95}
        )
        
        assert task.id == "task_123"
        assert task.task_type == "classification"
        assert task.status == TaskStatus.PROCESSING
        assert task.progress == 0.65
        assert task.result["success_rate"] == 0.95
    
    def test_task_info_progress_validation(self):
        """测试任务进度验证"""
        # 进度超出范围
        with pytest.raises(ValueError):
            TaskInfo(
                id="task_123",
                project_id="project_123",
                task_type="classification",
                status=TaskStatus.PROCESSING,
                progress=1.5  # 超出范围
            )


class TestSystemModels:
    """系统相关模型测试"""
    
    def test_project_statistics(self):
        """测试项目统计"""
        stats = ProjectStatistics(
            project_id="project_123",
            total_training_data=5000,
            total_models=3,
            total_predictions=10000,
            active_models=1,
            label_distribution={"政治": 1200, "军事": 1500, "经济": 1100, "文化": 1200},
            model_performance={"accuracy": 0.92, "f1_score": 0.91},
            recent_activity=[
                {"type": "training", "timestamp": "2025-09-09T10:00:00"},
                {"type": "prediction", "timestamp": "2025-09-09T11:00:00"}
            ],
            data_quality_score=0.88
        )
        
        assert stats.project_id == "project_123"
        assert stats.total_training_data == 5000
        assert stats.total_models == 3
        assert stats.active_models == 1
        assert stats.label_distribution["军事"] == 1500
        assert stats.model_performance["accuracy"] == 0.92
        assert len(stats.recent_activity) == 2
        assert stats.data_quality_score == 0.88
    
    def test_system_config(self):
        """测试系统配置"""
        config = SystemConfig(
            supported_classification_types=["topic", "era", "document_type", "sentiment"],
            supported_model_types=["svm", "random_forest", "bert", "roberta"],
            supported_feature_extractors=["tfidf", "word2vec", "bert"],
            predefined_labels={
                "topic": ["政治", "军事", "经济", "文化"],
                "era": ["先秦", "秦汉", "魏晋", "唐宋", "明清"],
                "sentiment": ["积极", "消极", "中性"]
            },
            max_text_length=10000,
            max_batch_size=100,
            performance_thresholds={"min_accuracy": 0.8, "min_f1_score": 0.75}
        )
        
        assert "topic" in config.supported_classification_types
        assert "bert" in config.supported_model_types
        assert "tfidf" in config.supported_feature_extractors
        assert config.predefined_labels["topic"] == ["政治", "军事", "经济", "文化"]
        assert config.max_text_length == 10000
        assert config.max_batch_size == 100
        assert config.performance_thresholds["min_accuracy"] == 0.8
    
    def test_health_check(self):
        """测试健康检查"""
        health = HealthCheck(
            service="intelligent-classification-service",
            status="healthy",
            version="v1.2.0",
            timestamp=datetime.now(),
            dependencies={
                "storage-service": "healthy",
                "mongodb": "healthy",
                "redis": "healthy"
            },
            system_info={
                "memory_usage": "2.5GB",
                "cpu_usage": "15%",
                "disk_usage": "60%"
            }
        )
        
        assert health.service == "intelligent-classification-service"
        assert health.status == "healthy"
        assert health.version == "v1.2.0"
        assert health.dependencies["storage-service"] == "healthy"
        assert health.system_info["memory_usage"] == "2.5GB"


class TestAdvancedFeatureModels:
    """高级功能模型测试"""
    
    def test_model_export_request(self):
        """测试模型导出请求"""
        export_request = ModelExportRequest(
            model_id="model_123",
            export_format="onnx",
            include_metadata=True,
            include_training_data=False
        )
        
        assert export_request.model_id == "model_123"
        assert export_request.export_format == "onnx"
        assert export_request.include_metadata is True
        assert export_request.include_training_data is False
    
    def test_model_import_request(self):
        """测试模型导入请求"""
        import_request = ModelImportRequest(
            project_id="project_123",
            model_file="/path/to/model.pkl",
            model_name="导入的BERT模型",
            replace_existing=True
        )
        
        assert import_request.project_id == "project_123"
        assert import_request.model_file == "/path/to/model.pkl"
        assert import_request.model_name == "导入的BERT模型"
        assert import_request.replace_existing is True
    
    def test_ab_test_request(self):
        """测试A/B测试请求"""
        ab_test = ABTestRequest(
            project_id="project_123",
            model_a_id="model_123",
            model_b_id="model_456",
            test_data=[
                {"text_content": "测试文档1", "true_label": "政治"},
                {"text_content": "测试文档2", "true_label": "军事"}
            ],
            test_name="BERT vs RoBERTa 性能对比"
        )
        
        assert ab_test.project_id == "project_123"
        assert ab_test.model_a_id == "model_123"
        assert ab_test.model_b_id == "model_456"
        assert len(ab_test.test_data) == 2
        assert ab_test.test_name == "BERT vs RoBERTa 性能对比"
    
    def test_ab_test_result(self):
        """测试A/B测试结果"""
        ab_result = ABTestResult(
            test_id="test_123",
            model_a_performance={"accuracy": 0.90, "f1_score": 0.89},
            model_b_performance={"accuracy": 0.92, "f1_score": 0.91},
            winner="model_b",
            confidence_level=0.95,
            recommendation="建议使用模型B，性能显著优于模型A",
            detailed_comparison={
                "statistical_significance": True,
                "p_value": 0.03,
                "effect_size": 0.02
            }
        )
        
        assert ab_result.test_id == "test_123"
        assert ab_result.model_b_performance["accuracy"] == 0.92
        assert ab_result.winner == "model_b"
        assert ab_result.confidence_level == 0.95
        assert "建议使用模型B" in ab_result.recommendation
        assert ab_result.detailed_comparison["statistical_significance"] is True


class TestEdgeCasesAndValidation:
    """边界情况和验证测试"""
    
    def test_enum_string_conversion(self):
        """测试枚举字符串转换"""
        # 从字符串创建枚举
        classification_type = ClassificationType("topic")
        assert classification_type == ClassificationType.TOPIC
        
        model_type = ModelType("bert")
        assert model_type == ModelType.BERT
        
        feature_extractor = FeatureExtractorType("tfidf")
        assert feature_extractor == FeatureExtractorType.TFIDF
    
    def test_complex_nested_models(self):
        """测试复杂嵌套模型"""
        # 创建复杂的模型性能响应
        training_metrics = TrainingMetrics(
            accuracy=0.92, precision=0.91, recall=0.90, f1_score=0.905,
            cv_mean=0.918, cv_std=0.025,
            confusion_matrix=[[85, 5, 3, 2], [4, 88, 2, 1]],
            classification_report={"class1": {"precision": 0.9}}
        )
        
        model_info = ModelInfo(
            id="model_123", project_id="project_123", model_name="测试模型",
            model_type=ModelType.BERT, feature_extractor=FeatureExtractorType.BERT,
            model_version="v1.0", hyperparameters={}, status=TaskStatus.COMPLETED,
            is_active=True, created_at=datetime.now(), updated_at=datetime.now()
        )
        
        usage_stats = UsageStatistics(
            total_predictions=1000, avg_confidence=0.85, avg_processing_time=0.12,
            confidence_distribution={}, label_distribution={}, daily_usage={}, error_rate=0.02
        )
        
        performance_response = ModelPerformanceResponse(
            model_info=model_info,
            training_metrics=training_metrics,
            usage_statistics=usage_stats,
            performance_trend={"accuracy": [0.85, 0.88, 0.90, 0.92]},
            comparison_with_baseline={"accuracy_improvement": 0.05}
        )
        
        assert performance_response.model_info.model_type == ModelType.BERT
        assert performance_response.training_metrics.accuracy == 0.92
        assert performance_response.usage_statistics.total_predictions == 1000
        assert len(performance_response.performance_trend["accuracy"]) == 4
    
    def test_boundary_values(self):
        """测试边界值"""
        # 最小置信度
        result = ClassificationResult(
            task_id="task_123",
            predicted_label="测试",
            confidence_score=0.0,  # 最小值
            processing_time=0.001,  # 极小处理时间
            model_info={}
        )
        assert result.confidence_score == 0.0
        
        # 最大置信度
        result2 = ClassificationResult(
            task_id="task_456",
            predicted_label="测试",
            confidence_score=1.0,  # 最大值
            processing_time=999.999,  # 很长处理时间
            model_info={}
        )
        assert result2.confidence_score == 1.0
    
    def test_optional_fields_handling(self):
        """测试可选字段处理"""
        # 最小化分类结果
        minimal_result = ClassificationResult(
            task_id="task_123",
            predicted_label="测试",
            confidence_score=0.5,
            processing_time=0.1,
            model_info={}
        )
        
        assert minimal_result.document_id is None
        assert minimal_result.probability_distribution is None
        assert minimal_result.feature_importance is None
        assert minimal_result.explanation is None
    
    def test_unicode_and_chinese_text(self):
        """测试中文和Unicode文本处理"""
        # 包含中文的分类请求
        chinese_request = ClassificationRequest(
            project_id="项目_123",
            text_content="这是一篇关于中国古代历史文献的研究，涵盖了政治、经济、文化等多个方面的内容分析。文献中提到了许多历史事件和人物。",
            document_id="文档_456"
        )
        
        assert "中国古代历史" in chinese_request.text_content
        assert chinese_request.project_id == "项目_123"
        assert chinese_request.document_id == "文档_456"
        
        # 包含特殊字符的标签
        special_result = ClassificationResult(
            task_id="任务_123",
            predicted_label="历史/政治",
            confidence_score=0.88,
            processing_time=0.2,
            model_info={"模型类型": "BERT", "特征提取器": "sentence-transformer"}
        )
        
        assert special_result.predicted_label == "历史/政治"
        assert special_result.model_info["模型类型"] == "BERT"