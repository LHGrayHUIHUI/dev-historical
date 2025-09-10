"""
智能分类服务业务逻辑模块
提供模型训练和分类预测功能
"""

from .model_trainer import (
    ModelTrainer,
    TrainingConfig,
    create_model_trainer,
    quick_train_model
)

from .classification_service import (
    ClassificationService,
    PredictionConfig,
    classification_service,
    classify_document,
    classify_documents_batch
)

__all__ = [
    # 模型训练
    'ModelTrainer',
    'TrainingConfig', 
    'create_model_trainer',
    'quick_train_model',
    
    # 分类服务
    'ClassificationService',
    'PredictionConfig',
    'classification_service',
    'classify_document',
    'classify_documents_batch'
]