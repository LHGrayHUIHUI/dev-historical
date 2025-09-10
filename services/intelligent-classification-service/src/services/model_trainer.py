"""
智能分类服务模型训练器
支持多种机器学习算法的训练和评估
专门针对中文历史文本分类优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import joblib
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

# 机器学习模型
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# MLflow暂时禁用
MLFLOW_AVAILABLE = False

from ..utils.feature_extraction import BaseFeatureExtractor, FeatureExtractorFactory
from ..utils.text_preprocessing import ChineseTextPreprocessor
from ..config.settings import settings
from ..schemas.classification_schemas import ModelType, FeatureExtractorType, TrainingMetrics


@dataclass
class TrainingConfig:
    """训练配置类"""
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    random_state: int = 42
    min_samples_per_class: int = 5
    max_training_time: int = 1800  # 30分钟
    early_stopping: bool = True
    patience: int = 10
    enable_scaling: bool = True
    enable_mlflow: bool = True


class ModelTrainer:
    """机器学习模型训练器
    
    支持多种算法的统一训练接口
    包含完整的训练流水线和性能评估
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        
        # 模型注册表
        self._model_registry = {
            ModelType.SVM: self._create_svm_model,
            ModelType.RANDOM_FOREST: self._create_rf_model,
            ModelType.XGBOOST: self._create_xgb_model,
            ModelType.LIGHTGBM: self._create_lgb_model
        }
        
        # 初始化MLflow
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            self._init_mlflow()
        
        # 预处理器
        self.preprocessor = ChineseTextPreprocessor()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler() if self.config.enable_scaling else None
    
    def _init_mlflow(self):
        """初始化MLflow实验跟踪"""
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            self.logger.info("MLflow实验跟踪已初始化")
        except Exception as e:
            self.logger.warning(f"MLflow初始化失败: {e}")
            self.config.enable_mlflow = False
    
    def _create_svm_model(self, hyperparameters: Dict[str, Any]) -> SVC:
        """创建SVM模型"""
        default_params = settings.get_model_config('svm')
        default_params.update(hyperparameters)
        
        return SVC(
            C=default_params.get('C', 1.0),
            kernel=default_params.get('kernel', 'rbf'),
            gamma=default_params.get('gamma', 'scale'),
            probability=True,  # 启用概率预测
            random_state=self.config.random_state
        )
    
    def _create_rf_model(self, hyperparameters: Dict[str, Any]) -> RandomForestClassifier:
        """创建随机森林模型"""
        default_params = settings.get_model_config('random_forest')
        default_params.update(hyperparameters)
        
        return RandomForestClassifier(
            n_estimators=default_params.get('n_estimators', 100),
            max_depth=default_params.get('max_depth'),
            min_samples_split=default_params.get('min_samples_split', 2),
            min_samples_leaf=default_params.get('min_samples_leaf', 1),
            random_state=self.config.random_state,
            n_jobs=-1
        )
    
    def _create_xgb_model(self, hyperparameters: Dict[str, Any]) -> Optional[Any]:
        """创建XGBoost模型"""
        if not XGBOOST_AVAILABLE:
            self.logger.error("XGBoost未安装")
            return None
        
        try:
            import xgboost as xgb
            default_params = settings.get_model_config('xgboost')
            default_params.update(hyperparameters)
            
            return xgb.XGBClassifier(
                n_estimators=default_params.get('n_estimators', 100),
                max_depth=default_params.get('max_depth', 6),
                learning_rate=default_params.get('learning_rate', 0.1),
                subsample=default_params.get('subsample', 0.8),
                random_state=self.config.random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )
        except ImportError:
            self.logger.error("XGBoost导入失败")
            return None
    
    def _create_lgb_model(self, hyperparameters: Dict[str, Any]) -> Optional[Any]:
        """创建LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            self.logger.error("LightGBM未安装")
            return None
        
        try:
            import lightgbm as lgb
            default_params = settings.get_model_config('lightgbm')
            default_params.update(hyperparameters)
            
            return lgb.LGBMClassifier(
                n_estimators=default_params.get('n_estimators', 100),
                max_depth=default_params.get('max_depth', -1),
                learning_rate=default_params.get('learning_rate', 0.1),
                num_leaves=default_params.get('num_leaves', 31),
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            )
        except ImportError:
            self.logger.error("LightGBM导入失败")
            return None
    
    def prepare_data(
        self,
        texts: List[str],
        labels: List[str],
        feature_extractor: BaseFeatureExtractor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备训练数据
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.logger.info(f"开始准备训练数据，样本数量: {len(texts)}")
        
        # 检查数据质量
        if len(texts) != len(labels):
            raise ValueError("文本和标签数量不匹配")
        
        if len(texts) < self.config.min_samples_per_class * len(set(labels)):
            raise ValueError(f"样本数量不足，每个类别至少需要{self.config.min_samples_per_class}个样本")
        
        # 特征提取
        self.logger.info("开始特征提取...")
        if not feature_extractor.is_fitted:
            feature_extractor.fit(texts, labels)
        
        features = feature_extractor.transform(texts)
        self.logger.info(f"特征提取完成，特征维度: {features.shape}")
        
        # 标签编码
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_.tolist()
        
        # 特征标准化
        if self.scaler:
            features = self.scaler.fit_transform(features)
            self.logger.info("特征标准化完成")
        
        # 数据分割
        # 首先分离测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, encoded_labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=encoded_labels
        )
        
        # 再从临时数据中分离训练集和验证集
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        self.logger.info(f"数据分割完成 - 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingMetrics:
        """评估模型性能"""
        self.logger.info("开始模型评估...")
        
        # 测试集预测
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # 获取预测概率
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception as e:
                self.logger.warning(f"无法获取预测概率: {e}")
        
        # 计算基础指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 交叉验证
        cv_scores = []
        if X_val is not None and y_val is not None:
            # 使用验证集进行交叉验证
            kfold = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            cv_data = np.vstack([X_val, X_test])
            cv_labels = np.concatenate([y_val, y_test])
            cv_scores = cross_val_score(model, cv_data, cv_labels, cv=kfold, scoring='f1_weighted')
        
        cv_mean = np.mean(cv_scores) if cv_scores else 0.0
        cv_std = np.std(cv_scores) if cv_scores else 0.0
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # 分类报告
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 创建TrainingMetrics对象
        metrics = TrainingMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cv_mean=cv_mean,
            cv_std=cv_std,
            confusion_matrix=conf_matrix,
            classification_report=class_report
        )
        
        self.logger.info(f"模型评估完成 - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
        
        return metrics
    
    def train_model(
        self,
        texts: List[str],
        labels: List[str],
        model_type: ModelType,
        feature_extractor_type: FeatureExtractorType,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """训练模型的完整流程"""
        start_time = time.time()
        hyperparameters = hyperparameters or {}
        
        self.logger.info(f"开始训练模型: {model_type}, 特征提取器: {feature_extractor_type}")
        
        # 开始MLflow运行
        if self.config.enable_mlflow and MLFLOW_AVAILABLE:
            mlflow.start_run()
            mlflow.log_params({
                'model_type': model_type,
                'feature_extractor': feature_extractor_type,
                'num_samples': len(texts),
                'num_classes': len(set(labels))
            })
        
        try:
            # 创建特征提取器
            feature_extractor = FeatureExtractorFactory.create_extractor(
                feature_extractor_type, feature_config
            )
            
            # 准备数据
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
                texts, labels, feature_extractor
            )
            
            # 创建模型
            if model_type not in self._model_registry:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            model = self._model_registry[model_type](hyperparameters)
            if model is None:
                raise ValueError(f"无法创建模型: {model_type}")
            
            # 训练模型
            self.logger.info("开始模型训练...")
            
            # 对于支持early stopping的模型，使用验证集
            if hasattr(model, 'fit') and model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
                if model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.config.patience if self.config.early_stopping else None,
                        verbose=False
                    )
                elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
                    try:
                        import lightgbm as lgb
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(self.config.patience)] if self.config.early_stopping else None,
                            verbose=0
                        )
                    except ImportError:
                        # 回退到基础训练
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            self.logger.info(f"模型训练完成，用时: {training_time:.2f}秒")
            
            # 评估模型
            metrics = self.evaluate_model(model, X_test, y_test, X_val, y_val)
            
            # 记录MLflow指标
            if self.config.enable_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'cv_mean': metrics.cv_mean,
                    'cv_std': metrics.cv_std,
                    'training_time': training_time
                })
                
                # 保存模型
                mlflow.sklearn.log_model(model, "model")
                
                # 保存特征提取器
                mlflow.log_dict(feature_extractor.get_config(), "feature_extractor_config.json")
            
            # 创建训练结果
            result = {
                'model': model,
                'feature_extractor': feature_extractor,
                'metrics': metrics,
                'training_time': training_time,
                'model_type': model_type,
                'feature_extractor_type': feature_extractor_type,
                'hyperparameters': hyperparameters,
                'feature_config': feature_config or {},
                'class_names': self.class_names,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'data_shapes': {
                    'train': X_train.shape,
                    'val': X_val.shape,
                    'test': X_test.shape
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            if self.config.enable_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_param('error', str(e))
            raise
        
        finally:
            if self.config.enable_mlflow and MLFLOW_AVAILABLE:
                mlflow.end_run()
    
    def save_trained_model(self, training_result: Dict[str, Any], model_path: str):
        """保存训练好的模型"""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整的训练结果
        joblib.dump(training_result, model_path)
        
        # 保存模型元数据
        metadata = {
            'model_type': training_result['model_type'],
            'feature_extractor_type': training_result['feature_extractor_type'],
            'training_time': training_result['training_time'],
            'metrics': {
                'accuracy': training_result['metrics'].accuracy,
                'precision': training_result['metrics'].precision,
                'recall': training_result['metrics'].recall,
                'f1_score': training_result['metrics'].f1_score
            },
            'class_names': training_result['class_names'],
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"模型已保存到: {model_path}")
    
    @classmethod
    def load_trained_model(cls, model_path: str) -> Dict[str, Any]:
        """加载训练好的模型"""
        training_result = joblib.load(model_path)
        logging.getLogger(__name__).info(f"模型已从 {model_path} 加载")
        return training_result
    
    def batch_train_models(
        self,
        texts: List[str],
        labels: List[str],
        model_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量训练多个模型配置"""
        results = []
        total_configs = len(model_configs)
        
        self.logger.info(f"开始批量训练，配置数量: {total_configs}")
        
        for i, config in enumerate(model_configs):
            self.logger.info(f"训练配置 {i+1}/{total_configs}: {config.get('name', 'unnamed')}")
            
            try:
                result = self.train_model(
                    texts=texts,
                    labels=labels,
                    model_type=config['model_type'],
                    feature_extractor_type=config['feature_extractor_type'],
                    hyperparameters=config.get('hyperparameters'),
                    feature_config=config.get('feature_config')
                )
                
                result['config_name'] = config.get('name', f'config_{i}')
                results.append(result)
                
                self.logger.info(f"配置 {config.get('name')} 训练完成，F1分数: {result['metrics'].f1_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"配置 {config.get('name')} 训练失败: {e}")
                continue
        
        # 按F1分数排序
        results.sort(key=lambda x: x['metrics'].f1_score, reverse=True)
        
        self.logger.info(f"批量训练完成，成功训练模型数量: {len(results)}")
        
        return results
    
    def get_model_comparison(self, training_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """比较多个训练结果"""
        comparison_data = []
        
        for result in training_results:
            metrics = result['metrics']
            comparison_data.append({
                'config_name': result.get('config_name', 'unknown'),
                'model_type': result['model_type'],
                'feature_extractor': result['feature_extractor_type'],
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'cv_mean': metrics.cv_mean,
                'cv_std': metrics.cv_std,
                'training_time': result['training_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('f1_score', ascending=False)
        
        return df


# 便捷函数
def create_model_trainer(config: Optional[TrainingConfig] = None) -> ModelTrainer:
    """创建模型训练器"""
    return ModelTrainer(config)


def quick_train_model(
    texts: List[str],
    labels: List[str],
    model_type: ModelType = ModelType.RANDOM_FOREST,
    feature_extractor_type: FeatureExtractorType = FeatureExtractorType.TFIDF
) -> Dict[str, Any]:
    """快速训练模型"""
    trainer = create_model_trainer()
    return trainer.train_model(texts, labels, model_type, feature_extractor_type)