# Story 2.5: 智能分类服务

## 基本信息
- **Story ID**: 2.5
- **Epic**: Epic 2 - 数据处理和智能分类微服务
- **标题**: 智能分类服务
- **优先级**: 高
- **状态**: 待开发
- **预估工期**: 6-8天

## 用户故事
**作为** 文档管理员  
**我希望** 有一个智能的文档分类服务  
**以便** 自动对历史文档进行主题分类、时代分类、类型分类等多维度智能分类，提高文档组织和检索效率

## 需求描述
开发智能文档分类服务，支持基于机器学习的文档自动分类，包括主题分类、时代分类、文档类型分类、重要性评级等功能，为历史文档提供智能化的组织和管理能力。

## 技术实现

### 核心技术栈
- **后端框架**: FastAPI 0.104+ (Python)
- **机器学习框架**: 
  - scikit-learn 1.3+ (传统机器学习)
  - transformers 4.35+ (预训练模型)
  - torch 2.1+ (深度学习)
  - tensorflow 2.14+ (备选深度学习框架)
- **文本处理**: 
  - jieba 0.42+ (中文分词)
  - spaCy 3.7+ (NLP处理)
  - HanLP 2.1+ (中文NLP)
- **特征工程**: 
  - TF-IDF (词频-逆文档频率)
  - Word2Vec (词向量)
  - BERT (语义向量)
  - FastText (快速文本分类)
- **分类算法**: 
  - SVM (支持向量机)
  - Random Forest (随机森林)
  - XGBoost (梯度提升)
  - BERT Classifier (BERT分类器)
  - TextCNN (卷积神经网络)
- **数据库**: 
  - PostgreSQL (分类结果存储)
  - Redis (缓存)
  - Elasticsearch (文档检索)
- **消息队列**: RabbitMQ 3.12+
- **模型管理**: MLflow 2.8+

### 数据模型设计

#### 分类项目表 (classification_projects)
```sql
CREATE TABLE classification_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    classification_type VARCHAR(100) NOT NULL, -- topic, era, document_type, importance
    domain VARCHAR(100), -- 领域：历史、文学等
    language VARCHAR(10) DEFAULT 'zh',
    status VARCHAR(50) DEFAULT 'active', -- active, training, completed, archived
    model_config JSONB, -- 模型配置
    training_config JSONB, -- 训练配置
    performance_metrics JSONB, -- 性能指标
    class_labels JSONB, -- 分类标签
    feature_config JSONB, -- 特征配置
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 训练数据表 (training_data)
```sql
CREATE TABLE training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES classification_projects(id),
    document_id UUID REFERENCES documents(id),
    text_content TEXT NOT NULL,
    true_label VARCHAR(100) NOT NULL, -- 真实标签
    label_confidence FLOAT DEFAULT 1.0, -- 标签置信度
    data_source VARCHAR(100), -- manual, auto, imported
    annotator_id UUID REFERENCES users(id),
    annotation_time TIMESTAMP,
    text_features JSONB, -- 文本特征
    metadata JSONB, -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 分类模型表 (classification_models)
```sql
CREATE TABLE classification_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES classification_projects(id),
    model_name VARCHAR(200) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- svm, random_forest, xgboost, bert, textcnn
    model_version VARCHAR(50),
    model_path VARCHAR(500), -- 模型文件路径
    feature_extractor VARCHAR(100), -- tfidf, word2vec, bert
    hyperparameters JSONB, -- 超参数
    training_metrics JSONB, -- 训练指标
    validation_metrics JSONB, -- 验证指标
    test_metrics JSONB, -- 测试指标
    training_data_size INTEGER,
    training_time FLOAT, -- 训练时间（秒）
    status VARCHAR(50) DEFAULT 'training', -- training, completed, deployed, archived
    is_active BOOLEAN DEFAULT FALSE, -- 是否为活跃模型
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 分类任务表 (classification_tasks)
```sql
CREATE TABLE classification_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES classification_projects(id),
    model_id UUID REFERENCES classification_models(id),
    document_id UUID REFERENCES documents(id),
    text_content TEXT NOT NULL,
    task_type VARCHAR(50) DEFAULT 'single', -- single, batch
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 分类结果表 (classification_results)
```sql
CREATE TABLE classification_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES classification_tasks(id),
    project_id UUID REFERENCES classification_projects(id),
    document_id UUID REFERENCES documents(id),
    predicted_label VARCHAR(100) NOT NULL, -- 预测标签
    confidence_score FLOAT NOT NULL, -- 置信度
    probability_distribution JSONB, -- 概率分布
    feature_importance JSONB, -- 特征重要性
    explanation TEXT, -- 分类解释
    is_verified BOOLEAN DEFAULT FALSE, -- 是否已验证
    verified_by UUID REFERENCES users(id),
    verified_at TIMESTAMP,
    feedback_score INTEGER, -- 反馈评分 1-5
    feedback_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 服务架构

#### 智能分类服务主类
```python
# src/services/classification_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import jieba
import spacy
import pickle
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from datetime import datetime
import uuid
import time
import json
import os

class IntelligentClassificationService:
    def __init__(self):
        # 初始化NLP工具
        self.nlp_zh = spacy.load('zh_core_web_sm')
        self.tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-chinese')
        
        # 数据库连接
        self.db = DatabaseManager()
        self.elasticsearch = ElasticsearchClient()
        self.message_queue = RabbitMQClient()
        
        # 模型存储路径
        self.model_storage_path = '/app/models/classification'
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        # 特征提取器
        self.feature_extractors = {
            'tfidf': TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
            'word2vec': None,  # 需要预训练模型
            'bert': None       # 使用transformers
        }
        
        # 分类器
        self.classifiers = {
            'svm': SVC(probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42)
        }
        
        # 预定义分类标签
        self.predefined_labels = {
            'topic': ['政治', '军事', '经济', '文化', '社会', '科技', '宗教', '教育'],
            'era': ['先秦', '秦汉', '魏晋南北朝', '隋唐', '宋元', '明清', '近代', '现代'],
            'document_type': ['史书', '文集', '诗词', '奏疏', '碑刻', '档案', '日记', '书信'],
            'importance': ['极高', '高', '中', '低']
        }
    
    async def create_classification_project(self, 
                                          name: str,
                                          description: str,
                                          classification_type: str,
                                          domain: str,
                                          config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建分类项目
        
        Args:
            name: 项目名称
            description: 项目描述
            classification_type: 分类类型
            domain: 领域
            config: 配置信息
            
        Returns:
            项目信息
        """
        try:
            project_id = str(uuid.uuid4())
            
            # 获取预定义标签
            class_labels = self.predefined_labels.get(
                classification_type, 
                config.get('custom_labels', [])
            )
            
            project_data = {
                'id': project_id,
                'name': name,
                'description': description,
                'classification_type': classification_type,
                'domain': domain,
                'class_labels': class_labels,
                'model_config': config.get('model_config', {
                    'feature_extractor': 'tfidf',
                    'classifier': 'svm',
                    'test_size': 0.2,
                    'cv_folds': 5
                }),
                'training_config': config.get('training_config', {
                    'min_samples_per_class': 10,
                    'max_features': 10000,
                    'ngram_range': [1, 2]
                }),
                'feature_config': config.get('feature_config', {
                    'use_tfidf': True,
                    'use_word2vec': False,
                    'use_bert': False,
                    'text_preprocessing': True
                })
            }
            
            await self.db.insert('classification_projects', project_data)
            
            return {
                'success': True,
                'project_id': project_id,
                'message': '分类项目创建成功'
            }
            
        except Exception as e:
            logger.error(f"创建分类项目失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def add_training_data(self, 
                              project_id: str,
                              text_content: str,
                              true_label: str,
                              document_id: str = None,
                              confidence: float = 1.0) -> Dict[str, Any]:
        """
        添加训练数据
        
        Args:
            project_id: 项目ID
            text_content: 文本内容
            true_label: 真实标签
            document_id: 文档ID
            confidence: 标签置信度
            
        Returns:
            添加结果
        """
        try:
            # 验证项目存在
            project = await self.db.select_one(
                'classification_projects',
                {'id': project_id}
            )
            
            if not project:
                raise ValueError(f"项目不存在: {project_id}")
            
            # 验证标签有效性
            valid_labels = project['class_labels']
            if true_label not in valid_labels:
                raise ValueError(f"无效标签: {true_label}，有效标签: {valid_labels}")
            
            # 文本预处理
            processed_text = await self._preprocess_text(text_content)
            
            # 提取文本特征
            text_features = await self._extract_text_features(processed_text)
            
            training_data = {
                'id': str(uuid.uuid4()),
                'project_id': project_id,
                'document_id': document_id,
                'text_content': text_content,
                'true_label': true_label,
                'label_confidence': confidence,
                'data_source': 'manual',
                'text_features': text_features
            }
            
            await self.db.insert('training_data', training_data)
            
            return {
                'success': True,
                'message': '训练数据添加成功'
            }
            
        except Exception as e:
            logger.error(f"添加训练数据失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def train_classification_model(self, 
                                       project_id: str,
                                       model_type: str = 'svm',
                                       feature_extractor: str = 'tfidf',
                                       hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        训练分类模型
        
        Args:
            project_id: 项目ID
            model_type: 模型类型
            feature_extractor: 特征提取器
            hyperparameters: 超参数
            
        Returns:
            训练结果
        """
        try:
            model_id = str(uuid.uuid4())
            start_time = time.time()
            
            # 创建模型记录
            model_data = {
                'id': model_id,
                'project_id': project_id,
                'model_name': f"{model_type}_{feature_extractor}_{int(time.time())}",
                'model_type': model_type,
                'feature_extractor': feature_extractor,
                'hyperparameters': hyperparameters or {},
                'status': 'training'
            }
            await self.db.insert('classification_models', model_data)
            
            # 获取训练数据
            training_data = await self.db.select(
                'training_data',
                {'project_id': project_id}
            )
            
            if len(training_data) < 10:
                raise ValueError("训练数据不足，至少需要10条数据")
            
            # 准备训练数据
            texts = [item['text_content'] for item in training_data]
            labels = [item['true_label'] for item in training_data]
            
            # 文本预处理
            processed_texts = [await self._preprocess_text(text) for text in texts]
            
            # 特征提取
            if feature_extractor == 'tfidf':
                features = await self._extract_tfidf_features(processed_texts)
            elif feature_extractor == 'bert':
                features = await self._extract_bert_features(processed_texts)
            else:
                raise ValueError(f"不支持的特征提取器: {feature_extractor}")
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # 模型训练
            if model_type == 'svm':
                model = await self._train_svm_model(X_train, y_train, hyperparameters)
            elif model_type == 'random_forest':
                model = await self._train_rf_model(X_train, y_train, hyperparameters)
            elif model_type == 'xgboost':
                model = await self._train_xgb_model(X_train, y_train, hyperparameters)
            elif model_type == 'bert':
                model = await self._train_bert_model(processed_texts, labels, hyperparameters)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 模型评估
            if model_type != 'bert':
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            else:
                # BERT模型评估需要特殊处理
                y_pred, y_pred_proba = await self._evaluate_bert_model(
                    model, X_test, y_test
                )
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # 交叉验证
            if model_type != 'bert':
                cv_scores = cross_val_score(model, features, labels, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean, cv_std = 0.0, 0.0  # BERT模型跳过交叉验证
            
            # 保存模型
            model_path = os.path.join(
                self.model_storage_path, 
                f"{model_id}_{model_type}_{feature_extractor}.pkl"
            )
            
            if model_type != 'bert':
                joblib.dump({
                    'model': model,
                    'feature_extractor': self.feature_extractors[feature_extractor],
                    'labels': list(set(labels))
                }, model_path)
            else:
                # BERT模型保存需要特殊处理
                model.save_pretrained(model_path)
            
            # 更新模型记录
            training_time = time.time() - start_time
            await self.db.update(
                'classification_models',
                {'id': model_id},
                {
                    'model_path': model_path,
                    'training_metrics': {
                        'accuracy': accuracy,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std
                    },
                    'test_metrics': classification_rep,
                    'training_data_size': len(training_data),
                    'training_time': training_time,
                    'status': 'completed'
                }
            )
            
            # 记录到MLflow
            with mlflow.start_run():
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("feature_extractor", feature_extractor)
                mlflow.log_param("training_size", len(training_data))
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("cv_mean", cv_mean)
                mlflow.log_metric("cv_std", cv_std)
                
                if model_type != 'bert':
                    mlflow.sklearn.log_model(model, "model")
            
            return {
                'success': True,
                'model_id': model_id,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time,
                'message': '模型训练完成'
            }
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            await self.db.update(
                'classification_models',
                {'id': model_id},
                {'status': 'failed', 'error_message': str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 去除特殊字符
        import re
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)
        
        # 中文分词
        words = jieba.cut(text)
        
        # 去除停用词
        stopwords = self._load_stopwords()
        filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
        
        return ' '.join(filtered_words)
    
    def _load_stopwords(self) -> set:
        """
        加载停用词表
        
        Returns:
            停用词集合
        """
        # 简化的停用词表
        return {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还是', '为了', '这个', '可以', '但是'
        }
    
    async def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """
        提取文本特征
        
        Args:
            text: 文本内容
            
        Returns:
            文本特征
        """
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('。') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }
        
        # 使用spaCy提取更多特征
        doc = self.nlp_zh(text)
        features.update({
            'noun_count': len([token for token in doc if token.pos_ == 'NOUN']),
            'verb_count': len([token for token in doc if token.pos_ == 'VERB']),
            'adj_count': len([token for token in doc if token.pos_ == 'ADJ'])
        })
        
        return features
    
    async def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """
        提取TF-IDF特征
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF特征矩阵
        """
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        features = tfidf.fit_transform(texts)
        self.feature_extractors['tfidf'] = tfidf
        
        return features.toarray()
    
    async def _extract_bert_features(self, texts: List[str]) -> np.ndarray:
        """
        提取BERT特征
        
        Args:
            texts: 文本列表
            
        Returns:
            BERT特征矩阵
        """
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained('bert-base-chinese')
        tokenizer = self.tokenizer_bert
        
        features = []
        for text in texts:
            inputs = tokenizer(
                text, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用[CLS]标记的向量作为文本表示
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                features.append(cls_embedding.flatten())
        
        return np.array(features)
    
    async def _train_svm_model(self, 
                             X_train: np.ndarray, 
                             y_train: List[str],
                             hyperparameters: Dict[str, Any] = None) -> SVC:
        """
        训练SVM模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            hyperparameters: 超参数
            
        Returns:
            训练好的SVM模型
        """
        params = hyperparameters or {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        }
        
        model = SVC(probability=True, random_state=42, **params)
        model.fit(X_train, y_train)
        
        return model
    
    async def _train_rf_model(self, 
                            X_train: np.ndarray, 
                            y_train: List[str],
                            hyperparameters: Dict[str, Any] = None) -> RandomForestClassifier:
        """
        训练随机森林模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            hyperparameters: 超参数
            
        Returns:
            训练好的随机森林模型
        """
        params = hyperparameters or {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2
        }
        
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        
        return model
    
    async def _train_xgb_model(self, 
                             X_train: np.ndarray, 
                             y_train: List[str],
                             hyperparameters: Dict[str, Any] = None) -> xgb.XGBClassifier:
        """
        训练XGBoost模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            hyperparameters: 超参数
            
        Returns:
            训练好的XGBoost模型
        """
        params = hyperparameters or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
        
        model = xgb.XGBClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        
        return model
    
    async def classify_document(self, 
                              project_id: str,
                              text_content: str,
                              model_id: str = None) -> Dict[str, Any]:
        """
        对文档进行分类
        
        Args:
            project_id: 项目ID
            text_content: 文档内容
            model_id: 模型ID（可选，使用活跃模型）
            
        Returns:
            分类结果
        """
        try:
            task_id = str(uuid.uuid4())
            start_time = time.time()
            
            # 获取模型
            if model_id:
                model_info = await self.db.select_one(
                    'classification_models',
                    {'id': model_id, 'status': 'completed'}
                )
            else:
                model_info = await self.db.select_one(
                    'classification_models',
                    {'project_id': project_id, 'is_active': True, 'status': 'completed'}
                )
            
            if not model_info:
                raise ValueError("未找到可用的分类模型")
            
            # 创建分类任务
            task_data = {
                'id': task_id,
                'project_id': project_id,
                'model_id': model_info['id'],
                'text_content': text_content[:1000],  # 只保存前1000字符
                'status': 'processing'
            }
            await self.db.insert('classification_tasks', task_data)
            
            # 文本预处理
            processed_text = await self._preprocess_text(text_content)
            
            # 加载模型
            model_data = joblib.load(model_info['model_path'])
            model = model_data['model']
            feature_extractor = model_data['feature_extractor']
            labels = model_data['labels']
            
            # 特征提取
            if model_info['feature_extractor'] == 'tfidf':
                features = feature_extractor.transform([processed_text]).toarray()
            elif model_info['feature_extractor'] == 'bert':
                features = await self._extract_bert_features([processed_text])
            
            # 预测
            predicted_label = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            confidence_score = max(prediction_proba)
            
            # 构建概率分布
            probability_distribution = {
                label: float(prob) for label, prob in zip(labels, prediction_proba)
            }
            
            # 特征重要性（仅对树模型）
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                top_features = np.argsort(model.feature_importances_)[-10:]
                feature_names = feature_extractor.get_feature_names_out()
                feature_importance = {
                    feature_names[i]: float(model.feature_importances_[i]) 
                    for i in top_features
                }
            
            # 保存分类结果
            result_data = {
                'id': str(uuid.uuid4()),
                'task_id': task_id,
                'project_id': project_id,
                'predicted_label': predicted_label,
                'confidence_score': confidence_score,
                'probability_distribution': probability_distribution,
                'feature_importance': feature_importance,
                'explanation': f"基于{model_info['model_type']}模型预测，置信度{confidence_score:.2%}"
            }
            
            await self.db.insert('classification_results', result_data)
            
            # 更新任务状态
            processing_time = time.time() - start_time
            await self.db.update(
                'classification_tasks',
                {'id': task_id},
                {
                    'status': 'completed',
                    'completed_at': datetime.now(),
                    'processing_time': processing_time
                }
            )
            
            return {
                'success': True,
                'task_id': task_id,
                'predicted_label': predicted_label,
                'confidence_score': confidence_score,
                'probability_distribution': probability_distribution,
                'feature_importance': feature_importance,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"文档分类失败: {str(e)}")
            await self.db.update(
                'classification_tasks',
                {'id': task_id},
                {'status': 'failed', 'error_message': str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def batch_classify_documents(self, 
                                     project_id: str,
                                     documents: List[Dict[str, Any]],
                                     model_id: str = None) -> Dict[str, Any]:
        """
        批量分类文档
        
        Args:
            project_id: 项目ID
            documents: 文档列表
            model_id: 模型ID
            
        Returns:
            批量分类结果
        """
        try:
            batch_task_id = str(uuid.uuid4())
            start_time = time.time()
            
            results = []
            for doc in documents:
                result = await self.classify_document(
                    project_id=project_id,
                    text_content=doc['content'],
                    model_id=model_id
                )
                
                result['document_id'] = doc.get('id')
                result['document_title'] = doc.get('title', '')
                results.append(result)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'batch_task_id': batch_task_id,
                'total_documents': len(documents),
                'successful_classifications': len([r for r in results if r['success']]),
                'results': results,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"批量分类失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型性能指标
        
        Args:
            model_id: 模型ID
            
        Returns:
            性能指标
        """
        try:
            model_info = await self.db.select_one(
                'classification_models',
                {'id': model_id}
            )
            
            if not model_info:
                raise ValueError(f"模型不存在: {model_id}")
            
            # 获取分类结果统计
            results = await self.db.select(
                'classification_results',
                {'project_id': model_info['project_id']}
            )
            
            # 计算统计指标
            total_predictions = len(results)
            avg_confidence = np.mean([r['confidence_score'] for r in results]) if results else 0
            
            # 置信度分布
            confidence_distribution = {
                'high (>0.8)': len([r for r in results if r['confidence_score'] > 0.8]),
                'medium (0.5-0.8)': len([r for r in results if 0.5 <= r['confidence_score'] <= 0.8]),
                'low (<0.5)': len([r for r in results if r['confidence_score'] < 0.5])
            }
            
            # 标签分布
            label_distribution = {}
            for result in results:
                label = result['predicted_label']
                label_distribution[label] = label_distribution.get(label, 0) + 1
            
            return {
                'success': True,
                'model_info': {
                    'model_id': model_id,
                    'model_name': model_info['model_name'],
                    'model_type': model_info['model_type'],
                    'feature_extractor': model_info['feature_extractor'],
                    'training_time': model_info['training_time'],
                    'training_data_size': model_info['training_data_size']
                },
                'training_metrics': model_info['training_metrics'],
                'test_metrics': model_info['test_metrics'],
                'usage_statistics': {
                    'total_predictions': total_predictions,
                    'avg_confidence': avg_confidence,
                    'confidence_distribution': confidence_distribution,
                    'label_distribution': label_distribution
                }
            }
            
        except Exception as e:
            logger.error(f"获取模型性能失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
```

### API设计

#### 智能分类API
```python
# 创建分类项目
POST /api/v1/classification/projects
Content-Type: application/json
Request: {
    "name": "明朝文档主题分类",
    "description": "对明朝历史文档进行主题分类",
    "classification_type": "topic",
    "domain": "历史",
    "config": {
        "model_config": {
            "feature_extractor": "tfidf",
            "classifier": "svm"
        }
    }
}
Response: {
    "success": true,
    "project_id": "uuid",
    "message": "分类项目创建成功"
}

# 添加训练数据
POST /api/v1/classification/training-data
Content-Type: application/json
Request: {
    "project_id": "uuid",
    "text_content": "朱元璋建立明朝，定都南京...",
    "true_label": "政治",
    "confidence": 1.0
}
Response: {
    "success": true,
    "message": "训练数据添加成功"
}

# 训练模型
POST /api/v1/classification/train
Content-Type: application/json
Request: {
    "project_id": "uuid",
    "model_type": "svm",
    "feature_extractor": "tfidf",
    "hyperparameters": {
        "C": 1.0,
        "kernel": "rbf"
    }
}
Response: {
    "success": true,
    "model_id": "uuid",
    "accuracy": 0.85,
    "cv_mean": 0.82,
    "cv_std": 0.03,
    "training_time": 45.2
}

# 文档分类
POST /api/v1/classification/classify
Content-Type: application/json
Request: {
    "project_id": "uuid",
    "text_content": "永乐帝迁都北京，加强中央集权...",
    "model_id": "uuid"
}
Response: {
    "success": true,
    "predicted_label": "政治",
    "confidence_score": 0.92,
    "probability_distribution": {
        "政治": 0.92,
        "军事": 0.05,
        "经济": 0.02,
        "文化": 0.01
    },
    "processing_time": 0.15
}

# 批量分类
POST /api/v1/classification/batch-classify
Content-Type: application/json
Request: {
    "project_id": "uuid",
    "documents": [
        {
            "id": "doc1",
            "title": "明史·太祖本纪",
            "content": "朱元璋..."
        }
    ]
}
Response: {
    "success": true,
    "total_documents": 1,
    "successful_classifications": 1,
    "results": [...]
}

# 获取模型性能
GET /api/v1/classification/models/{model_id}/performance
Response: {
    "success": true,
    "model_info": {
        "model_id": "uuid",
        "model_name": "svm_tfidf_1234567890",
        "model_type": "svm",
        "feature_extractor": "tfidf"
    },
    "training_metrics": {
        "accuracy": 0.85,
        "cv_mean": 0.82
    },
    "usage_statistics": {
        "total_predictions": 1500,
        "avg_confidence": 0.78
    }
}
```

### 前端集成

#### Vue3智能分类组件
```vue
<!-- src/components/classification/IntelligentClassification.vue -->
<template>
  <div class="intelligent-classification">
    <!-- 项目管理 -->
    <el-card class="project-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span>分类项目管理</span>
          <el-button type="primary" @click="showCreateProject = true">
            创建项目
          </el-button>
        </div>
      </template>
      
      <el-table :data="projects" v-loading="projectsLoading">
        <el-table-column prop="name" label="项目名称" />
        <el-table-column prop="classification_type" label="分类类型" />
        <el-table-column prop="domain" label="领域" />
        <el-table-column prop="status" label="状态">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作">
          <template #default="{ row }">
            <el-button size="small" @click="selectProject(row)">选择</el-button>
            <el-button size="small" type="info" @click="viewProject(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 训练数据管理 -->
    <el-card v-if="selectedProject" class="training-card" shadow="hover">
      <template #header>
        <span>训练数据管理 - {{ selectedProject.name }}</span>
      </template>
      
      <div class="training-controls">
        <el-button type="primary" @click="showAddTrainingData = true">
          添加训练数据
        </el-button>
        <el-button type="success" @click="trainModel" :loading="trainingLoading">
          训练模型
        </el-button>
      </div>
      
      <el-table :data="trainingData" v-loading="trainingDataLoading">
        <el-table-column prop="text_content" label="文本内容" show-overflow-tooltip />
        <el-table-column prop="true_label" label="标签" />
        <el-table-column prop="label_confidence" label="置信度" />
        <el-table-column prop="data_source" label="来源" />
        <el-table-column label="操作">
          <template #default="{ row }">
            <el-button size="small" type="danger" @click="deleteTrainingData(row.id)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 模型管理 -->
    <el-card v-if="selectedProject" class="model-card" shadow="hover">
      <template #header>
        <span>模型管理</span>
      </template>
      
      <el-table :data="models" v-loading="modelsLoading">
        <el-table-column prop="model_name" label="模型名称" />
        <el-table-column prop="model_type" label="模型类型" />
        <el-table-column prop="feature_extractor" label="特征提取器" />
        <el-table-column prop="status" label="状态">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="准确率">
          <template #default="{ row }">
            <span v-if="row.training_metrics">
              {{ (row.training_metrics.accuracy * 100).toFixed(1) }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column label="操作">
          <template #default="{ row }">
            <el-button size="small" @click="setActiveModel(row)" 
                      :disabled="row.status !== 'completed'">
              设为活跃
            </el-button>
            <el-button size="small" type="info" @click="viewModelPerformance(row)">
              性能
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 文档分类 -->
    <el-card v-if="selectedProject" class="classification-card" shadow="hover">
      <template #header>
        <span>文档分类</span>
      </template>
      
      <div class="classification-controls">
        <el-radio-group v-model="classificationMode">
          <el-radio-button label="single">单文档分类</el-radio-button>
          <el-radio-button label="batch">批量分类</el-radio-button>
        </el-radio-group>
      </div>
      
      <!-- 单文档分类 -->
      <div v-if="classificationMode === 'single'" class="single-classification">
        <el-input
          v-model="singleText"
          type="textarea"
          :rows="6"
          placeholder="请输入要分类的文档内容"
        />
        <div class="classification-actions">
          <el-button type="primary" @click="classifySingleDocument" 
                    :loading="classificationLoading">
            开始分类
          </el-button>
        </div>
        
        <!-- 分类结果 -->
        <div v-if="singleResult" class="classification-result">
          <h4>分类结果</h4>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="预测标签">
              <el-tag type="success">{{ singleResult.predicted_label }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="置信度">
              <el-progress :percentage="singleResult.confidence_score * 100" 
                          :color="getConfidenceColor(singleResult.confidence_score)" />
            </el-descriptions-item>
          </el-descriptions>
          
          <!-- 概率分布 -->
          <div class="probability-distribution">
            <h5>概率分布</h5>
            <div v-for="(prob, label) in singleResult.probability_distribution" 
                 :key="label" class="prob-item">
              <span class="label">{{ label }}</span>
              <el-progress :percentage="prob * 100" :show-text="false" />
              <span class="percentage">{{ (prob * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 批量分类 -->
      <div v-if="classificationMode === 'batch'" class="batch-classification">
        <el-upload
          class="upload-demo"
          drag
          :auto-upload="false"
          :on-change="handleFileChange"
          accept=".txt,.doc,.docx,.pdf"
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            拖拽文件到此处或<em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              支持txt/doc/docx/pdf格式文件
            </div>
          </template>
        </el-upload>
        
        <div class="batch-actions">
          <el-button type="primary" @click="batchClassifyDocuments" 
                    :loading="batchClassificationLoading"
                    :disabled="!batchFiles.length">
            批量分类
          </el-button>
        </div>
        
        <!-- 批量结果 -->
        <div v-if="batchResults" class="batch-results">
          <h4>批量分类结果</h4>
          <el-table :data="batchResults.results">
            <el-table-column prop="document_title" label="文档标题" />
            <el-table-column prop="predicted_label" label="预测标签">
              <template #default="{ row }">
                <el-tag>{{ row.predicted_label }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="confidence_score" label="置信度">
              <template #default="{ row }">
                <el-progress :percentage="row.confidence_score * 100" 
                            :show-text="false" />
                <span>{{ (row.confidence_score * 100).toFixed(1) }}%</span>
              </template>
            </el-table-column>
            <el-table-column label="操作">
              <template #default="{ row }">
                <el-button size="small" @click="viewDetailResult(row)">详情</el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-card>

    <!-- 创建项目对话框 -->
    <el-dialog v-model="showCreateProject" title="创建分类项目" width="600px">
      <el-form :model="newProject" label-width="120px">
        <el-form-item label="项目名称" required>
          <el-input v-model="newProject.name" placeholder="请输入项目名称" />
        </el-form-item>
        <el-form-item label="项目描述">
          <el-input v-model="newProject.description" type="textarea" :rows="3" />
        </el-form-item>
        <el-form-item label="分类类型" required>
          <el-select v-model="newProject.classification_type" placeholder="请选择分类类型">
            <el-option label="主题分类" value="topic" />
            <el-option label="时代分类" value="era" />
            <el-option label="文档类型" value="document_type" />
            <el-option label="重要性评级" value="importance" />
          </el-select>
        </el-form-item>
        <el-form-item label="领域">
          <el-input v-model="newProject.domain" placeholder="如：历史、文学等" />
        </el-form-item>
        <el-form-item label="模型配置">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-select v-model="newProject.feature_extractor" placeholder="特征提取器">
                <el-option label="TF-IDF" value="tfidf" />
                <el-option label="BERT" value="bert" />
              </el-select>
            </el-col>
            <el-col :span="12">
              <el-select v-model="newProject.classifier" placeholder="分类器">
                <el-option label="SVM" value="svm" />
                <el-option label="随机森林" value="random_forest" />
                <el-option label="XGBoost" value="xgboost" />
              </el-select>
            </el-col>
          </el-row>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showCreateProject = false">取消</el-button>
        <el-button type="primary" @click="createProject" :loading="createProjectLoading">
          创建
        </el-button>
      </template>
    </el-dialog>

    <!-- 添加训练数据对话框 -->
    <el-dialog v-model="showAddTrainingData" title="添加训练数据" width="700px">
      <el-form :model="newTrainingData" label-width="100px">
        <el-form-item label="文本内容" required>
          <el-input v-model="newTrainingData.text_content" 
                   type="textarea" :rows="6" 
                   placeholder="请输入文档内容" />
        </el-form-item>
        <el-form-item label="标签" required>
          <el-select v-model="newTrainingData.true_label" placeholder="请选择标签">
            <el-option v-for="label in selectedProject?.class_labels" 
                      :key="label" :label="label" :value="label" />
          </el-select>
        </el-form-item>
        <el-form-item label="置信度">
          <el-slider v-model="newTrainingData.confidence" 
                    :min="0.1" :max="1" :step="0.1" 
                    show-input />
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showAddTrainingData = false">取消</el-button>
        <el-button type="primary" @click="addTrainingData" :loading="addTrainingDataLoading">
          添加
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { classificationApi } from '@/api/classification'
import type { 
  ClassificationProject, 
  ClassificationModel, 
  TrainingData,
  ClassificationResult 
} from '@/types/classification'

// 响应式数据
const projects = ref<ClassificationProject[]>([])
const selectedProject = ref<ClassificationProject | null>(null)
const models = ref<ClassificationModel[]>([])
const trainingData = ref<TrainingData[]>([])

// 加载状态
const projectsLoading = ref(false)
const modelsLoading = ref(false)
const trainingDataLoading = ref(false)
const trainingLoading = ref(false)
const classificationLoading = ref(false)
const batchClassificationLoading = ref(false)
const createProjectLoading = ref(false)
const addTrainingDataLoading = ref(false)

// 对话框显示状态
const showCreateProject = ref(false)
const showAddTrainingData = ref(false)

// 分类模式
const classificationMode = ref('single')

// 单文档分类
const singleText = ref('')
const singleResult = ref<ClassificationResult | null>(null)

// 批量分类
const batchFiles = ref<File[]>([])
const batchResults = ref<any>(null)

// 新建项目表单
const newProject = reactive({
  name: '',
  description: '',
  classification_type: '',
  domain: '',
  feature_extractor: 'tfidf',
  classifier: 'svm'
})

// 新建训练数据表单
const newTrainingData = reactive({
  text_content: '',
  true_label: '',
  confidence: 1.0
})

/**
 * 组件挂载时加载数据
 */
onMounted(async () => {
  await loadProjects()
})

/**
 * 加载分类项目列表
 */
const loadProjects = async () => {
  try {
    projectsLoading.value = true
    const response = await classificationApi.getProjects()
    projects.value = response.data
  } catch (error) {
    ElMessage.error('加载项目列表失败')
    console.error('Load projects error:', error)
  } finally {
    projectsLoading.value = false
  }
}

/**
 * 选择项目
 */
const selectProject = async (project: ClassificationProject) => {
  selectedProject.value = project
  await Promise.all([
    loadModels(project.id),
    loadTrainingData(project.id)
  ])
}

/**
 * 加载模型列表
 */
const loadModels = async (projectId: string) => {
  try {
    modelsLoading.value = true
    const response = await classificationApi.getModels(projectId)
    models.value = response.data
  } catch (error) {
    ElMessage.error('加载模型列表失败')
    console.error('Load models error:', error)
  } finally {
    modelsLoading.value = false
  }
}

/**
 * 加载训练数据
 */
const loadTrainingData = async (projectId: string) => {
  try {
    trainingDataLoading.value = true
    const response = await classificationApi.getTrainingData(projectId)
    trainingData.value = response.data
  } catch (error) {
    ElMessage.error('加载训练数据失败')
    console.error('Load training data error:', error)
  } finally {
    trainingDataLoading.value = false
  }
}

/**
 * 创建项目
 */
const createProject = async () => {
  try {
    createProjectLoading.value = true
    
    const projectData = {
      name: newProject.name,
      description: newProject.description,
      classification_type: newProject.classification_type,
      domain: newProject.domain,
      config: {
        model_config: {
          feature_extractor: newProject.feature_extractor,
          classifier: newProject.classifier
        }
      }
    }
    
    await classificationApi.createProject(projectData)
    
    ElMessage.success('项目创建成功')
    showCreateProject.value = false
    
    // 重置表单
    Object.assign(newProject, {
      name: '',
      description: '',
      classification_type: '',
      domain: '',
      feature_extractor: 'tfidf',
      classifier: 'svm'
    })
    
    await loadProjects()
  } catch (error) {
    ElMessage.error('创建项目失败')
    console.error('Create project error:', error)
  } finally {
    createProjectLoading.value = false
  }
}

/**
 * 添加训练数据
 */
const addTrainingData = async () => {
  try {
    addTrainingDataLoading.value = true
    
    await classificationApi.addTrainingData({
      project_id: selectedProject.value!.id,
      text_content: newTrainingData.text_content,
      true_label: newTrainingData.true_label,
      confidence: newTrainingData.confidence
    })
    
    ElMessage.success('训练数据添加成功')
    showAddTrainingData.value = false
    
    // 重置表单
    Object.assign(newTrainingData, {
      text_content: '',
      true_label: '',
      confidence: 1.0
    })
    
    await loadTrainingData(selectedProject.value!.id)
  } catch (error) {
    ElMessage.error('添加训练数据失败')
    console.error('Add training data error:', error)
  } finally {
    addTrainingDataLoading.value = false
  }
}

/**
 * 训练模型
 */
const trainModel = async () => {
  try {
    await ElMessageBox.confirm(
      '训练模型可能需要较长时间，确定要开始训练吗？',
      '确认训练',
      {
        confirmButtonText: '开始训练',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    trainingLoading.value = true
    
    const response = await classificationApi.trainModel({
      project_id: selectedProject.value!.id,
      model_type: selectedProject.value!.model_config.classifier,
      feature_extractor: selectedProject.value!.model_config.feature_extractor
    })
    
    ElMessage.success(`模型训练完成，准确率: ${(response.data.accuracy * 100).toFixed(1)}%`)
    
    await loadModels(selectedProject.value!.id)
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('模型训练失败')
      console.error('Train model error:', error)
    }
  } finally {
    trainingLoading.value = false
  }
}

/**
 * 单文档分类
 */
const classifySingleDocument = async () => {
  try {
    classificationLoading.value = true
    
    const response = await classificationApi.classifyDocument({
      project_id: selectedProject.value!.id,
      text_content: singleText.value
    })
    
    singleResult.value = response.data
    ElMessage.success('文档分类完成')
  } catch (error) {
    ElMessage.error('文档分类失败')
    console.error('Classify document error:', error)
  } finally {
    classificationLoading.value = false
  }
}

/**
 * 批量分类文档
 */
const batchClassifyDocuments = async () => {
  try {
    batchClassificationLoading.value = true
    
    // 读取文件内容
    const documents = await Promise.all(
      batchFiles.value.map(async (file) => {
        const content = await readFileContent(file)
        return {
          id: file.name,
          title: file.name,
          content: content
        }
      })
    )
    
    const response = await classificationApi.batchClassifyDocuments({
      project_id: selectedProject.value!.id,
      documents: documents
    })
    
    batchResults.value = response.data
    ElMessage.success(`批量分类完成，成功分类 ${response.data.successful_classifications} 个文档`)
  } catch (error) {
    ElMessage.error('批量分类失败')
    console.error('Batch classify error:', error)
  } finally {
    batchClassificationLoading.value = false
  }
}

/**
 * 读取文件内容
 */
const readFileContent = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => resolve(e.target?.result as string)
    reader.onerror = reject
    reader.readAsText(file, 'utf-8')
  })
}

/**
 * 处理文件变化
 */
const handleFileChange = (file: any) => {
  batchFiles.value = [file.raw]
}

/**
 * 获取状态类型
 */
const getStatusType = (status: string) => {
  const statusMap: Record<string, string> = {
    'active': 'success',
    'training': 'warning',
    'completed': 'success',
    'failed': 'danger',
    'archived': 'info'
  }
  return statusMap[status] || 'info'
}

/**
 * 获取置信度颜色
 */
const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return '#67c23a'
  if (confidence >= 0.6) return '#e6a23c'
  return '#f56c6c'
}

/**
 * 设置活跃模型
 */
const setActiveModel = async (model: ClassificationModel) => {
  try {
    await classificationApi.setActiveModel(model.id)
    ElMessage.success('活跃模型设置成功')
    await loadModels(selectedProject.value!.id)
  } catch (error) {
    ElMessage.error('设置活跃模型失败')
    console.error('Set active model error:', error)
  }
}

/**
 * 查看模型性能
 */
const viewModelPerformance = async (model: ClassificationModel) => {
  try {
    const response = await classificationApi.getModelPerformance(model.id)
    // 显示性能详情对话框
    console.log('Model performance:', response.data)
  } catch (error) {
    ElMessage.error('获取模型性能失败')
    console.error('Get model performance error:', error)
  }
}
</script>

<style scoped>
.intelligent-classification {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.project-card,
.training-card,
.model-card,
.classification-card {
  margin-bottom: 20px;
}

.training-controls,
.classification-controls,
.classification-actions,
.batch-actions {
  margin-bottom: 20px;
}

.classification-result {
  margin-top: 20px;
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.probability-distribution {
  margin-top: 15px;
}

.prob-item {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.prob-item .label {
  width: 80px;
  margin-right: 10px;
}

.prob-item .el-progress {
  flex: 1;
  margin-right: 10px;
}

.prob-item .percentage {
  width: 50px;
  text-align: right;
}

.batch-results {
  margin-top: 20px;
}

.upload-demo {
  margin-bottom: 20px;
}
</style>
```

## 验收标准

### 功能性验收标准
1. **项目管理功能**
   - ✅ 能够创建不同类型的分类项目（主题、时代、文档类型、重要性）
   - ✅ 支持自定义分类标签和配置参数
   - ✅ 项目状态管理和权限控制

2. **训练数据管理**
   - ✅ 支持手动添加训练数据
   - ✅ 训练数据质量验证和标签一致性检查
   - ✅ 支持批量导入训练数据
   - ✅ 训练数据统计和分布分析

3. **模型训练功能**
   - ✅ 支持多种机器学习算法（SVM、随机森林、XGBoost、BERT）
   - ✅ 支持多种特征提取方法（TF-IDF、Word2Vec、BERT）
   - ✅ 自动模型评估和性能指标计算
   - ✅ 模型版本管理和比较

4. **文档分类功能**
   - ✅ 单文档实时分类
   - ✅ 批量文档分类处理
   - ✅ 分类结果置信度评估
   - ✅ 分类解释和特征重要性分析

5. **结果管理功能**
   - ✅ 分类结果存储和检索
   - ✅ 结果验证和反馈机制
   - ✅ 分类统计和报告生成

### 性能验收标准
1. **响应时间要求**
   - 单文档分类响应时间 < 2秒
   - 批量分类（100个文档）完成时间 < 5分钟
   - 模型训练时间（1000条数据）< 10分钟
   - API响应时间 < 500ms

2. **准确性要求**
   - 主题分类准确率 > 85%
   - 时代分类准确率 > 90%
   - 文档类型分类准确率 > 88%
   - 重要性评级准确率 > 80%

3. **并发性能**
   - 支持100个并发分类请求
   - 系统资源使用率 < 80%
   - 内存使用 < 4GB

4. **可扩展性**
   - 支持10万级训练数据
   - 支持100个分类项目
   - 支持50个并行模型训练

### 安全性验收标准
1. **数据安全**
   - 训练数据加密存储
   - 模型文件访问控制
   - 分类结果数据保护

2. **访问控制**
   - 基于角色的权限管理
   - API访问认证和授权
   - 操作日志记录

3. **模型安全**
   - 模型版本控制和回滚
   - 恶意输入检测和过滤
   - 模型性能监控和异常检测

## 业务价值

### 直接价值
1. **提升分类效率**
   - 自动化文档分类，减少90%人工分类工作量
   - 批量处理能力，支持大规模文档分类
   - 实时分类响应，提升用户体验

2. **提高分类准确性**
   - 机器学习算法保证分类一致性
   - 多维度分类提供全面文档标签
   - 置信度评估帮助质量控制

3. **降低运营成本**
   - 减少人工分类人力成本
   - 提高文档组织和检索效率
   - 降低分类错误带来的损失

### 间接价值
1. **知识管理优化**
   - 智能分类支持知识图谱构建
   - 提升文档检索和发现能力
   - 支持个性化推荐系统

2. **决策支持**
   - 分类统计支持业务分析
   - 文档趋势分析和预测
   - 内容质量评估和优化

3. **技术积累**
   - 机器学习技术能力建设
   - 文本处理和NLP技术积累
   - 为其他AI应用提供基础

## 依赖关系

### 前置依赖
1. **基础设施依赖**
   - Story 1.1: 微服务基础设施（Docker、Kubernetes）
   - Story 1.2: 用户认证授权服务
   - Story 1.4: 系统监控和日志服务

2. **数据依赖**
   - Story 1.3: 数据收集和存储服务
   - Story 2.2: NLP文本处理服务（文本预处理）

3. **技术依赖**
   - PostgreSQL数据库
   - Redis缓存服务
   - Elasticsearch搜索引擎
   - RabbitMQ消息队列

### 后置影响
1. **服务集成**
   - Story 3.1: 搜索引擎服务（分类标签检索）
   - Story 3.2: 推荐系统服务（基于分类的推荐）
   - Story 4.1: 数据可视化（分类统计图表）

2. **功能扩展**
   - 个性化分类模型
   - 多语言分类支持
   - 实时分类流处理

## 风险和缓解措施

### 技术风险
1. **模型准确性风险**
   - **风险**: 分类准确率不达标
   - **缓解**: 多算法对比测试，持续优化训练数据
   - **监控**: 实时准确率监控，用户反馈收集

2. **性能风险**
   - **风险**: 大规模数据处理性能瓶颈
   - **缓解**: 分布式处理，模型优化，缓存策略
   - **监控**: 性能指标监控，资源使用监控

3. **模型过拟合风险**
   - **风险**: 模型在新数据上表现差
   - **缓解**: 交叉验证，正则化，数据增强
   - **监控**: 验证集性能监控，A/B测试

### 业务风险
1. **数据质量风险**
   - **风险**: 训练数据质量影响模型效果
   - **缓解**: 数据质量检查，专家标注，多轮验证
   - **监控**: 数据质量指标，标注一致性检查

2. **用户接受度风险**
   - **风险**: 用户不信任自动分类结果
   - **缓解**: 提供分类解释，支持人工校正，渐进式部署
   - **监控**: 用户满意度调研，使用率统计

### 运维风险
1. **模型维护风险**
   - **风险**: 模型性能随时间衰减
   - **缓解**: 定期重训练，增量学习，性能监控
   - **监控**: 模型性能趋势，数据漂移检测

2. **资源消耗风险**
   - **风险**: 模型训练和推理消耗大量资源
   - **缓解**: 资源调度优化，模型压缩，边缘计算
   - **监控**: 资源使用监控，成本控制

## 开发任务分解

### 后端开发任务（4-5天）
1. **Day 1: 基础架构搭建**
   - 创建FastAPI项目结构
   - 配置数据库连接和模型
   - 实现基础的CRUD操作
   - 集成消息队列和缓存

2. **Day 2: 核心分类服务开发**
   - 实现文本预处理功能
   - 开发特征提取模块（TF-IDF、BERT）
   - 实现机器学习模型训练
   - 开发模型评估和验证

3. **Day 3: 分类功能实现**
   - 实现单文档分类API
   - 开发批量分类处理
   - 实现分类结果存储和检索
   - 添加分类解释和置信度计算

4. **Day 4: 模型管理功能**
   - 实现模型版本管理
   - 开发模型性能监控
   - 实现模型部署和切换
   - 集成MLflow模型管理

5. **Day 5: 优化和测试**
   - 性能优化和调试
   - 单元测试和集成测试
   - API文档完善
   - 部署配置和监控

### 前端开发任务（2-3天）
1. **Day 1: 基础组件开发**
   - 创建分类项目管理界面
   - 实现训练数据管理组件
   - 开发模型管理界面

2. **Day 2: 分类功能界面**
   - 实现单文档分类界面
   - 开发批量分类上传组件
   - 创建分类结果展示组件

3. **Day 3: 优化和集成**
   - 界面优化和用户体验改进
   - API集成和错误处理
   - 响应式设计和兼容性测试

### 测试和部署任务（1天）
1. **功能测试**
   - 分类准确性测试
   - 性能压力测试
   - 用户界面测试

2. **部署上线**
   - 生产环境部署
   - 监控配置
   - 用户培训和文档
```