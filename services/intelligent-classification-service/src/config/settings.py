"""
智能分类服务配置管理
无状态智能分类微服务配置
数据存储通过storage-service完成
"""

from pydantic_settings import BaseSettings
from typing import Dict, Any, List
import os


class Settings(BaseSettings):
    """智能分类服务配置（无状态架构）"""
    
    # 基础服务配置
    service_name: str = "intelligent-classification-service"
    service_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8007  # 使用8007端口避免与其他服务冲突
    api_prefix: str = "/api/v1"
    workers: int = 1
    
    # Storage Service配置（统一数据管理）
    storage_service_url: str = "http://localhost:8002"
    storage_service_timeout: int = 180  # 模型训练可能需要更长时间
    storage_service_retries: int = 3
    
    # 其他服务配置（可选调用）
    nlp_service_url: str = "http://localhost:8004"
    knowledge_graph_service_url: str = "http://localhost:8006"
    
    # 智能分类配置
    max_text_length: int = 10000  # 单次处理最大文本长度
    max_batch_size: int = 100  # 批量分类最大文档数
    classification_timeout: int = 300  # 5分钟分类超时
    max_concurrent_tasks: int = 5  # 最大并发任务数
    
    # 预定义分类标签
    predefined_labels: Dict[str, List[str]] = {
        "topic": ["政治", "军事", "经济", "文化", "社会", "科技", "宗教", "教育"],
        "era": ["先秦", "秦汉", "魏晋南北朝", "隋唐", "宋元", "明清", "近代", "现代"],
        "document_type": ["史书", "文集", "诗词", "奏疏", "碑刻", "档案", "日记", "书信"],
        "importance": ["极高", "高", "中", "低"],
        "sentiment": ["正面", "负面", "中性"],
        "genre": ["纪传体", "编年体", "纪事本末体", "政书", "杂史"]
    }
    
    # 机器学习模型配置
    ml_models: Dict[str, Dict[str, Any]] = {
        "svm": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": 42
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42
        }
    }
    
    # NLP模型配置
    nlp_models: Dict[str, str] = {
        "spacy_zh": "zh_core_web_sm",
        "spacy_en": "en_core_web_sm",
        "bert_chinese": "bert-base-chinese",
        "roberta_chinese": "hfl/chinese-roberta-wwm-ext",
        "sentence_transformer": "all-MiniLM-L6-v2"
    }
    
    # 特征提取配置
    feature_extraction: Dict[str, Any] = {
        "tfidf": {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": None
        },
        "word2vec": {
            "vector_size": 300,
            "window": 5,
            "min_count": 1,
            "workers": 4
        },
        "fasttext": {
            "vector_size": 300,
            "window": 5,
            "min_count": 1,
            "workers": 4,
            "sg": 1
        }
    }
    
    # 模型训练配置
    training_config: Dict[str, Any] = {
        "test_size": 0.2,
        "validation_size": 0.1,
        "cv_folds": 5,
        "random_state": 42,
        "min_samples_per_class": 5,
        "max_training_time": 1800,  # 30分钟
        "early_stopping": True,
        "patience": 10
    }
    
    # BERT训练配置
    bert_training: Dict[str, Any] = {
        "max_length": 512,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_steps": 0.1,
        "weight_decay": 0.01
    }
    
    # 文本预处理配置
    text_preprocessing: Dict[str, Any] = {
        "remove_punctuation": True,
        "remove_numbers": False,
        "remove_whitespace": True,
        "lowercase": True,
        "remove_stopwords": True,
        "min_word_length": 2,
        "max_word_length": 50
    }
    
    # 中文停用词
    chinese_stopwords: List[str] = [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", 
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", 
        "没有", "看", "好", "自己", "这", "那", "以", "为", "能", "可以", "这个",
        "但是", "因为", "所以", "如果", "虽然", "然而", "而且", "或者", "不过",
        "于是", "然后", "因此", "由于", "无论", "不管", "除了", "除非"
    ]
    
    # 模型性能阈值
    performance_thresholds: Dict[str, float] = {
        "min_accuracy": 0.7,
        "min_precision": 0.7,
        "min_recall": 0.7,
        "min_f1": 0.7,
        "confidence_threshold": 0.6
    }
    
    # 文件处理配置
    temp_dir: str = "/tmp/intelligent-classification"
    model_dir: str = "/tmp/models"
    cache_dir: str = "/tmp/cache"
    max_model_size: int = 500 * 1024 * 1024  # 500MB
    
    # 缓存配置（本地缓存，不使用外部Redis）
    enable_cache: bool = True
    cache_max_size: int = 1000  # 最大缓存条目数
    cache_ttl: int = 3600  # 1小时
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    log_file: str = "logs/intelligent_classification_service.log"
    
    # CORS配置
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ]
    
    # API限流配置
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # 健康检查配置
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # 多语言支持配置
    default_language: str = "zh"
    supported_languages: List[str] = ["zh", "en"]
    auto_detect_language: bool = True
    
    # MLflow配置
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "intelligent-classification"
    mlflow_enabled: bool = True
    
    # 模型版本管理
    max_model_versions: int = 10
    auto_archive_old_models: bool = True
    model_retention_days: int = 30
    
    # 性能监控配置
    enable_performance_monitoring: bool = True
    performance_log_interval: int = 300  # 5分钟
    
    # 批量处理配置
    batch_processing_enabled: bool = True
    max_batch_concurrent: int = 3
    batch_chunk_size: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @property
    def port(self) -> int:
        """获取服务端口"""
        return self.api_port
    
    @property
    def host(self) -> str:
        """获取服务主机"""
        return self.api_host
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取模型配置"""
        return self.ml_models.get(model_type, {})
    
    def get_nlp_model_path(self, model_type: str) -> str:
        """获取NLP模型路径"""
        return self.nlp_models.get(model_type, "")
    
    def get_feature_config(self, feature_type: str) -> Dict[str, Any]:
        """获取特征提取配置"""
        return self.feature_extraction.get(feature_type, {})
    
    def is_supported_classification_type(self, classification_type: str) -> bool:
        """检查是否为支持的分类类型"""
        return classification_type in self.predefined_labels
    
    def get_classification_labels(self, classification_type: str) -> List[str]:
        """获取分类标签"""
        return self.predefined_labels.get(classification_type, [])


# 全局配置实例
settings = Settings()