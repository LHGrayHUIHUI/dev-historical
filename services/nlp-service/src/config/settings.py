"""
NLP服务配置管理
无状态NLP文本处理微服务配置
数据存储通过storage-service完成
"""

from pydantic_settings import BaseSettings
from typing import Dict, Any, List
import os


class Settings(BaseSettings):
    """NLP服务配置（无状态架构）"""
    
    # 基础服务配置
    service_name: str = "nlp-service"
    service_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8004  # 使用8004端口避免与其他服务冲突
    api_prefix: str = "/api/v1"
    workers: int = 1
    
    # Storage Service配置（统一数据管理）
    storage_service_url: str = "http://localhost:8002"
    storage_service_timeout: int = 60  # NLP处理可能需要更长时间
    storage_service_retries: int = 3
    
    # File Processor配置（可选，如果需要调用其他文件处理功能）
    file_processor_url: str = "http://localhost:8001"
    
    # OCR Service配置（可选，如果需要调用OCR功能）
    ocr_service_url: str = "http://localhost:8003"
    
    # NLP引擎配置
    default_nlp_engine: str = "spacy"  # spacy, jieba, hanlp
    default_language: str = "zh"  # zh, en, zh-classical
    supported_languages: List[str] = ["zh", "en", "zh-classical"]
    max_text_length: int = 1000000  # 1MB文本
    max_batch_size: int = 50
    nlp_task_timeout: int = 300  # 5分钟
    max_concurrent_tasks: int = 4
    
    # NLP功能开关
    enable_segmentation: bool = True
    enable_pos_tagging: bool = True
    enable_ner: bool = True
    enable_sentiment_analysis: bool = True
    enable_keyword_extraction: bool = True
    enable_text_summarization: bool = True
    enable_text_similarity: bool = True
    
    # 模型配置
    spacy_model: str = "zh_core_web_sm"
    hanlp_tokenizer_model: str = "FINE_ELECTRA_SMALL_ZH"
    hanlp_ner_model: str = "MSRA_NER_ELECTRA_SMALL_ZH"
    sentiment_model: str = "uer/roberta-base-finetuned-chinanews-chinese"
    sentence_model: str = "shibing624/text2vec-base-chinese"
    
    # 分词配置
    jieba_dict_path: str = ""  # 自定义词典路径
    jieba_enable_parallel: bool = True
    jieba_parallel_workers: int = 4
    
    # 关键词提取配置
    keyword_extraction_methods: List[str] = ["tfidf", "textrank", "yake"]
    default_keyword_method: str = "textrank"
    max_keywords_count: int = 50
    
    # 摘要生成配置
    summary_methods: List[str] = ["extractive", "abstractive"]
    default_summary_method: str = "extractive"
    max_summary_sentences: int = 5
    summary_compression_ratio: float = 0.3
    
    # 临时文件配置
    temp_dir: str = "/tmp/nlp-service"
    temp_file_cleanup_interval: int = 3600
    temp_file_max_age: int = 7200
    
    # 缓存配置（本地缓存，不使用外部Redis）
    enable_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl: int = 3600
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = ""
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    enable_json_logs: bool = False
    log_request_id: bool = True
    
    # 开发环境特定配置
    debug_show_error_details: bool = True
    debug_reload_on_change: bool = True
    
    # GPU配置
    use_gpu: bool = False  # 在生产环境中可能开启
    gpu_device: int = 0
    
    # 模型路径配置
    models_dir: str = "./models"
    custom_dict_dir: str = "./dictionaries"
    
    class Config:
        env_prefix = "NLP_SERVICE_"
        env_file = ".env"


# 全局配置实例
settings = Settings()


def get_nlp_engine_config() -> Dict[str, Any]:
    """获取NLP引擎配置"""
    return {
        "spacy": {
            "model": settings.spacy_model,
            "disable": ["parser"] if not settings.enable_pos_tagging else [],
            "exclude": ["tagger"] if not settings.enable_pos_tagging else []
        },
        "jieba": {
            "dict_path": settings.jieba_dict_path,
            "enable_parallel": settings.jieba_enable_parallel,
            "parallel_workers": settings.jieba_parallel_workers
        },
        "hanlp": {
            "tokenizer_model": settings.hanlp_tokenizer_model,
            "ner_model": settings.hanlp_ner_model,
            "device": f"cuda:{settings.gpu_device}" if settings.use_gpu else "cpu"
        },
        "transformers": {
            "sentiment_model": settings.sentiment_model,
            "sentence_model": settings.sentence_model,
            "device": settings.gpu_device if settings.use_gpu else -1
        }
    }


def get_feature_config() -> Dict[str, bool]:
    """获取功能开关配置"""
    return {
        "segmentation": settings.enable_segmentation,
        "pos_tagging": settings.enable_pos_tagging,
        "ner": settings.enable_ner,
        "sentiment_analysis": settings.enable_sentiment_analysis,
        "keyword_extraction": settings.enable_keyword_extraction,
        "text_summarization": settings.enable_text_summarization,
        "text_similarity": settings.enable_text_similarity
    }