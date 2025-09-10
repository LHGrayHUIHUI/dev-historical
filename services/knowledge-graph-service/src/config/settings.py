"""
知识图谱构建服务配置管理
无状态知识图谱微服务配置
数据存储通过storage-service完成
"""

from pydantic_settings import BaseSettings
from typing import Dict, Any, List
import os


class Settings(BaseSettings):
    """知识图谱构建服务配置（无状态架构）"""
    
    # 基础服务配置
    service_name: str = "knowledge-graph-service"
    service_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8006  # 使用8006端口避免与其他服务冲突
    api_prefix: str = "/api/v1"
    workers: int = 1
    
    # Storage Service配置（统一数据管理）
    storage_service_url: str = "http://localhost:8002"
    storage_service_timeout: int = 180  # 知识图谱构建可能需要更长时间
    storage_service_retries: int = 3
    
    # 其他服务配置（可选调用）
    nlp_service_url: str = "http://localhost:8004"
    ocr_service_url: str = "http://localhost:8003"
    
    # 知识图谱构建配置
    max_text_length: int = 10000  # 单次处理最大文本长度
    max_batch_size: int = 50  # 批量处理最大文档数
    graph_construction_timeout: int = 600  # 10分钟
    max_concurrent_tasks: int = 3  # 知识图谱构建是计算密集型
    
    # 支持的实体类型
    supported_entity_types: List[str] = [
        "PERSON",       # 人物
        "LOCATION",     # 地点  
        "ORGANIZATION", # 组织
        "EVENT",        # 事件
        "TIME",         # 时间
        "CONCEPT",      # 概念
        "OBJECT",       # 物品
        "WORK",         # 作品
    ]
    
    # 支持的关系类型
    supported_relation_types: List[str] = [
        "出生于",    # BORN_IN
        "死于",      # DIED_IN
        "任职于",    # WORKED_AT
        "位于",      # LOCATED_IN
        "创建",      # FOUNDED
        "影响",      # INFLUENCED
        "参与",      # PARTICIPATED_IN
        "属于",      # BELONGS_TO
        "统治",      # RULED
        "继承",      # INHERITED
        "师从",      # LEARNED_FROM
        "包含",      # CONTAINS
    ]
    
    # NLP模型配置
    spacy_model_zh: str = "zh_core_web_sm"
    spacy_model_en: str = "en_core_web_sm"
    bert_model_name: str = "bert-base-chinese"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    # 实体识别配置
    entity_confidence_threshold: float = 0.75
    entity_similarity_threshold: float = 0.85  # 实体去重相似度阈值
    max_entity_length: int = 50  # 最大实体长度
    min_entity_length: int = 2   # 最小实体长度
    
    # 关系抽取配置
    relation_confidence_threshold: float = 0.70
    max_relation_distance: int = 100  # 实体间最大距离（字符）
    relation_pattern_file: str = "relation_patterns.json"
    
    # 图谱构建配置
    graph_max_nodes: int = 10000     # 单个图谱最大节点数
    graph_max_edges: int = 50000     # 单个图谱最大边数
    graph_clustering_threshold: float = 0.3  # 图聚类阈值
    graph_centrality_algorithms: List[str] = [
        "betweenness", "closeness", "degree", "eigenvector"
    ]
    
    # 概念挖掘配置
    topic_model_num_topics: int = 20
    topic_model_passes: int = 10
    word_embedding_dimensions: int = 300
    min_concept_frequency: int = 3
    
    # 文件处理配置
    temp_dir: str = "/tmp/knowledge-graph"
    upload_dir: str = "/tmp/uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    supported_text_formats: List[str] = [
        "txt", "md", "json", "csv"
    ]
    
    # 缓存配置（本地缓存，不使用外部Redis）
    enable_cache: bool = True
    cache_max_size: int = 1000  # 最大缓存条目数
    cache_ttl: int = 3600  # 1小时
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    log_file: str = "logs/knowledge_graph_service.log"
    
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
    
    # 实体链接配置
    entity_linking_enabled: bool = True
    entity_linking_threshold: float = 0.8
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    
    # 图谱质量评估配置
    quality_metrics: List[str] = [
        "completeness",    # 完整性
        "consistency",     # 一致性  
        "accuracy",        # 准确性
        "coverage",        # 覆盖率
        "connectivity"     # 连通性
    ]
    
    # 性能优化配置
    batch_processing_enabled: bool = True
    parallel_processing_enabled: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    
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
    
    def get_model_path(self, model_type: str) -> str:
        """获取模型路径"""
        model_paths = {
            "spacy_zh": self.spacy_model_zh,
            "spacy_en": self.spacy_model_en,
            "bert": self.bert_model_name,
            "sentence_transformer": self.sentence_transformer_model
        }
        return model_paths.get(model_type, "")
    
    def is_supported_entity_type(self, entity_type: str) -> bool:
        """检查是否为支持的实体类型"""
        return entity_type in self.supported_entity_types
    
    def is_supported_relation_type(self, relation_type: str) -> bool:
        """检查是否为支持的关系类型"""
        return relation_type in self.supported_relation_types


# 全局配置实例
settings = Settings()