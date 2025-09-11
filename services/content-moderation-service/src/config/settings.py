"""
内容审核服务配置管理

统一管理数据库连接、Redis缓存、AI模型、审核规则等配置
使用Pydantic Settings实现类型安全的配置加载
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, validator, Field
import os


class Settings(BaseSettings):
    """
    应用程序配置类
    
    支持从环境变量和.env文件加载配置
    提供配置验证和默认值设置
    """
    
    # 应用基础配置
    app_name: str = "Content Moderation Service"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8090
    
    # 数据库配置
    database_url: str = "postgresql://postgres:password@localhost:5433/historical_text_moderation"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Redis配置
    redis_url: str = "redis://localhost:6380/10"
    redis_max_connections: int = 100
    redis_socket_timeout: int = 5
    
    # Celery配置
    celery_broker_url: str = "redis://localhost:6380/11"
    celery_result_backend: str = "redis://localhost:6380/12"
    celery_task_timeout: int = 1800  # 30分钟
    celery_max_retries: int = 3
    
    # JWT配置
    jwt_secret_key: str = "moderation-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    
    # 文件处理配置
    max_file_size: int = 200 * 1024 * 1024  # 200MB
    supported_image_types: List[str] = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    supported_video_types: List[str] = ["video/mp4", "video/avi", "video/mov", "video/mkv"]
    supported_audio_types: List[str] = ["audio/mp3", "audio/wav", "audio/aac", "audio/ogg"]
    temp_storage_path: str = "/tmp/moderation"
    
    # AI模型配置
    text_model_name: str = "bert-base-chinese"
    image_model_path: str = "./models/nsfw_mobilenet_v2.h5"
    video_frame_sample_rate: int = 1  # 每秒采样帧数
    audio_segment_length: int = 10  # 音频分段长度(秒)
    
    # 审核阈值配置
    text_confidence_threshold: float = 0.8
    image_confidence_threshold: float = 0.7
    video_confidence_threshold: float = 0.75
    audio_confidence_threshold: float = 0.6
    
    # 敏感词配置
    sensitive_words_file: str = "./data/sensitive_words.txt"
    custom_dict_file: str = "./data/custom_dict.txt"
    
    # API限流配置
    rate_limit_per_minute: int = 1000
    rate_limit_per_hour: int = 10000
    
    # 监控配置
    enable_prometheus: bool = True
    prometheus_port: int = 9191
    log_level: str = "INFO"
    
    # 安全配置
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    allowed_hosts: List[str] = ["*"]
    
    # 外部服务配置
    ai_model_service_url: str = "http://localhost:8010"
    text_publishing_service_url: str = "http://localhost:8080"
    
    @validator("database_url")
    def validate_database_url(cls, v):
        """验证数据库连接URL格式"""
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("数据库URL必须以postgresql://或postgres://开头")
        return v
    
    @validator("redis_url")
    def validate_redis_url(cls, v):
        """验证Redis连接URL格式"""
        if not v.startswith("redis://"):
            raise ValueError("Redis URL必须以redis://开头")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是以下之一: {valid_levels}")
        return v.upper()
    
    @validator("text_confidence_threshold", "image_confidence_threshold", 
              "video_confidence_threshold", "audio_confidence_threshold")
    def validate_confidence_threshold(cls, v):
        """验证置信度阈值"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("置信度阈值必须在0.0到1.0之间")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ModerationRules:
    """
    审核规则配置类
    
    定义各种内容审核的规则和策略
    """
    
    # 文本审核规则
    TEXT_RULES = {
        "政治敏感": {
            "keywords": ["政治敏感词1", "政治敏感词2"],
            "severity": "critical",
            "action": "reject"
        },
        "色情暴力": {
            "keywords": ["色情词1", "暴力词1"],
            "severity": "high", 
            "action": "reject"
        },
        "垃圾广告": {
            "keywords": ["广告词1", "推广词1"],
            "severity": "medium",
            "action": "manual_review"
        },
        "虚假信息": {
            "patterns": [r"假消息模式1", r"谣言模式1"],
            "severity": "high",
            "action": "manual_review"
        }
    }
    
    # 图像审核规则
    IMAGE_RULES = {
        "色情内容": {
            "model": "nsfw_detection",
            "threshold": 0.8,
            "action": "reject"
        },
        "暴力血腥": {
            "model": "violence_detection", 
            "threshold": 0.7,
            "action": "reject"
        },
        "政治敏感": {
            "model": "politics_detection",
            "threshold": 0.6,
            "action": "manual_review"
        }
    }
    
    # 视频审核规则
    VIDEO_RULES = {
        "色情内容": {
            "frame_sampling": 5,  # 每5秒采样一帧
            "threshold": 0.8,
            "consecutive_frames": 3,  # 连续3帧违规才判定
            "action": "reject"
        },
        "暴力内容": {
            "frame_sampling": 3,
            "threshold": 0.7,
            "consecutive_frames": 2,
            "action": "reject"
        }
    }
    
    # 音频审核规则
    AUDIO_RULES = {
        "敏感言论": {
            "speech_to_text": True,
            "text_rules": TEXT_RULES,
            "confidence_threshold": 0.6
        },
        "背景音乐版权": {
            "fingerprint_matching": True,
            "database": "copyright_db",
            "action": "manual_review"
        }
    }
    
    @classmethod
    def get_rule_by_type(cls, content_type: str, rule_name: str) -> Dict[str, Any]:
        """
        根据内容类型和规则名称获取规则配置
        
        Args:
            content_type: 内容类型 (text, image, video, audio)
            rule_name: 规则名称
            
        Returns:
            Dict[str, Any]: 规则配置
        """
        rule_maps = {
            "text": cls.TEXT_RULES,
            "image": cls.IMAGE_RULES,
            "video": cls.VIDEO_RULES,
            "audio": cls.AUDIO_RULES
        }
        
        rules = rule_maps.get(content_type, {})
        return rules.get(rule_name, {})
    
    @classmethod
    def get_all_rules_by_type(cls, content_type: str) -> Dict[str, Dict[str, Any]]:
        """
        获取指定内容类型的所有规则
        
        Args:
            content_type: 内容类型
            
        Returns:
            Dict[str, Dict[str, Any]]: 所有规则配置
        """
        rule_maps = {
            "text": cls.TEXT_RULES,
            "image": cls.IMAGE_RULES,
            "video": cls.VIDEO_RULES,
            "audio": cls.AUDIO_RULES
        }
        
        return rule_maps.get(content_type, {})


class ContentLimits:
    """
    内容限制配置类
    
    定义各种内容的处理限制
    """
    
    # 文本限制
    MAX_TEXT_LENGTH = 50000  # 最大文本长度
    MAX_TEXT_LINES = 1000    # 最大文本行数
    
    # 图像限制
    MAX_IMAGE_WIDTH = 4096   # 最大图像宽度
    MAX_IMAGE_HEIGHT = 4096  # 最大图像高度
    MAX_IMAGE_PIXELS = 16777216  # 最大像素数 (4K)
    
    # 视频限制
    MAX_VIDEO_DURATION = 3600  # 最大视频长度(秒) 1小时
    MAX_VIDEO_BITRATE = 10000  # 最大比特率 (kbps)
    MAX_VIDEO_FPS = 60         # 最大帧率
    
    # 音频限制
    MAX_AUDIO_DURATION = 1800  # 最大音频长度(秒) 30分钟
    MAX_AUDIO_BITRATE = 320    # 最大比特率 (kbps)
    MAX_AUDIO_SAMPLE_RATE = 48000  # 最大采样率 (Hz)
    
    @classmethod
    def validate_content_limits(cls, content_type: str, content_info: Dict[str, Any]) -> bool:
        """
        验证内容是否超出限制
        
        Args:
            content_type: 内容类型
            content_info: 内容信息
            
        Returns:
            bool: 是否符合限制
        """
        if content_type == "text":
            text_length = content_info.get("length", 0)
            return text_length <= cls.MAX_TEXT_LENGTH
        
        elif content_type == "image":
            width = content_info.get("width", 0)
            height = content_info.get("height", 0)
            return (width <= cls.MAX_IMAGE_WIDTH and 
                   height <= cls.MAX_IMAGE_HEIGHT and
                   width * height <= cls.MAX_IMAGE_PIXELS)
        
        elif content_type == "video":
            duration = content_info.get("duration", 0)
            bitrate = content_info.get("bitrate", 0)
            fps = content_info.get("fps", 0)
            return (duration <= cls.MAX_VIDEO_DURATION and
                   bitrate <= cls.MAX_VIDEO_BITRATE and
                   fps <= cls.MAX_VIDEO_FPS)
        
        elif content_type == "audio":
            duration = content_info.get("duration", 0)
            bitrate = content_info.get("bitrate", 0)
            sample_rate = content_info.get("sample_rate", 0)
            return (duration <= cls.MAX_AUDIO_DURATION and
                   bitrate <= cls.MAX_AUDIO_BITRATE and
                   sample_rate <= cls.MAX_AUDIO_SAMPLE_RATE)
        
        return True


# 全局配置实例
settings = Settings()

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "json",
            "class": "logging.FileHandler",
            "filename": "logs/moderation_service.log",
        },
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["default", "file"],
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "sqlalchemy": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        }
    }
}