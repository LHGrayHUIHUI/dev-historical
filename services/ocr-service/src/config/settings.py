"""
OCR服务配置设置

精简的无状态OCR服务配置，专注于文本识别功能。
数据存储通过storage-service完成，本服务不直接连接数据库。

主要功能：
- OCR引擎配置
- 文件处理配置  
- 服务通信配置
- 基础安全配置

Author: OCR开发团队
Created: 2025-01-15
Version: 2.0.0 (无状态架构)
"""

from typing import List
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class OCRSettings(BaseSettings):
    """OCR引擎配置"""
    
    # 默认引擎设置
    DEFAULT_ENGINE: str = Field("paddleocr", description="默认OCR引擎")
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(0.8, ge=0.0, le=1.0, description="默认置信度阈值")
    DEFAULT_LANGUAGE_CODES: str = Field("zh,en", description="默认语言代码")
    
    # 文件处理配置
    MAX_FILE_SIZE: int = Field(50 * 1024 * 1024, description="最大文件大小(50MB)")
    MAX_BATCH_SIZE: int = Field(20, description="最大批量处理数量")
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        description="支持的图像格式"
    )
    
    # 处理配置
    MAX_CONCURRENT_TASKS: int = Field(4, description="最大并发任务数")
    TASK_TIMEOUT: int = Field(300, description="任务超时时间(秒)")
    ENABLE_PREPROCESSING: bool = Field(True, description="默认启用预处理")
    ENABLE_POSTPROCESSING: bool = Field(True, description="默认启用后处理")
    
    # 临时文件配置
    TEMP_DIR: str = Field("/tmp/ocr-service", description="临时文件目录")
    TEMP_FILE_CLEANUP_INTERVAL: int = Field(3600, description="临时文件清理间隔(秒)")
    TEMP_FILE_MAX_AGE: int = Field(7200, description="临时文件最大存在时间(秒)")
    
    @validator('SUPPORTED_IMAGE_FORMATS')
    def validate_image_formats(cls, v):
        """验证支持的图像格式"""
        if not v or not isinstance(v, list):
            raise ValueError("支持的图像格式不能为空")
        return [fmt.lower() for fmt in v]
    
    class Config:
        env_prefix = "OCR_"
        case_sensitive = True


class ServiceSettings(BaseSettings):
    """服务间通信配置"""
    
    # Storage Service配置 - 唯一的数据存储入口
    STORAGE_SERVICE_URL: str = Field(
        "http://localhost:8002",
        description="Storage服务URL"
    )
    STORAGE_SERVICE_TIMEOUT: int = Field(30, description="Storage服务超时时间")
    STORAGE_SERVICE_RETRIES: int = Field(3, description="Storage服务重试次数")
    
    # File Processor配置（如果需要调用其他文件处理功能）
    FILE_PROCESSOR_URL: str = Field(
        "http://localhost:8001", 
        description="File Processor服务URL"
    )
    
    class Config:
        env_prefix = "OCR_SERVICE_"
        case_sensitive = True


class Settings(BaseSettings):
    """主配置类"""
    
    # 基础配置
    SERVICE_NAME: str = Field("ocr-service", description="服务名称")
    SERVICE_VERSION: str = Field("2.0.0", description="服务版本")
    ENVIRONMENT: str = Field("development", description="运行环境")
    DEBUG: bool = Field(False, description="调试模式")
    
    # API配置
    API_HOST: str = Field("0.0.0.0", description="API主机地址")
    API_PORT: int = Field(8003, description="API端口")  # 修改为8003，避免与storage冲突
    API_PREFIX: str = Field("/api/v1", description="API路径前缀")
    
    # 工作配置
    WORKERS: int = Field(1, description="工作进程数量")
    
    # 子配置模块
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    services: ServiceSettings = Field(default_factory=ServiceSettings)
    
    # 日志配置（简化）
    LOG_LEVEL: str = Field("INFO", description="日志级别")
    LOG_FORMAT: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )
    ENABLE_JSON_LOGS: bool = Field(False, description="启用JSON格式日志")
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """验证运行环境"""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"无效的环境配置: {v}")
        return v.lower()
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"无效的日志级别: {v}")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """判断是否为开发环境"""
        return self.ENVIRONMENT == 'development'
    
    @property
    def is_production(self) -> bool:
        """判断是否为生产环境"""
        return self.ENVIRONMENT == 'production'
    
    def get_engine_config(self, engine_name: str) -> dict:
        """
        获取特定引擎的配置
        
        Args:
            engine_name: 引擎名称
            
        Returns:
            引擎配置字典
        """
        base_config = {
            'confidence_threshold': self.ocr.DEFAULT_CONFIDENCE_THRESHOLD,
            'language_codes': self.ocr.DEFAULT_LANGUAGE_CODES,
            'enable_preprocessing': self.ocr.ENABLE_PREPROCESSING,
            'enable_postprocessing': self.ocr.ENABLE_POSTPROCESSING,
        }
        
        # 引擎特定配置
        engine_configs = {
            'paddleocr': {
                'use_gpu': True,
                'use_angle_cls': True,
                'lang': 'ch',
            },
            'tesseract': {
                'oem': 3,
                'psm': 6,
                'lang': 'chi_sim+eng',
            },
            'easyocr': {
                'gpu': True,
                'lang_list': ['ch_sim', 'en'],
            }
        }
        
        if engine_name in engine_configs:
            base_config.update(engine_configs[engine_name])
        
        return base_config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例（单例模式）
    
    Returns:
        配置实例
    """
    return Settings()


# 导出主要配置实例
settings = get_settings()