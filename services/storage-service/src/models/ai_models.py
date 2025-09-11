"""
AI模型配置数据库模型

用于存储和管理AI模型的配置信息，包括API密钥、端点URL、系统提示语等
支持多种AI提供商：OpenAI、Google Gemini、Anthropic Claude、本地模型等
"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from sqlalchemy import String, Text, Boolean, Integer, Float, JSON, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class AIProviderEnum(str, Enum):
    """
    AI提供商枚举 - AI Provider Enum
    
    定义系统支持的AI模型提供商类型
    """
    openai = "openai"           # OpenAI GPT系列模型
    gemini = "gemini"           # Google Gemini系列模型  
    claude = "claude"           # Anthropic Claude系列模型
    local = "local"             # 本地部署模型
    azure_openai = "azure_openai"  # Azure OpenAI服务
    huggingface = "huggingface"    # HuggingFace模型


class ModelStatusEnum(str, Enum):
    """
    模型状态枚举 - Model Status Enum
    
    定义AI模型的配置状态
    """
    configured = "configured"       # 已配置完成，可以使用
    needs_api_key = "needs_api_key"  # 需要配置API密钥
    disabled = "disabled"           # 已禁用
    error = "error"                # 配置错误


class AIModelConfig(BaseModel):
    """
    AI模型配置表 - AI Model Configuration Table
    
    存储AI模型的完整配置信息，包括:
    - 基本信息：别名、提供商、模型名称
    - 连接配置：API密钥、端点URL、认证参数
    - 模型参数：最大token数、默认temperature、自定义参数
    - 管理信息：状态、描述、标签等
    
    设计特点:
    - API密钥加密存储，确保安全性
    - 支持JSON格式的自定义参数扩展
    - 完整的审计追踪（创建时间、更新时间）
    - 灵活的状态管理和错误处理
    """
    
    __tablename__ = "ai_model_configs"
    __table_args__ = {'comment': 'AI模型配置表'}
    
    # === 基本信息 ===
    alias: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="模型别名，用户自定义的模型标识符"
    )
    
    provider: Mapped[AIProviderEnum] = mapped_column(
        SQLEnum(AIProviderEnum, name="ai_provider_enum"),
        nullable=False,
        index=True,
        comment="AI提供商类型"
    )
    
    model_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="实际的模型名称，如gpt-4、gemini-1.5-pro等"
    )
    
    display_name: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
        comment="模型显示名称"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="模型描述信息"
    )
    
    # === 连接配置 ===
    api_key: Mapped[Optional[str]] = mapped_column(
        Text,  # 加密后的API密钥可能很长
        nullable=True,
        comment="加密存储的API密钥"
    )
    
    api_base: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="API基础URL端点"
    )
    
    api_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="API版本号"
    )
    
    # === 模型参数配置 ===
    max_tokens: Mapped[int] = mapped_column(
        Integer,
        default=4096,
        nullable=False,
        comment="最大输出token数量限制"
    )
    
    context_window: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="上下文窗口大小"
    )
    
    default_temperature: Mapped[float] = mapped_column(
        Float,
        default=0.7,
        nullable=False,
        comment="默认温度参数，控制生成随机性"
    )
    
    default_top_p: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="默认top_p参数"
    )
    
    # === 部署配置 ===
    is_local: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否为本地部署模型"
    )
    
    is_streaming_supported: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="是否支持流式响应"
    )
    
    # === 多模态支持配置 ===
    supports_files: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否支持文件上传（如PDF、Word等文档）"
    )
    
    supports_images: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否支持图片上传（如JPG、PNG、WebP等）"
    )
    
    supports_videos: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否支持视频上传（如MP4、AVI等）"
    )
    
    supports_audio: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="是否支持音频上传（如MP3、WAV等）"
    )
    
    supported_file_types: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="支持的文件类型配置，包含具体的MIME类型和大小限制"
    )
    
    max_file_size_mb: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="支持的最大文件大小（MB）"
    )
    
    # === 扩展配置 ===
    custom_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="自定义参数配置，JSON格式存储"
    )
    
    auth_headers: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="自定义认证头信息"
    )
    
    # === 管理信息 ===
    status: Mapped[ModelStatusEnum] = mapped_column(
        SQLEnum(ModelStatusEnum, name="model_status_enum"),
        default=ModelStatusEnum.needs_api_key,
        nullable=False,
        index=True,
        comment="模型配置状态"
    )
    
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="模型优先级，用于智能路由选择"
    )
    
    tags: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="模型标签，用于分类和管理"
    )
    
    # === 监控信息 ===
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="最后使用时间"
    )
    
    usage_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="使用次数统计"
    )
    
    error_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="错误次数统计"
    )
    
    last_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="最后一次错误信息"
    )
    
    # === 性能配置 ===
    request_timeout: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="请求超时时间（秒）"
    )
    
    max_retries: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
        comment="最大重试次数"
    )
    
    rate_limit_per_minute: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="每分钟请求限制"
    )
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"<AIModelConfig(alias='{self.alias}', provider='{self.provider}', status='{self.status}')>"
    
    def to_api_dict(self) -> Dict[str, Any]:
        """
        转换为API响应格式的字典
        
        隐藏敏感信息如API密钥的具体值，仅返回是否已配置的状态
        """
        return {
            "id": str(self.id),
            "alias": self.alias,
            "provider": self.provider.value,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "description": self.description,
            "has_api_key": bool(self.api_key),
            "api_base": self.api_base,
            "api_version": self.api_version,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "default_temperature": self.default_temperature,
            "default_top_p": self.default_top_p,
            "is_local": self.is_local,
            "is_streaming_supported": self.is_streaming_supported,
            # 多模态支持信息
            "multimodal_capabilities": {
                "supports_files": self.supports_files,
                "supports_images": self.supports_images,
                "supports_videos": self.supports_videos,
                "supports_audio": self.supports_audio,
                "supported_file_types": self.supported_file_types,
                "max_file_size_mb": self.max_file_size_mb
            },
            "status": self.status.value,
            "priority": self.priority,
            "tags": self.tags,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class SystemPromptTemplate(BaseModel):
    """
    系统提示语模板表 - System Prompt Template Table
    
    存储系统级别的提示语模板，用于：
    - 预定义常用的系统提示语
    - 支持不同场景的提示语管理
    - 提供模板变量替换功能
    - 支持多语言提示语
    
    使用场景:
    - 聊天机器人系统角色设定
    - 特定任务的指令模板
    - 多语言提示语管理
    - 提示语版本控制
    """
    
    __tablename__ = "system_prompt_templates"
    __table_args__ = {'comment': '系统提示语模板表'}
    
    # === 基本信息 ===
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
        comment="提示语模板名称，唯一标识符"
    )
    
    display_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="显示名称，用户友好的名称"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="提示语模板描述"
    )
    
    # === 提示语内容 ===
    prompt_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="提示语内容，支持模板变量"
    )
    
    language: Mapped[str] = mapped_column(
        String(10),
        default="zh-CN",
        nullable=False,
        index=True,
        comment="提示语语言，如zh-CN、en-US等"
    )
    
    # === 分类和标签 ===
    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="提示语分类，如'助手'、'翻译'、'写作'等"
    )
    
    tags: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="提示语标签，用于搜索和分类"
    )
    
    # === 模板变量 ===
    variables: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="模板变量定义，包含变量名、类型、默认值等"
    )
    
    example_variables: Mapped[Optional[Dict[str, str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="示例变量值，用于演示和测试"
    )
    
    # === 使用配置 ===
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="是否激活状态"
    )
    
    is_public: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="是否公开可用"
    )
    
    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="优先级，用于排序显示"
    )
    
    # === 使用统计 ===
    usage_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="使用次数统计"
    )
    
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="最后使用时间"
    )
    
    # === 版本控制 ===
    version: Mapped[str] = mapped_column(
        String(20),
        default="1.0.0",
        nullable=False,
        comment="模板版本号"
    )
    
    created_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="创建者"
    )
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"<SystemPromptTemplate(name='{self.name}', category='{self.category}', active={self.is_active})>"
    
    def render_prompt(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        渲染提示语模板，替换变量占位符
        
        Args:
            variables: 要替换的变量值字典
            
        Returns:
            渲染后的提示语内容
        """
        content = self.prompt_content
        
        if variables and self.variables:
            # 简单的模板变量替换，格式为 {{variable_name}}
            import re
            
            def replace_var(match):
                var_name = match.group(1)
                return str(variables.get(var_name, match.group(0)))
            
            content = re.sub(r'\{\{(\w+)\}\}', replace_var, content)
        
        return content
    
    def to_api_dict(self) -> Dict[str, Any]:
        """转换为API响应格式"""
        return {
            "id": str(self.id),
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "prompt_content": self.prompt_content,
            "language": self.language,
            "category": self.category,
            "tags": self.tags,
            "variables": self.variables,
            "example_variables": self.example_variables,
            "is_active": self.is_active,
            "is_public": self.is_public,
            "priority": self.priority,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "version": self.version,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }