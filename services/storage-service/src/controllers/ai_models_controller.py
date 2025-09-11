"""
AI模型配置控制器 - AI Models Configuration Controller

提供AI模型配置和系统提示语的REST API接口
包含完整的CRUD操作和管理功能

核心功能:
1. AI模型配置管理 - 增删改查API
2. 系统提示语管理 - 模板CRUD操作
3. 模型测试和验证 - 连接测试接口
4. 统计和监控 - 使用数据和分析
5. 批量操作 - 批量导入导出配置
"""

import logging
from typing import Dict, List, Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field

from ..database import get_async_session
from ..repositories.ai_models_repository import AIModelsRepository, SystemPromptRepository
from ..models.ai_models import AIProviderEnum, ModelStatusEnum


logger = logging.getLogger(__name__)


# === Pydantic模型定义 ===

class AIModelConfigRequest(BaseModel):
    """AI模型配置请求模型"""
    alias: str = Field(..., description="模型别名，系统内唯一标识", min_length=1, max_length=100)
    provider: AIProviderEnum = Field(..., description="AI提供商类型")
    model_name: str = Field(..., description="实际模型名称", min_length=1, max_length=200)
    display_name: Optional[str] = Field(None, description="显示名称", max_length=200)
    description: Optional[str] = Field(None, description="模型描述")
    api_key: Optional[str] = Field(None, description="API访问密钥")
    api_base: Optional[str] = Field(None, description="API基础URL", max_length=500)
    api_version: Optional[str] = Field(None, description="API版本", max_length=50)
    max_tokens: int = Field(4096, description="最大token数", ge=1, le=128000)
    context_window: Optional[int] = Field(None, description="上下文窗口大小", ge=1)
    default_temperature: float = Field(0.7, description="默认温度参数", ge=0.0, le=2.0)
    default_top_p: Optional[float] = Field(None, description="默认top_p参数", ge=0.0, le=1.0)
    is_local: bool = Field(False, description="是否为本地模型")
    is_streaming_supported: bool = Field(True, description="是否支持流式响应")
    # 多模态支持配置
    supports_files: bool = Field(False, description="是否支持文件上传")
    supports_images: bool = Field(False, description="是否支持图片上传")
    supports_videos: bool = Field(False, description="是否支持视频上传")
    supports_audio: bool = Field(False, description="是否支持音频上传")
    supported_file_types: Optional[Dict[str, Any]] = Field(None, description="支持的文件类型配置")
    max_file_size_mb: Optional[int] = Field(None, description="最大文件大小MB", ge=1)
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="自定义参数")
    auth_headers: Optional[Dict[str, str]] = Field(None, description="认证头信息")
    priority: int = Field(0, description="优先级", ge=0)
    tags: Optional[Dict[str, str]] = Field(None, description="标签")
    request_timeout: Optional[int] = Field(None, description="请求超时时间", ge=1)
    max_retries: int = Field(3, description="最大重试次数", ge=0)
    rate_limit_per_minute: Optional[int] = Field(None, description="每分钟请求限制", ge=1)


class AIModelConfigResponse(BaseModel):
    """AI模型配置响应模型"""
    id: str
    alias: str
    provider: str
    model_name: str
    display_name: Optional[str]
    description: Optional[str]
    has_api_key: bool
    api_base: Optional[str]
    api_version: Optional[str]
    max_tokens: int
    context_window: Optional[int]
    default_temperature: float
    default_top_p: Optional[float]
    is_local: bool
    is_streaming_supported: bool
    # 多模态支持信息
    multimodal_capabilities: Dict[str, Any]
    status: str
    priority: int
    tags: Optional[Dict[str, str]]
    last_used_at: Optional[str]
    usage_count: int
    error_count: int
    request_timeout: Optional[int]
    max_retries: int
    rate_limit_per_minute: Optional[int]
    created_at: str
    updated_at: str


class SystemPromptRequest(BaseModel):
    """系统提示语请求模型"""
    name: str = Field(..., description="模板名称", min_length=1, max_length=100)
    display_name: str = Field(..., description="显示名称", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="描述")
    prompt_content: str = Field(..., description="提示语内容", min_length=1)
    language: str = Field("zh-CN", description="语言代码", max_length=10)
    category: str = Field(..., description="分类", min_length=1, max_length=50)
    tags: Optional[Dict[str, str]] = Field(None, description="标签")
    variables: Optional[Dict[str, Any]] = Field(None, description="模板变量")
    example_variables: Optional[Dict[str, str]] = Field(None, description="示例变量")
    is_active: bool = Field(True, description="是否激活")
    is_public: bool = Field(True, description="是否公开")
    priority: int = Field(0, description="优先级", ge=0)
    version: str = Field("1.0.0", description="版本号", max_length=20)
    created_by: Optional[str] = Field(None, description="创建者", max_length=100)


class SystemPromptResponse(BaseModel):
    """系统提示语响应模型"""
    id: str
    name: str
    display_name: str
    description: Optional[str]
    prompt_content: str
    language: str
    category: str
    tags: Optional[Dict[str, str]]
    variables: Optional[Dict[str, Any]]
    example_variables: Optional[Dict[str, str]]
    is_active: bool
    is_public: bool
    priority: int
    usage_count: int
    last_used_at: Optional[str]
    version: str
    created_by: Optional[str]
    created_at: str
    updated_at: str


class AIModelsController:
    """
    AI模型配置控制器 - AI Models Configuration Controller
    
    提供完整的AI模型配置和系统提示语管理API：
    - 模型配置的CRUD操作
    - 系统提示语的管理功能
    - 批量操作和数据导入导出
    - 使用统计和监控信息
    - 模型测试和验证功能
    """
    
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/ai-models", tags=["ai-models"])
        self._logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        # === AI模型配置路由 ===
        
        @self.router.get("/configs",
                        summary="获取AI模型配置列表",
                        description="查询所有AI模型配置信息",
                        response_model=Dict[str, Any])
        async def get_model_configs(
            provider: Optional[AIProviderEnum] = Query(None, description="提供商过滤"),
            status: Optional[ModelStatusEnum] = Query(None, description="状态过滤"),
            skip: int = Query(0, description="跳过记录数", ge=0),
            limit: int = Query(100, description="返回记录数", ge=1, le=1000),
            session = Depends(get_async_session)
        ):
            """
            获取AI模型配置列表 - Get AI Model Configurations
            
            支持按提供商、状态等条件过滤，支持分页查询
            """
            try:
                repo = AIModelsRepository(session)
                
                if provider:
                    models = await repo.get_by_provider(provider, status)
                elif status:
                    # 按状态查询所有提供商的模型
                    all_models = await repo.get_all(skip, limit)
                    models = [m for m in all_models if m.status == status]
                else:
                    models = await repo.get_all(skip, limit)
                
                # 转换为响应格式
                model_list = []
                for model in models:
                    model_dict = model.to_api_dict()
                    model_list.append(AIModelConfigResponse(**model_dict))
                
                total = await repo.count()
                
                return {
                    "models": model_list,
                    "total": total,
                    "skip": skip,
                    "limit": limit
                }
            except Exception as e:
                self._logger.error(f"获取模型配置列表失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取模型配置列表失败: {str(e)}"
                )
        
        @self.router.post("/configs",
                         summary="创建AI模型配置",
                         description="添加新的AI模型配置",
                         response_model=AIModelConfigResponse,
                         status_code=status.HTTP_201_CREATED)
        async def create_model_config(
            request: AIModelConfigRequest,
            session = Depends(get_async_session)
        ):
            """
            创建新的AI模型配置 - Create AI Model Configuration
            """
            try:
                repo = AIModelsRepository(session)
                
                model_config = await repo.create_model_config(
                    alias=request.alias,
                    provider=request.provider,
                    model_name=request.model_name,
                    display_name=request.display_name,
                    description=request.description,
                    api_key=request.api_key,
                    api_base=request.api_base,
                    api_version=request.api_version,
                    max_tokens=request.max_tokens,
                    context_window=request.context_window,
                    default_temperature=request.default_temperature,
                    default_top_p=request.default_top_p,
                    is_local=request.is_local,
                    is_streaming_supported=request.is_streaming_supported,
                    # 多模态支持配置
                    supports_files=request.supports_files,
                    supports_images=request.supports_images,
                    supports_videos=request.supports_videos,
                    supports_audio=request.supports_audio,
                    supported_file_types=request.supported_file_types,
                    max_file_size_mb=request.max_file_size_mb,
                    custom_parameters=request.custom_parameters,
                    auth_headers=request.auth_headers,
                    priority=request.priority,
                    tags=request.tags,
                    request_timeout=request.request_timeout,
                    max_retries=request.max_retries,
                    rate_limit_per_minute=request.rate_limit_per_minute
                )
                
                model_dict = model_config.to_api_dict()
                return AIModelConfigResponse(**model_dict)
                
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            except Exception as e:
                self._logger.error(f"创建模型配置失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"创建模型配置失败: {str(e)}"
                )
        
        @self.router.get("/configs/{model_id}",
                        summary="获取模型配置详情",
                        description="根据ID获取特定模型配置",
                        response_model=AIModelConfigResponse)
        async def get_model_config(
            model_id: UUID,
            session = Depends(get_async_session)
        ):
            """
            获取模型配置详情 - Get Model Configuration Details
            """
            try:
                repo = AIModelsRepository(session)
                model_config = await repo.get_by_id(model_id)
                
                if not model_config:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"模型配置 {model_id} 不存在"
                    )
                
                model_dict = model_config.to_api_dict()
                return AIModelConfigResponse(**model_dict)
                
            except HTTPException:
                raise
            except Exception as e:
                self._logger.error(f"获取模型配置详情失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取模型配置详情失败: {str(e)}"
                )
        
        @self.router.put("/configs/{model_id}",
                        summary="更新模型配置",
                        description="更新指定的模型配置",
                        response_model=AIModelConfigResponse)
        async def update_model_config(
            model_id: UUID,
            request: AIModelConfigRequest,
            session = Depends(get_async_session)
        ):
            """
            更新模型配置 - Update Model Configuration
            """
            try:
                repo = AIModelsRepository(session)
                
                # 转换请求数据为字典
                update_data = request.dict(exclude_unset=True)
                
                model_config = await repo.update_model_config(model_id, **update_data)
                
                if not model_config:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"模型配置 {model_id} 不存在"
                    )
                
                model_dict = model_config.to_api_dict()
                return AIModelConfigResponse(**model_dict)
                
            except HTTPException:
                raise
            except Exception as e:
                self._logger.error(f"更新模型配置失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"更新模型配置失败: {str(e)}"
                )
        
        @self.router.delete("/configs/{model_id}",
                           summary="删除模型配置",
                           description="删除指定的模型配置",
                           status_code=status.HTTP_204_NO_CONTENT)
        async def delete_model_config(
            model_id: UUID,
            session = Depends(get_async_session)
        ):
            """
            删除模型配置 - Delete Model Configuration
            """
            try:
                repo = AIModelsRepository(session)
                success = await repo.delete(model_id)
                
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"模型配置 {model_id} 不存在"
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                self._logger.error(f"删除模型配置失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"删除模型配置失败: {str(e)}"
                )
        
        @self.router.get("/configs/by-alias/{alias}",
                        summary="根据别名获取模型配置",
                        description="使用模型别名查询配置",
                        response_model=AIModelConfigResponse)
        async def get_model_config_by_alias(
            alias: str,
            session = Depends(get_async_session)
        ):
            """
            根据别名获取模型配置 - Get Model Configuration by Alias
            """
            try:
                repo = AIModelsRepository(session)
                model_config = await repo.get_by_alias(alias)
                
                if not model_config:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"别名为 '{alias}' 的模型配置不存在"
                    )
                
                model_dict = model_config.to_api_dict()
                return AIModelConfigResponse(**model_dict)
                
            except HTTPException:
                raise
            except Exception as e:
                self._logger.error(f"根据别名获取模型配置失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"根据别名获取模型配置失败: {str(e)}"
                )
        
        @self.router.get("/active",
                        summary="获取可用模型列表",
                        description="获取所有已配置且可用的模型",
                        response_model=List[AIModelConfigResponse])
        async def get_active_models(
            session = Depends(get_async_session)
        ):
            """
            获取可用模型列表 - Get Active Models
            """
            try:
                repo = AIModelsRepository(session)
                models = await repo.get_active_models()
                
                model_list = []
                for model in models:
                    model_dict = model.to_api_dict()
                    model_list.append(AIModelConfigResponse(**model_dict))
                
                return model_list
                
            except Exception as e:
                self._logger.error(f"获取可用模型列表失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取可用模型列表失败: {str(e)}"
                )
        
        @self.router.get("/statistics",
                        summary="获取模型使用统计",
                        description="查询模型使用情况和统计数据",
                        response_model=Dict[str, Any])
        async def get_models_statistics(
            session = Depends(get_async_session)
        ):
            """
            获取模型使用统计 - Get Models Statistics
            """
            try:
                repo = AIModelsRepository(session)
                stats = await repo.get_models_statistics()
                return stats
                
            except Exception as e:
                self._logger.error(f"获取模型统计失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取模型统计失败: {str(e)}"
                )
        
        # === 系统提示语路由 ===
        
        @self.router.get("/prompts",
                        summary="获取系统提示语列表",
                        description="查询系统提示语模板",
                        response_model=Dict[str, Any])
        async def get_prompt_templates(
            category: Optional[str] = Query(None, description="分类过滤"),
            language: Optional[str] = Query(None, description="语言过滤"),
            active_only: bool = Query(True, description="仅返回激活的模板"),
            skip: int = Query(0, description="跳过记录数", ge=0),
            limit: int = Query(100, description="返回记录数", ge=1, le=1000),
            session = Depends(get_async_session)
        ):
            """
            获取系统提示语列表 - Get System Prompt Templates
            """
            try:
                repo = SystemPromptRepository(session)
                
                if category:
                    templates = await repo.get_by_category(category, language, active_only)
                else:
                    all_templates = await repo.get_all(skip, limit)
                    if active_only:
                        templates = [t for t in all_templates if t.is_active]
                    else:
                        templates = all_templates
                    
                    if language:
                        templates = [t for t in templates if t.language == language]
                
                # 转换为响应格式
                template_list = []
                for template in templates:
                    template_dict = template.to_api_dict()
                    template_list.append(SystemPromptResponse(**template_dict))
                
                total = await repo.count()
                
                return {
                    "templates": template_list,
                    "total": total,
                    "skip": skip,
                    "limit": limit
                }
            except Exception as e:
                self._logger.error(f"获取提示语模板列表失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取提示语模板列表失败: {str(e)}"
                )
        
        @self.router.post("/prompts",
                         summary="创建系统提示语",
                         description="添加新的系统提示语模板",
                         response_model=SystemPromptResponse,
                         status_code=status.HTTP_201_CREATED)
        async def create_prompt_template(
            request: SystemPromptRequest,
            session = Depends(get_async_session)
        ):
            """
            创建系统提示语模板 - Create System Prompt Template
            """
            try:
                repo = SystemPromptRepository(session)
                
                template = await repo.create_prompt_template(
                    name=request.name,
                    display_name=request.display_name,
                    prompt_content=request.prompt_content,
                    category=request.category,
                    language=request.language,
                    description=request.description,
                    variables=request.variables,
                    tags=request.tags,
                    example_variables=request.example_variables,
                    is_active=request.is_active,
                    is_public=request.is_public,
                    priority=request.priority,
                    version=request.version,
                    created_by=request.created_by
                )
                
                template_dict = template.to_api_dict()
                return SystemPromptResponse(**template_dict)
                
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            except Exception as e:
                self._logger.error(f"创建提示语模板失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"创建提示语模板失败: {str(e)}"
                )


def create_ai_models_controller() -> AIModelsController:
    """
    创建AI模型控制器实例 - Create AI Models Controller Instance
    
    工厂函数，用于创建和初始化AI模型配置控制器
    """
    return AIModelsController()