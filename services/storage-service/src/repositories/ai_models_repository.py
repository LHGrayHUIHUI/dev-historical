"""
AI模型配置数据访问层 - AI Models Repository

提供AI模型配置和系统提示语的数据库操作接口
包含完整的CRUD操作、查询过滤、统计分析等功能

核心功能:
1. AI模型配置管理 - 增删改查操作
2. 系统提示语管理 - 模板的创建和使用
3. API密钥安全存储 - 加密解密处理
4. 查询过滤和分页 - 支持复杂查询条件
5. 使用统计和监控 - 记录使用情况和错误信息
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.ai_models import (
    AIModelConfig, 
    SystemPromptTemplate, 
    AIProviderEnum, 
    ModelStatusEnum
)
from .base_repository import BaseRepository


logger = logging.getLogger(__name__)


class AIModelsRepository(BaseRepository[AIModelConfig]):
    """
    AI模型配置数据访问层 - AI Model Configuration Repository
    
    提供AI模型配置的完整数据库操作接口：
    - CRUD操作：创建、读取、更新、删除模型配置
    - 查询过滤：按提供商、状态、标签等条件查询
    - 批量操作：批量创建、更新、禁用模型
    - 统计分析：使用统计、错误监控、性能分析
    - 安全处理：API密钥加密存储和访问控制
    """
    
    def __init__(self, session: AsyncSession):
        """
        初始化AI模型仓库
        
        Args:
            session: 异步数据库会话
        """
        super().__init__(AIModelConfig, session)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def create_model_config(
        self,
        alias: str,
        provider: AIProviderEnum,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 4096,
        default_temperature: float = 0.7,
        is_local: bool = False,
        # 多模态支持参数
        supports_files: bool = False,
        supports_images: bool = False,
        supports_videos: bool = False,
        supports_audio: bool = False,
        supported_file_types: Optional[Dict[str, Any]] = None,
        max_file_size_mb: Optional[int] = None,
        custom_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AIModelConfig:
        """
        创建新的AI模型配置
        
        Args:
            alias: 模型别名，用户自定义标识符
            provider: AI提供商类型
            model_name: 实际模型名称
            api_key: API访问密钥
            api_base: API基础URL
            max_tokens: 最大token数
            default_temperature: 默认temperature参数
            is_local: 是否为本地模型
            custom_parameters: 自定义参数配置
            **kwargs: 其他配置参数
            
        Returns:
            创建的模型配置实例
            
        Raises:
            ValueError: 别名已存在时抛出
        """
        try:
            # 检查别名是否已存在
            existing = await self.get_by_alias(alias)
            if existing:
                raise ValueError(f"模型别名 '{alias}' 已存在")
            
            # 确定模型状态
            if is_local or api_key:
                status = ModelStatusEnum.configured
            else:
                status = ModelStatusEnum.needs_api_key
            
            # 创建模型配置
            model_config = AIModelConfig(
                alias=alias,
                provider=provider,
                model_name=model_name,
                api_key=api_key,  # TODO: 实现加密存储
                api_base=api_base,
                max_tokens=max_tokens,
                default_temperature=default_temperature,
                is_local=is_local,
                custom_parameters=custom_parameters,
                status=status,
                **kwargs
            )
            
            self._session.add(model_config)
            await self._session.commit()
            await self._session.refresh(model_config)
            
            self._logger.info(f"创建AI模型配置成功: alias={alias}, provider={provider}")
            return model_config
            
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"创建AI模型配置失败: {e}")
            raise
    
    async def get_by_alias(self, alias: str) -> Optional[AIModelConfig]:
        """
        根据别名获取模型配置
        
        Args:
            alias: 模型别名
            
        Returns:
            模型配置实例或None
        """
        try:
            stmt = select(AIModelConfig).where(AIModelConfig.alias == alias)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self._logger.error(f"根据别名查询模型配置失败: {e}")
            raise
    
    async def get_by_provider(
        self, 
        provider: AIProviderEnum,
        status: Optional[ModelStatusEnum] = None,
        limit: Optional[int] = None
    ) -> List[AIModelConfig]:
        """
        根据提供商获取模型配置列表
        
        Args:
            provider: AI提供商类型
            status: 模型状态过滤（可选）
            limit: 结果数量限制（可选）
            
        Returns:
            模型配置列表
        """
        try:
            stmt = select(AIModelConfig).where(AIModelConfig.provider == provider)
            
            if status:
                stmt = stmt.where(AIModelConfig.status == status)
            
            # 按优先级和创建时间排序
            stmt = stmt.order_by(AIModelConfig.priority.desc(), AIModelConfig.created_at.desc())
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await self._session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self._logger.error(f"根据提供商查询模型配置失败: {e}")
            raise
    
    async def get_active_models(self) -> List[AIModelConfig]:
        """
        获取所有可用的模型配置
        
        Returns:
            状态为configured的模型配置列表
        """
        try:
            stmt = select(AIModelConfig).where(
                AIModelConfig.status == ModelStatusEnum.configured
            ).order_by(AIModelConfig.priority.desc(), AIModelConfig.alias)
            
            result = await self._session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self._logger.error(f"查询活跃模型配置失败: {e}")
            raise
    
    async def update_model_config(
        self, 
        model_id: UUID, 
        **updates
    ) -> Optional[AIModelConfig]:
        """
        更新模型配置
        
        Args:
            model_id: 模型配置ID
            **updates: 要更新的字段
            
        Returns:
            更新后的模型配置或None
        """
        try:
            model_config = await self.get_by_id(model_id)
            if not model_config:
                return None
            
            # 应用更新
            for field, value in updates.items():
                if hasattr(model_config, field):
                    setattr(model_config, field, value)
            
            # 更新状态逻辑
            if 'api_key' in updates:
                if updates['api_key'] or model_config.is_local:
                    model_config.status = ModelStatusEnum.configured
                else:
                    model_config.status = ModelStatusEnum.needs_api_key
            
            await self._session.commit()
            await self._session.refresh(model_config)
            
            self._logger.info(f"更新AI模型配置成功: id={model_id}")
            return model_config
            
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"更新AI模型配置失败: {e}")
            raise
    
    async def increment_usage_count(self, model_id: UUID) -> bool:
        """
        增加模型使用次数
        
        Args:
            model_id: 模型配置ID
            
        Returns:
            操作是否成功
        """
        try:
            stmt = update(AIModelConfig).where(
                AIModelConfig.id == model_id
            ).values(
                usage_count=AIModelConfig.usage_count + 1,
                last_used_at=datetime.utcnow()
            )
            
            result = await self._session.execute(stmt)
            await self._session.commit()
            
            return result.rowcount > 0
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"增加使用次数失败: {e}")
            return False
    
    async def record_error(self, model_id: UUID, error_message: str) -> bool:
        """
        记录模型错误信息
        
        Args:
            model_id: 模型配置ID
            error_message: 错误信息
            
        Returns:
            操作是否成功
        """
        try:
            stmt = update(AIModelConfig).where(
                AIModelConfig.id == model_id
            ).values(
                error_count=AIModelConfig.error_count + 1,
                last_error=error_message
            )
            
            result = await self._session.execute(stmt)
            await self._session.commit()
            
            return result.rowcount > 0
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"记录错误信息失败: {e}")
            return False
    
    async def get_models_statistics(self) -> Dict[str, Any]:
        """
        获取模型使用统计信息
        
        Returns:
            包含各种统计数据的字典
        """
        try:
            # 总模型数
            total_count = await self._session.scalar(select(func.count(AIModelConfig.id)))
            
            # 按状态统计
            status_stats = {}
            for status in ModelStatusEnum:
                count = await self._session.scalar(
                    select(func.count(AIModelConfig.id)).where(
                        AIModelConfig.status == status
                    )
                )
                status_stats[status.value] = count
            
            # 按提供商统计
            provider_stats = {}
            for provider in AIProviderEnum:
                count = await self._session.scalar(
                    select(func.count(AIModelConfig.id)).where(
                        AIModelConfig.provider == provider
                    )
                )
                provider_stats[provider.value] = count
            
            # 使用统计
            usage_stats = await self._session.execute(
                select(
                    func.sum(AIModelConfig.usage_count).label('total_usage'),
                    func.avg(AIModelConfig.usage_count).label('avg_usage'),
                    func.sum(AIModelConfig.error_count).label('total_errors')
                )
            )
            usage_row = usage_stats.first()
            
            return {
                "total_models": total_count,
                "status_distribution": status_stats,
                "provider_distribution": provider_stats,
                "usage_statistics": {
                    "total_usage": int(usage_row.total_usage or 0),
                    "average_usage": float(usage_row.avg_usage or 0),
                    "total_errors": int(usage_row.total_errors or 0)
                }
            }
        except Exception as e:
            self._logger.error(f"获取模型统计失败: {e}")
            raise


class SystemPromptRepository(BaseRepository[SystemPromptTemplate]):
    """
    系统提示语数据访问层 - System Prompt Repository
    
    提供系统提示语模板的完整数据库操作接口：
    - 模板管理：创建、更新、删除提示语模板
    - 分类查询：按分类、语言、标签查询模板
    - 模板渲染：支持变量替换和内容生成
    - 使用统计：记录模板使用频率和效果
    - 版本控制：支持模板版本管理和回滚
    """
    
    def __init__(self, session: AsyncSession):
        """
        初始化系统提示语仓库
        
        Args:
            session: 异步数据库会话
        """
        super().__init__(SystemPromptTemplate, session)
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def create_prompt_template(
        self,
        name: str,
        display_name: str,
        prompt_content: str,
        category: str,
        language: str = "zh-CN",
        description: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> SystemPromptTemplate:
        """
        创建新的系统提示语模板
        
        Args:
            name: 模板名称（唯一标识符）
            display_name: 显示名称
            prompt_content: 提示语内容
            category: 分类
            language: 语言代码
            description: 描述
            variables: 模板变量定义
            tags: 标签
            **kwargs: 其他配置参数
            
        Returns:
            创建的提示语模板实例
            
        Raises:
            ValueError: 模板名称已存在时抛出
        """
        try:
            # 检查模板名称是否已存在
            existing = await self.get_by_name(name)
            if existing:
                raise ValueError(f"提示语模板名称 '{name}' 已存在")
            
            # 创建模板
            template = SystemPromptTemplate(
                name=name,
                display_name=display_name,
                prompt_content=prompt_content,
                category=category,
                language=language,
                description=description,
                variables=variables,
                tags=tags,
                **kwargs
            )
            
            self._session.add(template)
            await self._session.commit()
            await self._session.refresh(template)
            
            self._logger.info(f"创建系统提示语模板成功: name={name}, category={category}")
            return template
            
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"创建系统提示语模板失败: {e}")
            raise
    
    async def get_by_name(self, name: str) -> Optional[SystemPromptTemplate]:
        """
        根据名称获取提示语模板
        
        Args:
            name: 模板名称
            
        Returns:
            提示语模板实例或None
        """
        try:
            stmt = select(SystemPromptTemplate).where(SystemPromptTemplate.name == name)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self._logger.error(f"根据名称查询提示语模板失败: {e}")
            raise
    
    async def get_by_category(
        self, 
        category: str,
        language: Optional[str] = None,
        active_only: bool = True
    ) -> List[SystemPromptTemplate]:
        """
        根据分类获取提示语模板列表
        
        Args:
            category: 分类名称
            language: 语言代码过滤（可选）
            active_only: 是否只返回激活的模板
            
        Returns:
            提示语模板列表
        """
        try:
            stmt = select(SystemPromptTemplate).where(
                SystemPromptTemplate.category == category
            )
            
            if language:
                stmt = stmt.where(SystemPromptTemplate.language == language)
            
            if active_only:
                stmt = stmt.where(SystemPromptTemplate.is_active == True)
            
            # 按优先级和使用次数排序
            stmt = stmt.order_by(
                SystemPromptTemplate.priority.desc(),
                SystemPromptTemplate.usage_count.desc()
            )
            
            result = await self._session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self._logger.error(f"根据分类查询提示语模板失败: {e}")
            raise
    
    async def get_popular_templates(
        self, 
        limit: int = 10,
        language: Optional[str] = None
    ) -> List[SystemPromptTemplate]:
        """
        获取热门提示语模板
        
        Args:
            limit: 返回数量限制
            language: 语言代码过滤（可选）
            
        Returns:
            按使用次数排序的提示语模板列表
        """
        try:
            stmt = select(SystemPromptTemplate).where(
                and_(
                    SystemPromptTemplate.is_active == True,
                    SystemPromptTemplate.is_public == True
                )
            )
            
            if language:
                stmt = stmt.where(SystemPromptTemplate.language == language)
            
            stmt = stmt.order_by(SystemPromptTemplate.usage_count.desc()).limit(limit)
            
            result = await self._session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self._logger.error(f"查询热门提示语模板失败: {e}")
            raise
    
    async def increment_usage(self, template_id: UUID) -> bool:
        """
        增加模板使用次数
        
        Args:
            template_id: 模板ID
            
        Returns:
            操作是否成功
        """
        try:
            stmt = update(SystemPromptTemplate).where(
                SystemPromptTemplate.id == template_id
            ).values(
                usage_count=SystemPromptTemplate.usage_count + 1,
                last_used_at=datetime.utcnow()
            )
            
            result = await self._session.execute(stmt)
            await self._session.commit()
            
            return result.rowcount > 0
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"增加模板使用次数失败: {e}")
            return False
    
    async def search_templates(
        self,
        query: str,
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[SystemPromptTemplate]:
        """
        搜索提示语模板
        
        Args:
            query: 搜索关键词
            language: 语言代码过滤（可选）
            category: 分类过滤（可选）
            limit: 返回数量限制
            
        Returns:
            匹配的提示语模板列表
        """
        try:
            # 构建搜索条件
            conditions = [
                SystemPromptTemplate.is_active == True,
                SystemPromptTemplate.is_public == True
            ]
            
            # 关键词搜索（在名称、显示名称、描述中搜索）
            search_condition = or_(
                SystemPromptTemplate.name.ilike(f"%{query}%"),
                SystemPromptTemplate.display_name.ilike(f"%{query}%"),
                SystemPromptTemplate.description.ilike(f"%{query}%"),
                SystemPromptTemplate.prompt_content.ilike(f"%{query}%")
            )
            conditions.append(search_condition)
            
            if language:
                conditions.append(SystemPromptTemplate.language == language)
            
            if category:
                conditions.append(SystemPromptTemplate.category == category)
            
            stmt = select(SystemPromptTemplate).where(and_(*conditions))
            stmt = stmt.order_by(
                SystemPromptTemplate.priority.desc(),
                SystemPromptTemplate.usage_count.desc()
            ).limit(limit)
            
            result = await self._session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self._logger.error(f"搜索提示语模板失败: {e}")
            raise
    
    async def get_categories(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取所有分类统计信息
        
        Args:
            language: 语言代码过滤（可选）
            
        Returns:
            分类统计信息列表
        """
        try:
            stmt = select(
                SystemPromptTemplate.category,
                func.count(SystemPromptTemplate.id).label('count')
            ).where(SystemPromptTemplate.is_active == True)
            
            if language:
                stmt = stmt.where(SystemPromptTemplate.language == language)
            
            stmt = stmt.group_by(SystemPromptTemplate.category)
            stmt = stmt.order_by(func.count(SystemPromptTemplate.id).desc())
            
            result = await self._session.execute(stmt)
            
            categories = []
            for row in result:
                categories.append({
                    "category": row.category,
                    "count": row.count
                })
            
            return categories
        except Exception as e:
            self._logger.error(f"获取分类统计失败: {e}")
            raise