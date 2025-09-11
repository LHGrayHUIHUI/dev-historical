"""
基础数据访问层 - Base Repository

提供通用的数据库操作接口和基础CRUD功能
所有具体的仓库类都应该继承此基类
"""

import logging
from typing import Generic, TypeVar, Type, List, Optional, Any, Dict
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from ..models.base import BaseModel


# 泛型类型变量
ModelType = TypeVar("ModelType", bound=BaseModel)

logger = logging.getLogger(__name__)


class BaseRepository(Generic[ModelType]):
    """
    基础数据访问层 - Base Repository Class
    
    提供所有数据库模型的通用CRUD操作：
    - Create: 创建新记录
    - Read: 查询和读取记录
    - Update: 更新现有记录 
    - Delete: 删除记录
    - 分页查询和批量操作
    - 事务管理和错误处理
    """
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        """
        初始化基础仓库
        
        Args:
            model: 数据库模型类
            session: 异步数据库会话
        """
        self._model = model
        self._session = session
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def create(self, obj_data: Dict[str, Any]) -> ModelType:
        """
        创建新记录
        
        Args:
            obj_data: 对象数据字典
            
        Returns:
            创建的模型实例
        """
        try:
            obj = self._model(**obj_data)
            self._session.add(obj)
            await self._session.commit()
            await self._session.refresh(obj)
            return obj
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"创建记录失败: {e}")
            raise
    
    async def get_by_id(self, obj_id: UUID) -> Optional[ModelType]:
        """
        根据ID获取记录
        
        Args:
            obj_id: 记录ID
            
        Returns:
            模型实例或None
        """
        try:
            stmt = select(self._model).where(self._model.id == obj_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self._logger.error(f"根据ID查询记录失败: {e}")
            raise
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """
        获取所有记录
        
        Args:
            skip: 跳过的记录数
            limit: 返回的记录数限制
            order_by: 排序字段
            
        Returns:
            模型实例列表
        """
        try:
            stmt = select(self._model)
            
            if order_by:
                if hasattr(self._model, order_by):
                    stmt = stmt.order_by(getattr(self._model, order_by))
            else:
                stmt = stmt.order_by(self._model.created_at.desc())
            
            stmt = stmt.offset(skip).limit(limit)
            result = await self._session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self._logger.error(f"查询所有记录失败: {e}")
            raise
    
    async def update(self, obj_id: UUID, obj_data: Dict[str, Any]) -> Optional[ModelType]:
        """
        更新记录
        
        Args:
            obj_id: 记录ID
            obj_data: 更新数据字典
            
        Returns:
            更新后的模型实例或None
        """
        try:
            obj = await self.get_by_id(obj_id)
            if not obj:
                return None
            
            for field, value in obj_data.items():
                if hasattr(obj, field):
                    setattr(obj, field, value)
            
            await self._session.commit()
            await self._session.refresh(obj)
            return obj
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"更新记录失败: {e}")
            raise
    
    async def delete(self, obj_id: UUID) -> bool:
        """
        删除记录
        
        Args:
            obj_id: 记录ID
            
        Returns:
            删除是否成功
        """
        try:
            stmt = delete(self._model).where(self._model.id == obj_id)
            result = await self._session.execute(stmt)
            await self._session.commit()
            return result.rowcount > 0
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"删除记录失败: {e}")
            raise
    
    async def count(self, **filters) -> int:
        """
        统计记录数量
        
        Args:
            **filters: 过滤条件
            
        Returns:
            记录总数
        """
        try:
            stmt = select(func.count(self._model.id))
            
            # 应用过滤条件
            for field, value in filters.items():
                if hasattr(self._model, field):
                    stmt = stmt.where(getattr(self._model, field) == value)
            
            result = await self._session.execute(stmt)
            return result.scalar()
        except Exception as e:
            self._logger.error(f"统计记录数量失败: {e}")
            raise
    
    async def exists(self, obj_id: UUID) -> bool:
        """
        检查记录是否存在
        
        Args:
            obj_id: 记录ID
            
        Returns:
            记录是否存在
        """
        try:
            stmt = select(func.count(self._model.id)).where(self._model.id == obj_id)
            result = await self._session.execute(stmt)
            count = result.scalar()
            return count > 0
        except Exception as e:
            self._logger.error(f"检查记录存在性失败: {e}")
            raise
    
    async def batch_create(self, obj_data_list: List[Dict[str, Any]]) -> List[ModelType]:
        """
        批量创建记录
        
        Args:
            obj_data_list: 对象数据字典列表
            
        Returns:
            创建的模型实例列表
        """
        try:
            objects = [self._model(**obj_data) for obj_data in obj_data_list]
            self._session.add_all(objects)
            await self._session.commit()
            
            # 刷新所有对象以获取生成的ID
            for obj in objects:
                await self._session.refresh(obj)
            
            return objects
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"批量创建记录失败: {e}")
            raise
    
    async def batch_update(
        self, 
        updates: List[Dict[str, Any]], 
        id_field: str = "id"
    ) -> int:
        """
        批量更新记录
        
        Args:
            updates: 更新数据列表，每个字典必须包含ID字段
            id_field: ID字段名称
            
        Returns:
            更新的记录数量
        """
        try:
            updated_count = 0
            
            for update_data in updates:
                if id_field not in update_data:
                    continue
                
                obj_id = update_data.pop(id_field)
                stmt = update(self._model).where(
                    self._model.id == obj_id
                ).values(**update_data)
                
                result = await self._session.execute(stmt)
                updated_count += result.rowcount
            
            await self._session.commit()
            return updated_count
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"批量更新记录失败: {e}")
            raise
    
    async def batch_delete(self, obj_ids: List[UUID]) -> int:
        """
        批量删除记录
        
        Args:
            obj_ids: 记录ID列表
            
        Returns:
            删除的记录数量
        """
        try:
            stmt = delete(self._model).where(self._model.id.in_(obj_ids))
            result = await self._session.execute(stmt)
            await self._session.commit()
            return result.rowcount
        except Exception as e:
            await self._session.rollback()
            self._logger.error(f"批量删除记录失败: {e}")
            raise