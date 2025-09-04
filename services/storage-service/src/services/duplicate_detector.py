"""
重复文件检测服务

基于文件哈希值检测重复文件
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.dataset import Dataset
from ..utils.database import get_database_session

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """重复文件检测服务
    
    通过文件SHA256哈希值检测重复文件
    """
    
    def __init__(self):
        """初始化重复检测器"""
        logger.info("重复文件检测器初始化完成")
    
    async def check_duplicate(self, file_hash: str, user_id: str = None) -> Optional[Dataset]:
        """检查文件是否已存在
        
        Args:
            file_hash: 文件SHA256哈希值
            user_id: 用户ID，可选限制范围
            
        Returns:
            如果文件已存在，返回现有数据集；否则返回None
        """
        try:
            async with get_database_session() as session:
                # 构建查询
                query = select(Dataset).where(Dataset.file_hash == file_hash)
                
                # 可选择只在当前用户范围内查找
                if user_id:
                    query = query.where(Dataset.created_by == user_id)
                
                # 只查找已成功处理的文件
                query = query.where(Dataset.processing_status.in_(['completed', 'processing']))
                
                # 执行查询
                result = await session.execute(query)
                existing_dataset = result.scalar_one_or_none()
                
                if existing_dataset:
                    logger.info(
                        f"发现重复文件",
                        extra={
                            "file_hash": file_hash,
                            "existing_dataset_id": str(existing_dataset.id),
                            "existing_dataset_name": existing_dataset.name,
                            "user_id": user_id
                        }
                    )
                else:
                    logger.debug(f"未发现重复文件: {file_hash}")
                
                return existing_dataset
                
        except Exception as e:
            logger.error(f"检查重复文件失败: {str(e)}")
            return None
    
    async def check_duplicate_by_name(
        self, 
        file_name: str, 
        user_id: str, 
        source_id: str = None
    ) -> Optional[Dataset]:
        """通过文件名检查重复文件
        
        Args:
            file_name: 文件名
            user_id: 用户ID
            source_id: 数据源ID，可选
            
        Returns:
            如果文件已存在，返回现有数据集；否则返回None
        """
        try:
            async with get_database_session() as session:
                # 构建查询
                query = select(Dataset).where(
                    Dataset.name == file_name,
                    Dataset.created_by == user_id
                )
                
                # 可选择限制在特定数据源
                if source_id:
                    query = query.where(Dataset.source_id == source_id)
                
                # 执行查询
                result = await session.execute(query)
                existing_dataset = result.scalar_one_or_none()
                
                if existing_dataset:
                    logger.info(
                        f"发现同名文件",
                        extra={
                            "file_name": file_name,
                            "existing_dataset_id": str(existing_dataset.id),
                            "user_id": user_id,
                            "source_id": source_id
                        }
                    )
                
                return existing_dataset
                
        except Exception as e:
            logger.error(f"检查同名文件失败: {str(e)}")
            return None
    
    async def get_duplicate_files(self, user_id: str = None) -> list[Dataset]:
        """获取重复文件列表
        
        Args:
            user_id: 用户ID，可选限制范围
            
        Returns:
            重复文件的数据集列表
        """
        try:
            async with get_database_session() as session:
                # 查找有相同哈希值的文件
                subquery = select(Dataset.file_hash).group_by(Dataset.file_hash).having(
                    select.func.count(Dataset.file_hash) > 1
                )
                
                if user_id:
                    subquery = subquery.where(Dataset.created_by == user_id)
                
                # 获取所有重复文件
                query = select(Dataset).where(
                    Dataset.file_hash.in_(subquery)
                ).order_by(Dataset.file_hash, Dataset.created_at)
                
                if user_id:
                    query = query.where(Dataset.created_by == user_id)
                
                result = await session.execute(query)
                duplicate_files = result.scalars().all()
                
                logger.info(f"找到 {len(duplicate_files)} 个重复文件")
                return list(duplicate_files)
                
        except Exception as e:
            logger.error(f"获取重复文件列表失败: {str(e)}")
            return []
    
    async def remove_duplicate_by_id(self, dataset_id: str, user_id: str) -> bool:
        """删除指定的重复文件
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        try:
            async with get_database_session() as session:
                # 查找数据集
                query = select(Dataset).where(
                    Dataset.id == dataset_id,
                    Dataset.created_by == user_id
                )
                
                result = await session.execute(query)
                dataset = result.scalar_one_or_none()
                
                if not dataset:
                    logger.warning(f"未找到数据集: {dataset_id}")
                    return False
                
                # 删除数据集（级联删除相关内容）
                await session.delete(dataset)
                await session.commit()
                
                logger.info(f"删除重复文件成功: {dataset_id}")
                return True
                
        except Exception as e:
            logger.error(f"删除重复文件失败: {str(e)}")
            return False
    
    async def get_file_statistics(self, user_id: str = None) -> dict:
        """获取文件统计信息
        
        Args:
            user_id: 用户ID，可选限制范围
            
        Returns:
            统计信息字典
        """
        try:
            async with get_database_session() as session:
                # 总文件数
                total_query = select(Dataset.id)
                if user_id:
                    total_query = total_query.where(Dataset.created_by == user_id)
                
                total_result = await session.execute(total_query)
                total_files = len(total_result.scalars().all())
                
                # 唯一文件数（基于哈希值）
                unique_query = select(Dataset.file_hash).distinct()
                if user_id:
                    unique_query = unique_query.where(Dataset.created_by == user_id)
                
                unique_result = await session.execute(unique_query)
                unique_files = len([h for h in unique_result.scalars().all() if h is not None])
                
                # 重复文件数
                duplicate_files = total_files - unique_files
                
                # 总文件大小
                size_query = select(Dataset.file_size)
                if user_id:
                    size_query = size_query.where(Dataset.created_by == user_id)
                
                size_result = await session.execute(size_query)
                total_size = sum(size or 0 for size in size_result.scalars().all())
                
                stats = {
                    "total_files": total_files,
                    "unique_files": unique_files,
                    "duplicate_files": duplicate_files,
                    "duplicate_ratio": round(duplicate_files / total_files * 100, 2) if total_files > 0 else 0.0,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0.0
                }
                
                logger.info(f"文件统计信息: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"获取文件统计信息失败: {str(e)}")
            return {
                "total_files": 0,
                "unique_files": 0,
                "duplicate_files": 0,
                "duplicate_ratio": 0.0,
                "total_size_bytes": 0,
                "total_size_mb": 0.0
            }