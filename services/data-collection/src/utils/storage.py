"""
对象存储工具模块

提供MinIO对象存储的连接和操作功能
"""

import logging
from io import BytesIO
from typing import Dict, Any, Optional

from fastapi import UploadFile
from minio import Minio
from minio.error import S3Error

from ..config import get_settings

logger = logging.getLogger(__name__)

# 全局变量存储MinIO客户端
_storage_client: Minio = None


def init_storage_client() -> None:
    """初始化MinIO存储客户端"""
    global _storage_client
    
    settings = get_settings()
    
    _storage_client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    
    # 确保存储桶存在
    try:
        if not _storage_client.bucket_exists(settings.minio_bucket_name):
            _storage_client.make_bucket(settings.minio_bucket_name)
            logger.info(f"创建存储桶: {settings.minio_bucket_name}")
        else:
            logger.info(f"存储桶已存在: {settings.minio_bucket_name}")
    except S3Error as e:
        logger.error(f"初始化存储桶失败: {e}")
        raise
    
    logger.info("MinIO存储客户端初始化完成")


def get_storage_client() -> Minio:
    """获取MinIO客户端实例
    
    Returns:
        Minio: MinIO客户端
    """
    if not _storage_client:
        raise RuntimeError("存储客户端未初始化，请先调用 init_storage_client()")
    return _storage_client


async def check_storage_connection() -> bool:
    """检查存储服务连接状态
    
    Returns:
        bool: 连接是否正常
    """
    try:
        settings = get_settings()
        client = get_storage_client()
        client.bucket_exists(settings.minio_bucket_name)
        return True
    except Exception as e:
        logger.error(f"存储连接检查失败: {e}")
        return False


class MinIOClient:
    """MinIO客户端包装类
    
    提供更高级的文件操作接口
    """
    
    def __init__(self):
        self.client = get_storage_client()
        self.settings = get_settings()
        self.bucket_name = self.settings.minio_bucket_name
    
    async def upload_file(
        self, 
        file: UploadFile,
        object_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """上传文件到MinIO
        
        Args:
            file: 上传的文件对象
            object_name: 对象存储中的文件名
            metadata: 文件元数据
            
        Returns:
            上传结果信息
        """
        try:
            # 读取文件内容
            content = await file.read()
            content_stream = BytesIO(content)
            
            # 准备元数据
            file_metadata = {
                'Content-Type': file.content_type or 'application/octet-stream',
                'Content-Length': str(len(content)),
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            # 上传文件
            result = self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=content_stream,
                length=len(content),
                content_type=file.content_type,
                metadata=file_metadata
            )
            
            logger.info(f"文件上传成功: {object_name}")
            
            return {
                "success": True,
                "object_name": object_name,
                "etag": result.etag,
                "size": len(content),
                "bucket": self.bucket_name
            }
            
        except S3Error as e:
            logger.error(f"文件上传失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # 重置文件指针
            await file.seek(0)
    
    async def download_file(self, object_name: str) -> Optional[str]:
        """从MinIO下载文件到本地临时目录
        
        Args:
            object_name: 对象存储中的文件名
            
        Returns:
            本地文件路径，失败返回None
        """
        try:
            import tempfile
            import os
            
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            local_filename = os.path.join(temp_dir, os.path.basename(object_name))
            
            # 下载文件
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=local_filename
            )
            
            logger.info(f"文件下载成功: {object_name} -> {local_filename}")
            return local_filename
            
        except S3Error as e:
            logger.error(f"文件下载失败: {e}")
            return None
    
    def get_file_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """获取文件信息
        
        Args:
            object_name: 对象存储中的文件名
            
        Returns:
            文件信息字典，失败返回None
        """
        try:
            stat = self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            
            return {
                "size": stat.size,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "metadata": stat.metadata
            }
            
        except S3Error as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    def delete_file(self, object_name: str) -> bool:
        """删除文件
        
        Args:
            object_name: 对象存储中的文件名
            
        Returns:
            是否删除成功
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            
            logger.info(f"文件删除成功: {object_name}")
            return True
            
        except S3Error as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def list_files(self, prefix: str = None) -> list:
        """列出文件
        
        Args:
            prefix: 文件名前缀过滤
            
        Returns:
            文件对象列表
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            return [
                {
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag
                }
                for obj in objects
            ]
            
        except S3Error as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    def generate_presigned_url(
        self, 
        object_name: str, 
        expires: int = 3600,
        method: str = "GET"
    ) -> Optional[str]:
        """生成预签名URL
        
        Args:
            object_name: 对象存储中的文件名
            expires: 过期时间(秒)
            method: HTTP方法
            
        Returns:
            预签名URL，失败返回None
        """
        try:
            from datetime import timedelta
            
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=timedelta(seconds=expires)
            )
            
            logger.info(f"生成预签名URL成功: {object_name}")
            return url
            
        except S3Error as e:
            logger.error(f"生成预签名URL失败: {e}")
            return None