"""
数据采集服务核心业务逻辑

负责文件上传、存储和初步处理的主要服务类
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile
from prometheus_client import Counter, Gauge, Histogram
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import Dataset, DataSource, TextContent
from ..processors import (
    HTMLExtractor,
    ImageExtractor,
    PDFExtractor,
    PlainTextExtractor,
    WordExtractor,
)
from ..utils.database import get_database_session
from ..utils.storage import MinIOClient
from .duplicate_detector import DuplicateDetector
from .message_queue_service import RabbitMQClient
from .virus_scanner import VirusScanner

logger = logging.getLogger(__name__)

# 监控指标
file_upload_counter = Counter('file_uploads_total', 'Total file uploads', ['status', 'file_type'])
file_processing_duration = Histogram('file_processing_duration_seconds', 'File processing duration')
active_uploads = Gauge('active_uploads', 'Number of active uploads')
extraction_success_counter = Counter('text_extraction_success_total', 'Successful text extractions', ['file_type'])
extraction_error_counter = Counter('text_extraction_error_total', 'Failed text extractions', ['file_type', 'error_type'])


class DataCollectionService:
    """数据采集服务 - 负责文件上传、存储和初步处理"""
    
    def __init__(self):
        """初始化数据采集服务"""
        self.settings = get_settings()
        self.storage_client = MinIOClient()
        self.message_queue: Optional[RabbitMQClient] = None
        self.duplicate_detector = DuplicateDetector()
        
        # 初始化病毒扫描器
        virus_config = {
            'enabled': self.settings.virus_scan_enabled,
            'host': self.settings.clamav_host,
            'port': self.settings.clamav_port
        }
        self.virus_scanner = VirusScanner(virus_config)
        
        # 初始化文本提取器配置
        extractor_config = {
            'use_pdfplumber': True,
            'extract_images': False,
            'ocr_fallback': False  # 在测试环境中禁用OCR
        }
        
        # 初始化文本提取器
        self.extractors = {
            'application/pdf': PDFExtractor(extractor_config),
            'application/x-pdf': PDFExtractor(extractor_config),
            'application/msword': WordExtractor(extractor_config),
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': WordExtractor(extractor_config),
            'text/plain': PlainTextExtractor(extractor_config),
            'text/html': HTMLExtractor(extractor_config),
            'application/xhtml+xml': HTMLExtractor(extractor_config),
            'image/jpeg': ImageExtractor(extractor_config),
            'image/jpg': ImageExtractor(extractor_config),
            'image/png': ImageExtractor(extractor_config),
            'image/tiff': ImageExtractor(extractor_config),
            'image/bmp': ImageExtractor(extractor_config),
        }
        
        logger.info("数据采集服务初始化完成")
    
    def set_message_queue(self, message_queue: RabbitMQClient) -> None:
        """设置消息队列客户端
        
        Args:
            message_queue: RabbitMQ客户端实例
        """
        self.message_queue = message_queue
    
    async def upload_single_file(
        self,
        file: UploadFile,
        source_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """单文件上传处理
        
        Args:
            file: 上传的文件对象
            source_id: 数据源ID
            user_id: 用户ID
            metadata: 额外的元数据信息
            
        Returns:
            上传结果信息
            
        Raises:
            HTTPException: 文件验证失败或处理错误
        """
        active_uploads.inc()
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"开始处理文件上传: {file.filename}, 用户: {user_id}")
            
            # 1. 基础验证
            await self._validate_file(file)
            
            # 2. 计算文件哈希值
            file_hash = await self._calculate_file_hash(file)
            
            # 3. 检查重复文件
            existing_dataset = await self.duplicate_detector.check_duplicate(file_hash, user_id)
            if existing_dataset:
                logger.info(f"发现重复文件: {file_hash}, 现有数据集: {existing_dataset.id}")
                return {
                    "success": True,
                    "dataset_id": str(existing_dataset.id),
                    "message": "文件已存在，返回现有数据集",
                    "is_duplicate": True,
                    "filename": existing_dataset.name,
                    "file_size": existing_dataset.file_size,
                    "upload_status": "duplicate",
                    "processing_status": existing_dataset.processing_status
                }
            
            # 4. 病毒扫描
            scan_result = await self.virus_scanner.scan_file(file)
            if not scan_result.is_clean:
                file_upload_counter.labels(status='virus_detected', file_type=file.content_type).inc()
                raise HTTPException(
                    status_code=400, 
                    detail=f"文件安全检查失败: {scan_result.threat_name}"
                )
            
            # 5. 生成存储路径
            file_id = str(uuid.uuid4())
            file_path = self._generate_file_path(user_id, file_id, file.filename)
            
            # 6. 上传到对象存储
            storage_result = await self.storage_client.upload_file(
                file=file,
                object_name=file_path,
                metadata={
                    'user_id': user_id,
                    'source_id': source_id,
                    'original_filename': file.filename,
                    'content_type': file.content_type,
                    'upload_time': datetime.utcnow().isoformat(),
                    'file_hash': file_hash
                }
            )
            
            if not storage_result.get("success"):
                raise HTTPException(status_code=500, detail="文件存储失败")
            
            # 7. 创建数据集记录
            dataset = await self._create_dataset_record(
                file=file,
                source_id=source_id,
                file_path=file_path,
                file_hash=file_hash,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            # 8. 发送异步处理任务
            if self.message_queue and self.message_queue.is_connected:
                await self.message_queue.publish_processing_task(
                    dataset_id=str(dataset.id),
                    file_path=file_path,
                    file_type=file.content_type,
                    task_type='text_extraction'
                )
            else:
                logger.warning("消息队列未连接，无法发送处理任务")
            
            # 9. 记录成功指标
            file_upload_counter.labels(status='success', file_type=file.content_type).inc()
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            file_processing_duration.observe(processing_time)
            
            logger.info(
                f"文件上传成功: {file.filename}, 数据集ID: {dataset.id}, 处理时间: {processing_time}s"
            )
            
            return {
                "success": True,
                "dataset_id": str(dataset.id),
                "message": "文件上传成功，正在处理中",
                "filename": file.filename,
                "file_size": file.size,
                "upload_status": "uploaded",
                "processing_status": "pending",
                "file_info": {
                    "filename": file.filename,
                    "size": file.size,
                    "type": file.content_type,
                    "hash": file_hash
                },
                "estimated_processing_time": self._estimate_processing_time(file.size, file.content_type)
            }
            
        except HTTPException:
            file_upload_counter.labels(status='validation_failed', file_type=file.content_type or 'unknown').inc()
            raise
        except Exception as e:
            file_upload_counter.labels(status='error', file_type=file.content_type or 'unknown').inc()
            logger.error(f"文件上传失败: {file.filename}, 错误: {str(e)}")
            raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
        finally:
            active_uploads.dec()
    
    async def upload_batch_files(
        self,
        files: List[UploadFile],
        source_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """批量文件上传处理
        
        Args:
            files: 上传的文件列表
            source_id: 数据源ID
            user_id: 用户ID
            metadata: 额外的元数据信息
            
        Returns:
            批量上传结果信息
        """
        if len(files) > self.settings.max_batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"批量上传文件数量超过限制: {len(files)} > {self.settings.max_batch_size}"
            )
        
        logger.info(f"开始批量文件上传: {len(files)} 个文件, 用户: {user_id}")
        
        uploaded_files = []
        failed_files = []
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(self.settings.upload_concurrency)
        
        async def upload_with_semaphore(file: UploadFile) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.upload_single_file(file, source_id, user_id, metadata)
                    return {
                        "filename": file.filename,
                        "status": "success",
                        "dataset_id": result["dataset_id"],
                        "message": result["message"],
                        "is_duplicate": result.get("is_duplicate", False)
                    }
                except Exception as e:
                    return {
                        "filename": file.filename,
                        "status": "failed",
                        "error": str(e)
                    }
        
        # 并发处理所有文件
        upload_tasks = [upload_with_semaphore(file) for file in files]
        results = await asyncio.gather(*upload_tasks, return_exceptions=False)
        
        # 分类结果
        for result in results:
            if result.get("status") == "success":
                uploaded_files.append(result)
            else:
                failed_files.append(result)
        
        logger.info(
            f"批量文件上传完成: 总计 {len(files)}, 成功 {len(uploaded_files)}, 失败 {len(failed_files)}"
        )
        
        return {
            "success": True,
            "batch_id": str(uuid.uuid4()),
            "total_files": len(files),
            "successful_uploads": len(uploaded_files),
            "failed_uploads": len(failed_files),
            "uploaded_files": uploaded_files,
            "failed_files": failed_files
        }
    
    async def get_datasets(
        self,
        user_id: str,
        page: int = 1,
        size: int = 20,
        status: Optional[str] = None,
        source_id: Optional[str] = None,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取数据集列表
        
        Args:
            user_id: 用户ID
            page: 页码
            size: 每页大小
            status: 状态过滤
            source_id: 数据源ID过滤
            search: 搜索关键词
            
        Returns:
            数据集列表和分页信息
        """
        try:
            async with get_database_session() as session:
                # 构建查询
                query = select(Dataset).where(Dataset.created_by == user_id)
                
                # 添加过滤条件
                if status:
                    query = query.where(Dataset.processing_status == status)
                
                if source_id:
                    query = query.where(Dataset.source_id == source_id)
                
                if search:
                    query = query.where(Dataset.name.ilike(f"%{search}%"))
                
                # 计算总数
                count_result = await session.execute(query)
                total = len(count_result.scalars().all())
                
                # 分页查询
                query = query.order_by(Dataset.created_at.desc())
                query = query.offset((page - 1) * size).limit(size)
                
                result = await session.execute(query)
                datasets = result.scalars().all()
                
                # 转换为字典格式
                items = []
                for dataset in datasets:
                    item = dataset.dict()
                    items.append(item)
                
                return {
                    "items": items,
                    "total": total,
                    "page": page,
                    "size": size,
                    "total_pages": (total + size - 1) // size
                }
                
        except Exception as e:
            logger.error(f"获取数据集列表失败: {str(e)}")
            raise HTTPException(status_code=500, detail="获取数据集列表失败")
    
    async def get_dataset(
        self,
        dataset_id: str,
        user_id: str,
        include_content: bool = False
    ) -> Optional[Dict[str, Any]]:
        """获取单个数据集详情
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            include_content: 是否包含文本内容
            
        Returns:
            数据集详情，不存在返回None
        """
        try:
            async with get_database_session() as session:
                # 查询数据集
                query = select(Dataset).where(
                    Dataset.id == dataset_id,
                    Dataset.created_by == user_id
                )
                
                result = await session.execute(query)
                dataset = result.scalar_one_or_none()
                
                if not dataset:
                    return None
                
                # 转换为字典
                dataset_data = dataset.dict()
                
                # 包含文本内容（可选）
                if include_content:
                    content_query = select(TextContent).where(
                        TextContent.dataset_id == dataset_id
                    ).order_by(TextContent.page_number)
                    
                    content_result = await session.execute(content_query)
                    text_contents = content_result.scalars().all()
                    
                    dataset_data["text_contents"] = [
                        content.dict() for content in text_contents
                    ]
                
                return dataset_data
                
        except Exception as e:
            logger.error(f"获取数据集详情失败: {str(e)}")
            raise HTTPException(status_code=500, detail="获取数据集详情失败")
    
    async def get_processing_status(self, dataset_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集处理状态
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            
        Returns:
            处理状态信息
        """
        try:
            async with get_database_session() as session:
                query = select(Dataset).where(
                    Dataset.id == dataset_id,
                    Dataset.created_by == user_id
                )
                
                result = await session.execute(query)
                dataset = result.scalar_one_or_none()
                
                if not dataset:
                    return None
                
                return {
                    "dataset_id": str(dataset.id),
                    "processing_status": dataset.processing_status,
                    "error_message": dataset.error_message,
                    "text_count": dataset.text_count,
                    "total_words": dataset.total_words,
                    "total_chars": dataset.total_chars,
                    "processed_at": dataset.processed_at.isoformat() if dataset.processed_at else None,
                    "created_at": dataset.created_at.isoformat(),
                    "updated_at": dataset.updated_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"获取处理状态失败: {str(e)}")
            raise HTTPException(status_code=500, detail="获取处理状态失败")
    
    async def reprocess_dataset(self, dataset_id: str, user_id: str) -> bool:
        """重新处理数据集
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            
        Returns:
            是否成功加入重新处理队列
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
                    return False
                
                # 只能重新处理失败或完成的数据集
                if dataset.processing_status not in ['failed', 'completed']:
                    return False
                
                # 更新状态为pending
                await session.execute(
                    update(Dataset)
                    .where(Dataset.id == dataset_id)
                    .values(
                        processing_status='pending',
                        error_message=None,
                        updated_at=datetime.utcnow()
                    )
                )
                
                await session.commit()
                
                # 发送处理任务
                if self.message_queue and self.message_queue.is_connected:
                    await self.message_queue.publish_processing_task(
                        dataset_id=dataset_id,
                        file_path=dataset.file_path,
                        file_type=dataset.file_type,
                        task_type='text_extraction'
                    )
                
                logger.info(f"数据集重新处理任务已发送: {dataset_id}")
                return True
                
        except Exception as e:
            logger.error(f"重新处理数据集失败: {str(e)}")
            return False
    
    # 私有方法
    
    async def _validate_file(self, file: UploadFile) -> None:
        """验证上传文件
        
        Args:
            file: 上传的文件
            
        Raises:
            HTTPException: 文件验证失败
        """
        # 检查文件大小
        if file.size > self.settings.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"文件大小超过限制: {file.size} > {self.settings.max_file_size}"
            )
        
        # 检查文件类型
        if file.content_type not in self.settings.allowed_file_types:
            raise HTTPException(
                status_code=415, 
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        # 检查文件名
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        # 检查文件名长度和字符
        if len(file.filename) > 255:
            raise HTTPException(status_code=400, detail="文件名过长")
        
        # 检查危险文件扩展名
        dangerous_extensions = ['.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext in dangerous_extensions:
            raise HTTPException(status_code=400, detail="不允许的文件扩展名")
    
    async def _calculate_file_hash(self, file: UploadFile) -> str:
        """计算文件SHA256哈希值
        
        Args:
            file: 上传的文件
            
        Returns:
            文件的SHA256哈希值
        """
        hasher = hashlib.sha256()
        
        # 重置文件指针
        await file.seek(0)
        
        # 分块读取文件计算哈希
        while chunk := await file.read(8192):
            hasher.update(chunk)
        
        # 重置文件指针供后续使用
        await file.seek(0)
        
        return hasher.hexdigest()
    
    def _generate_file_path(self, user_id: str, file_id: str, filename: str) -> str:
        """生成文件存储路径
        
        Args:
            user_id: 用户ID
            file_id: 文件唯一ID
            filename: 原始文件名
            
        Returns:
            生成的文件存储路径
        """
        # 按日期和用户组织目录结构
        date_path = datetime.utcnow().strftime("%Y/%m/%d")
        safe_filename = self._sanitize_filename(filename)
        return f"uploads/{date_path}/{user_id}/{file_id}/{safe_filename}"
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除不安全字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            清理后的安全文件名
        """
        import re
        
        # 移除路径分隔符和特殊字符
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # 限制文件名长度
        if len(safe_name) > 255:
            name, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
            safe_name = name[:250] + ('.' + ext if ext else '')
        
        return safe_name
    
    def _estimate_processing_time(self, file_size: int, content_type: str) -> int:
        """估算文件处理时间（秒）
        
        Args:
            file_size: 文件大小（字节）
            content_type: 文件类型
            
        Returns:
            预估处理时间（秒）
        """
        # 基于文件类型和大小的处理时间估算
        base_time = {
            'application/pdf': 2,  # PDF每MB需要2秒
            'image/jpeg': 5,       # 图片OCR每MB需要5秒
            'image/png': 5,
            'text/plain': 0.1,     # 纯文本很快
            'application/msword': 1,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 1,
        }.get(content_type, 1)  # 默认每MB 1秒
        
        file_size_mb = file_size / (1024 * 1024)
        return max(int(file_size_mb * base_time), 5)  # 最少5秒
    
    async def _create_dataset_record(
        self,
        file: UploadFile,
        source_id: str,
        file_path: str,
        file_hash: str,
        user_id: str,
        metadata: Dict[str, Any]
    ) -> Dataset:
        """创建数据集记录
        
        Args:
            file: 上传的文件
            source_id: 数据源ID
            file_path: 文件存储路径
            file_hash: 文件哈希值
            user_id: 用户ID
            metadata: 元数据
            
        Returns:
            创建的数据集对象
        """
        async with get_database_session() as session:
            # 创建数据集记录
            dataset = Dataset(
                name=file.filename,
                description=metadata.get('description'),
                source_id=source_id,
                file_path=file_path,
                file_size=file.size,
                file_type=file.content_type,
                file_hash=file_hash,
                metadata=metadata,
                processing_status='pending',
                created_by=user_id
            )
            
            session.add(dataset)
            await session.commit()
            await session.refresh(dataset)
            
            logger.info(f"数据集记录已创建: {dataset.id}")
            return dataset