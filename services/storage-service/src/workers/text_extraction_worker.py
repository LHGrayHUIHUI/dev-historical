"""
文本提取工作器

处理异步文本提取任务的后台工作器
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import aio_pika
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Dataset, TextContent
from ..processors import (
    HTMLExtractor,
    ImageExtractor,
    PDFExtractor,
    PlainTextExtractor,
    TextExtractor,
    WordExtractor,
)
from ..services.message_queue_service import RabbitMQClient
from ..utils.database import get_database_session
from ..utils.storage import MinIOClient

logger = logging.getLogger(__name__)


class TextExtractionWorker:
    """文本提取工作器
    
    负责处理文本提取任务的后台工作器
    """
    
    def __init__(self, message_queue: RabbitMQClient):
        """初始化文本提取工作器
        
        Args:
            message_queue: RabbitMQ客户端
        """
        self.message_queue = message_queue
        self.storage_client = MinIOClient()
        self.running = False
        
        # 初始化文本提取器映射
        self.extractors = {
            'application/pdf': PDFExtractor(),
            'application/x-pdf': PDFExtractor(),
            'application/msword': WordExtractor(),
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': WordExtractor(),
            'text/plain': PlainTextExtractor(),
            'text/html': HTMLExtractor(),
            'application/xhtml+xml': HTMLExtractor(),
            'image/jpeg': ImageExtractor(),
            'image/jpg': ImageExtractor(),
            'image/png': ImageExtractor(),
            'image/tiff': ImageExtractor(),
            'image/bmp': ImageExtractor(),
        }
        
        logger.info("文本提取工作器初始化完成")
    
    async def start(self) -> None:
        """启动工作器"""
        try:
            # 确保消息队列已连接
            if not self.message_queue.is_connected:
                await self.message_queue.connect()
            
            # 设置消费者
            await self.message_queue.setup_consumer(
                queue_name='text_extraction',
                callback=self.process_extraction_task,
                consumer_tag='text_extraction_worker'
            )
            
            self.running = True
            logger.info("文本提取工作器已启动")
            
            # 保持运行
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"文本提取工作器启动失败: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """停止工作器"""
        self.running = False
        logger.info("文本提取工作器正在停止")
    
    async def process_extraction_task(self, message: aio_pika.IncomingMessage) -> None:
        """处理文本提取任务
        
        Args:
            message: 消息队列消息
        """
        task_start_time = datetime.utcnow()
        
        try:
            # 解析消息
            task_data = json.loads(message.body.decode())
            dataset_id = task_data['dataset_id']
            file_path = task_data['file_path']
            file_type = task_data['file_type']
            task_id = task_data.get('task_id', f"extract_{dataset_id}")
            
            logger.info(
                f"开始处理文本提取任务",
                extra={
                    "task_id": task_id,
                    "dataset_id": dataset_id,
                    "file_path": file_path,
                    "file_type": file_type
                }
            )
            
            # 更新数据集状态为处理中
            await self._update_dataset_status(dataset_id, 'processing')
            
            # 从存储下载文件
            local_file_path = await self.storage_client.download_file(file_path)
            if not local_file_path:
                raise Exception("文件下载失败")
            
            try:
                # 选择合适的提取器
                extractor = self.extractors.get(file_type)
                if not extractor:
                    raise Exception(f"不支持的文件类型: {file_type}")
                
                # 提取文本内容
                extracted_contents = await extractor.extract(local_file_path)
                
                if not extracted_contents:
                    logger.warning(f"未提取到文本内容: {dataset_id}")
                    extracted_contents = []
                
                # 保存提取的文本内容
                await self._save_text_contents(dataset_id, extracted_contents)
                
                # 更新数据集统计信息
                await self._update_dataset_statistics(dataset_id, extracted_contents)
                
                # 更新数据集状态为完成
                await self._update_dataset_status(dataset_id, 'completed')
                
                processing_time = (datetime.utcnow() - task_start_time).total_seconds()
                
                logger.info(
                    f"文本提取任务完成",
                    extra={
                        "task_id": task_id,
                        "dataset_id": dataset_id,
                        "extracted_count": len(extracted_contents),
                        "processing_time": processing_time
                    }
                )
                
            finally:
                # 清理临时文件
                if local_file_path and Path(local_file_path).exists():
                    Path(local_file_path).unlink(missing_ok=True)
                    
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"文本提取任务失败",
                extra={
                    "dataset_id": task_data.get('dataset_id') if 'task_data' in locals() else 'unknown',
                    "error": error_msg
                }
            )
            
            # 更新数据集状态为失败
            if 'dataset_id' in locals():
                await self._update_dataset_status(dataset_id, 'failed', error_msg)
            
            # 重新抛出异常以触发重试机制
            raise
    
    async def _update_dataset_status(
        self,
        dataset_id: str,
        status: str,
        error_message: str = None
    ) -> None:
        """更新数据集处理状态
        
        Args:
            dataset_id: 数据集ID
            status: 新状态
            error_message: 错误信息（如果有）
        """
        try:
            async with get_database_session() as session:
                update_data = {
                    'processing_status': status,
                    'updated_at': datetime.utcnow()
                }
                
                if status == 'completed':
                    update_data['processed_at'] = datetime.utcnow()
                elif error_message:
                    update_data['error_message'] = error_message
                
                query = update(Dataset).where(Dataset.id == dataset_id).values(**update_data)
                await session.execute(query)
                await session.commit()
                
                logger.debug(f"数据集状态已更新: {dataset_id} -> {status}")
                
        except Exception as e:
            logger.error(f"更新数据集状态失败: {str(e)}")
    
    async def _save_text_contents(
        self,
        dataset_id: str,
        contents: List[Dict[str, Any]]
    ) -> None:
        """保存提取的文本内容
        
        Args:
            dataset_id: 数据集ID
            contents: 提取的文本内容列表
        """
        try:
            async with get_database_session() as session:
                # 先删除已存在的文本内容（重新处理的情况）
                await session.execute(
                    "DELETE FROM text_contents WHERE dataset_id = :dataset_id",
                    {"dataset_id": dataset_id}
                )
                
                # 批量插入新的文本内容
                text_objects = []
                for content_data in contents:
                    text_obj = TextContent(
                        dataset_id=dataset_id,
                        title=content_data.get('title'),
                        content=content_data['content'],
                        page_number=content_data.get('page_number', 1),
                        word_count=content_data.get('word_count', 0),
                        char_count=content_data.get('char_count', 0),
                        language=content_data.get('language'),
                        confidence_score=content_data.get('confidence'),
                        quality_score=self._calculate_quality_score(content_data),
                        extracted_at=datetime.utcnow()
                    )
                    
                    # 计算基础统计（如果没有提供）
                    if not text_obj.word_count or not text_obj.char_count:
                        text_obj.calculate_basic_stats()
                    
                    # 检测语言（如果没有提供）
                    if not text_obj.language:
                        text_obj.detect_language()
                    
                    text_objects.append(text_obj)
                
                # 批量添加到数据库
                session.add_all(text_objects)
                await session.commit()
                
                logger.debug(f"已保存 {len(text_objects)} 个文本内容项")
                
        except Exception as e:
            logger.error(f"保存文本内容失败: {str(e)}")
            raise
    
    async def _update_dataset_statistics(
        self,
        dataset_id: str,
        contents: List[Dict[str, Any]]
    ) -> None:
        """更新数据集统计信息
        
        Args:
            dataset_id: 数据集ID
            contents: 文本内容列表
        """
        try:
            if not contents:
                return
            
            # 计算统计信息
            text_count = len(contents)
            total_words = sum(item.get('word_count', 0) for item in contents)
            total_chars = sum(item.get('char_count', 0) for item in contents)
            
            # 更新数据集统计
            async with get_database_session() as session:
                query = update(Dataset).where(Dataset.id == dataset_id).values(
                    text_count=text_count,
                    total_words=total_words,
                    total_chars=total_chars,
                    updated_at=datetime.utcnow()
                )
                await session.execute(query)
                await session.commit()
                
                logger.debug(
                    f"数据集统计已更新: {dataset_id}, "
                    f"文本数: {text_count}, 词数: {total_words}, 字符数: {total_chars}"
                )
                
        except Exception as e:
            logger.error(f"更新数据集统计失败: {str(e)}")
    
    def _calculate_quality_score(self, content_data: Dict[str, Any]) -> float:
        """计算文本质量评分
        
        Args:
            content_data: 文本内容数据（来自file-processor的响应）
            
        Returns:
            质量评分 (0-1)
        """
        # 适应file-processor的返回格式
        content = content_data.get('text_content', content_data.get('content', ''))
        success = content_data.get('success', True)
        warnings = content_data.get('warnings', [])
        processing_method = content_data.get('processing_method', '')
        
        if not content or not content.strip():
            return 0.0
        
        score = 1.0
        
        # 基于处理成功状态的评分
        if not success:
            score *= 0.2
        
        # 基于处理方法的评分
        if processing_method == 'failed':
            score *= 0.1
        elif processing_method == 'pypdf2' and 'pdfplumber提取失败' in str(warnings):
            score *= 0.8  # PyPDF2作为备选方案时稍微降分
        
        # 基于内容长度的评分
        content_length = len(content.strip())
        if content_length < 10:
            score *= 0.2
        elif content_length < 50:
            score *= 0.5
        elif content_length < 100:
            score *= 0.7
        elif content_length >= 1000:
            score *= 1.0  # 长内容得高分
        else:
            score *= 0.9
        
        # 基于警告数量的评分
        if warnings:
            warning_penalty = min(0.3, len(warnings) * 0.1)
            score *= (1.0 - warning_penalty)
        
        # 基于特殊字符比例的评分（针对中文文档优化）
        import re
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        # 统计英文字母
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        # 统计数字
        digit_chars = len(re.findall(r'[0-9]', content))
        # 统计标点符号（正常的）
        punctuation_chars = len(re.findall(r'[，。！？；：""''（）【】《》、\.,!?;:()\[\]"\'<>]', content))
        # 统计异常字符（乱码、特殊符号等）
        abnormal_chars = len(re.findall(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】《》、\.,!?;:()\[\]"\'<>-]', content))
        
        valid_chars = chinese_chars + english_chars + digit_chars + punctuation_chars
        if content_length > 0:
            abnormal_ratio = abnormal_chars / content_length
            valid_ratio = valid_chars / content_length
            
            # 异常字符太多说明提取质量差
            if abnormal_ratio > 0.3:
                score *= 0.3
            elif abnormal_ratio > 0.1:
                score *= 0.7
            
            # 有效字符比例高说明质量好
            if valid_ratio > 0.9:
                score *= 1.0
            elif valid_ratio > 0.7:
                score *= 0.9
            else:
                score *= 0.6
        
        # 基于文本完整性的评分（检查是否有明显的截断）
        if content.strip().endswith(('...', '…', '【省略】', '等等')):
            score *= 0.8
        
        return round(max(0.0, min(1.0, score)), 3)
    
    def get_extractor_for_file_type(self, file_type: str) -> TextExtractor:
        """根据文件类型获取对应的提取器
        
        Args:
            file_type: MIME类型
            
        Returns:
            文本提取器实例
            
        Raises:
            ValueError: 不支持的文件类型
        """
        extractor = self.extractors.get(file_type)
        if not extractor:
            raise ValueError(f"不支持的文件类型: {file_type}")
        return extractor
    
    def get_supported_file_types(self) -> List[str]:
        """获取支持的文件类型列表
        
        Returns:
            支持的MIME类型列表
        """
        return list(self.extractors.keys())
    
    async def test_extraction(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """测试文本提取功能
        
        Args:
            file_path: 测试文件路径
            file_type: 文件类型
            
        Returns:
            提取结果
        """
        try:
            extractor = self.get_extractor_for_file_type(file_type)
            
            start_time = datetime.utcnow()
            extracted_contents = await extractor.extract(file_path)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "file_type": file_type,
                "extracted_count": len(extracted_contents),
                "processing_time": processing_time,
                "total_chars": sum(item.get('char_count', 0) for item in extracted_contents),
                "total_words": sum(item.get('word_count', 0) for item in extracted_contents),
                "contents": extracted_contents[:3]  # 只返回前3个内容项作为示例
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_type": file_type
            }