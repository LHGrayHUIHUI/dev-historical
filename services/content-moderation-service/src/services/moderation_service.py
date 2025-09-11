"""
内容审核服务核心类

负责管理多媒体内容的智能审核
提供统一的审核接口和流程管理
"""

import logging
import hashlib
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from fastapi import HTTPException

from ..models.moderation_models import (
    ModerationTask, ModerationRule, SensitiveWord, Whitelist, Appeal,
    ContentType, ModerationStatus, RiskLevel
)
from ..analyzers.text_analyzer import TextAnalyzer
from ..analyzers.image_analyzer import ImageAnalyzer
from ..analyzers.video_analyzer import VideoAnalyzer
from ..analyzers.audio_analyzer import AudioAnalyzer
from ..config.settings import settings, ModerationRules, ContentLimits

logger = logging.getLogger(__name__)


class ModerationResult:
    """审核结果数据类"""
    
    def __init__(
        self,
        is_approved: bool,
        confidence_score: float,
        risk_level: RiskLevel,
        violation_types: List[str],
        details: Dict[str, Any],
        processing_time: float
    ):
        self.is_approved = is_approved
        self.confidence_score = confidence_score
        self.risk_level = risk_level
        self.violation_types = violation_types
        self.details = details
        self.processing_time = processing_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "is_approved": self.is_approved,
            "confidence_score": self.confidence_score,
            "risk_level": self.risk_level.value,
            "violation_types": self.violation_types,
            "details": self.details,
            "processing_time": self.processing_time
        }


class ContentModerationService:
    """
    内容审核服务主类
    
    负责管理多媒体内容的智能审核流程
    """
    
    def __init__(self, db_session: AsyncSession, redis_client=None):
        self.db = db_session
        self.redis = redis_client
        
        # 初始化分析器
        self.text_analyzer = TextAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        
        self.logger = logging.getLogger(__name__)
    
    async def submit_for_moderation(
        self, 
        content_id: str,
        content_type: ContentType,
        content_data: Dict[str, Any],
        user_id: Optional[str] = None,
        platform: Optional[str] = None
    ) -> str:
        """
        提交内容进行审核
        
        Args:
            content_id: 内容ID
            content_type: 内容类型
            content_data: 内容数据
            user_id: 用户ID
            platform: 来源平台
            
        Returns:
            str: 审核任务ID
        """
        try:
            # 验证内容限制
            if not ContentLimits.validate_content_limits(content_type.value, content_data):
                raise HTTPException(status_code=400, detail="内容超出处理限制")
            
            # 检查白名单
            if await self._check_whitelist(content_data, user_id):
                self.logger.info(f"内容 {content_id} 在白名单中，直接通过")
                return await self._create_approved_task(content_id, content_type, content_data, user_id, platform)
            
            # 创建审核任务
            task = await self._create_moderation_task(
                content_id=content_id,
                content_type=content_type,
                content_data=content_data,
                user_id=user_id,
                platform=platform
            )
            
            # 异步执行审核
            asyncio.create_task(self._process_moderation_task(str(task.id)))
            
            self.logger.info(f"审核任务创建成功: {task.id}")
            return str(task.id)
            
        except Exception as e:
            self.logger.error(f"提交审核失败: {e}")
            raise HTTPException(status_code=500, detail="提交审核失败")
    
    async def _create_moderation_task(
        self,
        content_id: str,
        content_type: ContentType,
        content_data: Dict[str, Any],
        user_id: Optional[str] = None,
        platform: Optional[str] = None
    ) -> ModerationTask:
        """
        创建审核任务记录
        
        Args:
            content_id: 内容ID
            content_type: 内容类型
            content_data: 内容数据
            user_id: 用户ID
            platform: 来源平台
            
        Returns:
            ModerationTask: 审核任务实例
        """
        # 生成内容哈希
        content_hash = self._generate_content_hash(content_data)
        
        # 检查是否已有相同内容的审核任务
        existing_task = await self._find_existing_task(content_hash)
        if existing_task:
            self.logger.info(f"发现重复内容，复用审核结果: {existing_task.id}")
            return existing_task
        
        # 创建新任务
        task = ModerationTask(
            content_id=content_id,
            content_type=content_type.value,
            content_url=content_data.get('url'),
            content_text=content_data.get('text'),
            content_hash=content_hash,
            source_platform=platform,
            user_id=UUID(user_id) if user_id else None,
            status=ModerationStatus.PENDING.value
        )
        
        self.db.add(task)
        await self.db.flush()  # 获取ID但不提交
        
        return task
    
    async def _create_approved_task(
        self,
        content_id: str,
        content_type: ContentType,
        content_data: Dict[str, Any],
        user_id: Optional[str] = None,
        platform: Optional[str] = None
    ) -> str:
        """
        创建自动通过的任务
        
        Args:
            content_id: 内容ID
            content_type: 内容类型  
            content_data: 内容数据
            user_id: 用户ID
            platform: 来源平台
            
        Returns:
            str: 任务ID
        """
        task = ModerationTask(
            content_id=content_id,
            content_type=content_type.value,
            content_url=content_data.get('url'),
            content_text=content_data.get('text'),
            content_hash=self._generate_content_hash(content_data),
            source_platform=platform,
            user_id=UUID(user_id) if user_id else None,
            status=ModerationStatus.APPROVED.value,
            final_result=ModerationStatus.APPROVED.value,
            confidence_score=1.0,
            risk_level=RiskLevel.LOW.value,
            auto_result={
                "approved_reason": "whitelist",
                "is_approved": True,
                "confidence_score": 1.0
            },
            reviewed_at=func.now()
        )
        
        self.db.add(task)
        await self.db.flush()
        
        return str(task.id)
    
    async def _process_moderation_task(self, task_id: str):
        """
        处理审核任务
        
        Args:
            task_id: 任务ID
        """
        start_time = datetime.utcnow()
        
        try:
            # 获取任务信息
            task = await self._get_task_by_id(task_id)
            if not task:
                self.logger.error(f"任务不存在: {task_id}")
                return
            
            # 更新状态为处理中
            task.status = ModerationStatus.PROCESSING.value
            await self.db.commit()
            
            # 执行审核
            result = await self._execute_moderation(task)
            
            # 计算处理时间
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 更新任务结果
            await self._update_task_result(task, result, processing_time)
            
            self.logger.info(f"任务处理完成: {task_id}, 结果: {result.is_approved}")
            
        except Exception as e:
            self.logger.error(f"任务处理失败: {task_id}, 错误: {e}")
            # 更新任务状态为失败
            await self._mark_task_failed(task_id, str(e))
    
    async def _execute_moderation(self, task: ModerationTask) -> ModerationResult:
        """
        执行内容审核
        
        Args:
            task: 审核任务
            
        Returns:
            ModerationResult: 审核结果
        """
        start_time = datetime.utcnow()
        
        try:
            if task.content_type == ContentType.TEXT.value:
                result = await self._moderate_text(task)
            elif task.content_type == ContentType.IMAGE.value:
                result = await self._moderate_image(task)
            elif task.content_type == ContentType.VIDEO.value:
                result = await self._moderate_video(task)
            elif task.content_type == ContentType.AUDIO.value:
                result = await self._moderate_audio(task)
            else:
                raise ValueError(f"不支持的内容类型: {task.content_type}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"审核执行失败: {e}")
            return ModerationResult(
                is_approved=False,
                confidence_score=0.0,
                risk_level=RiskLevel.HIGH,
                violation_types=["system_error"],
                details={"error": str(e)},
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def _moderate_text(self, task: ModerationTask) -> ModerationResult:
        """
        文本内容审核
        
        Args:
            task: 审核任务
            
        Returns:
            ModerationResult: 审核结果
        """
        if not task.content_text:
            return ModerationResult(
                is_approved=True,
                confidence_score=1.0,
                risk_level=RiskLevel.LOW,
                violation_types=[],
                details={"reason": "empty_content"},
                processing_time=0.0
            )
        
        # 使用文本分析器进行审核
        analysis_result = await self.text_analyzer.analyze(task.content_text)
        
        # 评估审核结果
        is_approved = analysis_result['is_safe']
        confidence_score = analysis_result['confidence']
        violation_types = analysis_result.get('violations', [])
        
        # 确定风险等级
        risk_level = self._determine_risk_level(confidence_score, violation_types)
        
        return ModerationResult(
            is_approved=is_approved,
            confidence_score=confidence_score,
            risk_level=risk_level,
            violation_types=violation_types,
            details=analysis_result,
            processing_time=0.0
        )
    
    async def _moderate_image(self, task: ModerationTask) -> ModerationResult:
        """
        图像内容审核
        
        Args:
            task: 审核任务
            
        Returns:
            ModerationResult: 审核结果
        """
        if not task.content_url:
            return ModerationResult(
                is_approved=False,
                confidence_score=0.0,
                risk_level=RiskLevel.HIGH,
                violation_types=["missing_content"],
                details={"error": "No image URL provided"},
                processing_time=0.0
            )
        
        # 使用图像分析器进行审核
        analysis_result = await self.image_analyzer.analyze(task.content_url)
        
        is_approved = analysis_result['is_safe']
        confidence_score = analysis_result['confidence']
        violation_types = analysis_result.get('violations', [])
        risk_level = self._determine_risk_level(confidence_score, violation_types)
        
        return ModerationResult(
            is_approved=is_approved,
            confidence_score=confidence_score,
            risk_level=risk_level,
            violation_types=violation_types,
            details=analysis_result,
            processing_time=0.0
        )
    
    async def _moderate_video(self, task: ModerationTask) -> ModerationResult:
        """
        视频内容审核
        
        Args:
            task: 审核任务
            
        Returns:
            ModerationResult: 审核结果  
        """
        if not task.content_url:
            return ModerationResult(
                is_approved=False,
                confidence_score=0.0,
                risk_level=RiskLevel.HIGH,
                violation_types=["missing_content"],
                details={"error": "No video URL provided"},
                processing_time=0.0
            )
        
        # 使用视频分析器进行审核
        analysis_result = await self.video_analyzer.analyze(task.content_url)
        
        is_approved = analysis_result['is_safe']
        confidence_score = analysis_result['confidence']
        violation_types = analysis_result.get('violations', [])
        risk_level = self._determine_risk_level(confidence_score, violation_types)
        
        return ModerationResult(
            is_approved=is_approved,
            confidence_score=confidence_score,
            risk_level=risk_level,
            violation_types=violation_types,
            details=analysis_result,
            processing_time=0.0
        )
    
    async def _moderate_audio(self, task: ModerationTask) -> ModerationResult:
        """
        音频内容审核
        
        Args:
            task: 审核任务
            
        Returns:
            ModerationResult: 审核结果
        """
        if not task.content_url:
            return ModerationResult(
                is_approved=False,
                confidence_score=0.0,
                risk_level=RiskLevel.HIGH,
                violation_types=["missing_content"],
                details={"error": "No audio URL provided"},
                processing_time=0.0
            )
        
        # 使用音频分析器进行审核
        analysis_result = await self.audio_analyzer.analyze(task.content_url)
        
        is_approved = analysis_result['is_safe']
        confidence_score = analysis_result['confidence']
        violation_types = analysis_result.get('violations', [])
        risk_level = self._determine_risk_level(confidence_score, violation_types)
        
        return ModerationResult(
            is_approved=is_approved,
            confidence_score=confidence_score,
            risk_level=risk_level,
            violation_types=violation_types,
            details=analysis_result,
            processing_time=0.0
        )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取审核任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, Any]: 任务状态信息
        """
        try:
            task = await self._get_task_by_id(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="审核任务不存在")
            
            return {
                "task_id": str(task.id),
                "content_id": task.content_id,
                "content_type": task.content_type,
                "status": task.status,
                "final_result": task.final_result,
                "confidence_score": float(task.confidence_score) if task.confidence_score else None,
                "risk_level": task.risk_level,
                "violation_types": task.violation_types or [],
                "auto_result": task.auto_result,
                "manual_result": task.manual_result,
                "processing_time": float(task.processing_time) if task.processing_time else None,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat(),
                "reviewed_at": task.reviewed_at.isoformat() if task.reviewed_at else None
            }
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"获取任务状态失败: {e}")
            raise HTTPException(status_code=500, detail="获取任务状态失败")
    
    async def get_batch_results(self, task_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取审核任务结果
        
        Args:
            task_ids: 任务ID列表
            
        Returns:
            Dict[str, Dict[str, Any]]: 任务结果字典
        """
        try:
            results = {}
            for task_id in task_ids:
                try:
                    results[task_id] = await self.get_task_status(task_id)
                except HTTPException:
                    results[task_id] = {"error": "任务不存在"}
                except Exception as e:
                    results[task_id] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"批量获取结果失败: {e}")
            raise HTTPException(status_code=500, detail="批量获取结果失败")
    
    # 辅助方法
    
    def _generate_content_hash(self, content_data: Dict[str, Any]) -> str:
        """
        生成内容哈希值
        
        Args:
            content_data: 内容数据
            
        Returns:
            str: 哈希值
        """
        # 提取关键内容生成哈希
        content_str = ""
        if 'text' in content_data:
            content_str += content_data['text']
        if 'url' in content_data:
            content_str += content_data['url']
        
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    async def _find_existing_task(self, content_hash: str) -> Optional[ModerationTask]:
        """
        查找已存在的相同内容审核任务
        
        Args:
            content_hash: 内容哈希值
            
        Returns:
            Optional[ModerationTask]: 已存在的任务
        """
        stmt = select(ModerationTask).where(
            and_(
                ModerationTask.content_hash == content_hash,
                ModerationTask.status.in_([
                    ModerationStatus.APPROVED.value,
                    ModerationStatus.REJECTED.value
                ])
            )
        ).order_by(desc(ModerationTask.created_at)).limit(1)
        
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _check_whitelist(
        self, 
        content_data: Dict[str, Any], 
        user_id: Optional[str] = None
    ) -> bool:
        """
        检查内容是否在白名单中
        
        Args:
            content_data: 内容数据
            user_id: 用户ID
            
        Returns:
            bool: 是否在白名单中
        """
        try:
            # 检查用户白名单
            if user_id:
                user_stmt = select(Whitelist).where(
                    and_(
                        Whitelist.type == 'user',
                        Whitelist.value == user_id,
                        Whitelist.is_active == True
                    )
                )
                user_result = await self.db.execute(user_stmt)
                if user_result.scalar_one_or_none():
                    return True
            
            # 检查内容哈希白名单
            content_hash = self._generate_content_hash(content_data)
            hash_stmt = select(Whitelist).where(
                and_(
                    Whitelist.type == 'hash',
                    Whitelist.value == content_hash,
                    Whitelist.is_active == True
                )
            )
            hash_result = await self.db.execute(hash_stmt)
            if hash_result.scalar_one_or_none():
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"检查白名单失败: {e}")
            return False
    
    def _determine_risk_level(self, confidence_score: float, violation_types: List[str]) -> RiskLevel:
        """
        根据置信度和违规类型确定风险等级
        
        Args:
            confidence_score: 置信度分数
            violation_types: 违规类型列表
            
        Returns:
            RiskLevel: 风险等级
        """
        # 检查是否包含高危违规类型
        critical_violations = {'politics', 'violence', 'pornography', 'terrorism'}
        if any(v in critical_violations for v in violation_types):
            return RiskLevel.CRITICAL
        
        # 根据置信度确定风险等级
        if confidence_score >= 0.9:
            return RiskLevel.CRITICAL if violation_types else RiskLevel.LOW
        elif confidence_score >= 0.7:
            return RiskLevel.HIGH
        elif confidence_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _get_task_by_id(self, task_id: str) -> Optional[ModerationTask]:
        """
        根据ID获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[ModerationTask]: 任务实例
        """
        stmt = select(ModerationTask).where(ModerationTask.id == UUID(task_id))
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _update_task_result(
        self, 
        task: ModerationTask, 
        result: ModerationResult, 
        processing_time: float
    ):
        """
        更新任务审核结果
        
        Args:
            task: 任务实例
            result: 审核结果
            processing_time: 处理时间
        """
        # 确定最终状态
        if result.confidence_score >= 0.8 and result.risk_level != RiskLevel.CRITICAL:
            # 高置信度自动审核
            final_status = ModerationStatus.APPROVED if result.is_approved else ModerationStatus.REJECTED
        else:
            # 需要人工审核
            final_status = ModerationStatus.MANUAL_REVIEW
        
        # 更新任务
        task.update_status(
            status=final_status.value,
            result=result.to_dict(),
            confidence_score=result.confidence_score,
            risk_level=result.risk_level.value,
            violation_types=result.violation_types
        )
        task.processing_time = processing_time
        
        await self.db.commit()
    
    async def _mark_task_failed(self, task_id: str, error_message: str):
        """
        标记任务失败
        
        Args:
            task_id: 任务ID
            error_message: 错误信息
        """
        try:
            task = await self._get_task_by_id(task_id)
            if task:
                task.status = "failed"
                task.auto_result = {"error": error_message}
                await self.db.commit()
        except Exception as e:
            self.logger.error(f"标记任务失败时出错: {e}")