"""
合规性检测引擎

提供全面的内容合规性检查，包括敏感词检测、政策合规、
版权检查、学术诚信检测等功能。
"""

import re
import time
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from loguru import logger

from ..models.quality_models import (
    ComplianceViolation, ComplianceCheckRequest, ComplianceCheckResult,
    ViolationType, ComplianceStatus
)
from ..config.settings import settings
from ..clients.storage_client import StorageServiceClient

class ComplianceEngine:
    """
    合规性检测引擎主类
    
    协调各种合规检测器，执行全面的内容合规性分析
    """
    
    def __init__(self, storage_client: StorageServiceClient):
        """
        初始化合规检测引擎
        
        Args:
            storage_client: 存储服务客户端
        """
        self.storage_client = storage_client
        
        # 初始化各种检测器
        self.detectors = {
            'sensitive_words': SensitiveWordDetector(storage_client),
            'policy': PolicyComplianceDetector(),
            'copyright': CopyrightDetector(),
            'academic_integrity': AcademicIntegrityDetector()
        }
        
        # 缓存敏感词列表
        self._sensitive_words_cache = {}
        self._cache_expiry = None
        
        logger.info("合规性检测引擎初始化完成")
    
    async def check_compliance(self, request: ComplianceCheckRequest) -> ComplianceCheckResult:
        """
        执行全面的合规性检查
        
        Args:
            request: 合规检测请求
            
        Returns:
            合规检测结果
        """
        start_time = time.time()
        logger.info(f"开始合规性检测，内容长度: {len(request.content)}")
        
        # 验证内容长度
        if len(request.content) > settings.MAX_CONTENT_LENGTH:
            raise ValueError(f"内容长度超过限制 ({settings.MAX_CONTENT_LENGTH})")
        
        # 并行执行各类检测
        detection_tasks = []
        for check_type in request.check_types:
            if check_type in self.detectors:
                detector = self.detectors[check_type]
                task = asyncio.create_task(
                    detector.detect(request.content, request.strict_mode),
                    name=f"{check_type}_compliance"
                )
                detection_tasks.append((check_type, task))
        
        # 等待所有检测完成
        all_violations = []
        policy_compliance = {}
        
        for check_type, task in detection_tasks:
            try:
                violations, compliance_info = await task
                all_violations.extend(violations)
                if compliance_info:
                    policy_compliance.update(compliance_info)
                logger.debug(f"{check_type}合规检测完成，发现{len(violations)}个违规")
            except Exception as e:
                logger.error(f"{check_type}合规检测失败: {e}")
                # 检测失败时添加警告状态
                policy_compliance[check_type] = "检测失败"
        
        # 计算风险评分
        risk_score = self._calculate_risk_score(all_violations)
        
        # 确定合规状态
        compliance_status = self._determine_compliance_status(risk_score, all_violations)
        
        # 生成整改建议
        recommendations = await self._generate_recommendations(all_violations)
        
        # 计算处理时间
        processing_time = int((time.time() - start_time) * 1000)
        
        # 创建结果对象
        result = ComplianceCheckResult(
            content_id=getattr(request, 'content_id', None),
            compliance_status=compliance_status,
            risk_score=risk_score,
            violations=all_violations,
            policy_compliance=policy_compliance,
            recommendations=recommendations,
            processing_time_ms=processing_time
        )
        
        logger.info(f"合规性检测完成，风险评分: {risk_score}，违规数: {len(all_violations)}")
        return result
    
    def _calculate_risk_score(self, violations: List[ComplianceViolation]) -> int:
        """
        计算风险评分
        
        Args:
            violations: 违规列表
            
        Returns:
            风险评分 (0-10)
        """
        if not violations:
            return 0
        
        risk_score = 0
        
        for violation in violations:
            # 基础分数
            base_score = violation.severity
            
            # 根据违规类型调整权重
            if violation.violation_type == ViolationType.SENSITIVE_WORD:
                weight = 1.5
            elif violation.violation_type == ViolationType.POLICY_VIOLATION:
                weight = 2.0
            elif violation.violation_type == ViolationType.COPYRIGHT_VIOLATION:
                weight = 2.5
            elif violation.violation_type == ViolationType.ACADEMIC_INTEGRITY:
                weight = 2.0
            else:
                weight = 1.0
            
            # 根据置信度调整
            confidence_factor = violation.confidence
            
            risk_score += base_score * weight * confidence_factor
        
        return min(10, int(risk_score))
    
    def _determine_compliance_status(self, 
                                   risk_score: int, 
                                   violations: List[ComplianceViolation]) -> ComplianceStatus:
        """
        确定合规状态
        
        Args:
            risk_score: 风险评分
            violations: 违规列表
            
        Returns:
            合规状态
        """
        # 检查是否有严重违规
        critical_violations = [
            v for v in violations 
            if v.severity >= 8 and v.confidence > 0.8
        ]
        
        if critical_violations:
            return ComplianceStatus.FAIL
        elif risk_score >= 7:
            return ComplianceStatus.FAIL
        elif risk_score >= 4 or any(v.severity >= 6 for v in violations):
            return ComplianceStatus.WARNING
        else:
            return ComplianceStatus.PASS
    
    async def _generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """
        生成整改建议
        
        Args:
            violations: 违规列表
            
        Returns:
            整改建议列表
        """
        recommendations = []
        
        # 按违规类型分组统计
        violation_counts = {}
        for violation in violations:
            violation_type = violation.violation_type
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        # 生成针对性建议
        if violation_counts.get(ViolationType.SENSITIVE_WORD, 0) > 0:
            recommendations.append("发现敏感词汇，请替换为更合适的表达")
        
        if violation_counts.get(ViolationType.POLICY_VIOLATION, 0) > 0:
            recommendations.append("内容可能违反相关政策，请仔细审查并修改")
        
        if violation_counts.get(ViolationType.COPYRIGHT_VIOLATION, 0) > 0:
            recommendations.append("可能存在版权风险，请确认内容的原创性或获得授权")
        
        if violation_counts.get(ViolationType.ACADEMIC_INTEGRITY, 0) > 0:
            recommendations.append("发现学术诚信问题，请确保内容的原创性和引用规范")
        
        # 根据严重程度生成建议
        high_severity_violations = [v for v in violations if v.severity >= 7]
        if high_severity_violations:
            recommendations.append("发现高风险问题，建议暂缓发布并进行人工审核")
        
        return recommendations[:5]  # 最多返回5条建议

class SensitiveWordDetector:
    """敏感词检测器"""
    
    def __init__(self, storage_client: StorageServiceClient):
        """
        初始化敏感词检测器
        
        Args:
            storage_client: 存储服务客户端
        """
        self.storage_client = storage_client
        self._word_cache = {}
        self._cache_expiry = None
        
        logger.debug("敏感词检测器初始化完成")
    
    async def detect(self, content: str, strict_mode: bool = False) -> tuple[List[ComplianceViolation], Dict[str, str]]:
        """
        检测敏感词汇
        
        Args:
            content: 待检测内容
            strict_mode: 严格模式
            
        Returns:
            (违规列表, 策略合规状态)
        """
        violations = []
        
        # 获取敏感词列表
        sensitive_words = await self._get_sensitive_words()
        
        # 检测敏感词
        for word_info in sensitive_words:
            word = word_info['word']
            category = word_info.get('category', 'general')
            severity = word_info.get('severity_level', 3)
            replacement = word_info.get('replacement_suggestion', '***')
            
            # 查找所有出现位置
            pattern = re.escape(word)
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # 上下文分析
                start_pos = max(0, match.start() - 10)
                end_pos = min(len(content), match.end() + 10)
                context = content[start_pos:end_pos]
                
                # 计算置信度
                confidence = self._calculate_confidence(word, context, category)
                
                if confidence > 0.5:  # 置信度阈值
                    violations.append(ComplianceViolation(
                        violation_type=ViolationType.SENSITIVE_WORD,
                        severity=severity,
                        position=match.start(),
                        length=len(word),
                        content=word,
                        description=f"检测到敏感词汇: {word}",
                        category=category,
                        action="replace",
                        suggestion=f"建议替换为: {replacement}",
                        confidence=confidence
                    ))
        
        # 策略合规状态
        policy_status = "pass" if not violations else ("warning" if len(violations) <= 3 else "fail")
        
        return violations, {"sensitive_word_check": policy_status}
    
    async def _get_sensitive_words(self) -> List[Dict[str, Any]]:
        """获取敏感词列表"""
        current_time = time.time()
        
        # 检查缓存是否过期
        if (self._cache_expiry is None or 
            current_time > self._cache_expiry or 
            not self._word_cache):
            
            try:
                # 从存储服务获取敏感词
                result = await self.storage_client.get_sensitive_words(active_only=True)
                self._word_cache = result.get('data', [])
                self._cache_expiry = current_time + 1800  # 缓存30分钟
                logger.debug(f"敏感词列表已更新，共{len(self._word_cache)}个词汇")
            except Exception as e:
                logger.error(f"获取敏感词列表失败: {e}")
                # 使用默认敏感词列表
                self._word_cache = self._get_default_sensitive_words()
        
        return self._word_cache
    
    def _get_default_sensitive_words(self) -> List[Dict[str, Any]]:
        """获取默认敏感词列表"""
        return [
            {
                "word": "违禁词1",
                "category": "political",
                "severity_level": 8,
                "replacement_suggestion": "合适表达"
            },
            {
                "word": "违禁词2", 
                "category": "social",
                "severity_level": 6,
                "replacement_suggestion": "其他表达"
            }
        ]
    
    def _calculate_confidence(self, word: str, context: str, category: str) -> float:
        """
        计算检测置信度
        
        Args:
            word: 敏感词
            context: 上下文
            category: 分类
            
        Returns:
            置信度 (0.0-1.0)
        """
        base_confidence = 0.8
        
        # 根据上下文调整置信度
        if len(word) <= 2:  # 短词的置信度较低
            base_confidence *= 0.7
        
        if word.isalpha():  # 纯字母词的置信度较高
            base_confidence *= 1.1
        
        # 根据分类调整
        if category == "political":
            base_confidence *= 1.2
        elif category == "social":
            base_confidence *= 1.0
        else:
            base_confidence *= 0.9
        
        return min(1.0, base_confidence)

class PolicyComplianceDetector:
    """政策合规检测器"""
    
    async def detect(self, content: str, strict_mode: bool = False) -> tuple[List[ComplianceViolation], Dict[str, str]]:
        """
        检测政策合规性
        
        Args:
            content: 待检测内容
            strict_mode: 严格模式
            
        Returns:
            (违规列表, 策略合规状态)
        """
        violations = []
        
        # 检查内容长度
        if len(content) > 50000:
            violations.append(ComplianceViolation(
                violation_type=ViolationType.POLICY_VIOLATION,
                severity=4,
                position=0,
                length=len(content),
                content="内容过长",
                description="内容长度超过建议限制",
                action="shorten",
                suggestion="建议适当缩减内容长度",
                confidence=1.0
            ))
        
        # 检查是否包含特定模式
        policy_patterns = [
            (r'广告|推广|营销', "可能包含商业推广内容", 5),
            (r'联系方式|电话|微信|QQ', "可能包含联系方式", 3),
            (r'链接|网址|http', "可能包含外部链接", 2)
        ]
        
        for pattern, description, severity in policy_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                violations.append(ComplianceViolation(
                    violation_type=ViolationType.POLICY_VIOLATION,
                    severity=severity,
                    position=match.start(),
                    length=match.end() - match.start(),
                    content=match.group(),
                    description=description,
                    action="review",
                    suggestion="请审查相关内容是否符合平台政策",
                    confidence=0.7
                ))
        
        # 策略合规状态
        policy_status = "pass" if not violations else "warning"
        
        return violations, {"policy_check": policy_status}

class CopyrightDetector:
    """版权检测器"""
    
    async def detect(self, content: str, strict_mode: bool = False) -> tuple[List[ComplianceViolation], Dict[str, str]]:
        """
        检测版权风险
        
        Args:
            content: 待检测内容
            strict_mode: 严格模式
            
        Returns:
            (违规列表, 策略合规状态)
        """
        violations = []
        
        # 检查是否包含版权声明
        copyright_patterns = [
            r'©\s*\d{4}',
            r'版权所有',
            r'copyright\s*\d{4}',
            r'保留所有权利'
        ]
        
        has_copyright = any(
            re.search(pattern, content, re.IGNORECASE) 
            for pattern in copyright_patterns
        )
        
        if has_copyright:
            violations.append(ComplianceViolation(
                violation_type=ViolationType.COPYRIGHT_VIOLATION,
                severity=6,
                position=0,
                length=0,
                content="版权声明",
                description="内容可能包含他人版权信息",
                action="verify",
                suggestion="请确认内容的原创性或版权授权",
                confidence=0.8
            ))
        
        # 检查引用格式
        citation_patterns = [
            r'引用自|来源于|摘自',
            r'\[.*\]',
            r'参考文献'
        ]
        
        has_citations = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in citation_patterns
        )
        
        if has_citations:
            # 引用是好事，但需要确保格式正确
            policy_status = "pass"
        else:
            policy_status = "warning" if violations else "pass"
        
        return violations, {"copyright_check": policy_status}

class AcademicIntegrityDetector:
    """学术诚信检测器"""
    
    async def detect(self, content: str, strict_mode: bool = False) -> tuple[List[ComplianceViolation], Dict[str, str]]:
        """
        检测学术诚信问题
        
        Args:
            content: 待检测内容
            strict_mode: 严格模式
            
        Returns:
            (违规列表, 策略合规状态)
        """
        violations = []
        
        # 检查文本相似度（简化版）
        # 在实际应用中，这里应该调用专业的查重服务
        
        # 检查是否有适当的引用
        has_proper_citations = await self._check_citations(content)
        
        if not has_proper_citations and len(content) > 1000:
            violations.append(ComplianceViolation(
                violation_type=ViolationType.ACADEMIC_INTEGRITY,
                severity=4,
                position=0,
                length=0,
                content="缺少引用",
                description="长篇内容缺少适当的引用或参考文献",
                action="add_citations",
                suggestion="建议添加相关引用和参考文献",
                confidence=0.6
            ))
        
        # 检查重复内容
        duplicate_issues = await self._check_duplicates(content)
        violations.extend(duplicate_issues)
        
        # 策略合规状态
        policy_status = "pass" if not violations else "warning"
        
        return violations, {"academic_integrity": policy_status}
    
    async def _check_citations(self, content: str) -> bool:
        """检查是否有适当的引用"""
        citation_indicators = [
            r'\[\d+\]',  # [1], [2] 等
            r'\(\d{4}\)',  # (2023) 等
            r'参考文献',
            r'引用',
            r'来源'
        ]
        
        return any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in citation_indicators
        )
    
    async def _check_duplicates(self, content: str) -> List[ComplianceViolation]:
        """检查重复内容"""
        violations = []
        
        # 简化的重复检查：检查段落内的重复句子
        sentences = re.split(r'[。！？]', content)
        sentence_counts = {}
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:  # 只检查长度超过10的句子
                if sentence in sentence_counts:
                    violations.append(ComplianceViolation(
                        violation_type=ViolationType.ACADEMIC_INTEGRITY,
                        severity=3,
                        position=content.find(sentence),
                        length=len(sentence),
                        content=sentence[:30] + "...",
                        description="发现重复句子",
                        action="review",
                        suggestion="请检查并删除重复内容",
                        confidence=0.9
                    ))
                else:
                    sentence_counts[sentence] = i
        
        return violations