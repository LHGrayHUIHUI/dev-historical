"""
质量检测引擎

提供多维度的内容质量检测功能，包括语法检测、逻辑分析、
格式检查、事实验证和学术标准评估。
"""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import jieba
import jieba.posseg as pseg
from loguru import logger

from ..models.quality_models import (
    QualityIssue, QualityMetrics, QualityCheckRequest, QualityCheckResult,
    IssueType, IssueSeverity
)
from ..config.settings import settings
from ..clients.storage_client import StorageServiceClient

class QualityDetectionEngine:
    """
    质量检测引擎主类
    
    负责协调各个检测器，执行全面的内容质量分析
    """
    
    def __init__(self, storage_client: StorageServiceClient):
        """
        初始化质量检测引擎
        
        Args:
            storage_client: 存储服务客户端
        """
        self.storage_client = storage_client
        
        # 初始化各种检测器
        self.detectors = {
            'grammar': GrammarDetector(),
            'logic': LogicDetector(),
            'format': FormatDetector(),
            'factual': FactualDetector(),
            'academic': AcademicDetector()
        }
        
        # 加载质量评分权重
        self.weights = {
            'grammar': settings.GRAMMAR_WEIGHT,
            'logic': settings.LOGIC_WEIGHT,
            'format': settings.FORMAT_WEIGHT,
            'factual': settings.FACTUAL_WEIGHT,
            'academic': settings.ACADEMIC_WEIGHT
        }
        
        logger.info("质量检测引擎初始化完成")
    
    async def check_quality(self, request: QualityCheckRequest) -> QualityCheckResult:
        """
        执行全面的质量检测
        
        Args:
            request: 质量检测请求
            
        Returns:
            质量检测结果
        """
        start_time = time.time()
        logger.info(f"开始质量检测，内容长度: {len(request.content)}")
        
        # 验证内容长度
        if len(request.content) > settings.MAX_CONTENT_LENGTH:
            raise ValueError(f"内容长度超过限制 ({settings.MAX_CONTENT_LENGTH})")
        
        # 预处理文本
        processed_content = await self._preprocess_content(request.content)
        
        # 并行执行各类检测
        detection_tasks = []
        for detector_name, detector in self.detectors.items():
            if request.check_options.get(f"{detector_name}_check", True):
                task = asyncio.create_task(
                    detector.detect(processed_content, request.content_type),
                    name=f"{detector_name}_detection"
                )
                detection_tasks.append((detector_name, task))
        
        # 等待所有检测完成
        detection_results = {}
        for detector_name, task in detection_tasks:
            try:
                issues, metrics = await task
                detection_results[detector_name] = (issues, metrics)
                logger.debug(f"{detector_name}检测完成，发现{len(issues)}个问题")
            except Exception as e:
                logger.error(f"{detector_name}检测失败: {e}")
                detection_results[detector_name] = ([], {f"{detector_name}_score": 50.0})
        
        # 合并检测结果
        all_issues = []
        all_metrics = {}
        
        for detector_name, (issues, metrics) in detection_results.items():
            all_issues.extend(issues)
            for key, value in metrics.items():
                all_metrics[f"{detector_name}_{key}"] = value
        
        # 计算综合质量指标
        quality_metrics = await self._calculate_quality_metrics(detection_results)
        
        # 生成改进建议
        suggestions = await self._generate_suggestions(all_issues, quality_metrics)
        
        # 生成自动修复方案
        auto_fixes = []
        if request.auto_fix and settings.AUTO_FIX_ENABLED:
            auto_fixes = await self._generate_auto_fixes(all_issues, request.content)
        
        # 计算处理时间
        processing_time = int((time.time() - start_time) * 1000)
        
        # 确定检测状态
        status = self._determine_status(quality_metrics.overall_score, all_issues)
        
        # 创建结果对象
        result = QualityCheckResult(
            content_id=getattr(request, 'content_id', None),
            overall_score=quality_metrics.overall_score,
            status=status,
            metrics=quality_metrics,
            issues=all_issues,
            suggestions=suggestions,
            auto_fixes=auto_fixes,
            processing_time_ms=processing_time
        )
        
        logger.info(f"质量检测完成，总分: {quality_metrics.overall_score:.1f}，问题数: {len(all_issues)}")
        return result
    
    async def _preprocess_content(self, content: str) -> str:
        """
        预处理文本内容
        
        Args:
            content: 原始内容
            
        Returns:
            预处理后的内容
        """
        # 去除多余空白
        content = re.sub(r'\s+', ' ', content.strip())
        
        # 规范化标点符号
        content = content.replace('，', '，').replace('。', '。')
        content = content.replace('？', '？').replace('！', '！')
        
        return content
    
    async def _calculate_quality_metrics(self, detection_results: Dict) -> QualityMetrics:
        """
        计算综合质量指标
        
        Args:
            detection_results: 各检测器的结果
            
        Returns:
            质量指标对象
        """
        # 从检测结果中提取各维度分数
        grammar_score = detection_results.get('grammar', ([], {}))[1].get('grammar_score', 80.0)
        logic_score = detection_results.get('logic', ([], {}))[1].get('logic_score', 80.0)
        format_score = detection_results.get('format', ([], {}))[1].get('format_score', 80.0)
        factual_score = detection_results.get('factual', ([], {}))[1].get('factual_score', 80.0)
        academic_score = detection_results.get('academic', ([], {}))[1].get('academic_score', 80.0)
        
        # 计算加权总分
        overall_score = (
            grammar_score * self.weights['grammar'] +
            logic_score * self.weights['logic'] +
            format_score * self.weights['format'] +
            factual_score * self.weights['factual'] +
            academic_score * self.weights['academic']
        )
        
        return QualityMetrics(
            grammar_score=grammar_score,
            logic_score=logic_score,
            format_score=format_score,
            factual_score=factual_score,
            academic_score=academic_score,
            overall_score=round(overall_score, 1)
        )
    
    async def _generate_suggestions(self, 
                                  issues: List[QualityIssue], 
                                  metrics: QualityMetrics) -> List[str]:
        """
        生成改进建议
        
        Args:
            issues: 检测到的问题列表
            metrics: 质量指标
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        # 按问题类型统计
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        # 基于分数生成建议
        if metrics.grammar_score < 70:
            suggestions.append("语法表达需要显著改善，建议仔细检查句法结构和用词准确性")
        elif metrics.grammar_score < 85:
            suggestions.append("语法表达可以进一步优化，注意主谓一致和标点使用")
        
        if metrics.logic_score < 70:
            suggestions.append("逻辑结构需要重新组织，确保论述的连贯性和一致性")
        elif metrics.logic_score < 85:
            suggestions.append("逻辑表达可以更加清晰，注意段落之间的过渡")
        
        if metrics.format_score < 70:
            suggestions.append("格式规范性需要大幅改善，请按照学术写作标准调整")
        elif metrics.format_score < 85:
            suggestions.append("格式可以进一步规范化，注意标题层次和引用格式")
        
        if metrics.academic_score < 70:
            suggestions.append("学术表达需要提升，建议使用更规范的学术用语")
        elif metrics.academic_score < 85:
            suggestions.append("学术规范性可以改善，注意专业术语的准确使用")
        
        # 基于问题数量生成建议
        if issue_counts.get(IssueType.GRAMMAR_ERROR, 0) > 5:
            suggestions.append("发现较多语法错误，建议使用语法检查工具辅助修改")
        
        if issue_counts.get(IssueType.FORMAT_VIOLATION, 0) > 3:
            suggestions.append("格式问题较多，建议参考标准模板进行调整")
        
        return suggestions[:5]  # 最多返回5条建议
    
    async def _generate_auto_fixes(self, 
                                 issues: List[QualityIssue], 
                                 content: str) -> List[Dict[str, Any]]:
        """
        生成自动修复方案
        
        Args:
            issues: 检测到的问题列表
            content: 原始内容
            
        Returns:
            自动修复方案列表
        """
        auto_fixes = []
        
        for issue in issues:
            if issue.auto_fixable and issue.confidence > 0.7:
                # 提取问题位置的文本
                start_pos = max(0, issue.position - 5)
                end_pos = min(len(content), issue.position + (issue.length or 10) + 5)
                context = content[start_pos:end_pos]
                
                fix_info = {
                    'issue_id': f"{issue.issue_type.value}_{issue.position}",
                    'position': issue.position,
                    'length': getattr(issue, 'length', 1),
                    'issue_type': issue.issue_type.value,
                    'description': issue.description,
                    'original_text': context,
                    'suggested_fix': issue.suggestion,
                    'confidence': issue.confidence,
                    'auto_apply': issue.confidence > 0.9
                }
                auto_fixes.append(fix_info)
        
        return auto_fixes
    
    def _determine_status(self, overall_score: float, issues: List[QualityIssue]) -> str:
        """
        确定检测状态
        
        Args:
            overall_score: 总体分数
            issues: 问题列表
            
        Returns:
            状态字符串
        """
        # 检查是否有严重问题
        critical_issues = [issue for issue in issues if issue.severity == IssueSeverity.CRITICAL]
        if critical_issues:
            return "critical_issues"
        
        # 基于分数确定状态
        if overall_score >= settings.AUTO_APPROVAL_THRESHOLD:
            return "pass"
        elif overall_score >= settings.HUMAN_REVIEW_THRESHOLD:
            return "needs_review"
        else:
            return "needs_major_revision"

class GrammarDetector:
    """语法检测器"""
    
    def __init__(self):
        """初始化语法检测器"""
        # 加载jieba分词器
        jieba.initialize()
        logger.debug("语法检测器初始化完成")
    
    async def detect(self, content: str, content_type: str) -> Tuple[List[QualityIssue], Dict[str, float]]:
        """
        检测语法错误
        
        Args:
            content: 待检测内容
            content_type: 内容类型
            
        Returns:
            (问题列表, 指标字典)
        """
        issues = []
        
        # 分句处理
        sentences = self._split_sentences(content)
        total_sentences = len(sentences)
        
        position = 0
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                position += len(sentence)
                continue
            
            # 检查句子语法
            sentence_issues = await self._check_sentence_grammar(sentence, position)
            issues.extend(sentence_issues)
            
            position += len(sentence)
        
        # 计算语法分数
        grammar_score = self._calculate_grammar_score(issues, total_sentences)
        
        metrics = {
            'grammar_score': grammar_score,
            'sentences_count': total_sentences,
            'issues_count': len(issues)
        }
        
        return issues, metrics
    
    def _split_sentences(self, content: str) -> List[str]:
        """分句"""
        # 按句号、问号、感叹号分句
        sentences = re.split(r'[。？！]', content)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _check_sentence_grammar(self, sentence: str, start_pos: int) -> List[QualityIssue]:
        """检查单个句子的语法"""
        issues = []
        
        # 句子长度检查
        if len(sentence) > 80:
            issues.append(QualityIssue(
                issue_type=IssueType.GRAMMAR_ERROR,
                severity=IssueSeverity.MEDIUM,
                position=start_pos,
                length=len(sentence),
                description="句子过长，建议拆分为多个短句",
                suggestion="将长句拆分为2-3个短句，提高可读性",
                auto_fixable=False,
                confidence=0.8
            ))
        
        # 标点符号检查
        comma_count = sentence.count('，')
        if comma_count > 4:
            issues.append(QualityIssue(
                issue_type=IssueType.GRAMMAR_ERROR,
                severity=IssueSeverity.LOW,
                position=start_pos,
                length=len(sentence),
                description="逗号使用过多",
                suggestion="减少逗号使用，适当使用分号或句号",
                auto_fixable=False,
                confidence=0.7
            ))
        
        # 词性分析
        words = list(pseg.cut(sentence))
        
        # 检查主谓结构
        has_noun = any(flag.startswith('n') for word, flag in words)
        has_verb = any(flag.startswith('v') for word, flag in words)
        
        if not has_noun or not has_verb:
            issues.append(QualityIssue(
                issue_type=IssueType.GRAMMAR_ERROR,
                severity=IssueSeverity.MEDIUM,
                position=start_pos,
                length=len(sentence),
                description="句子缺少主语或谓语",
                suggestion="补充完整的主谓结构",
                auto_fixable=False,
                confidence=0.6
            ))
        
        return issues
    
    def _calculate_grammar_score(self, issues: List[QualityIssue], total_sentences: int) -> float:
        """计算语法分数"""
        if total_sentences == 0:
            return 80.0
        
        base_score = 100.0
        
        # 根据问题严重程度扣分
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 15.0
            elif issue.severity == IssueSeverity.HIGH:
                base_score -= 10.0
            elif issue.severity == IssueSeverity.MEDIUM:
                base_score -= 5.0
            else:
                base_score -= 2.0
        
        return max(0.0, min(100.0, base_score))

class LogicDetector:
    """逻辑一致性检测器"""
    
    async def detect(self, content: str, content_type: str) -> Tuple[List[QualityIssue], Dict[str, float]]:
        """检测逻辑一致性"""
        issues = []
        
        # 检查时间逻辑
        time_issues = await self._check_temporal_logic(content)
        issues.extend(time_issues)
        
        # 检查因果关系
        causal_issues = await self._check_causal_logic(content)
        issues.extend(causal_issues)
        
        # 检查论述连贯性
        coherence_issues = await self._check_coherence(content)
        issues.extend(coherence_issues)
        
        # 计算逻辑分数
        logic_score = self._calculate_logic_score(issues)
        
        metrics = {
            'logic_score': logic_score,
            'time_issues': len(time_issues),
            'causal_issues': len(causal_issues),
            'coherence_issues': len(coherence_issues)
        }
        
        return issues, metrics
    
    async def _check_temporal_logic(self, content: str) -> List[QualityIssue]:
        """检查时间逻辑"""
        issues = []
        
        # 提取时间表达
        time_patterns = [
            r'\d{4}年', r'\d{1,2}月', r'\d{1,2}日',
            r'(明|清|唐|宋|元|汉)朝', r'(初|中|末)期',
            r'(公元前|公元)\d+年'
        ]
        
        time_mentions = []
        for pattern in time_patterns:
            for match in re.finditer(pattern, content):
                time_mentions.append((match.start(), match.group(), match.end()))
        
        # 简化的时间冲突检查
        if len(time_mentions) > 1:
            # 这里可以实现更复杂的时间逻辑检查
            pass
        
        return issues
    
    async def _check_causal_logic(self, content: str) -> List[QualityIssue]:
        """检查因果关系逻辑"""
        issues = []
        
        # 查找因果关系词
        causal_patterns = [
            r'因为.*所以', r'由于.*因此', r'既然.*就',
            r'导致', r'造成', r'引起', r'产生'
        ]
        
        for pattern in causal_patterns:
            for match in re.finditer(pattern, content):
                # 简化的因果逻辑检查
                pass
        
        return issues
    
    async def _check_coherence(self, content: str) -> List[QualityIssue]:
        """检查论述连贯性"""
        issues = []
        
        # 分段检查
        paragraphs = content.split('\n\n')
        
        if len(paragraphs) > 1:
            # 检查段落间的连接
            for i in range(len(paragraphs) - 1):
                current_para = paragraphs[i].strip()
                next_para = paragraphs[i + 1].strip()
                
                if current_para and next_para:
                    # 检查段落间是否有适当的过渡
                    transition_words = ['然而', '但是', '因此', '所以', '同时', '另外', '此外']
                    has_transition = any(word in next_para[:20] for word in transition_words)
                    
                    if not has_transition and len(paragraphs) > 3:
                        para_start = content.find(next_para)
                        issues.append(QualityIssue(
                            issue_type=IssueType.LOGIC_INCONSISTENCY,
                            severity=IssueSeverity.LOW,
                            position=para_start,
                            length=20,
                            description="段落间缺少过渡词语",
                            suggestion="添加适当的过渡词语增强连贯性",
                            auto_fixable=False,
                            confidence=0.6
                        ))
        
        return issues
    
    def _calculate_logic_score(self, issues: List[QualityIssue]) -> float:
        """计算逻辑分数"""
        base_score = 90.0
        
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 20.0
            elif issue.severity == IssueSeverity.HIGH:
                base_score -= 10.0
            elif issue.severity == IssueSeverity.MEDIUM:
                base_score -= 5.0
            else:
                base_score -= 2.0
        
        return max(0.0, min(100.0, base_score))

class FormatDetector:
    """格式检测器"""
    
    async def detect(self, content: str, content_type: str) -> Tuple[List[QualityIssue], Dict[str, float]]:
        """检测格式规范性"""
        issues = []
        
        # 检查标题格式
        title_issues = await self._check_title_format(content)
        issues.extend(title_issues)
        
        # 检查段落格式
        paragraph_issues = await self._check_paragraph_format(content)
        issues.extend(paragraph_issues)
        
        # 检查标点符号
        punctuation_issues = await self._check_punctuation(content)
        issues.extend(punctuation_issues)
        
        # 计算格式分数
        format_score = self._calculate_format_score(issues)
        
        metrics = {
            'format_score': format_score,
            'title_issues': len(title_issues),
            'paragraph_issues': len(paragraph_issues),
            'punctuation_issues': len(punctuation_issues)
        }
        
        return issues, metrics
    
    async def _check_title_format(self, content: str) -> List[QualityIssue]:
        """检查标题格式"""
        issues = []
        
        # 检查是否有标题
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        if first_line and len(first_line) < 5:
            issues.append(QualityIssue(
                issue_type=IssueType.FORMAT_VIOLATION,
                severity=IssueSeverity.LOW,
                position=0,
                length=len(first_line),
                description="标题过短",
                suggestion="标题应当简洁明确地概括主要内容",
                auto_fixable=False,
                confidence=0.7
            ))
        
        return issues
    
    async def _check_paragraph_format(self, content: str) -> List[QualityIssue]:
        """检查段落格式"""
        issues = []
        
        paragraphs = content.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            
            # 检查段落长度
            if len(para) > 500:
                para_start = content.find(para)
                issues.append(QualityIssue(
                    issue_type=IssueType.FORMAT_VIOLATION,
                    severity=IssueSeverity.MEDIUM,
                    position=para_start,
                    length=len(para),
                    description="段落过长",
                    suggestion="将长段落拆分为多个短段落",
                    auto_fixable=False,
                    confidence=0.8
                ))
        
        return issues
    
    async def _check_punctuation(self, content: str) -> List[QualityIssue]:
        """检查标点符号"""
        issues = []
        
        # 检查重复标点
        duplicate_patterns = [
            (r'，{2,}', '重复逗号'),
            (r'。{2,}', '重复句号'),
            (r'？{2,}', '重复问号'),
            (r'！{2,}', '重复感叹号')
        ]
        
        for pattern, description in duplicate_patterns:
            for match in re.finditer(pattern, content):
                issues.append(QualityIssue(
                    issue_type=IssueType.FORMAT_VIOLATION,
                    severity=IssueSeverity.LOW,
                    position=match.start(),
                    length=match.end() - match.start(),
                    description=description,
                    suggestion="删除多余的标点符号",
                    auto_fixable=True,
                    confidence=0.9
                ))
        
        return issues
    
    def _calculate_format_score(self, issues: List[QualityIssue]) -> float:
        """计算格式分数"""
        base_score = 95.0
        
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 15.0
            elif issue.severity == IssueSeverity.HIGH:
                base_score -= 8.0
            elif issue.severity == IssueSeverity.MEDIUM:
                base_score -= 4.0
            else:
                base_score -= 1.0
        
        return max(0.0, min(100.0, base_score))

class FactualDetector:
    """事实准确性检测器"""
    
    async def detect(self, content: str, content_type: str) -> Tuple[List[QualityIssue], Dict[str, float]]:
        """检测事实准确性"""
        issues = []
        
        # 这里可以实现更复杂的事实检查逻辑
        # 例如与历史数据库对比、检查数字的合理性等
        
        # 简化的数字合理性检查
        number_issues = await self._check_numbers(content)
        issues.extend(number_issues)
        
        # 计算事实分数
        factual_score = self._calculate_factual_score(issues)
        
        metrics = {
            'factual_score': factual_score,
            'number_issues': len(number_issues)
        }
        
        return issues, metrics
    
    async def _check_numbers(self, content: str) -> List[QualityIssue]:
        """检查数字的合理性"""
        issues = []
        
        # 查找年份
        year_pattern = r'\b(\d{1,4})年\b'
        for match in re.finditer(year_pattern, content):
            year = int(match.group(1))
            if year > 2024 or year < 1:
                issues.append(QualityIssue(
                    issue_type=IssueType.FACTUAL_ERROR,
                    severity=IssueSeverity.HIGH,
                    position=match.start(),
                    length=match.end() - match.start(),
                    description=f"年份{year}可能不正确",
                    suggestion="请核实年份的准确性",
                    auto_fixable=False,
                    confidence=0.8
                ))
        
        return issues
    
    def _calculate_factual_score(self, issues: List[QualityIssue]) -> float:
        """计算事实分数"""
        base_score = 85.0
        
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 25.0
            elif issue.severity == IssueSeverity.HIGH:
                base_score -= 15.0
            elif issue.severity == IssueSeverity.MEDIUM:
                base_score -= 8.0
            else:
                base_score -= 3.0
        
        return max(0.0, min(100.0, base_score))

class AcademicDetector:
    """学术标准检测器"""
    
    async def detect(self, content: str, content_type: str) -> Tuple[List[QualityIssue], Dict[str, float]]:
        """检测学术写作标准"""
        issues = []
        
        # 检查学术用词
        vocabulary_issues = await self._check_academic_vocabulary(content)
        issues.extend(vocabulary_issues)
        
        # 检查引用格式
        citation_issues = await self._check_citations(content)
        issues.extend(citation_issues)
        
        # 计算学术分数
        academic_score = self._calculate_academic_score(issues)
        
        metrics = {
            'academic_score': academic_score,
            'vocabulary_issues': len(vocabulary_issues),
            'citation_issues': len(citation_issues)
        }
        
        return issues, metrics
    
    async def _check_academic_vocabulary(self, content: str) -> List[QualityIssue]:
        """检查学术用词"""
        issues = []
        
        # 检查口语化表达
        colloquial_words = ['挺好的', '很棒', '超级', '特别地', '非常地']
        
        for word in colloquial_words:
            for match in re.finditer(re.escape(word), content):
                issues.append(QualityIssue(
                    issue_type=IssueType.ACADEMIC_STANDARD,
                    severity=IssueSeverity.LOW,
                    position=match.start(),
                    length=len(word),
                    description=f"'{word}'过于口语化",
                    suggestion="使用更正式的学术表达",
                    auto_fixable=False,
                    confidence=0.8
                ))
        
        return issues
    
    async def _check_citations(self, content: str) -> List[QualityIssue]:
        """检查引用格式"""
        issues = []
        
        # 简化的引用检查
        # 这里可以实现更复杂的引用格式检查
        
        return issues
    
    def _calculate_academic_score(self, issues: List[QualityIssue]) -> float:
        """计算学术分数"""
        base_score = 88.0
        
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 20.0
            elif issue.severity == IssueSeverity.HIGH:
                base_score -= 12.0
            elif issue.severity == IssueSeverity.MEDIUM:
                base_score -= 6.0
            else:
                base_score -= 2.0
        
        return max(0.0, min(100.0, base_score))