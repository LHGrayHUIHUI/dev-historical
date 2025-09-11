"""
文本内容分析器

基于NLP技术和机器学习模型的文本内容审核
支持敏感词检测、情感分析、政治敏感内容识别等功能
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional, Union, Set
import asyncio
from dataclasses import dataclass

from .base_analyzer import BaseAnalyzer, AnalysisResult, ViolationDetail, ViolationType, AnalysisStatus

logger = logging.getLogger(__name__)


@dataclass 
class TextAnalysisMetadata:
    """文本分析元数据"""
    text_length: int
    word_count: int
    sentence_count: int
    language: str
    encoding: str
    keywords: List[str]
    sentiment_score: float  # -1.0 到 1.0


class SensitiveWordDetector:
    """敏感词检测器"""
    
    def __init__(self, word_lists: Dict[str, List[str]] = None):
        """
        初始化敏感词检测器
        
        Args:
            word_lists: 按类别分组的敏感词列表
        """
        self.word_lists = word_lists or self._load_default_words()
        self.compiled_patterns = {}
        self._compile_patterns()
    
    def _load_default_words(self) -> Dict[str, List[str]]:
        """加载默认敏感词库"""
        return {
            ViolationType.POLITICS: [
                "政治敏感词示例1", "政治敏感词示例2", "敏感政治词汇"
            ],
            ViolationType.VIOLENCE: [
                "暴力词汇1", "血腥词汇", "恐怖主义", "杀害", "暴力行为"
            ],
            ViolationType.PORNOGRAPHY: [
                "色情词汇1", "成人内容", "不当词汇"
            ],
            ViolationType.SPAM: [
                "广告词汇", "推广", "营销", "免费赠送", "限时优惠"
            ],
            ViolationType.HATE: [
                "仇恨言论", "歧视词汇", "种族主义"
            ],
            ViolationType.DRUGS: [
                "毒品名称", "非法药物", "吸毒"
            ],
            ViolationType.GAMBLING: [
                "赌博词汇", "博彩", "彩票诈骗"
            ]
        }
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        for category, words in self.word_lists.items():
            # 创建不区分大小写的正则模式
            pattern = '|'.join(re.escape(word) for word in words)
            self.compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def detect(self, text: str) -> List[ViolationDetail]:
        """
        检测文本中的敏感词
        
        Args:
            text: 待检测文本
            
        Returns:
            List[ViolationDetail]: 违规详情列表
        """
        violations = []
        
        for category, pattern in self.compiled_patterns.items():
            matches = list(pattern.finditer(text))
            
            if matches:
                # 计算该类别的置信度
                confidence = min(1.0, len(matches) * 0.3)
                
                # 提取匹配位置信息
                locations = [{"start": m.start(), "end": m.end(), "text": m.group()} 
                           for m in matches[:5]]  # 最多记录5个位置
                
                violation = ViolationDetail(
                    type=category,
                    confidence=confidence,
                    description=f"检测到{len(matches)}个{category}相关敏感词",
                    evidence={"matches": [m.group() for m in matches[:3]]},
                    location={"positions": locations}
                )
                violations.append(violation)
        
        return violations


class TextPatternDetector:
    """文本模式检测器"""
    
    def __init__(self):
        """初始化模式检测器"""
        self.patterns = {
            ViolationType.FRAUD: [
                r'(?i)(免费|赠送).{0,10}(获得|领取).{0,20}(点击|联系)',
                r'(?i)(投资|理财).{0,15}(高收益|稳赚|保本)',
                r'(?i)(兼职|副业).{0,10}(日赚|月入).{0,10}\d+',
            ],
            ViolationType.SPAM: [
                r'(?i)(加微信|联系QQ).{0,15}[\d\-\+\(\)]{6,}',
                r'(?i)(优惠|折扣|特价).{0,20}(限时|今日|立即)',
                r'(?i)(点击|复制|转发).{0,10}(链接|网址)',
            ],
            ViolationType.PRIVACY: [
                r'\b\d{3}-\d{3}-\d{4}\b',  # 电话号码模式
                r'\b\d{15,18}\b',          # 身份证号码模式  
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
            ]
        }
        
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(p) for p in pattern_list]
    
    def detect(self, text: str) -> List[ViolationDetail]:
        """
        检测文本中的可疑模式
        
        Args:
            text: 待检测文本
            
        Returns:
            List[ViolationDetail]: 违规详情列表
        """
        violations = []
        
        for category, patterns in self.compiled_patterns.items():
            matches = []
            
            for pattern in patterns:
                pattern_matches = list(pattern.finditer(text))
                matches.extend(pattern_matches)
            
            if matches:
                confidence = min(0.9, len(matches) * 0.2)
                
                locations = [{"start": m.start(), "end": m.end(), "pattern": m.pattern} 
                           for m in matches[:3]]
                
                violation = ViolationDetail(
                    type=category,
                    confidence=confidence,
                    description=f"检测到{len(matches)}个{category}相关可疑模式",
                    evidence={"pattern_matches": len(matches)},
                    location={"positions": locations}
                )
                violations.append(violation)
        
        return violations


class SentimentAnalyzer:
    """情感分析器（简化版）"""
    
    def __init__(self):
        """初始化情感分析器"""
        # 简化的情感词典
        self.positive_words = {'好', '棒', '优秀', '完美', '喜欢', '爱', '快乐', '幸福', '满意'}
        self.negative_words = {'坏', '差', '糟糕', '恶心', '讨厌', '恨', '愤怒', '失望', '痛苦'}
        self.extreme_negative = {'杀', '死', '恨死', '去死', '该死'}
    
    def analyze(self, text: str) -> float:
        """
        分析文本情感
        
        Args:
            text: 待分析文本
            
        Returns:
            float: 情感分数 (-1.0 到 1.0)
        """
        # 简化的情感分析实现
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        extreme_count = sum(1 for word in self.extreme_negative if word in text)
        
        # 计算总分
        total_score = positive_count - negative_count - (extreme_count * 2)
        total_words = positive_count + negative_count + extreme_count
        
        if total_words == 0:
            return 0.0
        
        # 归一化到 -1.0 到 1.0
        normalized_score = total_score / max(total_words, 1)
        return max(-1.0, min(1.0, normalized_score))


class TextAnalyzer(BaseAnalyzer):
    """
    文本内容分析器
    
    提供全面的文本内容审核功能，包括：
    - 敏感词检测
    - 文本模式识别  
    - 情感分析
    - 语言检测
    - 内容质量评估
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化文本分析器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        
        self.max_text_length = self.config.get("max_text_length", 50000)
        self.min_text_length = self.config.get("min_text_length", 1)
        
        # 初始化检测器
        self.sensitive_word_detector = SensitiveWordDetector(
            self.config.get("custom_word_lists")
        )
        self.pattern_detector = TextPatternDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        logger.info("文本分析器初始化完成")
    
    def get_supported_types(self) -> List[str]:
        """获取支持的内容类型"""
        return [
            "text/plain",
            "text/html", 
            "text/markdown",
            "application/json",
            "text/csv"
        ]
    
    async def analyze(self, content: Union[str, bytes], metadata: Dict[str, Any] = None) -> AnalysisResult:
        """
        分析文本内容
        
        Args:
            content: 文本内容
            metadata: 元数据信息
            
        Returns:
            AnalysisResult: 分析结果
        """
        start_time = time.time()
        
        try:
            # 内容预处理
            text = await self._preprocess_content(content)
            if not text:
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "文本内容为空或无法处理"
                )
            
            # 内容验证
            if not self.validate_content(text, self.max_text_length):
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    f"文本长度超过限制 {self.max_text_length} 字符"
                )
            
            # 并发执行多种检测
            violations = []
            
            # 敏感词检测
            sensitive_violations = await asyncio.get_event_loop().run_in_executor(
                None, self.sensitive_word_detector.detect, text
            )
            violations.extend(sensitive_violations)
            
            # 模式检测
            pattern_violations = await asyncio.get_event_loop().run_in_executor(
                None, self.pattern_detector.detect, text
            )
            violations.extend(pattern_violations)
            
            # 情感分析
            sentiment_score = await asyncio.get_event_loop().run_in_executor(
                None, self.sentiment_analyzer.analyze, text
            )
            
            # 检查极端负面情感
            if sentiment_score < -0.7:
                violation = ViolationDetail(
                    type=ViolationType.HATE,
                    confidence=abs(sentiment_score),
                    description="检测到极端负面情感内容",
                    evidence={"sentiment_score": sentiment_score}
                )
                violations.append(violation)
            
            # 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(violations)
            
            # 生成分析元数据
            analysis_metadata = self._generate_metadata(text, sentiment_score, metadata)
            
            processing_time = time.time() - start_time
            
            return self.create_success_result(
                confidence=overall_confidence,
                violations=violations,
                processing_time=processing_time,
                metadata=analysis_metadata.__dict__
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"文本分析失败: {str(e)}")
            return self.create_error_result(
                AnalysisStatus.FAILED,
                f"分析过程中发生错误: {str(e)}",
                processing_time
            )
    
    async def _preprocess_content(self, content: Union[str, bytes]) -> str:
        """
        预处理内容
        
        Args:
            content: 原始内容
            
        Returns:
            str: 处理后的文本
        """
        if isinstance(content, bytes):
            # 尝试多种编码解析
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("无法解码字节内容")
        else:
            text = content
        
        # 清理文本
        text = text.strip()
        
        # 移除过多的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # HTML内容清理（如果需要）
        if '<html>' in text.lower() or '<body>' in text.lower():
            text = self._clean_html(text)
        
        return text
    
    def _clean_html(self, html_text: str) -> str:
        """
        清理HTML标签
        
        Args:
            html_text: HTML文本
            
        Returns:
            str: 清理后的纯文本
        """
        # 简化的HTML清理
        import re
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', html_text)
        
        # 解码HTML实体
        html_entities = {
            '&amp;': '&',
            '&lt;': '<', 
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'",
            '&nbsp;': ' '
        }
        
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        
        return text
    
    def _calculate_overall_confidence(self, violations: List[ViolationDetail]) -> float:
        """
        计算整体违规置信度
        
        Args:
            violations: 违规详情列表
            
        Returns:
            float: 整体置信度
        """
        if not violations:
            return 0.0
        
        # 使用加权平均，给严重违规更高权重
        total_weight = 0.0
        weighted_sum = 0.0
        
        weight_map = {
            ViolationType.POLITICS: 1.0,
            ViolationType.VIOLENCE: 1.0, 
            ViolationType.PORNOGRAPHY: 1.0,
            ViolationType.HATE: 0.9,
            ViolationType.DRUGS: 0.9,
            ViolationType.GAMBLING: 0.8,
            ViolationType.FRAUD: 0.8,
            ViolationType.SPAM: 0.6,
            ViolationType.PRIVACY: 0.7
        }
        
        for violation in violations:
            weight = weight_map.get(violation.type, 0.5)
            weighted_sum += violation.confidence * weight
            total_weight += weight
        
        return min(1.0, weighted_sum / total_weight) if total_weight > 0 else 0.0
    
    def _generate_metadata(self, text: str, sentiment_score: float, input_metadata: Dict[str, Any] = None) -> TextAnalysisMetadata:
        """
        生成分析元数据
        
        Args:
            text: 文本内容
            sentiment_score: 情感分数
            input_metadata: 输入元数据
            
        Returns:
            TextAnalysisMetadata: 分析元数据
        """
        # 基础统计
        text_length = len(text)
        word_count = len(text.split())
        sentence_count = len([s for s in re.split(r'[.!?。！？]', text) if s.strip()])
        
        # 语言检测（简化版）
        language = self._detect_language(text)
        
        # 提取关键词
        keywords = self.extract_keywords(text, max_keywords=10)
        
        # 编码检测
        encoding = input_metadata.get('encoding', 'utf-8') if input_metadata else 'utf-8'
        
        return TextAnalysisMetadata(
            text_length=text_length,
            word_count=word_count,
            sentence_count=sentence_count,
            language=language,
            encoding=encoding,
            keywords=keywords,
            sentiment_score=sentiment_score
        )
    
    def _detect_language(self, text: str) -> str:
        """
        检测文本语言（简化版）
        
        Args:
            text: 文本内容
            
        Returns:
            str: 语言代码
        """
        # 简化的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return 'zh-cn'
        else:
            return 'en'
    
    def batch_analyze_texts(self, texts: List[str]) -> List[AnalysisResult]:
        """
        批量分析文本
        
        Args:
            texts: 文本列表
            
        Returns:
            List[AnalysisResult]: 分析结果列表
        """
        results = []
        for text in texts:
            try:
                result = asyncio.run(self.analyze(text))
                results.append(result)
            except Exception as e:
                error_result = self.create_error_result(
                    AnalysisStatus.FAILED,
                    f"批量分析失败: {str(e)}"
                )
                results.append(error_result)
        
        return results
    
    def update_sensitive_words(self, category: str, words: List[str]):
        """
        更新敏感词库
        
        Args:
            category: 词汇类别
            words: 新增词汇列表
        """
        if category not in self.sensitive_word_detector.word_lists:
            self.sensitive_word_detector.word_lists[category] = []
        
        self.sensitive_word_detector.word_lists[category].extend(words)
        self.sensitive_word_detector._compile_patterns()
        
        logger.info(f"已更新 {category} 类别敏感词库，新增 {len(words)} 个词汇")