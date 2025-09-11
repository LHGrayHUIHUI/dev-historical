"""
基础分析器抽象类

定义所有内容分析器的通用接口和基础功能
提供分析结果的标准化格式和错误处理机制
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
import hashlib

logger = logging.getLogger(__name__)


class AnalysisStatus(str, Enum):
    """分析状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


class ViolationType(str, Enum):
    """违规类型枚举"""
    POLITICS = "politics"      # 政治敏感
    VIOLENCE = "violence"      # 暴力血腥
    PORNOGRAPHY = "pornography"  # 色情内容
    SPAM = "spam"             # 垃圾广告
    HATE = "hate"             # 仇恨言论
    DRUGS = "drugs"           # 毒品相关
    GAMBLING = "gambling"     # 赌博相关
    FRAUD = "fraud"           # 诈骗信息
    COPYRIGHT = "copyright"    # 版权违规
    PRIVACY = "privacy"       # 隐私泄露


@dataclass
class ViolationDetail:
    """违规详情"""
    type: ViolationType
    confidence: float
    description: str
    evidence: Optional[Dict[str, Any]] = None
    location: Optional[Dict[str, Any]] = None  # 位置信息(文本位置、图像区域等)


@dataclass
class AnalysisResult:
    """分析结果数据结构"""
    status: AnalysisStatus
    confidence: float  # 整体置信度 0.0-1.0
    is_violation: bool  # 是否违规
    risk_level: str  # low, medium, high, critical
    violations: List[ViolationDetail]  # 违规详情列表
    processing_time: float  # 处理时间(秒)
    analyzer_version: str  # 分析器版本
    metadata: Dict[str, Any]  # 额外元数据
    error_message: Optional[str] = None


class BaseAnalyzer(ABC):
    """
    内容分析器基类
    
    定义所有分析器必须实现的接口方法
    提供通用的结果处理和错误处理功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分析器
        
        Args:
            config: 分析器配置参数
        """
        self.config = config or {}
        self.version = "1.0.0"
        self.name = self.__class__.__name__
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.timeout = self.config.get("timeout", 30.0)
        
        logger.info(f"初始化 {self.name} 分析器，版本: {self.version}")
    
    @abstractmethod
    async def analyze(self, content: Union[str, bytes], metadata: Dict[str, Any] = None) -> AnalysisResult:
        """
        分析内容的抽象方法
        
        Args:
            content: 要分析的内容(文本或二进制数据)
            metadata: 内容元数据
            
        Returns:
            AnalysisResult: 分析结果
        """
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """
        获取支持的内容类型列表
        
        Returns:
            List[str]: 支持的MIME类型列表
        """
        pass
    
    def is_supported(self, content_type: str) -> bool:
        """
        检查是否支持指定的内容类型
        
        Args:
            content_type: MIME类型
            
        Returns:
            bool: 是否支持
        """
        supported_types = self.get_supported_types()
        return content_type in supported_types
    
    def calculate_content_hash(self, content: Union[str, bytes]) -> str:
        """
        计算内容哈希值
        
        Args:
            content: 内容数据
            
        Returns:
            str: SHA256哈希值
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def determine_risk_level(self, confidence: float, violations: List[ViolationDetail]) -> str:
        """
        根据置信度和违规类型确定风险等级
        
        Args:
            confidence: 整体置信度
            violations: 违规详情列表
            
        Returns:
            str: 风险等级 (low, medium, high, critical)
        """
        if not violations:
            return "low"
        
        # 检查是否有严重违规类型
        critical_types = {ViolationType.POLITICS, ViolationType.VIOLENCE, ViolationType.PORNOGRAPHY}
        has_critical = any(v.type in critical_types for v in violations)
        
        if has_critical and confidence > 0.8:
            return "critical"
        elif has_critical or confidence > 0.7:
            return "high" 
        elif confidence > 0.5:
            return "medium"
        else:
            return "low"
    
    def create_success_result(
        self, 
        confidence: float,
        violations: List[ViolationDetail],
        processing_time: float,
        metadata: Dict[str, Any] = None
    ) -> AnalysisResult:
        """
        创建成功分析结果
        
        Args:
            confidence: 置信度
            violations: 违规详情
            processing_time: 处理时间
            metadata: 元数据
            
        Returns:
            AnalysisResult: 分析结果
        """
        risk_level = self.determine_risk_level(confidence, violations)
        is_violation = confidence >= self.confidence_threshold and len(violations) > 0
        
        return AnalysisResult(
            status=AnalysisStatus.SUCCESS,
            confidence=confidence,
            is_violation=is_violation,
            risk_level=risk_level,
            violations=violations,
            processing_time=processing_time,
            analyzer_version=self.version,
            metadata=metadata or {}
        )
    
    def create_error_result(
        self, 
        status: AnalysisStatus,
        error_message: str,
        processing_time: float = 0.0
    ) -> AnalysisResult:
        """
        创建错误分析结果
        
        Args:
            status: 错误状态
            error_message: 错误消息
            processing_time: 处理时间
            
        Returns:
            AnalysisResult: 错误结果
        """
        return AnalysisResult(
            status=status,
            confidence=0.0,
            is_violation=False,
            risk_level="low",
            violations=[],
            processing_time=processing_time,
            analyzer_version=self.version,
            metadata={},
            error_message=error_message
        )
    
    async def analyze_with_timeout(self, content: Union[str, bytes], metadata: Dict[str, Any] = None) -> AnalysisResult:
        """
        带超时的分析方法
        
        Args:
            content: 要分析的内容
            metadata: 内容元数据
            
        Returns:
            AnalysisResult: 分析结果
        """
        import asyncio
        
        start_time = time.time()
        
        try:
            # 使用asyncio.wait_for设置超时
            result = await asyncio.wait_for(
                self.analyze(content, metadata),
                timeout=self.timeout
            )
            
            # 确保处理时间正确
            if result.processing_time == 0.0:
                result.processing_time = time.time() - start_time
            
            return result
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.error(f"{self.name} 分析超时，耗时: {processing_time:.2f}秒")
            return self.create_error_result(
                AnalysisStatus.TIMEOUT,
                f"分析超时，超过 {self.timeout} 秒限制",
                processing_time
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"{self.name} 分析失败: {str(e)}")
            return self.create_error_result(
                AnalysisStatus.FAILED,
                f"分析失败: {str(e)}",
                processing_time
            )
    
    def validate_content(self, content: Union[str, bytes], max_size: int = None) -> bool:
        """
        验证内容格式和大小
        
        Args:
            content: 内容数据
            max_size: 最大大小限制(字节)
            
        Returns:
            bool: 是否有效
        """
        if content is None:
            return False
        
        if max_size:
            content_size = len(content) if isinstance(content, (str, bytes)) else 0
            if content_size > max_size:
                logger.warning(f"内容大小 {content_size} 超过限制 {max_size}")
                return False
        
        return True
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        从文本中提取关键词
        
        Args:
            text: 文本内容
            max_keywords: 最大关键词数量
            
        Returns:
            List[str]: 关键词列表
        """
        # 简单的关键词提取实现
        # 实际项目中应该使用更复杂的NLP算法
        import re
        
        # 清理和分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词(简化版)
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 统计词频并排序
        from collections import Counter
        word_counts = Counter(keywords)
        
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """
        获取分析器信息
        
        Returns:
            Dict[str, Any]: 分析器信息
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_types": self.get_supported_types(),
            "confidence_threshold": self.confidence_threshold,
            "timeout": self.timeout,
            "config": self.config
        }