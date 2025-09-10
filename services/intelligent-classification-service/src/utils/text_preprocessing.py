"""
智能分类服务文本预处理模块
专门针对古代中文历史文献的文本清洗和预处理
优化对古汉语和繁简体文本的处理
"""

import re
import jieba
import unicodedata
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from ..config.settings import settings


@dataclass
class PreprocessingConfig:
    """文本预处理配置类"""
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_whitespace: bool = True
    lowercase: bool = True
    remove_stopwords: bool = True
    min_word_length: int = 2
    max_word_length: int = 50
    traditional_to_simplified: bool = True
    remove_html_tags: bool = True


class ChineseTextPreprocessor:
    """中文历史文本预处理器
    
    专门为古代中文文献设计的文本预处理工具
    支持繁简转换、古汉语处理、文言文分词优化
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # 初始化jieba分词器，添加历史文献专用词典
        self._init_jieba_dict()
        
        # 预编译正则表达式提升性能
        self._compile_regex_patterns()
        
        # 初始化繁简转换表
        self._init_traditional_simplified_mapping()
    
    def _init_jieba_dict(self):
        """初始化jieba分词器的历史文献专用词典"""
        # 历史人名词典
        historical_names = [
            "孔子", "老子", "孟子", "荀子", "庄子", "墨子", "韩非子",
            "秦始皇", "汉武帝", "唐太宗", "宋太祖", "康熙", "乾隆",
            "司马迁", "班固", "范仲淹", "欧阳修", "苏轼", "朱熹"
        ]
        
        # 历史地名词典
        historical_places = [
            "长安", "洛阳", "开封", "临安", "大都", "应天府", "顺天府",
            "关中", "江南", "河北", "河南", "山东", "山西", "陕西"
        ]
        
        # 官职名词典
        official_titles = [
            "丞相", "太尉", "御史大夫", "三公", "九卿", "刺史", "太守",
            "县令", "县长", "主簿", "长史", "司马", "参军", "录事"
        ]
        
        # 文言文常用词
        classical_words = [
            "之乎者也", "焉哉", "矣矣", "甚矣", "何如", "如何", "奈何",
            "所以", "是以", "故曰", "然则", "既然", "虽然", "固然"
        ]
        
        # 将专用词典添加到jieba
        all_words = historical_names + historical_places + official_titles + classical_words
        for word in all_words:
            jieba.add_word(word, freq=1000, tag='nr')  # 设置高频率优先分词
    
    def _compile_regex_patterns(self):
        """预编译常用正则表达式模式"""
        self.html_pattern = re.compile(r'<[^<]+?>', re.IGNORECASE)
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.number_pattern = re.compile(r'\d+')
        self.punctuation_pattern = re.compile(r'[^\w\s]', re.UNICODE)
        self.whitespace_pattern = re.compile(r'\s+')
        
        # 古文特殊符号处理
        self.classical_punctuation = re.compile(r'[。，、；：！？""''（）【】《》〈〉「」『』]')
    
    def _init_traditional_simplified_mapping(self):
        """初始化繁简转换映射表"""
        # 常用繁简字对照表（简化版）
        self.trad_to_simp = {
            '學': '学', '國': '国', '會': '会', '來': '来', '時': '时',
            '長': '长', '書': '书', '門': '门', '開': '开', '關': '关',
            '無': '无', '與': '与', '為': '为', '說': '说', '這': '这',
            '個': '个', '們': '们', '經': '经', '過': '过', '對': '对',
            '應': '应', '現': '现', '發': '发', '點': '点', '進': '进',
            '種': '种', '樣': '样', '買': '买', '賣': '卖', '錢': '钱',
            '車': '车', '軍': '军', '農': '农', '業': '业', '産': '产',
            '質': '质', '體': '体', '義': '义', '華': '华', '實': '实'
        }
    
    def traditional_to_simplified(self, text: str) -> str:
        """繁体转简体"""
        if not self.config.traditional_to_simplified:
            return text
        
        result = text
        for trad, simp in self.trad_to_simp.items():
            result = result.replace(trad, simp)
        return result
    
    def remove_html_tags(self, text: str) -> str:
        """移除HTML标签"""
        if not self.config.remove_html_tags:
            return text
        
        text = self.html_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Unicode标准化处理"""
        # 标准化Unicode编码，处理古文字符编码问题
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def remove_numbers(self, text: str) -> str:
        """移除数字（可选保留古代纪年）"""
        if not self.config.remove_numbers:
            return text
        
        # 保留古代纪年格式
        ancient_years = re.compile(r'(公元前?\d+年|民国\d+年|康熙\d+年|乾隆\d+年|光绪\d+年|宣统\d+年)')
        preserved_years = ancient_years.findall(text)
        
        # 移除一般数字
        text = self.number_pattern.sub(' ', text)
        
        # 恢复古代纪年
        for year in preserved_years:
            text = year + ' ' + text
        
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """移除标点符号（保留古文必要标点）"""
        if not self.config.remove_punctuation:
            return text
        
        # 保留重要的古文标点
        important_classical = ['。', '，']
        text_chars = []
        
        for char in text:
            if self.classical_punctuation.match(char):
                if char in important_classical:
                    text_chars.append(char)
                else:
                    text_chars.append(' ')
            elif self.punctuation_pattern.match(char):
                text_chars.append(' ')
            else:
                text_chars.append(char)
        
        return ''.join(text_chars)
    
    def clean_whitespace(self, text: str) -> str:
        """清理多余空白字符"""
        if not self.config.remove_whitespace:
            return text
        
        text = self.whitespace_pattern.sub(' ', text)
        return text.strip()
    
    def segment_text(self, text: str) -> List[str]:
        """中文分词
        
        使用jieba进行中文分词，针对古文优化
        """
        # 使用精确模式进行分词，适合古文处理
        words = jieba.cut(text, cut_all=False)
        return list(words)
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """移除停用词"""
        if not self.config.remove_stopwords:
            return words
        
        stopwords_set = set(settings.chinese_stopwords)
        
        # 添加古文特有停用词
        classical_stopwords = {
            '之', '其', '而', '以', '于', '乎', '也', '矣', '焉', '哉',
            '者', '所', '为', '与', '则', '若', '如', '何', '故',
            '是', '非', '或', '且', '又', '乃', '即', '既', '遂'
        }
        stopwords_set.update(classical_stopwords)
        
        filtered_words = [
            word for word in words 
            if word not in stopwords_set and word.strip()
        ]
        
        return filtered_words
    
    def filter_word_length(self, words: List[str]) -> List[str]:
        """根据词长过滤单词"""
        filtered_words = [
            word for word in words
            if self.config.min_word_length <= len(word) <= self.config.max_word_length
        ]
        return filtered_words
    
    def preprocess(self, text: str, return_tokens: bool = True) -> str | List[str]:
        """
        完整的文本预处理流水线
        
        Args:
            text: 输入文本
            return_tokens: 是否返回分词结果，False则返回处理后的文本字符串
            
        Returns:
            处理后的文本或词汇列表
        """
        if not text or not text.strip():
            return [] if return_tokens else ""
        
        self.logger.debug(f"开始预处理文本，长度: {len(text)}")
        
        # 步骤1: Unicode标准化
        text = self.normalize_unicode(text)
        
        # 步骤2: 繁简转换
        text = self.traditional_to_simplified(text)
        
        # 步骤3: 移除HTML标签和特殊内容
        text = self.remove_html_tags(text)
        
        # 步骤4: 移除数字（可选）
        text = self.remove_numbers(text)
        
        # 步骤5: 处理标点符号
        text = self.remove_punctuation(text)
        
        # 步骤6: 清理空白字符
        text = self.clean_whitespace(text)
        
        # 步骤7: 转小写（对中文影响不大，主要处理英文）
        if self.config.lowercase:
            text = text.lower()
        
        if not return_tokens:
            return text
        
        # 步骤8: 中文分词
        words = self.segment_text(text)
        
        # 步骤9: 移除停用词
        words = self.remove_stopwords(words)
        
        # 步骤10: 词长过滤
        words = self.filter_word_length(words)
        
        self.logger.debug(f"预处理完成，生成词汇数: {len(words)}")
        
        return words
    
    def batch_preprocess(self, texts: List[str], return_tokens: bool = True) -> List[str] | List[List[str]]:
        """批量预处理文本"""
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                self.logger.info(f"批量预处理进度: {i}/{total}")
            
            result = self.preprocess(text, return_tokens)
            results.append(result)
        
        self.logger.info(f"批量预处理完成，处理文本数: {total}")
        return results
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """获取文本统计信息"""
        original_length = len(text)
        processed_tokens = self.preprocess(text, return_tokens=True)
        processed_text = self.preprocess(text, return_tokens=False)
        
        stats = {
            'original_length': original_length,
            'processed_length': len(processed_text),
            'token_count': len(processed_tokens),
            'unique_tokens': len(set(processed_tokens)),
            'compression_ratio': len(processed_text) / original_length if original_length > 0 else 0,
            'avg_token_length': sum(len(token) for token in processed_tokens) / len(processed_tokens) if processed_tokens else 0
        }
        
        return stats


# 默认预处理器实例
default_preprocessor = ChineseTextPreprocessor()


def preprocess_text(text: str, config: Optional[PreprocessingConfig] = None) -> List[str]:
    """便捷的文本预处理函数"""
    if config:
        preprocessor = ChineseTextPreprocessor(config)
    else:
        preprocessor = default_preprocessor
    
    return preprocessor.preprocess(text, return_tokens=True)


def preprocess_text_to_string(text: str, config: Optional[PreprocessingConfig] = None) -> str:
    """便捷的文本预处理函数，返回字符串"""
    if config:
        preprocessor = ChineseTextPreprocessor(config)
    else:
        preprocessor = default_preprocessor
    
    return preprocessor.preprocess(text, return_tokens=False)