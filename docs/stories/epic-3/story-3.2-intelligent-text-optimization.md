# Story 3.2: 智能文本优化服务

## 基本信息
- **Epic**: Epic 3 - AI大模型服务和内容文本优化  
- **Story ID**: 3.2
- **优先级**: 高
- **预估工作量**: 2.5周
- **负责团队**: 后端开发团队 + NLP工程团队

## 用户故事

**作为** 内容编辑  
**我希望** 通过AI服务按照指定格式重新组织文本内容  
**以便于** 生成高质量的历史文本，提升内容的可读性和学术价值

## 需求描述

### 核心功能需求

1. **历史文本格式优化**
   - 按历史文档标准格式重新组织（时间-人物-地点-事件-感触）
   - 保持原意的基础上改善表达方式
   - 统一文档体例和用词规范
   - 增强文本的学术严谨性

2. **多模式文本重写**
   - **文本润色**: 优化语言表达，提升文本流畅度
   - **内容扩展**: 基于历史背景增加相关细节描述
   - **风格转换**: 转换为不同历史时期的文体风格
   - **现代化改写**: 将古文转换为现代汉语表达

3. **质量评估和版本管理**
   - 自动评估优化后文本的质量分数
   - 提供多个优化版本供选择对比
   - 记录优化历史和变更轨迹
   - 支持版本回滚和差异对比

4. **智能优化策略**
   - 基于文本类型自动选择优化策略
   - 学习用户偏好和反馈
   - 自适应调整优化参数
   - 提供优化建议和改进方向

5. **批量处理功能**
   - 支持大规模文档批量优化
   - 异步任务处理和进度跟踪
   - 优化结果统计和报告
   - 失败重试和错误处理

## 技术实现

### 核心技术栈

- **服务框架**: FastAPI + Python 3.11
- **AI集成**: 基于Story 3.1的AI大模型服务
- **NLP库**: jieba, HanLP, spaCy, transformers
- **质量评估**: BLEU, ROUGE, 自研质量评分模型
- **数据库**: PostgreSQL (元数据) + MongoDB (文本内容)
- **缓存**: Redis (优化结果缓存)
- **任务队列**: Celery + RabbitMQ
- **版本控制**: Git-like差异算法

### 系统架构设计

#### 服务架构图
```
┌─────────────────────────────────────────────────────────────┐
│                 智能文本优化服务架构                          │
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                        │
│  ├── 文本优化API ├── 批量处理API ├── 版本管理API              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   业务逻辑层                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 优化策略    │ 质量评估    │ 版本管理    │ 批量处理    │  │
│  │ 管理器      │ 引擎        │ 服务        │ 管理器      │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   核心处理层                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 文本分析    │ 优化执行    │ 质量检测    │ 结果生成    │  │
│  │ 引擎        │ 引擎        │ 引擎        │ 引擎        │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   数据存储层                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ PostgreSQL  │ MongoDB     │ Redis       │ 外部AI      │  │
│  │ (元数据)    │ (文本内容)  │ (缓存)      │ 服务        │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 数据库设计

**注意：**以下数据库表结构由storage-service统一管理，本服务通过API调用访问。

#### PostgreSQL元数据库
```sql
-- 优化任务表
CREATE TABLE optimization_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    source_document_id UUID NOT NULL,
    task_type VARCHAR(50) NOT NULL, -- polish, expand, style_convert, modernize
    optimization_mode VARCHAR(50) NOT NULL, -- historical_format, academic, literary
    parameters JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    priority INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    progress_percentage INTEGER DEFAULT 0
);

-- 优化版本表
CREATE TABLE optimization_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES optimization_tasks(id),
    version_number INTEGER NOT NULL,
    title VARCHAR(500),
    content_hash VARCHAR(64) NOT NULL, -- MongoDB文档的哈希值
    quality_score DECIMAL(5,3), -- 0-100分
    metrics JSONB, -- 详细质量指标
    ai_model_used VARCHAR(100),
    optimization_time_ms INTEGER,
    token_usage JSONB,
    is_selected BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 优化策略配置表
CREATE TABLE optimization_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    target_text_types TEXT[], -- 适用的文本类型
    prompt_template TEXT NOT NULL,
    parameters JSONB,
    quality_thresholds JSONB, -- 质量阈值配置
    is_active BOOLEAN DEFAULT true,
    usage_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,3) DEFAULT 0,
    avg_quality_improvement DECIMAL(5,3) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 质量评估规则表
CREATE TABLE quality_assessment_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- readability, academic, historical_accuracy
    weight DECIMAL(3,2) NOT NULL, -- 权重0-1
    parameters JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户偏好表
CREATE TABLE user_optimization_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    preferred_strategies JSONB, -- 偏好的优化策略
    quality_thresholds JSONB, -- 个人质量阈值
    style_preferences JSONB, -- 风格偏好
    feedback_history JSONB, -- 历史反馈数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 批量任务表
CREATE TABLE batch_optimization_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    job_name VARCHAR(200),
    source_document_ids UUID[],
    optimization_config JSONB,
    total_documents INTEGER,
    completed_documents INTEGER DEFAULT 0,
    failed_documents INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    estimated_completion_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_optimization_tasks_user_status ON optimization_tasks(user_id, status);
CREATE INDEX idx_optimization_tasks_created_at ON optimization_tasks(created_at);
CREATE INDEX idx_optimization_versions_task_id ON optimization_versions(task_id);
CREATE INDEX idx_batch_jobs_user_id ON batch_optimization_jobs(user_id);
```

#### MongoDB文本内容库

**注意：**文本内容存储由storage-service统一管理，本服务通过API访问。
```javascript
// 原始文档集合
{
  "_id": ObjectId,
  "document_id": "uuid",
  "title": "文档标题",
  "content": "原始文本内容",
  "metadata": {
    "author": "作者",
    "dynasty": "朝代", 
    "category": "分类",
    "source": "来源",
    "length": 1000,
    "language": "zh-CN"
  },
  "analysis": {
    "text_type": "史书", // 历史文本类型
    "writing_style": "文言文",
    "complexity_score": 0.8,
    "topics": ["政治", "军事"],
    "entities": [
      {
        "name": "朱元璋",
        "type": "人物",
        "start": 10,
        "end": 13
      }
    ]
  },
  "created_at": ISODate,
  "updated_at": ISODate
}

// 优化后文档集合
{
  "_id": ObjectId,
  "version_id": "uuid",
  "task_id": "uuid", 
  "original_document_id": "uuid",
  "optimized_content": {
    "title": "优化后标题",
    "content": "优化后内容",
    "summary": "内容摘要",
    "key_improvements": [
      "改进了语言表达的流畅性",
      "增加了历史背景描述", 
      "统一了用词规范"
    ]
  },
  "optimization_details": {
    "strategy_used": "historical_format",
    "model_used": "gpt-4",
    "prompt_used": "...",
    "processing_time_ms": 2500,
    "token_usage": {
      "prompt_tokens": 500,
      "completion_tokens": 800,
      "total_tokens": 1300
    }
  },
  "quality_metrics": {
    "overall_score": 85.5,
    "readability_score": 90.0,
    "academic_score": 82.0,
    "historical_accuracy": 88.0,
    "language_quality": 85.0,
    "structure_score": 87.0
  },
  "diff_analysis": {
    "changes_count": 15,
    "additions": 234, // 增加的字符数
    "deletions": 45,  // 删除的字符数
    "modifications": 12, // 修改的地方数
    "change_types": {
      "vocabulary": 8,
      "grammar": 3,
      "structure": 4
    }
  },
  "created_at": ISODate
}
```

### 核心服务实现

#### 文本优化引擎
```python
# text_optimization_engine.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import hashlib
from datetime import datetime

class OptimizationType(Enum):
    POLISH = "polish"           # 文本润色
    EXPAND = "expand"           # 内容扩展
    STYLE_CONVERT = "style_convert"  # 风格转换
    MODERNIZE = "modernize"     # 现代化改写

class OptimizationMode(Enum):
    HISTORICAL_FORMAT = "historical_format"  # 历史文档格式
    ACADEMIC = "academic"                     # 学术规范
    LITERARY = "literary"                     # 文学性
    SIMPLIFIED = "simplified"                 # 简化表达

@dataclass
class OptimizationRequest:
    """文本优化请求"""
    content: str
    optimization_type: OptimizationType
    optimization_mode: OptimizationMode
    target_length: Optional[int] = None
    style_reference: Optional[str] = None
    preserve_entities: bool = True
    quality_threshold: float = 80.0
    custom_instructions: Optional[str] = None

@dataclass
class OptimizationResult:
    """文本优化结果"""
    optimized_content: str
    quality_score: float
    improvements: List[str]
    metrics: Dict[str, float]
    processing_time_ms: int
    token_usage: Dict[str, int]
    model_used: str

class TextOptimizationEngine:
    """
    文本优化引擎核心类
    负责调用AI模型执行文本优化任务
    """
    
    def __init__(self, ai_service, quality_assessor, strategy_manager):
        self.ai_service = ai_service  # AI大模型服务
        self.quality_assessor = quality_assessor  # 质量评估器
        self.strategy_manager = strategy_manager  # 策略管理器
    
    async def optimize_text(self, request: OptimizationRequest) -> OptimizationResult:
        """
        执行文本优化
        
        Args:
            request: 优化请求参数
            
        Returns:
            优化结果
        """
        start_time = datetime.now()
        
        try:
            # 分析原始文本
            text_analysis = await self._analyze_text(request.content)
            
            # 选择优化策略
            strategy = await self.strategy_manager.select_strategy(
                text_analysis=text_analysis,
                optimization_type=request.optimization_type,
                optimization_mode=request.optimization_mode
            )
            
            # 构建优化提示
            prompt = await self._build_optimization_prompt(
                request, text_analysis, strategy
            )
            
            # 调用AI模型执行优化
            ai_response = await self.ai_service.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": strategy.system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=strategy.preferred_model,
                temperature=strategy.temperature,
                max_tokens=strategy.max_tokens
            )
            
            optimized_content = ai_response['choices'][0]['message']['content']
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # 质量评估
            quality_metrics = await self.quality_assessor.assess_quality(
                original_text=request.content,
                optimized_text=optimized_content,
                optimization_type=request.optimization_type
            )
            
            # 生成改进说明
            improvements = await self._generate_improvement_summary(
                request.content, optimized_content, quality_metrics
            )
            
            return OptimizationResult(
                optimized_content=optimized_content,
                quality_score=quality_metrics['overall_score'],
                improvements=improvements,
                metrics=quality_metrics,
                processing_time_ms=int(processing_time),
                token_usage=ai_response['usage'],
                model_used=ai_response['model']
            )
            
        except Exception as e:
            raise OptimizationError(f"文本优化失败: {str(e)}")
    
    async def _analyze_text(self, content: str) -> Dict[str, Any]:
        """分析文本特征"""
        return {
            'length': len(content),
            'complexity': await self._calculate_complexity(content),
            'style': await self._detect_writing_style(content),
            'entities': await self._extract_entities(content),
            'topics': await self._extract_topics(content),
            'language_features': await self._analyze_language_features(content)
        }
    
    async def _build_optimization_prompt(self, 
                                       request: OptimizationRequest,
                                       text_analysis: Dict,
                                       strategy) -> str:
        """构建优化提示词"""
        
        # 基础提示模板
        base_prompt = strategy.prompt_template
        
        # 根据优化类型定制提示
        if request.optimization_type == OptimizationType.POLISH:
            specific_instructions = """
请对以下历史文本进行润色优化：
1. 保持原文的核心内容和历史事实
2. 改善语言表达的流畅性和准确性
3. 统一用词规范，提升文本的学术性
4. 修正语法错误和表达不当之处
5. 保持历史文献的庄重感和严谨性
"""
        elif request.optimization_type == OptimizationType.EXPAND:
            specific_instructions = """
请对以下历史文本进行扩展优化：
1. 在保持原文核心内容的基础上增加相关细节
2. 补充必要的历史背景和上下文信息
3. 增强文本的完整性和可读性
4. 添加适当的人物描述和事件细节
5. 确保扩展内容符合历史事实
"""
        elif request.optimization_type == OptimizationType.STYLE_CONVERT:
            specific_instructions = f"""
请将以下历史文本转换为{request.optimization_mode.value}风格：
1. 调整文体风格以符合目标要求
2. 保持历史事实的准确性
3. 适应目标读者群体的阅读习惯
4. 保持文本的学术价值和历史意义
"""
        elif request.optimization_type == OptimizationType.MODERNIZE:
            specific_instructions = """
请将以下历史文本现代化改写：
1. 将文言文表达转换为现代汉语
2. 保持历史事实和核心内容不变
3. 使用现代读者容易理解的表达方式
4. 保留必要的历史术语和专有名词
5. 确保改写后的文本准确传达原意
"""
        
        # 组合完整提示
        full_prompt = f"""
{specific_instructions}

【原始文本】
{request.content}

【文本分析】
- 文本长度: {text_analysis['length']}字符
- 复杂度: {text_analysis['complexity']}
- 写作风格: {text_analysis['style']}
- 主要主题: {', '.join(text_analysis['topics'])}

【优化要求】
- 优化模式: {request.optimization_mode.value}
- 目标长度: {request.target_length or '不限制'}
- 质量阈值: {request.quality_threshold}分以上
"""
        
        if request.custom_instructions:
            full_prompt += f"\n【特殊要求】\n{request.custom_instructions}"
        
        full_prompt += "\n\n请提供优化后的文本："
        
        return full_prompt
    
    async def _generate_improvement_summary(self,
                                          original: str,
                                          optimized: str, 
                                          metrics: Dict) -> List[str]:
        """生成改进说明"""
        improvements = []
        
        # 根据质量指标生成改进说明
        if metrics.get('readability_improvement', 0) > 5:
            improvements.append("显著提升了文本的可读性和流畅度")
            
        if metrics.get('academic_improvement', 0) > 5:
            improvements.append("增强了文本的学术规范性和严谨性")
            
        if metrics.get('structure_improvement', 0) > 5:
            improvements.append("优化了文本结构和逻辑组织")
            
        if len(optimized) > len(original) * 1.2:
            improvements.append("适当扩展了内容，增加了相关细节和背景信息")
        elif len(optimized) < len(original) * 0.8:
            improvements.append("精简了表达，去除了冗余内容")
            
        if not improvements:
            improvements.append("对原文进行了语言润色和表达优化")
            
        return improvements

class QualityAssessor:
    """
    质量评估器
    对优化后的文本进行多维度质量评估
    """
    
    def __init__(self, rules_manager):
        self.rules_manager = rules_manager
    
    async def assess_quality(self,
                           original_text: str,
                           optimized_text: str,
                           optimization_type: OptimizationType) -> Dict[str, float]:
        """
        评估文本质量
        
        Args:
            original_text: 原始文本
            optimized_text: 优化后文本
            optimization_type: 优化类型
            
        Returns:
            质量评估结果
        """
        metrics = {}
        
        # 可读性评估
        metrics['readability_score'] = await self._assess_readability(optimized_text)
        metrics['readability_improvement'] = (
            metrics['readability_score'] - 
            await self._assess_readability(original_text)
        )
        
        # 学术规范性评估
        metrics['academic_score'] = await self._assess_academic_quality(optimized_text)
        metrics['academic_improvement'] = (
            metrics['academic_score'] -
            await self._assess_academic_quality(original_text)
        )
        
        # 历史准确性评估
        metrics['historical_accuracy'] = await self._assess_historical_accuracy(
            original_text, optimized_text
        )
        
        # 语言质量评估
        metrics['language_quality'] = await self._assess_language_quality(optimized_text)
        
        # 结构质量评估
        metrics['structure_score'] = await self._assess_structure_quality(optimized_text)
        metrics['structure_improvement'] = (
            metrics['structure_score'] -
            await self._assess_structure_quality(original_text)
        )
        
        # 内容完整性评估
        metrics['content_completeness'] = await self._assess_content_completeness(
            original_text, optimized_text
        )
        
        # 计算综合评分
        weights = await self.rules_manager.get_quality_weights(optimization_type)
        metrics['overall_score'] = sum(
            metrics[metric] * weights.get(metric, 0)
            for metric in ['readability_score', 'academic_score', 'historical_accuracy',
                          'language_quality', 'structure_score', 'content_completeness']
        )
        
        return metrics
    
    async def _assess_readability(self, text: str) -> float:
        """评估可读性"""
        # 实现基于多种指标的可读性评估
        # 包括句子长度、词汇难度、语法复杂性等
        import jieba
        
        sentences = text.split('。')
        avg_sentence_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
        
        words = list(jieba.cut(text))
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        
        # 简化的可读性评分算法
        readability = 100 - (avg_sentence_length * 0.5 + avg_word_length * 2)
        return max(0, min(100, readability))
    
    async def _assess_academic_quality(self, text: str) -> float:
        """评估学术质量"""
        # 评估用词规范性、表达严谨性、逻辑清晰度等
        academic_keywords = ['据', '史载', '考证', '研究表明', '学者认为']
        formal_expressions = ['因此', '然而', '此外', '综上所述']
        
        academic_score = 0
        for keyword in academic_keywords:
            if keyword in text:
                academic_score += 10
        
        for expr in formal_expressions:
            if expr in text:
                academic_score += 5
                
        return min(100, academic_score)
    
    async def _assess_historical_accuracy(self, original: str, optimized: str) -> float:
        """评估历史准确性"""
        # 检查关键历史信息是否保持一致
        import jieba.posseg as pseg
        
        # 提取人名、地名、朝代等关键信息
        original_entities = self._extract_historical_entities(original)
        optimized_entities = self._extract_historical_entities(optimized)
        
        # 计算实体保持率
        if not original_entities:
            return 100.0
            
        preserved_entities = set(original_entities) & set(optimized_entities)
        preservation_rate = len(preserved_entities) / len(original_entities)
        
        return preservation_rate * 100
    
    def _extract_historical_entities(self, text: str) -> List[str]:
        """提取历史实体"""
        import jieba.posseg as pseg
        
        entities = []
        words = pseg.cut(text)
        
        for word, flag in words:
            # 人名、地名等
            if flag in ['nr', 'ns', 'nt'] and len(word) > 1:
                entities.append(word)
        
        return entities

class OptimizationStrategyManager:
    """
    优化策略管理器
    管理不同的文本优化策略和配置
    """
    
    def __init__(self, storage_client):
        self.storage_client = storage_client
        self.strategies = {}
        
    async def load_strategies(self):
        """加载优化策略配置"""
        strategies_result = await self.storage_client.get_optimization_strategies(
            active_only=True
        )
        strategies = strategies_result.get('data', [])
        for strategy in strategies:
            self.strategies[strategy['name']] = strategy
    
    async def select_strategy(self,
                            text_analysis: Dict,
                            optimization_type: OptimizationType,
                            optimization_mode: OptimizationMode):
        """选择最佳优化策略"""
        
        # 根据文本类型和优化需求选择策略
        strategy_key = f"{optimization_type.value}_{optimization_mode.value}"
        
        if strategy_key in self.strategies:
            return self.strategies[strategy_key]
        
        # 回退到默认策略
        return self.strategies.get('default_strategy')

class BatchOptimizationManager:
    """
    批量优化管理器
    处理大规模文档的批量优化任务
    """
    
    def __init__(self, optimization_engine, storage_client, task_queue):
        self.optimization_engine = optimization_engine
        self.storage_client = storage_client
        self.task_queue = task_queue
    
    async def create_batch_job(self,
                             user_id: str,
                             document_ids: List[str],
                             optimization_config: Dict) -> str:
        """创建批量优化任务"""
        
        job_id = str(uuid.uuid4())
        
        # 通过storage-service保存任务信息
        await self.storage_client.create_batch_optimization_job({
            'job_id': job_id,
            'user_id': user_id,
            'job_name': optimization_config.get('job_name'),
            'source_document_ids': document_ids,
            'optimization_config': optimization_config,
            'total_documents': len(document_ids)
        })
        
        # 为每个文档创建子任务
        for doc_id in document_ids:
            await self.task_queue.enqueue('optimize_document', {
                'job_id': job_id,
                'document_id': doc_id,
                'config': optimization_config
            })
        
        return job_id
    
    async def process_document_optimization(self, task_data: Dict):
        """处理单个文档的优化任务"""
        try:
            job_id = task_data['job_id']
            document_id = task_data['document_id']
            config = task_data['config']
            
            # 通过storage-service获取文档内容
            document = await self.storage_client.get_document(document_id)
            
            # 执行优化
            request = OptimizationRequest(
                content=document['content'],
                optimization_type=OptimizationType(config['optimization_type']),
                optimization_mode=OptimizationMode(config['optimization_mode']),
                **config.get('parameters', {})
            )
            
            result = await self.optimization_engine.optimize_text(request)
            
            # 保存结果
            await self._save_optimization_result(job_id, document_id, result)
            
            # 更新任务进度
            await self._update_job_progress(job_id, completed=True)
            
        except Exception as e:
            await self._update_job_progress(job_id, failed=True, error=str(e))
    
    async def _update_job_progress(self, job_id: str, completed: bool = False, 
                                 failed: bool = False, error: str = None):
        """更新任务进度"""
        if completed:
            await self.storage_client.update_batch_job_progress(
                job_id=job_id,
                completed_increment=1
            )
        elif failed:
            await self.storage_client.update_batch_job_progress(
                job_id=job_id,
                failed_increment=1,
                error_message=error
            )
```

### API接口设计

#### 文本优化API
```python
# 单文档优化
POST /api/v1/optimization/optimize
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

Request:
{
    "content": "朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州...",
    "optimization_type": "polish", // polish, expand, style_convert, modernize
    "optimization_mode": "historical_format", // historical_format, academic, literary
    "parameters": {
        "target_length": null,
        "preserve_entities": true,
        "quality_threshold": 85.0,
        "custom_instructions": "请保持史书体例的庄重感"
    },
    "generate_versions": 3 // 生成多个版本
}

Response:
{
    "success": true,
    "data": {
        "task_id": "task_uuid",
        "versions": [
            {
                "version_id": "version_uuid_1",
                "content": "明太祖朱元璋，安徽濠州钟离县人。其祖上原居沛县，后迁至句容，再迁泗州...",
                "quality_score": 88.5,
                "metrics": {
                    "readability_score": 90.2,
                    "academic_score": 85.8,
                    "historical_accuracy": 92.1,
                    "language_quality": 87.3,
                    "structure_score": 89.0
                },
                "improvements": [
                    "统一了地名表达的现代化标准",
                    "改善了语言表达的流畅性",
                    "增强了文本的学术规范性"
                ],
                "processing_time_ms": 2300,
                "token_usage": {
                    "prompt_tokens": 450,
                    "completion_tokens": 380,
                    "total_tokens": 830
                }
            }
        ],
        "recommended_version": "version_uuid_1"
    }
}

# 批量优化
POST /api/v1/optimization/batch
Request:
{
    "document_ids": ["doc1", "doc2", "doc3"],
    "optimization_config": {
        "job_name": "史书文档批量优化",
        "optimization_type": "polish",
        "optimization_mode": "academic",
        "parameters": {...}
    }
}

Response:
{
    "success": true,
    "data": {
        "job_id": "job_uuid",
        "total_documents": 3,
        "estimated_completion_time": "2024-01-15T10:30:00Z"
    }
}

# 查询批量任务状态
GET /api/v1/optimization/batch/{job_id}/status
Response:
{
    "success": true,
    "data": {
        "job_id": "job_uuid",
        "status": "processing", // pending, processing, completed, failed
        "progress": {
            "total_documents": 100,
            "completed_documents": 65,
            "failed_documents": 2,
            "progress_percentage": 65
        },
        "estimated_remaining_time": "00:15:30",
        "results": {
            "successful_optimizations": 63,
            "average_quality_score": 86.7,
            "total_cost": 15.80
        }
    }
}
```

#### 版本管理API
```python
# 获取优化历史
GET /api/v1/optimization/tasks/{task_id}/versions
Response:
{
    "success": true,
    "data": {
        "task_id": "task_uuid",
        "original_content": "...",
        "versions": [
            {
                "version_id": "v1",
                "version_number": 1,
                "content": "...",
                "quality_score": 85.2,
                "created_at": "2024-01-15T10:00:00Z",
                "is_selected": false
            },
            {
                "version_id": "v2", 
                "version_number": 2,
                "content": "...",
                "quality_score": 88.7,
                "created_at": "2024-01-15T10:05:00Z",
                "is_selected": true
            }
        ]
    }
}

# 版本对比
GET /api/v1/optimization/compare?version1=v1&version2=v2
Response:
{
    "success": true,
    "data": {
        "comparison": {
            "version1": {...},
            "version2": {...},
            "differences": [
                {
                    "type": "modification",
                    "position": 15,
                    "original": "其先世家沛",
                    "modified": "其祖上原居沛县",
                    "reason": "地名表达现代化"
                }
            ],
            "quality_comparison": {
                "version1_score": 85.2,
                "version2_score": 88.7,
                "improvement_areas": ["学术规范性", "语言流畅度"]
            }
        }
    }
}

# 选择版本
POST /api/v1/optimization/tasks/{task_id}/select-version
Request:
{
    "version_id": "version_uuid"
}
```

## 验收标准

### 功能性验收标准

1. **文本优化功能**
   - ✅ 支持4种优化类型（润色、扩展、风格转换、现代化）
   - ✅ 支持4种优化模式（历史格式、学术、文学、简化）
   - ✅ 优化质量评分>85分
   - ✅ 历史事实准确性保持>95%

2. **质量评估系统**
   - ✅ 多维度质量评估（可读性、学术性、准确性等）
   - ✅ 自动生成改进说明
   - ✅ 质量对比分析
   - ✅ 个性化质量阈值

3. **版本管理功能**
   - ✅ 支持多版本生成和对比
   - ✅ 版本差异分析
   - ✅ 历史记录追踪
   - ✅ 版本回滚功能

4. **批量处理能力**
   - ✅ 支持1000+文档批量处理
   - ✅ 任务进度实时监控
   - ✅ 失败重试机制
   - ✅ 结果统计报告

### 性能验收标准

1. **响应性能**
   - ✅ 单文档优化<3秒
   - ✅ 批量任务启动<1秒
   - ✅ 质量评估<0.5秒
   - ✅ API响应时间<500ms

2. **处理能力**
   - ✅ 支持10万字长文档
   - ✅ 并发处理>50个任务
   - ✅ 日处理能力>10万文档
   - ✅ 系统可用性>99.5%

### 质量验收标准

1. **优化质量**
   - ✅ 平均质量提升>15分
   - ✅ 用户满意度>90%
   - ✅ 历史准确性>95%
   - ✅ 语言流畅度>90%

2. **系统稳定性**
   - ✅ 错误率<1%
   - ✅ 异常恢复时间<30秒
   - ✅ 数据一致性100%
   - ✅ 并发稳定性>99%

## 业务价值

### 直接价值
1. **内容质量提升**: 大幅改善历史文本的可读性和学术价值
2. **效率提升**: 自动化文本优化减少90%的人工编辑时间
3. **标准化**: 统一历史文档的体例和表达规范
4. **规模化**: 支持大规模历史文献的批量处理

### 间接价值
1. **学术价值**: 提升历史文献的学术研究价值
2. **传播价值**: 改善文本可读性促进历史文化传播
3. **平台价值**: 为历史文本平台提供核心竞争力
4. **商业价值**: 为专业文本优化服务奠定基础

## 总结

智能文本优化服务是历史文本项目的核心功能之一，通过先进的AI技术和专业的质量评估体系，实现了历史文献的智能化优化和标准化处理。该服务不仅能够显著提升文本质量，还为大规模历史文献的数字化处理提供了强有力的技术支撑。