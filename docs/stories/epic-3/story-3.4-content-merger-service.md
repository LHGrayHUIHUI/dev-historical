# Story 3.4: 多内容合并生成功能

## 基本信息
- **Epic**: Epic 3 - AI大模型服务和内容文本优化
- **Story ID**: 3.4
- **优先级**: 中
- **预估工作量**: 2周
- **负责团队**: 后端开发团队 + AI工程团队

## 用户故事

**作为** 内容创作者  
**我希望** 将多个相关内容智能合并生成新的综合内容  
**以便于** 创作更丰富的历史文本，提升内容的完整性和深度

## 需求描述

### 核心功能需求

1. **智能内容分析**
   - 主题提取和内容理解
   - 关键信息识别和分类
   - 内容相关性分析
   - 重复内容检测和去重
   - 信息完整性评估

2. **多样化合并策略**
   - **时间线整合**: 按时间顺序整合历史事件
   - **主题归并**: 按主题分类合并相关内容
   - **层次整合**: 按重要性层次组织内容
   - **逻辑关系构建**: 建立内容间的因果关系
   - **补充扩展**: 用相关内容补充主要内容

3. **智能生成算法**
   - 基于模板的内容生成
   - AI驱动的创意写作
   - 多文档摘要生成
   - 结构化内容组织
   - 个性化内容定制

4. **质量控制机制**
   - 内容一致性检查
   - 逻辑完整性验证
   - 信息准确性核实
   - 文本流畅度评估
   - 原创性检测

5. **交互式编辑**
   - 合并过程可视化
   - 用户参与决策
   - 实时预览和调整
   - 版本对比和选择
   - 手动干预和优化

## 技术实现

### 核心技术栈

- **服务框架**: FastAPI + Python 3.11
- **AI集成**: 基于Story 3.1的AI大模型服务
- **NLP处理**: spaCy, transformers, sentence-transformers
- **文本分析**: BERT, TF-IDF, TextRank
- **相似度计算**: 余弦相似度, Jaccard系数
- **数据库**: PostgreSQL (任务管理) + MongoDB (内容存储)
- **缓存**: Redis (分析结果缓存)
- **任务队列**: Celery + RabbitMQ

### 系统架构设计

#### 内容合并服务架构图
```
┌─────────────────────────────────────────────────────────────┐
│                 多内容合并生成服务架构                        │
├─────────────────────────────────────────────────────────────┤
│  API Layer                                                  │
│  ├── 合并任务API ├── 内容分析API ├── 生成配置API              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   内容分析层                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 主题提取    │ 关键信息    │ 相关性      │ 重复检测    │  │
│  │ 引擎        │ 识别器      │ 分析器      │ 引擎        │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   合并策略层                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 时间线      │ 主题归并    │ 层次整合    │ 逻辑构建    │  │
│  │ 整合器      │ 引擎        │ 引擎        │ 引擎        │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   内容生成层                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ AI文本      │ 模板生成    │ 摘要生成    │ 结构组织    │  │
│  │ 生成器      │ 引擎        │ 引擎        │ 引擎        │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   质量控制层                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 一致性      │ 完整性      │ 准确性      │ 流畅度      │  │
│  │ 检查器      │ 验证器      │ 核实器      │ 评估器      │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 数据库设计

**注意：**以下数据库表结构由storage-service统一管理，本服务通过API调用访问。

#### PostgreSQL任务管理数据库
```sql
-- 合并任务表
CREATE TABLE merge_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    task_name VARCHAR(200),
    source_content_ids UUID[] NOT NULL,
    merge_strategy VARCHAR(50) NOT NULL, -- timeline, topic, hierarchy, logic, supplement
    merge_config JSONB,
    target_length INTEGER, -- 目标长度
    target_style VARCHAR(50), -- 目标风格
    status VARCHAR(20) DEFAULT 'pending', -- pending, analyzing, merging, completed, failed
    progress_percentage INTEGER DEFAULT 0,
    result_content_id UUID,
    quality_score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- 内容分析结果表
CREATE TABLE content_analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- topic, entity, relation, sentiment
    analysis_result JSONB NOT NULL,
    confidence_score DECIMAL(5,3),
    analysis_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 合并策略配置表
CREATE TABLE merge_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(100) NOT NULL UNIQUE,
    strategy_type VARCHAR(50) NOT NULL,
    description TEXT,
    algorithm_config JSONB NOT NULL,
    template_config JSONB,
    quality_weights JSONB, -- 质量评估权重
    is_active BOOLEAN DEFAULT true,
    usage_count INTEGER DEFAULT 0,
    avg_quality_score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 内容关联关系表
CREATE TABLE content_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_content_id UUID NOT NULL,
    target_content_id UUID NOT NULL,
    relationship_type VARCHAR(50) NOT NULL, -- similar, supplement, contradiction, sequence
    similarity_score DECIMAL(5,3),
    relationship_strength DECIMAL(5,3),
    relationship_data JSONB, -- 关系的详细数据
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 合并操作日志表
CREATE TABLE merge_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES merge_tasks(id),
    operation_type VARCHAR(50) NOT NULL, -- analyze, extract, merge, generate, validate
    operation_data JSONB,
    processing_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户合并偏好表
CREATE TABLE user_merge_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    preferred_strategies JSONB,
    quality_thresholds JSONB,
    style_preferences JSONB,
    template_preferences JSONB,
    feedback_history JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_merge_tasks_user_status ON merge_tasks(user_id, status);
CREATE INDEX idx_content_analysis_content ON content_analysis_results(content_id);
CREATE INDEX idx_content_relationships_source ON content_relationships(source_content_id);
CREATE INDEX idx_merge_operations_task ON merge_operations(task_id);
```

#### MongoDB内容存储数据库

**注意：**内容数据存储由storage-service统一管理，本服务通过API访问。
```javascript
// 原始内容集合（继承自之前的设计）
{
  "_id": ObjectId,
  "content_id": "uuid",
  "title": "内容标题",
  "content": "原始内容文本",
  "metadata": {
    "author": "作者",
    "source": "来源",
    "category": "分类",
    "tags": ["标签1", "标签2"],
    "length": 1000,
    "created_date": "2024-01-15"
  },
  "analysis": {
    "topics": [
      {
        "topic": "政治改革",
        "relevance": 0.85,
        "keywords": ["变法", "改革", "政策"]
      }
    ],
    "entities": [
      {
        "name": "王安石",
        "type": "人物",
        "importance": 0.9,
        "mentions": [10, 45, 120]
      }
    ],
    "temporal_info": {
      "time_period": "北宋中期",
      "specific_dates": ["1069年", "1070年"],
      "temporal_order": 1
    },
    "key_points": [
      "王安石变法的背景",
      "新法的主要内容",
      "变法的影响"
    ]
  },
  "created_at": ISODate
}

// 合并结果集合
{
  "_id": ObjectId,
  "result_id": "uuid",
  "task_id": "uuid",
  "source_content_ids": ["uuid1", "uuid2", "uuid3"],
  "merged_content": {
    "title": "合并后标题",
    "content": "合并后的完整内容",
    "structure": {
      "sections": [
        {
          "title": "背景介绍",
          "content": "...",
          "source_contents": ["uuid1"],
          "merge_type": "direct"
        },
        {
          "title": "主要事件",
          "content": "...",
          "source_contents": ["uuid1", "uuid2"],
          "merge_type": "timeline_integration"
        }
      ],
      "transitions": [
        {
          "from_section": 0,
          "to_section": 1,
          "transition_text": "在此背景下，",
          "transition_type": "causal"
        }
      ]
    },
    "summary": "合并内容的摘要"
  },
  "merge_metadata": {
    "strategy_used": "timeline_integration",
    "merge_config": {...},
    "source_analysis": {
      "total_length": 5000,
      "overlap_percentage": 15.5,
      "complementarity_score": 0.78
    },
    "generation_info": {
      "ai_model_used": "gpt-4",
      "processing_time_ms": 8500,
      "token_usage": {
        "prompt_tokens": 2000,
        "completion_tokens": 1200
      }
    }
  },
  "quality_metrics": {
    "overall_score": 87.5,
    "consistency_score": 90.0,
    "completeness_score": 85.0,
    "fluency_score": 88.0,
    "originality_score": 82.0,
    "factual_accuracy": 92.0
  },
  "user_feedback": {
    "rating": 4.5,
    "comments": "合并效果很好，逻辑清晰",
    "suggested_improvements": []
  },
  "created_at": ISODate
}

// 合并过程记录集合
{
  "_id": ObjectId,
  "task_id": "uuid",
  "process_steps": [
    {
      "step": 1,
      "name": "内容分析",
      "start_time": ISODate,
      "end_time": ISODate,
      "results": {
        "topics_extracted": 5,
        "entities_identified": 12,
        "relationships_found": 8
      }
    },
    {
      "step": 2,
      "name": "相似度计算",
      "start_time": ISODate,
      "end_time": ISODate,
      "results": {
        "similarity_matrix": [[1.0, 0.65, 0.43], [0.65, 1.0, 0.52], [0.43, 0.52, 1.0]],
        "overlap_analysis": {...}
      }
    },
    {
      "step": 3,
      "name": "内容合并",
      "start_time": ISODate,
      "end_time": ISODate,
      "results": {
        "sections_created": 4,
        "transitions_added": 3,
        "content_reorganized": true
      }
    }
  ],
  "intermediate_results": {
    "content_outline": {...},
    "merge_decisions": [...],
    "quality_checks": [...]
  }
}
```

### 核心服务实现

#### 内容合并引擎
```python
# content_merger_engine.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import numpy as np

class MergeStrategy(Enum):
    TIMELINE = "timeline"         # 时间线整合
    TOPIC = "topic"              # 主题归并
    HIERARCHY = "hierarchy"       # 层次整合
    LOGIC = "logic"              # 逻辑关系
    SUPPLEMENT = "supplement"     # 补充扩展

class MergeMode(Enum):
    COMPREHENSIVE = "comprehensive"  # 全面合并
    SELECTIVE = "selective"         # 选择性合并
    SUMMARY = "summary"             # 摘要合并
    EXPANSION = "expansion"         # 扩展合并

@dataclass
class ContentItem:
    """内容项数据类"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    analysis: Dict[str, Any]

@dataclass
class MergeRequest:
    """合并请求数据类"""
    source_contents: List[ContentItem]
    strategy: MergeStrategy
    mode: MergeMode
    target_length: Optional[int] = None
    target_style: Optional[str] = None
    custom_config: Optional[Dict] = None

@dataclass
class MergeResult:
    """合并结果数据类"""
    merged_content: str
    title: str
    structure: Dict[str, Any]
    quality_metrics: Dict[str, float]
    merge_metadata: Dict[str, Any]

class ContentMergerEngine:
    """
    内容合并引擎核心类
    负责执行多内容的智能合并
    """
    
    def __init__(self, ai_service, analyzer, quality_assessor):
        self.ai_service = ai_service
        self.analyzer = analyzer
        self.quality_assessor = quality_assessor
        self.strategies = {
            MergeStrategy.TIMELINE: TimelineMerger(),
            MergeStrategy.TOPIC: TopicMerger(),
            MergeStrategy.HIERARCHY: HierarchyMerger(),
            MergeStrategy.LOGIC: LogicMerger(),
            MergeStrategy.SUPPLEMENT: SupplementMerger()
        }
    
    async def merge_contents(self, request: MergeRequest) -> MergeResult:
        """
        执行内容合并
        
        Args:
            request: 合并请求
            
        Returns:
            合并结果
        """
        start_time = datetime.now()
        
        try:
            # 1. 内容预分析
            analysis_results = await self._analyze_contents(request.source_contents)
            
            # 2. 关系分析
            relationships = await self._analyze_relationships(
                request.source_contents, analysis_results
            )
            
            # 3. 选择合并策略
            merger = self.strategies[request.strategy]
            
            # 4. 执行合并
            merge_plan = await merger.create_merge_plan(
                request.source_contents, analysis_results, relationships, request
            )
            
            merged_content = await merger.execute_merge(
                merge_plan, request, self.ai_service
            )
            
            # 5. 质量评估
            quality_metrics = await self.quality_assessor.assess_merge_quality(
                request.source_contents, merged_content, request
            )
            
            # 6. 生成结果
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MergeResult(
                merged_content=merged_content['content'],
                title=merged_content['title'],
                structure=merged_content['structure'],
                quality_metrics=quality_metrics,
                merge_metadata={
                    'strategy_used': request.strategy.value,
                    'processing_time_ms': processing_time,
                    'source_count': len(request.source_contents),
                    'merge_plan': merge_plan
                }
            )
            
        except Exception as e:
            raise MergeError(f"内容合并失败: {str(e)}")
    
    async def _analyze_contents(self, contents: List[ContentItem]) -> List[Dict]:
        """分析内容特征"""
        analysis_tasks = []
        
        for content in contents:
            task = self.analyzer.analyze_content(content)
            analysis_tasks.append(task)
        
        return await asyncio.gather(*analysis_tasks)
    
    async def _analyze_relationships(self, 
                                   contents: List[ContentItem],
                                   analyses: List[Dict]) -> Dict[str, Any]:
        """分析内容间关系"""
        relationships = {
            'similarity_matrix': [],
            'temporal_order': [],
            'topic_overlaps': [],
            'entity_connections': []
        }
        
        # 计算相似度矩阵
        similarity_matrix = []
        for i in range(len(contents)):
            row = []
            for j in range(len(contents)):
                if i == j:
                    row.append(1.0)
                else:
                    similarity = await self._calculate_similarity(
                        contents[i], contents[j], analyses[i], analyses[j]
                    )
                    row.append(similarity)
            similarity_matrix.append(row)
        
        relationships['similarity_matrix'] = similarity_matrix
        
        # 分析时间顺序
        temporal_info = [analysis.get('temporal_info', {}) for analysis in analyses]
        relationships['temporal_order'] = await self._analyze_temporal_order(temporal_info)
        
        # 分析主题重叠
        topics = [analysis.get('topics', []) for analysis in analyses]
        relationships['topic_overlaps'] = await self._analyze_topic_overlaps(topics)
        
        return relationships
    
    async def _calculate_similarity(self, content1: ContentItem, content2: ContentItem,
                                  analysis1: Dict, analysis2: Dict) -> float:
        """计算两个内容的相似度"""
        # 基于多个维度计算相似度
        topic_similarity = self._calculate_topic_similarity(
            analysis1.get('topics', []), analysis2.get('topics', [])
        )
        
        entity_similarity = self._calculate_entity_similarity(
            analysis1.get('entities', []), analysis2.get('entities', [])
        )
        
        content_similarity = await self._calculate_content_similarity(
            content1.content, content2.content
        )
        
        # 加权平均
        return (topic_similarity * 0.4 + entity_similarity * 0.3 + content_similarity * 0.3)

class TimelineMerger:
    """时间线整合器"""
    
    async def create_merge_plan(self, contents: List[ContentItem], 
                              analyses: List[Dict], relationships: Dict,
                              request: MergeRequest) -> Dict:
        """创建时间线合并计划"""
        plan = {
            'strategy': 'timeline',
            'sections': [],
            'transitions': []
        }
        
        # 根据时间顺序排序内容
        temporal_order = relationships.get('temporal_order', [])
        sorted_contents = self._sort_by_timeline(contents, analyses, temporal_order)
        
        # 创建时间段分组
        time_groups = self._group_by_time_period(sorted_contents, analyses)
        
        # 为每个时间段创建章节
        for time_period, group_contents in time_groups.items():
            section = {
                'title': f"{time_period}",
                'contents': [c.id for c in group_contents],
                'merge_type': 'chronological',
                'time_period': time_period
            }
            plan['sections'].append(section)
        
        # 创建章节间的过渡
        plan['transitions'] = self._create_temporal_transitions(plan['sections'])
        
        return plan
    
    async def execute_merge(self, plan: Dict, request: MergeRequest, ai_service) -> Dict:
        """执行时间线合并"""
        merged_sections = []
        
        for section in plan['sections']:
            # 获取该时间段的内容
            section_contents = [
                c for c in request.source_contents 
                if c.id in section['contents']
            ]
            
            # 生成该时间段的合并内容
            section_content = await self._merge_temporal_section(
                section_contents, section, ai_service
            )
            
            merged_sections.append({
                'title': section['title'],
                'content': section_content
            })
        
        # 组合所有章节
        full_content = await self._combine_sections(
            merged_sections, plan['transitions'], ai_service
        )
        
        return {
            'title': self._generate_timeline_title(request.source_contents),
            'content': full_content,
            'structure': {
                'sections': merged_sections,
                'transitions': plan['transitions']
            }
        }
    
    def _sort_by_timeline(self, contents: List[ContentItem], 
                         analyses: List[Dict], temporal_order: List) -> List[ContentItem]:
        """按时间线排序内容"""
        # 提取时间信息并排序
        content_with_time = []
        
        for i, content in enumerate(contents):
            temporal_info = analyses[i].get('temporal_info', {})
            time_score = self._calculate_time_score(temporal_info)
            content_with_time.append((content, time_score))
        
        # 按时间分数排序
        content_with_time.sort(key=lambda x: x[1])
        return [item[0] for item in content_with_time]
    
    async def _merge_temporal_section(self, contents: List[ContentItem], 
                                    section: Dict, ai_service) -> str:
        """合并单个时间段的内容"""
        if len(contents) == 1:
            return contents[0].content
        
        # 构建时间段合并提示
        prompt = f"""
请将以下关于{section['time_period']}的多个历史文档内容合并为一个连贯的段落：

"""
        
        for i, content in enumerate(contents, 1):
            prompt += f"文档{i}：\n{content.content}\n\n"
        
        prompt += """
合并要求：
1. 按时间顺序组织内容
2. 保持历史事实的准确性
3. 确保内容的逻辑连贯性
4. 避免重复信息
5. 突出该时期的重要特征

请提供合并后的内容：
"""
        
        response = await ai_service.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的历史文献编辑，擅长将多个历史文档按时间线整合。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response['choices'][0]['message']['content']

class TopicMerger:
    """主题归并器"""
    
    async def create_merge_plan(self, contents: List[ContentItem],
                              analyses: List[Dict], relationships: Dict,
                              request: MergeRequest) -> Dict:
        """创建主题合并计划"""
        plan = {
            'strategy': 'topic',
            'topics': [],
            'content_mapping': {}
        }
        
        # 提取所有主题
        all_topics = []
        for analysis in analyses:
            all_topics.extend(analysis.get('topics', []))
        
        # 主题聚类
        clustered_topics = self._cluster_topics(all_topics)
        
        # 为每个主题cluster创建合并计划
        for cluster in clustered_topics:
            topic_plan = {
                'main_topic': cluster['main_topic'],
                'sub_topics': cluster['sub_topics'],
                'related_contents': cluster['related_contents'],
                'importance_score': cluster['importance_score']
            }
            plan['topics'].append(topic_plan)
        
        return plan
    
    async def execute_merge(self, plan: Dict, request: MergeRequest, ai_service) -> Dict:
        """执行主题合并"""
        topic_sections = []
        
        # 按重要性排序主题
        sorted_topics = sorted(plan['topics'], 
                              key=lambda x: x['importance_score'], 
                              reverse=True)
        
        for topic in sorted_topics:
            # 获取该主题相关的内容
            related_contents = [
                c for c in request.source_contents
                if c.id in topic['related_contents']
            ]
            
            # 生成主题章节
            section_content = await self._merge_topic_section(
                related_contents, topic, ai_service
            )
            
            topic_sections.append({
                'title': topic['main_topic'],
                'content': section_content
            })
        
        # 组合所有主题章节
        full_content = await self._combine_topic_sections(
            topic_sections, ai_service
        )
        
        return {
            'title': self._generate_topic_title(plan['topics']),
            'content': full_content,
            'structure': {
                'sections': topic_sections,
                'topic_hierarchy': plan['topics']
            }
        }

class QualityAssessor:
    """合并质量评估器"""
    
    async def assess_merge_quality(self, source_contents: List[ContentItem],
                                 merged_result: Dict, request: MergeRequest) -> Dict[str, float]:
        """评估合并质量"""
        metrics = {}
        
        # 一致性评估
        metrics['consistency_score'] = await self._assess_consistency(
            merged_result['content']
        )
        
        # 完整性评估
        metrics['completeness_score'] = await self._assess_completeness(
            source_contents, merged_result['content']
        )
        
        # 流畅性评估
        metrics['fluency_score'] = await self._assess_fluency(
            merged_result['content']
        )
        
        # 原创性评估
        metrics['originality_score'] = await self._assess_originality(
            source_contents, merged_result['content']
        )
        
        # 事实准确性评估
        metrics['factual_accuracy'] = await self._assess_factual_accuracy(
            source_contents, merged_result['content']
        )
        
        # 综合评分
        weights = {
            'consistency_score': 0.25,
            'completeness_score': 0.25,
            'fluency_score': 0.20,
            'originality_score': 0.15,
            'factual_accuracy': 0.15
        }
        
        metrics['overall_score'] = sum(
            metrics[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return metrics
    
    async def _assess_consistency(self, content: str) -> float:
        """评估内容一致性"""
        # 检查内容的逻辑一致性
        sentences = content.split('。')
        consistency_issues = 0
        
        # 简化的一致性检查
        for i in range(len(sentences) - 1):
            current = sentences[i].strip()
            next_sent = sentences[i + 1].strip()
            
            if current and next_sent:
                # 检查时间逻辑一致性
                if self._has_temporal_inconsistency(current, next_sent):
                    consistency_issues += 1
        
        consistency_score = max(0, 100 - consistency_issues * 10)
        return min(100, consistency_score)
    
    async def _assess_completeness(self, source_contents: List[ContentItem],
                                 merged_content: str) -> float:
        """评估内容完整性"""
        # 检查关键信息是否被保留
        source_key_points = []
        for content in source_contents:
            key_points = content.analysis.get('key_points', [])
            source_key_points.extend(key_points)
        
        if not source_key_points:
            return 100.0
        
        # 检查合并内容中保留了多少关键点
        preserved_points = 0
        for point in source_key_points:
            if any(keyword in merged_content for keyword in point.split()[:3]):
                preserved_points += 1
        
        completeness_score = (preserved_points / len(source_key_points)) * 100
        return completeness_score

class BatchMergeManager:
    """批量合并管理器"""
    
    def __init__(self, merger_engine, storage_client, task_queue):
        self.merger_engine = merger_engine
        self.storage_client = storage_client
        self.task_queue = task_queue
    
    async def create_batch_merge_job(self, user_id: str,
                                   content_groups: List[List[str]],
                                   merge_config: Dict) -> str:
        """创建批量合并任务"""
        job_id = str(uuid.uuid4())
        
        # 通过storage-service保存批量任务信息
        await self.storage_client.create_batch_merge_job({
            'job_id': job_id,
            'user_id': user_id,
            'content_groups': content_groups,
            'merge_config': merge_config,
            'total_groups': len(content_groups)
        })
        
        # 为每个内容组创建合并子任务
        for i, content_group in enumerate(content_groups):
            await self.task_queue.enqueue('merge_content_group', {
                'job_id': job_id,
                'group_index': i,
                'content_ids': content_group,
                'merge_config': merge_config
            })
        
        return job_id
    
    async def process_content_group_merge(self, task_data: Dict):
        """处理单个内容组的合并任务"""
        try:
            job_id = task_data['job_id']
            content_ids = task_data['content_ids']
            merge_config = task_data['merge_config']
            
            # 通过storage-service获取内容数据
            contents = await self.storage_client.get_contents_by_ids(content_ids)
            
            # 创建合并请求
            request = MergeRequest(
                source_contents=contents,
                strategy=MergeStrategy(merge_config['strategy']),
                mode=MergeMode(merge_config['mode']),
                target_length=merge_config.get('target_length'),
                target_style=merge_config.get('target_style')
            )
            
            # 执行合并
            result = await self.merger_engine.merge_contents(request)
            
            # 通过storage-service保存结果
            await self.storage_client.save_merge_result(job_id, result)
            
            # 通过storage-service更新任务进度
            await self.storage_client.update_merge_job_progress(job_id, completed=True)
            
        except Exception as e:
            await self.storage_client.update_merge_job_progress(job_id, failed=True, error=str(e))
```

### API接口设计

#### 内容合并API
```python
# 创建合并任务
POST /api/v1/merge/create
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

Request:
{
    "source_content_ids": ["content1", "content2", "content3"],
    "strategy": "timeline", // timeline, topic, hierarchy, logic, supplement
    "mode": "comprehensive", // comprehensive, selective, summary, expansion
    "config": {
        "target_length": 3000,
        "target_style": "academic",
        "preserve_entities": true,
        "merge_overlaps": true,
        "quality_threshold": 85.0
    },
    "title": "王安石变法综述"
}

Response:
{
    "success": true,
    "data": {
        "task_id": "task_uuid",
        "status": "pending",
        "estimated_time_minutes": 5,
        "source_analysis": {
            "total_content_length": 8500,
            "estimated_overlap": 20,
            "complexity_score": 7.5
        }
    }
}

# 获取合并任务状态
GET /api/v1/merge/tasks/{task_id}/status
Response:
{
    "success": true,
    "data": {
        "task_id": "task_uuid",
        "status": "processing", // pending, analyzing, merging, completed, failed
        "progress_percentage": 65,
        "current_step": "content_generation",
        "estimated_remaining_time": "2 minutes",
        "steps_completed": [
            {
                "step": "content_analysis",
                "completed_at": "2024-01-15T10:05:00Z",
                "duration_ms": 2500
            },
            {
                "step": "relationship_analysis", 
                "completed_at": "2024-01-15T10:06:00Z",
                "duration_ms": 3200
            }
        ]
    }
}

# 获取合并结果
GET /api/v1/merge/tasks/{task_id}/result
Response:
{
    "success": true,
    "data": {
        "task_id": "task_uuid",
        "result": {
            "title": "王安石变法：背景、过程与影响综述",
            "content": "北宋中期，面对财政困难和社会矛盾...",
            "structure": {
                "sections": [
                    {
                        "title": "历史背景",
                        "content": "...",
                        "source_contents": ["content1"],
                        "word_count": 800
                    },
                    {
                        "title": "变法过程",
                        "content": "...",
                        "source_contents": ["content1", "content2"],
                        "word_count": 1200
                    }
                ],
                "transitions": [
                    {
                        "from_section": 0,
                        "to_section": 1,
                        "transition_text": "在此背景下，王安石开始了其变法之路。"
                    }
                ]
            },
            "quality_metrics": {
                "overall_score": 87.5,
                "consistency_score": 90.0,
                "completeness_score": 85.0,
                "fluency_score": 88.0,
                "originality_score": 82.0,
                "factual_accuracy": 92.0
            },
            "merge_metadata": {
                "strategy_used": "timeline",
                "source_count": 3,
                "processing_time_ms": 8500,
                "ai_model_used": "gpt-4",
                "token_usage": {
                    "total_tokens": 3500
                }
            }
        }
    }
}

# 批量合并
POST /api/v1/merge/batch
Request:
{
    "content_groups": [
        ["content1", "content2"],
        ["content3", "content4", "content5"],
        ["content6", "content7"]
    ],
    "merge_config": {
        "strategy": "topic",
        "mode": "summary",
        "target_length": 2000
    },
    "job_name": "历史文献批量合并"
}

Response:
{
    "success": true,
    "data": {
        "job_id": "job_uuid",
        "total_groups": 3,
        "estimated_completion_time": "15 minutes"
    }
}
```

#### 内容分析API
```python
# 内容关系分析
POST /api/v1/merge/analyze-relationships
Request:
{
    "content_ids": ["content1", "content2", "content3"]
}

Response:
{
    "success": true,
    "data": {
        "relationships": {
            "similarity_matrix": [
                [1.0, 0.75, 0.45],
                [0.75, 1.0, 0.62],
                [0.45, 0.62, 1.0]
            ],
            "temporal_order": [
                {"content_id": "content1", "time_score": 1069.5},
                {"content_id": "content2", "time_score": 1070.2},
                {"content_id": "content3", "time_score": 1076.8}
            ],
            "topic_overlaps": [
                {
                    "topic": "政治改革",
                    "contents": ["content1", "content2"],
                    "overlap_score": 0.85
                }
            ],
            "entity_connections": [
                {
                    "entity": "王安石",
                    "contents": ["content1", "content2", "content3"],
                    "connection_strength": 0.92
                }
            ]
        },
        "merge_recommendations": [
            {
                "strategy": "timeline",
                "confidence": 0.88,
                "reason": "内容具有明确的时间顺序"
            },
            {
                "strategy": "topic",
                "confidence": 0.75,
                "reason": "存在较强的主题关联性"
            }
        ]
    }
}

# 合并预览
POST /api/v1/merge/preview
Request:
{
    "content_ids": ["content1", "content2"],
    "strategy": "timeline",
    "preview_sections": 2
}

Response:
{
    "success": true,
    "data": {
        "preview": {
            "title": "预期合并标题",
            "sections": [
                {
                    "title": "背景介绍",
                    "preview_content": "北宋中期，社会矛盾日益尖锐...",
                    "estimated_length": 800
                },
                {
                    "title": "变法实施",
                    "preview_content": "1069年，王安石开始推行新法...",
                    "estimated_length": 1200
                }
            ],
            "estimated_quality": 85.0,
            "estimated_processing_time": "3-5分钟"
        }
    }
}
```

## 验收标准

### 功能性验收标准

1. **合并策略完整性**
   - ✅ 支持5种合并策略（时间线、主题、层次、逻辑、补充）
   - ✅ 支持4种合并模式（全面、选择性、摘要、扩展）
   - ✅ 策略选择准确率>85%
   - ✅ 模式适配度>90%

2. **内容分析能力**
   - ✅ 主题提取准确率>90%
   - ✅ 实体识别准确率>95%
   - ✅ 关系分析准确率>85%
   - ✅ 相似度计算误差<5%

3. **合并质量控制**
   - ✅ 合并内容质量评分>85分
   - ✅ 逻辑一致性>90%
   - ✅ 信息完整性>85%
   - ✅ 事实准确性>95%

4. **批量处理能力**
   - ✅ 支持100+内容组批量合并
   - ✅ 任务调度准确率>99%
   - ✅ 失败重试成功率>95%
   - ✅ 进度跟踪实时性<5秒

### 性能验收标准

1. **处理性能**
   - ✅ 单次合并处理时间<5分钟
   - ✅ 内容分析响应时间<30秒
   - ✅ 关系分析处理时间<1分钟
   - ✅ 批量任务启动时间<10秒

2. **系统性能**
   - ✅ 并发处理>20个合并任务
   - ✅ API响应时间<1秒
   - ✅ 系统可用性>99.5%
   - ✅ 缓存命中率>40%

### 质量验收标准

1. **合并质量**
   - ✅ 用户满意度>90%
   - ✅ 内容连贯性>85%
   - ✅ 信息价值提升>20%
   - ✅ 原创性评分>80%

2. **系统稳定性**
   - ✅ 错误率<1%
   - ✅ 数据一致性100%
   - ✅ 异常恢复时间<1分钟
   - ✅ 并发稳定性>99%

## 业务价值

### 直接价值
1. **内容丰富化**: 通过合并生成更完整、深入的历史文本
2. **创作效率**: 自动化内容合并减少80%的人工整理时间
3. **质量提升**: 智能合并提升内容的逻辑性和完整性
4. **知识整合**: 将分散的历史信息整合为系统化知识

### 间接价值
1. **学术价值**: 为历史研究提供更全面的文献资料
2. **教育价值**: 生成适合教学的综合性历史文本
3. **传播价值**: 创造更有吸引力的历史内容
4. **平台价值**: 提升平台的内容创作和管理能力

## 总结

多内容合并生成功能为历史文本项目提供了强大的内容整合能力，通过智能化的分析和合并算法，能够将分散的历史文献整合为连贯、完整的综合性内容。该功能不仅提升了内容的质量和价值，也为历史研究和教育提供了有力的工具支持。