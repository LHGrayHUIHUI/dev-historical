"""
内容合并引擎

该模块实现核心的内容合并算法，支持多种合并策略，
包括时间线整合、主题归并、层次整合、逻辑关系构建和补充扩展。
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import re

from ..models.merger_models import (
    ContentItem, MergeRequest, MergeResult, MergeStrategy, MergeMode,
    MergeSection, MergeTransition, MergeStructure, QualityMetrics, 
    MergeMetadata, TokenUsage, MergeError, ContentAnalysisError
)
from ..clients.ai_service_client import AIServiceClient
from .content_analyzer import ContentAnalyzer
from .quality_assessor import QualityAssessor
from ..config.settings import settings

logger = logging.getLogger(__name__)

class ContentMergerEngine:
    """
    内容合并引擎核心类
    
    负责执行多内容的智能合并，支持多种策略和模式
    """
    
    def __init__(self, 
                 ai_client: AIServiceClient,
                 analyzer: ContentAnalyzer,
                 quality_assessor: QualityAssessor):
        self.ai_client = ai_client
        self.analyzer = analyzer
        self.quality_assessor = quality_assessor
        
        # 初始化合并策略处理器
        self.strategies = {
            MergeStrategy.TIMELINE: TimelineMerger(ai_client),
            MergeStrategy.TOPIC: TopicMerger(ai_client),
            MergeStrategy.HIERARCHY: HierarchyMerger(ai_client),
            MergeStrategy.LOGIC: LogicMerger(ai_client),
            MergeStrategy.SUPPLEMENT: SupplementMerger(ai_client)
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
            logger.info(f"Starting content merge with strategy: {request.strategy}, mode: {request.mode}")
            
            # 1. 验证和预处理
            await self._validate_request(request)
            
            # 2. 内容分析
            logger.info("Analyzing content characteristics")
            analysis_results = await self._analyze_contents(request.source_contents)
            
            # 3. 关系分析
            logger.info("Analyzing content relationships")
            relationships = await self._analyze_relationships(
                request.source_contents, analysis_results
            )
            
            # 4. 选择和执行合并策略
            logger.info(f"Executing merge strategy: {request.strategy}")
            merger = self.strategies[request.strategy]
            
            # 创建合并计划
            merge_plan = await merger.create_merge_plan(
                request.source_contents, analysis_results, relationships, request
            )
            
            # 执行合并
            merged_content = await merger.execute_merge(
                merge_plan, request, self.ai_client
            )
            
            # 5. 质量评估
            logger.info("Assessing merge quality")
            quality_metrics = await self.quality_assessor.assess_merge_quality(
                request.source_contents, merged_content, request
            )
            
            # 6. 生成结果
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = MergeResult(
                title=merged_content['title'],
                content=merged_content['content'],
                structure=MergeStructure(
                    sections=merged_content['structure']['sections'],
                    transitions=merged_content['structure'].get('transitions', []),
                    total_word_count=len(merged_content['content'])
                ),
                quality_metrics=quality_metrics,
                merge_metadata=MergeMetadata(
                    strategy_used=request.strategy.value,
                    source_count=len(request.source_contents),
                    processing_time_ms=int(processing_time),
                    ai_model_used=settings.ai_model.default_model,
                    token_usage=merged_content.get('token_usage'),
                    merge_plan=merge_plan
                )
            )
            
            logger.info(f"Content merge completed successfully in {processing_time:.0f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Content merge failed: {str(e)}")
            raise MergeError(f"内容合并失败: {str(e)}")
    
    async def _validate_request(self, request: MergeRequest):
        """验证合并请求"""
        if not request.source_contents:
            raise MergeError("源内容列表不能为空")
        
        if len(request.source_contents) < 2:
            raise MergeError("至少需要2个源内容")
        
        if len(request.source_contents) > settings.merger_algorithm.max_source_contents:
            raise MergeError(f"源内容数量不能超过{settings.merger_algorithm.max_source_contents}个")
        
        # 检查内容长度
        total_length = sum(len(content.content) for content in request.source_contents)
        if total_length > settings.merger_algorithm.max_content_length:
            raise MergeError(f"内容总长度不能超过{settings.merger_algorithm.max_content_length}字符")
    
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
        
        # 分析实体连接
        entities = [analysis.get('entities', []) for analysis in analyses]
        relationships['entity_connections'] = await self._analyze_entity_connections(entities)
        
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
        weights = settings.merger_algorithm
        return (
            topic_similarity * weights.topic_similarity_weight +
            entity_similarity * weights.entity_similarity_weight +
            content_similarity * weights.content_similarity_weight
        )
    
    def _calculate_topic_similarity(self, topics1: List[Dict], topics2: List[Dict]) -> float:
        """计算主题相似度"""
        if not topics1 or not topics2:
            return 0.0
        
        # 提取主题名称和权重
        topic_names1 = {topic['topic']: topic.get('relevance', 1.0) for topic in topics1}
        topic_names2 = {topic['topic']: topic.get('relevance', 1.0) for topic in topics2}
        
        # 计算交集
        common_topics = set(topic_names1.keys()) & set(topic_names2.keys())
        
        if not common_topics:
            return 0.0
        
        # 加权相似度
        similarity_sum = 0.0
        weight_sum = 0.0
        
        for topic in common_topics:
            weight = min(topic_names1[topic], topic_names2[topic])
            similarity_sum += weight
            weight_sum += weight
        
        return similarity_sum / max(weight_sum, 1.0)
    
    def _calculate_entity_similarity(self, entities1: List[Dict], entities2: List[Dict]) -> float:
        """计算实体相似度"""
        if not entities1 or not entities2:
            return 0.0
        
        # 提取实体名称和重要性
        entity_names1 = {entity['name']: entity.get('importance', 1.0) for entity in entities1}
        entity_names2 = {entity['name']: entity.get('importance', 1.0) for entity in entities2}
        
        # 计算交集
        common_entities = set(entity_names1.keys()) & set(entity_names2.keys())
        
        if not common_entities:
            return 0.0
        
        # 加权相似度
        similarity_sum = 0.0
        weight_sum = 0.0
        
        for entity in common_entities:
            weight = min(entity_names1[entity], entity_names2[entity])
            similarity_sum += weight
            weight_sum += weight
        
        return similarity_sum / max(weight_sum, 1.0)
    
    async def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容文本相似度"""
        # 简化的文本相似度计算（实际应用中可以使用更复杂的算法）
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _analyze_temporal_order(self, temporal_info: List[Dict]) -> List[Dict]:
        """分析时间顺序"""
        temporal_order = []
        
        for i, info in enumerate(temporal_info):
            time_score = self._calculate_time_score(info)
            temporal_order.append({
                'content_index': i,
                'time_score': time_score,
                'time_period': info.get('time_period'),
                'specific_dates': info.get('specific_dates', [])
            })
        
        # 按时间分数排序
        temporal_order.sort(key=lambda x: x['time_score'])
        
        return temporal_order
    
    def _calculate_time_score(self, temporal_info: Dict) -> float:
        """计算时间分数用于排序"""
        if not temporal_info:
            return 0.0
        
        # 从时间信息中提取数字年份
        dates = temporal_info.get('specific_dates', [])
        if dates:
            years = []
            for date in dates:
                # 提取年份数字
                year_match = re.search(r'(\d{3,4})', str(date))
                if year_match:
                    years.append(int(year_match.group(1)))
            
            if years:
                return float(min(years))  # 使用最早的年份
        
        # 如果没有具体日期，尝试从时间段推断
        time_period = temporal_info.get('time_period', '')
        if time_period:
            return self._estimate_period_score(time_period)
        
        return 0.0
    
    def _estimate_period_score(self, time_period: str) -> float:
        """从时间段估算分数"""
        # 历史朝代和时期的大致时间映射
        period_scores = {
            '夏朝': 2000, '商朝': 1600, '周朝': 1046,
            '春秋': 770, '战国': 475, '秦朝': 221,
            '汉朝': 206, '三国': 220, '晋朝': 265,
            '南北朝': 420, '隋朝': 581, '唐朝': 618,
            '五代': 907, '宋朝': 960, '元朝': 1271,
            '明朝': 1368, '清朝': 1644, '近代': 1840,
            '现代': 1949
        }
        
        for period, score in period_scores.items():
            if period in time_period:
                return float(score)
        
        return 0.0
    
    async def _analyze_topic_overlaps(self, topics_list: List[List[Dict]]) -> List[Dict]:
        """分析主题重叠"""
        topic_contents = defaultdict(list)
        
        # 收集每个主题涉及的内容
        for content_idx, topics in enumerate(topics_list):
            for topic in topics:
                topic_name = topic['topic']
                topic_contents[topic_name].append({
                    'content_index': content_idx,
                    'relevance': topic.get('relevance', 1.0)
                })
        
        # 计算重叠
        overlaps = []
        for topic_name, content_list in topic_contents.items():
            if len(content_list) > 1:  # 至少在2个内容中出现
                overlap_score = sum(item['relevance'] for item in content_list) / len(content_list)
                overlaps.append({
                    'topic': topic_name,
                    'contents': [item['content_index'] for item in content_list],
                    'overlap_score': overlap_score
                })
        
        # 按重叠分数排序
        overlaps.sort(key=lambda x: x['overlap_score'], reverse=True)
        
        return overlaps
    
    async def _analyze_entity_connections(self, entities_list: List[List[Dict]]) -> List[Dict]:
        """分析实体连接"""
        entity_contents = defaultdict(list)
        
        # 收集每个实体涉及的内容
        for content_idx, entities in enumerate(entities_list):
            for entity in entities:
                entity_name = entity['name']
                entity_contents[entity_name].append({
                    'content_index': content_idx,
                    'importance': entity.get('importance', 1.0)
                })
        
        # 计算连接强度
        connections = []
        for entity_name, content_list in entity_contents.items():
            if len(content_list) > 1:  # 至少在2个内容中出现
                connection_strength = sum(item['importance'] for item in content_list) / len(content_list)
                connections.append({
                    'entity': entity_name,
                    'contents': [item['content_index'] for item in content_list],
                    'connection_strength': connection_strength
                })
        
        # 按连接强度排序
        connections.sort(key=lambda x: x['connection_strength'], reverse=True)
        
        return connections


class BaseMerger:
    """合并策略基类"""
    
    def __init__(self, ai_client: AIServiceClient):
        self.ai_client = ai_client
    
    async def create_merge_plan(self, contents: List[ContentItem], 
                              analyses: List[Dict], relationships: Dict,
                              request: MergeRequest) -> Dict:
        """创建合并计划（子类需要实现）"""
        raise NotImplementedError
    
    async def execute_merge(self, plan: Dict, request: MergeRequest, 
                          ai_client: AIServiceClient) -> Dict:
        """执行合并（子类需要实现）"""
        raise NotImplementedError


class TimelineMerger(BaseMerger):
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
        sorted_contents = self._sort_by_timeline(contents, temporal_order)
        
        # 创建时间段分组
        time_groups = self._group_by_time_period(sorted_contents, analyses, temporal_order)
        
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
    
    def _sort_by_timeline(self, contents: List[ContentItem], 
                         temporal_order: List[Dict]) -> List[ContentItem]:
        """按时间线排序内容"""
        if not temporal_order:
            return contents
        
        # 创建索引映射
        order_mapping = {item['content_index']: item['time_score'] for item in temporal_order}
        
        # 排序
        indexed_contents = [(i, content) for i, content in enumerate(contents)]
        indexed_contents.sort(key=lambda x: order_mapping.get(x[0], 0))
        
        return [content for _, content in indexed_contents]
    
    def _group_by_time_period(self, contents: List[ContentItem], 
                            analyses: List[Dict],
                            temporal_order: List[Dict]) -> Dict[str, List[ContentItem]]:
        """按时间段分组内容"""
        time_groups = defaultdict(list)
        
        for i, content in enumerate(contents):
            # 获取时间信息
            temporal_info = None
            for order_item in temporal_order:
                if order_item['content_index'] == i:
                    temporal_info = order_item
                    break
            
            if temporal_info and temporal_info.get('time_period'):
                time_period = temporal_info['time_period']
            else:
                # 使用默认分组
                time_period = "其他时期"
            
            time_groups[time_period].append(content)
        
        return dict(time_groups)
    
    def _create_temporal_transitions(self, sections: List[Dict]) -> List[Dict]:
        """创建时间顺序的过渡"""
        transitions = []
        
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            transition = {
                'from_section': i,
                'to_section': i + 1,
                'transition_text': f"随着时间的推移，从{current_section['time_period']}进入{next_section['time_period']}。",
                'transition_type': 'temporal'
            }
            transitions.append(transition)
        
        return transitions
    
    async def execute_merge(self, plan: Dict, request: MergeRequest, 
                          ai_client: AIServiceClient) -> Dict:
        """执行时间线合并"""
        merged_sections = []
        total_tokens = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        
        for section in plan['sections']:
            # 获取该时间段的内容
            section_contents = [
                c for c in request.source_contents 
                if c.id in section['contents']
            ]
            
            # 生成该时间段的合并内容
            section_content, tokens = await self._merge_temporal_section(
                section_contents, section, ai_client
            )
            
            # 累计令牌使用
            if tokens:
                total_tokens.prompt_tokens += tokens.prompt_tokens
                total_tokens.completion_tokens += tokens.completion_tokens
                total_tokens.total_tokens += tokens.total_tokens
            
            merged_sections.append(MergeSection(
                title=section['title'],
                content=section_content,
                source_contents=section['contents'],
                merge_type=section['merge_type'],
                word_count=len(section_content)
            ))
        
        # 组合所有章节
        full_content = await self._combine_sections(
            merged_sections, plan['transitions'], ai_client
        )
        
        # 生成标题
        title = await ai_client.generate_title(full_content[:1000], "historical")
        
        return {
            'title': title,
            'content': full_content,
            'structure': {
                'sections': merged_sections,
                'transitions': plan['transitions']
            },
            'token_usage': total_tokens
        }
    
    async def _merge_temporal_section(self, contents: List[ContentItem], 
                                    section: Dict, ai_client: AIServiceClient) -> Tuple[str, Optional[TokenUsage]]:
        """合并单个时间段的内容"""
        if len(contents) == 1:
            return contents[0].content, None
        
        # 提取内容文本
        content_texts = [content.content for content in contents]
        
        # 使用AI服务进行章节合并
        result = await ai_client.generate_section_merge(
            content_texts, section['time_period'], "chronological"
        )
        
        return result, None  # 这里应该返回实际的token usage
    
    async def _combine_sections(self, sections: List[MergeSection], 
                              transitions: List[Dict], 
                              ai_client: AIServiceClient) -> str:
        """组合所有章节"""
        combined_content = ""
        
        for i, section in enumerate(sections):
            combined_content += f"\n\n## {section.title}\n\n{section.content}"
            
            # 添加过渡
            if i < len(sections) - 1:
                # 查找对应的过渡
                transition_text = ""
                for transition in transitions:
                    if transition['from_section'] == i:
                        transition_text = transition['transition_text']
                        break
                
                if transition_text:
                    combined_content += f"\n\n{transition_text}"
        
        return combined_content.strip()


class TopicMerger(BaseMerger):
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
        for content_idx, analysis in enumerate(analyses):
            for topic in analysis.get('topics', []):
                all_topics.append({
                    'name': topic['topic'],
                    'relevance': topic.get('relevance', 1.0),
                    'content_index': content_idx
                })
        
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
    
    def _cluster_topics(self, topics: List[Dict]) -> List[Dict]:
        """对主题进行聚类"""
        # 简化的主题聚类实现
        topic_groups = defaultdict(list)
        
        for topic in topics:
            topic_groups[topic['name']].append(topic)
        
        clusters = []
        for topic_name, topic_list in topic_groups.items():
            # 计算重要性分数
            importance_score = sum(t['relevance'] for t in topic_list) / len(topic_list)
            
            # 获取相关内容
            related_contents = list(set(t['content_index'] for t in topic_list))
            
            clusters.append({
                'main_topic': topic_name,
                'sub_topics': [topic_name],  # 简化实现
                'related_contents': related_contents,
                'importance_score': importance_score
            })
        
        # 按重要性排序
        clusters.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return clusters
    
    async def execute_merge(self, plan: Dict, request: MergeRequest, 
                          ai_client: AIServiceClient) -> Dict:
        """执行主题合并"""
        topic_sections = []
        
        # 按重要性排序主题
        sorted_topics = sorted(plan['topics'], 
                              key=lambda x: x['importance_score'], 
                              reverse=True)
        
        for topic in sorted_topics:
            # 获取该主题相关的内容
            related_contents = [
                c for i, c in enumerate(request.source_contents)
                if i in topic['related_contents']
            ]
            
            # 生成主题章节
            section_content = await self._merge_topic_section(
                related_contents, topic, ai_client
            )
            
            topic_sections.append(MergeSection(
                title=topic['main_topic'],
                content=section_content,
                source_contents=[c.id for c in related_contents],
                merge_type='topic',
                word_count=len(section_content)
            ))
        
        # 组合所有主题章节
        full_content = await self._combine_topic_sections(
            topic_sections, ai_client
        )
        
        # 生成标题
        main_topics = [topic['main_topic'] for topic in sorted_topics[:3]]
        title = f"关于{' · '.join(main_topics)}的综述"
        
        return {
            'title': title,
            'content': full_content,
            'structure': {
                'sections': topic_sections,
                'topic_hierarchy': plan['topics']
            }
        }
    
    async def _merge_topic_section(self, contents: List[ContentItem], 
                                 topic: Dict, ai_client: AIServiceClient) -> str:
        """合并主题章节"""
        if len(contents) == 1:
            return contents[0].content
        
        content_texts = [content.content for content in contents]
        
        return await ai_client.generate_section_merge(
            content_texts, topic['main_topic'], "thematic"
        )
    
    async def _combine_topic_sections(self, sections: List[MergeSection], 
                                    ai_client: AIServiceClient) -> str:
        """组合主题章节"""
        combined_content = ""
        
        for section in sections:
            combined_content += f"\n\n## {section.title}\n\n{section.content}"
        
        return combined_content.strip()


# 其他合并策略的简化实现
class HierarchyMerger(TopicMerger):
    """层次整合器（继承主题合并器的基础功能）"""
    pass

class LogicMerger(TimelineMerger):
    """逻辑关系构建器（继承时间线合并器的基础功能）"""
    pass

class SupplementMerger(BaseMerger):
    """补充扩展器"""
    
    async def create_merge_plan(self, contents: List[ContentItem],
                              analyses: List[Dict], relationships: Dict,
                              request: MergeRequest) -> Dict:
        """创建补充合并计划"""
        return {
            'strategy': 'supplement',
            'main_content': 0,  # 使用第一个内容作为主要内容
            'supplements': list(range(1, len(contents)))
        }
    
    async def execute_merge(self, plan: Dict, request: MergeRequest, 
                          ai_client: AIServiceClient) -> Dict:
        """执行补充合并"""
        main_content = request.source_contents[plan['main_content']]
        supplement_contents = [
            request.source_contents[i] for i in plan['supplements']
        ]
        
        # 使用主要内容作为基础，补充其他内容
        all_texts = [main_content.content] + [c.content for c in supplement_contents]
        
        merged_content = await ai_client.generate_content_merge(
            all_texts, "supplement", request.mode.value
        )
        
        return {
            'title': merged_content.get('title', '补充合并内容'),
            'content': merged_content['content'],
            'structure': {
                'sections': [MergeSection(
                    title="补充合并内容",
                    content=merged_content['content'],
                    source_contents=[c.id for c in request.source_contents],
                    merge_type='supplement',
                    word_count=len(merged_content['content'])
                )],
                'transitions': []
            }
        }