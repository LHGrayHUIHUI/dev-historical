"""
知识图谱构建服务核心实现
实现实体识别、关系抽取、图谱构建等核心功能
无状态架构设计，专注于算法实现
"""

import asyncio
import time
import uuid
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import json

import spacy
import jieba
import jieba.posseg as pseg
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from gensim import corpora, models
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import pandas as pd
from loguru import logger

from ..config.settings import settings
from ..schemas.knowledge_graph_schemas import (
    EntityType, RelationType, ExtractionMethod,
    ExtractedEntity, ExtractedRelation, GraphNode, GraphEdge
)
from ..clients.storage_client import storage_client


class KnowledgeGraphService:
    """知识图谱构建服务核心类"""
    
    def __init__(self):
        self.nlp_models = {}
        self.ner_models = {}
        self.embedding_models = {}
        self.vectorizers = {}
        self.initialized = False
        
        # 实体类型映射
        self.entity_type_mapping = {
            'PERSON': '人物',
            'PER': '人物',
            'ORG': '组织',
            'GPE': '地理政治实体',
            'LOC': '地点',
            'LOCATION': '地点',
            'EVENT': '事件',
            'DATE': '时间',
            'TIME': '时间',
            'MONEY': '货币',
            'PERCENT': '百分比',
            'CARDINAL': '基数',
            'ORDINAL': '序数',
            'WORK_OF_ART': '作品',
            'PRODUCT': '物品'
        }
        
        # 中文关系模式
        self.relation_patterns = {
            r'(.+?)出生于(.+?)': ('出生于', 'PERSON', 'LOCATION'),
            r'(.+?)死于(.+?)': ('死于', 'PERSON', 'LOCATION'),
            r'(.+?)任职于(.+?)': ('任职于', 'PERSON', 'ORGANIZATION'),
            r'(.+?)在(.+?)任职': ('任职于', 'PERSON', 'ORGANIZATION'),
            r'(.+?)位于(.+?)': ('位于', 'LOCATION', 'LOCATION'),
            r'(.+?)创建了(.+?)': ('创建', 'PERSON', 'ORGANIZATION'),
            r'(.+?)建立了(.+?)': ('创建', 'PERSON', 'ORGANIZATION'),
            r'(.+?)统治(.+?)': ('统治', 'PERSON', 'LOCATION'),
            r'(.+?)的(.+?)': ('属于', 'ANY', 'ANY'),
            r'(.+?)影响了(.+?)': ('影响', 'PERSON', 'PERSON'),
            r'(.+?)师从(.+?)': ('师从', 'PERSON', 'PERSON'),
            r'(.+?)继承了(.+?)': ('继承', 'PERSON', 'ANY'),
            r'(.+?)参与(.+?)': ('参与', 'PERSON', 'EVENT'),
            r'(.+?)包含(.+?)': ('包含', 'ANY', 'ANY')
        }
        
        # 停用词
        self.stop_words = {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '或',
            '这', '那', '这个', '那个', '什么', '怎么', '为什么', '哪里'
        }
    
    async def initialize(self):
        """初始化NLP模型"""
        if self.initialized:
            return
        
        try:
            logger.info("正在初始化知识图谱服务...")
            
            # 初始化spaCy模型
            await self._load_spacy_models()
            
            # 初始化BERT模型（延迟加载）
            self.bert_model_name = settings.bert_model_name
            
            # 初始化句子transformer模型（延迟加载）
            self.sentence_transformer_name = settings.sentence_transformer_model
            
            # 初始化jieba
            jieba.initialize()
            
            self.initialized = True
            logger.info("知识图谱服务初始化完成")
            
        except Exception as e:
            logger.error(f"初始化知识图谱服务失败: {str(e)}")
            raise
    
    async def _load_spacy_models(self):
        """加载spaCy模型"""
        try:
            # 加载中文模型
            try:
                self.nlp_models['zh'] = spacy.load(settings.spacy_model_zh)
            except OSError:
                logger.warning(f"spaCy中文模型 {settings.spacy_model_zh} 未找到，使用基础模型")
                self.nlp_models['zh'] = None
            
            # 加载英文模型
            try:
                self.nlp_models['en'] = spacy.load(settings.spacy_model_en)
            except OSError:
                logger.warning(f"spaCy英文模型 {settings.spacy_model_en} 未找到，使用基础模型")
                self.nlp_models['en'] = None
                
        except Exception as e:
            logger.error(f"加载spaCy模型失败: {str(e)}")
            self.nlp_models = {'zh': None, 'en': None}
    
    def _get_bert_model(self):
        """获取BERT模型（懒加载）"""
        if 'bert' not in self.ner_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                model = AutoModelForTokenClassification.from_pretrained(self.bert_model_name)
                self.ner_models['bert'] = pipeline(
                    'ner',
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy='simple'
                )
            except Exception as e:
                logger.error(f"加载BERT模型失败: {str(e)}")
                self.ner_models['bert'] = None
        
        return self.ner_models['bert']
    
    def _get_sentence_transformer(self):
        """获取句子transformer模型（懒加载）"""
        if 'sentence_transformer' not in self.embedding_models:
            try:
                self.embedding_models['sentence_transformer'] = SentenceTransformer(
                    self.sentence_transformer_name
                )
            except Exception as e:
                logger.error(f"加载Sentence Transformer模型失败: {str(e)}")
                self.embedding_models['sentence_transformer'] = None
        
        return self.embedding_models['sentence_transformer']
    
    # ============ 项目管理 ============
    
    async def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        domain: str = "历史",
        language: str = "zh",
        entity_types: List[str] = None,
        relation_types: List[str] = None,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建知识图谱项目"""
        try:
            # 通过storage-service创建项目
            result = await storage_client.create_knowledge_graph_project(
                name=name,
                description=description,
                domain=domain,
                language=language,
                entity_types=entity_types or settings.supported_entity_types,
                relation_types=relation_types or settings.supported_relation_types,
                created_by=created_by
            )
            
            return {
                "success": True,
                "project_id": result.get("data", {}).get("id"),
                "message": "知识图谱项目创建成功"
            }
            
        except Exception as e:
            logger.error(f"创建知识图谱项目失败: {str(e)}")
            return {
                "success": False,
                "message": f"创建失败: {str(e)}"
            }
    
    # ============ 实体抽取 ============
    
    async def extract_entities(
        self,
        project_id: str,
        text: str,
        document_id: Optional[str] = None,
        method: ExtractionMethod = ExtractionMethod.SPACY_NER,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """从文本中抽取实体"""
        await self.initialize()
        
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        try:
            # 创建抽取任务记录
            await storage_client.create_entity_extraction_task(
                project_id=project_id,
                document_id=document_id,
                text_content=text,
                extraction_method=method.value,
                extraction_config=config or {}
            )
            
            # 更新任务状态为处理中
            await storage_client.update_entity_extraction_task(
                task_id=task_id,
                status="processing"
            )
            
            # 根据方法选择抽取器
            if method == ExtractionMethod.SPACY_NER:
                entities = await self._extract_entities_spacy(text, config)
            elif method == ExtractionMethod.BERT_NER:
                entities = await self._extract_entities_bert(text, config)
            elif method == ExtractionMethod.JIEBA_NER:
                entities = await self._extract_entities_jieba(text, config)
            else:
                raise ValueError(f"不支持的抽取方法: {method}")
            
            # 保存实体到storage-service
            saved_entities = []
            for entity_data in entities:
                try:
                    result = await storage_client.save_entity(
                        project_id=project_id,
                        entity_data=entity_data,
                        source_document_id=document_id
                    )
                    if result.get("success"):
                        saved_entities.append(entity_data)
                except Exception as e:
                    logger.warning(f"保存实体失败: {str(e)}")
            
            processing_time = time.time() - start_time
            
            # 更新任务状态为完成
            await storage_client.update_entity_extraction_task(
                task_id=task_id,
                status="completed",
                entities_found=len(saved_entities),
                processing_time=processing_time
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "entities_found": len(saved_entities),
                "entities": saved_entities,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"实体抽取失败: {str(e)}")
            
            # 更新任务状态为失败
            await storage_client.update_entity_extraction_task(
                task_id=task_id,
                status="failed",
                error_message=str(e)
            )
            
            return {
                "success": False,
                "message": f"实体抽取失败: {str(e)}",
                "task_id": task_id
            }
    
    async def _extract_entities_spacy(
        self,
        text: str,
        config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """使用spaCy进行实体抽取"""
        # 选择语言模型
        language = self._detect_language(text)
        nlp = self.nlp_models.get(language)
        
        if nlp is None:
            logger.warning(f"spaCy模型 {language} 不可用，回退到jieba")
            return await self._extract_entities_jieba(text, config)
        
        entities = []
        confidence_threshold = config.get('confidence_threshold', settings.entity_confidence_threshold) if config else settings.entity_confidence_threshold
        
        try:
            # 处理文本
            doc = nlp(text)
            
            for ent in doc.ents:
                # 过滤低置信度和短实体
                confidence = getattr(ent, 'confidence', 0.9)
                if confidence < confidence_threshold or len(ent.text.strip()) < 2:
                    continue
                
                entity_data = {
                    "name": ent.text.strip(),
                    "entity_type": ent.label_,
                    "start_pos": ent.start_char,
                    "end_pos": ent.end_char,
                    "confidence_score": confidence,
                    "context": text[max(0, ent.start_char-30):ent.end_char+30]
                }
                
                # 标准化实体
                normalized_entity = self._normalize_entity(entity_data)
                if normalized_entity:
                    entities.append(normalized_entity)
        
        except Exception as e:
            logger.error(f"spaCy实体抽取失败: {str(e)}")
            return []
        
        return entities
    
    async def _extract_entities_bert(
        self,
        text: str,
        config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """使用BERT进行实体抽取"""
        ner_model = self._get_bert_model()
        if ner_model is None:
            logger.warning("BERT模型不可用，回退到spaCy")
            return await self._extract_entities_spacy(text, config)
        
        entities = []
        confidence_threshold = config.get('confidence_threshold', settings.entity_confidence_threshold) if config else settings.entity_confidence_threshold
        
        try:
            # 分段处理长文本
            max_length = 512
            segments = [text[i:i+max_length] for i in range(0, len(text), max_length-50)]
            
            for segment_idx, segment in enumerate(segments):
                try:
                    results = ner_model(segment)
                    
                    for result in results:
                        if result['score'] < confidence_threshold:
                            continue
                        
                        # 计算在原文本中的位置
                        start_pos = segment_idx * (max_length - 50) + result['start']
                        end_pos = segment_idx * (max_length - 50) + result['end']
                        
                        entity_data = {
                            "name": result['word'].strip(),
                            "entity_type": result['entity_group'],
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                            "confidence_score": result['score'],
                            "context": text[max(0, start_pos-30):end_pos+30]
                        }
                        
                        # 标准化实体
                        normalized_entity = self._normalize_entity(entity_data)
                        if normalized_entity:
                            entities.append(normalized_entity)
                            
                except Exception as e:
                    logger.warning(f"处理文本段落失败: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"BERT实体抽取失败: {str(e)}")
            return []
        
        return entities
    
    async def _extract_entities_jieba(
        self,
        text: str,
        config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """使用jieba进行实体抽取"""
        entities = []
        
        try:
            # 词性标注
            words = list(pseg.cut(text))
            
            current_pos = 0
            
            # 词性到实体类型的映射
            pos_to_entity_type = {
                'nr': 'PERSON',      # 人名
                'ns': 'LOCATION',    # 地名
                'nt': 'ORGANIZATION', # 机构名
                'nz': 'CONCEPT',     # 其他专名
                't': 'TIME',         # 时间
                'm': 'CARDINAL'      # 数量
            }
            
            for word, pos in words:
                word_length = len(word)
                
                if pos in pos_to_entity_type and word_length >= 2 and word not in self.stop_words:
                    entity_data = {
                        "name": word.strip(),
                        "entity_type": pos_to_entity_type[pos],
                        "start_pos": current_pos,
                        "end_pos": current_pos + word_length,
                        "confidence_score": 0.8,  # jieba默认置信度
                        "context": text[max(0, current_pos-30):current_pos+word_length+30]
                    }
                    
                    # 标准化实体
                    normalized_entity = self._normalize_entity(entity_data)
                    if normalized_entity:
                        entities.append(normalized_entity)
                
                current_pos += word_length
        
        except Exception as e:
            logger.error(f"jieba实体抽取失败: {str(e)}")
            return []
        
        return entities
    
    def _normalize_entity(self, entity_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """实体标准化"""
        try:
            name = entity_data.get("name", "").strip()
            
            # 清理实体名称
            name = re.sub(r'[\s\n\r\t]+', '', name)
            name = re.sub(r'^[的地得了]', '', name)
            name = re.sub(r'[的地得了]$', '', name)
            
            # 过滤无效实体
            if (len(name) < settings.min_entity_length or 
                len(name) > settings.max_entity_length or 
                name.isdigit() or 
                name in self.stop_words):
                return None
            
            entity_data["name"] = name
            
            # 标准化实体类型
            entity_type = entity_data.get("entity_type", "")
            if entity_type in self.entity_type_mapping:
                entity_data["entity_type_zh"] = self.entity_type_mapping[entity_type]
            
            return entity_data
        
        except Exception as e:
            logger.warning(f"实体标准化失败: {str(e)}")
            return None
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        try:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            chinese_ratio = chinese_chars / len(text) if text else 0
            return 'zh' if chinese_ratio > 0.5 else 'en'
        except:
            return 'zh'
    
    # ============ 关系抽取 ============
    
    async def extract_relations(
        self,
        project_id: str,
        text: str,
        document_id: Optional[str] = None,
        method: ExtractionMethod = ExtractionMethod.PATTERN_BASED,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """从文本中抽取关系"""
        await self.initialize()
        
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        try:
            # 创建关系抽取任务记录
            await storage_client.create_relation_extraction_task(
                project_id=project_id,
                document_id=document_id,
                text_content=text,
                extraction_method=method.value,
                extraction_config=config or {}
            )
            
            # 先抽取实体
            entities_result = await self.extract_entities(
                project_id=project_id,
                text=text,
                document_id=document_id,
                method=ExtractionMethod.SPACY_NER,
                config=config
            )
            
            entities = entities_result.get("entities", [])
            
            # 根据方法抽取关系
            if method == ExtractionMethod.PATTERN_BASED:
                relations = self._extract_relations_pattern_based(text, entities, config)
            elif method == ExtractionMethod.DEPENDENCY_PARSING:
                relations = self._extract_relations_dependency_parsing(text, entities, config)
            else:
                relations = self._extract_relations_pattern_based(text, entities, config)
            
            # 保存关系到storage-service
            saved_relations = []
            for relation_data in relations:
                try:
                    # 构造关系数据
                    relation_record = {
                        "subject_entity_id": relation_data["subject_entity"].get("id"),
                        "predicate": relation_data["predicate"],
                        "object_entity_id": relation_data["object_entity"].get("id"),
                        "confidence_score": relation_data["confidence_score"],
                        "context": relation_data.get("context", ""),
                        "source_sentence": relation_data.get("source_sentence", "")
                    }
                    
                    result = await storage_client.save_relation(
                        project_id=project_id,
                        relation_data=relation_record,
                        source_document_id=document_id
                    )
                    
                    if result.get("success"):
                        saved_relations.append(relation_data)
                        
                except Exception as e:
                    logger.warning(f"保存关系失败: {str(e)}")
            
            processing_time = time.time() - start_time
            
            # 更新任务状态
            await storage_client.update_relation_extraction_task(
                task_id=task_id,
                status="completed",
                relations_found=len(saved_relations),
                processing_time=processing_time
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "relations_found": len(saved_relations),
                "relations": saved_relations,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"关系抽取失败: {str(e)}")
            
            await storage_client.update_relation_extraction_task(
                task_id=task_id,
                status="failed",
                error_message=str(e)
            )
            
            return {
                "success": False,
                "message": f"关系抽取失败: {str(e)}",
                "task_id": task_id
            }
    
    def _extract_relations_pattern_based(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """基于模式的关系抽取"""
        relations = []
        
        try:
            # 创建实体名称到实体的映射
            entity_map = {entity["name"]: entity for entity in entities}
            
            for pattern, (relation_type, subj_type, obj_type) in self.relation_patterns.items():
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    try:
                        subject_name = match.group(1).strip()
                        object_name = match.group(2).strip()
                        
                        # 查找对应的实体
                        subject_entity = entity_map.get(subject_name)
                        object_entity = entity_map.get(object_name)
                        
                        if subject_entity and object_entity:
                            # 检查实体类型是否匹配
                            if ((subj_type == 'ANY' or subject_entity.get('entity_type') == subj_type) and
                                (obj_type == 'ANY' or object_entity.get('entity_type') == obj_type)):
                                
                                relation = {
                                    "subject_entity": subject_entity,
                                    "predicate": relation_type,
                                    "object_entity": object_entity,
                                    "confidence_score": 0.8,
                                    "context": match.group(0),
                                    "source_sentence": self._extract_sentence(text, match.start())
                                }
                                
                                relations.append(relation)
                                
                    except Exception as e:
                        logger.warning(f"处理关系模式匹配失败: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"模式关系抽取失败: {str(e)}")
        
        return relations
    
    def _extract_relations_dependency_parsing(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """基于依存句法分析的关系抽取"""
        # 简化版本，主要使用模式匹配
        return self._extract_relations_pattern_based(text, entities, config)
    
    def _extract_sentence(self, text: str, position: int) -> str:
        """提取包含指定位置的句子"""
        try:
            sentence_end_chars = ['。', '！', '？', '.', '!', '?', '\n']
            
            # 向前查找句子开始
            start = position
            while start > 0 and text[start-1] not in sentence_end_chars:
                start -= 1
            
            # 向后查找句子结束
            end = position
            while end < len(text) and text[end] not in sentence_end_chars:
                end += 1
            
            if end < len(text):
                end += 1  # 包含句号
            
            return text[start:end].strip()
        except:
            return text[max(0, position-50):position+50]
    
    # ============ 批量处理 ============
    
    async def batch_extract(
        self,
        project_id: str,
        documents: List[Dict[str, str]],
        entity_extraction: bool = True,
        relation_extraction: bool = True,
        entity_method: ExtractionMethod = ExtractionMethod.SPACY_NER,
        relation_method: ExtractionMethod = ExtractionMethod.PATTERN_BASED,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """批量实体和关系抽取"""
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        processed_documents = 0
        failed_documents = 0
        total_entities = 0
        total_relations = 0
        
        try:
            for doc in documents:
                try:
                    document_id = doc.get("id")
                    text = doc.get("text", "")
                    
                    if not text.strip():
                        failed_documents += 1
                        continue
                    
                    # 实体抽取
                    if entity_extraction:
                        entity_result = await self.extract_entities(
                            project_id=project_id,
                            text=text,
                            document_id=document_id,
                            method=entity_method,
                            config=config
                        )
                        
                        if entity_result.get("success"):
                            total_entities += entity_result.get("entities_found", 0)
                    
                    # 关系抽取
                    if relation_extraction:
                        relation_result = await self.extract_relations(
                            project_id=project_id,
                            text=text,
                            document_id=document_id,
                            method=relation_method,
                            config=config
                        )
                        
                        if relation_result.get("success"):
                            total_relations += relation_result.get("relations_found", 0)
                    
                    processed_documents += 1
                    
                except Exception as e:
                    logger.warning(f"处理文档 {doc.get('id')} 失败: {str(e)}")
                    failed_documents += 1
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "batch_id": batch_id,
                "total_documents": len(documents),
                "processed_documents": processed_documents,
                "failed_documents": failed_documents,
                "total_entities": total_entities,
                "total_relations": total_relations,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"批量抽取失败: {str(e)}")
            return {
                "success": False,
                "message": f"批量抽取失败: {str(e)}",
                "batch_id": batch_id
            }
    
    # ============ 图谱构建 ============
    
    async def construct_graph(
        self,
        project_id: str,
        include_entities: bool = True,
        include_relations: bool = True,
        min_confidence: float = 0.7,
        max_nodes: int = 1000,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """构建知识图谱"""
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 创建图谱构建任务
            construction_config = {
                "include_entities": include_entities,
                "include_relations": include_relations,
                "min_confidence": min_confidence,
                "max_nodes": max_nodes,
                **(config or {})
            }
            
            await storage_client.create_graph_construction_task(
                project_id=project_id,
                construction_config=construction_config
            )
            
            nodes_count = 0
            edges_count = 0
            
            # 获取实体作为节点
            if include_entities:
                entities_result = await storage_client.list_entities(
                    project_id=project_id,
                    limit=max_nodes
                )
                
                if entities_result.get("success"):
                    entities = entities_result.get("data", [])
                    # 过滤低置信度实体
                    filtered_entities = [
                        e for e in entities 
                        if e.get("confidence_score", 0) >= min_confidence
                    ]
                    nodes_count = len(filtered_entities)
            
            # 获取关系作为边
            if include_relations:
                relations_result = await storage_client.list_relations(
                    project_id=project_id,
                    limit=max_nodes * 5  # 通常关系数比实体数多
                )
                
                if relations_result.get("success"):
                    relations = relations_result.get("data", [])
                    # 过滤低置信度关系
                    filtered_relations = [
                        r for r in relations 
                        if r.get("confidence_score", 0) >= min_confidence
                    ]
                    edges_count = len(filtered_relations)
            
            processing_time = time.time() - start_time
            
            # 更新构建任务状态
            await storage_client.update_graph_construction_task(
                task_id=task_id,
                status="completed",
                nodes_count=nodes_count,
                edges_count=edges_count,
                processing_time=processing_time
            )
            
            return {
                "success": True,
                "task_id": task_id,
                "nodes_count": nodes_count,
                "edges_count": edges_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"图谱构建失败: {str(e)}")
            
            await storage_client.update_graph_construction_task(
                task_id=task_id,
                status="failed",
                error_message=str(e)
            )
            
            return {
                "success": False,
                "message": f"图谱构建失败: {str(e)}",
                "task_id": task_id
            }
    
    # ============ 图谱查询 ============
    
    async def query_graph(
        self,
        project_id: str,
        query_type: str,
        parameters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """查询知识图谱"""
        start_time = time.time()
        
        try:
            query_params = {
                "query_type": query_type,
                "limit": limit,
                "offset": offset,
                **(parameters or {})
            }
            
            result = await storage_client.query_knowledge_graph(
                project_id=project_id,
                query_type=query_type,
                query_params=query_params
            )
            
            query_time = time.time() - start_time
            
            if result.get("success"):
                data = result.get("data", {})
                return {
                    "success": True,
                    "nodes": data.get("nodes", []),
                    "edges": data.get("edges", []),
                    "total_count": data.get("total_count", 0),
                    "query_time": query_time
                }
            else:
                return {
                    "success": False,
                    "message": result.get("message", "查询失败"),
                    "query_time": query_time
                }
                
        except Exception as e:
            logger.error(f"图谱查询失败: {str(e)}")
            return {
                "success": False,
                "message": f"查询失败: {str(e)}",
                "query_time": time.time() - start_time
            }


    async def cleanup(self):
        """清理资源"""
        try:
            self.logger.info("清理知识图谱服务资源...")
            # 清理缓存等资源
            if hasattr(self, 'cache'):
                self.cache.clear()
            self.logger.info("知识图谱服务资源清理完成")
        except Exception as e:
            self.logger.error(f"资源清理失败: {str(e)}")
    
    async def mine_concepts(self, documents: List[str], num_topics: int = 10, 
                          min_frequency: int = 3, language: str = "zh") -> Dict[str, Any]:
        """
        概念挖掘：从文档集合中发现主要概念和主题
        """
        try:
            start_time = time.time()
            self.logger.info(f"开始概念挖掘，文档数量: {len(documents)}")
            
            # 文本预处理
            processed_docs = []
            for doc in documents:
                if language == "zh":
                    # 中文分词和去停用词
                    words = jieba.cut(doc)
                    processed_words = [w for w in words if len(w) > 1 and w not in self._get_stopwords()]
                    processed_docs.append(" ".join(processed_words))
                else:
                    # 英文处理
                    processed_docs.append(doc.lower())
            
            # 使用LDA主题模型
            from gensim import corpora, models
            from gensim.models import LdaModel
            
            # 创建词典和语料库
            texts = [doc.split() for doc in processed_docs]
            dictionary = corpora.Dictionary(texts)
            dictionary.filter_extremes(no_below=min_frequency, no_above=0.8)
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            # 训练LDA模型
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # 提取主题和概念
            topics = []
            concepts = []
            
            for topic_id in range(num_topics):
                topic_words = lda_model.show_topic(topic_id, topn=10)
                topic_info = {
                    "topic_id": topic_id,
                    "words": [{"word": word, "probability": prob} for word, prob in topic_words],
                    "coherence_score": 0.0  # 可以计算主题一致性
                }
                topics.append(topic_info)
                
                # 将高概率词作为概念
                for word, prob in topic_words[:5]:  # 取前5个词作为核心概念
                    concepts.append({
                        "concept": word,
                        "topic_id": topic_id,
                        "probability": prob,
                        "frequency": dictionary.cfs.get(dictionary.token2id.get(word, -1), 0)
                    })
            
            # 概念关系挖掘（基于共现）
            concept_relations = []
            concept_words = [c["concept"] for c in concepts]
            
            for i, word1 in enumerate(concept_words):
                for j, word2 in enumerate(concept_words[i+1:], i+1):
                    # 计算共现频率
                    cooccurrence = self._calculate_cooccurrence(word1, word2, processed_docs)
                    if cooccurrence > 0:
                        concept_relations.append({
                            "source_concept": word1,
                            "target_concept": word2,
                            "relation_type": "共现",
                            "strength": cooccurrence / len(processed_docs)
                        })
            
            processing_time = time.time() - start_time
            
            result = {
                "concepts": concepts,
                "topics": topics,
                "concept_relations": concept_relations[:100],  # 限制返回数量
                "processing_time": processing_time,
                "num_documents": len(documents)
            }
            
            self.logger.info(f"概念挖掘完成，发现{len(concepts)}个概念")
            return result
            
        except Exception as e:
            self.logger.error(f"概念挖掘失败: {str(e)}")
            raise
    
    async def get_project_statistics(self, project_id: str) -> Optional[Dict[str, Any]]:
        """获取项目统计信息"""
        try:
            # 通过storage-service获取项目统计
            response = await self.storage_client.get_project_statistics(project_id)
            return response
            
        except Exception as e:
            self.logger.error(f"获取项目统计失败: {str(e)}")
            return None
    
    async def get_batch_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取批量任务状态"""
        try:
            # 通过storage-service获取任务状态
            response = await self.storage_client.get_batch_task_status(task_id)
            return response
            
        except Exception as e:
            self.logger.error(f"获取任务状态失败: {str(e)}")
            return None
    
    async def check_models_status(self) -> Dict[str, str]:
        """检查NLP模型状态"""
        try:
            status = {}
            
            # 检查spaCy模型
            try:
                import spacy
                nlp = spacy.load(self.settings.spacy_model_zh)
                status["spacy_zh"] = "healthy"
            except Exception:
                status["spacy_zh"] = "unavailable"
            
            # 检查BERT模型
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.settings.bert_model_name)
                status["bert"] = "healthy"
            except Exception:
                status["bert"] = "unavailable"
            
            # 检查sentence-transformer模型
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.settings.sentence_transformer_model)
                status["sentence_transformer"] = "healthy"
            except Exception:
                status["sentence_transformer"] = "unavailable"
            
            return status
            
        except Exception as e:
            self.logger.error(f"模型状态检查失败: {str(e)}")
            return {"error": str(e)}
    
    def _get_stopwords(self) -> set:
        """获取中文停用词"""
        # 这里应该从文件加载，现在返回基本停用词
        return {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", 
            "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", 
            "没有", "看", "好", "自己", "这", "那", "以", "为", "能", "可以"
        }
    
    def _calculate_cooccurrence(self, word1: str, word2: str, documents: List[str]) -> int:
        """计算两个词的共现次数"""
        cooccurrence = 0
        for doc in documents:
            if word1 in doc and word2 in doc:
                cooccurrence += 1
        return cooccurrence


# 全局知识图谱服务实例
knowledge_graph_service = KnowledgeGraphService()