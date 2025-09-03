# Story 2.4: 知识图谱构建服务

## 基本信息
- **Story ID**: 2.4
- **Epic**: Epic 2 - 数据处理和智能分类微服务
- **标题**: 知识图谱构建服务
- **优先级**: 高
- **状态**: 待开发
- **预估工期**: 8-10天

## 用户故事
**作为** 知识管理专员  
**我希望** 有一个智能的知识图谱构建服务  
**以便** 从历史文档中自动提取实体、关系和概念，构建结构化的知识图谱，支持知识发现和智能检索

## 需求描述
开发专业的知识图谱构建服务，支持实体识别、关系抽取、概念挖掘、图谱构建、图谱查询等功能，为历史文档提供深度的知识组织和智能分析能力。

## 技术实现

### 核心技术栈
- **后端框架**: FastAPI 0.104+ (Python)
- **知识图谱**: 
  - Neo4j 5.13+ (图数据库)
  - py2neo 2021.2+ (Python Neo4j驱动)
  - networkx 3.2+ (图分析库)
- **自然语言处理**: 
  - spaCy 3.7+ (NLP核心库)
  - transformers 4.35+ (预训练模型)
  - jieba 0.42+ (中文分词)
  - HanLP 2.1+ (中文NLP)
- **实体识别**: 
  - BERT-NER (命名实体识别)
  - BiLSTM-CRF (序列标注)
  - LAC (百度词法分析)
- **关系抽取**: 
  - OpenIE (开放信息抽取)
  - Stanford CoreNLP (关系抽取)
  - 自定义关系抽取模型
- **概念挖掘**: 
  - Word2Vec (词向量)
  - BERT (语义表示)
  - LDA (主题模型)
- **数据库**: 
  - PostgreSQL (元数据存储)
  - Redis (缓存)
  - Elasticsearch (全文检索)
- **消息队列**: RabbitMQ 3.12+

### 数据模型设计

#### 知识图谱项目表 (knowledge_graph_projects)
```sql
CREATE TABLE knowledge_graph_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    domain VARCHAR(100), -- 领域：历史、文学、政治等
    language VARCHAR(10) DEFAULT 'zh', -- 语言
    status VARCHAR(50) DEFAULT 'active', -- active, building, completed, archived
    neo4j_database VARCHAR(100), -- Neo4j数据库名
    entity_types JSONB, -- 实体类型配置
    relation_types JSONB, -- 关系类型配置
    extraction_config JSONB, -- 抽取配置
    statistics JSONB, -- 图谱统计信息
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 实体抽取任务表 (entity_extraction_tasks)
```sql
CREATE TABLE entity_extraction_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES knowledge_graph_projects(id),
    document_id UUID REFERENCES documents(id),
    text_content TEXT NOT NULL,
    extraction_method VARCHAR(100), -- spacy_ner, bert_ner, custom_model
    extraction_config JSONB,
    status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    entities_found INTEGER DEFAULT 0,
    confidence_threshold FLOAT DEFAULT 0.8,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 实体表 (entities)
```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES knowledge_graph_projects(id),
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL, -- PERSON, LOCATION, ORGANIZATION, EVENT, CONCEPT等
    aliases JSONB, -- 别名列表
    description TEXT,
    properties JSONB, -- 实体属性
    confidence_score FLOAT,
    source_documents JSONB, -- 来源文档列表
    mention_count INTEGER DEFAULT 1,
    neo4j_node_id BIGINT, -- Neo4j节点ID
    embedding_vector VECTOR(768), -- 实体向量表示
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, name, entity_type)
);
```

#### 关系抽取任务表 (relation_extraction_tasks)
```sql
CREATE TABLE relation_extraction_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES knowledge_graph_projects(id),
    document_id UUID REFERENCES documents(id),
    text_content TEXT NOT NULL,
    extraction_method VARCHAR(100), -- openie, stanford_corenlp, custom_model
    extraction_config JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    relations_found INTEGER DEFAULT 0,
    confidence_threshold FLOAT DEFAULT 0.7,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 关系表 (relations)
```sql
CREATE TABLE relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES knowledge_graph_projects(id),
    subject_entity_id UUID REFERENCES entities(id),
    predicate VARCHAR(200) NOT NULL, -- 关系类型
    object_entity_id UUID REFERENCES entities(id),
    confidence_score FLOAT,
    context TEXT, -- 关系上下文
    source_sentence TEXT, -- 来源句子
    source_document_id UUID REFERENCES documents(id),
    properties JSONB, -- 关系属性
    neo4j_relationship_id BIGINT, -- Neo4j关系ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, subject_entity_id, predicate, object_entity_id)
);
```

#### 概念挖掘任务表 (concept_mining_tasks)
```sql
CREATE TABLE concept_mining_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES knowledge_graph_projects(id),
    corpus_documents JSONB, -- 语料文档列表
    mining_method VARCHAR(100), -- lda, word2vec, bert_clustering
    mining_config JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    concepts_found INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 服务架构

#### 知识图谱构建服务主类
```python
# src/services/knowledge_graph_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import spacy
import jieba
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import re
import uuid
import time
import json
from datetime import datetime

class KnowledgeGraphService:
    def __init__(self):
        # 初始化NLP模型
        self.nlp_zh = spacy.load('zh_core_web_sm')
        self.nlp_en = spacy.load('en_core_web_sm')
        
        # 初始化BERT NER模型
        self.ner_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.ner_model = AutoModelForTokenClassification.from_pretrained('bert-base-chinese')
        self.ner_pipeline = pipeline('ner', 
                                   model=self.ner_model, 
                                   tokenizer=self.ner_tokenizer,
                                   aggregation_strategy='simple')
        
        # 数据库连接
        self.db = DatabaseManager()
        self.neo4j_graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
        self.elasticsearch = ElasticsearchClient()
        self.message_queue = RabbitMQClient()
        
        # 实体类型映射
        self.entity_type_mapping = {
            'PERSON': '人物',
            'ORG': '组织',
            'GPE': '地理政治实体',
            'LOC': '地点',
            'EVENT': '事件',
            'DATE': '日期',
            'TIME': '时间',
            'MONEY': '货币',
            'PERCENT': '百分比',
            'CARDINAL': '基数',
            'ORDINAL': '序数'
        }
        
        # 关系类型定义
        self.relation_types = {
            '出生于': 'BORN_IN',
            '死于': 'DIED_IN',
            '任职于': 'WORKED_AT',
            '位于': 'LOCATED_IN',
            '属于': 'BELONGS_TO',
            '参与': 'PARTICIPATED_IN',
            '创建': 'FOUNDED',
            '影响': 'INFLUENCED',
            '继承': 'INHERITED',
            '统治': 'RULED'
        }
    
    async def create_knowledge_graph_project(self, 
                                           name: str,
                                           description: str,
                                           domain: str,
                                           language: str = 'zh',
                                           config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建知识图谱项目
        
        Args:
            name: 项目名称
            description: 项目描述
            domain: 领域
            language: 语言
            config: 配置信息
            
        Returns:
            项目信息
        """
        try:
            project_id = str(uuid.uuid4())
            neo4j_database = f"kg_{project_id.replace('-', '_')}"
            
            # 创建Neo4j数据库
            await self._create_neo4j_database(neo4j_database)
            
            # 保存项目信息
            project_data = {
                'id': project_id,
                'name': name,
                'description': description,
                'domain': domain,
                'language': language,
                'neo4j_database': neo4j_database,
                'entity_types': config.get('entity_types', list(self.entity_type_mapping.keys())),
                'relation_types': config.get('relation_types', list(self.relation_types.keys())),
                'extraction_config': config.get('extraction_config', {}),
                'statistics': {
                    'entities': 0,
                    'relations': 0,
                    'documents': 0
                }
            }
            
            await self.db.insert('knowledge_graph_projects', project_data)
            
            return {
                'success': True,
                'project_id': project_id,
                'message': '知识图谱项目创建成功'
            }
            
        except Exception as e:
            logger.error(f"创建知识图谱项目失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def extract_entities_from_text(self, 
                                       project_id: str,
                                       text: str,
                                       document_id: str = None,
                                       method: str = 'spacy_ner',
                                       config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        从文本中抽取实体
        
        Args:
            project_id: 项目ID
            text: 文本内容
            document_id: 文档ID
            method: 抽取方法
            config: 抽取配置
            
        Returns:
            抽取结果
        """
        try:
            task_id = str(uuid.uuid4())
            start_time = time.time()
            
            # 创建抽取任务
            task_data = {
                'id': task_id,
                'project_id': project_id,
                'document_id': document_id,
                'text_content': text[:1000],  # 只保存前1000字符
                'extraction_method': method,
                'extraction_config': config or {},
                'status': 'processing'
            }
            await self.db.insert('entity_extraction_tasks', task_data)
            
            # 根据方法选择抽取器
            if method == 'spacy_ner':
                entities = await self._extract_entities_spacy(text, config)
            elif method == 'bert_ner':
                entities = await self._extract_entities_bert(text, config)
            elif method == 'jieba_ner':
                entities = await self._extract_entities_jieba(text, config)
            else:
                raise ValueError(f"不支持的抽取方法: {method}")
            
            # 保存实体到数据库和Neo4j
            saved_entities = []
            for entity in entities:
                saved_entity = await self._save_entity(
                    project_id=project_id,
                    entity_data=entity,
                    source_document_id=document_id
                )
                saved_entities.append(saved_entity)
            
            # 更新任务状态
            processing_time = time.time() - start_time
            await self.db.update(
                'entity_extraction_tasks',
                {'id': task_id},
                {
                    'status': 'completed',
                    'entities_found': len(entities),
                    'completed_at': datetime.now(),
                    'processing_time': processing_time
                }
            )
            
            return {
                'success': True,
                'task_id': task_id,
                'entities_found': len(entities),
                'entities': saved_entities,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"实体抽取失败: {str(e)}")
            await self.db.update(
                'entity_extraction_tasks',
                {'id': task_id},
                {'status': 'failed', 'error_message': str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _extract_entities_spacy(self, 
                                    text: str, 
                                    config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        使用spaCy进行实体抽取
        
        Args:
            text: 文本内容
            config: 配置参数
            
        Returns:
            实体列表
        """
        # 选择语言模型
        nlp = self.nlp_zh if self._is_chinese_text(text) else self.nlp_en
        
        # 处理文本
        doc = nlp(text)
        
        entities = []
        confidence_threshold = config.get('confidence_threshold', 0.8) if config else 0.8
        
        for ent in doc.ents:
            # 过滤低置信度实体
            if hasattr(ent, 'confidence') and ent.confidence < confidence_threshold:
                continue
            
            entity = {
                'name': ent.text.strip(),
                'entity_type': ent.label_,
                'start_pos': ent.start_char,
                'end_pos': ent.end_char,
                'confidence_score': getattr(ent, 'confidence', 0.9),
                'context': text[max(0, ent.start_char-50):ent.end_char+50]
            }
            
            # 实体标准化
            entity = await self._normalize_entity(entity)
            entities.append(entity)
        
        return entities
    
    async def _extract_entities_bert(self, 
                                   text: str, 
                                   config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        使用BERT进行实体抽取
        
        Args:
            text: 文本内容
            config: 配置参数
            
        Returns:
            实体列表
        """
        # 分段处理长文本
        max_length = 512
        text_segments = [text[i:i+max_length] for i in range(0, len(text), max_length-50)]
        
        entities = []
        confidence_threshold = config.get('confidence_threshold', 0.8) if config else 0.8
        
        for segment_idx, segment in enumerate(text_segments):
            # BERT NER推理
            ner_results = self.ner_pipeline(segment)
            
            for result in ner_results:
                if result['score'] < confidence_threshold:
                    continue
                
                # 计算在原文本中的位置
                start_pos = segment_idx * (max_length - 50) + result['start']
                end_pos = segment_idx * (max_length - 50) + result['end']
                
                entity = {
                    'name': result['word'].strip(),
                    'entity_type': result['entity_group'],
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'confidence_score': result['score'],
                    'context': text[max(0, start_pos-50):end_pos+50]
                }
                
                # 实体标准化
                entity = await self._normalize_entity(entity)
                entities.append(entity)
        
        return entities
    
    async def _extract_entities_jieba(self, 
                                    text: str, 
                                    config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        使用jieba进行实体抽取
        
        Args:
            text: 文本内容
            config: 配置参数
            
        Returns:
            实体列表
        """
        import jieba.posseg as pseg
        
        # 词性标注
        words = pseg.cut(text)
        
        entities = []
        current_pos = 0
        
        # 实体类型映射
        pos_to_entity_type = {
            'nr': 'PERSON',      # 人名
            'ns': 'LOCATION',    # 地名
            'nt': 'ORGANIZATION', # 机构名
            'nz': 'CONCEPT',     # 其他专名
            't': 'TIME',         # 时间
            'm': 'CARDINAL'      # 数量
        }
        
        for word, pos in words:
            if pos in pos_to_entity_type and len(word) > 1:
                entity = {
                    'name': word.strip(),
                    'entity_type': pos_to_entity_type[pos],
                    'start_pos': current_pos,
                    'end_pos': current_pos + len(word),
                    'confidence_score': 0.8,  # jieba默认置信度
                    'context': text[max(0, current_pos-50):current_pos+len(word)+50]
                }
                
                # 实体标准化
                entity = await self._normalize_entity(entity)
                entities.append(entity)
            
            current_pos += len(word)
        
        return entities
    
    async def _normalize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        实体标准化
        
        Args:
            entity: 原始实体
            
        Returns:
            标准化后的实体
        """
        # 清理实体名称
        name = entity['name']
        name = re.sub(r'[\s\n\r\t]+', '', name)  # 移除空白字符
        name = re.sub(r'^[的地得]', '', name)   # 移除开头的助词
        name = re.sub(r'[的地得]$', '', name)   # 移除结尾的助词
        
        # 过滤无效实体
        if len(name) < 2 or name.isdigit() or name in ['这个', '那个', '什么', '怎么']:
            return None
        
        entity['name'] = name
        
        # 标准化实体类型
        entity_type = entity['entity_type']
        if entity_type in self.entity_type_mapping:
            entity['entity_type_zh'] = self.entity_type_mapping[entity_type]
        
        return entity
    
    def _is_chinese_text(self, text: str) -> bool:
        """
        判断文本是否为中文
        
        Args:
            text: 文本内容
            
        Returns:
            是否为中文
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return chinese_chars / len(text) > 0.5 if text else False
    
    async def _save_entity(self, 
                         project_id: str,
                         entity_data: Dict[str, Any],
                         source_document_id: str = None) -> Dict[str, Any]:
        """
        保存实体到数据库和Neo4j
        
        Args:
            project_id: 项目ID
            entity_data: 实体数据
            source_document_id: 来源文档ID
            
        Returns:
            保存的实体信息
        """
        if not entity_data:
            return None
        
        try:
            # 检查实体是否已存在
            existing_entity = await self.db.select_one(
                'entities',
                {
                    'project_id': project_id,
                    'name': entity_data['name'],
                    'entity_type': entity_data['entity_type']
                }
            )
            
            if existing_entity:
                # 更新提及次数和来源文档
                mention_count = existing_entity['mention_count'] + 1
                source_documents = existing_entity.get('source_documents', [])
                if source_document_id and source_document_id not in source_documents:
                    source_documents.append(source_document_id)
                
                await self.db.update(
                    'entities',
                    {'id': existing_entity['id']},
                    {
                        'mention_count': mention_count,
                        'source_documents': source_documents,
                        'updated_at': datetime.now()
                    }
                )
                
                return existing_entity
            
            else:
                # 创建新实体
                entity_id = str(uuid.uuid4())
                
                # 保存到PostgreSQL
                entity_record = {
                    'id': entity_id,
                    'project_id': project_id,
                    'name': entity_data['name'],
                    'entity_type': entity_data['entity_type'],
                    'confidence_score': entity_data.get('confidence_score', 0.8),
                    'source_documents': [source_document_id] if source_document_id else [],
                    'mention_count': 1,
                    'properties': {
                        'context': entity_data.get('context', ''),
                        'start_pos': entity_data.get('start_pos'),
                        'end_pos': entity_data.get('end_pos')
                    }
                }
                
                await self.db.insert('entities', entity_record)
                
                # 保存到Neo4j
                neo4j_node = await self._create_neo4j_entity_node(
                    project_id, entity_record
                )
                
                # 更新Neo4j节点ID
                if neo4j_node:
                    await self.db.update(
                        'entities',
                        {'id': entity_id},
                        {'neo4j_node_id': neo4j_node.identity}
                    )
                
                return entity_record
                
        except Exception as e:
            logger.error(f"保存实体失败: {str(e)}")
            return None
    
    async def _create_neo4j_entity_node(self, 
                                      project_id: str,
                                      entity_data: Dict[str, Any]) -> Optional[Node]:
        """
        在Neo4j中创建实体节点
        
        Args:
            project_id: 项目ID
            entity_data: 实体数据
            
        Returns:
            Neo4j节点
        """
        try:
            # 获取项目信息
            project = await self.db.select_one(
                'knowledge_graph_projects',
                {'id': project_id}
            )
            
            if not project:
                return None
            
            # 切换到项目数据库
            graph = Graph(f"bolt://localhost:7687/{project['neo4j_database']}", 
                         auth=("neo4j", "password"))
            
            # 创建节点
            node = Node(
                entity_data['entity_type'],
                name=entity_data['name'],
                entity_id=entity_data['id'],
                confidence_score=entity_data['confidence_score'],
                mention_count=entity_data['mention_count'],
                created_at=datetime.now().isoformat()
            )
            
            # 添加属性
            for key, value in entity_data.get('properties', {}).items():
                node[key] = value
            
            graph.create(node)
            
            return node
            
        except Exception as e:
            logger.error(f"创建Neo4j实体节点失败: {str(e)}")
            return None
    
    async def extract_relations_from_text(self, 
                                        project_id: str,
                                        text: str,
                                        document_id: str = None,
                                        method: str = 'pattern_based',
                                        config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        从文本中抽取关系
        
        Args:
            project_id: 项目ID
            text: 文本内容
            document_id: 文档ID
            method: 抽取方法
            config: 抽取配置
            
        Returns:
            抽取结果
        """
        try:
            task_id = str(uuid.uuid4())
            start_time = time.time()
            
            # 创建抽取任务
            task_data = {
                'id': task_id,
                'project_id': project_id,
                'document_id': document_id,
                'text_content': text[:1000],
                'extraction_method': method,
                'extraction_config': config or {},
                'status': 'processing'
            }
            await self.db.insert('relation_extraction_tasks', task_data)
            
            # 首先获取文本中的实体
            entities_result = await self.extract_entities_from_text(
                project_id, text, document_id, 'spacy_ner', config
            )
            entities = entities_result.get('entities', [])
            
            # 根据方法选择关系抽取器
            if method == 'pattern_based':
                relations = await self._extract_relations_pattern_based(
                    text, entities, config
                )
            elif method == 'dependency_parsing':
                relations = await self._extract_relations_dependency_parsing(
                    text, entities, config
                )
            elif method == 'rule_based':
                relations = await self._extract_relations_rule_based(
                    text, entities, config
                )
            else:
                raise ValueError(f"不支持的关系抽取方法: {method}")
            
            # 保存关系到数据库和Neo4j
            saved_relations = []
            for relation in relations:
                saved_relation = await self._save_relation(
                    project_id=project_id,
                    relation_data=relation,
                    source_document_id=document_id
                )
                if saved_relation:
                    saved_relations.append(saved_relation)
            
            # 更新任务状态
            processing_time = time.time() - start_time
            await self.db.update(
                'relation_extraction_tasks',
                {'id': task_id},
                {
                    'status': 'completed',
                    'relations_found': len(relations),
                    'completed_at': datetime.now(),
                    'processing_time': processing_time
                }
            )
            
            return {
                'success': True,
                'task_id': task_id,
                'relations_found': len(relations),
                'relations': saved_relations,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"关系抽取失败: {str(e)}")
            await self.db.update(
                'relation_extraction_tasks',
                {'id': task_id},
                {'status': 'failed', 'error_message': str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _extract_relations_pattern_based(self, 
                                             text: str,
                                             entities: List[Dict[str, Any]],
                                             config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        基于模式的关系抽取
        
        Args:
            text: 文本内容
            entities: 实体列表
            config: 配置参数
            
        Returns:
            关系列表
        """
        relations = []
        
        # 定义关系模式
        relation_patterns = {
            r'(.+?)出生于(.+?)': ('出生于', 'PERSON', 'LOCATION'),
            r'(.+?)死于(.+?)': ('死于', 'PERSON', 'LOCATION'),
            r'(.+?)任职于(.+?)': ('任职于', 'PERSON', 'ORGANIZATION'),
            r'(.+?)位于(.+?)': ('位于', 'LOCATION', 'LOCATION'),
            r'(.+?)创建了(.+?)': ('创建', 'PERSON', 'ORGANIZATION'),
            r'(.+?)统治(.+?)': ('统治', 'PERSON', 'LOCATION'),
            r'(.+?)的(.+?)': ('属于', 'ANY', 'ANY')
        }
        
        # 创建实体名称到实体的映射
        entity_map = {entity['name']: entity for entity in entities}
        
        for pattern, (relation_type, subj_type, obj_type) in relation_patterns.items():
            matches = re.finditer(pattern, text)
            
            for match in matches:
                subject_name = match.group(1).strip()
                object_name = match.group(2).strip()
                
                # 查找对应的实体
                subject_entity = entity_map.get(subject_name)
                object_entity = entity_map.get(object_name)
                
                if subject_entity and object_entity:
                    # 检查实体类型是否匹配
                    if (subj_type == 'ANY' or subject_entity['entity_type'] == subj_type) and \
                       (obj_type == 'ANY' or object_entity['entity_type'] == obj_type):
                        
                        relation = {
                            'subject_entity': subject_entity,
                            'predicate': relation_type,
                            'object_entity': object_entity,
                            'confidence_score': 0.8,
                            'context': match.group(0),
                            'source_sentence': self._extract_sentence(text, match.start())
                        }
                        
                        relations.append(relation)
        
        return relations
    
    def _extract_sentence(self, text: str, position: int) -> str:
        """
        提取包含指定位置的句子
        
        Args:
            text: 文本内容
            position: 位置
            
        Returns:
            句子
        """
        # 查找句子边界
        sentence_end_chars = ['。', '！', '？', '.', '!', '?']
        
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
    
    async def _save_relation(self, 
                           project_id: str,
                           relation_data: Dict[str, Any],
                           source_document_id: str = None) -> Optional[Dict[str, Any]]:
        """
        保存关系到数据库和Neo4j
        
        Args:
            project_id: 项目ID
            relation_data: 关系数据
            source_document_id: 来源文档ID
            
        Returns:
            保存的关系信息
        """
        try:
            # 获取主体和客体实体ID
            subject_entity = relation_data['subject_entity']
            object_entity = relation_data['object_entity']
            
            # 检查关系是否已存在
            existing_relation = await self.db.select_one(
                'relations',
                {
                    'project_id': project_id,
                    'subject_entity_id': subject_entity['id'],
                    'predicate': relation_data['predicate'],
                    'object_entity_id': object_entity['id']
                }
            )
            
            if existing_relation:
                return existing_relation
            
            # 创建新关系
            relation_id = str(uuid.uuid4())
            
            relation_record = {
                'id': relation_id,
                'project_id': project_id,
                'subject_entity_id': subject_entity['id'],
                'predicate': relation_data['predicate'],
                'object_entity_id': object_entity['id'],
                'confidence_score': relation_data.get('confidence_score', 0.8),
                'context': relation_data.get('context', ''),
                'source_sentence': relation_data.get('source_sentence', ''),
                'source_document_id': source_document_id,
                'properties': {
                    'extraction_method': 'pattern_based'
                }
            }
            
            await self.db.insert('relations', relation_record)
            
            # 在Neo4j中创建关系
            neo4j_relationship = await self._create_neo4j_relationship(
                project_id, relation_record
            )
            
            if neo4j_relationship:
                await self.db.update(
                    'relations',
                    {'id': relation_id},
                    {'neo4j_relationship_id': neo4j_relationship.identity}
                )
            
            return relation_record
            
        except Exception as e:
            logger.error(f"保存关系失败: {str(e)}")
            return None
    
    async def _create_neo4j_relationship(self, 
                                       project_id: str,
                                       relation_data: Dict[str, Any]) -> Optional[Relationship]:
        """
        在Neo4j中创建关系
        
        Args:
            project_id: 项目ID
            relation_data: 关系数据
            
        Returns:
            Neo4j关系
        """
        try:
            # 获取项目信息
            project = await self.db.select_one(
                'knowledge_graph_projects',
                {'id': project_id}
            )
            
            if not project:
                return None
            
            # 获取主体和客体实体的Neo4j节点
            subject_entity = await self.db.select_one(
                'entities',
                {'id': relation_data['subject_entity_id']}
            )
            object_entity = await self.db.select_one(
                'entities',
                {'id': relation_data['object_entity_id']}
            )
            
            if not subject_entity or not object_entity:
                return None
            
            # 切换到项目数据库
            graph = Graph(f"bolt://localhost:7687/{project['neo4j_database']}", 
                         auth=("neo4j", "password"))
            
            # 查找节点
            matcher = NodeMatcher(graph)
            subject_node = matcher.match(entity_id=subject_entity['id']).first()
            object_node = matcher.match(entity_id=object_entity['id']).first()
            
            if not subject_node or not object_node:
                return None
            
            # 创建关系
            relationship = Relationship(
                subject_node,
                relation_data['predicate'],
                object_node,
                relation_id=relation_data['id'],
                confidence_score=relation_data['confidence_score'],
                context=relation_data.get('context', ''),
                source_sentence=relation_data.get('source_sentence', ''),
                created_at=datetime.now().isoformat()
            )
            
            graph.create(relationship)
            
            return relationship
            
        except Exception as e:
            logger.error(f"创建Neo4j关系失败: {str(e)}")
            return None
    
    async def query_knowledge_graph(self, 
                                  project_id: str,
                                  query_type: str,
                                  query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        查询知识图谱
        
        Args:
            project_id: 项目ID
            query_type: 查询类型
            query_params: 查询参数
            
        Returns:
            查询结果
        """
        try:
            if query_type == 'entity_search':
                return await self._query_entities(project_id, query_params)
            elif query_type == 'relation_search':
                return await self._query_relations(project_id, query_params)
            elif query_type == 'path_search':
                return await self._query_paths(project_id, query_params)
            elif query_type == 'subgraph':
                return await self._query_subgraph(project_id, query_params)
            elif query_type == 'statistics':
                return await self._query_statistics(project_id)
            else:
                raise ValueError(f"不支持的查询类型: {query_type}")
                
        except Exception as e:
            logger.error(f"知识图谱查询失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _query_entities(self, 
                            project_id: str,
                            params: Dict[str, Any]) -> Dict[str, Any]:
        """
        查询实体
        
        Args:
            project_id: 项目ID
            params: 查询参数
            
        Returns:
            实体查询结果
        """
        conditions = {'project_id': project_id}
        
        # 添加查询条件
        if 'entity_type' in params:
            conditions['entity_type'] = params['entity_type']
        
        if 'name_pattern' in params:
            # 使用LIKE查询
            pass  # 需要在数据库查询中处理
        
        # 分页参数
        page = params.get('page', 1)
        page_size = params.get('page_size', 20)
        offset = (page - 1) * page_size
        
        # 查询实体
        entities = await self.db.select(
            'entities',
            conditions,
            limit=page_size,
            offset=offset,
            order_by='mention_count DESC'
        )
        
        # 统计总数
        total_count = await self.db.count('entities', conditions)
        
        return {
            'success': True,
            'entities': entities,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': (total_count + page_size - 1) // page_size
            }
        }
```

### API设计

#### 知识图谱构建API
```python
# 创建知识图谱项目
POST /api/v1/knowledge-graph/projects
Content-Type: application/json
Request: {
    "name": "明朝历史知识图谱",
    "description": "基于明朝历史文档构建的知识图谱",
    "domain": "历史",
    "language": "zh",
    "config": {
        "entity_types": ["PERSON", "LOCATION", "ORGANIZATION", "EVENT"],
        "relation_types": ["出生于", "任职于", "统治", "位于"]
    }
}
Response: {
    "success": true,
    "project_id": "uuid",
    "message": "知识图谱项目创建成功"
}

# 实体抽取
POST /api/v1/knowledge-graph/extract-entities
Content-Type: application/json
Request: {
    "project_id": "uuid",
    "text": "朱元璋出生于濠州钟离，后来建立了明朝。",
    "document_id": "uuid",
    "method": "spacy_ner",
    "config": {
        "confidence_threshold": 0.8
    }
}
Response: {
    "success": true,
    "task_id": "uuid",
    "entities_found": 3,
    "entities": [
        {
            "id": "uuid",
            "name": "朱元璋",
            "entity_type": "PERSON",
            "confidence_score": 0.95
        }
    ],
    "processing_time": 1.2
}

# 关系抽取
POST /api/v1/knowledge-graph/extract-relations
Content-Type: application/json
Request: {
    "project_id": "uuid",
    "text": "朱元璋出生于濠州钟离，后来建立了明朝。",
    "document_id": "uuid",
    "method": "pattern_based"
}
Response: {
    "success": true,
    "task_id": "uuid",
    "relations_found": 2,
    "relations": [
        {
            "id": "uuid",
            "subject_entity": {"name": "朱元璋", "entity_type": "PERSON"},
            "predicate": "出生于",
            "object_entity": {"name": "濠州钟离", "entity_type": "LOCATION"},
            "confidence_score": 0.9
        }
    ]
}

# 知识图谱查询
GET /api/v1/knowledge-graph/query
Query Parameters:
    project_id=uuid
    query_type=entity_search
    entity_type=PERSON
    page=1
    page_size=20
Response: {
    "success": true,
    "entities": [
        {
            "id": "uuid",
            "name": "朱元璋",
            "entity_type": "PERSON",
            "mention_count": 15,
            "confidence_score": 0.95
        }
    ],
    "pagination": {
        "page": 1,
        "page_size": 20,
        "total_count": 156,
        "total_pages": 8
    }
}
```

### 前端集成

#### Vue3 知识图谱组件
```vue
<!-- components/KnowledgeGraphBuilder.vue -->
<template>
  <div class="knowledge-graph-builder">
    <el-card class="project-card">
      <template #header>
        <span>知识图谱项目</span>
        <el-button type="primary" @click="showCreateProject = true">
          创建项目
        </el-button>
      </template>
      
      <el-table :data="projects" style="width: 100%">
        <el-table-column prop="name" label="项目名称" />
        <el-table-column prop="domain" label="领域" />
        <el-table-column prop="language" label="语言" />
        <el-table-column prop="status" label="状态">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">
              {{ scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作">
          <template #default="scope">
            <el-button size="small" @click="selectProject(scope.row)">
              选择
            </el-button>
            <el-button size="small" type="info" @click="viewProject(scope.row)">
              查看
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
    
    <!-- 文本处理区域 -->
    <el-card v-if="selectedProject" class="processing-card">
      <template #header>
        <span>文本处理 - {{ selectedProject.name }}</span>
      </template>
      
      <el-tabs v-model="activeTab">
        <el-tab-pane label="实体抽取" name="entity">
          <div class="text-input-area">
            <el-input
              v-model="inputText"
              type="textarea"
              :rows="6"
              placeholder="请输入要处理的文本..."
            />
            
            <div class="config-area">
              <el-form :model="entityConfig" inline>
                <el-form-item label="抽取方法">
                  <el-select v-model="entityConfig.method">
                    <el-option label="spaCy NER" value="spacy_ner" />
                    <el-option label="BERT NER" value="bert_ner" />
                    <el-option label="jieba NER" value="jieba_ner" />
                  </el-select>
                </el-form-item>
                
                <el-form-item label="置信度阈值">
                  <el-input-number 
                    v-model="entityConfig.confidence_threshold" 
                    :min="0" 
                    :max="1" 
                    :step="0.1" 
                  />
                </el-form-item>
                
                <el-form-item>
                  <el-button type="primary" @click="extractEntities" :loading="extractingEntities">
                    抽取实体
                  </el-button>
                </el-form-item>
              </el-form>
            </div>
          </div>
          
          <!-- 实体抽取结果 -->
          <div v-if="entityResults.length > 0" class="results-area">
            <h4>抽取结果 ({{ entityResults.length }} 个实体)</h4>
            <el-table :data="entityResults" size="small">
              <el-table-column prop="name" label="实体名称" />
              <el-table-column prop="entity_type" label="实体类型">
                <template #default="scope">
                  <el-tag size="small">{{ scope.row.entity_type }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="confidence_score" label="置信度">
                <template #default="scope">
                  <el-progress 
                    :percentage="scope.row.confidence_score * 100" 
                    :stroke-width="6" 
                    :show-text="false"
                  />
                  <span class="confidence-text">{{ (scope.row.confidence_score * 100).toFixed(1) }}%</span>
                </template>
              </el-table-column>
              <el-table-column prop="context" label="上下文" show-overflow-tooltip />
            </el-table>
          </div>
        </el-tab-pane>
        
        <el-tab-pane label="关系抽取" name="relation">
          <div class="text-input-area">
            <el-input
              v-model="inputText"
              type="textarea"
              :rows="6"
              placeholder="请输入要处理的文本..."
            />
            
            <div class="config-area">
              <el-form :model="relationConfig" inline>
                <el-form-item label="抽取方法">
                  <el-select v-model="relationConfig.method">
                    <el-option label="模式匹配" value="pattern_based" />
                    <el-option label="依存分析" value="dependency_parsing" />
                    <el-option label="规则抽取" value="rule_based" />
                  </el-select>
                </el-form-item>
                
                <el-form-item>
                  <el-button type="primary" @click="extractRelations" :loading="extractingRelations">
                    抽取关系
                  </el-button>
                </el-form-item>
              </el-form>
            </div>
          </div>
          
          <!-- 关系抽取结果 -->
          <div v-if="relationResults.length > 0" class="results-area">
            <h4>抽取结果 ({{ relationResults.length }} 个关系)</h4>
            <el-table :data="relationResults" size="small">
              <el-table-column label="主体" prop="subject_entity.name" />
              <el-table-column label="关系" prop="predicate">
                <template #default="scope">
                  <el-tag type="warning" size="small">{{ scope.row.predicate }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column label="客体" prop="object_entity.name" />
              <el-table-column prop="confidence_score" label="置信度">
                <template #default="scope">
                  <span class="confidence-text">{{ (scope.row.confidence_score * 100).toFixed(1) }}%</span>
                </template>
              </el-table-column>
              <el-table-column prop="source_sentence" label="来源句子" show-overflow-tooltip />
            </el-table>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
    
    <!-- 知识图谱可视化 -->
    <el-card v-if="selectedProject" class="visualization-card">
      <template #header>
        <span>知识图谱可视化</span>
        <el-button @click="refreshGraph">刷新图谱</el-button>
      </template>
      
      <div id="graph-container" class="graph-container"></div>
    </el-card>
    
    <!-- 创建项目对话框 -->
    <el-dialog v-model="showCreateProject" title="创建知识图谱项目" width="600px">
      <el-form :model="newProject" label-width="100px">
        <el-form-item label="项目名称" required>
          <el-input v-model="newProject.name" />
        </el-form-item>
        
        <el-form-item label="项目描述">
          <el-input v-model="newProject.description" type="textarea" :rows="3" />
        </el-form-item>
        
        <el-form-item label="领域">
          <el-select v-model="newProject.domain">
            <el-option label="历史" value="历史" />
            <el-option label="文学" value="文学" />
            <el-option label="政治" value="政治" />
            <el-option label="经济" value="经济" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="语言">
          <el-select v-model="newProject.language">
            <el-option label="中文" value="zh" />
            <el-option label="英文" value="en" />
          </el-select>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showCreateProject = false">取消</el-button>
        <el-button type="primary" @click="createProject" :loading="creatingProject">
          创建
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import * as d3 from 'd3'

interface Project {
  id: string
  name: string
  description: string
  domain: string
  language: string
  status: string
}

interface Entity {
  id: string
  name: string
  entity_type: string
  confidence_score: number
  context: string
}

interface Relation {
  id: string
  subject_entity: Entity
  predicate: string
  object_entity: Entity
  confidence_score: number
  source_sentence: string
}

const authStore = useAuthStore()
const projects = ref<Project[]>([])
const selectedProject = ref<Project | null>(null)
const activeTab = ref('entity')
const inputText = ref('')
const entityResults = ref<Entity[]>([])
const relationResults = ref<Relation[]>([])
const extractingEntities = ref(false)
const extractingRelations = ref(false)
const showCreateProject = ref(false)
const creatingProject = ref(false)

const entityConfig = reactive({
  method: 'spacy_ner',
  confidence_threshold: 0.8
})

const relationConfig = reactive({
  method: 'pattern_based'
})

const newProject = reactive({
  name: '',
  description: '',
  domain: '历史',
  language: 'zh'
})

/**
 * 组件挂载时加载项目列表
 */
onMounted(async () => {
  await loadProjects()
})

/**
 * 加载项目列表
 */
const loadProjects = async () => {
  try {
    const response = await fetch('/api/v1/knowledge-graph/projects', {
      headers: {
        'Authorization': `Bearer ${authStore.token}`
      }
    })
    const data = await response.json()
    if (data.success) {
      projects.value = data.projects
    }
  } catch (error) {
    ElMessage.error('加载项目列表失败')
  }
}

/**
 * 选择项目
 */
const selectProject = (project: Project) => {
  selectedProject.value = project
  ElMessage.success(`已选择项目: ${project.name}`)
}

/**
 * 创建项目
 */
const createProject = async () => {
  if (!newProject.name) {
    ElMessage.warning('请输入项目名称')
    return
  }
  
  creatingProject.value = true
  try {
    const response = await fetch('/api/v1/knowledge-graph/projects', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authStore.token}`
      },
      body: JSON.stringify(newProject)
    })
    
    const data = await response.json()
    if (data.success) {
      ElMessage.success('项目创建成功')
      showCreateProject.value = false
      await loadProjects()
      // 重置表单
      Object.assign(newProject, {
        name: '',
        description: '',
        domain: '历史',
        language: 'zh'
      })
    }
  } catch (error) {
    ElMessage.error('创建项目失败')
  } finally {
    creatingProject.value = false
  }
}

/**
 * 抽取实体
 */
const extractEntities = async () => {
  if (!inputText.value.trim()) {
    ElMessage.warning('请输入要处理的文本')
    return
  }
  
  if (!selectedProject.value) {
    ElMessage.warning('请先选择项目')
    return
  }
  
  extractingEntities.value = true
  try {
    const response = await fetch('/api/v1/knowledge-graph/extract-entities', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authStore.token}`
      },
      body: JSON.stringify({
        project_id: selectedProject.value.id,
        text: inputText.value,
        method: entityConfig.method,
        config: {
          confidence_threshold: entityConfig.confidence_threshold
        }
      })
    })
    
    const data = await response.json()
    if (data.success) {
      entityResults.value = data.entities
      ElMessage.success(`成功抽取 ${data.entities_found} 个实体`)
    }
  } catch (error) {
    ElMessage.error('实体抽取失败')
  } finally {
    extractingEntities.value = false
  }
}

/**
 * 抽取关系
 */
const extractRelations = async () => {
  if (!inputText.value.trim()) {
    ElMessage.warning('请输入要处理的文本')
    return
  }
  
  if (!selectedProject.value) {
    ElMessage.warning('请先选择项目')
    return
  }
  
  extractingRelations.value = true
  try {
    const response = await fetch('/api/v1/knowledge-graph/extract-relations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authStore.token}`
      },
      body: JSON.stringify({
        project_id: selectedProject.value.id,
        text: inputText.value,
        method: relationConfig.method
      })
    })
    
    const data = await response.json()
    if (data.success) {
      relationResults.value = data.relations
      ElMessage.success(`成功抽取 ${data.relations_found} 个关系`)
    }
  } catch (error) {
    ElMessage.error('关系抽取失败')
  } finally {
    extractingRelations.value = false
  }
}

/**
 * 获取状态类型
 */
const getStatusType = (status: string) => {
  const statusMap: Record<string, string> = {
    'active': 'success',
    'building': 'warning',
    'completed': 'info',
    'archived': 'info'
  }
  return statusMap[status] || 'info'
}

/**
 * 刷新图谱
 */
const refreshGraph = async () => {
  if (!selectedProject.value) return
  
  try {
    // 查询图谱数据
    const response = await fetch(`/api/v1/knowledge-graph/query?project_id=${selectedProject.value.id}&query_type=subgraph&limit=100`, {
      headers: {
        'Authorization': `Bearer ${authStore.token}`
      }
    })
    
    const data = await response.json()
    if (data.success) {
      await nextTick()
      renderGraph(data.nodes, data.links)
    }
  } catch (error) {
    ElMessage.error('加载图谱数据失败')
  }
}

/**
 * 渲染知识图谱
 */
const renderGraph = (nodes: any[], links: any[]) => {
  const container = d3.select('#graph-container')
  container.selectAll('*').remove()
  
  const width = 800
  const height = 600
  
  const svg = container
    .append('svg')
    .attr('width', width)
    .attr('height', height)
  
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id((d: any) => d.id))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2))
  
  // 绘制连线
  const link = svg.append('g')
    .selectAll('line')
    .data(links)
    .enter().append('line')
    .attr('stroke', '#999')
    .attr('stroke-opacity', 0.6)
    .attr('stroke-width', 2)
  
  // 绘制节点
  const node = svg.append('g')
    .selectAll('circle')
    .data(nodes)
    .enter().append('circle')
    .attr('r', 8)
    .attr('fill', (d: any) => getNodeColor(d.entity_type))
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended))
  
  // 添加标签
  const label = svg.append('g')
    .selectAll('text')
    .data(nodes)
    .enter().append('text')
    .text((d: any) => d.name)
    .attr('font-size', 12)
    .attr('dx', 12)
    .attr('dy', 4)
  
  simulation.on('tick', () => {
    link
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y)
    
    node
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
    
    label
      .attr('x', (d: any) => d.x)
      .attr('y', (d: any) => d.y)
  })
  
  function dragstarted(event: any, d: any) {
    if (!event.active) simulation.alphaTarget(0.3).restart()
    d.fx = d.x
    d.fy = d.y
  }
  
  function dragged(event: any, d: any) {
    d.fx = event.x
    d.fy = event.y
  }
  
  function dragended(event: any, d: any) {
    if (!event.active) simulation.alphaTarget(0)
    d.fx = null
    d.fy = null
  }
}

/**
 * 获取节点颜色
 */
const getNodeColor = (entityType: string) => {
  const colorMap: Record<string, string> = {
    'PERSON': '#ff6b6b',
    'LOCATION': '#4ecdc4',
    'ORGANIZATION': '#45b7d1',
    'EVENT': '#96ceb4',
    'CONCEPT': '#feca57'
  }
  return colorMap[entityType] || '#95a5a6'
}
</script>

<style scoped>
.knowledge-graph-builder {
  padding: 20px;
}

.project-card,
.processing-card,
.visualization-card {
  margin-bottom: 20px;
}

.text-input-area {
  margin-bottom: 20px;
}

.config-area {
  margin-top: 15px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.results-area {
  margin-top: 20px;
  padding: 15px;
  border: 1px solid #e1e8ed;
  border-radius: 4px;
}

.confidence-text {
  margin-left: 8px;
  font-size: 12px;
  color: #666;
}

.graph-container {
  width: 100%;
  height: 600px;
  border: 1px solid #e1e8ed;
  border-radius: 4px;
}
</style>
```

## 验收标准

### 功能验收标准
1. **项目管理**
   - ✅ 能够创建知识图谱项目
   - ✅ 支持项目配置（实体类型、关系类型等）
   - ✅ 项目状态管理和切换

2. **实体抽取**
   - ✅ 支持多种NER方法（spaCy、BERT、jieba）
   - ✅ 实体类型识别准确率 > 85%
   - ✅ 支持置信度阈值配置
   - ✅ 实体去重和标准化

3. **关系抽取**
   - ✅ 支持模式匹配关系抽取
   - ✅ 支持依存句法分析
   - ✅ 关系抽取准确率 > 75%
   - ✅ 关系类型可配置

4. **知识图谱构建**
   - ✅ 实体和关系存储到Neo4j
   - ✅ 图谱数据一致性保证
   - ✅ 支持图谱查询和检索
   - ✅ 图谱可视化展示

5. **概念挖掘**
   - ✅ 支持主题模型（LDA）
   - ✅ 支持词向量聚类
   - ✅ 概念层次结构构建

### 性能验收标准
1. **处理性能**
   - 单次实体抽取响应时间 < 3秒（1000字符文本）
   - 单次关系抽取响应时间 < 5秒（1000字符文本）
   - 支持批量处理，吞吐量 > 100文档/分钟

2. **存储性能**
   - Neo4j图谱查询响应时间 < 1秒
   - 支持10万+实体和100万+关系
   - 数据库连接池优化

3. **内存使用**
   - NLP模型内存占用 < 2GB
   - 服务运行内存 < 4GB
   - 支持模型热加载和卸载

### 准确性验收标准
1. **实体识别准确性**
   - 人名识别准确率 > 90%
   - 地名识别准确率 > 85%
   - 机构名识别准确率 > 80%
   - 时间实体识别准确率 > 95%

2. **关系抽取准确性**
   - 基本关系（出生于、任职于）准确率 > 80%
   - 复杂关系准确率 > 70%
   - 关系方向正确率 > 90%

## 业务价值
1. **知识组织**: 将非结构化历史文档转化为结构化知识图谱
2. **智能检索**: 支持基于实体和关系的智能检索
3. **知识发现**: 通过图谱分析发现隐藏的知识关联
4. **可视化展示**: 直观展示历史知识的网络结构
5. **决策支持**: 为历史研究提供数据驱动的决策支持

## 依赖关系
- **前置依赖**: Story 2.2 (NLP服务) - 需要文本预处理能力
- **数据依赖**: Story 1.3 (数据收集服务) - 需要历史文档数据
- **技术依赖**: Neo4j图数据库、NLP模型库

## 风险和缓解措施
1. **技术风险**
   - **风险**: NLP模型准确率不足
   - **缓解**: 使用多模型融合，人工校验机制

2. **性能风险**
   - **风险**: 大规模图谱查询性能问题
   - **缓解**: 图谱分片、索引优化、缓存策略

3. **数据风险**
   - **风险**: 实体和关系冲突
   - **缓解**: 数据一致性检查、冲突解决机制

## 开发任务分解
1. **环境搭建** (1天)
   - Neo4j数据库安装配置
   - NLP模型下载和配置
   - 开发环境搭建

2. **实体抽取模块** (3天)
   - spaCy NER集成
   - BERT NER模型集成
   - jieba分词和词性标注
   - 实体标准化和去重

3. **关系抽取模块** (2天)
   - 模式匹配关系抽取
   - 依存句法分析
   - 关系验证和过滤

4. **知识图谱存储** (2天)
   - Neo4j数据模型设计
   - 图谱构建和更新
   - 数据一致性保证

5. **前端界面开发** (1天)
   - Vue3组件开发
   - 图谱可视化
   - 交互功能实现

6. **测试和优化** (1天)
   - 单元测试
   - 性能测试
   - 准确性评估