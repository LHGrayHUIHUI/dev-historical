"""
知识图谱服务数据模型单元测试

测试知识图谱服务的核心数据结构，包括：
- 实体和关系模型验证
- 图谱项目管理模型
- 抽取请求和响应模型
- 边界条件和验证规则

作者: Claude (测试架构师)
创建时间: 2025-09-09
版本: 1.0.0
"""

import pytest
from datetime import datetime
from uuid import uuid4
from pydantic import ValidationError
from typing import Dict, Any, List

from src.schemas.knowledge_graph_schemas import (
    EntityType, RelationType, ExtractionMethod, GraphTaskStatus, Language,
    BaseResponse, GraphProject, CreateGraphProjectRequest, CreateGraphProjectResponse,
    Entity, ExtractedEntity, Relation, ExtractedRelation,
    EntityExtractionRequest, EntityExtractionResponse,
    RelationExtractionRequest, RelationExtractionResponse,
    BatchExtractionRequest, BatchExtractionResponse,
    GraphConstructionRequest, GraphConstructionResponse,
    GraphQueryRequest, GraphNode, GraphEdge, GraphQueryResponse,
    GraphAnalysisRequest, GraphMetrics, GraphAnalysisResponse,
    TaskStatus, TaskStatusResponse, ConceptMiningRequest, ConceptTopic,
    ConceptMiningResponse, GraphStatistics, GraphStatisticsResponse
)


class TestEnumTypes:
    """枚举类型测试"""
    
    def test_entity_type_enum(self):
        """测试实体类型枚举"""
        assert EntityType.PERSON == "PERSON"
        assert EntityType.LOCATION == "LOCATION"
        assert EntityType.ORGANIZATION == "ORGANIZATION"
        assert EntityType.EVENT == "EVENT"
        assert EntityType.TIME == "TIME"
        assert EntityType.CONCEPT == "CONCEPT"
        assert EntityType.OBJECT == "OBJECT"
        assert EntityType.WORK == "WORK"
        
        # 测试枚举转换
        assert EntityType("PERSON") == EntityType.PERSON
        
        # 测试无效值
        with pytest.raises(ValueError):
            EntityType("INVALID_TYPE")
    
    def test_relation_type_enum(self):
        """测试关系类型枚举"""
        assert RelationType.BORN_IN == "出生于"
        assert RelationType.DIED_IN == "死于"
        assert RelationType.WORKED_AT == "任职于"
        assert RelationType.LOCATED_IN == "位于"
        assert RelationType.FOUNDED == "创建"
        assert RelationType.INFLUENCED == "影响"
        assert RelationType.PARTICIPATED_IN == "参与"
        assert RelationType.BELONGS_TO == "属于"
        assert RelationType.RULED == "统治"
        assert RelationType.INHERITED == "继承"
        assert RelationType.LEARNED_FROM == "师从"
        assert RelationType.CONTAINS == "包含"
    
    def test_extraction_method_enum(self):
        """测试抽取方法枚举"""
        assert ExtractionMethod.SPACY_NER == "spacy_ner"
        assert ExtractionMethod.BERT_NER == "bert_ner"
        assert ExtractionMethod.JIEBA_NER == "jieba_ner"
        assert ExtractionMethod.PATTERN_BASED == "pattern_based"
        assert ExtractionMethod.DEPENDENCY_PARSING == "dependency_parsing"
        assert ExtractionMethod.RULE_BASED == "rule_based"
    
    def test_graph_task_status_enum(self):
        """测试图谱任务状态枚举"""
        assert GraphTaskStatus.PENDING == "pending"
        assert GraphTaskStatus.PROCESSING == "processing"
        assert GraphTaskStatus.COMPLETED == "completed"
        assert GraphTaskStatus.FAILED == "failed"
        assert GraphTaskStatus.CANCELLED == "cancelled"
    
    def test_language_enum(self):
        """测试语言枚举"""
        assert Language.CHINESE == "zh"
        assert Language.ENGLISH == "en"


class TestGraphProjectModels:
    """图谱项目模型测试"""
    
    def test_graph_project_creation(self):
        """测试图谱项目创建"""
        project = GraphProject(
            name="历史人物知识图谱",
            description="中国历史人物关系图谱",
            domain="历史学",
            language=Language.CHINESE,
            entity_types=[EntityType.PERSON, EntityType.LOCATION, EntityType.EVENT],
            relation_types=[RelationType.BORN_IN, RelationType.WORKED_AT, RelationType.INFLUENCED]
        )
        
        assert project.name == "历史人物知识图谱"
        assert project.description == "中国历史人物关系图谱"
        assert project.domain == "历史学"
        assert project.language == Language.CHINESE
        assert len(project.entity_types) == 3
        assert len(project.relation_types) == 3
        assert EntityType.PERSON in project.entity_types
        assert RelationType.BORN_IN in project.relation_types
        assert project.id is not None
        assert isinstance(project.created_at, datetime)
    
    def test_create_graph_project_request(self):
        """测试创建图谱项目请求"""
        request = CreateGraphProjectRequest(
            name="古代文学图谱",
            description="古代文学作品和作者关系图谱",
            domain="文学",
            language=Language.CHINESE,
            entity_types=[EntityType.PERSON, EntityType.WORK],
            relation_types=[RelationType.INFLUENCED, RelationType.BELONGS_TO]
        )
        
        assert request.name == "古代文学图谱"
        assert request.domain == "文学"
        assert len(request.entity_types) == 2
        assert len(request.relation_types) == 2
    
    def test_create_project_request_validation(self):
        """测试创建项目请求验证"""
        # 测试空名称
        with pytest.raises(ValidationError) as exc_info:
            CreateGraphProjectRequest(
                name="",
                domain="测试"
            )
        
        assert "项目名称不能为空" in str(exc_info.value)
        
        # 测试只有空白字符的名称
        with pytest.raises(ValidationError) as exc_info:
            CreateGraphProjectRequest(
                name="   ",
                domain="测试"
            )
        
        assert "项目名称不能为空" in str(exc_info.value)
    
    def test_create_graph_project_response(self):
        """测试创建图谱项目响应"""
        project = GraphProject(
            name="测试项目",
            domain="测试域"
        )
        
        response = CreateGraphProjectResponse(
            success=True,
            message="项目创建成功",
            project_id=project.id,
            project=project
        )
        
        assert response.success is True
        assert response.project_id == project.id
        assert response.project.name == "测试项目"


class TestEntityModels:
    """实体模型测试"""
    
    def test_entity_creation(self):
        """测试实体创建"""
        entity = Entity(
            name="李白",
            entity_type=EntityType.PERSON,
            aliases=["太白", "李太白", "诗仙"],
            description="唐代著名诗人",
            properties={"朝代": "唐", "职业": "诗人"},
            confidence_score=0.95,
            mention_count=10,
            source_documents=["doc1", "doc2", "doc3"]
        )
        
        assert entity.name == "李白"
        assert entity.entity_type == EntityType.PERSON
        assert len(entity.aliases) == 3
        assert "太白" in entity.aliases
        assert entity.properties["朝代"] == "唐"
        assert entity.confidence_score == 0.95
        assert entity.mention_count == 10
        assert len(entity.source_documents) == 3
        assert entity.id is not None
    
    def test_extracted_entity_creation(self):
        """测试抽取实体创建"""
        extracted = ExtractedEntity(
            name="杜甫",
            entity_type=EntityType.PERSON,
            start_pos=10,
            end_pos=12,
            confidence_score=0.88,
            context="唐代诗人杜甫的作品"
        )
        
        assert extracted.name == "杜甫"
        assert extracted.entity_type == EntityType.PERSON
        assert extracted.start_pos == 10
        assert extracted.end_pos == 12
        assert extracted.confidence_score == 0.88
        assert extracted.context == "唐代诗人杜甫的作品"
    
    def test_extracted_entity_position_validation(self):
        """测试抽取实体位置验证"""
        # 测试结束位置必须大于开始位置
        with pytest.raises(ValidationError) as exc_info:
            ExtractedEntity(
                name="测试",
                entity_type=EntityType.PERSON,
                start_pos=10,
                end_pos=10,  # 等于开始位置
                confidence_score=0.8
            )
        
        assert "结束位置必须大于开始位置" in str(exc_info.value)
        
        with pytest.raises(ValidationError):
            ExtractedEntity(
                name="测试",
                entity_type=EntityType.PERSON,
                start_pos=15,
                end_pos=10,  # 小于开始位置
                confidence_score=0.8
            )
    
    def test_entity_confidence_validation(self):
        """测试实体置信度验证"""
        # 有效置信度
        entity = Entity(
            name="测试实体",
            entity_type=EntityType.CONCEPT,
            confidence_score=0.75
        )
        assert entity.confidence_score == 0.75
        
        # 边界值
        entity_min = Entity(
            name="测试实体",
            entity_type=EntityType.CONCEPT,
            confidence_score=0.0
        )
        assert entity_min.confidence_score == 0.0
        
        entity_max = Entity(
            name="测试实体",
            entity_type=EntityType.CONCEPT,
            confidence_score=1.0
        )
        assert entity_max.confidence_score == 1.0
        
        # 无效置信度
        with pytest.raises(ValidationError):
            Entity(
                name="测试实体",
                entity_type=EntityType.CONCEPT,
                confidence_score=-0.1
            )
        
        with pytest.raises(ValidationError):
            Entity(
                name="测试实体",
                entity_type=EntityType.CONCEPT,
                confidence_score=1.1
            )


class TestRelationModels:
    """关系模型测试"""
    
    def test_relation_creation(self):
        """测试关系创建"""
        relation = Relation(
            subject_entity_id="person1",
            predicate=RelationType.BORN_IN,
            object_entity_id="location1",
            confidence_score=0.9,
            context="李白出生于碎叶城",
            source_sentence="诗人李白出生于碎叶城",
            source_document_id="doc123",
            properties={"时间": "701年"}
        )
        
        assert relation.subject_entity_id == "person1"
        assert relation.predicate == RelationType.BORN_IN
        assert relation.object_entity_id == "location1"
        assert relation.confidence_score == 0.9
        assert relation.context == "李白出生于碎叶城"
        assert relation.source_sentence == "诗人李白出生于碎叶城"
        assert relation.source_document_id == "doc123"
        assert relation.properties["时间"] == "701年"
        assert relation.id is not None
    
    def test_extracted_relation_creation(self):
        """测试抽取关系创建"""
        subject = ExtractedEntity(
            name="白居易",
            entity_type=EntityType.PERSON,
            start_pos=0,
            end_pos=3,
            confidence_score=0.92
        )
        
        object_entity = ExtractedEntity(
            name="长安",
            entity_type=EntityType.LOCATION,
            start_pos=10,
            end_pos=12,
            confidence_score=0.88
        )
        
        extracted_relation = ExtractedRelation(
            subject_entity=subject,
            predicate=RelationType.WORKED_AT,
            object_entity=object_entity,
            confidence_score=0.85,
            context="白居易在长安任职",
            source_sentence="白居易曾在长安担任翰林学士"
        )
        
        assert extracted_relation.subject_entity.name == "白居易"
        assert extracted_relation.predicate == RelationType.WORKED_AT
        assert extracted_relation.object_entity.name == "长安"
        assert extracted_relation.confidence_score == 0.85
        assert extracted_relation.context == "白居易在长安任职"


class TestExtractionRequestModels:
    """抽取请求模型测试"""
    
    def test_entity_extraction_request(self):
        """测试实体抽取请求"""
        request = EntityExtractionRequest(
            project_id="project123",
            text="李白是唐代著名诗人，出生于碎叶城。",
            document_id="doc456",
            method=ExtractionMethod.SPACY_NER,
            config={"threshold": 0.8, "language": "zh"}
        )
        
        assert request.project_id == "project123"
        assert request.text == "李白是唐代著名诗人，出生于碎叶城。"
        assert request.document_id == "doc456"
        assert request.method == ExtractionMethod.SPACY_NER
        assert request.config["threshold"] == 0.8
    
    def test_entity_extraction_text_validation(self):
        """测试实体抽取文本验证"""
        # 测试空文本
        with pytest.raises(ValidationError) as exc_info:
            EntityExtractionRequest(
                project_id="project123",
                text=""
            )
        
        assert "待抽取文本不能为空" in str(exc_info.value)
        
        # 测试只有空白字符的文本
        with pytest.raises(ValidationError) as exc_info:
            EntityExtractionRequest(
                project_id="project123",
                text="   "
            )
        
        assert "待抽取文本不能为空" in str(exc_info.value)
    
    def test_relation_extraction_request(self):
        """测试关系抽取请求"""
        request = RelationExtractionRequest(
            project_id="project789",
            text="杜甫师从李白，深受其影响。",
            document_id="doc789",
            method=ExtractionMethod.PATTERN_BASED,
            config={"patterns": ["师从", "影响"]}
        )
        
        assert request.project_id == "project789"
        assert request.text == "杜甫师从李白，深受其影响。"
        assert request.method == ExtractionMethod.PATTERN_BASED
        assert "patterns" in request.config
    
    def test_batch_extraction_request(self):
        """测试批量抽取请求"""
        documents = [
            {"id": "doc1", "text": "李白是唐代诗人"},
            {"id": "doc2", "text": "杜甫师从李白"},
            {"id": "doc3", "text": "白居易生于洛阳"}
        ]
        
        request = BatchExtractionRequest(
            project_id="batch_project",
            documents=documents,
            entity_extraction=True,
            relation_extraction=True,
            entity_method=ExtractionMethod.BERT_NER,
            relation_method=ExtractionMethod.RULE_BASED
        )
        
        assert request.project_id == "batch_project"
        assert len(request.documents) == 3
        assert request.entity_extraction is True
        assert request.relation_extraction is True
        assert request.entity_method == ExtractionMethod.BERT_NER
        assert request.relation_method == ExtractionMethod.RULE_BASED
    
    def test_batch_extraction_documents_validation(self):
        """测试批量抽取文档验证"""
        # 测试空文档列表
        with pytest.raises(ValidationError) as exc_info:
            BatchExtractionRequest(
                project_id="project",
                documents=[]
            )
        
        assert "文档列表不能为空" in str(exc_info.value)
        
        # 测试缺少必需字段的文档
        with pytest.raises(ValidationError) as exc_info:
            BatchExtractionRequest(
                project_id="project",
                documents=[{"id": "doc1"}]  # 缺少text字段
            )
        
        assert "每个文档必须包含id和text字段" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            BatchExtractionRequest(
                project_id="project",
                documents=[{"text": "content"}]  # 缺少id字段
            )
        
        assert "每个文档必须包含id和text字段" in str(exc_info.value)


class TestResponseModels:
    """响应模型测试"""
    
    def test_entity_extraction_response(self):
        """测试实体抽取响应"""
        entities = [
            ExtractedEntity(
                name="苏轼",
                entity_type=EntityType.PERSON,
                start_pos=0,
                end_pos=2,
                confidence_score=0.95
            ),
            ExtractedEntity(
                name="宋朝",
                entity_type=EntityType.TIME,
                start_pos=10,
                end_pos=12,
                confidence_score=0.88
            )
        ]
        
        response = EntityExtractionResponse(
            success=True,
            task_id="task123",
            entities_found=2,
            entities=entities,
            processing_time=1.25
        )
        
        assert response.success is True
        assert response.task_id == "task123"
        assert response.entities_found == 2
        assert len(response.entities) == 2
        assert response.processing_time == 1.25
        assert response.entities[0].name == "苏轼"
    
    def test_relation_extraction_response(self):
        """测试关系抽取响应"""
        subject = ExtractedEntity(
            name="王安石", entity_type=EntityType.PERSON,
            start_pos=0, end_pos=3, confidence_score=0.9
        )
        object_entity = ExtractedEntity(
            name="变法", entity_type=EntityType.EVENT,
            start_pos=5, end_pos=7, confidence_score=0.85
        )
        
        relations = [
            ExtractedRelation(
                subject_entity=subject,
                predicate=RelationType.PARTICIPATED_IN,
                object_entity=object_entity,
                confidence_score=0.92
            )
        ]
        
        response = RelationExtractionResponse(
            success=True,
            task_id="rel_task456",
            relations_found=1,
            relations=relations,
            processing_time=2.1
        )
        
        assert response.success is True
        assert response.relations_found == 1
        assert len(response.relations) == 1
        assert response.relations[0].predicate == RelationType.PARTICIPATED_IN
    
    def test_batch_extraction_response(self):
        """测试批量抽取响应"""
        response = BatchExtractionResponse(
            success=True,
            batch_id="batch789",
            total_documents=100,
            processed_documents=98,
            failed_documents=2,
            total_entities=250,
            total_relations=180,
            processing_time=45.6
        )
        
        assert response.success is True
        assert response.batch_id == "batch789"
        assert response.total_documents == 100
        assert response.processed_documents == 98
        assert response.failed_documents == 2
        assert response.total_entities == 250
        assert response.total_relations == 180
        assert response.processing_time == 45.6


class TestGraphModels:
    """图谱模型测试"""
    
    def test_graph_construction_request(self):
        """测试图谱构建请求"""
        request = GraphConstructionRequest(
            project_id="graph_project",
            include_entities=True,
            include_relations=True,
            min_confidence=0.8,
            max_nodes=2000,
            config={"layout": "force_directed", "clustering": True}
        )
        
        assert request.project_id == "graph_project"
        assert request.include_entities is True
        assert request.include_relations is True
        assert request.min_confidence == 0.8
        assert request.max_nodes == 2000
        assert request.config["layout"] == "force_directed"
    
    def test_graph_construction_response(self):
        """测试图谱构建响应"""
        response = GraphConstructionResponse(
            success=True,
            task_id="construct_task",
            nodes_count=150,
            edges_count=280,
            processing_time=12.5
        )
        
        assert response.success is True
        assert response.task_id == "construct_task"
        assert response.nodes_count == 150
        assert response.edges_count == 280
        assert response.processing_time == 12.5
    
    def test_graph_node_creation(self):
        """测试图节点创建"""
        node = GraphNode(
            id="node_001",
            label="李白",
            entity_type="PERSON",
            properties={
                "朝代": "唐",
                "职业": "诗人",
                "别名": ["太白", "诗仙"]
            }
        )
        
        assert node.id == "node_001"
        assert node.label == "李白"
        assert node.entity_type == "PERSON"
        assert node.properties["朝代"] == "唐"
        assert "诗仙" in node.properties["别名"]
    
    def test_graph_edge_creation(self):
        """测试图边创建"""
        edge = GraphEdge(
            id="edge_001",
            source="person_001",
            target="location_001",
            relation_type="出生于",
            properties={
                "confidence": 0.95,
                "source_text": "李白出生于碎叶城"
            }
        )
        
        assert edge.id == "edge_001"
        assert edge.source == "person_001"
        assert edge.target == "location_001"
        assert edge.relation_type == "出生于"
        assert edge.properties["confidence"] == 0.95
    
    def test_graph_query_request(self):
        """测试图谱查询请求"""
        request = GraphQueryRequest(
            project_id="query_project",
            query_type="find_neighbors",
            parameters={"entity_id": "person_001", "depth": 2},
            limit=50,
            offset=0
        )
        
        assert request.project_id == "query_project"
        assert request.query_type == "find_neighbors"
        assert request.parameters["entity_id"] == "person_001"
        assert request.parameters["depth"] == 2
        assert request.limit == 50
        assert request.offset == 0
    
    def test_graph_query_response(self):
        """测试图谱查询响应"""
        nodes = [
            GraphNode(id="n1", label="节点1", entity_type="PERSON"),
            GraphNode(id="n2", label="节点2", entity_type="LOCATION")
        ]
        
        edges = [
            GraphEdge(id="e1", source="n1", target="n2", relation_type="位于")
        ]
        
        response = GraphQueryResponse(
            success=True,
            nodes=nodes,
            edges=edges,
            total_count=2,
            query_time=0.85
        )
        
        assert response.success is True
        assert len(response.nodes) == 2
        assert len(response.edges) == 1
        assert response.total_count == 2
        assert response.query_time == 0.85


class TestAnalysisModels:
    """分析模型测试"""
    
    def test_graph_analysis_request(self):
        """测试图谱分析请求"""
        request = GraphAnalysisRequest(
            project_id="analysis_project",
            analysis_type="centrality_analysis",
            parameters={
                "metrics": ["betweenness", "closeness", "pagerank"],
                "top_k": 20
            }
        )
        
        assert request.project_id == "analysis_project"
        assert request.analysis_type == "centrality_analysis"
        assert "betweenness" in request.parameters["metrics"]
        assert request.parameters["top_k"] == 20
    
    def test_graph_metrics_creation(self):
        """测试图谱指标创建"""
        metrics = GraphMetrics(
            node_count=1000,
            edge_count=2500,
            density=0.025,
            average_degree=5.0,
            clustering_coefficient=0.35,
            connected_components=3
        )
        
        assert metrics.node_count == 1000
        assert metrics.edge_count == 2500
        assert metrics.density == 0.025
        assert metrics.average_degree == 5.0
        assert metrics.clustering_coefficient == 0.35
        assert metrics.connected_components == 3
    
    def test_graph_analysis_response(self):
        """测试图谱分析响应"""
        metrics = GraphMetrics(
            node_count=500,
            edge_count=1200,
            density=0.048,
            average_degree=4.8,
            clustering_coefficient=0.42,
            connected_components=1
        )
        
        response = GraphAnalysisResponse(
            success=True,
            analysis_type="network_statistics",
            metrics=metrics,
            results={
                "most_central_nodes": ["李白", "杜甫", "白居易"],
                "communities": 5,
                "diameter": 8
            },
            processing_time=3.2
        )
        
        assert response.success is True
        assert response.analysis_type == "network_statistics"
        assert response.metrics.node_count == 500
        assert len(response.results["most_central_nodes"]) == 3
        assert response.results["communities"] == 5
        assert response.processing_time == 3.2


class TestTaskAndStatusModels:
    """任务和状态模型测试"""
    
    def test_task_status_creation(self):
        """测试任务状态创建"""
        task_status = TaskStatus(
            task_id="task_999",
            status=GraphTaskStatus.PROCESSING,
            progress=75.5,
            message="正在处理实体抽取"
        )
        
        assert task_status.task_id == "task_999"
        assert task_status.status == GraphTaskStatus.PROCESSING
        assert task_status.progress == 75.5
        assert task_status.message == "正在处理实体抽取"
        assert task_status.started_at is None
        assert task_status.completed_at is None
        assert task_status.error_message is None
    
    def test_task_status_with_times(self):
        """测试带时间信息的任务状态"""
        start_time = datetime.now()
        end_time = datetime.now()
        
        task_status = TaskStatus(
            task_id="task_completed",
            status=GraphTaskStatus.COMPLETED,
            progress=100.0,
            message="任务已完成",
            started_at=start_time,
            completed_at=end_time
        )
        
        assert task_status.status == GraphTaskStatus.COMPLETED
        assert task_status.progress == 100.0
        assert task_status.started_at == start_time
        assert task_status.completed_at == end_time
    
    def test_task_status_response(self):
        """测试任务状态响应"""
        task = TaskStatus(
            task_id="response_task",
            status=GraphTaskStatus.FAILED,
            progress=45.0,
            message="任务执行失败",
            error_message="内存不足导致处理失败"
        )
        
        response = TaskStatusResponse(
            success=False,
            message="任务状态查询成功",
            task=task
        )
        
        assert response.success is False
        assert response.task.task_id == "response_task"
        assert response.task.status == GraphTaskStatus.FAILED
        assert response.task.error_message == "内存不足导致处理失败"


class TestConceptMiningModels:
    """概念挖掘模型测试"""
    
    def test_concept_mining_request(self):
        """测试概念挖掘请求"""
        request = ConceptMiningRequest(
            project_id="concept_project",
            documents=["doc1", "doc2", "doc3", "doc4", "doc5"],
            num_topics=8,
            method="lda",
            config={
                "alpha": 0.1,
                "beta": 0.01,
                "iterations": 1000
            }
        )
        
        assert request.project_id == "concept_project"
        assert len(request.documents) == 5
        assert request.num_topics == 8
        assert request.method == "lda"
        assert request.config["alpha"] == 0.1
    
    def test_concept_topic_creation(self):
        """测试概念主题创建"""
        topic = ConceptTopic(
            id="topic_001",
            name="唐代诗歌",
            keywords=["诗歌", "唐代", "诗人", "文学", "作品"],
            weight=0.25,
            documents=["doc1", "doc3", "doc7"]
        )
        
        assert topic.id == "topic_001"
        assert topic.name == "唐代诗歌"
        assert len(topic.keywords) == 5
        assert "诗歌" in topic.keywords
        assert topic.weight == 0.25
        assert len(topic.documents) == 3
    
    def test_concept_mining_response(self):
        """测试概念挖掘响应"""
        topics = [
            ConceptTopic(
                id="topic_1",
                name="文学创作",
                keywords=["创作", "文学", "作品"],
                weight=0.3
            ),
            ConceptTopic(
                id="topic_2", 
                name="历史事件",
                keywords=["历史", "事件", "时代"],
                weight=0.25
            )
        ]
        
        response = ConceptMiningResponse(
            success=True,
            task_id="mining_task",
            topics=topics,
            processing_time=15.8
        )
        
        assert response.success is True
        assert response.task_id == "mining_task"
        assert len(response.topics) == 2
        assert response.topics[0].name == "文学创作"
        assert response.processing_time == 15.8


class TestStatisticsModels:
    """统计模型测试"""
    
    def test_graph_statistics_creation(self):
        """测试图谱统计创建"""
        entity_distribution = {
            "PERSON": 150,
            "LOCATION": 80,
            "ORGANIZATION": 45,
            "EVENT": 30
        }
        
        relation_distribution = {
            "出生于": 120,
            "任职于": 85,
            "影响": 95,
            "位于": 70
        }
        
        statistics = GraphStatistics(
            project_count=5,
            entity_count=305,
            relation_count=370,
            document_count=1200,
            entity_type_distribution=entity_distribution,
            relation_type_distribution=relation_distribution,
            average_confidence_score=0.82,
            processing_statistics={
                "avg_processing_time": 2.5,
                "success_rate": 0.96,
                "total_processed": 15000
            }
        )
        
        assert statistics.project_count == 5
        assert statistics.entity_count == 305
        assert statistics.relation_count == 370
        assert statistics.entity_type_distribution["PERSON"] == 150
        assert statistics.relation_type_distribution["出生于"] == 120
        assert statistics.average_confidence_score == 0.82
        assert statistics.processing_statistics["success_rate"] == 0.96
    
    def test_graph_statistics_response(self):
        """测试图谱统计响应"""
        stats = GraphStatistics(
            project_count=3,
            entity_count=200,
            relation_count=150,
            document_count=500,
            average_confidence_score=0.85
        )
        
        response = GraphStatisticsResponse(
            success=True,
            message="统计信息获取成功",
            statistics=stats
        )
        
        assert response.success is True
        assert response.statistics.project_count == 3
        assert response.statistics.entity_count == 200
        assert isinstance(response.generated_at, datetime)


class TestValidationAndEdgeCases:
    """验证和边界条件测试"""
    
    def test_confidence_score_boundaries(self):
        """测试置信度边界值"""
        # 测试最小值
        entity_min = Entity(
            name="测试",
            entity_type=EntityType.PERSON,
            confidence_score=0.0
        )
        assert entity_min.confidence_score == 0.0
        
        # 测试最大值
        entity_max = Entity(
            name="测试",
            entity_type=EntityType.PERSON,
            confidence_score=1.0
        )
        assert entity_max.confidence_score == 1.0
        
        # 测试超出范围
        with pytest.raises(ValidationError):
            Entity(
                name="测试",
                entity_type=EntityType.PERSON,
                confidence_score=-0.1
            )
        
        with pytest.raises(ValidationError):
            Entity(
                name="测试",
                entity_type=EntityType.PERSON,
                confidence_score=1.1
            )
    
    def test_progress_boundaries(self):
        """测试进度边界值"""
        # 有效进度值
        task = TaskStatus(
            task_id="test",
            status=GraphTaskStatus.PROCESSING,
            progress=50.0
        )
        assert task.progress == 50.0
        
        # 边界值
        task_min = TaskStatus(
            task_id="test",
            status=GraphTaskStatus.PENDING,
            progress=0.0
        )
        assert task_min.progress == 0.0
        
        task_max = TaskStatus(
            task_id="test",
            status=GraphTaskStatus.COMPLETED,
            progress=100.0
        )
        assert task_max.progress == 100.0
        
        # 超出范围
        with pytest.raises(ValidationError):
            TaskStatus(
                task_id="test",
                status=GraphTaskStatus.PROCESSING,
                progress=-1.0
            )
        
        with pytest.raises(ValidationError):
            TaskStatus(
                task_id="test",
                status=GraphTaskStatus.PROCESSING,
                progress=101.0
            )
    
    def test_string_length_limits(self):
        """测试字符串长度限制"""
        # 测试项目名称长度
        long_name = "x" * 201
        with pytest.raises(ValidationError):
            CreateGraphProjectRequest(
                name=long_name,
                domain="test"
            )
        
        # 测试域名长度
        long_domain = "x" * 101
        with pytest.raises(ValidationError):
            CreateGraphProjectRequest(
                name="test",
                domain=long_domain
            )
        
        # 测试实体名称长度
        long_entity_name = "x" * 501
        with pytest.raises(ValidationError):
            Entity(
                name=long_entity_name,
                entity_type=EntityType.PERSON,
                confidence_score=0.8
            )
        
        # 测试文本长度限制
        long_text = "x" * 10001
        with pytest.raises(ValidationError):
            EntityExtractionRequest(
                project_id="test",
                text=long_text
            )
    
    def test_enum_value_conversion(self):
        """测试枚举值转换"""
        # 测试字符串到枚举的转换
        entity = Entity(
            name="测试实体",
            entity_type="PERSON",  # 字符串形式
            confidence_score=0.8
        )
        assert entity.entity_type == EntityType.PERSON
        
        # 测试关系类型转换
        relation = Relation(
            subject_entity_id="e1",
            predicate="出生于",  # 字符串形式
            object_entity_id="e2",
            confidence_score=0.9
        )
        assert relation.predicate == RelationType.BORN_IN