# Story 3.1: 搜索引擎服务

## 基本信息
- **Epic**: Epic 3 - 智能搜索和推荐系统
- **Story ID**: 3.1
- **优先级**: 高
- **预估工作量**: 5天
- **负责团队**: 后端开发团队 + 前端开发团队

## 用户故事

**作为** 历史文本研究人员  
**我希望** 能够使用智能搜索引擎快速查找相关的历史文档和内容  
**以便于** 高效地进行文献检索、内容发现和研究分析

## 需求描述

### 核心功能需求
1. **全文搜索功能**
   - 支持中文分词和语义搜索
   - 多字段搜索（标题、内容、标签、作者等）
   - 模糊搜索和精确匹配
   - 搜索结果高亮显示

2. **高级搜索功能**
   - 布尔搜索（AND、OR、NOT）
   - 范围搜索（时间、数值）
   - 通配符和正则表达式搜索
   - 字段特定搜索

3. **智能搜索功能**
   - 搜索建议和自动补全
   - 拼写纠错和同义词扩展
   - 语义相似度搜索
   - 个性化搜索排序

4. **搜索结果管理**
   - 多维度排序（相关性、时间、热度）
   - 分页和无限滚动
   - 搜索结果过滤和筛选
   - 搜索历史和收藏

5. **搜索分析功能**
   - 搜索统计和热词分析
   - 用户搜索行为分析
   - 搜索性能监控
   - A/B测试支持

## 技术实现

### 核心技术栈
- **搜索引擎**: Elasticsearch 8.x
- **后端框架**: FastAPI
- **数据库**: PostgreSQL (元数据), Redis (缓存)
- **消息队列**: RabbitMQ
- **NLP库**: jieba, HanLP, transformers
- **机器学习**: scikit-learn, sentence-transformers
- **监控**: Prometheus, Grafana

### 数据模型设计

#### Elasticsearch索引结构
```json
{
  "mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "title": {
        "type": "text",
        "analyzer": "ik_max_word",
        "search_analyzer": "ik_smart",
        "fields": {
          "keyword": {"type": "keyword"}
        }
      },
      "content": {
        "type": "text",
        "analyzer": "ik_max_word",
        "search_analyzer": "ik_smart"
      },
      "summary": {
        "type": "text",
        "analyzer": "ik_max_word"
      },
      "author": {"type": "keyword"},
      "dynasty": {"type": "keyword"},
      "era": {"type": "keyword"},
      "category": {"type": "keyword"},
      "tags": {"type": "keyword"},
      "created_time": {"type": "date"},
      "updated_time": {"type": "date"},
      "document_type": {"type": "keyword"},
      "importance_score": {"type": "float"},
      "quality_score": {"type": "float"},
      "view_count": {"type": "integer"},
      "like_count": {"type": "integer"},
      "embedding_vector": {
        "type": "dense_vector",
        "dims": 768
      },
      "keywords": {"type": "keyword"},
      "entities": {
        "type": "nested",
        "properties": {
          "name": {"type": "keyword"},
          "type": {"type": "keyword"},
          "confidence": {"type": "float"}
        }
      },
      "location": {"type": "geo_point"},
      "language": {"type": "keyword"},
      "source": {"type": "keyword"}
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "ik_max_word": {
          "type": "ik_max_word"
        },
        "ik_smart": {
          "type": "ik_smart"
        }
      }
    }
  }
}
```

#### PostgreSQL数据模型
```sql
-- 搜索配置表
CREATE TABLE search_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    config_data JSONB NOT NULL,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 搜索历史表
CREATE TABLE search_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    query_text TEXT NOT NULL,
    search_type VARCHAR(50) DEFAULT 'basic',
    filters JSONB,
    results_count INTEGER,
    response_time_ms INTEGER,
    clicked_results JSONB,
    session_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 搜索建议表
CREATE TABLE search_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text VARCHAR(200) NOT NULL,
    suggestion_text VARCHAR(200) NOT NULL,
    suggestion_type VARCHAR(50) DEFAULT 'completion',
    frequency INTEGER DEFAULT 1,
    score FLOAT DEFAULT 0.0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 热门搜索表
CREATE TABLE popular_searches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text VARCHAR(200) NOT NULL,
    search_count INTEGER DEFAULT 1,
    period_type VARCHAR(20) DEFAULT 'daily', -- daily, weekly, monthly
    period_date DATE NOT NULL,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户搜索偏好表
CREATE TABLE user_search_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    preferred_categories JSONB,
    preferred_eras JSONB,
    search_behavior JSONB,
    personalization_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_search_history_user_id ON search_history(user_id);
CREATE INDEX idx_search_history_created_at ON search_history(created_at);
CREATE INDEX idx_search_suggestions_query ON search_suggestions(query_text);
CREATE INDEX idx_popular_searches_period ON popular_searches(period_date, period_type);
CREATE INDEX idx_user_preferences_user_id ON user_search_preferences(user_id);
```

### 服务架构设计

#### 搜索服务核心组件
```python
# search_service.py
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import jieba
import re

class SearchService:
    """
    搜索引擎核心服务类
    提供全文搜索、语义搜索、智能建议等功能
    """
    
    def __init__(self, es_client: Elasticsearch, embedding_model: SentenceTransformer):
        self.es = es_client
        self.embedding_model = embedding_model
        self.index_name = "historical_documents"
    
    async def basic_search(self, 
                          query: str, 
                          page: int = 1, 
                          size: int = 20,
                          filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        基础全文搜索功能
        
        Args:
            query: 搜索查询字符串
            page: 页码
            size: 每页结果数
            filters: 过滤条件
        
        Returns:
            搜索结果字典
        """
        # 构建搜索查询
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^3",
                                    "content^2", 
                                    "summary^2",
                                    "author",
                                    "tags"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "filter": self._build_filters(filters) if filters else []
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    },
                    "summary": {}
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"importance_score": {"order": "desc"}},
                {"created_time": {"order": "desc"}}
            ],
            "from": (page - 1) * size,
            "size": size
        }
        
        # 执行搜索
        response = await self.es.search(
            index=self.index_name,
            body=search_body
        )
        
        return self._format_search_results(response)
    
    async def semantic_search(self, 
                             query: str, 
                             page: int = 1, 
                             size: int = 20,
                             similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        语义搜索功能
        
        Args:
            query: 搜索查询字符串
            page: 页码
            size: 每页结果数
            similarity_threshold: 相似度阈值
        
        Returns:
            语义搜索结果
        """
        # 生成查询向量
        query_vector = self.embedding_model.encode(query).tolist()
        
        search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    },
                    "min_score": similarity_threshold + 1.0
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            },
            "from": (page - 1) * size,
            "size": size
        }
        
        response = await self.es.search(
            index=self.index_name,
            body=search_body
        )
        
        return self._format_search_results(response)
    
    async def advanced_search(self, 
                             search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        高级搜索功能
        
        Args:
            search_params: 高级搜索参数
        
        Returns:
            搜索结果
        """
        query_clauses = []
        
        # 处理不同类型的搜索条件
        if search_params.get('title'):
            query_clauses.append({
                "match": {
                    "title": {
                        "query": search_params['title'],
                        "boost": 2.0
                    }
                }
            })
        
        if search_params.get('content'):
            query_clauses.append({
                "match": {
                    "content": search_params['content']
                }
            })
        
        if search_params.get('author'):
            query_clauses.append({
                "term": {
                    "author": search_params['author']
                }
            })
        
        if search_params.get('date_range'):
            query_clauses.append({
                "range": {
                    "created_time": search_params['date_range']
                }
            })
        
        # 构建布尔查询
        search_body = {
            "query": {
                "bool": {
                    "must": query_clauses,
                    "filter": self._build_filters(search_params.get('filters'))
                }
            },
            "highlight": {
                "fields": {
                    "title": {},
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            },
            "sort": self._build_sort(search_params.get('sort')),
            "from": search_params.get('from', 0),
            "size": search_params.get('size', 20)
        }
        
        response = await self.es.search(
            index=self.index_name,
            body=search_body
        )
        
        return self._format_search_results(response)
    
    async def get_suggestions(self, 
                             query: str, 
                             size: int = 10) -> List[str]:
        """
        获取搜索建议
        
        Args:
            query: 部分查询字符串
            size: 建议数量
        
        Returns:
            建议列表
        """
        search_body = {
            "suggest": {
                "text": query,
                "completion_suggest": {
                    "completion": {
                        "field": "title.suggest",
                        "size": size,
                        "skip_duplicates": True
                    }
                },
                "phrase_suggest": {
                    "phrase": {
                        "field": "content",
                        "size": size,
                        "gram_size": 3,
                        "direct_generator": [{
                            "field": "content",
                            "suggest_mode": "missing"
                        }]
                    }
                }
            }
        }
        
        response = await self.es.search(
            index=self.index_name,
            body=search_body
        )
        
        suggestions = []
        
        # 处理自动补全建议
        for suggest in response['suggest']['completion_suggest']:
            for option in suggest['options']:
                suggestions.append(option['text'])
        
        # 处理短语建议
        for suggest in response['suggest']['phrase_suggest']:
            for option in suggest['options']:
                suggestions.append(option['text'])
        
        return list(set(suggestions))[:size]
    
    async def get_aggregations(self, 
                              query: str = "*", 
                              agg_fields: List[str] = None) -> Dict[str, Any]:
        """
        获取聚合统计信息
        
        Args:
            query: 搜索查询
            agg_fields: 聚合字段列表
        
        Returns:
            聚合结果
        """
        if agg_fields is None:
            agg_fields = ['category', 'dynasty', 'era', 'author']
        
        aggregations = {}
        for field in agg_fields:
            aggregations[f"{field}_stats"] = {
                "terms": {
                    "field": field,
                    "size": 20
                }
            }
        
        search_body = {
            "query": {
                "query_string": {
                    "query": query
                }
            },
            "aggs": aggregations,
            "size": 0
        }
        
        response = await self.es.search(
            index=self.index_name,
            body=search_body
        )
        
        return response['aggregations']
    
    def _build_filters(self, filters: Optional[Dict]) -> List[Dict]:
        """
        构建过滤条件
        
        Args:
            filters: 过滤参数字典
        
        Returns:
            Elasticsearch过滤条件列表
        """
        if not filters:
            return []
        
        filter_clauses = []
        
        # 分类过滤
        if filters.get('categories'):
            filter_clauses.append({
                "terms": {
                    "category": filters['categories']
                }
            })
        
        # 朝代过滤
        if filters.get('dynasties'):
            filter_clauses.append({
                "terms": {
                    "dynasty": filters['dynasties']
                }
            })
        
        # 时间范围过滤
        if filters.get('date_range'):
            filter_clauses.append({
                "range": {
                    "created_time": filters['date_range']
                }
            })
        
        # 重要性评分过滤
        if filters.get('importance_range'):
            filter_clauses.append({
                "range": {
                    "importance_score": filters['importance_range']
                }
            })
        
        return filter_clauses
    
    def _build_sort(self, sort_params: Optional[Dict]) -> List[Dict]:
        """
        构建排序条件
        
        Args:
            sort_params: 排序参数
        
        Returns:
            Elasticsearch排序条件列表
        """
        if not sort_params:
            return [
                {"_score": {"order": "desc"}},
                {"importance_score": {"order": "desc"}}
            ]
        
        sort_clauses = []
        
        for field, order in sort_params.items():
            if field == 'relevance':
                sort_clauses.append({"_score": {"order": order}})
            elif field == 'date':
                sort_clauses.append({"created_time": {"order": order}})
            elif field == 'importance':
                sort_clauses.append({"importance_score": {"order": order}})
            elif field == 'popularity':
                sort_clauses.append({"view_count": {"order": order}})
            else:
                sort_clauses.append({field: {"order": order}})
        
        return sort_clauses
    
    def _format_search_results(self, response: Dict) -> Dict[str, Any]:
        """
        格式化搜索结果
        
        Args:
            response: Elasticsearch响应
        
        Returns:
            格式化后的搜索结果
        """
        hits = response['hits']
        
        results = {
            'total': hits['total']['value'],
            'max_score': hits['max_score'],
            'took': response['took'],
            'documents': []
        }
        
        for hit in hits['hits']:
            doc = {
                'id': hit['_id'],
                'score': hit['_score'],
                'source': hit['_source'],
                'highlight': hit.get('highlight', {})
            }
            results['documents'].append(doc)
        
        return results

class SearchAnalyticsService:
    """
    搜索分析服务
    提供搜索统计、用户行为分析等功能
    """
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def record_search(self, 
                           user_id: str, 
                           query: str, 
                           search_type: str,
                           results_count: int,
                           response_time: int,
                           session_id: str = None,
                           ip_address: str = None) -> None:
        """
        记录搜索历史
        
        Args:
            user_id: 用户ID
            query: 搜索查询
            search_type: 搜索类型
            results_count: 结果数量
            response_time: 响应时间
            session_id: 会话ID
            ip_address: IP地址
        """
        search_record = {
            'user_id': user_id,
            'query_text': query,
            'search_type': search_type,
            'results_count': results_count,
            'response_time_ms': response_time,
            'session_id': session_id,
            'ip_address': ip_address
        }
        
        # 插入搜索历史记录
        await self.db.execute(
            "INSERT INTO search_history (user_id, query_text, search_type, results_count, response_time_ms, session_id, ip_address) VALUES (:user_id, :query_text, :search_type, :results_count, :response_time_ms, :session_id, :ip_address)",
            search_record
        )
        
        # 更新热门搜索统计
        await self._update_popular_searches(query)
    
    async def get_search_analytics(self, 
                                  period: str = 'daily',
                                  limit: int = 100) -> Dict[str, Any]:
        """
        获取搜索分析数据
        
        Args:
            period: 统计周期 (daily, weekly, monthly)
            limit: 结果限制
        
        Returns:
            搜索分析数据
        """
        # 获取热门搜索词
        popular_queries = await self.db.fetch_all(
            "SELECT query_text, search_count FROM popular_searches WHERE period_type = :period ORDER BY search_count DESC LIMIT :limit",
            {'period': period, 'limit': limit}
        )
        
        # 获取搜索趋势
        search_trends = await self.db.fetch_all(
            "SELECT DATE(created_at) as date, COUNT(*) as search_count FROM search_history WHERE created_at >= NOW() - INTERVAL '30 days' GROUP BY DATE(created_at) ORDER BY date"
        )
        
        # 获取用户搜索行为统计
        user_stats = await self.db.fetch_one(
            "SELECT COUNT(DISTINCT user_id) as unique_users, AVG(results_count) as avg_results, AVG(response_time_ms) as avg_response_time FROM search_history WHERE created_at >= NOW() - INTERVAL '7 days'"
        )
        
        return {
            'popular_queries': [dict(row) for row in popular_queries],
            'search_trends': [dict(row) for row in search_trends],
            'user_statistics': dict(user_stats) if user_stats else {}
        }
    
    async def _update_popular_searches(self, query: str) -> None:
        """
        更新热门搜索统计
        
        Args:
            query: 搜索查询
        """
        today = datetime.now().date()
        
        # 更新或插入今日搜索统计
        await self.db.execute(
            """
            INSERT INTO popular_searches (query_text, search_count, period_type, period_date)
            VALUES (:query, 1, 'daily', :date)
            ON CONFLICT (query_text, period_type, period_date)
            DO UPDATE SET search_count = popular_searches.search_count + 1, updated_at = CURRENT_TIMESTAMP
            """,
            {'query': query, 'date': today}
        )
```

### API设计

#### 基础搜索API
```python
# 基础搜索
POST /api/v1/search/basic
Content-Type: application/json
Request: {
    "query": "明朝历史",
    "page": 1,
    "size": 20,
    "filters": {
        "categories": ["史书", "传记"],
        "dynasties": ["明朝"],
        "date_range": {
            "gte": "1368-01-01",
            "lte": "1644-12-31"
        }
    },
    "sort": {
        "relevance": "desc",
        "importance": "desc"
    }
}
Response: {
    "success": true,
    "data": {
        "total": 1250,
        "max_score": 8.5,
        "took": 45,
        "documents": [
            {
                "id": "doc_123",
                "score": 8.5,
                "source": {
                    "title": "明史·太祖本纪",
                    "content": "朱元璋...",
                    "author": "张廷玉",
                    "dynasty": "明朝",
                    "category": "史书",
                    "importance_score": 9.2
                },
                "highlight": {
                    "title": ["<em>明朝</em>史·太祖本纪"],
                    "content": ["<em>明朝</em>开国皇帝朱元璋..."]
                }
            }
        ]
    },
    "pagination": {
        "current_page": 1,
        "total_pages": 63,
        "has_next": true,
        "has_prev": false
    }
}

# 语义搜索
POST /api/v1/search/semantic
Content-Type: application/json
Request: {
    "query": "古代皇帝治国理政",
    "page": 1,
    "size": 20,
    "similarity_threshold": 0.7
}
Response: {
    "success": true,
    "data": {
        "total": 856,
        "documents": [...],
        "semantic_similarity": true
    }
}

# 高级搜索
POST /api/v1/search/advanced
Content-Type: application/json
Request: {
    "title": "本纪",
    "content": "治国",
    "author": "司马迁",
    "date_range": {
        "gte": "1000-01-01",
        "lte": "2000-12-31"
    },
    "boolean_query": {
        "must": ["皇帝", "政治"],
        "should": ["改革", "变法"],
        "must_not": ["战争"]
    },
    "page": 1,
    "size": 20
}
Response: {
    "success": true,
    "data": {
        "total": 342,
        "documents": [...]
    }
}

# 搜索建议
GET /api/v1/search/suggestions?q=明朝&size=10
Response: {
    "success": true,
    "suggestions": [
        "明朝历史",
        "明朝皇帝",
        "明朝政治",
        "明朝经济",
        "明朝文化"
    ]
}

# 搜索统计
GET /api/v1/search/aggregations?query=明朝
Response: {
    "success": true,
    "aggregations": {
        "category_stats": {
            "buckets": [
                {"key": "史书", "doc_count": 450},
                {"key": "传记", "doc_count": 320},
                {"key": "文学", "doc_count": 180}
            ]
        },
        "dynasty_stats": {
            "buckets": [
                {"key": "明朝", "doc_count": 1250}
            ]
        }
    }
}

# 搜索分析
GET /api/v1/search/analytics?period=daily&limit=50
Response: {
    "success": true,
    "data": {
        "popular_queries": [
            {"query_text": "明朝历史", "search_count": 156},
            {"query_text": "唐诗宋词", "search_count": 134}
        ],
        "search_trends": [
            {"date": "2024-01-15", "search_count": 1250},
            {"date": "2024-01-16", "search_count": 1380}
        ],
        "user_statistics": {
            "unique_users": 456,
            "avg_results": 23.5,
            "avg_response_time": 125.6
        }
    }
}
```

### 前端集成

#### Vue3搜索组件
```vue
<!-- src/components/search/IntelligentSearch.vue -->
<template>
  <div class="intelligent-search">
    <!-- 搜索输入框 -->
    <div class="search-container">
      <el-autocomplete
        v-model="searchQuery"
        :fetch-suggestions="getSuggestions"
        :trigger-on-focus="false"
        placeholder="请输入搜索关键词"
        class="search-input"
        size="large"
        @select="handleSuggestionSelect"
        @keyup.enter="performSearch"
      >
        <template #prefix>
          <el-icon><Search /></el-icon>
        </template>
        <template #suffix>
          <el-button 
            type="primary" 
            @click="performSearch"
            :loading="searchLoading"
          >
            搜索
          </el-button>
        </template>
      </el-autocomplete>
      
      <!-- 搜索模式切换 -->
      <div class="search-modes">
        <el-radio-group v-model="searchMode" size="small">
          <el-radio-button label="basic">基础搜索</el-radio-button>
          <el-radio-button label="semantic">语义搜索</el-radio-button>
          <el-radio-button label="advanced">高级搜索</el-radio-button>
        </el-radio-group>
      </div>
    </div>

    <!-- 高级搜索面板 -->
    <el-collapse v-if="searchMode === 'advanced'" v-model="advancedPanelOpen">
      <el-collapse-item title="高级搜索选项" name="advanced">
        <el-form :model="advancedForm" label-width="100px">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="标题">
                <el-input v-model="advancedForm.title" placeholder="在标题中搜索" />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="作者">
                <el-input v-model="advancedForm.author" placeholder="作者姓名" />
              </el-form-item>
            </el-col>
          </el-row>
          
          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="朝代">
                <el-select v-model="advancedForm.dynasties" multiple placeholder="选择朝代">
                  <el-option v-for="dynasty in dynastyOptions" 
                            :key="dynasty" :label="dynasty" :value="dynasty" />
                </el-select>
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="分类">
                <el-select v-model="advancedForm.categories" multiple placeholder="选择分类">
                  <el-option v-for="category in categoryOptions" 
                            :key="category" :label="category" :value="category" />
                </el-select>
              </el-form-item>
            </el-col>
          </el-row>
          
          <el-row :gutter="20">
            <el-col :span="12">
              <el-form-item label="时间范围">
                <el-date-picker
                  v-model="advancedForm.dateRange"
                  type="daterange"
                  range-separator="至"
                  start-placeholder="开始日期"
                  end-placeholder="结束日期"
                  format="YYYY-MM-DD"
                  value-format="YYYY-MM-DD"
                />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="重要性">
                <el-slider
                  v-model="advancedForm.importanceRange"
                  range
                  :min="0"
                  :max="10"
                  :step="0.1"
                  show-input
                />
              </el-form-item>
            </el-col>
          </el-row>
        </el-form>
      </el-collapse-item>
    </el-collapse>

    <!-- 搜索过滤器 -->
    <div v-if="searchResults" class="search-filters">
      <div class="filter-section">
        <span class="filter-label">快速过滤:</span>
        <el-tag
          v-for="filter in quickFilters"
          :key="filter.key"
          :type="filter.active ? 'primary' : 'info'"
          @click="toggleQuickFilter(filter)"
          class="filter-tag"
        >
          {{ filter.label }} ({{ filter.count }})
        </el-tag>
      </div>
      
      <div class="sort-section">
        <span class="sort-label">排序:</span>
        <el-select v-model="sortBy" @change="applySorting">
          <el-option label="相关性" value="relevance" />
          <el-option label="时间" value="date" />
          <el-option label="重要性" value="importance" />
          <el-option label="热度" value="popularity" />
        </el-select>
        <el-select v-model="sortOrder" @change="applySorting">
          <el-option label="降序" value="desc" />
          <el-option label="升序" value="asc" />
        </el-select>
      </div>
    </div>

    <!-- 搜索结果 -->
    <div v-if="searchResults" class="search-results">
      <div class="results-header">
        <span class="results-info">
          找到 {{ searchResults.total }} 个结果 (用时 {{ searchResults.took }}ms)
        </span>
        <div class="view-options">
          <el-radio-group v-model="viewMode" size="small">
            <el-radio-button label="list">列表</el-radio-button>
            <el-radio-button label="grid">网格</el-radio-button>
          </el-radio-group>
        </div>
      </div>
      
      <!-- 列表视图 -->
      <div v-if="viewMode === 'list'" class="results-list">
        <div 
          v-for="doc in searchResults.documents" 
          :key="doc.id"
          class="result-item"
          @click="viewDocument(doc)"
        >
          <div class="result-header">
            <h3 class="result-title" v-html="getHighlightedText(doc, 'title')" />
            <div class="result-meta">
              <el-tag size="small">{{ doc.source.category }}</el-tag>
              <el-tag size="small" type="info">{{ doc.source.dynasty }}</el-tag>
              <span class="result-score">相关度: {{ doc.score.toFixed(2) }}</span>
            </div>
          </div>
          
          <div class="result-content">
            <p v-html="getHighlightedText(doc, 'content')" />
          </div>
          
          <div class="result-footer">
            <span class="result-author">作者: {{ doc.source.author }}</span>
            <span class="result-date">{{ formatDate(doc.source.created_time) }}</span>
            <div class="result-actions">
              <el-button size="small" @click.stop="saveDocument(doc)">收藏</el-button>
              <el-button size="small" @click.stop="shareDocument(doc)">分享</el-button>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 网格视图 -->
      <div v-else class="results-grid">
        <el-row :gutter="20">
          <el-col 
            v-for="doc in searchResults.documents" 
            :key="doc.id"
            :xs="24" :sm="12" :md="8" :lg="6"
          >
            <el-card 
              class="result-card"
              @click="viewDocument(doc)"
              shadow="hover"
            >
              <template #header>
                <div class="card-header">
                  <span class="card-title" v-html="getHighlightedText(doc, 'title')" />
                  <el-tag size="small">{{ doc.source.category }}</el-tag>
                </div>
              </template>
              
              <div class="card-content">
                <p v-html="getHighlightedText(doc, 'content')" />
              </div>
              
              <template #footer>
                <div class="card-footer">
                  <span class="card-author">{{ doc.source.author }}</span>
                  <el-rate 
                    v-model="doc.source.importance_score" 
                    :max="10" 
                    disabled 
                    show-score
                  />
                </div>
              </template>
            </el-card>
          </el-col>
        </el-row>
      </div>
      
      <!-- 分页 -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="searchResults.total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </div>

    <!-- 搜索历史和热门搜索 -->
    <div v-if="!searchResults" class="search-suggestions-panel">
      <el-row :gutter="20">
        <el-col :span="12">
          <el-card title="搜索历史" shadow="hover">
            <div class="history-list">
              <div 
                v-for="history in searchHistory" 
                :key="history.id"
                class="history-item"
                @click="searchQuery = history.query_text; performSearch()"
              >
                <span class="history-query">{{ history.query_text }}</span>
                <span class="history-time">{{ formatTime(history.created_at) }}</span>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :span="12">
          <el-card title="热门搜索" shadow="hover">
            <div class="popular-list">
              <el-tag 
                v-for="popular in popularSearches" 
                :key="popular.query_text"
                class="popular-tag"
                @click="searchQuery = popular.query_text; performSearch()"
              >
                {{ popular.query_text }} ({{ popular.search_count }})
              </el-tag>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import { searchApi } from '@/api/search'
import type { SearchResult, SearchHistory, PopularSearch } from '@/types/search'

// 响应式数据
const searchQuery = ref('')
const searchMode = ref('basic')
const searchLoading = ref(false)
const searchResults = ref<SearchResult | null>(null)
const searchHistory = ref<SearchHistory[]>([])
const popularSearches = ref<PopularSearch[]>([])

// 高级搜索
const advancedPanelOpen = ref(['advanced'])
const advancedForm = reactive({
  title: '',
  author: '',
  dynasties: [],
  categories: [],
  dateRange: [],
  importanceRange: [0, 10]
})

// 过滤和排序
const quickFilters = ref([])
const sortBy = ref('relevance')
const sortOrder = ref('desc')
const viewMode = ref('list')

// 分页
const currentPage = ref(1)
const pageSize = ref(20)

// 选项数据
const dynastyOptions = ref(['唐朝', '宋朝', '明朝', '清朝', '汉朝'])
const categoryOptions = ref(['史书', '传记', '文学', '哲学', '政治'])

/**
 * 组件挂载时加载数据
 */
onMounted(async () => {
  await Promise.all([
    loadSearchHistory(),
    loadPopularSearches()
  ])
})

/**
 * 执行搜索
 */
const performSearch = async () => {
  if (!searchQuery.value.trim()) {
    ElMessage.warning('请输入搜索关键词')
    return
  }
  
  try {
    searchLoading.value = true
    
    let searchParams: any = {
      query: searchQuery.value,
      page: currentPage.value,
      size: pageSize.value
    }
    
    // 根据搜索模式构建参数
    if (searchMode.value === 'advanced') {
      searchParams = {
        ...searchParams,
        ...advancedForm,
        filters: {
          categories: advancedForm.categories,
          dynasties: advancedForm.dynasties,
          date_range: advancedForm.dateRange.length === 2 ? {
            gte: advancedForm.dateRange[0],
            lte: advancedForm.dateRange[1]
          } : undefined,
          importance_range: {
            gte: advancedForm.importanceRange[0],
            lte: advancedForm.importanceRange[1]
          }
        },
        sort: {
          [sortBy.value]: sortOrder.value
        }
      }
    } else {
      searchParams.sort = {
        [sortBy.value]: sortOrder.value
      }
    }
    
    // 调用相应的搜索API
    let response
    if (searchMode.value === 'semantic') {
      response = await searchApi.semanticSearch(searchParams)
    } else if (searchMode.value === 'advanced') {
      response = await searchApi.advancedSearch(searchParams)
    } else {
      response = await searchApi.basicSearch(searchParams)
    }
    
    searchResults.value = response.data
    
    // 更新快速过滤器
    await updateQuickFilters()
    
    ElMessage.success(`找到 ${response.data.total} 个相关结果`)
  } catch (error) {
    ElMessage.error('搜索失败，请重试')
    console.error('Search error:', error)
  } finally {
    searchLoading.value = false
  }
}

/**
 * 获取搜索建议
 */
const getSuggestions = async (queryString: string, callback: Function) => {
  if (!queryString) {
    callback([])
    return
  }
  
  try {
    const response = await searchApi.getSuggestions(queryString)
    const suggestions = response.data.map((text: string) => ({ value: text }))
    callback(suggestions)
  } catch (error) {
    console.error('Get suggestions error:', error)
    callback([])
  }
}

/**
 * 处理建议选择
 */
const handleSuggestionSelect = (item: any) => {
  searchQuery.value = item.value
  performSearch()
}

/**
 * 更新快速过滤器
 */
const updateQuickFilters = async () => {
  try {
    const response = await searchApi.getAggregations(searchQuery.value)
    const aggregations = response.data
    
    quickFilters.value = []
    
    // 处理分类聚合
    if (aggregations.category_stats) {
      aggregations.category_stats.buckets.forEach((bucket: any) => {
        quickFilters.value.push({
          key: `category:${bucket.key}`,
          label: bucket.key,
          count: bucket.doc_count,
          active: false,
          type: 'category'
        })
      })
    }
    
    // 处理朝代聚合
    if (aggregations.dynasty_stats) {
      aggregations.dynasty_stats.buckets.forEach((bucket: any) => {
        quickFilters.value.push({
          key: `dynasty:${bucket.key}`,
          label: bucket.key,
          count: bucket.doc_count,
          active: false,
          type: 'dynasty'
        })
      })
    }
  } catch (error) {
    console.error('Update quick filters error:', error)
  }
}

/**
 * 切换快速过滤器
 */
const toggleQuickFilter = (filter: any) => {
  filter.active = !filter.active
  // 重新执行搜索
  performSearch()
}

/**
 * 应用排序
 */
const applySorting = () => {
  if (searchResults.value) {
    performSearch()
  }
}

/**
 * 处理页面变化
 */
const handlePageChange = (page: number) => {
  currentPage.value = page
  performSearch()
}

/**
 * 处理页面大小变化
 */
const handleSizeChange = (size: number) => {
  pageSize.value = size
  currentPage.value = 1
  performSearch()
}

/**
 * 获取高亮文本
 */
const getHighlightedText = (doc: any, field: string) => {
  if (doc.highlight && doc.highlight[field]) {
    return doc.highlight[field].join('...')
  }
  return doc.source[field] || ''
}

/**
 * 查看文档详情
 */
const viewDocument = (doc: any) => {
  // 跳转到文档详情页
  console.log('View document:', doc)
}

/**
 * 收藏文档
 */
const saveDocument = async (doc: any) => {
  try {
    // 调用收藏API
    ElMessage.success('文档已收藏')
  } catch (error) {
    ElMessage.error('收藏失败')
  }
}

/**
 * 分享文档
 */
const shareDocument = (doc: any) => {
  // 实现分享功能
  console.log('Share document:', doc)
}

/**
 * 加载搜索历史
 */
const loadSearchHistory = async () => {
  try {
    const response = await searchApi.getSearchHistory()
    searchHistory.value = response.data
  } catch (error) {
    console.error('Load search history error:', error)
  }
}

/**
 * 加载热门搜索
 */
const loadPopularSearches = async () => {
  try {
    const response = await searchApi.getPopularSearches()
    popularSearches.value = response.data
  } catch (error) {
    console.error('Load popular searches error:', error)
  }
}

/**
 * 格式化日期
 */
const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString('zh-CN')
}

/**
 * 格式化时间
 */
const formatTime = (dateString: string) => {
  return new Date(dateString).toLocaleString('zh-CN')
}
</script>

<style scoped>
.intelligent-search {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.search-container {
  margin-bottom: 20px;
}

.search-input {
  width: 100%;
  margin-bottom: 10px;
}

.search-modes {
  text-align: center;
}

.search-filters {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.filter-section,
.sort-section {
  display: flex;
  align-items: center;
  gap: 10px;
}

.filter-tag {
  cursor: pointer;
  margin-right: 8px;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.results-info {
  color: #666;
  font-size: 14px;
}

.result-item {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 15px;
  cursor: pointer;
  transition: all 0.3s;
}

.result-item:hover {
  border-color: #409eff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 10px;
}

.result-title {
  margin: 0;
  color: #303133;
  font-size: 18px;
  font-weight: 600;
}

.result-meta {
  display: flex;
  align-items: center;
  gap: 10px;
}

.result-score {
  color: #909399;
  font-size: 12px;
}

.result-content {
  margin: 15px 0;
  color: #606266;
  line-height: 1.6;
}

.result-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #909399;
  font-size: 14px;
}

.result-actions {
  display: flex;
  gap: 10px;
}

.results-grid {
  margin-bottom: 20px;
}

.result-card {
  cursor: pointer;
  transition: all 0.3s;
}

.result-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-weight: 600;
  color: #303133;
}

.card-content {
  color: #606266;
  line-height: 1.6;
  height: 60px;
  overflow: hidden;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pagination-container {
  text-align: center;
  margin-top: 30px;
}

.search-suggestions-panel {
  margin-top: 40px;
}

.history-list,
.popular-list {
  max-height: 300px;
  overflow-y: auto;
}

.history-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
  cursor: pointer;
}

.history-item:hover {
  background-color: #f5f7fa;
}

.history-query {
  color: #303133;
}

.history-time {
  color: #909399;
  font-size: 12px;
}

.popular-tag {
  margin: 5px;
  cursor: pointer;
}

.popular-tag:hover {
  background-color: #409eff;
  color: white;
}
</style>
```

## 验收标准

### 功能性验收标准
1. **基础搜索功能**
   - ✅ 支持中文分词和全文搜索
   - ✅ 多字段搜索（标题、内容、作者等）
   - ✅ 搜索结果高亮显示
   - ✅ 模糊搜索和精确匹配

2. **高级搜索功能**
   - ✅ 布尔搜索（AND、OR、NOT）
   - ✅ 范围搜索（时间、数值）
   - ✅ 字段特定搜索
   - ✅ 通配符和正则表达式搜索

3. **智能搜索功能**
   - ✅ 搜索建议和自动补全
   - ✅ 拼写纠错和同义词扩展
   - ✅ 语义相似度搜索
   - ✅ 个性化搜索排序

4. **搜索结果管理**
   - ✅ 多维度排序（相关性、时间、热度）
   - ✅ 分页和无限滚动
   - ✅ 搜索结果过滤和筛选
   - ✅ 搜索历史和收藏

5. **搜索分析功能**
   - ✅ 搜索统计和热词分析
   - ✅ 用户搜索行为分析
   - ✅ 搜索性能监控

### 性能验收标准
1. **响应时间**
   - ✅ 基础搜索响应时间 < 200ms
   - ✅ 语义搜索响应时间 < 500ms
   - ✅ 高级搜索响应时间 < 1s
   - ✅ 搜索建议响应时间 < 100ms

2. **并发性能**
   - ✅ 支持1000+并发搜索请求
   - ✅ 系统可用性 > 99.9%
   - ✅ 搜索准确率 > 95%

3. **可扩展性**
   - ✅ 支持千万级文档索引
   - ✅ 支持水平扩展
   - ✅ 索引更新延迟 < 5分钟

### 安全验收标准
1. **访问控制**
   - ✅ 用户身份验证和授权
   - ✅ 搜索权限控制
   - ✅ 敏感信息过滤

2. **数据安全**
   - ✅ 搜索日志加密存储
   - ✅ 用户隐私保护
   - ✅ 防止SQL注入和XSS攻击

## 业务价值

### 直接价值
1. **提升用户体验**
   - 快速准确的文档检索
   - 智能搜索建议和纠错
   - 个性化搜索结果

2. **提高研究效率**
   - 语义搜索发现相关内容
   - 高级搜索精确定位
   - 搜索历史和收藏管理

3. **数据洞察**
   - 用户搜索行为分析
   - 热门内容发现
   - 内容质量评估

### 间接价值
1. **平台竞争力**
   - 先进的搜索技术
   - 优秀的用户体验
   - 数据驱动的优化

2. **商业化潜力**
   - 精准内容推荐
   - 用户行为分析
   - 付费搜索服务

## 依赖关系

### 前置依赖
1. **基础设施**
   - ✅ Elasticsearch集群部署
   - ✅ PostgreSQL数据库
   - ✅ Redis缓存服务
   - ✅ RabbitMQ消息队列

2. **数据准备**
   - ✅ 历史文档数据导入
   - ✅ 文档元数据标准化
   - ✅ 索引结构设计

3. **服务依赖**
   - ✅ 用户认证服务
   - ✅ 数据收集服务
   - ✅ NLP处理服务

### 后续依赖
1. **推荐系统**
   - 基于搜索行为的推荐
   - 相关文档推荐
   - 个性化内容推送

2. **分析系统**
   - 搜索效果分析
   - 用户行为分析
   - 内容热度分析

## 风险评估

### 技术风险
1. **性能风险**
   - **风险**: Elasticsearch性能瓶颈
   - **影响**: 搜索响应慢，用户体验差
   - **缓解**: 集群优化，索引分片，缓存策略

2. **数据风险**
   - **风险**: 索引数据不一致
   - **影响**: 搜索结果不准确
   - **缓解**: 数据同步机制，定期校验

3. **扩展风险**
   - **风险**: 数据量增长超预期
   - **影响**: 系统性能下降
   - **缓解**: 弹性扩展，分层存储

### 业务风险
1. **用户体验风险**
   - **风险**: 搜索结果不相关
   - **影响**: 用户满意度下降
   - **缓解**: 算法优化，用户反馈机制

2. **数据质量风险**
   - **风险**: 源数据质量差
   - **影响**: 搜索效果不佳
   - **缓解**: 数据清洗，质量监控

### 安全风险
1. **隐私风险**
   - **风险**: 用户搜索隐私泄露
   - **影响**: 法律合规问题
   - **缓解**: 数据脱敏，访问控制

2. **攻击风险**
   - **风险**: 恶意搜索攻击
   - **影响**: 系统可用性下降
   - **缓解**: 限流防护，异常检测

## 开发任务分解

### 后端开发任务

#### 阶段1: 基础搜索服务 (2天)
1. **Elasticsearch集群配置**
   - 安装和配置Elasticsearch
   - 设计文档索引结构
   - 配置中文分词器
   - 创建索引模板

2. **基础搜索API开发**
   - 实现SearchService核心类
   - 开发基础搜索接口
   - 实现搜索结果格式化
   - 添加搜索日志记录

#### 阶段2: 高级搜索功能 (1.5天)
1. **高级搜索实现**
   - 布尔查询构建
   - 范围搜索实现
   - 字段特定搜索
   - 通配符搜索支持

2. **搜索建议服务**
   - 自动补全功能
   - 拼写纠错实现
   - 搜索建议API
   - 热门搜索统计

#### 阶段3: 智能搜索功能 (1.5天)
1. **语义搜索实现**
   - 集成sentence-transformers
   - 向量相似度搜索
   - 语义搜索API
   - 相似度阈值调优

2. **搜索分析服务**
   - 搜索历史记录
   - 用户行为分析
   - 搜索统计API
   - 性能监控集成

### 前端开发任务

#### 阶段1: 基础搜索界面 (1天)
1. **搜索组件开发**
   - 搜索输入框组件
   - 搜索模式切换
   - 基础搜索功能
   - 搜索结果展示

2. **结果展示优化**
   - 搜索结果高亮
   - 分页组件集成
   - 加载状态处理
   - 错误处理机制

#### 阶段2: 高级搜索界面 (1天)
1. **高级搜索面板**
   - 高级搜索表单
   - 过滤条件组件
   - 排序选项界面
   - 搜索历史展示

2. **搜索体验优化**
   - 搜索建议集成
   - 快速过滤标签
   - 搜索结果统计
   - 响应式设计

#### 阶段3: 智能功能集成 (1天)
1. **智能搜索功能**
   - 语义搜索界面
   - 个性化推荐
   - 搜索分析图表
   - 用户偏好设置

2. **用户体验提升**
   - 搜索性能优化
   - 界面交互优化
   - 移动端适配
   - 无障碍访问支持

### 测试任务

#### 功能测试 (0.5天)
1. **搜索功能测试**
   - 基础搜索测试
   - 高级搜索测试
   - 语义搜索测试
   - 搜索建议测试

2. **界面交互测试**
   - 用户界面测试
   - 响应式测试
   - 兼容性测试
   - 无障碍测试

#### 性能测试 (0.5天)
1. **搜索性能测试**
   - 响应时间测试
   - 并发性能测试
   - 大数据量测试
   - 内存使用测试

2. **系统稳定性测试**
   - 长时间运行测试
   - 异常情况测试
   - 恢复能力测试
   - 监控告警测试

### 部署任务 (0.5天)
1. **环境配置**
   - 生产环境部署
   - 配置文件管理
   - 环境变量设置
   - 服务启动脚本

2. **监控配置**
   - 性能监控配置
   - 日志收集配置
   - 告警规则设置
   - 健康检查配置

## 总结

本用户故事实现了一个功能完整的智能搜索引擎服务，包括基础搜索、高级搜索、语义搜索和智能建议等核心功能。通过Elasticsearch强大的搜索能力和现代化的前端界面，为用户提供了高效、准确、智能的文档检索体验。

该服务不仅满足了基本的搜索需求，还通过语义搜索、个性化推荐和搜索分析等高级功能，为历史文本研究提供了强有力的技术支持，显著提升了研究效率和用户体验。