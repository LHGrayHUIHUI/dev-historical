# Story 3.2: 智能推荐系统

## 基本信息
- **Epic**: Epic 3 - 智能搜索和推荐系统
- **Story ID**: 3.2
- **优先级**: 高
- **预估工作量**: 6天
- **负责团队**: 后端开发团队 + 算法团队 + 前端开发团队

## 用户故事

**作为** 历史文本研究人员  
**我希望** 系统能够基于我的研究兴趣和行为智能推荐相关的历史文档和内容  
**以便于** 发现更多有价值的研究资料，提高研究效率和深度

## 需求描述

### 核心功能需求
1. **基于内容的推荐**
   - 文档相似度推荐
   - 主题相关性推荐
   - 关键词匹配推荐
   - 作者作品推荐

2. **协同过滤推荐**
   - 用户行为相似性推荐
   - 基于评分的推荐
   - 隐式反馈推荐
   - 社交网络推荐

3. **混合推荐算法**
   - 多算法融合
   - 动态权重调整
   - 冷启动处理
   - 实时推荐更新

4. **个性化推荐**
   - 用户画像构建
   - 兴趣标签管理
   - 推荐偏好设置
   - 推荐解释机制

5. **推荐场景支持**
   - 首页推荐
   - 相关文档推荐
   - 搜索结果推荐
   - 阅读推荐

## 技术实现

### 核心技术栈
- **推荐引擎**: Apache Spark MLlib, scikit-learn
- **机器学习**: TensorFlow, PyTorch, LightGBM
- **特征工程**: pandas, numpy, scikit-learn
- **向量计算**: Faiss, Annoy, sentence-transformers
- **实时计算**: Apache Kafka, Redis Streams
- **数据存储**: PostgreSQL, MongoDB, Redis
- **图数据库**: Neo4j (用户关系图)
- **监控**: MLflow, Prometheus, Grafana

### 数据模型设计

#### PostgreSQL数据模型
```sql
-- 用户画像表
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    demographics JSONB, -- 人口统计信息
    interests JSONB, -- 兴趣标签和权重
    behavior_features JSONB, -- 行为特征
    preference_settings JSONB, -- 推荐偏好设置
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户行为表
CREATE TABLE user_behaviors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    document_id UUID,
    behavior_type VARCHAR(50) NOT NULL, -- view, like, share, download, bookmark
    behavior_value FLOAT DEFAULT 1.0, -- 行为权重
    session_id VARCHAR(100),
    duration_seconds INTEGER,
    context_data JSONB, -- 上下文信息
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 文档特征表
CREATE TABLE document_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    content_features JSONB, -- 内容特征向量
    topic_features JSONB, -- 主题特征
    quality_score FLOAT DEFAULT 0.0,
    popularity_score FLOAT DEFAULT 0.0,
    freshness_score FLOAT DEFAULT 0.0,
    embedding_vector FLOAT[], -- 文档向量表示
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 推荐结果表
CREATE TABLE recommendation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    document_id UUID,
    recommendation_type VARCHAR(50), -- content_based, collaborative, hybrid
    score FLOAT NOT NULL,
    rank_position INTEGER,
    algorithm_version VARCHAR(50),
    context_info JSONB,
    is_clicked BOOLEAN DEFAULT false,
    is_liked BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 推荐模型表
CREATE TABLE recommendation_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- content_based, collaborative, hybrid
    model_version VARCHAR(50) NOT NULL,
    model_path TEXT, -- 模型文件路径
    model_config JSONB, -- 模型配置
    performance_metrics JSONB, -- 性能指标
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 推荐实验表
CREATE TABLE recommendation_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name VARCHAR(100) NOT NULL,
    experiment_type VARCHAR(50), -- a_b_test, multi_armed_bandit
    model_configs JSONB, -- 实验模型配置
    traffic_allocation JSONB, -- 流量分配
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'draft', -- draft, running, completed, stopped
    results JSONB, -- 实验结果
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户反馈表
CREATE TABLE recommendation_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    recommendation_id UUID REFERENCES recommendation_results(id),
    feedback_type VARCHAR(50), -- like, dislike, not_interested, report
    feedback_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_user_behaviors_user_id ON user_behaviors(user_id);
CREATE INDEX idx_user_behaviors_document_id ON user_behaviors(document_id);
CREATE INDEX idx_user_behaviors_type ON user_behaviors(behavior_type);
CREATE INDEX idx_user_behaviors_created_at ON user_behaviors(created_at);
CREATE INDEX idx_document_features_document_id ON document_features(document_id);
CREATE INDEX idx_recommendation_results_user_id ON recommendation_results(user_id);
CREATE INDEX idx_recommendation_results_created_at ON recommendation_results(created_at);
CREATE INDEX idx_recommendation_feedback_user_id ON recommendation_feedback(user_id);
```

### 推荐算法架构

#### 基于内容的推荐算法
```python
# content_based_recommender.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import jieba
from typing import List, Dict, Tuple

class ContentBasedRecommender:
    """
    基于内容的推荐算法
    通过分析文档内容特征进行推荐
    """
    
    def __init__(self, embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=self._get_chinese_stopwords(),
            tokenizer=self._chinese_tokenizer
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.document_features = None
        self.document_embeddings = None
        self.document_ids = None
    
    def _get_chinese_stopwords(self) -> List[str]:
        """
        获取中文停用词列表
        
        Returns:
            中文停用词列表
        """
        # 这里应该加载实际的中文停用词文件
        return ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这']
    
    def _chinese_tokenizer(self, text: str) -> List[str]:
        """
        中文分词器
        
        Args:
            text: 输入文本
        
        Returns:
            分词结果列表
        """
        return list(jieba.cut(text))
    
    def fit(self, documents: pd.DataFrame) -> None:
        """
        训练推荐模型
        
        Args:
            documents: 文档数据框，包含id, title, content, category等字段
        """
        self.document_ids = documents['id'].tolist()
        
        # 构建文档文本特征
        document_texts = documents.apply(
            lambda row: f"{row['title']} {row['content']} {row.get('category', '')} {row.get('tags', '')}",
            axis=1
        ).tolist()
        
        # TF-IDF特征
        tfidf_features = self.tfidf_vectorizer.fit_transform(document_texts)
        
        # 语义嵌入特征
        embeddings = self.embedding_model.encode(document_texts)
        
        # 合并特征
        self.document_features = np.hstack([
            tfidf_features.toarray(),
            embeddings
        ])
        
        self.document_embeddings = embeddings
        
        print(f"训练完成，处理了 {len(documents)} 个文档")
    
    def recommend_by_document(self, 
                             document_id: str, 
                             top_k: int = 10,
                             exclude_same_author: bool = False) -> List[Tuple[str, float]]:
        """
        基于文档相似度推荐
        
        Args:
            document_id: 目标文档ID
            top_k: 推荐数量
            exclude_same_author: 是否排除同作者文档
        
        Returns:
            推荐结果列表 [(document_id, similarity_score)]
        """
        if document_id not in self.document_ids:
            return []
        
        doc_index = self.document_ids.index(document_id)
        doc_features = self.document_features[doc_index].reshape(1, -1)
        
        # 计算相似度
        similarities = cosine_similarity(doc_features, self.document_features)[0]
        
        # 排序并获取top_k
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if len(recommendations) >= top_k:
                break
            
            if idx == doc_index:  # 排除自身
                continue
            
            sim_doc_id = self.document_ids[idx]
            similarity_score = similarities[idx]
            
            recommendations.append((sim_doc_id, similarity_score))
        
        return recommendations
    
    def recommend_by_user_profile(self, 
                                 user_interests: Dict[str, float],
                                 user_history: List[str],
                                 top_k: int = 20) -> List[Tuple[str, float]]:
        """
        基于用户画像推荐
        
        Args:
            user_interests: 用户兴趣标签和权重
            user_history: 用户历史浏览文档ID列表
            top_k: 推荐数量
        
        Returns:
            推荐结果列表
        """
        if not user_history:
            return self._recommend_popular_documents(top_k)
        
        # 基于用户历史构建用户画像向量
        user_vector = self._build_user_vector(user_history, user_interests)
        
        # 计算与所有文档的相似度
        similarities = cosine_similarity(user_vector.reshape(1, -1), self.document_features)[0]
        
        # 排序并过滤已浏览文档
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if len(recommendations) >= top_k:
                break
            
            doc_id = self.document_ids[idx]
            if doc_id in user_history:  # 排除已浏览文档
                continue
            
            similarity_score = similarities[idx]
            recommendations.append((doc_id, similarity_score))
        
        return recommendations
    
    def _build_user_vector(self, 
                          user_history: List[str], 
                          user_interests: Dict[str, float]) -> np.ndarray:
        """
        构建用户特征向量
        
        Args:
            user_history: 用户历史文档列表
            user_interests: 用户兴趣权重
        
        Returns:
            用户特征向量
        """
        # 获取用户历史文档的特征向量
        history_vectors = []
        for doc_id in user_history:
            if doc_id in self.document_ids:
                doc_index = self.document_ids.index(doc_id)
                history_vectors.append(self.document_features[doc_index])
        
        if not history_vectors:
            return np.zeros(self.document_features.shape[1])
        
        # 计算用户向量（历史文档特征的加权平均）
        user_vector = np.mean(history_vectors, axis=0)
        
        # 可以根据用户兴趣进一步调整向量
        # 这里简化处理，实际可以更复杂
        
        return user_vector
    
    def _recommend_popular_documents(self, top_k: int) -> List[Tuple[str, float]]:
        """
        推荐热门文档（冷启动处理）
        
        Args:
            top_k: 推荐数量
        
        Returns:
            推荐结果列表
        """
        # 这里应该基于文档的热度、质量等指标排序
        # 简化处理，返回前top_k个文档
        recommendations = []
        for i in range(min(top_k, len(self.document_ids))):
            recommendations.append((self.document_ids[i], 1.0))
        
        return recommendations

class CollaborativeFilteringRecommender:
    """
    协同过滤推荐算法
    基于用户行为相似性进行推荐
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, user_behaviors: pd.DataFrame) -> None:
        """
        训练协同过滤模型
        
        Args:
            user_behaviors: 用户行为数据，包含user_id, document_id, rating等字段
        """
        # 构建用户-物品评分矩阵
        self.user_item_matrix = user_behaviors.pivot_table(
            index='user_id',
            columns='document_id',
            values='behavior_value',
            fill_value=0
        )
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # 计算用户相似度矩阵
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix.values)
        
        # 计算物品相似度矩阵
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T.values)
        
        print(f"协同过滤模型训练完成，用户数: {len(self.user_ids)}, 物品数: {len(self.item_ids)}")
    
    def recommend_user_based(self, 
                           user_id: str, 
                           top_k: int = 20,
                           neighbor_size: int = 50) -> List[Tuple[str, float]]:
        """
        基于用户的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            top_k: 推荐数量
            neighbor_size: 邻居用户数量
        
        Returns:
            推荐结果列表
        """
        if user_id not in self.user_ids:
            return []
        
        user_index = self.user_ids.index(user_id)
        user_similarities = self.user_similarity_matrix[user_index]
        
        # 找到最相似的邻居用户
        similar_users = np.argsort(user_similarities)[::-1][1:neighbor_size+1]
        
        # 计算推荐分数
        recommendations = {}
        user_ratings = self.user_item_matrix.iloc[user_index]
        
        for similar_user_idx in similar_users:
            similarity = user_similarities[similar_user_idx]
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]
            
            for item_idx, rating in enumerate(similar_user_ratings):
                if rating > 0 and user_ratings.iloc[item_idx] == 0:  # 用户未评分的物品
                    item_id = self.item_ids[item_idx]
                    if item_id not in recommendations:
                        recommendations[item_id] = 0
                    recommendations[item_id] += similarity * rating
        
        # 排序并返回top_k
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_k]
    
    def recommend_item_based(self, 
                           user_id: str, 
                           top_k: int = 20) -> List[Tuple[str, float]]:
        """
        基于物品的协同过滤推荐
        
        Args:
            user_id: 目标用户ID
            top_k: 推荐数量
        
        Returns:
            推荐结果列表
        """
        if user_id not in self.user_ids:
            return []
        
        user_index = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_index]
        
        recommendations = {}
        
        # 对用户评分过的每个物品
        for item_idx, rating in enumerate(user_ratings):
            if rating > 0:
                item_similarities = self.item_similarity_matrix[item_idx]
                
                # 找到相似物品
                for similar_item_idx, similarity in enumerate(item_similarities):
                    if similar_item_idx != item_idx and user_ratings.iloc[similar_item_idx] == 0:
                        item_id = self.item_ids[similar_item_idx]
                        if item_id not in recommendations:
                            recommendations[item_id] = 0
                        recommendations[item_id] += similarity * rating
        
        # 排序并返回top_k
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:top_k]

class HybridRecommender:
    """
    混合推荐算法
    融合多种推荐算法的结果
    """
    
    def __init__(self, 
                 content_recommender: ContentBasedRecommender,
                 collaborative_recommender: CollaborativeFilteringRecommender):
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.weights = {
            'content_based': 0.4,
            'collaborative_user': 0.3,
            'collaborative_item': 0.3
        }
    
    def recommend(self, 
                 user_id: str,
                 user_profile: Dict,
                 top_k: int = 20) -> List[Tuple[str, float, str]]:
        """
        混合推荐
        
        Args:
            user_id: 用户ID
            user_profile: 用户画像信息
            top_k: 推荐数量
        
        Returns:
            推荐结果列表 [(document_id, score, algorithm)]
        """
        all_recommendations = {}
        
        # 基于内容的推荐
        try:
            content_recs = self.content_recommender.recommend_by_user_profile(
                user_profile.get('interests', {}),
                user_profile.get('history', []),
                top_k * 2
            )
            
            for doc_id, score in content_recs:
                if doc_id not in all_recommendations:
                    all_recommendations[doc_id] = {'scores': {}, 'total': 0}
                all_recommendations[doc_id]['scores']['content_based'] = score
                all_recommendations[doc_id]['total'] += score * self.weights['content_based']
        except Exception as e:
            print(f"Content-based recommendation error: {e}")
        
        # 基于用户的协同过滤推荐
        try:
            user_cf_recs = self.collaborative_recommender.recommend_user_based(user_id, top_k * 2)
            
            for doc_id, score in user_cf_recs:
                if doc_id not in all_recommendations:
                    all_recommendations[doc_id] = {'scores': {}, 'total': 0}
                all_recommendations[doc_id]['scores']['collaborative_user'] = score
                all_recommendations[doc_id]['total'] += score * self.weights['collaborative_user']
        except Exception as e:
            print(f"User-based collaborative filtering error: {e}")
        
        # 基于物品的协同过滤推荐
        try:
            item_cf_recs = self.collaborative_recommender.recommend_item_based(user_id, top_k * 2)
            
            for doc_id, score in item_cf_recs:
                if doc_id not in all_recommendations:
                    all_recommendations[doc_id] = {'scores': {}, 'total': 0}
                all_recommendations[doc_id]['scores']['collaborative_item'] = score
                all_recommendations[doc_id]['total'] += score * self.weights['collaborative_item']
        except Exception as e:
            print(f"Item-based collaborative filtering error: {e}")
        
        # 排序并返回结果
        sorted_recommendations = sorted(
            all_recommendations.items(), 
            key=lambda x: x[1]['total'], 
            reverse=True
        )
        
        results = []
        for doc_id, rec_info in sorted_recommendations[:top_k]:
            # 确定主要推荐算法
            main_algorithm = max(rec_info['scores'].items(), key=lambda x: x[1])[0]
            results.append((doc_id, rec_info['total'], main_algorithm))
        
        return results
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        更新算法权重
        
        Args:
            new_weights: 新的权重配置
        """
        self.weights.update(new_weights)
        
        # 确保权重和为1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight
```

### 推荐服务架构

#### 推荐服务核心类
```python
# recommendation_service.py
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from redis import Redis

class RecommendationService:
    """
    推荐服务核心类
    提供统一的推荐接口和服务管理
    """
    
    def __init__(self, 
                 db_session: AsyncSession,
                 redis_client: Redis,
                 content_recommender: ContentBasedRecommender,
                 collaborative_recommender: CollaborativeFilteringRecommender,
                 hybrid_recommender: HybridRecommender):
        self.db = db_session
        self.redis = redis_client
        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.hybrid_recommender = hybrid_recommender
        
        # 缓存配置
        self.cache_ttl = 3600  # 1小时
        self.cache_prefix = "rec:"
    
    async def get_recommendations(self, 
                                user_id: str,
                                recommendation_type: str = 'hybrid',
                                context: Optional[Dict] = None,
                                top_k: int = 20) -> Dict[str, Any]:
        """
        获取用户推荐
        
        Args:
            user_id: 用户ID
            recommendation_type: 推荐类型 (content, collaborative, hybrid)
            context: 推荐上下文信息
            top_k: 推荐数量
        
        Returns:
            推荐结果字典
        """
        # 检查缓存
        cache_key = f"{self.cache_prefix}{user_id}:{recommendation_type}:{top_k}"
        cached_result = self.redis.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # 获取用户画像
        user_profile = await self._get_user_profile(user_id)
        
        # 根据推荐类型生成推荐
        if recommendation_type == 'content':
            recommendations = await self._get_content_based_recommendations(
                user_profile, top_k
            )
        elif recommendation_type == 'collaborative':
            recommendations = await self._get_collaborative_recommendations(
                user_id, top_k
            )
        else:  # hybrid
            recommendations = await self._get_hybrid_recommendations(
                user_id, user_profile, top_k
            )
        
        # 增强推荐结果
        enhanced_recommendations = await self._enhance_recommendations(
            recommendations, user_id, context
        )
        
        # 记录推荐结果
        await self._record_recommendations(user_id, enhanced_recommendations, recommendation_type)
        
        # 缓存结果
        result = {
            'user_id': user_id,
            'recommendation_type': recommendation_type,
            'recommendations': enhanced_recommendations,
            'generated_at': datetime.now().isoformat(),
            'total_count': len(enhanced_recommendations)
        }
        
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(result, default=str))
        
        return result
    
    async def get_similar_documents(self, 
                                  document_id: str,
                                  top_k: int = 10) -> List[Dict[str, Any]]:
        """
        获取相似文档推荐
        
        Args:
            document_id: 文档ID
            top_k: 推荐数量
        
        Returns:
            相似文档列表
        """
        # 检查缓存
        cache_key = f"{self.cache_prefix}similar:{document_id}:{top_k}"
        cached_result = self.redis.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # 获取相似文档
        similar_docs = self.content_recommender.recommend_by_document(
            document_id, top_k
        )
        
        # 增强结果信息
        enhanced_docs = await self._enhance_document_recommendations(similar_docs)
        
        # 缓存结果
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(enhanced_docs, default=str))
        
        return enhanced_docs
    
    async def record_user_feedback(self, 
                                 user_id: str,
                                 recommendation_id: str,
                                 feedback_type: str,
                                 feedback_reason: Optional[str] = None) -> None:
        """
        记录用户反馈
        
        Args:
            user_id: 用户ID
            recommendation_id: 推荐ID
            feedback_type: 反馈类型
            feedback_reason: 反馈原因
        """
        feedback_data = {
            'user_id': user_id,
            'recommendation_id': recommendation_id,
            'feedback_type': feedback_type,
            'feedback_reason': feedback_reason
        }
        
        await self.db.execute(
            "INSERT INTO recommendation_feedback (user_id, recommendation_id, feedback_type, feedback_reason) VALUES (:user_id, :recommendation_id, :feedback_type, :feedback_reason)",
            feedback_data
        )
        
        # 更新用户画像
        await self._update_user_profile_from_feedback(user_id, feedback_type, feedback_reason)
        
        # 清除相关缓存
        await self._clear_user_cache(user_id)
    
    async def update_user_behavior(self, 
                                 user_id: str,
                                 document_id: str,
                                 behavior_type: str,
                                 behavior_value: float = 1.0,
                                 context_data: Optional[Dict] = None) -> None:
        """
        更新用户行为数据
        
        Args:
            user_id: 用户ID
            document_id: 文档ID
            behavior_type: 行为类型
            behavior_value: 行为权重
            context_data: 上下文数据
        """
        behavior_data = {
            'user_id': user_id,
            'document_id': document_id,
            'behavior_type': behavior_type,
            'behavior_value': behavior_value,
            'context_data': json.dumps(context_data) if context_data else None
        }
        
        await self.db.execute(
            "INSERT INTO user_behaviors (user_id, document_id, behavior_type, behavior_value, context_data) VALUES (:user_id, :document_id, :behavior_type, :behavior_value, :context_data)",
            behavior_data
        )
        
        # 异步更新用户画像
        asyncio.create_task(self._update_user_profile(user_id))
        
        # 清除相关缓存
        await self._clear_user_cache(user_id)
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户画像
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户画像字典
        """
        # 从数据库获取用户画像
        profile_result = await self.db.fetch_one(
            "SELECT * FROM user_profiles WHERE user_id = :user_id",
            {'user_id': user_id}
        )
        
        if not profile_result:
            # 创建默认用户画像
            return await self._create_default_user_profile(user_id)
        
        # 获取用户历史行为
        history_result = await self.db.fetch_all(
            "SELECT document_id, behavior_type, behavior_value FROM user_behaviors WHERE user_id = :user_id ORDER BY created_at DESC LIMIT 100",
            {'user_id': user_id}
        )
        
        profile = {
            'user_id': user_id,
            'interests': profile_result['interests'] or {},
            'demographics': profile_result['demographics'] or {},
            'behavior_features': profile_result['behavior_features'] or {},
            'preference_settings': profile_result['preference_settings'] or {},
            'history': [row['document_id'] for row in history_result]
        }
        
        return profile
    
    async def _get_content_based_recommendations(self, 
                                               user_profile: Dict,
                                               top_k: int) -> List[Tuple[str, float, str]]:
        """
        获取基于内容的推荐
        
        Args:
            user_profile: 用户画像
            top_k: 推荐数量
        
        Returns:
            推荐结果列表
        """
        recommendations = self.content_recommender.recommend_by_user_profile(
            user_profile.get('interests', {}),
            user_profile.get('history', []),
            top_k
        )
        
        return [(doc_id, score, 'content_based') for doc_id, score in recommendations]
    
    async def _get_collaborative_recommendations(self, 
                                               user_id: str,
                                               top_k: int) -> List[Tuple[str, float, str]]:
        """
        获取协同过滤推荐
        
        Args:
            user_id: 用户ID
            top_k: 推荐数量
        
        Returns:
            推荐结果列表
        """
        # 尝试基于用户的协同过滤
        user_cf_recs = self.collaborative_recommender.recommend_user_based(user_id, top_k // 2)
        
        # 尝试基于物品的协同过滤
        item_cf_recs = self.collaborative_recommender.recommend_item_based(user_id, top_k // 2)
        
        # 合并结果
        all_recs = []
        all_recs.extend([(doc_id, score, 'collaborative_user') for doc_id, score in user_cf_recs])
        all_recs.extend([(doc_id, score, 'collaborative_item') for doc_id, score in item_cf_recs])
        
        # 去重并排序
        unique_recs = {}
        for doc_id, score, algorithm in all_recs:
            if doc_id not in unique_recs or score > unique_recs[doc_id][1]:
                unique_recs[doc_id] = (doc_id, score, algorithm)
        
        sorted_recs = sorted(unique_recs.values(), key=lambda x: x[1], reverse=True)
        
        return sorted_recs[:top_k]
    
    async def _get_hybrid_recommendations(self, 
                                        user_id: str,
                                        user_profile: Dict,
                                        top_k: int) -> List[Tuple[str, float, str]]:
        """
        获取混合推荐
        
        Args:
            user_id: 用户ID
            user_profile: 用户画像
            top_k: 推荐数量
        
        Returns:
            推荐结果列表
        """
        return self.hybrid_recommender.recommend(user_id, user_profile, top_k)
    
    async def _enhance_recommendations(self, 
                                     recommendations: List[Tuple[str, float, str]],
                                     user_id: str,
                                     context: Optional[Dict]) -> List[Dict[str, Any]]:
        """
        增强推荐结果信息
        
        Args:
            recommendations: 原始推荐结果
            user_id: 用户ID
            context: 上下文信息
        
        Returns:
            增强后的推荐结果
        """
        enhanced_recs = []
        
        for i, (doc_id, score, algorithm) in enumerate(recommendations):
            # 获取文档详细信息
            doc_info = await self._get_document_info(doc_id)
            
            if doc_info:
                enhanced_rec = {
                    'document_id': doc_id,
                    'score': score,
                    'algorithm': algorithm,
                    'rank': i + 1,
                    'document_info': doc_info,
                    'explanation': self._generate_explanation(algorithm, doc_info, context)
                }
                enhanced_recs.append(enhanced_rec)
        
        return enhanced_recs
    
    async def _enhance_document_recommendations(self, 
                                              similar_docs: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """
        增强文档推荐结果
        
        Args:
            similar_docs: 相似文档列表
        
        Returns:
            增强后的推荐结果
        """
        enhanced_docs = []
        
        for i, (doc_id, similarity) in enumerate(similar_docs):
            doc_info = await self._get_document_info(doc_id)
            
            if doc_info:
                enhanced_doc = {
                    'document_id': doc_id,
                    'similarity_score': similarity,
                    'rank': i + 1,
                    'document_info': doc_info
                }
                enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs
    
    async def _get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档详细信息
        
        Args:
            document_id: 文档ID
        
        Returns:
            文档信息字典
        """
        # 这里应该调用文档服务获取详细信息
        # 简化处理，返回基本信息
        doc_result = await self.db.fetch_one(
            "SELECT id, title, author, category, created_time FROM documents WHERE id = :doc_id",
            {'doc_id': document_id}
        )
        
        if doc_result:
            return dict(doc_result)
        
        return None
    
    def _generate_explanation(self, 
                            algorithm: str, 
                            doc_info: Dict, 
                            context: Optional[Dict]) -> str:
        """
        生成推荐解释
        
        Args:
            algorithm: 推荐算法
            doc_info: 文档信息
            context: 上下文信息
        
        Returns:
            推荐解释文本
        """
        explanations = {
            'content_based': f"因为您对{doc_info.get('category', '此类')}内容感兴趣",
            'collaborative_user': "因为与您兴趣相似的用户也喜欢这个内容",
            'collaborative_item': "因为这与您之前浏览的内容相似",
            'hybrid': "基于您的兴趣和行为综合推荐"
        }
        
        return explanations.get(algorithm, "系统智能推荐")
    
    async def _record_recommendations(self, 
                                    user_id: str,
                                    recommendations: List[Dict],
                                    recommendation_type: str) -> None:
        """
        记录推荐结果
        
        Args:
            user_id: 用户ID
            recommendations: 推荐结果
            recommendation_type: 推荐类型
        """
        for rec in recommendations:
            rec_data = {
                'user_id': user_id,
                'document_id': rec['document_id'],
                'recommendation_type': recommendation_type,
                'score': rec['score'],
                'rank_position': rec['rank'],
                'algorithm_version': rec['algorithm']
            }
            
            await self.db.execute(
                "INSERT INTO recommendation_results (user_id, document_id, recommendation_type, score, rank_position, algorithm_version) VALUES (:user_id, :document_id, :recommendation_type, :score, :rank_position, :algorithm_version)",
                rec_data
            )
    
    async def _update_user_profile(self, user_id: str) -> None:
        """
        更新用户画像
        
        Args:
            user_id: 用户ID
        """
        # 获取用户最近行为
        recent_behaviors = await self.db.fetch_all(
            "SELECT * FROM user_behaviors WHERE user_id = :user_id AND created_at >= :since ORDER BY created_at DESC",
            {
                'user_id': user_id,
                'since': datetime.now() - timedelta(days=30)
            }
        )
        
        # 分析用户兴趣
        interests = self._analyze_user_interests(recent_behaviors)
        
        # 计算行为特征
        behavior_features = self._calculate_behavior_features(recent_behaviors)
        
        # 更新用户画像
        await self.db.execute(
            "UPDATE user_profiles SET interests = :interests, behavior_features = :behavior_features, updated_at = CURRENT_TIMESTAMP WHERE user_id = :user_id",
            {
                'user_id': user_id,
                'interests': json.dumps(interests),
                'behavior_features': json.dumps(behavior_features)
            }
        )
    
    def _analyze_user_interests(self, behaviors: List[Dict]) -> Dict[str, float]:
        """
        分析用户兴趣
        
        Args:
            behaviors: 用户行为列表
        
        Returns:
            兴趣标签和权重字典
        """
        interests = {}
        
        # 基于用户行为分析兴趣
        for behavior in behaviors:
            # 这里应该根据文档的分类、标签等信息分析兴趣
            # 简化处理
            behavior_weight = {
                'view': 1.0,
                'like': 2.0,
                'share': 3.0,
                'download': 2.5,
                'bookmark': 3.0
            }.get(behavior['behavior_type'], 1.0)
            
            # 这里需要获取文档的分类信息来更新兴趣
            # 简化处理，直接使用行为类型
            category = 'general'  # 实际应该从文档信息中获取
            
            if category not in interests:
                interests[category] = 0
            interests[category] += behavior_weight * behavior['behavior_value']
        
        # 归一化兴趣权重
        total_weight = sum(interests.values())
        if total_weight > 0:
            for category in interests:
                interests[category] /= total_weight
        
        return interests
    
    def _calculate_behavior_features(self, behaviors: List[Dict]) -> Dict[str, Any]:
        """
        计算行为特征
        
        Args:
            behaviors: 用户行为列表
        
        Returns:
            行为特征字典
        """
        if not behaviors:
            return {}
        
        behavior_counts = {}
        for behavior in behaviors:
            behavior_type = behavior['behavior_type']
            if behavior_type not in behavior_counts:
                behavior_counts[behavior_type] = 0
            behavior_counts[behavior_type] += 1
        
        total_behaviors = len(behaviors)
        
        features = {
            'total_behaviors': total_behaviors,
            'behavior_distribution': behavior_counts,
            'activity_level': 'high' if total_behaviors > 50 else 'medium' if total_behaviors > 10 else 'low',
            'last_activity': behaviors[0]['created_at'].isoformat() if behaviors else None
        }
        
        return features
    
    async def _create_default_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        创建默认用户画像
        
        Args:
            user_id: 用户ID
        
        Returns:
            默认用户画像
        """
        default_profile = {
            'user_id': user_id,
            'interests': {},
            'demographics': {},
            'behavior_features': {},
            'preference_settings': {
                'recommendation_frequency': 'daily',
                'content_types': ['all'],
                'explanation_level': 'basic'
            },
            'history': []
        }
        
        # 插入数据库
        await self.db.execute(
            "INSERT INTO user_profiles (user_id, interests, demographics, behavior_features, preference_settings) VALUES (:user_id, :interests, :demographics, :behavior_features, :preference_settings)",
            {
                'user_id': user_id,
                'interests': json.dumps(default_profile['interests']),
                'demographics': json.dumps(default_profile['demographics']),
                'behavior_features': json.dumps(default_profile['behavior_features']),
                'preference_settings': json.dumps(default_profile['preference_settings'])
            }
        )
        
        return default_profile
    
    async def _update_user_profile_from_feedback(self, 
                                               user_id: str,
                                               feedback_type: str,
                                               feedback_reason: Optional[str]) -> None:
        """
        根据用户反馈更新用户画像
        
        Args:
            user_id: 用户ID
            feedback_type: 反馈类型
            feedback_reason: 反馈原因
        """
        # 根据反馈调整用户兴趣权重
        # 这里可以实现更复杂的反馈学习机制
        pass
    
    async def _clear_user_cache(self, user_id: str) -> None:
        """
        清除用户相关缓存
        
        Args:
            user_id: 用户ID
        """
        # 清除用户推荐缓存
        cache_pattern = f"{self.cache_prefix}{user_id}:*"
        keys = self.redis.keys(cache_pattern)
        if keys:
            self.redis.delete(*keys)
```

## API设计

### 推荐API接口

#### 1. 获取用户推荐
```http
GET /api/v1/recommendations/users/{user_id}
```

**请求参数:**
```json
{
  "type": "hybrid",  // content, collaborative, hybrid
  "limit": 20,
  "context": {
    "page": "homepage",
    "current_document_id": "doc_123"
  }
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "user_id": "user_123",
    "recommendation_type": "hybrid",
    "recommendations": [
      {
        "document_id": "doc_456",
        "score": 0.95,
        "algorithm": "content_based",
        "rank": 1,
        "document_info": {
          "id": "doc_456",
          "title": "明代历史文献研究",
          "author": "张三",
          "category": "历史研究",
          "created_time": "2024-01-15T10:30:00Z"
        },
        "explanation": "因为您对历史研究内容感兴趣"
      }
    ],
    "generated_at": "2024-01-20T15:30:00Z",
    "total_count": 20
  }
}
```

#### 2. 获取相似文档推荐
```http
GET /api/v1/recommendations/documents/{document_id}/similar
```

**请求参数:**
```json
{
  "limit": 10,
  "exclude_same_author": false
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "document_id": "doc_123",
    "similar_documents": [
      {
        "document_id": "doc_789",
        "similarity_score": 0.88,
        "rank": 1,
        "document_info": {
          "id": "doc_789",
          "title": "清代文献整理方法",
          "author": "李四",
          "category": "文献学"
        }
      }
    ],
    "total_count": 10
  }
}
```

#### 3. 记录用户行为
```http
POST /api/v1/recommendations/behaviors
```

**请求体:**
```json
{
  "user_id": "user_123",
  "document_id": "doc_456",
  "behavior_type": "view",  // view, like, share, download, bookmark
  "behavior_value": 1.0,
  "context_data": {
    "source": "search_result",
    "position": 3,
    "session_id": "session_789"
  }
}
```

#### 4. 用户反馈
```http
POST /api/v1/recommendations/feedback
```

**请求体:**
```json
{
  "user_id": "user_123",
  "recommendation_id": "rec_456",
  "feedback_type": "like",  // like, dislike, not_interested, report
  "feedback_reason": "内容很有价值"
}
```

#### 5. 获取用户画像
```http
GET /api/v1/recommendations/users/{user_id}/profile
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "user_id": "user_123",
    "interests": {
      "历史研究": 0.4,
      "文献学": 0.3,
      "考古学": 0.2,
      "古代文学": 0.1
    },
    "behavior_features": {
      "total_behaviors": 156,
      "activity_level": "high",
      "behavior_distribution": {
        "view": 120,
        "like": 25,
        "share": 8,
        "download": 3
      }
    },
    "preference_settings": {
      "recommendation_frequency": "daily",
      "content_types": ["research_paper", "historical_document"],
      "explanation_level": "detailed"
    }
  }
}
```

#### 6. 更新推荐偏好
```http
PUT /api/v1/recommendations/users/{user_id}/preferences
```

**请求体:**
```json
{
  "recommendation_frequency": "weekly",
  "content_types": ["all"],
  "explanation_level": "basic",
  "algorithm_weights": {
    "content_based": 0.5,
    "collaborative_user": 0.3,
    "collaborative_item": 0.2
  }
}
```

## 前端集成

### Vue3 推荐组件

#### 推荐列表组件
```vue
<template>
  <div class="recommendation-container">
    <!-- 推荐设置 -->
    <div class="recommendation-settings" v-if="showSettings">
      <el-card class="settings-card">
        <template #header>
          <div class="card-header">
            <span>推荐设置</span>
            <el-button type="text" @click="showSettings = false">收起</el-button>
          </div>
        </template>
        
        <el-form :model="preferences" label-width="120px">
          <el-form-item label="推荐类型">
            <el-select v-model="preferences.type" @change="onPreferenceChange">
              <el-option label="智能推荐" value="hybrid"></el-option>
              <el-option label="内容相似" value="content"></el-option>
              <el-option label="协同过滤" value="collaborative"></el-option>
            </el-select>
          </el-form-item>
          
          <el-form-item label="推荐数量">
            <el-slider 
              v-model="preferences.limit" 
              :min="5" 
              :max="50" 
              :step="5"
              @change="onPreferenceChange"
            ></el-slider>
          </el-form-item>
          
          <el-form-item label="内容类型">
            <el-checkbox-group v-model="preferences.contentTypes" @change="onPreferenceChange">
              <el-checkbox label="research_paper">研究论文</el-checkbox>
              <el-checkbox label="historical_document">历史文献</el-checkbox>
              <el-checkbox label="ancient_book">古籍</el-checkbox>
              <el-checkbox label="archaeological_report">考古报告</el-checkbox>
            </el-checkbox-group>
          </el-form-item>
          
          <el-form-item label="解释详细度">
            <el-radio-group v-model="preferences.explanationLevel" @change="onPreferenceChange">
              <el-radio label="basic">简单</el-radio>
              <el-radio label="detailed">详细</el-radio>
            </el-radio-group>
          </el-form-item>
        </el-form>
      </el-card>
    </div>
    
    <!-- 推荐列表 -->
    <div class="recommendation-list">
      <div class="list-header">
        <h3>为您推荐</h3>
        <div class="header-actions">
          <el-button 
            type="text" 
            icon="el-icon-setting" 
            @click="showSettings = !showSettings"
          >
            设置
          </el-button>
          <el-button 
            type="text" 
            icon="el-icon-refresh" 
            @click="refreshRecommendations"
            :loading="loading"
          >
            刷新
          </el-button>
        </div>
      </div>
      
      <el-skeleton :loading="loading" animated>
        <template #template>
          <div v-for="i in 5" :key="i" class="recommendation-skeleton">
            <el-skeleton-item variant="image" style="width: 100px; height: 80px;"></el-skeleton-item>
            <div class="skeleton-content">
              <el-skeleton-item variant="h3" style="width: 60%;"></el-skeleton-item>
              <el-skeleton-item variant="text" style="width: 80%;"></el-skeleton-item>
              <el-skeleton-item variant="text" style="width: 40%;"></el-skeleton-item>
            </div>
          </div>
        </template>
        
        <template #default>
          <div class="recommendations-grid">
            <div 
              v-for="(item, index) in recommendations" 
              :key="item.document_id"
              class="recommendation-item"
              @click="onItemClick(item)"
            >
              <div class="item-rank">{{ item.rank }}</div>
              
              <div class="item-thumbnail">
                <img 
                  :src="getDocumentThumbnail(item.document_info)" 
                  :alt="item.document_info.title"
                  @error="onImageError"
                >
              </div>
              
              <div class="item-content">
                <h4 class="item-title">{{ item.document_info.title }}</h4>
                <p class="item-author">{{ item.document_info.author }}</p>
                <p class="item-category">{{ item.document_info.category }}</p>
                
                <div class="item-meta">
                  <span class="algorithm-tag" :class="getAlgorithmClass(item.algorithm)">
                    {{ getAlgorithmLabel(item.algorithm) }}
                  </span>
                  <span class="score">{{ (item.score * 100).toFixed(0) }}%</span>
                </div>
                
                <div class="item-explanation" v-if="preferences.explanationLevel === 'detailed'">
                  <el-tooltip :content="item.explanation" placement="top">
                    <i class="el-icon-info"></i>
                  </el-tooltip>
                  <span>{{ item.explanation }}</span>
                </div>
              </div>
              
              <div class="item-actions">
                <el-button 
                  type="text" 
                  icon="el-icon-star-off" 
                  @click.stop="onLike(item)"
                  :class="{ 'liked': item.isLiked }"
                >
                </el-button>
                <el-button 
                  type="text" 
                  icon="el-icon-share" 
                  @click.stop="onShare(item)"
                >
                </el-button>
                <el-dropdown @command="onMoreAction" trigger="click">
                  <el-button type="text" icon="el-icon-more"></el-button>
                  <template #dropdown>
                    <el-dropdown-menu>
                      <el-dropdown-item :command="{action: 'not_interested', item}">不感兴趣</el-dropdown-item>
                      <el-dropdown-item :command="{action: 'report', item}">举报</el-dropdown-item>
                    </el-dropdown-menu>
                  </template>
                </el-dropdown>
              </div>
            </div>
          </div>
          
          <!-- 加载更多 -->
          <div class="load-more" v-if="hasMore">
            <el-button 
              @click="loadMore" 
              :loading="loadingMore"
              type="primary"
              plain
            >
              加载更多
            </el-button>
          </div>
        </template>
      </el-skeleton>
    </div>
    
    <!-- 用户画像展示 -->
    <div class="user-profile" v-if="showProfile">
      <el-card class="profile-card">
        <template #header>
          <div class="card-header">
            <span>我的兴趣画像</span>
            <el-button type="text" @click="showProfile = false">收起</el-button>
          </div>
        </template>
        
        <div class="interests-chart">
          <div class="chart-title">兴趣分布</div>
          <div class="interests-list">
            <div 
              v-for="(weight, interest) in userProfile.interests" 
              :key="interest"
              class="interest-item"
            >
              <span class="interest-label">{{ interest }}</span>
              <div class="interest-bar">
                <div 
                  class="interest-fill" 
                  :style="{ width: (weight * 100) + '%' }"
                ></div>
              </div>
              <span class="interest-value">{{ (weight * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
        
        <div class="behavior-stats">
          <div class="chart-title">行为统计</div>
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-value">{{ userProfile.behavior_features?.total_behaviors || 0 }}</div>
              <div class="stat-label">总行为数</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">{{ userProfile.behavior_features?.activity_level || 'low' }}</div>
              <div class="stat-label">活跃度</div>
            </div>
          </div>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useRecommendationStore } from '@/stores/recommendation'
import { useUserStore } from '@/stores/user'
import { useRouter } from 'vue-router'

// Props
interface Props {
  userId?: string
  context?: Record<string, any>
  showUserProfile?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showUserProfile: false
})

// Stores
const recommendationStore = useRecommendationStore()
const userStore = useUserStore()
const router = useRouter()

// Reactive data
const loading = ref(false)
const loadingMore = ref(false)
const showSettings = ref(false)
const showProfile = ref(props.showUserProfile)
const recommendations = ref([])
const userProfile = ref({})
const hasMore = ref(true)
const currentPage = ref(1)

const preferences = reactive({
  type: 'hybrid',
  limit: 20,
  contentTypes: ['research_paper', 'historical_document'],
  explanationLevel: 'basic'
})

// Computed
const currentUserId = computed(() => {
  return props.userId || userStore.currentUser?.id
})

// Methods
const loadRecommendations = async (reset = false) => {
  if (!currentUserId.value) return
  
  try {
    loading.value = reset
    loadingMore.value = !reset
    
    const params = {
      type: preferences.type,
      limit: preferences.limit,
      context: props.context
    }
    
    const result = await recommendationStore.getUserRecommendations(
      currentUserId.value,
      params
    )
    
    if (reset) {
      recommendations.value = result.recommendations
      currentPage.value = 1
    } else {
      recommendations.value.push(...result.recommendations)
      currentPage.value++
    }
    
    hasMore.value = result.recommendations.length === preferences.limit
    
  } catch (error) {
    console.error('加载推荐失败:', error)
    ElMessage.error('加载推荐失败，请稍后重试')
  } finally {
    loading.value = false
    loadingMore.value = false
  }
}

const loadUserProfile = async () => {
  if (!currentUserId.value) return
  
  try {
    userProfile.value = await recommendationStore.getUserProfile(currentUserId.value)
  } catch (error) {
    console.error('加载用户画像失败:', error)
  }
}

const refreshRecommendations = () => {
  loadRecommendations(true)
}

const loadMore = () => {
  loadRecommendations(false)
}

const onPreferenceChange = async () => {
  // 保存用户偏好
  try {
    await recommendationStore.updateUserPreferences(currentUserId.value, preferences)
    // 重新加载推荐
    loadRecommendations(true)
  } catch (error) {
    console.error('更新偏好失败:', error)
    ElMessage.error('更新偏好失败')
  }
}

const onItemClick = async (item: any) => {
  // 记录点击行为
  await recordBehavior(item.document_id, 'view')
  
  // 跳转到文档详情页
  router.push({
    name: 'DocumentDetail',
    params: { id: item.document_id }
  })
}

const onLike = async (item: any) => {
  try {
    await recommendationStore.submitFeedback(
      currentUserId.value,
      item.recommendation_id,
      'like'
    )
    
    item.isLiked = !item.isLiked
    ElMessage.success(item.isLiked ? '已添加到喜欢' : '已取消喜欢')
    
    // 记录行为
    await recordBehavior(item.document_id, 'like')
    
  } catch (error) {
    console.error('操作失败:', error)
    ElMessage.error('操作失败，请稍后重试')
  }
}

const onShare = async (item: any) => {
  try {
    // 复制分享链接
    const shareUrl = `${window.location.origin}/documents/${item.document_id}`
    await navigator.clipboard.writeText(shareUrl)
    
    ElMessage.success('分享链接已复制到剪贴板')
    
    // 记录分享行为
    await recordBehavior(item.document_id, 'share')
    
  } catch (error) {
    console.error('分享失败:', error)
    ElMessage.error('分享失败')
  }
}

const onMoreAction = async (command: any) => {
  const { action, item } = command
  
  if (action === 'not_interested') {
    try {
      await recommendationStore.submitFeedback(
        currentUserId.value,
        item.recommendation_id,
        'not_interested',
        '用户标记为不感兴趣'
      )
      
      // 从推荐列表中移除
      const index = recommendations.value.findIndex(rec => rec.document_id === item.document_id)
      if (index > -1) {
        recommendations.value.splice(index, 1)
      }
      
      ElMessage.success('已标记为不感兴趣')
      
    } catch (error) {
      console.error('操作失败:', error)
      ElMessage.error('操作失败')
    }
  } else if (action === 'report') {
    ElMessageBox.prompt('请说明举报原因', '举报内容', {
      confirmButtonText: '提交',
      cancelButtonText: '取消'
    }).then(async ({ value }) => {
      try {
        await recommendationStore.submitFeedback(
          currentUserId.value,
          item.recommendation_id,
          'report',
          value
        )
        
        ElMessage.success('举报已提交，我们会尽快处理')
        
      } catch (error) {
        console.error('举报失败:', error)
        ElMessage.error('举报失败')
      }
    })
  }
}

const recordBehavior = async (documentId: string, behaviorType: string) => {
  try {
    await recommendationStore.recordUserBehavior({
      user_id: currentUserId.value,
      document_id: documentId,
      behavior_type: behaviorType,
      context_data: {
        source: 'recommendation',
        page: 'recommendation_list'
      }
    })
  } catch (error) {
    console.error('记录行为失败:', error)
  }
}

const getDocumentThumbnail = (docInfo: any) => {
  return docInfo.thumbnail || '/images/default-document.png'
}

const onImageError = (event: Event) => {
  const target = event.target as HTMLImageElement
  target.src = '/images/default-document.png'
}

const getAlgorithmClass = (algorithm: string) => {
  const classMap = {
    'content_based': 'content-tag',
    'collaborative_user': 'collaborative-tag',
    'collaborative_item': 'collaborative-tag',
    'hybrid': 'hybrid-tag'
  }
  return classMap[algorithm] || 'default-tag'
}

const getAlgorithmLabel = (algorithm: string) => {
  const labelMap = {
    'content_based': '内容推荐',
    'collaborative_user': '用户协同',
    'collaborative_item': '物品协同',
    'hybrid': '智能推荐'
  }
  return labelMap[algorithm] || '系统推荐'
}

// Lifecycle
onMounted(() => {
  loadRecommendations(true)
  if (showProfile.value) {
    loadUserProfile()
  }
})
</script>

<style scoped>
.recommendation-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.recommendation-settings {
  margin-bottom: 20px;
}

.settings-card {
  border-radius: 8px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.list-header h3 {
  margin: 0;
  color: #303133;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.recommendation-skeleton {
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
  padding: 15px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
}

.skeleton-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.recommendations-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.recommendation-item {
  display: flex;
  gap: 15px;
  padding: 15px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
}

.recommendation-item:hover {
  border-color: #409eff;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.item-rank {
  position: absolute;
  top: -8px;
  left: -8px;
  width: 24px;
  height: 24px;
  background: #409eff;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
}

.item-thumbnail {
  width: 80px;
  height: 60px;
  border-radius: 4px;
  overflow: hidden;
  flex-shrink: 0;
}

.item-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.item-content {
  flex: 1;
  min-width: 0;
}

.item-title {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-author,
.item-category {
  margin: 0 0 4px 0;
  font-size: 14px;
  color: #909399;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 8px 0;
}

.algorithm-tag {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.content-tag {
  background: #e1f3d8;
  color: #67c23a;
}

.collaborative-tag {
  background: #fdf6ec;
  color: #e6a23c;
}

.hybrid-tag {
  background: #ecf5ff;
  color: #409eff;
}

.score {
  font-size: 12px;
  color: #909399;
  font-weight: 500;
}

.item-explanation {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
}

.item-actions {
  display: flex;
  flex-direction: column;
  gap: 5px;
  align-items: center;
}

.item-actions .el-button.liked {
  color: #f56c6c;
}

.load-more {
  text-align: center;
  margin-top: 20px;
}

.user-profile {
  margin-top: 30px;
}

.profile-card {
  border-radius: 8px;
}

.chart-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 15px;
}

.interests-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.interest-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.interest-label {
  width: 80px;
  font-size: 14px;
  color: #606266;
  flex-shrink: 0;
}

.interest-bar {
  flex: 1;
  height: 8px;
  background: #f5f7fa;
  border-radius: 4px;
  overflow: hidden;
}

.interest-fill {
  height: 100%;
  background: linear-gradient(90deg, #409eff, #67c23a);
  transition: width 0.3s ease;
}

.interest-value {
  width: 50px;
  text-align: right;
  font-size: 12px;
  color: #909399;
  flex-shrink: 0;
}

.behavior-stats {
  margin-top: 30px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 20px;
}

.stat-item {
  text-align: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #409eff;
  margin-bottom: 5px;
}

.stat-label {
  font-size: 14px;
  color: #909399;
}
</style>
```

#### 相似文档推荐组件
```vue
<template>
  <div class="similar-documents">
    <div class="section-header">
      <h4>相关推荐</h4>
      <el-button type="text" @click="refreshSimilar" :loading="loading">
        <i class="el-icon-refresh"></i>
      </el-button>
    </div>
    
    <el-skeleton :loading="loading" animated>
      <template #template>
        <div v-for="i in 3" :key="i" class="similar-skeleton">
          <el-skeleton-item variant="image" style="width: 60px; height: 45px;"></el-skeleton-item>
          <div class="skeleton-content">
            <el-skeleton-item variant="h4" style="width: 80%;"></el-skeleton-item>
            <el-skeleton-item variant="text" style="width: 60%;"></el-skeleton-item>
          </div>
        </div>
      </template>
      
      <template #default>
        <div class="similar-list">
          <div 
            v-for="(doc, index) in similarDocuments" 
            :key="doc.document_id"
            class="similar-item"
            @click="onDocumentClick(doc)"
          >
            <div class="item-thumbnail">
              <img 
                :src="getDocumentThumbnail(doc.document_info)" 
                :alt="doc.document_info.title"
                @error="onImageError"
              >
            </div>
            
            <div class="item-content">
              <h5 class="item-title">{{ doc.document_info.title }}</h5>
              <p class="item-author">{{ doc.document_info.author }}</p>
              <div class="item-similarity">
                <span class="similarity-label">相似度:</span>
                <span class="similarity-value">{{ (doc.similarity_score * 100).toFixed(0) }}%</span>
              </div>
            </div>
            
            <div class="item-rank">{{ index + 1 }}</div>
          </div>
        </div>
        
        <div class="view-more" v-if="similarDocuments.length > 0">
          <el-button type="text" @click="viewAllSimilar">
            查看更多相似文档
            <i class="el-icon-arrow-right"></i>
          </el-button>
        </div>
      </template>
    </el-skeleton>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRecommendationStore } from '@/stores/recommendation'
import { useRouter } from 'vue-router'

// Props
interface Props {
  documentId: string
  limit?: number
}

const props = withDefaults(defineProps<Props>(), {
  limit: 5
})

// Stores
const recommendationStore = useRecommendationStore()
const router = useRouter()

// Reactive data
const loading = ref(false)
const similarDocuments = ref([])

// Methods
const loadSimilarDocuments = async () => {
  if (!props.documentId) return
  
  try {
    loading.value = true
    
    const result = await recommendationStore.getSimilarDocuments(
      props.documentId,
      { limit: props.limit }
    )
    
    similarDocuments.value = result.similar_documents
    
  } catch (error) {
    console.error('加载相似文档失败:', error)
  } finally {
    loading.value = false
  }
}

const refreshSimilar = () => {
  loadSimilarDocuments()
}

const onDocumentClick = (doc: any) => {
  router.push({
    name: 'DocumentDetail',
    params: { id: doc.document_id }
  })
}

const viewAllSimilar = () => {
  router.push({
    name: 'SimilarDocuments',
    params: { id: props.documentId }
  })
}

const getDocumentThumbnail = (docInfo: any) => {
  return docInfo.thumbnail || '/images/default-document.png'
}

const onImageError = (event: Event) => {
  const target = event.target as HTMLImageElement
  target.src = '/images/default-document.png'
}

// Watchers
watch(() => props.documentId, () => {
  loadSimilarDocuments()
})

// Lifecycle
onMounted(() => {
  loadSimilarDocuments()
})
</script>

<style scoped>
.similar-documents {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.section-header h4 {
  margin: 0;
  color: #303133;
  font-size: 16px;
}

.similar-skeleton {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.skeleton-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.similar-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.similar-item {
  display: flex;
  gap: 12px;
  padding: 12px;
  border: 1px solid #ebeef5;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
}

.similar-item:hover {
  border-color: #409eff;
  background: #f8f9fa;
}

.item-thumbnail {
  width: 50px;
  height: 40px;
  border-radius: 4px;
  overflow: hidden;
  flex-shrink: 0;
}

.item-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.item-content {
  flex: 1;
  min-width: 0;
}

.item-title {
  margin: 0 0 4px 0;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-author {
  margin: 0 0 4px 0;
  font-size: 12px;
  color: #909399;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-similarity {
  display: flex;
  align-items: center;
  gap: 5px;
}

.similarity-label {
  font-size: 12px;
  color: #909399;
}

.similarity-value {
  font-size: 12px;
  color: #409eff;
  font-weight: 500;
}

.item-rank {
  position: absolute;
  top: -6px;
  right: -6px;
  width: 18px;
  height: 18px;
  background: #409eff;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: bold;
}

.view-more {
  text-align: center;
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #ebeef5;
}
</style>
```

## 验收标准

### 功能性验收标准

#### 基础推荐功能
- [ ] **内容推荐**: 系统能够基于文档内容相似度进行推荐
- [ ] **协同过滤**: 系统能够基于用户行为相似性进行推荐
- [ ] **混合推荐**: 系统能够融合多种算法提供综合推荐
- [ ] **个性化推荐**: 系统能够根据用户画像提供个性化推荐
- [ ] **相似文档**: 系统能够为指定文档推荐相似内容

#### 用户画像功能
- [ ] **兴趣分析**: 系统能够基于用户行为分析兴趣偏好
- [ ] **行为追踪**: 系统能够记录和分析用户各种行为
- [ ] **画像更新**: 用户画像能够实时更新和优化
- [ ] **偏好设置**: 用户能够自定义推荐偏好和参数
- [ ] **冷启动处理**: 系统能够为新用户提供合理的初始推荐

#### 反馈机制
- [ ] **用户反馈**: 用户能够对推荐结果进行反馈（喜欢/不喜欢）
- [ ] **反馈学习**: 系统能够基于用户反馈优化推荐算法
- [ ] **推荐解释**: 系统能够为推荐结果提供解释说明
- [ ] **不感兴趣**: 用户能够标记不感兴趣的内容
- [ ] **举报功能**: 用户能够举报不当推荐内容

#### 推荐场景
- [ ] **首页推荐**: 在首页展示个性化推荐内容
- [ ] **搜索推荐**: 在搜索结果中提供相关推荐
- [ ] **阅读推荐**: 在文档阅读页面推荐相关内容
- [ ] **分类推荐**: 在分类页面提供该分类的热门推荐
- [ ] **历史推荐**: 基于用户历史行为推荐相关内容

### 性能验收标准

#### 响应时间要求
- [ ] **推荐生成**: 个人推荐生成时间 < 500ms
- [ ] **相似文档**: 相似文档推荐时间 < 200ms
- [ ] **用户画像**: 用户画像查询时间 < 100ms
- [ ] **行为记录**: 用户行为记录时间 < 50ms
- [ ] **反馈处理**: 用户反馈处理时间 < 100ms

#### 并发性能要求
- [ ] **并发用户**: 支持1000+并发用户同时获取推荐
- [ ] **推荐吞吐**: 推荐服务QPS > 500
- [ ] **缓存命中**: 推荐缓存命中率 > 80%
- [ ] **数据库性能**: 数据库查询响应时间 < 100ms
- [ ] **内存使用**: 推荐服务内存使用率 < 80%

#### 准确性要求
- [ ] **推荐准确率**: 推荐内容准确率 > 70%
- [ ] **点击率**: 推荐内容点击率 > 15%
- [ ] **用户满意度**: 用户对推荐的满意度 > 80%
- [ ] **多样性**: 推荐结果具有足够的多样性
- [ ] **新颖性**: 推荐结果包含用户未接触过的新内容

### 安全性验收标准

#### 数据安全
- [ ] **隐私保护**: 用户行为数据加密存储和传输
- [ ] **访问控制**: 推荐数据访问权限控制
- [ ] **数据脱敏**: 敏感用户信息脱敏处理
- [ ] **审计日志**: 完整的推荐操作审计日志
- [ ] **数据备份**: 推荐数据定期备份和恢复

#### 内容安全
- [ ] **内容过滤**: 过滤不当或违规推荐内容
- [ ] **举报处理**: 及时处理用户举报的问题内容
- [ ] **算法公平**: 推荐算法避免歧视和偏见
- [ ] **透明度**: 推荐算法具有一定的可解释性
- [ ] **用户控制**: 用户能够控制推荐内容类型

## 业务价值

### 用户体验提升
1. **个性化体验**: 为每个用户提供个性化的内容推荐
2. **发现效率**: 帮助用户快速发现感兴趣的历史文献
3. **学习路径**: 为研究人员提供系统的学习和研究路径
4. **时间节省**: 减少用户查找相关资料的时间成本

### 平台价值创造
1. **用户粘性**: 提高用户在平台的停留时间和使用频率
2. **内容消费**: 增加平台内容的浏览和使用量
3. **用户增长**: 通过优质推荐吸引更多用户使用平台
4. **数据价值**: 积累用户行为数据，为平台优化提供依据

### 学术研究支持
1. **研究发现**: 帮助研究人员发现相关的研究资料
2. **跨领域连接**: 促进不同研究领域之间的知识连接
3. **研究趋势**: 通过推荐数据分析研究热点和趋势
4. **协作促进**: 基于兴趣相似性促进研究人员之间的协作

## 依赖关系

### 前置依赖
1. **用户认证服务** (Story 1.2): 需要用户身份信息进行个性化推荐
2. **数据收集服务** (Story 1.3): 需要文档数据作为推荐内容源
3. **搜索引擎服务** (Story 3.1): 与搜索功能集成，提供搜索推荐

### 后续依赖
1. **数据分析服务**: 推荐数据为数据分析提供用户行为数据
2. **内容管理**: 推荐结果影响内容的展示和管理策略
3. **用户运营**: 推荐数据支持用户运营和个性化营销

### 外部依赖
1. **机器学习平台**: 依赖ML平台进行模型训练和部署
2. **大数据处理**: 依赖大数据平台处理用户行为数据
3. **缓存服务**: 依赖Redis等缓存服务提高推荐性能
4. **消息队列**: 依赖消息队列处理异步推荐任务

## 风险和挑战

### 技术风险
1. **算法复杂性**: 推荐算法实现和调优的复杂性
2. **性能瓶颈**: 大规模用户和数据下的性能挑战
3. **数据质量**: 用户行为数据质量影响推荐效果
4. **冷启动问题**: 新用户和新内容的推荐挑战

### 业务风险
1. **用户接受度**: 用户对推荐结果的接受度和满意度
2. **推荐偏见**: 算法可能产生的推荐偏见和过滤泡沫
3. **隐私担忧**: 用户对个人数据使用的隐私担忧
4. **内容质量**: 推荐内容的质量和相关性保证

### 运营风险
1. **算法维护**: 推荐算法的持续优化和维护成本
2. **数据合规**: 用户数据收集和使用的合规性要求
3. **系统稳定**: 推荐系统的稳定性和可用性保证
4. **成本控制**: 推荐系统的计算和存储成本控制

## 开发任务分解

### 后端开发任务

#### 阶段一：基础架构 (2天)
- [ ] 设计推荐系统数据库模型
- [ ] 创建推荐服务基础架构
- [ ] 实现用户行为数据收集接口
- [ ] 搭建推荐算法框架
- [ ] 配置Redis缓存和消息队列

#### 阶段二：推荐算法 (3天)
- [ ] 实现基于内容的推荐算法
- [ ] 实现协同过滤推荐算法
- [ ] 实现混合推荐算法
- [ ] 开发用户画像构建功能
- [ ] 实现推荐结果排序和过滤

#### 阶段三：API开发 (1天)
- [ ] 开发用户推荐API接口
- [ ] 开发相似文档推荐API
- [ ] 开发用户行为记录API
- [ ] 开发用户反馈API
- [ ] 开发用户画像查询API

### 前端开发任务

#### 阶段一：基础组件 (1天)
- [ ] 创建推荐列表组件
- [ ] 创建相似文档组件
- [ ] 创建用户画像展示组件
- [ ] 实现推荐设置界面
- [ ] 集成推荐数据状态管理

#### 阶段二：交互功能 (1天)
- [ ] 实现推荐内容点击跳转
- [ ] 实现用户反馈功能
- [ ] 实现推荐偏好设置
- [ ] 实现推荐刷新和加载更多
- [ ] 实现推荐解释展示

#### 阶段三：页面集成 (1天)
- [ ] 在首页集成推荐组件
- [ ] 在搜索页面集成推荐
- [ ] 在文档详情页集成相似推荐
- [ ] 在用户中心集成画像展示
- [ ] 优化推荐组件响应式设计

### 测试任务

#### 单元测试 (1天)
- [ ] 推荐算法单元测试
- [ ] API接口单元测试
- [ ] 前端组件单元测试
- [ ] 数据处理函数测试
- [ ] 缓存功能测试

#### 集成测试 (1天)
- [ ] 推荐服务集成测试
- [ ] 前后端集成测试
- [ ] 数据库集成测试
- [ ] 缓存集成测试
- [ ] 消息队列集成测试

#### 性能测试 (1天)
- [ ] 推荐生成性能测试
- [ ] 并发用户推荐测试
- [ ] 数据库查询性能测试
- [ ] 缓存性能测试
- [ ] 前端组件性能测试

### 部署任务

#### 环境准备 (0.5天)
- [ ] 配置推荐服务部署环境
- [ ] 配置机器学习运行环境
- [ ] 配置Redis和消息队列
- [ ] 配置监控和日志系统
- [ ] 准备推荐模型部署

#### 部署上线 (0.5天)
- [ ] 部署推荐服务到测试环境
- [ ] 部署前端推荐组件
- [ ] 配置推荐数据同步
- [ ] 验证推荐功能正常运行
- [ ] 部署到生产环境并监控
```