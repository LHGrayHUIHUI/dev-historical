# 用户故事 3.5: 用户行为分析服务

## 故事描述

**作为** 系统管理员和产品经理  
**我希望** 有一个用户行为分析服务  
**以便于** 深入了解用户使用模式、优化产品功能、提升用户体验，并为产品决策提供数据支持

## 详细需求

### 核心功能

1. **用户行为追踪**
   - 页面访问记录
   - 功能使用统计
   - 用户操作路径分析
   - 停留时间统计

2. **用户画像分析**
   - 用户基本信息分析
   - 使用习惯分析
   - 偏好特征提取
   - 用户分群

3. **实时监控**
   - 在线用户统计
   - 实时访问量监控
   - 异常行为检测
   - 系统性能监控

4. **数据可视化**
   - 行为数据仪表板
   - 趋势分析图表
   - 用户流量分析
   - 自定义报表

5. **智能分析**
   - 用户流失预测
   - 功能使用推荐
   - 异常行为识别
   - 个性化内容推荐

## 技术栈

### 后端技术栈

- **Web框架**: FastAPI
- **数据处理**: Pandas, NumPy, SciPy
- **机器学习**: scikit-learn, XGBoost, LightGBM
- **时间序列分析**: Prophet, statsmodels
- **实时处理**: Apache Kafka, Redis Streams
- **任务队列**: Celery, RQ
- **数据库**: 
  - ClickHouse（时间序列数据）
  - PostgreSQL（用户数据）
  - Redis（缓存和会话）
  - InfluxDB（监控数据）
- **消息队列**: RabbitMQ, Apache Kafka
- **监控**: Prometheus, Grafana
- **日志**: ELK Stack (Elasticsearch, Logstash, Kibana)

### 前端技术栈

- **框架**: Vue 3 + TypeScript
- **状态管理**: Pinia
- **路由**: Vue Router 4
- **UI组件**: Element Plus
- **图表库**: ECharts, D3.js, Chart.js
- **实时通信**: Socket.IO
- **数据可视化**: Vue-ECharts, @vue/composition-api
- **地图可视化**: Leaflet, MapBox
- **构建工具**: Vite
- **代码质量**: ESLint, Prettier

## 数据模型设计

### ClickHouse 时间序列数据表

```sql
-- 用户行为事件表
CREATE TABLE user_events (
    event_id String,
    user_id String,
    session_id String,
    event_type String,
    event_name String,
    page_url String,
    page_title String,
    referrer String,
    user_agent String,
    ip_address String,
    country String,
    city String,
    device_type String,
    browser String,
    os String,
    screen_resolution String,
    event_properties String, -- JSON格式
    timestamp DateTime64(3),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, timestamp)
TTL timestamp + INTERVAL 2 YEAR;

-- 页面访问统计表
CREATE TABLE page_views (
    page_url String,
    page_title String,
    user_id String,
    session_id String,
    visit_duration UInt32, -- 秒
    bounce_rate Float32,
    entry_page Boolean,
    exit_page Boolean,
    timestamp DateTime64(3),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (page_url, timestamp);

-- 用户会话表
CREATE TABLE user_sessions (
    session_id String,
    user_id String,
    start_time DateTime64(3),
    end_time DateTime64(3),
    duration UInt32, -- 秒
    page_views UInt32,
    events_count UInt32,
    bounce Boolean,
    conversion Boolean,
    device_info String, -- JSON格式
    location_info String, -- JSON格式
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(start_time)
ORDER BY (user_id, start_time);
```

### PostgreSQL 用户数据表

```sql
-- 用户画像表
CREATE TABLE user_profiles (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100),
    email VARCHAR(255),
    registration_date TIMESTAMP,
    last_login TIMESTAMP,
    total_sessions INTEGER DEFAULT 0,
    total_page_views INTEGER DEFAULT 0,
    total_events INTEGER DEFAULT 0,
    avg_session_duration FLOAT DEFAULT 0,
    preferred_features JSONB,
    user_segment VARCHAR(50),
    risk_score FLOAT DEFAULT 0,
    lifetime_value FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户分群表
CREATE TABLE user_segments (
    segment_id VARCHAR(50) PRIMARY KEY,
    segment_name VARCHAR(100) NOT NULL,
    description TEXT,
    criteria JSONB, -- 分群条件
    user_count INTEGER DEFAULT 0,
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 用户分群关系表
CREATE TABLE user_segment_members (
    user_id VARCHAR(50),
    segment_id VARCHAR(50),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, segment_id),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id),
    FOREIGN KEY (segment_id) REFERENCES user_segments(segment_id)
);

-- 分析任务表
CREATE TABLE analysis_tasks (
    task_id VARCHAR(50) PRIMARY KEY,
    task_name VARCHAR(100) NOT NULL,
    task_type VARCHAR(50), -- cohort, funnel, retention, etc.
    parameters JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    result JSONB,
    created_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
```

## 服务架构

### 核心服务类

```python
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import clickhouse_connect
import asyncio

class UserBehaviorAnalytics:
    """
    用户行为分析核心服务类
    提供用户行为数据收集、分析和洞察功能
    """
    
    def __init__(self, clickhouse_client, postgres_client, redis_client):
        self.ch_client = clickhouse_client
        self.pg_client = postgres_client
        self.redis_client = redis_client
        self.scaler = StandardScaler()
    
    async def track_event(self, event_data: Dict[str, Any]) -> bool:
        """
        记录用户行为事件
        
        Args:
            event_data: 事件数据字典
            
        Returns:
            bool: 记录是否成功
        """
        try:
            # 数据验证和清洗
            cleaned_data = self._clean_event_data(event_data)
            
            # 实时写入ClickHouse
            await self._insert_event(cleaned_data)
            
            # 更新Redis缓存
            await self._update_realtime_stats(cleaned_data)
            
            # 触发实时分析
            await self._trigger_realtime_analysis(cleaned_data)
            
            return True
            
        except Exception as e:
            print(f"事件记录失败: {e}")
            return False
    
    def _clean_event_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗和验证事件数据
        """
        cleaned = {
            'event_id': event_data.get('event_id', self._generate_event_id()),
            'user_id': event_data.get('user_id', 'anonymous'),
            'session_id': event_data.get('session_id'),
            'event_type': event_data.get('event_type', 'page_view'),
            'event_name': event_data.get('event_name'),
            'page_url': event_data.get('page_url'),
            'page_title': event_data.get('page_title'),
            'referrer': event_data.get('referrer'),
            'user_agent': event_data.get('user_agent'),
            'ip_address': event_data.get('ip_address'),
            'timestamp': event_data.get('timestamp', datetime.now()),
            'event_properties': event_data.get('properties', {})
        }
        
        # 解析地理位置信息
        if cleaned['ip_address']:
            location = self._get_location_from_ip(cleaned['ip_address'])
            cleaned.update(location)
        
        # 解析设备信息
        if cleaned['user_agent']:
            device_info = self._parse_user_agent(cleaned['user_agent'])
            cleaned.update(device_info)
        
        return cleaned
    
    async def _insert_event(self, event_data: Dict[str, Any]):
        """
        将事件数据插入ClickHouse
        """
        query = """
        INSERT INTO user_events (
            event_id, user_id, session_id, event_type, event_name,
            page_url, page_title, referrer, user_agent, ip_address,
            country, city, device_type, browser, os, screen_resolution,
            event_properties, timestamp
        ) VALUES
        """
        
        values = (
            event_data['event_id'],
            event_data['user_id'],
            event_data['session_id'],
            event_data['event_type'],
            event_data['event_name'],
            event_data['page_url'],
            event_data['page_title'],
            event_data['referrer'],
            event_data['user_agent'],
            event_data['ip_address'],
            event_data.get('country'),
            event_data.get('city'),
            event_data.get('device_type'),
            event_data.get('browser'),
            event_data.get('os'),
            event_data.get('screen_resolution'),
            json.dumps(event_data['event_properties']),
            event_data['timestamp']
        )
        
        self.ch_client.insert(query, [values])
    
    async def get_user_behavior_summary(self, user_id: str, 
                                      days: int = 30) -> Dict[str, Any]:
        """
        获取用户行为摘要
        
        Args:
            user_id: 用户ID
            days: 分析天数
            
        Returns:
            Dict: 用户行为摘要数据
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 查询用户事件数据
        events_query = """
        SELECT 
            event_type,
            event_name,
            page_url,
            COUNT(*) as event_count,
            COUNT(DISTINCT session_id) as session_count,
            MIN(timestamp) as first_event,
            MAX(timestamp) as last_event
        FROM user_events 
        WHERE user_id = %(user_id)s 
            AND timestamp >= %(start_date)s 
            AND timestamp <= %(end_date)s
        GROUP BY event_type, event_name, page_url
        ORDER BY event_count DESC
        """
        
        events_df = pd.read_sql(
            events_query, 
            self.ch_client,
            params={
                'user_id': user_id,
                'start_date': start_date,
                'end_date': end_date
            }
        )
        
        # 查询会话数据
        sessions_query = """
        SELECT 
            COUNT(*) as total_sessions,
            AVG(duration) as avg_session_duration,
            SUM(page_views) as total_page_views,
            SUM(events_count) as total_events,
            COUNT(CASE WHEN bounce = 1 THEN 1 END) as bounce_sessions
        FROM user_sessions 
        WHERE user_id = %(user_id)s 
            AND start_time >= %(start_date)s 
            AND start_time <= %(end_date)s
        """
        
        sessions_result = self.ch_client.query(
            sessions_query,
            parameters={
                'user_id': user_id,
                'start_date': start_date,
                'end_date': end_date
            }
        )
        
        sessions_data = sessions_result.result_rows[0] if sessions_result.result_rows else [0] * 5
        
        # 计算用户活跃度
        activity_score = self._calculate_activity_score(events_df, sessions_data)
        
        # 识别用户偏好
        preferences = self._identify_user_preferences(events_df)
        
        return {
            'user_id': user_id,
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days
            },
            'activity_summary': {
                'total_sessions': sessions_data[0],
                'avg_session_duration': round(sessions_data[1], 2),
                'total_page_views': sessions_data[2],
                'total_events': sessions_data[3],
                'bounce_rate': round(sessions_data[4] / max(sessions_data[0], 1), 2),
                'activity_score': activity_score
            },
            'top_events': events_df.head(10).to_dict('records'),
            'preferences': preferences,
            'user_segment': await self._get_user_segment(user_id)
        }
    
    def _calculate_activity_score(self, events_df: pd.DataFrame, 
                                sessions_data: List) -> float:
        """
        计算用户活跃度评分
        """
        if events_df.empty or sessions_data[0] == 0:
            return 0.0
        
        # 基于多个维度计算活跃度
        session_score = min(sessions_data[0] / 10, 1.0)  # 会话数量
        duration_score = min(sessions_data[1] / 1800, 1.0)  # 平均会话时长(30分钟为满分)
        engagement_score = min(sessions_data[2] / 50, 1.0)  # 页面浏览数
        diversity_score = min(len(events_df) / 20, 1.0)  # 事件类型多样性
        
        # 加权计算总分
        total_score = (
            session_score * 0.3 +
            duration_score * 0.3 +
            engagement_score * 0.2 +
            diversity_score * 0.2
        ) * 100
        
        return round(total_score, 2)
    
    def _identify_user_preferences(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        识别用户偏好特征
        """
        if events_df.empty:
            return {}
        
        # 分析页面偏好
        page_preferences = events_df.groupby('page_url')['event_count'].sum().sort_values(ascending=False)
        
        # 分析功能偏好
        feature_preferences = events_df.groupby('event_name')['event_count'].sum().sort_values(ascending=False)
        
        # 分析使用时间模式
        time_pattern = self._analyze_time_pattern(events_df)
        
        return {
            'top_pages': page_preferences.head(5).to_dict(),
            'top_features': feature_preferences.head(5).to_dict(),
            'time_pattern': time_pattern,
            'engagement_level': self._classify_engagement_level(events_df)
        }
    
    async def perform_cohort_analysis(self, start_date: datetime, 
                                    end_date: datetime) -> Dict[str, Any]:
        """
        执行队列分析
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 队列分析结果
        """
        # 获取用户首次访问数据
        first_visit_query = """
        SELECT 
            user_id,
            MIN(DATE(timestamp)) as first_visit_date
        FROM user_events 
        WHERE timestamp >= %(start_date)s 
            AND timestamp <= %(end_date)s
            AND user_id != 'anonymous'
        GROUP BY user_id
        """
        
        first_visits_df = pd.read_sql(
            first_visit_query,
            self.ch_client,
            params={'start_date': start_date, 'end_date': end_date}
        )
        
        # 获取用户后续访问数据
        subsequent_visits_query = """
        SELECT 
            user_id,
            DATE(timestamp) as visit_date
        FROM user_events 
        WHERE timestamp >= %(start_date)s 
            AND timestamp <= %(end_date)s
            AND user_id != 'anonymous'
        GROUP BY user_id, DATE(timestamp)
        """
        
        visits_df = pd.read_sql(
            subsequent_visits_query,
            self.ch_client,
            params={'start_date': start_date, 'end_date': end_date}
        )
        
        # 构建队列分析矩阵
        cohort_matrix = self._build_cohort_matrix(first_visits_df, visits_df)
        
        return {
            'cohort_matrix': cohort_matrix.to_dict(),
            'retention_rates': self._calculate_retention_rates(cohort_matrix),
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        }
    
    async def perform_funnel_analysis(self, funnel_steps: List[str], 
                                    days: int = 30) -> Dict[str, Any]:
        """
        执行漏斗分析
        
        Args:
            funnel_steps: 漏斗步骤列表
            days: 分析天数
            
        Returns:
            Dict: 漏斗分析结果
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        funnel_data = []
        
        for i, step in enumerate(funnel_steps):
            if i == 0:
                # 第一步：所有进入漏斗的用户
                query = """
                SELECT COUNT(DISTINCT user_id) as user_count
                FROM user_events 
                WHERE event_name = %(step)s
                    AND timestamp >= %(start_date)s 
                    AND timestamp <= %(end_date)s
                """
            else:
                # 后续步骤：完成前一步的用户中完成当前步骤的用户
                query = """
                WITH step_users AS (
                    SELECT DISTINCT user_id
                    FROM user_events 
                    WHERE event_name = %(prev_step)s
                        AND timestamp >= %(start_date)s 
                        AND timestamp <= %(end_date)s
                )
                SELECT COUNT(DISTINCT e.user_id) as user_count
                FROM user_events e
                INNER JOIN step_users su ON e.user_id = su.user_id
                WHERE e.event_name = %(step)s
                    AND e.timestamp >= %(start_date)s 
                    AND e.timestamp <= %(end_date)s
                """
            
            params = {
                'step': step,
                'start_date': start_date,
                'end_date': end_date
            }
            
            if i > 0:
                params['prev_step'] = funnel_steps[i-1]
            
            result = self.ch_client.query(query, parameters=params)
            user_count = result.result_rows[0][0] if result.result_rows else 0
            
            conversion_rate = 0
            if i > 0 and funnel_data:
                conversion_rate = user_count / funnel_data[0]['user_count'] * 100
            
            funnel_data.append({
                'step': step,
                'step_index': i,
                'user_count': user_count,
                'conversion_rate': round(conversion_rate, 2)
            })
        
        return {
            'funnel_steps': funnel_data,
            'overall_conversion_rate': round(
                funnel_data[-1]['user_count'] / max(funnel_data[0]['user_count'], 1) * 100, 2
            ) if funnel_data else 0,
            'analysis_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days
            }
        }
    
    async def segment_users(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        用户分群分析
        
        Args:
            criteria: 分群条件
            
        Returns:
            Dict: 分群结果
        """
        # 构建查询条件
        where_conditions = []
        params = {}
        
        if criteria.get('activity_level'):
            where_conditions.append("total_sessions >= %(min_sessions)s")
            params['min_sessions'] = criteria['activity_level'].get('min_sessions', 1)
        
        if criteria.get('registration_period'):
            where_conditions.append(
                "registration_date >= %(reg_start)s AND registration_date <= %(reg_end)s"
            )
            params['reg_start'] = criteria['registration_period']['start']
            params['reg_end'] = criteria['registration_period']['end']
        
        if criteria.get('engagement_score'):
            where_conditions.append("avg_session_duration >= %(min_duration)s")
            params['min_duration'] = criteria['engagement_score'].get('min_duration', 60)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # 查询符合条件的用户
        query = f"""
        SELECT 
            user_id,
            username,
            total_sessions,
            avg_session_duration,
            user_segment,
            lifetime_value
        FROM user_profiles 
        WHERE {where_clause}
        ORDER BY total_sessions DESC
        """
        
        users_df = pd.read_sql(query, self.pg_client, params=params)
        
        # 如果启用机器学习分群
        if criteria.get('use_ml_clustering', False):
            clustered_users = self._perform_ml_clustering(users_df)
            return clustered_users
        
        return {
            'segment_criteria': criteria,
            'user_count': len(users_df),
            'users': users_df.to_dict('records'),
            'segment_stats': {
                'avg_sessions': users_df['total_sessions'].mean(),
                'avg_duration': users_df['avg_session_duration'].mean(),
                'avg_lifetime_value': users_df['lifetime_value'].mean()
            }
        }
    
    def _perform_ml_clustering(self, users_df: pd.DataFrame) -> Dict[str, Any]:
        """
        使用机器学习进行用户聚类分析
        """
        if users_df.empty:
            return {'clusters': [], 'user_count': 0}
        
        # 准备特征数据
        features = ['total_sessions', 'avg_session_duration', 'lifetime_value']
        X = users_df[features].fillna(0)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # K-means聚类
        n_clusters = min(5, len(users_df))  # 最多5个聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # 添加聚类标签
        users_df['cluster'] = cluster_labels
        
        # 分析每个聚类的特征
        cluster_analysis = []
        for cluster_id in range(n_clusters):
            cluster_users = users_df[users_df['cluster'] == cluster_id]
            
            cluster_analysis.append({
                'cluster_id': cluster_id,
                'user_count': len(cluster_users),
                'characteristics': {
                    'avg_sessions': cluster_users['total_sessions'].mean(),
                    'avg_duration': cluster_users['avg_session_duration'].mean(),
                    'avg_lifetime_value': cluster_users['lifetime_value'].mean()
                },
                'users': cluster_users.to_dict('records')
            })
        
        return {
            'clustering_method': 'K-Means',
            'n_clusters': n_clusters,
            'clusters': cluster_analysis,
            'total_users': len(users_df)
        }
    
    async def get_realtime_stats(self) -> Dict[str, Any]:
        """
        获取实时统计数据
        
        Returns:
            Dict: 实时统计数据
        """
        # 从Redis获取实时数据
        online_users = await self.redis_client.scard("online_users")
        
        # 获取今日统计
        today = datetime.now().date()
        today_stats_query = """
        SELECT 
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(*) as total_events,
            COUNT(DISTINCT page_url) as unique_pages
        FROM user_events 
        WHERE DATE(timestamp) = %(today)s
        """
        
        today_result = self.ch_client.query(
            today_stats_query,
            parameters={'today': today}
        )
        
        today_data = today_result.result_rows[0] if today_result.result_rows else [0] * 4
        
        # 获取热门页面
        popular_pages_query = """
        SELECT 
            page_url,
            page_title,
            COUNT(*) as views
        FROM user_events 
        WHERE DATE(timestamp) = %(today)s
            AND event_type = 'page_view'
        GROUP BY page_url, page_title
        ORDER BY views DESC
        LIMIT 10
        """
        
        popular_pages_result = self.ch_client.query(
            popular_pages_query,
            parameters={'today': today}
        )
        
        popular_pages = [
            {
                'page_url': row[0],
                'page_title': row[1],
                'views': row[2]
            }
            for row in popular_pages_result.result_rows
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'online_users': online_users,
            'today_stats': {
                'unique_users': today_data[0],
                'total_sessions': today_data[1],
                'total_events': today_data[2],
                'unique_pages': today_data[3]
            },
            'popular_pages': popular_pages
        }

class UserSegmentationService:
    """
    用户分群服务
    提供用户分群创建、管理和分析功能
    """
    
    def __init__(self, postgres_client, analytics_service):
        self.pg_client = postgres_client
        self.analytics = analytics_service
    
    async def create_segment(self, segment_data: Dict[str, Any]) -> str:
        """
        创建用户分群
        
        Args:
            segment_data: 分群数据
            
        Returns:
            str: 分群ID
        """
        segment_id = f"segment_{int(datetime.now().timestamp())}"
        
        # 插入分群定义
        insert_query = """
        INSERT INTO user_segments (
            segment_id, segment_name, description, criteria, created_by
        ) VALUES (%(segment_id)s, %(name)s, %(description)s, %(criteria)s, %(created_by)s)
        """
        
        await self.pg_client.execute(
            insert_query,
            {
                'segment_id': segment_id,
                'name': segment_data['name'],
                'description': segment_data.get('description', ''),
                'criteria': json.dumps(segment_data['criteria']),
                'created_by': segment_data.get('created_by', 'system')
            }
        )
        
        # 计算分群用户
        await self._calculate_segment_users(segment_id, segment_data['criteria'])
        
        return segment_id
    
    async def _calculate_segment_users(self, segment_id: str, criteria: Dict[str, Any]):
        """
        计算分群用户并更新关系表
        """
        # 使用分析服务获取符合条件的用户
        segment_result = await self.analytics.segment_users(criteria)
        
        if segment_result['user_count'] > 0:
            # 清除旧的分群关系
            delete_query = "DELETE FROM user_segment_members WHERE segment_id = %(segment_id)s"
            await self.pg_client.execute(delete_query, {'segment_id': segment_id})
            
            # 插入新的分群关系
            insert_members_query = """
            INSERT INTO user_segment_members (user_id, segment_id)
            VALUES (%(user_id)s, %(segment_id)s)
            """
            
            for user in segment_result['users']:
                await self.pg_client.execute(
                    insert_members_query,
                    {'user_id': user['user_id'], 'segment_id': segment_id}
                )
            
            # 更新分群用户数量
            update_count_query = """
            UPDATE user_segments 
            SET user_count = %(count)s, updated_at = CURRENT_TIMESTAMP
            WHERE segment_id = %(segment_id)s
            """
            
            await self.pg_client.execute(
                update_count_query,
                {'count': segment_result['user_count'], 'segment_id': segment_id}
            )

class PredictiveAnalytics:
    """
    预测分析服务
    提供用户流失预测、生命周期价值预测等功能
    """
    
    def __init__(self, clickhouse_client, postgres_client):
        self.ch_client = clickhouse_client
        self.pg_client = postgres_client
    
    async def predict_churn_risk(self, user_id: str) -> Dict[str, Any]:
        """
        预测用户流失风险
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 流失风险预测结果
        """
        # 获取用户历史行为特征
        features = await self._extract_churn_features(user_id)
        
        if not features:
            return {
                'user_id': user_id,
                'churn_risk': 'unknown',
                'risk_score': 0,
                'confidence': 0,
                'factors': []
            }
        
        # 计算流失风险评分
        risk_score = self._calculate_churn_score(features)
        
        # 确定风险等级
        if risk_score >= 0.7:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # 识别关键风险因素
        risk_factors = self._identify_risk_factors(features)
        
        return {
            'user_id': user_id,
            'churn_risk': risk_level,
            'risk_score': round(risk_score, 3),
            'confidence': 0.85,  # 模型置信度
            'factors': risk_factors,
            'recommendations': self._generate_retention_recommendations(risk_factors)
        }
    
    async def _extract_churn_features(self, user_id: str) -> Dict[str, float]:
        """
        提取用户流失预测特征
        """
        # 获取用户最近30天的行为数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # 查询用户活动特征
        activity_query = """
        SELECT 
            COUNT(DISTINCT DATE(timestamp)) as active_days,
            COUNT(DISTINCT session_id) as total_sessions,
            AVG(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) as page_view_ratio,
            COUNT(*) as total_events,
            MAX(timestamp) as last_activity
        FROM user_events 
        WHERE user_id = %(user_id)s 
            AND timestamp >= %(start_date)s 
            AND timestamp <= %(end_date)s
        """
        
        activity_result = self.ch_client.query(
            activity_query,
            parameters={
                'user_id': user_id,
                'start_date': start_date,
                'end_date': end_date
            }
        )
        
        if not activity_result.result_rows:
            return {}
        
        activity_data = activity_result.result_rows[0]
        
        # 计算特征
        days_since_last_activity = (end_date - activity_data[4]).days if activity_data[4] else 30
        
        features = {
            'active_days_ratio': activity_data[0] / 30,  # 活跃天数比例
            'avg_sessions_per_day': activity_data[1] / max(activity_data[0], 1),  # 日均会话数
            'page_view_ratio': activity_data[2],  # 页面浏览比例
            'total_events': activity_data[3],  # 总事件数
            'days_since_last_activity': days_since_last_activity,  # 距离最后活动天数
            'engagement_decline': self._calculate_engagement_decline(user_id)  # 参与度下降趋势
        }
        
        return features
    
    def _calculate_churn_score(self, features: Dict[str, float]) -> float:
        """
        计算流失风险评分
        """
        # 简化的流失风险评分模型
        score = 0
        
        # 活跃度因素
        if features['active_days_ratio'] < 0.1:
            score += 0.3
        elif features['active_days_ratio'] < 0.3:
            score += 0.2
        
        # 最后活动时间
        if features['days_since_last_activity'] > 14:
            score += 0.4
        elif features['days_since_last_activity'] > 7:
            score += 0.2
        
        # 参与度下降
        if features['engagement_decline'] > 0.5:
            score += 0.3
        elif features['engagement_decline'] > 0.2:
            score += 0.1
        
        return min(score, 1.0)
```

## API设计

### 1. 事件追踪
```http
POST /api/v1/analytics/events
```

**请求体:**
```json
{
  "event_type": "page_view",
  "event_name": "view_document",
  "user_id": "user_123",
  "session_id": "session_abc",
  "page_url": "/documents/123",
  "page_title": "历史文献详情",
  "properties": {
    "document_id": "doc_123",
    "category": "明清史料",
    "reading_time": 120
  },
  "timestamp": "2024-01-20T10:30:00Z"
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "event_id": "event_xyz789",
    "status": "recorded",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2. 用户行为摘要
```http
GET /api/v1/analytics/users/{user_id}/summary?days={days}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "user_id": "user_123",
    "analysis_period": {
      "start_date": "2023-12-21T00:00:00Z",
      "end_date": "2024-01-20T00:00:00Z",
      "days": 30
    },
    "activity_summary": {
      "total_sessions": 15,
      "avg_session_duration": 1200.5,
      "total_page_views": 85,
      "total_events": 156,
      "bounce_rate": 0.2,
      "activity_score": 78.5
    },
    "top_events": [
      {
        "event_type": "page_view",
        "event_name": "view_document",
        "page_url": "/documents",
        "event_count": 45
      }
    ],
    "preferences": {
      "top_pages": {
        "/documents": 45,
        "/search": 23
      },
      "top_features": {
        "document_search": 30,
        "document_view": 45
      },
      "engagement_level": "high"
    },
    "user_segment": "power_user"
  }
}
```

### 3. 队列分析
```http
POST /api/v1/analytics/cohort
```

**请求体:**
```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "period_type": "weekly"
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "cohort_matrix": {
      "2024-01-01": [100, 65, 45, 32],
      "2024-01-08": [120, 78, 52],
      "2024-01-15": [95, 61],
      "2024-01-22": [110]
    },
    "retention_rates": {
      "week_1": 0.65,
      "week_2": 0.45,
      "week_3": 0.32
    },
    "analysis_period": {
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-01-31T23:59:59Z"
    }
  }
}
```

### 4. 漏斗分析
```http
POST /api/v1/analytics/funnel
```

**请求体:**
```json
{
  "funnel_steps": [
    "page_visit",
    "search_document",
    "view_document",
    "download_document"
  ],
  "days": 30
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "funnel_steps": [
      {
        "step": "page_visit",
        "step_index": 0,
        "user_count": 1000,
        "conversion_rate": 0
      },
      {
        "step": "search_document",
        "step_index": 1,
        "user_count": 650,
        "conversion_rate": 65.0
      },
      {
        "step": "view_document",
        "step_index": 2,
        "user_count": 420,
        "conversion_rate": 42.0
      },
      {
        "step": "download_document",
        "step_index": 3,
        "user_count": 180,
        "conversion_rate": 18.0
      }
    ],
    "overall_conversion_rate": 18.0,
    "analysis_period": {
      "start_date": "2023-12-21T00:00:00Z",
      "end_date": "2024-01-20T00:00:00Z",
      "days": 30
    }
  }
}
```

### 5. 用户分群
```http
POST /api/v1/analytics/segments
```

**请求体:**
```json
{
  "name": "高价值用户",
  "description": "活跃度高且使用时间长的用户",
  "criteria": {
    "activity_level": {
      "min_sessions": 10
    },
    "engagement_score": {
      "min_duration": 300
    },
    "use_ml_clustering": true
  },
  "created_by": "admin"
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "segment_id": "segment_1705747200",
    "user_count": 245,
    "clusters": [
      {
        "cluster_id": 0,
        "user_count": 89,
        "characteristics": {
          "avg_sessions": 25.5,
          "avg_duration": 1800.2,
          "avg_lifetime_value": 150.0
        }
      }
    ]
  }
}
```

### 6. 实时统计
```http
GET /api/v1/analytics/realtime
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "timestamp": "2024-01-20T10:30:00Z",
    "online_users": 156,
    "today_stats": {
      "unique_users": 1250,
      "total_sessions": 2100,
      "total_events": 15600,
      "unique_pages": 45
    },
    "popular_pages": [
      {
        "page_url": "/documents",
        "page_title": "文档列表",
        "views": 890
      },
      {
        "page_url": "/search",
        "page_title": "搜索页面",
        "views": 567
      }
    ]
  }
}
```

### 7. 流失风险预测
```http
GET /api/v1/analytics/users/{user_id}/churn-risk
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "user_id": "user_123",
    "churn_risk": "medium",
    "risk_score": 0.65,
    "confidence": 0.85,
    "factors": [
      {
        "factor": "low_activity",
        "impact": 0.3,
        "description": "用户活跃度较低"
      },
      {
        "factor": "engagement_decline",
        "impact": 0.25,
        "description": "参与度呈下降趋势"
      }
    ],
    "recommendations": [
      "发送个性化内容推荐",
      "提供新功能引导",
      "发送重新激活邮件"
    ]
  }
}
```

## 前端组件设计

### 1. 用户行为分析仪表板组件

```vue
<template>
  <div class="analytics-dashboard">
    <!-- 顶部统计卡片 -->
    <el-row :gutter="20" class="stats-cards">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-value">{{ realtimeStats.online_users }}</div>
            <div class="stat-label">在线用户</div>
            <div class="stat-trend positive">
              <el-icon><ArrowUp /></el-icon>
              +12%
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-value">{{ realtimeStats.today_stats.unique_users }}</div>
            <div class="stat-label">今日访客</div>
            <div class="stat-trend positive">
              <el-icon><ArrowUp /></el-icon>
              +8%
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-value">{{ realtimeStats.today_stats.total_sessions }}</div>
            <div class="stat-label">今日会话</div>
            <div class="stat-trend negative">
              <el-icon><ArrowDown /></el-icon>
              -3%
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-value">{{ realtimeStats.today_stats.total_events }}</div>
            <div class="stat-label">今日事件</div>
            <div class="stat-trend positive">
              <el-icon><ArrowUp /></el-icon>
              +15%
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 图表区域 -->
    <el-row :gutter="20" class="charts-section">
      <!-- 访问趋势图 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>访问趋势</span>
              <el-select v-model="trendPeriod" size="small" style="width: 120px">
                <el-option label="今日" value="today" />
                <el-option label="7天" value="7days" />
                <el-option label="30天" value="30days" />
              </el-select>
            </div>
          </template>
          <div ref="trendChart" style="height: 300px;"></div>
        </el-card>
      </el-col>

      <!-- 用户分布图 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>用户分布</span>
          </template>
          <div ref="distributionChart" style="height: 300px;"></div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 热门页面和用户行为 -->
    <el-row :gutter="20" class="content-section">
      <!-- 热门页面 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>热门页面</span>
          </template>
          <el-table :data="realtimeStats.popular_pages" style="width: 100%">
            <el-table-column prop="page_title" label="页面标题" />
            <el-table-column prop="views" label="访问量" width="100">
              <template #default="{ row }">
                <el-tag type="primary">{{ row.views }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="趋势" width="80">
              <template #default>
                <div class="trend-indicator positive">
                  <el-icon><TrendCharts /></el-icon>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>

      <!-- 用户分群 -->
      <el-col :span="12">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>用户分群</span>
              <el-button type="primary" size="small" @click="showSegmentDialog = true">
                创建分群
              </el-button>
            </div>
          </template>
          <div class="segment-list">
            <div v-for="segment in userSegments" :key="segment.segment_id" class="segment-item">
              <div class="segment-info">
                <div class="segment-name">{{ segment.segment_name }}</div>
                <div class="segment-count">{{ segment.user_count }} 用户</div>
              </div>
              <div class="segment-actions">
                <el-button type="text" @click="viewSegmentDetails(segment)">
                  查看详情
                </el-button>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 分析工具 -->
    <el-row :gutter="20" class="analysis-tools">
      <el-col :span="8">
        <el-card class="tool-card" @click="openCohortAnalysis">
          <div class="tool-content">
            <el-icon class="tool-icon"><DataAnalysis /></el-icon>
            <div class="tool-title">队列分析</div>
            <div class="tool-description">分析用户留存情况</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="tool-card" @click="openFunnelAnalysis">
          <div class="tool-content">
            <el-icon class="tool-icon"><Funnel /></el-icon>
            <div class="tool-title">漏斗分析</div>
            <div class="tool-description">分析转化路径</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="tool-card" @click="openChurnPrediction">
          <div class="tool-content">
            <el-icon class="tool-icon"><Warning /></el-icon>
            <div class="tool-title">流失预测</div>
            <div class="tool-description">预测用户流失风险</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 创建分群对话框 -->
    <el-dialog v-model="showSegmentDialog" title="创建用户分群" width="600px">
      <el-form :model="segmentForm" label-width="120px">
        <el-form-item label="分群名称">
          <el-input v-model="segmentForm.name" placeholder="请输入分群名称" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="segmentForm.description" type="textarea" placeholder="请输入分群描述" />
        </el-form-item>
        <el-form-item label="分群条件">
          <div class="criteria-builder">
            <el-form-item label="最小会话数">
              <el-input-number v-model="segmentForm.criteria.activity_level.min_sessions" :min="1" />
            </el-form-item>
            <el-form-item label="最小停留时间">
              <el-input-number v-model="segmentForm.criteria.engagement_score.min_duration" :min="60" />
            </el-form-item>
            <el-form-item label="使用机器学习">
              <el-switch v-model="segmentForm.criteria.use_ml_clustering" />
            </el-form-item>
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showSegmentDialog = false">取消</el-button>
        <el-button type="primary" @click="createSegment" :loading="creating">
          创建分群
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import { analyticsApi } from '@/api/analytics'

// 响应式数据
const realtimeStats = ref({
  online_users: 0,
  today_stats: {
    unique_users: 0,
    total_sessions: 0,
    total_events: 0,
    unique_pages: 0
  },
  popular_pages: []
})

const userSegments = ref([])
const trendPeriod = ref('7days')
const showSegmentDialog = ref(false)
const creating = ref(false)

// 图表引用
const trendChart = ref()
const distributionChart = ref()

// 表单数据
const segmentForm = reactive({
  name: '',
  description: '',
  criteria: {
    activity_level: {
      min_sessions: 5
    },
    engagement_score: {
      min_duration: 300
    },
    use_ml_clustering: false
  }
})

/**
 * 加载实时统计数据
 */
const loadRealtimeStats = async () => {
  try {
    const response = await analyticsApi.getRealtimeStats()
    realtimeStats.value = response.data
  } catch (error) {
    console.error('加载实时统计失败:', error)
    ElMessage.error('加载实时统计数据失败')
  }
}

/**
 * 加载用户分群列表
 */
const loadUserSegments = async () => {
  try {
    const response = await analyticsApi.getUserSegments()
    userSegments.value = response.data
  } catch (error) {
    console.error('加载用户分群失败:', error)
  }
}

/**
 * 初始化访问趋势图表
 */
const initTrendChart = () => {
  const chart = echarts.init(trendChart.value)
  
  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: ['访问量', '用户数']
    },
    xAxis: {
      type: 'category',
      data: ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: '访问量',
        type: 'line',
        data: [820, 932, 901, 934, 1290, 1330, 1320],
        smooth: true
      },
      {
        name: '用户数',
        type: 'line',
        data: [620, 732, 701, 734, 1090, 1130, 1120],
        smooth: true
      }
    ]
  }
  
  chart.setOption(option)
}

/**
 * 初始化用户分布图表
 */
const initDistributionChart = () => {
  const chart = echarts.init(distributionChart.value)
  
  const option = {
    tooltip: {
      trigger: 'item'
    },
    legend: {
      orient: 'vertical',
      left: 'left'
    },
    series: [
      {
        name: '用户分布',
        type: 'pie',
        radius: '50%',
        data: [
          { value: 1048, name: '新用户' },
          { value: 735, name: '活跃用户' },
          { value: 580, name: '沉睡用户' },
          { value: 484, name: '流失用户' }
        ],
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  }
  
  chart.setOption(option)
}

/**
 * 创建用户分群
 */
const createSegment = async () => {
  if (!segmentForm.name.trim()) {
    ElMessage.warning('请输入分群名称')
    return
  }
  
  creating.value = true
  try {
    await analyticsApi.createSegment(segmentForm)
    ElMessage.success('分群创建成功')
    showSegmentDialog.value = false
    await loadUserSegments()
    
    // 重置表单
    Object.assign(segmentForm, {
      name: '',
      description: '',
      criteria: {
        activity_level: { min_sessions: 5 },
        engagement_score: { min_duration: 300 },
        use_ml_clustering: false
      }
    })
  } catch (error) {
    console.error('创建分群失败:', error)
    ElMessage.error('创建分群失败')
  } finally {
    creating.value = false
  }
}

/**
 * 查看分群详情
 */
const viewSegmentDetails = (segment: any) => {
  // 跳转到分群详情页面
  console.log('查看分群详情:', segment)
}

/**
 * 打开队列分析
 */
const openCohortAnalysis = () => {
  // 跳转到队列分析页面
  console.log('打开队列分析')
}

/**
 * 打开漏斗分析
 */
const openFunnelAnalysis = () => {
  // 跳转到漏斗分析页面
  console.log('打开漏斗分析')
}

/**
 * 打开流失预测
 */
const openChurnPrediction = () => {
  // 跳转到流失预测页面
  console.log('打开流失预测')
}

// 生命周期
onMounted(async () => {
  await loadRealtimeStats()
  await loadUserSegments()
  
  nextTick(() => {
    initTrendChart()
    initDistributionChart()
  })
  
  // 设置定时刷新
  setInterval(loadRealtimeStats, 30000) // 30秒刷新一次
})
</script>

<style scoped>
.analytics-dashboard {
  padding: 20px;
}

.stats-cards {
  margin-bottom: 20px;
}

.stat-card {
  text-align: center;
}

.stat-content {
  padding: 10px;
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #409eff;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.stat-trend {
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
}

.stat-trend.positive {
  color: #67c23a;
}

.stat-trend.negative {
  color: #f56c6c;
}

.charts-section {
  margin-bottom: 20px;
}

.content-section {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.trend-indicator {
  display: flex;
  justify-content: center;
}

.trend-indicator.positive {
  color: #67c23a;
}

.segment-list {
  max-height: 300px;
  overflow-y: auto;
}

.segment-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #ebeef5;
}

.segment-item:last-child {
  border-bottom: none;
}

.segment-name {
  font-weight: 500;
  margin-bottom: 4px;
}

.segment-count {
  font-size: 12px;
  color: #666;
}

.analysis-tools {
  margin-top: 20px;
}

.tool-card {
  cursor: pointer;
  transition: all 0.3s;
}

.tool-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.tool-content {
  text-align: center;
  padding: 20px;
}

.tool-icon {
  font-size: 32px;
  color: #409eff;
  margin-bottom: 12px;
}

.tool-title {
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 8px;
}

.tool-description {
  font-size: 12px;
  color: #666;
}

.criteria-builder {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 16px;
  background-color: #fafafa;
}
</style>
```

### 2. 漏斗分析组件

```vue
<template>
  <div class="funnel-analysis">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>漏斗分析</span>
          <el-button type="primary" @click="showConfigDialog = true">
            配置漏斗
          </el-button>
        </div>
      </template>
      
      <!-- 漏斗图表 -->
      <div ref="funnelChart" style="height: 400px;"></div>
      
      <!-- 漏斗数据表格 -->
      <el-table :data="funnelData.funnel_steps" style="width: 100%; margin-top: 20px;">
        <el-table-column prop="step" label="步骤" />
        <el-table-column prop="user_count" label="用户数" width="120">
          <template #default="{ row }">
            <el-tag type="primary">{{ row.user_count }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="conversion_rate" label="转化率" width="120">
          <template #default="{ row }">
            <span :class="getConversionRateClass(row.conversion_rate)">
              {{ row.conversion_rate }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column label="流失率" width="120">
          <template #default="{ row, $index }">
            <span v-if="$index > 0" class="churn-rate">
              {{ calculateChurnRate(row, $index) }}%
            </span>
            <span v-else>-</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
    
    <!-- 配置对话框 -->
    <el-dialog v-model="showConfigDialog" title="配置漏斗分析" width="600px">
      <el-form :model="funnelConfig" label-width="120px">
        <el-form-item label="分析周期">
          <el-select v-model="funnelConfig.days" style="width: 200px">
            <el-option label="7天" :value="7" />
            <el-option label="30天" :value="30" />
            <el-option label="90天" :value="90" />
          </el-select>
        </el-form-item>
        <el-form-item label="漏斗步骤">
          <div class="funnel-steps-config">
            <div v-for="(step, index) in funnelConfig.funnel_steps" :key="index" class="step-item">
              <el-input v-model="funnelConfig.funnel_steps[index]" placeholder="请输入步骤名称" />
              <el-button 
                type="danger" 
                size="small" 
                @click="removeStep(index)"
                :disabled="funnelConfig.funnel_steps.length <= 2"
              >
                删除
              </el-button>
            </div>
            <el-button type="primary" size="small" @click="addStep">
              添加步骤
            </el-button>
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showConfigDialog = false">取消</el-button>
        <el-button type="primary" @click="runFunnelAnalysis" :loading="analyzing">
          开始分析
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import { analyticsApi } from '@/api/analytics'

// 响应式数据
const funnelData = ref({
  funnel_steps: [],
  overall_conversion_rate: 0,
  analysis_period: {}
})

const showConfigDialog = ref(false)
const analyzing = ref(false)

// 图表引用
const funnelChart = ref()

// 配置数据
const funnelConfig = reactive({
  days: 30,
  funnel_steps: [
    'page_visit',
    'search_document',
    'view_document',
    'download_document'
  ]
})

/**
 * 运行漏斗分析
 */
const runFunnelAnalysis = async () => {
  if (funnelConfig.funnel_steps.length < 2) {
    ElMessage.warning('至少需要2个步骤')
    return
  }
  
  analyzing.value = true
  try {
    const response = await analyticsApi.performFunnelAnalysis(funnelConfig)
    funnelData.value = response.data
    showConfigDialog.value = false
    
    nextTick(() => {
      initFunnelChart()
    })
    
    ElMessage.success('漏斗分析完成')
  } catch (error) {
    console.error('漏斗分析失败:', error)
    ElMessage.error('漏斗分析失败')
  } finally {
    analyzing.value = false
  }
}

/**
 * 初始化漏斗图表
 */
const initFunnelChart = () => {
  const chart = echarts.init(funnelChart.value)
  
  const data = funnelData.value.funnel_steps.map(step => ({
    name: step.step,
    value: step.user_count
  }))
  
  const option = {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    series: [
      {
        name: '漏斗分析',
        type: 'funnel',
        left: '10%',
        top: 60,
        bottom: 60,
        width: '80%',
        min: 0,
        max: Math.max(...data.map(d => d.value)),
        minSize: '0%',
        maxSize: '100%',
        sort: 'descending',
        gap: 2,
        label: {
          show: true,
          position: 'inside'
        },
        labelLine: {
          length: 10,
          lineStyle: {
            width: 1,
            type: 'solid'
          }
        },
        itemStyle: {
          borderColor: '#fff',
          borderWidth: 1
        },
        emphasis: {
          label: {
            fontSize: 20
          }
        },
        data: data
      }
    ]
  }
  
  chart.setOption(option)
}

/**
 * 添加步骤
 */
const addStep = () => {
  funnelConfig.funnel_steps.push('')
}

/**
 * 删除步骤
 */
const removeStep = (index: number) => {
  funnelConfig.funnel_steps.splice(index, 1)
}

/**
 * 获取转化率样式类
 */
const getConversionRateClass = (rate: number) => {
  if (rate >= 50) return 'high-conversion'
  if (rate >= 20) return 'medium-conversion'
  return 'low-conversion'
}

/**
 * 计算流失率
 */
const calculateChurnRate = (row: any, index: number) => {
  if (index === 0) return 0
  const prevStep = funnelData.value.funnel_steps[index - 1]
  const churnRate = ((prevStep.user_count - row.user_count) / prevStep.user_count) * 100
  return Math.round(churnRate * 100) / 100
}

// 生命周期
onMounted(() => {
  // 默认运行一次分析
  runFunnelAnalysis()
})
</script>

<style scoped>
.funnel-analysis {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.funnel-steps-config {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 16px;
  background-color: #fafafa;
}

.step-item {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
  align-items: center;
}

.step-item:last-child {
  margin-bottom: 0;
}

.high-conversion {
  color: #67c23a;
  font-weight: bold;
}

.medium-conversion {
  color: #e6a23c;
  font-weight: bold;
}

.low-conversion {
  color: #f56c6c;
  font-weight: bold;
}

.churn-rate {
  color: #f56c6c;
}
</style>
```

### 3. 队列分析组件

```vue
<template>
  <div class="cohort-analysis">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>队列分析</span>
          <div class="header-controls">
            <el-date-picker
              v-model="dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              format="YYYY-MM-DD"
              value-format="YYYY-MM-DD"
              @change="runCohortAnalysis"
            />
            <el-select v-model="periodType" @change="runCohortAnalysis" style="width: 120px; margin-left: 10px;">
              <el-option label="按周" value="weekly" />
              <el-option label="按月" value="monthly" />
            </el-select>
          </div>
        </div>
      </template>
      
      <!-- 队列矩阵热力图 -->
      <div ref="cohortHeatmap" style="height: 400px;"></div>
      
      <!-- 留存率趋势图 -->
      <div ref="retentionTrend" style="height: 300px; margin-top: 20px;"></div>
      
      <!-- 队列数据表格 -->
      <el-table :data="cohortTableData" style="width: 100%; margin-top: 20px;">
        <el-table-column prop="cohort" label="队列" width="120" />
        <el-table-column 
          v-for="(period, index) in retentionPeriods" 
          :key="index"
          :prop="`period_${index}`"
          :label="period"
          width="80"
        >
          <template #default="{ row }">
            <span :class="getRetentionClass(row[`period_${index}`])">
              {{ row[`period_${index}`] ? row[`period_${index}`] + '%' : '-' }}
            </span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import { analyticsApi } from '@/api/analytics'

// 响应式数据
const cohortData = ref({
  cohort_matrix: {},
  retention_rates: {},
  analysis_period: {}
})

const dateRange = ref([new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), new Date()])
const periodType = ref('weekly')
const analyzing = ref(false)

// 图表引用
const cohortHeatmap = ref()
const retentionTrend = ref()

// 计算属性
const cohortTableData = computed(() => {
  const matrix = cohortData.value.cohort_matrix
  const data = []
  
  for (const [cohort, values] of Object.entries(matrix)) {
    const row = { cohort }
    values.forEach((value, index) => {
      row[`period_${index}`] = Math.round((value / values[0]) * 100)
    })
    data.push(row)
  }
  
  return data
})

const retentionPeriods = computed(() => {
  const maxPeriods = Math.max(
    ...Object.values(cohortData.value.cohort_matrix).map(arr => arr.length)
  )
  
  return Array.from({ length: maxPeriods }, (_, i) => {
    if (periodType.value === 'weekly') {
      return i === 0 ? '第0周' : `第${i}周`
    } else {
      return i === 0 ? '第0月' : `第${i}月`
    }
  })
})

/**
 * 运行队列分析
 */
const runCohortAnalysis = async () => {
  if (!dateRange.value || dateRange.value.length !== 2) {
    ElMessage.warning('请选择日期范围')
    return
  }
  
  analyzing.value = true
  try {
    const response = await analyticsApi.performCohortAnalysis({
      start_date: dateRange.value[0],
      end_date: dateRange.value[1],
      period_type: periodType.value
    })
    
    cohortData.value = response.data
    
    nextTick(() => {
      initCohortHeatmap()
      initRetentionTrend()
    })
    
    ElMessage.success('队列分析完成')
  } catch (error) {
    console.error('队列分析失败:', error)
    ElMessage.error('队列分析失败')
  } finally {
    analyzing.value = false
  }
}

/**
 * 初始化队列热力图
 */
const initCohortHeatmap = () => {
  const chart = echarts.init(cohortHeatmap.value)
  
  const matrix = cohortData.value.cohort_matrix
  const cohorts = Object.keys(matrix)
  const maxPeriods = Math.max(...Object.values(matrix).map(arr => arr.length))
  
  const data = []
  cohorts.forEach((cohort, cohortIndex) => {
    const values = matrix[cohort]
    values.forEach((value, periodIndex) => {
      const retentionRate = Math.round((value / values[0]) * 100)
      data.push([periodIndex, cohortIndex, retentionRate])
    })
  })
  
  const option = {
    tooltip: {
      position: 'top',
      formatter: function (params) {
        return `队列: ${cohorts[params.data[1]]}<br/>周期: ${params.data[0]}<br/>留存率: ${params.data[2]}%`
      }
    },
    grid: {
      height: '50%',
      top: '10%'
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: maxPeriods }, (_, i) => `周期${i}`),
      splitArea: {
        show: true
      }
    },
    yAxis: {
      type: 'category',
      data: cohorts,
      splitArea: {
        show: true
      }
    },
    visualMap: {
      min: 0,
      max: 100,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '15%',
      inRange: {
        color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffcc', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
      }
    },
    series: [
      {
        name: '留存率',
        type: 'heatmap',
        data: data,
        label: {
          show: true
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  }
  
  chart.setOption(option)
}

/**
 * 初始化留存率趋势图
 */
const initRetentionTrend = () => {
  const chart = echarts.init(retentionTrend.value)
  
  const retentionRates = cohortData.value.retention_rates
  const periods = Object.keys(retentionRates)
  const rates = Object.values(retentionRates).map(rate => Math.round(rate * 100))
  
  const option = {
    title: {
      text: '平均留存率趋势',
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      formatter: '{b}: {c}%'
    },
    xAxis: {
      type: 'category',
      data: periods
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series: [
      {
        name: '留存率',
        type: 'line',
        data: rates,
        smooth: true,
        areaStyle: {
          opacity: 0.3
        },
        lineStyle: {
          width: 3
        },
        itemStyle: {
          color: '#409eff'
        }
      }
    ]
  }
  
  chart.setOption(option)
}

/**
 * 获取留存率样式类
 */
const getRetentionClass = (rate: number) => {
  if (!rate) return ''
  if (rate >= 70) return 'high-retention'
  if (rate >= 40) return 'medium-retention'
  return 'low-retention'
}

// 生命周期
onMounted(() => {
  runCohortAnalysis()
})
</script>

<style scoped>
.cohort-analysis {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-controls {
  display: flex;
  align-items: center;
}

.high-retention {
  color: #67c23a;
  font-weight: bold;
}

.medium-retention {
  color: #e6a23c;
  font-weight: bold;
}

.low-retention {
  color: #f56c6c;
  font-weight: bold;
}
</style>
```

## 验收标准

### 功能性验收标准

1. **事件追踪功能**
   - ✅ 能够准确记录用户行为事件
   - ✅ 支持实时事件数据收集
   - ✅ 事件数据完整性验证
   - ✅ 支持自定义事件属性
   - ✅ 异常事件处理和容错

2. **用户行为分析**
   - ✅ 生成准确的用户行为摘要
   - ✅ 计算用户活跃度评分
   - ✅ 识别用户偏好特征
   - ✅ 用户分群功能正常
   - ✅ 支持多维度行为分析

3. **队列分析**
   - ✅ 准确计算用户留存率
   - ✅ 生成队列分析矩阵
   - ✅ 支持不同时间周期分析
   - ✅ 留存率趋势可视化
   - ✅ 队列对比分析功能

4. **漏斗分析**
   - ✅ 准确计算转化率
   - ✅ 识别转化瓶颈
   - ✅ 支持自定义漏斗步骤
   - ✅ 流失原因分析
   - ✅ 漏斗优化建议

5. **预测分析**
   - ✅ 用户流失风险预测
   - ✅ 风险因素识别
   - ✅ 预测准确率 ≥ 80%
   - ✅ 个性化留存建议
   - ✅ 预测模型持续优化

### 性能验收标准

1. **数据处理性能**
   - ✅ 事件记录响应时间 < 100ms
   - ✅ 实时统计更新延迟 < 5秒
   - ✅ 支持每秒1000+事件处理
   - ✅ 大数据量查询响应时间 < 3秒
   - ✅ 并发用户支持 ≥ 500

2. **存储性能**
   - ✅ ClickHouse写入性能 ≥ 10万条/秒
   - ✅ 数据压缩率 ≥ 70%
   - ✅ 查询性能优化
   - ✅ 数据分区策略有效
   - ✅ TTL策略正确执行

3. **缓存性能**
   - ✅ Redis缓存命中率 ≥ 90%
   - ✅ 缓存更新策略合理
   - ✅ 热点数据快速访问
   - ✅ 缓存失效处理正确
   - ✅ 内存使用优化

### 安全性验收标准

1. **数据安全**
   - ✅ 用户隐私数据脱敏
   - ✅ 敏感信息加密存储
   - ✅ 数据访问权限控制
   - ✅ 审计日志完整
   - ✅ 数据备份和恢复

2. **接口安全**
   - ✅ API身份认证
   - ✅ 请求频率限制
   - ✅ 输入数据验证
   - ✅ SQL注入防护
   - ✅ XSS攻击防护

## 业务价值

### 直接价值

1. **用户洞察**
   - 深入了解用户行为模式
   - 识别高价值用户群体
   - 发现产品使用痛点
   - 优化用户体验路径

2. **产品优化**
   - 基于数据驱动的产品决策
   - 功能使用情况分析
   - 页面性能优化指导
   - 用户界面改进建议

3. **运营效率**
   - 精准用户分群营销
   - 个性化内容推荐
   - 用户流失预警
   - 运营活动效果评估

### 间接价值

1. **商业智能**
   - 支持战略决策制定
   - 市场趋势分析
   - 竞争优势识别
   - 业务增长机会发现

2. **风险管控**
   - 异常行为检测
   - 安全风险预警
   - 系统性能监控
   - 业务连续性保障

## 依赖关系

### 技术依赖

1. **基础设施**
   - ClickHouse集群部署
   - PostgreSQL数据库
   - Redis缓存集群
   - Kafka消息队列

2. **开发框架**
   - FastAPI后端框架
   - Vue 3前端框架
   - ECharts图表库
   - Element Plus UI组件

3. **数据科学**
   - scikit-learn机器学习库
   - Pandas数据处理
   - NumPy数值计算
   - Prophet时间序列分析

### 业务依赖

1. **数据源**
   - 用户管理系统
   - 内容管理系统
   - 搜索服务
   - 推荐系统

2. **外部服务**
   - 地理位置服务
   - 邮件通知服务
   - 短信服务
   - 第三方分析工具

### 环境依赖

1. **开发环境**
   - Python 3.9+
   - Node.js 16+
   - Docker容器化
   - Kubernetes编排

2. **监控工具**
   - Prometheus监控
   - Grafana可视化
   - ELK日志分析
   - Jaeger链路追踪

## 风险评估

### 技术风险

1. **数据量风险** (中等)
   - **风险描述**: 大数据量可能影响查询性能
   - **影响程度**: 中等
   - **缓解措施**: 
     - 实施数据分区策略
     - 优化查询索引
     - 使用数据预聚合
     - 实施TTL策略

2. **实时性风险** (中等)
   - **风险描述**: 实时数据处理可能出现延迟
   - **影响程度**: 中等
   - **缓解措施**:
     - 使用流处理技术
     - 实施数据缓存策略
     - 优化消息队列配置
     - 监控处理延迟

3. **存储成本风险** (低)
   - **风险描述**: 长期数据存储成本增长
   - **影响程度**: 低
   - **缓解措施**:
     - 实施数据生命周期管理
     - 使用数据压缩技术
     - 冷热数据分离存储
     - 定期数据清理

### 业务风险

1. **隐私合规风险** (高)
   - **风险描述**: 用户数据收集可能涉及隐私问题
   - **影响程度**: 高
   - **缓解措施**:
     - 严格遵循GDPR等法规
     - 实施数据脱敏处理
     - 提供用户数据控制选项
     - 定期隐私合规审计

2. **数据准确性风险** (中等)
   - **风险描述**: 数据质量问题可能影响分析结果
   - **影响程度**: 中等
   - **缓解措施**:
     - 实施数据验证规则
     - 建立数据质量监控
     - 定期数据清洗
     - 异常数据告警

### 运营风险

1. **系统可用性风险** (中等)
   - **风险描述**: 系统故障可能影响数据收集
   - **影响程度**: 中等
   - **缓解措施**:
     - 实施高可用架构
     - 建立故障转移机制
     - 定期备份数据
     - 24/7监控告警

## 开发任务分解

### 后端开发任务 (预计16周)

#### 第一阶段：基础架构 (4周)
1. **数据模型设计** (1周)
   - ClickHouse表结构设计
   - PostgreSQL表结构设计
   - 数据关系建模
   - 索引策略设计

2. **核心服务开发** (2周)
   - UserBehaviorAnalytics服务
   - 事件追踪功能
   - 数据清洗和验证
   - 基础查询接口

3. **数据存储层** (1周)
   - ClickHouse连接和操作
   - PostgreSQL连接和操作
   - Redis缓存实现
   - 数据库迁移脚本

#### 第二阶段：分析功能 (6周)
1. **用户行为分析** (2周)
   - 用户行为摘要生成
   - 活跃度评分算法
   - 偏好特征识别
   - 用户画像构建

2. **队列分析** (2周)
   - 队列分析算法实现
   - 留存率计算
   - 队列矩阵生成
   - 趋势分析功能

3. **漏斗分析** (2周)
   - 漏斗分析算法
   - 转化率计算
   - 流失点识别
   - 优化建议生成

#### 第三阶段：高级功能 (4周)
1. **用户分群** (2周)
   - 分群算法实现
   - 机器学习聚类
   - 分群管理功能
   - 动态分群更新

2. **预测分析** (2周)
   - 流失预测模型
   - 特征工程
   - 模型训练和评估
   - 预测结果解释

#### 第四阶段：优化和部署 (2周)
1. **性能优化** (1周)
   - 查询性能优化
   - 缓存策略优化
   - 并发处理优化
   - 内存使用优化

2. **部署和监控** (1周)
   - Docker容器化
   - Kubernetes部署配置
   - 监控指标配置
   - 日志收集配置

### 前端开发任务 (预计12周)

#### 第一阶段：基础组件 (4周)
1. **项目初始化** (1周)
   - Vue 3项目搭建
   - TypeScript配置
   - 路由和状态管理
   - UI组件库集成

2. **API接口封装** (1周)
   - HTTP客户端配置
   - API接口定义
   - 错误处理机制
   - 请求拦截器

3. **基础组件开发** (2周)
   - 统计卡片组件
   - 图表基础组件
   - 表格组件
   - 对话框组件

#### 第二阶段：核心功能 (6周)
1. **仪表板开发** (2周)
   - 实时统计展示
   - 趋势图表
   - 用户分布图
   - 热门页面列表

2. **分析工具** (2周)
   - 漏斗分析组件
   - 队列分析组件
   - 用户分群组件
   - 预测分析组件

3. **数据可视化** (2周)
   - ECharts图表集成
   - 交互式图表
   - 图表配置功能
   - 图表导出功能

#### 第三阶段：用户体验 (2周)
1. **交互优化** (1周)
   - 响应式设计
   - 加载状态处理
   - 错误状态处理
   - 用户反馈机制

2. **性能优化** (1周)
   - 组件懒加载
   - 图表渲染优化
   - 内存泄漏防护
   - 打包优化

### 测试任务 (预计6周)

#### 单元测试 (2周)
- 后端服务单元测试
- 前端组件单元测试
- 工具函数测试
- 测试覆盖率 ≥ 80%

#### 集成测试 (2周)
- API接口测试
- 数据库集成测试
- 缓存集成测试
- 消息队列测试

#### 端到端测试 (2周)
- 用户行为模拟测试
- 分析功能测试
- 性能压力测试
- 兼容性测试

### 部署任务 (预计2周)

#### 环境准备 (1周)
- 生产环境配置
- 数据库部署
- 缓存集群部署
- 监控系统部署

#### 上线部署 (1周)
- 应用部署
- 数据迁移
- 功能验证
- 性能监控

## 总结

用户行为分析服务是历史文本项目的重要组成部分，通过全面的用户行为数据收集、分析和洞察，为产品优化和运营决策提供强有力的数据支持。该服务采用现代化的技术栈，具备高性能、高可用性和强扩展性的特点，能够满足大规模用户行为分析的需求。