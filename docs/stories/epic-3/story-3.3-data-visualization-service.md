# Story 3.3: 数据可视化分析服务

## 用户故事描述

**作为** 历史文献研究人员和数据分析师  
**我希望** 能够通过直观的图表和可视化界面分析历史文献数据  
**以便于** 发现数据中的模式、趋势和关联关系，支持研究决策和成果展示

## 核心技术栈

### 后端技术
- **Web框架**: FastAPI (Python)
- **数据处理**: Pandas, NumPy, SciPy
- **可视化引擎**: Matplotlib, Plotly, Seaborn
- **统计分析**: scikit-learn, statsmodels
- **图数据库**: Neo4j (知识图谱可视化)
- **时序数据库**: InfluxDB (时间序列分析)
- **数据库**: PostgreSQL, MongoDB
- **缓存**: Redis
- **消息队列**: RabbitMQ
- **任务调度**: Celery

### 前端技术
- **框架**: Vue 3 + TypeScript
- **可视化库**: D3.js, ECharts, Chart.js
- **图表组件**: Vue-ECharts, @vue/composition-api
- **3D可视化**: Three.js
- **地图可视化**: Leaflet, MapBox
- **UI组件**: Element Plus
- **状态管理**: Pinia

## 数据模型设计

### 可视化项目模型
```python
class VisualizationProject(BaseModel):
    """可视化项目模型"""
    id: str = Field(default_factory=lambda: f"viz_proj_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="项目名称")
    description: Optional[str] = Field(None, description="项目描述")
    user_id: str = Field(..., description="创建用户ID")
    data_sources: List[str] = Field(default_factory=list, description="数据源列表")
    charts: List[str] = Field(default_factory=list, description="图表ID列表")
    dashboards: List[str] = Field(default_factory=list, description="仪表板ID列表")
    settings: Dict[str, Any] = Field(default_factory=dict, description="项目设置")
    created_time: datetime = Field(default_factory=datetime.utcnow)
    updated_time: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="active", description="项目状态")
```

### 图表配置模型
```python
class ChartConfig(BaseModel):
    """图表配置模型"""
    id: str = Field(default_factory=lambda: f"chart_{uuid.uuid4().hex[:8]}")
    project_id: str = Field(..., description="所属项目ID")
    name: str = Field(..., description="图表名称")
    chart_type: str = Field(..., description="图表类型")
    data_source: str = Field(..., description="数据源")
    query_config: Dict[str, Any] = Field(..., description="查询配置")
    chart_options: Dict[str, Any] = Field(..., description="图表选项")
    filters: List[Dict[str, Any]] = Field(default_factory=list, description="过滤器")
    dimensions: List[str] = Field(default_factory=list, description="维度字段")
    metrics: List[str] = Field(default_factory=list, description="指标字段")
    created_time: datetime = Field(default_factory=datetime.utcnow)
    updated_time: datetime = Field(default_factory=datetime.utcnow)
```

### 仪表板模型
```python
class Dashboard(BaseModel):
    """仪表板模型"""
    id: str = Field(default_factory=lambda: f"dashboard_{uuid.uuid4().hex[:8]}")
    project_id: str = Field(..., description="所属项目ID")
    name: str = Field(..., description="仪表板名称")
    description: Optional[str] = Field(None, description="仪表板描述")
    layout: Dict[str, Any] = Field(..., description="布局配置")
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="图表配置")
    filters: List[Dict[str, Any]] = Field(default_factory=list, description="全局过滤器")
    refresh_interval: Optional[int] = Field(None, description="刷新间隔(秒)")
    is_public: bool = Field(default=False, description="是否公开")
    created_time: datetime = Field(default_factory=datetime.utcnow)
    updated_time: datetime = Field(default_factory=datetime.utcnow)
```

### 数据分析任务模型
```python
class AnalysisTask(BaseModel):
    """数据分析任务模型"""
    id: str = Field(default_factory=lambda: f"analysis_{uuid.uuid4().hex[:8]}")
    project_id: str = Field(..., description="所属项目ID")
    name: str = Field(..., description="任务名称")
    analysis_type: str = Field(..., description="分析类型")
    data_source: str = Field(..., description="数据源")
    parameters: Dict[str, Any] = Field(..., description="分析参数")
    status: str = Field(default="pending", description="任务状态")
    progress: float = Field(default=0.0, description="执行进度")
    result_data: Optional[Dict[str, Any]] = Field(None, description="分析结果")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_time: datetime = Field(default_factory=datetime.utcnow)
    started_time: Optional[datetime] = Field(None, description="开始时间")
    completed_time: Optional[datetime] = Field(None, description="完成时间")
```

## 服务架构设计

### 数据可视化服务核心类
```python
class DataVisualizationService:
    """数据可视化服务"""
    
    def __init__(self):
        self.db = get_database()
        self.redis = get_redis_client()
        self.chart_generator = ChartGenerator()
        self.data_processor = DataProcessor()
        self.analysis_engine = AnalysisEngine()
    
    async def create_project(self, project_data: dict) -> str:
        """创建可视化项目"""
        project = VisualizationProject(**project_data)
        
        # 保存项目信息
        await self.db.visualization_projects.insert_one(project.dict())
        
        # 初始化项目工作空间
        await self._initialize_project_workspace(project.id)
        
        return project.id
    
    async def create_chart(self, chart_data: dict) -> dict:
        """创建图表"""
        chart_config = ChartConfig(**chart_data)
        
        # 验证数据源
        await self._validate_data_source(chart_config.data_source)
        
        # 生成图表数据
        chart_result = await self.chart_generator.generate_chart(
            chart_config.chart_type,
            chart_config.data_source,
            chart_config.query_config,
            chart_config.chart_options
        )
        
        # 保存图表配置
        await self.db.chart_configs.insert_one(chart_config.dict())
        
        # 缓存图表数据
        cache_key = f"chart_data:{chart_config.id}"
        await self.redis.setex(
            cache_key, 
            3600,  # 1小时缓存
            json.dumps(chart_result)
        )
        
        return {
            "chart_id": chart_config.id,
            "chart_data": chart_result,
            "config": chart_config.dict()
        }
    
    async def create_dashboard(self, dashboard_data: dict) -> str:
        """创建仪表板"""
        dashboard = Dashboard(**dashboard_data)
        
        # 验证图表配置
        for chart_config in dashboard.charts:
            await self._validate_chart_config(chart_config)
        
        # 保存仪表板
        await self.db.dashboards.insert_one(dashboard.dict())
        
        return dashboard.id
    
    async def execute_analysis(self, analysis_data: dict) -> str:
        """执行数据分析"""
        task = AnalysisTask(**analysis_data)
        
        # 保存任务信息
        await self.db.analysis_tasks.insert_one(task.dict())
        
        # 异步执行分析任务
        await self._execute_analysis_async(task.id)
        
        return task.id
    
    async def get_chart_data(self, chart_id: str, filters: dict = None) -> dict:
        """获取图表数据"""
        # 尝试从缓存获取
        cache_key = f"chart_data:{chart_id}"
        if filters:
            cache_key += f":{hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()}"
        
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
        
        # 从数据库获取图表配置
        chart_config = await self.db.chart_configs.find_one({"id": chart_id})
        if not chart_config:
            raise ValueError(f"Chart {chart_id} not found")
        
        # 应用过滤器
        query_config = chart_config["query_config"].copy()
        if filters:
            query_config.update(filters)
        
        # 生成图表数据
        chart_data = await self.chart_generator.generate_chart(
            chart_config["chart_type"],
            chart_config["data_source"],
            query_config,
            chart_config["chart_options"]
        )
        
        # 缓存结果
        await self.redis.setex(cache_key, 1800, json.dumps(chart_data))  # 30分钟缓存
        
        return chart_data
    
    async def get_dashboard_data(self, dashboard_id: str, filters: dict = None) -> dict:
        """获取仪表板数据"""
        dashboard = await self.db.dashboards.find_one({"id": dashboard_id})
        if not dashboard:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard_data = {
            "dashboard": dashboard,
            "charts": []
        }
        
        # 并行获取所有图表数据
        chart_tasks = []
        for chart_config in dashboard["charts"]:
            chart_id = chart_config.get("chart_id")
            if chart_id:
                chart_filters = {**filters} if filters else {}
                chart_filters.update(chart_config.get("filters", {}))
                
                task = self.get_chart_data(chart_id, chart_filters)
                chart_tasks.append(task)
        
        chart_results = await asyncio.gather(*chart_tasks, return_exceptions=True)
        
        for i, result in enumerate(chart_results):
            if isinstance(result, Exception):
                dashboard_data["charts"].append({
                    "error": str(result),
                    "chart_config": dashboard["charts"][i]
                })
            else:
                dashboard_data["charts"].append({
                    "data": result,
                    "chart_config": dashboard["charts"][i]
                })
        
        return dashboard_data
    
    async def _initialize_project_workspace(self, project_id: str):
        """初始化项目工作空间"""
        # 创建项目目录结构
        workspace_path = f"/data/visualization/{project_id}"
        os.makedirs(workspace_path, exist_ok=True)
        os.makedirs(f"{workspace_path}/charts", exist_ok=True)
        os.makedirs(f"{workspace_path}/dashboards", exist_ok=True)
        os.makedirs(f"{workspace_path}/exports", exist_ok=True)
    
    async def _validate_data_source(self, data_source: str):
        """验证数据源"""
        # 检查数据源是否存在和可访问
        if data_source.startswith("db:"):
            # 数据库数据源
            table_name = data_source.replace("db:", "")
            result = await self.db[table_name].find_one()
            if not result:
                raise ValueError(f"Data source {data_source} is empty or not accessible")
        elif data_source.startswith("api:"):
            # API数据源
            # 验证API可访问性
            pass
        else:
            raise ValueError(f"Unsupported data source type: {data_source}")
    
    async def _validate_chart_config(self, chart_config: dict):
        """验证图表配置"""
        required_fields = ["chart_id", "position", "size"]
        for field in required_fields:
            if field not in chart_config:
                raise ValueError(f"Missing required field: {field}")
    
    async def _execute_analysis_async(self, task_id: str):
        """异步执行分析任务"""
        # 使用Celery异步执行
        from .tasks import execute_analysis_task
        execute_analysis_task.delay(task_id)
```

### 图表生成器
```python
class ChartGenerator:
    """图表生成器"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.chart_renderers = {
            "line": self._render_line_chart,
            "bar": self._render_bar_chart,
            "pie": self._render_pie_chart,
            "scatter": self._render_scatter_chart,
            "heatmap": self._render_heatmap,
            "treemap": self._render_treemap,
            "network": self._render_network_chart,
            "timeline": self._render_timeline_chart,
            "map": self._render_map_chart,
            "wordcloud": self._render_wordcloud
        }
    
    async def generate_chart(self, chart_type: str, data_source: str, 
                           query_config: dict, chart_options: dict) -> dict:
        """生成图表数据"""
        # 获取原始数据
        raw_data = await self.data_processor.get_data(data_source, query_config)
        
        # 数据预处理
        processed_data = await self.data_processor.process_data(
            raw_data, 
            query_config.get("transformations", [])
        )
        
        # 渲染图表
        renderer = self.chart_renderers.get(chart_type)
        if not renderer:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        chart_data = await renderer(processed_data, chart_options)
        
        return {
            "type": chart_type,
            "data": chart_data,
            "options": chart_options,
            "metadata": {
                "data_count": len(processed_data),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    async def _render_line_chart(self, data: List[dict], options: dict) -> dict:
        """渲染折线图"""
        x_field = options.get("x_field")
        y_fields = options.get("y_fields", [])
        
        series_data = []
        for y_field in y_fields:
            series = {
                "name": y_field,
                "data": []
            }
            
            for item in data:
                series["data"].append({
                    "x": item.get(x_field),
                    "y": item.get(y_field)
                })
            
            series_data.append(series)
        
        return {
            "xAxis": {
                "type": "category",
                "data": [item.get(x_field) for item in data]
            },
            "yAxis": {
                "type": "value"
            },
            "series": series_data
        }
    
    async def _render_bar_chart(self, data: List[dict], options: dict) -> dict:
        """渲染柱状图"""
        category_field = options.get("category_field")
        value_field = options.get("value_field")
        
        categories = [item.get(category_field) for item in data]
        values = [item.get(value_field) for item in data]
        
        return {
            "xAxis": {
                "type": "category",
                "data": categories
            },
            "yAxis": {
                "type": "value"
            },
            "series": [{
                "type": "bar",
                "data": values
            }]
        }
    
    async def _render_pie_chart(self, data: List[dict], options: dict) -> dict:
        """渲染饼图"""
        name_field = options.get("name_field")
        value_field = options.get("value_field")
        
        pie_data = []
        for item in data:
            pie_data.append({
                "name": item.get(name_field),
                "value": item.get(value_field)
            })
        
        return {
            "series": [{
                "type": "pie",
                "data": pie_data
            }]
        }
    
    async def _render_network_chart(self, data: List[dict], options: dict) -> dict:
        """渲染网络图"""
        nodes = []
        links = []
        
        # 处理节点数据
        node_field = options.get("node_field")
        for item in data:
            if node_field in item:
                nodes.append({
                    "id": item[node_field],
                    "name": item.get("name", item[node_field]),
                    "category": item.get("category", 0),
                    "value": item.get("value", 1)
                })
        
        # 处理连接数据
        if "relationships" in data[0]:
            for item in data:
                for rel in item.get("relationships", []):
                    links.append({
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                        "value": rel.get("weight", 1)
                    })
        
        return {
            "series": [{
                "type": "graph",
                "layout": "force",
                "data": nodes,
                "links": links,
                "roam": True,
                "force": {
                    "repulsion": 1000,
                    "edgeLength": 50
                }
            }]
        }
```

### 数据处理器
```python
class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.db = get_database()
        self.transformers = {
            "filter": self._apply_filter,
            "group": self._apply_grouping,
            "aggregate": self._apply_aggregation,
            "sort": self._apply_sorting,
            "limit": self._apply_limit,
            "join": self._apply_join
        }
    
    async def get_data(self, data_source: str, query_config: dict) -> List[dict]:
        """获取数据"""
        if data_source.startswith("db:"):
            return await self._get_database_data(data_source, query_config)
        elif data_source.startswith("api:"):
            return await self._get_api_data(data_source, query_config)
        elif data_source.startswith("file:"):
            return await self._get_file_data(data_source, query_config)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    async def process_data(self, data: List[dict], transformations: List[dict]) -> List[dict]:
        """处理数据"""
        result = data.copy()
        
        for transformation in transformations:
            transform_type = transformation.get("type")
            transformer = self.transformers.get(transform_type)
            
            if transformer:
                result = await transformer(result, transformation.get("params", {}))
            else:
                logger.warning(f"Unknown transformation type: {transform_type}")
        
        return result
    
    async def _get_database_data(self, data_source: str, query_config: dict) -> List[dict]:
        """从数据库获取数据"""
        collection_name = data_source.replace("db:", "")
        collection = self.db[collection_name]
        
        # 构建查询条件
        query = query_config.get("query", {})
        projection = query_config.get("projection")
        sort = query_config.get("sort")
        limit = query_config.get("limit")
        
        cursor = collection.find(query, projection)
        
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        
        return await cursor.to_list(length=None)
    
    async def _apply_filter(self, data: List[dict], params: dict) -> List[dict]:
        """应用过滤器"""
        conditions = params.get("conditions", [])
        
        filtered_data = []
        for item in data:
            match = True
            for condition in conditions:
                field = condition.get("field")
                operator = condition.get("operator")
                value = condition.get("value")
                
                if not self._evaluate_condition(item.get(field), operator, value):
                    match = False
                    break
            
            if match:
                filtered_data.append(item)
        
        return filtered_data
    
    async def _apply_grouping(self, data: List[dict], params: dict) -> List[dict]:
        """应用分组"""
        group_by = params.get("group_by", [])
        aggregations = params.get("aggregations", [])
        
        # 使用pandas进行分组操作
        df = pd.DataFrame(data)
        
        if group_by:
            grouped = df.groupby(group_by)
            
            result_data = []
            for name, group in grouped:
                group_result = {}
                
                # 设置分组字段值
                if isinstance(name, tuple):
                    for i, field in enumerate(group_by):
                        group_result[field] = name[i]
                else:
                    group_result[group_by[0]] = name
                
                # 应用聚合函数
                for agg in aggregations:
                    field = agg.get("field")
                    func = agg.get("function")
                    alias = agg.get("alias", f"{func}_{field}")
                    
                    if func == "count":
                        group_result[alias] = len(group)
                    elif func == "sum":
                        group_result[alias] = group[field].sum()
                    elif func == "avg":
                        group_result[alias] = group[field].mean()
                    elif func == "min":
                        group_result[alias] = group[field].min()
                    elif func == "max":
                        group_result[alias] = group[field].max()
                
                result_data.append(group_result)
            
            return result_data
        
        return data
    
    def _evaluate_condition(self, field_value, operator: str, condition_value) -> bool:
        """评估条件"""
        if operator == "eq":
            return field_value == condition_value
        elif operator == "ne":
            return field_value != condition_value
        elif operator == "gt":
            return field_value > condition_value
        elif operator == "gte":
            return field_value >= condition_value
        elif operator == "lt":
            return field_value < condition_value
        elif operator == "lte":
            return field_value <= condition_value
        elif operator == "in":
            return field_value in condition_value
        elif operator == "contains":
            return condition_value in str(field_value)
        else:
            return False
```

## API设计

### 可视化项目API

#### 1. 创建可视化项目
```http
POST /api/v1/visualization/projects
```

**请求体:**
```json
{
  "name": "历史文献分析项目",
  "description": "分析明清时期历史文献的时间分布和主题趋势",
  "data_sources": ["db:documents", "db:authors"],
  "settings": {
    "theme": "light",
    "auto_refresh": true,
    "refresh_interval": 300
  }
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "project_id": "viz_proj_abc123",
    "name": "历史文献分析项目",
    "created_time": "2024-01-20T10:30:00Z"
  }
}
```

#### 2. 创建图表
```http
POST /api/v1/visualization/charts
```

**请求体:**
```json
{
  "project_id": "viz_proj_abc123",
  "name": "文献年代分布",
  "chart_type": "line",
  "data_source": "db:documents",
  "query_config": {
    "query": {},
    "transformations": [
      {
        "type": "group",
        "params": {
          "group_by": ["year"],
          "aggregations": [
            {
              "field": "id",
              "function": "count",
              "alias": "document_count"
            }
          ]
        }
      },
      {
        "type": "sort",
        "params": {
          "field": "year",
          "order": "asc"
        }
      }
    ]
  },
  "chart_options": {
    "x_field": "year",
    "y_fields": ["document_count"],
    "title": "历史文献年代分布趋势",
    "x_axis_label": "年份",
    "y_axis_label": "文献数量"
  }
}
```

#### 3. 创建仪表板
```http
POST /api/v1/visualization/dashboards
```

**请求体:**
```json
{
  "project_id": "viz_proj_abc123",
  "name": "历史文献分析仪表板",
  "description": "综合展示历史文献的各项统计指标",
  "layout": {
    "type": "grid",
    "columns": 12,
    "row_height": 100
  },
  "charts": [
    {
      "chart_id": "chart_abc123",
      "position": {"x": 0, "y": 0},
      "size": {"w": 6, "h": 4},
      "filters": {}
    },
    {
      "chart_id": "chart_def456",
      "position": {"x": 6, "y": 0},
      "size": {"w": 6, "h": 4},
      "filters": {}
    }
  ],
  "filters": [
    {
      "field": "category",
      "type": "select",
      "options": ["历史研究", "文献学", "考古学"]
    }
  ],
  "refresh_interval": 300
}
```

#### 4. 获取图表数据
```http
GET /api/v1/visualization/charts/{chart_id}/data
```

**查询参数:**
```
?filters={"category": "历史研究"}&refresh=false
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "type": "line",
    "data": {
      "xAxis": {
        "type": "category",
        "data": ["1400", "1450", "1500", "1550", "1600"]
      },
      "yAxis": {
        "type": "value"
      },
      "series": [
        {
          "name": "document_count",
          "data": [12, 25, 45, 67, 89]
        }
      ]
    },
    "options": {
      "title": "历史文献年代分布趋势",
      "x_axis_label": "年份",
      "y_axis_label": "文献数量"
    },
    "metadata": {
      "data_count": 5,
      "generated_at": "2024-01-20T15:30:00Z"
    }
  }
}
```

#### 5. 执行数据分析
```http
POST /api/v1/visualization/analysis
```

**请求体:**
```json
{
  "project_id": "viz_proj_abc123",
  "name": "文献主题聚类分析",
  "analysis_type": "clustering",
  "data_source": "db:documents",
  "parameters": {
    "algorithm": "kmeans",
    "n_clusters": 5,
    "features": ["title", "content"],
    "preprocessing": {
      "text_vectorization": "tfidf",
      "max_features": 1000
    }
  }
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "analysis_xyz789",
    "status": "pending",
    "estimated_time": 300
  }
}
```

#### 6. 获取分析结果
```http
GET /api/v1/visualization/analysis/{task_id}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "analysis_xyz789",
    "status": "completed",
    "progress": 100,
    "result_data": {
      "clusters": [
        {
          "cluster_id": 0,
          "label": "政治史料",
          "documents": 45,
          "keywords": ["政治", "朝廷", "官员", "制度"]
        },
        {
          "cluster_id": 1,
          "label": "经济文献",
          "documents": 32,
          "keywords": ["经济", "贸易", "农业", "商业"]
        }
      ],
      "visualization_data": {
        "type": "scatter",
        "data": [
          {"x": 0.2, "y": 0.8, "cluster": 0, "document_id": "doc_1"},
          {"x": 0.3, "y": 0.7, "cluster": 0, "document_id": "doc_2"}
        ]
      }
    },
    "completed_time": "2024-01-20T15:35:00Z"
  }
}
```

## 前端集成

### Vue3 数据可视化组件

#### 图表编辑器组件
```vue
<template>
  <div class="chart-editor">
    <!-- 图表配置面板 -->
    <div class="config-panel">
      <el-card class="config-card">
        <template #header>
          <div class="card-header">
            <span>图表配置</span>
            <el-button type="primary" @click="generateChart" :loading="generating">
              生成图表
            </el-button>
          </div>
        </template>
        
        <el-form :model="chartConfig" label-width="120px">
          <el-form-item label="图表名称">
            <el-input v-model="chartConfig.name" placeholder="请输入图表名称"></el-input>
          </el-form-item>
          
          <el-form-item label="图表类型">
            <el-select v-model="chartConfig.chart_type" @change="onChartTypeChange">
              <el-option label="折线图" value="line"></el-option>
              <el-option label="柱状图" value="bar"></el-option>
              <el-option label="饼图" value="pie"></el-option>
              <el-option label="散点图" value="scatter"></el-option>
              <el-option label="热力图" value="heatmap"></el-option>
              <el-option label="网络图" value="network"></el-option>
              <el-option label="时间轴" value="timeline"></el-option>
              <el-option label="地图" value="map"></el-option>
              <el-option label="词云" value="wordcloud"></el-option>
            </el-select>
          </el-form-item>
          
          <el-form-item label="数据源">
            <el-select v-model="chartConfig.data_source" @change="onDataSourceChange">
              <el-option 
                v-for="source in dataSources" 
                :key="source.id" 
                :label="source.name" 
                :value="source.id"
              ></el-option>
            </el-select>
          </el-form-item>
          
          <!-- 动态字段配置 -->
          <div v-if="chartConfig.chart_type === 'line' || chartConfig.chart_type === 'bar'">
            <el-form-item label="X轴字段">
              <el-select v-model="chartConfig.chart_options.x_field">
                <el-option 
                  v-for="field in availableFields" 
                  :key="field" 
                  :label="field" 
                  :value="field"
                ></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="Y轴字段">
              <el-select v-model="chartConfig.chart_options.y_fields" multiple>
                <el-option 
                  v-for="field in availableFields" 
                  :key="field" 
                  :label="field" 
                  :value="field"
                ></el-option>
              </el-select>
            </el-form-item>
          </div>
          
          <div v-if="chartConfig.chart_type === 'pie'">
            <el-form-item label="名称字段">
              <el-select v-model="chartConfig.chart_options.name_field">
                <el-option 
                  v-for="field in availableFields" 
                  :key="field" 
                  :label="field" 
                  :value="field"
                ></el-option>
              </el-select>
            </el-form-item>
            
            <el-form-item label="数值字段">
              <el-select v-model="chartConfig.chart_options.value_field">
                <el-option 
                  v-for="field in availableFields" 
                  :key="field" 
                  :label="field" 
                  :value="field"
                ></el-option>
              </el-select>
            </el-form-item>
          </div>
          
          <!-- 数据转换配置 -->
          <el-form-item label="数据转换">
            <div class="transformations">
              <div 
                v-for="(transform, index) in chartConfig.query_config.transformations" 
                :key="index"
                class="transformation-item"
              >
                <el-select v-model="transform.type" style="width: 120px;">
                  <el-option label="过滤" value="filter"></el-option>
                  <el-option label="分组" value="group"></el-option>
                  <el-option label="聚合" value="aggregate"></el-option>
                  <el-option label="排序" value="sort"></el-option>
                  <el-option label="限制" value="limit"></el-option>
                </el-select>
                
                <!-- 分组配置 -->
                <div v-if="transform.type === 'group'" class="transform-config">
                  <el-select 
                    v-model="transform.params.group_by" 
                    multiple 
                    placeholder="选择分组字段"
                    style="width: 200px;"
                  >
                    <el-option 
                      v-for="field in availableFields" 
                      :key="field" 
                      :label="field" 
                      :value="field"
                    ></el-option>
                  </el-select>
                  
                  <div class="aggregations">
                    <div 
                      v-for="(agg, aggIndex) in transform.params.aggregations" 
                      :key="aggIndex"
                      class="aggregation-item"
                    >
                      <el-select v-model="agg.field" placeholder="字段" style="width: 100px;">
                        <el-option 
                          v-for="field in availableFields" 
                          :key="field" 
                          :label="field" 
                          :value="field"
                        ></el-option>
                      </el-select>
                      
                      <el-select v-model="agg.function" placeholder="函数" style="width: 80px;">
                        <el-option label="计数" value="count"></el-option>
                        <el-option label="求和" value="sum"></el-option>
                        <el-option label="平均" value="avg"></el-option>
                        <el-option label="最小" value="min"></el-option>
                        <el-option label="最大" value="max"></el-option>
                      </el-select>
                      
                      <el-input 
                        v-model="agg.alias" 
                        placeholder="别名" 
                        style="width: 100px;"
                      ></el-input>
                      
                      <el-button 
                        type="danger" 
                        icon="el-icon-delete" 
                        size="small" 
                        @click="removeAggregation(transform.params.aggregations, aggIndex)"
                      ></el-button>
                    </div>
                    
                    <el-button 
                      type="primary" 
                      icon="el-icon-plus" 
                      size="small" 
                      @click="addAggregation(transform.params.aggregations)"
                    >
                      添加聚合
                    </el-button>
                  </div>
                </div>
                
                <el-button 
                  type="danger" 
                  icon="el-icon-delete" 
                  size="small" 
                  @click="removeTransformation(index)"
                ></el-button>
              </div>
              
              <el-button 
                type="primary" 
                icon="el-icon-plus" 
                size="small" 
                @click="addTransformation"
              >
                添加转换
              </el-button>
            </div>
          </el-form-item>
          
          <!-- 图表样式配置 -->
          <el-form-item label="图表标题">
            <el-input v-model="chartConfig.chart_options.title"></el-input>
          </el-form-item>
          
          <el-form-item label="主题">
            <el-select v-model="chartConfig.chart_options.theme">
              <el-option label="默认" value="default"></el-option>
              <el-option label="深色" value="dark"></el-option>
              <el-option label="浅色" value="light"></el-option>
            </el-select>
          </el-form-item>
        </el-form>
      </el-card>
    </div>
    
    <!-- 图表预览区域 -->
    <div class="preview-panel">
      <el-card class="preview-card">
        <template #header>
          <div class="card-header">
            <span>图表预览</span>
            <div class="preview-actions">
              <el-button 
                type="text" 
                icon="el-icon-refresh" 
                @click="refreshChart"
                :loading="refreshing"
              >
                刷新
              </el-button>
              <el-button 
                type="text" 
                icon="el-icon-download" 
                @click="exportChart"
              >
                导出
              </el-button>
            </div>
          </div>
        </template>
        
        <div class="chart-container" ref="chartContainer">
          <div v-if="!chartData" class="empty-chart">
            <el-empty description="请配置图表参数并生成图表"></el-empty>
          </div>
          
          <div v-else class="chart-wrapper">
            <!-- ECharts图表 -->
            <div 
              v-if="isEChartsType(chartConfig.chart_type)"
              ref="echartsContainer"
              class="echarts-chart"
            ></div>
            
            <!-- D3.js图表 -->
            <div 
              v-else-if="isD3Type(chartConfig.chart_type)"
              ref="d3Container"
              class="d3-chart"
            ></div>
            
            <!-- 其他类型图表 -->
            <div v-else class="custom-chart">
              <component 
                :is="getChartComponent(chartConfig.chart_type)"
                :data="chartData"
                :options="chartConfig.chart_options"
              ></component>
            </div>
          </div>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, nextTick, watch } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import * as d3 from 'd3'
import { useVisualizationStore } from '@/stores/visualization'

// Props
interface Props {
  projectId: string
  chartId?: string
}

const props = defineProps<Props>()

// Stores
const visualizationStore = useVisualizationStore()

// Reactive data
const generating = ref(false)
const refreshing = ref(false)
const chartData = ref(null)
const dataSources = ref([])
const availableFields = ref([])
const echartsInstance = ref(null)

const chartConfig = reactive({
  name: '',
  chart_type: 'line',
  data_source: '',
  query_config: {
    query: {},
    transformations: []
  },
  chart_options: {
    title: '',
    theme: 'default',
    x_field: '',
    y_fields: [],
    name_field: '',
    value_field: ''
  }
})

// Refs
const chartContainer = ref(null)
const echartsContainer = ref(null)
const d3Container = ref(null)

// Methods
const loadDataSources = async () => {
  try {
    const sources = await visualizationStore.getDataSources(props.projectId)
    dataSources.value = sources
  } catch (error) {
    console.error('加载数据源失败:', error)
    ElMessage.error('加载数据源失败')
  }
}

const loadAvailableFields = async (dataSource: string) => {
  try {
    const fields = await visualizationStore.getDataSourceFields(dataSource)
    availableFields.value = fields
  } catch (error) {
    console.error('加载字段失败:', error)
    ElMessage.error('加载字段失败')
  }
}

const generateChart = async () => {
  if (!chartConfig.data_source) {
    ElMessage.warning('请选择数据源')
    return
  }
  
  try {
    generating.value = true
    
    const result = await visualizationStore.createChart({
      project_id: props.projectId,
      ...chartConfig
    })
    
    chartData.value = result.chart_data
    
    await nextTick()
    renderChart()
    
    ElMessage.success('图表生成成功')
    
  } catch (error) {
    console.error('生成图表失败:', error)
    ElMessage.error('生成图表失败')
  } finally {
    generating.value = false
  }
}

const renderChart = () => {
  if (!chartData.value) return
  
  if (isEChartsType(chartConfig.chart_type)) {
    renderEChartsChart()
  } else if (isD3Type(chartConfig.chart_type)) {
    renderD3Chart()
  }
}

const renderEChartsChart = () => {
  if (!echartsContainer.value) return
  
  // 销毁现有实例
  if (echartsInstance.value) {
    echartsInstance.value.dispose()
  }
  
  // 创建新实例
  echartsInstance.value = echarts.init(echartsContainer.value)
  
  // 设置配置
  const option = {
    title: {
      text: chartConfig.chart_options.title
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: chartData.value.data.series?.map(s => s.name) || []
    },
    ...chartData.value.data
  }
  
  echartsInstance.value.setOption(option)
  
  // 响应式调整
  window.addEventListener('resize', () => {
    echartsInstance.value?.resize()
  })
}

const renderD3Chart = () => {
  if (!d3Container.value) return
  
  // 清空容器
  d3.select(d3Container.value).selectAll('*').remove()
  
  if (chartConfig.chart_type === 'network') {
    renderNetworkChart()
  }
}

const renderNetworkChart = () => {
  const container = d3.select(d3Container.value)
  const width = 800
  const height = 600
  
  const svg = container
    .append('svg')
    .attr('width', width)
    .attr('height', height)
  
  const data = chartData.value.data.series[0]
  const nodes = data.data
  const links = data.links
  
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2))
  
  const link = svg.append('g')
    .selectAll('line')
    .data(links)
    .enter().append('line')
    .attr('stroke', '#999')
    .attr('stroke-opacity', 0.6)
    .attr('stroke-width', d => Math.sqrt(d.value))
  
  const node = svg.append('g')
    .selectAll('circle')
    .data(nodes)
    .enter().append('circle')
    .attr('r', d => Math.sqrt(d.value) * 3)
    .attr('fill', '#69b3a2')
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended))
  
  node.append('title')
    .text(d => d.name)
  
  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)
    
    node
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
  })
  
  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart()
    d.fx = d.x
    d.fy = d.y
  }
  
  function dragged(event, d) {
    d.fx = event.x
    d.fy = event.y
  }
  
  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0)
    d.fx = null
    d.fy = null
  }
}

const refreshChart = async () => {
  if (!props.chartId) {
    await generateChart()
    return
  }
  
  try {
    refreshing.value = true
    
    const result = await visualizationStore.getChartData(props.chartId, { refresh: true })
    chartData.value = result
    
    await nextTick()
    renderChart()
    
  } catch (error) {
    console.error('刷新图表失败:', error)
    ElMessage.error('刷新图表失败')
  } finally {
    refreshing.value = false
  }
}

const exportChart = () => {
  if (echartsInstance.value) {
    const url = echartsInstance.value.getDataURL({
      type: 'png',
      pixelRatio: 2,
      backgroundColor: '#fff'
    })
    
    const link = document.createElement('a')
    link.download = `${chartConfig.name || 'chart'}.png`
    link.href = url
    link.click()
  }
}

const onChartTypeChange = () => {
  // 重置图表选项
  chartConfig.chart_options = {
    title: chartConfig.chart_options.title,
    theme: chartConfig.chart_options.theme
  }
}

const onDataSourceChange = () => {
  if (chartConfig.data_source) {
    loadAvailableFields(chartConfig.data_source)
  }
}

const addTransformation = () => {
  chartConfig.query_config.transformations.push({
    type: 'filter',
    params: {}
  })
}

const removeTransformation = (index: number) => {
  chartConfig.query_config.transformations.splice(index, 1)
}

const addAggregation = (aggregations: any[]) => {
  aggregations.push({
    field: '',
    function: 'count',
    alias: ''
  })
}

const removeAggregation = (aggregations: any[], index: number) => {
  aggregations.splice(index, 1)
}

const isEChartsType = (type: string) => {
  return ['line', 'bar', 'pie', 'scatter', 'heatmap'].includes(type)
}

const isD3Type = (type: string) => {
  return ['network', 'timeline'].includes(type)
}

const getChartComponent = (type: string) => {
  // 返回自定义组件
  const components = {
    'wordcloud': 'WordCloudChart',
    'map': 'MapChart'
  }
  return components[type] || 'div'
}

// Watchers
watch(() => chartConfig.chart_type, () => {
  chartData.value = null
})

// Lifecycle
onMounted(() => {
  loadDataSources()
  
  if (props.chartId) {
    // 加载现有图表配置
    // loadChartConfig(props.chartId)
  }
})
</script>

<style scoped>
.chart-editor {
  display: flex;
  height: 100vh;
  gap: 20px;
  padding: 20px;
}

.config-panel {
  width: 400px;
  flex-shrink: 0;
}

.preview-panel {
  flex: 1;
  min-width: 0;
}

.config-card,
.preview-card {
  height: 100%;
  border-radius: 8px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.preview-actions {
  display: flex;
  gap: 10px;
}

.chart-container {
  height: calc(100% - 60px);
  min-height: 400px;
}

.empty-chart {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.chart-wrapper {
  height: 100%;
}

.echarts-chart,
.d3-chart {
  width: 100%;
  height: 100%;
}

.transformations {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 10px;
}

.transformation-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.transform-config {
  flex: 1;
  margin-left: 10px;
}

.aggregations {
  margin-top: 10px;
}

.aggregation-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
}
</style>
```

#### 仪表板组件
```vue
<template>
  <div class="dashboard-container">
    <!-- 仪表板头部 -->
    <div class="dashboard-header">
      <div class="header-left">
        <h2>{{ dashboard?.name || '仪表板' }}</h2>
        <p class="dashboard-description">{{ dashboard?.description }}</p>
      </div>
      
      <div class="header-actions">
        <!-- 全局过滤器 -->
        <div class="global-filters">
          <el-select 
            v-for="filter in dashboard?.filters || []"
            :key="filter.field"
            v-model="globalFilters[filter.field]"
            :placeholder="`选择${filter.field}`"
            @change="applyGlobalFilters"
            style="width: 150px; margin-right: 10px;"
          >
            <el-option 
              v-for="option in filter.options" 
              :key="option" 
              :label="option" 
              :value="option"
            ></el-option>
          </el-select>
        </div>
        
        <div class="action-buttons">
          <el-button 
            type="text" 
            icon="el-icon-refresh" 
            @click="refreshDashboard"
            :loading="refreshing"
          >
            刷新
          </el-button>
          
          <el-button 
            type="text" 
            icon="el-icon-setting" 
            @click="showSettings = true"
          >
            设置
          </el-button>
          
          <el-button 
            type="text" 
            icon="el-icon-share" 
            @click="shareDashboard"
          >
            分享
          </el-button>
        </div>
      </div>
    </div>
    
    <!-- 仪表板内容 -->
    <div class="dashboard-content" v-loading="loading">
      <grid-layout
        :layout="layout"
        :col-num="12"
        :row-height="100"
        :is-draggable="editMode"
        :is-resizable="editMode"
        :vertical-compact="true"
        :margin="[10, 10]"
        @layout-updated="onLayoutUpdated"
      >
        <grid-item
          v-for="item in layout"
          :key="item.i"
          :x="item.x"
          :y="item.y"
          :w="item.w"
          :h="item.h"
          :i="item.i"
        >
          <div class="chart-item">
            <div class="chart-header">
              <span class="chart-title">{{ getChartTitle(item.i) }}</span>
              <div class="chart-actions" v-if="editMode">
                <el-button 
                  type="text" 
                  icon="el-icon-edit" 
                  size="small"
                  @click="editChart(item.i)"
                ></el-button>
                <el-button 
                  type="text" 
                  icon="el-icon-delete" 
                  size="small"
                  @click="removeChart(item.i)"
                ></el-button>
              </div>
            </div>
            
            <div class="chart-content">
              <div v-if="chartErrors[item.i]" class="chart-error">
                <el-alert 
                  :title="chartErrors[item.i]" 
                  type="error" 
                  :closable="false"
                ></el-alert>
              </div>
              
              <div v-else-if="!chartData[item.i]" class="chart-loading">
                <el-skeleton :rows="3" animated></el-skeleton>
              </div>
              
              <div v-else class="chart-wrapper">
                <component 
                  :is="getChartComponent(item.i)"
                  :data="chartData[item.i]"
                  :options="getChartOptions(item.i)"
                  @chart-click="onChartClick"
                ></component>
              </div>
            </div>
          </div>
        </grid-item>
      </grid-layout>
    </div>
    
    <!-- 设置对话框 -->
    <el-dialog 
      v-model="showSettings" 
      title="仪表板设置" 
      width="600px"
    >
      <el-form :model="dashboardSettings" label-width="120px">
        <el-form-item label="自动刷新">
          <el-switch v-model="dashboardSettings.auto_refresh"></el-switch>
        </el-form-item>
        
        <el-form-item label="刷新间隔" v-if="dashboardSettings.auto_refresh">
          <el-select v-model="dashboardSettings.refresh_interval">
            <el-option label="30秒" :value="30"></el-option>
            <el-option label="1分钟" :value="60"></el-option>
            <el-option label="5分钟" :value="300"></el-option>
            <el-option label="10分钟" :value="600"></el-option>
          </el-select>
        </el-form-item>
        
        <el-form-item label="编辑模式">
          <el-switch v-model="editMode"></el-switch>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showSettings = false">取消</el-button>
        <el-button type="primary" @click="saveSettings">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue'
import { GridLayout, GridItem } from 'vue-grid-layout'
import { ElMessage } from 'element-plus'
import { useVisualizationStore } from '@/stores/visualization'

// Props
interface Props {
  dashboardId: string
}

const props = defineProps<Props>()

// Stores
const visualizationStore = useVisualizationStore()

// Reactive data
const loading = ref(false)
const refreshing = ref(false)
const showSettings = ref(false)
const editMode = ref(false)
const dashboard = ref(null)
const layout = ref([])
const chartData = ref({})
const chartErrors = ref({})
const globalFilters = reactive({})
const refreshTimer = ref(null)

const dashboardSettings = reactive({
  auto_refresh: false,
  refresh_interval: 300
})

// Methods
const loadDashboard = async () => {
  try {
    loading.value = true
    
    const result = await visualizationStore.getDashboardData(
      props.dashboardId, 
      globalFilters
    )
    
    dashboard.value = result.dashboard
    
    // 设置布局
    layout.value = dashboard.value.charts.map((chart, index) => ({
      i: chart.chart_id,
      x: chart.position.x,
      y: chart.position.y,
      w: chart.size.w,
      h: chart.size.h
    }))
    
    // 设置图表数据
    result.charts.forEach((chartResult, index) => {
      const chartId = dashboard.value.charts[index].chart_id
      
      if (chartResult.error) {
        chartErrors.value[chartId] = chartResult.error
      } else {
        chartData.value[chartId] = chartResult.data
      }
    })
    
    // 设置仪表板配置
    if (dashboard.value.refresh_interval) {
      dashboardSettings.auto_refresh = true
      dashboardSettings.refresh_interval = dashboard.value.refresh_interval
      startAutoRefresh()
    }
    
  } catch (error) {
    console.error('加载仪表板失败:', error)
    ElMessage.error('加载仪表板失败')
  } finally {
    loading.value = false
  }
}

const refreshDashboard = async () => {
  try {
    refreshing.value = true
    await loadDashboard()
    ElMessage.success('仪表板已刷新')
  } catch (error) {
    ElMessage.error('刷新失败')
  } finally {
    refreshing.value = false
  }
}

const applyGlobalFilters = async () => {
  await loadDashboard()
}

const startAutoRefresh = () => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
  }
  
  if (dashboardSettings.auto_refresh) {
    refreshTimer.value = setInterval(() => {
      refreshDashboard()
    }, dashboardSettings.refresh_interval * 1000)
  }
}

const stopAutoRefresh = () => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
    refreshTimer.value = null
  }
}

const saveSettings = () => {
  startAutoRefresh()
  showSettings.value = false
  ElMessage.success('设置已保存')
}

const shareDashboard = () => {
  const url = `${window.location.origin}/dashboard/${props.dashboardId}`
  navigator.clipboard.writeText(url)
  ElMessage.success('分享链接已复制到剪贴板')
}

const getChartTitle = (chartId: string) => {
  const chart = dashboard.value?.charts.find(c => c.chart_id === chartId)
  return chart?.name || '图表'
}

const getChartComponent = (chartId: string) => {
  const chartType = chartData.value[chartId]?.type
  const components = {
    'line': 'LineChart',
    'bar': 'BarChart',
    'pie': 'PieChart',
    'scatter': 'ScatterChart',
    'heatmap': 'HeatmapChart',
    'network': 'NetworkChart',
    'timeline': 'TimelineChart',
    'map': 'MapChart',
    'wordcloud': 'WordCloudChart'
  }
  return components[chartType] || 'div'
}

const getChartOptions = (chartId: string) => {
  const chart = dashboard.value?.charts.find(c => c.chart_id === chartId)
  return chart?.chart_options || {}
}

const onLayoutUpdated = (newLayout) => {
  layout.value = newLayout
  // 保存布局更改
  saveDashboardLayout()
}

const saveDashboardLayout = async () => {
  try {
    const updatedCharts = dashboard.value.charts.map(chart => {
      const layoutItem = layout.value.find(item => item.i === chart.chart_id)
      return {
        ...chart,
        position: { x: layoutItem.x, y: layoutItem.y },
        size: { w: layoutItem.w, h: layoutItem.h }
      }
    })
    
    await visualizationStore.updateDashboard(props.dashboardId, {
      charts: updatedCharts
    })
    
  } catch (error) {
    console.error('保存布局失败:', error)
  }
}

const editChart = (chartId: string) => {
  // 跳转到图表编辑页面
  // router.push(`/chart/edit/${chartId}`)
}

const removeChart = async (chartId: string) => {
  try {
    const updatedCharts = dashboard.value.charts.filter(c => c.chart_id !== chartId)
    
    await visualizationStore.updateDashboard(props.dashboardId, {
      charts: updatedCharts
    })
    
    // 更新本地数据
    dashboard.value.charts = updatedCharts
    layout.value = layout.value.filter(item => item.i !== chartId)
    delete chartData.value[chartId]
    delete chartErrors.value[chartId]
    
    ElMessage.success('图表已删除')
    
  } catch (error) {
    console.error('删除图表失败:', error)
    ElMessage.error('删除图表失败')
  }
}

const onChartClick = (event) => {
  // 处理图表点击事件
  console.log('Chart clicked:', event)
}

// Lifecycle
onMounted(() => {
  loadDashboard()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>

<style scoped>
.dashboard-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e4e7ed;
  background: #fff;
}

.header-left h2 {
  margin: 0 0 5px 0;
  color: #303133;
}

.dashboard-description {
  margin: 0;
  color: #909399;
  font-size: 14px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 20px;
}

.global-filters {
  display: flex;
  align-items: center;
}

.action-buttons {
  display: flex;
  gap: 10px;
}

.dashboard-content {
  flex: 1;
  padding: 20px;
  background: #f5f7fa;
  overflow: auto;
}

.chart-item {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px 10px;
  border-bottom: 1px solid #e4e7ed;
}

.chart-title {
  font-weight: 600;
  color: #303133;
}

.chart-actions {
  display: flex;
  gap: 5px;
}

.chart-content {
  flex: 1;
  padding: 20px;
  min-height: 0;
}

.chart-wrapper {
  height: 100%;
}

.chart-error,
.chart-loading {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
```

## 验收标准

### 功能性验收标准

1. **项目管理功能**
   - ✅ 能够创建、编辑、删除可视化项目
   - ✅ 支持项目权限管理和共享
   - ✅ 提供项目模板和快速创建功能

2. **图表创建功能**
   - ✅ 支持10种以上图表类型（折线图、柱状图、饼图、散点图、热力图、网络图、时间轴、地图、词云等）
   - ✅ 提供可视化的图表配置界面
   - ✅ 支持数据源选择和字段映射
   - ✅ 支持数据转换和聚合操作

3. **仪表板功能**
   - ✅ 支持拖拽式布局设计
   - ✅ 支持图表组合和联动
   - ✅ 提供全局过滤器功能
   - ✅ 支持自动刷新和实时更新

4. **数据分析功能**
   - ✅ 支持统计分析（描述性统计、相关性分析）
   - ✅ 支持机器学习分析（聚类、分类、回归）
   - ✅ 支持时间序列分析
   - ✅ 支持文本分析和主题建模

5. **数据处理功能**
   - ✅ 支持多种数据源（数据库、API、文件）
   - ✅ 支持数据清洗和预处理
   - ✅ 支持数据转换和聚合
   - ✅ 支持实时数据流处理

### 性能验收标准

1. **响应时间**
   - 图表生成时间 < 5秒（1万条数据）
   - 仪表板加载时间 < 10秒（包含5个图表）
   - 数据查询响应时间 < 3秒
   - 实时数据更新延迟 < 2秒

2. **并发性能**
   - 支持100个并发用户同时访问
   - 支持50个并发图表生成请求
   - 系统资源使用率 < 80%

3. **数据处理能力**
   - 支持单次处理100万条数据记录
   - 支持10GB以上数据文件导入
   - 内存使用优化，避免OOM错误

### 安全性验收标准

1. **访问控制**
   - ✅ 用户身份认证和授权
   - ✅ 项目级别的权限控制
   - ✅ 数据源访问权限验证

2. **数据安全**
   - ✅ 敏感数据脱敏处理
   - ✅ 数据传输加密（HTTPS）
   - ✅ 数据存储加密

3. **系统安全**
   - ✅ SQL注入防护
   - ✅ XSS攻击防护
   - ✅ CSRF攻击防护

## 业务价值

### 直接价值
1. **提升研究效率**: 通过可视化分析，研究人员能够快速发现数据中的模式和趋势
2. **降低技术门槛**: 提供无代码的可视化工具，让非技术人员也能进行数据分析
3. **增强决策支持**: 通过直观的图表和仪表板，为研究决策提供数据支撑
4. **促进协作交流**: 支持仪表板分享和协作，促进团队间的知识共享

### 间接价值
1. **数据驱动文化**: 培养基于数据的研究和决策习惯
2. **知识发现**: 通过数据挖掘和可视化，发现隐藏的研究价值
3. **成果展示**: 为学术报告和论文提供高质量的可视化图表
4. **教学支持**: 为历史文献教学提供直观的数据展示工具

## 依赖关系

### 技术依赖
- **数据存储**: 依赖PostgreSQL、MongoDB等数据库服务
- **搜索引擎**: 依赖Elasticsearch进行数据检索
- **缓存服务**: 依赖Redis进行数据缓存
- **消息队列**: 依赖RabbitMQ进行异步任务处理
- **文件存储**: 依赖MinIO进行图表和报告存储

### 业务依赖
- **用户管理**: 依赖用户认证和权限管理系统
- **数据源**: 依赖历史文献数据、OCR结果、NLP分析结果等
- **API网关**: 依赖API网关进行请求路由和限流

### 外部依赖
- **可视化库**: ECharts、D3.js等前端可视化库
- **机器学习**: scikit-learn、TensorFlow等ML库
- **地图服务**: 地理可视化需要地图API支持

## 风险评估

### 技术风险
1. **性能风险**: 大数据量可视化可能导致性能问题
   - **缓解措施**: 数据分页、缓存优化、异步加载

2. **兼容性风险**: 不同浏览器的可视化效果差异
   - **缓解措施**: 使用成熟的可视化库，进行跨浏览器测试

3. **复杂性风险**: 可视化配置过于复杂，用户学习成本高
   - **缓解措施**: 提供模板和向导，简化配置流程

### 业务风险
1. **数据质量风险**: 源数据质量问题影响可视化效果
   - **缓解措施**: 数据质量检查和清洗机制

2. **用户接受度风险**: 用户可能不习惯使用可视化工具
   - **缓解措施**: 提供培训和文档，逐步推广使用

### 运维风险
1. **资源消耗风险**: 复杂图表生成消耗大量计算资源
   - **缓解措施**: 资源监控和自动扩缩容

2. **数据安全风险**: 可视化结果可能泄露敏感信息
   - **缓解措施**: 数据脱敏和访问控制

## 开发任务分解

### 后端开发任务

#### Phase 1: 核心服务开发 (2周)
1. **数据可视化服务基础架构**
   - 设计服务架构和数据模型
   - 实现项目管理功能
   - 实现数据源管理功能

2. **图表生成引擎**
   - 实现基础图表类型（折线图、柱状图、饼图）
   - 实现数据处理和转换功能
   - 实现图表配置管理

#### Phase 2: 高级功能开发 (3周)
1. **高级图表类型**
   - 实现散点图、热力图、网络图
   - 实现地图可视化和词云
   - 实现时间轴和3D可视化

2. **数据分析引擎**
   - 实现统计分析功能
   - 实现机器学习分析
   - 实现时间序列分析

#### Phase 3: 仪表板和优化 (2周)
1. **仪表板功能**
   - 实现仪表板创建和管理
   - 实现图表联动和过滤
   - 实现自动刷新和实时更新

2. **性能优化**
   - 实现数据缓存机制
   - 优化查询性能
   - 实现异步任务处理

### 前端开发任务

#### Phase 1: 基础组件开发 (2周)
1. **图表编辑器**
   - 实现图表配置界面
   - 实现数据源选择和字段映射
   - 实现图表预览功能

2. **基础图表组件**
   - 实现ECharts图表组件
   - 实现图表交互功能
   - 实现图表导出功能

#### Phase 2: 高级组件开发 (3周)
1. **仪表板设计器**
   - 实现拖拽式布局编辑
   - 实现图表组合和排列
   - 实现响应式布局

2. **高级图表组件**
   - 实现D3.js网络图组件
   - 实现地图可视化组件
   - 实现3D可视化组件

#### Phase 3: 用户体验优化 (1周)
1. **交互优化**
   - 实现图表联动和钻取
   - 实现全局过滤器
   - 实现实时数据更新

2. **界面优化**
   - 优化移动端适配
   - 实现主题切换
   - 优化加载性能

### 测试任务 (1周)
1. **单元测试**: 核心算法和组件测试
2. **集成测试**: API接口和数据流测试
3. **性能测试**: 大数据量和并发测试
4. **用户测试**: 可用性和用户体验测试

### 部署任务 (0.5周)
1. **环境配置**: 生产环境部署配置
2. **监控配置**: 性能监控和日志配置
3. **文档编写**: API文档和用户手册
4. **培训准备**: 用户培训材料准备