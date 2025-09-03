# 用户故事 4.5: 数据分析与报告服务

## 基本信息

- **Epic**: Epic 4 - 智能运营与分析系统
- **故事编号**: Story 4.5
- **故事标题**: 数据分析与报告服务
- **优先级**: 高
- **估算点数**: 21
- **负责团队**: 数据分析团队 + 前端团队

## 用户故事描述

**作为** 内容运营人员和管理者  
**我希望** 能够查看详细的数据分析报告和可视化图表  
**以便于** 了解内容发布效果、用户互动情况、平台表现等关键指标，为运营决策提供数据支持

## 详细需求

### 核心功能

1. **数据收集与处理**
   - 自动收集各平台发布数据
   - 实时同步用户互动数据
   - 数据清洗和标准化处理
   - 异常数据检测和处理

2. **多维度数据分析**
   - 内容表现分析（阅读量、点赞、评论、转发）
   - 平台对比分析
   - 时间趋势分析
   - 用户行为分析
   - 内容类型效果分析

3. **可视化报告**
   - 实时数据仪表板
   - 自定义报告生成
   - 图表类型多样化（折线图、柱状图、饼图、热力图等）
   - 报告导出功能（PDF、Excel）

4. **智能洞察**
   - 趋势预测分析
   - 异常检测和告警
   - 优化建议生成
   - 竞品对比分析

## 技术栈

### 后端技术栈

- **Web框架**: FastAPI
- **数据处理**: Pandas, NumPy, Apache Spark
- **机器学习**: scikit-learn, TensorFlow, PyTorch
- **时间序列分析**: Prophet, statsmodels
- **数据库**: 
  - PostgreSQL (关系型数据)
  - InfluxDB (时间序列数据)
  - ClickHouse (OLAP分析)
  - Redis (缓存)
- **消息队列**: Apache Kafka, RabbitMQ
- **任务调度**: Celery, APScheduler
- **监控**: Prometheus, Grafana
- **日志**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **容器化**: Docker, Kubernetes

### 前端技术栈

- **框架**: Vue 3 + TypeScript
- **状态管理**: Pinia
- **路由**: Vue Router 4
- **UI组件库**: Element Plus
- **组合式API**: @vue/composition-api
- **HTTP客户端**: Axios
- **日期处理**: Day.js
- **工具库**: Lodash
- **图表库**: 
  - ECharts (主要图表库)
  - Chart.js (轻量级图表)
  - D3.js (自定义可视化)
  - Vue-ECharts (Vue集成)
- **构建工具**: Vite
- **代码规范**: ESLint, Prettier
- **测试**: Vitest

## 数据模型设计

### PostgreSQL 数据模型

```sql
-- 分析任务表
CREATE TABLE analysis_tasks (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    task_type VARCHAR(50) NOT NULL, -- 'content_analysis', 'platform_comparison', 'trend_analysis'
    config JSONB NOT NULL, -- 分析配置
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    result_data JSONB, -- 分析结果
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 报告模板表
CREATE TABLE report_templates (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    template_type VARCHAR(50) NOT NULL, -- 'dashboard', 'periodic_report', 'custom_report'
    config JSONB NOT NULL, -- 模板配置
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 生成的报告表
CREATE TABLE generated_reports (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    template_id INTEGER REFERENCES report_templates(id),
    name VARCHAR(200) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    time_range JSONB NOT NULL, -- 时间范围
    filters JSONB, -- 筛选条件
    data JSONB NOT NULL, -- 报告数据
    file_path VARCHAR(500), -- 导出文件路径
    status VARCHAR(20) DEFAULT 'generating', -- 'generating', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 数据源配置表
CREATE TABLE data_sources (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR(200) NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- 'platform_api', 'database', 'file_upload'
    config JSONB NOT NULL, -- 数据源配置
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'inactive', 'error'
    last_sync_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 告警规则表
CREATE TABLE alert_rules (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    metric_type VARCHAR(100) NOT NULL, -- 监控指标类型
    condition_config JSONB NOT NULL, -- 告警条件配置
    notification_config JSONB NOT NULL, -- 通知配置
    is_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 告警记录表
CREATE TABLE alert_records (
    id SERIAL PRIMARY KEY,
    rule_id INTEGER REFERENCES alert_rules(id),
    metric_value DECIMAL(15,4),
    threshold_value DECIMAL(15,4),
    alert_level VARCHAR(20) NOT NULL, -- 'info', 'warning', 'critical'
    message TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);
```

### InfluxDB 时间序列数据模型

```sql
-- 内容指标数据
-- measurement: content_metrics
-- tags: platform, content_type, user_id, content_id
-- fields: views, likes, comments, shares, engagement_rate
-- time: timestamp

-- 平台指标数据
-- measurement: platform_metrics
-- tags: platform, user_id
-- fields: total_posts, total_views, total_engagement, follower_count
-- time: timestamp

-- 用户行为数据
-- measurement: user_behavior
-- tags: user_id, action_type, platform
-- fields: action_count, session_duration
-- time: timestamp
```

### Redis 缓存结构

```python
# 实时数据缓存
REAL_TIME_METRICS = "analytics:realtime:{user_id}:{metric_type}"

# 报告缓存
REPORT_CACHE = "analytics:report:{report_id}"

# 分析结果缓存
ANALYSIS_CACHE = "analytics:analysis:{task_id}"

# 用户仪表板配置缓存
DASHBOARD_CONFIG = "analytics:dashboard:{user_id}"

# 热门内容缓存
TRENDING_CONTENT = "analytics:trending:{platform}:{time_range}"
```

## 服务架构设计

### 核心服务类

```python
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import asyncio

class AnalyticsService:
    """
    数据分析服务核心类
    负责数据收集、处理、分析和报告生成
    """
    
    def __init__(self, db_manager, cache_manager, ml_engine):
        self.db = db_manager
        self.cache = cache_manager
        self.ml_engine = ml_engine
        self.data_processor = DataProcessor()
        self.report_generator = ReportGenerator()
    
    async def collect_platform_data(self, user_id: int, platforms: List[str], 
                                  start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        收集指定平台的数据
        
        Args:
            user_id: 用户ID
            platforms: 平台列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            收集到的数据字典
        """
        collected_data = {}
        
        for platform in platforms:
            try:
                # 从各平台API收集数据
                platform_data = await self._fetch_platform_data(
                    platform, user_id, start_date, end_date
                )
                
                # 数据清洗和标准化
                cleaned_data = self.data_processor.clean_data(platform_data)
                
                # 存储到时间序列数据库
                await self._store_time_series_data(platform, cleaned_data)
                
                collected_data[platform] = cleaned_data
                
            except Exception as e:
                print(f"Failed to collect data from {platform}: {e}")
                collected_data[platform] = None
        
        return collected_data
    
    async def analyze_content_performance(self, user_id: int, 
                                        content_ids: List[int] = None,
                                        time_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """
        分析内容表现
        
        Args:
            user_id: 用户ID
            content_ids: 内容ID列表，为空则分析所有内容
            time_range: 时间范围
            
        Returns:
            内容表现分析结果
        """
        # 获取内容数据
        content_data = await self._get_content_metrics(
            user_id, content_ids, time_range
        )
        
        if content_data.empty:
            return {"error": "No data available for analysis"}
        
        # 计算基础指标
        basic_metrics = self._calculate_basic_metrics(content_data)
        
        # 趋势分析
        trend_analysis = self._analyze_trends(content_data)
        
        # 内容分类分析
        category_analysis = self._analyze_by_category(content_data)
        
        # 平台对比分析
        platform_comparison = self._compare_platforms(content_data)
        
        # 异常检测
        anomalies = self._detect_anomalies(content_data)
        
        return {
            "basic_metrics": basic_metrics,
            "trend_analysis": trend_analysis,
            "category_analysis": category_analysis,
            "platform_comparison": platform_comparison,
            "anomalies": anomalies,
            "generated_at": datetime.now().isoformat()
        }
    
    async def generate_insights(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成智能洞察
        
        Args:
            analysis_data: 分析数据
            
        Returns:
            洞察列表
        """
        insights = []
        
        # 趋势洞察
        trend_insights = self._generate_trend_insights(analysis_data.get("trend_analysis", {}))
        insights.extend(trend_insights)
        
        # 异常洞察
        anomaly_insights = self._generate_anomaly_insights(analysis_data.get("anomalies", []))
        insights.extend(anomaly_insights)
        
        # 优化建议
        optimization_suggestions = self._generate_optimization_suggestions(analysis_data)
        insights.extend(optimization_suggestions)
        
        # 按重要性排序
        insights.sort(key=lambda x: x.get("importance", 0), reverse=True)
        
        return insights
    
    async def create_custom_report(self, user_id: int, template_id: int, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建自定义报告
        
        Args:
            user_id: 用户ID
            template_id: 模板ID
            config: 报告配置
            
        Returns:
            报告生成结果
        """
        # 获取模板配置
        template = await self.db.get_report_template(template_id)
        if not template or template["user_id"] != user_id:
            raise ValueError("Template not found or access denied")
        
        # 合并配置
        merged_config = {**template["config"], **config}
        
        # 收集数据
        data = await self._collect_report_data(user_id, merged_config)
        
        # 生成报告
        report = await self.report_generator.generate_report(
            template["template_type"], data, merged_config
        )
        
        # 保存报告
        report_id = await self._save_generated_report(
            user_id, template_id, report, merged_config
        )
        
        return {
            "report_id": report_id,
            "report": report,
            "status": "completed"
        }
    
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算基础指标
        """
        return {
            "total_views": int(data["views"].sum()),
            "total_likes": int(data["likes"].sum()),
            "total_comments": int(data["comments"].sum()),
            "total_shares": int(data["shares"].sum()),
            "avg_engagement_rate": float(data["engagement_rate"].mean()),
            "content_count": len(data),
            "top_performing_content": data.nlargest(5, "engagement_rate")[["content_id", "title", "engagement_rate"]].to_dict("records")
        }
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        趋势分析
        """
        # 按日期分组计算趋势
        daily_metrics = data.groupby(data["publish_date"].dt.date).agg({
            "views": "sum",
            "likes": "sum",
            "comments": "sum",
            "shares": "sum",
            "engagement_rate": "mean"
        }).reset_index()
        
        # 计算增长率
        for metric in ["views", "likes", "comments", "shares"]:
            daily_metrics[f"{metric}_growth"] = daily_metrics[metric].pct_change() * 100
        
        return {
            "daily_trends": daily_metrics.to_dict("records"),
            "overall_growth": {
                "views": float(daily_metrics["views_growth"].mean()),
                "likes": float(daily_metrics["likes_growth"].mean()),
                "comments": float(daily_metrics["comments_growth"].mean()),
                "shares": float(daily_metrics["shares_growth"].mean())
            }
        }
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        异常检测
        """
        anomalies = []
        
        for metric in ["views", "likes", "comments", "engagement_rate"]:
            # 使用IQR方法检测异常值
            Q1 = data[metric].quantile(0.25)
            Q3 = data[metric].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 检测异常高值
            high_anomalies = data[data[metric] > upper_bound]
            for _, row in high_anomalies.iterrows():
                anomalies.append({
                    "type": "high_value",
                    "metric": metric,
                    "content_id": row["content_id"],
                    "value": float(row[metric]),
                    "threshold": float(upper_bound),
                    "severity": "high" if row[metric] > upper_bound * 1.5 else "medium"
                })
            
            # 检测异常低值
            low_anomalies = data[data[metric] < lower_bound]
            for _, row in low_anomalies.iterrows():
                anomalies.append({
                    "type": "low_value",
                    "metric": metric,
                    "content_id": row["content_id"],
                    "value": float(row[metric]),
                    "threshold": float(lower_bound),
                    "severity": "high" if row[metric] < lower_bound * 0.5 else "medium"
                })
        
        return anomalies

class DataProcessor:
    """
    数据处理器
    负责数据清洗、转换和标准化
    """
    
    def clean_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        数据清洗
        
        Args:
            raw_data: 原始数据
            
        Returns:
            清洗后的数据
        """
        cleaned_data = raw_data.copy()
        
        # 移除空值和异常值
        if "metrics" in cleaned_data:
            metrics = cleaned_data["metrics"]
            for metric in metrics:
                if isinstance(metric.get("value"), (int, float)):
                    # 移除负值（除了增长率）
                    if metric["name"] not in ["growth_rate", "change_rate"] and metric["value"] < 0:
                        metric["value"] = 0
                    
                    # 移除异常大的值
                    if metric["value"] > 1e10:
                        metric["value"] = None
        
        # 标准化时间格式
        if "timestamp" in cleaned_data:
            try:
                cleaned_data["timestamp"] = pd.to_datetime(cleaned_data["timestamp"]).isoformat()
            except:
                cleaned_data["timestamp"] = datetime.now().isoformat()
        
        return cleaned_data
    
    def normalize_metrics(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        指标标准化
        
        Args:
            data: 数据DataFrame
            metrics: 需要标准化的指标列表
            
        Returns:
            标准化后的数据
        """
        normalized_data = data.copy()
        scaler = StandardScaler()
        
        for metric in metrics:
            if metric in data.columns:
                normalized_data[f"{metric}_normalized"] = scaler.fit_transform(
                    data[[metric]]
                ).flatten()
        
        return normalized_data

class ReportGenerator:
    """
    报告生成器
    负责生成各种类型的报告
    """
    
    async def generate_report(self, report_type: str, data: Dict[str, Any], 
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成报告
        
        Args:
            report_type: 报告类型
            data: 报告数据
            config: 报告配置
            
        Returns:
            生成的报告
        """
        if report_type == "dashboard":
            return await self._generate_dashboard_report(data, config)
        elif report_type == "periodic_report":
            return await self._generate_periodic_report(data, config)
        elif report_type == "custom_report":
            return await self._generate_custom_report(data, config)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
    
    async def _generate_dashboard_report(self, data: Dict[str, Any], 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成仪表板报告
        """
        return {
            "type": "dashboard",
            "title": config.get("title", "数据仪表板"),
            "sections": [
                {
                    "id": "overview",
                    "title": "概览",
                    "widgets": [
                        {
                            "type": "metric_card",
                            "title": "总浏览量",
                            "value": data.get("total_views", 0),
                            "trend": data.get("views_trend", 0)
                        },
                        {
                            "type": "metric_card",
                            "title": "总互动量",
                            "value": data.get("total_engagement", 0),
                            "trend": data.get("engagement_trend", 0)
                        }
                    ]
                },
                {
                    "id": "charts",
                    "title": "图表分析",
                    "widgets": [
                        {
                            "type": "line_chart",
                            "title": "趋势分析",
                            "data": data.get("trend_data", [])
                        },
                        {
                            "type": "bar_chart",
                            "title": "平台对比",
                            "data": data.get("platform_data", [])
                        }
                    ]
                }
            ],
            "generated_at": datetime.now().isoformat()
        }
    
    async def _generate_periodic_report(self, data: Dict[str, Any], 
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成周期性报告
        """
        period = config.get("period", "weekly")
        
        return {
            "type": "periodic_report",
            "period": period,
            "title": f"{period.title()} 运营报告",
            "summary": {
                "key_metrics": data.get("key_metrics", {}),
                "highlights": data.get("highlights", []),
                "insights": data.get("insights", [])
            },
            "detailed_analysis": {
                "content_performance": data.get("content_analysis", {}),
                "platform_comparison": data.get("platform_comparison", {}),
                "audience_insights": data.get("audience_insights", {})
            },
            "recommendations": data.get("recommendations", []),
            "generated_at": datetime.now().isoformat()
        }
```

## API设计

### 数据分析API

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

@router.post("/analyze/content")
async def analyze_content_performance(
    request: ContentAnalysisRequest,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    分析内容表现
    """
    try:
        result = await analytics_service.analyze_content_performance(
            user_id=current_user.id,
            content_ids=request.content_ids,
            time_range=request.time_range
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/platform")
async def analyze_platform_performance(
    request: PlatformAnalysisRequest,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    分析平台表现
    """
    try:
        result = await analytics_service.analyze_platform_performance(
            user_id=current_user.id,
            platforms=request.platforms,
            time_range=request.time_range
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights")
async def get_insights(
    time_range: str = Query("7d", description="时间范围"),
    insight_type: Optional[str] = Query(None, description="洞察类型"),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取智能洞察
    """
    try:
        insights = await analytics_service.get_insights(
            user_id=current_user.id,
            time_range=time_range,
            insight_type=insight_type
        )
        return {"success": True, "data": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/realtime")
async def get_realtime_metrics(
    platforms: List[str] = Query([]),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取实时指标
    """
    try:
        metrics = await analytics_service.get_realtime_metrics(
            user_id=current_user.id,
            platforms=platforms
        )
        return {"success": True, "data": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 报告管理API

```python
@router.post("/reports")
async def create_report(
    request: CreateReportRequest,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    创建报告
    """
    try:
        result = await analytics_service.create_custom_report(
            user_id=current_user.id,
            template_id=request.template_id,
            config=request.config
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports")
async def get_reports(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    report_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取报告列表
    """
    try:
        reports = await analytics_service.get_user_reports(
            user_id=current_user.id,
            page=page,
            size=size,
            report_type=report_type
        )
        return {"success": True, "data": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{report_id}")
async def get_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取报告详情
    """
    try:
        report = await analytics_service.get_report(
            report_id=report_id,
            user_id=current_user.id
        )
        return {"success": True, "data": report}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Report not found")

@router.post("/reports/{report_id}/export")
async def export_report(
    report_id: int,
    format: str = Query("pdf", description="导出格式"),
    current_user: User = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    导出报告
    """
    try:
        file_path = await analytics_service.export_report(
            report_id=report_id,
            user_id=current_user.id,
            format=format
        )
        return {"success": True, "data": {"file_path": file_path}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 前端组件设计

### 1. 数据分析主页面

```vue
<template>
  <div class="analytics-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">数据分析</h1>
        <p class="page-description">深入了解您的内容表现和用户互动情况</p>
      </div>
      <div class="header-actions">
        <el-button type="primary" @click="showCreateReportDialog = true">
          <el-icon><DocumentAdd /></el-icon>
          创建报告
        </el-button>
        <el-button @click="refreshData" :loading="refreshing">
          <el-icon><Refresh /></el-icon>
          刷新数据
        </el-button>
      </div>
    </div>

    <!-- 时间范围选择 -->
    <div class="time-range-section">
      <el-card class="time-range-card">
        <div class="time-range-content">
          <div class="range-selector">
            <el-radio-group v-model="selectedTimeRange" @change="handleTimeRangeChange">
              <el-radio-button label="7d">最近7天</el-radio-button>
              <el-radio-button label="30d">最近30天</el-radio-button>
              <el-radio-button label="90d">最近90天</el-radio-button>
              <el-radio-button label="custom">自定义</el-radio-button>
            </el-radio-group>
          </div>
          <div v-if="selectedTimeRange === 'custom'" class="custom-range">
            <el-date-picker
              v-model="customTimeRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              format="YYYY-MM-DD"
              value-format="YYYY-MM-DD"
              @change="handleCustomRangeChange"
            />
          </div>
          <div class="platform-filter">
            <el-select
              v-model="selectedPlatforms"
              multiple
              placeholder="选择平台"
              style="width: 200px"
              @change="handlePlatformChange"
            >
              <el-option
                v-for="platform in availablePlatforms"
                :key="platform.value"
                :label="platform.label"
                :value="platform.value"
              />
            </el-select>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 关键指标卡片 -->
    <div class="metrics-section">
      <el-row :gutter="20">
        <el-col :span="6" v-for="metric in keyMetrics" :key="metric.key">
          <el-card class="metric-card" :class="`metric-${metric.key}`">
            <div class="metric-content">
              <div class="metric-icon" :class="metric.key">
                <el-icon><component :is="metric.icon" /></el-icon>
              </div>
              <div class="metric-info">
                <div class="metric-value">{{ formatNumber(metric.value) }}</div>
                <div class="metric-label">{{ metric.label }}</div>
                <div class="metric-trend" :class="metric.trend > 0 ? 'positive' : 'negative'">
                  <el-icon>
                    <ArrowUp v-if="metric.trend > 0" />
                    <ArrowDown v-else />
                  </el-icon>
                  {{ Math.abs(metric.trend) }}%
                </div>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 图表分析区域 -->
    <div class="charts-section">
      <el-row :gutter="20">
        <!-- 趋势分析图表 -->
        <el-col :span="12">
          <el-card class="chart-card">
            <template #header>
              <div class="chart-header">
                <span class="chart-title">趋势分析</span>
                <el-dropdown @command="handleTrendMetricChange">
                  <el-button text>
                    {{ currentTrendMetric.label }}
                    <el-icon><ArrowDown /></el-icon>
                  </el-button>
                  <template #dropdown>
                    <el-dropdown-menu>
                      <el-dropdown-item
                        v-for="metric in trendMetricOptions"
                        :key="metric.value"
                        :command="metric.value"
                      >
                        {{ metric.label }}
                      </el-dropdown-item>
                    </el-dropdown-menu>
                  </template>
                </el-dropdown>
              </div>
            </template>
            <div class="chart-container">
              <v-chart
                :option="trendChartOption"
                :loading="chartsLoading"
                style="height: 300px"
              />
            </div>
          </el-card>
        </el-col>

        <!-- 平台对比图表 -->
        <el-col :span="12">
          <el-card class="chart-card">
            <template #header>
              <span class="chart-title">平台对比</span>
            </template>
            <div class="chart-container">
              <v-chart
                :option="platformChartOption"
                :loading="chartsLoading"
                style="height: 300px"
              />
            </div>
          </el-card>
        </el-col>
      </el-row>

      <el-row :gutter="20" style="margin-top: 20px">
        <!-- 内容类型分析 -->
        <el-col :span="8">
          <el-card class="chart-card">
            <template #header>
              <span class="chart-title">内容类型分析</span>
            </template>
            <div class="chart-container">
              <v-chart
                :option="contentTypeChartOption"
                :loading="chartsLoading"
                style="height: 250px"
              />
            </div>
          </el-card>
        </el-col>

        <!-- 互动率分布 -->
        <el-col :span="8">
          <el-card class="chart-card">
            <template #header>
              <span class="chart-title">互动率分布</span>
            </template>
            <div class="chart-container">
              <v-chart
                :option="engagementDistributionOption"
                :loading="chartsLoading"
                style="height: 250px"
              />
            </div>
          </el-card>
        </el-col>

        <!-- 发布时间热力图 -->
        <el-col :span="8">
          <el-card class="chart-card">
            <template #header>
              <span class="chart-title">发布时间热力图</span>
            </template>
            <div class="chart-container">
              <v-chart
                :option="timeHeatmapOption"
                :loading="chartsLoading"
                style="height: 250px"
              />
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 智能洞察 -->
    <div class="insights-section">
      <el-card class="insights-card">
        <template #header>
          <div class="insights-header">
            <span class="insights-title">
              <el-icon><Lightbulb /></el-icon>
              智能洞察
            </span>
            <el-button text @click="refreshInsights" :loading="insightsLoading">
              <el-icon><Refresh /></el-icon>
              刷新洞察
            </el-button>
          </div>
        </template>
        <div class="insights-content">
          <div v-if="insights.length === 0" class="no-insights">
            <el-empty description="暂无洞察数据" />
          </div>
          <div v-else class="insights-list">
            <div
              v-for="insight in insights"
              :key="insight.id"
              class="insight-item"
              :class="`insight-${insight.type}`"
            >
              <div class="insight-icon">
                <el-icon>
                  <TrendCharts v-if="insight.type === 'trend'" />
                  <Warning v-else-if="insight.type === 'anomaly'" />
                  <Lightbulb v-else-if="insight.type === 'suggestion'" />
                  <DataAnalysis v-else />
                </el-icon>
              </div>
              <div class="insight-content">
                <div class="insight-title">{{ insight.title }}</div>
                <div class="insight-description">{{ insight.description }}</div>
                <div v-if="insight.action" class="insight-action">
                  <el-button size="small" type="primary" @click="handleInsightAction(insight)">
                    {{ insight.action.label }}
                  </el-button>
                </div>
              </div>
              <div class="insight-importance">
                <el-tag
                  :type="insight.importance === 'high' ? 'danger' : insight.importance === 'medium' ? 'warning' : 'info'"
                  size="small"
                >
                  {{ insight.importance === 'high' ? '高' : insight.importance === 'medium' ? '中' : '低' }}
                </el-tag>
              </div>
            </div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 热门内容 -->
    <div class="top-content-section">
      <el-card class="top-content-card">
        <template #header>
          <div class="top-content-header">
            <span class="top-content-title">热门内容</span>
            <el-radio-group v-model="topContentMetric" size="small" @change="handleTopContentMetricChange">
              <el-radio-button label="views">浏览量</el-radio-button>
              <el-radio-button label="engagement">互动率</el-radio-button>
              <el-radio-button label="shares">分享数</el-radio-button>
            </el-radio-group>
          </div>
        </template>
        <div class="top-content-list">
          <div
            v-for="(content, index) in topContent"
            :key="content.id"
            class="content-item"
          >
            <div class="content-rank">{{ index + 1 }}</div>
            <div class="content-info">
              <div class="content-title">{{ content.title }}</div>
              <div class="content-meta">
                <span class="content-platform">{{ content.platform }}</span>
                <span class="content-date">{{ formatDate(content.publish_date) }}</span>
              </div>
            </div>
            <div class="content-metrics">
              <div class="metric-item">
                <span class="metric-label">浏览</span>
                <span class="metric-value">{{ formatNumber(content.views) }}</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">互动率</span>
                <span class="metric-value">{{ (content.engagement_rate * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 创建报告对话框 -->
    <CreateReportDialog
      v-model="showCreateReportDialog"
      @report-created="handleReportCreated"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  DocumentAdd,
  Refresh,
  ArrowUp,
  ArrowDown,
  Lightbulb,
  TrendCharts,
  Warning,
  DataAnalysis
} from '@element-plus/icons-vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import {
  CanvasRenderer
} from 'echarts/renderers'
import {
  LineChart,
  BarChart,
  PieChart,
  HeatmapChart
} from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  CalendarComponent,
  VisualMapComponent
} from 'echarts/components'
import { analyticsApi } from '@/api/analytics'
import CreateReportDialog from './components/CreateReportDialog.vue'

// 注册ECharts组件
use([
  CanvasRenderer,
  LineChart,
  BarChart,
  PieChart,
  HeatmapChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  CalendarComponent,
  VisualMapComponent
])

// 响应式数据
const refreshing = ref(false)
const chartsLoading = ref(false)
const insightsLoading = ref(false)
const showCreateReportDialog = ref(false)

// 时间范围
const selectedTimeRange = ref('7d')
const customTimeRange = ref([])

// 平台筛选
const selectedPlatforms = ref(['weibo', 'wechat'])
const availablePlatforms = [
  { value: 'weibo', label: '微博' },
  { value: 'wechat', label: '微信公众号' },
  { value: 'douyin', label: '抖音' },
  { value: 'toutiao', label: '今日头条' }
]

// 关键指标
const keyMetrics = ref([
  {
    key: 'views',
    label: '总浏览量',
    value: 0,
    trend: 0,
    icon: 'View'
  },
  {
    key: 'engagement',
    label: '总互动量',
    value: 0,
    trend: 0,
    icon: 'ChatDotRound'
  },
  {
    key: 'content',
    label: '发布内容',
    value: 0,
    trend: 0,
    icon: 'Document'
  },
  {
    key: 'rate',
    label: '平均互动率',
    value: 0,
    trend: 0,
    icon: 'TrendCharts'
  }
])

// 趋势分析
const currentTrendMetric = ref({ value: 'views', label: '浏览量' })
const trendMetricOptions = [
  { value: 'views', label: '浏览量' },
  { value: 'likes', label: '点赞数' },
  { value: 'comments', label: '评论数' },
  { value: 'shares', label: '分享数' },
  { value: 'engagement_rate', label: '互动率' }
]

// 图表配置
const trendChartOption = ref({})
const platformChartOption = ref({})
const contentTypeChartOption = ref({})
const engagementDistributionOption = ref({})
const timeHeatmapOption = ref({})

// 智能洞察
const insights = ref([])

// 热门内容
const topContentMetric = ref('views')
const topContent = ref([])

// 计算属性
const timeRangeParams = computed(() => {
  if (selectedTimeRange.value === 'custom' && customTimeRange.value.length === 2) {
    return {
      start_date: customTimeRange.value[0],
      end_date: customTimeRange.value[1]
    }
  }
  
  const days = parseInt(selectedTimeRange.value.replace('d', ''))
  const endDate = new Date()
  const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000)
  
  return {
    start_date: startDate.toISOString().split('T')[0],
    end_date: endDate.toISOString().split('T')[0]
  }
})

// 监听器
watch([selectedTimeRange, selectedPlatforms], () => {
  loadAnalyticsData()
}, { deep: true })

// 生命周期
onMounted(() => {
  loadAnalyticsData()
})

// 方法
const loadAnalyticsData = async () => {
  try {
    chartsLoading.value = true
    
    // 并行加载数据
    const [metricsData, trendsData, platformData, contentData, insightsData] = await Promise.all([
      analyticsApi.getKeyMetrics({
        ...timeRangeParams.value,
        platforms: selectedPlatforms.value
      }),
      analyticsApi.getTrendAnalysis({
        ...timeRangeParams.value,
        platforms: selectedPlatforms.value,
        metric: currentTrendMetric.value.value
      }),
      analyticsApi.getPlatformComparison({
        ...timeRangeParams.value,
        platforms: selectedPlatforms.value
      }),
      analyticsApi.getContentAnalysis({
        ...timeRangeParams.value,
        platforms: selectedPlatforms.value
      }),
      analyticsApi.getInsights({
        ...timeRangeParams.value,
        platforms: selectedPlatforms.value
      })
    ])
    
    // 更新数据
    updateKeyMetrics(metricsData.data)
    updateTrendChart(trendsData.data)
    updatePlatformChart(platformData.data)
    updateContentTypeChart(contentData.data)
    updateEngagementDistribution(contentData.data)
    updateTimeHeatmap(contentData.data)
    insights.value = insightsData.data
    
    // 加载热门内容
    await loadTopContent()
    
  } catch (error) {
    ElMessage.error('加载数据失败')
    console.error('Failed to load analytics data:', error)
  } finally {
    chartsLoading.value = false
  }
}

const updateKeyMetrics = (data: any) => {
  keyMetrics.value = [
    {
      key: 'views',
      label: '总浏览量',
      value: data.total_views || 0,
      trend: data.views_trend || 0,
      icon: 'View'
    },
    {
      key: 'engagement',
      label: '总互动量',
      value: data.total_engagement || 0,
      trend: data.engagement_trend || 0,
      icon: 'ChatDotRound'
    },
    {
      key: 'content',
      label: '发布内容',
      value: data.content_count || 0,
      trend: data.content_trend || 0,
      icon: 'Document'
    },
    {
      key: 'rate',
      label: '平均互动率',
      value: data.avg_engagement_rate || 0,
      trend: data.engagement_rate_trend || 0,
      icon: 'TrendCharts'
    }
  ]
}

const updateTrendChart = (data: any) => {
  trendChartOption.value = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data: selectedPlatforms.value.map(p => availablePlatforms.find(ap => ap.value === p)?.label || p)
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: data.dates || []
    },
    yAxis: {
      type: 'value'
    },
    series: selectedPlatforms.value.map(platform => ({
      name: availablePlatforms.find(p => p.value === platform)?.label || platform,
      type: 'line',
      smooth: true,
      data: data[platform] || []
    }))
  }
}

const updatePlatformChart = (data: any) => {
  platformChartOption.value = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    legend: {
      data: ['浏览量', '点赞数', '评论数', '分享数']
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: selectedPlatforms.value.map(p => availablePlatforms.find(ap => ap.value === p)?.label || p)
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: '浏览量',
        type: 'bar',
        data: selectedPlatforms.value.map(p => data.platforms?.[p]?.views || 0)
      },
      {
        name: '点赞数',
        type: 'bar',
        data: selectedPlatforms.value.map(p => data.platforms?.[p]?.likes || 0)
      },
      {
        name: '评论数',
        type: 'bar',
        data: selectedPlatforms.value.map(p => data.platforms?.[p]?.comments || 0)
      },
      {
        name: '分享数',
        type: 'bar',
        data: selectedPlatforms.value.map(p => data.platforms?.[p]?.shares || 0)
      }
    ]
  }
}

const updateContentTypeChart = (data: any) => {
  const contentTypes = data.content_types || []
  
  contentTypeChartOption.value = {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      data: contentTypes.map((item: any) => item.name)
    },
    series: [
      {
        name: '内容类型',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        label: {
          show: false,
          position: 'center'
        },
        emphasis: {
          label: {
            show: true,
            fontSize: '18',
            fontWeight: 'bold'
          }
        },
        labelLine: {
          show: false
        },
        data: contentTypes
      }
    ]
  }
}

const updateEngagementDistribution = (data: any) => {
  const distribution = data.engagement_distribution || []
  
  engagementDistributionOption.value = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: distribution.map((item: any) => item.range)
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: '内容数量',
        type: 'bar',
        data: distribution.map((item: any) => item.count),
        itemStyle: {
          color: '#409EFF'
        }
      }
    ]
  }
}

const updateTimeHeatmap = (data: any) => {
  const heatmapData = data.time_heatmap || []
  
  timeHeatmapOption.value = {
    tooltip: {
      position: 'top',
      formatter: function (params: any) {
        return `${params.data[1]}:00 - ${params.data[0]}<br/>发布量: ${params.data[2]}`
      }
    },
    grid: {
      height: '50%',
      top: '10%'
    },
    xAxis: {
      type: 'category',
      data: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
      splitArea: {
        show: true
      }
    },
    yAxis: {
      type: 'category',
      data: Array.from({ length: 24 }, (_, i) => i),
      splitArea: {
        show: true
      }
    },
    visualMap: {
      min: 0,
      max: 10,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '15%'
    },
    series: [
      {
        name: '发布量',
        type: 'heatmap',
        data: heatmapData,
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
}

const loadTopContent = async () => {
  try {
    const response = await analyticsApi.getTopContent({
      ...timeRangeParams.value,
      platforms: selectedPlatforms.value,
      metric: topContentMetric.value,
      limit: 10
    })
    topContent.value = response.data
  } catch (error) {
    console.error('Failed to load top content:', error)
  }
}

const refreshData = async () => {
  refreshing.value = true
  try {
    await loadAnalyticsData()
    ElMessage.success('数据已刷新')
  } catch (error) {
    ElMessage.error('刷新数据失败')
  } finally {
    refreshing.value = false
  }
}

const refreshInsights = async () => {
  insightsLoading.value = true
  try {
    const response = await analyticsApi.getInsights({
      ...timeRangeParams.value,
      platforms: selectedPlatforms.value
    })
    insights.value = response.data
    ElMessage.success('洞察已刷新')
  } catch (error) {
    ElMessage.error('刷新洞察失败')
  } finally {
    insightsLoading.value = false
  }
}

const handleTimeRangeChange = () => {
  if (selectedTimeRange.value !== 'custom') {
    customTimeRange.value = []
  }
}

const handleCustomRangeChange = () => {
  // 自动触发数据加载
}

const handlePlatformChange = () => {
  // 自动触发数据加载
}

const handleTrendMetricChange = (metric: string) => {
  const option = trendMetricOptions.find(opt => opt.value === metric)
  if (option) {
    currentTrendMetric.value = option
    loadAnalyticsData()
  }
}

const handleTopContentMetricChange = () => {
  loadTopContent()
}

const handleInsightAction = (insight: any) => {
  if (insight.action?.type === 'navigate') {
    // 导航到相关页面
    console.log('Navigate to:', insight.action.target)
  } else if (insight.action?.type === 'optimize') {
    // 执行优化操作
    console.log('Execute optimization:', insight.action.params)
  }
}

const handleReportCreated = (report: any) => {
  ElMessage.success('报告创建成功')
  // 可以导航到报告详情页面
}

const formatNumber = (num: number) => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}

const formatDate = (date: string) => {
  return new Date(date).toLocaleDateString('zh-CN')
}
</script>

<style scoped>
.analytics-page {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding: 24px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header-content h1 {
  margin: 0 0 8px 0;
  color: #303133;
  font-size: 24px;
  font-weight: 600;
}

.header-content p {
  margin: 0;
  color: #909399;
  font-size: 14px;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.time-range-section {
  margin-bottom: 24px;
}

.time-range-card {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.time-range-content {
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}

.custom-range {
  margin-left: 20px;
}

.metrics-section {
  margin-bottom: 24px;
}

.metric-card {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
}

.metric-card:hover {
  transform: translateY(-2px);
}

.metric-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.metric-icon {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
}

.metric-icon.views {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.metric-icon.engagement {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.metric-icon.content {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.metric-icon.rate {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.metric-info {
  flex: 1;
}

.metric-value {
  font-size: 24px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}

.metric-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 8px;
}

.metric-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
}

.metric-trend.positive {
  color: #67c23a;
}

.metric-trend.negative {
  color: #f56c6c;
}

.charts-section {
  margin-bottom: 24px;
}

.chart-card {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.chart-container {
  padding: 16px 0;
}

.insights-section {
  margin-bottom: 24px;
}

.insights-card {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.insights-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.insights-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.insights-content {
  padding: 16px 0;
}

.no-insights {
  text-align: center;
  padding: 40px 0;
}

.insights-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.insight-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #409eff;
}

.insight-item.insight-trend {
  border-left-color: #67c23a;
}

.insight-item.insight-anomaly {
  border-left-color: #f56c6c;
}

.insight-item.insight-suggestion {
  border-left-color: #e6a23c;
}

.insight-icon {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #409eff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.insight-content {
  flex: 1;
}

.insight-title {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}

.insight-description {
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
  margin-bottom: 8px;
}

.insight-action {
  margin-top: 8px;
}

.insight-importance {
  flex-shrink: 0;
}

.top-content-section {
  margin-bottom: 24px;
}

.top-content-card {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.top-content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.top-content-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.top-content-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px 0;
}

.content-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  background: #f8f9fa;
  border-radius: 8px;
  transition: background-color 0.2s;
}

.content-item:hover {
  background: #ecf5ff;
}

.content-rank {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #409eff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  flex-shrink: 0;
}

.content-info {
  flex: 1;
}

.content-title {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.content-meta {
  display: flex;
  gap: 12px;
  font-size: 12px;
  color: #909399;
}

.content-metrics {
  display: flex;
  gap: 16px;
  flex-shrink: 0;
}

.metric-item {
  text-align: center;
}

.metric-item .metric-label {
  font-size: 11px;
  color: #909399;
  margin-bottom: 2px;
}

.metric-item .metric-value {
  font-size: 13px;
  font-weight: 600;
  color: #303133;
}
</style>
```

### 2. 创建报告对话框组件

```vue
<template>
  <el-dialog
    v-model="visible"
    title="创建报告"
    width="600px"
    :before-close="handleClose"
  >
    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-width="100px"
    >
      <el-form-item label="报告名称" prop="name">
        <el-input
          v-model="form.name"
          placeholder="请输入报告名称"
          maxlength="100"
          show-word-limit
        />
      </el-form-item>
      
      <el-form-item label="报告类型" prop="type">
        <el-select v-model="form.type" placeholder="请选择报告类型" style="width: 100%">
          <el-option label="仪表板报告" value="dashboard" />
          <el-option label="周期性报告" value="periodic" />
          <el-option label="自定义报告" value="custom" />
        </el-select>
      </el-form-item>
      
      <el-form-item label="时间范围" prop="timeRange">
        <el-date-picker
          v-model="form.timeRange"
          type="daterange"
          range-separator="至"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          format="YYYY-MM-DD"
          value-format="YYYY-MM-DD"
          style="width: 100%"
        />
      </el-form-item>
      
      <el-form-item label="包含平台" prop="platforms">
        <el-checkbox-group v-model="form.platforms">
          <el-checkbox label="weibo">微博</el-checkbox>
          <el-checkbox label="wechat">微信公众号</el-checkbox>
          <el-checkbox label="douyin">抖音</el-checkbox>
          <el-checkbox label="toutiao">今日头条</el-checkbox>
        </el-checkbox-group>
      </el-form-item>
      
      <el-form-item label="包含指标" prop="metrics">
        <el-checkbox-group v-model="form.metrics">
          <el-checkbox label="views">浏览量</el-checkbox>
          <el-checkbox label="likes">点赞数</el-checkbox>
          <el-checkbox label="comments">评论数</el-checkbox>
          <el-checkbox label="shares">分享数</el-checkbox>
          <el-checkbox label="engagement_rate">互动率</el-checkbox>
        </el-checkbox-group>
      </el-form-item>
      
      <el-form-item label="报告描述">
        <el-input
          v-model="form.description"
          type="textarea"
          :rows="3"
          placeholder="请输入报告描述（可选）"
          maxlength="500"
          show-word-limit
        />
      </el-form-item>
    </el-form>
    
    <template #footer>
      <div class="dialog-footer">
        <el-button @click="handleClose">取消</el-button>
        <el-button type="primary" @click="handleSubmit" :loading="submitting">
          创建报告
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import type { FormInstance, FormRules } from 'element-plus'
import { analyticsApi } from '@/api/analytics'

interface Props {
  modelValue: boolean
}

interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'report-created', report: any): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const formRef = ref<FormInstance>()
const submitting = ref(false)

const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const form = reactive({
  name: '',
  type: 'dashboard',
  timeRange: [],
  platforms: ['weibo', 'wechat'],
  metrics: ['views', 'likes', 'comments', 'engagement_rate'],
  description: ''
})

const rules: FormRules = {
  name: [
    { required: true, message: '请输入报告名称', trigger: 'blur' },
    { min: 2, max: 100, message: '长度在 2 到 100 个字符', trigger: 'blur' }
  ],
  type: [
    { required: true, message: '请选择报告类型', trigger: 'change' }
  ],
  timeRange: [
    { required: true, message: '请选择时间范围', trigger: 'change' }
  ],
  platforms: [
    { type: 'array', required: true, message: '请至少选择一个平台', trigger: 'change' }
  ],
  metrics: [
    { type: 'array', required: true, message: '请至少选择一个指标', trigger: 'change' }
  ]
}

const handleSubmit = async () => {
  if (!formRef.value) return
  
  try {
    await formRef.value.validate()
    
    submitting.value = true
    
    const response = await analyticsApi.createReport({
      name: form.name,
      type: form.type,
      config: {
        time_range: {
          start_date: form.timeRange[0],
          end_date: form.timeRange[1]
        },
        platforms: form.platforms,
        metrics: form.metrics,
        description: form.description
      }
    })
    
    ElMessage.success('报告创建成功')
    emit('report-created', response.data)
    handleClose()
    
  } catch (error) {
    console.error('Failed to create report:', error)
    ElMessage.error('创建报告失败')
  } finally {
    submitting.value = false
  }
}

const handleClose = () => {
  visible.value = false
  resetForm()
}

const resetForm = () => {
  if (formRef.value) {
    formRef.value.resetFields()
  }
  Object.assign(form, {
    name: '',
    type: 'dashboard',
    timeRange: [],
    platforms: ['weibo', 'wechat'],
    metrics: ['views', 'likes', 'comments', 'engagement_rate'],
    description: ''
  })
}
</script>
```

## 验收标准

### 功能验收标准

1. **数据收集与处理**
   - [ ] 能够自动从各平台API收集数据
   - [ ] 支持实时数据同步和批量数据导入
   - [ ] 数据清洗准确率达到95%以上
   - [ ] 异常数据能够被正确识别和处理

2. **数据分析功能**
   - [ ] 支持多维度数据分析（时间、平台、内容类型等）
   - [ ] 提供趋势分析和对比分析功能
   - [ ] 支持自定义分析指标和维度
   - [ ] 分析结果准确性达到98%以上

3. **可视化报告**
   - [ ] 提供丰富的图表类型（折线图、柱状图、饼图、热力图等）
   - [ ] 支持交互式图表操作
   - [ ] 报告生成时间不超过30秒
   - [ ] 支持PDF和Excel格式导出

4. **智能洞察**
   - [ ] 能够自动识别数据趋势和异常
   - [ ] 提供个性化的优化建议
   - [ ] 洞察准确率达到85%以上
   - [ ] 支持洞察结果的反馈和学习

### 性能验收标准

1. **响应时间**
   - 仪表板加载时间 ≤ 3秒
   - 图表渲染时间 ≤ 2秒
   - 报告生成时间 ≤ 30秒
   - API响应时间 ≤ 1秒

2. **并发处理**
   - 支持100个并发用户同时访问
   - 支持10个并发报告生成任务
   - 数据处理吞吐量 ≥ 1000条/秒

3. **数据处理能力**
   - 支持处理100万条历史数据
   - 实时数据延迟 ≤ 5分钟
   - 数据存储压缩率 ≥ 70%

### 安全验收标准

1. **数据安全**
   - 所有敏感数据必须加密存储
   - 支持数据脱敏和匿名化
   - 数据传输使用HTTPS加密
   - 定期数据备份和恢复测试

2. **访问控制**
   - 基于角色的权限控制
   - 支持数据行级权限控制
   - 操作日志完整记录
   - 支持单点登录(SSO)

## 业务价值

### 直接业务价值

1. **运营效率提升**
   - 减少手动数据收集和分析时间80%
   - 提高决策制定速度50%
   - 降低运营成本30%

2. **数据驱动决策**
   - 提供准确的数据支持
   - 识别最佳发布时间和内容类型
   - 优化内容策略和平台选择

3. **竞争优势**
   - 快速响应市场变化
   - 精准的用户洞察
   - 个性化的内容推荐

### 间接业务价值

1. **品牌影响力**
   - 提升内容质量和用户参与度
   - 增强品牌在各平台的影响力
   - 建立数据驱动的运营文化

2. **用户体验**
   - 提供更相关的内容
   - 提高用户满意度
   - 增加用户粘性和忠诚度

## 依赖关系

### 技术依赖

1. **基础设施依赖**
   - 高性能计算集群
   - 大容量存储系统
   - 稳定的网络连接
   - 监控和告警系统

2. **第三方服务依赖**
   - 各平台API服务
   - 云服务提供商
   - 数据库服务
   - 缓存服务

3. **内部系统依赖**
   - 用户认证系统
   - 内容管理系统
   - 账号管理系统
   - 通知服务

### 业务依赖

1. **数据源依赖**
   - 各平台账号授权
   - API访问权限
   - 数据质量保证
   - 历史数据迁移

2. **团队依赖**
   - 数据分析师
   - 运营人员
   - 技术支持团队
   - 产品经理

### 环境依赖

1. **开发环境**
   - Python 3.9+开发环境
   - Node.js 16+前端环境
   - Docker容器化环境
   - CI/CD流水线

2. **生产环境**
   - Kubernetes集群
   - 负载均衡器
   - 数据库集群
   - 监控系统

## 风险评估

### 技术风险

1. **高风险**
   - **数据处理性能瓶颈**
     - 影响：系统响应缓慢，用户体验差
     - 概率：中等
     - 缓解措施：性能优化、分布式处理、缓存策略
   
   - **第三方API限制**
     - 影响：数据收集不完整，分析结果不准确
     - 概率：高
     - 缓解措施：多API源、数据缓存、降级策略

2. **中等风险**
   - **机器学习模型准确性**
     - 影响：洞察质量下降，用户信任度降低
     - 概率：中等
     - 缓解措施：模型持续训练、A/B测试、人工审核
   
   - **数据安全风险**
     - 影响：数据泄露，法律风险
     - 概率：低
     - 缓解措施：加密存储、访问控制、安全审计

### 业务风险

1. **高风险**
   - **用户接受度低**
     - 影响：功能使用率低，投资回报率差
     - 概率：中等
     - 缓解措施：用户培训、界面优化、功能简化
   
   - **数据质量问题**
     - 影响：分析结果不可信，决策错误
     - 概率：中等
     - 缓解措施：数据验证、清洗规则、质量监控

2. **中等风险**
   - **竞争对手压力**
     - 影响：市场份额下降，功能优势丧失
     - 概率：中等
     - 缓解措施：持续创新、差异化功能、快速迭代

### 运营风险

1. **中等风险**
   - **系统维护复杂性**
     - 影响：运维成本高，故障恢复时间长
     - 概率：中等
     - 缓解措施：自动化运维、监控告警、文档完善
   
   - **人员技能要求高**
     - 影响：招聘困难，培训成本高
     - 概率：中等
     - 缓解措施：技能培训、知识分享、外部合作

## 开发任务分解

### 后端开发任务

#### 阶段1：基础架构搭建（预计4周）

1. **数据库设计与搭建**（1周）
   - PostgreSQL数据库设计
   - InfluxDB时间序列数据库搭建
   - ClickHouse OLAP数据库配置
   - Redis缓存系统部署

2. **核心服务框架**（2周）
   - FastAPI项目结构搭建
   - 数据库连接和ORM配置
   - 缓存服务集成
   - 日志和监控系统集成

3. **数据收集模块**（1周）
   - 平台API适配器开发
   - 数据收集调度器
   - 数据清洗和验证
   - 异常处理机制

#### 阶段2：分析引擎开发（预计6周）

1. **数据处理引擎**（2周）
   - 数据预处理管道
   - 指标计算引擎
   - 聚合和统计功能
   - 数据质量监控

2. **分析算法实现**（2周）
   - 趋势分析算法
   - 异常检测算法
   - 相关性分析
   - 预测模型

3. **智能洞察引擎**（2周）
   - 规则引擎开发
   - 机器学习模型集成
   - 洞察生成逻辑
   - 建议推荐系统

#### 阶段3：API和服务开发（预计4周）

1. **分析API开发**（2周）
   - 数据查询API
   - 分析任务API
   - 实时指标API
   - 洞察获取API

2. **报告服务开发**（2周）
   - 报告模板管理
   - 报告生成引擎
   - 导出功能
   - 定时报告

### 前端开发任务

#### 阶段1：基础组件开发（预计3周）

1. **项目架构搭建**（1周）
   - Vue3 + TypeScript项目初始化
   - 路由和状态管理配置
   - UI组件库集成
   - 构建和部署配置

2. **通用组件开发**（1周）
   - 图表组件封装
   - 数据表格组件
   - 筛选器组件
   - 导出组件

3. **布局和导航**（1周）
   - 主布局设计
   - 导航菜单
   - 面包屑导航
   - 响应式适配

#### 阶段2：核心页面开发（预计5周）

1. **数据仪表板**（2周）
   - 关键指标卡片
   - 趋势图表
   - 实时数据展示
   - 交互式筛选

2. **分析报告页面**（2周）
   - 多维度分析界面
   - 自定义图表配置
   - 数据钻取功能
   - 对比分析视图

3. **智能洞察页面**（1周）
   - 洞察列表展示
   - 洞察详情查看
   - 建议操作界面
   - 反馈收集

#### 阶段3：高级功能开发（预计4周）

1. **报告管理**（2周）
   - 报告创建向导
   - 模板管理界面
   - 报告列表和搜索
   - 报告分享功能

2. **数据导出和打印**（1周）
   - 多格式导出
   - 打印优化
   - 批量操作
   - 进度显示

3. **个性化设置**（1周）
   - 仪表板自定义
   - 用户偏好设置
   - 主题切换
   - 快捷操作

### 测试任务

#### 单元测试（预计2周）
- 后端服务单元测试
- 前端组件单元测试
- 工具函数测试
- 测试覆盖率达到80%

#### 集成测试（预计2周）
- API集成测试
- 数据库集成测试
- 第三方服务集成测试
- 端到端测试

#### 性能测试（预计1周）
- 负载测试
- 压力测试
- 数据处理性能测试
- 前端性能优化

### 部署任务

#### 环境准备（预计1周）
- 生产环境配置
- 数据库部署
- 缓存系统部署
- 监控系统配置

#### 应用部署（预计1周）
- 后端服务部署
- 前端应用部署
- 负载均衡配置
- SSL证书配置

## 时间估算

### 总体时间安排

- **后端开发**：14周
- **前端开发**：12周
- **测试阶段**：5周
- **部署上线**：2周
- **总计**：约16-18周（考虑并行开发）

### 关键里程碑

1. **第4周**：基础架构完成
2. **第8周**：核心分析功能完成
3. **第12周**：前端主要页面完成
4. **第14周**：功能开发完成，开始测试
5. **第16周**：测试完成，准备上线
6. **第18周**：正式上线运行

## 人力资源需求

### 开发团队配置

1. **后端开发**：3人
   - 1名高级后端工程师（架构设计、核心功能）
   - 1名中级后端工程师（API开发、数据处理）
   - 1名初级后端工程师（辅助开发、测试）

2. **前端开发**：2人
   - 1名高级前端工程师（架构设计、核心组件）
   - 1名中级前端工程师（页面开发、交互实现）

3. **数据工程**：1人
   - 1名数据工程师（数据建模、算法实现）

4. **测试工程**：1人
   - 1名测试工程师（测试用例、自动化测试）

5. **运维工程**：1人
   - 1名运维工程师（部署、监控、维护）

### 总人力投入

- **总人数**：8人
- **总人月**：约32人月
- **平均开发周期**：4个月

## 成功指标

### 技术指标

1. **性能指标**
   - 系统可用性 ≥ 99.5%
   - 平均响应时间 ≤ 2秒
   - 数据处理准确率 ≥ 98%
   - 并发用户支持 ≥ 100人

2. **质量指标**
   - 代码测试覆盖率 ≥ 80%
   - 生产环境bug数量 ≤ 5个/月
   - 安全漏洞数量 = 0
   - 数据丢失率 = 0

### 业务指标

1. **使用指标**
   - 日活跃用户数 ≥ 50人
   - 报告生成数量 ≥ 100个/月
   - 用户满意度 ≥ 4.0/5.0
   - 功能使用率 ≥ 70%

2. **效果指标**
   - 数据分析效率提升 ≥ 80%
   - 决策制定时间缩短 ≥ 50%
   - 运营成本降低 ≥ 30%
   - ROI ≥ 200%

### 运营指标

1. **维护指标**
   - 平均故障恢复时间 ≤ 30分钟
   - 系统维护时间 ≤ 4小时/月
   - 用户支持响应时间 ≤ 2小时
   - 知识库完整度 ≥ 90%

2. **发展指标**
   - 新功能发布频率 ≥ 1次/月
   - 用户反馈处理率 ≥ 95%
   - 技术债务控制 ≤ 10%
   - 团队技能提升 ≥ 20%