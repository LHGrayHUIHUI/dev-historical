# Story 4.4: 自动化内容排程服务

## 用户故事概述

**作为** 内容运营人员  
**我希望** 能够自动化地安排和发布内容到多个平台  
**以便** 提高内容发布效率，确保内容在最佳时间发布，并减少手动操作的工作量

## 功能需求

### 核心功能

1. **智能排程算法**
   - 基于历史数据分析最佳发布时间
   - 支持多平台发布时间优化
   - 考虑用户活跃度和互动率
   - 支持节假日和特殊事件调整

2. **内容排程管理**
   - 创建和编辑排程任务
   - 支持单次和循环排程
   - 批量导入排程计划
   - 排程冲突检测和解决

3. **自动化发布**
   - 定时自动发布内容
   - 支持多平台同步发布
   - 发布状态实时监控
   - 发布失败自动重试

4. **排程优化建议**
   - AI驱动的发布时间建议
   - 内容类型匹配优化
   - 平台特性适配建议
   - 竞争对手分析集成

## 技术架构

### 核心技术栈

#### 后端技术栈

- **Web框架**: FastAPI 0.104+
- **任务调度**: Celery 5.3+ + Redis 7.0+
- **定时任务**: APScheduler 3.10+
- **机器学习**: scikit-learn 1.3+, pandas 2.1+, numpy 1.24+
- **数据分析**: matplotlib 3.7+, seaborn 0.12+
- **数据库**: PostgreSQL 15+ (主数据库), InfluxDB 2.7+ (时序数据)
- **缓存**: Redis 7.0+ (缓存和任务队列)
- **消息队列**: RabbitMQ 3.12+ / Apache Kafka 3.5+
- **监控**: Prometheus 2.47+, Grafana 10.1+
- **日志**: ELK Stack (Elasticsearch 8.9+, Logstash 8.9+, Kibana 8.9+)
- **容器化**: Docker 24.0+, Kubernetes 1.28+

#### 前端技术栈

- **框架**: Vue 3.3+ with Composition API
- **语言**: TypeScript 5.2+
- **状态管理**: Pinia 2.1+
- **路由**: Vue Router 4.2+
- **UI组件库**: Element Plus 2.4+
- **组合式API**: @vue/composition-api
- **HTTP客户端**: Axios 1.5+
- **日期处理**: Day.js 1.11+
- **工具库**: Lodash 4.17+
- **图表库**: ECharts 5.4+, Chart.js 4.4+
- **Vue图表**: Vue-ECharts 6.6+
- **构建工具**: Vite 4.4+
- **代码规范**: ESLint 8.49+, Prettier 3.0+
- **测试框架**: Vitest 0.34+

### 数据模型设计

#### PostgreSQL 数据模型

```sql
-- 排程任务表
CREATE TABLE scheduling_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    content_id UUID,
    platforms TEXT[] NOT NULL, -- 目标平台列表
    schedule_type VARCHAR(20) NOT NULL, -- 'once', 'daily', 'weekly', 'monthly'
    schedule_time TIMESTAMP WITH TIME ZONE NOT NULL,
    timezone VARCHAR(50) DEFAULT 'UTC',
    repeat_config JSONB, -- 重复配置
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'scheduled', 'published', 'failed', 'cancelled'
    auto_optimize BOOLEAN DEFAULT true, -- 是否启用自动优化
    optimization_config JSONB, -- 优化配置
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    scheduled_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE
);

-- 发布历史表
CREATE TABLE publish_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES scheduling_tasks(id),
    platform VARCHAR(50) NOT NULL,
    content_id UUID,
    scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    actual_publish_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL, -- 'success', 'failed', 'retrying'
    platform_post_id VARCHAR(200), -- 平台返回的帖子ID
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    engagement_data JSONB, -- 互动数据
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 最佳发布时间分析表
CREATE TABLE optimal_publish_times (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50) NOT NULL,
    content_type VARCHAR(50),
    day_of_week INTEGER, -- 0-6 (周日到周六)
    hour_of_day INTEGER, -- 0-23
    engagement_score DECIMAL(5,2), -- 互动评分
    sample_size INTEGER, -- 样本数量
    confidence_level DECIMAL(3,2), -- 置信度
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(platform, content_type, day_of_week, hour_of_day)
);

-- 排程模板表
CREATE TABLE schedule_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    template_config JSONB NOT NULL, -- 模板配置
    platforms TEXT[] NOT NULL,
    is_public BOOLEAN DEFAULT false,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 排程冲突记录表
CREATE TABLE schedule_conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES scheduling_tasks(id),
    conflict_with_task_id UUID REFERENCES scheduling_tasks(id),
    platform VARCHAR(50) NOT NULL,
    conflict_time TIMESTAMP WITH TIME ZONE NOT NULL,
    conflict_type VARCHAR(50) NOT NULL, -- 'time_overlap', 'rate_limit', 'content_similar'
    resolution_strategy VARCHAR(50), -- 'delay', 'cancel', 'merge'
    resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### Redis 数据结构

```python
# 任务队列
scheduling_queue = "scheduling:tasks:pending"
retry_queue = "scheduling:tasks:retry"

# 任务状态缓存
task_status_cache = "scheduling:task:{task_id}:status"

# 平台发布限制
platform_rate_limit = "scheduling:platform:{platform}:rate_limit"

# 最佳发布时间缓存
optimal_times_cache = "scheduling:optimal_times:{platform}:{content_type}"

# 用户排程统计
user_schedule_stats = "scheduling:user:{user_id}:stats"

# 实时发布监控
publish_monitor = "scheduling:monitor:publishing"
```

### 服务架构设计

#### 核心服务类

```python
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from celery import Celery

class ScheduleType(Enum):
    """排程类型枚举"""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SchedulingTask:
    """排程任务数据类"""
    id: str
    user_id: str
    title: str
    content_id: str
    platforms: List[str]
    schedule_type: ScheduleType
    schedule_time: datetime
    timezone: str = "UTC"
    repeat_config: Optional[Dict] = None
    auto_optimize: bool = True
    status: TaskStatus = TaskStatus.PENDING

class ContentSchedulingService:
    """内容排程服务核心类"""
    
    def __init__(self, db_session, redis_client, celery_app):
        self.db = db_session
        self.redis = redis_client
        self.celery = celery_app
        self.scheduler = AsyncIOScheduler()
        self.optimizer = ScheduleOptimizer()
        
    async def create_scheduling_task(
        self, 
        user_id: str,
        title: str,
        content_id: str,
        platforms: List[str],
        schedule_time: datetime,
        schedule_type: ScheduleType = ScheduleType.ONCE,
        **kwargs
    ) -> SchedulingTask:
        """
        创建排程任务
        
        Args:
            user_id: 用户ID
            title: 任务标题
            content_id: 内容ID
            platforms: 目标平台列表
            schedule_time: 排程时间
            schedule_type: 排程类型
            **kwargs: 其他配置参数
            
        Returns:
            SchedulingTask: 创建的排程任务
        """
        # 创建任务对象
        task = SchedulingTask(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            content_id=content_id,
            platforms=platforms,
            schedule_type=schedule_type,
            schedule_time=schedule_time,
            **kwargs
        )
        
        # 检查排程冲突
        conflicts = await self._check_schedule_conflicts(task)
        if conflicts:
            await self._resolve_conflicts(task, conflicts)
        
        # 优化发布时间
        if task.auto_optimize:
            optimized_time = await self.optimizer.optimize_publish_time(
                platforms=platforms,
                original_time=schedule_time,
                content_type=await self._get_content_type(content_id)
            )
            task.schedule_time = optimized_time
        
        # 保存到数据库
        await self._save_task_to_db(task)
        
        # 添加到调度器
        await self._schedule_task(task)
        
        return task
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 任务状态信息
        """
        # 从缓存获取状态
        cached_status = await self.redis.get(f"scheduling:task:{task_id}:status")
        if cached_status:
            return json.loads(cached_status)
        
        # 从数据库获取
        task = await self._get_task_from_db(task_id)
        if not task:
            raise TaskNotFoundError(f"Task {task_id} not found")
        
        status_info = {
            "task_id": task_id,
            "status": task.status.value,
            "schedule_time": task.schedule_time.isoformat(),
            "platforms": task.platforms,
            "progress": await self._calculate_task_progress(task)
        }
        
        # 缓存状态信息
        await self.redis.setex(
            f"scheduling:task:{task_id}:status",
            300,  # 5分钟过期
            json.dumps(status_info)
        )
        
        return status_info
    
    async def cancel_task(self, task_id: str, user_id: str) -> bool:
        """
        取消排程任务
        
        Args:
            task_id: 任务ID
            user_id: 用户ID
            
        Returns:
            bool: 是否成功取消
        """
        task = await self._get_task_from_db(task_id)
        if not task or task.user_id != user_id:
            return False
        
        if task.status in [TaskStatus.PUBLISHED, TaskStatus.CANCELLED]:
            return False
        
        # 从调度器移除
        self.scheduler.remove_job(task_id)
        
        # 更新状态
        task.status = TaskStatus.CANCELLED
        await self._update_task_in_db(task)
        
        # 清除缓存
        await self.redis.delete(f"scheduling:task:{task_id}:status")
        
        return True
    
    async def get_user_tasks(
        self, 
        user_id: str, 
        status: Optional[TaskStatus] = None,
        platform: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取用户的排程任务列表
        
        Args:
            user_id: 用户ID
            status: 任务状态过滤
            platform: 平台过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Dict]: 任务列表
        """
        query_conditions = {"user_id": user_id}
        
        if status:
            query_conditions["status"] = status.value
        
        tasks = await self._query_tasks_from_db(
            conditions=query_conditions,
            platform=platform,
            limit=limit,
            offset=offset
        )
        
        return [await self._task_to_dict(task) for task in tasks]
    
    async def _check_schedule_conflicts(self, task: SchedulingTask) -> List[Dict]:
        """
        检查排程冲突
        
        Args:
            task: 排程任务
            
        Returns:
            List[Dict]: 冲突列表
        """
        conflicts = []
        
        # 检查时间冲突
        time_conflicts = await self._check_time_conflicts(task)
        conflicts.extend(time_conflicts)
        
        # 检查平台限制冲突
        rate_limit_conflicts = await self._check_rate_limit_conflicts(task)
        conflicts.extend(rate_limit_conflicts)
        
        # 检查内容相似性冲突
        content_conflicts = await self._check_content_similarity_conflicts(task)
        conflicts.extend(content_conflicts)
        
        return conflicts
    
    async def _schedule_task(self, task: SchedulingTask):
        """
        将任务添加到调度器
        
        Args:
            task: 排程任务
        """
        if task.schedule_type == ScheduleType.ONCE:
            self.scheduler.add_job(
                self._execute_publish_task,
                'date',
                run_date=task.schedule_time,
                args=[task.id],
                id=task.id
            )
        elif task.schedule_type == ScheduleType.DAILY:
            self.scheduler.add_job(
                self._execute_publish_task,
                'cron',
                hour=task.schedule_time.hour,
                minute=task.schedule_time.minute,
                args=[task.id],
                id=task.id
            )
        # 其他重复类型的处理...
    
    async def _execute_publish_task(self, task_id: str):
        """
        执行发布任务
        
        Args:
            task_id: 任务ID
        """
        task = await self._get_task_from_db(task_id)
        if not task:
            return
        
        # 更新状态为发布中
        task.status = TaskStatus.PUBLISHING
        await self._update_task_in_db(task)
        
        # 异步发布到各平台
        publish_results = await asyncio.gather(
            *[self._publish_to_platform(task, platform) 
              for platform in task.platforms],
            return_exceptions=True
        )
        
        # 处理发布结果
        success_count = sum(1 for result in publish_results 
                          if not isinstance(result, Exception))
        
        if success_count == len(task.platforms):
            task.status = TaskStatus.PUBLISHED
        elif success_count > 0:
            task.status = TaskStatus.PUBLISHED  # 部分成功也算发布
        else:
            task.status = TaskStatus.FAILED
        
        task.published_at = datetime.utcnow()
        await self._update_task_in_db(task)
        
        # 记录发布历史
        await self._record_publish_history(task, publish_results)

class ScheduleOptimizer:
    """排程优化器"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    async def optimize_publish_time(
        self, 
        platforms: List[str],
        original_time: datetime,
        content_type: str
    ) -> datetime:
        """
        优化发布时间
        
        Args:
            platforms: 目标平台列表
            original_time: 原始时间
            content_type: 内容类型
            
        Returns:
            datetime: 优化后的时间
        """
        if not self.is_trained:
            await self._train_model()
        
        best_times = []
        
        for platform in platforms:
            optimal_time = await self._get_optimal_time_for_platform(
                platform=platform,
                content_type=content_type,
                reference_time=original_time
            )
            best_times.append(optimal_time)
        
        # 选择综合最优时间
        return self._select_best_compromise_time(best_times, original_time)
    
    async def _train_model(self):
        """
        训练优化模型
        """
        # 获取历史数据
        historical_data = await self._get_historical_engagement_data()
        
        if len(historical_data) < 100:  # 数据不足
            self.is_trained = False
            return
        
        # 特征工程
        features = self._extract_features(historical_data)
        targets = historical_data['engagement_score']
        
        # 训练模型
        self.model.fit(features, targets)
        self.is_trained = True
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        提取特征
        
        Args:
            data: 历史数据
            
        Returns:
            pd.DataFrame: 特征数据
        """
        features = pd.DataFrame()
        
        # 时间特征
        features['hour'] = data['publish_time'].dt.hour
        features['day_of_week'] = data['publish_time'].dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # 平台特征
        platform_dummies = pd.get_dummies(data['platform'], prefix='platform')
        features = pd.concat([features, platform_dummies], axis=1)
        
        # 内容类型特征
        content_dummies = pd.get_dummies(data['content_type'], prefix='content')
        features = pd.concat([features, content_dummies], axis=1)
        
        return features

class TaskProcessor:
    """任务处理器"""
    
    def __init__(self, publishing_service, notification_service):
        self.publishing_service = publishing_service
        self.notification_service = notification_service
    
    async def process_scheduled_task(self, task_id: str):
        """
        处理排程任务
        
        Args:
            task_id: 任务ID
        """
        try:
            # 获取任务详情
            task = await self._get_task_details(task_id)
            
            # 验证任务状态
            if not self._validate_task_for_execution(task):
                return
            
            # 执行发布
            results = await self._execute_publishing(task)
            
            # 处理结果
            await self._handle_publishing_results(task, results)
            
        except Exception as e:
            await self._handle_task_error(task_id, e)
    
    async def _execute_publishing(self, task: Dict) -> List[Dict]:
        """
        执行发布操作
        
        Args:
            task: 任务信息
            
        Returns:
            List[Dict]: 发布结果
        """
        results = []
        
        for platform in task['platforms']:
            try:
                result = await self.publishing_service.publish_content(
                    platform=platform,
                    content_id=task['content_id'],
                    publish_config=task.get('publish_config', {})
                )
                results.append({
                    'platform': platform,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'platform': platform,
                    'success': False,
                    'error': str(e)
                })
        
        return results
```

### API接口设计

#### 排程管理接口

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime

router = APIRouter(prefix="/api/v1/scheduling", tags=["scheduling"])

@router.post("/tasks", response_model=SchedulingTaskResponse)
async def create_scheduling_task(
    request: CreateSchedulingTaskRequest,
    current_user: User = Depends(get_current_user),
    scheduling_service: ContentSchedulingService = Depends(get_scheduling_service)
):
    """
    创建排程任务
    
    Args:
        request: 创建请求
        current_user: 当前用户
        scheduling_service: 排程服务
        
    Returns:
        SchedulingTaskResponse: 创建的任务信息
    """
    try:
        task = await scheduling_service.create_scheduling_task(
            user_id=current_user.id,
            title=request.title,
            content_id=request.content_id,
            platforms=request.platforms,
            schedule_time=request.schedule_time,
            schedule_type=request.schedule_type,
            repeat_config=request.repeat_config,
            auto_optimize=request.auto_optimize
        )
        
        return SchedulingTaskResponse.from_task(task)
        
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/tasks/{task_id}", response_model=SchedulingTaskDetailResponse)
async def get_task_detail(
    task_id: str,
    current_user: User = Depends(get_current_user),
    scheduling_service: ContentSchedulingService = Depends(get_scheduling_service)
):
    """
    获取任务详情
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        scheduling_service: 排程服务
        
    Returns:
        SchedulingTaskDetailResponse: 任务详情
    """
    try:
        task_detail = await scheduling_service.get_task_detail(
            task_id=task_id,
            user_id=current_user.id
        )
        
        return SchedulingTaskDetailResponse.from_dict(task_detail)
        
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")

@router.get("/tasks", response_model=List[SchedulingTaskResponse])
async def get_user_tasks(
    status: Optional[str] = Query(None, description="任务状态过滤"),
    platform: Optional[str] = Query(None, description="平台过滤"),
    limit: int = Query(50, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    current_user: User = Depends(get_current_user),
    scheduling_service: ContentSchedulingService = Depends(get_scheduling_service)
):
    """
    获取用户任务列表
    
    Args:
        status: 状态过滤
        platform: 平台过滤
        limit: 数量限制
        offset: 偏移量
        current_user: 当前用户
        scheduling_service: 排程服务
        
    Returns:
        List[SchedulingTaskResponse]: 任务列表
    """
    task_status = TaskStatus(status) if status else None
    
    tasks = await scheduling_service.get_user_tasks(
        user_id=current_user.id,
        status=task_status,
        platform=platform,
        limit=limit,
        offset=offset
    )
    
    return [SchedulingTaskResponse.from_dict(task) for task in tasks]

@router.put("/tasks/{task_id}", response_model=SchedulingTaskResponse)
async def update_scheduling_task(
    task_id: str,
    request: UpdateSchedulingTaskRequest,
    current_user: User = Depends(get_current_user),
    scheduling_service: ContentSchedulingService = Depends(get_scheduling_service)
):
    """
    更新排程任务
    
    Args:
        task_id: 任务ID
        request: 更新请求
        current_user: 当前用户
        scheduling_service: 排程服务
        
    Returns:
        SchedulingTaskResponse: 更新后的任务
    """
    try:
        updated_task = await scheduling_service.update_task(
            task_id=task_id,
            user_id=current_user.id,
            update_data=request.dict(exclude_unset=True)
        )
        
        return SchedulingTaskResponse.from_task(updated_task)
        
    except TaskNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/tasks/{task_id}")
async def cancel_scheduling_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    scheduling_service: ContentSchedulingService = Depends(get_scheduling_service)
):
    """
    取消排程任务
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        scheduling_service: 排程服务
        
    Returns:
        dict: 操作结果
    """
    success = await scheduling_service.cancel_task(
        task_id=task_id,
        user_id=current_user.id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
    
    return {"message": "Task cancelled successfully"}

@router.post("/tasks/batch", response_model=BatchSchedulingResponse)
async def create_batch_scheduling(
    request: BatchSchedulingRequest,
    current_user: User = Depends(get_current_user),
    scheduling_service: ContentSchedulingService = Depends(get_scheduling_service)
):
    """
    批量创建排程任务
    
    Args:
        request: 批量创建请求
        current_user: 当前用户
        scheduling_service: 排程服务
        
    Returns:
        BatchSchedulingResponse: 批量创建结果
    """
    results = await scheduling_service.create_batch_tasks(
        user_id=current_user.id,
        tasks_data=request.tasks,
        batch_config=request.batch_config
    )
    
    return BatchSchedulingResponse(
        total_tasks=len(request.tasks),
        successful_tasks=len([r for r in results if r.success]),
        failed_tasks=len([r for r in results if not r.success]),
        results=results
    )
```

#### 优化建议接口

```python
@router.get("/optimization/suggestions", response_model=OptimizationSuggestionsResponse)
async def get_optimization_suggestions(
    platforms: List[str] = Query(..., description="目标平台列表"),
    content_type: str = Query(..., description="内容类型"),
    preferred_time: Optional[datetime] = Query(None, description="偏好时间"),
    current_user: User = Depends(get_current_user),
    optimizer: ScheduleOptimizer = Depends(get_optimizer)
):
    """
    获取排程优化建议
    
    Args:
        platforms: 目标平台
        content_type: 内容类型
        preferred_time: 偏好时间
        current_user: 当前用户
        optimizer: 优化器
        
    Returns:
        OptimizationSuggestionsResponse: 优化建议
    """
    suggestions = await optimizer.get_optimization_suggestions(
        user_id=current_user.id,
        platforms=platforms,
        content_type=content_type,
        preferred_time=preferred_time
    )
    
    return OptimizationSuggestionsResponse(
        suggestions=suggestions,
        confidence_score=suggestions.get('confidence', 0.0),
        explanation=suggestions.get('explanation', '')
    )

@router.get("/analytics/performance", response_model=SchedulingAnalyticsResponse)
async def get_scheduling_analytics(
    start_date: datetime = Query(..., description="开始日期"),
    end_date: datetime = Query(..., description="结束日期"),
    platforms: Optional[List[str]] = Query(None, description="平台过滤"),
    current_user: User = Depends(get_current_user),
    analytics_service: SchedulingAnalyticsService = Depends(get_analytics_service)
):
    """
    获取排程分析数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        platforms: 平台过滤
        current_user: 当前用户
        analytics_service: 分析服务
        
    Returns:
        SchedulingAnalyticsResponse: 分析数据
    """
    analytics = await analytics_service.get_scheduling_analytics(
        user_id=current_user.id,
        start_date=start_date,
        end_date=end_date,
        platforms=platforms
    )
    
    return SchedulingAnalyticsResponse.from_dict(analytics)
```

## 前端组件设计

### 1. 排程管理主页面

```vue
<template>
  <div class="scheduling-management">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">内容排程</h1>
        <p class="page-description">智能化内容发布排程，提升运营效率</p>
      </div>
      <div class="header-actions">
        <el-button type="primary" @click="showCreateDialog = true">
          <el-icon><Plus /></el-icon>
          创建排程
        </el-button>
        <el-button @click="showBatchDialog = true">
          <el-icon><Upload /></el-icon>
          批量导入
        </el-button>
        <el-button @click="showTemplateDialog = true">
          <el-icon><Document /></el-icon>
          排程模板
        </el-button>
      </div>
    </div>

    <!-- 统计卡片 -->
    <el-row :gutter="20" class="stats-cards">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon total">
              <el-icon><Calendar /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.totalTasks }}</div>
              <div class="stat-label">总排程任务</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon scheduled">
              <el-icon><Clock /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.scheduledTasks }}</div>
              <div class="stat-label">待发布任务</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon published">
              <el-icon><Check /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.publishedToday }}</div>
              <div class="stat-label">今日已发布</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-content">
            <div class="stat-icon success-rate">
              <el-icon><TrendCharts /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ stats.successRate }}%</div>
              <div class="stat-label">发布成功率</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 筛选和搜索 -->
    <el-card class="filter-section">
      <el-row :gutter="20">
        <el-col :span="6">
          <el-select
            v-model="filters.status"
            placeholder="任务状态"
            clearable
            @change="handleFilterChange"
          >
            <el-option label="全部状态" value="" />
            <el-option label="待发布" value="scheduled" />
            <el-option label="发布中" value="publishing" />
            <el-option label="已发布" value="published" />
            <el-option label="发布失败" value="failed" />
            <el-option label="已取消" value="cancelled" />
          </el-select>
        </el-col>
        <el-col :span="6">
          <el-select
            v-model="filters.platform"
            placeholder="发布平台"
            clearable
            @change="handleFilterChange"
          >
            <el-option label="全部平台" value="" />
            <el-option label="微博" value="weibo" />
            <el-option label="微信公众号" value="wechat" />
            <el-option label="抖音" value="douyin" />
            <el-option label="今日头条" value="toutiao" />
          </el-select>
        </el-col>
        <el-col :span="6">
          <el-date-picker
            v-model="filters.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            @change="handleFilterChange"
          />
        </el-col>
        <el-col :span="6">
          <el-input
            v-model="filters.keyword"
            placeholder="搜索任务标题或内容"
            @input="handleSearch"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </el-col>
      </el-row>
    </el-card>

    <!-- 任务列表 -->
    <el-card class="tasks-section">
      <template #header>
        <div class="section-header">
          <span>排程任务</span>
          <div class="header-actions">
            <el-button-group>
              <el-button
                :type="viewMode === 'list' ? 'primary' : 'default'"
                @click="viewMode = 'list'"
              >
                <el-icon><List /></el-icon>
              </el-button>
              <el-button
                :type="viewMode === 'calendar' ? 'primary' : 'default'"
                @click="viewMode = 'calendar'"
              >
                <el-icon><Calendar /></el-icon>
              </el-button>
            </el-button-group>
          </div>
        </div>
      </template>

      <!-- 列表视图 -->
      <div v-if="viewMode === 'list'">
        <el-table
          :data="tasks"
          v-loading="loading"
          @selection-change="handleSelectionChange"
        >
          <el-table-column type="selection" width="55" />
          <el-table-column label="任务标题" min-width="200">
            <template #default="{ row }">
              <div class="task-title">
                <span>{{ row.title }}</span>
                <el-tag
                  v-if="row.auto_optimize"
                  size="small"
                  type="success"
                  class="optimize-tag"
                >
                  智能优化
                </el-tag>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="发布平台" width="150">
            <template #default="{ row }">
              <div class="platforms">
                <el-tag
                  v-for="platform in row.platforms"
                  :key="platform"
                  size="small"
                  class="platform-tag"
                >
                  {{ getPlatformName(platform) }}
                </el-tag>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="排程时间" width="180">
            <template #default="{ row }">
              <div class="schedule-time">
                <div>{{ formatDateTime(row.schedule_time) }}</div>
                <div class="time-info">
                  <span v-if="row.schedule_type !== 'once'" class="repeat-info">
                    {{ getRepeatText(row.schedule_type) }}
                  </span>
                </div>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="状态" width="120">
            <template #default="{ row }">
              <el-tag
                :type="getStatusType(row.status)"
                :effect="row.status === 'publishing' ? 'plain' : 'dark'"
              >
                <el-icon v-if="row.status === 'publishing'" class="is-loading">
                  <Loading />
                </el-icon>
                {{ getStatusText(row.status) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="进度" width="150">
            <template #default="{ row }">
              <div v-if="row.status === 'publishing'" class="progress-info">
                <el-progress
                  :percentage="row.progress || 0"
                  :stroke-width="6"
                  :show-text="false"
                />
                <span class="progress-text">{{ row.progress || 0 }}%</span>
              </div>
              <div v-else-if="row.status === 'published'" class="publish-results">
                <span class="success-count">{{ row.success_count || 0 }}</span>
                <span class="total-count">/ {{ row.platforms.length }}</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="200" fixed="right">
            <template #default="{ row }">
              <el-button
                size="small"
                @click="viewTaskDetail(row)"
              >
                查看
              </el-button>
              <el-button
                v-if="canEditTask(row)"
                size="small"
                type="primary"
                @click="editTask(row)"
              >
                编辑
              </el-button>
              <el-button
                v-if="canCancelTask(row)"
                size="small"
                type="danger"
                @click="cancelTask(row)"
              >
                取消
              </el-button>
            </template>
          </el-table-column>
        </el-table>

        <!-- 批量操作 -->
        <div v-if="selectedTasks.length > 0" class="batch-actions">
          <el-alert
            :title="`已选择 ${selectedTasks.length} 个任务`"
            type="info"
            show-icon
            :closable="false"
          >
            <template #default>
              <el-button size="small" @click="batchCancel">
                批量取消
              </el-button>
              <el-button size="small" @click="batchReschedule">
                批量重新排程
              </el-button>
              <el-button size="small" @click="batchExport">
                导出选中
              </el-button>
            </template>
          </el-alert>
        </div>

        <!-- 分页 -->
        <div class="pagination">
          <el-pagination
            v-model:current-page="pagination.page"
            v-model:page-size="pagination.size"
            :total="pagination.total"
            :page-sizes="[10, 20, 50, 100]"
            layout="total, sizes, prev, pager, next, jumper"
            @size-change="handleSizeChange"
            @current-change="handlePageChange"
          />
        </div>
      </div>

      <!-- 日历视图 -->
      <div v-else-if="viewMode === 'calendar'" class="calendar-view">
        <el-calendar v-model="calendarDate">
          <template #date-cell="{ data }">
            <div class="calendar-cell">
              <div class="date-number">{{ data.day.split('-').pop() }}</div>
              <div v-if="getTasksForDate(data.day).length > 0" class="tasks-indicator">
                <div
                  v-for="task in getTasksForDate(data.day).slice(0, 3)"
                  :key="task.id"
                  class="task-dot"
                  :class="`task-dot--${task.status}`"
                  @click="viewTaskDetail(task)"
                >
                  <el-tooltip :content="task.title" placement="top">
                    <div class="dot"></div>
                  </el-tooltip>
                </div>
                <div
                  v-if="getTasksForDate(data.day).length > 3"
                  class="more-tasks"
                >
                  +{{ getTasksForDate(data.day).length - 3 }}
                </div>
              </div>
            </div>
          </template>
        </el-calendar>
      </div>
    </el-card>

    <!-- 创建排程对话框 -->
    <CreateSchedulingDialog
      v-model="showCreateDialog"
      @task-created="handleTaskCreated"
    />

    <!-- 任务详情对话框 -->
    <TaskDetailDialog
      v-model="showDetailDialog"
      :task="selectedTask"
      @task-updated="handleTaskUpdated"
    />

    <!-- 批量导入对话框 -->
    <BatchImportDialog
      v-model="showBatchDialog"
      @tasks-imported="handleTasksImported"
    />

    <!-- 排程模板对话框 -->
    <ScheduleTemplateDialog
      v-model="showTemplateDialog"
      @template-applied="handleTemplateApplied"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Plus,
  Upload,
  Document,
  Calendar,
  Clock,
  Check,
  TrendCharts,
  Search,
  List,
  Loading
} from '@element-plus/icons-vue'
import { schedulingApi } from '@/api/scheduling'
import CreateSchedulingDialog from './components/CreateSchedulingDialog.vue'
import TaskDetailDialog from './components/TaskDetailDialog.vue'
import BatchImportDialog from './components/BatchImportDialog.vue'
import ScheduleTemplateDialog from './components/ScheduleTemplateDialog.vue'

// 响应式数据
const loading = ref(false)
const viewMode = ref('list')
const showCreateDialog = ref(false)
const showDetailDialog = ref(false)
const showBatchDialog = ref(false)
const showTemplateDialog = ref(false)
const selectedTask = ref(null)
const selectedTasks = ref([])
const calendarDate = ref(new Date())

// 统计数据
const stats = reactive({
  totalTasks: 0,
  scheduledTasks: 0,
  publishedToday: 0,
  successRate: 0
})

// 筛选条件
const filters = reactive({
  status: '',
  platform: '',
  dateRange: null,
  keyword: ''
})

// 任务列表
const tasks = ref([])

// 分页
const pagination = reactive({
  page: 1,
  size: 20,
  total: 0
})

// 计算属性
const filteredTasks = computed(() => {
  return tasks.value.filter(task => {
    if (filters.status && task.status !== filters.status) return false
    if (filters.platform && !task.platforms.includes(filters.platform)) return false
    if (filters.keyword && !task.title.toLowerCase().includes(filters.keyword.toLowerCase())) return false
    return true
  })
})

// 监听器
watch([filters], () => {
  pagination.page = 1
  loadTasks()
}, { deep: true })

// 生命周期
onMounted(() => {
  loadStats()
  loadTasks()
  // 定时刷新状态
  setInterval(refreshTaskStatus, 30000)
})

// 方法
const loadStats = async () => {
  try {
    const response = await schedulingApi.getStats()
    Object.assign(stats, response.data)
  } catch (error) {
    console.error('Failed to load stats:', error)
  }
}

const loadTasks = async () => {
  try {
    loading.value = true
    const params = {
      page: pagination.page,
      size: pagination.size,
      status: filters.status || undefined,
      platform: filters.platform || undefined,
      keyword: filters.keyword || undefined,
      start_date: filters.dateRange?.[0],
      end_date: filters.dateRange?.[1]
    }
    
    const response = await schedulingApi.getTasks(params)
    tasks.value = response.data.items
    pagination.total = response.data.total
    
  } catch (error) {
    ElMessage.error('加载任务列表失败')
  } finally {
    loading.value = false
  }
}

const refreshTaskStatus = async () => {
  // 只刷新进行中的任务状态
  const activeTasks = tasks.value.filter(task => 
    ['scheduled', 'publishing'].includes(task.status)
  )
  
  for (const task of activeTasks) {
    try {
      const response = await schedulingApi.getTaskStatus(task.id)
      const index = tasks.value.findIndex(t => t.id === task.id)
      if (index !== -1) {
        Object.assign(tasks.value[index], response.data)
      }
    } catch (error) {
      console.error(`Failed to refresh task ${task.id} status:`, error)
    }
  }
}

const handleFilterChange = () => {
  loadTasks()
}

const handleSearch = debounce(() => {
  loadTasks()
}, 500)

const handleSelectionChange = (selection) => {
  selectedTasks.value = selection
}

const handlePageChange = (page) => {
  pagination.page = page
  loadTasks()
}

const handleSizeChange = (size) => {
  pagination.size = size
  pagination.page = 1
  loadTasks()
}

const viewTaskDetail = (task) => {
  selectedTask.value = task
  showDetailDialog.value = true
}

const editTask = (task) => {
  selectedTask.value = task
  showCreateDialog.value = true
}

const cancelTask = async (task) => {
  try {
    await ElMessageBox.confirm(
      `确定要取消排程任务 "${task.title}" 吗？`,
      '确认取消',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await schedulingApi.cancelTask(task.id)
    ElMessage.success('任务已取消')
    loadTasks()
    
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('取消任务失败')
    }
  }
}

const batchCancel = async () => {
  try {
    await ElMessageBox.confirm(
      `确定要取消选中的 ${selectedTasks.value.length} 个任务吗？`,
      '批量取消',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    const taskIds = selectedTasks.value.map(task => task.id)
    await schedulingApi.batchCancelTasks(taskIds)
    
    ElMessage.success('批量取消成功')
    selectedTasks.value = []
    loadTasks()
    
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('批量取消失败')
    }
  }
}

const batchReschedule = () => {
  // 实现批量重新排程逻辑
  ElMessage.info('批量重新排程功能开发中')
}

const batchExport = () => {
  // 实现批量导出逻辑
  ElMessage.info('批量导出功能开发中')
}

const getTasksForDate = (date) => {
  return tasks.value.filter(task => {
    const taskDate = new Date(task.schedule_time).toISOString().split('T')[0]
    return taskDate === date
  })
}

const handleTaskCreated = (task) => {
  ElMessage.success('排程任务创建成功')
  loadTasks()
  loadStats()
}

const handleTaskUpdated = (task) => {
  ElMessage.success('任务更新成功')
  loadTasks()
}

const handleTasksImported = (result) => {
  ElMessage.success(`成功导入 ${result.success_count} 个任务`)
  loadTasks()
  loadStats()
}

const handleTemplateApplied = (template) => {
  ElMessage.success('模板应用成功')
  loadTasks()
}

const canEditTask = (task) => {
  return ['pending', 'scheduled'].includes(task.status)
}

const canCancelTask = (task) => {
  return ['pending', 'scheduled', 'publishing'].includes(task.status)
}

const getPlatformName = (platform) => {
  const nameMap = {
    weibo: '微博',
    wechat: '微信',
    douyin: '抖音',
    toutiao: '头条'
  }
  return nameMap[platform] || platform
}

const getStatusType = (status) => {
  const typeMap = {
    pending: 'info',
    scheduled: 'warning',
    publishing: 'primary',
    published: 'success',
    failed: 'danger',
    cancelled: 'info'
  }
  return typeMap[status] || 'info'
}

const getStatusText = (status) => {
  const textMap = {
    pending: '待排程',
    scheduled: '已排程',
    publishing: '发布中',
    published: '已发布',
    failed: '发布失败',
    cancelled: '已取消'
  }
  return textMap[status] || status
}

const getRepeatText = (type) => {
  const textMap = {
    daily: '每日重复',
    weekly: '每周重复',
    monthly: '每月重复'
  }
  return textMap[type] || type
}

const formatDateTime = (dateTime) => {
  return new Date(dateTime).toLocaleString('zh-CN')
}

// 防抖函数
function debounce(func, wait) {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}
</script>

<style scoped>
.scheduling-management {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
}

.header-content {
  flex: 1;
}

.page-title {
  font-size: 28px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 8px 0;
}

.page-description {
  font-size: 16px;
  color: #6b7280;
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.stats-cards {
  margin-bottom: 24px;
}

.stat-card {
  border: none;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.stat-content {
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
}

.stat-icon.total {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stat-icon.scheduled {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.stat-icon.published {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.stat-icon.success-rate {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.stat-info {
  flex: 1;
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1;
}

.stat-label {
  font-size: 14px;
  color: #6b7280;
  margin-top: 4px;
}

.filter-section {
  margin-bottom: 24px;
}

.tasks-section {
  min-height: 600px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.optimize-tag {
  font-size: 10px;
}

.platforms {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.platform-tag {
  font-size: 12px;
}

.schedule-time {
  font-size: 14px;
}

.time-info {
  margin-top: 4px;
  font-size: 12px;
  color: #6b7280;
}

.repeat-info {
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
}

.progress-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.progress-text {
  font-size: 12px;
  color: #6b7280;
}

.publish-results {
  font-size: 14px;
}

.success-count {
  color: #10b981;
  font-weight: 600;
}

.total-count {
  color: #6b7280;
}

.batch-actions {
  margin-top: 16px;
}

.pagination {
  margin-top: 24px;
  display: flex;
  justify-content: center;
}

.calendar-view {
  min-height: 600px;
}

.calendar-cell {
  height: 100%;
  padding: 4px;
}

.date-number {
  font-weight: 600;
  margin-bottom: 4px;
}

.tasks-indicator {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
}

.task-dot {
  cursor: pointer;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #e5e7eb;
}

.task-dot--scheduled .dot {
  background: #f59e0b;
}

.task-dot--publishing .dot {
  background: #3b82f6;
  animation: pulse 2s infinite;
}

.task-dot--published .dot {
  background: #10b981;
}

.task-dot--failed .dot {
  background: #ef4444;
}

.more-tasks {
  font-size: 10px;
  color: #6b7280;
  margin-top: 2px;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>
```

### 2. 创建排程对话框组件

```vue
<template>
  <el-dialog
    v-model="visible"
    :title="isEdit ? '编辑排程任务' : '创建排程任务'"
    width="800px"
    :close-on-click-modal="false"
    @close="handleClose"
  >
    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-width="120px"
      @submit.prevent
    >
      <el-row :gutter="20">
        <el-col :span="12">
          <el-form-item label="任务标题" prop="title">
            <el-input
              v-model="form.title"
              placeholder="请输入任务标题"
              maxlength="100"
              show-word-limit
            />
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="内容选择" prop="content_id">
            <el-select
              v-model="form.content_id"
              placeholder="选择要发布的内容"
              filterable
              remote
              :remote-method="searchContent"
              :loading="contentLoading"
              style="width: 100%"
            >
              <el-option
                v-for="content in contentOptions"
                :key="content.id"
                :label="content.title"
                :value="content.id"
              >
                <div class="content-option">
                  <span class="content-title">{{ content.title }}</span>
                  <span class="content-type">{{ content.type }}</span>
                </div>
              </el-option>
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <el-form-item label="发布平台" prop="platforms">
        <el-checkbox-group v-model="form.platforms">
          <el-checkbox
            v-for="platform in availablePlatforms"
            :key="platform.value"
            :label="platform.value"
            :disabled="!platform.enabled"
          >
            <div class="platform-option">
              <el-icon class="platform-icon">
                <component :is="platform.icon" />
              </el-icon>
              <span>{{ platform.label }}</span>
              <el-tag v-if="!platform.enabled" size="small" type="info">
                未配置
              </el-tag>
            </div>
          </el-checkbox>
        </el-checkbox-group>
      </el-form-item>

      <el-row :gutter="20">
        <el-col :span="12">
          <el-form-item label="排程类型" prop="schedule_type">
            <el-radio-group v-model="form.schedule_type">
              <el-radio label="once">单次发布</el-radio>
              <el-radio label="daily">每日重复</el-radio>
              <el-radio label="weekly">每周重复</el-radio>
              <el-radio label="monthly">每月重复</el-radio>
            </el-radio-group>
          </el-form-item>
        </el-col>
        <el-col :span="12">
          <el-form-item label="时区设置" prop="timezone">
            <el-select v-model="form.timezone" placeholder="选择时区">
              <el-option
                v-for="tz in timezones"
                :key="tz.value"
                :label="tz.label"
                :value="tz.value"
              />
            </el-select>
          </el-form-item>
        </el-col>
      </el-row>

      <el-form-item label="排程时间" prop="schedule_time">
        <el-date-picker
          v-model="form.schedule_time"
          type="datetime"
          placeholder="选择排程时间"
          format="YYYY-MM-DD HH:mm:ss"
          value-format="YYYY-MM-DD HH:mm:ss"
          :disabled-date="disabledDate"
          style="width: 100%"
        />
      </el-form-item>

      <el-form-item v-if="form.schedule_type !== 'once'" label="重复配置">
        <div class="repeat-config">
          <template v-if="form.schedule_type === 'weekly'">
            <el-checkbox-group v-model="form.repeat_config.weekdays">
              <el-checkbox
                v-for="(day, index) in weekdays"
                :key="index"
                :label="index"
              >
                {{ day }}
              </el-checkbox>
            </el-checkbox-group>
          </template>
          
          <template v-if="form.schedule_type === 'monthly'">
            <el-radio-group v-model="form.repeat_config.monthly_type">
              <el-radio label="date">按日期</el-radio>
              <el-radio label="weekday">按星期</el-radio>
            </el-radio-group>
            
            <div v-if="form.repeat_config.monthly_type === 'date'" class="config-item">
              <el-input-number
                v-model="form.repeat_config.day_of_month"
                :min="1"
                :max="31"
                placeholder="日期"
              />
              <span>号</span>
            </div>
            
            <div v-else class="config-item">
              <el-select v-model="form.repeat_config.week_of_month" placeholder="第几周">
                <el-option label="第一周" value="1" />
                <el-option label="第二周" value="2" />
                <el-option label="第三周" value="3" />
                <el-option label="第四周" value="4" />
                <el-option label="最后一周" value="-1" />
              </el-select>
              <el-select v-model="form.repeat_config.weekday" placeholder="星期几">
                <el-option
                  v-for="(day, index) in weekdays"
                  :key="index"
                  :label="day"
                  :value="index"
                />
              </el-select>
            </div>
          </template>
          
          <div class="config-item">
            <label>结束时间：</label>
            <el-date-picker
              v-model="form.repeat_config.end_date"
              type="date"
              placeholder="选择结束日期（可选）"
              format="YYYY-MM-DD"
              value-format="YYYY-MM-DD"
            />
          </div>
        </div>
      </el-form-item>

      <el-form-item>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-checkbox v-model="form.auto_optimize">
              启用智能优化
            </el-checkbox>
            <el-tooltip content="系统将根据历史数据自动优化发布时间以获得更好的互动效果">
              <el-icon class="info-icon"><QuestionFilled /></el-icon>
            </el-tooltip>
          </el-col>
          <el-col :span="12">
            <el-button
              v-if="form.auto_optimize"
              type="text"
              @click="getOptimizationSuggestions"
              :loading="optimizationLoading"
            >
              获取优化建议
            </el-button>
          </el-col>
        </el-row>
      </el-form-item>

      <el-form-item v-if="optimizationSuggestions" label="优化建议">
        <el-alert
          :title="optimizationSuggestions.title"
          type="info"
          show-icon
          :closable="false"
        >
          <template #default>
            <div class="optimization-content">
              <p>{{ optimizationSuggestions.description }}</p>
              <div class="suggested-times">
                <div
                  v-for="suggestion in optimizationSuggestions.suggestions"
                  :key="suggestion.time"
                  class="time-suggestion"
                  @click="applySuggestion(suggestion)"
                >
                  <div class="suggestion-time">{{ suggestion.time }}</div>
                  <div class="suggestion-score">预期互动率: {{ suggestion.score }}%</div>
                  <div class="suggestion-reason">{{ suggestion.reason }}</div>
                </div>
              </div>
            </div>
          </template>
        </el-alert>
      </el-form-item>

      <el-form-item label="描述">
        <el-input
          v-model="form.description"
          type="textarea"
          :rows="3"
          placeholder="任务描述（可选）"
          maxlength="500"
          show-word-limit
        />
      </el-form-item>
    </el-form>

    <template #footer>
      <div class="dialog-footer">
        <el-button @click="handleClose">取消</el-button>
        <el-button
          type="primary"
          @click="handleSubmit"
          :loading="submitting"
        >
          {{ isEdit ? '更新' : '创建' }}排程
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { QuestionFilled } from '@element-plus/icons-vue'
import { schedulingApi } from '@/api/scheduling'
import { contentApi } from '@/api/content'

// Props
interface Props {
  modelValue: boolean
  task?: any
}

const props = withDefaults(defineProps<Props>(), {
  task: null
})

// Emits
const emit = defineEmits(['update:modelValue', 'task-created', 'task-updated'])

// 响应式数据
const formRef = ref()
const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const isEdit = computed(() => !!props.task)
const submitting = ref(false)
const contentLoading = ref(false)
const optimizationLoading = ref(false)
const contentOptions = ref([])
const optimizationSuggestions = ref(null)

// 表单数据
const form = reactive({
  title: '',
  content_id: '',
  platforms: [],
  schedule_type: 'once',
  schedule_time: '',
  timezone: 'Asia/Shanghai',
  repeat_config: {
    weekdays: [],
    monthly_type: 'date',
    day_of_month: 1,
    week_of_month: '1',
    weekday: 0,
    end_date: ''
  },
  auto_optimize: true,
  description: ''
})

// 表单验证规则
const rules = {
  title: [
    { required: true, message: '请输入任务标题', trigger: 'blur' },
    { min: 1, max: 100, message: '标题长度在 1 到 100 个字符', trigger: 'blur' }
  ],
  content_id: [
    { required: true, message: '请选择要发布的内容', trigger: 'change' }
  ],
  platforms: [
    { type: 'array', required: true, message: '请选择至少一个发布平台', trigger: 'change' },
    { type: 'array', min: 1, message: '请选择至少一个发布平台', trigger: 'change' }
  ],
  schedule_type: [
    { required: true, message: '请选择排程类型', trigger: 'change' }
  ],
  schedule_time: [
    { required: true, message: '请选择排程时间', trigger: 'change' }
  ],
  timezone: [
    { required: true, message: '请选择时区', trigger: 'change' }
  ]
}

// 可用平台
const availablePlatforms = ref([
  { value: 'weibo', label: '微博', icon: 'Weibo', enabled: true },
  { value: 'wechat', label: '微信公众号', icon: 'Wechat', enabled: true },
  { value: 'douyin', label: '抖音', icon: 'Douyin', enabled: false },
  { value: 'toutiao', label: '今日头条', icon: 'Toutiao', enabled: true }
])

// 时区选项
const timezones = [
  { value: 'Asia/Shanghai', label: '北京时间 (UTC+8)' },
  { value: 'UTC', label: '协调世界时 (UTC+0)' },
  { value: 'America/New_York', label: '纽约时间 (UTC-5)' },
  { value: 'Europe/London', label: '伦敦时间 (UTC+0)' }
]

// 星期选项
const weekdays = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']

// 监听器
watch(() => props.task, (newTask) => {
  if (newTask) {
    Object.assign(form, {
      ...newTask,
      repeat_config: newTask.repeat_config || form.repeat_config
    })
  }
}, { immediate: true })

watch(() => props.modelValue, (visible) => {
  if (visible && !props.task) {
    resetForm()
  }
})

// 方法
const resetForm = () => {
  Object.assign(form, {
    title: '',
    content_id: '',
    platforms: [],
    schedule_type: 'once',
    schedule_time: '',
    timezone: 'Asia/Shanghai',
    repeat_config: {
      weekdays: [],
      monthly_type: 'date',
      day_of_month: 1,
      week_of_month: '1',
      weekday: 0,
      end_date: ''
    },
    auto_optimize: true,
    description: ''
  })
  optimizationSuggestions.value = null
  nextTick(() => {
    formRef.value?.clearValidate()
  })
}

const searchContent = async (query: string) => {
  if (!query) return
  
  try {
    contentLoading.value = true
    const response = await contentApi.searchContent({ keyword: query, limit: 20 })
    contentOptions.value = response.data.items
  } catch (error) {
    console.error('Failed to search content:', error)
  } finally {
    contentLoading.value = false
  }
}

const disabledDate = (time: Date) => {
  return time.getTime() < Date.now() - 24 * 60 * 60 * 1000
}

const getOptimizationSuggestions = async () => {
  if (!form.platforms.length || !form.schedule_time) {
    ElMessage.warning('请先选择发布平台和排程时间')
    return
  }
  
  try {
    optimizationLoading.value = true
    const response = await schedulingApi.getOptimizationSuggestions({
      platforms: form.platforms,
      content_type: 'article', // 根据实际内容类型设置
      preferred_time: form.schedule_time
    })
    optimizationSuggestions.value = response.data
  } catch (error) {
    ElMessage.error('获取优化建议失败')
  } finally {
    optimizationLoading.value = false
  }
}

const applySuggestion = (suggestion: any) => {
  form.schedule_time = suggestion.time
  ElMessage.success('已应用优化建议')
}

const handleSubmit = async () => {
  try {
    await formRef.value.validate()
    
    submitting.value = true
    
    const submitData = {
      ...form,
      repeat_config: form.schedule_type === 'once' ? null : form.repeat_config
    }
    
    if (isEdit.value) {
      await schedulingApi.updateTask(props.task.id, submitData)
      emit('task-updated', submitData)
    } else {
      const response = await schedulingApi.createTask(submitData)
      emit('task-created', response.data)
    }
    
    handleClose()
    
  } catch (error) {
    if (error !== 'validation failed') {
      ElMessage.error(isEdit.value ? '更新任务失败' : '创建任务失败')
    }
  } finally {
    submitting.value = false
  }
}

const handleClose = () => {
  visible.value = false
  if (!isEdit.value) {
    resetForm()
  }
}
</script>

<style scoped>
.content-option {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.content-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.content-type {
  font-size: 12px;
  color: #909399;
  margin-left: 8px;
}

.platform-option {
  display: flex;
  align-items: center;
  gap: 8px;
}

.platform-icon {
  font-size: 16px;
}

.repeat-config {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  padding: 16px;
  background: #fafafa;
}

.config-item {
  margin-top: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.info-icon {
  margin-left: 4px;
  color: #909399;
  cursor: help;
}

.optimization-content {
  margin-top: 8px;
}

.suggested-times {
  display: flex;
  gap: 12px;
  margin-top: 12px;
  flex-wrap: wrap;
}

.time-suggestion {
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  padding: 12px;
  cursor: pointer;
  transition: all 0.3s;
  background: white;
  min-width: 160px;
}

.time-suggestion:hover {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
}

.suggestion-time {
  font-weight: 600;
  color: #303133;
  margin-bottom: 4px;
}

.suggestion-score {
  font-size: 12px;
  color: #67c23a;
  margin-bottom: 4px;
}

.suggestion-reason {
  font-size: 12px;
  color: #909399;
}

.dialog-footer {
  text-align: right;
}
</style>
```

## 验收标准

### 功能验收标准

1. **排程任务管理**
   - ✅ 能够创建单次和重复排程任务
   - ✅ 支持多平台同时排程
   - ✅ 提供任务编辑和取消功能
   - ✅ 实时显示任务执行状态和进度

2. **智能优化功能**
   - ✅ 基于历史数据提供发布时间优化建议
   - ✅ 支持用户手动启用/禁用优化功能
   - ✅ 显示优化建议的置信度和原因

3. **排程冲突处理**
   - ✅ 自动检测时间冲突、频率限制冲突
   - ✅ 提供冲突解决策略（延迟、取消、合并）
   - ✅ 记录冲突处理历史

4. **批量操作**
   - ✅ 支持批量创建、取消、重新排程
   - ✅ 提供批量导入/导出功能
   - ✅ 支持排程模板的创建和应用

### 性能验收标准

1. **响应时间**
   - 任务创建响应时间 < 2秒
   - 任务列表加载时间 < 3秒
   - 状态更新延迟 < 30秒

2. **并发处理**
   - 支持同时处理1000+排程任务
   - 支持100+用户同时操作
   - 任务执行准确率 > 99.5%

3. **系统稳定性**
   - 服务可用性 > 99.9%
   - 任务调度精度误差 < 30秒
   - 系统恢复时间 < 5分钟

### 安全验收标准

1. **权限控制**
   - ✅ 用户只能管理自己的排程任务
   - ✅ 平台账号权限验证
   - ✅ 操作日志完整记录

2. **数据安全**
   - ✅ 敏感信息加密存储
   - ✅ API接口安全认证
   - ✅ 防止恶意批量操作

## 商业价值

### 直接价值

1. **效率提升**
   - 减少90%的手动发布操作时间
   - 提高内容发布的及时性和准确性
   - 支持7x24小时自动化运营

2. **成本节约**
   - 减少人工运营成本60%
   - 降低发布错误率80%
   - 提高运营团队工作效率

3. **效果优化**
   - 基于数据的最佳发布时间选择
   - 提升内容互动率15-25%
   - 增加内容曝光量和影响力

### 间接价值

1. **用户体验**
   - 提供一致的内容发布体验
   - 减少用户等待时间
   - 提高平台活跃度

2. **数据洞察**
   - 积累发布效果数据
   - 支持运营策略优化
   - 为AI推荐提供数据基础

3. **业务扩展**
   - 支持多平台业务拓展
   - 提供标准化运营流程
   - 为企业客户提供专业服务

## 依赖关系

### 技术依赖

1. **基础服务**
   - 用户认证服务
   - 内容管理服务
   - 多平台发布服务
   - 账号管理服务

2. **第三方服务**
   - 各平台API接口
   - 消息队列服务
   - 监控告警服务
   - 数据分析服务

3. **数据依赖**
   - 历史发布数据
   - 用户行为数据
   - 平台限制规则
   - 内容分类数据

### 业务依赖

1. **运营流程**
   - 内容审核流程
   - 发布规范制定
   - 应急处理机制

2. **团队协作**
   - 产品团队需求确认
   - 运营团队使用培训
   - 技术团队维护支持

### 环境依赖

1. **基础设施**
   - 高可用服务器集群
   - 稳定的网络连接
   - 充足的存储空间

2. **监控体系**
   - 实时监控系统
   - 日志收集分析
   - 告警通知机制

## 风险评估

### 技术风险

1. **高风险**
   - 任务调度精度问题
   - 平台API变更影响
   - 大规模并发处理

2. **中风险**
   - 数据一致性问题
   - 系统性能瓶颈
   - 第三方服务依赖

3. **低风险**
   - 界面交互优化
   - 功能扩展需求
   - 用户体验改进

### 业务风险

1. **高风险**
   - 发布内容合规性
   - 平台政策变化
   - 用户数据安全

2. **中风险**
   - 用户接受度
   - 竞争对手压力
   - 运营成本控制

3. **低风险**
   - 功能使用率
   - 用户反馈处理
   - 产品迭代速度

### 运营风险

1. **高风险**
   - 系统故障影响
   - 数据丢失风险
   - 安全漏洞威胁

2. **中风险**
   - 人员技能要求
   - 维护成本增加
   - 用户培训需求

3. **低风险**
   - 功能使用指导
   - 用户支持服务
   - 产品推广策略

## 开发任务分解

### 后端开发任务

#### 第一阶段：核心服务开发 (2周)

1. **排程服务核心** (5天)
   - 实现ContentSchedulingService类
   - 任务创建、更新、删除功能
   - 任务状态管理和查询
   - 数据库模型设计和实现

2. **调度引擎开发** (4天)
   - 集成APScheduler调度器
   - 实现任务执行逻辑
   - 错误处理和重试机制
   - 任务监控和日志记录

3. **API接口开发** (3天)
   - 排程管理REST API
   - 请求验证和响应格式
   - 错误处理和状态码
   - API文档编写

4. **数据库设计** (2天)
   - PostgreSQL表结构设计
   - Redis缓存结构设计
   - 数据迁移脚本
   - 索引优化

#### 第二阶段：智能优化功能 (1.5周)

1. **优化算法开发** (4天)
   - 实现ScheduleOptimizer类
   - 机器学习模型训练
   - 特征工程和数据处理
   - 预测准确性验证

2. **优化建议API** (3天)
   - 优化建议生成接口
   - 历史数据分析接口
   - 性能指标统计接口
   - 建议应用接口

#### 第三阶段：高级功能开发 (1.5周)

1. **冲突检测处理** (3天)
   - 时间冲突检测算法
   - 频率限制检测
   - 内容相似性分析
   - 冲突解决策略

2. **批量操作功能** (3天)
   - 批量任务创建
   - 批量状态更新
   - 导入导出功能
   - 模板管理系统

3. **监控和告警** (1天)
   - 系统监控指标
   - 告警规则配置
   - 性能数据收集
   - 健康检查接口

### 前端开发任务

#### 第一阶段：基础页面开发 (1.5周)

1. **主页面开发** (3天)
   - 排程管理主页面布局
   - 统计卡片组件
   - 筛选和搜索功能
   - 任务列表展示

2. **任务操作功能** (3天)
   - 任务详情查看
   - 任务编辑功能
   - 任务取消操作
   - 批量操作界面

#### 第二阶段：创建和编辑功能 (1.5周)

1. **创建排程对话框** (4天)
   - 表单设计和验证
   - 平台选择组件
   - 时间选择器
   - 重复配置界面

2. **优化建议功能** (2天)
   - 优化建议展示
   - 建议应用交互
   - 置信度可视化
   - 历史对比分析

#### 第三阶段：高级功能和优化 (1周)

1. **日历视图** (2天)
   - 日历组件集成
   - 任务标记显示
   - 日期点击交互
   - 月份切换功能

2. **批量操作界面** (2天)
   - 批量选择功能
   - 批量操作按钮
   - 进度显示
   - 结果反馈

3. **用户体验优化** (1天)
   - 加载状态优化
   - 错误提示改进
   - 响应式布局
   - 性能优化

### 测试任务

#### 单元测试 (1周)

1. **后端单元测试** (3天)
   - 服务类测试
   - API接口测试
   - 数据库操作测试
   - 算法逻辑测试

2. **前端单元测试** (2天)
   - 组件功能测试
   - 状态管理测试
   - 工具函数测试
   - 用户交互测试

#### 集成测试 (1周)

1. **API集成测试** (2天)
   - 接口联调测试
   - 数据流测试
   - 错误场景测试
   - 性能压力测试

2. **端到端测试** (3天)
   - 完整流程测试
   - 用户场景测试
   - 浏览器兼容测试
   - 移动端适配测试

### 部署任务

#### 环境准备 (0.5周)

1. **基础设施** (2天)
   - 服务器环境配置
   - 数据库部署
   - 缓存服务配置
   - 监控系统部署

2. **CI/CD配置** (1天)
   - 构建流水线
   - 自动化测试
   - 部署脚本
   - 回滚机制

## 时间估算

### 总体时间安排

- **后端开发**: 5周
- **前端开发**: 4周
- **测试阶段**: 2周
- **部署上线**: 0.5周
- **总计**: 6.5周 (考虑并行开发)

### 人力资源需求

1. **后端开发工程师**: 2人
2. **前端开发工程师**: 2人
3. **测试工程师**: 1人
4. **DevOps工程师**: 1人
5. **产品经理**: 1人 (兼职)
6. **UI/UX设计师**: 1人 (兼职)

### 关键里程碑

1. **第2周末**: 后端核心服务完成
2. **第3周末**: 前端基础页面完成
3. **第4周末**: 智能优化功能完成
4. **第5周末**: 所有功能开发完成
5. **第6周末**: 测试完成，准备上线
6. **第7周**: 正式上线运行

## 成功指标

### 技术指标

1. **功能完整性**: 100%需求实现
2. **代码质量**: 测试覆盖率 > 80%
3. **性能指标**: 响应时间 < 2秒
4. **稳定性**: 系统可用性 > 99.9%
5. **安全性**: 0安全漏洞

### 业务指标

1. **用户采用率**: 30天内50%用户使用
2. **任务成功率**: 排程任务执行成功率 > 95%
3. **效率提升**: 发布效率提升 > 80%
4. **用户满意度**: 用户评分 > 4.5/5
5. **错误率**: 用户操作错误率 < 5%

### 运营指标

1. **系统监控**: 24/7监控覆盖
2. **响应时间**: 问题响应时间 < 1小时
3. **文档完整性**: 100%功能有文档
4. **培训效果**: 用户培训通过率 > 90%
5. **维护成本**: 月维护成本控制在预算内