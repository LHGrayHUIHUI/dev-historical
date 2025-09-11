-- 自动内容调度服务数据库初始化脚本

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 设置时区
SET timezone = 'UTC';

-- 创建枚举类型
DO $$ BEGIN
    CREATE TYPE task_status AS ENUM ('pending', 'scheduled', 'running', 'completed', 'failed', 'cancelled', 'paused', 'retrying');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE task_type AS ENUM ('single', 'recurring', 'batch', 'template');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE conflict_type AS ENUM ('time_overlap', 'resource_conflict', 'platform_limit', 'content_duplicate', 'user_preference');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE conflict_severity AS ENUM ('low', 'medium', 'high', 'critical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE optimization_status AS ENUM ('not_optimized', 'optimizing', 'optimized', 'optimization_failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 创建调度任务表
CREATE TABLE IF NOT EXISTS scheduling_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- 任务配置
    task_type task_type NOT NULL DEFAULT 'single',
    status task_status NOT NULL DEFAULT 'pending',
    
    -- 内容信息
    content_id VARCHAR(255),
    content_title VARCHAR(500),
    content_body TEXT,
    content_metadata JSONB,
    
    -- 调度时间
    scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    actual_execution_time TIMESTAMP WITH TIME ZONE,
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 平台配置
    target_platforms TEXT[],
    platform_configs JSONB,
    
    -- 循环任务配置
    recurrence_rule VARCHAR(255),
    recurrence_end_date TIMESTAMP WITH TIME ZONE,
    max_occurrences INTEGER,
    occurrence_count INTEGER DEFAULT 0,
    
    -- 优化配置
    optimization_enabled BOOLEAN DEFAULT true,
    optimization_status optimization_status DEFAULT 'not_optimized',
    optimization_score FLOAT,
    original_scheduled_time TIMESTAMP WITH TIME ZONE,
    
    -- 执行配置
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    max_retries INTEGER DEFAULT 3 CHECK (max_retries >= 0),
    current_retries INTEGER DEFAULT 0,
    retry_delay INTEGER DEFAULT 300,
    
    -- 模板配置
    template_id UUID REFERENCES scheduling_templates(id),
    is_template_instance BOOLEAN DEFAULT false,
    
    CONSTRAINT check_occurrence_count CHECK (occurrence_count >= 0)
);

-- 创建调度模板表
CREATE TABLE IF NOT EXISTS scheduling_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- 模板配置
    template_config JSONB NOT NULL,
    default_platforms TEXT[],
    default_timing JSONB,
    
    -- 使用统计
    usage_count INTEGER DEFAULT 0,
    last_used_time TIMESTAMP WITH TIME ZONE,
    
    -- 时间戳
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 状态
    is_active BOOLEAN DEFAULT true,
    is_public BOOLEAN DEFAULT false,
    
    CONSTRAINT uq_user_template_name UNIQUE (user_id, name)
);

-- 创建调度冲突表
CREATE TABLE IF NOT EXISTS scheduling_conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES scheduling_tasks(id) ON DELETE CASCADE,
    
    -- 冲突信息
    conflict_type conflict_type NOT NULL,
    severity conflict_severity NOT NULL,
    description TEXT,
    
    -- 冲突对象
    conflicted_task_id UUID REFERENCES scheduling_tasks(id),
    conflicted_resource VARCHAR(255),
    platform_name VARCHAR(50),
    
    -- 冲突详情
    conflict_details JSONB,
    suggested_resolution JSONB,
    
    -- 处理状态
    is_resolved BOOLEAN DEFAULT false,
    resolution_method VARCHAR(255),
    resolved_time TIMESTAMP WITH TIME ZONE,
    resolved_by INTEGER,
    
    -- 时间戳
    detected_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建任务执行日志表
CREATE TABLE IF NOT EXISTS task_execution_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES scheduling_tasks(id) ON DELETE CASCADE,
    
    -- 执行信息
    execution_start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    execution_end_time TIMESTAMP WITH TIME ZONE,
    execution_duration FLOAT,
    
    -- 执行状态
    status task_status NOT NULL,
    platform_results JSONB,
    
    -- 错误信息
    error_message TEXT,
    error_details JSONB,
    stack_trace TEXT,
    
    -- 执行统计
    successful_platforms INTEGER DEFAULT 0,
    failed_platforms INTEGER DEFAULT 0
);

-- 创建优化日志表
CREATE TABLE IF NOT EXISTS optimization_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES scheduling_tasks(id) ON DELETE CASCADE,
    
    -- 优化信息
    optimization_type VARCHAR(50) NOT NULL,
    original_scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    optimized_scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- 优化指标
    optimization_score FLOAT,
    predicted_engagement FLOAT,
    predicted_reach FLOAT,
    confidence_score FLOAT,
    
    -- 优化详情
    optimization_factors JSONB,
    model_version VARCHAR(50),
    model_params JSONB,
    
    -- 时间戳
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建调度队列表
CREATE TABLE IF NOT EXISTS scheduling_queues (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES scheduling_tasks(id) ON DELETE CASCADE,
    
    -- 队列信息
    queue_name VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 5,
    
    -- 调度信息
    scheduled_execution_time TIMESTAMP WITH TIME ZONE NOT NULL,
    actual_execution_time TIMESTAMP WITH TIME ZONE,
    
    -- 状态信息
    status VARCHAR(50) DEFAULT 'queued',
    retry_count INTEGER DEFAULT 0,
    
    -- 时间戳
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_task_queue UNIQUE (task_id)
);

-- 创建平台指标表
CREATE TABLE IF NOT EXISTS platform_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    platform_name VARCHAR(50) NOT NULL,
    user_id INTEGER NOT NULL,
    account_id VARCHAR(255),
    
    -- 时间维度
    metric_date TIMESTAMP WITH TIME ZONE NOT NULL,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    
    -- 参与度指标
    total_posts INTEGER DEFAULT 0,
    total_views INTEGER DEFAULT 0,
    total_likes INTEGER DEFAULT 0,
    total_shares INTEGER DEFAULT 0,
    total_comments INTEGER DEFAULT 0,
    total_clicks INTEGER DEFAULT 0,
    
    -- 计算指标
    engagement_rate FLOAT DEFAULT 0.0,
    click_through_rate FLOAT DEFAULT 0.0,
    conversion_rate FLOAT DEFAULT 0.0,
    reach_rate FLOAT DEFAULT 0.0,
    
    -- 质量指标
    average_post_score FLOAT,
    user_satisfaction FLOAT,
    
    -- 时间戳
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_platform_metric_key UNIQUE (platform_name, user_id, account_id, metric_date, hour_of_day)
);

-- 创建内容性能表
CREATE TABLE IF NOT EXISTS content_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES scheduling_tasks(id) ON DELETE CASCADE,
    content_id VARCHAR(255),
    
    -- 平台信息
    platform_name VARCHAR(50) NOT NULL,
    platform_post_id VARCHAR(255),
    
    -- 发布信息
    published_time TIMESTAMP WITH TIME ZONE,
    scheduled_time TIMESTAMP WITH TIME ZONE,
    time_variance FLOAT,
    
    -- 内容特征
    content_type VARCHAR(50),
    content_length INTEGER,
    has_media BOOLEAN DEFAULT false,
    media_count INTEGER DEFAULT 0,
    hashtag_count INTEGER DEFAULT 0,
    mention_count INTEGER DEFAULT 0,
    
    -- 性能数据（24小时内）
    views_1h INTEGER DEFAULT 0,
    views_6h INTEGER DEFAULT 0,
    views_24h INTEGER DEFAULT 0,
    
    likes_1h INTEGER DEFAULT 0,
    likes_6h INTEGER DEFAULT 0,
    likes_24h INTEGER DEFAULT 0,
    
    shares_1h INTEGER DEFAULT 0,
    shares_6h INTEGER DEFAULT 0,
    shares_24h INTEGER DEFAULT 0,
    
    comments_1h INTEGER DEFAULT 0,
    comments_6h INTEGER DEFAULT 0,
    comments_24h INTEGER DEFAULT 0,
    
    -- 计算指标
    peak_engagement_time TIMESTAMP WITH TIME ZONE,
    engagement_score FLOAT,
    virality_score FLOAT,
    quality_score FLOAT,
    
    -- 预测vs实际
    predicted_performance JSONB,
    actual_performance JSONB,
    prediction_accuracy FLOAT,
    
    -- 时间戳
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_content_platform_performance UNIQUE (task_id, platform_name)
);

-- 创建用户行为模式表
CREATE TABLE IF NOT EXISTS user_behavior_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL,
    platform_name VARCHAR(50),
    
    -- 时间模式
    preferred_hours INTEGER[],
    preferred_days INTEGER[],
    peak_engagement_hours INTEGER[],
    
    -- 内容偏好
    preferred_content_types JSONB,
    optimal_content_length JSONB,
    media_preference JSONB,
    
    -- 频率模式
    average_posting_frequency FLOAT,
    optimal_posting_interval INTEGER,
    
    -- 性能模式
    average_engagement_rate FLOAT,
    best_performing_times JSONB,
    worst_performing_times JSONB,
    
    -- 统计信息
    total_posts_analyzed INTEGER DEFAULT 0,
    pattern_confidence FLOAT,
    last_analysis_date TIMESTAMP WITH TIME ZONE,
    
    -- 时间戳
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_user_platform_behavior UNIQUE (user_id, platform_name)
);

-- 创建ML模型指标表
CREATE TABLE IF NOT EXISTS ml_model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- 训练信息
    training_start_time TIMESTAMP WITH TIME ZONE,
    training_end_time TIMESTAMP WITH TIME ZONE,
    training_duration FLOAT,
    training_samples INTEGER,
    validation_samples INTEGER,
    
    -- 模型参数
    model_parameters JSONB,
    feature_importance JSONB,
    
    -- 性能指标
    accuracy_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    mse_score FLOAT,
    r2_score FLOAT,
    
    -- 交叉验证指标
    cv_mean_score FLOAT,
    cv_std_score FLOAT,
    
    -- 业务指标
    prediction_accuracy FLOAT,
    improvement_rate FLOAT,
    
    -- 模型状态
    is_active BOOLEAN DEFAULT false,
    deployment_time TIMESTAMP WITH TIME ZONE,
    
    -- 时间戳
    created_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_model_version UNIQUE (model_name, model_version)
);

-- 创建系统指标表
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(50) NOT NULL,
    service_name VARCHAR(100),
    
    -- 时间维度
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 系统指标
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_usage FLOAT,
    network_io JSONB,
    
    -- 数据库指标
    db_connections INTEGER,
    db_query_time FLOAT,
    cache_hit_rate FLOAT,
    
    -- API指标
    api_request_count INTEGER,
    api_response_time FLOAT,
    api_error_rate FLOAT,
    
    -- 任务队列指标
    queue_length INTEGER,
    task_processing_time FLOAT,
    task_success_rate FLOAT,
    
    -- 业务指标
    active_users INTEGER,
    scheduled_tasks INTEGER,
    completed_tasks INTEGER,
    failed_tasks INTEGER
);

-- 创建索引以提高查询性能
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_tasks_user_id ON scheduling_tasks(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_tasks_status ON scheduling_tasks(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_tasks_scheduled_time ON scheduling_tasks(scheduled_time);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_tasks_created_time ON scheduling_tasks(created_time DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_tasks_platforms ON scheduling_tasks USING gin(target_platforms);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_templates_user_id ON scheduling_templates(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_templates_name ON scheduling_templates(name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_templates_usage ON scheduling_templates(usage_count);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_conflicts_task_id ON scheduling_conflicts(task_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_conflicts_type ON scheduling_conflicts(conflict_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_conflicts_severity ON scheduling_conflicts(severity);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_conflicts_platform ON scheduling_conflicts(platform_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_conflicts_resolved ON scheduling_conflicts(is_resolved);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_execution_logs_task_id ON task_execution_logs(task_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_execution_logs_status ON task_execution_logs(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_execution_logs_start_time ON task_execution_logs(execution_start_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_optimization_logs_task_id ON optimization_logs(task_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_optimization_logs_type ON optimization_logs(optimization_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_optimization_logs_score ON optimization_logs(optimization_score);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_queues_execution_time ON scheduling_queues(scheduled_execution_time);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_queues_priority ON scheduling_queues(priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_queues_status ON scheduling_queues(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scheduling_queues_queue_name ON scheduling_queues(queue_name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_platform_metrics_platform ON platform_metrics(platform_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_platform_metrics_user_id ON platform_metrics(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_platform_metrics_date ON platform_metrics(metric_date);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_platform_metrics_hour ON platform_metrics(hour_of_day);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_platform_metrics_engagement ON platform_metrics(engagement_rate);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_content_performance_task_id ON content_performance(task_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_content_performance_platform ON content_performance(platform_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_content_performance_published_time ON content_performance(published_time);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_content_performance_engagement ON content_performance(engagement_score);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_behavior_user_id ON user_behavior_patterns(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_behavior_platform ON user_behavior_patterns(platform_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_behavior_confidence ON user_behavior_patterns(pattern_confidence);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_name ON ml_model_metrics(model_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_version ON ml_model_metrics(model_version);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_accuracy ON ml_model_metrics(accuracy_score);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_model_metrics_active ON ml_model_metrics(is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_type ON system_metrics(metric_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_service ON system_metrics(service_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- 创建视图以简化查询
CREATE OR REPLACE VIEW task_with_conflicts AS
SELECT 
    t.*,
    COALESCE(c.conflict_count, 0) as conflict_count,
    COALESCE(c.critical_conflicts, 0) as critical_conflicts
FROM scheduling_tasks t
LEFT JOIN (
    SELECT 
        task_id,
        COUNT(*) as conflict_count,
        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_conflicts
    FROM scheduling_conflicts
    WHERE is_resolved = false
    GROUP BY task_id
) c ON t.id = c.task_id;

-- 创建函数和触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_time = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为相关表创建更新时间触发器
DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_scheduling_tasks_updated_at ON scheduling_tasks;
    CREATE TRIGGER update_scheduling_tasks_updated_at
        BEFORE UPDATE ON scheduling_tasks
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_scheduling_templates_updated_at ON scheduling_templates;
    CREATE TRIGGER update_scheduling_templates_updated_at
        BEFORE UPDATE ON scheduling_templates
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_scheduling_queues_updated_at ON scheduling_queues;
    CREATE TRIGGER update_scheduling_queues_updated_at
        BEFORE UPDATE ON scheduling_queues
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_platform_metrics_updated_at ON platform_metrics;
    CREATE TRIGGER update_platform_metrics_updated_at
        BEFORE UPDATE ON platform_metrics
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_content_performance_updated_at ON content_performance;
    CREATE TRIGGER update_content_performance_updated_at
        BEFORE UPDATE ON content_performance
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_user_behavior_patterns_updated_at ON user_behavior_patterns;
    CREATE TRIGGER update_user_behavior_patterns_updated_at
        BEFORE UPDATE ON user_behavior_patterns
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 输出初始化完成信息
DO $$ BEGIN
    RAISE NOTICE '自动内容调度服务数据库初始化完成';
    RAISE NOTICE '创建了调度任务相关的所有表和索引';
END $$;