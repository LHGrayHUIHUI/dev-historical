-- 分析报告服务PostgreSQL初始化脚本

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS historical_text_analytics;

-- 切换到目标数据库
\c historical_text_analytics;

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 创建用户和权限
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'analytics_user') THEN
        CREATE USER analytics_user WITH PASSWORD 'analytics_pass';
    END IF;
END
$$;

-- 授权
GRANT ALL PRIVILEGES ON DATABASE historical_text_analytics TO analytics_user;
GRANT ALL ON SCHEMA public TO analytics_user;

-- 创建分析任务状态类型
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'analysis_task_status_enum') THEN
        CREATE TYPE analysis_task_status_enum AS ENUM (
            'pending', 'running', 'completed', 'failed', 'cancelled', 'paused'
        );
    END IF;
END
$$;

-- 创建分析任务类型枚举
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'analysis_task_type_enum') THEN
        CREATE TYPE analysis_task_type_enum AS ENUM (
            'content_performance', 'platform_comparison', 'trend_analysis', 
            'user_behavior', 'anomaly_detection', 'forecast', 'custom'
        );
    END IF;
END
$$;

-- 创建告警严重程度枚举
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_severity_enum') THEN
        CREATE TYPE alert_severity_enum AS ENUM ('low', 'medium', 'high', 'critical');
    END IF;
END
$$;

-- 创建分析任务表
CREATE TABLE IF NOT EXISTS analysis_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    task_type analysis_task_type_enum NOT NULL,
    status analysis_task_status_enum NOT NULL DEFAULT 'pending',
    
    -- 用户信息
    user_id VARCHAR(50) NOT NULL,
    created_by VARCHAR(100),
    
    -- 配置信息
    config JSONB DEFAULT '{}',
    parameters JSONB DEFAULT '{}',
    filters JSONB DEFAULT '{}',
    
    -- 时间范围
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    
    -- 执行信息
    scheduled_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- 结果信息
    result_data JSONB DEFAULT '{}',
    error_message TEXT,
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    
    -- 优先级和资源
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    estimated_duration INTEGER,
    actual_duration INTEGER,
    
    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- 创建报告模板表
CREATE TABLE IF NOT EXISTS report_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    
    -- 用户信息
    user_id VARCHAR(50),
    is_public BOOLEAN DEFAULT FALSE,
    
    -- 模板配置
    template_config JSONB NOT NULL DEFAULT '{}',
    chart_configs JSONB DEFAULT '{}',
    layout_config JSONB DEFAULT '{}',
    
    -- 数据配置
    data_sources JSONB DEFAULT '{}',
    default_filters JSONB DEFAULT '{}',
    
    -- 样式配置
    theme VARCHAR(50) DEFAULT 'default',
    custom_styles JSONB DEFAULT '{}',
    
    -- 使用统计
    usage_count INTEGER DEFAULT 0,
    
    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- 创建生成报告表
CREATE TABLE IF NOT EXISTS generated_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- 关联信息
    analysis_task_id UUID REFERENCES analysis_tasks(id),
    template_id UUID REFERENCES report_templates(id),
    
    -- 基本信息
    title VARCHAR(255) NOT NULL,
    description TEXT,
    user_id VARCHAR(50) NOT NULL,
    
    -- 报告内容
    content_data JSONB DEFAULT '{}',
    chart_data JSONB DEFAULT '{}',
    summary TEXT,
    
    -- 时间范围
    report_period_start TIMESTAMP WITH TIME ZONE,
    report_period_end TIMESTAMP WITH TIME ZONE,
    
    -- 文件信息
    file_path VARCHAR(500),
    file_size INTEGER,
    file_format VARCHAR(20),
    
    -- 状态信息
    generation_status VARCHAR(20) DEFAULT 'generating',
    is_shared BOOLEAN DEFAULT FALSE,
    
    -- 访问统计
    view_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    
    -- 审计字段
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- 创建数据源配置表
CREATE TABLE IF NOT EXISTS data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    source_type VARCHAR(50) NOT NULL,
    
    -- 连接配置
    connection_config JSONB NOT NULL DEFAULT '{}',
    credentials JSONB DEFAULT '{}',
    
    -- 数据配置
    schema_config JSONB DEFAULT '{}',
    refresh_interval INTEGER DEFAULT 3600,
    
    -- 状态信息
    is_active BOOLEAN DEFAULT TRUE,
    last_sync_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(20) DEFAULT 'pending',
    
    -- 用户信息
    user_id VARCHAR(50) NOT NULL,
    
    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- 创建告警规则表
CREATE TABLE IF NOT EXISTS alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- 规则配置
    metric_name VARCHAR(100) NOT NULL,
    condition VARCHAR(50) NOT NULL,
    threshold_value DOUBLE PRECISION NOT NULL,
    
    -- 时间窗口
    evaluation_window INTEGER DEFAULT 300,
    evaluation_frequency INTEGER DEFAULT 60,
    
    -- 告警配置
    severity alert_severity_enum NOT NULL,
    notification_channels JSONB DEFAULT '{}',
    
    -- 过滤条件
    filters JSONB DEFAULT '{}',
    
    -- 状态信息
    is_active BOOLEAN DEFAULT TRUE,
    last_triggered_at TIMESTAMP WITH TIME ZONE,
    trigger_count INTEGER DEFAULT 0,
    
    -- 用户信息
    user_id VARCHAR(50) NOT NULL,
    
    -- 审计字段
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- 创建告警历史表
CREATE TABLE IF NOT EXISTS alert_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_rule_id UUID NOT NULL REFERENCES alert_rules(id),
    
    -- 告警信息
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL,
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- 告警数据
    actual_value DOUBLE PRECISION,
    threshold_value DOUBLE PRECISION,
    severity alert_severity_enum NOT NULL,
    
    -- 通知状态
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channels JSONB DEFAULT '{}',
    
    -- 用户信息
    user_id VARCHAR(50) NOT NULL
);

-- 创建索引
-- 分析任务索引
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_user_status ON analysis_tasks(user_id, status);
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_type_created ON analysis_tasks(task_type, created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_scheduled ON analysis_tasks(scheduled_at) WHERE scheduled_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_deleted ON analysis_tasks(deleted_at);

-- 报告模板索引
CREATE INDEX IF NOT EXISTS idx_report_templates_user_public ON report_templates(user_id, is_public);
CREATE INDEX IF NOT EXISTS idx_report_templates_category ON report_templates(category);
CREATE INDEX IF NOT EXISTS idx_report_templates_deleted ON report_templates(deleted_at);

-- 生成报告索引
CREATE INDEX IF NOT EXISTS idx_generated_reports_user_generated ON generated_reports(user_id, generated_at);
CREATE INDEX IF NOT EXISTS idx_generated_reports_task ON generated_reports(analysis_task_id);
CREATE INDEX IF NOT EXISTS idx_generated_reports_expires ON generated_reports(expires_at) WHERE expires_at IS NOT NULL;

-- 数据源索引
CREATE INDEX IF NOT EXISTS idx_data_sources_user_active ON data_sources(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_data_sources_type ON data_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_data_sources_deleted ON data_sources(deleted_at);

-- 告警规则索引
CREATE INDEX IF NOT EXISTS idx_alert_rules_user_active ON alert_rules(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_alert_rules_metric ON alert_rules(metric_name);
CREATE INDEX IF NOT EXISTS idx_alert_rules_last_triggered ON alert_rules(last_triggered_at);
CREATE INDEX IF NOT EXISTS idx_alert_rules_deleted ON alert_rules(deleted_at);

-- 告警历史索引
CREATE INDEX IF NOT EXISTS idx_alert_history_rule_triggered ON alert_history(alert_rule_id, triggered_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_user ON alert_history(user_id);

-- 创建唯一约束
ALTER TABLE data_sources ADD CONSTRAINT uk_data_sources_user_name 
    UNIQUE (user_id, name) WHERE deleted_at IS NULL;

-- 创建触发器函数 - 更新时间戳
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为相关表创建触发器
CREATE TRIGGER update_analysis_tasks_updated_at 
    BEFORE UPDATE ON analysis_tasks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_report_templates_updated_at 
    BEFORE UPDATE ON report_templates 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_sources_updated_at 
    BEFORE UPDATE ON data_sources 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alert_rules_updated_at 
    BEFORE UPDATE ON alert_rules 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入示例数据
INSERT INTO report_templates (name, description, category, is_public, template_config) VALUES
(
    '内容表现分析报告',
    '分析内容在各平台的表现指标',
    'content_analysis',
    TRUE,
    '{
        "sections": [
            {"type": "summary", "title": "概览"},
            {"type": "chart", "title": "表现趋势", "chart_type": "line"},
            {"type": "table", "title": "详细数据"}
        ],
        "theme": "professional"
    }'
),
(
    '平台对比分析报告',
    '对比不同平台的数据表现',
    'platform_comparison',
    TRUE,
    '{
        "sections": [
            {"type": "summary", "title": "平台概览"},
            {"type": "chart", "title": "平台对比", "chart_type": "bar"},
            {"type": "insights", "title": "洞察建议"}
        ],
        "theme": "modern"
    }'
);

-- 创建一些示例告警规则
INSERT INTO alert_rules (name, description, metric_name, condition, threshold_value, severity, user_id) VALUES
(
    '浏览量异常下降',
    '当浏览量比前一天下降超过30%时触发告警',
    'daily_views',
    'less_than',
    0.7,
    'high',
    'system'
),
(
    '参与度过低',
    '当参与度低于1%时触发告警',
    'engagement_rate',
    'less_than',
    0.01,
    'medium',
    'system'
);

-- 设置表权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO analytics_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO analytics_user;

-- 完成初始化
SELECT 'PostgreSQL database initialization completed successfully!' as status;