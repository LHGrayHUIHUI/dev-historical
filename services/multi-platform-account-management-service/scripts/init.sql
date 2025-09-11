-- 多平台账号管理服务数据库初始化脚本

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 创建数据库（如果不存在）
-- 注意：这个脚本会在数据库创建后运行，所以数据库已经存在

-- 设置时区
SET timezone = 'UTC';

-- 创建枚举类型
DO $$ BEGIN
    CREATE TYPE platform_type AS ENUM ('social_media', 'blog', 'news', 'short_video', 'content');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE account_status AS ENUM ('active', 'suspended', 'expired', 'error', 'pending');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE verification_status AS ENUM ('unverified', 'verified', 'pending_verification');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE account_type AS ENUM ('personal', 'business', 'creator', 'organization');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE sync_type AS ENUM ('profile', 'stats', 'posts', 'followers', 'full');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE sync_status AS ENUM ('success', 'failed', 'partial', 'in_progress');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE permission_type AS ENUM ('read', 'write', 'admin', 'publish', 'manage');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 插入初始平台配置数据
INSERT INTO platforms (name, display_name, platform_type, api_base_url, oauth_config, rate_limits, features, is_active) 
VALUES 
    -- 新浪微博
    ('weibo', '新浪微博', 'social_media', 'https://api.weibo.com', 
     '{
        "authorize_url": "https://api.weibo.com/oauth2/authorize",
        "token_url": "https://api.weibo.com/oauth2/access_token",
        "user_info_url": "https://api.weibo.com/2/users/show.json",
        "client_id": "your_weibo_client_id",
        "client_secret": "your_weibo_client_secret",
        "scope": "read,write"
     }'::jsonb,
     '{
        "window_seconds": 3600,
        "max_requests": 1000,
        "min_interval_seconds": 1
     }'::jsonb,
     '{
        "supports_followers": true,
        "supports_posts": true,
        "supports_stats": true,
        "max_posts_per_request": 200
     }'::jsonb,
     true),
     
    -- 微信公众号
    ('wechat', '微信公众号', 'social_media', 'https://api.weixin.qq.com',
     '{
        "authorize_url": "https://open.weixin.qq.com/connect/oauth2/authorize",
        "token_url": "https://api.weixin.qq.com/sns/oauth2/access_token",
        "user_info_url": "https://api.weixin.qq.com/cgi-bin/user/info",
        "client_id": "your_wechat_appid",
        "client_secret": "your_wechat_secret",
        "scope": "snsapi_base"
     }'::jsonb,
     '{
        "window_seconds": 3600,
        "max_requests": 500,
        "min_interval_seconds": 2
     }'::jsonb,
     '{
        "supports_followers": true,
        "supports_posts": false,
        "supports_stats": true,
        "max_followers_per_request": 100
     }'::jsonb,
     true),
     
    -- 抖音
    ('douyin', '抖音', 'short_video', 'https://open.douyin.com',
     '{
        "authorize_url": "https://open.douyin.com/platform/oauth/connect",
        "token_url": "https://open.douyin.com/oauth/access_token",
        "user_info_url": "https://open.douyin.com/oauth/userinfo",
        "client_id": "your_douyin_client_key",
        "client_secret": "your_douyin_client_secret",
        "scope": "user_info,video.list"
     }'::jsonb,
     '{
        "window_seconds": 3600,
        "max_requests": 800,
        "min_interval_seconds": 1
     }'::jsonb,
     '{
        "supports_followers": true,
        "supports_posts": true,
        "supports_stats": true,
        "max_videos_per_request": 20
     }'::jsonb,
     true),
     
    -- 今日头条
    ('toutiao', '今日头条', 'news', 'https://open.toutiao.com',
     '{
        "authorize_url": "https://open.toutiao.com/oauth/authorize",
        "token_url": "https://open.toutiao.com/oauth/token",
        "user_info_url": "https://open.toutiao.com/oauth/userinfo",
        "client_id": "your_toutiao_client_id",
        "client_secret": "your_toutiao_client_secret",
        "scope": "user_info,article.list"
     }'::jsonb,
     '{
        "window_seconds": 3600,
        "max_requests": 600,
        "min_interval_seconds": 2
     }'::jsonb,
     '{
        "supports_followers": false,
        "supports_posts": true,
        "supports_stats": true,
        "max_articles_per_request": 20
     }'::jsonb,
     true),
     
    -- 百家号
    ('baijiahao', '百家号', 'content', 'https://openapi.baidu.com',
     '{
        "authorize_url": "https://openapi.baidu.com/oauth/2.0/authorize",
        "token_url": "https://openapi.baidu.com/oauth/2.0/token",
        "user_info_url": "https://openapi.baidu.com/rest/2.0/cambrian/app_info",
        "client_id": "your_baijiahao_client_id",
        "client_secret": "your_baijiahao_client_secret",
        "scope": "basic,article"
     }'::jsonb,
     '{
        "window_seconds": 3600,
        "max_requests": 400,
        "min_interval_seconds": 3
     }'::jsonb,
     '{
        "supports_followers": false,
        "supports_posts": true,
        "supports_stats": true,
        "max_articles_per_request": 20
     }'::jsonb,
     true)
ON CONFLICT (name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    platform_type = EXCLUDED.platform_type,
    api_base_url = EXCLUDED.api_base_url,
    oauth_config = EXCLUDED.oauth_config,
    rate_limits = EXCLUDED.rate_limits,
    features = EXCLUDED.features,
    is_active = EXCLUDED.is_active,
    updated_at = CURRENT_TIMESTAMP;

-- 创建索引以提高查询性能
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_user_id ON accounts(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_platform_id ON accounts(platform_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_status ON accounts(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_created_at ON accounts(created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_credentials_account_id ON account_credentials(account_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_credentials_expires_at ON account_credentials(expires_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_sync_logs_account_id ON account_sync_logs(account_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_sync_logs_created_at ON account_sync_logs(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_sync_logs_status ON account_sync_logs(status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_permissions_user_id ON account_permissions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_account_permissions_account_id ON account_permissions(account_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_usage_stats_platform_id ON api_usage_stats(platform_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_usage_stats_request_date ON api_usage_stats(request_date DESC);

-- 创建视图以简化查询
CREATE OR REPLACE VIEW account_with_platform AS
SELECT 
    a.*,
    p.name as platform_name,
    p.display_name as platform_display_name,
    p.platform_type
FROM accounts a
JOIN platforms p ON a.platform_id = p.id;

-- 创建函数和触发器
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为相关表创建更新时间触发器
DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_platforms_updated_at ON platforms;
    CREATE TRIGGER update_platforms_updated_at
        BEFORE UPDATE ON platforms
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_accounts_updated_at ON accounts;
    CREATE TRIGGER update_accounts_updated_at
        BEFORE UPDATE ON accounts
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    DROP TRIGGER IF EXISTS update_account_credentials_updated_at ON account_credentials;
    CREATE TRIGGER update_account_credentials_updated_at
        BEFORE UPDATE ON account_credentials
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 创建示例用户数据（可选，用于测试）
-- 注意：在生产环境中应该删除这部分
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'accounts' AND table_schema = 'public') THEN
        -- 插入示例账号（仅在开发环境）
        INSERT INTO accounts (platform_id, user_id, account_name, account_id, display_name, bio, follower_count, following_count, post_count, verification_status, account_type, status)
        SELECT 
            p.id,
            1001, -- 示例用户ID
            '示例账号_' || p.name,
            'example_' || p.name || '_id',
            '示例' || p.display_name || '账号',
            '这是一个示例账号，用于测试',
            1000 + (p.id * 100),
            500 + (p.id * 50),
            100 + (p.id * 20),
            'unverified',
            'personal',
            'active'
        FROM platforms p
        WHERE p.is_active = true
        ON CONFLICT (platform_id, user_id, account_name) DO NOTHING;
    END IF;
END $$;

-- 输出初始化完成信息
DO $$ BEGIN
    RAISE NOTICE '多平台账号管理服务数据库初始化完成';
    RAISE NOTICE '已创建 % 个平台配置', (SELECT COUNT(*) FROM platforms WHERE is_active = true);
END $$;