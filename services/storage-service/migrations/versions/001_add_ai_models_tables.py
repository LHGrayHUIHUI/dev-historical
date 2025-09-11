"""Add AI models and system prompts tables

Revision ID: 001_ai_models
Revises: 
Create Date: 2025-09-10 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_ai_models'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create AI models configuration and system prompts tables"""
    
    # 创建AI提供商枚举类型
    ai_provider_enum = postgresql.ENUM(
        'openai', 'gemini', 'claude', 'local', 'azure_openai', 'huggingface',
        name='ai_provider_enum'
    )
    ai_provider_enum.create(op.get_bind(), checkfirst=True)
    
    # 创建模型状态枚举类型
    model_status_enum = postgresql.ENUM(
        'configured', 'needs_api_key', 'disabled', 'error',
        name='model_status_enum'
    )
    model_status_enum.create(op.get_bind(), checkfirst=True)
    
    # 创建AI模型配置表
    op.create_table(
        'ai_model_configs',
        # 基础字段
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        
        # 基本信息
        sa.Column('alias', sa.String(100), unique=True, nullable=False, index=True, comment='模型别名，用户自定义的模型标识符'),
        sa.Column('provider', ai_provider_enum, nullable=False, index=True, comment='AI提供商类型'),
        sa.Column('model_name', sa.String(200), nullable=False, comment='实际的模型名称，如gpt-4、gemini-1.5-pro等'),
        sa.Column('display_name', sa.String(200), nullable=True, comment='模型显示名称'),
        sa.Column('description', sa.Text, nullable=True, comment='模型描述信息'),
        
        # 连接配置
        sa.Column('api_key', sa.Text, nullable=True, comment='加密存储的API密钥'),
        sa.Column('api_base', sa.String(500), nullable=True, comment='API基础URL端点'),
        sa.Column('api_version', sa.String(50), nullable=True, comment='API版本号'),
        
        # 模型参数配置
        sa.Column('max_tokens', sa.Integer, default=4096, nullable=False, comment='最大输出token数量限制'),
        sa.Column('context_window', sa.Integer, nullable=True, comment='上下文窗口大小'),
        sa.Column('default_temperature', sa.Float, default=0.7, nullable=False, comment='默认温度参数，控制生成随机性'),
        sa.Column('default_top_p', sa.Float, nullable=True, comment='默认top_p参数'),
        
        # 部署配置
        sa.Column('is_local', sa.Boolean, default=False, nullable=False, comment='是否为本地部署模型'),
        sa.Column('is_streaming_supported', sa.Boolean, default=True, nullable=False, comment='是否支持流式响应'),
        
        # 扩展配置
        sa.Column('custom_parameters', postgresql.JSON, nullable=True, comment='自定义参数配置，JSON格式存储'),
        sa.Column('auth_headers', postgresql.JSON, nullable=True, comment='自定义认证头信息'),
        
        # 管理信息
        sa.Column('status', model_status_enum, default='needs_api_key', nullable=False, index=True, comment='模型配置状态'),
        sa.Column('priority', sa.Integer, default=0, nullable=False, comment='模型优先级，用于智能路由选择'),
        sa.Column('tags', postgresql.JSON, nullable=True, comment='模型标签，用于分类和管理'),
        
        # 监控信息
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True, comment='最后使用时间'),
        sa.Column('usage_count', sa.Integer, default=0, nullable=False, comment='使用次数统计'),
        sa.Column('error_count', sa.Integer, default=0, nullable=False, comment='错误次数统计'),
        sa.Column('last_error', sa.Text, nullable=True, comment='最后一次错误信息'),
        
        # 性能配置
        sa.Column('request_timeout', sa.Integer, nullable=True, comment='请求超时时间（秒）'),
        sa.Column('max_retries', sa.Integer, default=3, nullable=False, comment='最大重试次数'),
        sa.Column('rate_limit_per_minute', sa.Integer, nullable=True, comment='每分钟请求限制'),
        
        comment='AI模型配置表'
    )
    
    # 创建系统提示语模板表
    op.create_table(
        'system_prompt_templates',
        # 基础字段
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        
        # 基本信息
        sa.Column('name', sa.String(100), unique=True, nullable=False, index=True, comment='提示语模板名称，唯一标识符'),
        sa.Column('display_name', sa.String(200), nullable=False, comment='显示名称，用户友好的名称'),
        sa.Column('description', sa.Text, nullable=True, comment='提示语模板描述'),
        
        # 提示语内容
        sa.Column('prompt_content', sa.Text, nullable=False, comment='提示语内容，支持模板变量'),
        sa.Column('language', sa.String(10), default='zh-CN', nullable=False, index=True, comment='提示语语言，如zh-CN、en-US等'),
        
        # 分类和标签
        sa.Column('category', sa.String(50), nullable=False, index=True, comment='提示语分类，如\'助手\'、\'翻译\'、\'写作\'等'),
        sa.Column('tags', postgresql.JSON, nullable=True, comment='提示语标签，用于搜索和分类'),
        
        # 模板变量
        sa.Column('variables', postgresql.JSON, nullable=True, comment='模板变量定义，包含变量名、类型、默认值等'),
        sa.Column('example_variables', postgresql.JSON, nullable=True, comment='示例变量值，用于演示和测试'),
        
        # 使用配置
        sa.Column('is_active', sa.Boolean, default=True, nullable=False, index=True, comment='是否激活状态'),
        sa.Column('is_public', sa.Boolean, default=True, nullable=False, comment='是否公开可用'),
        sa.Column('priority', sa.Integer, default=0, nullable=False, comment='优先级，用于排序显示'),
        
        # 使用统计
        sa.Column('usage_count', sa.Integer, default=0, nullable=False, comment='使用次数统计'),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True, comment='最后使用时间'),
        
        # 版本控制
        sa.Column('version', sa.String(20), default='1.0.0', nullable=False, comment='模板版本号'),
        sa.Column('created_by', sa.String(100), nullable=True, comment='创建者'),
        
        comment='系统提示语模板表'
    )
    
    # 创建索引
    op.create_index('ix_ai_model_configs_provider_status', 'ai_model_configs', ['provider', 'status'])
    op.create_index('ix_ai_model_configs_priority_created', 'ai_model_configs', ['priority', 'created_at'])
    op.create_index('ix_system_prompt_templates_category_lang', 'system_prompt_templates', ['category', 'language'])
    op.create_index('ix_system_prompt_templates_active_public', 'system_prompt_templates', ['is_active', 'is_public'])


def downgrade() -> None:
    """Drop AI models configuration and system prompts tables"""
    
    # 删除索引
    op.drop_index('ix_system_prompt_templates_active_public')
    op.drop_index('ix_system_prompt_templates_category_lang')
    op.drop_index('ix_ai_model_configs_priority_created')
    op.drop_index('ix_ai_model_configs_provider_status')
    
    # 删除表
    op.drop_table('system_prompt_templates')
    op.drop_table('ai_model_configs')
    
    # 删除枚举类型
    op.execute('DROP TYPE IF EXISTS model_status_enum')
    op.execute('DROP TYPE IF EXISTS ai_provider_enum')