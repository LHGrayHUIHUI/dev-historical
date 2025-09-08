-- OCR服务数据库迁移脚本
-- 创建OCR相关的数据表和索引
-- 版本: 001
-- 创建时间: 2025-01-15
-- 描述: 初始化OCR任务、结果和配置表

-- 创建任务状态枚举类型
CREATE TYPE task_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');

-- 创建OCR引擎枚举类型
CREATE TYPE ocr_engine AS ENUM ('paddleocr', 'tesseract', 'easyocr');

-- 创建OCR任务表
CREATE TABLE ocr_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID,
    image_path VARCHAR(500) NOT NULL,
    image_size JSONB,
    processing_status task_status DEFAULT 'pending' NOT NULL,
    ocr_engine ocr_engine DEFAULT 'paddleocr' NOT NULL,
    confidence_threshold FLOAT DEFAULT 0.8 NOT NULL,
    language_codes VARCHAR(100) DEFAULT 'zh,en' NOT NULL,
    preprocessing_config JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_by UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- 添加表注释
COMMENT ON TABLE ocr_tasks IS 'OCR识别任务表';
COMMENT ON COLUMN ocr_tasks.id IS '任务ID';
COMMENT ON COLUMN ocr_tasks.dataset_id IS '数据集ID';
COMMENT ON COLUMN ocr_tasks.image_path IS '图像文件路径';
COMMENT ON COLUMN ocr_tasks.image_size IS '图像尺寸信息';
COMMENT ON COLUMN ocr_tasks.processing_status IS '处理状态';
COMMENT ON COLUMN ocr_tasks.ocr_engine IS '使用的OCR引擎';
COMMENT ON COLUMN ocr_tasks.confidence_threshold IS '置信度阈值';
COMMENT ON COLUMN ocr_tasks.language_codes IS '支持的语言代码';
COMMENT ON COLUMN ocr_tasks.preprocessing_config IS '预处理配置参数';
COMMENT ON COLUMN ocr_tasks.started_at IS '处理开始时间';
COMMENT ON COLUMN ocr_tasks.completed_at IS '处理完成时间';
COMMENT ON COLUMN ocr_tasks.error_message IS '错误信息';
COMMENT ON COLUMN ocr_tasks.created_by IS '创建者用户ID';
COMMENT ON COLUMN ocr_tasks.created_at IS '创建时间';
COMMENT ON COLUMN ocr_tasks.updated_at IS '更新时间';

-- 创建OCR结果表
CREATE TABLE ocr_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL UNIQUE REFERENCES ocr_tasks(id) ON DELETE CASCADE,
    text_content TEXT NOT NULL,
    confidence_score FLOAT NOT NULL,
    bounding_boxes JSONB,
    text_blocks JSONB,
    language_detected VARCHAR(50),
    word_count INTEGER DEFAULT 0 NOT NULL,
    char_count INTEGER DEFAULT 0 NOT NULL,
    processing_time FLOAT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- 全文搜索生成列
    text_vector tsvector GENERATED ALWAYS AS (to_tsvector('chinese', text_content)) STORED
);

-- 添加表注释
COMMENT ON TABLE ocr_results IS 'OCR识别结果表';
COMMENT ON COLUMN ocr_results.id IS '结果ID';
COMMENT ON COLUMN ocr_results.task_id IS '关联的任务ID';
COMMENT ON COLUMN ocr_results.text_content IS '识别的完整文本内容';
COMMENT ON COLUMN ocr_results.confidence_score IS '整体置信度分数';
COMMENT ON COLUMN ocr_results.bounding_boxes IS '文字区域边界框坐标';
COMMENT ON COLUMN ocr_results.text_blocks IS '文本块详细信息';
COMMENT ON COLUMN ocr_results.language_detected IS '检测到的语言';
COMMENT ON COLUMN ocr_results.word_count IS '词语数量';
COMMENT ON COLUMN ocr_results.char_count IS '字符数量';
COMMENT ON COLUMN ocr_results.processing_time IS '处理时间（秒）';
COMMENT ON COLUMN ocr_results.metadata IS '额外的元数据信息';
COMMENT ON COLUMN ocr_results.text_vector IS '全文搜索向量';

-- 创建OCR配置表
CREATE TABLE ocr_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    engine ocr_engine NOT NULL,
    config JSONB NOT NULL,
    is_default BOOLEAN DEFAULT false NOT NULL,
    is_active BOOLEAN DEFAULT true NOT NULL,
    created_by UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- 添加表注释
COMMENT ON TABLE ocr_configs IS 'OCR配置预设表';
COMMENT ON COLUMN ocr_configs.id IS '配置ID';
COMMENT ON COLUMN ocr_configs.name IS '配置名称';
COMMENT ON COLUMN ocr_configs.description IS '配置描述';
COMMENT ON COLUMN ocr_configs.engine IS 'OCR引擎类型';
COMMENT ON COLUMN ocr_configs.config IS '引擎配置参数JSON';
COMMENT ON COLUMN ocr_configs.is_default IS '是否为默认配置';
COMMENT ON COLUMN ocr_configs.is_active IS '是否启用';
COMMENT ON COLUMN ocr_configs.created_by IS '创建者用户ID';

-- 创建索引
-- OCR任务表索引
CREATE INDEX idx_ocr_tasks_status ON ocr_tasks(processing_status);
CREATE INDEX idx_ocr_tasks_created_by ON ocr_tasks(created_by);
CREATE INDEX idx_ocr_tasks_engine ON ocr_tasks(ocr_engine);
CREATE INDEX idx_ocr_tasks_created_at ON ocr_tasks(created_at DESC);
CREATE INDEX idx_ocr_tasks_dataset ON ocr_tasks(dataset_id) WHERE dataset_id IS NOT NULL;

-- OCR结果表索引
CREATE INDEX idx_ocr_results_task_id ON ocr_results(task_id);
CREATE INDEX idx_ocr_results_confidence ON ocr_results(confidence_score);
CREATE INDEX idx_ocr_results_char_count ON ocr_results(char_count);
CREATE INDEX idx_ocr_results_created_at ON ocr_results(created_at DESC);

-- 全文搜索索引
CREATE INDEX idx_ocr_results_text_search ON ocr_results USING GIN(text_vector);
CREATE INDEX idx_ocr_results_text_content ON ocr_results USING GIN(text_content gin_trgm_ops);

-- OCR配置表索引
CREATE INDEX idx_ocr_configs_engine ON ocr_configs(engine);
CREATE INDEX idx_ocr_configs_active ON ocr_configs(is_active) WHERE is_active = true;
CREATE INDEX idx_ocr_configs_default ON ocr_configs(is_default) WHERE is_default = true;

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为各表添加更新时间触发器
CREATE TRIGGER update_ocr_tasks_updated_at 
    BEFORE UPDATE ON ocr_tasks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ocr_results_updated_at 
    BEFORE UPDATE ON ocr_results 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ocr_configs_updated_at 
    BEFORE UPDATE ON ocr_configs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认OCR配置
INSERT INTO ocr_configs (name, description, engine, config, is_default, is_active, created_by) VALUES 
(
    'PaddleOCR默认配置',
    'PaddleOCR引擎的默认配置，适用于中文和英文文档识别',
    'paddleocr',
    '{
        "use_angle_cls": true,
        "lang": "ch",
        "use_gpu": true,
        "det_model_dir": null,
        "rec_model_dir": null,
        "cls_model_dir": null,
        "preprocessing": {
            "grayscale": true,
            "denoise": true,
            "enhance_contrast": true,
            "deskew": true,
            "binarize": false,
            "resize": false,
            "scale_factor": 2.0
        },
        "post_processing": {
            "traditional_to_simplified": true,
            "remove_extra_whitespace": true,
            "normalize_punctuation": true,
            "spell_check": false
        }
    }',
    true,
    true,
    '00000000-0000-0000-0000-000000000000'
),
(
    'Tesseract中文配置',
    'Tesseract引擎的中文优化配置',
    'tesseract',
    '{
        "lang": "chi_sim+eng",
        "oem": 3,
        "psm": 6,
        "config": "--dpi 300",
        "preprocessing": {
            "grayscale": true,
            "denoise": true,
            "enhance_contrast": true,
            "deskew": true,
            "binarize": true,
            "resize": true,
            "scale_factor": 2.0
        },
        "post_processing": {
            "traditional_to_simplified": true,
            "remove_extra_whitespace": true,
            "normalize_punctuation": true,
            "spell_check": false
        }
    }',
    false,
    true,
    '00000000-0000-0000-0000-000000000000'
),
(
    'EasyOCR多语言配置',
    'EasyOCR引擎的多语言支持配置',
    'easyocr',
    '{
        "lang_list": ["ch_sim", "ch_tra", "en"],
        "gpu": true,
        "detail": 1,
        "preprocessing": {
            "grayscale": false,
            "denoise": true,
            "enhance_contrast": true,
            "deskew": true,
            "binarize": false,
            "resize": false,
            "scale_factor": 1.0
        },
        "post_processing": {
            "traditional_to_simplified": true,
            "remove_extra_whitespace": true,
            "normalize_punctuation": true,
            "spell_check": false
        }
    }',
    false,
    true,
    '00000000-0000-0000-0000-000000000000'
);

-- 创建视图：任务统计
CREATE VIEW ocr_task_stats AS
SELECT 
    processing_status,
    ocr_engine,
    COUNT(*) as task_count,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_processing_time,
    DATE_TRUNC('day', created_at) as date
FROM ocr_tasks 
WHERE started_at IS NOT NULL
GROUP BY processing_status, ocr_engine, DATE_TRUNC('day', created_at);

COMMENT ON VIEW ocr_task_stats IS 'OCR任务统计视图';

-- 创建函数：搜索OCR结果
CREATE OR REPLACE FUNCTION search_ocr_results(
    search_query TEXT,
    user_id UUID DEFAULT NULL,
    limit_count INTEGER DEFAULT 20,
    offset_count INTEGER DEFAULT 0
)
RETURNS TABLE(
    id UUID,
    task_id UUID,
    text_content TEXT,
    confidence_score FLOAT,
    char_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    rank FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.id,
        r.task_id,
        r.text_content,
        r.confidence_score,
        r.char_count,
        r.created_at,
        ts_rank(r.text_vector, plainto_tsquery('chinese', search_query)) as rank
    FROM ocr_results r
    JOIN ocr_tasks t ON r.task_id = t.id
    WHERE 
        r.text_vector @@ plainto_tsquery('chinese', search_query)
        AND (user_id IS NULL OR t.created_by = user_id)
    ORDER BY rank DESC, r.created_at DESC
    LIMIT limit_count OFFSET offset_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_ocr_results IS 'OCR结果全文搜索函数';