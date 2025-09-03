"""
内容数据模型定义
用于数据源服务中的内容结构化存储和传输
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class ContentStatus(str, Enum):
    """内容状态枚举"""
    PENDING = "pending"      # 待处理
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 处理失败
    REJECTED = "rejected"    # 已拒绝


class ContentSource(str, Enum):
    """内容来源枚举"""
    TOUTIAO = "toutiao"      # 今日头条
    BAIJIAHAO = "baijiahao"  # 百家号
    XIAOHONGSHU = "xiaohongshu"  # 小红书
    MANUAL = "manual"        # 手动添加
    RSS = "rss"             # RSS订阅
    API = "api"             # API接入


class ContentType(str, Enum):
    """内容类型枚举"""
    ARTICLE = "article"      # 文章
    VIDEO = "video"         # 视频
    IMAGE = "image"         # 图片
    AUDIO = "audio"         # 音频
    OTHER = "other"         # 其他


class ContentBase(BaseModel):
    """内容基础模型"""
    title: str = Field(..., description="内容标题", max_length=500)
    content: str = Field(..., description="内容正文")
    source: ContentSource = Field(..., description="内容来源")
    author: Optional[str] = Field(None, description="作者", max_length=100)
    source_url: Optional[HttpUrl] = Field(None, description="原始链接")
    publish_time: Optional[datetime] = Field(None, description="发布时间")
    content_type: ContentType = Field(ContentType.ARTICLE, description="内容类型")
    
    # 媒体文件
    images: List[HttpUrl] = Field(default_factory=list, description="相关图片链接")
    videos: List[HttpUrl] = Field(default_factory=list, description="相关视频链接")
    
    # 元数据
    keywords: List[str] = Field(default_factory=list, description="关键词")
    tags: List[str] = Field(default_factory=list, description="标签")
    category: Optional[str] = Field(None, description="分类", max_length=100)
    
    # 统计数据
    view_count: Optional[int] = Field(None, description="浏览量", ge=0)
    like_count: Optional[int] = Field(None, description="点赞量", ge=0)
    comment_count: Optional[int] = Field(None, description="评论量", ge=0)
    share_count: Optional[int] = Field(None, description="分享量", ge=0)
    
    # 质量评分 (0-100)
    quality_score: Optional[float] = Field(None, description="质量评分", ge=0, le=100)
    
    @validator("content")
    def validate_content_length(cls, v):
        """验证内容长度"""
        if len(v.strip()) < 10:
            raise ValueError("内容长度不能少于10个字符")
        return v.strip()
    
    @validator("keywords", "tags")
    def validate_lists_length(cls, v):
        """验证列表长度"""
        return v[:20]  # 最多20个关键词/标签


class ContentCreate(ContentBase):
    """创建内容请求模型"""
    # 爬虫相关配置
    crawler_config: Optional[Dict[str, Any]] = Field(None, description="爬虫配置")
    priority: int = Field(1, description="处理优先级", ge=1, le=10)
    
    # 处理选项
    auto_process: bool = Field(True, description="是否自动处理")
    extract_images: bool = Field(True, description="是否提取图片")
    generate_summary: bool = Field(True, description="是否生成摘要")


class ContentUpdate(BaseModel):
    """更新内容请求模型"""
    title: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = None
    author: Optional[str] = Field(None, max_length=100)
    keywords: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = Field(None, max_length=100)
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    
    @validator("keywords", "tags")
    def validate_lists_length(cls, v):
        """验证列表长度"""
        if v is not None:
            return v[:20]
        return v


class ContentInDB(ContentBase):
    """数据库中的内容模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="唯一标识")
    status: ContentStatus = Field(ContentStatus.PENDING, description="处理状态")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    processed_at: Optional[datetime] = Field(None, description="处理完成时间")
    
    # 处理信息
    processor_version: Optional[str] = Field(None, description="处理器版本")
    processing_time: Optional[float] = Field(None, description="处理耗时(秒)", ge=0)
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 统计信息
    download_count: int = Field(0, description="下载次数", ge=0)
    last_accessed: Optional[datetime] = Field(None, description="最后访问时间")
    
    # 内容指纹 (用于去重)
    content_hash: Optional[str] = Field(None, description="内容哈希")
    similarity_hash: Optional[str] = Field(None, description="相似性哈希")
    
    # 额外元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    class Config:
        """Pydantic配置"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContentResponse(ContentInDB):
    """内容响应模型"""
    # 可以添加响应特定的字段
    summary: Optional[str] = Field(None, description="内容摘要", max_length=1000)
    extracted_entities: List[str] = Field(default_factory=list, description="提取的实体")
    sentiment_score: Optional[float] = Field(None, description="情感分数", ge=-1, le=1)


class ContentBatchCreate(BaseModel):
    """批量创建内容请求模型"""
    contents: List[ContentCreate] = Field(..., description="内容列表", max_items=100)
    batch_name: Optional[str] = Field(None, description="批次名称", max_length=200)
    auto_deduplicate: bool = Field(True, description="是否自动去重")
    
    @validator("contents")
    def validate_contents_not_empty(cls, v):
        """验证内容列表不为空"""
        if not v:
            raise ValueError("内容列表不能为空")
        return v


class ContentBatchResponse(BaseModel):
    """批量操作响应模型"""
    batch_id: str = Field(..., description="批次ID")
    total_count: int = Field(..., description="总数量")
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")
    created_ids: List[str] = Field(..., description="成功创建的内容ID列表")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="错误列表")


class ContentFilter(BaseModel):
    """内容过滤条件模型"""
    source: Optional[ContentSource] = None
    content_type: Optional[ContentType] = None
    status: Optional[ContentStatus] = None
    author: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    
    # 时间范围
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # 数值范围
    min_quality_score: Optional[float] = Field(None, ge=0, le=100)
    max_quality_score: Optional[float] = Field(None, ge=0, le=100)
    min_view_count: Optional[int] = Field(None, ge=0)
    
    # 分页
    skip: int = Field(0, ge=0, description="跳过数量")
    limit: int = Field(20, ge=1, le=100, description="限制数量")
    
    # 排序
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="排序顺序")


class ContentStats(BaseModel):
    """内容统计模型"""
    total_count: int = Field(..., description="总数量")
    status_counts: Dict[str, int] = Field(..., description="按状态统计")
    source_counts: Dict[str, int] = Field(..., description="按来源统计")
    type_counts: Dict[str, int] = Field(..., description="按类型统计")
    
    # 时间统计
    today_count: int = Field(..., description="今日新增")
    week_count: int = Field(..., description="本周新增")
    month_count: int = Field(..., description="本月新增")
    
    # 质量统计
    avg_quality_score: Optional[float] = Field(None, description="平均质量分")
    high_quality_count: int = Field(..., description="高质量内容数量(>80分)")
    
    # 处理统计
    avg_processing_time: Optional[float] = Field(None, description="平均处理时间(秒)")
    success_rate: float = Field(..., description="处理成功率", ge=0, le=1)