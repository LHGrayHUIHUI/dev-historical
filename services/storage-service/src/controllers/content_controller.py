"""
内容管理控制器
统一管理所有内容数据的CRUD操作和业务逻辑
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging
import httpx
from pydantic import BaseModel, HttpUrl, Field

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/content", tags=["内容管理"])


class ContentBase(BaseModel):
    """内容基础模型"""
    title: str = Field(..., description="内容标题", max_length=500)
    content: str = Field(..., description="内容正文")
    source: str = Field(..., description="内容来源")
    author: Optional[str] = Field(None, description="作者", max_length=100)
    source_url: Optional[HttpUrl] = Field(None, description="原始链接")
    
    # 媒体文件URL引用
    images: List[HttpUrl] = Field(default_factory=list, description="图片文件URL链接")
    videos: List[HttpUrl] = Field(default_factory=list, description="视频文件URL链接")
    
    # 元数据
    keywords: List[str] = Field(default_factory=list, description="关键词")
    tags: List[str] = Field(default_factory=list, description="标签")
    category: Optional[str] = Field(None, description="分类", max_length=100)


class ContentCreate(ContentBase):
    """创建内容请求模型"""
    priority: int = Field(1, description="处理优先级", ge=1, le=10)
    auto_process: bool = Field(True, description="是否自动处理")
    generate_summary: bool = Field(True, description="是否生成摘要")


class ContentResponse(BaseModel):
    """内容响应模型"""
    id: str
    title: str
    content: str
    source: str
    author: Optional[str]
    source_url: Optional[str]
    images: List[str]
    videos: List[str]
    keywords: List[str]
    tags: List[str]
    category: Optional[str]
    status: str
    created_at: str
    updated_at: str
    summary: Optional[str] = None


class ContentFilter(BaseModel):
    """内容过滤条件模型"""
    source: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None
    skip: int = Field(0, ge=0, description="跳过数量")
    limit: int = Field(20, ge=1, le=100, description="限制数量")
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="排序顺序")


# 模拟内容存储（生产环境应该使用数据库）
content_store: Dict[str, ContentResponse] = {}


@router.post("/", response_model=ContentResponse)
async def create_content(content_data: ContentCreate):
    """
    创建新的内容记录
    """
    try:
        content_id = str(uuid.uuid4())
        
        # 创建内容记录
        content_response = ContentResponse(
            id=content_id,
            title=content_data.title,
            content=content_data.content,
            source=content_data.source,
            author=content_data.author,
            source_url=str(content_data.source_url) if content_data.source_url else None,
            images=[str(img) for img in content_data.images],
            videos=[str(vid) for vid in content_data.videos],
            keywords=content_data.keywords,
            tags=content_data.tags,
            category=content_data.category,
            status="active",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            summary=f"自动生成的摘要：{content_data.title}" if content_data.generate_summary else None
        )
        
        # 保存到存储
        content_store[content_id] = content_response
        
        logger.info(f"内容创建成功: {content_id} - {content_data.title}")
        
        return content_response
        
    except Exception as e:
        logger.error(f"创建内容失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建内容失败: {str(e)}")


@router.post("/with-files", response_model=ContentResponse)
async def create_content_with_files(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    content: str = Form(...),
    source: str = Form("manual"),
    author: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),  # 逗号分隔的关键词
    tags: Optional[str] = Form(None),      # 逗号分隔的标签
    image_files: List[UploadFile] = File(default=[]),
    video_files: List[UploadFile] = File(default=[])
):
    """
    创建内容并处理关联的文件
    这个端点会先处理文件，然后创建内容记录
    """
    try:
        content_id = str(uuid.uuid4())
        image_urls = []
        video_urls = []
        
        # 处理图片文件
        for image_file in image_files:
            if image_file.filename:
                # TODO: 调用file-processor服务处理图片
                # 这里应该调用 POST http://file-processor:8000/api/v1/process/image-ocr
                # async with httpx.AsyncClient() as client:
                #     files = {"file": (image_file.filename, image_file.file, image_file.content_type)}
                #     response = await client.post("http://file-processor:8000/api/v1/process/image-ocr", files=files)
                #     result = response.json()
                
                # 模拟文件处理和存储
                file_url = f"http://localhost:9001/historical-images/{datetime.now().strftime('%Y%m%d')}/{content_id}_{image_file.filename}"
                image_urls.append(file_url)
                logger.info(f"图片文件已处理: {image_file.filename} -> {file_url}")
        
        # 处理视频文件
        for video_file in video_files:
            if video_file.filename:
                # TODO: 存储视频文件到MinIO
                file_url = f"http://localhost:9001/historical-videos/{datetime.now().strftime('%Y%m%d')}/{content_id}_{video_file.filename}"
                video_urls.append(file_url)
                logger.info(f"视频文件已处理: {video_file.filename} -> {file_url}")
        
        # 解析关键词和标签
        keywords_list = [k.strip() for k in keywords.split(",")] if keywords else []
        tags_list = [t.strip() for t in tags.split(",")] if tags else []
        
        # 创建内容记录
        content_response = ContentResponse(
            id=content_id,
            title=title,
            content=content,
            source=source,
            author=author,
            source_url=None,
            images=image_urls,
            videos=video_urls,
            keywords=keywords_list,
            tags=tags_list,
            category=category,
            status="active",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            summary=f"包含{len(image_urls)}个图片和{len(video_urls)}个视频的内容"
        )
        
        # 保存到存储
        content_store[content_id] = content_response
        
        logger.info(f"多媒体内容创建成功: {content_id} - {title}")
        
        return content_response
        
    except Exception as e:
        logger.error(f"创建多媒体内容失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建内容失败: {str(e)}")


@router.get("/", response_model=List[ContentResponse])
async def get_contents(
    skip: int = 0,
    limit: int = 20,
    source: Optional[str] = None,
    category: Optional[str] = None,
    author: Optional[str] = None
):
    """
    获取内容列表，支持分页和过滤
    """
    try:
        contents = list(content_store.values())
        
        # 应用过滤器
        if source:
            contents = [c for c in contents if c.source == source]
        if category:
            contents = [c for c in contents if c.category == category]
        if author:
            contents = [c for c in contents if c.author == author]
        
        # 分页
        total = len(contents)
        contents = contents[skip:skip + limit]
        
        logger.info(f"获取内容列表: 总数{total}, 返回{len(contents)}条")
        
        return contents
        
    except Exception as e:
        logger.error(f"获取内容列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取内容列表失败")


@router.get("/{content_id}", response_model=ContentResponse)
async def get_content(content_id: str):
    """
    获取单个内容详情
    """
    if content_id not in content_store:
        raise HTTPException(status_code=404, detail="内容不存在")
    
    return content_store[content_id]


@router.put("/{content_id}", response_model=ContentResponse)
async def update_content(content_id: str, update_data: ContentCreate):
    """
    更新内容信息
    """
    if content_id not in content_store:
        raise HTTPException(status_code=404, detail="内容不存在")
    
    try:
        existing_content = content_store[content_id]
        
        # 更新字段
        updated_content = ContentResponse(
            id=content_id,
            title=update_data.title,
            content=update_data.content,
            source=update_data.source,
            author=update_data.author,
            source_url=str(update_data.source_url) if update_data.source_url else existing_content.source_url,
            images=[str(img) for img in update_data.images],
            videos=[str(vid) for vid in update_data.videos],
            keywords=update_data.keywords,
            tags=update_data.tags,
            category=update_data.category,
            status=existing_content.status,
            created_at=existing_content.created_at,
            updated_at=datetime.now().isoformat(),
            summary=existing_content.summary
        )
        
        content_store[content_id] = updated_content
        
        logger.info(f"内容更新成功: {content_id}")
        
        return updated_content
        
    except Exception as e:
        logger.error(f"更新内容失败: {e}")
        raise HTTPException(status_code=500, detail="更新内容失败")


@router.delete("/{content_id}")
async def delete_content(content_id: str):
    """
    删除内容
    """
    if content_id not in content_store:
        raise HTTPException(status_code=404, detail="内容不存在")
    
    try:
        deleted_content = content_store.pop(content_id)
        
        # TODO: 同时删除关联的文件
        # for image_url in deleted_content.images:
        #     await delete_file_from_storage(image_url)
        # for video_url in deleted_content.videos:
        #     await delete_file_from_storage(video_url)
        
        logger.info(f"内容删除成功: {content_id}")
        
        return {"success": True, "message": "内容删除成功", "deleted_id": content_id}
        
    except Exception as e:
        logger.error(f"删除内容失败: {e}")
        raise HTTPException(status_code=500, detail="删除内容失败")


@router.get("/search/")
async def search_contents(
    q: str,
    skip: int = 0,
    limit: int = 20
):
    """
    内容搜索
    """
    try:
        if not q or len(q.strip()) < 2:
            raise HTTPException(status_code=400, detail="搜索关键词至少2个字符")
        
        query = q.lower().strip()
        results = []
        
        # 在标题和内容中搜索
        for content in content_store.values():
            if (query in content.title.lower() or 
                query in content.content.lower() or
                any(query in keyword.lower() for keyword in content.keywords) or
                any(query in tag.lower() for tag in content.tags)):
                results.append(content)
        
        # 分页
        total = len(results)
        results = results[skip:skip + limit]
        
        logger.info(f"搜索内容: 关键词'{q}', 找到{total}条结果")
        
        return {
            "success": True,
            "data": {
                "query": q,
                "total": total,
                "results": results,
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "has_more": skip + limit < total
                }
            }
        }
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail="搜索失败")


@router.get("/stats/")
async def get_content_stats():
    """
    获取内容统计信息
    """
    try:
        contents = list(content_store.values())
        total_count = len(contents)
        
        # 按来源统计
        source_counts = {}
        category_counts = {}
        
        for content in contents:
            source_counts[content.source] = source_counts.get(content.source, 0) + 1
            if content.category:
                category_counts[content.category] = category_counts.get(content.category, 0) + 1
        
        # 媒体文件统计
        total_images = sum(len(content.images) for content in contents)
        total_videos = sum(len(content.videos) for content in contents)
        
        return {
            "success": True,
            "data": {
                "total_content": total_count,
                "source_distribution": source_counts,
                "category_distribution": category_counts,
                "media_stats": {
                    "total_images": total_images,
                    "total_videos": total_videos,
                    "content_with_media": len([c for c in contents if c.images or c.videos])
                },
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")