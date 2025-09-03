"""
内容管理API接口
提供内容的手动添加、查询、更新和删除接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
import uuid
import csv
import json
import io
from pydantic import BaseModel, Field

from ..models.content import (
    ContentCreate,
    ContentUpdate,
    ContentResponse,
    ContentFilter,
    ContentBatchCreate,
    ContentBatchResponse,
    ContentStats,
    ContentSource,
    ContentType,
    ContentStatus
)
from ..database.database import get_database_manager, DatabaseManager
from ..config.settings import get_settings

# 创建路由器
router = APIRouter(prefix="/content", tags=["内容管理"])


class ContentCreateRequest(ContentCreate):
    """内容创建请求模型 - 扩展基础模型"""
    pass


class ContentListResponse(BaseModel):
    """内容列表响应模型"""
    items: List[ContentResponse]
    total: int
    page: int
    size: int
    pages: int


@router.post("/", response_model=Dict[str, Any], summary="添加单个内容")
async def create_content(
    content: ContentCreateRequest,
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    手动添加单个内容
    
    - **title**: 内容标题
    - **content**: 内容正文
    - **source**: 内容来源
    - **author**: 作者（可选）
    - **source_url**: 原始链接（可选）
    - **publish_time**: 发布时间（可选）
    - **content_type**: 内容类型（默认为文章）
    """
    try:
        collection = await db_manager.get_mongodb_collection("contents")
        
        # 创建内容对象
        content_dict = content.dict()
        content_dict['id'] = str(uuid.uuid4())
        content_dict['status'] = ContentStatus.PENDING
        content_dict['created_at'] = datetime.now()
        content_dict['updated_at'] = datetime.now()
        
        # 生成内容哈希用于去重
        content_text = f"{content.title}{content.content}"
        content_dict['content_hash'] = str(hash(content_text))
        
        # 检查是否已存在相同内容
        existing = await collection.find_one({"content_hash": content_dict['content_hash']})
        if existing:
            return {
                "success": False,
                "message": "内容已存在",
                "data": {"existing_id": existing['id']}
            }
        
        # 插入到数据库
        result = await collection.insert_one(content_dict)
        
        if result.inserted_id:
            return {
                "success": True,
                "data": {
                    "id": content_dict['id'],
                    "title": content.title,
                    "status": ContentStatus.PENDING,
                    "created_at": content_dict['created_at']
                },
                "message": "内容添加成功"
            }
        else:
            raise HTTPException(status_code=500, detail="内容保存失败")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加内容失败: {str(e)}")


@router.post("/batch", response_model=Dict[str, Any], summary="批量添加内容")
async def create_batch_content(
    batch_request: ContentBatchCreate,
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    批量添加多个内容
    
    - **contents**: 内容列表（最多100个）
    - **batch_name**: 批次名称（可选）
    - **auto_deduplicate**: 是否自动去重（默认为true）
    """
    try:
        collection = await db_manager.get_mongodb_collection("contents")
        
        batch_id = str(uuid.uuid4())
        created_ids = []
        errors = []
        
        for i, content in enumerate(batch_request.contents):
            try:
                # 创建内容对象
                content_dict = content.dict()
                content_dict['id'] = str(uuid.uuid4())
                content_dict['status'] = ContentStatus.PENDING
                content_dict['created_at'] = datetime.now()
                content_dict['updated_at'] = datetime.now()
                content_dict['batch_id'] = batch_id
                
                # 生成内容哈希
                content_text = f"{content.title}{content.content}"
                content_dict['content_hash'] = str(hash(content_text))
                
                # 检查去重
                if batch_request.auto_deduplicate:
                    existing = await collection.find_one({"content_hash": content_dict['content_hash']})
                    if existing:
                        errors.append({
                            "index": i,
                            "title": content.title[:50] + "...",
                            "error": "内容已存在",
                            "existing_id": existing['id']
                        })
                        continue
                
                # 插入数据库
                result = await collection.insert_one(content_dict)
                if result.inserted_id:
                    created_ids.append(content_dict['id'])
                else:
                    errors.append({
                        "index": i,
                        "title": content.title[:50] + "...",
                        "error": "保存失败"
                    })
                    
            except Exception as e:
                errors.append({
                    "index": i,
                    "title": getattr(content, 'title', 'Unknown')[:50] + "...",
                    "error": str(e)
                })
        
        # 创建批量响应
        response = ContentBatchResponse(
            batch_id=batch_id,
            total_count=len(batch_request.contents),
            success_count=len(created_ids),
            failed_count=len(errors),
            created_ids=created_ids,
            errors=errors
        )
        
        return {
            "success": True,
            "data": response.dict(),
            "message": f"批量处理完成，成功: {response.success_count}, 失败: {response.failed_count}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量添加内容失败: {str(e)}")


@router.post("/upload", response_model=Dict[str, Any], summary="文件导入内容")
async def upload_content_file(
    file: UploadFile = File(..., description="上传文件（支持CSV、JSON格式）"),
    batch_name: Optional[str] = Form(None, description="批次名称"),
    auto_deduplicate: bool = Form(True, description="是否自动去重"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    通过文件导入内容（支持CSV和JSON格式）
    
    CSV格式要求：
    - title: 标题（必填）
    - content: 内容（必填）  
    - source: 来源（必填）
    - author: 作者（可选）
    - source_url: 原始链接（可选）
    - keywords: 关键词（逗号分隔，可选）
    
    JSON格式要求：
    - 数组格式，每个对象包含上述字段
    """
    try:
        if file.content_type not in ['text/csv', 'application/json', 'text/plain']:
            raise HTTPException(status_code=400, detail="不支持的文件格式，请上传CSV或JSON文件")
        
        # 读取文件内容
        content_bytes = await file.read()
        content_str = content_bytes.decode('utf-8')
        
        contents = []
        
        # 解析CSV文件
        if file.filename.endswith('.csv') or file.content_type == 'text/csv':
            csv_reader = csv.DictReader(io.StringIO(content_str))
            
            for row in csv_reader:
                try:
                    # 验证必填字段
                    if not all(row.get(field) for field in ['title', 'content', 'source']):
                        continue
                    
                    # 处理关键词
                    keywords = []
                    if row.get('keywords'):
                        keywords = [k.strip() for k in row['keywords'].split(',') if k.strip()]
                    
                    # 处理内容来源
                    try:
                        source = ContentSource(row['source'].lower())
                    except ValueError:
                        source = ContentSource.MANUAL
                    
                    content_obj = ContentCreate(
                        title=row['title'],
                        content=row['content'],
                        source=source,
                        author=row.get('author', ''),
                        source_url=row.get('source_url'),
                        keywords=keywords,
                        content_type=ContentType.ARTICLE
                    )
                    contents.append(content_obj)
                    
                except Exception as e:
                    print(f"解析CSV行失败: {e}")
                    continue
        
        # 解析JSON文件
        elif file.filename.endswith('.json') or file.content_type == 'application/json':
            try:
                json_data = json.loads(content_str)
                
                if not isinstance(json_data, list):
                    raise HTTPException(status_code=400, detail="JSON文件必须是数组格式")
                
                for item in json_data:
                    try:
                        # 验证必填字段
                        if not all(item.get(field) for field in ['title', 'content', 'source']):
                            continue
                        
                        # 处理内容来源
                        try:
                            source = ContentSource(item['source'].lower())
                        except ValueError:
                            source = ContentSource.MANUAL
                        
                        # 处理内容类型
                        try:
                            content_type = ContentType(item.get('content_type', 'article').lower())
                        except ValueError:
                            content_type = ContentType.ARTICLE
                        
                        content_obj = ContentCreate(
                            title=item['title'],
                            content=item['content'],
                            source=source,
                            author=item.get('author', ''),
                            source_url=item.get('source_url'),
                            keywords=item.get('keywords', []),
                            content_type=content_type,
                            tags=item.get('tags', []),
                            category=item.get('category')
                        )
                        contents.append(content_obj)
                        
                    except Exception as e:
                        print(f"解析JSON项失败: {e}")
                        continue
                        
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="JSON文件格式错误")
        
        if not contents:
            raise HTTPException(status_code=400, detail="文件中没有找到有效的内容数据")
        
        # 创建批量请求
        batch_request = ContentBatchCreate(
            contents=contents,
            batch_name=batch_name or f"文件导入_{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            auto_deduplicate=auto_deduplicate
        )
        
        # 调用批量创建接口
        return await create_batch_content(batch_request, db_manager)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件导入失败: {str(e)}")


@router.get("/", response_model=Dict[str, Any], summary="获取内容列表")
async def get_content_list(
    # 过滤参数
    status: Optional[ContentStatus] = Query(None, description="按状态过滤"),
    source: Optional[ContentSource] = Query(None, description="按来源过滤"),
    content_type: Optional[ContentType] = Query(None, description="按类型过滤"),
    author: Optional[str] = Query(None, description="按作者过滤"),
    category: Optional[str] = Query(None, description="按分类过滤"),
    keywords: Optional[str] = Query(None, description="关键词搜索（逗号分隔）"),
    
    # 时间范围
    start_date: Optional[date] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    
    # 质量分数范围
    min_quality_score: Optional[float] = Query(None, description="最小质量分数", ge=0, le=100),
    max_quality_score: Optional[float] = Query(None, description="最大质量分数", ge=0, le=100),
    
    # 浏览量范围
    min_view_count: Optional[int] = Query(None, description="最小浏览量", ge=0),
    
    # 分页参数
    page: int = Query(1, description="页码", ge=1),
    size: int = Query(20, description="每页数量", ge=1, le=100),
    
    # 排序参数
    sort_by: str = Query("created_at", description="排序字段"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="排序顺序"),
    
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    获取内容列表，支持多种过滤和排序选项
    """
    try:
        collection = await db_manager.get_mongodb_collection("contents")
        
        # 构建查询条件
        query = {}
        
        if status:
            query["status"] = status
        if source:
            query["source"] = source
        if content_type:
            query["content_type"] = content_type
        if author:
            query["author"] = {"$regex": author, "$options": "i"}
        if category:
            query["category"] = category
        
        # 关键词搜索
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            if keyword_list:
                query["$or"] = [
                    {"title": {"$regex": "|".join(keyword_list), "$options": "i"}},
                    {"content": {"$regex": "|".join(keyword_list), "$options": "i"}},
                    {"keywords": {"$in": keyword_list}}
                ]
        
        # 时间范围过滤
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = datetime.combine(start_date, datetime.min.time())
            if end_date:
                date_query["$lte"] = datetime.combine(end_date, datetime.max.time())
            query["created_at"] = date_query
        
        # 质量分数范围
        if min_quality_score is not None or max_quality_score is not None:
            quality_query = {}
            if min_quality_score is not None:
                quality_query["$gte"] = min_quality_score
            if max_quality_score is not None:
                quality_query["$lte"] = max_quality_score
            query["quality_score"] = quality_query
        
        # 浏览量范围
        if min_view_count is not None:
            query["view_count"] = {"$gte": min_view_count}
        
        # 排序
        sort_direction = 1 if sort_order == "asc" else -1
        sort_spec = [(sort_by, sort_direction)]
        
        # 获取总数
        total = await collection.count_documents(query)
        
        # 分页查询
        skip = (page - 1) * size
        cursor = collection.find(query).sort(sort_spec).skip(skip).limit(size)
        
        items = []
        async for doc in cursor:
            # 转换为响应模型
            content_response = ContentResponse(
                **doc,
                summary=doc.get('summary'),
                extracted_entities=doc.get('extracted_entities', []),
                sentiment_score=doc.get('sentiment_score')
            )
            items.append(content_response)
        
        # 计算总页数
        pages = (total + size - 1) // size
        
        return {
            "success": True,
            "data": {
                "items": [item.dict() for item in items],
                "total": total,
                "page": page,
                "size": size,
                "pages": pages
            },
            "message": "获取内容列表成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取内容列表失败: {str(e)}")


@router.get("/{content_id}", response_model=Dict[str, Any], summary="获取内容详情")
async def get_content_detail(
    content_id: str = Path(..., description="内容ID"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    获取指定内容的详细信息
    """
    try:
        collection = await db_manager.get_mongodb_collection("contents")
        
        # 查找内容
        doc = await collection.find_one({"id": content_id})
        if not doc:
            raise HTTPException(status_code=404, detail="内容不存在")
        
        # 更新最后访问时间
        await collection.update_one(
            {"id": content_id},
            {
                "$set": {"last_accessed": datetime.now()},
                "$inc": {"download_count": 1}
            }
        )
        
        # 转换为响应模型
        content_response = ContentResponse(
            **doc,
            summary=doc.get('summary'),
            extracted_entities=doc.get('extracted_entities', []),
            sentiment_score=doc.get('sentiment_score')
        )
        
        return {
            "success": True,
            "data": content_response.dict(),
            "message": "获取内容详情成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取内容详情失败: {str(e)}")


@router.put("/{content_id}", response_model=Dict[str, Any], summary="更新内容")
async def update_content(
    content_id: str = Path(..., description="内容ID"),
    update_data: ContentUpdate = ...,
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    更新指定内容的信息
    """
    try:
        collection = await db_manager.get_mongodb_collection("contents")
        
        # 检查内容是否存在
        existing = await collection.find_one({"id": content_id})
        if not existing:
            raise HTTPException(status_code=404, detail="内容不存在")
        
        # 构建更新数据
        update_dict = {}
        for field, value in update_data.dict(exclude_unset=True).items():
            if value is not None:
                update_dict[field] = value
        
        if not update_dict:
            return {
                "success": True,
                "message": "没有需要更新的字段",
                "data": {"id": content_id}
            }
        
        update_dict["updated_at"] = datetime.now()
        
        # 执行更新
        result = await collection.update_one(
            {"id": content_id},
            {"$set": update_dict}
        )
        
        if result.modified_count > 0:
            return {
                "success": True,
                "data": {"id": content_id, "modified_fields": list(update_dict.keys())},
                "message": "内容更新成功"
            }
        else:
            return {
                "success": True,
                "message": "内容没有变化",
                "data": {"id": content_id}
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新内容失败: {str(e)}")


@router.delete("/{content_id}", response_model=Dict[str, Any], summary="删除内容")
async def delete_content(
    content_id: str = Path(..., description="内容ID"),
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    删除指定的内容
    """
    try:
        collection = await db_manager.get_mongodb_collection("contents")
        
        # 检查内容是否存在
        existing = await collection.find_one({"id": content_id})
        if not existing:
            raise HTTPException(status_code=404, detail="内容不存在")
        
        # 删除内容
        result = await collection.delete_one({"id": content_id})
        
        if result.deleted_count > 0:
            return {
                "success": True,
                "data": {"id": content_id, "title": existing.get("title", "")},
                "message": "内容删除成功"
            }
        else:
            raise HTTPException(status_code=500, detail="删除操作失败")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除内容失败: {str(e)}")


@router.get("/statistics/overview", response_model=Dict[str, Any], summary="获取内容统计")
async def get_content_statistics(
    db_manager: DatabaseManager = Depends(get_database_manager)
):
    """
    获取内容的统计信息
    """
    try:
        collection = await db_manager.get_mongodb_collection("contents")
        
        # 总数统计
        total_count = await collection.count_documents({})
        
        # 按状态统计
        status_pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        status_cursor = collection.aggregate(status_pipeline)
        status_counts = {doc["_id"]: doc["count"] async for doc in status_cursor}
        
        # 按来源统计
        source_pipeline = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}}
        ]
        source_cursor = collection.aggregate(source_pipeline)
        source_counts = {doc["_id"]: doc["count"] async for doc in source_cursor}
        
        # 按类型统计
        type_pipeline = [
            {"$group": {"_id": "$content_type", "count": {"$sum": 1}}}
        ]
        type_cursor = collection.aggregate(type_pipeline)
        type_counts = {doc["_id"]: doc["count"] async for doc in type_cursor}
        
        # 时间统计
        now = datetime.now()
        today = datetime(now.year, now.month, now.day)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        today_count = await collection.count_documents({"created_at": {"$gte": today}})
        week_count = await collection.count_documents({"created_at": {"$gte": week_ago}})
        month_count = await collection.count_documents({"created_at": {"$gte": month_ago}})
        
        # 质量统计
        quality_pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_quality": {"$avg": "$quality_score"},
                    "high_quality_count": {
                        "$sum": {
                            "$cond": [{"$gt": ["$quality_score", 80]}, 1, 0]
                        }
                    }
                }
            }
        ]
        quality_cursor = collection.aggregate(quality_pipeline)
        quality_stats = await quality_cursor.to_list(length=1)
        
        avg_quality_score = None
        high_quality_count = 0
        if quality_stats:
            avg_quality_score = quality_stats[0].get("avg_quality")
            high_quality_count = quality_stats[0].get("high_quality_count", 0)
        
        # 处理统计
        processing_stats = await collection.aggregate([
            {
                "$group": {
                    "_id": None,
                    "avg_processing_time": {"$avg": "$processing_time"},
                    "success_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$status", "completed"]}, 1, 0]
                        }
                    }
                }
            }
        ]).to_list(length=1)
        
        avg_processing_time = None
        success_rate = 0.0
        if processing_stats:
            avg_processing_time = processing_stats[0].get("avg_processing_time")
            success_count = processing_stats[0].get("success_count", 0)
            success_rate = (success_count / total_count) if total_count > 0 else 0.0
        
        # 构建统计响应
        stats = ContentStats(
            total_count=total_count,
            status_counts=status_counts,
            source_counts=source_counts,
            type_counts=type_counts,
            today_count=today_count,
            week_count=week_count,
            month_count=month_count,
            avg_quality_score=avg_quality_score,
            high_quality_count=high_quality_count,
            avg_processing_time=avg_processing_time,
            success_rate=success_rate
        )
        
        return {
            "success": True,
            "data": stats.dict(),
            "message": "获取统计信息成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")