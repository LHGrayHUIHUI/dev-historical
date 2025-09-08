"""
NLP控制器
FastAPI路由处理器，处理NLP相关的HTTP请求
无状态架构，数据存储通过storage-service完成
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, status
from loguru import logger

from ..schemas.nlp_schemas import *
from ..services.nlp_service import NLPService
from ..clients.storage_client import StorageServiceClient, storage_client
from ..config.settings import settings


# 创建路由器
router = APIRouter(prefix="/api/v1/nlp", tags=["NLP"])

# NLP服务实例（全局单例）
nlp_service: Optional[NLPService] = None


async def get_nlp_service() -> NLPService:
    """获取NLP服务实例"""
    global nlp_service
    if nlp_service is None:
        nlp_service = NLPService()
        await nlp_service.initialize_models()
    return nlp_service


async def get_storage_client() -> StorageServiceClient:
    """获取storage客户端实例"""
    return storage_client


# ============ 文本处理接口 ============

@router.post("/segment", response_model=SegmentationResponse)
async def segment_text_endpoint(
    request: TextProcessRequest,
    nlp: NLPService = Depends(get_nlp_service),
    storage: StorageServiceClient = Depends(get_storage_client),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """文本分词接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 如果是异步模式，创建任务记录
        if request.async_mode:
            await storage.create_nlp_task(
                dataset_id=request.dataset_id or task_id,
                text_content=request.text,
                processing_type="segmentation",
                nlp_model=request.engine or settings.default_nlp_engine,
                language=request.language,
                config=request.config
            )
            
            # 后台处理
            background_tasks.add_task(
                _process_segmentation_async,
                task_id, request.text, request.engine or "jieba", 
                request.language, request.config, nlp, storage
            )
            
            return SegmentationResponse(
                success=True,
                message="分词任务已创建，请通过task_id查询结果",
                task_id=task_id,
                processing_type=ProcessingType.SEGMENTATION,
                processing_time=0,
                result=SegmentationResult(
                    original_text="",
                    segmented_text="",
                    words=[],
                    word_count=0,
                    unique_word_count=0,
                    method=""
                )
            )
        
        # 同步处理
        result = await nlp.segment_text(
            text=request.text,
            method=request.engine or "jieba",
            language=request.language,
            config=request.config
        )
        
        processing_time = time.time() - start_time
        
        # 保存结果到storage-service
        if request.dataset_id:
            task_record = await storage.create_nlp_task(
                dataset_id=request.dataset_id,
                text_content=request.text,
                processing_type="segmentation",
                nlp_model=request.engine or settings.default_nlp_engine,
                language=request.language,
                config=request.config
            )
            
            task_id = task_record.get("task_id", task_id)
            
            await storage.save_segmentation_result(
                task_id=task_id,
                original_text=result.original_text,
                segmented_text=result.segmented_text,
                words=[word.dict() for word in result.words],
                segmentation_method=result.method
            )
            
            await storage.update_nlp_task_status(
                task_id=task_id,
                status="completed",
                processing_time=processing_time
            )
        
        return SegmentationResponse(
            success=True,
            message="分词完成",
            task_id=task_id,
            processing_type=ProcessingType.SEGMENTATION,
            processing_time=processing_time,
            result=result
        )
        
    except Exception as e:
        logger.error(f"分词处理失败: {str(e)}")
        
        # 更新任务状态为失败
        if request.async_mode and request.dataset_id:
            try:
                await storage.update_nlp_task_status(
                    task_id=task_id,
                    status="failed",
                    error_message=str(e)
                )
            except:
                pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分词处理失败: {str(e)}"
        )


@router.post("/pos-tagging", response_model=PosTaggingResponse)
async def pos_tagging_endpoint(
    request: TextProcessRequest,
    nlp: NLPService = Depends(get_nlp_service),
    storage: StorageServiceClient = Depends(get_storage_client),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """词性标注接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 异步模式处理
        if request.async_mode:
            await storage.create_nlp_task(
                dataset_id=request.dataset_id or task_id,
                text_content=request.text,
                processing_type="pos_tagging",
                nlp_model=request.engine or settings.default_nlp_engine,
                language=request.language,
                config=request.config
            )
            
            background_tasks.add_task(
                _process_pos_tagging_async,
                task_id, request.text, request.engine or "jieba",
                request.language, request.config, nlp, storage
            )
            
            return PosTaggingResponse(
                success=True,
                message="词性标注任务已创建",
                task_id=task_id,
                processing_type=ProcessingType.POS_TAGGING,
                processing_time=0,
                result=PosTaggingResult(
                    words_with_pos=[],
                    pos_distribution={},
                    method=""
                )
            )
        
        # 同步处理
        result = await nlp.pos_tagging(
            text=request.text,
            method=request.engine or "jieba",
            language=request.language,
            config=request.config
        )
        
        processing_time = time.time() - start_time
        
        # 保存结果
        if request.dataset_id:
            task_record = await storage.create_nlp_task(
                dataset_id=request.dataset_id,
                text_content=request.text,
                processing_type="pos_tagging",
                nlp_model=request.engine or settings.default_nlp_engine,
                language=request.language,
                config=request.config
            )
            
            task_id = task_record.get("task_id", task_id)
            
            await storage.save_pos_tagging_result(
                task_id=task_id,
                words_with_pos=[word.dict() for word in result.words_with_pos],
                pos_distribution=result.pos_distribution,
                tagging_method=result.method
            )
            
            await storage.update_nlp_task_status(
                task_id=task_id,
                status="completed",
                processing_time=processing_time
            )
        
        return PosTaggingResponse(
            success=True,
            message="词性标注完成",
            task_id=task_id,
            processing_type=ProcessingType.POS_TAGGING,
            processing_time=processing_time,
            result=result
        )
        
    except Exception as e:
        logger.error(f"词性标注失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"词性标注失败: {str(e)}"
        )


@router.post("/ner", response_model=NERResponse)
async def ner_endpoint(
    request: TextProcessRequest,
    nlp: NLPService = Depends(get_nlp_service),
    storage: StorageServiceClient = Depends(get_storage_client),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """命名实体识别接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        result = await nlp.named_entity_recognition(
            text=request.text,
            model=request.engine or "spacy",
            language=request.language,
            config=request.config
        )
        
        processing_time = time.time() - start_time
        
        # 保存结果
        if request.dataset_id:
            task_record = await storage.create_nlp_task(
                dataset_id=request.dataset_id,
                text_content=request.text,
                processing_type="ner",
                nlp_model=request.engine or "spacy",
                language=request.language,
                config=request.config
            )
            
            task_id = task_record.get("task_id", task_id)
            
            await storage.save_ner_result(
                task_id=task_id,
                entities=[entity.dict() for entity in result.entities],
                entity_types=result.entity_types,
                ner_model=result.model
            )
            
            await storage.update_nlp_task_status(
                task_id=task_id,
                status="completed",
                processing_time=processing_time
            )
        
        return NERResponse(
            success=True,
            message="命名实体识别完成",
            task_id=task_id,
            processing_type=ProcessingType.NER,
            processing_time=processing_time,
            result=result
        )
        
    except Exception as e:
        logger.error(f"NER处理失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NER处理失败: {str(e)}"
        )


@router.post("/sentiment", response_model=SentimentResponse)
async def sentiment_analysis_endpoint(
    request: TextProcessRequest,
    nlp: NLPService = Depends(get_nlp_service),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """情感分析接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        result = await nlp.sentiment_analysis(
            text=request.text,
            model=request.engine or "transformers",
            language=request.language,
            config=request.config
        )
        
        processing_time = time.time() - start_time
        
        # 保存结果
        if request.dataset_id:
            task_record = await storage.create_nlp_task(
                dataset_id=request.dataset_id,
                text_content=request.text,
                processing_type="sentiment",
                nlp_model=request.engine or "transformers",
                language=request.language,
                config=request.config
            )
            
            task_id = task_record.get("task_id", task_id)
            
            await storage.save_sentiment_result(
                task_id=task_id,
                sentiment_label=result.sentiment.label,
                sentiment_score=result.sentiment.score,
                confidence=result.sentiment.confidence,
                emotion_details=result.sentiment.emotions,
                sentiment_model=result.model
            )
            
            await storage.update_nlp_task_status(
                task_id=task_id,
                status="completed",
                processing_time=processing_time
            )
        
        return SentimentResponse(
            success=True,
            message="情感分析完成",
            task_id=task_id,
            processing_type=ProcessingType.SENTIMENT,
            processing_time=processing_time,
            result=result
        )
        
    except Exception as e:
        logger.error(f"情感分析失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"情感分析失败: {str(e)}"
        )


@router.post("/keywords", response_model=KeywordResponse)
async def extract_keywords_endpoint(
    request: TextProcessRequest,
    nlp: NLPService = Depends(get_nlp_service),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """关键词提取接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 从配置中获取参数
        method = request.config.get('method', 'textrank')
        top_k = request.config.get('top_k', 20)
        
        result = await nlp.extract_keywords(
            text=request.text,
            method=method,
            top_k=top_k,
            language=request.language,
            config=request.config
        )
        
        processing_time = time.time() - start_time
        
        # 保存结果
        if request.dataset_id:
            task_record = await storage.create_nlp_task(
                dataset_id=request.dataset_id,
                text_content=request.text,
                processing_type="keywords",
                nlp_model=method,
                language=request.language,
                config=request.config
            )
            
            task_id = task_record.get("task_id", task_id)
            
            await storage.save_keyword_result(
                task_id=task_id,
                keywords=[keyword.dict() for keyword in result.keywords],
                extraction_method=result.method
            )
            
            await storage.update_nlp_task_status(
                task_id=task_id,
                status="completed",
                processing_time=processing_time
            )
        
        return KeywordResponse(
            success=True,
            message="关键词提取完成",
            task_id=task_id,
            processing_type=ProcessingType.KEYWORDS,
            processing_time=processing_time,
            result=result
        )
        
    except Exception as e:
        logger.error(f"关键词提取失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关键词提取失败: {str(e)}"
        )


@router.post("/summary", response_model=SummaryResponse)
async def text_summarization_endpoint(
    request: TextProcessRequest,
    nlp: NLPService = Depends(get_nlp_service),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """文本摘要接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 从配置中获取参数
        method = request.config.get('method', 'extractive')
        max_sentences = request.config.get('max_sentences', 5)
        compression_ratio = request.config.get('compression_ratio', 0.3)
        
        result = await nlp.text_summarization(
            text=request.text,
            method=method,
            max_sentences=max_sentences,
            compression_ratio=compression_ratio,
            language=request.language,
            config=request.config
        )
        
        processing_time = time.time() - start_time
        
        # 保存结果
        if request.dataset_id:
            task_record = await storage.create_nlp_task(
                dataset_id=request.dataset_id,
                text_content=request.text,
                processing_type="summary",
                nlp_model=method,
                language=request.language,
                config=request.config
            )
            
            task_id = task_record.get("task_id", task_id)
            
            await storage.save_summary_result(
                task_id=task_id,
                original_length=result.original_length,
                summary_text=result.summary_text,
                summary_method=result.method
            )
            
            await storage.update_nlp_task_status(
                task_id=task_id,
                status="completed",
                processing_time=processing_time
            )
        
        return SummaryResponse(
            success=True,
            message="文本摘要完成",
            task_id=task_id,
            processing_type=ProcessingType.SUMMARY,
            processing_time=processing_time,
            result=result
        )
        
    except Exception as e:
        logger.error(f"文本摘要失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文本摘要失败: {str(e)}"
        )


@router.post("/similarity", response_model=SimilarityResponse)
async def text_similarity_endpoint(
    text1: str,
    text2: str,
    method: str = "sentence_transformer",
    language: str = "zh",
    nlp: NLPService = Depends(get_nlp_service)
):
    """文本相似度计算接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        result = await nlp.text_similarity(
            text1=text1,
            text2=text2,
            method=method,
            language=language
        )
        
        processing_time = time.time() - start_time
        
        return SimilarityResponse(
            success=True,
            message="相似度计算完成",
            task_id=task_id,
            processing_type=ProcessingType.SIMILARITY,
            processing_time=processing_time,
            result=result
        )
        
    except Exception as e:
        logger.error(f"相似度计算失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"相似度计算失败: {str(e)}"
        )


# ============ 批量处理接口 ============

@router.post("/batch", response_model=BatchProcessingResponse)
async def batch_process_endpoint(
    request: BatchProcessRequest,
    nlp: NLPService = Depends(get_nlp_service),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """批量文本处理接口"""
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        results = await nlp.batch_process(
            texts=request.texts,
            processing_type=request.processing_type,
            config=request.config
        )
        
        processing_time = time.time() - start_time
        
        # 统计成功和失败数量
        success_count = sum(1 for r in results if r.get('success', False))
        failed_count = len(results) - success_count
        
        return BatchProcessingResponse(
            success=True,
            message=f"批量处理完成，成功: {success_count}, 失败: {failed_count}",
            task_id=task_id,
            processing_type=request.processing_type,
            processing_time=processing_time,
            results=results,
            success_count=success_count,
            failed_count=failed_count
        )
        
    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量处理失败: {str(e)}"
        )


# ============ 任务管理接口 ============

@router.get("/tasks/{task_id}", response_model=NLPTaskResponse)
async def get_nlp_task(
    task_id: str,
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """获取NLP任务详情"""
    
    try:
        task_data = await storage.get_nlp_task(task_id)
        
        return NLPTaskResponse(
            success=True,
            message="任务信息获取成功",
            task=NLPTask(**task_data)
        )
        
    except Exception as e:
        logger.error(f"获取任务失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {task_id}"
        )


@router.get("/tasks", response_model=NLPTaskListResponse)
async def get_nlp_tasks(
    dataset_id: Optional[str] = Query(None),
    processing_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """获取NLP任务列表"""
    
    try:
        offset = (page - 1) * size
        
        tasks_data = await storage.get_nlp_tasks(
            dataset_id=dataset_id,
            processing_type=processing_type,
            status=status,
            limit=size,
            offset=offset
        )
        
        tasks = [NLPTask(**task) for task in tasks_data.get('tasks', [])]
        
        return NLPTaskListResponse(
            success=True,
            message="任务列表获取成功",
            tasks=tasks,
            total=tasks_data.get('total', 0),
            page=page,
            size=size
        )
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务列表失败: {str(e)}"
        )


# ============ 引擎信息接口 ============

@router.get("/engines", response_model=NLPEnginesResponse)
async def get_available_engines(nlp: NLPService = Depends(get_nlp_service)):
    """获取可用的NLP引擎列表"""
    
    try:
        engines = nlp.get_available_engines()
        
        return NLPEnginesResponse(
            success=True,
            message="引擎列表获取成功",
            engines=engines
        )
        
    except Exception as e:
        logger.error(f"获取引擎列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取引擎列表失败: {str(e)}"
        )


# ============ 统计接口 ============

@router.get("/statistics", response_model=NLPStatisticsResponse)
async def get_nlp_statistics(
    dataset_id: Optional[str] = Query(None),
    processing_type: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    storage: StorageServiceClient = Depends(get_storage_client)
):
    """获取NLP处理统计信息"""
    
    try:
        stats_data = await storage.get_nlp_statistics(
            dataset_id=dataset_id,
            processing_type=processing_type,
            date_from=date_from,
            date_to=date_to
        )
        
        return NLPStatisticsResponse(
            success=True,
            message="统计信息获取成功",
            statistics=NLPStatistics(**stats_data)
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


# ============ 异步处理任务函数 ============

async def _process_segmentation_async(
    task_id: str,
    text: str,
    engine: str,
    language: str,
    config: Dict,
    nlp: NLPService,
    storage: StorageServiceClient
):
    """异步分词处理"""
    try:
        await storage.update_nlp_task_status(task_id, "processing")
        
        start_time = time.time()
        result = await nlp.segment_text(text, engine, language, config)
        processing_time = time.time() - start_time
        
        await storage.save_segmentation_result(
            task_id=task_id,
            original_text=result.original_text,
            segmented_text=result.segmented_text,
            words=[word.dict() for word in result.words],
            segmentation_method=result.method
        )
        
        await storage.update_nlp_task_status(
            task_id=task_id,
            status="completed",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"异步分词任务失败: {str(e)}")
        await storage.update_nlp_task_status(
            task_id=task_id,
            status="failed",
            error_message=str(e)
        )


async def _process_pos_tagging_async(
    task_id: str,
    text: str,
    engine: str,
    language: str,
    config: Dict,
    nlp: NLPService,
    storage: StorageServiceClient
):
    """异步词性标注处理"""
    try:
        await storage.update_nlp_task_status(task_id, "processing")
        
        start_time = time.time()
        result = await nlp.pos_tagging(text, engine, language, config)
        processing_time = time.time() - start_time
        
        await storage.save_pos_tagging_result(
            task_id=task_id,
            words_with_pos=[word.dict() for word in result.words_with_pos],
            pos_distribution=result.pos_distribution,
            tagging_method=result.method
        )
        
        await storage.update_nlp_task_status(
            task_id=task_id,
            status="completed",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"异步词性标注任务失败: {str(e)}")
        await storage.update_nlp_task_status(
            task_id=task_id,
            status="failed",
            error_message=str(e)
        )