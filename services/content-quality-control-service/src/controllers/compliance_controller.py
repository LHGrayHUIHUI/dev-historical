"""
合规检测控制器

提供合规性检测相关的API端点，包括合规检测、敏感词管理、
政策合规检查等功能。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional

from loguru import logger

from ..models.quality_models import (
    ComplianceCheckRequest, ComplianceCheckResult, DataResponse, ErrorResponse
)
from ..services.compliance_engine import ComplianceEngine
from ..clients.storage_client import StorageServiceClient, get_storage_client
from ..config.settings import settings

# 创建路由器
router = APIRouter(prefix="/compliance", tags=["合规检测"])

@router.post("/check",
             response_model=DataResponse,
             summary="合规性检测",
             description="对内容进行全面的合规性检查")
async def check_compliance(
    request: ComplianceCheckRequest,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    合规性检测
    
    对提供的内容进行全面的合规性检查，包括敏感词检测、
    政策合规、版权检查、学术诚信等方面的分析。
    """
    try:
        logger.info(f"开始合规检测，内容长度: {len(request.content)}")
        
        # 创建合规检测引擎
        compliance_engine = ComplianceEngine(storage_client)
        
        # 执行合规检测
        result = await compliance_engine.check_compliance(request)
        
        # 保存检测结果
        if settings.ENABLE_RESULT_CACHE:
            try:
                await storage_client.save_compliance_check_result(result)
                logger.debug(f"合规检测结果已保存: {result.check_id}")
            except Exception as e:
                logger.warning(f"保存合规检测结果失败: {e}")
        
        # 准备响应数据
        response_data = {
            "check_id": result.check_id,
            "compliance_status": result.compliance_status.value,
            "risk_score": result.risk_score,
            "violations": [
                {
                    "type": violation.violation_type.value,
                    "severity": violation.severity,
                    "position": violation.position,
                    "content": violation.content,
                    "description": violation.description,
                    "category": violation.category,
                    "action": violation.action,
                    "suggestion": violation.suggestion,
                    "confidence": violation.confidence
                }
                for violation in result.violations
            ],
            "policy_compliance": result.policy_compliance,
            "recommendations": result.recommendations,
            "processing_time_ms": result.processing_time_ms
        }
        
        return DataResponse(
            success=True,
            message="合规检测完成",
            data=response_data
        )
        
    except ValueError as e:
        logger.error(f"合规检测参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"合规检测失败: {e}")
        raise HTTPException(status_code=500, detail="合规检测服务异常")

@router.get("/result/{check_id}",
            response_model=DataResponse,
            summary="获取合规检测结果",
            description="根据检测ID获取合规检测结果")
async def get_compliance_result(
    check_id: str,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取合规检测结果
    
    根据检测ID返回详细的合规检测结果。
    """
    try:
        logger.info(f"获取合规检测结果: check_id={check_id}")
        
        # 通过storage-service获取检测结果
        result = await storage_client.get_compliance_check_result(check_id)
        
        if not result.get("data"):
            raise HTTPException(status_code=404, detail="检测结果不存在")
        
        return DataResponse(
            success=True,
            message="合规检测结果获取成功",
            data=result.get("data")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取合规检测结果失败: {e}")
        raise HTTPException(status_code=500, detail="获取合规检测结果失败")

@router.get("/sensitive-words",
            response_model=DataResponse,
            summary="获取敏感词列表",
            description="获取当前激活的敏感词列表")
async def get_sensitive_words(
    category: Optional[str] = None,
    active_only: bool = True,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取敏感词列表
    
    返回当前激活的敏感词列表，可按分类过滤。
    """
    try:
        logger.info(f"获取敏感词列表: category={category}, active_only={active_only}")
        
        # 通过storage-service获取敏感词列表
        result = await storage_client.get_sensitive_words(category, active_only)
        
        return DataResponse(
            success=True,
            message="敏感词列表获取成功",
            data=result.get("data", [])
        )
        
    except Exception as e:
        logger.error(f"获取敏感词列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取敏感词列表失败")

@router.post("/sensitive-words",
             response_model=DataResponse,
             summary="添加敏感词",
             description="添加新的敏感词到词库")
async def add_sensitive_word(
    word_data: Dict[str, Any],
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    添加敏感词
    
    向敏感词库中添加新的敏感词汇。
    """
    try:
        logger.info(f"添加敏感词: {word_data.get('word', 'unknown')}")
        
        # 验证必需字段
        if "word" not in word_data:
            raise ValueError("敏感词内容不能为空")
        
        # 设置默认值
        word_data.setdefault("category", "general")
        word_data.setdefault("severity_level", 5)
        word_data.setdefault("is_active", True)
        
        # 通过storage-service添加敏感词
        result = await storage_client.add_sensitive_word(word_data)
        
        return DataResponse(
            success=True,
            message="敏感词添加成功",
            data=result.get("data", {})
        )
        
    except ValueError as e:
        logger.error(f"添加敏感词参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"添加敏感词失败: {e}")
        raise HTTPException(status_code=500, detail="添加敏感词失败")

@router.get("/rules",
            response_model=DataResponse,
            summary="获取合规规则",
            description="获取合规检测规则列表")
async def get_compliance_rules(
    rule_type: Optional[str] = None,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取合规规则
    
    返回当前激活的合规检测规则列表。
    """
    try:
        logger.info(f"获取合规规则: rule_type={rule_type}")
        
        # 通过storage-service获取合规规则
        result = await storage_client.get_compliance_rules(rule_type)
        
        return DataResponse(
            success=True,
            message="合规规则获取成功",
            data=result.get("data", [])
        )
        
    except Exception as e:
        logger.error(f"获取合规规则失败: {e}")
        raise HTTPException(status_code=500, detail="获取合规规则失败")

@router.get("/statistics",
            response_model=DataResponse,
            summary="合规统计",
            description="获取合规检测统计数据")
async def get_compliance_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    获取合规统计数据
    
    返回指定时间范围内的合规检测统计信息。
    """
    try:
        logger.info(f"获取合规统计数据: {start_date} - {end_date}")
        
        # 通过storage-service获取统计数据
        stats_result = await storage_client.get_compliance_statistics(start_date, end_date)
        
        return DataResponse(
            success=True,
            message="合规统计数据获取成功",
            data=stats_result.get("data", {})
        )
        
    except Exception as e:
        logger.error(f"获取合规统计数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取合规统计数据失败")

@router.post("/batch-check",
             response_model=DataResponse,
             summary="批量合规检测",
             description="对多个内容进行批量合规检测")
async def batch_compliance_check(
    contents: List[str],
    check_types: List[str] = None,
    strict_mode: bool = False,
    storage_client: StorageServiceClient = Depends(get_storage_client)
) -> DataResponse:
    """
    批量合规检测
    
    对多个内容进行并行合规检测。
    """
    try:
        logger.info(f"开始批量合规检测，内容数量: {len(contents)}")
        
        # 验证批量大小
        if len(contents) > settings.MAX_BATCH_SIZE:
            raise ValueError(f"批量大小超过限制 ({settings.MAX_BATCH_SIZE})")
        
        # 设置默认检测类型
        if check_types is None:
            check_types = ["sensitive_words", "policy", "copyright", "academic_integrity"]
        
        # 创建合规检测引擎
        compliance_engine = ComplianceEngine(storage_client)
        
        # 并行处理所有内容
        tasks = []
        for i, content in enumerate(contents):
            request = ComplianceCheckRequest(
                content=content,
                check_types=check_types,
                strict_mode=strict_mode
            )
            task = compliance_engine.check_compliance(request)
            tasks.append(task)
        
        # 等待所有检测完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        batch_results = []
        failed_items = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_items += 1
                batch_results.append({
                    "index": i,
                    "success": False,
                    "error": str(result)
                })
            else:
                batch_results.append({
                    "index": i,
                    "success": True,
                    "check_id": result.check_id,
                    "compliance_status": result.compliance_status.value,
                    "risk_score": result.risk_score,
                    "violations_count": len(result.violations)
                })
        
        success_rate = (len(contents) - failed_items) / len(contents) if contents else 0
        
        return DataResponse(
            success=True,
            message=f"批量合规检测完成，成功率: {success_rate:.2%}",
            data={
                "total_items": len(contents),
                "successful_items": len(contents) - failed_items,
                "failed_items": failed_items,
                "success_rate": success_rate,
                "results": batch_results
            }
        )
        
    except ValueError as e:
        logger.error(f"批量合规检测参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量合规检测失败: {e}")
        raise HTTPException(status_code=500, detail="批量合规检测服务异常")