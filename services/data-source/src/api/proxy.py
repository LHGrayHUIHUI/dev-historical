"""
代理管理API接口
提供代理的获取、测试、统计和管理接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ..proxy.proxy_manager import (
    get_proxy_manager,
    ProxyManager,
    ProxyInfo,
    ProxyStatus,
    ProxyQuality
)

# 创建路由器
router = APIRouter(prefix="/proxy", tags=["代理管理"])


class ProxyInfoResponse(BaseModel):
    """代理信息响应模型"""
    proxy_id: str
    host: str
    port: int
    protocol: str
    status: ProxyStatus
    quality: ProxyQuality
    country: Optional[str]
    region: Optional[str]
    provider: Optional[str]
    success_count: int
    failure_count: int
    total_requests: int
    success_rate: float
    avg_response_time: float
    is_available: bool
    created_at: str
    updated_at: str
    
    @classmethod
    def from_proxy_info(cls, proxy: ProxyInfo) -> "ProxyInfoResponse":
        """从ProxyInfo对象创建响应模型"""
        return cls(
            proxy_id=proxy.proxy_id,
            host=proxy.host,
            port=proxy.port,
            protocol=proxy.protocol,
            status=proxy.status,
            quality=proxy.quality,
            country=proxy.country,
            region=proxy.region,
            provider=proxy.provider,
            success_count=proxy.success_count,
            failure_count=proxy.failure_count,
            total_requests=proxy.total_requests,
            success_rate=proxy.success_rate,
            avg_response_time=proxy.avg_response_time,
            is_available=proxy.is_available,
            created_at=proxy.created_at.isoformat(),
            updated_at=proxy.updated_at.isoformat()
        )


class ProxyListResponse(BaseModel):
    """代理列表响应模型"""
    items: List[ProxyInfoResponse]
    total: int
    active: int
    banned: int


class ProxyTestRequest(BaseModel):
    """代理测试请求模型"""
    host: str = Field(..., description="代理主机地址")
    port: int = Field(..., description="代理端口", ge=1, le=65535)
    username: Optional[str] = Field(None, description="用户名")
    password: Optional[str] = Field(None, description="密码")
    protocol: str = Field("http", description="协议类型", pattern="^(http|https|socks4|socks5)$")


@router.get("/", response_model=Dict[str, Any], summary="获取代理列表")
async def get_proxy_list(
    status: Optional[ProxyStatus] = Query(None, description="按状态过滤"),
    quality: Optional[ProxyQuality] = Query(None, description="按质量过滤"),
    provider: Optional[str] = Query(None, description="按供应商过滤"),
    country: Optional[str] = Query(None, description="按国家过滤"),
    page: int = Query(1, description="页码", ge=1),
    size: int = Query(20, description="每页数量", ge=1, le=100),
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    获取代理列表，支持过滤和分页
    """
    try:
        # 获取所有代理
        all_proxies = list(proxy_manager.proxies.values())
        
        # 过滤
        filtered_proxies = all_proxies
        if status:
            filtered_proxies = [p for p in filtered_proxies if p.status == status]
        if quality:
            filtered_proxies = [p for p in filtered_proxies if p.quality == quality]
        if provider:
            filtered_proxies = [p for p in filtered_proxies if p.provider == provider]
        if country:
            filtered_proxies = [p for p in filtered_proxies if p.country == country]
        
        # 排序（按成功率倒序）
        filtered_proxies.sort(key=lambda x: x.success_rate, reverse=True)
        
        # 分页
        total = len(filtered_proxies)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        page_proxies = filtered_proxies[start_idx:end_idx]
        
        # 转换为响应模型
        proxy_responses = [ProxyInfoResponse.from_proxy_info(proxy) for proxy in page_proxies]
        
        # 统计信息
        active_count = len([p for p in filtered_proxies if p.status == ProxyStatus.ACTIVE])
        banned_count = len([p for p in filtered_proxies if p.status == ProxyStatus.BANNED])
        
        response = ProxyListResponse(
            items=proxy_responses,
            total=total,
            active=active_count,
            banned=banned_count
        )
        
        return {
            "success": True,
            "data": {
                "items": [item.dict() for item in response.items],
                "total": response.total,
                "active": response.active,
                "banned": response.banned,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            },
            "message": "获取代理列表成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取代理列表失败: {str(e)}")


@router.get("/active", response_model=Dict[str, Any], summary="获取可用代理")
async def get_active_proxies(
    limit: int = Query(10, description="返回数量限制", ge=1, le=100),
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    获取当前可用的代理列表
    """
    try:
        available_proxies = []
        
        for proxy_id in proxy_manager.active_proxies[:limit]:
            if proxy_id in proxy_manager.proxies:
                proxy = proxy_manager.proxies[proxy_id]
                if proxy.is_available:
                    available_proxies.append(ProxyInfoResponse.from_proxy_info(proxy))
        
        return {
            "success": True,
            "data": {
                "items": [proxy.dict() for proxy in available_proxies],
                "count": len(available_proxies)
            },
            "message": "获取可用代理成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取可用代理失败: {str(e)}")


@router.get("/best", response_model=Dict[str, Any], summary="获取最佳代理")
async def get_best_proxy(
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    获取当前评分最高的代理
    """
    try:
        best_proxy = proxy_manager.get_proxy()
        
        if not best_proxy:
            return {
                "success": False,
                "message": "当前没有可用的代理",
                "data": None
            }
        
        proxy_response = ProxyInfoResponse.from_proxy_info(best_proxy)
        
        return {
            "success": True,
            "data": proxy_response.dict(),
            "message": "获取最佳代理成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取最佳代理失败: {str(e)}")


@router.post("/test", response_model=Dict[str, Any], summary="测试代理")
async def test_proxy(
    test_request: ProxyTestRequest,
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    测试指定代理的可用性
    """
    try:
        # 创建临时代理对象
        import hashlib
        proxy_id = hashlib.md5(f"{test_request.host}:{test_request.port}".encode()).hexdigest()
        
        from ..proxy.proxy_manager import ProxyInfo, ProxyStatus, ProxyQuality
        temp_proxy = ProxyInfo(
            proxy_id=proxy_id,
            host=test_request.host,
            port=test_request.port,
            username=test_request.username,
            password=test_request.password,
            protocol=test_request.protocol,
            status=ProxyStatus.INACTIVE,
            quality=ProxyQuality.UNKNOWN
        )
        
        # 执行测试
        success = await proxy_manager.test_proxy(temp_proxy)
        
        return {
            "success": True,
            "data": {
                "proxy": f"{test_request.host}:{test_request.port}",
                "test_result": success,
                "status": temp_proxy.status,
                "response_time": temp_proxy.avg_response_time,
                "error_message": None if success else "代理连接失败"
            },
            "message": "代理测试完成"
        }
        
    except Exception as e:
        return {
            "success": False,
            "data": {
                "proxy": f"{test_request.host}:{test_request.port}",
                "test_result": False,
                "error_message": str(e)
            },
            "message": "代理测试失败"
        }


@router.post("/refresh", response_model=Dict[str, Any], summary="刷新代理列表")
async def refresh_proxy_list(
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    从供应商刷新代理列表
    """
    try:
        old_count = len(proxy_manager.proxies)
        await proxy_manager.refresh_proxies()
        new_count = len(proxy_manager.proxies)
        
        added_count = new_count - old_count
        
        return {
            "success": True,
            "data": {
                "old_count": old_count,
                "new_count": new_count,
                "added_count": added_count,
                "active_count": len(proxy_manager.active_proxies)
            },
            "message": f"代理列表刷新完成，新增 {added_count} 个代理"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新代理列表失败: {str(e)}")


@router.post("/{proxy_id}/ban", response_model=Dict[str, Any], summary="禁用代理")
async def ban_proxy(
    proxy_id: str,
    reason: Optional[str] = Query(None, description="禁用原因"),
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    手动禁用指定的代理
    """
    try:
        if proxy_id not in proxy_manager.proxies:
            raise HTTPException(status_code=404, detail="代理不存在")
        
        proxy_manager.mark_proxy_failed(proxy_id, reason or "手动禁用")
        
        return {
            "success": True,
            "data": {
                "proxy_id": proxy_id,
                "status": "banned",
                "reason": reason or "手动禁用"
            },
            "message": "代理已被禁用"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"禁用代理失败: {str(e)}")


@router.post("/{proxy_id}/unban", response_model=Dict[str, Any], summary="解禁代理")
async def unban_proxy(
    proxy_id: str,
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    解禁被禁用的代理
    """
    try:
        if proxy_id not in proxy_manager.proxies:
            raise HTTPException(status_code=404, detail="代理不存在")
        
        proxy = proxy_manager.proxies[proxy_id]
        if proxy.status != ProxyStatus.BANNED:
            raise HTTPException(status_code=400, detail="代理未被禁用")
        
        # 重新测试代理
        success = await proxy_manager.test_proxy(proxy)
        
        if success:
            proxy_manager.banned_proxies.discard(proxy_id)
            if proxy_id not in proxy_manager.active_proxies:
                proxy_manager.active_proxies.append(proxy_id)
        
        return {
            "success": True,
            "data": {
                "proxy_id": proxy_id,
                "status": proxy.status,
                "test_result": success
            },
            "message": "代理解禁完成" if success else "代理解禁失败，仍无法连接"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解禁代理失败: {str(e)}")


@router.delete("/{proxy_id}", response_model=Dict[str, Any], summary="删除代理")
async def delete_proxy(
    proxy_id: str,
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    从系统中删除指定的代理
    """
    try:
        if proxy_id not in proxy_manager.proxies:
            raise HTTPException(status_code=404, detail="代理不存在")
        
        proxy = proxy_manager.proxies[proxy_id]
        proxy_info = f"{proxy.host}:{proxy.port}"
        
        # 从各个列表中移除
        if proxy_id in proxy_manager.active_proxies:
            proxy_manager.active_proxies.remove(proxy_id)
        proxy_manager.banned_proxies.discard(proxy_id)
        
        # 从内存中删除
        del proxy_manager.proxies[proxy_id]
        
        # TODO: 从数据库中删除
        
        return {
            "success": True,
            "data": {
                "proxy_id": proxy_id,
                "proxy_info": proxy_info
            },
            "message": "代理删除成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除代理失败: {str(e)}")


@router.get("/statistics", response_model=Dict[str, Any], summary="获取代理统计")
async def get_proxy_statistics(
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    获取代理系统的统计信息
    """
    try:
        stats = proxy_manager.get_proxy_statistics()
        
        return {
            "success": True,
            "data": stats,
            "message": "获取代理统计成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取代理统计失败: {str(e)}")


@router.get("/health", response_model=Dict[str, Any], summary="代理健康检查")
async def proxy_health_check(
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """
    检查代理系统的健康状态
    """
    try:
        stats = proxy_manager.get_proxy_statistics()
        
        # 健康检查逻辑
        health_status = "healthy"
        issues = []
        
        if stats["active_proxies"] == 0:
            health_status = "critical"
            issues.append("没有可用的代理")
        elif stats["active_proxies"] < 5:
            health_status = "warning"
            issues.append("可用代理数量较少")
        
        if stats["average_success_rate"] < 50:
            health_status = "warning"
            issues.append("代理平均成功率较低")
        
        banned_rate = (stats["banned_proxies"] / stats["total_proxies"]) * 100 if stats["total_proxies"] > 0 else 0
        if banned_rate > 30:
            health_status = "warning"
            issues.append("代理封禁率较高")
        
        return {
            "success": True,
            "data": {
                "status": health_status,
                "issues": issues,
                "statistics": stats,
                "recommendations": [
                    "定期刷新代理列表",
                    "监控代理成功率",
                    "及时处理失效代理"
                ] if issues else []
            },
            "message": "代理健康检查完成"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代理健康检查失败: {str(e)}")