"""
微服务健康检查实现
提供标准的健康检查功能

功能：
- 服务存活检查（Liveness Probe）
- 服务就绪检查（Readiness Probe）
- 依赖服务健康检查
- 系统资源监控
- 自定义健康检查项
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import asyncio
import psutil
import logging
import time
import aiohttp
import asyncpg
import aioredis


class HealthStatus(BaseModel):
    """健康状态响应模型"""
    status: str = Field(..., description="健康状态", example="healthy")
    timestamp: datetime = Field(..., description="检查时间")
    version: str = Field(..., description="服务版本")
    uptime: float = Field(..., description="运行时间(秒)")
    checks: Dict[str, Any] = Field(..., description="检查项详情")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class CheckResult(BaseModel):
    """单个检查项结果"""
    status: str = Field(..., description="检查状态")
    message: Optional[str] = Field(None, description="状态消息")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    duration_ms: Optional[float] = Field(None, description="检查耗时(毫秒)")


class HealthChecker:
    """健康检查器 - 实现标准的健康检查逻辑"""
    
    def __init__(self, app_version: str = "1.0.0", service_name: str = "microservice"):
        """
        初始化健康检查器
        
        Args:
            app_version: 应用版本
            service_name: 服务名称
        """
        self.app_version = app_version
        self.service_name = service_name
        self.start_time = datetime.utcnow()
        self.logger = logging.getLogger(__name__)
        self.checks = {}
        self._check_cache = {}
        self._cache_ttl = 30  # 检查结果缓存时间(秒)
    
    def add_check(self, 
                  name: str, 
                  check_func: Callable, 
                  critical: bool = True,
                  timeout: float = 5.0,
                  cache: bool = True):
        """
        添加健康检查项
        
        Args:
            name: 检查项名称
            check_func: 检查函数
            critical: 是否为关键检查项
            timeout: 超时时间(秒)
            cache: 是否缓存结果
        """
        self.checks[name] = {
            "func": check_func,
            "critical": critical,
            "timeout": timeout,
            "cache": cache
        }
        self.logger.info(f"Added health check: {name} (critical={critical})")
    
    def remove_check(self, name: str):
        """移除健康检查项"""
        if name in self.checks:
            del self.checks[name]
            if name in self._check_cache:
                del self._check_cache[name]
            self.logger.info(f"Removed health check: {name}")
    
    async def perform_health_check(self) -> HealthStatus:
        """执行完整的健康检查"""
        check_results = {}
        overall_status = "healthy"
        
        # 执行所有检查项
        for check_name, check_config in self.checks.items():
            result = await self._execute_check(check_name, check_config)
            check_results[check_name] = result
            
            # 判断整体状态
            if result.status == "fail" and check_config["critical"]:
                overall_status = "unhealthy"
            elif result.status == "fail" and overall_status == "healthy":
                overall_status = "degraded"
        
        # 计算运行时间
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # 获取系统资源信息
        system_info = self._get_system_info()
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=self.app_version,
            uptime=uptime,
            checks=check_results,
            metadata={
                "system": system_info,
                "service_name": self.service_name
            }
        )
    
    async def perform_liveness_check(self) -> Dict[str, Any]:
        """执行存活检查 - 用于Kubernetes liveness probe"""
        return {
            "status": "alive",
            "timestamp": datetime.utcnow(),
            "uptime": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    async def perform_readiness_check(self) -> Dict[str, Any]:
        """执行就绪检查 - 用于Kubernetes readiness probe"""
        # 只检查关键的依赖服务
        critical_checks = {
            name: config for name, config in self.checks.items() 
            if config["critical"]
        }
        
        for check_name, check_config in critical_checks.items():
            result = await self._execute_check(check_name, check_config)
            if result.status == "fail":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service not ready: {check_name} check failed"
                )
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow(),
            "checks_passed": len(critical_checks)
        }
    
    async def _execute_check(self, name: str, config: Dict) -> CheckResult:
        """执行单个检查项"""
        # 检查缓存
        if config.get("cache", True):
            cached_result = self._get_cached_result(name)
            if cached_result:
                return cached_result
        
        start_time = time.time()
        
        try:
            # 执行检查函数，支持超时
            check_func = config["func"]
            timeout = config.get("timeout", 5.0)
            
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, check_func),
                    timeout=timeout
                )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # 处理结果
            if isinstance(result, dict):
                if result.get("status") in ["pass", "ok", True]:
                    check_result = CheckResult(
                        status="pass",
                        details=result,
                        duration_ms=duration_ms
                    )
                else:
                    check_result = CheckResult(
                        status="fail",
                        message=result.get("message", "Check failed"),
                        details=result,
                        duration_ms=duration_ms
                    )
            elif result in [True, "ok", "pass"]:
                check_result = CheckResult(
                    status="pass",
                    duration_ms=duration_ms
                )
            else:
                check_result = CheckResult(
                    status="fail",
                    message=str(result),
                    duration_ms=duration_ms
                )
                
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            check_result = CheckResult(
                status="fail",
                message=f"Check timed out after {config.get('timeout', 5.0)}s",
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            check_result = CheckResult(
                status="fail",
                message=str(e),
                duration_ms=duration_ms
            )
        
        # 缓存结果
        if config.get("cache", True):
            self._cache_result(name, check_result)
        
        return check_result
    
    def _get_cached_result(self, name: str) -> Optional[CheckResult]:
        """获取缓存的检查结果"""
        if name in self._check_cache:
            cached_at, result = self._check_cache[name]
            if (time.time() - cached_at) < self._cache_ttl:
                return result
        return None
    
    def _cache_result(self, name: str, result: CheckResult):
        """缓存检查结果"""
        self._check_cache[name] = (time.time(), result)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统资源信息"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "percent": psutil.virtual_memory().percent,
                    "available": psutil.virtual_memory().available,
                    "total": psutil.virtual_memory().total
                },
                "disk": {
                    "percent": psutil.disk_usage('/').percent,
                    "free": psutil.disk_usage('/').free,
                    "total": psutil.disk_usage('/').total
                },
                "network": {
                    "connections": len(psutil.net_connections())
                }
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {"error": "Unable to get system info"}


# 常用健康检查函数
class CommonHealthChecks:
    """常用健康检查函数集合"""
    
    @staticmethod
    async def check_database_postgresql(connection_string: str) -> Dict[str, Any]:
        """检查PostgreSQL数据库连接"""
        try:
            start_time = time.time()
            conn = await asyncpg.connect(connection_string)
            await conn.execute("SELECT 1")
            await conn.close()
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "pass",
                "connection": "ok",
                "response_time_ms": round(duration_ms, 2)
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    @staticmethod
    async def check_redis(redis_url: str) -> Dict[str, Any]:
        """检查Redis连接"""
        try:
            start_time = time.time()
            redis = aioredis.from_url(redis_url)
            await redis.ping()
            await redis.close()
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "pass",
                "connection": "ok",
                "response_time_ms": round(duration_ms, 2)
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    @staticmethod
    async def check_http_service(url: str, timeout: float = 5.0) -> Dict[str, Any]:
        """检查HTTP服务"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    return {
                        "status": "pass" if response.status < 500 else "fail",
                        "http_status": response.status,
                        "response_time_ms": round(duration_ms, 2)
                    }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    @staticmethod
    async def check_consul(consul_url: str = "http://localhost:8500") -> Dict[str, Any]:
        """检查Consul服务"""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{consul_url}/v1/status/leader") as response:
                    if response.status == 200:
                        leader = await response.text()
                        duration_ms = (time.time() - start_time) * 1000
                        
                        return {
                            "status": "pass",
                            "leader": leader.strip('"') if leader else "unknown",
                            "response_time_ms": round(duration_ms, 2)
                        }
                    else:
                        return {
                            "status": "fail",
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    @staticmethod
    def check_disk_space(path: str = "/", threshold: float = 80.0) -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            disk_usage = psutil.disk_usage(path)
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            return {
                "status": "pass" if usage_percent < threshold else "fail",
                "usage_percent": round(usage_percent, 2),
                "free_bytes": disk_usage.free,
                "total_bytes": disk_usage.total,
                "threshold": threshold
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }
    
    @staticmethod
    def check_memory_usage(threshold: float = 80.0) -> Dict[str, Any]:
        """检查内存使用率"""
        try:
            memory = psutil.virtual_memory()
            
            return {
                "status": "pass" if memory.percent < threshold else "fail",
                "usage_percent": memory.percent,
                "available_bytes": memory.available,
                "total_bytes": memory.total,
                "threshold": threshold
            }
        except Exception as e:
            return {
                "status": "fail",
                "error": str(e)
            }


# FastAPI健康检查端点集成
def setup_health_endpoints(app: FastAPI, health_checker: HealthChecker):
    """设置FastAPI健康检查端点"""
    
    @app.get("/health", response_model=HealthStatus)
    async def health_check():
        """完整健康检查端点"""
        return await health_checker.perform_health_check()
    
    @app.get("/health/live")
    async def liveness_check():
        """存活检查端点 - 用于Kubernetes liveness probe"""
        return await health_checker.perform_liveness_check()
    
    @app.get("/health/ready")
    async def readiness_check():
        """就绪检查端点 - 用于Kubernetes readiness probe"""
        return await health_checker.perform_readiness_check()
    
    @app.get("/health/checks")
    async def list_checks():
        """列出所有检查项"""
        return {
            "checks": list(health_checker.checks.keys()),
            "total": len(health_checker.checks)
        }