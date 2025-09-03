"""
健康检查器单元测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import time

from services.core.health.health_checker import (
    HealthChecker,
    CommonHealthChecks,
    CheckResult,
    HealthStatus
)


@pytest.fixture
def health_checker():
    """健康检查器夹具"""
    return HealthChecker(app_version="1.0.0", service_name="test-service")


@pytest.mark.asyncio
async def test_health_checker_initialization(health_checker):
    """测试健康检查器初始化"""
    assert health_checker.app_version == "1.0.0"
    assert health_checker.service_name == "test-service"
    assert health_checker.start_time is not None
    assert len(health_checker.checks) == 0


def test_add_remove_check(health_checker):
    """测试添加和移除检查项"""
    def dummy_check():
        return True
    
    # 测试添加检查项
    health_checker.add_check("test_check", dummy_check, critical=True, timeout=5.0)
    
    assert "test_check" in health_checker.checks
    assert health_checker.checks["test_check"]["critical"] is True
    assert health_checker.checks["test_check"]["timeout"] == 5.0
    
    # 测试移除检查项
    health_checker.remove_check("test_check")
    assert "test_check" not in health_checker.checks


@pytest.mark.asyncio
async def test_execute_successful_sync_check(health_checker):
    """测试执行成功的同步检查"""
    def successful_check():
        return {"status": "pass", "message": "All good"}
    
    health_checker.add_check("success_check", successful_check)
    config = health_checker.checks["success_check"]
    
    result = await health_checker._execute_check("success_check", config)
    
    assert isinstance(result, CheckResult)
    assert result.status == "pass"
    assert result.duration_ms is not None
    assert result.duration_ms > 0


@pytest.mark.asyncio
async def test_execute_successful_async_check(health_checker):
    """测试执行成功的异步检查"""
    async def async_successful_check():
        await asyncio.sleep(0.01)  # 模拟异步操作
        return True
    
    health_checker.add_check("async_success_check", async_successful_check)
    config = health_checker.checks["async_success_check"]
    
    result = await health_checker._execute_check("async_success_check", config)
    
    assert result.status == "pass"
    assert result.duration_ms > 10  # 至少10ms（因为sleep了0.01秒）


@pytest.mark.asyncio
async def test_execute_failing_check(health_checker):
    """测试执行失败的检查"""
    def failing_check():
        return {"status": "fail", "message": "Something went wrong"}
    
    health_checker.add_check("fail_check", failing_check)
    config = health_checker.checks["fail_check"]
    
    result = await health_checker._execute_check("fail_check", config)
    
    assert result.status == "fail"
    assert "Something went wrong" in str(result.message)


@pytest.mark.asyncio
async def test_execute_timeout_check(health_checker):
    """测试检查超时"""
    async def slow_check():
        await asyncio.sleep(2)  # 超过默认超时时间
        return True
    
    health_checker.add_check("slow_check", slow_check, timeout=0.1)  # 0.1秒超时
    config = health_checker.checks["slow_check"]
    
    result = await health_checker._execute_check("slow_check", config)
    
    assert result.status == "fail"
    assert "timed out" in result.message.lower()


@pytest.mark.asyncio
async def test_execute_exception_check(health_checker):
    """测试检查抛出异常"""
    def exception_check():
        raise ValueError("Test exception")
    
    health_checker.add_check("exception_check", exception_check)
    config = health_checker.checks["exception_check"]
    
    result = await health_checker._execute_check("exception_check", config)
    
    assert result.status == "fail"
    assert "Test exception" in result.message


@pytest.mark.asyncio
async def test_perform_health_check_all_pass(health_checker):
    """测试所有检查项都通过的健康检查"""
    def check1():
        return True
    
    def check2():
        return {"status": "pass", "details": "OK"}
    
    health_checker.add_check("check1", check1, critical=True)
    health_checker.add_check("check2", check2, critical=False)
    
    status = await health_checker.perform_health_check()
    
    assert isinstance(status, HealthStatus)
    assert status.status == "healthy"
    assert status.version == "1.0.0"
    assert status.uptime > 0
    assert "check1" in status.checks
    assert "check2" in status.checks
    assert status.metadata["service_name"] == "test-service"


@pytest.mark.asyncio
async def test_perform_health_check_critical_fail(health_checker):
    """测试关键检查项失败"""
    def critical_fail():
        return False
    
    def non_critical_pass():
        return True
    
    health_checker.add_check("critical", critical_fail, critical=True)
    health_checker.add_check("non_critical", non_critical_pass, critical=False)
    
    status = await health_checker.perform_health_check()
    
    assert status.status == "unhealthy"  # 关键检查失败导致不健康


@pytest.mark.asyncio
async def test_perform_health_check_non_critical_fail(health_checker):
    """测试非关键检查项失败"""
    def critical_pass():
        return True
    
    def non_critical_fail():
        return False
    
    health_checker.add_check("critical", critical_pass, critical=True)
    health_checker.add_check("non_critical", non_critical_fail, critical=False)
    
    status = await health_checker.perform_health_check()
    
    assert status.status == "degraded"  # 非关键检查失败导致降级


@pytest.mark.asyncio
async def test_liveness_check(health_checker):
    """测试存活检查"""
    result = await health_checker.perform_liveness_check()
    
    assert result["status"] == "alive"
    assert "timestamp" in result
    assert "uptime" in result
    assert result["uptime"] >= 0


@pytest.mark.asyncio
async def test_readiness_check_success(health_checker):
    """测试就绪检查成功"""
    def ready_check():
        return True
    
    health_checker.add_check("database", ready_check, critical=True)
    
    result = await health_checker.perform_readiness_check()
    
    assert result["status"] == "ready"
    assert result["checks_passed"] == 1


@pytest.mark.asyncio 
async def test_readiness_check_failure(health_checker):
    """测试就绪检查失败"""
    def not_ready_check():
        return False
    
    health_checker.add_check("database", not_ready_check, critical=True)
    
    with pytest.raises(Exception):  # 应该抛出HTTPException
        await health_checker.perform_readiness_check()


def test_check_result_caching(health_checker):
    """测试检查结果缓存"""
    call_count = 0
    
    def counting_check():
        nonlocal call_count
        call_count += 1
        return True
    
    health_checker.add_check("cached_check", counting_check, cache=True)
    health_checker._cache_ttl = 1  # 1秒缓存
    
    # 第一次调用
    result1 = health_checker._get_cached_result("cached_check")
    assert result1 is None  # 没有缓存
    
    # 缓存结果
    check_result = CheckResult(status="pass", duration_ms=10.0)
    health_checker._cache_result("cached_check", check_result)
    
    # 第二次调用应该返回缓存
    result2 = health_checker._get_cached_result("cached_check")
    assert result2 is not None
    assert result2.status == "pass"


class TestCommonHealthChecks:
    """通用健康检查函数测试"""
    
    @pytest.mark.asyncio
    async def test_check_redis_success(self):
        """测试Redis连接检查成功"""
        with patch('aioredis.from_url') as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            result = await CommonHealthChecks.check_redis("redis://localhost:6379")
            
            assert result["status"] == "pass"
            assert result["connection"] == "ok"
            assert "response_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_check_redis_failure(self):
        """测试Redis连接检查失败"""
        with patch('aioredis.from_url') as mock_from_url:
            mock_from_url.side_effect = Exception("Connection failed")
            
            result = await CommonHealthChecks.check_redis("redis://localhost:6379")
            
            assert result["status"] == "fail"
            assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_check_http_service_success(self):
        """测试HTTP服务检查成功"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_session_instance = AsyncMock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            result = await CommonHealthChecks.check_http_service("http://localhost:8080/health")
            
            assert result["status"] == "pass"
            assert result["http_status"] == 200
    
    @pytest.mark.asyncio
    async def test_check_http_service_server_error(self):
        """测试HTTP服务返回服务器错误"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 503  # Service Unavailable
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            
            mock_session_instance = AsyncMock()
            mock_session_instance.get.return_value = mock_response
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            
            result = await CommonHealthChecks.check_http_service("http://localhost:8080/health")
            
            assert result["status"] == "fail"
            assert result["http_status"] == 503
    
    def test_check_disk_space_ok(self):
        """测试磁盘空间检查正常"""
        with patch('psutil.disk_usage') as mock_disk_usage:
            # 模拟60%的磁盘使用率
            mock_disk_usage.return_value = Mock(used=600, total=1000, free=400)
            
            result = CommonHealthChecks.check_disk_space(threshold=80.0)
            
            assert result["status"] == "pass"
            assert result["usage_percent"] == 60.0
            assert result["threshold"] == 80.0
    
    def test_check_disk_space_over_threshold(self):
        """测试磁盘空间超过阈值"""
        with patch('psutil.disk_usage') as mock_disk_usage:
            # 模拟90%的磁盘使用率
            mock_disk_usage.return_value = Mock(used=900, total=1000, free=100)
            
            result = CommonHealthChecks.check_disk_space(threshold=80.0)
            
            assert result["status"] == "fail"
            assert result["usage_percent"] == 90.0
    
    def test_check_memory_usage_ok(self):
        """测试内存使用率检查正常"""
        with patch('psutil.virtual_memory') as mock_memory:
            # 模拟60%的内存使用率
            mock_memory.return_value = Mock(
                percent=60.0, 
                available=400*1024*1024, 
                total=1000*1024*1024
            )
            
            result = CommonHealthChecks.check_memory_usage(threshold=80.0)
            
            assert result["status"] == "pass"
            assert result["usage_percent"] == 60.0
    
    def test_check_memory_usage_over_threshold(self):
        """测试内存使用率超过阈值"""
        with patch('psutil.virtual_memory') as mock_memory:
            # 模拟90%的内存使用率
            mock_memory.return_value = Mock(
                percent=90.0,
                available=100*1024*1024,
                total=1000*1024*1024
            )
            
            result = CommonHealthChecks.check_memory_usage(threshold=80.0)
            
            assert result["status"] == "fail"
            assert result["usage_percent"] == 90.0