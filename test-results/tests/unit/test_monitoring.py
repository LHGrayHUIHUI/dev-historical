"""
监控系统单元测试模块

此模块包含针对监控中间件、监控控制器、业务指标收集器
等监控组件的完整单元测试套件。

测试覆盖范围：
- PrometheusMetricsMiddleware功能测试
- BusinessMetricsCollector指标记录测试
- MonitoringController API端点测试
- 错误处理和边界条件测试

Author: 开发团队
Created: 2025-09-03
Version: 1.0.0
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import FastAPI, Request, Response, HTTPException
from prometheus_client import CollectorRegistry, REGISTRY
from fastapi.testclient import TestClient
import asyncio
import json
import time
from datetime import datetime, timedelta

# 导入被测试的模块
from services.core.monitoring.metrics_middleware import (
    PrometheusMetricsMiddleware,
    BusinessMetricsCollector,
    get_business_metrics,
    generate_metrics
)
from services.core.monitoring.monitoring_controller import (
    router as monitoring_router,
    health_check,
    readiness_check,
    get_metrics,
    get_system_info,
    get_service_metrics,
    get_service_status
)


@pytest.fixture(autouse=True)
def clear_registry():
    """自动清理Prometheus registry以避免指标重复注册"""
    # 在每个测试前清理registry中的所有收集器
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass  # 忽略已经被移除的收集器
    yield
    # 测试后再次清理
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass


class TestPrometheusMetricsMiddleware:
    """Prometheus监控中间件测试类
    
    测试HTTP请求监控指标的收集、处理和记录功能，
    包括正常请求处理、异常处理、指标标准化等场景。
    """
    
    @pytest.fixture
    def mock_app(self):
        """创建模拟ASGI应用"""
        return Mock()
    
    @pytest.fixture
    def middleware(self, mock_app):
        """创建监控中间件实例"""
        return PrometheusMetricsMiddleware(
            app=mock_app,
            service_name="test-service"
        )
    
    @pytest.fixture
    def mock_request(self):
        """创建模拟HTTP请求"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/v1/documents/123"
        request.client.host = "127.0.0.1"
        return request
    
    @pytest.fixture
    def mock_response(self):
        """创建模拟HTTP响应"""
        response = Mock(spec=Response)
        response.status_code = 200
        response.headers = {}
        return response
    
    def test_middleware_initialization(self, mock_app):
        """测试中间件初始化
        
        验证中间件实例创建时各种指标收集器的正确初始化。
        """
        middleware = PrometheusMetricsMiddleware(mock_app, "test-service")
        
        # 验证服务名称设置
        assert middleware.service_name == "test-service"
        
        # 验证指标收集器实例化
        assert middleware.http_requests_total is not None
        assert middleware.http_request_duration is not None
        assert middleware.active_connections is not None
        assert middleware.application_info is not None
    
    @pytest.mark.asyncio
    async def test_successful_request_processing(
        self, 
        middleware, 
        mock_request, 
        mock_response
    ):
        """测试成功请求处理流程
        
        验证正常HTTP请求的监控指标收集和记录过程。
        """
        # 模拟下一个中间件处理函数
        async def mock_call_next(request):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return mock_response
        
        # 执行请求处理
        result = await middleware.dispatch(mock_request, mock_call_next)
        
        # 验证响应结果
        assert result == mock_response
        assert "X-Service-Name" in result.headers
        assert "X-Response-Time" in result.headers
        assert result.headers["X-Service-Name"] == "test-service"
        
        # 验证响应时间格式
        response_time = result.headers["X-Response-Time"]
        assert response_time.endswith("s")
        assert float(response_time[:-1]) > 0
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, middleware, mock_request):
        """测试异常处理流程
        
        验证请求处理过程中发生异常时的监控指标记录。
        """
        # 模拟抛出异常的处理函数
        async def mock_call_next_with_error(request):
            await asyncio.sleep(0.05)
            raise ValueError("模拟处理异常")
        
        # 执行请求处理并期待异常
        with pytest.raises(ValueError, match="模拟处理异常"):
            await middleware.dispatch(mock_request, mock_call_next_with_error)
    
    def test_endpoint_normalization(self, middleware):
        """测试端点路径标准化
        
        验证动态参数路径的标准化处理功能。
        """
        test_cases = [
            ("/api/v1/documents/123", "/api/v1/documents/{id}"),
            ("/users/456/profile", "/users/{id}/profile"),
            (
                "/api/v1/files/abc123def456.pdf", 
                "/api/v1/files/file.{ext}"
            ),
            (
                "/documents/550e8400-e29b-41d4-a716-446655440000",
                "/documents/{uuid}"
            ),
            ("/static/images/logo.png", "/static/images/file.{ext}"),
            ("/api/v1/users", "/api/v1/users")  # 无变化情况
        ]
        
        for original_path, expected_path in test_cases:
            normalized = middleware._normalize_endpoint(original_path)
            assert normalized == expected_path, \
                f"路径 {original_path} 标准化失败，期待 {expected_path}，实际 {normalized}"
    
    def test_metrics_recording(self, middleware):
        """测试指标记录功能
        
        验证HTTP请求指标的正确记录和统计。
        """
        # 模拟指标记录
        with patch.object(middleware.http_requests_total, 'labels') as mock_counter_labels:
            with patch.object(middleware.http_request_duration, 'labels') as mock_histogram_labels:
                mock_counter = Mock()
                mock_histogram = Mock()
                mock_counter_labels.return_value = mock_counter
                mock_histogram_labels.return_value = mock_histogram
                
                # 执行指标记录
                middleware._record_request_metrics(
                    method="GET",
                    endpoint="/api/v1/test",
                    status="200",
                    duration=0.15
                )
                
                # 验证计数器调用
                mock_counter_labels.assert_called_once_with(
                    method="GET",
                    endpoint="/api/v1/test",
                    status="200",
                    service="test-service"
                )
                mock_counter.inc.assert_called_once()
                
                # 验证直方图调用
                mock_histogram_labels.assert_called_once_with(
                    method="GET",
                    endpoint="/api/v1/test",
                    status="200",
                    service="test-service"
                )
                mock_histogram.observe.assert_called_once_with(0.15)


class TestBusinessMetricsCollector:
    """业务指标收集器测试类
    
    测试业务特定指标的收集和记录功能，包括文件处理、
    OCR操作、认证事件等业务场景的监控指标。
    """
    
    @pytest.fixture
    def metrics_collector(self):
        """创建业务指标收集器实例"""
        return BusinessMetricsCollector("test-service")
    
    def test_collector_initialization(self, metrics_collector):
        """测试指标收集器初始化
        
        验证业务指标收集器实例创建时的正确初始化。
        """
        assert metrics_collector.service_name == "test-service"
        assert metrics_collector.file_uploads_total is not None
        assert metrics_collector.text_processing_duration is not None
        assert metrics_collector.ocr_operations_total is not None
        assert metrics_collector.file_virus_scan_total is not None
        assert metrics_collector.text_processing_queue_size is not None
        assert metrics_collector.auth_login_attempts_total is not None
    
    def test_file_upload_metrics(self, metrics_collector):
        """测试文件上传指标记录
        
        验证文件上传成功、失败等状态的指标记录功能。
        """
        with patch.object(metrics_collector.file_uploads_total, 'labels') as mock_labels:
            mock_counter = Mock()
            mock_labels.return_value = mock_counter
            
            # 记录成功上传
            metrics_collector.record_file_upload("success", "pdf")
            
            mock_labels.assert_called_with(
                status="success",
                file_type="pdf",
                service="test-service"
            )
            mock_counter.inc.assert_called_once()
            
            # 重置mock
            mock_labels.reset_mock()
            mock_counter.reset_mock()
            
            # 记录失败上传
            metrics_collector.record_file_upload("error", "docx")
            
            mock_labels.assert_called_with(
                status="error",
                file_type="docx",
                service="test-service"
            )
            mock_counter.inc.assert_called_once()
    
    def test_text_processing_metrics(self, metrics_collector):
        """测试文本处理指标记录
        
        验证文本处理时间和状态的指标记录功能。
        """
        with patch.object(metrics_collector.text_processing_duration, 'labels') as mock_labels:
            mock_histogram = Mock()
            mock_labels.return_value = mock_histogram
            
            # 记录处理时间
            metrics_collector.record_text_processing(
                operation="extract",
                status="success", 
                duration=2.5
            )
            
            mock_labels.assert_called_with(
                operation="extract",
                status="success",
                service="test-service"
            )
            mock_histogram.observe.assert_called_once_with(2.5)
    
    def test_ocr_operation_metrics(self, metrics_collector):
        """测试OCR操作指标记录
        
        验证OCR识别操作的成功、失败状态记录功能。
        """
        with patch.object(metrics_collector.ocr_operations_total, 'labels') as mock_labels:
            mock_counter = Mock()
            mock_labels.return_value = mock_counter
            
            # 记录成功OCR操作
            metrics_collector.record_ocr_operation("success", "zh")
            
            mock_labels.assert_called_with(
                status="success",
                language="zh",
                service="test-service"
            )
            mock_counter.inc.assert_called_once()
    
    def test_virus_scan_metrics(self, metrics_collector):
        """测试病毒扫描指标记录
        
        验证文件安全扫描结果的指标记录功能。
        """
        with patch.object(metrics_collector.file_virus_scan_total, 'labels') as mock_labels:
            mock_counter = Mock()
            mock_labels.return_value = mock_counter
            
            # 记录清洁扫描结果
            metrics_collector.record_virus_scan("clean")
            
            mock_labels.assert_called_with(
                result="clean",
                service="test-service"
            )
            mock_counter.inc.assert_called_once()
    
    def test_queue_size_updates(self, metrics_collector):
        """测试队列大小指标更新
        
        验证处理队列大小的实时监控功能。
        """
        with patch.object(metrics_collector.text_processing_queue_size, 'labels') as mock_labels:
            mock_gauge = Mock()
            mock_labels.return_value = mock_gauge
            
            # 更新队列大小
            metrics_collector.update_queue_size("ocr", 25)
            
            mock_labels.assert_called_with(
                queue_type="ocr",
                service="test-service"
            )
            mock_gauge.set.assert_called_once_with(25)
    
    def test_auth_attempt_metrics(self, metrics_collector):
        """测试认证尝试指标记录
        
        验证用户登录尝试的成功、失败状态记录功能。
        """
        with patch.object(metrics_collector.auth_login_attempts_total, 'labels') as mock_labels:
            mock_counter = Mock()
            mock_labels.return_value = mock_counter
            
            # 记录认证失败
            metrics_collector.record_auth_attempt("failed", "password")
            
            mock_labels.assert_called_with(
                status="failed",
                method="password",
                service="test-service"
            )
            mock_counter.inc.assert_called_once()
    
    def test_get_business_metrics_singleton(self):
        """测试业务指标收集器单例模式
        
        验证全局指标收集器实例的单例模式实现。
        """
        # 首次获取实例
        instance1 = get_business_metrics("service1")
        assert instance1 is not None
        assert instance1.service_name == "service1"
        
        # 再次获取应返回同一实例
        instance2 = get_business_metrics("service2")
        assert instance1 is instance2
        assert instance2.service_name == "service1"  # 应保持原有服务名


class TestMonitoringController:
    """监控控制器测试类
    
    测试监控API端点的功能，包括健康检查、系统信息查询、
    指标数据获取等REST API的正确性和错误处理。
    """
    
    @pytest.fixture
    def app(self):
        """创建FastAPI测试应用"""
        app = FastAPI()
        app.include_router(monitoring_router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self):
        """测试健康检查端点
        
        验证健康检查API返回正确的服务状态信息。
        """
        with patch('services.core.monitoring.monitoring_controller.get_service_name') as mock_name:
            with patch('services.core.monitoring.monitoring_controller.get_service_version') as mock_version:
                with patch('services.core.monitoring.monitoring_controller.get_environment') as mock_env:
                    mock_name.return_value = "test-service"
                    mock_version.return_value = "1.0.0"
                    mock_env.return_value = "test"
                    
                    result = await health_check()
                    
                    assert result.status == "healthy"
                    assert result.service_name == "test-service"
                    assert result.version == "1.0.0"
                    assert result.environment == "test"
                    assert result.uptime_seconds >= 0
                    assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_readiness_check_endpoint(self):
        """测试就绪检查端点
        
        验证就绪检查API对服务依赖状态的检查功能。
        """
        result = await readiness_check()
        
        assert result["status"] == "ready"
        assert "timestamp" in result
        assert "checks" in result
        assert result["checks"]["service"] == "ready"
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """测试指标获取端点
        
        验证Prometheus格式指标数据的正确生成和返回。
        """
        with patch('services.core.monitoring.monitoring_controller.generate_latest') as mock_generate:
            mock_generate.return_value = b"# Prometheus metrics data"
            
            response = await get_metrics()
            
            assert response.status_code == 200
            assert b"# Prometheus metrics data" in response.body
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_system_info_endpoint(self):
        """测试系统信息端点
        
        验证系统硬件和资源信息的查询功能。
        """
        with patch('psutil.virtual_memory') as mock_memory:
            with patch('psutil.cpu_count') as mock_cpu_count:
                with patch('psutil.disk_partitions') as mock_partitions:
                    with patch('psutil.disk_usage') as mock_disk_usage:
                        with patch('platform.node') as mock_hostname:
                            with patch('platform.system') as mock_system:
                                with patch('platform.release') as mock_release:
                                    # 配置模拟数据
                                    mock_memory_obj = Mock()
                                    mock_memory_obj.total = 8589934592  # 8GB
                                    mock_memory_obj.available = 4294967296  # 4GB
                                    mock_memory_obj.percent = 50.0
                                    mock_memory.return_value = mock_memory_obj
                                    
                                    mock_cpu_count.return_value = 4
                                    mock_partitions.return_value = []
                                    mock_hostname.return_value = "test-host"
                                    mock_system.return_value = "Linux"
                                    mock_release.return_value = "5.4.0"
                                    
                                    result = await get_system_info()
                                    
                                    assert result.hostname == "test-host"
                                    assert result.platform == "Linux 5.4.0"
                                    assert result.cpu_count == 4
                                    assert result.memory_total == 8589934592
                                    assert result.memory_available == 4294967296
                                    assert result.memory_percent == 50.0
    
    @pytest.mark.asyncio
    async def test_service_metrics_endpoint(self):
        """测试服务指标汇总端点
        
        验证服务关键指标的汇总查询功能。
        """
        with patch('services.core.monitoring.monitoring_controller.get_service_name') as mock_name:
            mock_name.return_value = "test-service"
            
            result = await get_service_metrics()
            
            assert isinstance(result, list)
            assert len(result) == 1
            
            service_metric = result[0]
            assert service_metric.service_name == "test-service"
            assert service_metric.uptime_seconds >= 0
            assert service_metric.http_requests_total >= 0
            assert service_metric.error_rate >= 0.0
    
    @pytest.mark.asyncio
    async def test_service_status_endpoint(self):
        """测试服务状态概览端点
        
        验证服务整体状态信息的汇总查询功能。
        """
        with patch('services.core.monitoring.monitoring_controller.get_service_name') as mock_name:
            with patch('services.core.monitoring.monitoring_controller.get_service_version') as mock_version:
                with patch('services.core.monitoring.monitoring_controller.get_environment') as mock_env:
                    with patch('psutil.virtual_memory') as mock_memory:
                        with patch('psutil.cpu_percent') as mock_cpu:
                            mock_name.return_value = "test-service"
                            mock_version.return_value = "1.0.0"
                            mock_env.return_value = "test"
                            
                            mock_memory_obj = Mock()
                            mock_memory_obj.percent = 60.0
                            mock_memory_obj.available = 3221225472  # 3GB
                            mock_memory.return_value = mock_memory_obj
                            mock_cpu.return_value = 25.5
                            
                            result = await get_service_status()
                            
                            assert result["service"]["name"] == "test-service"
                            assert result["service"]["version"] == "1.0.0"
                            assert result["service"]["environment"] == "test"
                            assert result["service"]["status"] == "running"
                            assert result["system"]["cpu_percent"] == 25.5
                            assert result["system"]["memory_percent"] == 60.0
    
    def test_health_check_api_endpoint(self, client):
        """测试健康检查HTTP API端点
        
        验证通过HTTP客户端访问健康检查API的功能。
        """
        with patch('services.core.monitoring.monitoring_controller.get_service_name') as mock_name:
            with patch('services.core.monitoring.monitoring_controller.get_service_version') as mock_version:
                with patch('services.core.monitoring.monitoring_controller.get_environment') as mock_env:
                    mock_name.return_value = "test-service"
                    mock_version.return_value = "1.0.0"
                    mock_env.return_value = "test"
                    
                    response = client.get("/monitoring/health")
                    
                    assert response.status_code == 200
                    data = response.json()
                    
                    assert data["status"] == "healthy"
                    assert data["service_name"] == "test-service"
                    assert data["version"] == "1.0.0"
                    assert data["environment"] == "test"
    
    def test_metrics_api_endpoint(self, client):
        """测试指标获取HTTP API端点
        
        验证通过HTTP客户端获取Prometheus指标的功能。
        """
        with patch('services.core.monitoring.monitoring_controller.generate_latest') as mock_generate:
            mock_generate.return_value = b"# TYPE http_requests_total counter\n"
            
            response = client.get("/monitoring/metrics")
            
            assert response.status_code == 200
            assert b"# TYPE http_requests_total counter" in response.content
    
    def test_error_handling_in_health_check(self, client):
        """测试健康检查错误处理
        
        验证健康检查API在异常情况下的错误处理机制。
        """
        with patch('services.core.monitoring.monitoring_controller.get_service_name') as mock_name:
            mock_name.side_effect = Exception("服务名称获取失败")
            
            response = client.get("/monitoring/health")
            
            assert response.status_code == 500
            data = response.json()
            assert "健康检查失败" in data["detail"]


class TestIntegrationScenarios:
    """集成测试场景类
    
    测试监控组件之间的集成功能，包括中间件与控制器的
    协同工作、指标数据的端到端流转等复杂场景。
    """
    
    @pytest.fixture
    def app_with_monitoring(self):
        """创建包含监控功能的完整应用"""
        app = FastAPI(title="测试应用")
        
        # 添加监控中间件
        app.add_middleware(PrometheusMetricsMiddleware, service_name="test-app")
        
        # 添加监控路由
        app.include_router(monitoring_router)
        
        # 添加测试路由
        @app.get("/test")
        async def test_endpoint():
            return {"message": "测试成功"}
        
        @app.get("/test/error")
        async def test_error_endpoint():
            raise HTTPException(status_code=500, detail="模拟错误")
        
        return app
    
    @pytest.fixture
    def client_with_monitoring(self, app_with_monitoring):
        """创建包含监控功能的测试客户端"""
        return TestClient(app_with_monitoring)
    
    def test_end_to_end_request_monitoring(self, client_with_monitoring):
        """测试端到端请求监控
        
        验证从HTTP请求到指标收集的完整监控流程。
        """
        # 发送正常请求
        response = client_with_monitoring.get("/test")
        assert response.status_code == 200
        assert response.json() == {"message": "测试成功"}
        
        # 验证监控响应头
        assert "X-Service-Name" in response.headers
        assert "X-Response-Time" in response.headers
        assert response.headers["X-Service-Name"] == "test-app"
        
        # 获取监控指标
        metrics_response = client_with_monitoring.get("/monitoring/metrics")
        assert metrics_response.status_code == 200
        
        # 验证指标数据包含请求统计
        metrics_data = metrics_response.content.decode()
        assert "http_requests_total" in metrics_data
        assert "http_request_duration_seconds" in metrics_data
    
    def test_error_request_monitoring(self, client_with_monitoring):
        """测试错误请求监控
        
        验证错误请求的监控指标收集和记录。
        """
        # 发送会触发错误的请求
        response = client_with_monitoring.get("/test/error")
        assert response.status_code == 500
        
        # 验证仍有监控响应头
        assert "X-Service-Name" in response.headers
        assert "X-Response-Time" in response.headers
        
        # 获取监控指标验证错误统计
        metrics_response = client_with_monitoring.get("/monitoring/metrics")
        assert metrics_response.status_code == 200
    
    def test_concurrent_request_monitoring(self, client_with_monitoring):
        """测试并发请求监控
        
        验证在高并发情况下监控系统的稳定性和准确性。
        """
        import concurrent.futures
        
        def send_request():
            try:
                response = client_with_monitoring.get("/test")
                return response.status_code
            except Exception as e:
                # 记录错误但继续测试
                print(f"请求失败: {e}")
                return 0
        
        # 发送并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request) for _ in range(10)]
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=5)
                    results.append(result)
                except Exception as e:
                    print(f"获取结果失败: {e}")
                    results.append(0)
        
        # 验证大部分请求都成功（允许少量失败）
        successful_requests = [status for status in results if status == 200]
        assert len(successful_requests) >= 8  # 至少80%成功率
        assert len(results) == 10
        
        # 验证监控指标正常生成
        metrics_response = client_with_monitoring.get("/monitoring/metrics")
        assert metrics_response.status_code == 200


if __name__ == "__main__":
    """运行测试套件
    
    执行完整的监控系统单元测试，包括覆盖率统计。
    """
    pytest.main([
        __file__,
        "-v",
        "--cov=services.core.monitoring",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80"
    ])