"""
监控服务单元测试

测试监控服务的核心功能，包括：
- 服务初始化和配置
- 监控组件集成
- API端点功能
- 告警和日志功能

Author: 开发团队
Created: 2025-09-04
Version: 1.0.0
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from services.core.monitoring.monitoring_service import MonitoringService, create_monitoring_service
from services.core.monitoring.alert_service import AlertSeverity, AlertStatus
from services.core.monitoring.logging_service import LogLevel


class TestMonitoringService:
    """监控服务测试类"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return {
            "service": {
                "name": "test-monitoring-service",
                "version": "1.0.0",
                "environment": "test",
                "host": "0.0.0.0",
                "port": 8004,
                "debug": True
            },
            "metrics": {
                "enabled": True,
                "prometheus_enabled": True,
                "business_metrics_enabled": True,
                "collect_interval": 15
            },
            "tracing": {
                "enabled": False,  # 测试中禁用以避免外部依赖
                "jaeger_endpoint": "http://localhost:14268/api/traces",
                "sampling_rate": 0.1,
                "service_name": "test-service"
            },
            "alerting": {
                "enabled": True,
                "check_interval": 60,
                "email_enabled": False,
                "slack_enabled": False,
                "smtp_server": None,
                "smtp_port": 587,
                "smtp_username": None,
                "smtp_password": None,
                "alert_emails": [],
                "slack_webhook_url": None
            },
            "logging": {
                "enabled": True,
                "level": "DEBUG",
                "elasticsearch_enabled": False,
                "elasticsearch_url": "http://localhost:9200",
                "log_dir": "test_logs",
                "max_file_size": 10,
                "backup_count": 3
            }
        }
    
    @pytest.fixture
    def monitoring_service(self, mock_config):
        """创建监控服务实例"""
        return MonitoringService(
            service_name="test-monitoring-service",
            config=mock_config
        )
    
    def test_monitoring_service_initialization(self, monitoring_service, mock_config):
        """测试监控服务初始化"""
        assert monitoring_service.service_name == "test-monitoring-service"
        assert monitoring_service.config == mock_config
        assert not monitoring_service.is_running
        assert monitoring_service.app is None
    
    def test_load_default_config(self):
        """测试默认配置加载"""
        service = MonitoringService()
        
        assert "service" in service.config
        assert "metrics" in service.config
        assert "tracing" in service.config
        assert "alerting" in service.config
        assert "logging" in service.config
        
        # 验证必需字段
        assert service.config["service"]["name"] == "monitoring-service"
        assert service.config["metrics"]["enabled"] is True
        assert service.config["logging"]["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    @pytest.mark.asyncio
    async def test_initialize_logging(self, monitoring_service):
        """测试日志服务初始化"""
        with patch('services.core.monitoring.monitoring_service.get_logging_service') as mock_get_logging:
            mock_logging_service = AsyncMock()
            mock_get_logging.return_value = mock_logging_service
            
            await monitoring_service._initialize_logging()
            
            assert monitoring_service.logging_service == mock_logging_service
            mock_get_logging.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_alerting(self, monitoring_service):
        """测试告警管理初始化"""
        with patch('services.core.monitoring.monitoring_service.get_alert_manager') as mock_get_alert_manager:
            mock_alert_manager = AsyncMock()
            mock_get_alert_manager.return_value = mock_alert_manager
            
            with patch('services.core.monitoring.monitoring_service.create_default_alert_rules') as mock_create_rules:
                mock_rules = [MagicMock()]
                mock_create_rules.return_value = mock_rules
                
                await monitoring_service._initialize_alerting()
                
                assert monitoring_service.alert_manager == mock_alert_manager
                mock_alert_manager.add_rule.assert_called()
                mock_alert_manager.start_monitoring.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_fastapi_app(self, monitoring_service):
        """测试FastAPI应用创建"""
        with patch('services.core.monitoring.monitoring_service.get_business_metrics'):
            await monitoring_service._create_fastapi_app()
            
            assert monitoring_service.app is not None
            assert monitoring_service.app.title == "历史文本处理项目 - 监控服务"
            assert monitoring_service.app.version == monitoring_service.config["service"]["version"]
    
    @pytest.mark.asyncio
    async def test_monitoring_endpoints(self, monitoring_service):
        """测试监控API端点"""
        # 模拟告警管理器
        mock_alert_manager = AsyncMock()
        monitoring_service.alert_manager = mock_alert_manager
        
        # 模拟活跃告警
        mock_alert = MagicMock()
        mock_alert.to_dict.return_value = {
            "id": "test-alert-1",
            "status": "active",
            "severity": "warning",
            "message": "测试告警"
        }
        mock_alert_manager.get_active_alerts.return_value = [mock_alert]
        
        await monitoring_service._create_fastapi_app()
        
        # 测试应用已创建
        assert monitoring_service.app is not None
        
        # 创建测试客户端
        client = TestClient(monitoring_service.app)
        
        # 测试根路径
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "test-monitoring-service"
        assert "timestamp" in data
        
        # 测试健康检查
        response = client.get("/api/v1/monitoring/health")
        assert response.status_code == 200
        
        # 测试获取告警
        response = client.get("/api/v1/monitoring/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert data["total"] == 1
    
    def test_create_monitoring_service_factory(self):
        """测试监控服务工厂函数"""
        service = create_monitoring_service(
            service_name="factory-test-service",
            config={"test": "config"}
        )
        
        assert isinstance(service, MonitoringService)
        assert service.service_name == "factory-test-service"
        assert service.config["test"] == "config"


class TestMonitoringIntegration:
    """监控系统集成测试类"""
    
    @pytest.fixture
    def integration_config(self):
        """集成测试配置"""
        return {
            "service": {
                "name": "integration-monitoring-service",
                "version": "1.0.0",
                "environment": "test",
                "host": "127.0.0.1",
                "port": 0,  # 使用随机端口
                "debug": True
            },
            "metrics": {"enabled": True, "prometheus_enabled": True},
            "tracing": {"enabled": False},
            "alerting": {"enabled": False},  # 集成测试中简化告警
            "logging": {"enabled": True, "level": "INFO", "elasticsearch_enabled": False}
        }
    
    @pytest.mark.asyncio
    async def test_full_service_lifecycle(self, integration_config):
        """测试完整的服务生命周期"""
        service = MonitoringService(config=integration_config)
        
        try:
            # 初始化服务
            await service.initialize()
            
            # 验证初始化后状态
            assert service.app is not None
            assert service.logging_service is not None
            
            # 验证应用路由
            client = TestClient(service.app)
            
            # 测试基本端点
            response = client.get("/")
            assert response.status_code == 200
            
            response = client.get("/api/v1/monitoring/health")
            assert response.status_code == 200
            
            response = client.get("/api/v1/monitoring/system")
            assert response.status_code == 200
            
            # 测试指标端点
            response = client.get("/api/v1/monitoring/metrics")
            assert response.status_code == 200
            
        finally:
            # 确保服务正确关闭
            await service.shutdown()
    
    @pytest.mark.asyncio 
    async def test_monitoring_middleware_integration(self, integration_config):
        """测试监控中间件集成"""
        service = MonitoringService(config=integration_config)
        
        try:
            await service.initialize()
            client = TestClient(service.app)
            
            # 发送几个请求以生成指标
            for i in range(5):
                response = client.get("/api/v1/monitoring/health")
                assert response.status_code == 200
            
            # 检查Prometheus指标
            response = client.get("/api/v1/monitoring/metrics")
            assert response.status_code == 200
            
            metrics_data = response.content.decode('utf-8')
            
            # 验证基本指标存在
            assert "http_requests_total" in metrics_data
            assert "http_request_duration_seconds" in metrics_data
            
        finally:
            await service.shutdown()


class TestMonitoringComponentsIntegration:
    """监控组件集成测试"""
    
    @pytest.mark.asyncio
    async def test_alert_manager_integration(self):
        """测试告警管理器集成"""
        from services.core.monitoring.alert_service import get_alert_manager, AlertRule
        
        alert_manager = get_alert_manager()
        
        # 创建测试告警规则
        test_rule = AlertRule(
            name="test_rule",
            query="up",
            condition="== 0",
            duration=60,
            severity=AlertSeverity.WARNING,
            summary="测试告警",
            description="这是一个测试告警规则"
        )
        
        alert_manager.add_rule(test_rule)
        
        # 验证规则已添加
        assert "test_rule" in alert_manager.rules
        assert alert_manager.rules["test_rule"].name == "test_rule"
        assert alert_manager.rules["test_rule"].severity == AlertSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_logging_service_integration(self):
        """测试日志服务集成"""
        from services.core.monitoring.logging_service import create_default_log_config, get_logging_service
        
        config = create_default_log_config(
            service_name="test-service",
            log_level=LogLevel.DEBUG,
            elasticsearch_enabled=False
        )
        
        logging_service = get_logging_service(config)
        
        # 测试日志记录器
        logger = logging_service.bind_context(component="test")
        assert logger is not None
        
        # 测试请求上下文绑定
        request_logger = logging_service.bind_request_context(
            request_id="test-req-123",
            user_id="test-user-456"
        )
        assert request_logger is not None
    
    def test_business_metrics_integration(self):
        """测试业务指标集成"""
        from services.core.monitoring.metrics_middleware import get_business_metrics
        
        metrics = get_business_metrics("test-service")
        
        # 测试文件上传指标
        metrics.record_file_upload("success", "pdf")
        metrics.record_file_upload("error", "docx")
        
        # 测试文本处理指标
        metrics.record_text_processing("extract", "success", 2.5)
        metrics.record_text_processing("analyze", "error", 1.2)
        
        # 测试OCR操作指标
        metrics.record_ocr_operation("success", "zh")
        metrics.record_ocr_operation("low_confidence", "en")
        
        # 测试病毒扫描指标
        metrics.record_virus_scan("clean")
        
        # 测试队列大小更新
        metrics.update_queue_size("ocr", 5)
        metrics.update_queue_size("text_analysis", 12)
        
        # 验证指标收集器存在
        assert metrics.service_name == "test-service"


@pytest.mark.integration
class TestMonitoringEndToEnd:
    """端到端监控测试"""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self):
        """测试完整的监控工作流程"""
        # 创建配置，启用所有功能（除了外部依赖）
        config = {
            "service": {
                "name": "e2e-monitoring-service",
                "version": "1.0.0",
                "environment": "test",
                "host": "127.0.0.1",
                "port": 0,
                "debug": True
            },
            "metrics": {"enabled": True, "prometheus_enabled": True},
            "tracing": {"enabled": False},  # 避免Jaeger依赖
            "alerting": {"enabled": True, "email_enabled": False, "slack_enabled": False},
            "logging": {"enabled": True, "level": "INFO", "elasticsearch_enabled": False}
        }
        
        service = MonitoringService(config=config)
        
        try:
            # 1. 初始化服务
            await service.initialize()
            assert service.is_running is False  # 还未启动HTTP服务
            
            # 2. 验证所有组件已初始化
            assert service.app is not None
            assert service.logging_service is not None
            assert service.alert_manager is not None
            
            # 3. 创建测试客户端并测试所有端点
            client = TestClient(service.app)
            
            # 根路径
            response = client.get("/")
            assert response.status_code == 200
            
            # 健康检查
            response = client.get("/api/v1/monitoring/health")
            assert response.status_code == 200
            
            # 系统信息
            response = client.get("/api/v1/monitoring/system")
            assert response.status_code == 200
            
            # 指标端点
            response = client.get("/api/v1/monitoring/metrics")
            assert response.status_code == 200
            
            # 告警端点
            response = client.get("/api/v1/monitoring/alerts")
            assert response.status_code == 200
            
            # 服务状态
            response = client.get("/api/v1/monitoring/status")
            assert response.status_code == 200
            
            print("✅ 端到端监控测试通过")
            
        finally:
            await service.shutdown()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])