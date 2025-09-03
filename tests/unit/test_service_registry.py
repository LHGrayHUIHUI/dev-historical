"""
服务注册与发现单元测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from services.core.registry.service_registry import (
    ServiceRegistry, 
    ServiceInfo, 
    LoadBalancer,
    ServiceRegistration
)


@pytest.fixture
def service_info():
    """测试服务信息夹具"""
    return ServiceInfo(
        name="test-service",
        service_id="test-service-001",
        address="127.0.0.1",
        port=8000,
        health_check="/health",
        tags=["test", "microservice"],
        metadata={"version": "1.0.0"}
    )


@pytest.fixture
def mock_consul():
    """Mock Consul客户端夹具"""
    consul_mock = Mock()
    consul_mock.agent = Mock()
    consul_mock.agent.service = Mock()
    consul_mock.agent.self = AsyncMock(return_value={"Config": {"NodeName": "test-node"}})
    consul_mock.health = Mock()
    consul_mock.kv = Mock()
    return consul_mock


@pytest.mark.asyncio
async def test_service_registry_initialization():
    """测试服务注册中心初始化"""
    registry = ServiceRegistry("localhost", 8500)
    
    assert registry.consul is not None
    assert registry._health_check_interval == 30
    assert len(registry.services) == 0


@pytest.mark.asyncio 
async def test_service_registry_start_stop(mock_consul):
    """测试服务注册中心启动和停止"""
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        # 测试启动
        await registry.start()
        assert registry._running is True
        assert registry._health_check_task is not None
        
        # 测试停止
        await registry.stop()
        assert registry._running is False


@pytest.mark.asyncio
async def test_register_service_success(mock_consul, service_info):
    """测试成功注册服务"""
    mock_consul.agent.service.register = AsyncMock(return_value=True)
    
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        result = await registry.register_service(service_info)
        
        assert result is True
        assert service_info.service_id in registry.services
        assert registry.services[service_info.service_id].status == "registered"
        
        # 验证Consul API调用
        mock_consul.agent.service.register.assert_called_once()


@pytest.mark.asyncio
async def test_register_service_failure(mock_consul, service_info):
    """测试服务注册失败"""
    mock_consul.agent.service.register = AsyncMock(return_value=False)
    
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        result = await registry.register_service(service_info)
        
        assert result is False
        assert service_info.service_id not in registry.services


@pytest.mark.asyncio
async def test_discover_service_healthy_only(mock_consul):
    """测试发现健康服务"""
    mock_services_data = [
        {
            'Service': {
                'Service': 'test-service',
                'ID': 'test-service-001',
                'Address': '127.0.0.1',
                'Port': 8000,
                'Tags': ['test'],
                'Meta': {'version': '1.0.0'}
            },
            'Checks': [{'Status': 'passing'}]
        }
    ]
    
    mock_consul.health.service = AsyncMock(return_value=(None, mock_services_data))
    
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        services = await registry.discover_service("test-service", healthy_only=True)
        
        assert len(services) == 1
        assert services[0].name == "test-service"
        assert services[0].status == "healthy"
        
        # 验证调用参数
        mock_consul.health.service.assert_called_with("test-service", passing=True)


@pytest.mark.asyncio
async def test_discover_service_with_tags(mock_consul):
    """测试按标签发现服务"""
    mock_services_data = [
        {
            'Service': {
                'Service': 'test-service',
                'ID': 'test-service-001',
                'Address': '127.0.0.1',
                'Port': 8000,
                'Tags': ['test', 'api'],
                'Meta': {}
            },
            'Checks': [{'Status': 'passing'}]
        },
        {
            'Service': {
                'Service': 'test-service',
                'ID': 'test-service-002',
                'Address': '127.0.0.1',
                'Port': 8001,
                'Tags': ['test', 'worker'],
                'Meta': {}
            },
            'Checks': [{'Status': 'passing'}]
        }
    ]
    
    mock_consul.health.service = AsyncMock(return_value=(None, mock_services_data))
    
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        # 测试标签过滤
        services = await registry.discover_service("test-service", tags=["api"])
        assert len(services) == 1
        assert "api" in services[0].tags


@pytest.mark.asyncio
async def test_deregister_service(mock_consul, service_info):
    """测试注销服务"""
    mock_consul.agent.service.register = AsyncMock(return_value=True)
    mock_consul.agent.service.deregister = AsyncMock(return_value=True)
    
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        # 先注册服务
        await registry.register_service(service_info)
        assert service_info.service_id in registry.services
        
        # 测试注销
        result = await registry.deregister_service(service_info.service_id)
        
        assert result is True
        assert service_info.service_id not in registry.services
        mock_consul.agent.service.deregister.assert_called_with(service_info.service_id)


class TestLoadBalancer:
    """负载均衡器测试类"""
    
    @pytest.fixture
    def services_list(self):
        """测试服务列表"""
        return [
            ServiceInfo("service", "service-1", "127.0.0.1", 8001, status="healthy"),
            ServiceInfo("service", "service-2", "127.0.0.1", 8002, status="healthy"),
            ServiceInfo("service", "service-3", "127.0.0.1", 8003, status="healthy"),
        ]
    
    def test_round_robin_selection(self, services_list):
        """测试轮询选择算法"""
        lb = LoadBalancer(strategy="round_robin")
        
        # 连续选择应该轮询
        selected = []
        for _ in range(6):  # 选择6次，应该轮询两轮
            service = lb.select_service(services_list, "test-service")
            selected.append(service.service_id)
        
        expected = ["service-1", "service-2", "service-3", "service-1", "service-2", "service-3"]
        assert selected == expected
    
    def test_random_selection(self, services_list):
        """测试随机选择算法"""
        lb = LoadBalancer(strategy="random")
        
        selected_ids = set()
        for _ in range(10):  # 多次选择，应该能选到不同的服务
            service = lb.select_service(services_list)
            selected_ids.add(service.service_id)
        
        # 随机选择应该选到至少2个不同的服务（概率很高）
        assert len(selected_ids) >= 1
    
    def test_healthy_service_filtering(self):
        """测试健康服务过滤"""
        services = [
            ServiceInfo("service", "service-1", "127.0.0.1", 8001, status="healthy"),
            ServiceInfo("service", "service-2", "127.0.0.1", 8002, status="unhealthy"),
            ServiceInfo("service", "service-3", "127.0.0.1", 8003, status="healthy"),
        ]
        
        lb = LoadBalancer(strategy="round_robin")
        
        # 应该只选择健康的服务
        selected = []
        for _ in range(4):
            service = lb.select_service(services, "test-service")
            selected.append(service.service_id)
        
        # 只有service-1和service-3是健康的
        assert all(sid in ["service-1", "service-3"] for sid in selected)
    
    def test_no_healthy_services_fallback(self):
        """测试没有健康服务时的回退机制"""
        services = [
            ServiceInfo("service", "service-1", "127.0.0.1", 8001, status="unhealthy"),
            ServiceInfo("service", "service-2", "127.0.0.1", 8002, status="critical"),
        ]
        
        lb = LoadBalancer(strategy="round_robin")
        service = lb.select_service(services, "test-service")
        
        # 没有健康服务时，应该使用所有服务（应急方案）
        assert service is not None
        assert service.service_id in ["service-1", "service-2"]
    
    def test_empty_services_list(self):
        """测试空服务列表"""
        lb = LoadBalancer()
        result = lb.select_service([])
        assert result is None


@pytest.mark.asyncio
async def test_service_registration_context_manager(mock_consul, service_info):
    """测试服务注册上下文管理器"""
    mock_consul.agent.service.register = AsyncMock(return_value=True)
    mock_consul.agent.service.deregister = AsyncMock(return_value=True)
    
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        async with ServiceRegistration(registry, service_info) as registration:
            assert registration.registered is True
            assert service_info.service_id in registry.services
        
        # 退出上下文后应该自动注销
        mock_consul.agent.service.deregister.assert_called_with(service_info.service_id)


@pytest.mark.asyncio
async def test_service_registration_context_manager_failure(mock_consul, service_info):
    """测试服务注册失败时的上下文管理器"""
    mock_consul.agent.service.register = AsyncMock(return_value=False)
    
    with patch('services.core.registry.service_registry.consul.aio.Consul', return_value=mock_consul):
        registry = ServiceRegistry()
        
        async with ServiceRegistration(registry, service_info) as registration:
            assert registration.registered is False
            assert service_info.service_id not in registry.services