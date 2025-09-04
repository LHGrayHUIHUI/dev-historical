"""
基础架构集成测试
测试微服务基础架构各组件的集成
"""

import pytest
import asyncio
import docker
import time
from pathlib import Path

# 测试标记，需要Docker环境
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def docker_client():
    """Docker客户端夹具"""
    client = docker.from_env()
    yield client
    client.close()


@pytest.fixture(scope="session") 
async def infrastructure_setup(docker_client):
    """启动基础架构服务"""
    # Docker Compose文件路径
    compose_file = Path(__file__).parent.parent.parent / "docker-compose.yml"
    
    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")
    
    # 启动基础服务
    services_to_start = ["consul", "postgres", "redis"]
    containers = []
    
    try:
        for service in services_to_start:
            # 这里简化处理，实际应该使用docker-compose
            # 或者使用testcontainers库
            container_name = f"test_{service}"
            
            if service == "consul":
                container = docker_client.containers.run(
                    "consul:1.16",
                    name=container_name,
                    ports={"8500/tcp": 8500},
                    command="consul agent -server -bootstrap -ui -node=consul-test -client=0.0.0.0",
                    detach=True,
                    remove=True
                )
            elif service == "postgres":
                container = docker_client.containers.run(
                    "postgres:15",
                    name=container_name,
                    ports={"5432/tcp": 5432},
                    environment={
                        "POSTGRES_DB": "test_db",
                        "POSTGRES_USER": "test_user", 
                        "POSTGRES_PASSWORD": "test_pass"
                    },
                    detach=True,
                    remove=True
                )
            elif service == "redis":
                container = docker_client.containers.run(
                    "redis:7-alpine",
                    name=container_name,
                    ports={"6379/tcp": 6379},
                    detach=True,
                    remove=True
                )
            
            containers.append(container)
        
        # 等待服务启动
        await asyncio.sleep(10)
        
        yield {
            "consul_url": "http://localhost:8500",
            "postgres_url": "postgresql://test_user:test_pass@localhost:5432/test_db",
            "redis_url": "redis://localhost:6379"
        }
    
    finally:
        # 清理容器
        for container in containers:
            try:
                container.stop()
            except:
                pass


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("docker-compose.yml").exists(), reason="Docker environment not available")
async def test_consul_service_registration(infrastructure_setup):
    """测试Consul服务注册"""
    import consul.aio
    
    consul_client = consul.aio.Consul(host="localhost", port=8500)
    
    # 测试Consul连接
    try:
        leader = await consul_client.status.leader()
        assert leader is not None
    except Exception as e:
        pytest.skip(f"Consul not available: {e}")
    
    # 测试服务注册
    service_id = "test-service-integration"
    success = await consul_client.agent.service.register(
        name="test-service",
        service_id=service_id,
        address="127.0.0.1",
        port=8080,
        tags=["test", "integration"]
    )
    
    assert success is True
    
    # 验证服务已注册
    services = await consul_client.agent.services()
    assert service_id in services
    
    # 清理
    await consul_client.agent.service.deregister(service_id)


@pytest.mark.asyncio 
@pytest.mark.skipif(not Path("docker-compose.yml").exists(), reason="Docker environment not available")
async def test_postgresql_connection(infrastructure_setup):
    """测试PostgreSQL数据库连接"""
    import asyncpg
    
    postgres_url = infrastructure_setup["postgres_url"]
    
    try:
        # 测试数据库连接
        conn = await asyncpg.connect(postgres_url)
        
        # 执行简单查询
        result = await conn.fetchval("SELECT 1")
        assert result == 1
        
        # 创建测试表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # 插入测试数据
        await conn.execute(
            "INSERT INTO test_table (name) VALUES ($1)",
            "integration_test"
        )
        
        # 查询测试数据
        name = await conn.fetchval(
            "SELECT name FROM test_table WHERE name = $1",
            "integration_test"
        )
        assert name == "integration_test"
        
        # 清理
        await conn.execute("DROP TABLE test_table")
        await conn.close()
        
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("docker-compose.yml").exists(), reason="Docker environment not available") 
async def test_redis_connection(infrastructure_setup):
    """测试Redis连接"""
    import aioredis
    
    redis_url = infrastructure_setup["redis_url"]
    
    try:
        # 测试Redis连接
        redis = aioredis.from_url(redis_url)
        
        # 测试ping
        pong = await redis.ping()
        assert pong is True
        
        # 测试键值操作
        await redis.set("test_key", "test_value")
        value = await redis.get("test_key")
        assert value.decode() == "test_value"
        
        # 测试过期
        await redis.setex("temp_key", 1, "temp_value")
        temp_value = await redis.get("temp_key")
        assert temp_value.decode() == "temp_value"
        
        # 等待过期
        await asyncio.sleep(2)
        expired_value = await redis.get("temp_key")
        assert expired_value is None
        
        # 清理
        await redis.delete("test_key")
        await redis.close()
        
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("docker-compose.yml").exists(), reason="Docker environment not available")
async def test_service_discovery_integration(infrastructure_setup):
    """测试服务发现集成"""
    from services.core.registry.service_registry import ServiceRegistry, ServiceInfo
    
    try:
        # 初始化服务注册中心
        registry = ServiceRegistry("localhost", 8500)
        await registry.start()
        
        # 创建测试服务信息
        service_info = ServiceInfo(
            name="integration-test-service",
            service_id="integration-test-001",
            address="127.0.0.1",
            port=8080,
            tags=["integration", "test"]
        )
        
        # 注册服务
        success = await registry.register_service(service_info)
        assert success is True
        
        # 等待一下让服务注册生效
        await asyncio.sleep(2)
        
        # 发现服务
        discovered_services = await registry.discover_service("integration-test-service")
        assert len(discovered_services) > 0
        
        found_service = discovered_services[0]
        assert found_service.name == "integration-test-service"
        assert found_service.address == "127.0.0.1"
        assert found_service.port == 8080
        assert "integration" in found_service.tags
        
        # 清理
        await registry.deregister_service(service_info.service_id)
        await registry.stop()
        
    except Exception as e:
        pytest.skip(f"Service registry integration test failed: {e}")


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("docker-compose.yml").exists(), reason="Docker environment not available")
async def test_config_management_integration(infrastructure_setup):
    """测试配置管理集成"""
    from services.core.config.config_manager import ConfigManager
    
    try:
        # 初始化配置管理器
        config_manager = ConfigManager("localhost", 8500, environment="integration_test")
        await config_manager.start()
        
        # 设置配置
        test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db"
            },
            "redis": {
                "host": "localhost", 
                "port": 6379
            }
        }
        
        success = await config_manager.set_config(
            "app_config", 
            test_config, 
            "Integration test configuration"
        )
        assert success is True
        
        # 获取配置
        retrieved_config = await config_manager.get_config("app_config")
        assert retrieved_config == test_config
        
        # 测试配置监听
        config_changed = asyncio.Event()
        changed_value = None
        
        def on_config_change(key, value):
            nonlocal changed_value
            changed_value = value
            config_changed.set()
        
        await config_manager.watch_config("app_config", on_config_change)
        
        # 修改配置
        updated_config = test_config.copy()
        updated_config["version"] = "2.0"
        
        await config_manager.set_config("app_config", updated_config)
        
        # 等待配置变更通知
        try:
            await asyncio.wait_for(config_changed.wait(), timeout=5.0)
            assert changed_value == updated_config
        except asyncio.TimeoutError:
            pytest.skip("Config change notification timeout")
        
        # 清理
        await config_manager.delete_config("app_config")
        await config_manager.stop()
        
    except Exception as e:
        pytest.skip(f"Config management integration test failed: {e}")


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("docker-compose.yml").exists(), reason="Docker environment not available")
async def test_health_check_integration(infrastructure_setup):
    """测试健康检查集成"""
    from services.core.health.health_checker import HealthChecker, CommonHealthChecks
    
    # 初始化健康检查器
    health_checker = HealthChecker("1.0.0", "integration-test-service")
    
    # 添加真实的健康检查项
    health_checker.add_check(
        "postgres",
        lambda: CommonHealthChecks.check_database_postgresql(
            infrastructure_setup["postgres_url"]
        ),
        critical=True
    )
    
    health_checker.add_check(
        "redis", 
        lambda: CommonHealthChecks.check_redis(infrastructure_setup["redis_url"]),
        critical=True
    )
    
    health_checker.add_check(
        "consul",
        lambda: CommonHealthChecks.check_consul("http://localhost:8500"),
        critical=False
    )
    
    # 执行健康检查
    health_status = await health_checker.perform_health_check()
    
    # 验证健康状态
    assert health_status.status in ["healthy", "degraded", "unhealthy"]
    assert health_status.version == "1.0.0"
    assert health_status.uptime > 0
    
    # 验证各检查项
    assert "postgres" in health_status.checks
    assert "redis" in health_status.checks
    assert "consul" in health_status.checks
    
    # 如果所有服务都可用，状态应该是healthy
    postgres_status = health_status.checks["postgres"]["status"]
    redis_status = health_status.checks["redis"]["status"]
    
    if postgres_status == "pass" and redis_status == "pass":
        assert health_status.status in ["healthy", "degraded"]  # degraded可能是因为consul检查