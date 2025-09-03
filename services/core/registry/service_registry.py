"""
服务注册与发现实现
基于Consul的微服务注册中心

功能：
- 服务注册与注销
- 服务发现与负载均衡  
- 健康检查与监控
- 故障转移支持
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import consul.aio
import logging
import json
import random


@dataclass
class ServiceInfo:
    """服务信息数据类"""
    name: str
    service_id: str
    address: str
    port: int
    health_check: str = "/health"
    tags: List[str] = None
    metadata: Dict = None
    status: str = "unknown"
    last_check: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class ServiceRegistry:
    """服务注册中心 - 实现服务发现和健康检查"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        """
        初始化服务注册中心
        
        Args:
            consul_host: Consul服务器地址
            consul_port: Consul服务器端口
        """
        self.consul = consul.aio.Consul(host=consul_host, port=consul_port)
        self.services: Dict[str, ServiceInfo] = {}
        self.logger = logging.getLogger(__name__)
        self._health_check_interval = 30  # 健康检查间隔(秒)
        self._health_check_task = None
        self._running = False
    
    async def start(self) -> None:
        """启动服务注册中心"""
        try:
            # 测试Consul连接
            await self.consul.agent.self()
            
            self._running = True
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            self.logger.info("Service registry started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start service registry: {e}")
            raise
    
    async def stop(self) -> None:
        """停止服务注册中心"""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        # 注销所有已注册的服务
        for service_info in self.services.values():
            await self._deregister_service(service_info.service_id)
            
        self.logger.info("Service registry stopped")
    
    async def register_service(self, service_info: ServiceInfo) -> bool:
        """
        注册服务到Consul和本地缓存
        
        Args:
            service_info: 服务信息
            
        Returns:
            bool: 注册是否成功
        """
        try:
            # 构建健康检查配置
            health_check = consul.Check.http(
                url=f"http://{service_info.address}:{service_info.port}{service_info.health_check}",
                interval="10s",
                timeout="5s",
                deregister="30s"
            )
            
            # 注册到Consul
            success = await self.consul.agent.service.register(
                name=service_info.name,
                service_id=service_info.service_id,
                address=service_info.address,
                port=service_info.port,
                tags=service_info.tags,
                meta=service_info.metadata,
                check=health_check
            )
            
            if success:
                # 更新本地缓存
                service_info.status = "registered"
                service_info.last_check = datetime.utcnow()
                self.services[service_info.service_id] = service_info
                
                self.logger.info(
                    f"Service registered: {service_info.name}[{service_info.service_id}] "
                    f"at {service_info.address}:{service_info.port}"
                )
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to register service {service_info.service_id}: {e}")
            return False
    
    async def discover_service(self, 
                             service_name: str, 
                             healthy_only: bool = True, 
                             tags: List[str] = None) -> List[ServiceInfo]:
        """
        发现服务实例
        
        Args:
            service_name: 服务名称
            healthy_only: 是否只返回健康的服务
            tags: 标签过滤
            
        Returns:
            List[ServiceInfo]: 服务实例列表
        """
        try:
            # 从Consul获取服务实例
            if healthy_only:
                _, services = await self.consul.health.service(service_name, passing=True)
            else:
                _, services = await self.consul.health.service(service_name)
            
            discovered_services = []
            
            for service_data in services:
                service = service_data['Service']
                checks = service_data.get('Checks', [])
                
                # 检查标签过滤
                if tags and not all(tag in service.get('Tags', []) for tag in tags):
                    continue
                
                # 确定服务状态
                status = "healthy"
                for check in checks:
                    if check['Status'] != 'passing':
                        status = "unhealthy" if check['Status'] == 'warning' else "critical"
                        break
                
                service_info = ServiceInfo(
                    name=service['Service'],
                    service_id=service['ID'],
                    address=service['Address'],
                    port=service['Port'],
                    tags=service.get('Tags', []),
                    metadata=service.get('Meta', {}),
                    status=status,
                    last_check=datetime.utcnow()
                )
                
                discovered_services.append(service_info)
            
            self.logger.debug(f"Discovered {len(discovered_services)} instances for service {service_name}")
            return discovered_services
            
        except Exception as e:
            self.logger.error(f"Failed to discover service {service_name}: {e}")
            return []
    
    async def deregister_service(self, service_id: str) -> bool:
        """
        注销服务
        
        Args:
            service_id: 服务ID
            
        Returns:
            bool: 注销是否成功
        """
        try:
            success = await self._deregister_service(service_id)
            if success and service_id in self.services:
                del self.services[service_id]
                self.logger.info(f"Service deregistered: {service_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_id}: {e}")
            return False
    
    async def _deregister_service(self, service_id: str) -> bool:
        """内部注销服务方法"""
        try:
            success = await self.consul.agent.service.deregister(service_id)
            return success
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_id}: {e}")
            return False
    
    async def _periodic_health_check(self) -> None:
        """定期健康检查任务"""
        while self._running:
            try:
                # 检查本地缓存中的服务状态
                for service_id, service_info in self.services.items():
                    try:
                        # 从Consul获取最新的健康状态
                        _, checks = await self.consul.health.service(
                            service_info.name, 
                            passing=False
                        )
                        
                        # 更新服务状态
                        for check_data in checks:
                            if check_data['Service']['ID'] == service_id:
                                checks_status = check_data.get('Checks', [])
                                status = "healthy"
                                for check in checks_status:
                                    if check['Status'] != 'passing':
                                        status = "unhealthy" if check['Status'] == 'warning' else "critical"
                                        break
                                
                                service_info.status = status
                                service_info.last_check = datetime.utcnow()
                                break
                        
                    except Exception as e:
                        self.logger.warning(f"Health check failed for service {service_id}: {e}")
                
                await asyncio.sleep(self._health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # 错误后等待5秒重试


class LoadBalancer:
    """负载均衡器 - 实现多种负载均衡算法"""
    
    def __init__(self, strategy: str = "round_robin"):
        """
        初始化负载均衡器
        
        Args:
            strategy: 负载均衡策略 (round_robin, random, least_connections)
        """
        self.strategy = strategy
        self._round_robin_counters: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
    
    def select_service(self, services: List[ServiceInfo], service_name: str = None) -> Optional[ServiceInfo]:
        """
        根据负载均衡策略选择服务实例
        
        Args:
            services: 可用服务列表
            service_name: 服务名称 (用于round_robin计数器)
            
        Returns:
            Optional[ServiceInfo]: 选中的服务实例
        """
        if not services:
            return None
        
        # 过滤健康的服务
        healthy_services = [s for s in services if s.status == "healthy"]
        if not healthy_services:
            # 如果没有健康的服务，使用所有服务（应急方案）
            healthy_services = services
            self.logger.warning(f"No healthy services available for {service_name}, using all services")
        
        if self.strategy == "round_robin":
            return self._round_robin_select(healthy_services, service_name or "default")
        elif self.strategy == "random":
            return self._random_select(healthy_services)
        elif self.strategy == "least_connections":
            return self._least_connections_select(healthy_services)
        else:
            # 默认返回第一个服务
            return healthy_services[0]
    
    def _round_robin_select(self, services: List[ServiceInfo], service_name: str) -> ServiceInfo:
        """轮询选择算法"""
        if service_name not in self._round_robin_counters:
            self._round_robin_counters[service_name] = 0
        
        index = self._round_robin_counters[service_name] % len(services)
        self._round_robin_counters[service_name] += 1
        
        return services[index]
    
    def _random_select(self, services: List[ServiceInfo]) -> ServiceInfo:
        """随机选择算法"""
        return random.choice(services)
    
    def _least_connections_select(self, services: List[ServiceInfo]) -> ServiceInfo:
        """最少连接选择算法 (简化实现)"""
        # 实际实现中应该维护每个服务的连接数统计
        # 这里简化为返回第一个服务
        return services[0]


# 服务注册装饰器
class ServiceRegistration:
    """服务注册装饰器和上下文管理器"""
    
    def __init__(self, registry: ServiceRegistry, service_info: ServiceInfo):
        """
        初始化服务注册管理器
        
        Args:
            registry: 服务注册中心实例
            service_info: 服务信息
        """
        self.registry = registry
        self.service_info = service_info
        self.registered = False
    
    async def __aenter__(self):
        """异步上下文管理器进入"""
        success = await self.registry.register_service(self.service_info)
        if success:
            self.registered = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.registered:
            await self.registry.deregister_service(self.service_info.service_id)