"""
分布式配置管理实现
基于Consul KV存储的配置中心

功能：
- 分布式配置存储与读取
- 配置变更实时监听
- 配置缓存与同步
- 多环境配置支持
"""

import json
import asyncio
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import consul.aio
import logging


@dataclass
class ConfigItem:
    """配置项数据类"""
    key: str
    value: Any
    version: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    description: str = ""
    environment: str = "default"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class ConfigManager:
    """分布式配置管理器 - 基于Consul KV存储"""
    
    def __init__(self, 
                 consul_host: str = "localhost", 
                 consul_port: int = 8500,
                 prefix: str = "config/",
                 environment: str = "development"):
        """
        初始化配置管理器
        
        Args:
            consul_host: Consul服务器地址
            consul_port: Consul服务器端口
            prefix: 配置键前缀
            environment: 环境标识
        """
        self.consul = consul.aio.Consul(host=consul_host, port=consul_port)
        self.prefix = prefix
        self.environment = environment
        self.config_cache: Dict[str, ConfigItem] = {}
        self.watchers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
        self._watch_tasks: Dict[str, asyncio.Task] = {}
        self._cache_ttl = 300  # 缓存TTL(秒)
        self._running = False
    
    async def start(self) -> None:
        """启动配置管理器"""
        try:
            # 测试Consul连接
            await self.consul.agent.self()
            
            # 加载所有配置
            await self._load_all_configs()
            
            self._running = True
            self.logger.info(f"Config manager started for environment: {self.environment}")
            
        except Exception as e:
            self.logger.error(f"Failed to start config manager: {e}")
            raise
    
    async def stop(self) -> None:
        """停止配置管理器"""
        self._running = False
        
        # 取消所有监听任务
        for task in self._watch_tasks.values():
            task.cancel()
            
        for task in self._watch_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        self._watch_tasks.clear()
        self.logger.info("Config manager stopped")
    
    async def get_config(self, 
                        key: str, 
                        default: Any = None, 
                        use_cache: bool = True) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            use_cache: 是否使用缓存
            
        Returns:
            Any: 配置值
        """
        try:
            full_key = self._get_full_key(key)
            
            # 检查缓存
            if use_cache and key in self.config_cache:
                config_item = self.config_cache[key]
                # 检查缓存是否过期
                if (datetime.utcnow() - config_item.updated_at).seconds < self._cache_ttl:
                    return config_item.value
            
            # 从Consul获取
            _, data = await self.consul.kv.get(full_key)
            if data:
                config_value = json.loads(data['Value'].decode())
                
                # 更新缓存
                config_item = ConfigItem(
                    key=key,
                    value=config_value,
                    version=data.get('ModifyIndex', 0),
                    updated_at=datetime.utcnow(),
                    environment=self.environment
                )
                self.config_cache[key] = config_item
                
                return config_value
            
            return default
            
        except Exception as e:
            self.logger.error(f"Failed to get config {key}: {e}")
            return default
    
    async def set_config(self, 
                        key: str, 
                        value: Any, 
                        description: str = "") -> bool:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            description: 配置描述
            
        Returns:
            bool: 设置是否成功
        """
        try:
            full_key = self._get_full_key(key)
            
            # 构建配置元数据
            config_metadata = {
                "value": value,
                "description": description,
                "environment": self.environment,
                "updated_at": datetime.utcnow().isoformat(),
                "updated_by": "config_manager"
            }
            
            # 序列化配置值
            config_data = json.dumps(config_metadata, ensure_ascii=False, indent=2)
            
            # 保存到Consul
            success = await self.consul.kv.put(full_key, config_data)
            
            if success:
                # 更新本地缓存
                config_item = ConfigItem(
                    key=key,
                    value=value,
                    description=description,
                    environment=self.environment,
                    updated_at=datetime.utcnow()
                )
                self.config_cache[key] = config_item
                
                self.logger.info(f"Config updated: {key}")
                
                # 触发变更回调
                await self._notify_config_change(key, value)
                
                return True
            
            return False
             
        except Exception as e:
            self.logger.error(f"Failed to set config {key}: {e}")
            return False
    
    async def delete_config(self, key: str) -> bool:
        """
        删除配置
        
        Args:
            key: 配置键
            
        Returns:
            bool: 删除是否成功
        """
        try:
            full_key = self._get_full_key(key)
            success = await self.consul.kv.delete(full_key)
            
            if success:
                # 从缓存中移除
                if key in self.config_cache:
                    del self.config_cache[key]
                
                self.logger.info(f"Config deleted: {key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete config {key}: {e}")
            return False
    
    async def list_configs(self, pattern: str = None) -> Dict[str, Any]:
        """
        列出配置
        
        Args:
            pattern: 匹配模式
            
        Returns:
            Dict[str, Any]: 配置字典
        """
        try:
            search_key = self.prefix
            if pattern:
                search_key = f"{self.prefix}{pattern}"
            
            _, configs = await self.consul.kv.get(search_key, recurse=True)
            
            result = {}
            if configs:
                for config_data in configs:
                    key = config_data['Key'].replace(self.prefix, '')
                    try:
                        config_metadata = json.loads(config_data['Value'].decode())
                        if isinstance(config_metadata, dict) and 'value' in config_metadata:
                            result[key] = config_metadata['value']
                        else:
                            result[key] = config_metadata
                    except json.JSONDecodeError:
                        # 兼容旧格式
                        result[key] = config_data['Value'].decode()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to list configs: {e}")
            return {}
    
    async def watch_config(self, 
                          key: str, 
                          callback: Callable[[str, Any], None]) -> None:
        """
        监听配置变化
        
        Args:
            key: 配置键
            callback: 变更回调函数
        """
        if key not in self.watchers:
            self.watchers[key] = []
        
        self.watchers[key].append(callback)
        
        # 启动监听任务
        if key not in self._watch_tasks:
            self._watch_tasks[key] = asyncio.create_task(
                self._watch_key_changes(key)
            )
        
        self.logger.info(f"Added watcher for config key: {key}")
    
    async def unwatch_config(self, key: str, callback: Callable = None) -> None:
        """
        取消配置监听
        
        Args:
            key: 配置键
            callback: 特定的回调函数，如果为None则移除所有回调
        """
        if key in self.watchers:
            if callback:
                if callback in self.watchers[key]:
                    self.watchers[key].remove(callback)
            else:
                self.watchers[key].clear()
            
            # 如果没有回调了，取消监听任务
            if not self.watchers[key] and key in self._watch_tasks:
                self._watch_tasks[key].cancel()
                try:
                    await self._watch_tasks[key]
                except asyncio.CancelledError:
                    pass
                del self._watch_tasks[key]
    
    async def _load_all_configs(self) -> None:
        """加载所有配置到缓存"""
        try:
            _, configs = await self.consul.kv.get(self.prefix, recurse=True)
            
            if configs:
                for config_data in configs:
                    key = config_data['Key'].replace(self.prefix, '')
                    
                    try:
                        config_metadata = json.loads(config_data['Value'].decode())
                        if isinstance(config_metadata, dict) and 'value' in config_metadata:
                            value = config_metadata['value']
                            description = config_metadata.get('description', '')
                        else:
                            value = config_metadata
                            description = ''
                    except json.JSONDecodeError:
                        # 兼容旧格式
                        value = config_data['Value'].decode()
                        description = ''
                    
                    config_item = ConfigItem(
                        key=key,
                        value=value,
                        version=config_data.get('ModifyIndex', 0),
                        description=description,
                        environment=self.environment,
                        updated_at=datetime.utcnow()
                    )
                    self.config_cache[key] = config_item
                
                self.logger.info(f"Loaded {len(configs)} configurations")
                
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
    
    async def _watch_key_changes(self, key: str) -> None:
        """监听单个配置键的变化"""
        full_key = self._get_full_key(key)
        index = None
        
        while self._running:
            try:
                # 使用阻塞查询监听变化
                index, data = await self.consul.kv.get(
                    full_key, 
                    index=index, 
                    wait='30s'
                )
                
                if data:
                    try:
                        config_metadata = json.loads(data['Value'].decode())
                        if isinstance(config_metadata, dict) and 'value' in config_metadata:
                            new_value = config_metadata['value']
                        else:
                            new_value = config_metadata
                    except json.JSONDecodeError:
                        new_value = data['Value'].decode()
                    
                    # 更新缓存
                    if key in self.config_cache:
                        old_value = self.config_cache[key].value
                        self.config_cache[key].value = new_value
                        self.config_cache[key].updated_at = datetime.utcnow()
                        
                        # 如果值发生变化，触发回调
                        if old_value != new_value:
                            await self._notify_config_change(key, new_value)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Config watch error for key {key}: {e}")
                await asyncio.sleep(5)  # 错误后等待5秒重试
    
    async def _notify_config_change(self, key: str, value: Any) -> None:
        """通知配置变更"""
        if key in self.watchers:
            for callback in self.watchers[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, value)
                    else:
                        callback(key, value)
                except Exception as e:
                    self.logger.error(f"Config callback error for {key}: {e}")
    
    def _get_full_key(self, key: str) -> str:
        """获取完整的配置键"""
        return f"{self.prefix}{self.environment}/{key}"


# 服务配置类示例
class ServiceConfig:
    """服务配置类 - 演示如何使用ConfigManager"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化服务配置
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    async def get_database_config(self) -> Dict[str, Any]:
        """获取数据库配置"""
        return await self.config_manager.get_config(
            "database",
            default={
                "host": "localhost",
                "port": 5432,
                "database": "historical_text",
                "username": "postgres",
                "password": "password",
                "pool_size": 10,
                "max_overflow": 20,
                "echo": False
            }
        )
    
    async def get_redis_config(self) -> Dict[str, Any]:
        """获取Redis配置"""
        return await self.config_manager.get_config(
            "redis",
            default={
                "host": "localhost",
                "port": 6379,
                "database": 0,
                "password": None,
                "max_connections": 100,
                "retry_on_timeout": True,
                "health_check_interval": 30
            }
        )
    
    async def get_api_gateway_config(self) -> Dict[str, Any]:
        """获取API网关配置"""
        return await self.config_manager.get_config(
            "api_gateway",
            default={
                "host": "localhost",
                "port": 8000,
                "admin_port": 8001,
                "timeout": {
                    "connect": 5000,
                    "send": 60000,
                    "read": 60000
                },
                "rate_limit": {
                    "requests_per_minute": 100,
                    "requests_per_hour": 1000,
                    "requests_per_day": 10000
                },
                "cors": {
                    "allowed_origins": ["http://localhost:3000"],
                    "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
                    "allowed_headers": ["*"],
                    "max_age": 3600
                }
            }
        )
    
    async def get_consul_config(self) -> Dict[str, Any]:
        """获取Consul配置"""
        return await self.config_manager.get_config(
            "consul",
            default={
                "host": "localhost",
                "port": 8500,
                "datacenter": "dc1",
                "token": None,
                "scheme": "http",
                "verify": True,
                "timeout": 10
            }
        )
    
    async def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return await self.config_manager.get_config(
            "logging",
            default={
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": ["console", "file"],
                "file": {
                    "filename": "logs/application.log",
                    "max_bytes": 10485760,  # 10MB
                    "backup_count": 5
                },
                "elasticsearch": {
                    "enabled": False,
                    "host": "localhost",
                    "port": 9200,
                    "index": "historical-text-logs"
                }
            }
        )