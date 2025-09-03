"""
代理管理器
负责代理IP的获取、验证、轮换和管理，提供反封禁能力
"""

import asyncio
import aiohttp
import random
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from urllib.parse import urlparse
from loguru import logger
import hashlib

from ..config.settings import get_settings
from ..database.database import get_database_manager


class ProxyStatus(str, Enum):
    """代理状态枚举"""
    ACTIVE = "active"           # 活跃可用
    INACTIVE = "inactive"       # 不可用
    TESTING = "testing"         # 测试中
    BANNED = "banned"           # 被封禁
    TIMEOUT = "timeout"         # 超时
    ERROR = "error"             # 错误


class ProxyQuality(str, Enum):
    """代理质量等级"""
    UNKNOWN = "unknown"
    LOW = "low"                # 免费代理
    MEDIUM = "medium"          # 付费低级代理
    HIGH = "high"              # 付费高级代理
    PREMIUM = "premium"        # 专业代理


@dataclass
class ProxyInfo:
    """代理信息"""
    proxy_id: str
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: str = "http"              # http, https, socks4, socks5
    status: ProxyStatus = ProxyStatus.INACTIVE
    quality: ProxyQuality = ProxyQuality.UNKNOWN
    country: Optional[str] = None
    region: Optional[str] = None
    provider: Optional[str] = None
    
    # 性能统计
    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 限制设置
    max_requests_per_hour: int = 100
    current_hour_requests: int = 0
    hour_reset_time: datetime = field(default_factory=datetime.now)
    
    @property
    def url(self) -> str:
        """代理URL"""
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.success_count / self.total_requests) * 100
    
    @property
    def is_available(self) -> bool:
        """是否可用"""
        if self.status != ProxyStatus.ACTIVE:
            return False
        
        # 检查是否达到小时请求限制
        if datetime.now() - self.hour_reset_time > timedelta(hours=1):
            self.current_hour_requests = 0
            self.hour_reset_time = datetime.now()
        
        return self.current_hour_requests < self.max_requests_per_hour
    
    def update_stats(self, success: bool, response_time: float):
        """更新统计信息"""
        self.total_requests += 1
        self.current_hour_requests += 1
        
        if success:
            self.success_count += 1
            self.last_success_time = datetime.now()
            self.status = ProxyStatus.ACTIVE
            
            # 更新平均响应时间
            if self.avg_response_time == 0:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = (self.avg_response_time + response_time) / 2
        else:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # 失败率过高时标记为不可用
            if self.success_rate < 50 and self.total_requests > 10:
                self.status = ProxyStatus.INACTIVE
        
        self.updated_at = datetime.now()


class ProxyProvider:
    """代理供应商基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', False)
    
    async def fetch_proxies(self) -> List[ProxyInfo]:
        """获取代理列表 - 子类需要实现"""
        raise NotImplementedError("子类必须实现fetch_proxies方法")


class FreeProxyProvider(ProxyProvider):
    """免费代理供应商"""
    
    async def fetch_proxies(self) -> List[ProxyInfo]:
        """从免费代理源获取代理列表"""
        proxies = []
        urls = self.config.get('urls', [])
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for url in urls:
                try:
                    logger.info(f"从 {url} 获取免费代理...")
                    proxies_from_url = await self._fetch_from_url(session, url)
                    proxies.extend(proxies_from_url)
                    
                    # 防止请求过于频繁
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"从 {url} 获取代理失败: {e}")
        
        logger.info(f"从免费源获取到 {len(proxies)} 个代理")
        return proxies
    
    async def _fetch_from_url(self, session: aiohttp.ClientSession, url: str) -> List[ProxyInfo]:
        """从指定URL获取代理"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                text = await response.text()
                return self._parse_proxy_text(text)
                
        except Exception as e:
            logger.error(f"请求代理URL失败 {url}: {e}")
            return []
    
    def _parse_proxy_text(self, text: str) -> List[ProxyInfo]:
        """解析代理文本"""
        proxies = []
        
        # 匹配IP:Port格式的代理
        proxy_pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5})')
        matches = proxy_pattern.findall(text)
        
        for match in matches:
            host, port = match
            try:
                proxy_id = hashlib.md5(f"{host}:{port}".encode()).hexdigest()
                proxy = ProxyInfo(
                    proxy_id=proxy_id,
                    host=host,
                    port=int(port),
                    protocol="http",
                    status=ProxyStatus.INACTIVE,
                    quality=ProxyQuality.LOW,
                    provider=self.name
                )
                proxies.append(proxy)
            except ValueError:
                continue
        
        return proxies


class PremiumProxyProvider(ProxyProvider):
    """付费代理供应商"""
    
    async def fetch_proxies(self) -> List[ProxyInfo]:
        """从付费代理API获取代理列表"""
        if not self.config.get('api_url') or not self.config.get('api_key'):
            logger.warning("付费代理配置不完整")
            return []
        
        proxies = []
        api_url = self.config['api_url']
        api_key = self.config['api_key']
        
        try:
            headers = {'Authorization': f'Bearer {api_key}'}
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        proxies = self._parse_premium_response(data)
        
        except Exception as e:
            logger.error(f"获取付费代理失败: {e}")
        
        return proxies
    
    def _parse_premium_response(self, data: Dict) -> List[ProxyInfo]:
        """解析付费代理API响应"""
        proxies = []
        
        for item in data.get('proxies', []):
            try:
                proxy_id = hashlib.md5(f"{item['host']}:{item['port']}".encode()).hexdigest()
                proxy = ProxyInfo(
                    proxy_id=proxy_id,
                    host=item['host'],
                    port=int(item['port']),
                    username=item.get('username'),
                    password=item.get('password'),
                    protocol=item.get('protocol', 'http'),
                    status=ProxyStatus.ACTIVE,
                    quality=ProxyQuality.HIGH,
                    country=item.get('country'),
                    region=item.get('region'),
                    provider=self.name,
                    max_requests_per_hour=item.get('max_requests_per_hour', 500)
                )
                proxies.append(proxy)
            except (KeyError, ValueError) as e:
                logger.warning(f"解析付费代理数据失败: {e}")
        
        return proxies


class ProxyManager:
    """代理管理器 - 核心代理管理类"""
    
    def __init__(self):
        self.proxies: Dict[str, ProxyInfo] = {}
        self.active_proxies: List[str] = []
        self.banned_proxies: Set[str] = set()
        self.providers: Dict[str, ProxyProvider] = {}
        self.settings = get_settings()
        self.current_proxy_index = 0
        self.last_rotation_time = datetime.now()
        
        # 测试目标URL列表
        self.test_urls = [
            "http://httpbin.org/ip",
            "https://api.ipify.org?format=json",
            "http://ip-api.com/json"
        ]
        
        self._initialize_providers()
    
    def _initialize_providers(self):
        """初始化代理供应商"""
        proxy_config = self.settings.proxy.proxy_providers
        
        for name, config in proxy_config.items():
            if not config.get('enabled', False):
                continue
            
            if name == 'free_proxy':
                self.providers[name] = FreeProxyProvider(name, config)
            elif name == 'premium_proxy':
                self.providers[name] = PremiumProxyProvider(name, config)
            else:
                logger.warning(f"未知的代理供应商类型: {name}")
        
        logger.info(f"初始化了 {len(self.providers)} 个代理供应商")
    
    async def initialize(self):
        """初始化代理管理器"""
        logger.info("初始化代理管理器...")
        
        # 从数据库加载已有代理
        await self._load_proxies_from_db()
        
        # 如果没有可用代理，获取新代理
        if not self.active_proxies:
            await self.refresh_proxies()
        
        # 启动后台任务
        asyncio.create_task(self._background_tasks())
        
        logger.info(f"代理管理器初始化完成，可用代理: {len(self.active_proxies)}")
    
    async def refresh_proxies(self):
        """刷新代理列表"""
        logger.info("开始刷新代理列表...")
        
        new_proxies = []
        for provider in self.providers.values():
            try:
                proxies = await provider.fetch_proxies()
                new_proxies.extend(proxies)
            except Exception as e:
                logger.error(f"从供应商 {provider.name} 获取代理失败: {e}")
        
        # 添加新代理到管理器
        added_count = 0
        for proxy in new_proxies:
            if proxy.proxy_id not in self.proxies:
                self.proxies[proxy.proxy_id] = proxy
                added_count += 1
        
        logger.info(f"新增 {added_count} 个代理，总计 {len(self.proxies)} 个代理")
        
        # 测试新代理
        await self._test_new_proxies(new_proxies)
        
        # 保存到数据库
        await self._save_proxies_to_db()
    
    async def _test_new_proxies(self, proxies: List[ProxyInfo]):
        """测试新代理的可用性"""
        logger.info(f"开始测试 {len(proxies)} 个新代理...")
        
        # 并发测试，但限制并发数量
        semaphore = asyncio.Semaphore(20)
        tasks = []
        
        for proxy in proxies:
            task = asyncio.create_task(self._test_proxy_with_semaphore(semaphore, proxy))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计测试结果
        active_count = 0
        for i, result in enumerate(results):
            if isinstance(result, bool) and result:
                self.active_proxies.append(proxies[i].proxy_id)
                active_count += 1
        
        logger.info(f"代理测试完成，激活 {active_count} 个可用代理")
    
    async def _test_proxy_with_semaphore(self, semaphore: asyncio.Semaphore, proxy: ProxyInfo) -> bool:
        """使用信号量限制并发的代理测试"""
        async with semaphore:
            return await self.test_proxy(proxy)
    
    async def test_proxy(self, proxy: ProxyInfo) -> bool:
        """测试单个代理的可用性"""
        proxy.status = ProxyStatus.TESTING
        
        test_url = random.choice(self.test_urls)
        start_time = time.time()
        
        try:
            connector = aiohttp.TCPConnector()
            timeout = aiohttp.ClientTimeout(total=self.settings.proxy.proxy_test_timeout)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(
                    test_url,
                    proxy=proxy.url,
                    headers={'User-Agent': random.choice(self.settings.crawler.user_agents)}
                ) as response:
                    
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        proxy.update_stats(True, response_time)
                        logger.debug(f"代理测试成功: {proxy.host}:{proxy.port} ({response_time:.2f}s)")
                        return True
                    else:
                        proxy.update_stats(False, response_time)
                        logger.debug(f"代理测试失败: {proxy.host}:{proxy.port} (状态码: {response.status})")
                        return False
        
        except asyncio.TimeoutError:
            proxy.status = ProxyStatus.TIMEOUT
            proxy.update_stats(False, self.settings.proxy.proxy_test_timeout)
            logger.debug(f"代理测试超时: {proxy.host}:{proxy.port}")
            return False
        
        except Exception as e:
            proxy.status = ProxyStatus.ERROR
            proxy.update_stats(False, time.time() - start_time)
            logger.debug(f"代理测试异常: {proxy.host}:{proxy.port}, 错误: {e}")
            return False
    
    def get_proxy(self) -> Optional[ProxyInfo]:
        """获取一个可用的代理"""
        available_proxies = [
            proxy_id for proxy_id in self.active_proxies
            if proxy_id in self.proxies and self.proxies[proxy_id].is_available
        ]
        
        if not available_proxies:
            logger.warning("没有可用的代理")
            return None
        
        # 根据质量和成功率选择代理
        proxy_scores = []
        for proxy_id in available_proxies:
            proxy = self.proxies[proxy_id]
            
            # 计算代理得分（质量 + 成功率 + 响应时间）
            quality_score = {
                ProxyQuality.PREMIUM: 100,
                ProxyQuality.HIGH: 80,
                ProxyQuality.MEDIUM: 60,
                ProxyQuality.LOW: 40,
                ProxyQuality.UNKNOWN: 20
            }.get(proxy.quality, 20)
            
            success_score = min(proxy.success_rate, 100)
            
            # 响应时间越小得分越高
            speed_score = max(0, 100 - proxy.avg_response_time * 10)
            
            total_score = quality_score * 0.4 + success_score * 0.4 + speed_score * 0.2
            proxy_scores.append((proxy_id, total_score))
        
        # 选择得分最高的代理（加入一些随机性）
        proxy_scores.sort(key=lambda x: x[1], reverse=True)
        top_proxies = proxy_scores[:min(5, len(proxy_scores))]  # 取前5个
        
        selected_proxy_id = random.choice(top_proxies)[0]
        return self.proxies[selected_proxy_id]
    
    def rotate_proxy(self) -> Optional[ProxyInfo]:
        """轮换代理"""
        current_time = datetime.now()
        
        # 检查是否需要轮换
        if (current_time - self.last_rotation_time).seconds < self.settings.proxy.proxy_check_interval:
            return self.get_proxy()
        
        self.last_rotation_time = current_time
        logger.debug("执行代理轮换...")
        
        return self.get_proxy()
    
    def mark_proxy_failed(self, proxy_id: str, reason: str = ""):
        """标记代理失败"""
        if proxy_id not in self.proxies:
            return
        
        proxy = self.proxies[proxy_id]
        proxy.failure_count += 1
        proxy.last_failure_time = datetime.now()
        
        # 失败次数过多时禁用代理
        if proxy.failure_count >= self.settings.proxy.proxy_failure_threshold:
            proxy.status = ProxyStatus.BANNED
            self.banned_proxies.add(proxy_id)
            
            if proxy_id in self.active_proxies:
                self.active_proxies.remove(proxy_id)
            
            logger.warning(f"代理已被禁用: {proxy.host}:{proxy.port}, 原因: {reason}")
    
    def get_proxy_statistics(self) -> Dict[str, Any]:
        """获取代理统计信息"""
        total_proxies = len(self.proxies)
        active_proxies = len(self.active_proxies)
        banned_proxies = len(self.banned_proxies)
        
        # 按质量分组统计
        quality_stats = {}
        for quality in ProxyQuality:
            count = len([p for p in self.proxies.values() if p.quality == quality])
            quality_stats[quality.value] = count
        
        # 计算平均成功率
        total_success_rate = 0
        if self.proxies:
            total_success_rate = sum(p.success_rate for p in self.proxies.values()) / len(self.proxies)
        
        return {
            "total_proxies": total_proxies,
            "active_proxies": active_proxies,
            "banned_proxies": banned_proxies,
            "quality_distribution": quality_stats,
            "average_success_rate": round(total_success_rate, 2),
            "providers": list(self.providers.keys())
        }
    
    async def _background_tasks(self):
        """后台任务：定期检查和刷新代理"""
        while True:
            try:
                # 每10分钟检查一次代理状态
                await asyncio.sleep(600)
                
                logger.info("执行代理后台维护任务...")
                
                # 重新测试失败的代理
                await self._retest_failed_proxies()
                
                # 每小时刷新代理列表
                current_hour = datetime.now().hour
                if current_hour % 1 == 0:  # 每小时执行一次
                    await self.refresh_proxies()
            
            except Exception as e:
                logger.error(f"代理后台任务异常: {e}")
    
    async def _retest_failed_proxies(self):
        """重新测试失败的代理"""
        failed_proxies = [
            proxy for proxy in self.proxies.values()
            if proxy.status in [ProxyStatus.INACTIVE, ProxyStatus.ERROR, ProxyStatus.TIMEOUT]
            and proxy.proxy_id not in self.banned_proxies
        ]
        
        if not failed_proxies:
            return
        
        logger.info(f"重新测试 {len(failed_proxies)} 个失败的代理...")
        
        for proxy in failed_proxies[:10]:  # 限制重测数量
            success = await self.test_proxy(proxy)
            if success and proxy.proxy_id not in self.active_proxies:
                self.active_proxies.append(proxy.proxy_id)
    
    async def _load_proxies_from_db(self):
        """从数据库加载代理"""
        try:
            db_manager = await get_database_manager()
            collection = await db_manager.get_mongodb_collection("proxies")
            
            cursor = collection.find({})
            async for doc in cursor:
                proxy = ProxyInfo(
                    proxy_id=doc['proxy_id'],
                    host=doc['host'],
                    port=doc['port'],
                    username=doc.get('username'),
                    password=doc.get('password'),
                    protocol=doc.get('protocol', 'http'),
                    status=ProxyStatus(doc.get('status', ProxyStatus.INACTIVE)),
                    quality=ProxyQuality(doc.get('quality', ProxyQuality.UNKNOWN)),
                    country=doc.get('country'),
                    region=doc.get('region'),
                    provider=doc.get('provider'),
                    success_count=doc.get('success_count', 0),
                    failure_count=doc.get('failure_count', 0),
                    total_requests=doc.get('total_requests', 0),
                    avg_response_time=doc.get('avg_response_time', 0.0),
                    created_at=doc.get('created_at', datetime.now()),
                    updated_at=doc.get('updated_at', datetime.now())
                )
                
                self.proxies[proxy.proxy_id] = proxy
                
                if proxy.status == ProxyStatus.ACTIVE:
                    self.active_proxies.append(proxy.proxy_id)
                elif proxy.status == ProxyStatus.BANNED:
                    self.banned_proxies.add(proxy.proxy_id)
            
            logger.info(f"从数据库加载 {len(self.proxies)} 个代理")
        
        except Exception as e:
            logger.error(f"从数据库加载代理失败: {e}")
    
    async def _save_proxies_to_db(self):
        """保存代理到数据库"""
        try:
            db_manager = await get_database_manager()
            collection = await db_manager.get_mongodb_collection("proxies")
            
            for proxy in self.proxies.values():
                proxy_dict = {
                    "proxy_id": proxy.proxy_id,
                    "host": proxy.host,
                    "port": proxy.port,
                    "username": proxy.username,
                    "password": proxy.password,
                    "protocol": proxy.protocol,
                    "status": proxy.status,
                    "quality": proxy.quality,
                    "country": proxy.country,
                    "region": proxy.region,
                    "provider": proxy.provider,
                    "success_count": proxy.success_count,
                    "failure_count": proxy.failure_count,
                    "total_requests": proxy.total_requests,
                    "avg_response_time": proxy.avg_response_time,
                    "last_success_time": proxy.last_success_time,
                    "last_failure_time": proxy.last_failure_time,
                    "created_at": proxy.created_at,
                    "updated_at": proxy.updated_at
                }
                
                await collection.replace_one(
                    {"proxy_id": proxy.proxy_id},
                    proxy_dict,
                    upsert=True
                )
        
        except Exception as e:
            logger.error(f"保存代理到数据库失败: {e}")


# 全局代理管理器实例
proxy_manager = ProxyManager()


async def get_proxy_manager() -> ProxyManager:
    """获取代理管理器实例"""
    return proxy_manager