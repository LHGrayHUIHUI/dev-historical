"""
爬虫管理器
负责管理和调度各平台爬虫，支持多平台并发爬取和任务调度
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
import aiohttp
from loguru import logger
import signal
import sys

from ..config.settings import get_settings
from ..database.database import get_database_manager
from ..models.content import ContentCreate, ContentSource, ContentStatus


class CrawlerStatus(str, Enum):
    """爬虫状态枚举"""
    IDLE = "idle"           # 空闲
    RUNNING = "running"     # 运行中
    PAUSED = "paused"       # 暂停
    STOPPED = "stopped"     # 已停止
    ERROR = "error"         # 错误状态
    FINISHED = "finished"   # 已完成


class TaskPriority(int, Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class CrawlerConfig:
    """爬虫配置"""
    platform: ContentSource
    keywords: List[str]
    max_pages: int = 10
    interval: float = 5.0           # 请求间隔(秒)
    timeout: int = 300              # 超时时间(秒)
    priority: TaskPriority = TaskPriority.NORMAL
    proxy_enabled: bool = True
    retry_attempts: int = 3
    custom_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}


@dataclass 
class CrawlerTask:
    """爬虫任务"""
    task_id: str
    config: CrawlerConfig
    status: CrawlerStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: float = 0.0           # 进度百分比 0-100
    total_items: int = 0            # 总计划抓取数量
    success_items: int = 0          # 成功抓取数量
    failed_items: int = 0           # 失败抓取数量
    error_message: Optional[str] = None
    current_page: int = 0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.success_items + self.failed_items
        if total == 0:
            return 0.0
        return (self.success_items / total) * 100
    
    @property
    def duration(self) -> Optional[timedelta]:
        """运行时长"""
        if self.started_at is None:
            return None
        end_time = self.finished_at or datetime.now()
        return end_time - self.started_at


class BaseCrawler:
    """爬虫基类 - 定义爬虫接口"""
    
    def __init__(self, task: CrawlerTask, session: aiohttp.ClientSession):
        self.task = task
        self.session = session
        self.is_running = False
        self.should_stop = False
        
    async def crawl(self) -> None:
        """爬取主逻辑 - 子类需要实现"""
        raise NotImplementedError("子类必须实现crawl方法")
    
    async def stop(self) -> None:
        """停止爬取"""
        self.should_stop = True
        logger.info(f"爬虫任务 {self.task.task_id} 收到停止信号")
    
    async def pause(self) -> None:
        """暂停爬取"""
        self.task.status = CrawlerStatus.PAUSED
        logger.info(f"爬虫任务 {self.task.task_id} 已暂停")
    
    async def resume(self) -> None:
        """恢复爬取"""
        if self.task.status == CrawlerStatus.PAUSED:
            self.task.status = CrawlerStatus.RUNNING
            logger.info(f"爬虫任务 {self.task.task_id} 已恢复")
    
    def update_progress(self, current: int, total: int):
        """更新进度"""
        self.task.progress = (current / total) * 100 if total > 0 else 0
        self.task.total_items = total


class ToutiaoCrawler(BaseCrawler):
    """今日头条爬虫实现"""
    
    async def crawl(self) -> None:
        """今日头条爬取逻辑"""
        self.task.status = CrawlerStatus.RUNNING
        self.task.started_at = datetime.now()
        
        try:
            keywords = self.task.config.keywords
            max_pages = self.task.config.max_pages
            
            logger.info(f"开始爬取今日头条，关键词: {keywords}, 最大页数: {max_pages}")
            
            for keyword in keywords:
                if self.should_stop:
                    break
                    
                for page in range(1, max_pages + 1):
                    if self.should_stop:
                        break
                    
                    # 等待暂停状态结束
                    while self.task.status == CrawlerStatus.PAUSED:
                        await asyncio.sleep(1)
                    
                    try:
                        # 模拟爬取逻辑（实际需要根据今日头条API实现）
                        await self._crawl_page(keyword, page)
                        
                        # 更新进度
                        current_progress = (page / max_pages) * 100
                        self.task.progress = current_progress
                        self.task.current_page = page
                        
                        # 请求间隔
                        await asyncio.sleep(self.task.config.interval)
                        
                    except Exception as e:
                        logger.error(f"爬取今日头条第{page}页失败: {e}")
                        self.task.failed_items += 1
            
            self.task.status = CrawlerStatus.FINISHED
            self.task.finished_at = datetime.now()
            logger.info(f"今日头条爬取完成，成功: {self.task.success_items}, 失败: {self.task.failed_items}")
            
        except Exception as e:
            self.task.status = CrawlerStatus.ERROR
            self.task.error_message = str(e)
            self.task.finished_at = datetime.now()
            logger.error(f"今日头条爬取异常: {e}")
    
    async def _crawl_page(self, keyword: str, page: int):
        """爬取单页数据"""
        # 这里是模拟实现，实际需要根据今日头条的接口来实现
        url = f"https://www.toutiao.com/api/search/content/"
        params = {
            "keyword": keyword,
            "page": page,
            "count": 20
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.toutiao.com/"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    await self._process_articles(data.get('data', []))
                else:
                    logger.warning(f"今日头条请求失败，状态码: {response.status}")
                    self.task.failed_items += 1
                    
        except asyncio.TimeoutError:
            logger.warning(f"今日头条请求超时: {url}")
            self.task.failed_items += 1
        except Exception as e:
            logger.error(f"今日头条请求异常: {e}")
            self.task.failed_items += 1
    
    async def _process_articles(self, articles: List[Dict]):
        """处理文章数据"""
        db_manager = await get_database_manager()
        collection = await db_manager.get_mongodb_collection("contents")
        
        for article in articles:
            try:
                # 提取文章数据
                content_data = ContentCreate(
                    title=article.get('title', ''),
                    content=article.get('content', ''),
                    source=ContentSource.TOUTIAO,
                    author=article.get('author', ''),
                    source_url=article.get('url'),
                    keywords=self.task.config.keywords,
                    publish_time=datetime.fromtimestamp(article.get('publish_time', 0)) if article.get('publish_time') else None
                )
                
                # 保存到数据库
                content_dict = content_data.dict()
                content_dict['status'] = ContentStatus.PENDING
                content_dict['created_at'] = datetime.now()
                content_dict['id'] = str(uuid.uuid4())
                
                await collection.insert_one(content_dict)
                self.task.success_items += 1
                
                logger.debug(f"保存今日头条文章: {content_data.title}")
                
            except Exception as e:
                logger.error(f"处理今日头条文章失败: {e}")
                self.task.failed_items += 1


class BaijiahaoCrawler(BaseCrawler):
    """百家号爬虫实现"""
    
    async def crawl(self) -> None:
        """百家号爬取逻辑"""
        self.task.status = CrawlerStatus.RUNNING
        self.task.started_at = datetime.now()
        
        try:
            # 百家号爬取逻辑实现
            logger.info("开始爬取百家号...")
            
            # 模拟爬取过程
            for i in range(self.task.config.max_pages):
                if self.should_stop:
                    break
                
                while self.task.status == CrawlerStatus.PAUSED:
                    await asyncio.sleep(1)
                
                # 模拟数据爬取
                await self._simulate_crawl_page(i + 1)
                await asyncio.sleep(self.task.config.interval)
            
            self.task.status = CrawlerStatus.FINISHED
            self.task.finished_at = datetime.now()
            
        except Exception as e:
            self.task.status = CrawlerStatus.ERROR
            self.task.error_message = str(e)
            self.task.finished_at = datetime.now()
            logger.error(f"百家号爬取异常: {e}")
    
    async def _simulate_crawl_page(self, page: int):
        """模拟爬取页面"""
        # 这里是模拟实现，实际需要根据百家号接口实现
        self.task.success_items += 5  # 模拟每页爬取5篇文章
        self.task.progress = (page / self.task.config.max_pages) * 100
        self.task.current_page = page
        logger.info(f"模拟爬取百家号第{page}页完成")


class XiaohongshuCrawler(BaseCrawler):
    """小红书爬虫实现"""
    
    async def crawl(self) -> None:
        """小红书爬取逻辑"""
        self.task.status = CrawlerStatus.RUNNING
        self.task.started_at = datetime.now()
        
        try:
            logger.info("开始爬取小红书...")
            
            # 模拟爬取过程
            for i in range(self.task.config.max_pages):
                if self.should_stop:
                    break
                
                while self.task.status == CrawlerStatus.PAUSED:
                    await asyncio.sleep(1)
                
                await self._simulate_crawl_page(i + 1)
                await asyncio.sleep(self.task.config.interval)
            
            self.task.status = CrawlerStatus.FINISHED
            self.task.finished_at = datetime.now()
            
        except Exception as e:
            self.task.status = CrawlerStatus.ERROR
            self.task.error_message = str(e)
            self.task.finished_at = datetime.now()
            logger.error(f"小红书爬取异常: {e}")
    
    async def _simulate_crawl_page(self, page: int):
        """模拟爬取页面"""
        self.task.success_items += 3  # 模拟每页爬取3篇文章
        self.task.progress = (page / self.task.config.max_pages) * 100
        self.task.current_page = page
        logger.info(f"模拟爬取小红书第{page}页完成")


class CrawlerManager:
    """爬虫管理器 - 核心管理类"""
    
    def __init__(self):
        self.tasks: Dict[str, CrawlerTask] = {}
        self.crawlers: Dict[str, BaseCrawler] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.settings = get_settings()
        self.is_running = True
        
        # 爬虫类型映射
        self.crawler_classes = {
            ContentSource.TOUTIAO: ToutiaoCrawler,
            ContentSource.BAIJIAHAO: BaijiahaoCrawler,
            ContentSource.XIAOHONGSHU: XiaohongshuCrawler,
        }
        
        # 设置信号处理
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """处理关闭信号"""
        logger.info(f"收到关闭信号 {signum}，开始停止所有爬虫...")
        asyncio.create_task(self.stop_all_crawlers())
    
    async def initialize(self):
        """初始化管理器"""
        # 创建HTTP会话
        timeout = aiohttp.ClientTimeout(total=self.settings.crawler.crawler_timeout)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=100),
            headers={"User-Agent": self.settings.crawler.user_agents[0]}
        )
        logger.info("爬虫管理器初始化完成")
    
    async def cleanup(self):
        """清理资源"""
        await self.stop_all_crawlers()
        if self.session:
            await self.session.close()
        logger.info("爬虫管理器资源清理完成")
    
    async def create_task(self, config: CrawlerConfig) -> str:
        """创建爬虫任务"""
        task_id = str(uuid.uuid4())
        
        # 创建任务对象
        task = CrawlerTask(
            task_id=task_id,
            config=config,
            status=CrawlerStatus.IDLE,
            created_at=datetime.now()
        )
        
        self.tasks[task_id] = task
        
        # 保存任务到数据库
        await self._save_task_to_db(task)
        
        logger.info(f"创建爬虫任务: {task_id}, 平台: {config.platform}")
        return task_id
    
    async def start_task(self, task_id: str) -> bool:
        """启动爬虫任务"""
        if task_id not in self.tasks:
            logger.error(f"任务不存在: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        if task.status != CrawlerStatus.IDLE:
            logger.warning(f"任务状态不允许启动: {task_id}, 当前状态: {task.status}")
            return False
        
        # 检查并发限制
        running_count = len([t for t in self.tasks.values() if t.status == CrawlerStatus.RUNNING])
        if running_count >= self.settings.crawler.max_concurrent_crawlers:
            logger.warning(f"达到最大并发限制: {running_count}")
            return False
        
        try:
            # 创建爬虫实例
            crawler_class = self.crawler_classes.get(task.config.platform)
            if not crawler_class:
                raise ValueError(f"不支持的平台: {task.config.platform}")
            
            crawler = crawler_class(task, self.session)
            self.crawlers[task_id] = crawler
            
            # 启动爬虫任务
            asyncio.create_task(self._run_crawler(task_id, crawler))
            
            logger.info(f"启动爬虫任务: {task_id}")
            return True
            
        except Exception as e:
            task.status = CrawlerStatus.ERROR
            task.error_message = str(e)
            logger.error(f"启动爬虫任务失败: {task_id}, 错误: {e}")
            return False
    
    async def _run_crawler(self, task_id: str, crawler: BaseCrawler):
        """运行爬虫任务"""
        try:
            await crawler.crawl()
        except Exception as e:
            logger.error(f"爬虫任务执行异常: {task_id}, 错误: {e}")
            crawler.task.status = CrawlerStatus.ERROR
            crawler.task.error_message = str(e)
            crawler.task.finished_at = datetime.now()
        finally:
            # 清理爬虫实例
            if task_id in self.crawlers:
                del self.crawlers[task_id]
            
            # 更新数据库
            await self._save_task_to_db(crawler.task)
    
    async def stop_task(self, task_id: str) -> bool:
        """停止爬虫任务"""
        if task_id not in self.tasks:
            logger.error(f"任务不存在: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        if task.status not in [CrawlerStatus.RUNNING, CrawlerStatus.PAUSED]:
            logger.warning(f"任务状态不允许停止: {task_id}, 当前状态: {task.status}")
            return False
        
        if task_id in self.crawlers:
            await self.crawlers[task_id].stop()
        
        task.status = CrawlerStatus.STOPPED
        task.finished_at = datetime.now()
        
        logger.info(f"停止爬虫任务: {task_id}")
        return True
    
    async def pause_task(self, task_id: str) -> bool:
        """暂停爬虫任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status != CrawlerStatus.RUNNING:
            return False
        
        if task_id in self.crawlers:
            await self.crawlers[task_id].pause()
        
        return True
    
    async def resume_task(self, task_id: str) -> bool:
        """恢复爬虫任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status != CrawlerStatus.PAUSED:
            return False
        
        if task_id in self.crawlers:
            await self.crawlers[task_id].resume()
        
        return True
    
    def get_task_status(self, task_id: str) -> Optional[CrawlerTask]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[CrawlerTask]:
        """获取所有任务"""
        return list(self.tasks.values())
    
    def get_running_tasks(self) -> List[CrawlerTask]:
        """获取运行中的任务"""
        return [task for task in self.tasks.values() if task.status == CrawlerStatus.RUNNING]
    
    async def stop_all_crawlers(self):
        """停止所有爬虫"""
        logger.info("开始停止所有爬虫任务...")
        
        tasks = []
        for task_id in list(self.crawlers.keys()):
            tasks.append(self.stop_task(task_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.is_running = False
        logger.info("所有爬虫任务已停止")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tasks = len(self.tasks)
        running_tasks = len([t for t in self.tasks.values() if t.status == CrawlerStatus.RUNNING])
        finished_tasks = len([t for t in self.tasks.values() if t.status == CrawlerStatus.FINISHED])
        error_tasks = len([t for t in self.tasks.values() if t.status == CrawlerStatus.ERROR])
        
        total_success = sum(task.success_items for task in self.tasks.values())
        total_failed = sum(task.failed_items for task in self.tasks.values())
        
        return {
            "total_tasks": total_tasks,
            "running_tasks": running_tasks,
            "finished_tasks": finished_tasks,
            "error_tasks": error_tasks,
            "total_success_items": total_success,
            "total_failed_items": total_failed,
            "overall_success_rate": (total_success / (total_success + total_failed) * 100) if (total_success + total_failed) > 0 else 0
        }
    
    async def _save_task_to_db(self, task: CrawlerTask):
        """保存任务到数据库"""
        try:
            db_manager = await get_database_manager()
            collection = await db_manager.get_mongodb_collection("crawler_tasks")
            
            task_dict = {
                "task_id": task.task_id,
                "platform": task.config.platform,
                "keywords": task.config.keywords,
                "status": task.status,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "finished_at": task.finished_at,
                "progress": task.progress,
                "total_items": task.total_items,
                "success_items": task.success_items,
                "failed_items": task.failed_items,
                "error_message": task.error_message,
                "current_page": task.current_page,
                "config": task.config.__dict__
            }
            
            await collection.replace_one(
                {"task_id": task.task_id},
                task_dict,
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"保存任务到数据库失败: {e}")


# 全局爬虫管理器实例
crawler_manager = CrawlerManager()


async def get_crawler_manager() -> CrawlerManager:
    """获取爬虫管理器实例"""
    return crawler_manager