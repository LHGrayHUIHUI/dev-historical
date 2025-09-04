"""
监控服务主入口文件

此模块提供监控系统的统一入口点，负责初始化和协调
所有监控组件的工作，包括指标收集、告警管理、日志收集、
链路追踪等功能。

主要功能：
- 统一初始化所有监控组件
- 提供监控系统的生命周期管理
- 协调各监控组件之间的配置和依赖
- 提供统一的监控API接口
- 管理监控数据的收集和存储

Author: 开发团队
Created: 2025-09-04
Version: 1.0.0
"""

import asyncio
import os
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

# 导入监控组件
from .metrics_middleware import PrometheusMetricsMiddleware, get_business_metrics
from .monitoring_controller import router as monitoring_router
from .tracing_service import initialize_tracing_for_service, get_tracing_service
from .alert_service import get_alert_manager, create_default_alert_rules, EmailNotification, SlackNotification
from .logging_service import get_logging_service, create_default_log_config, LogLevel

# 获取结构化日志记录器
logger = structlog.get_logger()

class MonitoringService:
    """监控服务主类
    
    负责协调和管理所有监控组件的初始化、配置和运行。
    提供统一的监控服务入口点。
    
    Attributes:
        service_name: 监控服务名称
        config: 监控配置
        app: FastAPI应用实例
        tracing_service: 链路追踪服务
        alert_manager: 告警管理器
        logging_service: 日志服务
        is_running: 服务运行状态
    """
    
    def __init__(
        self,
        service_name: str = "monitoring-service",
        config: Optional[Dict[str, Any]] = None
    ):
        """初始化监控服务
        
        Args:
            service_name: 服务名称
            config: 监控配置字典
        """
        self.service_name = service_name
        self.config = config or self._load_default_config()
        self.app: Optional[FastAPI] = None
        self.tracing_service = None
        self.alert_manager = None
        self.logging_service = None
        self.is_running = False
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(
            "监控服务初始化完成",
            service_name=self.service_name,
            config_keys=list(self.config.keys())
        )
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置
        
        Returns:
            Dict[str, Any]: 默认配置字典
        """
        return {
            # 服务配置
            "service": {
                "name": "monitoring-service",
                "version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "environment": os.getenv("ENVIRONMENT", "development"),
                "host": os.getenv("SERVICE_HOST", "0.0.0.0"),
                "port": int(os.getenv("SERVICE_PORT", "8004")),
                "debug": os.getenv("DEBUG", "false").lower() == "true"
            },
            
            # 指标配置
            "metrics": {
                "enabled": True,
                "prometheus_enabled": True,
                "business_metrics_enabled": True,
                "collect_interval": 15  # 秒
            },
            
            # 链路追踪配置
            "tracing": {
                "enabled": os.getenv("TRACING_ENABLED", "true").lower() == "true",
                "jaeger_endpoint": os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
                "sampling_rate": float(os.getenv("JAEGER_SAMPLING_RATE", "0.1")),
                "service_name": os.getenv("SERVICE_NAME", "monitoring-service")
            },
            
            # 告警配置
            "alerting": {
                "enabled": True,
                "check_interval": 60,  # 秒
                "email_enabled": os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true",
                "slack_enabled": os.getenv("SLACK_ALERTS_ENABLED", "false").lower() == "true",
                "smtp_server": os.getenv("SMTP_SERVER"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "smtp_username": os.getenv("SMTP_USERNAME"),
                "smtp_password": os.getenv("SMTP_PASSWORD"),
                "alert_emails": os.getenv("ALERT_EMAILS", "").split(",") if os.getenv("ALERT_EMAILS") else [],
                "slack_webhook_url": os.getenv("SLACK_WEBHOOK_URL")
            },
            
            # 日志配置
            "logging": {
                "enabled": True,
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "elasticsearch_enabled": os.getenv("ELASTICSEARCH_ENABLED", "false").lower() == "true",
                "elasticsearch_url": os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
                "log_dir": os.getenv("LOG_DIR", "logs"),
                "max_file_size": int(os.getenv("LOG_MAX_FILE_SIZE", "100")),  # MB
                "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "10"))
            }
        }
    
    async def initialize(self):
        """初始化所有监控组件"""
        try:
            # 1. 初始化日志服务
            await self._initialize_logging()
            
            # 2. 初始化链路追踪
            await self._initialize_tracing()
            
            # 3. 初始化告警管理
            await self._initialize_alerting()
            
            # 4. 创建FastAPI应用
            await self._create_fastapi_app()
            
            # 5. 启动后台任务
            await self._start_background_tasks()
            
            logger.info("所有监控组件初始化完成")
            
        except Exception as e:
            logger.error(
                "监控服务初始化失败",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _initialize_logging(self):
        """初始化日志服务"""
        if not self.config["logging"]["enabled"]:
            logger.info("日志服务已禁用")
            return
        
        try:
            log_config = create_default_log_config(
                service_name=self.service_name,
                log_level=LogLevel(self.config["logging"]["level"]),
                elasticsearch_enabled=self.config["logging"]["elasticsearch_enabled"]
            )
            
            # 更新配置
            log_config.elasticsearch_url = self.config["logging"]["elasticsearch_url"]
            log_config.log_dir = self.config["logging"]["log_dir"]
            log_config.max_file_size = self.config["logging"]["max_file_size"]
            log_config.backup_count = self.config["logging"]["backup_count"]
            
            self.logging_service = get_logging_service(log_config)
            logger.info("日志服务初始化完成")
            
        except Exception as e:
            logger.error("日志服务初始化失败", error=str(e))
            raise
    
    async def _initialize_tracing(self):
        """初始化链路追踪服务"""
        if not self.config["tracing"]["enabled"]:
            logger.info("链路追踪已禁用")
            return
        
        try:
            self.tracing_service = initialize_tracing_for_service(
                service_name=self.config["tracing"]["service_name"],
                jaeger_endpoint=self.config["tracing"]["jaeger_endpoint"],
                sampling_rate=self.config["tracing"]["sampling_rate"]
            )
            logger.info("链路追踪服务初始化完成")
            
        except Exception as e:
            logger.error("链路追踪服务初始化失败", error=str(e))
            raise
    
    async def _initialize_alerting(self):
        """初始化告警管理"""
        if not self.config["alerting"]["enabled"]:
            logger.info("告警管理已禁用")
            return
        
        try:
            self.alert_manager = get_alert_manager()
            
            # 添加默认告警规则
            default_rules = create_default_alert_rules()
            for rule in default_rules:
                self.alert_manager.add_rule(rule)
            
            # 配置通知渠道
            await self._setup_notification_channels()
            
            # 启动告警监控
            await self.alert_manager.start_monitoring()
            
            logger.info("告警管理初始化完成")
            
        except Exception as e:
            logger.error("告警管理初始化失败", error=str(e))
            raise
    
    async def _setup_notification_channels(self):
        """设置通知渠道"""
        try:
            # 配置邮件通知
            if self.config["alerting"]["email_enabled"]:
                email_config = {
                    "enabled": True,
                    "smtp_server": self.config["alerting"]["smtp_server"],
                    "smtp_port": self.config["alerting"]["smtp_port"],
                    "username": self.config["alerting"]["smtp_username"],
                    "password": self.config["alerting"]["smtp_password"],
                    "to_emails": self.config["alerting"]["alert_emails"]
                }
                
                email_channel = EmailNotification("email", email_config)
                self.alert_manager.add_notification_channel(email_channel)
                logger.info("邮件通知渠道已配置")
            
            # 配置Slack通知
            if self.config["alerting"]["slack_enabled"]:
                slack_config = {
                    "enabled": True,
                    "webhook_url": self.config["alerting"]["slack_webhook_url"]
                }
                
                slack_channel = SlackNotification("slack", slack_config)
                self.alert_manager.add_notification_channel(slack_channel)
                logger.info("Slack通知渠道已配置")
                
        except Exception as e:
            logger.error("通知渠道配置失败", error=str(e))
    
    async def _create_fastapi_app(self):
        """创建FastAPI应用"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # 启动逻辑
            logger.info("监控服务启动中...")
            yield
            # 关闭逻辑
            logger.info("监控服务关闭中...")
            await self.shutdown()
        
        self.app = FastAPI(
            title="历史文本处理项目 - 监控服务",
            description="提供完整的系统监控、告警、日志和链路追踪功能",
            version=self.config["service"]["version"],
            lifespan=lifespan,
            debug=self.config["service"]["debug"]
        )
        
        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 添加监控中间件
        if self.config["metrics"]["enabled"]:
            metrics_middleware = PrometheusMetricsMiddleware(
                self.app,
                service_name=self.service_name
            )
            self.app.add_middleware(metrics_middleware.__class__, middleware=metrics_middleware)
        
        # 链路追踪中间件
        if self.tracing_service:
            self.tracing_service.instrument_fastapi(self.app)
        
        # 添加路由
        self.app.include_router(monitoring_router, prefix="/api/v1")
        
        # 根路径健康检查
        @self.app.get("/")
        async def root():
            return {
                "service": self.service_name,
                "version": self.config["service"]["version"],
                "status": "running" if self.is_running else "starting",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 添加监控特定的API端点
        await self._add_monitoring_endpoints()
        
        logger.info("FastAPI应用创建完成")
    
    async def _add_monitoring_endpoints(self):
        """添加监控特定的API端点"""
        
        @self.app.get("/api/v1/monitoring/alerts")
        async def get_alerts():
            """获取活跃告警"""
            if not self.alert_manager:
                raise HTTPException(status_code=503, detail="告警管理未启用")
            
            active_alerts = self.alert_manager.get_active_alerts()
            return {
                "alerts": [alert.to_dict() for alert in active_alerts],
                "total": len(active_alerts),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.post("/api/v1/monitoring/alerts/{alert_id}/silence")
        async def silence_alert(alert_id: str, duration: int = 3600, reason: str = ""):
            """静默告警"""
            if not self.alert_manager:
                raise HTTPException(status_code=503, detail="告警管理未启用")
            
            self.alert_manager.suppress_alert(alert_id, duration, reason)
            return {"message": f"告警 {alert_id} 已静默 {duration} 秒"}
        
        @self.app.get("/api/v1/monitoring/logs/search")
        async def search_logs(query: str = "*", size: int = 100):
            """搜索日志"""
            if not self.logging_service:
                raise HTTPException(status_code=503, detail="日志服务未启用")
            
            try:
                logs = await self.logging_service.search_logs(
                    query=query,
                    size=size
                )
                return {
                    "logs": logs,
                    "total": len(logs),
                    "query": query
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/monitoring/traces/{trace_id}")
        async def get_trace(trace_id: str):
            """获取链路追踪详情"""
            if not self.tracing_service:
                raise HTTPException(status_code=503, detail="链路追踪未启用")
            
            # 这里应该实现从Jaeger查询trace的逻辑
            return {
                "trace_id": trace_id,
                "message": "链路追踪查询功能待实现"
            }
    
    async def _start_background_tasks(self):
        """启动后台任务"""
        # 启动指标收集任务
        if self.config["metrics"]["enabled"]:
            asyncio.create_task(self._metrics_collection_task())
        
        # 启动日志清理任务
        if self.logging_service:
            asyncio.create_task(self._log_cleanup_task())
    
    async def _metrics_collection_task(self):
        """指标收集后台任务"""
        while self.is_running:
            try:
                # 更新业务指标
                business_metrics = get_business_metrics(self.service_name)
                
                # 记录监控服务自身的指标
                business_metrics.record_text_processing(
                    "monitoring",
                    "success", 
                    1.0  # 示例处理时间
                )
                
                await asyncio.sleep(self.config["metrics"]["collect_interval"])
                
            except Exception as e:
                logger.error("指标收集任务异常", error=str(e))
                await asyncio.sleep(60)  # 异常时等待1分钟
    
    async def _log_cleanup_task(self):
        """日志清理后台任务"""
        while self.is_running:
            try:
                # 每天清理一次过期日志
                await asyncio.sleep(86400)  # 24小时
                
                if self.logging_service:
                    await self.logging_service.cleanup_old_logs(retention_days=30)
                
            except Exception as e:
                logger.error("日志清理任务异常", error=str(e))
    
    async def start(self):
        """启动监控服务"""
        try:
            await self.initialize()
            self.is_running = True
            
            import uvicorn
            
            config = uvicorn.Config(
                app=self.app,
                host=self.config["service"]["host"],
                port=self.config["service"]["port"],
                log_level="info" if not self.config["service"]["debug"] else "debug",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            logger.info(
                "监控服务启动完成",
                host=self.config["service"]["host"],
                port=self.config["service"]["port"]
            )
            
            await server.serve()
            
        except Exception as e:
            logger.error("监控服务启动失败", error=str(e))
            await self.shutdown()
            sys.exit(1)
    
    async def shutdown(self):
        """关闭监控服务"""
        logger.info("正在关闭监控服务...")
        
        self.is_running = False
        
        try:
            # 停止告警监控
            if self.alert_manager:
                await self.alert_manager.stop_monitoring()
            
            # 关闭链路追踪
            if self.tracing_service:
                self.tracing_service.shutdown()
            
            # 关闭日志服务
            if self.logging_service:
                await self.logging_service.close()
            
            logger.info("监控服务已关闭")
            
        except Exception as e:
            logger.error("监控服务关闭异常", error=str(e))
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，正在关闭监控服务...")
        asyncio.create_task(self.shutdown())

def create_monitoring_service(
    service_name: str = "monitoring-service",
    config: Optional[Dict[str, Any]] = None
) -> MonitoringService:
    """创建监控服务实例
    
    Args:
        service_name: 服务名称
        config: 配置字典
        
    Returns:
        MonitoringService: 监控服务实例
    """
    return MonitoringService(service_name=service_name, config=config)

async def main():
    """主入口函数"""
    # 创建并启动监控服务
    service = create_monitoring_service()
    await service.start()

if __name__ == "__main__":
    # 运行监控服务
    asyncio.run(main())