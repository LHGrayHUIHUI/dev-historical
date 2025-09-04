"""
告警管理服务模块

此模块提供完整的告警管理功能，包括告警规则定义、
告警触发检测、通知发送、告警静默等功能。

主要功能：
- 告警规则管理
- 实时指标监控和告警触发
- 多渠道通知发送（邮件、Slack、钉钉等）
- 告警聚合和抑制
- 告警历史记录

Author: 开发团队
Created: 2025-09-04
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import structlog
from prometheus_client import CollectorRegistry, generate_latest
import os

# 获取结构化日志记录器
logger = structlog.get_logger()

class AlertSeverity(Enum):
    """告警严重程度枚举"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class AlertStatus(Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """告警规则定义
    
    定义告警的触发条件、严重程度、通知配置等信息。
    
    Attributes:
        name: 告警规则名称
        query: Prometheus查询表达式
        condition: 告警触发条件
        duration: 持续时间阈值
        severity: 告警严重程度
        summary: 告警摘要模板
        description: 告警描述模板
        labels: 告警标签
        annotations: 告警注解
        enabled: 是否启用
    """
    name: str
    query: str
    condition: str  # 例如: "> 0.8", "< 100"
    duration: int  # 持续时间（秒）
    severity: AlertSeverity
    summary: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        """后初始化处理"""
        # 添加默认标签
        if 'severity' not in self.labels:
            self.labels['severity'] = self.severity.value

@dataclass
class Alert:
    """告警实例
    
    表示一个具体的告警事件，包含告警详情、状态、时间等信息。
    
    Attributes:
        id: 告警唯一标识
        rule_name: 关联的告警规则名称
        status: 告警状态
        severity: 告警严重程度
        summary: 告警摘要
        description: 告警详细描述
        labels: 告警标签
        annotations: 告警注解
        starts_at: 告警开始时间
        ends_at: 告警结束时间
        updated_at: 最后更新时间
        value: 触发告警的指标值
        fingerprint: 告警指纹（用于去重）
    """
    id: str
    rule_name: str
    status: AlertStatus
    severity: AlertSeverity
    summary: str
    description: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: datetime
    ends_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    value: Optional[float] = None
    fingerprint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'status': self.status.value,
            'severity': self.severity.value,
            'summary': self.summary,
            'description': self.description,
            'labels': self.labels,
            'annotations': self.annotations,
            'starts_at': self.starts_at.isoformat(),
            'ends_at': self.ends_at.isoformat() if self.ends_at else None,
            'updated_at': self.updated_at.isoformat(),
            'value': self.value,
            'fingerprint': self.fingerprint
        }

class NotificationChannel:
    """通知渠道基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    async def send_notification(self, alert: Alert) -> bool:
        """发送通知（子类需要实现）
        
        Args:
            alert: 告警对象
            
        Returns:
            bool: 发送是否成功
        """
        raise NotImplementedError

class EmailNotification(NotificationChannel):
    """邮件通知渠道"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """发送邮件通知
        
        Args:
            alert: 告警对象
            
        Returns:
            bool: 发送是否成功
        """
        if not self.enabled:
            return True
        
        try:
            smtp_server = self.config.get('smtp_server')
            smtp_port = self.config.get('smtp_port', 587)
            username = self.config.get('username')
            password = self.config.get('password')
            to_emails = self.config.get('to_emails', [])
            
            if not all([smtp_server, username, password, to_emails]):
                logger.warning("邮件配置不完整，跳过发送")
                return False
            
            # 构建邮件内容
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.summary}"
            
            # 邮件正文
            body = f"""
告警详情：

规则名称：{alert.rule_name}
严重程度：{alert.severity.value}
状态：{alert.status.value}
开始时间：{alert.starts_at}
结束时间：{alert.ends_at or '持续中'}

摘要：{alert.summary}
描述：{alert.description}

标签：{json.dumps(alert.labels, ensure_ascii=False, indent=2)}

指标值：{alert.value}
"""
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info(
                "邮件通知发送成功",
                alert_id=alert.id,
                to_emails=to_emails
            )
            return True
            
        except Exception as e:
            logger.error(
                "邮件通知发送失败",
                alert_id=alert.id,
                error=str(e),
                error_type=type(e).__name__
            )
            return False

class SlackNotification(NotificationChannel):
    """Slack通知渠道"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """发送Slack通知
        
        Args:
            alert: 告警对象
            
        Returns:
            bool: 发送是否成功
        """
        if not self.enabled:
            return True
        
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                logger.warning("Slack webhook_url未配置，跳过发送")
                return False
            
            # 构建Slack消息
            color_map = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.INFO: "good"
            }
            
            payload = {
                "text": f"🚨 {alert.severity.value.upper()} Alert",
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": alert.summary,
                    "text": alert.description,
                    "fields": [
                        {"title": "规则", "value": alert.rule_name, "short": True},
                        {"title": "状态", "value": alert.status.value, "short": True},
                        {"title": "开始时间", "value": alert.starts_at.strftime('%Y-%m-%d %H:%M:%S'), "short": True},
                        {"title": "指标值", "value": str(alert.value), "short": True}
                    ],
                    "footer": "监控系统",
                    "ts": int(alert.starts_at.timestamp())
                }]
            }
            
            # 发送HTTP请求
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(
                            "Slack通知发送成功",
                            alert_id=alert.id
                        )
                        return True
                    else:
                        logger.error(
                            "Slack通知发送失败",
                            alert_id=alert.id,
                            status_code=response.status
                        )
                        return False
                        
        except Exception as e:
            logger.error(
                "Slack通知发送失败",
                alert_id=alert.id,
                error=str(e),
                error_type=type(e).__name__
            )
            return False

class AlertManager:
    """告警管理器
    
    负责告警规则管理、告警检测、通知发送等核心功能。
    
    Attributes:
        rules: 告警规则字典
        active_alerts: 活跃告警字典
        notification_channels: 通知渠道列表
        check_interval: 检查间隔（秒）
        alert_history: 告警历史记录
    """
    
    def __init__(self, check_interval: int = 60):
        """初始化告警管理器
        
        Args:
            check_interval: 告警检查间隔（秒）
        """
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: List[NotificationChannel] = []
        self.check_interval = check_interval
        self.alert_history: List[Alert] = []
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        
        logger.info(
            "告警管理器初始化完成",
            check_interval=self.check_interval
        )
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则
        
        Args:
            rule: 告警规则对象
        """
        self.rules[rule.name] = rule
        logger.info(
            "告警规则已添加",
            rule_name=rule.name,
            severity=rule.severity.value
        )
    
    def remove_rule(self, rule_name: str):
        """删除告警规则
        
        Args:
            rule_name: 告警规则名称
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info("告警规则已删除", rule_name=rule_name)
    
    def add_notification_channel(self, channel: NotificationChannel):
        """添加通知渠道
        
        Args:
            channel: 通知渠道对象
        """
        self.notification_channels.append(channel)
        logger.info(
            "通知渠道已添加",
            channel_name=channel.name,
            channel_type=type(channel).__name__
        )
    
    async def start_monitoring(self):
        """开始告警监控"""
        if self._running:
            logger.warning("告警监控已在运行")
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._monitoring_loop())
        logger.info("告警监控已启动")
    
    async def stop_monitoring(self):
        """停止告警监控"""
        if not self._running:
            return
        
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("告警监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "告警检查异常",
                    error=str(e),
                    error_type=type(e).__name__
                )
                await asyncio.sleep(self.check_interval)
    
    async def _check_alerts(self):
        """检查所有告警规则"""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(
                    "告警规则评估失败",
                    rule_name=rule_name,
                    error=str(e)
                )
    
    async def _evaluate_rule(self, rule: AlertRule):
        """评估告警规则
        
        Args:
            rule: 告警规则
        """
        # 这里应该查询Prometheus获取指标值
        # 为了演示，使用模拟数据
        metric_value = await self._query_metric(rule.query)
        
        if metric_value is None:
            return
        
        # 检查是否触发告警条件
        triggered = self._evaluate_condition(metric_value, rule.condition)
        
        alert_id = f"{rule.name}_{hash(str(rule.labels))}"
        existing_alert = self.active_alerts.get(alert_id)
        
        if triggered:
            if existing_alert is None:
                # 创建新告警
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    status=AlertStatus.ACTIVE,
                    severity=rule.severity,
                    summary=rule.summary,
                    description=rule.description,
                    labels=rule.labels.copy(),
                    annotations=rule.annotations.copy(),
                    starts_at=datetime.utcnow(),
                    value=metric_value,
                    fingerprint=alert_id
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # 发送通知
                await self._send_notifications(alert)
                
                logger.warning(
                    "新告警触发",
                    alert_id=alert_id,
                    rule_name=rule.name,
                    value=metric_value
                )
            else:
                # 更新现有告警
                existing_alert.updated_at = datetime.utcnow()
                existing_alert.value = metric_value
        
        else:
            if existing_alert is not None:
                # 解决告警
                existing_alert.status = AlertStatus.RESOLVED
                existing_alert.ends_at = datetime.utcnow()
                existing_alert.updated_at = datetime.utcnow()
                
                # 发送解决通知
                await self._send_notifications(existing_alert)
                
                # 从活跃告警中移除
                del self.active_alerts[alert_id]
                
                logger.info(
                    "告警已解决",
                    alert_id=alert_id,
                    rule_name=rule.name
                )
    
    async def _query_metric(self, query: str) -> Optional[float]:
        """查询指标值（模拟实现）
        
        Args:
            query: Prometheus查询表达式
            
        Returns:
            Optional[float]: 指标值
        """
        # 在实际实现中，这里应该查询Prometheus
        # 现在返回模拟数据
        import random
        return random.random()
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """评估告警条件
        
        Args:
            value: 指标值
            condition: 条件表达式（如 "> 0.8", "< 100"）
            
        Returns:
            bool: 是否满足告警条件
        """
        try:
            # 简单的条件解析和评估
            condition = condition.strip()
            
            if condition.startswith('>='):
                threshold = float(condition[2:].strip())
                return value >= threshold
            elif condition.startswith('<='):
                threshold = float(condition[2:].strip())
                return value <= threshold
            elif condition.startswith('>'):
                threshold = float(condition[1:].strip())
                return value > threshold
            elif condition.startswith('<'):
                threshold = float(condition[1:].strip())
                return value < threshold
            elif condition.startswith('=='):
                threshold = float(condition[2:].strip())
                return abs(value - threshold) < 1e-9
            elif condition.startswith('!='):
                threshold = float(condition[2:].strip())
                return abs(value - threshold) >= 1e-9
            
            return False
            
        except Exception as e:
            logger.error(
                "条件评估失败",
                condition=condition,
                value=value,
                error=str(e)
            )
            return False
    
    async def _send_notifications(self, alert: Alert):
        """发送告警通知
        
        Args:
            alert: 告警对象
        """
        if not self.notification_channels:
            logger.debug("无可用通知渠道")
            return
        
        # 并发发送所有通知渠道
        tasks = []
        for channel in self.notification_channels:
            if channel.enabled:
                task = asyncio.create_task(channel.send_notification(alert))
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if r is True)
            total_count = len(tasks)
            
            logger.info(
                "告警通知发送完成",
                alert_id=alert.id,
                success_count=success_count,
                total_count=total_count
            )
    
    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None,
                         rule_name: Optional[str] = None) -> List[Alert]:
        """获取活跃告警列表
        
        Args:
            severity: 过滤严重程度
            rule_name: 过滤规则名称
            
        Returns:
            List[Alert]: 告警列表
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if rule_name:
            alerts = [a for a in alerts if a.rule_name == rule_name]
        
        return alerts
    
    def get_alert_history(self, 
                         limit: int = 100,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """获取告警历史记录
        
        Args:
            limit: 返回数量限制
            severity: 过滤严重程度
            
        Returns:
            List[Alert]: 历史告警列表
        """
        alerts = self.alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # 按时间倒序排序
        alerts.sort(key=lambda x: x.starts_at, reverse=True)
        
        return alerts[:limit]
    
    def suppress_alert(self, alert_id: str, duration: int, reason: str = ""):
        """静默告警
        
        Args:
            alert_id: 告警ID
            duration: 静默时长（秒）
            reason: 静默原因
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.updated_at = datetime.utcnow()
            
            # 添加静默信息到注解
            alert.annotations['suppressed_until'] = (
                datetime.utcnow() + timedelta(seconds=duration)
            ).isoformat()
            alert.annotations['suppressed_reason'] = reason
            
            logger.info(
                "告警已静默",
                alert_id=alert_id,
                duration=duration,
                reason=reason
            )

# 全局告警管理器实例
_alert_manager: Optional[AlertManager] = None

def get_alert_manager() -> AlertManager:
    """获取全局告警管理器实例
    
    Returns:
        AlertManager: 告警管理器实例
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager

def create_default_alert_rules() -> List[AlertRule]:
    """创建默认告警规则
    
    Returns:
        List[AlertRule]: 默认告警规则列表
    """
    return [
        AlertRule(
            name="ServiceDown",
            query="up",
            condition="== 0",
            duration=60,
            severity=AlertSeverity.CRITICAL,
            summary="服务不可用",
            description="服务 {{ $labels.job }} 已停止响应超过1分钟",
            labels={"alertname": "ServiceDown"},
            annotations={"runbook_url": "https://example.com/runbooks/service-down"}
        ),
        AlertRule(
            name="HighErrorRate",
            query="rate(http_requests_total{status=~\"5..\"}[5m])",
            condition="> 0.1",
            duration=120,
            severity=AlertSeverity.WARNING,
            summary="错误率过高",
            description="服务 {{ $labels.job }} 5分钟内错误率超过10%",
            labels={"alertname": "HighErrorRate"}
        ),
        AlertRule(
            name="HighResponseTime",
            query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            condition="> 1",
            duration=180,
            severity=AlertSeverity.WARNING,
            summary="响应时间过长",
            description="服务 {{ $labels.job }} 95%分位响应时间超过1秒",
            labels={"alertname": "HighResponseTime"}
        )
    ]