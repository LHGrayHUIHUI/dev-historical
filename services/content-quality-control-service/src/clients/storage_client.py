"""
Storage Service 客户端

提供与存储服务的HTTP通信接口，包括质量检测结果存储、
合规检测结果存储、审核任务管理、敏感词库管理等功能。
"""

import httpx
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from ..config.settings import settings
from ..models.quality_models import (
    QualityCheckResult, ComplianceCheckResult, ReviewTask,
    ContentReviewRecord, ErrorResponse
)

class StorageServiceError(Exception):
    """存储服务异常类"""
    def __init__(self, message: str, status_code: int = None, response_data: Dict = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        super().__init__(message)

class StorageServiceClient:
    """
    Storage Service HTTP客户端
    
    负责与storage-service进行所有数据交互，包括：
    - 质量检测结果的CRUD操作
    - 合规检测结果的CRUD操作  
    - 审核任务管理
    - 敏感词库管理
    - 质量规则配置管理
    """
    
    def __init__(self):
        """初始化客户端"""
        self.base_url = settings.STORAGE_SERVICE_URL.rstrip('/')
        self.timeout = settings.STORAGE_SERVICE_TIMEOUT
        self.max_retries = settings.STORAGE_SERVICE_RETRIES
        
        # 创建HTTP客户端
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"{settings.SERVICE_NAME}/{settings.SERVICE_VERSION}"
            }
        )
        
        logger.info(f"Storage Service客户端初始化完成: {self.base_url}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def close(self):
        """关闭客户端连接"""
        await self.client.aclose()
        logger.info("Storage Service客户端连接已关闭")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException))
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        发起HTTP请求的通用方法
        
        Args:
            method: HTTP方法
            endpoint: API端点
            **kwargs: 请求参数
            
        Returns:
            响应数据字典
            
        Raises:
            StorageServiceError: 请求失败时抛出
        """
        url = f"{endpoint}"
        
        try:
            response = await self.client.request(method, url, **kwargs)
            
            # 记录请求日志
            logger.debug(f"Storage Service请求: {method} {url} -> {response.status_code}")
            
            # 检查响应状态
            if response.status_code >= 400:
                error_data = {}
                try:
                    error_data = response.json()
                except:
                    error_data = {"detail": response.text}
                
                raise StorageServiceError(
                    f"Storage Service请求失败: {response.status_code}",
                    status_code=response.status_code,
                    response_data=error_data
                )
            
            # 解析响应数据
            return response.json()
            
        except httpx.ConnectError as e:
            logger.error(f"Storage Service连接失败: {e}")
            raise StorageServiceError(f"无法连接到Storage Service: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Storage Service请求超时: {e}")
            raise StorageServiceError(f"Storage Service请求超时: {e}")
        except Exception as e:
            logger.error(f"Storage Service请求异常: {e}")
            raise StorageServiceError(f"Storage Service请求异常: {e}")
    
    # ==================== 质量检测结果管理 ====================
    
    async def save_quality_check_result(self, result: QualityCheckResult) -> Dict[str, Any]:
        """
        保存质量检测结果
        
        Args:
            result: 质量检测结果对象
            
        Returns:
            保存结果
        """
        logger.info(f"保存质量检测结果: {result.check_id}")
        
        # 转换为存储格式
        data = {
            "check_id": result.check_id,
            "content_id": result.content_id,
            "check_type": "quality",
            "overall_score": result.overall_score,
            "detected_issues": [issue.dict() for issue in result.issues],
            "suggestions": result.suggestions,
            "auto_fixes_applied": result.auto_fixes,
            "check_duration_ms": result.processing_time_ms
        }
        
        return await self._make_request(
            "POST",
            "/api/v1/quality/check-results",
            json=data
        )
    
    async def get_quality_check_result(self, check_id: str) -> Dict[str, Any]:
        """
        获取质量检测结果
        
        Args:
            check_id: 检测ID
            
        Returns:
            质量检测结果
        """
        return await self._make_request(
            "GET",
            f"/api/v1/quality/check-results/{check_id}"
        )
    
    async def get_quality_history(self, content_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        获取内容的质量检测历史
        
        Args:
            content_id: 内容ID
            limit: 返回数量限制
            
        Returns:
            质量检测历史
        """
        return await self._make_request(
            "GET",
            f"/api/v1/quality/history/{content_id}",
            params={"limit": limit}
        )
    
    # ==================== 合规检测结果管理 ====================
    
    async def save_compliance_check_result(self, result: ComplianceCheckResult) -> Dict[str, Any]:
        """
        保存合规检测结果
        
        Args:
            result: 合规检测结果对象
            
        Returns:
            保存结果
        """
        logger.info(f"保存合规检测结果: {result.check_id}")
        
        # 转换为存储格式
        data = {
            "check_id": result.check_id,
            "content_id": result.content_id,
            "compliance_status": result.compliance_status.value,
            "risk_score": result.risk_score,
            "violations": [violation.dict() for violation in result.violations],
            "recommendations": result.recommendations,
            "reviewed_by": None,  # 自动检测
            "review_notes": None
        }
        
        return await self._make_request(
            "POST",
            "/api/v1/compliance/check-results",
            json=data
        )
    
    async def get_compliance_check_result(self, check_id: str) -> Dict[str, Any]:
        """
        获取合规检测结果
        
        Args:
            check_id: 检测ID
            
        Returns:
            合规检测结果
        """
        return await self._make_request(
            "GET",
            f"/api/v1/compliance/check-results/{check_id}"
        )
    
    # ==================== 敏感词库管理 ====================
    
    async def get_sensitive_words(self, 
                                category: Optional[str] = None,
                                active_only: bool = True) -> Dict[str, Any]:
        """
        获取敏感词列表
        
        Args:
            category: 敏感词分类
            active_only: 仅获取激活的敏感词
            
        Returns:
            敏感词列表
        """
        params = {}
        if category:
            params["category"] = category
        if active_only:
            params["active_only"] = "true"
        
        return await self._make_request(
            "GET",
            "/api/v1/compliance/sensitive-words",
            params=params
        )
    
    async def add_sensitive_word(self, word_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加敏感词
        
        Args:
            word_data: 敏感词数据
            
        Returns:
            添加结果
        """
        return await self._make_request(
            "POST",
            "/api/v1/compliance/sensitive-words",
            json=word_data
        )
    
    # ==================== 质量规则管理 ====================
    
    async def get_quality_rules(self, rule_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取质量检测规则
        
        Args:
            rule_type: 规则类型
            
        Returns:
            质量规则列表
        """
        params = {}
        if rule_type:
            params["rule_type"] = rule_type
        
        return await self._make_request(
            "GET",
            "/api/v1/quality/rules",
            params=params
        )
    
    async def get_compliance_rules(self, rule_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取合规检测规则
        
        Args:
            rule_type: 规则类型
            
        Returns:
            合规规则列表
        """
        params = {}
        if rule_type:
            params["rule_type"] = rule_type
        
        return await self._make_request(
            "GET",
            "/api/v1/compliance/rules",
            params=params
        )
    
    # ==================== 审核任务管理 ====================
    
    async def create_review_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建审核任务
        
        Args:
            task_data: 任务数据
            
        Returns:
            创建结果，包含任务ID
        """
        logger.info(f"创建审核任务: content_id={task_data.get('content_id')}")
        
        return await self._make_request(
            "POST",
            "/api/v1/review/tasks",
            json=task_data
        )
    
    async def get_review_task(self, task_id: str) -> Dict[str, Any]:
        """
        获取审核任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务详情
        """
        return await self._make_request(
            "GET",
            f"/api/v1/review/tasks/{task_id}"
        )
    
    async def update_review_task(self, task_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新审核任务
        
        Args:
            task_id: 任务ID
            update_data: 更新数据
            
        Returns:
            更新结果
        """
        return await self._make_request(
            "PUT",
            f"/api/v1/review/tasks/{task_id}",
            json=update_data
        )
    
    async def get_review_tasks(self, 
                             status: Optional[str] = None,
                             assigned_to: Optional[str] = None,
                             priority: Optional[str] = None,
                             page: int = 1,
                             per_page: int = 20) -> Dict[str, Any]:
        """
        获取审核任务列表
        
        Args:
            status: 任务状态
            assigned_to: 分配的审核员
            priority: 优先级
            page: 页码
            per_page: 每页数量
            
        Returns:
            任务列表
        """
        params = {
            "page": page,
            "per_page": per_page
        }
        
        if status:
            params["status"] = status
        if assigned_to:
            params["assigned_to"] = assigned_to
        if priority:
            params["priority"] = priority
        
        return await self._make_request(
            "GET",
            "/api/v1/review/tasks",
            params=params
        )
    
    async def submit_review_decision(self, 
                                   task_id: str, 
                                   decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提交审核决策
        
        Args:
            task_id: 任务ID
            decision_data: 决策数据
            
        Returns:
            提交结果
        """
        logger.info(f"提交审核决策: task_id={task_id}, decision={decision_data.get('decision')}")
        
        return await self._make_request(
            "POST",
            f"/api/v1/review/tasks/{task_id}/decision",
            json=decision_data
        )
    
    # ==================== 工作流管理 ====================
    
    async def get_review_workflows(self, content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取审核工作流列表
        
        Args:
            content_type: 内容类型
            
        Returns:
            工作流列表
        """
        params = {}
        if content_type:
            params["content_type"] = content_type
        
        return await self._make_request(
            "GET",
            "/api/v1/review/workflows",
            params=params
        )
    
    async def get_active_workflows(self) -> Dict[str, Any]:
        """
        获取激活的工作流
        
        Returns:
            激活的工作流列表
        """
        return await self._make_request(
            "GET",
            "/api/v1/review/workflows",
            params={"active_only": "true"}
        )
    
    # ==================== 审核记录管理 ====================
    
    async def save_review_record(self, record: ContentReviewRecord) -> Dict[str, Any]:
        """
        保存内容审核记录
        
        Args:
            record: 审核记录对象
            
        Returns:
            保存结果
        """
        logger.info(f"保存审核记录: {record.record_id}")
        
        # 将记录转换为存储格式
        record_data = record.dict()
        
        return await self._make_request(
            "POST",
            "/api/v1/review/records",
            json=record_data
        )
    
    async def get_review_record(self, content_id: str) -> Dict[str, Any]:
        """
        获取内容审核记录
        
        Args:
            content_id: 内容ID
            
        Returns:
            审核记录
        """
        return await self._make_request(
            "GET",
            f"/api/v1/review/records/{content_id}"
        )
    
    # ==================== 统计分析 ====================
    
    async def get_quality_statistics(self, 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取质量统计数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            质量统计数据
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return await self._make_request(
            "GET",
            "/api/v1/quality/statistics",
            params=params
        )
    
    async def get_compliance_statistics(self, 
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取合规统计数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            合规统计数据
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return await self._make_request(
            "GET",
            "/api/v1/compliance/statistics",
            params=params
        )
    
    async def get_review_statistics(self, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取审核统计数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            审核统计数据
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return await self._make_request(
            "GET",
            "/api/v1/review/statistics",
            params=params
        )
    
    # ==================== 健康检查 ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        检查Storage Service健康状态
        
        Returns:
            健康状态信息
        """
        try:
            return await self._make_request("GET", "/health")
        except Exception as e:
            logger.error(f"Storage Service健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# 创建全局客户端实例
storage_client = StorageServiceClient()

# 依赖注入函数
async def get_storage_client() -> StorageServiceClient:
    """获取Storage Service客户端实例"""
    return storage_client