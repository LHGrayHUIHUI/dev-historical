"""
状态监控API控制器
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from ..services.ai_service import get_ai_service, AIServiceError
from ..services.usage_tracker import get_usage_tracker
from ..services.account_monitor import get_account_monitor
from ..services.model_router import get_model_router


class StatusController:
    """状态监控API控制器"""
    
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/status", tags=["status"])
        self._logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.get("/health")
        async def health_check() -> Dict[str, Any]:
            """
            健康检查接口
            
            Returns:
                服务健康状态
            """
            try:
                health_status = {
                    'status': 'healthy',
                    'timestamp': self._get_current_timestamp(),
                    'service': 'ai-model-service',
                    'version': '1.0.0',
                    'checks': {}
                }
                
                # 检查AI服务状态
                try:
                    ai_service = await get_ai_service()
                    service_status = await ai_service.get_service_status()
                    health_status['checks']['ai_service'] = {
                        'status': 'healthy' if service_status.get('initialized') else 'unhealthy',
                        'details': service_status
                    }
                except Exception as e:
                    health_status['checks']['ai_service'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                
                # 检查模型路由器状态
                try:
                    model_router = await get_model_router()
                    router_stats = await model_router.get_router_statistics()
                    health_status['checks']['model_router'] = {
                        'status': 'healthy',
                        'details': router_stats
                    }
                except Exception as e:
                    health_status['checks']['model_router'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                
                # 检查使用跟踪器状态
                try:
                    usage_tracker = await get_usage_tracker()
                    health_status['checks']['usage_tracker'] = {
                        'status': 'healthy' if usage_tracker.is_running else 'unhealthy',
                        'details': {
                            'running': usage_tracker.is_running,
                            'batch_size': len(usage_tracker.batch_records)
                        }
                    }
                except Exception as e:
                    health_status['checks']['usage_tracker'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                
                # 总体状态判断
                unhealthy_checks = [check for check in health_status['checks'].values() 
                                  if check['status'] == 'unhealthy']
                if unhealthy_checks:
                    health_status['status'] = 'unhealthy'
                
                return health_status
                
            except Exception as e:
                self._logger.error(f"Health check error: {e}")
                return {
                    'status': 'unhealthy',
                    'timestamp': self._get_current_timestamp(),
                    'service': 'ai-model-service',
                    'error': str(e)
                }
        
        @self.router.get("/metrics")
        async def get_metrics() -> Dict[str, Any]:
            """
            获取服务指标
            
            Returns:
                服务性能指标
            """
            try:
                self._logger.info("Getting service metrics")
                
                metrics = {
                    'timestamp': self._get_current_timestamp(),
                    'service': 'ai-model-service',
                    'metrics': {}
                }
                
                # AI服务指标
                try:
                    ai_service = await get_ai_service()
                    service_status = await ai_service.get_service_status()
                    metrics['metrics']['ai_service'] = service_status
                except Exception as e:
                    metrics['metrics']['ai_service'] = {'error': str(e)}
                
                # 使用统计指标
                try:
                    usage_tracker = await get_usage_tracker()
                    usage_summary = await usage_tracker.get_usage_summary(period='1h')
                    metrics['metrics']['usage_1h'] = usage_summary
                    
                    usage_summary_24h = await usage_tracker.get_usage_summary(period='24h')
                    metrics['metrics']['usage_24h'] = usage_summary_24h
                except Exception as e:
                    metrics['metrics']['usage'] = {'error': str(e)}
                
                # 账号健康指标
                try:
                    account_monitor = await get_account_monitor()
                    health_summary = await account_monitor.get_health_summary()
                    metrics['metrics']['account_health'] = health_summary
                except Exception as e:
                    metrics['metrics']['account_health'] = {'error': str(e)}
                
                self._logger.info("Service metrics retrieved successfully")
                
                return metrics
                
            except Exception as e:
                self._logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=f"获取指标失败: {e}")
        
        @self.router.get("/usage")
        async def get_usage_stats(
            period: str = Query('24h', description="统计周期 (1h, 24h, 7d, 30d)"),
            provider: Optional[str] = Query(None, description="筛选提供商"),
            model_id: Optional[str] = Query(None, description="筛选模型")
        ) -> Dict[str, Any]:
            """
            获取使用统计
            
            Args:
                period: 统计周期
                provider: 可选的提供商筛选
                model_id: 可选的模型筛选
                
            Returns:
                使用统计数据
            """
            try:
                self._logger.info(f"Getting usage stats: period={period}, provider={provider}, model_id={model_id}")
                
                # 验证周期参数
                valid_periods = ['1h', '24h', '7d', '30d']
                if period not in valid_periods:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"无效的统计周期: {period}, 支持的周期: {valid_periods}"
                    )
                
                usage_tracker = await get_usage_tracker()
                usage_summary = await usage_tracker.get_usage_summary(
                    period=period, 
                    provider=provider, 
                    model_id=model_id
                )
                
                self._logger.info("Usage stats retrieved successfully")
                
                return usage_summary
                
            except HTTPException:
                raise
            except Exception as e:
                self._logger.error(f"Error getting usage stats: {e}")
                raise HTTPException(status_code=500, detail=f"获取使用统计失败: {e}")
        
        @self.router.get("/performance")
        async def get_performance_metrics(
            period: str = Query('24h', description="统计周期"),
            model_id: Optional[str] = Query(None, description="筛选模型")
        ) -> Dict[str, Any]:
            """
            获取性能指标
            
            Args:
                period: 统计周期
                model_id: 可选的模型筛选
                
            Returns:
                性能指标数据
            """
            try:
                self._logger.info(f"Getting performance metrics: period={period}, model_id={model_id}")
                
                usage_tracker = await get_usage_tracker()
                performance_metrics = await usage_tracker.get_performance_metrics(
                    period=period,
                    model_id=model_id
                )
                
                self._logger.info("Performance metrics retrieved successfully")
                
                return performance_metrics
                
            except Exception as e:
                self._logger.error(f"Error getting performance metrics: {e}")
                raise HTTPException(status_code=500, detail=f"获取性能指标失败: {e}")
        
        @self.router.get("/cost")
        async def get_cost_analysis(
            period: str = Query('30d', description="分析周期"),
            group_by: str = Query('provider', description="分组方式 (provider, model, user)")
        ) -> Dict[str, Any]:
            """
            获取成本分析
            
            Args:
                period: 分析周期
                group_by: 分组方式
                
            Returns:
                成本分析数据
            """
            try:
                self._logger.info(f"Getting cost analysis: period={period}, group_by={group_by}")
                
                # 验证分组参数
                valid_group_by = ['provider', 'model', 'user']
                if group_by not in valid_group_by:
                    raise HTTPException(
                        status_code=400,
                        detail=f"无效的分组方式: {group_by}, 支持的方式: {valid_group_by}"
                    )
                
                usage_tracker = await get_usage_tracker()
                cost_analysis = await usage_tracker.get_cost_analysis(
                    period=period,
                    group_by=group_by
                )
                
                self._logger.info("Cost analysis retrieved successfully")
                
                return cost_analysis
                
            except HTTPException:
                raise
            except Exception as e:
                self._logger.error(f"Error getting cost analysis: {e}")
                raise HTTPException(status_code=500, detail=f"获取成本分析失败: {e}")
        
        @self.router.get("/accounts")
        async def get_account_status() -> Dict[str, Any]:
            """
            获取账号状态
            
            Returns:
                账号健康状态汇总
            """
            try:
                self._logger.info("Getting account status")
                
                account_monitor = await get_account_monitor()
                health_summary = await account_monitor.get_health_summary()
                
                self._logger.info("Account status retrieved successfully")
                
                return health_summary
                
            except Exception as e:
                self._logger.error(f"Error getting account status: {e}")
                raise HTTPException(status_code=500, detail=f"获取账号状态失败: {e}")
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


# 创建控制器实例
def create_status_controller() -> StatusController:
    """创建状态监控控制器实例"""
    return StatusController()