"""
AI模型路由器

负责智能选择最佳的模型和账号组合，实现负载均衡和故障转移
"""

import asyncio
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

import redis.asyncio as aioredis

from ..models.ai_models import ModelConfig, APIAccount, ModelProvider
from ..config.settings import get_settings
from .storage_client import get_storage_client


class ModelRouterError(Exception):
    """模型路由器错误"""
    pass


@dataclass
class RoutingRequest:
    """路由请求"""
    model_name: Optional[str] = None
    provider: Optional[str] = None
    requirements: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    priority: int = 1


@dataclass 
class RoutingResult:
    """路由结果"""
    model: ModelConfig
    account: APIAccount
    routing_strategy: str
    selection_reason: str


class ModelRouter:
    """
    AI模型路由器
    负责选择最佳的模型和账号组合
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._logger = logging.getLogger(__name__)
        
        # 缓存
        self.redis_client = None
        self.models_cache: Dict[str, ModelConfig] = {}
        self.accounts_cache: Dict[str, APIAccount] = {}
        self.routing_strategies: Dict[str, Dict[str, Any]] = {}
        
        # 缓存TTL
        self.cache_ttl_models = self.settings.cache_ttl_models
        self.cache_ttl_accounts = self.settings.cache_ttl_accounts
        
        # 健康检查
        self.last_health_check = {}
        self.health_check_interval = timedelta(seconds=self.settings.health_check_interval)
        
    async def initialize(self):
        """初始化路由器"""
        # 连接Redis
        if self.settings.redis_url:
            try:
                self.redis_client = aioredis.from_url(self.settings.redis_url)
                await self.redis_client.ping()
                self._logger.info("Redis connected successfully")
            except Exception as e:
                self._logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # 加载初始数据
        await self._load_models()
        await self._load_accounts()
        await self._load_routing_strategies()
        
        self._logger.info(f"Model router initialized with {len(self.models_cache)} models and {len(self.accounts_cache)} accounts")
    
    async def select_model_account(self, request: RoutingRequest) -> RoutingResult:
        """
        选择最佳的模型和账号组合
        
        Args:
            request: 路由请求
            
        Returns:
            RoutingResult: 路由结果
        """
        try:
            # 1. 筛选可用模型
            available_models = await self._filter_available_models(request)
            
            if not available_models:
                raise ModelRouterError("没有可用的模型满足要求")
            
            # 2. 根据策略选择模型
            selected_model, strategy_used = await self._select_model_by_strategy(available_models, request)
            
            # 3. 为选定模型选择最佳账号
            selected_account, selection_reason = await self._select_account_for_model(selected_model, request)
            
            # 4. 记录选择结果
            await self._record_selection(selected_model, selected_account, strategy_used)
            
            result = RoutingResult(
                model=selected_model,
                account=selected_account,
                routing_strategy=strategy_used,
                selection_reason=selection_reason
            )
            
            self._logger.debug(f"Selected model {selected_model.name} with account {selected_account.account_name}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Model selection failed: {str(e)}")
            raise ModelRouterError(f"模型选择失败: {str(e)}")
    
    async def _filter_available_models(self, request: RoutingRequest) -> List[ModelConfig]:
        """筛选可用模型"""
        available_models = []
        
        # 确保缓存是最新的
        await self._refresh_cache_if_needed()
        
        for model in self.models_cache.values():
            # 检查模型是否活跃
            if not model.is_active:
                continue
            
            # 检查模型名称匹配
            if request.model_name and model.name != request.model_name:
                continue
            
            # 检查提供商匹配
            if request.provider and model.provider.value != request.provider:
                continue
            
            # 检查特殊能力需求
            if request.requirements:
                if not self._check_model_capabilities(model, request.requirements):
                    continue
            
            # 检查是否有可用账号
            if await self._has_available_account(model):
                available_models.append(model)
        
        return available_models
    
    async def _select_model_by_strategy(self, models: List[ModelConfig], request: RoutingRequest) -> Tuple[ModelConfig, str]:
        """根据路由策略选择模型"""
        strategy_name = self.settings.default_routing_strategy
        strategy = self.routing_strategies.get(strategy_name, {'strategy_type': 'priority'})
        strategy_type = strategy.get('strategy_type', 'priority')
        
        if strategy_type == 'priority':
            # 按优先级选择
            selected = max(models, key=lambda m: m.priority)
            return selected, 'priority'
            
        elif strategy_type == 'cost_based':
            # 按成本选择（优先选择成本低的）
            selected = min(models, key=lambda m: m.cost_per_1k_tokens)
            return selected, 'cost_based'
            
        elif strategy_type == 'round_robin':
            # 轮询选择
            selected = await self._round_robin_select(models)
            return selected, 'round_robin'
            
        elif strategy_type == 'weighted':
            # 按权重随机选择
            selected = await self._weighted_select(models)
            return selected, 'weighted'
            
        elif strategy_type == 'health_based':
            # 基于健康评分选择
            selected = await self._health_based_select(models)
            return selected, 'health_based'
            
        else:
            # 默认选择第一个
            return models[0], 'default'
    
    async def _select_account_for_model(self, model: ModelConfig, request: RoutingRequest) -> Tuple[APIAccount, str]:
        """为指定模型选择最佳账号"""
        available_accounts = []
        
        # 获取该模型的所有可用账号
        for account in self.accounts_cache.values():
            if (account.provider == model.provider and 
                account.status.value == 'active' and
                account.health_score > 0.5):
                available_accounts.append(account)
        
        if not available_accounts:
            raise ModelRouterError(f"模型 {model.name} 没有可用账号")
        
        # 选择策略：综合健康分数、剩余配额、错误率
        best_account = max(available_accounts, key=lambda a: self._calculate_account_score(a))
        
        # 生成选择原因
        reasons = []
        if best_account.health_score > 0.8:
            reasons.append("高健康评分")
        
        quota_usage = best_account.quota_used / max(best_account.quota_limit, 1)
        if quota_usage < 0.8:
            reasons.append("配额充足")
        
        if best_account.error_count < 5:
            reasons.append("错误率低")
        
        selection_reason = ", ".join(reasons) if reasons else "默认选择"
        
        return best_account, selection_reason
    
    def _calculate_account_score(self, account: APIAccount) -> float:
        """计算账号综合评分"""
        # 健康分数权重40%
        health_factor = account.health_score * 0.4
        
        # 剩余配额权重30%
        quota_usage = account.quota_used / max(account.quota_limit, 1)
        quota_factor = max(0, 1 - quota_usage) * 0.3
        
        # 错误率权重20% 
        error_factor = max(0, 1 - account.error_count / 10) * 0.2
        
        # 最后使用时间权重10%（越近使用越好，表明账号稳定）
        time_factor = 0.1
        if account.last_used_at:
            hours_since_use = (datetime.now() - account.last_used_at).total_seconds() / 3600
            time_factor = max(0, 1 - hours_since_use / 24) * 0.1  # 24小时内使用过的账号得分更高
        
        return health_factor + quota_factor + error_factor + time_factor
    
    async def _round_robin_select(self, models: List[ModelConfig]) -> ModelConfig:
        """轮询选择模型"""
        if not self.redis_client:
            return random.choice(models)
        
        try:
            counter_key = f"{self.settings.cache_prefix}round_robin_counter"
            counter = await self.redis_client.get(counter_key)
            
            if counter is None:
                counter = 0
            else:
                counter = int(counter)
            
            # 选择模型
            selected = models[counter % len(models)]
            
            # 更新计数器
            await self.redis_client.set(counter_key, counter + 1, ex=3600)
            
            return selected
            
        except Exception as e:
            self._logger.warning(f"Round robin selection failed, using random: {e}")
            return random.choice(models)
    
    async def _weighted_select(self, models: List[ModelConfig]) -> ModelConfig:
        """按权重选择模型"""
        weights = [model.priority for model in models]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(models)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return models[i]
        
        return models[-1]
    
    async def _health_based_select(self, models: List[ModelConfig]) -> ModelConfig:
        """基于健康评分选择模型"""
        model_health_scores = {}
        
        for model in models:
            # 获取模型的平均账号健康分数
            model_accounts = [a for a in self.accounts_cache.values() 
                             if a.provider == model.provider and a.status.value == 'active']
            
            if model_accounts:
                avg_health = sum(a.health_score for a in model_accounts) / len(model_accounts)
                model_health_scores[model.id] = avg_health
            else:
                model_health_scores[model.id] = 0.0
        
        # 选择健康分数最高的模型
        best_model = max(models, key=lambda m: model_health_scores.get(m.id, 0.0))
        return best_model
    
    def _check_model_capabilities(self, model: ModelConfig, requirements: Dict[str, Any]) -> bool:
        """检查模型是否满足能力需求"""
        for capability, required in requirements.items():
            if required and not model.capabilities.get(capability, False):
                return False
        return True
    
    async def _has_available_account(self, model: ModelConfig) -> bool:
        """检查模型是否有可用账号"""
        for account in self.accounts_cache.values():
            if (account.provider == model.provider and 
                account.status.value == 'active' and
                account.health_score > 0.5):
                return True
        return False
    
    async def _load_models(self):
        """加载模型配置"""
        try:
            storage_client = await get_storage_client()
            models_data = await storage_client.get_ai_models(active_only=True)
            
            self.models_cache = {}
            for model_data in models_data:
                model = ModelConfig(**model_data)
                self.models_cache[model.id] = model
                
            self._logger.info(f"Loaded {len(self.models_cache)} models")
            
        except Exception as e:
            self._logger.error(f"Failed to load models: {e}")
            raise
    
    async def _load_accounts(self):
        """加载账号配置"""
        try:
            storage_client = await get_storage_client()
            accounts_data = await storage_client.get_api_accounts(status='active')
            
            self.accounts_cache = {}
            for account_data in accounts_data:
                account = APIAccount(**account_data)
                self.accounts_cache[account.id] = account
                
            self._logger.info(f"Loaded {len(self.accounts_cache)} accounts")
            
        except Exception as e:
            self._logger.error(f"Failed to load accounts: {e}")
            raise
    
    async def _load_routing_strategies(self):
        """加载路由策略"""
        try:
            storage_client = await get_storage_client()
            strategies_data = await storage_client.get_routing_strategies(active_only=True)
            
            self.routing_strategies = {}
            for strategy_data in strategies_data:
                self.routing_strategies[strategy_data['name']] = strategy_data
                
            # 确保有默认策略
            if 'default' not in self.routing_strategies:
                self.routing_strategies['default'] = {
                    'name': 'default',
                    'strategy_type': self.settings.default_routing_strategy,
                    'config': {}
                }
                
            self._logger.info(f"Loaded {len(self.routing_strategies)} routing strategies")
            
        except Exception as e:
            self._logger.warning(f"Failed to load routing strategies: {e}")
            # 使用默认策略
            self.routing_strategies = {
                'default': {
                    'name': 'default', 
                    'strategy_type': self.settings.default_routing_strategy,
                    'config': {}
                }
            }
    
    async def _refresh_cache_if_needed(self):
        """根据需要刷新缓存"""
        now = datetime.now()
        
        # 检查模型缓存
        models_key = f"{self.settings.cache_prefix}models_last_update"
        if self.redis_client:
            try:
                last_update = await self.redis_client.get(models_key)
                if not last_update or (now - datetime.fromisoformat(last_update.decode())).seconds > self.cache_ttl_models:
                    await self._load_models()
                    await self.redis_client.set(models_key, now.isoformat(), ex=self.cache_ttl_models)
            except Exception:
                pass
        
        # 检查账号缓存
        accounts_key = f"{self.settings.cache_prefix}accounts_last_update"
        if self.redis_client:
            try:
                last_update = await self.redis_client.get(accounts_key)
                if not last_update or (now - datetime.fromisoformat(last_update.decode())).seconds > self.cache_ttl_accounts:
                    await self._load_accounts()
                    await self.redis_client.set(accounts_key, now.isoformat(), ex=self.cache_ttl_accounts)
            except Exception:
                pass
    
    async def _record_selection(self, model: ModelConfig, account: APIAccount, strategy: str):
        """记录选择结果"""
        if not self.redis_client:
            return
        
        try:
            # 记录选择统计
            selection_key = f"{self.settings.cache_prefix}selections:{model.id}:{account.id}"
            await self.redis_client.incr(selection_key, 1)
            await self.redis_client.expire(selection_key, 86400)  # 24小时过期
            
            # 记录策略使用统计
            strategy_key = f"{self.settings.cache_prefix}strategy_usage:{strategy}"
            await self.redis_client.incr(strategy_key, 1)
            await self.redis_client.expire(strategy_key, 86400)
            
        except Exception as e:
            self._logger.warning(f"Failed to record selection: {e}")
    
    async def get_router_statistics(self) -> Dict[str, Any]:
        """获取路由器统计信息"""
        stats = {
            'models_count': len(self.models_cache),
            'accounts_count': len(self.accounts_cache),
            'strategies_count': len(self.routing_strategies),
            'cache_status': 'connected' if self.redis_client else 'disconnected'
        }
        
        if self.redis_client:
            try:
                # 获取选择统计
                selection_keys = await self.redis_client.keys(f"{self.settings.cache_prefix}selections:*")
                stats['recent_selections'] = len(selection_keys)
                
                # 获取策略使用统计
                strategy_keys = await self.redis_client.keys(f"{self.settings.cache_prefix}strategy_usage:*")
                strategy_usage = {}
                for key in strategy_keys:
                    strategy_name = key.decode().split(':')[-1]
                    count = await self.redis_client.get(key)
                    strategy_usage[strategy_name] = int(count) if count else 0
                
                stats['strategy_usage'] = strategy_usage
                
            except Exception as e:
                self._logger.warning(f"Failed to get router statistics: {e}")
        
        return stats
    
    async def invalidate_cache(self):
        """清除缓存"""
        self.models_cache.clear()
        self.accounts_cache.clear()
        self.routing_strategies.clear()
        
        if self.redis_client:
            try:
                # 清除Redis中的相关键
                keys = await self.redis_client.keys(f"{self.settings.cache_prefix}*")
                if keys:
                    await self.redis_client.delete(*keys)
                    self._logger.info("Cache invalidated")
            except Exception as e:
                self._logger.warning(f"Failed to clear Redis cache: {e}")
        
        # 重新加载数据
        await self._load_models()
        await self._load_accounts()
        await self._load_routing_strategies()


# 全局路由器实例
_model_router_instance = None

async def get_model_router() -> ModelRouter:
    """获取模型路由器实例"""
    global _model_router_instance
    if _model_router_instance is None:
        _model_router_instance = ModelRouter()
        await _model_router_instance.initialize()
    return _model_router_instance