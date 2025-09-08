"""
OCR引擎工厂类

提供OCR引擎的统一创建和管理接口，支持动态引擎选择、
配置管理、引擎池等高级功能。

主要功能：
- 统一的引擎创建接口
- 动态引擎选择和切换
- 引擎配置验证和管理
- 引擎健康检查
- 引擎性能监控

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from enum import Enum
import time
from dataclasses import dataclass

from .base_engine import BaseOCREngine
from .paddleocr_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine
from .easyocr_engine import EasyOCREngine

# 配置日志
logger = logging.getLogger(__name__)


class OCREngineType(str, Enum):
    """OCR引擎类型枚举"""
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"


@dataclass
class EnginePerformanceMetrics:
    """引擎性能指标"""
    engine_name: str                    # 引擎名称
    total_recognitions: int = 0         # 总识别次数
    total_processing_time: float = 0.0  # 总处理时间
    success_count: int = 0              # 成功次数
    failure_count: int = 0              # 失败次数
    avg_processing_time: float = 0.0    # 平均处理时间
    avg_confidence: float = 0.0         # 平均置信度
    last_used: Optional[float] = None   # 最后使用时间
    
    def update_metrics(
        self, 
        processing_time: float, 
        success: bool, 
        confidence: float = 0.0
    ):
        """
        更新性能指标
        
        Args:
            processing_time: 处理时间
            success: 是否成功
            confidence: 置信度
        """
        self.total_recognitions += 1
        self.total_processing_time += processing_time
        self.last_used = time.time()
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # 计算平均值
        self.avg_processing_time = self.total_processing_time / self.total_recognitions
        
        if success and confidence > 0:
            # 更新平均置信度（只考虑成功的识别）
            total_confidence = self.avg_confidence * (self.success_count - 1) + confidence
            self.avg_confidence = total_confidence / self.success_count
    
    @property
    def success_rate(self) -> float:
        """获取成功率"""
        if self.total_recognitions == 0:
            return 0.0
        return self.success_count / self.total_recognitions
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'engine_name': self.engine_name,
            'total_recognitions': self.total_recognitions,
            'total_processing_time': self.total_processing_time,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'avg_processing_time': self.avg_processing_time,
            'avg_confidence': self.avg_confidence,
            'success_rate': self.success_rate,
            'last_used': self.last_used
        }


class OCREngineFactory:
    """
    OCR引擎工厂类
    
    提供OCR引擎的创建、管理和监控功能。支持多种引擎类型
    的动态创建和配置，以及引擎性能监控。
    """
    
    # 引擎类映射
    ENGINE_CLASSES: Dict[OCREngineType, Type[BaseOCREngine]] = {
        OCREngineType.PADDLEOCR: PaddleOCREngine,
        OCREngineType.TESSERACT: TesseractEngine,
        OCREngineType.EASYOCR: EasyOCREngine,
    }
    
    def __init__(self):
        """初始化引擎工厂"""
        self._engine_instances: Dict[str, BaseOCREngine] = {}
        self._engine_configs: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: Dict[str, EnginePerformanceMetrics] = {}
        self._default_engine_type = OCREngineType.PADDLEOCR
        self._lock = asyncio.Lock()
        
        logger.info("OCR引擎工厂初始化完成")
    
    async def create_engine(
        self,
        engine_type: Union[OCREngineType, str],
        config: Optional[Dict[str, Any]] = None,
        instance_name: Optional[str] = None
    ) -> BaseOCREngine:
        """
        创建OCR引擎实例
        
        Args:
            engine_type: 引擎类型
            config: 引擎配置
            instance_name: 实例名称（用于缓存）
            
        Returns:
            OCR引擎实例
        """
        # 类型转换
        if isinstance(engine_type, str):
            try:
                engine_type = OCREngineType(engine_type.lower())
            except ValueError:
                raise ValueError(f"不支持的引擎类型: {engine_type}")
        
        # 生成实例名称
        if instance_name is None:
            instance_name = f"{engine_type.value}_{int(time.time())}"
        
        async with self._lock:
            try:
                # 检查是否已存在实例
                if instance_name in self._engine_instances:
                    logger.info(f"返回已存在的引擎实例: {instance_name}")
                    return self._engine_instances[instance_name]
                
                # 获取引擎类
                engine_class = self.ENGINE_CLASSES.get(engine_type)
                if not engine_class:
                    raise ValueError(f"未找到引擎类: {engine_type}")
                
                # 创建引擎实例
                logger.info(f"创建新的引擎实例: {instance_name} ({engine_type.value})")
                engine_instance = engine_class(config)
                
                # 初始化引擎
                if not await engine_instance.initialize():
                    raise RuntimeError(f"引擎初始化失败: {instance_name}")
                
                # 缓存实例和配置
                self._engine_instances[instance_name] = engine_instance
                self._engine_configs[instance_name] = config or {}
                
                # 初始化性能指标
                self._performance_metrics[instance_name] = EnginePerformanceMetrics(
                    engine_name=engine_type.value
                )
                
                logger.info(f"引擎实例创建成功: {instance_name}")
                return engine_instance
                
            except Exception as e:
                logger.error(f"创建引擎实例失败: {str(e)}")
                raise
    
    async def get_engine(
        self, 
        instance_name: str
    ) -> Optional[BaseOCREngine]:
        """
        获取已创建的引擎实例
        
        Args:
            instance_name: 实例名称
            
        Returns:
            引擎实例或None
        """
        async with self._lock:
            return self._engine_instances.get(instance_name)
    
    async def get_or_create_engine(
        self,
        engine_type: Union[OCREngineType, str],
        config: Optional[Dict[str, Any]] = None,
        instance_name: Optional[str] = None
    ) -> BaseOCREngine:
        """
        获取或创建引擎实例
        
        Args:
            engine_type: 引擎类型
            config: 引擎配置
            instance_name: 实例名称
            
        Returns:
            引擎实例
        """
        if instance_name and instance_name in self._engine_instances:
            return self._engine_instances[instance_name]
        
        return await self.create_engine(engine_type, config, instance_name)
    
    async def get_default_engine(self) -> BaseOCREngine:
        """
        获取默认引擎实例
        
        Returns:
            默认引擎实例
        """
        default_name = f"default_{self._default_engine_type.value}"
        return await self.get_or_create_engine(
            self._default_engine_type,
            instance_name=default_name
        )
    
    def set_default_engine_type(self, engine_type: Union[OCREngineType, str]):
        """
        设置默认引擎类型
        
        Args:
            engine_type: 引擎类型
        """
        if isinstance(engine_type, str):
            engine_type = OCREngineType(engine_type.lower())
        
        self._default_engine_type = engine_type
        logger.info(f"默认引擎类型设置为: {engine_type.value}")
    
    async def remove_engine(self, instance_name: str) -> bool:
        """
        移除引擎实例
        
        Args:
            instance_name: 实例名称
            
        Returns:
            是否成功移除
        """
        async with self._lock:
            try:
                if instance_name in self._engine_instances:
                    del self._engine_instances[instance_name]
                    del self._engine_configs[instance_name]
                    if instance_name in self._performance_metrics:
                        del self._performance_metrics[instance_name]
                    
                    logger.info(f"引擎实例已移除: {instance_name}")
                    return True
                else:
                    logger.warning(f"引擎实例不存在: {instance_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"移除引擎实例失败: {str(e)}")
                return False
    
    async def list_engines(self) -> List[Dict[str, Any]]:
        """
        列出所有引擎实例
        
        Returns:
            引擎实例信息列表
        """
        async with self._lock:
            engine_list = []
            
            for instance_name, engine in self._engine_instances.items():
                engine_info = {
                    'instance_name': instance_name,
                    'engine_type': engine.engine_name,
                    'engine_version': engine.engine_version,
                    'initialized': engine.is_initialized,
                    'config': self._engine_configs.get(instance_name, {})
                }
                
                # 添加性能指标
                if instance_name in self._performance_metrics:
                    engine_info['performance'] = self._performance_metrics[instance_name].to_dict()
                
                engine_list.append(engine_info)
            
            return engine_list
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        对所有引擎进行健康检查
        
        Returns:
            健康检查结果字典
        """
        health_results = {}
        
        async with self._lock:
            # 创建健康检查任务
            tasks = []
            instance_names = []
            
            for instance_name, engine in self._engine_instances.items():
                tasks.append(engine.health_check())
                instance_names.append(instance_name)
            
            # 并发执行健康检查
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for instance_name, result in zip(instance_names, results):
                    if isinstance(result, Exception):
                        health_results[instance_name] = False
                        logger.error(f"引擎健康检查异常 {instance_name}: {str(result)}")
                    else:
                        health_results[instance_name] = bool(result)
        
        return health_results
    
    def update_performance_metrics(
        self,
        instance_name: str,
        processing_time: float,
        success: bool,
        confidence: float = 0.0
    ):
        """
        更新引擎性能指标
        
        Args:
            instance_name: 实例名称
            processing_time: 处理时间
            success: 是否成功
            confidence: 置信度
        """
        if instance_name in self._performance_metrics:
            self._performance_metrics[instance_name].update_metrics(
                processing_time, success, confidence
            )
    
    def get_performance_metrics(
        self, 
        instance_name: str
    ) -> Optional[EnginePerformanceMetrics]:
        """
        获取引擎性能指标
        
        Args:
            instance_name: 实例名称
            
        Returns:
            性能指标或None
        """
        return self._performance_metrics.get(instance_name)
    
    def get_all_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有引擎的性能指标
        
        Returns:
            性能指标字典
        """
        return {
            name: metrics.to_dict() 
            for name, metrics in self._performance_metrics.items()
        }
    
    async def select_best_engine(
        self,
        criteria: str = "success_rate"
    ) -> Optional[Tuple[str, BaseOCREngine]]:
        """
        根据性能指标选择最佳引擎
        
        Args:
            criteria: 选择标准 (success_rate, avg_processing_time, avg_confidence)
            
        Returns:
            最佳引擎的名称和实例
        """
        async with self._lock:
            if not self._performance_metrics:
                return None
            
            best_instance_name = None
            best_score = None
            
            for instance_name, metrics in self._performance_metrics.items():
                if metrics.total_recognitions == 0:
                    continue  # 跳过未使用的引擎
                
                if criteria == "success_rate":
                    score = metrics.success_rate
                    if best_score is None or score > best_score:
                        best_score = score
                        best_instance_name = instance_name
                        
                elif criteria == "avg_processing_time":
                    score = metrics.avg_processing_time
                    if best_score is None or score < best_score:  # 时间越短越好
                        best_score = score
                        best_instance_name = instance_name
                        
                elif criteria == "avg_confidence":
                    score = metrics.avg_confidence
                    if best_score is None or score > best_score:
                        best_score = score
                        best_instance_name = instance_name
            
            if best_instance_name:
                return best_instance_name, self._engine_instances[best_instance_name]
            
            return None
    
    @classmethod
    def get_supported_engines(cls) -> List[str]:
        """
        获取支持的引擎类型列表
        
        Returns:
            支持的引擎类型列表
        """
        return [engine_type.value for engine_type in cls.ENGINE_CLASSES.keys()]
    
    @classmethod
    def validate_engine_type(cls, engine_type: str) -> bool:
        """
        验证引擎类型是否支持
        
        Args:
            engine_type: 引擎类型字符串
            
        Returns:
            是否支持
        """
        return engine_type.lower() in [e.value for e in cls.ENGINE_CLASSES.keys()]
    
    async def cleanup(self):
        """清理所有引擎实例和资源"""
        async with self._lock:
            try:
                # 清理所有引擎实例
                for instance_name in list(self._engine_instances.keys()):
                    try:
                        del self._engine_instances[instance_name]
                    except:
                        pass
                
                # 清理所有数据结构
                self._engine_instances.clear()
                self._engine_configs.clear()
                self._performance_metrics.clear()
                
                logger.info("OCR引擎工厂清理完成")
                
            except Exception as e:
                logger.error(f"引擎工厂清理失败: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        try:
            # 尝试清理资源（可能在事件循环已关闭时失败）
            asyncio.create_task(self.cleanup())
        except:
            pass


# 全局引擎工厂实例
_global_factory: Optional[OCREngineFactory] = None


def get_engine_factory() -> OCREngineFactory:
    """
    获取全局引擎工厂实例
    
    Returns:
        引擎工厂实例
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = OCREngineFactory()
    return _global_factory


async def create_engine_from_config(
    engine_config: Dict[str, Any]
) -> BaseOCREngine:
    """
    根据配置创建引擎实例
    
    Args:
        engine_config: 包含engine_type和其他配置的字典
        
    Returns:
        引擎实例
    """
    factory = get_engine_factory()
    
    engine_type = engine_config.get('engine_type', 'paddleocr')
    instance_name = engine_config.get('instance_name')
    config = {k: v for k, v in engine_config.items() if k not in ['engine_type', 'instance_name']}
    
    return await factory.create_engine(engine_type, config, instance_name)