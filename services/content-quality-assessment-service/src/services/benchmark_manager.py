"""
质量基准管理器

管理质量评估基准，包括基准创建、更新、对比分析等功能。
支持不同内容类型的专业化基准设置和合规性检查。
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import json

from ..config.settings import settings
from ..models.assessment_models import (
    QualityBenchmark, BenchmarkComparison, QualityAssessmentResult,
    QualityDimension, ContentType
)
from ..clients.storage_client import storage_client

logger = logging.getLogger(__name__)

class QualityBenchmarkManager:
    """质量基准管理器"""
    
    def __init__(self):
        self.default_benchmarks = settings.benchmark.default_benchmarks
        self.max_custom_benchmarks = settings.benchmark.max_custom_benchmarks
        self.validation_enabled = settings.benchmark.benchmark_validation_enabled
        
        # 基准权重配置
        self.dimension_priorities = {
            ContentType.HISTORICAL_DOCUMENT: {
                QualityDimension.ACCURACY: 0.3,
                QualityDimension.COMPLETENESS: 0.25,
                QualityDimension.COHERENCE: 0.2,
                QualityDimension.RELEVANCE: 0.15,
                QualityDimension.READABILITY: 0.1
            },
            ContentType.ACADEMIC_PAPER: {
                QualityDimension.ACCURACY: 0.35,
                QualityDimension.COHERENCE: 0.25,
                QualityDimension.COMPLETENESS: 0.2,
                QualityDimension.RELEVANCE: 0.15,
                QualityDimension.READABILITY: 0.05
            },
            ContentType.EDUCATIONAL_CONTENT: {
                QualityDimension.READABILITY: 0.3,
                QualityDimension.ACCURACY: 0.25,
                QualityDimension.RELEVANCE: 0.2,
                QualityDimension.COMPLETENESS: 0.15,
                QualityDimension.COHERENCE: 0.1
            }
        }
    
    async def create_benchmark(self, benchmark: QualityBenchmark) -> str:
        """创建质量基准"""
        try:
            # 验证基准数据
            if self.validation_enabled:
                await self._validate_benchmark(benchmark)
            
            # 设置创建时间
            benchmark.created_at = datetime.now()
            benchmark.updated_at = datetime.now()
            benchmark.usage_count = 0
            
            # 通过storage-service存储
            benchmark_data = benchmark.dict()
            response = await storage_client.create_quality_benchmark(benchmark_data)
            
            if response.get('success'):
                logger.info(f"Created quality benchmark: {benchmark.benchmark_id}")
                return benchmark.benchmark_id
            else:
                raise Exception(f"Failed to create benchmark: {response.get('message', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Failed to create quality benchmark: {str(e)}")
            raise
    
    async def get_benchmark(self, benchmark_id: str) -> Optional[QualityBenchmark]:
        """获取质量基准"""
        try:
            response = await storage_client.get_quality_benchmark(benchmark_id)
            
            if response.get('success') and response.get('data'):
                benchmark_data = response['data']
                return QualityBenchmark(**benchmark_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get quality benchmark {benchmark_id}: {str(e)}")
            return None
    
    async def update_benchmark(self, 
                             benchmark_id: str, 
                             update_data: Dict[str, Any]) -> bool:
        """更新质量基准"""
        try:
            # 获取现有基准
            existing_benchmark = await self.get_benchmark(benchmark_id)
            if not existing_benchmark:
                raise ValueError(f"Benchmark not found: {benchmark_id}")
            
            # 更新时间
            update_data['updated_at'] = datetime.now()
            
            # 验证更新数据
            if self.validation_enabled and 'dimension_standards' in update_data:
                await self._validate_dimension_standards(update_data['dimension_standards'])
            
            # 执行更新
            response = await storage_client.update_quality_benchmark(benchmark_id, update_data)
            
            if response.get('success'):
                logger.info(f"Updated quality benchmark: {benchmark_id}")
                return True
            else:
                raise Exception(f"Failed to update benchmark: {response.get('message', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Failed to update quality benchmark {benchmark_id}: {str(e)}")
            return False
    
    async def list_benchmarks(self,
                            content_type: Optional[ContentType] = None,
                            is_active: Optional[bool] = None,
                            limit: int = 50,
                            offset: int = 0) -> List[QualityBenchmark]:
        """列出质量基准"""
        try:
            content_type_str = content_type.value if content_type else None
            
            response = await storage_client.list_quality_benchmarks(
                content_type=content_type_str,
                is_active=is_active,
                limit=limit,
                offset=offset
            )
            
            if response.get('success') and response.get('data'):
                benchmarks = []
                for benchmark_data in response['data']:
                    try:
                        benchmark = QualityBenchmark(**benchmark_data)
                        benchmarks.append(benchmark)
                    except Exception as e:
                        logger.warning(f"Failed to parse benchmark data: {str(e)}")
                
                logger.debug(f"Retrieved {len(benchmarks)} quality benchmarks")
                return benchmarks
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to list quality benchmarks: {str(e)}")
            return []
    
    async def delete_benchmark(self, benchmark_id: str) -> bool:
        """删除质量基准"""
        try:
            response = await storage_client.delete_quality_benchmark(benchmark_id)
            
            if response.get('success'):
                logger.info(f"Deleted quality benchmark: {benchmark_id}")
                return True
            else:
                raise Exception(f"Failed to delete benchmark: {response.get('message', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Failed to delete quality benchmark {benchmark_id}: {str(e)}")
            return False
    
    async def compare_with_benchmark(self, 
                                   assessment_result: QualityAssessmentResult,
                                   benchmark_id: str) -> BenchmarkComparison:
        """与基准对比"""
        try:
            # 获取基准
            benchmark = await self.get_benchmark(benchmark_id)
            if not benchmark:
                raise ValueError(f"Benchmark not found: {benchmark_id}")
            
            # 更新基准使用统计
            await self._update_benchmark_usage(benchmark_id)
            
            # 计算各维度差异
            comparison_details = {}
            improvement_gaps = []
            priority_improvements = []
            dimension_compliance = []
            
            for metric in assessment_result.metrics:
                dimension = metric.dimension
                if dimension in benchmark.dimension_standards:
                    standard = benchmark.dimension_standards[dimension]
                    actual = metric.score
                    gap = actual - standard
                    
                    comparison_details[dimension] = {
                        "standard": standard,
                        "actual": actual,
                        "gap": gap,
                        "meets_standard": actual >= standard,
                        "performance_ratio": actual / standard if standard > 0 else 1.0
                    }
                    
                    if actual < standard:
                        gap_description = (
                            f"{dimension.value}得分{actual:.1f}，低于基准{standard:.1f}，"
                            f"需提升{standard - actual:.1f}分"
                        )
                        improvement_gaps.append(gap_description)
                        
                        # 根据维度优先级确定改进优先级
                        content_type = ContentType(assessment_result.content_type)
                        dimension_priority = self.dimension_priorities.get(content_type, {})
                        priority_weight = dimension_priority.get(dimension, 0.1)
                        
                        if priority_weight > 0.2:  # 高优先级维度
                            priority_improvements.append(gap_description)
                    
                    # 计算合规度
                    compliance_ratio = min(1.0, actual / standard if standard > 0 else 1.0)
                    dimension_compliance.append(compliance_ratio)
            
            # 整体合规性检查
            meets_overall = assessment_result.overall_score >= benchmark.overall_standard
            meets_dimensions = all(
                detail["meets_standard"] for detail in comparison_details.values()
            )
            meets_standard = meets_overall and meets_dimensions
            
            # 计算合规度得分
            if dimension_compliance:
                compliance_score = np.mean(dimension_compliance) * 100
            else:
                compliance_score = 0.0
            
            # 估算改进工作量
            estimated_effort = self._estimate_improvement_effort(comparison_details)
            
            # 构建对比结果
            comparison = BenchmarkComparison(
                content_id=assessment_result.content_id,
                benchmark_id=benchmark_id,
                assessment_result=assessment_result,
                comparison_details=comparison_details,
                meets_standard=meets_standard,
                compliance_score=compliance_score,
                improvement_gaps=improvement_gaps,
                priority_improvements=priority_improvements,
                estimated_effort=estimated_effort,
                comparison_time=datetime.now(),
                benchmark_version="1.0"
            )
            
            # 保存对比结果
            try:
                await storage_client.save_benchmark_comparison(comparison)
            except Exception as e:
                logger.warning(f"Failed to save benchmark comparison: {str(e)}")
            
            logger.info(f"Benchmark comparison completed for content {assessment_result.content_id}")
            return comparison
            
        except Exception as e:
            logger.error(f"Benchmark comparison failed: {str(e)}")
            raise
    
    async def get_default_benchmark(self, content_type: ContentType) -> Optional[QualityBenchmark]:
        """获取默认基准"""
        try:
            # 查找该内容类型的默认基准
            benchmarks = await self.list_benchmarks(
                content_type=content_type,
                is_active=True
            )
            
            # 优先返回标记为默认的基准
            for benchmark in benchmarks:
                if benchmark.is_default:
                    return benchmark
            
            # 如果没有默认基准，创建一个
            if content_type.value in self.default_benchmarks:
                default_config = self.default_benchmarks[content_type.value]
                
                # 构建维度标准
                dimension_standards = {}
                for dim_name, standard in default_config.items():
                    if dim_name != "overall":
                        try:
                            dimension = QualityDimension(dim_name)
                            dimension_standards[dimension] = standard
                        except ValueError:
                            continue
                
                # 创建默认基准
                default_benchmark = QualityBenchmark(
                    name=f"默认{content_type.value}基准",
                    description=f"系统默认的{content_type.value}质量基准",
                    content_type=content_type,
                    target_audience="通用",
                    dimension_standards=dimension_standards,
                    overall_standard=default_config.get("overall", 80.0),
                    is_default=True,
                    created_by="system"
                )
                
                benchmark_id = await self.create_benchmark(default_benchmark)
                default_benchmark.benchmark_id = benchmark_id
                
                return default_benchmark
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get default benchmark for {content_type}: {str(e)}")
            return None
    
    async def analyze_benchmark_performance(self, 
                                          benchmark_id: str,
                                          period_days: int = 30) -> Dict[str, Any]:
        """分析基准性能"""
        try:
            # 获取基准信息
            benchmark = await self.get_benchmark(benchmark_id)
            if not benchmark:
                raise ValueError(f"Benchmark not found: {benchmark_id}")
            
            # 这里应该通过storage service获取使用该基准的评估统计数据
            # 暂时返回基础分析
            analysis = {
                "benchmark_id": benchmark_id,
                "benchmark_name": benchmark.name,
                "content_type": benchmark.content_type,
                "usage_count": benchmark.usage_count,
                "last_used": benchmark.last_used,
                "analysis_period_days": period_days,
                "generated_at": datetime.now()
            }
            
            # 基准有效性分析
            analysis["effectiveness"] = self._analyze_benchmark_effectiveness(benchmark)
            
            # 建议更新
            analysis["recommendations"] = self._generate_benchmark_recommendations(benchmark)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze benchmark performance: {str(e)}")
            return {}
    
    # ==================== 验证和辅助方法 ====================
    
    async def _validate_benchmark(self, benchmark: QualityBenchmark):
        """验证基准数据"""
        # 检查维度标准
        await self._validate_dimension_standards(benchmark.dimension_standards)
        
        # 检查整体标准
        if not 0 <= benchmark.overall_standard <= 100:
            raise ValueError(f"整体标准{benchmark.overall_standard}必须在0-100范围内")
        
        # 检查名称唯一性
        existing_benchmarks = await self.list_benchmarks(
            content_type=benchmark.content_type,
            is_active=True
        )
        
        for existing in existing_benchmarks:
            if existing.name == benchmark.name and existing.benchmark_id != benchmark.benchmark_id:
                raise ValueError(f"基准名称'{benchmark.name}'已存在")
    
    async def _validate_dimension_standards(self, dimension_standards: Dict[QualityDimension, float]):
        """验证维度标准"""
        for dimension, standard in dimension_standards.items():
            if not 0 <= standard <= 100:
                raise ValueError(f"维度{dimension.value}的标准{standard}必须在0-100范围内")
        
        # 检查维度合理性
        if len(dimension_standards) == 0:
            raise ValueError("至少需要设置一个维度标准")
    
    async def _update_benchmark_usage(self, benchmark_id: str):
        """更新基准使用统计"""
        try:
            update_data = {
                "last_used": datetime.now(),
                "usage_count": {"$inc": 1}  # 假设storage service支持递增操作
            }
            
            await storage_client.update_quality_benchmark(benchmark_id, update_data)
            
        except Exception as e:
            logger.warning(f"Failed to update benchmark usage: {str(e)}")
    
    def _estimate_improvement_effort(self, comparison_details: Dict) -> str:
        """估算改进工作量"""
        try:
            total_gap = 0
            gap_count = 0
            
            for detail in comparison_details.values():
                if detail["gap"] < 0:  # 低于标准
                    total_gap += abs(detail["gap"])
                    gap_count += 1
            
            if gap_count == 0:
                return "无需改进"
            
            avg_gap = total_gap / gap_count
            
            if avg_gap > 20:
                return "大量工作"
            elif avg_gap > 10:
                return "适中工作"
            elif avg_gap > 5:
                return "少量工作"
            else:
                return "微调即可"
                
        except Exception as e:
            logger.error(f"Failed to estimate improvement effort: {str(e)}")
            return "无法估算"
    
    def _analyze_benchmark_effectiveness(self, benchmark: QualityBenchmark) -> Dict[str, Any]:
        """分析基准有效性"""
        effectiveness = {
            "overall_rating": "良好",  # 简化分析
            "standard_appropriateness": "适中",
            "usage_frequency": "正常" if benchmark.usage_count > 0 else "低",
            "last_review": benchmark.updated_at,
            "needs_update": (datetime.now() - benchmark.updated_at).days > 90
        }
        
        # 标准合理性分析
        standards = list(benchmark.dimension_standards.values())
        if standards:
            avg_standard = np.mean(standards)
            std_standard = np.std(standards)
            
            if avg_standard > 90:
                effectiveness["standard_appropriateness"] = "过高"
            elif avg_standard < 60:
                effectiveness["standard_appropriateness"] = "过低"
            
            if std_standard > 15:
                effectiveness["consistency"] = "标准差异较大"
            else:
                effectiveness["consistency"] = "标准一致性好"
        
        return effectiveness
    
    def _generate_benchmark_recommendations(self, benchmark: QualityBenchmark) -> List[str]:
        """生成基准建议"""
        recommendations = []
        
        # 基于使用情况的建议
        if benchmark.usage_count == 0:
            recommendations.append("该基准尚未被使用，建议推广使用或考虑删除")
        elif benchmark.usage_count > 100:
            recommendations.append("该基准使用频繁，建议定期审查和优化")
        
        # 基于更新时间的建议
        days_since_update = (datetime.now() - benchmark.updated_at).days
        if days_since_update > 90:
            recommendations.append("基准超过3个月未更新，建议审查是否需要调整")
        
        # 基于标准设置的建议
        standards = list(benchmark.dimension_standards.values())
        if standards:
            avg_standard = np.mean(standards)
            if avg_standard > 85:
                recommendations.append("基准标准偏高，可能导致大部分内容难以达标")
            elif avg_standard < 65:
                recommendations.append("基准标准偏低，可能无法有效筛选质量问题")
        
        return recommendations

# 全局基准管理器实例
benchmark_manager = QualityBenchmarkManager()