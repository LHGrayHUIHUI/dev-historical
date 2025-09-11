"""
AI Model Service 客户端

与ai-model-service通信的HTTP客户端，提供AI模型调用能力，
支持文本分析、质量评估、建议生成等AI功能。
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime
import json

from ..config.settings import settings
from ..models.assessment_models import TokenUsage

logger = logging.getLogger(__name__)

class AIServiceClient:
    """AI模型服务HTTP客户端"""
    
    def __init__(self):
        self.base_url = settings.external_services.ai_model_service_url
        self.timeout = settings.external_services.ai_model_service_timeout
        self.retries = settings.external_services.ai_model_service_retries
        self._session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._close_session()
    
    async def _create_session(self):
        """创建HTTP会话"""
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "content-quality-assessment-service/1.0.0"
                }
            )
    
    async def _close_session(self):
        """关闭HTTP会话"""
        if self._session:
            await self._session.aclose()
            self._session = None
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, method: str, endpoint: str, 
                          data: Optional[Dict] = None,
                          params: Optional[Dict] = None) -> Dict[str, Any]:
        """发送HTTP请求"""
        if not self._session:
            await self._create_session()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == "GET":
                response = await self._session.get(url, params=params)
            elif method.upper() == "POST":
                response = await self._session.post(url, json=data, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"AI service request successful: {response.status_code}")
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"AI service HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"AI service request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"AI service unexpected error: {str(e)}")
            raise
    
    # ==================== 基础AI调用 ====================
    
    async def chat_completion(self, 
                            messages: List[Dict[str, str]],
                            model: Optional[str] = None,
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            **kwargs) -> Dict[str, Any]:
        """聊天完成接口"""
        try:
            data = {
                "messages": messages,
                "model": model or "gemini-1.5-flash",
                "temperature": temperature or 0.3,
                "max_tokens": max_tokens or 2000,
                **kwargs
            }
            
            response = await self._make_request("POST", "/api/v1/chat/completions", data)
            
            if response.get("success") and response.get("data"):
                return response["data"]
            
            raise Exception("AI service returned invalid response")
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise
    
    # ==================== 质量评估AI功能 ====================
    
    async def analyze_readability(self, content: str, content_type: str) -> Dict[str, Any]:
        """AI可读性分析"""
        try:
            system_prompt = """你是一个专业的文本可读性分析专家。
请分析给定文本的可读性，包括：
1. 句子结构复杂度
2. 词汇难度水平
3. 语言风格适宜性
4. 表达清晰度

请返回JSON格式的分析结果。"""
            
            user_prompt = f"""请分析以下{content_type}内容的可读性：

{content[:2000]}

返回格式：
{{
    "complexity_score": 75.0,
    "clarity_score": 85.0,
    "style_appropriateness": 80.0,
    "vocabulary_level": "中级",
    "sentence_structure_score": 70.0,
    "overall_readability": 77.5,
    "issues": ["句子过长", "专业术语较多"],
    "suggestions": ["简化长句", "增加术语解释"]
}}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
            
            return {
                "ai_analysis": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {str(e)}")
            raise
    
    async def analyze_accuracy(self, content: str, content_type: str) -> Dict[str, Any]:
        """AI准确性分析"""
        try:
            system_prompt = """你是一个专业的内容准确性分析专家，特别擅长历史文献的事实核查。
请分析给定文本的准确性，包括：
1. 事实一致性检查
2. 逻辑合理性验证
3. 引用和数据准确性
4. 专业术语使用正确性

请返回JSON格式的分析结果。"""
            
            user_prompt = f"""请分析以下{content_type}内容的准确性：

{content[:2000]}

返回格式：
{{
    "factual_consistency": 85.0,
    "logical_coherence": 80.0,
    "data_accuracy": 90.0,
    "terminology_correctness": 85.0,
    "overall_accuracy": 85.0,
    "potential_issues": ["某个日期可能有误", "引用格式不标准"],
    "fact_check_results": ["已验证的事实", "需要核实的内容"],
    "suggestions": ["核实具体日期", "标准化引用格式"]
}}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            return {
                "ai_analysis": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Accuracy analysis failed: {str(e)}")
            raise
    
    async def analyze_completeness(self, content: str, content_type: str) -> Dict[str, Any]:
        """AI完整性分析"""
        try:
            system_prompt = """你是一个专业的内容完整性分析专家。
请分析给定文本的完整性，包括：
1. 结构完整性（是否包含必要的章节和要素）
2. 信息完整性（关键信息是否缺失）
3. 逻辑完整性（论述是否完整闭合）
4. 背景信息充分性

请返回JSON格式的分析结果。"""
            
            user_prompt = f"""请分析以下{content_type}内容的完整性：

{content[:2000]}

返回格式：
{{
    "structure_completeness": 80.0,
    "information_completeness": 75.0,
    "logical_completeness": 85.0,
    "background_sufficiency": 70.0,
    "overall_completeness": 77.5,
    "missing_elements": ["缺少结论部分", "背景信息不足"],
    "incomplete_aspects": ["某些论点缺乏支撑", "时间线不够清晰"],
    "suggestions": ["补充结论段落", "增加背景介绍", "完善时间线描述"]
}}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
            
            return {
                "ai_analysis": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Completeness analysis failed: {str(e)}")
            raise
    
    async def analyze_coherence(self, content: str, content_type: str) -> Dict[str, Any]:
        """AI连贯性分析"""
        try:
            system_prompt = """你是一个专业的文本连贯性分析专家。
请分析给定文本的连贯性，包括：
1. 逻辑流程的连贯性
2. 段落间的过渡自然性
3. 论点的一致性
4. 叙述的流畅性

请返回JSON格式的分析结果。"""
            
            user_prompt = f"""请分析以下{content_type}内容的连贯性：

{content[:2000]}

返回格式：
{{
    "logical_flow": 85.0,
    "transition_quality": 80.0,
    "argument_consistency": 90.0,
    "narrative_smoothness": 75.0,
    "overall_coherence": 82.5,
    "coherence_issues": ["段落跳跃过大", "缺少过渡句"],
    "inconsistencies": ["前后观点略有矛盾"],
    "suggestions": ["增加过渡段落", "统一论述立场", "优化段落顺序"]
}}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
            
            return {
                "ai_analysis": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Coherence analysis failed: {str(e)}")
            raise
    
    async def analyze_relevance(self, content: str, content_type: str, 
                              target_audience: Optional[str] = None) -> Dict[str, Any]:
        """AI相关性分析"""
        try:
            audience_info = f"目标受众：{target_audience}" if target_audience else "通用受众"
            
            system_prompt = f"""你是一个专业的内容相关性分析专家。
请分析给定文本的相关性，包括：
1. 主题相关性（内容是否切题）
2. 受众相关性（是否适合目标受众）
3. 时代相关性（是否具有时代意义）
4. 实用相关性（是否有实际价值）

当前分析上下文：{audience_info}

请返回JSON格式的分析结果。"""
            
            user_prompt = f"""请分析以下{content_type}内容的相关性：

{content[:2000]}

返回格式：
{{
    "topic_relevance": 90.0,
    "audience_relevance": 85.0,
    "temporal_relevance": 80.0,
    "practical_relevance": 75.0,
    "overall_relevance": 82.5,
    "relevance_strengths": ["主题明确", "适合目标读者"],
    "relevance_gaps": ["时代背景说明不足", "实用性有限"],
    "suggestions": ["增加现代意义阐述", "提供实际应用场景", "优化受众针对性"]
}}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            
            return {
                "ai_analysis": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Relevance analysis failed: {str(e)}")
            raise
    
    # ==================== 综合分析和建议生成 ====================
    
    async def generate_quality_summary(self, 
                                     metrics_data: Dict[str, Any],
                                     content_type: str) -> Dict[str, Any]:
        """生成质量评估总结"""
        try:
            system_prompt = """你是一个专业的内容质量评估专家。
基于提供的多维度评估数据，生成一份简洁而全面的质量评估总结报告。
包括优势亮点、主要问题、改进建议和整体评价。"""
            
            user_prompt = f"""基于以下{content_type}的质量评估数据，生成评估总结：

评估数据：
{json.dumps(metrics_data, ensure_ascii=False, indent=2)}

请生成包含以下内容的总结：
1. 整体质量评价（2-3句话）
2. 主要优势（3-5个要点）
3. 需要改进的方面（3-5个要点）
4. 具体改进建议（5-8个建议）
5. 质量提升重点（按优先级排序）

返回格式：
{{
    "overall_assessment": "整体评价文字",
    "strengths": ["优势1", "优势2", "优势3"],
    "weaknesses": ["问题1", "问题2", "问题3"],
    "recommendations": ["建议1", "建议2", "建议3"],
    "improvement_priorities": ["优先级1", "优先级2", "优先级3"]
}}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.4,
                max_tokens=2000
            )
            
            return {
                "summary": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Quality summary generation failed: {str(e)}")
            raise
    
    async def generate_improvement_plan(self,
                                      assessment_result: Dict[str, Any],
                                      target_grade: str = "A") -> Dict[str, Any]:
        """生成质量改进计划"""
        try:
            system_prompt = f"""你是一个专业的内容质量改进顾问。
基于当前的质量评估结果，制定一个实用的改进计划，目标是达到{target_grade}级质量标准。
计划应该具体、可操作、有优先级。"""
            
            user_prompt = f"""基于以下质量评估结果，制定改进计划（目标等级：{target_grade}）：

当前评估结果：
{json.dumps(assessment_result, ensure_ascii=False, indent=2)}

请制定包含以下内容的改进计划：
1. 改进目标和预期效果
2. 具体改进步骤（按优先级排序）
3. 每个步骤的预估工作量和时间
4. 改进效果的衡量指标
5. 风险点和注意事项

返回格式：
{{
    "improvement_goal": "改进目标描述",
    "expected_score_increase": 15.5,
    "action_steps": [
        {{
            "step": "步骤1",
            "priority": "高",
            "effort": "2小时",
            "expected_impact": "提升5分"
        }}
    ],
    "success_metrics": ["指标1", "指标2"],
    "risks_and_notes": ["风险1", "注意事项1"]
}}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=2500
            )
            
            return {
                "improvement_plan": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Improvement plan generation failed: {str(e)}")
            raise
    
    # ==================== 健康检查 ====================
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            response = await self._make_request("GET", "/health")
            return response.get("status") == "healthy"
            
        except Exception as e:
            logger.error(f"AI service health check failed: {str(e)}")
            return False

# 全局客户端实例
ai_service_client = AIServiceClient()