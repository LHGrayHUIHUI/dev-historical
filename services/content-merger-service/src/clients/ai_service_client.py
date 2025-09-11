"""
AI模型服务客户端

该模块提供与ai-model-service的HTTP通信接口，
支持文本生成、内容分析、创意写作等AI功能。
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime
import json

from ..config.settings import settings
from ..models.merger_models import MergeError, TokenUsage

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
                    "User-Agent": "content-merger-service/1.0.0"
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
            raise MergeError(f"AI service error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"AI service request error: {str(e)}")
            raise MergeError(f"Failed to connect to AI service: {str(e)}")
        except Exception as e:
            logger.error(f"AI service unexpected error: {str(e)}")
            raise MergeError(f"AI service request failed: {str(e)}")
    
    # 文本生成接口
    
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
                "model": model or settings.ai_model.default_model,
                "temperature": temperature or settings.ai_model.default_temperature,
                "max_tokens": max_tokens or settings.ai_model.default_max_tokens,
                **kwargs
            }
            
            response = await self._make_request("POST", "/api/v1/chat/completions", data)
            
            if response.get("success") and response.get("data"):
                return response["data"]
            
            raise MergeError("AI service returned invalid response")
            
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise MergeError(f"Text generation failed: {str(e)}")
    
    async def generate_content_merge(self, 
                                   source_contents: List[str],
                                   strategy: str,
                                   mode: str,
                                   instructions: Optional[str] = None,
                                   target_length: Optional[int] = None,
                                   target_style: Optional[str] = None) -> Dict[str, Any]:
        """生成内容合并"""
        try:
            # 构建合并提示语
            system_prompt = self._build_merge_system_prompt(strategy, mode)
            user_prompt = self._build_merge_user_prompt(
                source_contents, instructions, target_length, target_style
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 根据策略调整AI参数
            temperature = self._get_temperature_for_strategy(strategy, mode)
            max_tokens = min(4000, target_length * 2 if target_length else 4000)
            
            result = await self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "model_used": result.get("model", "unknown"),
                "token_usage": TokenUsage(**result.get("usage", {}))
            }
            
        except Exception as e:
            logger.error(f"Content merge generation failed: {str(e)}")
            raise MergeError(f"Failed to generate merged content: {str(e)}")
    
    async def generate_section_merge(self,
                                   section_contents: List[str],
                                   section_theme: str,
                                   merge_type: str = "chronological") -> str:
        """生成章节合并"""
        try:
            system_prompt = f"""你是一个专业的历史文献编辑专家，擅长将多个相关内容合并为连贯的章节。
请遵循以下原则：
1. 保持历史事实的准确性
2. 确保内容的逻辑连贯性  
3. 避免重复信息
4. 突出该章节的主题：{section_theme}
5. 使用{merge_type}的组织方式"""
            
            user_prompt = f"""请将以下关于"{section_theme}"的多个内容合并为一个连贯的章节：

"""
            
            for i, content in enumerate(section_contents, 1):
                user_prompt += f"内容{i}：\n{content}\n\n"
            
            user_prompt += """
请提供合并后的章节内容，要求：
1. 内容连贯流畅
2. 逻辑清晰
3. 信息完整
4. 符合学术写作规范"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=3000
            )
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Section merge generation failed: {str(e)}")
            raise MergeError(f"Failed to generate section merge: {str(e)}")
    
    async def generate_transitions(self, 
                                 from_section: str,
                                 to_section: str,
                                 transition_type: str = "logical") -> str:
        """生成章节过渡"""
        try:
            system_prompt = """你是一个专业的文本编辑专家，擅长创建流畅的章节过渡。
请生成简洁而自然的过渡语句，确保前后章节的逻辑连接。"""
            
            user_prompt = f"""请为以下两个章节生成{transition_type}过渡语句：

前一章节结尾：
{from_section[-200:]}

后一章节开头：
{to_section[:200]}

请提供1-2句自然的过渡语句。"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.4,
                max_tokens=200
            )
            
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Transition generation failed: {str(e)}")
            return ""  # 过渡生成失败不影响主流程
    
    async def generate_title(self, 
                           content: str,
                           style: str = "academic") -> str:
        """生成标题"""
        try:
            system_prompt = f"""你是一个专业的标题生成专家，擅长为{style}风格的历史文本创建合适的标题。
标题要求：
1. 准确概括内容主题
2. 符合{style}写作规范
3. 简洁明了
4. 具有吸引力"""
            
            user_prompt = f"""请为以下内容生成一个合适的标题：

{content[:1000]}

请只提供标题，不要其他说明。"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=100
            )
            
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Title generation failed: {str(e)}")
            return "合并内容"  # 默认标题
    
    async def analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """分析内容结构"""
        try:
            system_prompt = """你是一个专业的文本结构分析专家。
请分析给定文本的结构，包括：
1. 主要段落主题
2. 逻辑组织方式
3. 信息层次
4. 关键转折点

请以JSON格式返回分析结果。"""
            
            user_prompt = f"""请分析以下文本的结构：

{content}

返回格式：
{{
    "main_themes": ["主题1", "主题2"],
    "organization_type": "时间顺序/主题分类/逻辑论证",
    "key_sections": [
        {{"title": "章节标题", "start_pos": 0, "length": 100}}
    ],
    "transition_points": [50, 150, 300]
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
            
            # 尝试解析JSON响应
            response_text = result["choices"][0]["message"]["content"]
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse structure analysis JSON")
                return {
                    "main_themes": [],
                    "organization_type": "unknown",
                    "key_sections": [],
                    "transition_points": []
                }
            
        except Exception as e:
            logger.error(f"Content structure analysis failed: {str(e)}")
            return {}
    
    # 辅助方法
    
    def _build_merge_system_prompt(self, strategy: str, mode: str) -> str:
        """构建合并系统提示语"""
        base_prompt = settings.ai_model.system_prompt_prefix
        
        strategy_prompts = {
            "timeline": "擅长按时间顺序整合历史内容，确保时间逻辑的准确性",
            "topic": "擅长按主题归并相关内容，突出主题的完整性和深度",
            "hierarchy": "擅长按重要性层次组织内容，突出核心信息",
            "logic": "擅长构建内容间的逻辑关系，强调因果关系和推理过程",
            "supplement": "擅长用相关内容补充和丰富主要内容"
        }
        
        mode_prompts = {
            "comprehensive": "进行全面深入的内容合并，保留所有重要信息",
            "selective": "选择性合并最重要的内容，突出核心信息",
            "summary": "生成简洁的摘要式合并，提炼关键要点",
            "expansion": "在合并基础上适当扩展，增加深度和广度"
        }
        
        return f"""{base_prompt}，{strategy_prompts.get(strategy, '')}。
当前任务模式：{mode_prompts.get(mode, '')}。

请遵循以下原则：
1. 确保历史事实的准确性
2. 保持内容的逻辑连贯性
3. 避免重复和冗余信息
4. 维持适当的学术写作规范
5. 根据指定策略和模式进行合并"""
    
    def _build_merge_user_prompt(self, 
                               source_contents: List[str],
                               instructions: Optional[str],
                               target_length: Optional[int],
                               target_style: Optional[str]) -> str:
        """构建合并用户提示语"""
        prompt = "请将以下多个内容进行智能合并：\n\n"
        
        for i, content in enumerate(source_contents, 1):
            prompt += f"内容{i}：\n{content}\n\n"
        
        prompt += "合并要求：\n"
        
        if target_length:
            prompt += f"1. 目标长度：约{target_length}字\n"
        
        if target_style:
            prompt += f"2. 目标风格：{target_style}\n"
        
        if instructions:
            prompt += f"3. 特殊要求：{instructions}\n"
        
        prompt += """
请提供合并后的完整内容，确保：
- 信息完整准确
- 逻辑清晰连贯
- 语言流畅自然
- 符合要求的风格和长度"""
        
        return prompt
    
    def _get_temperature_for_strategy(self, strategy: str, mode: str) -> float:
        """根据策略获取合适的温度参数"""
        # 基础温度
        base_temp = settings.ai_model.default_temperature
        
        # 策略调整
        strategy_adjustments = {
            "timeline": -0.1,  # 时间线需要更准确
            "topic": 0.0,      # 主题合并平衡
            "hierarchy": -0.1, # 层次需要逻辑性
            "logic": -0.2,     # 逻辑关系最保守
            "supplement": 0.1  # 补充可以更创新
        }
        
        # 模式调整
        mode_adjustments = {
            "comprehensive": 0.0,   # 全面合并平衡
            "selective": -0.1,      # 选择性更保守
            "summary": -0.2,        # 摘要最保守
            "expansion": 0.2        # 扩展更创新
        }
        
        final_temp = base_temp + strategy_adjustments.get(strategy, 0) + mode_adjustments.get(mode, 0)
        return max(0.1, min(1.0, final_temp))  # 确保在有效范围内
    
    # 健康检查
    
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