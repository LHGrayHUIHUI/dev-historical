"""
AI模型服务客户端 - AI Model Service Client

负责与AI模型服务通信，调用各种AI模型执行文本优化任务
支持多模型路由、负载均衡和容错处理

核心功能:
1. 文本优化AI调用
2. 模型选择和路由
3. 响应解析和处理
4. 错误处理和重试
5. 性能监控和统计
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import httpx

from ..config.settings import get_settings
from ..models.optimization_models import OptimizationType, OptimizationMode


logger = logging.getLogger(__name__)


class AIModelServiceError(Exception):
    """AI模型服务客户端错误"""
    pass


class AIModelServiceClient:
    """
    AI模型服务HTTP客户端
    
    提供与AI模型服务通信的接口，用于执行文本优化任务
    支持异步操作、模型路由和容错处理
    """
    
    def __init__(self):
        """初始化AI模型服务客户端"""
        self.settings = get_settings()
        self.base_url = self.settings.ai_model_service_url.rstrip('/')
        self.timeout = self.settings.ai_model_service_timeout
        self.max_retries = 3
        
        # HTTP客户端配置
        self.client_config = {
            "timeout": httpx.Timeout(self.timeout),
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            params: 查询参数
            retries: 重试次数
            
        Returns:
            响应数据
            
        Raises:
            AIModelServiceError: 请求失败时抛出
        """
        if retries is None:
            retries = self.max_retries
        
        url = f"{self.base_url}{endpoint}"
        self._logger.debug(f"Making {method} request to {url}")
        
        for attempt in range(retries + 1):
            try:
                timeout = httpx.Timeout(self.timeout)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code in [200, 201]:
                        return response.json()
                    elif response.status_code == 404:
                        raise AIModelServiceError(f"资源未找到: {endpoint}")
                    elif response.status_code == 422:
                        error_data = response.json()
                        raise AIModelServiceError(f"请求验证失败: {error_data}")
                    else:
                        response.raise_for_status()
                        
            except httpx.ConnectError as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"AI模型服务连接失败，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise AIModelServiceError(f"无法连接到AI模型服务: {e}")
            except httpx.TimeoutException as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"AI模型服务请求超时，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise AIModelServiceError(f"AI模型服务请求超时: {e}")
            except Exception as e:
                if attempt < retries:
                    wait_time = 2 ** attempt
                    self._logger.warning(f"AI模型服务请求失败，{wait_time}秒后重试: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise AIModelServiceError(f"AI模型服务请求失败: {e}")
        
        raise AIModelServiceError("达到最大重试次数")
    
    # === AI模型调用接口 ===
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        调用聊天完成API
        
        Args:
            messages: 消息列表
            model: 指定模型(可选)
            temperature: 生成温度
            max_tokens: 最大token数
            top_p: top_p参数
            stream: 是否流式返回
            
        Returns:
            AI模型响应
        """
        try:
            data = {
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }
            
            if model:
                data["model"] = model
            if max_tokens:
                data["max_tokens"] = max_tokens
            if top_p:
                data["top_p"] = top_p
            
            response = await self._make_request("POST", "/api/v1/chat/completions", data=data)
            
            self._logger.info(f"AI模型调用成功，使用模型: {response.get('model', 'unknown')}")
            return response
            
        except Exception as e:
            self._logger.error(f"AI模型调用失败: {e}")
            raise
    
    async def optimize_text_with_ai(
        self,
        content: str,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        使用AI模型优化文本
        
        Args:
            content: 原始文本内容
            system_prompt: 系统提示词
            user_prompt: 用户提示词  
            model: 指定模型
            temperature: 生成温度
            max_tokens: 最大token数
            
        Returns:
            优化后的文本和相关信息
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 解析响应
            if response.get("choices") and len(response["choices"]) > 0:
                optimized_content = response["choices"][0]["message"]["content"]
                
                result = {
                    "optimized_content": optimized_content,
                    "model_used": response.get("model", "unknown"),
                    "token_usage": response.get("usage", {}),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "response_metadata": {
                        "created": response.get("created"),
                        "id": response.get("id")
                    }
                }
                
                self._logger.info(f"文本优化完成，输出长度: {len(optimized_content)} 字符")
                return result
            else:
                raise AIModelServiceError("AI模型返回空响应")
                
        except Exception as e:
            self._logger.error(f"AI文本优化失败: {e}")
            raise
    
    async def generate_optimization_prompt(
        self,
        original_content: str,
        optimization_type: OptimizationType,
        optimization_mode: OptimizationMode,
        text_analysis: Dict[str, Any],
        custom_instructions: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成优化提示词
        
        Args:
            original_content: 原始内容
            optimization_type: 优化类型
            optimization_mode: 优化模式
            text_analysis: 文本分析结果
            custom_instructions: 自定义指令
            
        Returns:
            包含系统提示词和用户提示词的字典
        """
        try:
            # 构建系统提示词
            system_prompt = self._build_system_prompt(optimization_type, optimization_mode)
            
            # 构建用户提示词
            user_prompt = self._build_user_prompt(
                original_content, 
                optimization_type, 
                optimization_mode,
                text_analysis,
                custom_instructions
            )
            
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }
            
        except Exception as e:
            self._logger.error(f"生成优化提示词失败: {e}")
            raise AIModelServiceError(f"生成优化提示词失败: {e}")
    
    def _build_system_prompt(
        self, 
        optimization_type: OptimizationType, 
        optimization_mode: OptimizationMode
    ) -> str:
        """
        构建系统提示词
        
        Args:
            optimization_type: 优化类型
            optimization_mode: 优化模式
            
        Returns:
            系统提示词
        """
        base_prompt = """你是一个专业的历史文本优化专家，精通古代文献的整理和现代化处理。你的任务是按照指定要求优化历史文本，确保：

1. 保持历史事实的准确性和完整性
2. 遵循学术规范和专业标准
3. 提升文本的可读性和表达质量
4. 保留原文的历史价值和文化内涵

请严格按照用户的要求执行优化任务。"""

        # 根据优化类型添加特定指导
        type_specific = {
            OptimizationType.POLISH: """
专注于语言润色：
- 改善语言表达的流畅性和准确性
- 统一用词规范，提升文本的学术性
- 修正语法错误和表达不当之处
- 保持历史文献的庄重感和严谨性""",
            
            OptimizationType.EXPAND: """
专注于内容扩展：
- 在保持原文核心内容的基础上增加相关细节
- 补充必要的历史背景和上下文信息
- 增强文本的完整性和可读性
- 确保扩展内容符合历史事实""",
            
            OptimizationType.STYLE_CONVERT: """
专注于风格转换：
- 调整文体风格以符合目标要求
- 保持历史事实的准确性
- 适应目标读者群体的阅读习惯
- 保持文本的学术价值和历史意义""",
            
            OptimizationType.MODERNIZE: """
专注于现代化改写：
- 将文言文表达转换为现代汉语
- 保持历史事实和核心内容不变
- 使用现代读者容易理解的表达方式
- 保留必要的历史术语和专有名词"""
        }
        
        # 根据优化模式添加特定要求
        mode_specific = {
            OptimizationMode.HISTORICAL_FORMAT: "按照历史文档标准格式重新组织（时间-人物-地点-事件-感触），保持史书体例。",
            OptimizationMode.ACADEMIC: "遵循现代学术规范，提升文本的学术严谨性和专业性。",
            OptimizationMode.LITERARY: "增强文本的文学性和艺术表达力，提升阅读体验。",
            OptimizationMode.SIMPLIFIED: "简化表达，提高普通读者的理解度和可读性。"
        }
        
        return f"{base_prompt}\n\n{type_specific[optimization_type]}\n\n优化模式要求：{mode_specific[optimization_mode]}"
    
    def _build_user_prompt(
        self,
        original_content: str,
        optimization_type: OptimizationType,
        optimization_mode: OptimizationMode,
        text_analysis: Dict[str, Any],
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        构建用户提示词
        
        Args:
            original_content: 原始内容
            optimization_type: 优化类型
            optimization_mode: 优化模式
            text_analysis: 文本分析结果
            custom_instructions: 自定义指令
            
        Returns:
            用户提示词
        """
        prompt_parts = [
            f"请按照{optimization_type.value}类型和{optimization_mode.value}模式优化以下历史文本：",
            "",
            "【原始文本】",
            original_content,
            "",
            "【文本分析】"
        ]
        
        # 添加文本分析信息
        if text_analysis:
            prompt_parts.extend([
                f"- 文本长度: {text_analysis.get('length', 0)}字符",
                f"- 复杂度: {text_analysis.get('complexity', 'unknown')}",
                f"- 写作风格: {text_analysis.get('style', 'unknown')}",
                f"- 主要主题: {', '.join(text_analysis.get('topics', []))}"
            ])
        
        prompt_parts.extend(["", "【优化要求】"])
        
        # 添加优化要求
        if optimization_type == OptimizationType.POLISH:
            prompt_parts.append("- 保持原文的核心内容和历史事实不变")
            prompt_parts.append("- 改善语言表达的流畅性和准确性")
            prompt_parts.append("- 统一用词规范，提升文本的学术性")
            
        elif optimization_type == OptimizationType.EXPAND:
            prompt_parts.append("- 在保持原文核心内容基础上适当扩展细节")
            prompt_parts.append("- 补充相关的历史背景和上下文信息")
            prompt_parts.append("- 确保所有扩展内容符合历史事实")
            
        elif optimization_type == OptimizationType.STYLE_CONVERT:
            prompt_parts.append(f"- 转换为{optimization_mode.value}风格")
            prompt_parts.append("- 调整文体以符合目标读者群体")
            prompt_parts.append("- 保持历史价值和学术意义")
            
        elif optimization_type == OptimizationType.MODERNIZE:
            prompt_parts.append("- 将古文表达转换为现代汉语")
            prompt_parts.append("- 保持历史事实和核心内容不变")
            prompt_parts.append("- 保留重要的历史术语和专有名词")
        
        # 添加自定义指令
        if custom_instructions:
            prompt_parts.extend([
                "",
                "【特殊要求】",
                custom_instructions
            ])
        
        prompt_parts.extend([
            "",
            "请提供优化后的文本，确保高质量和准确性。"
        ])
        
        return "\n".join(prompt_parts)
    
    # === 模型管理接口 ===
    
    async def get_available_models(self) -> Dict[str, Any]:
        """
        获取可用的AI模型列表
        
        Returns:
            可用模型列表
        """
        try:
            response = await self._make_request("GET", "/api/v1/models")
            return response
        except Exception as e:
            self._logger.error(f"获取可用模型失败: {e}")
            raise
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取指定模型的详细信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息
        """
        try:
            response = await self._make_request("GET", f"/api/v1/models/{model_name}")
            return response
        except Exception as e:
            self._logger.error(f"获取模型信息失败 (model={model_name}): {e}")
            raise
    
    async def select_best_model(
        self,
        optimization_type: OptimizationType,
        content_length: int,
        quality_requirements: Optional[Dict[str, float]] = None
    ) -> str:
        """
        根据任务类型和要求选择最佳模型
        
        Args:
            optimization_type: 优化类型
            content_length: 内容长度
            quality_requirements: 质量要求
            
        Returns:
            推荐的模型名称
        """
        try:
            data = {
                "task_type": "text_optimization",
                "optimization_type": optimization_type.value,
                "content_length": content_length,
                "quality_requirements": quality_requirements or {}
            }
            
            response = await self._make_request("POST", "/api/v1/models/select", data=data)
            model_name = response.get("recommended_model")
            
            if not model_name:
                # 回退到默认模型
                model_name = "gemini-1.5-pro"
                self._logger.warning("未能获取推荐模型，使用默认模型")
            
            self._logger.info(f"选择模型: {model_name} (优化类型: {optimization_type.value})")
            return model_name
            
        except Exception as e:
            self._logger.error(f"模型选择失败: {e}")
            # 返回默认模型
            return "gemini-1.5-pro"
    
    # === 健康检查 ===
    
    async def health_check(self) -> bool:
        """
        检查AI模型服务健康状态
        
        Returns:
            健康状态
        """
        try:
            await self._make_request("GET", "/health")
            return True
        except Exception as e:
            self._logger.warning(f"AI模型服务健康检查失败: {e}")
            return False