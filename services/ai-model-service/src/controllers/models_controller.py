"""
模型管理API控制器
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query

from ..services.ai_service import get_ai_service, AIServiceError


class ModelsController:
    """模型管理API控制器"""
    
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/models", tags=["models"])
        self._logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.router.get("/")
        async def list_models(
            provider: Optional[str] = Query(None, description="筛选指定提供商的模型")
        ) -> Dict[str, List[Dict[str, Any]]]:
            """
            获取可用模型列表
            
            Args:
                provider: 可选的提供商筛选 (openai, claude, baidu, alibaba, tencent, zhipu)
                
            Returns:
                按提供商分组的模型列表
            """
            try:
                self._logger.info(f"Listing models, provider filter: {provider}")
                
                # 验证提供商参数
                if provider:
                    supported_providers = ['openai', 'claude', 'baidu', 'alibaba', 'tencent', 'zhipu']
                    if provider not in supported_providers:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"不支持的提供商: {provider}, 支持的提供商: {supported_providers}"
                        )
                
                ai_service = await get_ai_service()
                models = await ai_service.list_available_models(provider=provider)
                
                self._logger.info(f"Listed models successfully, total providers: {len(models)}")
                
                return models
                
            except AIServiceError as e:
                self._logger.error(f"AI service error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            
            except Exception as e:
                self._logger.error(f"Unexpected error listing models: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
        
        @self.router.get("/providers")
        async def list_providers() -> Dict[str, Any]:
            """
            获取支持的AI提供商列表
            
            Returns:
                支持的提供商信息
            """
            try:
                self._logger.info("Listing supported providers")
                
                ai_service = await get_ai_service()
                supported_providers = ai_service.adapter_factory.get_supported_providers()
                
                providers_info = {
                    'supported_providers': [provider.value for provider in supported_providers],
                    'provider_details': {}
                }
                
                # 添加提供商详细信息
                provider_descriptions = {
                    'openai': {
                        'name': 'OpenAI',
                        'description': 'OpenAI GPT系列模型',
                        'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
                    },
                    'claude': {
                        'name': 'Anthropic Claude',
                        'description': 'Anthropic Claude系列模型',
                        'models': ['claude-3-sonnet', 'claude-3-opus', 'claude-3-haiku']
                    },
                    'baidu': {
                        'name': '百度文心一言',
                        'description': '百度文心大模型',
                        'models': ['ernie-4.0-turbo', 'ernie-3.5', 'ernie-turbo']
                    },
                    'alibaba': {
                        'name': '阿里云通义千问',
                        'description': '阿里云通义千问大模型',
                        'models': ['qwen-plus', 'qwen-turbo', 'qwen-max']
                    },
                    'tencent': {
                        'name': '腾讯混元',
                        'description': '腾讯混元大模型',
                        'models': ['hunyuan-pro', 'hunyuan-standard', 'hunyuan-lite']
                    },
                    'zhipu': {
                        'name': '智谱AI',
                        'description': '智谱GLM系列模型',
                        'models': ['glm-4', 'glm-4-air', 'glm-3-turbo']
                    }
                }
                
                for provider in supported_providers:
                    provider_name = provider.value
                    if provider_name in provider_descriptions:
                        providers_info['provider_details'][provider_name] = provider_descriptions[provider_name]
                
                self._logger.info(f"Listed {len(supported_providers)} supported providers")
                
                return providers_info
                
            except Exception as e:
                self._logger.error(f"Unexpected error listing providers: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
        
        @self.router.get("/{provider}/models")
        async def list_provider_models(provider: str) -> List[Dict[str, Any]]:
            """
            获取指定提供商的模型列表
            
            Args:
                provider: 提供商名称
                
            Returns:
                该提供商的模型列表
            """
            try:
                self._logger.info(f"Listing models for provider: {provider}")
                
                ai_service = await get_ai_service()
                models_dict = await ai_service.list_available_models(provider=provider)
                
                if provider not in models_dict:
                    raise HTTPException(status_code=404, detail=f"提供商 {provider} 不存在或无可用模型")
                
                models_list = models_dict[provider]
                
                self._logger.info(f"Listed {len(models_list)} models for provider {provider}")
                
                return models_list
                
            except HTTPException:
                raise
            except AIServiceError as e:
                self._logger.error(f"AI service error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self._logger.error(f"Unexpected error listing provider models: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
        
        @self.router.get("/capabilities")
        async def get_model_capabilities() -> Dict[str, Any]:
            """
            获取模型能力说明
            
            Returns:
                模型能力描述
            """
            try:
                capabilities_info = {
                    'common_capabilities': {
                        'chat': '支持多轮对话',
                        'streaming': '支持流式输出',
                        'function_calling': '支持函数调用',
                        'json_mode': '支持JSON格式输出',
                        'vision': '支持图像理解'
                    },
                    'provider_capabilities': {
                        'openai': {
                            'strengths': ['强大的推理能力', '丰富的知识库', '良好的代码生成'],
                            'limitations': ['可能存在幻觉', '知识截止时间限制']
                        },
                        'claude': {
                            'strengths': ['长文本处理', '安全性强', '逻辑推理优秀'],
                            'limitations': ['响应速度较慢', '某些创意任务表现一般']
                        },
                        'baidu': {
                            'strengths': ['中文理解优秀', '本土化程度高', '合规性好'],
                            'limitations': ['英文能力相对较弱', '创意性有待提升']
                        },
                        'alibaba': {
                            'strengths': ['中文处理强', '电商领域专业', '响应速度快'],
                            'limitations': ['国际化内容理解有限', '复杂推理能力待提升']
                        },
                        'tencent': {
                            'strengths': ['游戏内容专业', '社交媒体理解', '多模态能力'],
                            'limitations': ['通用性有待加强', '学术内容相对较弱']
                        },
                        'zhipu': {
                            'strengths': ['科研背景强', '数学推理好', '代码能力优秀'],
                            'limitations': ['商业化程度不足', '生态相对较小']
                        }
                    },
                    'parameter_explanations': {
                        'temperature': {
                            'description': '控制输出的随机性和创造性',
                            'range': '0.0-2.0',
                            'recommendation': '0.7适合一般对话，1.0适合创意写作，0.1适合事实查询'
                        },
                        'top_p': {
                            'description': '核采样参数，控制候选词汇范围',
                            'range': '0.0-1.0',
                            'recommendation': '0.9是常用值，与temperature配合使用'
                        },
                        'max_tokens': {
                            'description': '最大输出token数量',
                            'range': '1-模型上限',
                            'recommendation': '根据需要设置，过小可能截断，过大消耗更多资源'
                        }
                    }
                }
                
                self._logger.info("Returned model capabilities information")
                
                return capabilities_info
                
            except Exception as e:
                self._logger.error(f"Unexpected error getting capabilities: {e}")
                raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")


# 创建控制器实例
def create_models_controller() -> ModelsController:
    """创建模型管理控制器实例"""
    return ModelsController()