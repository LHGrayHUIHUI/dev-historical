"""
简化的AI模型服务 - Simplified AI Model Service
专门用于Docker测试和快速部署的轻量级AI服务

功能特点:
- 无外部存储依赖，纯内存运行
- 直接集成Gemini API，响应快速
- 简化的模型管理和路由逻辑
- 完整的错误处理和日志记录
- 支持健康检查和服务监控

设计目标:
- Docker容器化部署优化
- 最小化启动时间和资源占用
- 保持API接口兼容性
- 提供完整的测试环境
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models.requests import ChatMessage, ChatCompletionRequest


class AIServiceError(Exception):
    """
    AI服务异常类 - AI Service Exception
    
    用于标识AI服务相关的错误和异常情况。
    继承自Python内置的Exception类，支持标准异常处理流程。
    """
    pass


class SimplifiedAIModelService:
    """
    简化的AI模型服务 - Simplified AI Model Service
    
    专门用于Docker测试环境的轻量级AI服务实现。
    移除了复杂的外部依赖，专注于核心AI对话功能。
    
    核心特性:
    1. 纯内存运行，无数据库依赖
    2. 直接集成Google Gemini API
    3. 简化的模型管理逻辑
    4. 快速启动和响应
    5. 完整的错误处理
    
    适用场景:
    - Docker容器化测试
    - 快速原型验证
    - CI/CD集成测试
    - 开发环境调试
    - 功能演示
    
    技术限制:
    - 不支持模型配置持久化
    - 仅支持Gemini提供商
    - 简化的负载均衡逻辑
    - 内存存储，重启后丢失配置
    """
    
    def __init__(self):
        """
        初始化简化AI服务
        
        配置项:
        - logger: 日志记录器，用于记录服务运行状态
        - initialized: 服务初始化状态标志
        - gemini_api_key: 预设的Gemini API密钥（测试用）
        
        注意: 生产环境应使用环境变量管理API密钥
        """
        self._logger = logging.getLogger(__name__)
        self.initialized = False
        self.gemini_api_key = "AIzaSyCrpXFxpEbsKjrHOCQ0oR2dUtMRjys3_-w"  # 测试用密钥
    
    async def initialize(self):
        """
        简化服务初始化 - Simplified Service Initialization
        
        执行服务启动必要的初始化操作，包括:
        1. 检查服务是否已初始化
        2. 记录初始化日志
        3. 设置服务状态标志
        4. 错误处理和恢复机制
        
        特点:
        - 快速启动，无外部依赖检查
        - 即使初始化失败也允许服务运行
        - 适合测试和开发环境
        - 支持重复调用，避免重复初始化
        
        异常处理:
        初始化失败时不会阻止服务启动，而是记录警告并
        继续以简化模式运行，保证服务可用性。
        """
        if self.initialized:
            return
        
        try:
            self._logger.info("初始化简化AI模型服务...")
            self.initialized = True
            self._logger.info("简化AI模型服务初始化成功")
            
        except Exception as e:
            self._logger.error(f"初始化失败: {e}")
            # 允许继续运行
            self.initialized = True
            self._logger.warning("简化模式启动")
    
    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        AI聊天完成核心接口 - AI Chat Completion Core Interface
        
        功能描述:
        处理聊天完成请求，通过Gemini API获取AI回复。
        这是服务的核心功能，支持完整的对话交互。
        
        处理流程:
        1. 确保服务已初始化
        2. 记录请求信息和参数
        3. 创建Gemini适配器实例
        4. 构建模型和账户配置对象
        5. 调用适配器执行AI请求
        6. 标准化响应格式
        7. 记录响应时间和状态
        
        参数:
            request: ChatCompletionRequest - 聊天完成请求对象
                - model: 指定的AI模型名称
                - messages: 对话消息列表
                - temperature: 生成随机性控制
                - max_tokens: 最大输出token数
        
        返回:
            Dict[str, Any]: 标准化的AI响应对象
                成功响应包含:
                - id: 响应唯一标识符
                - object: "chat.completion"
                - model: 实际使用的模型名称
                - choices: AI回复选择列表
                - usage: token使用统计
                - metadata: 响应元数据和性能信息
                
                错误响应包含:
                - error: 错误信息描述
                - provider: AI提供商标识
                - simplified_mode: 简化模式标识
        
        特殊处理:
        - 自动创建简单配置对象，避免复杂依赖
        - 支持默认模型fallback (gemini-1.5-flash)
        - 完整的错误处理和日志记录
        - 响应时间测量和性能监控
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            self._logger.info(f"处理聊天完成请求: model={request.model or 'gemini-1.5-flash'}")
            
            # 使用简化的Gemini适配器
            from ..adapters.simple_gemini_adapter import SimpleGeminiAdapter
            
            adapter = SimpleGeminiAdapter()
            
            # 创建简单配置对象
            model_config = type('SimpleModel', (), {
                'model_id': request.model or 'gemini-1.5-flash',
                'max_tokens': request.max_tokens or 1000
            })()
            
            account_config = type('SimpleAccount', (), {
                'api_key': self.gemini_api_key
            })()
            
            # 调用适配器
            start_time = datetime.now()
            response = await adapter.chat_completion(
                model_config, 
                account_config, 
                request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            # 标准化响应
            if isinstance(response, dict) and 'error' in response:
                return {
                    'error': response['error'],
                    'provider': 'gemini',
                    'model': model_config.model_id,
                    'response_time_ms': response.get('response_time_ms', response_time),
                    'simplified_mode': True
                }
            
            return {
                'id': response.get('id', f'ai-{hash(str(request.messages)) % 1000000}'),
                'object': 'chat.completion',
                'created': response.get('created', int(datetime.now().timestamp())),
                'model': model_config.model_id,
                'provider': 'gemini',
                'choices': response.get('choices', []),
                'usage': response.get('usage', {}),
                'metadata': {
                    'response_time_ms': response.get('response_time_ms', response_time),
                    'model_used': model_config.model_id,
                    'provider_used': 'gemini',
                    'routing_strategy': 'direct',
                    'simplified_mode': True
                }
            }
            
        except Exception as e:
            self._logger.error(f"聊天完成失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'聊天完成失败: {str(e)}'}
    
    async def chat_completion_stream(self, request: ChatCompletionRequest):
        """
        流式聊天完成接口 - Streaming Chat Completion Interface
        
        功能说明:
        简化服务暂不支持流式响应，返回错误信息提示。
        在完整版本的服务中，此接口会提供实时流式AI响应。
        
        参数:
            request: ChatCompletionRequest - 聊天完成请求对象
        
        返回:
            异步生成器，产出错误信息
        
        未来扩展:
        - 实现真正的流式响应
        - 支持Server-Sent Events格式
        - 提供实时token生成
        - 支持取消和中断机制
        """
        yield {'error': 'Stream mode not implemented in simplified service'}
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        获取可用AI模型列表 - Get Available AI Models List
        
        功能描述:
        返回系统中所有可用的AI模型配置信息。
        优先从模型管理控制器获取动态配置，失败时返回默认模型。
        
        数据来源:
        1. 首先尝试从模型管理控制器获取已配置模型
        2. 过滤出状态为"configured"的可用模型
        3. 失败时返回预设的默认模型列表
        
        返回模型信息包含:
        - id: 模型唯一标识符
        - alias: 模型别名
        - name: 模型显示名称
        - provider: AI提供商名称
        - description: 模型描述信息
        - context_window: 上下文窗口大小
        - max_tokens: 最大token数限制
        - is_local: 是否为本地模型
        - simplified_mode: 简化模式标识
        
        错误处理:
        当无法从模型管理控制器获取配置时，返回默认的
        Gemini模型配置，确保服务基本可用性。
        
        使用场景:
        - API客户端查询可用模型
        - 前端显示模型选择列表
        - 服务状态和能力检查
        """
        try:
            # 从模型管理控制器获取配置的模型
            from ..controllers.models_management_controller import create_models_management_controller
            
            management_controller = create_models_management_controller()
            models_data = management_controller._models_config
            
            available_models = []
            for model_id, config in models_data.items():
                # 只返回已配置好的模型
                if config.get("status") == "configured":
                    available_models.append({
                        'id': model_id,
                        'alias': config.get('alias', model_id),
                        'name': config.get('model_name', model_id),
                        'provider': config.get('provider', 'unknown'),
                        'description': f"{config.get('provider', 'Unknown')} - {config.get('model_name', 'Unknown Model')}",
                        'context_window': config.get('max_tokens', 4096),
                        'max_tokens': config.get('max_tokens', 4096),
                        'is_local': config.get('is_local', False),
                        'api_base': config.get('api_base'),
                        'simplified_mode': True
                    })
            
            return available_models
            
        except Exception as e:
            self._logger.error(f"Failed to get available models: {e}")
            # 返回默认模型
            return [
                {
                    'id': 'gemini-flash',
                    'alias': 'gemini-flash',
                    'name': 'Gemini 1.5 Flash',
                    'provider': 'gemini',
                    'description': 'Google Gemini 1.5 Flash model',
                    'context_window': 8192,
                    'max_tokens': 8192,
                    'is_local': False,
                    'simplified_mode': True
                }
            ]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        服务健康检查 - Service Health Check
        
        功能描述:
        返回AI服务的健康状态和基本信息。
        用于监控系统、负载均衡器和容器编排平台的健康探测。
        
        检查内容:
        - 服务基本状态 (healthy)
        - 服务标识信息
        - 运行模式标识 (simplified)
        - 初始化状态
        - 当前时间戳
        - 支持的AI提供商列表
        
        响应字段:
        - status: "healthy" - 服务健康状态
        - service: "ai-model-service" - 服务标识
        - mode: "simplified" - 运行模式
        - initialized: 服务初始化状态
        - timestamp: 检查时间的ISO格式时间戳
        - supported_providers: 支持的AI提供商列表
        
        特点:
        - 轻量级检查，响应迅速
        - 无外部依赖验证
        - 始终返回健康状态
        - 提供基本运行信息
        
        使用场景:
        - Docker容器健康检查
        - Kubernetes探针监控
        - 负载均衡器健康探测
        - 系统监控和告警
        """
        return {
            'status': 'healthy',
            'service': 'ai-model-service',
            'mode': 'simplified',
            'initialized': self.initialized,
            'timestamp': datetime.now().isoformat(),
            'supported_providers': ['gemini']
        }


# 全局实例管理 - Global Instance Management
_simplified_ai_service_instance = None


async def get_ai_service():
    """
    获取简化AI服务实例 - Get Simplified AI Service Instance
    
    功能描述:
    单例模式工厂函数，用于获取全局唯一的AI服务实例。
    确保整个应用程序使用同一个服务实例，避免重复初始化。
    
    工作流程:
    1. 检查全局实例是否已存在
    2. 不存在时创建新的服务实例
    3. 自动执行服务初始化
    4. 返回可用的服务实例
    
    设计模式:
    - 单例模式: 确保全局唯一实例
    - 懒加载: 首次调用时才创建实例
    - 异步初始化: 支持异步操作
    
    返回:
        SimplifiedAIModelService: 已初始化的AI服务实例
    
    使用场景:
    - FastAPI应用启动时获取服务实例
    - 控制器中调用AI功能
    - 生命周期管理和依赖注入
    
    优势:
    - 节省内存和资源
    - 保持状态一致性
    - 简化依赖管理
    - 支持服务复用
    
    注意事项:
    - 全局变量在多线程环境下需要谨慎使用
    - 测试环境可能需要重置实例
    - 服务实例包含状态信息
    """
    global _simplified_ai_service_instance
    
    if _simplified_ai_service_instance is None:
        _simplified_ai_service_instance = SimplifiedAIModelService()
        await _simplified_ai_service_instance.initialize()
    
    return _simplified_ai_service_instance