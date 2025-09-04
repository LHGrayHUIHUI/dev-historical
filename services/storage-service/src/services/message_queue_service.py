"""
消息队列服务

基于RabbitMQ的异步消息处理
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust

logger = logging.getLogger(__name__)


class RabbitMQClient:
    """RabbitMQ消息队列客户端
    
    提供消息发布、消费和队列管理功能
    """
    
    def __init__(self, connection_url: str):
        """初始化RabbitMQ客户端
        
        Args:
            connection_url: RabbitMQ连接URL
        """
        self.connection_url = connection_url
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.exchanges: Dict[str, aio_pika.Exchange] = {}
        self.queues: Dict[str, aio_pika.Queue] = {}
        
        logger.info("RabbitMQ客户端初始化完成")
    
    async def connect(self) -> None:
        """建立RabbitMQ连接"""
        try:
            # 建立连接
            self.connection = await connect_robust(
                self.connection_url,
                heartbeat=60,
                blocked_connection_timeout=300
            )
            
            # 创建通道
            self.channel = await self.connection.channel()
            
            # 设置QoS
            await self.channel.set_qos(prefetch_count=10)
            
            # 声明交换机
            await self._declare_exchanges()
            
            # 声明队列
            await self._declare_queues()
            
            logger.info("RabbitMQ连接建立成功")
            
        except Exception as e:
            logger.error(f"RabbitMQ连接失败: {str(e)}")
            raise
    
    async def _declare_exchanges(self) -> None:
        """声明交换机"""
        # 文本处理交换机
        self.exchanges['text_processing'] = await self.channel.declare_exchange(
            'text_processing',
            ExchangeType.DIRECT,
            durable=True
        )
        
        # 死信交换机
        self.exchanges['dlx'] = await self.channel.declare_exchange(
            'dlx',
            ExchangeType.DIRECT,
            durable=True
        )
        
        logger.info("交换机声明完成")
    
    async def _declare_queues(self) -> None:
        """声明队列"""
        # 文本提取队列
        self.queues['text_extraction'] = await self.channel.declare_queue(
            'text_extraction',
            durable=True,
            arguments={
                'x-max-retries': 3,
                'x-message-ttl': 3600000,  # 1小时TTL
                'x-dead-letter-exchange': 'dlx',
                'x-dead-letter-routing-key': 'text_extraction_failed'
            }
        )
        
        # 文件处理队列
        self.queues['file_processing'] = await self.channel.declare_queue(
            'file_processing',
            durable=True,
            arguments={
                'x-max-retries': 3,
                'x-message-ttl': 3600000,
                'x-dead-letter-exchange': 'dlx',
                'x-dead-letter-routing-key': 'file_processing_failed'
            }
        )
        
        # 死信队列
        self.queues['text_extraction_failed'] = await self.channel.declare_queue(
            'text_extraction_failed',
            durable=True
        )
        
        self.queues['file_processing_failed'] = await self.channel.declare_queue(
            'file_processing_failed',
            durable=True
        )
        
        # 绑定队列到交换机
        await self._bind_queues()
        
        logger.info("队列声明完成")
    
    async def _bind_queues(self) -> None:
        """绑定队列到交换机"""
        # 绑定文本提取队列
        await self.queues['text_extraction'].bind(
            self.exchanges['text_processing'],
            routing_key='extract'
        )
        
        # 绑定文件处理队列
        await self.queues['file_processing'].bind(
            self.exchanges['text_processing'],
            routing_key='process'
        )
        
        # 绑定死信队列
        await self.queues['text_extraction_failed'].bind(
            self.exchanges['dlx'],
            routing_key='text_extraction_failed'
        )
        
        await self.queues['file_processing_failed'].bind(
            self.exchanges['dlx'],
            routing_key='file_processing_failed'
        )
        
        logger.info("队列绑定完成")
    
    async def publish_processing_task(
        self,
        dataset_id: str,
        file_path: str,
        file_type: str,
        task_type: str = 'text_extraction',
        priority: int = 5,
        **kwargs
    ) -> None:
        """发布文件处理任务
        
        Args:
            dataset_id: 数据集ID
            file_path: 文件路径
            file_type: 文件类型
            task_type: 任务类型
            priority: 任务优先级 (1-10, 10最高)
            **kwargs: 其他任务参数
        """
        try:
            # 构建消息体
            message_body = {
                'task_id': f"{task_type}_{dataset_id}_{datetime.utcnow().timestamp()}",
                'task_type': task_type,
                'dataset_id': dataset_id,
                'file_path': file_path,
                'file_type': file_type,
                'created_at': datetime.utcnow().isoformat(),
                'retry_count': 0,
                **kwargs
            }
            
            # 创建消息
            message = Message(
                json.dumps(message_body).encode('utf-8'),
                priority=priority,
                delivery_mode=DeliveryMode.PERSISTENT,
                headers={
                    'task_type': task_type,
                    'dataset_id': dataset_id,
                    'created_at': message_body['created_at']
                },
                expiration=3600000  # 1小时过期
            )
            
            # 选择路由键
            routing_key = 'extract' if task_type == 'text_extraction' else 'process'
            
            # 发布消息
            await self.exchanges['text_processing'].publish(
                message,
                routing_key=routing_key
            )
            
            logger.info(
                f"任务发布成功",
                extra={
                    "task_type": task_type,
                    "dataset_id": dataset_id,
                    "file_path": file_path,
                    "priority": priority
                }
            )
            
        except Exception as e:
            logger.error(f"发布处理任务失败: {str(e)}")
            raise
    
    async def setup_consumer(
        self,
        queue_name: str,
        callback: Callable,
        consumer_tag: str = None,
        auto_ack: bool = False
    ) -> None:
        """设置消息消费者
        
        Args:
            queue_name: 队列名称
            callback: 消息处理回调函数
            consumer_tag: 消费者标签
            auto_ack: 是否自动确认消息
        """
        try:
            queue = self.queues.get(queue_name)
            if not queue:
                raise ValueError(f"队列 {queue_name} 不存在")
            
            # 包装回调函数以处理异常
            async def wrapped_callback(message: aio_pika.IncomingMessage):
                async with message.process(ignore_processed=True):
                    try:
                        await callback(message)
                    except Exception as e:
                        logger.error(
                            f"消息处理失败",
                            extra={
                                "queue": queue_name,
                                "error": str(e),
                                "message_id": message.message_id
                            }
                        )
                        # 重新抛出异常以触发重试机制
                        raise
            
            # 开始消费消息
            await queue.consume(
                wrapped_callback,
                consumer_tag=consumer_tag or f"{queue_name}_consumer"
            )
            
            logger.info(
                f"消费者设置成功",
                extra={
                    "queue": queue_name,
                    "consumer_tag": consumer_tag
                }
            )
            
        except Exception as e:
            logger.error(f"设置消费者失败: {str(e)}")
            raise
    
    async def publish_message(
        self,
        exchange_name: str,
        routing_key: str,
        message_body: Dict[str, Any],
        priority: int = 5,
        **kwargs
    ) -> None:
        """发布通用消息
        
        Args:
            exchange_name: 交换机名称
            routing_key: 路由键
            message_body: 消息体
            priority: 优先级
            **kwargs: 其他消息属性
        """
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"交换机 {exchange_name} 不存在")
            
            # 添加时间戳
            message_body['published_at'] = datetime.utcnow().isoformat()
            
            # 创建消息
            message = Message(
                json.dumps(message_body).encode('utf-8'),
                priority=priority,
                delivery_mode=DeliveryMode.PERSISTENT,
                **kwargs
            )
            
            # 发布消息
            await exchange.publish(message, routing_key=routing_key)
            
            logger.info(f"消息发布成功: {exchange_name}/{routing_key}")
            
        except Exception as e:
            logger.error(f"发布消息失败: {str(e)}")
            raise
    
    async def get_queue_info(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """获取队列信息
        
        Args:
            queue_name: 队列名称
            
        Returns:
            队列信息字典，失败返回None
        """
        try:
            queue = self.queues.get(queue_name)
            if not queue:
                return None
            
            # 获取队列状态
            info = await self.channel.queue_declare(queue_name, passive=True)
            
            return {
                "name": queue_name,
                "message_count": info.method.message_count,
                "consumer_count": info.method.consumer_count
            }
            
        except Exception as e:
            logger.error(f"获取队列信息失败: {str(e)}")
            return None
    
    async def purge_queue(self, queue_name: str) -> bool:
        """清空队列
        
        Args:
            queue_name: 队列名称
            
        Returns:
            是否清空成功
        """
        try:
            queue = self.queues.get(queue_name)
            if not queue:
                return False
            
            await queue.purge()
            logger.info(f"队列 {queue_name} 清空成功")
            return True
            
        except Exception as e:
            logger.error(f"清空队列失败: {str(e)}")
            return False
    
    async def close(self) -> None:
        """关闭连接"""
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                logger.info("RabbitMQ连接已关闭")
        except Exception as e:
            logger.error(f"关闭RabbitMQ连接失败: {str(e)}")
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return (
            self.connection is not None 
            and not self.connection.is_closed 
            and self.channel is not None 
            and not self.channel.is_closed
        )