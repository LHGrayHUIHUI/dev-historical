# 编码规范文档

## 1. 概述

本文档定义了历史文本优化项目的编码规范和最佳实践，确保代码的一致性、可读性和可维护性。项目采用微服务架构，主要使用Python 3.11+和Vue3+TypeScript技术栈。

### 核心原则
- **一致性优先**: 保持整个项目的编码风格一致
- **中文注释**: 所有注释必须使用中文，注释密度≥30%
- **测试驱动**: 单元测试覆盖率≥80%，集成测试验证端到端功能
- **安全第一**: 遵循安全编码最佳实践，防止常见漏洞
- **性能考虑**: 编写高效的代码，合理使用缓存和异步处理

## 2. Python 编码规范

### 2.1 基础规范

遵循 PEP 8 规范，并使用以下工具进行代码格式化和检查：

```bash
# 代码格式化
black --line-length=88 src/
isort --profile=black src/

# 代码检查
flake8 src/
mypy src/
bandit -r src/
```

### 2.2 命名规范

```python
# 模块和包名：小写字母，用下划线分隔
import data_processor
from services.ocr_service import OCRProcessor

# 类名：PascalCase（大驼峰命名法）
class DocumentProcessor:
    """文档处理器类"""
    pass

class UserAuthenticationService:
    """用户认证服务类"""
    pass

# 函数和变量名：snake_case（小写加下划线）
def extract_text_content(document_path: str) -> str:
    """从文档中提取文本内容"""
    pass

user_id = "12345"
document_metadata = {}

# 常量：全大写，用下划线分隔
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
DEFAULT_TIMEOUT = 30
API_VERSION = "v1"

# 私有属性和方法：单下划线开头
class DocumentService:
    def __init__(self):
        self._cache = {}  # 私有属性
        
    def _validate_document(self, doc):  # 私有方法
        """验证文档格式"""
        pass
```

### 2.3 类型提示

所有公共方法必须包含完整的类型提示：

```python
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID

class DocumentProcessor:
    """文档处理器
    
    负责处理各种格式的文档，提取文本内容
    并进行预处理和清洗操作。
    """
    
    def process_document(
        self,
        file_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理文档并返回结果
        
        Args:
            file_path: 文档文件路径
            options: 可选的处理参数
            
        Returns:
            包含处理结果的字典，包含以下键：
            - text: 提取的文本内容
            - metadata: 文档元数据
            - status: 处理状态
            
        Raises:
            FileNotFoundError: 文件不存在时抛出
            ProcessingError: 处理过程中发生错误时抛出
        """
        pass
    
    async def async_process_batch(
        self,
        file_paths: List[str],
        callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """异步批量处理文档
        
        Args:
            file_paths: 文档文件路径列表
            callback: 可选的进度回调函数
            
        Returns:
            处理结果列表
        """
        pass
```

### 2.4 文档字符串规范

使用Google风格的文档字符串，必须使用中文：

```python
class DataSourceService:
    """数据源服务类
    
    提供多平台数据获取功能，支持今日头条、百家号、
    小红书等平台的内容爬取和数据整理。
    
    Attributes:
        client: HTTP客户端实例
        proxy_manager: 代理管理器
        rate_limiter: 频率限制器
    
    Example:
        >>> service = DataSourceService()
        >>> result = await service.fetch_content('https://example.com')
        >>> print(result['title'])
        '示例标题'
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化数据源服务
        
        Args:
            config: 服务配置字典，包含以下键值：
                - timeout: 请求超时时间（秒）
                - max_retries: 最大重试次数
                - proxy_enabled: 是否启用代理
                - user_agents: 用户代理列表
        
        Raises:
            ValueError: 配置参数无效时抛出
            ConnectionError: 网络连接失败时抛出
        """
        self.config = config
        self._validate_config()
    
    async def fetch_content(
        self, 
        url: str, 
        platform: str = "auto"
    ) -> Optional[Dict[str, Any]]:
        """从指定URL获取内容
        
        根据平台类型自动选择合适的解析器，提取文章标题、
        内容、发布时间等信息。支持智能重试和错误恢复。
        
        Args:
            url: 目标URL地址
            platform: 平台类型，支持 'toutiao'、'baijiahao'、
                     'xiaohongshu' 或 'auto'（自动检测）
        
        Returns:
            包含内容信息的字典，包含以下键：
            - title: 文章标题
            - content: 文章正文
            - author: 作者信息
            - publish_time: 发布时间
            - url: 原始URL
            - platform: 平台标识
            
            如果获取失败返回None
        
        Raises:
            InvalidUrlError: URL格式无效时抛出
            PlatformNotSupportedError: 不支持的平台时抛出
            RateLimitExceededError: 超出频率限制时抛出
        """
        pass
```

### 2.5 错误处理规范

```python
# 自定义异常类
class HistoricalTextError(Exception):
    """项目基础异常类"""
    pass

class DocumentProcessingError(HistoricalTextError):
    """文档处理异常"""
    pass

class OCRError(DocumentProcessingError):
    """OCR识别异常"""
    pass

# 错误处理模式
from contextlib import asynccontextmanager
from loguru import logger
import traceback

class DocumentService:
    """文档服务类"""
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """处理文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            处理结果字典
        """
        try:
            # 参数验证
            if not file_path or not isinstance(file_path, str):
                raise ValueError("文件路径参数无效")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 业务逻辑处理
            result = await self._do_process(file_path)
            logger.info(f"文档处理成功: {file_path}")
            
            return result
            
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            raise
            
        except DocumentProcessingError as e:
            logger.error(f"文档处理失败: {file_path}, 错误: {e}")
            raise
            
        except Exception as e:
            # 记录完整的错误堆栈
            logger.error(f"处理文档时发生未知错误: {file_path}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise DocumentProcessingError(f"文档处理失败: {str(e)}")
    
    @asynccontextmanager
    async def _transaction_context(self):
        """事务上下文管理器"""
        transaction = await self.db.begin()
        try:
            yield transaction
            await transaction.commit()
            logger.debug("事务提交成功")
        except Exception as e:
            await transaction.rollback()
            logger.error(f"事务回滚: {e}")
            raise
```

### 2.6 异步编程规范

```python
import asyncio
from typing import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

class AsyncDocumentProcessor:
    """异步文档处理器"""
    
    def __init__(self, max_concurrent: int = 10):
        """初始化异步处理器
        
        Args:
            max_concurrent: 最大并发处理数量
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """处理单个文件
        
        使用信号量控制并发数量，避免资源过度占用
        """
        async with self.semaphore:
            return await self._do_process_file(file_path)
    
    async def process_files_batch(
        self, 
        file_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """批量处理文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            处理结果列表
        """
        tasks = [
            self.process_file(file_path) 
            for file_path in file_paths
        ]
        
        # 使用 asyncio.gather 进行并发处理
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"处理文件失败: {file_paths[i]}, 错误: {result}")
                processed_results.append({
                    "file_path": file_paths[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def stream_process_files(
        self, 
        file_paths: AsyncIterator[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式处理文件
        
        适用于大量文件的处理场景，避免内存占用过高
        """
        async for file_path in file_paths:
            try:
                result = await self.process_file(file_path)
                yield result
            except Exception as e:
                logger.error(f"流式处理失败: {file_path}, 错误: {e}")
                yield {
                    "file_path": file_path,
                    "success": False,
                    "error": str(e)
                }
```

### 2.7 配置管理规范

```python
from pydantic import BaseSettings, Field, validator
from typing import List, Optional
from enum import Enum

class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class DatabaseSettings(BaseSettings):
    """数据库配置"""
    
    host: str = Field(..., description="数据库主机地址")
    port: int = Field(5432, description="数据库端口")
    database: str = Field(..., description="数据库名称")
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    max_connections: int = Field(20, description="最大连接数")
    
    @validator('port')
    def validate_port(cls, v):
        """验证端口号"""
        if not 1 <= v <= 65535:
            raise ValueError('端口号必须在1-65535范围内')
        return v
    
    @property
    def connection_url(self) -> str:
        """获取数据库连接URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class ServiceSettings(BaseSettings):
    """服务配置"""
    
    # 服务基础配置
    service_name: str = Field("historical-text-service", description="服务名称")
    host: str = Field("0.0.0.0", description="监听主机")
    port: int = Field(8000, description="监听端口")
    debug: bool = Field(False, description="调试模式")
    log_level: LogLevel = Field(LogLevel.INFO, description="日志级别")
    
    # 数据库配置
    database: DatabaseSettings
    
    # Redis配置
    redis_url: str = Field("redis://localhost:6379", description="Redis连接URL")
    
    # 文件处理配置
    max_file_size: int = Field(50 * 1024 * 1024, description="最大文件大小（字节）")
    allowed_file_types: List[str] = Field(
        ["pdf", "docx", "txt", "jpg", "png"],
        description="允许的文件类型"
    )
    
    # 并发配置
    max_concurrent_tasks: int = Field(10, description="最大并发任务数")
    request_timeout: int = Field(30, description="请求超时时间（秒）")
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False

# 使用配置
settings = ServiceSettings()
```

## 3. FastAPI 开发规范

### 3.1 路由组织

```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import List, Optional

# 创建路由器
router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    responses={404: {"description": "文档未找到"}}
)

security = HTTPBearer()

# 请求模型
class DocumentCreateRequest(BaseModel):
    """文档创建请求模型"""
    
    title: str = Field(..., min_length=1, max_length=200, description="文档标题")
    content: Optional[str] = Field(None, description="文档内容")
    tags: List[str] = Field(default_factory=list, description="文档标签")
    
    class Config:
        """模型配置"""
        schema_extra = {
            "example": {
                "title": "示例文档",
                "content": "文档内容示例",
                "tags": ["历史", "文献"]
            }
        }

# 响应模型
class DocumentResponse(BaseModel):
    """文档响应模型"""
    
    id: str = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    status: str = Field(..., description="处理状态")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        """模型配置"""
        orm_mode = True

# 依赖注入
async def get_current_user(token: str = Depends(security)) -> User:
    """获取当前用户"""
    try:
        user = await verify_token(token.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证令牌"
            )
        return user
    except Exception as e:
        logger.error(f"用户认证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证失败"
        )

async def get_document_service() -> DocumentService:
    """获取文档服务实例"""
    return DocumentService()

# 路由处理器
@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建新文档",
    description="创建新的文档记录，支持多种格式的文件上传"
)
async def create_document(
    request: DocumentCreateRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """创建新文档
    
    此端点用于创建新的文档记录。文档创建后会进入
    处理队列，异步执行OCR和文本提取操作。
    
    Args:
        request: 文档创建请求
        current_user: 当前登录用户
        service: 文档服务实例
        
    Returns:
        创建的文档信息
        
    Raises:
        HTTPException: 当文档创建失败时抛出
    """
    try:
        document = await service.create_document(
            title=request.title,
            content=request.content,
            tags=request.tags,
            user_id=current_user.id
        )
        
        logger.info(f"用户 {current_user.id} 创建了文档: {document.id}")
        return DocumentResponse(**document.dict())
        
    except ValidationError as e:
        logger.error(f"文档创建验证失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"数据验证失败: {str(e)}"
        )
    except Exception as e:
        logger.error(f"文档创建失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档创建失败"
        )

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="获取文档详情",
    description="根据文档ID获取文档的详细信息"
)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    """获取文档详情"""
    try:
        document = await service.get_document(document_id, current_user.id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档 {document_id} 不存在"
            )
        
        return DocumentResponse(**document.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档失败: {document_id}, 错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文档失败"
        )
```

### 3.2 中间件规范

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件
    
    记录所有HTTP请求的详细信息，包括请求时间、
    响应时间、状态码等，便于监控和调试。
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """处理请求并记录日志
        
        Args:
            request: HTTP请求对象
            call_next: 下一个处理函数
            
        Returns:
            HTTP响应对象
        """
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 记录请求信息
        logger.info(
            f"请求开始 - ID: {request_id}, "
            f"方法: {request.method}, "
            f"路径: {request.url.path}, "
            f"客户端IP: {request.client.host}"
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            logger.info(
                f"请求完成 - ID: {request_id}, "
                f"状态码: {response.status_code}, "
                f"处理时间: {process_time:.3f}s"
            )
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # 记录错误信息
            process_time = time.time() - start_time
            logger.error(
                f"请求失败 - ID: {request_id}, "
                f"错误: {str(e)}, "
                f"处理时间: {process_time:.3f}s"
            )
            raise
```

## 4. 数据库操作规范

### 4.1 SQLAlchemy模型定义

```python
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class TimestampMixin:
    """时间戳混入类"""
    
    created_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        nullable=False,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow, 
        nullable=False,
        comment="更新时间"
    )

class Document(Base, TimestampMixin):
    """文档模型
    
    存储历史文档的基本信息和处理状态，
    支持多种文档格式和批量处理。
    """
    
    __tablename__ = "documents"
    __table_args__ = {'comment': '文档表'}
    
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        comment="文档ID"
    )
    title = Column(
        String(200), 
        nullable=False, 
        index=True,
        comment="文档标题"
    )
    content = Column(
        Text, 
        comment="文档原始内容"
    )
    processed_content = Column(
        Text, 
        comment="处理后的内容"
    )
    file_path = Column(
        String(500), 
        comment="文件存储路径"
    )
    file_size = Column(
        Integer, 
        comment="文件大小（字节）"
    )
    mime_type = Column(
        String(100), 
        comment="文件MIME类型"
    )
    status = Column(
        String(20), 
        default="uploaded", 
        index=True,
        comment="处理状态"
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id"), 
        nullable=False,
        comment="所属用户ID"
    )
    
    # 关系定义
    user = relationship("User", back_populates="documents")
    tags = relationship("DocumentTag", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        """对象字符串表示"""
        return f"<Document(id={self.id}, title='{self.title}', status='{self.status}')>"

class DocumentTag(Base, TimestampMixin):
    """文档标签模型"""
    
    __tablename__ = "document_tags"
    __table_args__ = {'comment': '文档标签表'}
    
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id"), 
        nullable=False
    )
    tag_name = Column(
        String(50), 
        nullable=False,
        comment="标签名称"
    )
    
    # 关系定义
    document = relationship("Document", back_populates="tags")
```

### 4.2 数据访问对象（DAO）规范

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any

class BaseDAO:
    """数据访问对象基类
    
    提供通用的数据库操作方法，包括CRUD操作、
    分页查询、事务管理等功能。
    """
    
    def __init__(self, session: AsyncSession):
        """初始化DAO
        
        Args:
            session: 异步数据库会话
        """
        self.session = session

class DocumentDAO(BaseDAO):
    """文档数据访问对象
    
    负责处理文档相关的所有数据库操作，
    包括增删改查、状态更新、标签管理等。
    """
    
    async def create(self, document_data: Dict[str, Any]) -> Document:
        """创建文档
        
        Args:
            document_data: 文档数据字典
            
        Returns:
            创建的文档对象
            
        Raises:
            DatabaseError: 数据库操作失败时抛出
        """
        try:
            document = Document(**document_data)
            self.session.add(document)
            await self.session.flush()  # 获取ID但不提交事务
            await self.session.refresh(document)
            
            logger.info(f"创建文档成功: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"创建文档失败: {e}")
            await self.session.rollback()
            raise DatabaseError(f"创建文档失败: {str(e)}")
    
    async def get_by_id(
        self, 
        document_id: str, 
        include_tags: bool = False
    ) -> Optional[Document]:
        """根据ID获取文档
        
        Args:
            document_id: 文档ID
            include_tags: 是否包含标签信息
            
        Returns:
            文档对象，不存在时返回None
        """
        try:
            query = select(Document).where(Document.id == document_id)
            
            if include_tags:
                query = query.options(selectinload(Document.tags))
            
            result = await self.session.execute(query)
            document = result.scalar_one_or_none()
            
            if document:
                logger.debug(f"查询文档成功: {document_id}")
            else:
                logger.warning(f"文档不存在: {document_id}")
                
            return document
            
        except Exception as e:
            logger.error(f"查询文档失败: {document_id}, 错误: {e}")
            raise DatabaseError(f"查询文档失败: {str(e)}")
    
    async def list_by_user(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Document]:
        """获取用户的文档列表
        
        Args:
            user_id: 用户ID
            status: 可选的状态过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            文档列表
        """
        try:
            query = select(Document).where(Document.user_id == user_id)
            
            if status:
                query = query.where(Document.status == status)
            
            query = query.order_by(Document.created_at.desc())
            query = query.limit(limit).offset(offset)
            
            result = await self.session.execute(query)
            documents = result.scalars().all()
            
            logger.debug(f"查询用户文档: {user_id}, 返回 {len(documents)} 条记录")
            return documents
            
        except Exception as e:
            logger.error(f"查询用户文档失败: {user_id}, 错误: {e}")
            raise DatabaseError(f"查询用户文档失败: {str(e)}")
    
    async def update_status(
        self, 
        document_id: str, 
        status: str,
        processed_content: Optional[str] = None
    ) -> bool:
        """更新文档状态
        
        Args:
            document_id: 文档ID
            status: 新状态
            processed_content: 可选的处理后内容
            
        Returns:
            更新成功返回True，否则返回False
        """
        try:
            query = select(Document).where(Document.id == document_id)
            result = await self.session.execute(query)
            document = result.scalar_one_or_none()
            
            if not document:
                logger.warning(f"要更新的文档不存在: {document_id}")
                return False
            
            document.status = status
            document.updated_at = datetime.utcnow()
            
            if processed_content:
                document.processed_content = processed_content
            
            await self.session.flush()
            logger.info(f"更新文档状态成功: {document_id} -> {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"更新文档状态失败: {document_id}, 错误: {e}")
            raise DatabaseError(f"更新文档状态失败: {str(e)}")
```

## 5. 测试规范

### 5.1 单元测试规范

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import tempfile
import os

class TestDocumentService:
    """文档服务单元测试类
    
    测试文档服务的所有核心功能，包括文档创建、
    处理、状态更新等操作的正确性。
    """
    
    @pytest.fixture
    def mock_session(self):
        """模拟数据库会话"""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session
    
    @pytest.fixture
    def document_service(self, mock_session):
        """文档服务实例"""
        return DocumentService(session=mock_session)
    
    @pytest.fixture
    def sample_document_data(self):
        """示例文档数据"""
        return {
            "title": "测试文档",
            "content": "这是测试内容",
            "user_id": "123e4567-e89b-12d3-a456-426614174000",
            "tags": ["测试", "示例"]
        }
    
    async def test_create_document_success(
        self, 
        document_service, 
        sample_document_data,
        mock_session
    ):
        """测试成功创建文档
        
        验证文档创建的正常流程，包括数据验证、
        数据库操作、返回结果等步骤。
        """
        # 准备测试数据
        expected_id = "doc-12345"
        mock_session.add = Mock()
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()
        
        # 模拟数据库返回
        with patch.object(Document, '__init__', return_value=None):
            with patch.object(Document, 'id', expected_id):
                # 执行测试
                result = await document_service.create_document(sample_document_data)
                
                # 验证结果
                assert result is not None
                assert result.title == sample_document_data["title"]
                
                # 验证数据库操作
                mock_session.add.assert_called_once()
                mock_session.flush.assert_called_once()
                mock_session.refresh.assert_called_once()
    
    async def test_create_document_validation_error(
        self, 
        document_service
    ):
        """测试文档创建时的数据验证错误
        
        验证当输入数据不符合要求时，服务能够
        正确抛出验证异常。
        """
        # 准备无效数据
        invalid_data = {
            "title": "",  # 空标题
            "user_id": "invalid-uuid"  # 无效的UUID
        }
        
        # 执行测试并验证异常
        with pytest.raises(ValidationError) as exc_info:
            await document_service.create_document(invalid_data)
        
        assert "标题不能为空" in str(exc_info.value)
    
    @pytest.mark.parametrize("status,expected_result", [
        ("processing", True),
        ("completed", True),
        ("failed", True),
        ("invalid_status", False),
    ])
    async def test_update_document_status(
        self,
        document_service,
        status,
        expected_result
    ):
        """参数化测试文档状态更新
        
        验证不同状态值的更新操作结果
        """
        document_id = "test-doc-id"
        
        with patch.object(document_service, '_validate_status') as mock_validate:
            mock_validate.return_value = expected_result
            
            result = await document_service.update_status(document_id, status)
            assert result == expected_result
    
    async def test_process_document_with_retry(
        self,
        document_service,
        mock_session
    ):
        """测试文档处理的重试机制
        
        验证当处理失败时，系统能够按照配置
        的重试策略进行重试操作。
        """
        document_id = "test-doc-id"
        
        # 模拟前两次调用失败，第三次成功
        side_effects = [
            ProcessingError("临时错误"),
            ProcessingError("临时错误"),
            {"status": "completed", "content": "处理完成"}
        ]
        
        with patch.object(document_service, '_do_process') as mock_process:
            mock_process.side_effect = side_effects
            
            result = await document_service.process_document(document_id)
            
            # 验证重试次数
            assert mock_process.call_count == 3
            assert result["status"] == "completed"
```

### 5.2 集成测试规范

```python
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

class TestDocumentAPI:
    """文档API集成测试类
    
    测试文档相关API的端到端功能，验证请求响应、
    数据库交互、业务流程等完整功能链路。
    """
    
    @pytest.fixture(scope="class")
    def event_loop(self):
        """事件循环fixture"""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture(scope="class")
    async def test_db_engine(self):
        """测试数据库引擎"""
        engine = create_async_engine(
            "postgresql+asyncpg://test:test@localhost:5432/test_db",
            echo=False
        )
        
        # 创建表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield engine
        
        # 清理
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        await engine.dispose()
    
    @pytest.fixture
    async def test_session(self, test_db_engine):
        """测试数据库会话"""
        async_session = sessionmaker(
            test_db_engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
        
        async with async_session() as session:
            yield session
            await session.rollback()
    
    @pytest.fixture
    async def test_client(self, test_session):
        """测试客户端"""
        def override_get_db():
            return test_session
        
        app.dependency_overrides[get_db_session] = override_get_db
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
        
        app.dependency_overrides.clear()
    
    @pytest.fixture
    async def auth_token(self, test_client):
        """获取认证令牌"""
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        
        return response.json()["access_token"]
    
    async def test_create_document_success(self, test_client, auth_token):
        """测试成功创建文档的完整流程
        
        包括请求认证、数据验证、数据库写入、
        响应返回等完整的API调用过程。
        """
        # 准备测试数据
        document_data = {
            "title": "集成测试文档",
            "content": "这是集成测试的内容",
            "tags": ["测试", "集成"]
        }
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # 发送请求
        response = await test_client.post(
            "/api/v1/documents/",
            json=document_data,
            headers=headers
        )
        
        # 验证响应
        assert response.status_code == 201
        result = response.json()
        
        assert result["title"] == document_data["title"]
        assert result["status"] == "uploaded"
        assert "id" in result
        assert "created_at" in result
        
        # 验证数据库中的数据
        document_id = result["id"]
        get_response = await test_client.get(
            f"/api/v1/documents/{document_id}",
            headers=headers
        )
        
        assert get_response.status_code == 200
        stored_doc = get_response.json()
        assert stored_doc["title"] == document_data["title"]
    
    async def test_document_processing_workflow(
        self, 
        test_client, 
        auth_token
    ):
        """测试文档处理工作流
        
        验证从文档上传到处理完成的完整业务流程，
        包括状态变更、异步处理等关键步骤。
        """
        # 1. 创建文档
        document_data = {
            "title": "工作流测试文档",
            "content": "工作流测试内容"
        }
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        create_response = await test_client.post(
            "/api/v1/documents/",
            json=document_data,
            headers=headers
        )
        
        assert create_response.status_code == 201
        document_id = create_response.json()["id"]
        
        # 2. 启动处理
        process_response = await test_client.post(
            f"/api/v1/documents/{document_id}/process",
            headers=headers
        )
        
        assert process_response.status_code == 202
        assert "task_id" in process_response.json()
        
        # 3. 检查状态变更
        status_response = await test_client.get(
            f"/api/v1/documents/{document_id}",
            headers=headers
        )
        
        assert status_response.status_code == 200
        assert status_response.json()["status"] in ["processing", "completed"]
        
        # 4. 等待处理完成（模拟异步处理）
        await asyncio.sleep(1)
        
        final_response = await test_client.get(
            f"/api/v1/documents/{document_id}",
            headers=headers
        )
        
        assert final_response.status_code == 200
        final_doc = final_response.json()
        
        # 验证处理结果
        assert final_doc["status"] in ["completed", "failed"]
        if final_doc["status"] == "completed":
            assert "processed_content" in final_doc
```

## 6. 日志规范

### 6.1 日志配置

```python
from loguru import logger
import sys
from datetime import datetime
import os

# 日志配置
def configure_logging(service_name: str, log_level: str = "INFO"):
    """配置日志系统
    
    Args:
        service_name: 服务名称
        log_level: 日志级别
    """
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # 文件输出格式
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{extra[request_id]} | "
        "{message}"
    )
    
    # 控制台处理器
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True
    )
    
    # 文件处理器 - 按日期轮转
    log_dir = f"logs/{service_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger.add(
        f"{log_dir}/app.log",
        format=file_format,
        level=log_level,
        rotation="00:00",  # 每天轮转
        retention="30 days",  # 保留30天
        compression="zip",  # 压缩旧日志
        encoding="utf-8"
    )
    
    # 错误日志单独文件
    logger.add(
        f"{log_dir}/error.log",
        format=file_format,
        level="ERROR",
        rotation="100 MB",
        retention="90 days",
        compression="zip",
        encoding="utf-8"
    )
    
    # 性能日志
    logger.add(
        f"{log_dir}/performance.log",
        format=file_format,
        level="DEBUG",
        filter=lambda record: "performance" in record["extra"],
        rotation="1 week",
        retention="4 weeks",
        compression="zip",
        encoding="utf-8"
    )
    
    logger.info(f"{service_name} 日志系统初始化完成")

# 日志上下文管理
from contextvars import ContextVar
import uuid

request_id_var: ContextVar[str] = ContextVar('request_id', default='')

def bind_request_id() -> str:
    """绑定请求ID到日志上下文"""
    request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    logger.configure(extra={"request_id": request_id})
    return request_id

def get_request_id() -> str:
    """获取当前请求ID"""
    return request_id_var.get()

# 性能监控装饰器
import functools
import time

def log_performance(func_name: str = None):
    """性能监控装饰器
    
    Args:
        func_name: 可选的函数名称，用于日志记录
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.bind(performance=True).info(
                    f"性能监控 - {name} 执行完成, "
                    f"耗时: {duration:.3f}s, "
                    f"请求ID: {get_request_id()}"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.bind(performance=True).error(
                    f"性能监控 - {name} 执行失败, "
                    f"耗时: {duration:.3f}s, "
                    f"错误: {str(e)}, "
                    f"请求ID: {get_request_id()}"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.bind(performance=True).info(
                    f"性能监控 - {name} 执行完成, "
                    f"耗时: {duration:.3f}s, "
                    f"请求ID: {get_request_id()}"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.bind(performance=True).error(
                    f"性能监控 - {name} 执行失败, "
                    f"耗时: {duration:.3f}s, "
                    f"错误: {str(e)}, "
                    f"请求ID: {get_request_id()}"
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 使用示例
@log_performance("文档处理")
async def process_document(document_id: str) -> Dict[str, Any]:
    """处理文档"""
    logger.info(f"开始处理文档: {document_id}")
    
    try:
        # 处理逻辑
        result = await _do_process(document_id)
        
        logger.info(f"文档处理完成: {document_id}")
        return result
        
    except Exception as e:
        logger.error(f"文档处理失败: {document_id}, 错误: {e}")
        raise
```

## 7. 安全规范

### 7.1 输入验证

```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import re
from html import escape
import bleach

class SecurityValidationMixin:
    """安全验证混入类"""
    
    @validator('*', pre=True)
    def sanitize_strings(cls, v):
        """清理字符串输入"""
        if isinstance(v, str):
            # HTML转义
            v = escape(v)
            # 移除潜在危险字符
            v = bleach.clean(v, strip=True)
            # 限制长度
            if len(v) > 10000:
                raise ValueError("输入内容过长")
        return v

class DocumentInput(BaseModel, SecurityValidationMixin):
    """文档输入模型"""
    
    title: str = Field(..., min_length=1, max_length=200)
    content: Optional[str] = Field(None, max_length=50000)
    tags: List[str] = Field(default_factory=list, max_items=10)
    
    @validator('title')
    def validate_title(cls, v):
        """验证标题"""
        if not v.strip():
            raise ValueError("标题不能为空")
        
        # 禁止特殊字符
        if re.search(r'[<>"\'\n\r\t]', v):
            raise ValueError("标题包含非法字符")
        
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        """验证标签"""
        if not isinstance(v, list):
            raise ValueError("标签必须是列表格式")
        
        validated_tags = []
        for tag in v:
            if not isinstance(tag, str):
                continue
            
            tag = tag.strip()
            if not tag:
                continue
                
            if len(tag) > 20:
                raise ValueError("标签长度不能超过20字符")
            
            if re.search(r'[<>"\'\n\r\t]', tag):
                raise ValueError("标签包含非法字符")
            
            validated_tags.append(tag)
        
        return validated_tags[:10]  # 最多10个标签
```

### 7.2 认证和授权

```python
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import secrets

class SecurityManager:
    """安全管理器"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """初始化安全管理器
        
        Args:
            secret_key: JWT密钥
            algorithm: 加密算法
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def create_access_token(
        self, 
        data: dict, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌
        
        Args:
            data: 要编码的数据
            expires_delta: 可选的过期时间间隔
            
        Returns:
            JWT令牌字符串
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        to_encode.update({"jti": secrets.token_urlsafe(16)})  # JWT ID
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.secret_key, 
            algorithm=self.algorithm
        )
        
        logger.debug(f"创建访问令牌成功, 过期时间: {expire}")
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        """验证令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            解码后的数据，验证失败返回None
        """
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # 检查过期时间
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                logger.warning("令牌已过期")
                return None
            
            logger.debug("令牌验证成功")
            return payload
            
        except JWTError as e:
            logger.warning(f"令牌验证失败: {e}")
            return None
    
    def hash_password(self, password: str) -> str:
        """密码哈希
        
        Args:
            password: 明文密码
            
        Returns:
            哈希后的密码
        """
        if len(password) < 8:
            raise ValueError("密码长度不能少于8位")
        
        if not re.search(r'[A-Z]', password):
            raise ValueError("密码必须包含大写字母")
        
        if not re.search(r'[a-z]', password):
            raise ValueError("密码必须包含小写字母")
        
        if not re.search(r'[0-9]', password):
            raise ValueError("密码必须包含数字")
        
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码
        
        Args:
            plain_password: 明文密码
            hashed_password: 哈希密码
            
        Returns:
            验证结果
        """
        return self.pwd_context.verify(plain_password, hashed_password)

# 权限检查装饰器
from functools import wraps
from fastapi import HTTPException, status

def require_permission(permission: str):
    """权限检查装饰器
    
    Args:
        permission: 所需权限
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 从请求上下文获取当前用户
            current_user = get_current_user()
            
            if not current_user:
                logger.warning(f"未认证用户尝试访问需要权限 {permission} 的资源")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="未认证用户"
                )
            
            # 检查权限
            if not has_permission(current_user, permission):
                logger.warning(
                    f"用户 {current_user.id} 尝试访问权限不足的资源: {permission}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限不足，需要权限: {permission}"
                )
            
            logger.debug(f"用户 {current_user.id} 权限验证通过: {permission}")
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
```

## 8. 版本控制规范

### 8.1 Git提交规范

使用Conventional Commits规范：

```bash
# 提交格式
<类型>[可选的作用域]: <描述>

[可选的正文]

[可选的脚注]

# 类型说明
feat: 新功能
fix: 修复问题
docs: 文档更新
style: 代码格式调整（不影响功能）
refactor: 代码重构
perf: 性能优化
test: 测试相关
chore: 构建工具或辅助工具的变动

# 示例
feat(epic-1): 添加文档上传功能

实现了支持PDF、Word、图片格式的文档上传功能，
包括文件验证、格式检查和安全扫描。

Closes #123
```

### 8.2 分支管理策略

```bash
# 主要分支
main        # 生产环境分支，只接受release和hotfix合并
develop     # 开发主分支，集成所有功能
release/*   # 发布准备分支
hotfix/*    # 紧急修复分支

# 功能分支
feature/epic-1/story-1.1    # 按Epic和Story组织
feature/epic-2/story-2.3
bugfix/fix-ocr-error        # 缺陷修复

# 分支操作示例
git checkout develop
git checkout -b feature/epic-1/story-1.4
# 开发完成后
git checkout develop
git merge --no-ff feature/epic-1/story-1.4
git push origin develop
```

---

**文档版本**: v1.0  
**最后更新**: 2025-09-03  
**负责人**: 开发团队  
**审核人**: 架构师