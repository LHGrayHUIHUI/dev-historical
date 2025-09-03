# Story 4.2: 内容审核服务

## 用户故事描述

**作为** 内容管理员  
**我希望** 有一个智能的内容审核系统  
**以便** 能够自动检测和过滤不当内容，确保发布内容的合规性和质量

## 详细需求

### 功能需求

1. **智能内容检测**
   - 文本内容敏感词检测
   - 图片内容识别和审核
   - 视频内容分析和检测
   - 音频内容识别和过滤

2. **多维度审核规则**
   - 政治敏感内容检测
   - 色情暴力内容识别
   - 广告垃圾信息过滤
   - 版权侵权内容检测
   - 虚假信息识别

3. **审核流程管理**
   - 自动审核和人工审核结合
   - 审核结果分级处理
   - 审核历史记录追踪
   - 申诉和复审机制

4. **规则配置管理**
   - 敏感词库管理
   - 审核规则自定义
   - 白名单和黑名单管理
   - 审核策略动态调整

### 非功能需求

1. **性能要求**
   - 文本审核响应时间 < 1秒
   - 图片审核响应时间 < 3秒
   - 视频审核响应时间 < 30秒
   - 支持并发审核请求 > 1000/分钟

2. **准确性要求**
   - 敏感内容检测准确率 > 95%
   - 误报率 < 5%
   - 漏报率 < 2%

3. **可用性要求**
   - 系统可用性 > 99.9%
   - 支持7x24小时服务
   - 故障恢复时间 < 5分钟

## 技术栈

### 后端技术栈

#### 核心框架
- **FastAPI**: Web框架，提供高性能API服务
- **Celery**: 异步任务队列，处理耗时的审核任务
- **Redis**: 缓存和消息队列
- **RQ**: 轻量级任务队列，处理快速审核任务

#### AI/ML框架
- **TensorFlow**: 深度学习框架
- **PyTorch**: 机器学习框架
- **Transformers**: 预训练模型库
- **OpenCV**: 计算机视觉库
- **Pillow**: 图像处理库
- **scikit-learn**: 机器学习工具包

#### 文本处理
- **jieba**: 中文分词
- **NLTK**: 自然语言处理
- **spaCy**: 高级NLP库
- **TextBlob**: 文本分析库

#### 数据存储
- **PostgreSQL**: 主数据库，存储审核记录和配置
- **MongoDB**: 文档数据库，存储非结构化数据
- **InfluxDB**: 时序数据库，存储审核统计数据
- **MinIO**: 对象存储，存储媒体文件

#### 消息队列
- **RabbitMQ**: 消息队列服务
- **Apache Kafka**: 流数据处理

#### 监控和日志
- **Prometheus**: 指标收集
- **Grafana**: 数据可视化
- **ELK Stack**: 日志收集和分析
- **Jaeger**: 分布式追踪

### 前端技术栈

#### 核心框架
- **Vue 3**: 前端框架
- **TypeScript**: 类型安全的JavaScript
- **Pinia**: 状态管理
- **Vue Router 4**: 路由管理

#### UI组件库
- **Element Plus**: UI组件库
- **@vue/composition-api**: 组合式API

#### 工具库
- **Axios**: HTTP客户端
- **Day.js**: 日期处理
- **Lodash**: 工具函数库

#### 图表和可视化
- **ECharts**: 数据可视化
- **Chart.js**: 图表库
- **Vue-ECharts**: Vue ECharts组件

#### 开发工具
- **Vite**: 构建工具
- **ESLint**: 代码检查
- **Prettier**: 代码格式化
- **Vitest**: 单元测试

## 数据模型设计

### PostgreSQL 数据模型

```sql
-- 审核任务表
CREATE TABLE moderation_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) NOT NULL, -- text, image, video, audio
    content_url TEXT,
    content_text TEXT,
    source_platform VARCHAR(100),
    user_id UUID,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, processing, approved, rejected, manual_review
    auto_result JSONB, -- 自动审核结果
    manual_result JSONB, -- 人工审核结果
    final_result VARCHAR(50), -- approved, rejected
    confidence_score DECIMAL(5,4),
    risk_level VARCHAR(20), -- low, medium, high
    violation_types TEXT[], -- 违规类型数组
    reviewer_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP WITH TIME ZONE
);

-- 审核规则表
CREATE TABLE moderation_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    rule_type VARCHAR(50) NOT NULL, -- keyword, regex, ml_model, api_call
    content_types TEXT[] NOT NULL, -- 适用的内容类型
    rule_config JSONB NOT NULL, -- 规则配置
    severity VARCHAR(20) NOT NULL DEFAULT 'medium', -- low, medium, high, critical
    action VARCHAR(50) NOT NULL DEFAULT 'flag', -- flag, block, manual_review
    is_active BOOLEAN DEFAULT true,
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 敏感词库表
CREATE TABLE sensitive_words (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    word VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL, -- politics, violence, pornography, spam, etc.
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    is_regex BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 白名单表
CREATE TABLE whitelists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL, -- user, domain, keyword, ip
    value VARCHAR(500) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- 申诉记录表
CREATE TABLE appeals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES moderation_tasks(id),
    user_id UUID NOT NULL,
    reason TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, approved, rejected
    reviewer_id UUID,
    reviewer_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP WITH TIME ZONE
);

-- 审核统计表
CREATE TABLE moderation_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    content_type VARCHAR(50),
    platform VARCHAR(100),
    total_tasks INTEGER DEFAULT 0,
    auto_approved INTEGER DEFAULT 0,
    auto_rejected INTEGER DEFAULT 0,
    manual_review INTEGER DEFAULT 0,
    appeals INTEGER DEFAULT 0,
    avg_processing_time DECIMAL(10,2), -- 平均处理时间（秒）
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_moderation_tasks_status ON moderation_tasks(status);
CREATE INDEX idx_moderation_tasks_content_type ON moderation_tasks(content_type);
CREATE INDEX idx_moderation_tasks_created_at ON moderation_tasks(created_at);
CREATE INDEX idx_moderation_tasks_user_id ON moderation_tasks(user_id);
CREATE INDEX idx_sensitive_words_category ON sensitive_words(category);
CREATE INDEX idx_sensitive_words_word ON sensitive_words(word);
CREATE INDEX idx_appeals_task_id ON appeals(task_id);
CREATE INDEX idx_appeals_user_id ON appeals(user_id);
CREATE INDEX idx_moderation_stats_date ON moderation_stats(date);
```

### Redis 数据模型

```python
# 审核任务缓存
moderation_task:{task_id} = {
    "id": "task_uuid",
    "status": "processing",
    "progress": 75,
    "current_step": "image_analysis",
    "estimated_time": 30
}

# 敏感词缓存
sensitive_words:{category} = ["word1", "word2", "word3"]

# 审核规则缓存
moderation_rules:active = {
    "text_rules": [...],
    "image_rules": [...],
    "video_rules": [...]
}

# 用户审核统计
user_moderation_stats:{user_id}:{date} = {
    "total_submissions": 10,
    "approved": 8,
    "rejected": 2,
    "violation_types": ["spam", "inappropriate"]
}

# API限流
api_rate_limit:{user_id}:{endpoint} = {
    "count": 100,
    "window_start": 1640995200,
    "limit": 1000
}
```

## 服务架构设计

### 核心服务类

```python
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class ModerationStatus(Enum):
    """审核状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ModerationResult:
    """审核结果数据类"""
    is_approved: bool
    confidence_score: float
    risk_level: RiskLevel
    violation_types: List[str]
    details: Dict[str, Any]
    processing_time: float

class ContentModerationService:
    """内容审核服务主类"""
    
    def __init__(self, db_session, redis_client, ml_models):
        self.db = db_session
        self.redis = redis_client
        self.ml_models = ml_models
        self.text_analyzer = TextAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    async def submit_for_moderation(self, 
                                  content_id: str,
                                  content_type: ContentType,
                                  content_data: Dict[str, Any],
                                  user_id: str,
                                  platform: str = None) -> str:
        """提交内容进行审核"""
        try:
            # 创建审核任务
            task = await self._create_moderation_task(
                content_id=content_id,
                content_type=content_type,
                content_data=content_data,
                user_id=user_id,
                platform=platform
            )
            
            # 异步执行审核
            asyncio.create_task(self._process_moderation_task(task.id))
            
            self.logger.info(f"Moderation task created: {task.id}")
            return task.id
            
        except Exception as e:
            self.logger.error(f"Failed to submit content for moderation: {e}")
            raise
    
    async def _process_moderation_task(self, task_id: str) -> None:
        """处理审核任务"""
        try:
            # 更新任务状态为处理中
            await self._update_task_status(task_id, ModerationStatus.PROCESSING)
            
            # 获取任务详情
            task = await self._get_task_by_id(task_id)
            
            # 根据内容类型选择分析器
            analyzer = self._get_analyzer(task.content_type)
            
            # 执行自动审核
            result = await analyzer.analyze(task.content_data)
            
            # 保存审核结果
            await self._save_moderation_result(task_id, result)
            
            # 根据结果决定最终状态
            final_status = self._determine_final_status(result)
            await self._update_task_status(task_id, final_status)
            
            # 发送通知
            await self._send_moderation_notification(task_id, final_status)
            
        except Exception as e:
            self.logger.error(f"Failed to process moderation task {task_id}: {e}")
            await self._update_task_status(task_id, ModerationStatus.MANUAL_REVIEW)
    
    async def get_moderation_result(self, task_id: str) -> Dict[str, Any]:
        """获取审核结果"""
        try:
            task = await self._get_task_by_id(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            return {
                "task_id": task.id,
                "status": task.status,
                "result": task.final_result,
                "confidence_score": task.confidence_score,
                "risk_level": task.risk_level,
                "violation_types": task.violation_types,
                "created_at": task.created_at,
                "reviewed_at": task.reviewed_at
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get moderation result: {e}")
            raise
    
    async def submit_appeal(self, 
                          task_id: str, 
                          user_id: str, 
                          reason: str) -> str:
        """提交申诉"""
        try:
            appeal = await self._create_appeal(
                task_id=task_id,
                user_id=user_id,
                reason=reason
            )
            
            # 通知审核员
            await self._notify_reviewers(appeal.id)
            
            self.logger.info(f"Appeal submitted: {appeal.id}")
            return appeal.id
            
        except Exception as e:
            self.logger.error(f"Failed to submit appeal: {e}")
            raise
    
    def _get_analyzer(self, content_type: ContentType):
        """根据内容类型获取分析器"""
        analyzers = {
            ContentType.TEXT: self.text_analyzer,
            ContentType.IMAGE: self.image_analyzer,
            ContentType.VIDEO: self.video_analyzer,
            ContentType.AUDIO: self.audio_analyzer
        }
        return analyzers.get(content_type)
    
    def _determine_final_status(self, result: ModerationResult) -> ModerationStatus:
        """根据审核结果确定最终状态"""
        if result.risk_level == RiskLevel.CRITICAL:
            return ModerationStatus.REJECTED
        elif result.risk_level == RiskLevel.HIGH:
            return ModerationStatus.MANUAL_REVIEW
        elif result.confidence_score > 0.9:
            return ModerationStatus.APPROVED if result.is_approved else ModerationStatus.REJECTED
        else:
            return ModerationStatus.MANUAL_REVIEW

class TextAnalyzer:
    """文本内容分析器"""
    
    def __init__(self):
        self.sensitive_words_detector = SensitiveWordsDetector()
        self.ml_classifier = MLTextClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    async def analyze(self, content_data: Dict[str, Any]) -> ModerationResult:
        """分析文本内容"""
        text = content_data.get('text', '')
        start_time = time.time()
        
        # 敏感词检测
        sensitive_result = await self.sensitive_words_detector.detect(text)
        
        # 机器学习分类
        ml_result = await self.ml_classifier.classify(text)
        
        # 情感分析
        sentiment_result = await self.sentiment_analyzer.analyze(text)
        
        # 综合分析结果
        violation_types = []
        risk_score = 0.0
        
        if sensitive_result['violations']:
            violation_types.extend(sensitive_result['types'])
            risk_score = max(risk_score, sensitive_result['max_severity'])
        
        if ml_result['is_violation']:
            violation_types.extend(ml_result['categories'])
            risk_score = max(risk_score, ml_result['confidence'])
        
        # 确定风险等级
        risk_level = self._calculate_risk_level(risk_score)
        
        # 判断是否通过
        is_approved = len(violation_types) == 0 and risk_score < 0.3
        
        processing_time = time.time() - start_time
        
        return ModerationResult(
            is_approved=is_approved,
            confidence_score=1.0 - risk_score,
            risk_level=risk_level,
            violation_types=violation_types,
            details={
                'sensitive_words': sensitive_result,
                'ml_classification': ml_result,
                'sentiment': sentiment_result
            },
            processing_time=processing_time
        )
    
    def _calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """计算风险等级"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

class ImageAnalyzer:
    """图片内容分析器"""
    
    def __init__(self):
        self.nsfw_detector = NSFWDetector()
        self.object_detector = ObjectDetector()
        self.text_extractor = ImageTextExtractor()
        self.face_detector = FaceDetector()
    
    async def analyze(self, content_data: Dict[str, Any]) -> ModerationResult:
        """分析图片内容"""
        image_url = content_data.get('url')
        start_time = time.time()
        
        # 下载图片
        image_data = await self._download_image(image_url)
        
        # NSFW检测
        nsfw_result = await self.nsfw_detector.detect(image_data)
        
        # 物体检测
        object_result = await self.object_detector.detect(image_data)
        
        # 文字提取和分析
        text_result = await self.text_extractor.extract_and_analyze(image_data)
        
        # 人脸检测
        face_result = await self.face_detector.detect(image_data)
        
        # 综合分析结果
        violation_types = []
        risk_score = 0.0
        
        if nsfw_result['is_nsfw']:
            violation_types.append('nsfw')
            risk_score = max(risk_score, nsfw_result['confidence'])
        
        if object_result['violations']:
            violation_types.extend(object_result['types'])
            risk_score = max(risk_score, object_result['max_confidence'])
        
        if text_result['violations']:
            violation_types.extend(text_result['types'])
            risk_score = max(risk_score, text_result['risk_score'])
        
        # 确定风险等级
        risk_level = self._calculate_risk_level(risk_score)
        
        # 判断是否通过
        is_approved = len(violation_types) == 0 and risk_score < 0.3
        
        processing_time = time.time() - start_time
        
        return ModerationResult(
            is_approved=is_approved,
            confidence_score=1.0 - risk_score,
            risk_level=risk_level,
            violation_types=violation_types,
            details={
                'nsfw_detection': nsfw_result,
                'object_detection': object_result,
                'text_extraction': text_result,
                'face_detection': face_result
            },
            processing_time=processing_time
        )

class RuleEngine:
    """规则引擎"""
    
    def __init__(self, db_session, redis_client):
        self.db = db_session
        self.redis = redis_client
        self.rules_cache = {}
    
    async def load_rules(self) -> None:
        """加载审核规则"""
        try:
            # 从数据库加载规则
            rules = await self._fetch_active_rules()
            
            # 按内容类型分组
            self.rules_cache = {
                'text': [r for r in rules if 'text' in r.content_types],
                'image': [r for r in rules if 'image' in r.content_types],
                'video': [r for r in rules if 'video' in r.content_types],
                'audio': [r for r in rules if 'audio' in r.content_types]
            }
            
            # 缓存到Redis
            await self.redis.setex(
                'moderation_rules:active',
                3600,  # 1小时过期
                json.dumps(self.rules_cache, default=str)
            )
            
        except Exception as e:
            logging.error(f"Failed to load moderation rules: {e}")
            raise
    
    async def evaluate_content(self, 
                             content_type: ContentType, 
                             content_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估内容是否违规"""
        violations = []
        rules = self.rules_cache.get(content_type.value, [])
        
        for rule in rules:
            try:
                result = await self._evaluate_rule(rule, content_data)
                if result['is_violation']:
                    violations.append({
                        'rule_id': rule.id,
                        'rule_name': rule.name,
                        'severity': rule.severity,
                        'confidence': result['confidence'],
                        'details': result['details']
                    })
            except Exception as e:
                logging.error(f"Failed to evaluate rule {rule.id}: {e}")
        
        return violations
```

## API设计

### 内容审核API

```python
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/moderation", tags=["moderation"])

class ModerationRequest(BaseModel):
    """审核请求模型"""
    content_id: str
    content_type: str  # text, image, video, audio
    content_data: dict
    platform: Optional[str] = None
    priority: Optional[str] = "normal"  # low, normal, high, urgent

class ModerationResponse(BaseModel):
    """审核响应模型"""
    task_id: str
    status: str
    message: str

@router.post("/submit", response_model=ModerationResponse)
async def submit_content_for_moderation(
    request: ModerationRequest,
    current_user: dict = Depends(get_current_user),
    moderation_service: ContentModerationService = Depends(get_moderation_service)
):
    """提交内容进行审核"""
    try:
        task_id = await moderation_service.submit_for_moderation(
            content_id=request.content_id,
            content_type=ContentType(request.content_type),
            content_data=request.content_data,
            user_id=current_user['id'],
            platform=request.platform
        )
        
        return ModerationResponse(
            task_id=task_id,
            status="submitted",
            message="Content submitted for moderation successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/result/{task_id}")
async def get_moderation_result(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    moderation_service: ContentModerationService = Depends(get_moderation_service)
):
    """获取审核结果"""
    try:
        result = await moderation_service.get_moderation_result(task_id)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-file")
async def upload_file_for_moderation(
    file: UploadFile = File(...),
    content_type: str = "image",
    current_user: dict = Depends(get_current_user),
    moderation_service: ContentModerationService = Depends(get_moderation_service)
):
    """上传文件进行审核"""
    try:
        # 保存文件
        file_url = await save_uploaded_file(file)
        
        # 提交审核
        task_id = await moderation_service.submit_for_moderation(
            content_id=f"upload_{file.filename}",
            content_type=ContentType(content_type),
            content_data={"url": file_url, "filename": file.filename},
            user_id=current_user['id']
        )
        
        return {
            "task_id": task_id,
            "file_url": file_url,
            "status": "submitted"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks")
async def get_moderation_tasks(
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    page: int = 1,
    size: int = 20,
    current_user: dict = Depends(get_current_user),
    moderation_service: ContentModerationService = Depends(get_moderation_service)
):
    """获取审核任务列表"""
    try:
        tasks = await moderation_service.get_user_tasks(
            user_id=current_user['id'],
            status=status,
            content_type=content_type,
            page=page,
            size=size
        )
        return tasks
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 申诉管理API

```python
class AppealRequest(BaseModel):
    """申诉请求模型"""
    task_id: str
    reason: str
    additional_info: Optional[dict] = None

@router.post("/appeal", response_model=dict)
async def submit_appeal(
    request: AppealRequest,
    current_user: dict = Depends(get_current_user),
    moderation_service: ContentModerationService = Depends(get_moderation_service)
):
    """提交申诉"""
    try:
        appeal_id = await moderation_service.submit_appeal(
            task_id=request.task_id,
            user_id=current_user['id'],
            reason=request.reason
        )
        
        return {
            "appeal_id": appeal_id,
            "status": "submitted",
            "message": "Appeal submitted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/appeals")
async def get_user_appeals(
    status: Optional[str] = None,
    page: int = 1,
    size: int = 20,
    current_user: dict = Depends(get_current_user),
    moderation_service: ContentModerationService = Depends(get_moderation_service)
):
    """获取用户申诉列表"""
    try:
        appeals = await moderation_service.get_user_appeals(
            user_id=current_user['id'],
            status=status,
            page=page,
            size=size
        )
        return appeals
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 规则管理API

```python
class RuleRequest(BaseModel):
    """规则请求模型"""
    name: str
    description: Optional[str] = None
    rule_type: str  # keyword, regex, ml_model, api_call
    content_types: List[str]
    rule_config: dict
    severity: str = "medium"  # low, medium, high, critical
    action: str = "flag"  # flag, block, manual_review

@router.post("/rules", dependencies=[Depends(require_admin)])
async def create_moderation_rule(
    request: RuleRequest,
    current_user: dict = Depends(get_current_user),
    rule_service: RuleManagementService = Depends(get_rule_service)
):
    """创建审核规则"""
    try:
        rule_id = await rule_service.create_rule(
            name=request.name,
            description=request.description,
            rule_type=request.rule_type,
            content_types=request.content_types,
            rule_config=request.rule_config,
            severity=request.severity,
            action=request.action,
            created_by=current_user['id']
        )
        
        return {
            "rule_id": rule_id,
            "status": "created",
            "message": "Moderation rule created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rules")
async def get_moderation_rules(
    rule_type: Optional[str] = None,
    content_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    page: int = 1,
    size: int = 20,
    rule_service: RuleManagementService = Depends(get_rule_service)
):
    """获取审核规则列表"""
    try:
        rules = await rule_service.get_rules(
            rule_type=rule_type,
            content_type=content_type,
            is_active=is_active,
            page=page,
            size=size
        )
        return rules
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/rules/{rule_id}", dependencies=[Depends(require_admin)])
async def update_moderation_rule(
    rule_id: str,
    request: RuleRequest,
    current_user: dict = Depends(get_current_user),
    rule_service: RuleManagementService = Depends(get_rule_service)
):
    """更新审核规则"""
    try:
        await rule_service.update_rule(
            rule_id=rule_id,
            **request.dict()
        )
        
        return {
            "status": "updated",
            "message": "Moderation rule updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rules/{rule_id}", dependencies=[Depends(require_admin)])
async def delete_moderation_rule(
    rule_id: str,
    rule_service: RuleManagementService = Depends(get_rule_service)
):
    """删除审核规则"""
    try:
        await rule_service.delete_rule(rule_id)
        
        return {
            "status": "deleted",
            "message": "Moderation rule deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 统计分析API

```python
@router.get("/stats/overview")
async def get_moderation_overview(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    platform: Optional[str] = None,
    stats_service: ModerationStatsService = Depends(get_stats_service)
):
    """获取审核概览统计"""
    try:
        stats = await stats_service.get_overview_stats(
            start_date=start_date,
            end_date=end_date,
            platform=platform
        )
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/trends")
async def get_moderation_trends(
    period: str = "7d",  # 1d, 7d, 30d, 90d
    content_type: Optional[str] = None,
    stats_service: ModerationStatsService = Depends(get_stats_service)
):
    """获取审核趋势数据"""
    try:
        trends = await stats_service.get_trend_stats(
            period=period,
            content_type=content_type
        )
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/violations")
async def get_violation_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stats_service: ModerationStatsService = Depends(get_stats_service)
):
    """获取违规类型统计"""
    try:
        violations = await stats_service.get_violation_stats(
            start_date=start_date,
            end_date=end_date
        )
        return violations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 前端组件设计

### 1. 内容审核主页面

```vue
<template>
  <div class="moderation-dashboard">
    <!-- 统计卡片 -->
    <div class="stats-cards">
      <el-row :gutter="20">
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-value">{{ stats.totalTasks }}</div>
              <div class="stat-label">总审核任务</div>
            </div>
            <div class="stat-icon">
              <el-icon><Document /></el-icon>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-value">{{ stats.approvedTasks }}</div>
              <div class="stat-label">已通过</div>
            </div>
            <div class="stat-icon success">
              <el-icon><Check /></el-icon>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-value">{{ stats.rejectedTasks }}</div>
              <div class="stat-label">已拒绝</div>
            </div>
            <div class="stat-icon danger">
              <el-icon><Close /></el-icon>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card class="stat-card">
            <div class="stat-content">
              <div class="stat-value">{{ stats.pendingTasks }}</div>
              <div class="stat-label">待审核</div>
            </div>
            <div class="stat-icon warning">
              <el-icon><Clock /></el-icon>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 筛选和操作栏 -->
    <el-card class="filter-card">
      <div class="filter-row">
        <div class="filter-left">
          <el-select v-model="filters.status" placeholder="选择状态" clearable>
            <el-option label="全部" value="" />
            <el-option label="待审核" value="pending" />
            <el-option label="审核中" value="processing" />
            <el-option label="已通过" value="approved" />
            <el-option label="已拒绝" value="rejected" />
            <el-option label="人工审核" value="manual_review" />
          </el-select>
          
          <el-select v-model="filters.contentType" placeholder="内容类型" clearable>
            <el-option label="全部" value="" />
            <el-option label="文本" value="text" />
            <el-option label="图片" value="image" />
            <el-option label="视频" value="video" />
            <el-option label="音频" value="audio" />
          </el-select>
          
          <el-date-picker
            v-model="filters.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            format="YYYY-MM-DD"
            value-format="YYYY-MM-DD"
          />
          
          <el-input
            v-model="filters.keyword"
            placeholder="搜索内容ID或用户ID"
            clearable
            style="width: 200px"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </div>
        
        <div class="filter-right">
          <el-button type="primary" @click="searchTasks">
            <el-icon><Search /></el-icon>
            搜索
          </el-button>
          <el-button @click="resetFilters">
            <el-icon><Refresh /></el-icon>
            重置
          </el-button>
          <el-button type="success" @click="showSubmitDialog = true">
            <el-icon><Plus /></el-icon>
            提交审核
          </el-button>
        </div>
      </div>
    </el-card>

    <!-- 任务列表 -->
    <el-card class="table-card">
      <el-table
        v-loading="loading"
        :data="tasks"
        stripe
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="55" />
        
        <el-table-column prop="contentId" label="内容ID" width="150">
          <template #default="{ row }">
            <el-link type="primary" @click="viewTaskDetail(row)">
              {{ row.contentId }}
            </el-link>
          </template>
        </el-table-column>
        
        <el-table-column prop="contentType" label="类型" width="100">
          <template #default="{ row }">
            <el-tag :type="getContentTypeColor(row.contentType)">
              {{ getContentTypeLabel(row.contentType) }}
            </el-tag>
          </template>
        </el-table-column>
        
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusColor(row.status)">
              {{ getStatusLabel(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        
        <el-table-column prop="riskLevel" label="风险等级" width="100">
          <template #default="{ row }">
            <el-tag :type="getRiskLevelColor(row.riskLevel)" v-if="row.riskLevel">
              {{ getRiskLevelLabel(row.riskLevel) }}
            </el-tag>
          </template>
        </el-table-column>
        
        <el-table-column prop="confidenceScore" label="置信度" width="100">
          <template #default="{ row }">
            <span v-if="row.confidenceScore">
              {{ (row.confidenceScore * 100).toFixed(1) }}%
            </span>
          </template>
        </el-table-column>
        
        <el-table-column prop="violationTypes" label="违规类型" width="200">
          <template #default="{ row }">
            <el-tag
              v-for="type in row.violationTypes"
              :key="type"
              size="small"
              type="danger"
              style="margin-right: 5px"
            >
              {{ type }}
            </el-tag>
          </template>
        </el-table-column>
        
        <el-table-column prop="platform" label="平台" width="100" />
        
        <el-table-column prop="createdAt" label="创建时间" width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.createdAt) }}
          </template>
        </el-table-column>
        
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button size="small" @click="viewTaskDetail(row)">
              查看
            </el-button>
            <el-button
              v-if="row.status === 'rejected'"
              size="small"
              type="warning"
              @click="showAppealDialog(row)"
            >
              申诉
            </el-button>
            <el-button
              v-if="canResubmit(row)"
              size="small"
              type="success"
              @click="resubmitTask(row)"
            >
              重新提交
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <!-- 分页 -->
      <div class="pagination-wrapper">
        <el-pagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.size"
          :total="pagination.total"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 提交审核对话框 -->
    <SubmitModerationDialog
      v-model="showSubmitDialog"
      @success="handleSubmitSuccess"
    />
    
    <!-- 任务详情对话框 -->
    <TaskDetailDialog
      v-model="showDetailDialog"
      :task="selectedTask"
    />
    
    <!-- 申诉对话框 -->
    <AppealDialog
      v-model="showAppealDialog"
      :task="selectedTask"
      @success="handleAppealSuccess"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Document, Check, Close, Clock, Search, Refresh, Plus } from '@element-plus/icons-vue'
import { getModerationTasks, getModerationStats } from '@/api/moderation'
import SubmitModerationDialog from './components/SubmitModerationDialog.vue'
import TaskDetailDialog from './components/TaskDetailDialog.vue'
import AppealDialog from './components/AppealDialog.vue'
import { formatDateTime } from '@/utils/date'

// 响应式数据
const loading = ref(false)
const tasks = ref([])
const selectedTasks = ref([])
const selectedTask = ref(null)
const showSubmitDialog = ref(false)
const showDetailDialog = ref(false)
const showAppealDialog = ref(false)

// 统计数据
const stats = reactive({
  totalTasks: 0,
  approvedTasks: 0,
  rejectedTasks: 0,
  pendingTasks: 0
})

// 筛选条件
const filters = reactive({
  status: '',
  contentType: '',
  dateRange: [],
  keyword: ''
})

// 分页数据
const pagination = reactive({
  page: 1,
  size: 20,
  total: 0
})

// 计算属性
const hasSelection = computed(() => selectedTasks.value.length > 0)

// 生命周期
onMounted(() => {
  loadStats()
  loadTasks()
})

// 方法
const loadStats = async () => {
  try {
    const response = await getModerationStats()
    Object.assign(stats, response.data)
  } catch (error) {
    console.error('Failed to load stats:', error)
  }
}

const loadTasks = async () => {
  loading.value = true
  try {
    const params = {
      ...filters,
      page: pagination.page,
      size: pagination.size
    }
    
    const response = await getModerationTasks(params)
    tasks.value = response.data.items
    pagination.total = response.data.total
  } catch (error) {
    ElMessage.error('加载任务列表失败')
    console.error('Failed to load tasks:', error)
  } finally {
    loading.value = false
  }
}

const searchTasks = () => {
  pagination.page = 1
  loadTasks()
}

const resetFilters = () => {
  Object.assign(filters, {
    status: '',
    contentType: '',
    dateRange: [],
    keyword: ''
  })
  searchTasks()
}

const handleSelectionChange = (selection: any[]) => {
  selectedTasks.value = selection
}

const handleSizeChange = (size: number) => {
  pagination.size = size
  loadTasks()
}

const handleCurrentChange = (page: number) => {
  pagination.page = page
  loadTasks()
}

const viewTaskDetail = (task: any) => {
  selectedTask.value = task
  showDetailDialog.value = true
}

const showAppealDialog = (task: any) => {
  selectedTask.value = task
  showAppealDialog.value = true
}

const canResubmit = (task: any) => {
  return ['rejected', 'manual_review'].includes(task.status)
}

const resubmitTask = async (task: any) => {
  try {
    await ElMessageBox.confirm(
      '确定要重新提交此任务进行审核吗？',
      '确认操作',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    // 调用重新提交API
    // await resubmitModerationTask(task.id)
    
    ElMessage.success('任务已重新提交')
    loadTasks()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('重新提交失败')
    }
  }
}

const handleSubmitSuccess = () => {
  loadStats()
  loadTasks()
}

const handleAppealSuccess = () => {
  loadTasks()
}

// 工具方法
const getContentTypeLabel = (type: string) => {
  const labels = {
    text: '文本',
    image: '图片',
    video: '视频',
    audio: '音频'
  }
  return labels[type] || type
}

const getContentTypeColor = (type: string) => {
  const colors = {
    text: '',
    image: 'success',
    video: 'warning',
    audio: 'info'
  }
  return colors[type] || ''
}

const getStatusLabel = (status: string) => {
  const labels = {
    pending: '待审核',
    processing: '审核中',
    approved: '已通过',
    rejected: '已拒绝',
    manual_review: '人工审核'
  }
  return labels[status] || status
}

const getStatusColor = (status: string) => {
  const colors = {
    pending: 'info',
    processing: 'warning',
    approved: 'success',
    rejected: 'danger',
    manual_review: 'warning'
  }
  return colors[status] || ''
}

const getRiskLevelLabel = (level: string) => {
  const labels = {
    low: '低',
    medium: '中',
    high: '高',
    critical: '严重'
  }
  return labels[level] || level
}

const getRiskLevelColor = (level: string) => {
  const colors = {
    low: 'success',
    medium: 'warning',
    high: 'danger',
    critical: 'danger'
  }
  return colors[level] || ''
}
</script>

<style scoped>
.moderation-dashboard {
  padding: 20px;
}

.stats-cards {
  margin-bottom: 20px;
}

.stat-card {
  position: relative;
  overflow: hidden;
}

.stat-card :deep(.el-card__body) {
  padding: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #303133;
  line-height: 1;
}

.stat-label {
  font-size: 14px;
  color: #909399;
  margin-top: 8px;
}

.stat-icon {
  font-size: 40px;
  color: #409eff;
  opacity: 0.8;
}

.stat-icon.success {
  color: #67c23a;
}

.stat-icon.danger {
  color: #f56c6c;
}

.stat-icon.warning {
  color: #e6a23c;
}

.filter-card {
  margin-bottom: 20px;
}

.filter-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 15px;
}

.filter-left {
  display: flex;
  gap: 15px;
  flex: 1;
}

.filter-right {
  display: flex;
  gap: 10px;
}

.table-card {
  margin-bottom: 20px;
}

.pagination-wrapper {
  margin-top: 20px;
  text-align: right;
}

.el-select {
  width: 150px;
}
</style>
```

### 2. 提交审核对话框

```vue
<template>
  <el-dialog
    v-model="visible"
    title="提交内容审核"
    width="600px"
    :before-close="handleClose"
  >
    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-width="100px"
    >
      <el-form-item label="内容类型" prop="contentType">
        <el-radio-group v-model="form.contentType">
          <el-radio label="text">文本</el-radio>
          <el-radio label="image">图片</el-radio>
          <el-radio label="video">视频</el-radio>
          <el-radio label="audio">音频</el-radio>
        </el-radio-group>
      </el-form-item>
      
      <el-form-item label="内容ID" prop="contentId">
        <el-input
          v-model="form.contentId"
          placeholder="请输入内容唯一标识"
        />
      </el-form-item>
      
      <!-- 文本内容 -->
      <el-form-item
        v-if="form.contentType === 'text'"
        label="文本内容"
        prop="textContent"
      >
        <el-input
          v-model="form.textContent"
          type="textarea"
          :rows="6"
          placeholder="请输入要审核的文本内容"
          maxlength="5000"
          show-word-limit
        />
      </el-form-item>
      
      <!-- 文件上传 -->
      <el-form-item
        v-if="form.contentType !== 'text'"
        label="文件上传"
        prop="fileUrl"
      >
        <el-upload
          ref="uploadRef"
          :action="uploadAction"
          :headers="uploadHeaders"
          :on-success="handleUploadSuccess"
          :on-error="handleUploadError"
          :before-upload="beforeUpload"
          :file-list="fileList"
          :limit="1"
          drag
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            拖拽文件到此处或<em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              {{ getUploadTip() }}
            </div>
          </template>
        </el-upload>
      </el-form-item>
      
      <el-form-item label="来源平台" prop="platform">
        <el-select v-model="form.platform" placeholder="选择来源平台" clearable>
          <el-option label="微博" value="weibo" />
          <el-option label="微信" value="wechat" />
          <el-option label="抖音" value="douyin" />
          <el-option label="头条" value="toutiao" />
          <el-option label="百家号" value="baijiahao" />
          <el-option label="其他" value="other" />
        </el-select>
      </el-form-item>
      
      <el-form-item label="优先级" prop="priority">
        <el-select v-model="form.priority" placeholder="选择优先级">
          <el-option label="低" value="low" />
          <el-option label="普通" value="normal" />
          <el-option label="高" value="high" />
          <el-option label="紧急" value="urgent" />
        </el-select>
      </el-form-item>
      
      <el-form-item label="备注" prop="notes">
        <el-input
          v-model="form.notes"
          type="textarea"
          :rows="3"
          placeholder="可选：添加审核备注信息"
          maxlength="500"
          show-word-limit
        />
      </el-form-item>
    </el-form>
    
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="handleClose">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="handleSubmit">
          提交审核
        </el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { submitModerationTask } from '@/api/moderation'
import { getUploadToken } from '@/api/upload'

// Props
interface Props {
  modelValue: boolean
}

const props = defineProps<Props>()

// Emits
interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'success', taskId: string): void
}

const emit = defineEmits<Emits>()

// 响应式数据
const formRef = ref()
const uploadRef = ref()
const submitting = ref(false)
const fileList = ref([])

const form = reactive({
  contentType: 'text',
  contentId: '',
  textContent: '',
  fileUrl: '',
  platform: '',
  priority: 'normal',
  notes: ''
})

// 表单验证规则
const rules = {
  contentType: [
    { required: true, message: '请选择内容类型', trigger: 'change' }
  ],
  contentId: [
    { required: true, message: '请输入内容ID', trigger: 'blur' },
    { min: 1, max: 255, message: '长度在 1 到 255 个字符', trigger: 'blur' }
  ],
  textContent: [
    {
      validator: (rule: any, value: string, callback: Function) => {
        if (form.contentType === 'text' && !value) {
          callback(new Error('请输入文本内容'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ],
  fileUrl: [
    {
      validator: (rule: any, value: string, callback: Function) => {
        if (form.contentType !== 'text' && !value) {
          callback(new Error('请上传文件'))
        } else {
          callback()
        }
      },
      trigger: 'change'
    }
  ]
}

// 计算属性
const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const uploadAction = computed(() => {
  return '/api/v1/upload/file'
})

const uploadHeaders = computed(() => {
  return {
    'Authorization': `Bearer ${localStorage.getItem('token')}`
  }
})

// 监听内容类型变化
watch(() => form.contentType, (newType) => {
  // 清空相关字段
  form.textContent = ''
  form.fileUrl = ''
  fileList.value = []
  
  // 清除验证错误
  formRef.value?.clearValidate(['textContent', 'fileUrl'])
})

// 方法
const getUploadTip = () => {
  const tips = {
    image: '支持 jpg、png、gif 格式，文件大小不超过 10MB',
    video: '支持 mp4、avi、mov 格式，文件大小不超过 100MB',
    audio: '支持 mp3、wav、aac 格式，文件大小不超过 50MB'
  }
  return tips[form.contentType] || ''
}

const beforeUpload = (file: File) => {
  const isValidType = checkFileType(file)
  const isValidSize = checkFileSize(file)
  
  if (!isValidType) {
    ElMessage.error('文件格式不正确')
    return false
  }
  
  if (!isValidSize) {
    ElMessage.error('文件大小超出限制')
    return false
  }
  
  return true
}

const checkFileType = (file: File) => {
  const typeMap = {
    image: ['image/jpeg', 'image/png', 'image/gif'],
    video: ['video/mp4', 'video/avi', 'video/quicktime'],
    audio: ['audio/mpeg', 'audio/wav', 'audio/aac']
  }
  
  const allowedTypes = typeMap[form.contentType] || []
  return allowedTypes.includes(file.type)
}

const checkFileSize = (file: File) => {
  const sizeMap = {
    image: 10 * 1024 * 1024, // 10MB
    video: 100 * 1024 * 1024, // 100MB
    audio: 50 * 1024 * 1024 // 50MB
  }
  
  const maxSize = sizeMap[form.contentType] || 10 * 1024 * 1024
  return file.size <= maxSize
}

const handleUploadSuccess = (response: any, file: any) => {
  form.fileUrl = response.data.url
  ElMessage.success('文件上传成功')
}

const handleUploadError = (error: any) => {
  ElMessage.error('文件上传失败')
  console.error('Upload error:', error)
}

const handleSubmit = async () => {
  try {
    await formRef.value.validate()
    
    submitting.value = true
    
    const contentData = form.contentType === 'text' 
      ? { text: form.textContent }
      : { url: form.fileUrl }
    
    const response = await submitModerationTask({
      contentId: form.contentId,
      contentType: form.contentType,
      contentData,
      platform: form.platform,
      priority: form.priority,
      notes: form.notes
    })
    
    ElMessage.success('审核任务提交成功')
    emit('success', response.data.taskId)
    handleClose()
    
  } catch (error) {
    if (error.message) {
      ElMessage.error(error.message)
    }
  } finally {
    submitting.value = false
  }
}

const handleClose = () => {
  // 重置表单
  formRef.value?.resetFields()
  fileList.value = []
  
  visible.value = false
}
</script>

<style scoped>
.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.el-upload {
  width: 100%;
}

.el-upload__tip {
  color: #909399;
  font-size: 12px;
  margin-top: 5px;
}
</style>
```

### 3. 任务详情对话框

```vue
<template>
  <el-dialog
    v-model="visible"
    title="审核任务详情"
    width="800px"
    :before-close="handleClose"
  >
    <div v-if="task" class="task-detail">
      <!-- 基本信息 -->
      <el-card class="info-card">
        <template #header>
          <span class="card-title">基本信息</span>
        </template>
        
        <el-descriptions :column="2" border>
          <el-descriptions-item label="任务ID">
            {{ task.id }}
          </el-descriptions-item>
          <el-descriptions-item label="内容ID">
            {{ task.contentId }}
          </el-descriptions-item>
          <el-descriptions-item label="内容类型">
            <el-tag :type="getContentTypeColor(task.contentType)">
              {{ getContentTypeLabel(task.contentType) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusColor(task.status)">
              {{ getStatusLabel(task.status) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="风险等级">
            <el-tag :type="getRiskLevelColor(task.riskLevel)" v-if="task.riskLevel">
              {{ getRiskLevelLabel(task.riskLevel) }}
            </el-tag>
            <span v-else>-</span>
          </el-descriptions-item>
          <el-descriptions-item label="置信度">
            <span v-if="task.confidenceScore">
              {{ (task.confidenceScore * 100).toFixed(1) }}%
            </span>
            <span v-else>-</span>
          </el-descriptions-item>
          <el-descriptions-item label="来源平台">
            {{ task.platform || '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ formatDateTime(task.createdAt) }}
          </el-descriptions-item>
          <el-descriptions-item label="审核时间">
            {{ task.reviewedAt ? formatDateTime(task.reviewedAt) : '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="处理时长">
            {{ getProcessingDuration(task) }}
          </el-descriptions-item>
        </el-descriptions>
      </el-card>
      
      <!-- 内容预览 -->
      <el-card class="content-card">
        <template #header>
          <span class="card-title">内容预览</span>
        </template>
        
        <!-- 文本内容 -->
        <div v-if="task.contentType === 'text'" class="text-content">
          <el-input
            :model-value="task.contentText"
            type="textarea"
            :rows="8"
            readonly
          />
        </div>
        
        <!-- 图片内容 -->
        <div v-else-if="task.contentType === 'image'" class="image-content">
          <el-image
            :src="task.contentUrl"
            :preview-src-list="[task.contentUrl]"
            fit="contain"
            style="max-width: 100%; max-height: 400px"
          />
        </div>
        
        <!-- 视频内容 -->
        <div v-else-if="task.contentType === 'video'" class="video-content">
          <video
            :src="task.contentUrl"
            controls
            style="max-width: 100%; max-height: 400px"
          >
            您的浏览器不支持视频播放
          </video>
        </div>
        
        <!-- 音频内容 -->
        <div v-else-if="task.contentType === 'audio'" class="audio-content">
          <audio :src="task.contentUrl" controls style="width: 100%">
            您的浏览器不支持音频播放
          </audio>
        </div>
      </el-card>
      
      <!-- 审核结果 -->
      <el-card v-if="task.autoResult || task.manualResult" class="result-card">
        <template #header>
          <span class="card-title">审核结果</span>
        </template>
        
        <!-- 自动审核结果 -->
        <div v-if="task.autoResult" class="auto-result">
          <h4>自动审核结果</h4>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="审核结论">
              <el-tag :type="task.autoResult.isApproved ? 'success' : 'danger'">
                {{ task.autoResult.isApproved ? '通过' : '拒绝' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="置信度">
              {{ (task.autoResult.confidenceScore * 100).toFixed(1) }}%
            </el-descriptions-item>
            <el-descriptions-item label="违规类型" v-if="task.violationTypes?.length">
              <el-tag
                v-for="type in task.violationTypes"
                :key="type"
                type="danger"
                size="small"
                style="margin-right: 5px"
              >
                {{ type }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="处理时间">
              {{ task.autoResult.processingTime?.toFixed(2) }}秒
            </el-descriptions-item>
          </el-descriptions>
          
          <!-- 详细分析结果 -->
          <div v-if="task.autoResult.details" class="analysis-details">
            <h5>详细分析</h5>
            <el-collapse>
              <el-collapse-item
                v-for="(detail, key) in task.autoResult.details"
                :key="key"
                :title="getAnalysisTitle(key)"
              >
                <pre>{{ JSON.stringify(detail, null, 2) }}</pre>
              </el-collapse-item>
            </el-collapse>
          </div>
        </div>
        
        <!-- 人工审核结果 -->
        <div v-if="task.manualResult" class="manual-result">
          <h4>人工审核结果</h4>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="审核结论">
              <el-tag :type="task.manualResult.isApproved ? 'success' : 'danger'">
                {{ task.manualResult.isApproved ? '通过' : '拒绝' }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="审核员">
              {{ task.manualResult.reviewerName || task.reviewerId }}
            </el-descriptions-item>
            <el-descriptions-item label="审核意见">
              {{ task.manualResult.comments || '-' }}
            </el-descriptions-item>
            <el-descriptions-item label="审核时间">
              {{ formatDateTime(task.manualResult.reviewedAt) }}
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </el-card>
      
      <!-- 申诉记录 -->
      <el-card v-if="appeals.length" class="appeal-card">
        <template #header>
          <span class="card-title">申诉记录</span>
        </template>
        
        <el-timeline>
          <el-timeline-item
            v-for="appeal in appeals"
            :key="appeal.id"
            :timestamp="formatDateTime(appeal.createdAt)"
            placement="top"
          >
            <el-card>
              <div class="appeal-item">
                <div class="appeal-header">
                  <span class="appeal-status">
                    <el-tag :type="getAppealStatusColor(appeal.status)">
                      {{ getAppealStatusLabel(appeal.status) }}
                    </el-tag>
                  </span>
                </div>
                <div class="appeal-content">
                  <p><strong>申诉理由：</strong>{{ appeal.reason }}</p>
                  <p v-if="appeal.reviewerNotes">
                    <strong>处理意见：</strong>{{ appeal.reviewerNotes }}
                  </p>
                </div>
              </div>
            </el-card>
          </el-timeline-item>
        </el-timeline>
      </el-card>
    </div>
    
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="handleClose">关闭</el-button>
        <el-button
          v-if="task?.status === 'rejected'"
          type="warning"
          @click="handleAppeal"
        >
          申诉
        </el-button>
        <el-button
          v-if="canResubmit"
          type="success"
          @click="handleResubmit"
        >
          重新提交
        </el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { getTaskAppeals } from '@/api/moderation'
import { formatDateTime } from '@/utils/date'

// Props
interface Props {
  modelValue: boolean
  task: any
}

const props = defineProps<Props>()

// Emits
interface Emits {
  (e: 'update:modelValue', value: boolean): void
  (e: 'appeal', task: any): void
  (e: 'resubmit', task: any): void
}

const emit = defineEmits<Emits>()

// 响应式数据
const appeals = ref([])

// 计算属性
const visible = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const canResubmit = computed(() => {
  return props.task && ['rejected', 'manual_review'].includes(props.task.status)
})

// 监听任务变化
watch(() => props.task, async (newTask) => {
  if (newTask) {
    await loadAppeals(newTask.id)
  }
}, { immediate: true })

// 方法
const loadAppeals = async (taskId: string) => {
  try {
    const response = await getTaskAppeals(taskId)
    appeals.value = response.data
  } catch (error) {
    console.error('Failed to load appeals:', error)
  }
}

const getProcessingDuration = (task: any) => {
  if (!task.reviewedAt) return '-'
  
  const start = new Date(task.createdAt)
  const end = new Date(task.reviewedAt)
  const duration = (end.getTime() - start.getTime()) / 1000
  
  if (duration < 60) {
    return `${duration.toFixed(0)}秒`
  } else if (duration < 3600) {
    return `${(duration / 60).toFixed(1)}分钟`
  } else {
    return `${(duration / 3600).toFixed(1)}小时`
  }
}

const getAnalysisTitle = (key: string) => {
  const titles = {
    sensitive_words: '敏感词检测',
    ml_classification: '机器学习分类',
    sentiment: '情感分析',
    nsfw_detection: 'NSFW检测',
    object_detection: '物体检测',
    text_extraction: '文字提取',
    face_detection: '人脸检测'
  }
  return titles[key] || key
}

const getAppealStatusLabel = (status: string) => {
  const labels = {
    pending: '待处理',
    approved: '已通过',
    rejected: '已拒绝'
  }
  return labels[status] || status
}

const getAppealStatusColor = (status: string) => {
  const colors = {
    pending: 'warning',
    approved: 'success',
    rejected: 'danger'
  }
  return colors[status] || ''
}

const handleAppeal = () => {
  emit('appeal', props.task)
}

const handleResubmit = () => {
  emit('resubmit', props.task)
}

const handleClose = () => {
  visible.value = false
}

// 工具方法（与主页面相同）
const getContentTypeLabel = (type: string) => {
  const labels = {
    text: '文本',
    image: '图片',
    video: '视频',
    audio: '音频'
  }
  return labels[type] || type
}

const getContentTypeColor = (type: string) => {
  const colors = {
    text: '',
    image: 'success',
    video: 'warning',
    audio: 'info'
  }
  return colors[type] || ''
}

const getStatusLabel = (status: string) => {
  const labels = {
    pending: '待审核',
    processing: '审核中',
    approved: '已通过',
    rejected: '已拒绝',
    manual_review: '人工审核'
  }
  return labels[status] || status
}

const getStatusColor = (status: string) => {
  const colors = {
    pending: 'info',
    processing: 'warning',
    approved: 'success',
    rejected: 'danger',
    manual_review: 'warning'
  }
  return colors[status] || ''
}

const getRiskLevelLabel = (level: string) => {
  const labels = {
    low: '低',
    medium: '中',
    high: '高',
    critical: '严重'
  }
  return labels[level] || level
}

const getRiskLevelColor = (level: string) => {
  const colors = {
    low: 'success',
    medium: 'warning',
    high: 'danger',
    critical: 'danger'
  }
  return colors[level] || ''
}
</script>

<style scoped>
.task-detail {
  max-height: 70vh;
  overflow-y: auto;
}

.info-card,
.content-card,
.result-card,
.appeal-card {
  margin-bottom: 20px;
}

.card-title {
  font-weight: bold;
  font-size: 16px;
}

.text-content {
  margin-top: 10px;
}

.image-content,
.video-content,
.audio-content {
  text-align: center;
  margin-top: 10px;
}

.auto-result,
.manual-result {
  margin-bottom: 20px;
}

.auto-result h4,
.manual-result h4 {
  margin-bottom: 15px;
  color: #303133;
}

.analysis-details {
  margin-top: 20px;
}

.analysis-details h5 {
  margin-bottom: 10px;
  color: #606266;
}

.analysis-details pre {
  background-color: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  font-size: 12px;
  max-height: 200px;
  overflow-y: auto;
}

.appeal-item {
  padding: 10px;
}

.appeal-header {
  margin-bottom: 10px;
}

.appeal-content p {
  margin: 5px 0;
  line-height: 1.5;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>
```

## 验收标准

### 功能验收标准

1. **内容检测功能**
   - ✅ 支持文本、图片、视频、音频四种内容类型的审核
   - ✅ 敏感词检测准确率 > 95%
   - ✅ 图片NSFW检测准确率 > 90%
   - ✅ 视频内容分析覆盖关键帧
   - ✅ 音频内容转文字并进行文本审核

2. **审核流程管理**
   - ✅ 自动审核和人工审核无缝衔接
   - ✅ 支持审核结果分级处理
   - ✅ 完整的审核历史记录
   - ✅ 申诉和复审机制完善

3. **规则配置管理**
   - ✅ 支持敏感词库动态更新
   - ✅ 审核规则可视化配置
   - ✅ 白名单和黑名单管理
   - ✅ 规则优先级和权重设置

4. **统计分析功能**
   - ✅ 实时审核统计数据
   - ✅ 违规类型分析报告
   - ✅ 审核效率分析
   - ✅ 趋势分析和预警

### 性能验收标准

1. **响应时间**
   - 文本审核响应时间 < 1秒
   - 图片审核响应时间 < 3秒
   - 视频审核响应时间 < 30秒
   - 音频审核响应时间 < 15秒

2. **并发处理**
   - 支持1000个并发审核请求
   - 任务队列处理能力 > 500任务/分钟
   - 系统资源利用率 < 80%

3. **准确性指标**
   - 敏感内容检测准确率 > 95%
   - 误报率 < 5%
   - 漏报率 < 2%
   - 人工审核一致性 > 90%

4. **可用性要求**
   - 系统可用性 > 99.9%
   - 故障恢复时间 < 5分钟
   - 数据备份和恢复机制完善

### 安全验收标准

1. **数据安全**
   - 审核内容加密存储
   - 敏感数据脱敏处理
   - 访问日志完整记录
   - 定期安全扫描和评估

2. **权限控制**
   - 基于角色的访问控制
   - API接口权限验证
   - 操作审计日志
   - 敏感操作二次确认

3. **合规要求**
   - 符合数据保护法规
   - 用户隐私保护
   - 内容审核标准合规
   - 定期合规性检查

## 业务价值

### 直接价值

1. **风险控制**
   - 降低内容违规风险90%
   - 减少人工审核成本70%
   - 提高审核效率500%
   - 降低平台处罚风险

2. **用户体验**
   - 提供安全的内容环境
   - 减少不良内容曝光
   - 提升平台信誉度
   - 增强用户信任感

3. **运营效率**
   - 自动化审核流程
   - 减少人力投入
   - 提高处理速度
   - 降低运营成本

### 间接价值

1. **数据洞察**
   - 内容质量分析
   - 用户行为洞察
   - 违规趋势预测
   - 审核策略优化

2. **业务支撑**
   - 支持业务快速扩展
   - 降低合规风险
   - 提升品牌形象
   - 增强竞争优势

3. **技术积累**
   - AI技术能力提升
   - 数据处理经验
   - 系统架构优化
   - 团队技能提升

## 依赖关系

### 技术依赖

1. **基础设施**
   - PostgreSQL 数据库
   - MongoDB 文档数据库
   - Redis 缓存服务
   - MinIO 对象存储
   - RabbitMQ 消息队列

2. **AI/ML服务**
   - TensorFlow/PyTorch 模型
   - 预训练语言模型
   - 图像识别服务
   - 语音识别服务
   - 第三方AI API

3. **监控服务**
   - Prometheus 指标收集
   - Grafana 数据可视化
   - ELK 日志分析
   - Jaeger 链路追踪

### 业务依赖

1. **内容来源**
   - 用户生成内容
   - 第三方内容导入
   - 批量内容处理
   - 实时内容流

2. **审核标准**
   - 平台内容政策
   - 法律法规要求
   - 行业标准规范
   - 社区准则

3. **人力资源**
   - 专业审核员
   - 技术支持团队
   - 产品运营团队
   - 法务合规团队

### 环境依赖

1. **开发环境**
   - Python 3.11+
   - Node.js 18+
   - Docker & Kubernetes
   - GPU计算资源

2. **测试环境**
   - 测试数据集
   - 模拟审核场景
   - 性能测试工具
   - 安全测试工具

3. **生产环境**
   - 高性能服务器
   - 负载均衡器
   - CDN加速服务
   - 备份存储系统

## 风险评估

### 技术风险

1. **AI模型准确性** (高风险)
   - **风险描述**: AI模型误判导致审核错误
   - **影响程度**: 可能导致合规内容被误删或违规内容漏过
   - **缓解措施**:
     - 持续优化模型训练数据
     - 建立多模型融合机制
     - 设置人工审核兜底
     - 定期评估模型性能

2. **系统性能瓶颈** (中风险)
   - **风险描述**: 高并发场景下系统响应缓慢
   - **影响程度**: 影响用户体验和审核效率
   - **缓解措施**:
     - 实现分布式处理架构
     - 优化算法和数据结构
     - 增加缓存机制
     - 水平扩展能力

3. **数据安全风险** (高风险)
   - **风险描述**: 审核内容泄露或被恶意访问
   - **影响程度**: 可能导致用户隐私泄露和法律风险
   - **缓解措施**:
     - 数据加密存储和传输
     - 严格的访问控制
     - 定期安全审计
     - 数据脱敏处理

### 业务风险

1. **审核标准变化** (中风险)
   - **风险描述**: 法规政策变化导致审核标准调整
   - **影响程度**: 需要快速调整审核规则和模型
   - **缓解措施**:
     - 建立灵活的规则配置系统
     - 密切关注政策动态
     - 快速响应机制
     - 专业法务支持

2. **误审申诉处理** (中风险)
   - **风险描述**: 大量误审导致申诉积压
   - **影响程度**: 影响用户满意度和运营效率
   - **缓解措施**:
     - 优化审核算法准确性
     - 建立高效申诉处理流程
     - 增加人工审核资源
     - 用户教育和沟通

3. **内容多样性挑战** (中风险)
   - **风险描述**: 新型内容形式难以有效审核
   - **影响程度**: 可能出现审核盲区
   - **缓解措施**:
     - 持续技术研发投入
     - 建立快速适应机制
     - 行业合作和经验分享
     - 用户举报机制

### 运营风险

1. **人员依赖** (低风险)
   - **风险描述**: 关键技术人员流失
   - **影响程度**: 可能影响系统维护和优化
   - **缓解措施**:
     - 完善技术文档
     - 知识转移机制
     - 团队梯队建设
     - 技术培训计划

2. **第三方服务依赖** (中风险)
   - **风险描述**: 依赖的AI服务不稳定
   - **影响程度**: 可能影响审核功能可用性
   - **缓解措施**:
     - 多服务商备选方案
     - 自研核心能力
     - 服务监控和告警
     - SLA保障协议

## 开发任务分解

### 后端开发任务

#### 阶段一：基础架构 (预计 3 周)

1. **数据库设计与实现** (5天)
   - 设计PostgreSQL表结构
   - 创建MongoDB集合结构
   - 实现数据模型和ORM映射
   - 创建索引和约束
   - 数据迁移脚本

2. **核心服务框架** (5天)
   - 搭建FastAPI项目结构
   - 配置数据库连接池
   - 集成Redis缓存
   - 配置Celery任务队列
   - 实现基础中间件和异常处理

3. **AI模型集成** (5天)
   - 集成文本分析模型
   - 集成图像识别模型
   - 集成视频分析模型
   - 集成音频识别模型
   - 模型性能优化

#### 阶段二：核心功能 (预计 4 周)

1. **内容审核服务** (7天)
   - 实现ContentModerationService类
   - 文本分析器实现
   - 图像分析器实现
   - 视频分析器实现
   - 音频分析器实现

2. **规则引擎** (5天)
   - 实现RuleEngine类
   - 敏感词检测器
   - 规则配置管理
   - 规则评估算法
   - 规则缓存机制

3. **任务处理系统** (5天)
   - 异步任务处理
   - 任务状态管理
   - 重试机制
   - 错误处理和日志
   - 性能监控

4. **申诉管理系统** (3天)
   - 申诉提交处理
   - 申诉审核流程
   - 申诉状态管理
   - 通知机制

#### 阶段三：API接口 (预计 2 周)

1. **审核管理API** (5天)
   - 内容提交审核接口
   - 审核结果查询接口
   - 任务列表接口
   - 文件上传接口
   - 批量操作接口

2. **规则管理API** (3天)
   - 规则CRUD接口
   - 敏感词管理接口
   - 白名单管理接口
   - 规则测试接口

3. **统计分析API** (3天)
   - 审核统计接口
   - 趋势分析接口
   - 违规分析接口
   - 性能指标接口

4. **API文档和测试** (3天)
   - OpenAPI文档生成
   - 接口单元测试
   - 集成测试
   - 性能测试

#### 阶段四：优化和监控 (预计 1 周)

1. **性能优化** (3天)
   - 数据库查询优化
   - 缓存策略优化
   - 算法性能优化
   - 并发处理优化

2. **监控和告警** (2天)
   - Prometheus指标收集
   - 日志结构化
   - 告警规则配置
   - 健康检查接口

3. **安全加固** (2天)
   - 输入验证加强
   - 权限控制完善
   - 数据加密实现
   - 安全审计日志

### 前端开发任务

#### 阶段一：项目搭建 (预计 1 周)

1. **项目初始化** (2天)
   - Vue3 + TypeScript项目搭建
   - 配置构建工具和代码规范
   - 集成Element Plus和图标库
   - 配置路由和状态管理

2. **基础组件** (2天)
   - 布局组件开发
   - 通用工具函数
   - API客户端配置
   - 类型定义文件

3. **样式和主题** (1天)
   - 主题色彩配置
   - 响应式布局
   - 组件样式规范
   - 图标和字体配置

#### 阶段二：核心页面 (预计 3 周)

1. **审核管理主页面** (5天)
   - 统计卡片组件
   - 任务列表组件
   - 筛选和搜索功能
   - 分页和排序
   - 批量操作功能

2. **提交审核功能** (4天)
   - 提交审核对话框
   - 文件上传组件
   - 表单验证
   - 内容预览功能

3. **任务详情页面** (4天)
   - 任务信息展示
   - 内容预览组件
   - 审核结果展示
   - 申诉记录展示

4. **申诉管理功能** (3天)
   - 申诉提交对话框
   - 申诉列表页面
   - 申诉详情展示
   - 申诉状态管理

5. **规则管理页面** (5天)
   - 规则列表页面
   - 规则编辑对话框
   - 敏感词管理
   - 白名单管理
   - 规则测试功能

#### 阶段三：统计分析 (预计 1 周)

1. **统计仪表板** (3天)
   - 审核概览统计
   - 趋势图表展示
   - 违规类型分析
   - 实时数据更新

2. **报表功能** (2天)
   - 数据导出功能
   - 报表生成
   - 图表交互
   - 数据筛选

#### 阶段四：优化和测试 (预计 1 周)

1. **性能优化** (2天)
   - 组件懒加载
   - 图片优化
   - 打包优化
   - 缓存策略

2. **用户体验优化** (2天)
   - 加载状态处理
   - 错误提示优化
   - 交互动画
   - 响应式适配

3. **测试和调试** (1天)
   - 单元测试
   - 端到端测试
   - 浏览器兼容性测试
   - 性能测试

### 测试任务

#### 单元测试 (预计 1.5 周)

1. **后端单元测试** (5天)
   - 服务类测试
   - API接口测试
   - 数据库操作测试
   - 算法逻辑测试
   - 模型集成测试

2. **前端单元测试** (3天)
   - 组件测试
   - 工具函数测试
   - 状态管理测试
   - API调用测试

#### 集成测试 (预计 1.5 周)

1. **API集成测试** (3天)
   - 接口联调测试
   - 数据流测试
   - 错误场景测试
   - 性能压力测试

2. **端到端测试** (3天)
   - 用户流程测试
   - 跨浏览器测试
   - 移动端适配测试
   - 审核流程完整性测试

3. **AI模型测试** (2天)
   - 模型准确性测试
   - 性能基准测试
   - 边界情况测试
   - 模型稳定性测试

### 部署任务

#### 环境搭建 (预计 1 周)

1. **容器化** (2天)
   - 编写Dockerfile
   - Docker Compose配置
   - 环境变量配置
   - 容器优化

2. **CI/CD配置** (2天)
   - 构建流水线
   - 自动化测试
   - 部署脚本
   - 回滚机制

3. **监控配置** (2天)
   - 日志收集配置
   - 指标监控配置
   - 告警规则配置
   - 性能监控

4. **安全配置** (1天)
   - 网络安全配置
   - 访问控制配置
   - 证书配置
   - 安全扫描

## 时间估算

### 总体时间安排

- **后端开发**: 10 周
- **前端开发**: 6 周
- **测试**: 3 周
- **部署**: 1 周
- **总计**: 12 周 (考虑并行开发)

### 人力资源需求

- **后端开发工程师**: 3人
- **AI/ML工程师**: 2人
- **前端开发工程师**: 2人
- **测试工程师**: 2人
- **DevOps工程师**: 1人
- **产品经理**: 1人
- **UI/UX设计师**: 1人

### 关键里程碑

1. **第3周末**: 基础架构和AI模型集成完成
2. **第6周末**: 核心审核功能完成
3. **第8周末**: API接口和前端主要页面完成
4. **第10周末**: 功能开发完成，开始测试
5. **第12周末**: 测试完成，部署上线

## 成功指标

### 技术指标

1. **功能完整性**: 100%需求实现
2. **代码覆盖率**: >85%
3. **API响应时间**: 文本<1s, 图片<3s, 视频<30s
4. **系统可用性**: >99.9%
5. **并发处理能力**: >1000请求/分钟

### 业务指标

1. **审核准确率**: >95%
2. **误报率**: <5%
3. **漏报率**: <2%
4. **用户满意度**: >4.0/5
5. **处理效率提升**: >500%

### 运营指标

1. **系统稳定性**: 无重大故障
2. **响应时间**: 问题解决<2小时
3. **文档完整性**: 100%功能有文档
4. **团队技能**: 100%成员掌握核心技术
5. **知识传承**: 完整的技术文档和培训材料