# Story 3.3: 内容质量控制服务

## 基本信息
- **Epic**: Epic 3 - AI大模型服务和内容文本优化
- **Story ID**: 3.3
- **优先级**: 中
- **预估工作量**: 2周
- **负责团队**: 后端开发团队 + 质量工程团队

## 用户故事

**作为** 内容审核员  
**我希望** 自动化的内容质量控制服务  
**以便于** 确保发布内容的质量和合规性，降低人工审核成本，提升审核效率

## 需求描述

### 核心功能需求

1. **多维度质量检测**
   - 语法正确性检查和自动纠错
   - 逻辑一致性和内容连贯性分析
   - 事实准确性验证（基于历史数据库）
   - 文本结构和格式规范检查
   - 学术写作规范性评估

2. **合规性审核**
   - 敏感词汇检测和过滤
   - 政策法规合规性检查
   - 版权侵权风险评估
   - 学术诚信检查（抄袭检测）
   - 内容适宜性评级

3. **智能风险评估**
   - 内容风险等级自动评分
   - 潜在问题预警和标记
   - 人工审核优先级排序
   - 风险趋势分析和预测
   - 自动化处理建议

4. **自动修复建议**
   - 语法错误自动修正
   - 格式标准化建议
   - 内容完善性建议
   - 替代表达推荐
   - 质量提升路径规划

5. **审核工作流集成**
   - 多级审核流程管理
   - 审核任务智能分配
   - 审核结果统计分析
   - 审核员绩效评估
   - 审核标准持续优化

## 技术实现

### 核心技术栈

- **服务框架**: FastAPI + Python 3.11
- **NLP处理**: spaCy, jieba, HanLP, BERT
- **质量检测**: 自研算法 + 规则引擎
- **合规检测**: 敏感词库 + 机器学习模型
- **数据库**: PostgreSQL (规则配置) + MongoDB (审核记录)
- **缓存**: Redis (检测结果缓存)
- **任务队列**: Celery + RabbitMQ
- **机器学习**: scikit-learn, TensorFlow

### 系统架构设计

#### 质量控制架构图
```
┌─────────────────────────────────────────────────────────────┐
│                 内容质量控制服务架构                          │
├─────────────────────────────────────────────────────────────┤
│  API Gateway                                                │
│  ├── 质量检测API ├── 合规审核API ├── 工作流API               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    质量检测引擎                              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 语法检测    │ 逻辑分析    │ 事实验证    │ 格式检查    │  │
│  │ 模块        │ 模块        │ 模块        │ 模块        │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    合规检测引擎                              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 敏感词      │ 政策合规    │ 版权检查    │ 学术诚信    │  │
│  │ 检测        │ 检查        │ 模块        │ 检测        │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    风险评估引擎                              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ 风险计算    │ 优先级      │ 修复建议    │ 工作流      │  │
│  │ 模块        │ 排序        │ 生成器      │ 管理器      │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    数据存储层                                │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ PostgreSQL  │ MongoDB     │ Redis       │ 规则库      │  │
│  │ (配置规则)  │ (审核记录)  │ (缓存)      │ (知识库)    │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 数据库设计

**注意：**以下数据库表结构由storage-service统一管理，本服务通过API调用访问。

#### PostgreSQL配置数据库
```sql
-- 质量检测规则表
CREATE TABLE quality_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- grammar, logic, format, academic
    category VARCHAR(50), -- 规则分类
    description TEXT,
    rule_config JSONB NOT NULL, -- 规则具体配置
    severity_level INTEGER DEFAULT 1, -- 1-5，严重程度
    is_active BOOLEAN DEFAULT true,
    auto_fix_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 合规检测规则表
CREATE TABLE compliance_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- sensitive_word, policy, copyright, plagiarism
    compliance_category VARCHAR(50), -- 合规类别
    rule_pattern TEXT, -- 规则模式或关键词
    action_type VARCHAR(30) DEFAULT 'warn', -- block, warn, flag, auto_replace
    replacement_text TEXT, -- 自动替换文本
    risk_score INTEGER DEFAULT 1, -- 1-10，风险评分
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 审核工作流配置表
CREATE TABLE review_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name VARCHAR(100) NOT NULL,
    content_type VARCHAR(50), -- 适用的内容类型
    workflow_steps JSONB NOT NULL, -- 工作流步骤配置
    auto_approval_threshold DECIMAL(5,2), -- 自动通过阈值
    human_review_threshold DECIMAL(5,2), -- 人工审核阈值
    escalation_rules JSONB, -- 升级规则
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 质量检测结果表
CREATE TABLE quality_check_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    check_type VARCHAR(50) NOT NULL,
    overall_score DECIMAL(5,2), -- 总体质量分数
    detected_issues JSONB, -- 检测到的问题
    suggestions JSONB, -- 改进建议
    auto_fixes_applied JSONB, -- 自动修复的内容
    check_duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 合规检测结果表
CREATE TABLE compliance_check_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    compliance_status VARCHAR(20) DEFAULT 'pending', -- pass, fail, warning, pending
    risk_score INTEGER, -- 风险评分
    violations JSONB, -- 违规详情
    recommendations JSONB, -- 整改建议
    reviewed_by UUID, -- 审核人员ID
    review_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP
);

-- 审核任务表
CREATE TABLE review_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    workflow_id UUID REFERENCES review_workflows(id),
    current_step INTEGER DEFAULT 1,
    assigned_reviewer UUID, -- 分配的审核员
    task_status VARCHAR(20) DEFAULT 'pending', -- pending, in_progress, completed, rejected
    priority_score INTEGER DEFAULT 5, -- 1-10，优先级评分
    estimated_review_time INTEGER, -- 预估审核时间（分钟）
    actual_review_time INTEGER, -- 实际审核时间
    review_notes TEXT,
    decision_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assigned_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- 敏感词库表
CREATE TABLE sensitive_words (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    word VARCHAR(200) NOT NULL,
    category VARCHAR(50), -- 敏感词分类
    severity_level INTEGER DEFAULT 1, -- 严重程度
    replacement_suggestion VARCHAR(200), -- 替换建议
    context_rules JSONB, -- 上下文规则
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_quality_rules_type ON quality_rules(rule_type);
CREATE INDEX idx_compliance_rules_type ON compliance_rules(rule_type);
CREATE INDEX idx_quality_check_content ON quality_check_results(content_id);
CREATE INDEX idx_compliance_check_content ON compliance_check_results(content_id);
CREATE INDEX idx_review_tasks_status ON review_tasks(task_status);
CREATE INDEX idx_review_tasks_assigned ON review_tasks(assigned_reviewer);
CREATE INDEX idx_sensitive_words_word ON sensitive_words(word);
```

#### MongoDB审核记录数据库

**注意：**审核记录数据由storage-service统一管理，本服务通过API访问。
```javascript
// 内容审核记录集合
{
  "_id": ObjectId,
  "content_id": "uuid",
  "original_content": "原始内容",
  "processed_content": "处理后内容", 
  "quality_analysis": {
    "grammar_score": 85.5,
    "logic_score": 78.2,
    "format_score": 92.0,
    "academic_score": 80.5,
    "overall_score": 84.1,
    "issues": [
      {
        "type": "grammar_error",
        "position": 25,
        "description": "主谓不一致",
        "suggestion": "建议修改为...",
        "severity": "medium",
        "auto_fixable": true
      }
    ]
  },
  "compliance_analysis": {
    "status": "warning",
    "risk_score": 3,
    "violations": [
      {
        "type": "sensitive_word",
        "word": "敏感词",
        "position": 45,
        "category": "政治敏感",
        "action": "replace",
        "suggestion": "替代词"
      }
    ],
    "policy_compliance": {
      "content_policy": "pass",
      "academic_integrity": "pass",
      "copyright_check": "warning"
    }
  },
  "review_history": [
    {
      "reviewer_id": "reviewer_uuid",
      "step": 1,
      "action": "approve_with_changes",
      "notes": "需要修改部分表述",
      "timestamp": ISODate,
      "time_spent_minutes": 15
    }
  ],
  "final_decision": {
    "status": "approved", // approved, rejected, needs_revision
    "decision_maker": "reviewer_uuid",
    "decision_time": ISODate,
    "final_notes": "经修改后符合发布标准"
  },
  "metrics": {
    "total_review_time": 25, // 分钟
    "auto_fixes_applied": 3,
    "human_interventions": 1,
    "quality_improvement": 12.5 // 质量提升分数
  },
  "created_at": ISODate,
  "updated_at": ISODate
}

// 审核统计集合
{
  "_id": ObjectId,
  "date": ISODate,
  "period": "daily", // daily, weekly, monthly
  "statistics": {
    "total_reviews": 150,
    "auto_approved": 120,
    "human_reviewed": 30,
    "rejected": 5,
    "average_quality_score": 86.3,
    "average_review_time": 18.5,
    "common_issues": [
      {"type": "grammar_error", "count": 45},
      {"type": "format_issue", "count": 23}
    ],
    "reviewer_performance": [
      {
        "reviewer_id": "uuid",
        "reviews_completed": 25,
        "average_time": 15.2,
        "accuracy_rate": 0.94
      }
    ]
  }
}
```

### 核心服务实现

#### 质量检测引擎
```python
# quality_control_engine.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import re
from datetime import datetime

class IssueType(Enum):
    GRAMMAR_ERROR = "grammar_error"
    LOGIC_INCONSISTENCY = "logic_inconsistency"
    FORMAT_VIOLATION = "format_violation"
    FACTUAL_ERROR = "factual_error"
    ACADEMIC_STANDARD = "academic_standard"

class IssueSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityIssue:
    """质量问题数据类"""
    issue_type: IssueType
    severity: IssueSeverity
    position: int
    description: str
    suggestion: str
    auto_fixable: bool
    confidence: float

@dataclass
class QualityCheckResult:
    """质量检测结果"""
    overall_score: float
    issues: List[QualityIssue]
    suggestions: List[str]
    auto_fixes: List[Dict]
    metrics: Dict[str, float]

class QualityDetectionEngine:
    """
    质量检测引擎
    负责多维度的内容质量检测
    """
    
    def __init__(self, storage_client, nlp_processor):
        self.storage_client = storage_client
        self.nlp_processor = nlp_processor
        self.detectors = {
            'grammar': GrammarDetector(),
            'logic': LogicDetector(), 
            'format': FormatDetector(),
            'factual': FactualDetector(),
            'academic': AcademicDetector()
        }
    
    async def check_quality(self, content: str, content_type: str = "general") -> QualityCheckResult:
        """
        执行全面的质量检测
        
        Args:
            content: 待检测内容
            content_type: 内容类型
            
        Returns:
            质量检测结果
        """
        start_time = datetime.now()
        
        # 预处理文本
        processed_content = await self._preprocess_content(content)
        
        # 并行执行各类检测
        detection_tasks = []
        for detector_name, detector in self.detectors.items():
            task = detector.detect(processed_content, content_type)
            detection_tasks.append(task)
        
        detection_results = await asyncio.gather(*detection_tasks)
        
        # 合并检测结果
        all_issues = []
        metrics = {}
        
        for i, (detector_name, results) in enumerate(zip(self.detectors.keys(), detection_results)):
            issues, detector_metrics = results
            all_issues.extend(issues)
            metrics.update({f"{detector_name}_{k}": v for k, v in detector_metrics.items()})
        
        # 计算总体评分
        overall_score = await self._calculate_overall_score(all_issues, metrics)
        
        # 生成改进建议
        suggestions = await self._generate_suggestions(all_issues)
        
        # 生成自动修复方案
        auto_fixes = await self._generate_auto_fixes(all_issues, content)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        metrics['processing_time_ms'] = processing_time
        
        return QualityCheckResult(
            overall_score=overall_score,
            issues=all_issues,
            suggestions=suggestions,
            auto_fixes=auto_fixes,
            metrics=metrics
        )
    
    async def _calculate_overall_score(self, issues: List[QualityIssue], metrics: Dict) -> float:
        """计算总体质量评分"""
        base_score = 100.0
        
        # 根据问题严重程度扣分
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 10.0
            elif issue.severity == IssueSeverity.HIGH:
                base_score -= 5.0
            elif issue.severity == IssueSeverity.MEDIUM:
                base_score -= 2.0
            elif issue.severity == IssueSeverity.LOW:
                base_score -= 0.5
        
        return max(0.0, min(100.0, base_score))
    
    async def _generate_suggestions(self, issues: List[QualityIssue]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 按问题类型分组统计
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        # 生成针对性建议
        if issue_counts.get(IssueType.GRAMMAR_ERROR, 0) > 3:
            suggestions.append("建议仔细检查语法错误，注意主谓一致和时态使用")
        
        if issue_counts.get(IssueType.FORMAT_VIOLATION, 0) > 2:
            suggestions.append("请按照学术写作规范调整文档格式")
        
        if issue_counts.get(IssueType.LOGIC_INCONSISTENCY, 0) > 1:
            suggestions.append("注意检查内容逻辑的一致性和连贯性")
        
        return suggestions
    
    async def _generate_auto_fixes(self, issues: List[QualityIssue], content: str) -> List[Dict]:
        """生成自动修复方案"""
        auto_fixes = []
        
        for issue in issues:
            if issue.auto_fixable:
                fix_info = {
                    'position': issue.position,
                    'issue_type': issue.issue_type.value,
                    'original': self._extract_text_at_position(content, issue.position),
                    'suggested_fix': issue.suggestion,
                    'confidence': issue.confidence
                }
                auto_fixes.append(fix_info)
        
        return auto_fixes

class GrammarDetector:
    """语法检测器"""
    
    async def detect(self, content: str, content_type: str) -> tuple[List[QualityIssue], Dict]:
        """检测语法错误"""
        issues = []
        metrics = {'grammar_score': 90.0}
        
        # 使用NLP库进行语法分析
        import jieba.posseg as pseg
        
        sentences = content.split('。')
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # 检查句子结构
            grammar_issues = await self._check_sentence_grammar(sentence, i)
            issues.extend(grammar_issues)
        
        # 计算语法分数
        if issues:
            grammar_score = max(60.0, 100.0 - len(issues) * 5)
            metrics['grammar_score'] = grammar_score
        
        return issues, metrics
    
    async def _check_sentence_grammar(self, sentence: str, sentence_index: int) -> List[QualityIssue]:
        """检查单个句子的语法"""
        issues = []
        
        # 简化的语法检查规则
        if len(sentence) > 100:
            issues.append(QualityIssue(
                issue_type=IssueType.GRAMMAR_ERROR,
                severity=IssueSeverity.MEDIUM,
                position=sentence_index * 50,  # 估算位置
                description="句子过长，建议拆分",
                suggestion="将长句拆分为多个短句",
                auto_fixable=False,
                confidence=0.8
            ))
        
        # 检查标点符号
        if sentence.count('，') > 5:
            issues.append(QualityIssue(
                issue_type=IssueType.GRAMMAR_ERROR,
                severity=IssueSeverity.LOW,
                position=sentence_index * 50,
                description="逗号使用过多",
                suggestion="减少逗号使用，适当使用其他标点符号",
                auto_fixable=False,
                confidence=0.7
            ))
        
        return issues

class LogicDetector:
    """逻辑一致性检测器"""
    
    async def detect(self, content: str, content_type: str) -> tuple[List[QualityIssue], Dict]:
        """检测逻辑一致性"""
        issues = []
        metrics = {'logic_score': 85.0}
        
        # 检查时间逻辑
        time_issues = await self._check_temporal_logic(content)
        issues.extend(time_issues)
        
        # 检查因果关系
        causal_issues = await self._check_causal_logic(content)
        issues.extend(causal_issues)
        
        # 检查前后矛盾
        contradiction_issues = await self._check_contradictions(content)
        issues.extend(contradiction_issues)
        
        return issues, metrics
    
    async def _check_temporal_logic(self, content: str) -> List[QualityIssue]:
        """检查时间逻辑"""
        issues = []
        
        # 提取时间表达
        time_patterns = [
            r'\d{4}年', r'\d{1,2}月', r'\d{1,2}日',
            r'(明|清|唐|宋|元|汉)朝', r'(初|中|末)期'
        ]
        
        time_mentions = []
        for pattern in time_patterns:
            for match in re.finditer(pattern, content):
                time_mentions.append((match.start(), match.group()))
        
        # 简化的时间逻辑检查
        if len(time_mentions) > 3:
            # 检查是否有明显的时间冲突
            pass  # 这里可以实现更复杂的时间逻辑检查
        
        return issues

class ComplianceEngine:
    """
    合规检测引擎
    负责内容的合规性检查
    """
    
    def __init__(self, storage_client, policy_checker):
        self.storage_client = storage_client
        self.policy_checker = policy_checker
    
    async def check_compliance(self, content: str) -> Dict[str, Any]:
        """
        执行合规性检查
        
        Args:
            content: 待检查内容
            
        Returns:
            合规检查结果
        """
        result = {
            'status': 'pass',
            'risk_score': 0,
            'violations': [],
            'recommendations': []
        }
        
        # 敏感词检测
        sensitive_violations = await self._check_sensitive_words(content)
        result['violations'].extend(sensitive_violations)
        
        # 政策合规检查
        policy_violations = await self._check_policy_compliance(content)
        result['violations'].extend(policy_violations)
        
        # 版权检查
        copyright_violations = await self._check_copyright(content)
        result['violations'].extend(copyright_violations)
        
        # 学术诚信检查
        integrity_violations = await self._check_academic_integrity(content)
        result['violations'].extend(integrity_violations)
        
        # 计算风险评分
        result['risk_score'] = self._calculate_risk_score(result['violations'])
        
        # 确定合规状态
        if result['risk_score'] >= 8:
            result['status'] = 'fail'
        elif result['risk_score'] >= 5:
            result['status'] = 'warning'
        else:
            result['status'] = 'pass'
        
        # 生成建议
        result['recommendations'] = await self._generate_compliance_recommendations(
            result['violations']
        )
        
        return result
    
    async def _check_sensitive_words(self, content: str) -> List[Dict]:
        """检查敏感词"""
        violations = []
        
        # 通过storage-service获取敏感词列表
        sensitive_words_result = await self.storage_client.get_sensitive_words(active_only=True)
        sensitive_words = sensitive_words_result.get('data', [])
        
        for word_info in sensitive_words:
            word = word_info['word']
            if word in content:
                # 找到所有出现位置
                for match in re.finditer(re.escape(word), content):
                    violations.append({
                        'type': 'sensitive_word',
                        'word': word,
                        'position': match.start(),
                        'category': word_info.get('category', 'unknown'),
                        'severity': word_info.get('severity_level', 1),
                        'action': 'replace',
                        'suggestion': word_info.get('replacement_suggestion', '***')
                    })
        
        return violations
    
    async def _check_policy_compliance(self, content: str) -> List[Dict]:
        """检查政策合规性"""
        violations = []
        
        # 这里可以实现具体的政策合规检查逻辑
        # 例如检查是否包含违规内容类型、是否符合出版规范等
        
        return violations
    
    async def _check_copyright(self, content: str) -> List[Dict]:
        """检查版权风险"""
        violations = []
        
        # 实现版权风险评估逻辑
        # 可以通过相似度检索检查是否存在抄袭
        
        return violations
    
    def _calculate_risk_score(self, violations: List[Dict]) -> int:
        """计算风险评分"""
        risk_score = 0
        
        for violation in violations:
            severity = violation.get('severity', 1)
            if violation['type'] == 'sensitive_word':
                risk_score += severity * 2
            elif violation['type'] == 'policy_violation':
                risk_score += severity * 3
            elif violation['type'] == 'copyright_violation':
                risk_score += severity * 4
        
        return min(10, risk_score)

class ReviewWorkflowManager:
    """
    审核工作流管理器
    管理内容审核的工作流程
    """
    
    def __init__(self, storage_client, task_queue):
        self.storage_client = storage_client
        self.task_queue = task_queue
    
    async def create_review_task(self, 
                               content_id: str,
                               quality_result: QualityCheckResult,
                               compliance_result: Dict) -> str:
        """
        创建审核任务
        
        Args:
            content_id: 内容ID
            quality_result: 质量检测结果
            compliance_result: 合规检测结果
            
        Returns:
            任务ID
        """
        # 计算优先级评分
        priority_score = self._calculate_priority(quality_result, compliance_result)
        
        # 选择合适的工作流
        workflow = await self._select_workflow(quality_result, compliance_result)
        
        # 通过storage-service创建审核任务
        task_result = await self.storage_client.create_review_task({
            'content_id': content_id,
            'workflow_id': workflow['id'],
            'priority_score': priority_score,
            'estimated_review_time': workflow.get('estimated_time', 30)
        })
        task_id = task_result.get('task_id')
        
        # 如果符合自动审核条件，直接处理
        if (quality_result.overall_score >= workflow.get('auto_approval_threshold', 90) and
            compliance_result['risk_score'] <= workflow.get('auto_approval_risk', 2)):
            await self._auto_approve_content(task_id, quality_result, compliance_result)
        else:
            # 分配给人工审核
            await self._assign_to_reviewer(task_id, priority_score)
        
        return task_id
    
    def _calculate_priority(self, 
                          quality_result: QualityCheckResult,
                          compliance_result: Dict) -> int:
        """计算审核优先级"""
        priority = 5  # 基础优先级
        
        # 质量分数越低，优先级越高
        if quality_result.overall_score < 70:
            priority += 3
        elif quality_result.overall_score < 80:
            priority += 1
        
        # 风险分数越高，优先级越高
        risk_score = compliance_result.get('risk_score', 0)
        if risk_score >= 7:
            priority += 4
        elif risk_score >= 4:
            priority += 2
        
        # 关键问题增加优先级
        critical_issues = [
            issue for issue in quality_result.issues
            if issue.severity == IssueSeverity.CRITICAL
        ]
        priority += len(critical_issues)
        
        return min(10, priority)
    
    async def _select_workflow(self, quality_result, compliance_result) -> Dict:
        """选择合适的审核工作流"""
        # 通过storage-service根据内容特征选择工作流
        workflow_result = await self.storage_client.get_active_workflows()
        workflows = workflow_result.get('data', [])
        workflow = workflows[0] if workflows else {}
        return dict(workflow) if workflow else {}
```

### API接口设计

#### 质量检测API
```python
# 内容质量检测
POST /api/v1/quality/check
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

Request:
{
    "content": "朱元璋，濠州钟离人也。其先世家沛，徙句容，再徙泗州...",
    "content_type": "historical_text",
    "check_options": {
        "grammar_check": true,
        "logic_check": true,
        "format_check": true,
        "factual_check": true,
        "academic_check": true
    },
    "auto_fix": true
}

Response:
{
    "success": true,
    "data": {
        "check_id": "check_uuid",
        "overall_score": 84.5,
        "status": "needs_review", // pass, needs_review, critical_issues
        "quality_analysis": {
            "grammar_score": 88.0,
            "logic_score": 82.5,
            "format_score": 90.0,
            "factual_score": 85.0,
            "academic_score": 80.0
        },
        "issues": [
            {
                "type": "grammar_error",
                "severity": "medium",
                "position": 25,
                "description": "句子结构不够完整",
                "suggestion": "建议在此处添加适当的连接词",
                "auto_fixable": true,
                "confidence": 0.85
            }
        ],
        "suggestions": [
            "建议优化部分语法表达",
            "可以增强学术写作的规范性"
        ],
        "auto_fixes": [
            {
                "position": 25,
                "original": "其先世家沛",
                "suggested_fix": "其祖先世居沛县",
                "confidence": 0.9
            }
        ],
        "processing_time_ms": 1500
    }
}

# 合规性检查
POST /api/v1/compliance/check
Request:
{
    "content": "文档内容...",
    "check_types": ["sensitive_words", "policy", "copyright", "academic_integrity"]
}

Response:
{
    "success": true,
    "data": {
        "compliance_status": "warning", // pass, warning, fail
        "risk_score": 3,
        "violations": [
            {
                "type": "sensitive_word",
                "word": "敏感词汇",
                "position": 45,
                "category": "政治敏感",
                "severity": 2,
                "action": "replace",
                "suggestion": "建议替换词汇"
            }
        ],
        "policy_compliance": {
            "content_policy": "pass",
            "academic_integrity": "pass", 
            "copyright_check": "warning"
        },
        "recommendations": [
            "建议替换检测到的敏感词汇",
            "需要进一步核实版权信息"
        ]
    }
}
```

#### 审核工作流API
```python
# 创建审核任务
POST /api/v1/review/tasks
Request:
{
    "content_id": "content_uuid",
    "quality_result": {...},
    "compliance_result": {...},
    "priority": "high", // low, medium, high, urgent
    "assigned_reviewer": "reviewer_uuid" // 可选
}

Response:
{
    "success": true,
    "data": {
        "task_id": "task_uuid",
        "status": "pending",
        "priority_score": 8,
        "estimated_review_time": 25,
        "workflow_steps": [
            {
                "step": 1,
                "name": "初步审核",
                "assignee": "reviewer_uuid",
                "estimated_time": 15
            },
            {
                "step": 2, 
                "name": "专家复审",
                "estimated_time": 10
            }
        ]
    }
}

# 获取审核任务列表
GET /api/v1/review/tasks?status=pending&assigned_to=reviewer_uuid
Response:
{
    "success": true,
    "data": {
        "tasks": [
            {
                "task_id": "task_uuid",
                "content_id": "content_uuid",
                "title": "文档标题",
                "priority_score": 8,
                "status": "pending",
                "estimated_time": 25,
                "created_at": "2024-01-15T10:00:00Z",
                "quality_summary": {
                    "overall_score": 84.5,
                    "major_issues": 2
                },
                "compliance_summary": {
                    "risk_score": 3,
                    "violations_count": 1
                }
            }
        ],
        "pagination": {
            "total": 50,
            "page": 1,
            "per_page": 20
        }
    }
}

# 提交审核结果
POST /api/v1/review/tasks/{task_id}/decision
Request:
{
    "decision": "approve_with_changes", // approve, reject, approve_with_changes, escalate
    "notes": "建议修改第二段的表述方式",
    "required_changes": [
        {
            "position": 120,
            "description": "修改语法错误",
            "suggestion": "具体修改建议"
        }
    ],
    "review_time_minutes": 18
}

Response:
{
    "success": true,
    "data": {
        "task_id": "task_uuid",
        "new_status": "approved_with_changes",
        "next_steps": [
            "等待作者修改",
            "修改完成后自动进入二次审核"
        ]
    }
}
```

## 验收标准

### 功能性验收标准

1. **质量检测功能**
   - ✅ 支持语法、逻辑、格式、事实、学术5个维度检测
   - ✅ 检测准确率>90%
   - ✅ 自动修复建议准确率>80%
   - ✅ 支持批量质量检测

2. **合规性检测**
   - ✅ 敏感词检测准确率>95%
   - ✅ 政策合规检查覆盖率>90%
   - ✅ 版权风险评估准确率>85%
   - ✅ 学术诚信检测准确率>90%

3. **工作流管理**
   - ✅ 支持多级审核流程
   - ✅ 智能任务分配
   - ✅ 审核进度实时跟踪
   - ✅ 审核效率提升>50%

### 性能验收标准

1. **检测性能**
   - ✅ 质量检测响应时间<2秒
   - ✅ 合规检测响应时间<1秒
   - ✅ 批量检测处理能力>100文档/分钟
   - ✅ 系统并发支持>200请求

2. **系统性能**
   - ✅ 系统可用性>99.5%
   - ✅ 检测结果缓存命中率>60%
   - ✅ API响应时间<500ms
   - ✅ 错误率<0.5%

### 质量验收标准

1. **检测质量**
   - ✅ 假阳性率<5%
   - ✅ 假阴性率<3%
   - ✅ 检测一致性>95%
   - ✅ 用户满意度>90%

2. **审核效率**
   - ✅ 人工审核工作量减少>60%
   - ✅ 审核周期缩短>40%
   - ✅ 自动通过率>70%
   - ✅ 审核准确性>95%

## 业务价值

### 直接价值
1. **效率提升**: 自动化质量检测和合规审核大幅提升工作效率
2. **成本降低**: 减少人工审核成本60%以上
3. **质量保证**: 确保发布内容的高质量和合规性
4. **风险控制**: 有效降低内容发布的法律和声誉风险

### 间接价值
1. **标准化**: 建立统一的内容质量和合规标准
2. **数据积累**: 积累大量质量和合规数据用于持续优化
3. **用户信任**: 提升用户对平台内容质量的信任度
4. **竞争优势**: 建立内容质量管理的技术壁垒

## 总结

内容质量控制服务通过多维度的自动化检测和智能化的审核工作流，为历史文本项目提供了全面的质量保障体系。该服务不仅能够有效提升内容质量和合规性，还能显著降低人工审核成本，为平台的健康发展奠定了坚实基础。