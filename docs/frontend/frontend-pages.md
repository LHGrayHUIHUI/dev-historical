# 历史文本优化项目 - 前端页面设计

## 概述

本文档详细描述历史文本优化项目的8个核心页面设计，包括功能规范、组件结构和技术实现。每个页面都基于PRD需求进行设计，确保功能完整性和用户体验。

## 页面架构

### 页面导航结构
```
├── 仪表板 (Dashboard) - /dashboard
├── 数据源管理 (Data Source) - /data-source
├── 内容管理 (Content Management) - /content
├── 发布管理 (Publish Management) - /publish
├── 客户管理 (Customer Management) - /customer
├── 系统设置 (System Settings) - /settings
├── AI文本优化监控 (AI Optimization) - /ai-optimization
└── 系统监控运维 (System Monitoring) - /monitoring
```

## 1. 仪表板页面 (Dashboard)

### 功能描述
系统总览和关键指标展示，实时监控所有微服务状态

### 主要功能
- 每日数据获取和处理统计
- 微服务健康状态监控
- 收益统计和趋势分析
- 实时任务进度和队列状态
- 系统性能指标展示

### 主要组件
- 数据统计卡片组
- 微服务状态监控面板
- 收益分析图表
- 任务队列进度条
- 实时活动时间线

### 技术实现
```typescript
// 仪表板数据接口
interface DashboardData {
  // 内容统计
  totalArticles: number;
  processedToday: number;
  publishedCount: number;
  pendingReview: number;
  
  // 微服务状态
  serviceStatus: {
    dataSource: 'healthy' | 'warning' | 'error';
    aiModel: 'healthy' | 'warning' | 'error';
    contentProcessing: 'healthy' | 'warning' | 'error';
    publishing: 'healthy' | 'warning' | 'error';
    customerMessage: 'healthy' | 'warning' | 'error';
  };
  
  // 收益数据
  revenue: {
    today: number;
    thisMonth: number;
    trend: number[];
  };
  
  // 任务队列
  taskQueues: {
    processing: number;
    publishing: number;
    aiOptimization: number;
  };
  
  recentActivities: Activity[];
}

// 组件示例
<template>
  <div class="dashboard">
    <div class="stats-grid">
      <StatisticsCards :data="statistics" />
      <ServiceStatusPanel :services="serviceStatus" />
    </div>
    <div class="charts-section">
      <RevenueChart :data="revenueData" />
      <TaskQueueMonitor :queues="taskQueues" />
    </div>
    <ActivityTimeline :activities="activities" />
  </div>
</template>
```

### 页面布局
```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useDashboard } from '@/composables/useDashboard'

const { 
  dashboardData, 
  loading, 
  refreshData, 
  realTimeUpdates 
} = useDashboard()

onMounted(() => {
  refreshData()
  realTimeUpdates.start()
})
</script>

<template>
  <div class="dashboard-container">
    <!-- 顶部统计卡片 -->
    <div class="stats-overview">
      <el-row :gutter="20">
        <el-col :span="6">
          <StatCard 
            title="今日处理"
            :value="dashboardData.processedToday"
            icon="DocumentChecked"
            color="#409EFF"
          />
        </el-col>
        <el-col :span="6">
          <StatCard 
            title="待审核"
            :value="dashboardData.pendingReview"
            icon="Clock"
            color="#E6A23C"
          />
        </el-col>
        <el-col :span="6">
          <StatCard 
            title="已发布"
            :value="dashboardData.publishedCount"
            icon="Check"
            color="#67C23A"
          />
        </el-col>
        <el-col :span="6">
          <StatCard 
            title="总收益"
            :value="dashboardData.revenue.thisMonth"
            icon="Money"
            color="#F56C6C"
            format="currency"
          />
        </el-col>
      </el-row>
    </div>

    <!-- 中间监控面板 -->
    <div class="monitoring-section">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="微服务状态监控">
            <ServiceStatusGrid :services="dashboardData.serviceStatus" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="任务队列">
            <TaskQueuePanel :queues="dashboardData.taskQueues" />
          </el-card>
        </el-col>
      </el-row>
    </div>

    <!-- 底部图表和活动 -->
    <div class="analytics-section">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="收益趋势分析">
            <RevenueChart :data="dashboardData.revenue" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="最近活动">
            <ActivityTimeline :activities="dashboardData.recentActivities" />
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>
```

## 2. 数据源管理页面 (Data Source)

### 功能描述
多平台爬虫配置、数据获取监控和手动内容添加，对应PRD中的FR1数据获取模块

### 主要功能
- 支持今日头条、百家号、小红书等主流平台爬虫配置
- 实时监控各平台数据获取状态和反封禁策略
- 手动添加内容的Web界面，支持文本和图片上传
- 预留其他数据获取渠道的扩展接口配置
- 数据获取性能分析和统计报表
- 爬虫任务调度和队列管理

### 关键组件
```typescript
// 数据源配置接口
interface DataSourceConfig {
  id: string;
  platform: 'toutiao' | 'baijiahao' | 'xiaohongshu' | 'custom';
  name: string;
  enabled: boolean;
  config: {
    apiKey?: string;
    crawlInterval: number;
    maxPages: number;
    keywords: string[];
    filters: {
      minLength: number;
      excludeKeywords: string[];
    };
  };
  status: 'active' | 'paused' | 'error';
  lastRun: Date;
  totalCrawled: number;
}

// 手动内容上传接口
interface ManualContent {
  title: string;
  content: string;
  images: File[];
  source: string;
  tags: string[];
  category: string;
}
```

### 页面布局
```vue
<template>
  <div class="data-source-management">
    <!-- 平台配置区域 -->
    <div class="platform-configs">
      <el-card>
        <template #header>
          <div class="card-header">
            <span>爬虫平台配置</span>
            <el-button type="primary" @click="handleAddPlatform">
              添加平台
            </el-button>
          </div>
        </template>
        
        <div class="platform-grid">
          <PlatformConfigCard 
            v-for="platform in platforms" 
            :key="platform.id"
            :config="platform"
            @update="handleConfigUpdate"
            @toggle="handleTogglePlatform"
            @test="handleTestConnection"
          />
        </div>
      </el-card>
    </div>
    
    <!-- 手动上传区域 -->
    <div class="manual-upload">
      <el-card>
        <template #header>
          <span>手动添加内容</span>
        </template>
        
        <ManualContentForm 
          @submit="handleManualUpload"
          :uploading="uploadLoading"
        />
      </el-card>
    </div>
    
    <!-- 监控面板 -->
    <div class="monitoring-panel">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="实时监控">
            <CrawlerStatusMonitor :platforms="platforms" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="数据统计">
            <DataSourceMetrics :metrics="sourceMetrics" />
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>
```

## 3. 内容管理页面 (Content Management)

### 功能描述
内容分类、审核、编辑和质量控制，对应PRD中的FR2数据整理存储和FR3内容审核去重

### 主要功能
- 按固定格式存储和展示获取的内容
- 相同图片描述的文章分类管理
- 文本内容相似度检测和分类展示
- 基于OpenCV的图片内容一致性检测结果
- 按日期顺序展示每日获取的数据
- 相同内容的自动检测和标记
- 重复内容的删除和保留决策支持
- 内容质量评估和审核工作流
- AI文本优化处理状态跟踪

### 关键组件
```typescript
// 内容项接口
interface ContentItem {
  id: string;
  title: string;
  content: string;
  images: string[];
  source: string;
  platform: 'toutiao' | 'baijiahao' | 'xiaohongshu' | 'manual';
  category: string;
  status: 'pending' | 'approved' | 'rejected' | 'processing' | 'ai_optimizing' | 'duplicate';
  qualityScore: number;
  
  // 相似度检测结果
  similarity: {
    textSimilarity: number;
    imageSimilarity: number;
    duplicateIds: string[];
  };
  
  // AI优化状态
  aiOptimization: {
    status: 'pending' | 'processing' | 'completed' | 'failed';
    optimizedContent?: string;
    optimizedAt?: Date;
    format: 'time_person_place_feeling';
  };
  
  // 分类信息
  classification: {
    autoCategory: string;
    confidence: number;
    tags: string[];
  };
  
  createdAt: Date;
  updatedAt: Date;
  processedAt?: Date;
}
```

### 页面布局
```vue
<template>
  <div class="content-management">
    <!-- 管理头部 -->
    <div class="management-header">
      <el-card>
        <SearchFilters 
          @filter="handleFilter" 
          :categories="categories"
          :platforms="platforms"
        />
        <DailyReviewReport :date="selectedDate" />
      </el-card>
    </div>
    
    <!-- 内容网格 -->
    <div class="content-grid">
      <el-row :gutter="20">
        <el-col :span="selectedContent ? 16 : 24">
          <el-card>
            <ContentTable 
              :items="contentList" 
              :loading="loading"
              @select="handleSelect"
              @edit="handleEdit"
              @preview="handlePreview"
              @ai-optimize="handleAIOptimize"
              @merge="handleMerge"
            />
          </el-card>
        </el-col>
        
        <el-col v-if="selectedContent" :span="8">
          <el-card title="相似内容分析">
            <SimilarityPanel 
              :content="selectedContent"
              :similar-items="similarItems"
              @merge="handleMergeSimilar"
            />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 批量操作 -->
    <div class="batch-operations">
      <el-card>
        <BatchOperations 
          :selected="selectedItems"
          @approve="handleBatchApprove"
          @reject="handleBatchReject"
          @ai-optimize="handleBatchAIOptimize"
          @delete-duplicates="handleDeleteDuplicates"
        />
      </el-card>
    </div>
    
    <!-- AI优化进度监控 -->
    <div class="ai-optimization-monitor">
      <el-card title="AI优化进度">
        <AIOptimizationMonitor 
          :queue="aiOptimizationQueue"
          @retry="handleRetryOptimization"
        />
      </el-card>
    </div>
  </div>
</template>
```

## 4. 发布管理页面 (Publish Management)

### 功能描述
多平台发布配置、状态跟踪和统计分析，对应PRD中的FR5内容发布管理

### 主要功能
- 处理后内容按日期存储和管理
- YouTube等平台的自动发布接口配置
- 手动发布平台的邮件通知功能
- 发布状态跟踪和管理
- 支持至少5个平台的API接入
- 定时发布和批量发布功能
- 发布失败重试机制
- 发布队列和优先级管理
- 发布效果统计和收益分析

### 关键组件
```typescript
// 发布平台配置接口
interface PublishPlatform {
  id: string;
  name: string;
  type: 'auto' | 'manual';
  enabled: boolean;
  config: {
    apiKey?: string;
    apiSecret?: string;
    accessToken?: string;
    webhookUrl?: string;
    publishSettings: {
      autoPublish: boolean;
      scheduleEnabled: boolean;
      defaultTags: string[];
      contentFormat: string;
    };
  };
  status: 'connected' | 'disconnected' | 'error';
  lastPublish?: Date;
  totalPublished: number;
  successRate: number;
}

// 发布任务接口
interface PublishTask {
  id: string;
  contentId: string;
  platforms: string[];
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'scheduled';
  scheduledAt?: Date;
  publishedAt?: Date;
  priority: 'low' | 'medium' | 'high';
  retryCount: number;
  results: {
    platform: string;
    status: 'success' | 'failed';
    publishedUrl?: string;
    error?: string;
    publishedAt?: Date;
  }[];
}

// 发布统计接口
interface PublishStats {
  totalPublished: number;
  successRate: number;
  platformStats: {
    platform: string;
    published: number;
    views: number;
    engagement: number;
    revenue: number;
  }[];
  dailyStats: {
    date: string;
    published: number;
    revenue: number;
  }[];
}
```

### 页面布局
```vue
<template>
  <div class="publish-management">
    <!-- 平台配置 -->
    <div class="platform-config">
      <el-card>
        <template #header>
          <div class="card-header">
            <span>发布平台配置</span>
            <el-button type="primary" @click="handleAddPlatform">
              添加平台
            </el-button>
          </div>
        </template>
        
        <PlatformConfigGrid 
          :platforms="platforms"
          @update="handlePlatformUpdate"
          @test-connection="handleTestConnection"
        />
      </el-card>
    </div>
    
    <!-- 发布队列 -->
    <div class="publish-queue">
      <el-card title="发布队列">
        <PublishQueueTable 
          :tasks="publishTasks"
          @retry="handleRetryPublish"
          @cancel="handleCancelPublish"
          @schedule="handleSchedulePublish"
        />
      </el-card>
    </div>
    
    <!-- 发布统计 -->
    <div class="publish-stats">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="发布统计">
            <PublishStatsCharts :stats="publishStats" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="收益分析">
            <RevenueAnalysis :revenue="revenueData" />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 批量操作 -->
    <div class="batch-operations">
      <el-card title="批量发布">
        <BatchPublishPanel 
          :selected-content="selectedContent"
          @publish="handleBatchPublish"
          @schedule="handleBatchSchedule"
        />
      </el-card>
    </div>
  </div>
</template>
```

## 5. 客户管理页面 (Customer Management)

### 功能描述
客户档案管理、消息发送和收益统计，对应PRD中的客户消息服务和收益管理

### 主要功能
- 客户档案管理和服务记录
- 客户分群和标签管理功能
- 支持邮件、短信、推送等多渠道消息发送
- 消息模板管理和定制化功能
- 批量发送和定时发送
- 客户满意度调查和反馈管理
- 收益统计和分析仪表板
- 多维度报表和趋势分析
- 收益预测和商业智能分析

### 关键组件
```typescript
// 客户信息接口
interface Customer {
  id: string;
  name: string;
  email: string;
  phone?: string;
  tags: string[];
  segment: string;
  status: 'active' | 'inactive' | 'vip';
  totalRevenue: number;
  lastContact: Date;
  preferences: {
    contentTypes: string[];
    frequency: 'daily' | 'weekly' | 'monthly';
    channels: ('email' | 'sms' | 'push')[];
  };
  metrics: {
    engagementRate: number;
    conversionRate: number;
    avgOrderValue: number;
  };
}

// 消息模板接口
interface MessageTemplate {
  id: string;
  name: string;
  type: 'email' | 'sms' | 'push';
  subject?: string;
  content: string;
  variables: string[];
  status: 'active' | 'draft';
  usage: number;
}
```

### 页面布局
```vue
<template>
  <div class="customer-management">
    <!-- 客户概览 -->
    <div class="customer-overview">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="客户指标">
            <CustomerMetrics :metrics="customerMetrics" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="收益分析">
            <RevenueAnalytics :revenue="revenueData" />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 客户列表 -->
    <div class="customer-list">
      <el-card title="客户管理">
        <CustomerTable 
          :customers="customers"
          @edit="handleEditCustomer"
          @send-message="handleSendMessage"
          @view-details="handleViewDetails"
        />
      </el-card>
    </div>
    
    <!-- 消息中心 -->
    <div class="message-center">
      <el-row :gutter="20">
        <el-col :span="12">
          <el-card title="消息模板">
            <MessageTemplates 
              :templates="messageTemplates"
              @edit="handleEditTemplate"
              @delete="handleDeleteTemplate"
            />
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card title="批量发送">
            <BulkMessageSender 
              :selected-customers="selectedCustomers"
              :templates="messageTemplates"
              @send="handleBulkSend"
            />
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>
```

## 6. 系统设置页面 (System Settings)

### 功能描述
系统配置、用户权限和监控设置，对应PRD中的AI大模型服务配置和系统管理

### 主要功能
- 独立AI大模型服务配置
- 支持多平台API接入(OpenAI、Claude、文心一言、通义千问等)
- 账号池管理和自动轮换机制
- 多模型切换和负载均衡配置
- Key值管理和安全存储
- API调用监控和费用统计
- 用户权限控制和角色管理
- 系统参数设置和配置管理
- 监控告警配置和通知设置
- 备份恢复和灾难恢复管理

### 关键组件
```typescript
// AI模型配置接口
interface AIModelConfig {
  id: string;
  provider: 'openai' | 'claude' | 'wenxin' | 'tongyi' | 'custom';
  modelName: string;
  apiKey: string;
  endpoint?: string;
  enabled: boolean;
  priority: number;
  rateLimit: {
    requestsPerMinute: number;
    tokensPerMinute: number;
  };
  cost: {
    inputTokenPrice: number;
    outputTokenPrice: number;
  };
  usage: {
    totalRequests: number;
    totalTokens: number;
    totalCost: number;
    lastUsed: Date;
  };
}

// 系统配置接口
interface SystemConfig {
  ai: {
    defaultModel: string;
    fallbackModels: string[];
    optimizationTemplate: string;
    qualityThreshold: number;
  };
  processing: {
    batchSize: number;
    maxRetries: number;
    timeoutSeconds: number;
  };
  storage: {
    retentionDays: number;
    backupEnabled: boolean;
    backupSchedule: string;
  };
  notifications: {
    emailEnabled: boolean;
    webhookUrl?: string;
    alertThresholds: {
      errorRate: number;
      responseTime: number;
    };
  };
}
```

### 页面布局
```vue
<template>
  <div class="system-settings">
    <!-- AI模型配置 -->
    <div class="ai-model-config">
      <el-card>
        <template #header>
          <div class="card-header">
            <span>AI模型配置</span>
            <el-button type="primary" @click="handleAddModel">
              添加模型
            </el-button>
          </div>
        </template>
        
        <AIModelGrid 
          :models="aiModels"
          @update="handleModelUpdate"
          @test="handleModelTest"
          @delete="handleDeleteModel"
        />
        
        <div class="model-usage">
          <ModelUsageStats :usage="modelUsage" />
        </div>
      </el-card>
    </div>
    
    <!-- 用户管理 -->
    <div class="user-management">
      <el-card title="用户权限管理">
        <UserRoleTable 
          :users="users"
          :roles="roles"
          @update-role="handleUpdateRole"
          @add-user="handleAddUser"
        />
      </el-card>
    </div>
    
    <!-- 系统配置 -->
    <div class="system-config">
      <el-card title="系统参数">
        <ConfigurationForm 
          :config="systemConfig"
          @save="handleSaveConfig"
        />
      </el-card>
    </div>
    
    <!-- 监控配置 -->
    <div class="monitoring-config">
      <el-card title="监控告警">
        <AlertConfigPanel 
          :alerts="alertConfig"
          @update="handleUpdateAlerts"
        />
      </el-card>
    </div>
  </div>
</template>
```

## 7. AI文本优化监控页面 (AI Optimization Monitor)

### 功能描述
AI文本优化处理监控，对应PRD中的FR4内容文本优化处理

### 主要功能
- 图片优化和水印去除功能监控
- 基于大模型的文本文本优化状态跟踪
- 按照"时间、人物、地点、感触"格式重写文本的进度监控
- 多个相似内容合并生成单一最终版本的管理
- AI模型调用监控和费用统计
- 文本优化质量评估和人工审核
- 处理队列管理和优先级调整

### 关键组件
```typescript
// AI优化任务接口
interface AIOptimizationTask {
  id: string;
  contentId: string;
  type: 'text_rewrite' | 'image_process' | 'content_merge';
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'reviewing';
  priority: 'low' | 'medium' | 'high';
  
  // 原始内容
  originalContent: {
    title: string;
    content: string;
    images: string[];
  };
  
  // 优化配置
  optimizationConfig: {
    format: 'time_person_place_feeling';
    model: string;
    temperature: number;
    maxTokens: number;
    customPrompt?: string;
  };
  
  // 优化结果
  result?: {
    optimizedTitle: string;
    optimizedContent: string;
    processedImages: string[];
    qualityScore: number;
    aiCost: number;
    processingTime: number;
  };
  
  // 审核信息
  review?: {
    status: 'approved' | 'rejected' | 'needs_revision';
    reviewer: string;
    feedback: string;
    reviewedAt: Date;
  };
  
  createdAt: Date;
  completedAt?: Date;
  error?: string;
}

// 内容合并任务接口
interface ContentMergeTask {
  id: string;
  sourceContentIds: string[];
  mergeStrategy: 'best_quality' | 'comprehensive' | 'custom';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: {
    mergedContent: ContentItem;
    mergeReport: {
      selectedSources: string[];
      mergeReason: string;
      qualityImprovement: number;
    };
  };
}
```

### 页面布局
```vue
<template>
  <div class="ai-optimization-monitor">
    <!-- 优化概览 -->
    <div class="optimization-overview">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="任务指标">
            <AITaskMetrics :metrics="taskMetrics" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="模型使用统计">
            <ModelUsageStats :usage="modelUsage" />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 任务队列 -->
    <div class="task-queue">
      <el-card title="优化任务队列">
        <AITaskTable 
          :tasks="optimizationTasks"
          @retry="handleRetryTask"
          @cancel="handleCancelTask"
          @review="handleReviewTask"
          @view-details="handleViewTaskDetails"
        />
      </el-card>
    </div>
    
    <!-- 合并管理 -->
    <div class="merge-management">
      <el-card title="内容合并管理">
        <ContentMergePanel 
          :merge-tasks="mergeTasks"
          @create-merge="handleCreateMerge"
          @approve-merge="handleApproveMerge"
        />
      </el-card>
    </div>
    
    <!-- 质量控制 -->
    <div class="quality-control">
      <el-card title="质量控制">
        <QualityReviewPanel 
          :pending-reviews="pendingReviews"
          @approve="handleApproveContent"
          @reject="handleRejectContent"
          @request-revision="handleRequestRevision"
        />
      </el-card>
    </div>
    
    <!-- 优化设置 -->
    <div class="optimization-settings">
      <el-card title="优化设置">
        <OptimizationConfigForm 
          :config="optimizationConfig"
          @save="handleSaveConfig"
        />
      </el-card>
    </div>
  </div>
</template>
```

## 8. 系统监控和运维界面 (System Monitoring & Operations)

### 功能描述
系统监控和运维管理界面，对应PRD中的系统监控和运维需求

### 主要功能
- 微服务健康状态监控和告警
- 系统性能指标实时监控（CPU、内存、网络、磁盘）
- 错误日志收集和分析
- 依赖服务状态跟踪
- 自动化运维操作（重启、扩容、回滚）
- 备份和灾难恢复管理
- API调用监控和限流管理
- 数据库连接池和查询性能监控

### 关键组件
```typescript
// 系统健康状态接口
interface SystemHealth {
  overall: 'healthy' | 'warning' | 'critical';
  services: ServiceHealth[];
  infrastructure: InfrastructureHealth;
  lastUpdated: Date;
}

interface ServiceHealth {
  name: string;
  status: 'up' | 'down' | 'degraded';
  version: string;
  uptime: number;
  responseTime: number;
  errorRate: number;
  instances: number;
  activeInstances: number;
  dependencies: string[];
  metrics: {
    cpu: number;
    memory: number;
    requests: number;
    errors: number;
  };
}

interface InfrastructureHealth {
  database: {
    status: 'connected' | 'disconnected' | 'slow';
    connections: number;
    maxConnections: number;
    queryTime: number;
    slowQueries: number;
  };
  messageQueue: {
    status: 'healthy' | 'backlog' | 'error';
    queueSize: number;
    processingRate: number;
    deadLetters: number;
  };
  cache: {
    status: 'healthy' | 'degraded';
    hitRate: number;
    memoryUsage: number;
    evictions: number;
  };
  storage: {
    diskUsage: number;
    availableSpace: number;
    ioLatency: number;
  };
}

// 运维操作接口
interface OperationTask {
  id: string;
  type: 'restart' | 'scale' | 'deploy' | 'rollback' | 'backup';
  target: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  parameters: Record<string, any>;
  startedAt: Date;
  completedAt?: Date;
  result?: {
    success: boolean;
    message: string;
    logs: string[];
  };
  operator: string;
}

// 告警规则接口
interface AlertRule {
  id: string;
  name: string;
  metric: string;
  condition: 'gt' | 'lt' | 'eq';
  threshold: number;
  duration: number;
  severity: 'info' | 'warning' | 'critical';
  enabled: boolean;
  notifications: {
    email: string[];
    webhook?: string;
    slack?: string;
  };
}
```

### 页面布局
```vue
<template>
  <div class="system-monitoring">
    <!-- 监控概览 -->
    <div class="monitoring-overview">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="系统健康状态">
            <SystemHealthCard :health="systemHealth" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="告警摘要">
            <AlertSummary :alerts="activeAlerts" />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 服务网格 -->
    <div class="service-grid">
      <el-card title="微服务状态">
        <ServiceStatusGrid 
          :services="services"
          @restart="handleRestartService"
          @scale="handleScaleService"
          @view-logs="handleViewLogs"
        />
      </el-card>
    </div>
    
    <!-- 基础设施监控 -->
    <div class="infrastructure-monitoring">
      <el-row :gutter="20">
        <el-col :span="8">
          <el-card title="基础设施指标">
            <InfrastructureMetrics :metrics="infrastructureMetrics" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="数据库监控">
            <DatabaseMonitor :database="databaseHealth" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="消息队列">
            <MessageQueueMonitor :queue="queueHealth" />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 性能图表 -->
    <div class="performance-charts">
      <el-card title="性能指标">
        <MetricsChart 
          :data="performanceData"
          :time-range="selectedTimeRange"
          @time-range-change="handleTimeRangeChange"
        />
      </el-card>
    </div>
    
    <!-- 运维面板 -->
    <div class="operations-panel">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="运维操作历史">
            <OperationHistory :tasks="operationTasks" />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="快速操作">
            <QuickActions 
              @backup="handleBackup"
              @health-check="handleHealthCheck"
              @clear-cache="handleClearCache"
            />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 告警管理 -->
    <div class="alert-management">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="告警规则">
            <AlertRuleEditor 
              :rules="alertRules"
              @save="handleSaveAlertRule"
              @delete="handleDeleteAlertRule"
            />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="告警历史">
            <AlertHistory :alerts="alertHistory" />
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 日志分析 -->
    <div class="log-analysis">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card title="日志查看器">
            <LogViewer 
              :logs="systemLogs"
              :filters="logFilters"
              @filter-change="handleLogFilterChange"
            />
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card title="错误分析">
            <ErrorAnalysis :errors="errorStats" />
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>
```

## 页面间导航和状态管理

### 路由配置
```typescript
// router/index.ts
const routes = [
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/pages/dashboard/DashboardPage.vue'),
    meta: { title: '仪表板', icon: 'Dashboard' }
  },
  {
    path: '/data-source',
    name: 'DataSource',
    component: () => import('@/pages/data-source/DataSourcePage.vue'),
    meta: { title: '数据源管理', icon: 'DataLine' }
  },
  {
    path: '/content',
    name: 'Content',
    component: () => import('@/pages/content/ContentPage.vue'),
    meta: { title: '内容管理', icon: 'Document' }
  },
  {
    path: '/publish',
    name: 'Publish',
    component: () => import('@/pages/publish/PublishPage.vue'),
    meta: { title: '发布管理', icon: 'Upload' }
  },
  {
    path: '/customer',
    name: 'Customer',
    component: () => import('@/pages/customer/CustomerPage.vue'),
    meta: { title: '客户管理', icon: 'User' }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@/pages/settings/SettingsPage.vue'),
    meta: { title: '系统设置', icon: 'Setting' }
  },
  {
    path: '/ai-optimization',
    name: 'AIOptimization',
    component: () => import('@/pages/ai-optimization/AIOptimizationPage.vue'),
    meta: { title: 'AI优化监控', icon: 'MagicStick' }
  },
  {
    path: '/monitoring',
    name: 'Monitoring',
    component: () => import('@/pages/monitoring/MonitoringPage.vue'),
    meta: { title: '系统监控', icon: 'Monitor' }
  }
]
```

### 全局状态管理
```typescript
// stores/app.ts
export const useAppStore = defineStore('app', () => {
  const currentPage = ref('dashboard')
  const breadcrumbs = ref<BreadcrumbItem[]>([])
  const notifications = ref<Notification[]>([])
  
  const setCurrentPage = (page: string) => {
    currentPage.value = page
  }
  
  const updateBreadcrumbs = (items: BreadcrumbItem[]) => {
    breadcrumbs.value = items
  }
  
  const addNotification = (notification: Notification) => {
    notifications.value.unshift(notification)
  }
  
  return {
    currentPage,
    breadcrumbs,
    notifications,
    setCurrentPage,
    updateBreadcrumbs,
    addNotification
  }
})
```

---

*本文档详细描述了历史文本优化项目的8个核心页面设计，为前端开发提供了完整的功能规范和技术实现指导。*