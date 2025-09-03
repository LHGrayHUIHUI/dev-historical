# 前端组件设计规范

## 概述

本文档定义了历史文本优化项目前端应用的组件设计规范，包括基础组件、业务组件、图表组件等的设计标准和实现示例。

## 组件分类

### 1. 基础组件 (Basic Components)
基础UI组件，提供通用的界面元素

### 2. 业务组件 (Business Components)
结合具体业务逻辑的复合组件

### 3. 图表组件 (Chart Components)
数据可视化相关组件

### 4. 监控组件 (Monitoring Components)
系统监控和状态展示组件

---

## 1. 基础组件 (Basic Components)

### 按钮组件 (Button)

```typescript
// components/common/Button.vue
interface ButtonProps {
  type?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: string;
}

<template>
  <el-button
    :type="type"
    :size="size"
    :disabled="disabled"
    :loading="loading"
    :icon="icon"
    @click="handleClick"
  >
    <slot />
  </el-button>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<ButtonProps>(), {
  type: 'primary',
  size: 'medium',
  disabled: false,
  loading: false
});

const emit = defineEmits<{
  click: [event: MouseEvent];
}>();

const handleClick = (event: MouseEvent) => {
  if (!props.disabled && !props.loading) {
    emit('click', event);
  }
};
</script>
```

### 数据表格组件 (DataTable)

```typescript
// components/common/DataTable.vue
interface TableColumn {
  key: string;
  label: string;
  width?: string;
  sortable?: boolean;
  formatter?: (value: any, row: any) => string;
}

interface TableProps {
  data: any[];
  columns: TableColumn[];
  loading?: boolean;
  pagination?: {
    page: number;
    pageSize: number;
    total: number;
  };
  selectable?: boolean;
}

<template>
  <div class="data-table">
    <el-table
      :data="data"
      :loading="loading"
      @selection-change="handleSelectionChange"
      @sort-change="handleSortChange"
    >
      <el-table-column
        v-if="selectable"
        type="selection"
        width="55"
      />
      
      <el-table-column
        v-for="column in columns"
        :key="column.key"
        :prop="column.key"
        :label="column.label"
        :width="column.width"
        :sortable="column.sortable"
      >
        <template #default="{ row }" v-if="column.formatter">
          {{ column.formatter(row[column.key], row) }}
        </template>
      </el-table-column>
      
      <el-table-column label="操作" width="200">
        <template #default="{ row }">
          <slot name="actions" :row="row" />
        </template>
      </el-table-column>
    </el-table>
    
    <el-pagination
      v-if="pagination"
      :current-page="pagination.page"
      :page-size="pagination.pageSize"
      :total="pagination.total"
      @current-change="handlePageChange"
      @size-change="handleSizeChange"
    />
  </div>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<TableProps>(), {
  loading: false,
  selectable: false
});

const emit = defineEmits<{
  'selection-change': [selection: any[]];
  'sort-change': [{ column: any; prop: string; order: string }];
  'page-change': [page: number];
  'size-change': [size: number];
}>();

const handleSelectionChange = (selection: any[]) => {
  emit('selection-change', selection);
};

const handleSortChange = (sortInfo: any) => {
  emit('sort-change', sortInfo);
};

const handlePageChange = (page: number) => {
  emit('page-change', page);
};

const handleSizeChange = (size: number) => {
  emit('size-change', size);
};
</script>
```

### 表单组件 (Form)

```typescript
// components/common/Form.vue
interface FormField {
  key: string;
  label: string;
  type: 'input' | 'select' | 'textarea' | 'date' | 'number';
  required?: boolean;
  options?: { label: string; value: any }[];
  placeholder?: string;
  rules?: any[];
}

interface FormProps {
  fields: FormField[];
  modelValue: Record<string, any>;
  labelWidth?: string;
}

<template>
  <el-form
    ref="formRef"
    :model="modelValue"
    :label-width="labelWidth"
    @submit.prevent="handleSubmit"
  >
    <el-form-item
      v-for="field in fields"
      :key="field.key"
      :label="field.label"
      :prop="field.key"
      :rules="field.rules"
    >
      <!-- 输入框 -->
      <el-input
        v-if="field.type === 'input'"
        v-model="modelValue[field.key]"
        :placeholder="field.placeholder"
      />
      
      <!-- 选择器 -->
      <el-select
        v-else-if="field.type === 'select'"
        v-model="modelValue[field.key]"
        :placeholder="field.placeholder"
      >
        <el-option
          v-for="option in field.options"
          :key="option.value"
          :label="option.label"
          :value="option.value"
        />
      </el-select>
      
      <!-- 文本域 -->
      <el-input
        v-else-if="field.type === 'textarea'"
        v-model="modelValue[field.key]"
        type="textarea"
        :placeholder="field.placeholder"
      />
      
      <!-- 日期选择器 -->
      <el-date-picker
        v-else-if="field.type === 'date'"
        v-model="modelValue[field.key]"
        type="date"
        :placeholder="field.placeholder"
      />
      
      <!-- 数字输入框 -->
      <el-input-number
        v-else-if="field.type === 'number'"
        v-model="modelValue[field.key]"
        :placeholder="field.placeholder"
      />
    </el-form-item>
    
    <el-form-item>
      <slot name="actions" />
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const props = withDefaults(defineProps<FormProps>(), {
  labelWidth: '120px'
});

const emit = defineEmits<{
  'update:modelValue': [value: Record<string, any>];
  submit: [value: Record<string, any>];
}>;

const formRef = ref();

const handleSubmit = async () => {
  try {
    await formRef.value.validate();
    emit('submit', props.modelValue);
  } catch (error) {
    console.error('表单验证失败:', error);
  }
};

// 暴露验证方法
defineExpose({
  validate: () => formRef.value.validate(),
  resetFields: () => formRef.value.resetFields()
});
</script>
```

---

## 2. 业务组件 (Business Components)

### 内容卡片组件 (ContentCard)

```typescript
// components/business/ContentCard.vue
interface ContentItem {
  id: string;
  title: string;
  content: string;
  status: 'pending' | 'approved' | 'rejected';
  author: string;
  createdAt: Date;
  tags: string[];
}

interface ContentCardProps {
  item: ContentItem;
  showActions?: boolean;
}

<template>
  <div class="content-card">
    <div class="card-header">
      <h3 class="card-title">{{ item.title }}</h3>
      <div class="card-status">
        <el-tag :type="getStatusType(item.status)">
          {{ getStatusText(item.status) }}
        </el-tag>
      </div>
    </div>
    
    <div class="card-body">
      <p class="card-content">{{ truncatedContent }}</p>
      <div class="card-meta">
        <span class="author">作者: {{ item.author }}</span>
        <span class="date">{{ formatDate(item.createdAt) }}</span>
      </div>
      <div class="card-tags">
        <el-tag
          v-for="tag in item.tags"
          :key="tag"
          size="small"
          effect="plain"
        >
          {{ tag }}
        </el-tag>
      </div>
    </div>
    
    <div class="card-actions" v-if="showActions">
      <el-button size="small" @click="handleView">查看</el-button>
      <el-button size="small" type="primary" @click="handleEdit">编辑</el-button>
      <el-button size="small" type="success" @click="handleApprove">审核</el-button>
      <el-button size="small" type="danger" @click="handleDelete">删除</el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = withDefaults(defineProps<ContentCardProps>(), {
  showActions: true
});

const emit = defineEmits<{
  view: [id: string];
  edit: [id: string];
  approve: [id: string];
  delete: [id: string];
}>();

const truncatedContent = computed(() => {
  return props.item.content.length > 100 
    ? props.item.content.slice(0, 100) + '...' 
    : props.item.content;
});

const getStatusType = (status: string) => {
  const statusMap = {
    pending: 'warning',
    approved: 'success',
    rejected: 'danger'
  };
  return statusMap[status] || 'info';
};

const getStatusText = (status: string) => {
  const statusMap = {
    pending: '待审核',
    approved: '已审核',
    rejected: '已拒绝'
  };
  return statusMap[status] || '未知';
};

const formatDate = (date: Date) => {
  return new Date(date).toLocaleDateString();
};

const handleView = () => emit('view', props.item.id);
const handleEdit = () => emit('edit', props.item.id);
const handleApprove = () => emit('approve', props.item.id);
const handleDelete = () => emit('delete', props.item.id);
</script>

<style scoped lang="scss">
.content-card {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .card-header {
    @apply flex justify-between items-center mb-3;
    
    .card-title {
      @apply text-lg font-semibold text-gray-900;
    }
  }
  
  .card-body {
    @apply mb-4;
    
    .card-content {
      @apply text-gray-700 mb-2;
    }
    
    .card-meta {
      @apply flex justify-between text-sm text-gray-500 mb-2;
    }
    
    .card-tags {
      @apply flex flex-wrap gap-1;
    }
  }
  
  .card-actions {
    @apply flex justify-end space-x-2;
  }
}
</style>
```

### 仪表盘图表 (GaugeChart)

```typescript
// components/charts/GaugeChart.vue
interface GaugeConfig {
  min: number;
  max: number;
  value: number;
  unit: string;
  thresholds: {
    value: number;
    color: string;
    label: string;
  }[];
}

interface GaugeChartProps {
  title: string;
  config: GaugeConfig;
  size?: number;
}

<template>
  <div class="gauge-chart">
    <div class="gauge-header">
      <h3>{{ title }}</h3>
    </div>
    
    <div class="gauge-container">
      <svg :width="size" :height="size" class="gauge-svg">
        <!-- 背景圆弧 -->
        <path
          :d="backgroundArc"
          :stroke="backgroundColor"
          :stroke-width="strokeWidth"
          fill="none"
        />
        
        <!-- 数值圆弧 -->
        <path
          :d="valueArc"
          :stroke="getValueColor(config.value)"
          :stroke-width="strokeWidth"
          fill="none"
          stroke-linecap="round"
        />
        
        <!-- 指针 -->
        <line
          :x1="centerX"
          :y1="centerY"
          :x2="needleX"
          :y2="needleY"
          stroke="#333"
          stroke-width="2"
        />
        
        <!-- 中心点 -->
        <circle
          :cx="centerX"
          :cy="centerY"
          r="4"
          fill="#333"
        />
      </svg>
      
      <div class="gauge-value">
        <span class="value">{{ config.value }}</span>
        <span class="unit">{{ config.unit }}</span>
      </div>
      
      <div class="gauge-thresholds">
        <div 
          v-for="threshold in config.thresholds"
          :key="threshold.value"
          class="threshold"
        >
          <span 
            class="threshold-color"
            :style="{ backgroundColor: threshold.color }"
          ></span>
          <span class="threshold-label">{{ threshold.label }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = withDefaults(defineProps<GaugeChartProps>(), {
  size: 200
});

const centerX = computed(() => props.size / 2);
const centerY = computed(() => props.size / 2);
const radius = computed(() => props.size / 2 - 20);
const strokeWidth = computed(() => 10);
const backgroundColor = computed(() => '#e5e7eb');

const backgroundArc = computed(() => {
  const startAngle = -Math.PI * 0.75;
  const endAngle = Math.PI * 0.75;
  
  const x1 = centerX.value + radius.value * Math.cos(startAngle);
  const y1 = centerY.value + radius.value * Math.sin(startAngle);
  const x2 = centerX.value + radius.value * Math.cos(endAngle);
  const y2 = centerY.value + radius.value * Math.sin(endAngle);
  
  return `M ${x1} ${y1} A ${radius.value} ${radius.value} 0 1 1 ${x2} ${y2}`;
});

const valueArc = computed(() => {
  const startAngle = -Math.PI * 0.75;
  const valueRatio = (props.config.value - props.config.min) / (props.config.max - props.config.min);
  const endAngle = startAngle + (Math.PI * 1.5 * valueRatio);
  
  const x1 = centerX.value + radius.value * Math.cos(startAngle);
  const y1 = centerY.value + radius.value * Math.sin(startAngle);
  const x2 = centerX.value + radius.value * Math.cos(endAngle);
  const y2 = centerY.value + radius.value * Math.sin(endAngle);
  
  const largeArcFlag = valueRatio > 0.5 ? 1 : 0;
  
  return `M ${x1} ${y1} A ${radius.value} ${radius.value} 0 ${largeArcFlag} 1 ${x2} ${y2}`;
});

const needleX = computed(() => {
  const valueRatio = (props.config.value - props.config.min) / (props.config.max - props.config.min);
  const angle = -Math.PI * 0.75 + (Math.PI * 1.5 * valueRatio);
  return centerX.value + (radius.value - 10) * Math.cos(angle);
});

const needleY = computed(() => {
  const valueRatio = (props.config.value - props.config.min) / (props.config.max - props.config.min);
  const angle = -Math.PI * 0.75 + (Math.PI * 1.5 * valueRatio);
  return centerY.value + (radius.value - 10) * Math.sin(angle);
});

const getValueColor = (value: number) => {
  const threshold = props.config.thresholds
    .slice()
    .reverse()
    .find(t => value >= t.value);
  
  return threshold?.color || '#3b82f6';
};
</script>

<style scoped lang="scss">
.gauge-chart {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4 text-center;
  
  .gauge-header {
    @apply mb-4;
    
    h3 {
      @apply text-lg font-semibold text-gray-900;
    }
  }
  
  .gauge-container {
    @apply relative;
  }
  
  .gauge-svg {
    @apply mx-auto;
  }
  
  .gauge-value {
    @apply absolute inset-0 flex flex-col items-center justify-center;
    
    .value {
      @apply text-3xl font-bold text-gray-900;
    }
    
    .unit {
      @apply text-sm text-gray-600;
    }
  }
  
  .gauge-thresholds {
    @apply flex justify-center space-x-4 mt-4;
    
    .threshold {
      @apply flex items-center space-x-1;
      
      .threshold-color {
        @apply w-3 h-3 rounded-full;
      }
      
      .threshold-label {
        @apply text-xs text-gray-600;
      }
    }
  }
}
</style>
```

---

## 4. 组合式函数规范 (Composable Function Specification)

### 表格数据管理 (useTable)

```typescript
// composables/useTable.ts
import { ref, reactive, computed } from 'vue';

export interface TableOptions {
  pageSize?: number;
  sortable?: boolean;
  filterable?: boolean;
}

export interface TableState {
  data: any[];
  loading: boolean;
  pagination: {
    page: number;
    pageSize: number;
    total: number;
  };
  selection: any[];
  filters: Record<string, any>;
  sort: {
    prop: string;
    order: 'ascending' | 'descending' | null;
  };
}

/**
 * 表格数据管理组合式函数
 * @param fetchFn 数据获取函数
 * @param options 表格配置选项
 */
export function useTable(
  fetchFn: (params: any) => Promise<{ data: any[]; total: number }>,
  options: TableOptions = {}
) {
  const state = reactive<TableState>({
    data: [],
    loading: false,
    pagination: {
      page: 1,
      pageSize: options.pageSize || 20,
      total: 0
    },
    selection: [],
    filters: {},
    sort: {
      prop: '',
      order: null
    }
  });
  
  const hasSelection = computed(() => state.selection.length > 0);
  const hasFilters = computed(() => Object.keys(state.filters).length > 0);
  
  const fetchData = async (params: any = {}) => {
    state.loading = true;
    try {
      const response = await fetchFn({
        ...params,
        ...state.filters,
        page: state.pagination.page,
        pageSize: state.pagination.pageSize,
        sortProp: state.sort.prop,
        sortOrder: state.sort.order
      });
      
      state.data = response.data;
      state.pagination.total = response.total;
    } catch (error) {
      console.error('获取数据失败:', error);
      state.data = [];
      state.pagination.total = 0;
    } finally {
      state.loading = false;
    }
  };
  
  const handlePageChange = (page: number) => {
    state.pagination.page = page;
    fetchData();
  };
  
  const handleSizeChange = (size: number) => {
    state.pagination.pageSize = size;
    state.pagination.page = 1;
    fetchData();
  };
  
  const handleSelectionChange = (selection: any[]) => {
    state.selection = selection;
  };
  
  const handleSortChange = ({ prop, order }: { prop: string; order: string }) => {
    state.sort.prop = prop;
    state.sort.order = order as 'ascending' | 'descending' | null;
    state.pagination.page = 1;
    fetchData();
  };
  
  const setFilters = (filters: Record<string, any>) => {
    state.filters = { ...filters };
    state.pagination.page = 1;
    fetchData();
  };
  
  const clearFilters = () => {
    state.filters = {};
    state.pagination.page = 1;
    fetchData();
  };
  
  const refresh = () => {
    fetchData();
  };
  
  const reset = () => {
    state.pagination.page = 1;
    state.filters = {};
    state.sort = { prop: '', order: null };
    state.selection = [];
    fetchData();
  };
  
  return {
    state: readonly(state),
    hasSelection,
    hasFilters,
    fetchData,
    handlePageChange,
    handleSizeChange,
    handleSelectionChange,
    handleSortChange,
    setFilters,
    clearFilters,
    refresh,
    reset
  };
}
```

### 表单验证管理 (useForm)

```typescript
// composables/useForm.ts
import { ref, reactive, computed } from 'vue';
import type { FormInstance, FormRules } from 'element-plus';

export interface FormOptions {
  initialValues?: Record<string, any>;
  rules?: FormRules;
  validateOnChange?: boolean;
}

/**
 * 表单管理组合式函数
 * @param options 表单配置选项
 */
export function useForm(options: FormOptions = {}) {
  const formRef = ref<FormInstance>();
  const formData = reactive(options.initialValues || {});
  const errors = ref<Record<string, string>>({});
  const isSubmitting = ref(false);
  
  const isValid = computed(() => {
    return Object.keys(errors.value).length === 0;
  });
  
  const hasErrors = computed(() => {
    return Object.keys(errors.value).length > 0;
  });
  
  const validate = async (): Promise<boolean> => {
    if (!formRef.value) return false;
    
    try {
      await formRef.value.validate();
      errors.value = {};
      return true;
    } catch (error) {
      return false;
    }
  };
  
  const validateField = async (field: string): Promise<boolean> => {
    if (!formRef.value) return false;
    
    try {
      await formRef.value.validateField(field);
      delete errors.value[field];
      return true;
    } catch (error) {
      errors.value[field] = error as string;
      return false;
    }
  };
  
  const resetFields = () => {
    if (formRef.value) {
      formRef.value.resetFields();
    }
    errors.value = {};
  };
  
  const clearValidate = () => {
    if (formRef.value) {
      formRef.value.clearValidate();
    }
    errors.value = {};
  };
  
  const setFieldValue = (field: string, value: any) => {
    formData[field] = value;
    
    if (options.validateOnChange) {
      validateField(field);
    }
  };
  
  const setFieldError = (field: string, message: string) => {
    errors.value[field] = message;
  };
  
  const submit = async (submitFn: (data: any) => Promise<any>) => {
    const isFormValid = await validate();
    if (!isFormValid) return;
    
    isSubmitting.value = true;
    try {
      const result = await submitFn(formData);
      return result;
    } catch (error) {
      throw error;
    } finally {
      isSubmitting.value = false;
    }
  };
  
  return {
    formRef,
    formData,
    errors: readonly(errors),
    isValid,
    hasErrors,
    isSubmitting: readonly(isSubmitting),
    validate,
    validateField,
    resetFields,
    clearValidate,
    setFieldValue,
    setFieldError,
    submit
  };
}
```

### API请求管理 (useApi)

```typescript
// composables/useApi.ts
import { ref, computed } from 'vue';

export interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

/**
 * API请求管理组合式函数
 * @param apiFn API请求函数
 */
export function useApi<T>(
  apiFn: (...args: any[]) => Promise<T>
) {
  const state = ref<ApiState<T>>({
    data: null,
    loading: false,
    error: null
  });
  
  const isLoading = computed(() => state.value.loading);
  const hasError = computed(() => state.value.error !== null);
  const hasData = computed(() => state.value.data !== null);
  
  const execute = async (...args: any[]): Promise<T | null> => {
    state.value.loading = true;
    state.value.error = null;
    
    try {
      const result = await apiFn(...args);
      state.value.data = result;
      return result;
    } catch (error) {
      state.value.error = error as Error;
      return null;
    } finally {
      state.value.loading = false;
    }
  };
  
  const reset = () => {
    state.value = {
      data: null,
      loading: false,
      error: null
    };
  };
  
  return {
    state: readonly(state),
    isLoading,
    hasError,
    hasData,
    execute,
    reset
  };
}
```

---

## 组件使用指南

### 1. 组件命名规范
- 组件文件使用 PascalCase 命名：`ContentCard.vue`
- 组件注册使用 PascalCase：`<ContentCard />`
- 组件属性使用 camelCase：`showActions`

### 2. 组件通信规范
- 父子组件通信：使用 props 和 emits
- 跨组件通信：使用 Pinia 状态管理
- 全局事件：使用 mitt 事件总线

### 3. 样式规范
- 使用 Tailwind CSS 工具类
- 组件内部样式使用 scoped
- 全局样式定义在 `styles/` 目录

### 4. 类型安全
- 所有组件必须定义 TypeScript 接口
- Props 和 Emits 必须有明确的类型定义
- 使用泛型提高组件复用性

### 5. 性能优化
- 大数据列表使用虚拟滚动
- 图表组件使用 Canvas 渲染
- 合理使用 v-memo 和 v-once
- 组件懒加载和代码分割

---

## 总结

本文档定义了历史文本优化项目的前端组件设计规范，包括：

1. **基础组件**：Button、DataTable、Form 等通用UI组件
2. **业务组件**：ContentCard、StatusMonitor、AITaskCard 等业务相关组件
3. **图表组件**：MetricsChart、RealtimeChart、HeatmapChart、GaugeChart 等数据可视化组件
4. **组合式函数**：useTable、useForm、useApi 等逻辑复用函数

通过遵循这些规范，可以确保组件的一致性、可维护性和可复用性，提高开发效率和代码质量。

---

**文档版本**: v1.0  
**最后更新**: 2024年1月  
**维护团队**: 前端开发组
```

### 状态监控组件 (StatusMonitor)

```typescript
// components/business/StatusMonitor.vue
interface StatusItem {
  id: string;
  name: string;
  status: 'healthy' | 'warning' | 'error';
  value: string;
  description?: string;
  lastUpdated: Date;
}

interface StatusMonitorProps {
  title: string;
  items: StatusItem[];
  refreshInterval?: number;
}

<template>
  <div class="status-monitor">
    <div class="monitor-header">
      <h3 class="monitor-title">{{ title }}</h3>
      <div class="monitor-actions">
        <el-button size="small" @click="handleRefresh">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>
    </div>
    
    <div class="monitor-grid">
      <div 
        v-for="item in items"
        :key="item.id"
        class="monitor-item"
        :class="`status-${item.status}`"
      >
        <div class="item-header">
          <span class="item-name">{{ item.name }}</span>
          <div class="item-status">
            <el-icon :class="getStatusIcon(item.status)">
              <component :is="getStatusIcon(item.status)" />
            </el-icon>
          </div>
        </div>
        
        <div class="item-value">{{ item.value }}</div>
        
        <div class="item-description" v-if="item.description">
          {{ item.description }}
        </div>
        
        <div class="item-updated">
          最后更新: {{ formatTime(item.lastUpdated) }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';
import { Refresh, CircleCheck, Warning, CircleClose } from '@element-plus/icons-vue';

const props = withDefaults(defineProps<StatusMonitorProps>(), {
  refreshInterval: 30000 // 30秒
});

const emit = defineEmits<{
  refresh: [];
}>;

let intervalId: number | null = null;

const getStatusIcon = (status: string) => {
  const iconMap = {
    healthy: CircleCheck,
    warning: Warning,
    error: CircleClose
  };
  return iconMap[status] || CircleCheck;
};

const formatTime = (date: Date) => {
  return new Date(date).toLocaleTimeString();
};

const handleRefresh = () => {
  emit('refresh');
};

onMounted(() => {
  if (props.refreshInterval > 0) {
    intervalId = setInterval(() => {
      handleRefresh();
    }, props.refreshInterval);
  }
});

onUnmounted(() => {
  if (intervalId) {
    clearInterval(intervalId);
  }
});
</script>

<style scoped lang="scss">
.status-monitor {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .monitor-header {
    @apply flex justify-between items-center mb-4;
    
    .monitor-title {
      @apply text-lg font-semibold text-gray-900;
    }
  }
  
  .monitor-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4;
  }
  
  .monitor-item {
    @apply p-3 rounded-lg border;
    
    &.status-healthy {
      @apply border-green-200 bg-green-50;
    }
    
    &.status-warning {
      @apply border-yellow-200 bg-yellow-50;
    }
    
    &.status-error {
      @apply border-red-200 bg-red-50;
    }
    
    .item-header {
      @apply flex justify-between items-center mb-2;
      
      .item-name {
        @apply font-medium text-gray-900;
      }
      
      .item-status {
        @apply text-lg;
        
        &.status-healthy { @apply text-green-600; }
        &.status-warning { @apply text-yellow-600; }
        &.status-error { @apply text-red-600; }
      }
    }
    
    .item-value {
      @apply text-2xl font-bold text-gray-900 mb-1;
    }
    
    .item-description {
      @apply text-sm text-gray-600 mb-2;
    }
    
    .item-updated {
      @apply text-xs text-gray-500;
    }
  }
}
</style>
```

### AI优化任务组件 (AITaskCard)

```typescript
// components/business/AITaskCard.vue
interface AITask {
  id: string;
  title: string;
  type: 'optimization' | 'analysis' | 'generation';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: Date;
  endTime?: Date;
  inputCount: number;
  outputCount: number;
  errorMessage?: string;
}

interface AITaskCardProps {
  task: AITask;
}

<template>
  <div class="ai-task-card">
    <div class="task-header">
      <div class="task-info">
        <h4 class="task-title">{{ task.title }}</h4>
        <el-tag :type="getTaskTypeColor(task.type)" size="small">
          {{ getTaskTypeText(task.type) }}
        </el-tag>
      </div>
      <div class="task-status">
        <el-tag :type="getStatusColor(task.status)">
          {{ getStatusText(task.status) }}
        </el-tag>
      </div>
    </div>
    
    <div class="task-progress" v-if="task.status === 'running'">
      <el-progress 
        :percentage="task.progress" 
        :status="task.progress === 100 ? 'success' : undefined"
      />
    </div>
    
    <div class="task-stats">
      <div class="stat-item">
        <span class="stat-label">输入数量:</span>
        <span class="stat-value">{{ task.inputCount }}</span>
      </div>
      <div class="stat-item" v-if="task.outputCount > 0">
        <span class="stat-label">输出数量:</span>
        <span class="stat-value">{{ task.outputCount }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">开始时间:</span>
        <span class="stat-value">{{ formatDateTime(task.startTime) }}</span>
      </div>
      <div class="stat-item" v-if="task.endTime">
        <span class="stat-label">结束时间:</span>
        <span class="stat-value">{{ formatDateTime(task.endTime) }}</span>
      </div>
    </div>
    
    <div class="task-error" v-if="task.status === 'failed' && task.errorMessage">
      <el-alert
        :title="task.errorMessage"
        type="error"
        :closable="false"
        show-icon
      />
    </div>
    
    <div class="task-actions">
      <el-button 
        size="small" 
        @click="handleView"
      >
        查看详情
      </el-button>
      <el-button 
        v-if="task.status === 'completed'"
        size="small" 
        type="primary"
        @click="handleDownload"
      >
        下载结果
      </el-button>
      <el-button 
        v-if="task.status === 'running'"
        size="small" 
        type="danger"
        @click="handleCancel"
      >
        取消任务
      </el-button>
      <el-button 
        v-if="task.status === 'failed'"
        size="small" 
        type="warning"
        @click="handleRetry"
      >
        重试
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<AITaskCardProps>();

const emit = defineEmits<{
  view: [id: string];
  download: [id: string];
  cancel: [id: string];
  retry: [id: string];
}>();

const getTaskTypeColor = (type: string) => {
  const colorMap = {
    optimization: 'primary',
    analysis: 'success',
    generation: 'warning'
  };
  return colorMap[type] || 'info';
};

const getTaskTypeText = (type: string) => {
  const textMap = {
    optimization: '文本优化',
    analysis: '相似性分析',
    generation: '内容生成'
  };
  return textMap[type] || '未知类型';
};

const getStatusColor = (status: string) => {
  const colorMap = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  };
  return colorMap[status] || 'info';
};

const getStatusText = (status: string) => {
  const textMap = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败'
  };
  return textMap[status] || '未知状态';
};

const formatDateTime = (date: Date) => {
  return new Date(date).toLocaleString();
};

const handleView = () => emit('view', props.task.id);
const handleDownload = () => emit('download', props.task.id);
const handleCancel = () => emit('cancel', props.task.id);
const handleRetry = () => emit('retry', props.task.id);
</script>

<style scoped lang="scss">
.ai-task-card {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .task-header {
    @apply flex justify-between items-start mb-3;
    
    .task-info {
      @apply flex items-center space-x-2;
      
      .task-title {
        @apply text-base font-medium text-gray-900;
      }
    }
  }
  
  .task-progress {
    @apply mb-3;
  }
  
  .task-stats {
    @apply grid grid-cols-2 gap-2 mb-3 text-sm;
    
    .stat-item {
      @apply flex justify-between;
      
      .stat-label {
        @apply text-gray-600;
      }
      
      .stat-value {
        @apply font-medium text-gray-900;
      }
    }
  }
  
  .task-error {
    @apply mb-3;
  }
  
  .task-actions {
    @apply flex justify-end space-x-2;
  }
}
</style>
```

### 相似性分析组件 (SimilarityPanel)

```typescript
// components/business/SimilarityPanel.vue
interface SimilarityGroup {
  id: string;
  similarity: number;
  items: {
    id: string;
    title: string;
    content: string;
    source: string;
  }[];
}

interface SimilarityPanelProps {
  groups: SimilarityGroup[];
  threshold?: number;
}

<template>
  <div class="similarity-panel">
    <div class="panel-header">
      <h3 class="panel-title">相似内容分析</h3>
      <div class="panel-controls">
        <span class="threshold-label">相似度阈值:</span>
        <el-slider
          v-model="currentThreshold"
          :min="0"
          :max="100"
          :step="5"
          show-input
          @change="handleThresholdChange"
        />
      </div>
    </div>
    
    <div class="similarity-groups">
      <div 
        v-for="group in filteredGroups"
        :key="group.id"
        class="similarity-group"
      >
        <div class="group-header">
          <div class="group-info">
            <span class="similarity-score">相似度: {{ group.similarity }}%</span>
            <span class="item-count">{{ group.items.length }} 个项目</span>
          </div>
          <div class="group-actions">
            <el-button size="small" @click="handleMerge(group.id)">
              合并内容
            </el-button>
            <el-button size="small" @click="handleKeepAll(group.id)">
              保留全部
            </el-button>
          </div>
        </div>
        
        <div class="group-items">
          <div 
            v-for="item in group.items"
            :key="item.id"
            class="similarity-item"
          >
            <div class="item-header">
              <h5 class="item-title">{{ item.title }}</h5>
              <span class="item-source">来源: {{ item.source }}</span>
            </div>
            <p class="item-content">{{ truncateContent(item.content) }}</p>
            <div class="item-actions">
              <el-button size="small" type="text" @click="handleViewItem(item.id)">
                查看详情
              </el-button>
              <el-button size="small" type="text" @click="handleSelectPrimary(group.id, item.id)">
                设为主要
              </el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="panel-summary" v-if="filteredGroups.length === 0">
      <el-empty description="没有找到相似内容" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

const props = withDefaults(defineProps<SimilarityPanelProps>(), {
  threshold: 70
});

const emit = defineEmits<{
  merge: [groupId: string];
  keepAll: [groupId: string];
  viewItem: [itemId: string];
  selectPrimary: [groupId: string, itemId: string];
  thresholdChange: [threshold: number];
}>();

const currentThreshold = ref(props.threshold);

const filteredGroups = computed(() => {
  return props.groups.filter(group => group.similarity >= currentThreshold.value);
});

const truncateContent = (content: string, maxLength = 150) => {
  return content.length > maxLength 
    ? content.slice(0, maxLength) + '...' 
    : content;
};

const handleThresholdChange = (value: number) => {
  emit('thresholdChange', value);
};

const handleMerge = (groupId: string) => {
  emit('merge', groupId);
};

const handleKeepAll = (groupId: string) => {
  emit('keepAll', groupId);
};

const handleViewItem = (itemId: string) => {
  emit('viewItem', itemId);
};

const handleSelectPrimary = (groupId: string, itemId: string) => {
  emit('selectPrimary', groupId, itemId);
};
</script>

<style scoped lang="scss">
.similarity-panel {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .panel-header {
    @apply flex justify-between items-center mb-4 pb-4 border-b border-gray-200;
    
    .panel-title {
      @apply text-lg font-semibold text-gray-900;
    }
    
    .panel-controls {
      @apply flex items-center space-x-3;
      
      .threshold-label {
        @apply text-sm text-gray-600;
      }
    }
  }
  
  .similarity-groups {
    @apply space-y-4;
  }
  
  .similarity-group {
    @apply border border-gray-200 rounded-lg p-4;
    
    .group-header {
      @apply flex justify-between items-center mb-3;
      
      .group-info {
        @apply flex items-center space-x-4;
        
        .similarity-score {
          @apply font-medium text-blue-600;
        }
        
        .item-count {
          @apply text-sm text-gray-500;
        }
      }
      
      .group-actions {
        @apply flex space-x-2;
      }
    }
    
    .group-items {
      @apply space-y-3;
    }
    
    .similarity-item {
      @apply bg-gray-50 rounded-lg p-3;
      
      .item-header {
        @apply flex justify-between items-start mb-2;
        
        .item-title {
          @apply font-medium text-gray-900;
        }
        
        .item-source {
          @apply text-xs text-gray-500;
        }
      }
      
      .item-content {
        @apply text-sm text-gray-700 mb-2;
      }
      
      .item-actions {
        @apply flex space-x-2;
      }
    }
  }
  
  .panel-summary {
    @apply text-center py-8;
  }
}
</style>
```

### 平台连接器组件 (PlatformConnector)

```typescript
// components/business/PlatformConnector.vue
interface Platform {
  id: string;
  name: string;
  type: 'social' | 'cms' | 'ecommerce' | 'blog';
  status: 'connected' | 'disconnected' | 'error';
  lastSync?: Date;
  config: Record<string, any>;
  capabilities: string[];
}

interface PlatformConnectorProps {
  platforms: Platform[];
}

<template>
  <div class="platform-connector">
    <div class="connector-header">
      <h3 class="connector-title">平台连接管理</h3>
      <el-button type="primary" @click="handleAddPlatform">
        添加平台
      </el-button>
    </div>
    
    <div class="platform-grid">
      <div 
        v-for="platform in platforms"
        :key="platform.id"
        class="platform-card"
        :class="`status-${platform.status}`"
      >
        <div class="platform-header">
          <div class="platform-info">
            <h4 class="platform-name">{{ platform.name }}</h4>
            <el-tag :type="getPlatformTypeColor(platform.type)" size="small">
              {{ getPlatformTypeText(platform.type) }}
            </el-tag>
          </div>
          <div class="platform-status">
            <el-icon :class="getStatusIconClass(platform.status)">
              <component :is="getStatusIcon(platform.status)" />
            </el-icon>
          </div>
        </div>
        
        <div class="platform-details">
          <div class="detail-item" v-if="platform.lastSync">
            <span class="detail-label">最后同步:</span>
            <span class="detail-value">{{ formatDateTime(platform.lastSync) }}</span>
          </div>
          
          <div class="platform-capabilities">
            <span class="capabilities-label">支持功能:</span>
            <div class="capabilities-list">
              <el-tag 
                v-for="capability in platform.capabilities"
                :key="capability"
                size="small"
                effect="plain"
              >
                {{ capability }}
              </el-tag>
            </div>
          </div>
        </div>
        
        <div class="platform-actions">
          <el-button 
            size="small" 
            @click="handleTest(platform.id)"
          >
            测试连接
          </el-button>
          <el-button 
            size="small" 
            @click="handleConfigure(platform.id)"
          >
            配置
          </el-button>
          <el-button 
            v-if="platform.status === 'connected'"
            size="small" 
            type="success"
            @click="handleSync(platform.id)"
          >
            同步
          </el-button>
          <el-button 
            size="small" 
            type="danger"
            @click="handleDisconnect(platform.id)"
          >
            断开
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { CircleCheck, Warning, CircleClose } from '@element-plus/icons-vue';

const props = defineProps<PlatformConnectorProps>();

const emit = defineEmits<{
  addPlatform: [];
  test: [platformId: string];
  configure: [platformId: string];
  sync: [platformId: string];
  disconnect: [platformId: string];
}>();

const getPlatformTypeColor = (type: string) => {
  const colorMap = {
    social: 'primary',
    cms: 'success',
    ecommerce: 'warning',
    blog: 'info'
  };
  return colorMap[type] || 'info';
};

const getPlatformTypeText = (type: string) => {
  const textMap = {
    social: '社交媒体',
    cms: '内容管理',
    ecommerce: '电商平台',
    blog: '博客平台'
  };
  return textMap[type] || '未知类型';
};

const getStatusIcon = (status: string) => {
  const iconMap = {
    connected: CircleCheck,
    disconnected: Warning,
    error: CircleClose
  };
  return iconMap[status] || Warning;
};

const getStatusIconClass = (status: string) => {
  const classMap = {
    connected: 'text-green-500',
    disconnected: 'text-gray-400',
    error: 'text-red-500'
  };
  return classMap[status] || 'text-gray-400';
};

const formatDateTime = (date: Date) => {
  return new Date(date).toLocaleString();
};

const handleAddPlatform = () => emit('addPlatform');
const handleTest = (platformId: string) => emit('test', platformId);
const handleConfigure = (platformId: string) => emit('configure', platformId);
const handleSync = (platformId: string) => emit('sync', platformId);
const handleDisconnect = (platformId: string) => emit('disconnect', platformId);
</script>

<style scoped lang="scss">
.platform-connector {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .connector-header {
    @apply flex justify-between items-center mb-4;
    
    .connector-title {
      @apply text-lg font-semibold text-gray-900;
    }
  }
  
  .platform-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4;
  }
  
  .platform-card {
    @apply border rounded-lg p-4;
    
    &.status-connected {
      @apply border-green-200 bg-green-50;
    }
    
    &.status-disconnected {
      @apply border-gray-200 bg-gray-50;
    }
    
    &.status-error {
      @apply border-red-200 bg-red-50;
    }
    
    .platform-header {
      @apply flex justify-between items-start mb-3;
      
      .platform-info {
        @apply flex flex-col space-y-1;
        
        .platform-name {
          @apply font-medium text-gray-900;
        }
      }
      
      .platform-status {
        @apply text-lg;
      }
    }
    
    .platform-details {
      @apply mb-3 space-y-2;
      
      .detail-item {
        @apply flex justify-between text-sm;
        
        .detail-label {
          @apply text-gray-600;
        }
        
        .detail-value {
          @apply font-medium text-gray-900;
        }
      }
      
      .platform-capabilities {
        .capabilities-label {
          @apply text-sm text-gray-600 block mb-1;
        }
        
        .capabilities-list {
          @apply flex flex-wrap gap-1;
        }
      }
    }
    
    .platform-actions {
      @apply flex flex-wrap gap-2;
    }
  }
}
</style>
```

---

## 3. 图表组件 (Chart Components)

### 统计图表组件 (MetricsChart)

```typescript
// components/charts/MetricsChart.vue
interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string;
    borderColor?: string;
    borderWidth?: number;
  }[];
}

interface MetricsChartProps {
  title: string;
  type: 'line' | 'bar' | 'pie' | 'doughnut';
  data: ChartData;
  height?: number;
  showTimeRange?: boolean;
}

<template>
  <div class="metrics-chart">
    <div class="chart-header">
      <h3 class="chart-title">{{ title }}</h3>
      <div class="chart-controls" v-if="showTimeRange">
        <el-select v-model="timeRange" @change="handleTimeRangeChange">
          <el-option label="最近7天" value="7d" />
          <el-option label="最近30天" value="30d" />
          <el-option label="最近90天" value="90d" />
          <el-option label="最近1年" value="1y" />
        </el-select>
      </div>
    </div>
    
    <div class="chart-container" :style="{ height: `${height}px` }">
      <canvas ref="chartCanvas"></canvas>
    </div>
    
    <div class="chart-legend">
      <div 
        v-for="dataset in data.datasets"
        :key="dataset.label"
        class="legend-item"
      >
        <span 
          class="legend-color"
          :style="{ backgroundColor: dataset.backgroundColor }"
        ></span>
        <span class="legend-label">{{ dataset.label }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted } from 'vue';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const props = withDefaults(defineProps<MetricsChartProps>(), {
  height: 300,
  showTimeRange: true
});

const emit = defineEmits<{
  timeRangeChange: [range: string];
}>();

const chartCanvas = ref<HTMLCanvasElement>();
const timeRange = ref('30d');
let chartInstance: Chart | null = null;

const createChart = () => {
  if (!chartCanvas.value) return;
  
  if (chartInstance) {
    chartInstance.destroy();
  }
  
  chartInstance = new Chart(chartCanvas.value, {
    type: props.type,
    data: props.data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: props.type === 'pie' || props.type === 'doughnut' ? {} : {
        y: {
          beginAtZero: true
        }
      }
    }
  });
};

const handleTimeRangeChange = (range: string) => {
  emit('timeRangeChange', range);
};

watch(() => props.data, () => {
  if (chartInstance) {
    chartInstance.data = props.data;
    chartInstance.update();
  }
}, { deep: true });

onMounted(() => {
  createChart();
});

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.destroy();
  }
});
</script>

<style scoped lang="scss">
.metrics-chart {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .chart-header {
    @apply flex justify-between items-center mb-4;
    
    .chart-title {
      @apply text-lg font-semibold text-gray-900;
    }
  }
  
  .chart-container {
    @apply relative;
  }
  
  .chart-legend {
    @apply flex flex-wrap justify-center gap-4 mt-4;
    
    .legend-item {
      @apply flex items-center space-x-2;
      
      .legend-color {
        @apply w-3 h-3 rounded-full;
      }
      
      .legend-label {
        @apply text-sm text-gray-700;
      }
    }
  }
}
</style>
```

### 实时监控图表 (RealtimeChart)

```typescript
// components/charts/RealtimeChart.vue
interface RealtimeData {
  timestamp: Date;
  value: number;
  metric: string;
}

interface MetricCard {
  name: string;
  label: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
}

interface RealtimeChartProps {
  title: string;
  data: RealtimeData[];
  metrics: MetricCard[];
  maxDataPoints?: number;
}

<template>
  <div class="realtime-chart">
    <div class="chart-header">
      <h3>{{ title }}</h3>
      <div class="chart-controls">
        <el-button 
          :type="isPlaying ? 'danger' : 'primary'"
          @click="togglePlayback"
        >
          {{ isPlaying ? '暂停' : '开始' }}
        </el-button>
        <el-button @click="clearData">清空数据</el-button>
      </div>
    </div>
    
    <div class="chart-metrics">
      <div class="metric-card" v-for="metric in metrics" :key="metric.name">
        <div class="metric-value">{{ metric.value }}</div>
        <div class="metric-label">{{ metric.label }}</div>
        <div class="metric-trend" :class="metric.trend">
          <el-icon><component :is="getTrendIcon(metric.trend)" /></el-icon>
          {{ metric.change }}%
        </div>
      </div>
    </div>
    
    <div class="chart-container">
      <canvas ref="realtimeCanvas"></canvas>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue';
import { Chart, registerables } from 'chart.js';
import { ArrowUp, ArrowDown, Minus } from '@element-plus/icons-vue';

Chart.register(...registerables);

const props = withDefaults(defineProps<RealtimeChartProps>(), {
  maxDataPoints: 50
});

const emit = defineEmits<{
  play: [];
  pause: [];
  clear: [];
}>();

const realtimeCanvas = ref<HTMLCanvasElement>();
const isPlaying = ref(false);
let chartInstance: Chart | null = null;
let updateInterval: number | null = null;

const getTrendIcon = (trend: string) => {
  const iconMap = {
    up: ArrowUp,
    down: ArrowDown,
    stable: Minus
  };
  return iconMap[trend] || Minus;
};

const createChart = () => {
  if (!realtimeCanvas.value) return;
  
  chartInstance = new Chart(realtimeCanvas.value, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: '实时数据',
        data: [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 0
      },
      scales: {
        x: {
          type: 'time',
          time: {
            displayFormats: {
              second: 'HH:mm:ss'
            }
          }
        },
        y: {
          beginAtZero: true
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
};

const updateChart = () => {
  if (!chartInstance || props.data.length === 0) return;
  
  const latestData = props.data.slice(-props.maxDataPoints);
  
  chartInstance.data.labels = latestData.map(d => d.timestamp);
  chartInstance.data.datasets[0].data = latestData.map(d => d.value);
  chartInstance.update('none');
};

const togglePlayback = () => {
  isPlaying.value = !isPlaying.value;
  
  if (isPlaying.value) {
    emit('play');
    updateInterval = setInterval(updateChart, 1000);
  } else {
    emit('pause');
    if (updateInterval) {
      clearInterval(updateInterval);
      updateInterval = null;
    }
  }
};

const clearData = () => {
  emit('clear');
  if (chartInstance) {
    chartInstance.data.labels = [];
    chartInstance.data.datasets[0].data = [];
    chartInstance.update();
  }
};

onMounted(() => {
  createChart();
});

onUnmounted(() => {
  if (updateInterval) {
    clearInterval(updateInterval);
  }
  if (chartInstance) {
    chartInstance.destroy();
  }
});
</script>

<style scoped lang="scss">
.realtime-chart {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .chart-header {
    @apply flex justify-between items-center mb-4;
    
    h3 {
      @apply text-lg font-semibold text-gray-900;
    }
    
    .chart-controls {
      @apply flex space-x-2;
    }
  }
  
  .chart-metrics {
    @apply grid grid-cols-1 md:grid-cols-3 gap-4 mb-4;
    
    .metric-card {
      @apply bg-gray-50 rounded-lg p-3 text-center;
      
      .metric-value {
        @apply text-2xl font-bold text-gray-900;
      }
      
      .metric-label {
        @apply text-sm text-gray-600 mt-1;
      }
      
      .metric-trend {
        @apply flex items-center justify-center space-x-1 mt-2 text-sm;
        
        &.up {
          @apply text-green-600;
        }
        
        &.down {
          @apply text-red-600;
        }
        
        &.stable {
          @apply text-gray-600;
        }
      }
    }
  }
  
  .chart-container {
    @apply relative h-64;
  }
}
</style>
```

### 热力图组件 (HeatmapChart)

```typescript
// components/charts/HeatmapChart.vue
interface HeatmapData {
  x: string;
  y: string;
  value: number;
  color?: string;
}

interface HeatmapChartProps {
  title: string;
  data: HeatmapData[];
  colorScheme?: 'blue' | 'green' | 'red' | 'purple';
}

<template>
  <div class="heatmap-chart">
    <div class="chart-header">
      <h3>{{ title }}</h3>
      <div class="chart-filters">
        <el-date-picker
          v-model="dateRange"
          type="daterange"
          range-separator="至"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          @change="handleDateRangeChange"
        />
      </div>
    </div>
    
    <div class="heatmap-container">
      <div class="heatmap-grid">
        <div 
          v-for="(row, rowIndex) in heatmapGrid"
          :key="rowIndex"
          class="heatmap-row"
        >
          <div 
            v-for="(cell, colIndex) in row"
            :key="colIndex"
            class="heatmap-cell"
            :style="{ backgroundColor: getCellColor(cell.value) }"
            :title="`${cell.x} - ${cell.y}: ${cell.value}`"
            @click="handleCellClick(cell)"
          >
            <span class="cell-value">{{ cell.value }}</span>
          </div>
        </div>
      </div>
      
      <div class="heatmap-legend">
        <div class="legend-scale">
          <span class="scale-min">{{ minValue }}</span>
          <div class="scale-gradient" :class="`gradient-${colorScheme}`"></div>
          <span class="scale-max">{{ maxValue }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

const props = withDefaults(defineProps<HeatmapChartProps>(), {
  colorScheme: 'blue'
});

const emit = defineEmits<{
  cellClick: [cell: HeatmapData];
  dateRangeChange: [range: [Date, Date] | null];
}>();

const dateRange = ref<[Date, Date] | null>(null);

const minValue = computed(() => {
  return Math.min(...props.data.map(d => d.value));
});

const maxValue = computed(() => {
  return Math.max(...props.data.map(d => d.value));
});

const heatmapGrid = computed(() => {
  const xValues = [...new Set(props.data.map(d => d.x))];
  const yValues = [...new Set(props.data.map(d => d.y))];
  
  return yValues.map(y => {
    return xValues.map(x => {
      const cell = props.data.find(d => d.x === x && d.y === y);
      return cell || { x, y, value: 0 };
    });
  });
});

const getCellColor = (value: number) => {
  const intensity = (value - minValue.value) / (maxValue.value - minValue.value);
  
  const colorSchemes = {
    blue: `rgba(59, 130, 246, ${intensity})`,
    green: `rgba(34, 197, 94, ${intensity})`,
    red: `rgba(239, 68, 68, ${intensity})`,
    purple: `rgba(147, 51, 234, ${intensity})`
  };
  
  return colorSchemes[props.colorScheme] || colorSchemes.blue;
};

const handleCellClick = (cell: HeatmapData) => {
  emit('cellClick', cell);
};

const handleDateRangeChange = (range: [Date, Date] | null) => {
  emit('dateRangeChange', range);
};
</script>

<style scoped lang="scss">
.heatmap-chart {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  .chart-header {
    @apply flex justify-between items-center mb-4;
    
    h3 {
      @apply text-lg font-semibold text-gray-900;
    }
  }
  
  .heatmap-container {
    @apply space-y-4;
  }
  
  .heatmap-grid {
    @apply space-y-1;
    
    .heatmap-row {
      @apply flex space-x-1;
    }
    
    .heatmap-cell {
      @apply w-8 h-8 flex items-center justify-center text-xs font-medium cursor-pointer rounded transition-all hover:scale-110;
      
      .cell-value {
        @apply text-white drop-shadow-sm;
      }
    }
  }
  
  .heatmap-legend {
    @apply flex justify-center;
    
    .legend-scale {
      @apply flex items-center space-x-2;
      
      .scale-min,
      .scale-max {
        @apply text-sm text-gray-600;
      }
      
      .scale-gradient {
        @apply w-32 h-4 rounded;
        
        &.gradient-blue {
          background: linear-gradient(to right, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 1));
        }
        
        &.gradient-green {
          background: linear-gradient(to right, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 1));
        }
        
        &.gradient-red {
          background: linear-gradient(to right, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 1));
        }
        
        &.gradient-purple {
          background: linear-gradient(to right, rgba(147, 51, 234, 0.1), rgba(147, 51, 234, 1));
        }
      }
    }
  }
}
</style>