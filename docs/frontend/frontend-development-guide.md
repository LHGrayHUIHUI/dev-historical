# 前端开发规范指南

## 目录

1. [代码规范](#代码规范)
2. [开发工作流](#开发工作流)
3. [测试策略](#测试策略)
4. [性能优化](#性能优化)
5. [错误处理规范](#错误处理规范)
6. [部署配置](#部署配置)
7. [开发工具配置](#开发工具配置)
8. [最佳实践](#最佳实践)

---

## 代码规范

### TypeScript编码规范

```typescript
// ✅ 推荐的TypeScript代码风格

// 1. 接口定义
interface UserInfo {
  id: string;
  username: string;
  email: string;
  createdAt: Date;
  updatedAt: Date;
}

// 2. 类型定义
type UserStatus = 'active' | 'inactive' | 'pending';
type ApiResponse<T> = {
  code: number;
  message: string;
  data: T;
};

// 3. 函数定义
const fetchUserInfo = async (userId: string): Promise<UserInfo> => {
  try {
    const response = await httpClient.get<ApiResponse<UserInfo>>(`/users/${userId}`);
    return response.data;
  } catch (error) {
    console.error('获取用户信息失败:', error);
    throw error;
  }
};

// 4. 组件Props定义
interface ButtonProps {
  type?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  onClick?: (event: MouseEvent) => void;
}

// 5. 枚举定义
enum ContentStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}
```

### Vue组件编码规范

```vue
<!-- ✅ 推荐的Vue组件结构 -->
<template>
  <div class="user-card">
    <!-- 使用语义化的HTML标签 -->
    <header class="user-card__header">
      <h3 class="user-card__title">{{ user.username }}</h3>
      <span class="user-card__status" :class="statusClass">
        {{ statusText }}
      </span>
    </header>
    
    <main class="user-card__content">
      <p class="user-card__email">{{ user.email }}</p>
      <time class="user-card__date" :datetime="user.createdAt.toISOString()">
        注册时间：{{ formatDate(user.createdAt) }}
      </time>
    </main>
    
    <footer class="user-card__actions">
      <el-button 
        type="primary" 
        size="small"
        :loading="loading"
        @click="handleEdit"
      >
        编辑
      </el-button>
      <el-button 
        type="danger" 
        size="small"
        :disabled="user.status === 'active'"
        @click="handleDelete"
      >
        删除
      </el-button>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { ElButton, ElMessage } from 'element-plus';
import { formatDate } from '@/utils/date';

// Props定义
interface Props {
  user: UserInfo;
  readonly?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  readonly: false,
});

// Emits定义
interface Emits {
  edit: [user: UserInfo];
  delete: [userId: string];
}

const emit = defineEmits<Emits>();

// 响应式数据
const loading = ref(false);

// 计算属性
const statusClass = computed(() => ({
  'user-card__status--active': props.user.status === 'active',
  'user-card__status--inactive': props.user.status === 'inactive',
  'user-card__status--pending': props.user.status === 'pending',
}));

const statusText = computed(() => {
  const statusMap = {
    active: '活跃',
    inactive: '非活跃',
    pending: '待审核',
  };
  return statusMap[props.user.status] || '未知';
});

// 方法定义
const handleEdit = () => {
  if (props.readonly) {
    ElMessage.warning('只读模式下无法编辑');
    return;
  }
  emit('edit', props.user);
};

const handleDelete = async () => {
  if (props.readonly) {
    ElMessage.warning('只读模式下无法删除');
    return;
  }
  
  try {
    loading.value = true;
    // 这里应该调用删除API
    await new Promise(resolve => setTimeout(resolve, 1000)); // 模拟API调用
    emit('delete', props.user.id);
    ElMessage.success('删除成功');
  } catch (error) {
    ElMessage.error('删除失败');
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped lang="scss">
.user-card {
  @apply bg-white rounded-lg shadow-sm border border-gray-200 p-4;
  
  &__header {
    @apply flex justify-between items-center mb-3;
  }
  
  &__title {
    @apply text-lg font-semibold text-gray-900 m-0;
  }
  
  &__status {
    @apply px-2 py-1 rounded-full text-xs font-medium;
    
    &--active {
      @apply bg-green-100 text-green-800;
    }
    
    &--inactive {
      @apply bg-gray-100 text-gray-800;
    }
    
    &--pending {
      @apply bg-yellow-100 text-yellow-800;
    }
  }
  
  &__content {
    @apply mb-4;
  }
  
  &__email {
    @apply text-gray-600 mb-2;
  }
  
  &__date {
    @apply text-sm text-gray-500;
  }
  
  &__actions {
    @apply flex space-x-2;
  }
}
</style>
```

### CSS/SCSS编码规范

```scss
// ✅ 推荐的SCSS代码风格

// 1. 变量定义
$primary-color: #3b82f6;
$secondary-color: #6b7280;
$success-color: #10b981;
$warning-color: #f59e0b;
$error-color: #ef4444;

$font-size-xs: 0.75rem;
$font-size-sm: 0.875rem;
$font-size-base: 1rem;
$font-size-lg: 1.125rem;
$font-size-xl: 1.25rem;

$spacing-xs: 0.25rem;
$spacing-sm: 0.5rem;
$spacing-md: 1rem;
$spacing-lg: 1.5rem;
$spacing-xl: 2rem;

// 2. Mixin定义
@mixin button-base {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: $spacing-sm $spacing-md;
  border: 1px solid transparent;
  border-radius: 0.375rem;
  font-size: $font-size-sm;
  font-weight: 500;
  line-height: 1.5;
  text-decoration: none;
  cursor: pointer;
  transition: all 0.2s ease-in-out;
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
}

@mixin button-variant($bg-color, $text-color, $border-color: $bg-color) {
  background-color: $bg-color;
  color: $text-color;
  border-color: $border-color;
  
  &:hover:not(:disabled) {
    background-color: darken($bg-color, 10%);
    border-color: darken($border-color, 10%);
  }
  
  &:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba($bg-color, 0.25);
  }
}

// 3. 组件样式
.btn {
  @include button-base;
  
  &--primary {
    @include button-variant($primary-color, white);
  }
  
  &--secondary {
    @include button-variant(white, $secondary-color, $secondary-color);
  }
  
  &--success {
    @include button-variant($success-color, white);
  }
  
  &--warning {
    @include button-variant($warning-color, white);
  }
  
  &--danger {
    @include button-variant($error-color, white);
  }
  
  &--small {
    padding: $spacing-xs $spacing-sm;
    font-size: $font-size-xs;
  }
  
  &--large {
    padding: $spacing-md $spacing-lg;
    font-size: $font-size-lg;
  }
}

// 4. 响应式设计
.container {
  width: 100%;
  margin: 0 auto;
  padding: 0 $spacing-md;
  
  @media (min-width: 640px) {
    max-width: 640px;
  }
  
  @media (min-width: 768px) {
    max-width: 768px;
    padding: 0 $spacing-lg;
  }
  
  @media (min-width: 1024px) {
    max-width: 1024px;
  }
  
  @media (min-width: 1280px) {
    max-width: 1280px;
  }
}
```

---

## 开发工作流

### Git工作流规范

```bash
# 1. 分支命名规范
feature/user-authentication    # 新功能开发
bugfix/login-validation-error  # Bug修复
hotfix/security-patch         # 紧急修复
refactor/api-client-structure # 代码重构
docs/api-documentation        # 文档更新

# 2. 提交信息规范
git commit -m "feat: 添加用户认证功能"
git commit -m "fix: 修复登录验证错误"
git commit -m "docs: 更新API文档"
git commit -m "style: 调整代码格式"
git commit -m "refactor: 重构HTTP客户端"
git commit -m "test: 添加用户服务单元测试"
git commit -m "chore: 更新依赖包版本"

# 3. 开发流程
# 创建功能分支
git checkout -b feature/content-management

# 开发过程中定期提交
git add .
git commit -m "feat: 添加内容列表组件"

# 推送到远程分支
git push origin feature/content-management

# 创建Pull Request
# 代码审查通过后合并到主分支
```

### 代码审查清单

```markdown
## 代码审查清单

### 功能性
- [ ] 功能是否按需求正确实现
- [ ] 边界条件是否正确处理
- [ ] 错误处理是否完善
- [ ] 用户体验是否良好

### 代码质量
- [ ] 代码是否遵循项目规范
- [ ] 变量和函数命名是否清晰
- [ ] 代码是否有适当的注释
- [ ] 是否有重复代码需要提取

### 性能
- [ ] 是否有性能问题
- [ ] 大数据量处理是否优化
- [ ] 是否有内存泄漏风险
- [ ] 网络请求是否合理

### 安全性
- [ ] 用户输入是否正确验证
- [ ] 敏感信息是否正确处理
- [ ] 权限控制是否到位
- [ ] XSS和CSRF防护是否完善

### 测试
- [ ] 是否有对应的单元测试
- [ ] 测试覆盖率是否达标
- [ ] 集成测试是否通过
- [ ] 手动测试是否完成
```

---

## 测试策略

### 单元测试

```typescript
// tests/utils/date.test.ts
import { describe, it, expect } from 'vitest';
import { formatDate, isValidDate, getRelativeTime } from '@/utils/date';

describe('日期工具函数', () => {
  describe('formatDate', () => {
    it('应该正确格式化日期', () => {
      const date = new Date('2024-01-15T10:30:00Z');
      expect(formatDate(date)).toBe('2024-01-15');
      expect(formatDate(date, 'YYYY-MM-DD HH:mm')).toBe('2024-01-15 10:30');
    });
    
    it('应该处理无效日期', () => {
      expect(formatDate(null)).toBe('');
      expect(formatDate(undefined)).toBe('');
      expect(formatDate(new Date('invalid'))).toBe('');
    });
  });
  
  describe('isValidDate', () => {
    it('应该正确验证日期', () => {
      expect(isValidDate(new Date())).toBe(true);
      expect(isValidDate(new Date('2024-01-15'))).toBe(true);
      expect(isValidDate(new Date('invalid'))).toBe(false);
      expect(isValidDate(null)).toBe(false);
    });
  });
  
  describe('getRelativeTime', () => {
    it('应该返回相对时间', () => {
      const now = new Date();
      const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
      const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      
      expect(getRelativeTime(oneHourAgo)).toBe('1小时前');
      expect(getRelativeTime(oneDayAgo)).toBe('1天前');
    });
  });
});
```

### 组件测试

```typescript
// tests/components/UserCard.test.ts
import { describe, it, expect, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { ElButton } from 'element-plus';
import UserCard from '@/components/UserCard.vue';

const mockUser = {
  id: '1',
  username: 'testuser',
  email: 'test@example.com',
  status: 'active' as const,
  createdAt: new Date('2024-01-01'),
  updatedAt: new Date('2024-01-01'),
};

describe('UserCard组件', () => {
  it('应该正确渲染用户信息', () => {
    const wrapper = mount(UserCard, {
      props: { user: mockUser },
      global: {
        components: { ElButton },
      },
    });
    
    expect(wrapper.find('.user-card__title').text()).toBe('testuser');
    expect(wrapper.find('.user-card__email').text()).toBe('test@example.com');
    expect(wrapper.find('.user-card__status').text()).toBe('活跃');
  });
  
  it('应该在只读模式下禁用操作按钮', () => {
    const wrapper = mount(UserCard, {
      props: { user: mockUser, readonly: true },
      global: {
        components: { ElButton },
      },
    });
    
    const editButton = wrapper.find('[data-testid="edit-button"]');
    const deleteButton = wrapper.find('[data-testid="delete-button"]');
    
    expect(editButton.attributes('disabled')).toBeDefined();
    expect(deleteButton.attributes('disabled')).toBeDefined();
  });
  
  it('应该正确触发编辑事件', async () => {
    const wrapper = mount(UserCard, {
      props: { user: mockUser },
      global: {
        components: { ElButton },
      },
    });
    
    await wrapper.find('[data-testid="edit-button"]').trigger('click');
    
    expect(wrapper.emitted('edit')).toBeTruthy();
    expect(wrapper.emitted('edit')?.[0]).toEqual([mockUser]);
  });
  
  it('应该正确处理删除操作', async () => {
    const wrapper = mount(UserCard, {
      props: { user: mockUser },
      global: {
        components: { ElButton },
      },
    });
    
    await wrapper.find('[data-testid="delete-button"]').trigger('click');
    
    // 等待异步操作完成
    await wrapper.vm.$nextTick();
    
    expect(wrapper.emitted('delete')).toBeTruthy();
    expect(wrapper.emitted('delete')?.[0]).toEqual([mockUser.id]);
  });
});
```

### 集成测试

```typescript
// tests/integration/content-management.test.ts
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import ContentList from '@/views/Content/List.vue';
import { useContentStore } from '@/stores/content';
import * as contentApi from '@/api/content';

// Mock API
vi.mock('@/api/content', () => ({
  contentApi: {
    getContentList: vi.fn(),
    deleteContent: vi.fn(),
  },
}));

describe('内容管理集成测试', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });
  
  it('应该正确加载和显示内容列表', async () => {
    const mockContentList = {
      data: [
        {
          id: '1',
          title: '测试内容1',
          status: 'completed',
          createdAt: '2024-01-01T00:00:00Z',
        },
        {
          id: '2',
          title: '测试内容2',
          status: 'pending',
          createdAt: '2024-01-02T00:00:00Z',
        },
      ],
      total: 2,
      page: 1,
      pageSize: 20,
    };
    
    vi.mocked(contentApi.contentApi.getContentList).mockResolvedValue(mockContentList);
    
    const wrapper = mount(ContentList, {
      global: {
        plugins: [createPinia()],
      },
    });
    
    const store = useContentStore();
    await store.fetchContentList();
    
    await wrapper.vm.$nextTick();
    
    expect(contentApi.contentApi.getContentList).toHaveBeenCalledWith({
      page: 1,
      pageSize: 20,
    });
    
    expect(store.contentList).toEqual(mockContentList.data);
    expect(store.pagination.total).toBe(2);
  });
  
  it('应该正确处理内容删除', async () => {
    vi.mocked(contentApi.contentApi.deleteContent).mockResolvedValue();
    
    const store = useContentStore();
    store.contentList = [
      {
        id: '1',
        title: '测试内容',
        status: 'completed',
        createdAt: '2024-01-01T00:00:00Z',
      } as any,
    ];
    
    await store.deleteContent('1');
    
    expect(contentApi.contentApi.deleteContent).toHaveBeenCalledWith('1');
    expect(store.contentList).toHaveLength(0);
  });
});
```

### E2E测试

```typescript
// tests/e2e/content-workflow.spec.ts
import { test, expect } from '@playwright/test';

test.describe('内容管理工作流', () => {
  test.beforeEach(async ({ page }) => {
    // 登录
    await page.goto('/login');
    await page.fill('[data-testid="username"]', 'admin');
    await page.fill('[data-testid="password"]', 'password');
    await page.click('[data-testid="login-button"]');
    await expect(page).toHaveURL('/dashboard');
  });
  
  test('应该能够创建、编辑和删除内容', async ({ page }) => {
    // 导航到内容管理页面
    await page.click('[data-testid="nav-content"]');
    await expect(page).toHaveURL('/content/list');
    
    // 创建新内容
    await page.click('[data-testid="create-content-button"]');
    await expect(page).toHaveURL('/content/create');
    
    await page.fill('[data-testid="content-title"]', '测试内容标题');
    await page.fill('[data-testid="content-text"]', '这是一段测试内容文本');
    await page.click('[data-testid="submit-button"]');
    
    // 验证内容创建成功
    await expect(page).toHaveURL('/content/list');
    await expect(page.locator('[data-testid="content-item"]')).toContainText('测试内容标题');
    
    // 编辑内容
    await page.click('[data-testid="edit-content-button"]');
    await page.fill('[data-testid="content-title"]', '修改后的标题');
    await page.click('[data-testid="submit-button"]');
    
    // 验证内容编辑成功
    await expect(page.locator('[data-testid="content-item"]')).toContainText('修改后的标题');
    
    // 删除内容
    await page.click('[data-testid="delete-content-button"]');
    await page.click('[data-testid="confirm-delete-button"]');
    
    // 验证内容删除成功
    await expect(page.locator('[data-testid="content-item"]')).not.toBeVisible();
  });
  
  test('应该能够进行AI优化', async ({ page }) => {
    // 假设已有内容存在
    await page.goto('/content/list');
    
    // 点击优化按钮
    await page.click('[data-testid="optimize-content-button"]');
    
    // 选择优化选项
    await page.selectOption('[data-testid="optimization-type"]', 'comprehensive');
    await page.click('[data-testid="start-optimization-button"]');
    
    // 等待优化完成
    await expect(page.locator('[data-testid="optimization-status"]')).toContainText('优化中');
    
    // 模拟等待优化完成（实际测试中可能需要更长时间）
    await page.waitForTimeout(5000);
    
    // 验证优化结果
    await expect(page.locator('[data-testid="optimization-status"]')).toContainText('优化完成');
    await expect(page.locator('[data-testid="optimized-text"]')).toBeVisible();
  });
});
```

---

## 性能优化

### 代码分割和懒加载

```typescript
// router/index.ts - 路由懒加载
const routes = [
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('@/views/Dashboard.vue'),
  },
  {
    path: '/content',
    name: 'Content',
    component: () => import('@/views/Content/index.vue'),
    children: [
      {
        path: 'list',
        component: () => import('@/views/Content/List.vue'),
      },
      {
        path: 'create',
        component: () => import('@/views/Content/Create.vue'),
      },
    ],
  },
];

// 组件懒加载
// components/LazyComponent.vue
<template>
  <Suspense>
    <template #default>
      <AsyncComponent v-if="shouldLoad" />
    </template>
    <template #fallback>
      <div class="loading-placeholder">
        <el-skeleton :rows="5" animated />
      </div>
    </template>
  </Suspense>
</template>

<script setup lang="ts">
import { defineAsyncComponent, ref, onMounted } from 'vue';

const shouldLoad = ref(false);

const AsyncComponent = defineAsyncComponent(() => 
  import('@/components/HeavyComponent.vue')
);

onMounted(() => {
  // 延迟加载重型组件
  setTimeout(() => {
    shouldLoad.value = true;
  }, 100);
});
</script>
```

### 虚拟滚动优化

```vue
<!-- components/VirtualList.vue -->
<template>
  <div 
    ref="containerRef"
    class="virtual-list"
    :style="{ height: `${containerHeight}px` }"
    @scroll="handleScroll"
  >
    <div 
      class="virtual-list__spacer"
      :style="{ height: `${totalHeight}px` }"
    >
      <div 
        class="virtual-list__content"
        :style="{ transform: `translateY(${offsetY}px)` }"
      >
        <div
          v-for="item in visibleItems"
          :key="item.id"
          class="virtual-list__item"
          :style="{ height: `${itemHeight}px` }"
        >
          <slot :item="item" :index="item.index" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';

interface Props {
  items: any[];
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
}

const props = withDefaults(defineProps<Props>(), {
  overscan: 5,
});

const containerRef = ref<HTMLElement>();
const scrollTop = ref(0);

const totalHeight = computed(() => props.items.length * props.itemHeight);

const startIndex = computed(() => {
  return Math.max(0, Math.floor(scrollTop.value / props.itemHeight) - props.overscan);
});

const endIndex = computed(() => {
  const visibleCount = Math.ceil(props.containerHeight / props.itemHeight);
  return Math.min(
    props.items.length - 1,
    startIndex.value + visibleCount + props.overscan * 2
  );
});

const visibleItems = computed(() => {
  return props.items
    .slice(startIndex.value, endIndex.value + 1)
    .map((item, index) => ({
      ...item,
      index: startIndex.value + index,
    }));
});

const offsetY = computed(() => startIndex.value * props.itemHeight);

const handleScroll = (event: Event) => {
  const target = event.target as HTMLElement;
  scrollTop.value = target.scrollTop;
};

let resizeObserver: ResizeObserver;

onMounted(() => {
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => {
      // 处理容器大小变化
    });
    resizeObserver.observe(containerRef.value);
  }
});

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect();
  }
});
</script>

<style scoped lang="scss">
.virtual-list {
  overflow-y: auto;
  position: relative;
  
  &__spacer {
    position: relative;
  }
  
  &__content {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
  }
  
  &__item {
    border-bottom: 1px solid #f0f0f0;
  }
}
</style>
```

### 图片优化

```vue
<!-- components/OptimizedImage.vue -->
<template>
  <div class="optimized-image">
    <img
      v-if="shouldLoad"
      :src="currentSrc"
      :alt="alt"
      :loading="loading"
      :class="imageClass"
      @load="handleLoad"
      @error="handleError"
    />
    <div v-else-if="showPlaceholder" class="image-placeholder">
      <el-skeleton-item variant="image" :style="placeholderStyle" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';

interface Props {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  lazy?: boolean;
  placeholder?: boolean;
  quality?: number;
  format?: 'webp' | 'jpeg' | 'png';
}

const props = withDefaults(defineProps<Props>(), {
  lazy: true,
  placeholder: true,
  quality: 80,
  format: 'webp',
});

const shouldLoad = ref(!props.lazy);
const showPlaceholder = ref(props.placeholder);
const isLoaded = ref(false);
const hasError = ref(false);

const currentSrc = computed(() => {
  if (hasError.value) {
    return '/images/placeholder-error.svg';
  }
  
  let url = props.src;
  
  // 添加图片优化参数
  const params = new URLSearchParams();
  if (props.width) params.append('w', props.width.toString());
  if (props.height) params.append('h', props.height.toString());
  params.append('q', props.quality.toString());
  params.append('f', props.format);
  
  if (params.toString()) {
    url += (url.includes('?') ? '&' : '?') + params.toString();
  }
  
  return url;
});

const imageClass = computed(() => ({
  'optimized-image__img': true,
  'optimized-image__img--loaded': isLoaded.value,
  'optimized-image__img--error': hasError.value,
}));

const placeholderStyle = computed(() => ({
  width: props.width ? `${props.width}px` : '100%',
  height: props.height ? `${props.height}px` : '200px',
}));

const handleLoad = () => {
  isLoaded.value = true;
  showPlaceholder.value = false;
};

const handleError = () => {
  hasError.value = true;
  showPlaceholder.value = false;
};

onMounted(() => {
  if (props.lazy) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            shouldLoad.value = true;
            observer.disconnect();
          }
        });
      },
      { threshold: 0.1 }
    );
    
    const element = document.querySelector('.optimized-image');
    if (element) {
      observer.observe(element);
    }
  }
});
</script>

<style scoped lang="scss">
.optimized-image {
  position: relative;
  overflow: hidden;
  
  &__img {
    width: 100%;
    height: auto;
    opacity: 0;
    transition: opacity 0.3s ease;
    
    &--loaded {
      opacity: 1;
    }
    
    &--error {
      opacity: 0.5;
    }
  }
  
  .image-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #f5f5f5;
  }
}
</style>
```

---

## 错误处理规范

### 全局错误处理

```typescript
// utils/error-handler.ts
import { ElMessage, ElNotification } from 'element-plus';
import { useUserStore } from '@/stores/user';
import router from '@/router';

export class AppError extends Error {
  public code: string;
  public statusCode: number;
  public isOperational: boolean;
  
  constructor(
    message: string,
    code: string = 'UNKNOWN_ERROR',
    statusCode: number = 500,
    isOperational: boolean = true
  ) {
    super(message);
    this.name = 'AppError';
    this.code = code;
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    
    Error.captureStackTrace(this, this.constructor);
  }
}

export class ValidationError extends AppError {
  public fields: Record<string, string[]>;
  
  constructor(message: string, fields: Record<string, string[]> = {}) {
    super(message, 'VALIDATION_ERROR', 422);
    this.fields = fields;
  }
}

export class NetworkError extends AppError {
  constructor(message: string = '网络连接失败') {
    super(message, 'NETWORK_ERROR', 0);
  }
}

export class AuthenticationError extends AppError {
  constructor(message: string = '认证失败') {
    super(message, 'AUTHENTICATION_ERROR', 401);
  }
}

export class AuthorizationError extends AppError {
  constructor(message: string = '权限不足') {
    super(message, 'AUTHORIZATION_ERROR', 403);
  }
}

// 全局错误处理器
export class ErrorHandler {
  private static instance: ErrorHandler;
  
  public static getInstance(): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler();
    }
    return ErrorHandler.instance;
  }
  
  public handle(error: Error | AppError, context?: string): void {
    console.error(`[${context || 'Global'}] Error:`, error);
    
    if (error instanceof AppError) {
      this.handleAppError(error);
    } else {
      this.handleUnknownError(error);
    }
    
    // 发送错误报告到监控服务
    this.reportError(error, context);
  }
  
  private handleAppError(error: AppError): void {
    switch (error.code) {
      case 'AUTHENTICATION_ERROR':
        this.handleAuthenticationError(error);
        break;
      case 'AUTHORIZATION_ERROR':
        this.handleAuthorizationError(error);
        break;
      case 'VALIDATION_ERROR':
        this.handleValidationError(error as ValidationError);
        break;
      case 'NETWORK_ERROR':
        this.handleNetworkError(error);
        break;
      default:
        this.showErrorMessage(error.message);
    }
  }
  
  private handleAuthenticationError(error: AppError): void {
    ElMessage.error(error.message);
    const userStore = useUserStore();
    userStore.logout();
    router.push('/login');
  }
  
  private handleAuthorizationError(error: AppError): void {
    ElMessage.error(error.message);
    router.push('/403');
  }
  
  private handleValidationError(error: ValidationError): void {
    const firstError = Object.values(error.fields)[0]?.[0];
    ElMessage.error(firstError || error.message);
  }
  
  private handleNetworkError(error: AppError): void {
    ElNotification({
      title: '网络错误',
      message: error.message,
      type: 'error',
      duration: 5000,
    });
  }
  
  private handleUnknownError(error: Error): void {
    console.error('未知错误:', error);
    ElMessage.error('系统发生未知错误，请稍后重试');
  }
  
  private showErrorMessage(message: string): void {
    ElMessage.error(message);
  }
  
  private reportError(error: Error, context?: string): void {
    // 发送错误报告到监控服务（如Sentry）
    if (import.meta.env.PROD) {
      // Sentry.captureException(error, { tags: { context } });
    }
  }
}

// Vue错误处理插件
export function setupErrorHandler(app: any) {
  const errorHandler = ErrorHandler.getInstance();
  
  // 捕获Vue组件错误
  app.config.errorHandler = (error: Error, instance: any, info: string) => {
    errorHandler.handle(error, `Vue Component: ${info}`);
  };
  
  // 捕获未处理的Promise拒绝
  window.addEventListener('unhandledrejection', (event) => {
    errorHandler.handle(new Error(event.reason), 'Unhandled Promise Rejection');
    event.preventDefault();
  });
  
  // 捕获全局JavaScript错误
  window.addEventListener('error', (event) => {
    errorHandler.handle(event.error, 'Global JavaScript Error');
  });
}
```

### 错误边界组件

```vue
<!-- components/ErrorBoundary.vue -->
<template>
  <div class="error-boundary">
    <div v-if="hasError" class="error-boundary__content">
      <div class="error-boundary__icon">
        <el-icon size="48" color="#f56565">
          <WarningFilled />
        </el-icon>
      </div>
      
      <h3 class="error-boundary__title">{{ errorTitle }}</h3>
      <p class="error-boundary__message">{{ errorMessage }}</p>
      
      <div class="error-boundary__actions">
        <el-button type="primary" @click="handleRetry">
          重试
        </el-button>
        <el-button @click="handleGoHome">
          返回首页
        </el-button>
      </div>
      
      <details v-if="showDetails" class="error-boundary__details">
        <summary>错误详情</summary>
        <pre>{{ errorDetails }}</pre>
      </details>
    </div>
    
    <slot v-else />
  </div>
</template>

<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue';
import { ElButton, ElIcon } from 'element-plus';
import { WarningFilled } from '@element-plus/icons-vue';
import router from '@/router';

interface Props {
  fallbackTitle?: string;
  fallbackMessage?: string;
  showDetails?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  fallbackTitle: '页面加载失败',
  fallbackMessage: '抱歉，页面遇到了一些问题。请尝试刷新页面或联系技术支持。',
  showDetails: false,
});

interface Emits {
  error: [error: Error];
  retry: [];
}

const emit = defineEmits<Emits>();

const hasError = ref(false);
const errorTitle = ref('');
const errorMessage = ref('');
const errorDetails = ref('');

onErrorCaptured((error: Error) => {
  hasError.value = true;
  errorTitle.value = props.fallbackTitle;
  errorMessage.value = props.fallbackMessage;
  errorDetails.value = error.stack || error.message;
  
  emit('error', error);
  
  // 阻止错误继续向上传播
  return false;
});

const handleRetry = () => {
  hasError.value = false;
  errorTitle.value = '';
  errorMessage.value = '';
  errorDetails.value = '';
  emit('retry');
};

const handleGoHome = () => {
  router.push('/');
};
</script>

<style scoped lang="scss">
.error-boundary {
  &__content {
    @apply flex flex-col items-center justify-center min-h-96 p-8 text-center;
  }
  
  &__icon {
    @apply mb-4;
  }
  
  &__title {
    @apply text-xl font-semibold text-gray-900 mb-2;
  }
  
  &__message {
    @apply text-gray-600 mb-6 max-w-md;
  }
  
  &__actions {
    @apply flex space-x-4 mb-6;
  }
  
  &__details {
    @apply text-left max-w-2xl w-full;
    
    summary {
      @apply cursor-pointer text-sm text-gray-500 mb-2;
    }
    
    pre {
      @apply bg-gray-100 p-4 rounded text-xs overflow-auto max-h-40;
    }
  }
}
</style>
```

---

## 部署配置

### Dockerfile

```dockerfile
# 多阶段构建
FROM node:18-alpine AS builder

# 设置工作目录
WORKDIR /app

# 复制package文件
COPY package*.json ./
COPY pnpm-lock.yaml ./

# 安装pnpm
RUN npm install -g pnpm

# 安装依赖
RUN pnpm install --frozen-lockfile

# 复制源代码
COPY . .

# 构建应用
RUN pnpm build

# 生产阶段
FROM nginx:alpine

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

# 暴露端口
EXPOSE 80

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
```

### Nginx配置

```nginx
# nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    # 基础配置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # 服务器配置
    server {
        listen 80;
        server_name localhost;
        root /usr/share/nginx/html;
        index index.html;
        
        # 安全头
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
        
        # 静态资源缓存
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            access_log off;
        }
        
        # API代理
        location /api/ {
            proxy_pass http://backend:3000/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 超时设置
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # SPA路由支持
        location / {
            try_files $uri $uri/ /index.html;
        }
        
        # 健康检查
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

### CI/CD配置

```yaml
# .github/workflows/deploy.yml
name: Deploy Frontend

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '18'
  PNPM_VERSION: '8'

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          
      - name: Setup pnpm
        uses: pnpm/action-setup@v2
        with:
          version: ${{ env.PNPM_VERSION }}
          
      - name: Get pnpm store directory
        id: pnpm-cache
        shell: bash
        run: |
          echo "STORE_PATH=$(pnpm store path)" >> $GITHUB_OUTPUT
          
      - name: Setup pnpm cache
        uses: actions/cache@v3
        with:
          path: ${{ steps.pnpm-cache.outputs.STORE_PATH }}
          key: ${{ runner.os }}-pnpm-store-${{ hashFiles('**/pnpm-lock.yaml') }}
          restore-keys: |
            ${{ runner.os }}-pnpm-store-
            
      - name: Install dependencies
        run: pnpm install --frozen-lockfile
        
      - name: Run linting
        run: pnpm lint
        
      - name: Run type checking
        run: pnpm type-check
        
      - name: Run unit tests
        run: pnpm test:unit
        
      - name: Run E2E tests
        run: pnpm test:e2e
        
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info
          
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          
      - name: Setup pnpm
        uses: pnpm/action-setup@v2
        with:
          version: ${{ env.PNPM_VERSION }}
          
      - name: Install dependencies
        run: pnpm install --frozen-lockfile
        
      - name: Build application
        run: pnpm build
        env:
          VITE_API_BASE_URL: ${{ secrets.API_BASE_URL }}
          VITE_APP_VERSION: ${{ github.sha }}
          
      - name: Build Docker image
        run: |
          docker build -t frontend:${{ github.sha }} .
          docker tag frontend:${{ github.sha }} frontend:latest
          
      - name: Login to Docker Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ secrets.DOCKER_REGISTRY }}
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
          
      - name: Push Docker image
        run: |
          docker push ${{ secrets.DOCKER_REGISTRY }}/frontend:${{ github.sha }}
          docker push ${{ secrets.DOCKER_REGISTRY }}/frontend:latest
          
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@v0.1.5
        with:
          host: ${{ secrets.DEPLOY_HOST }}
          username: ${{ secrets.DEPLOY_USER }}
          key: ${{ secrets.DEPLOY_KEY }}
          script: |
            cd /opt/app
            docker-compose pull frontend
            docker-compose up -d frontend
            docker system prune -f
```

---

## 开发工具配置

### VSCode配置

```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true,
    "source.organizeImports": true
  },
  "typescript.preferences.importModuleSpecifier": "relative",
  "typescript.suggest.autoImports": true,
  "vue.codeActions.enabled": true,
  "vue.complete.casing.tags": "kebab",
  "vue.complete.casing.props": "camel",
  "emmet.includeLanguages": {
    "vue-html": "html"
  },
  "files.associations": {
    "*.vue": "vue"
  },
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/.git": true
  }
}

// .vscode/extensions.json
{
  "recommendations": [
    "Vue.volar",
    "Vue.vscode-typescript-vue-plugin",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "ms-vscode.vscode-typescript-next",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense"
  ]
}
```

### ESLint配置

```javascript
// .eslintrc.js
module.exports = {
  root: true,
  env: {
    node: true,
    browser: true,
    es2022: true,
  },
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'plugin:vue/vue3-recommended',
    'plugin:@typescript-eslint/recommended',
    '@vue/typescript/recommended',
    'prettier',
  ],
  parser: 'vue-eslint-parser',
  parserOptions: {
    parser: '@typescript-eslint/parser',
    ecmaVersion: 2022,
    sourceType: 'module',
  },
  plugins: ['@typescript-eslint', 'vue'],
  rules: {
    // Vue规则
    'vue/multi-word-component-names': 'off',
    'vue/no-v-html': 'warn',
    'vue/require-default-prop': 'error',
    'vue/require-prop-types': 'error',
    'vue/component-name-in-template-casing': ['error', 'PascalCase'],
    'vue/custom-event-name-casing': ['error', 'camelCase'],
    
    // TypeScript规则
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    '@typescript-eslint/no-non-null-assertion': 'warn',
    
    // 通用规则
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'prefer-const': 'error',
    'no-var': 'error',
    'object-shorthand': 'error',
    'prefer-template': 'error',
  },
  overrides: [
    {
      files: ['**/__tests__/**/*', '**/*.{test,spec}.*'],
      env: {
        jest: true,
      },
      rules: {
        '@typescript-eslint/no-explicit-any': 'off',
      },
    },
  ],
};
```

### Prettier配置

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100,
  "bracketSpacing": true,
  "arrowParens": "avoid",
  "endOfLine": "lf",
  "vueIndentScriptAndStyle": false
}
```

---

## 最佳实践

### 1. 组件设计原则
- **单一职责**：每个组件只负责一个功能
- **可复用性**：设计通用的、可配置的组件
- **可测试性**：组件应该易于测试
- **性能优化**：合理使用响应式数据和计算属性

### 2. 状态管理原则
- **最小化状态**：只在store中存储必要的状态
- **单向数据流**：保持数据流向的可预测性
- **状态归一化**：避免数据冗余和不一致
- **合理分割**：按功能模块划分store

### 3. 代码组织原则
- **模块化**：按功能划分文件和目录
- **命名规范**：使用清晰、一致的命名
- **依赖管理**：合理管理组件间的依赖关系
- **文档完善**：为复杂逻辑添加注释和文档

### 4. 性能优化原则
- **懒加载**：按需加载组件和资源
- **缓存策略**：合理使用浏览器缓存
- **虚拟化**：对大数据量列表使用虚拟滚动
- **防抖节流**：优化高频事件处理

### 5. 安全最佳实践
- **输入验证**：对所有用户输入进行验证
- **XSS防护**：避免直接渲染用户输入的HTML
- **CSRF防护**：使用CSRF令牌保护表单
- **敏感信息**：不在前端存储敏感数据

### 6. 可访问性原则
- **语义化HTML**：使用正确的HTML标签
- **键盘导航**：支持键盘操作
- **屏幕阅读器**：添加适当的ARIA属性
- **颜色对比**：确保足够的颜色对比度

### 7. 国际化原则
- **文本外化**：将所有文本提取到语言文件
- **格式化**：正确处理日期、数字格式
- **RTL支持**：考虑从右到左的语言
- **字体支持**：确保字体支持目标语言

---

## 总结

本开发规范指南涵盖了前端开发的各个方面，从代码规范到部署配置，旨在确保项目的质量、可维护性和团队协作效率。

### 关键要点

1. **代码质量**：遵循TypeScript和Vue编码规范，使用ESLint和Prettier保持代码一致性
2. **测试覆盖**：实施完整的测试策略，包括单元测试、集成测试和E2E测试
3. **性能优化**：采用代码分割、懒加载、虚拟滚动等技术提升应用性能
4. **错误处理**：建立完善的错误处理机制，提供良好的用户体验
5. **部署自动化**：使用Docker和CI/CD实现自动化部署
6. **开发工具**：配置合适的开发工具提高开发效率
7. **最佳实践**：遵循组件设计、状态管理、安全性等最佳实践

### 持续改进

开发规范应该是一个持续演进的过程，团队应该：

- 定期回顾和更新规范
- 收集团队反馈和建议
- 关注技术发展趋势
- 分享最佳实践和经验

通过遵循这些规范和最佳实践，我们可以构建高质量、可维护、高性能的前端应用。