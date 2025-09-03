# 前端API接口设计文档

## 目录

1. [HTTP客户端配置](#http客户端配置)
2. [API接口设计](#api接口设计)
3. [状态管理设计](#状态管理设计)
4. [路由设计](#路由设计)
5. [错误处理](#错误处理)
6. [请求拦截器](#请求拦截器)
7. [响应处理](#响应处理)

---

## HTTP客户端配置

### Axios配置

```typescript
// utils/http.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ElMessage } from 'element-plus';
import { useUserStore } from '@/stores/user';
import router from '@/router';

// 基础配置
const baseConfig: AxiosRequestConfig = {
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
};

// 创建axios实例
const httpClient: AxiosInstance = axios.create(baseConfig);

// 请求拦截器
httpClient.interceptors.request.use(
  (config) => {
    const userStore = useUserStore();
    const token = userStore.token;
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // 添加请求时间戳
    config.metadata = { startTime: new Date() };
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
httpClient.interceptors.response.use(
  (response: AxiosResponse) => {
    const { data, config } = response;
    
    // 计算请求耗时
    if (config.metadata?.startTime) {
      const duration = new Date().getTime() - config.metadata.startTime.getTime();
      console.log(`API请求耗时: ${duration}ms - ${config.url}`);
    }
    
    // 统一处理响应格式
    if (data.code === 200) {
      return data.data;
    } else {
      ElMessage.error(data.message || '请求失败');
      return Promise.reject(new Error(data.message));
    }
  },
  (error) => {
    const { response } = error;
    
    if (response) {
      switch (response.status) {
        case 401:
          ElMessage.error('登录已过期，请重新登录');
          const userStore = useUserStore();
          userStore.logout();
          router.push('/login');
          break;
        case 403:
          ElMessage.error('没有权限访问该资源');
          break;
        case 404:
          ElMessage.error('请求的资源不存在');
          break;
        case 500:
          ElMessage.error('服务器内部错误');
          break;
        default:
          ElMessage.error(response.data?.message || '网络错误');
      }
    } else {
      ElMessage.error('网络连接失败');
    }
    
    return Promise.reject(error);
  }
);

export default httpClient;
```

---

## API接口设计

### 用户认证API

```typescript
// api/auth.ts
import httpClient from '@/utils/http';

export interface LoginRequest {
  username: string;
  password: string;
  captcha?: string;
}

export interface LoginResponse {
  token: string;
  refreshToken: string;
  user: {
    id: string;
    username: string;
    email: string;
    role: string;
    permissions: string[];
  };
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export const authApi = {
  // 用户登录
  login: (data: LoginRequest): Promise<LoginResponse> => {
    return httpClient.post('/auth/login', data);
  },
  
  // 用户注册
  register: (data: RegisterRequest): Promise<void> => {
    return httpClient.post('/auth/register', data);
  },
  
  // 刷新token
  refreshToken: (refreshToken: string): Promise<{ token: string }> => {
    return httpClient.post('/auth/refresh', { refreshToken });
  },
  
  // 用户登出
  logout: (): Promise<void> => {
    return httpClient.post('/auth/logout');
  },
  
  // 获取用户信息
  getUserInfo: (): Promise<LoginResponse['user']> => {
    return httpClient.get('/auth/user');
  },
  
  // 修改密码
  changePassword: (data: {
    oldPassword: string;
    newPassword: string;
  }): Promise<void> => {
    return httpClient.put('/auth/password', data);
  },
};
```

### 内容管理API

```typescript
// api/content.ts
import httpClient from '@/utils/http';

export interface Content {
  id: string;
  title: string;
  originalText: string;
  optimizedText?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  similarity?: number;
  aiScore?: number;
  tags: string[];
  sourceId: string;
  createdAt: string;
  updatedAt: string;
}

export interface ContentListParams {
  page: number;
  pageSize: number;
  keyword?: string;
  status?: string;
  sourceId?: string;
  startDate?: string;
  endDate?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface ContentListResponse {
  data: Content[];
  total: number;
  page: number;
  pageSize: number;
}

export interface CreateContentRequest {
  title: string;
  originalText: string;
  tags?: string[];
  sourceId: string;
}

export interface OptimizeContentRequest {
  contentId: string;
  optimizationType: 'grammar' | 'style' | 'clarity' | 'comprehensive';
  targetAudience?: string;
  tone?: string;
}

export const contentApi = {
  // 获取内容列表
  getContentList: (params: ContentListParams): Promise<ContentListResponse> => {
    return httpClient.get('/content', { params });
  },
  
  // 获取内容详情
  getContentById: (id: string): Promise<Content> => {
    return httpClient.get(`/content/${id}`);
  },
  
  // 创建内容
  createContent: (data: CreateContentRequest): Promise<Content> => {
    return httpClient.post('/content', data);
  },
  
  // 更新内容
  updateContent: (id: string, data: Partial<Content>): Promise<Content> => {
    return httpClient.put(`/content/${id}`, data);
  },
  
  // 删除内容
  deleteContent: (id: string): Promise<void> => {
    return httpClient.delete(`/content/${id}`);
  },
  
  // 批量删除内容
  batchDeleteContent: (ids: string[]): Promise<void> => {
    return httpClient.delete('/content/batch', { data: { ids } });
  },
  
  // AI优化内容
  optimizeContent: (data: OptimizeContentRequest): Promise<{
    taskId: string;
    estimatedTime: number;
  }> => {
    return httpClient.post('/content/optimize', data);
  },
  
  // 获取优化任务状态
  getOptimizeTaskStatus: (taskId: string): Promise<{
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
    result?: string;
    error?: string;
  }> => {
    return httpClient.get(`/content/optimize/${taskId}`);
  },
  
  // 相似度检测
  checkSimilarity: (contentId: string, targetText: string): Promise<{
    similarity: number;
    details: {
      sentence: string;
      similarity: number;
      source: string;
    }[];
  }> => {
    return httpClient.post(`/content/${contentId}/similarity`, { targetText });
  },
};
```

### 数据源管理API

```typescript
// api/datasource.ts
import httpClient from '@/utils/http';

export interface DataSource {
  id: string;
  name: string;
  type: 'database' | 'api' | 'file' | 'web';
  config: Record<string, any>;
  status: 'active' | 'inactive' | 'error';
  lastSyncTime?: string;
  totalRecords: number;
  createdAt: string;
  updatedAt: string;
}

export interface CreateDataSourceRequest {
  name: string;
  type: DataSource['type'];
  config: Record<string, any>;
}

export interface SyncDataSourceRequest {
  sourceId: string;
  fullSync?: boolean;
  filters?: Record<string, any>;
}

export const dataSourceApi = {
  // 获取数据源列表
  getDataSources: (): Promise<DataSource[]> => {
    return httpClient.get('/datasource');
  },
  
  // 获取数据源详情
  getDataSourceById: (id: string): Promise<DataSource> => {
    return httpClient.get(`/datasource/${id}`);
  },
  
  // 创建数据源
  createDataSource: (data: CreateDataSourceRequest): Promise<DataSource> => {
    return httpClient.post('/datasource', data);
  },
  
  // 更新数据源
  updateDataSource: (id: string, data: Partial<DataSource>): Promise<DataSource> => {
    return httpClient.put(`/datasource/${id}`, data);
  },
  
  // 删除数据源
  deleteDataSource: (id: string): Promise<void> => {
    return httpClient.delete(`/datasource/${id}`);
  },
  
  // 测试数据源连接
  testConnection: (config: Record<string, any>): Promise<{
    success: boolean;
    message: string;
    details?: any;
  }> => {
    return httpClient.post('/datasource/test', { config });
  },
  
  // 同步数据源
  syncDataSource: (data: SyncDataSourceRequest): Promise<{
    taskId: string;
    estimatedTime: number;
  }> => {
    return httpClient.post('/datasource/sync', data);
  },
  
  // 获取同步任务状态
  getSyncTaskStatus: (taskId: string): Promise<{
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
    processedRecords: number;
    totalRecords: number;
    error?: string;
  }> => {
    return httpClient.get(`/datasource/sync/${taskId}`);
  },
};
```

### 发布管理API

```typescript
// api/publish.ts
import httpClient from '@/utils/http';

export interface PublishPlatform {
  id: string;
  name: string;
  type: 'wechat' | 'weibo' | 'douyin' | 'xiaohongshu' | 'zhihu';
  config: Record<string, any>;
  status: 'connected' | 'disconnected' | 'error';
  lastPublishTime?: string;
  totalPublished: number;
}

export interface PublishTask {
  id: string;
  contentId: string;
  platformIds: string[];
  scheduledTime?: string;
  status: 'draft' | 'scheduled' | 'publishing' | 'published' | 'failed';
  publishResults: {
    platformId: string;
    status: 'success' | 'failed';
    publishedUrl?: string;
    error?: string;
  }[];
  createdAt: string;
  updatedAt: string;
}

export interface CreatePublishTaskRequest {
  contentId: string;
  platformIds: string[];
  scheduledTime?: string;
  publishSettings?: Record<string, any>;
}

export const publishApi = {
  // 获取发布平台列表
  getPlatforms: (): Promise<PublishPlatform[]> => {
    return httpClient.get('/publish/platforms');
  },
  
  // 连接发布平台
  connectPlatform: (platformId: string, config: Record<string, any>): Promise<void> => {
    return httpClient.post(`/publish/platforms/${platformId}/connect`, { config });
  },
  
  // 断开发布平台
  disconnectPlatform: (platformId: string): Promise<void> => {
    return httpClient.post(`/publish/platforms/${platformId}/disconnect`);
  },
  
  // 获取发布任务列表
  getPublishTasks: (params: {
    page: number;
    pageSize: number;
    status?: string;
    platformId?: string;
  }): Promise<{
    data: PublishTask[];
    total: number;
  }> => {
    return httpClient.get('/publish/tasks', { params });
  },
  
  // 创建发布任务
  createPublishTask: (data: CreatePublishTaskRequest): Promise<PublishTask> => {
    return httpClient.post('/publish/tasks', data);
  },
  
  // 立即发布
  publishNow: (taskId: string): Promise<void> => {
    return httpClient.post(`/publish/tasks/${taskId}/publish`);
  },
  
  // 取消发布任务
  cancelPublishTask: (taskId: string): Promise<void> => {
    return httpClient.post(`/publish/tasks/${taskId}/cancel`);
  },
  
  // 获取发布统计
  getPublishStats: (params: {
    startDate: string;
    endDate: string;
    platformId?: string;
  }): Promise<{
    totalTasks: number;
    successfulTasks: number;
    failedTasks: number;
    platformStats: {
      platformId: string;
      platformName: string;
      totalTasks: number;
      successRate: number;
    }[];
  }> => {
    return httpClient.get('/publish/stats', { params });
  },
};
```

---

## 状态管理设计

### 用户状态管理

```typescript
// stores/user.ts
import { defineStore } from 'pinia';
import { authApi, type LoginResponse } from '@/api/auth';

interface UserState {
  token: string | null;
  refreshToken: string | null;
  user: LoginResponse['user'] | null;
  permissions: string[];
  isLoggedIn: boolean;
}

export const useUserStore = defineStore('user', {
  state: (): UserState => ({
    token: localStorage.getItem('token'),
    refreshToken: localStorage.getItem('refreshToken'),
    user: null,
    permissions: [],
    isLoggedIn: false,
  }),
  
  getters: {
    hasPermission: (state) => (permission: string) => {
      return state.permissions.includes(permission);
    },
    
    hasAnyPermission: (state) => (permissions: string[]) => {
      return permissions.some(permission => state.permissions.includes(permission));
    },
    
    isAdmin: (state) => {
      return state.user?.role === 'admin';
    },
  },
  
  actions: {
    async login(credentials: { username: string; password: string }) {
      try {
        const response = await authApi.login(credentials);
        
        this.token = response.token;
        this.refreshToken = response.refreshToken;
        this.user = response.user;
        this.permissions = response.user.permissions;
        this.isLoggedIn = true;
        
        // 保存到本地存储
        localStorage.setItem('token', response.token);
        localStorage.setItem('refreshToken', response.refreshToken);
        
        return response;
      } catch (error) {
        this.logout();
        throw error;
      }
    },
    
    async logout() {
      try {
        if (this.token) {
          await authApi.logout();
        }
      } catch (error) {
        console.error('登出失败:', error);
      } finally {
        this.token = null;
        this.refreshToken = null;
        this.user = null;
        this.permissions = [];
        this.isLoggedIn = false;
        
        // 清除本地存储
        localStorage.removeItem('token');
        localStorage.removeItem('refreshToken');
      }
    },
    
    async refreshUserToken() {
      if (!this.refreshToken) {
        throw new Error('没有刷新令牌');
      }
      
      try {
        const response = await authApi.refreshToken(this.refreshToken);
        this.token = response.token;
        localStorage.setItem('token', response.token);
        return response.token;
      } catch (error) {
        this.logout();
        throw error;
      }
    },
    
    async getUserInfo() {
      try {
        const user = await authApi.getUserInfo();
        this.user = user;
        this.permissions = user.permissions;
        this.isLoggedIn = true;
        return user;
      } catch (error) {
        this.logout();
        throw error;
      }
    },
  },
});
```

### 内容管理状态

```typescript
// stores/content.ts
import { defineStore } from 'pinia';
import { contentApi, type Content, type ContentListParams } from '@/api/content';

interface ContentState {
  contentList: Content[];
  currentContent: Content | null;
  loading: boolean;
  pagination: {
    page: number;
    pageSize: number;
    total: number;
  };
  filters: Partial<ContentListParams>;
  optimizeTasks: Map<string, {
    taskId: string;
    status: string;
    progress: number;
  }>;
}

export const useContentStore = defineStore('content', {
  state: (): ContentState => ({
    contentList: [],
    currentContent: null,
    loading: false,
    pagination: {
      page: 1,
      pageSize: 20,
      total: 0,
    },
    filters: {},
    optimizeTasks: new Map(),
  }),
  
  getters: {
    pendingContents: (state) => {
      return state.contentList.filter(content => content.status === 'pending');
    },
    
    completedContents: (state) => {
      return state.contentList.filter(content => content.status === 'completed');
    },
    
    hasOptimizeTask: (state) => (contentId: string) => {
      return state.optimizeTasks.has(contentId);
    },
  },
  
  actions: {
    async fetchContentList(params?: Partial<ContentListParams>) {
      this.loading = true;
      try {
        const queryParams = {
          ...this.filters,
          ...params,
          page: this.pagination.page,
          pageSize: this.pagination.pageSize,
        };
        
        const response = await contentApi.getContentList(queryParams);
        
        this.contentList = response.data;
        this.pagination.total = response.total;
        this.pagination.page = response.page;
        this.pagination.pageSize = response.pageSize;
      } catch (error) {
        console.error('获取内容列表失败:', error);
        throw error;
      } finally {
        this.loading = false;
      }
    },
    
    async fetchContentById(id: string) {
      try {
        const content = await contentApi.getContentById(id);
        this.currentContent = content;
        return content;
      } catch (error) {
        console.error('获取内容详情失败:', error);
        throw error;
      }
    },
    
    async createContent(data: any) {
      try {
        const content = await contentApi.createContent(data);
        this.contentList.unshift(content);
        this.pagination.total += 1;
        return content;
      } catch (error) {
        console.error('创建内容失败:', error);
        throw error;
      }
    },
    
    async updateContent(id: string, data: Partial<Content>) {
      try {
        const updatedContent = await contentApi.updateContent(id, data);
        
        const index = this.contentList.findIndex(content => content.id === id);
        if (index !== -1) {
          this.contentList[index] = updatedContent;
        }
        
        if (this.currentContent?.id === id) {
          this.currentContent = updatedContent;
        }
        
        return updatedContent;
      } catch (error) {
        console.error('更新内容失败:', error);
        throw error;
      }
    },
    
    async deleteContent(id: string) {
      try {
        await contentApi.deleteContent(id);
        
        const index = this.contentList.findIndex(content => content.id === id);
        if (index !== -1) {
          this.contentList.splice(index, 1);
          this.pagination.total -= 1;
        }
        
        if (this.currentContent?.id === id) {
          this.currentContent = null;
        }
      } catch (error) {
        console.error('删除内容失败:', error);
        throw error;
      }
    },
    
    async optimizeContent(contentId: string, options: any) {
      try {
        const response = await contentApi.optimizeContent({
          contentId,
          ...options,
        });
        
        // 保存优化任务信息
        this.optimizeTasks.set(contentId, {
          taskId: response.taskId,
          status: 'pending',
          progress: 0,
        });
        
        return response;
      } catch (error) {
        console.error('启动内容优化失败:', error);
        throw error;
      }
    },
    
    async checkOptimizeTaskStatus(contentId: string) {
      const task = this.optimizeTasks.get(contentId);
      if (!task) return;
      
      try {
        const status = await contentApi.getOptimizeTaskStatus(task.taskId);
        
        // 更新任务状态
        this.optimizeTasks.set(contentId, {
          ...task,
          status: status.status,
          progress: status.progress,
        });
        
        // 如果任务完成，更新内容
        if (status.status === 'completed' && status.result) {
          await this.updateContent(contentId, {
            optimizedText: status.result,
            status: 'completed',
          });
          
          // 移除任务记录
          this.optimizeTasks.delete(contentId);
        }
        
        return status;
      } catch (error) {
        console.error('检查优化任务状态失败:', error);
        throw error;
      }
    },
    
    setFilters(filters: Partial<ContentListParams>) {
      this.filters = { ...this.filters, ...filters };
      this.pagination.page = 1;
    },
    
    clearFilters() {
      this.filters = {};
      this.pagination.page = 1;
    },
    
    setPage(page: number) {
      this.pagination.page = page;
    },
    
    setPageSize(pageSize: number) {
      this.pagination.pageSize = pageSize;
      this.pagination.page = 1;
    },
  },
});
```

---

## 路由设计

### 路由配置

```typescript
// router/index.ts
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router';
import { useUserStore } from '@/stores/user';
import { ElMessage } from 'element-plus';

// 路由配置
const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: {
      title: '用户登录',
      requiresAuth: false,
      hideInMenu: true,
    },
  },
  {
    path: '/',
    name: 'Layout',
    component: () => import('@/layouts/MainLayout.vue'),
    redirect: '/dashboard',
    meta: {
      requiresAuth: true,
    },
    children: [
      {
        path: '/dashboard',
        name: 'Dashboard',
        component: () => import('@/views/Dashboard.vue'),
        meta: {
          title: '仪表盘',
          icon: 'dashboard',
          requiresAuth: true,
        },
      },
      {
        path: '/datasource',
        name: 'DataSource',
        component: () => import('@/views/DataSource/index.vue'),
        meta: {
          title: '数据源管理',
          icon: 'database',
          requiresAuth: true,
          permissions: ['datasource:read'],
        },
        children: [
          {
            path: '/datasource/list',
            name: 'DataSourceList',
            component: () => import('@/views/DataSource/List.vue'),
            meta: {
              title: '数据源列表',
              requiresAuth: true,
              permissions: ['datasource:read'],
            },
          },
          {
            path: '/datasource/create',
            name: 'DataSourceCreate',
            component: () => import('@/views/DataSource/Create.vue'),
            meta: {
              title: '创建数据源',
              requiresAuth: true,
              permissions: ['datasource:create'],
            },
          },
          {
            path: '/datasource/:id/edit',
            name: 'DataSourceEdit',
            component: () => import('@/views/DataSource/Edit.vue'),
            meta: {
              title: '编辑数据源',
              requiresAuth: true,
              permissions: ['datasource:update'],
            },
          },
        ],
      },
      {
        path: '/content',
        name: 'Content',
        component: () => import('@/views/Content/index.vue'),
        meta: {
          title: '内容管理',
          icon: 'document',
          requiresAuth: true,
          permissions: ['content:read'],
        },
        children: [
          {
            path: '/content/list',
            name: 'ContentList',
            component: () => import('@/views/Content/List.vue'),
            meta: {
              title: '内容列表',
              requiresAuth: true,
              permissions: ['content:read'],
            },
          },
          {
            path: '/content/create',
            name: 'ContentCreate',
            component: () => import('@/views/Content/Create.vue'),
            meta: {
              title: '创建内容',
              requiresAuth: true,
              permissions: ['content:create'],
            },
          },
          {
            path: '/content/:id',
            name: 'ContentDetail',
            component: () => import('@/views/Content/Detail.vue'),
            meta: {
              title: '内容详情',
              requiresAuth: true,
              permissions: ['content:read'],
            },
          },
          {
            path: '/content/:id/edit',
            name: 'ContentEdit',
            component: () => import('@/views/Content/Edit.vue'),
            meta: {
              title: '编辑内容',
              requiresAuth: true,
              permissions: ['content:update'],
            },
          },
        ],
      },
      {
        path: '/publish',
        name: 'Publish',
        component: () => import('@/views/Publish/index.vue'),
        meta: {
          title: '发布管理',
          icon: 'share',
          requiresAuth: true,
          permissions: ['publish:read'],
        },
        children: [
          {
            path: '/publish/tasks',
            name: 'PublishTasks',
            component: () => import('@/views/Publish/Tasks.vue'),
            meta: {
              title: '发布任务',
              requiresAuth: true,
              permissions: ['publish:read'],
            },
          },
          {
            path: '/publish/platforms',
            name: 'PublishPlatforms',
            component: () => import('@/views/Publish/Platforms.vue'),
            meta: {
              title: '发布平台',
              requiresAuth: true,
              permissions: ['publish:platform'],
            },
          },
        ],
      },
      {
        path: '/customers',
        name: 'Customers',
        component: () => import('@/views/Customers.vue'),
        meta: {
          title: '客户管理',
          icon: 'users',
          requiresAuth: true,
          permissions: ['customer:read'],
        },
      },
      {
        path: '/settings',
        name: 'Settings',
        component: () => import('@/views/Settings.vue'),
        meta: {
          title: '系统设置',
          icon: 'settings',
          requiresAuth: true,
          permissions: ['system:settings'],
        },
      },
      {
        path: '/ai-monitoring',
        name: 'AIMonitoring',
        component: () => import('@/views/AIMonitoring.vue'),
        meta: {
          title: 'AI文本优化监控',
          icon: 'monitor',
          requiresAuth: true,
          permissions: ['ai:monitor'],
        },
      },
      {
        path: '/system-monitoring',
        name: 'SystemMonitoring',
        component: () => import('@/views/SystemMonitoring.vue'),
        meta: {
          title: '系统监控运维',
          icon: 'server',
          requiresAuth: true,
          permissions: ['system:monitor'],
        },
      },
    ],
  },
  {
    path: '/403',
    name: 'Forbidden',
    component: () => import('@/views/Error/403.vue'),
    meta: {
      title: '访问被拒绝',
      hideInMenu: true,
    },
  },
  {
    path: '/404',
    name: 'NotFound',
    component: () => import('@/views/Error/404.vue'),
    meta: {
      title: '页面不存在',
      hideInMenu: true,
    },
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/404',
  },
];

// 创建路由实例
const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition;
    } else {
      return { top: 0 };
    }
  },
});

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const userStore = useUserStore();
  
  // 设置页面标题
  if (to.meta.title) {
    document.title = `${to.meta.title} - 历史文本优化系统`;
  }
  
  // 检查是否需要认证
  if (to.meta.requiresAuth) {
    if (!userStore.token) {
      ElMessage.warning('请先登录');
      next({ name: 'Login', query: { redirect: to.fullPath } });
      return;
    }
    
    // 如果有token但没有用户信息，尝试获取用户信息
    if (!userStore.user) {
      try {
        await userStore.getUserInfo();
      } catch (error) {
        ElMessage.error('获取用户信息失败，请重新登录');
        next({ name: 'Login' });
        return;
      }
    }
    
    // 检查权限
    if (to.meta.permissions && Array.isArray(to.meta.permissions)) {
      const hasPermission = userStore.hasAnyPermission(to.meta.permissions);
      if (!hasPermission) {
        ElMessage.error('没有访问权限');
        next({ name: 'Forbidden' });
        return;
      }
    }
  }
  
  // 如果已登录用户访问登录页，重定向到首页
  if (to.name === 'Login' && userStore.isLoggedIn) {
    next({ name: 'Dashboard' });
    return;
  }
  
  next();
});

export default router;
```

### 路由工具函数

```typescript
// utils/route.ts
import type { RouteRecordRaw } from 'vue-router';
import { useUserStore } from '@/stores/user';

/**
 * 过滤路由，根据用户权限显示菜单
 */
export function filterRoutes(routes: RouteRecordRaw[]): RouteRecordRaw[] {
  const userStore = useUserStore();
  
  return routes.filter(route => {
    // 隐藏不在菜单中显示的路由
    if (route.meta?.hideInMenu) {
      return false;
    }
    
    // 检查权限
    if (route.meta?.permissions) {
      const hasPermission = userStore.hasAnyPermission(route.meta.permissions);
      if (!hasPermission) {
        return false;
      }
    }
    
    // 递归过滤子路由
    if (route.children) {
      route.children = filterRoutes(route.children);
    }
    
    return true;
  });
}

/**
 * 生成面包屑导航
 */
export function generateBreadcrumbs(route: any): Array<{ name: string; path?: string }> {
  const breadcrumbs: Array<{ name: string; path?: string }> = [];
  
  // 递归查找父级路由
  function findParents(currentRoute: any, allRoutes: RouteRecordRaw[]) {
    for (const r of allRoutes) {
      if (r.name === currentRoute.name) {
        if (r.meta?.title) {
          breadcrumbs.unshift({
            name: r.meta.title,
            path: r.path,
          });
        }
        return true;
      }
      
      if (r.children) {
        const found = findParents(currentRoute, r.children);
        if (found && r.meta?.title) {
          breadcrumbs.unshift({
            name: r.meta.title,
            path: r.path,
          });
          return true;
        }
      }
    }
    return false;
  }
  
  return breadcrumbs;
}

/**
 * 检查路由权限
 */
export function hasRoutePermission(route: RouteRecordRaw): boolean {
  const userStore = useUserStore();
  
  if (!route.meta?.permissions) {
    return true;
  }
  
  return userStore.hasAnyPermission(route.meta.permissions);
}
```

---

## 错误处理

### API错误处理

```typescript
// utils/error.ts
import { ElMessage, ElNotification } from 'element-plus';

export class ApiError extends Error {
  public code: number;
  public data?: any;
  
  constructor(message: string, code: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.data = data;
  }
}

/**
 * 统一错误处理函数
 */
export function handleError(error: any, showMessage = true) {
  console.error('错误详情:', error);
  
  let message = '未知错误';
  let title = '错误';
  
  if (error instanceof ApiError) {
    message = error.message;
    title = `错误 ${error.code}`;
  } else if (error.response) {
    // HTTP错误
    const { status, data } = error.response;
    message = data?.message || `HTTP ${status} 错误`;
    title = `网络错误 ${status}`;
  } else if (error.request) {
    // 网络错误
    message = '网络连接失败，请检查网络设置';
    title = '网络错误';
  } else {
    // 其他错误
    message = error.message || '操作失败';
  }
  
  if (showMessage) {
    if (error.code >= 500) {
      // 服务器错误使用通知
      ElNotification({
        title,
        message,
        type: 'error',
        duration: 5000,
      });
    } else {
      // 客户端错误使用消息
      ElMessage.error(message);
    }
  }
  
  return { message, title, code: error.code || 0 };
}

/**
 * 异步操作错误处理装饰器
 */
export function withErrorHandling<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  showMessage = true
): T {
  return (async (...args: any[]) => {
    try {
      return await fn(...args);
    } catch (error) {
      handleError(error, showMessage);
      throw error;
    }
  }) as T;
}

/**
 * 表单验证错误处理
 */
export function handleValidationError(error: any) {
  if (error.response?.status === 422) {
    const errors = error.response.data?.errors || {};
    const messages = Object.values(errors).flat();
    
    if (messages.length > 0) {
      ElMessage.error(messages[0] as string);
    }
    
    return errors;
  }
  
  handleError(error);
  return {};
}
```

---

## 总结

本文档定义了历史文本优化项目的前端API接口设计，包括：

1. **HTTP客户端配置**：基于Axios的请求拦截器和响应处理
2. **API接口设计**：用户认证、内容管理、数据源管理、发布管理等核心API
3. **状态管理设计**：基于Pinia的用户状态和内容状态管理
4. **路由设计**：包含权限控制和导航守卫的Vue Router配置
5. **错误处理**：统一的错误处理机制和用户友好的错误提示

通过这些设计，确保前端应用具有良好的数据流管理、用户体验和错误处理能力。

---

**文档版本**: v1.0  
**最后更新**: 2024年1月  
**维护团队**: 前端开发组