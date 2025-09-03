# 性能与扩展设计文档

## 1. 性能设计概述

历史文本漂洗项目采用多层次的性能优化策略，从前端到后端、从应用层到数据层，全方位提升系统性能。通过合理的架构设计、缓存策略、数据库优化和资源管理，确保系统在高并发场景下的稳定运行。

### 1.1 性能目标

| 性能指标 | 目标值 | 说明 |
|----------|--------|------|
| 响应时间 | < 200ms | API平均响应时间 |
| 吞吐量 | > 1000 QPS | 系统并发处理能力 |
| 可用性 | 99.9% | 系统年度可用性 |
| 文档处理 | < 30s | 单个文档OCR处理时间 |
| 搜索延迟 | < 100ms | 全文搜索响应时间 |
| 并发用户 | > 10000 | 同时在线用户数 |

### 1.2 性能监控体系

```mermaid
graph TB
    subgraph "性能监控层"
        APM[APM监控]
        METRICS[指标收集]
        TRACING[链路追踪]
        LOGGING[日志分析]
    end
    
    subgraph "应用层监控"
        API_METRICS[API性能指标]
        BUSINESS_METRICS[业务指标]
        ERROR_TRACKING[错误追踪]
    end
    
    subgraph "基础设施监控"
        RESOURCE_MONITOR[资源监控]
        NETWORK_MONITOR[网络监控]
        STORAGE_MONITOR[存储监控]
    end
    
    subgraph "告警系统"
        ALERT_RULES[告警规则]
        NOTIFICATION[通知系统]
        ESCALATION[升级策略]
    end
    
    APM --> API_METRICS
    METRICS --> BUSINESS_METRICS
    TRACING --> ERROR_TRACKING
    
    API_METRICS --> RESOURCE_MONITOR
    BUSINESS_METRICS --> NETWORK_MONITOR
    ERROR_TRACKING --> STORAGE_MONITOR
    
    RESOURCE_MONITOR --> ALERT_RULES
    NETWORK_MONITOR --> NOTIFICATION
    STORAGE_MONITOR --> ESCALATION
```

## 2. 前端性能优化

### 2.1 资源优化策略

```javascript
// webpack.config.js - 前端构建优化
const path = require('path');
const CompressionPlugin = require('compression-webpack-plugin');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
  mode: 'production',
  
  // 代码分割配置
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          chunks: 'all',
          priority: 10
        },
        common: {
          name: 'common',
          minChunks: 2,
          chunks: 'all',
          priority: 5
        }
      }
    },
    
    // 代码压缩
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true,
            drop_debugger: true
          }
        }
      })
    ]
  },
  
  // 插件配置
  plugins: [
    // Gzip压缩
    new CompressionPlugin({
      algorithm: 'gzip',
      test: /\.(js|css|html|svg)$/,
      threshold: 8192,
      minRatio: 0.8
    }),
    
    // 包分析
    new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      openAnalyzer: false
    })
  ],
  
  // 模块解析优化
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      'vue$': 'vue/dist/vue.runtime.esm-bundler.js'
    },
    extensions: ['.js', '.vue', '.json', '.ts']
  }
};
```

### 2.2 Vue3性能优化

```vue
<!-- DocumentList.vue - 虚拟滚动优化 -->
<template>
  <div class="document-list">
    <!-- 虚拟滚动列表 -->
    <RecycleScroller
      class="scroller"
      :items="documents"
      :item-size="120"
      key-field="id"
      v-slot="{ item, index }"
    >
      <DocumentItem
        :key="item.id"
        :document="item"
        :index="index"
        @click="handleDocumentClick"
      />
    </RecycleScroller>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { RecycleScroller } from 'vue-virtual-scroller';
import DocumentItem from './DocumentItem.vue';
import { useDocumentStore } from '@/stores/document';
import { debounce } from 'lodash-es';

// 响应式数据
const documentStore = useDocumentStore();
const searchQuery = ref('');
const loading = ref(false);

// 计算属性 - 使用缓存
const documents = computed(() => {
  return documentStore.filteredDocuments;
});

// 防抖搜索
const debouncedSearch = debounce(async (query: string) => {
  loading.value = true;
  try {
    await documentStore.searchDocuments(query);
  } finally {
    loading.value = false;
  }
}, 300);

// 监听搜索变化
watch(searchQuery, (newQuery) => {
  debouncedSearch(newQuery);
});

// 事件处理
const handleDocumentClick = (document: Document) => {
  // 使用路由懒加载
  router.push({
    name: 'DocumentDetail',
    params: { id: document.id }
  });
};

// 生命周期
onMounted(async () => {
  // 预加载关键数据
  await documentStore.loadDocuments();
});
</script>

<style scoped>
.document-list {
  height: 100%;
  overflow: hidden;
}

.scroller {
  height: 100%;
}

/* 使用CSS containment优化 */
.document-item {
  contain: layout style paint;
}
</style>
```

### 2.3 缓存策略

```typescript
// cache.ts - 前端缓存管理
import { LRUCache } from 'lru-cache';

class CacheManager {
  private apiCache: LRUCache<string, any>;
  private imageCache: LRUCache<string, string>;
  
  constructor() {
    // API响应缓存
    this.apiCache = new LRUCache({
      max: 500,
      ttl: 1000 * 60 * 5, // 5分钟
      allowStale: true,
      updateAgeOnGet: true
    });
    
    // 图片缓存
    this.imageCache = new LRUCache({
      max: 100,
      ttl: 1000 * 60 * 30, // 30分钟
      sizeCalculation: (value) => value.length
    });
  }
  
  // API缓存方法
  async getCachedApiResponse<T>(key: string, fetcher: () => Promise<T>): Promise<T> {
    const cached = this.apiCache.get(key);
    if (cached) {
      return cached;
    }
    
    const result = await fetcher();
    this.apiCache.set(key, result);
    return result;
  }
  
  // 图片预加载
  async preloadImage(url: string): Promise<string> {
    const cached = this.imageCache.get(url);
    if (cached) {
      return cached;
    }
    
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.imageCache.set(url, url);
        resolve(url);
      };
      img.onerror = reject;
      img.src = url;
    });
  }
  
  // 清理缓存
  clearCache(pattern?: string) {
    if (pattern) {
      for (const key of this.apiCache.keys()) {
        if (key.includes(pattern)) {
          this.apiCache.delete(key);
        }
      }
    } else {
      this.apiCache.clear();
      this.imageCache.clear();
    }
  }
}

export const cacheManager = new CacheManager();

// HTTP拦截器中使用缓存
import axios from 'axios';

axios.interceptors.request.use((config) => {
  // 为GET请求添加缓存键
  if (config.method === 'get') {
    const cacheKey = `${config.url}?${new URLSearchParams(config.params).toString()}`;
    config.metadata = { cacheKey };
  }
  return config;
});

axios.interceptors.response.use((response) => {
  // 缓存GET请求响应
  if (response.config.method === 'get' && response.config.metadata?.cacheKey) {
    cacheManager.getCachedApiResponse(
      response.config.metadata.cacheKey,
      () => Promise.resolve(response.data)
    );
  }
  return response;
});
```

## 3. 后端性能优化

### 3.1 数据库优化

```sql
-- 数据库索引优化
-- 用户表索引
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_users_username ON users(username);
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at DESC);
CREATE INDEX CONCURRENTLY idx_users_active ON users(is_active) WHERE is_active = true;

-- 文档表复合索引
CREATE INDEX CONCURRENTLY idx_documents_owner_status ON documents(owner_id, status);
CREATE INDEX CONCURRENTLY idx_documents_category_created ON documents(category_id, created_at DESC);
CREATE INDEX CONCURRENTLY idx_documents_search ON documents USING gin(to_tsvector('chinese', title || ' ' || description));

-- 文档内容表分区
CREATE TABLE document_contents (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- 按月分区
CREATE TABLE document_contents_2024_01 PARTITION OF document_contents
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE document_contents_2024_02 PARTITION OF document_contents
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 查询优化视图
CREATE MATERIALIZED VIEW document_stats AS
SELECT 
    d.category_id,
    c.name as category_name,
    COUNT(*) as document_count,
    AVG(d.file_size) as avg_file_size,
    MAX(d.created_at) as latest_document
FROM documents d
JOIN categories c ON d.category_id = c.id
WHERE d.status = 'completed'
GROUP BY d.category_id, c.name;

-- 定期刷新统计视图
CREATE OR REPLACE FUNCTION refresh_document_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY document_stats;
END;
$$ LANGUAGE plpgsql;

-- 定时任务
SELECT cron.schedule('refresh-stats', '0 */6 * * *', 'SELECT refresh_document_stats();');
```

### 3.2 连接池优化

```python
# database.py - 数据库连接池配置
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os

class DatabaseConfig:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '20'))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '30'))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', '3600'))
    
    def create_engine(self):
        return create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=True,  # 连接健康检查
            echo=False,
            future=True
        )

# 读写分离配置
class DatabaseManager:
    def __init__(self):
        self.write_engine = self._create_write_engine()
        self.read_engines = self._create_read_engines()
        self.SessionLocal = sessionmaker()
    
    def _create_write_engine(self):
        write_url = os.getenv('DATABASE_WRITE_URL')
        return create_engine(
            write_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600
        )
    
    def _create_read_engines(self):
        read_urls = os.getenv('DATABASE_READ_URLS', '').split(',')
        engines = []
        for url in read_urls:
            if url.strip():
                engine = create_engine(
                    url.strip(),
                    poolclass=QueuePool,
                    pool_size=15,
                    max_overflow=25,
                    pool_timeout=30,
                    pool_recycle=3600
                )
                engines.append(engine)
        return engines
    
    def get_write_session(self):
        """获取写数据库会话"""
        return self.SessionLocal(bind=self.write_engine)
    
    def get_read_session(self):
        """获取读数据库会话（负载均衡）"""
        import random
        if self.read_engines:
            engine = random.choice(self.read_engines)
            return self.SessionLocal(bind=engine)
        return self.get_write_session()

db_manager = DatabaseManager()
```

### 3.3 缓存层设计

```python
# cache.py - 多级缓存系统
import redis
import pickle
import hashlib
from typing import Any, Optional, Union
from functools import wraps
import asyncio
from datetime import timedelta

class CacheManager:
    def __init__(self):
        # Redis集群配置
        self.redis_cluster = redis.RedisCluster(
            host='localhost',
            port=7000,
            decode_responses=False,
            skip_full_coverage_check=True,
            health_check_interval=30
        )
        
        # 本地缓存
        self.local_cache = {}
        self.local_cache_ttl = {}
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        # 先检查本地缓存
        if key in self.local_cache:
            import time
            if time.time() < self.local_cache_ttl.get(key, 0):
                return self.local_cache[key]
            else:
                del self.local_cache[key]
                del self.local_cache_ttl[key]
        
        # 检查Redis缓存
        try:
            data = await self.redis_cluster.get(key)
            if data:
                value = pickle.loads(data)
                # 回填本地缓存
                self._set_local_cache(key, value, 60)  # 本地缓存1分钟
                return value
        except Exception as e:
            print(f"Redis cache error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存值"""
        try:
            # 设置Redis缓存
            data = pickle.dumps(value)
            await self.redis_cluster.setex(key, ttl, data)
            
            # 设置本地缓存
            local_ttl = min(ttl, 300)  # 本地缓存最多5分钟
            self._set_local_cache(key, value, local_ttl)
            
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def _set_local_cache(self, key: str, value: Any, ttl: int):
        """设置本地缓存"""
        import time
        self.local_cache[key] = value
        self.local_cache_ttl[key] = time.time() + ttl
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            # 删除本地缓存
            self.local_cache.pop(key, None)
            self.local_cache_ttl.pop(key, None)
            
            # 删除Redis缓存
            await self.redis_cluster.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """按模式清理缓存"""
        try:
            keys = await self.redis_cluster.keys(pattern)
            if keys:
                return await self.redis_cluster.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache clear error: {e}")
            return 0

cache_manager = CacheManager()

# 缓存装饰器
def cached(prefix: str, ttl: int = 3600, key_func=None):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._generate_key(prefix, *args, **kwargs)
            
            # 尝试从缓存获取
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 缓存结果
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# 使用示例
@cached(prefix="user", ttl=1800)
async def get_user_by_id(user_id: str):
    # 从数据库获取用户信息
    pass

@cached(prefix="document_list", ttl=300, key_func=lambda user_id, page, limit: f"docs:{user_id}:{page}:{limit}")
async def get_user_documents(user_id: str, page: int, limit: int):
    # 获取用户文档列表
    pass
```

### 3.4 异步处理优化

```python
# async_processor.py - 异步任务处理
import asyncio
import aiohttp
import aioboto3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
import uvloop

class AsyncProcessor:
    def __init__(self):
        # 使用uvloop提升性能
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        # 线程池用于I/O密集型任务
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        
        # 进程池用于CPU密集型任务
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
        # HTTP会话池
        self.http_session = None
    
    async def __aenter__(self):
        # 创建HTTP会话
        connector = aiohttp.TCPConnector(
            limit=100,  # 总连接池大小
            limit_per_host=30,  # 每个主机的连接数
            ttl_dns_cache=300,  # DNS缓存时间
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_session:
            await self.http_session.close()
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
    
    async def process_documents_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理文档"""
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(10)
        
        async def process_single_document(doc):
            async with semaphore:
                return await self._process_document(doc)
        
        # 并发处理所有文档
        tasks = [process_single_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常结果
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Document {documents[i]['id']} processing failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def _process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个文档"""
        doc_id = document['id']
        file_path = document['file_path']
        
        try:
            # 并行执行多个处理步骤
            ocr_task = self._run_ocr(file_path)
            thumbnail_task = self._generate_thumbnail(file_path)
            metadata_task = self._extract_metadata(file_path)
            
            # 等待所有任务完成
            ocr_result, thumbnail_url, metadata = await asyncio.gather(
                ocr_task, thumbnail_task, metadata_task
            )
            
            # 更新文档状态
            await self._update_document_status(doc_id, {
                'status': 'completed',
                'ocr_result': ocr_result,
                'thumbnail_url': thumbnail_url,
                'metadata': metadata
            })
            
            return {
                'id': doc_id,
                'status': 'success',
                'ocr_result': ocr_result,
                'thumbnail_url': thumbnail_url,
                'metadata': metadata
            }
            
        except Exception as e:
            await self._update_document_status(doc_id, {
                'status': 'failed',
                'error_message': str(e)
            })
            raise
    
    async def _run_ocr(self, file_path: str) -> Dict[str, Any]:
        """运行OCR识别（CPU密集型任务）"""
        loop = asyncio.get_event_loop()
        
        # 在进程池中执行OCR
        result = await loop.run_in_executor(
            self.process_pool,
            self._ocr_worker,
            file_path
        )
        
        return result
    
    def _ocr_worker(self, file_path: str) -> Dict[str, Any]:
        """OCR工作进程"""
        import pytesseract
        from PIL import Image
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            
            return {
                'text': text,
                'confidence': 0.95,  # 模拟置信度
                'language': 'chi_sim'
            }
        except Exception as e:
            raise Exception(f"OCR processing failed: {str(e)}")
    
    async def _generate_thumbnail(self, file_path: str) -> str:
        """生成缩略图（I/O密集型任务）"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行缩略图生成
        thumbnail_path = await loop.run_in_executor(
            self.thread_pool,
            self._thumbnail_worker,
            file_path
        )
        
        # 上传到对象存储
        thumbnail_url = await self._upload_to_s3(thumbnail_path)
        
        return thumbnail_url
    
    def _thumbnail_worker(self, file_path: str) -> str:
        """缩略图生成工作线程"""
        from PIL import Image
        import os
        
        try:
            image = Image.open(file_path)
            image.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            thumbnail_path = f"/tmp/thumbnail_{os.path.basename(file_path)}"
            image.save(thumbnail_path, 'JPEG', quality=85)
            
            return thumbnail_path
        except Exception as e:
            raise Exception(f"Thumbnail generation failed: {str(e)}")
    
    async def _upload_to_s3(self, file_path: str) -> str:
        """上传文件到S3"""
        session = aioboto3.Session()
        
        async with session.client('s3') as s3:
            bucket_name = 'document-thumbnails'
            key = f"thumbnails/{os.path.basename(file_path)}"
            
            await s3.upload_file(file_path, bucket_name, key)
            
            return f"https://{bucket_name}.s3.amazonaws.com/{key}"
    
    async def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取文件元数据"""
        loop = asyncio.get_event_loop()
        
        metadata = await loop.run_in_executor(
            self.thread_pool,
            self._metadata_worker,
            file_path
        )
        
        return metadata
    
    def _metadata_worker(self, file_path: str) -> Dict[str, Any]:
        """元数据提取工作线程"""
        import os
        from datetime import datetime
        
        stat = os.stat(file_path)
        
        return {
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'file_type': os.path.splitext(file_path)[1].lower()
        }
    
    async def _update_document_status(self, doc_id: str, update_data: Dict[str, Any]):
        """更新文档状态"""
        # 这里应该调用数据库更新接口
        pass

# 使用示例
async def main():
    documents = [
        {'id': '1', 'file_path': '/path/to/doc1.pdf'},
        {'id': '2', 'file_path': '/path/to/doc2.jpg'},
        # ... 更多文档
    ]
    
    async with AsyncProcessor() as processor:
        results = await processor.process_documents_batch(documents)
        print(f"Processed {len(results)} documents successfully")

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. 扩展性设计

### 4.1 水平扩展架构

```mermaid
graph TB
    subgraph "负载均衡层"
        LB[负载均衡器]
        CDN[CDN]
    end
    
    subgraph "应用层集群"
        APP1[应用实例1]
        APP2[应用实例2]
        APP3[应用实例3]
        APPN[应用实例N]
    end
    
    subgraph "服务层集群"
        AUTH_SVC[认证服务集群]
        DOC_SVC[文档服务集群]
        SEARCH_SVC[搜索服务集群]
        AI_SVC[AI服务集群]
    end
    
    subgraph "数据层集群"
        PG_MASTER[(PostgreSQL主)]
        PG_SLAVE1[(PostgreSQL从1)]
        PG_SLAVE2[(PostgreSQL从2)]
        
        REDIS_CLUSTER[(Redis集群)]
        
        ES_CLUSTER[(Elasticsearch集群)]
        
        MONGO_CLUSTER[(MongoDB集群)]
    end
    
    CDN --> LB
    LB --> APP1
    LB --> APP2
    LB --> APP3
    LB --> APPN
    
    APP1 --> AUTH_SVC
    APP2 --> DOC_SVC
    APP3 --> SEARCH_SVC
    APPN --> AI_SVC
    
    AUTH_SVC --> PG_MASTER
    DOC_SVC --> PG_SLAVE1
    SEARCH_SVC --> ES_CLUSTER
    AI_SVC --> MONGO_CLUSTER
    
    PG_MASTER --> PG_SLAVE1
    PG_MASTER --> PG_SLAVE2
    
    AUTH_SVC --> REDIS_CLUSTER
    DOC_SVC --> REDIS_CLUSTER
```

### 4.2 自动扩缩容配置

```yaml
# kubernetes/hpa.yaml - 水平Pod自动扩缩容
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: document-service-hpa
  namespace: historical-text
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: document-service
  
  minReplicas: 3
  maxReplicas: 20
  
  metrics:
  # CPU使用率
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # 内存使用率
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # 自定义指标 - 请求队列长度
  - type: Pods
    pods:
      metric:
        name: request_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  
  # 扩缩容行为配置
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
    
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min

---
# VPA配置 - 垂直Pod自动扩缩容
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: document-service-vpa
  namespace: historical-text
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: document-service
  
  updatePolicy:
    updateMode: "Auto"
  
  resourcePolicy:
    containerPolicies:
    - containerName: document-service
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 2
        memory: 4Gi
      controlledResources: ["cpu", "memory"]
```

### 4.3 数据库分片策略

```python
# sharding.py - 数据库分片实现
from typing import Dict, List, Any
import hashlib
import asyncpg
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class DatabaseSharding:
    def __init__(self, shard_configs: List[Dict[str, str]]):
        self.shards = {}
        self.shard_count = len(shard_configs)
        
        # 初始化分片连接
        for i, config in enumerate(shard_configs):
            engine = create_engine(
                config['url'],
                pool_size=10,
                max_overflow=20,
                pool_timeout=30
            )
            self.shards[i] = {
                'engine': engine,
                'session': sessionmaker(bind=engine),
                'config': config
            }
    
    def get_shard_key(self, user_id: str) -> int:
        """根据用户ID计算分片键"""
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return hash_value % self.shard_count
    
    def get_shard_session(self, user_id: str):
        """获取用户对应的分片会话"""
        shard_key = self.get_shard_key(user_id)
        return self.shards[shard_key]['session']()
    
    def get_all_shards(self):
        """获取所有分片会话（用于全局查询）"""
        return [shard['session']() for shard in self.shards.values()]
    
    async def execute_on_all_shards(self, query: str, params: Dict[str, Any] = None) -> List[Any]:
        """在所有分片上执行查询"""
        results = []
        
        for shard in self.shards.values():
            async with asyncpg.connect(shard['config']['url']) as conn:
                result = await conn.fetch(query, **(params or {}))
                results.extend(result)
        
        return results
    
    async def migrate_user_data(self, user_id: str, target_shard: int):
        """迁移用户数据到目标分片"""
        current_shard = self.get_shard_key(user_id)
        
        if current_shard == target_shard:
            return  # 无需迁移
        
        # 开始数据迁移
        source_session = self.shards[current_shard]['session']()
        target_session = self.shards[target_shard]['session']()
        
        try:
            # 1. 复制用户数据
            user_data = source_session.query(User).filter_by(id=user_id).first()
            if user_data:
                target_session.merge(user_data)
            
            # 2. 复制用户文档
            documents = source_session.query(Document).filter_by(owner_id=user_id).all()
            for doc in documents:
                target_session.merge(doc)
            
            # 3. 提交目标分片
            target_session.commit()
            
            # 4. 删除源分片数据
            source_session.query(Document).filter_by(owner_id=user_id).delete()
            source_session.query(User).filter_by(id=user_id).delete()
            source_session.commit()
            
        except Exception as e:
            source_session.rollback()
            target_session.rollback()
            raise e
        finally:
            source_session.close()
            target_session.close()

# 分片配置
shard_configs = [
    {'url': 'postgresql://user:pass@shard1:5432/db', 'name': 'shard1'},
    {'url': 'postgresql://user:pass@shard2:5432/db', 'name': 'shard2'},
    {'url': 'postgresql://user:pass@shard3:5432/db', 'name': 'shard3'},
    {'url': 'postgresql://user:pass@shard4:5432/db', 'name': 'shard4'}
]

db_sharding = DatabaseSharding(shard_configs)

# 分片路由中间件
class ShardingMiddleware:
    def __init__(self, app, db_sharding):
        self.app = app
        self.db_sharding = db_sharding
    
    def __call__(self, environ, start_response):
        # 从请求中提取用户ID
        user_id = self.extract_user_id(environ)
        
        if user_id:
            # 设置当前请求的分片会话
            environ['db_session'] = self.db_sharding.get_shard_session(user_id)
        
        return self.app(environ, start_response)
    
    def extract_user_id(self, environ) -> str:
        # 从JWT token或session中提取用户ID
        # 实现具体的用户ID提取逻辑
        pass
```

---

**文档版本**: v1.0  
**最后更新**: 2024-01-01  
**维护者**: 架构团队