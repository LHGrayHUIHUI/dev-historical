# 🐳 Docker部署配置指南

## 📋 概述

历史文本项目采用微服务架构，所有服务都基于Docker容器化部署。本文档详细说明了Docker Hub集成、镜像构建、推送和部署的完整流程。

---

## 🔐 Docker Hub账户配置

### 1. Docker Hub账户设置

#### 创建Docker Hub账户
1. 访问 [Docker Hub](https://hub.docker.com/)
2. 创建账户或登录现有账户
3. 建议使用项目专用账户：`historical-text-project`

#### 访问令牌配置
```bash
# 登录Docker Hub
docker login

# 或使用访问令牌登录（推荐）
docker login -u <username> -p <access_token>
```

### 2. 环境变量配置

创建 `.env.docker` 文件：
```bash
# Docker Hub配置
DOCKER_HUB_USERNAME=historical-text-project
DOCKER_HUB_PASSWORD=your_access_token_here
DOCKER_HUB_REGISTRY=docker.io

# 镜像标签配置
PROJECT_NAME=historical-text
VERSION_TAG=latest
BUILD_DATE=$(date +%Y%m%d)

# 镜像仓库前缀
IMAGE_PREFIX=${DOCKER_HUB_USERNAME}/${PROJECT_NAME}
```

---

## 🏗️ 微服务Docker化配置

### 微服务镜像构建配置

#### 1. 数据源服务 (Data Source Service)
```dockerfile
# services/data-source/Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY .env.example .env

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "src.main"]
```

#### 2. 基础设施服务配置
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # 数据源服务
  data-source-service:
    image: ${IMAGE_PREFIX}-data-source:${VERSION_TAG}
    build:
      context: ./services/data-source
      dockerfile: Dockerfile
      args:
        - BUILD_DATE=${BUILD_DATE}
        - VERSION=${VERSION_TAG}
    container_name: historical-text-data-source
    restart: unless-stopped
    environment:
      - SERVICE_ENVIRONMENT=production
      - SERVICE_HOST=0.0.0.0
      - SERVICE_PORT=8000
    ports:
      - "8000:8000"
    networks:
      - historical-text-network
    depends_on:
      - mongodb
      - redis
    volumes:
      - ./logs/data-source:/app/logs
    labels:
      - "com.historical-text.service=data-source"
      - "com.historical-text.version=${VERSION_TAG}"

  # MongoDB数据库
  mongodb:
    image: mongo:5.0
    container_name: historical-text-mongodb
    restart: unless-stopped
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_ROOT_USER}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_ROOT_PASSWORD}
      - MONGO_INITDB_DATABASE=historical_text_prod
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
      - ./docker/mongodb/init.js:/docker-entrypoint-initdb.d/init.js:ro
    networks:
      - historical-text-network

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: historical-text-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - historical-text-network

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: historical-text-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - historical-text-network
    depends_on:
      - data-source-service

volumes:
  mongodb_data:
  redis_data:

networks:
  historical-text-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## 🚀 自动化构建和推送脚本

### 1. 镜像构建脚本

创建 `scripts/build-images.sh`：
```bash
#!/bin/bash

# 历史文本项目 - Docker镜像构建脚本
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 加载环境变量
if [ -f .env.docker ]; then
    source .env.docker
    log_info "已加载Docker环境配置"
else
    log_error ".env.docker 文件不存在"
    exit 1
fi

# 构建参数
BUILD_DATE=$(date +%Y%m%d-%H%M%S)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION_TAG=${VERSION_TAG:-latest}

log_info "开始构建历史文本项目Docker镜像..."
log_info "构建时间: $BUILD_DATE"
log_info "Git提交: $GIT_COMMIT"
log_info "版本标签: $VERSION_TAG"

# 微服务列表
SERVICES=(
    "data-source:services/data-source"
    "data-storage:services/data-storage"
    "data-processing:services/data-processing"
    "ai-model:services/ai-model"
    "text-optimization:services/text-optimization"
    "content-publishing:services/content-publishing"
    "customer-messaging:services/customer-messaging"
)

# 构建镜像函数
build_service() {
    local service_name=$1
    local service_path=$2
    local image_name="${IMAGE_PREFIX}-${service_name}"
    
    log_info "构建服务: $service_name"
    log_info "服务路径: $service_path"
    log_info "镜像名称: $image_name:$VERSION_TAG"
    
    if [ ! -d "$service_path" ]; then
        log_warning "服务目录不存在，跳过: $service_path"
        return 0
    fi
    
    # 检查Dockerfile
    if [ ! -f "$service_path/Dockerfile" ]; then
        log_warning "Dockerfile不存在，跳过: $service_path/Dockerfile"
        return 0
    fi
    
    # 构建镜像
    docker build \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        --build-arg VERSION="$VERSION_TAG" \
        -t "$image_name:$VERSION_TAG" \
        -t "$image_name:latest" \
        -f "$service_path/Dockerfile" \
        "$service_path"
    
    if [ $? -eq 0 ]; then
        log_success "✅ $service_name 构建成功"
        
        # 显示镜像信息
        docker images "$image_name:$VERSION_TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    else
        log_error "❌ $service_name 构建失败"
        return 1
    fi
}

# 构建所有服务
log_info "开始构建微服务镜像..."
for service in "${SERVICES[@]}"; do
    IFS=':' read -r service_name service_path <<< "$service"
    build_service "$service_name" "$service_path"
done

# 构建前端应用
if [ -d "frontend" ]; then
    log_info "构建前端应用..."
    docker build \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION_TAG" \
        -t "${IMAGE_PREFIX}-frontend:$VERSION_TAG" \
        -t "${IMAGE_PREFIX}-frontend:latest" \
        frontend/
    
    if [ $? -eq 0 ]; then
        log_success "✅ 前端应用构建成功"
    else
        log_error "❌ 前端应用构建失败"
    fi
fi

# 构建完成统计
log_info "📊 构建完成统计:"
echo "----------------------------------------"
docker images "${IMAGE_PREFIX}-*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -20

log_success "🎉 所有镜像构建完成！"
log_info "下一步: 运行 './scripts/push-images.sh' 推送镜像到Docker Hub"
```

### 2. 镜像推送脚本

创建 `scripts/push-images.sh`：
```bash
#!/bin/bash

# 历史文本项目 - Docker镜像推送脚本
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 加载环境变量
if [ -f .env.docker ]; then
    source .env.docker
else
    log_error ".env.docker 文件不存在"
    exit 1
fi

# 登录Docker Hub
log_info "登录Docker Hub..."
if [ -n "$DOCKER_HUB_PASSWORD" ]; then
    echo "$DOCKER_HUB_PASSWORD" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
else
    docker login -u "$DOCKER_HUB_USERNAME"
fi

if [ $? -ne 0 ]; then
    log_error "Docker Hub登录失败"
    exit 1
fi

log_success "Docker Hub登录成功"

# 推送函数
push_image() {
    local image_name=$1
    local tag=$2
    
    log_info "推送镜像: $image_name:$tag"
    
    docker push "$image_name:$tag"
    
    if [ $? -eq 0 ]; then
        log_success "✅ $image_name:$tag 推送成功"
    else
        log_error "❌ $image_name:$tag 推送失败"
        return 1
    fi
}

# 获取所有本地构建的镜像
IMAGES=$(docker images "${IMAGE_PREFIX}-*" --format "{{.Repository}}")

if [ -z "$IMAGES" ]; then
    log_error "未找到需要推送的镜像"
    log_info "请先运行 './scripts/build-images.sh' 构建镜像"
    exit 1
fi

# 推送所有镜像
log_info "开始推送镜像到Docker Hub..."
for image in $IMAGES; do
    # 推送版本标签
    push_image "$image" "$VERSION_TAG"
    
    # 推送latest标签
    push_image "$image" "latest"
done

log_success "🎉 所有镜像推送完成！"

# 显示推送的镜像列表
log_info "📊 已推送的镜像:"
echo "----------------------------------------"
for image in $IMAGES; do
    echo "🐳 $image:$VERSION_TAG"
    echo "🐳 $image:latest"
done
```

### 3. 一键部署脚本

创建 `scripts/deploy.sh`：
```bash
#!/bin/bash

# 历史文本项目 - 一键部署脚本
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 参数解析
ENVIRONMENT=${1:-development}
ACTION=${2:-up}

log_info "历史文本项目一键部署"
log_info "环境: $ENVIRONMENT"
log_info "操作: $ACTION"

# 环境配置文件
case $ENVIRONMENT in
    "development")
        COMPOSE_FILE="docker-compose.yml"
        ;;
    "production")
        COMPOSE_FILE="docker-compose.production.yml"
        ;;
    "testing")
        COMPOSE_FILE="docker-compose.test.yml"
        ;;
    *)
        log_error "未知环境: $ENVIRONMENT"
        echo "支持的环境: development, production, testing"
        exit 1
        ;;
esac

# 检查Docker Compose文件
if [ ! -f "$COMPOSE_FILE" ]; then
    log_error "配置文件不存在: $COMPOSE_FILE"
    exit 1
fi

# 加载环境变量
ENV_FILE=".env.${ENVIRONMENT}"
if [ -f "$ENV_FILE" ]; then
    export $(cat "$ENV_FILE" | grep -v '#' | xargs)
    log_info "已加载环境配置: $ENV_FILE"
fi

# 执行部署操作
case $ACTION in
    "up")
        log_info "启动服务..."
        docker-compose -f "$COMPOSE_FILE" up -d
        ;;
    "down")
        log_info "停止服务..."
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    "restart")
        log_info "重启服务..."
        docker-compose -f "$COMPOSE_FILE" restart
        ;;
    "logs")
        log_info "查看日志..."
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    "status")
        log_info "服务状态..."
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "build")
        log_info "构建并启动..."
        docker-compose -f "$COMPOSE_FILE" up -d --build
        ;;
    *)
        log_error "未知操作: $ACTION"
        echo "支持的操作: up, down, restart, logs, status, build"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    log_success "✅ 操作完成: $ACTION"
else
    log_error "❌ 操作失败: $ACTION"
    exit 1
fi

# 显示服务状态
if [ "$ACTION" = "up" ] || [ "$ACTION" = "build" ]; then
    sleep 5
    log_info "📊 服务状态:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    log_info "🔗 服务访问地址:"
    echo "----------------------------------------"
    echo "🌐 数据源服务: http://localhost:8000"
    echo "📚 API文档: http://localhost:8000/docs"
    echo "💾 MongoDB: mongodb://localhost:27017"
    echo "🔴 Redis: redis://localhost:6379"
    echo "📊 监控面板: http://localhost:3000"
fi
```

---

## 🔧 Docker Compose配置

### 1. 开发环境配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  data-source-service:
    build:
      context: ./services/data-source
      dockerfile: Dockerfile
    container_name: historical-text-data-source-dev
    restart: unless-stopped
    environment:
      - SERVICE_ENVIRONMENT=development
      - SERVICE_HOST=0.0.0.0
      - SERVICE_PORT=8000
      - DB_MONGODB_URL=mongodb://mongodb:27017
      - DB_REDIS_URL=redis://redis:6379
    ports:
      - "8000:8000"
    volumes:
      - ./services/data-source/src:/app/src
      - ./logs:/app/logs
    networks:
      - historical-text-network
    depends_on:
      - mongodb
      - redis

  mongodb:
    image: mongo:5.0
    container_name: historical-text-mongodb-dev
    restart: unless-stopped
    ports:
      - "27017:27017"
    volumes:
      - mongodb_dev_data:/data/db
    networks:
      - historical-text-network

  redis:
    image: redis:7-alpine
    container_name: historical-text-redis-dev
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
    networks:
      - historical-text-network

volumes:
  mongodb_dev_data:
  redis_dev_data:

networks:
  historical-text-network:
    driver: bridge
```

### 2. 测试环境配置
```yaml
# docker-compose.test.yml
version: '3.8'

services:
  data-source-service-test:
    build:
      context: ./services/data-source
      dockerfile: Dockerfile
      target: test
    container_name: historical-text-data-source-test
    environment:
      - SERVICE_ENVIRONMENT=testing
      - DB_MONGODB_URL=mongodb://mongodb-test:27017
      - DB_REDIS_URL=redis://redis-test:6379
    networks:
      - historical-text-test-network
    depends_on:
      - mongodb-test
      - redis-test
    command: pytest tests/ -v --cov=src

  mongodb-test:
    image: mongo:5.0
    container_name: historical-text-mongodb-test
    tmpfs:
      - /data/db
    networks:
      - historical-text-test-network

  redis-test:
    image: redis:7-alpine
    container_name: historical-text-redis-test
    tmpfs:
      - /data
    networks:
      - historical-text-test-network

networks:
  historical-text-test-network:
    driver: bridge
```

---

## 📦 镜像管理最佳实践

### 1. 镜像标签策略
```bash
# 版本标签格式
${IMAGE_PREFIX}-${SERVICE}:${VERSION}

# 示例
historical-text-project/historical-text-data-source:v1.0.0
historical-text-project/historical-text-data-source:latest
historical-text-project/historical-text-data-source:20250903
```

### 2. 镜像优化
```dockerfile
# 多阶段构建示例
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim as runner
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/

# 添加到PATH
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "-m", "src.main"]
```

### 3. 镜像安全扫描
```bash
# 使用Docker Scout扫描
docker scout quickview ${IMAGE_NAME}:${TAG}

# 使用Trivy扫描
trivy image ${IMAGE_NAME}:${TAG}
```

---

## 🚀 快速开始

### 1. 初始化配置
```bash
# 克隆项目
git clone <repository-url>
cd "Historical Text Project"

# 复制环境配置
cp .env.docker.example .env.docker
# 编辑配置文件，设置Docker Hub账户信息

# 赋予脚本执行权限
chmod +x scripts/*.sh
```

### 2. 构建和推送
```bash
# 构建所有镜像
./scripts/build-images.sh

# 推送到Docker Hub
./scripts/push-images.sh
```

### 3. 部署运行
```bash
# 开发环境
./scripts/deploy.sh development up

# 生产环境
./scripts/deploy.sh production up

# 查看状态
./scripts/deploy.sh development status
```

---

## 📊 监控和维护

### 1. 服务健康检查
```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f data-source-service

# 健康检查
curl http://localhost:8000/health
```

### 2. 镜像清理
```bash
# 清理未使用的镜像
docker image prune -f

# 清理所有未使用的资源
docker system prune -af
```

### 3. 备份和恢复
```bash
# 备份数据卷
docker run --rm -v mongodb_data:/data -v $(pwd):/backup busybox tar czf /backup/mongodb_backup.tar.gz /data

# 恢复数据卷
docker run --rm -v mongodb_data:/data -v $(pwd):/backup busybox tar xzf /backup/mongodb_backup.tar.gz -C /
```

---

## 🔍 故障排除

### 常见问题和解决方案

1. **镜像推送失败**
   ```bash
   # 检查登录状态
   docker info | grep Username
   
   # 重新登录
   docker login
   ```

2. **容器启动失败**
   ```bash
   # 查看详细错误
   docker-compose logs <service-name>
   
   # 检查端口占用
   netstat -tulpn | grep <port>
   ```

3. **网络连接问题**
   ```bash
   # 检查网络
   docker network ls
   docker network inspect <network-name>
   ```

---

## 📚 相关文档

- [Docker官方文档](https://docs.docker.com/)
- [Docker Compose文档](https://docs.docker.com/compose/)
- [Docker Hub使用指南](https://docs.docker.com/docker-hub/)
- [项目架构文档](docs/architecture/)

---

*最后更新: 2025-09-03*