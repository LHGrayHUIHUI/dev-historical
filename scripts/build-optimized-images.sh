#!/bin/bash

# 历史文本项目 - 优化Docker镜像构建脚本
# 使用优化的Dockerfile快速构建生产级镜像

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 历史文本项目 - 优化镜像构建${NC}"
echo "=================================================="

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 加载环境变量
if [ -f ".env.docker" ]; then
    source .env.docker
    echo -e "${GREEN}✅ Docker配置加载成功${NC}"
else
    echo -e "${RED}❌ 未找到.env.docker配置文件${NC}"
    exit 1
fi

# 设置构建参数
BUILD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VERSION_TAG=${VERSION_TAG:-"v1.0"}
REGISTRY_PREFIX="historicaltext/historical-text"

echo -e "${BLUE}📋 构建配置:${NC}"
echo "  项目名称: $PROJECT_NAME"
echo "  版本标签: $VERSION_TAG"
echo "  构建时间: $BUILD_TIMESTAMP"
echo "  Registry: $REGISTRY_PREFIX"
echo ""

# 构建函数
build_optimized_service() {
    local service_name=$1
    local service_path=$2
    
    echo -e "${YELLOW}🔨 构建优化镜像: $service_name${NC}"
    
    if [ ! -d "$service_path" ]; then
        echo -e "${RED}❌ 服务目录不存在: $service_path${NC}"
        return 1
    fi
    
    if [ ! -f "$service_path/Dockerfile.optimized" ]; then
        echo -e "${RED}❌ 优化Dockerfile不存在: $service_path/Dockerfile.optimized${NC}"
        return 1
    fi
    
    # 构建生产镜像
    docker build \
        --file "$service_path/Dockerfile.optimized" \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION_TAG" \
        --build-arg GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --tag "$REGISTRY_PREFIX-$service_name:$VERSION_TAG" \
        --tag "$REGISTRY_PREFIX-$service_name:latest" \
        "$service_path"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $service_name 生产镜像构建成功${NC}"
        
        # 构建开发镜像
        docker build \
            --file "$service_path/Dockerfile.optimized" \
            --target development \
            --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
            --build-arg VERSION="$VERSION_TAG-dev" \
            --tag "$REGISTRY_PREFIX-$service_name:$VERSION_TAG-dev" \
            --tag "$REGISTRY_PREFIX-$service_name:dev" \
            "$service_path"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ $service_name 开发镜像构建成功${NC}"
        else
            echo -e "${YELLOW}⚠️  $service_name 开发镜像构建失败${NC}"
        fi
    else
        echo -e "${RED}❌ $service_name 生产镜像构建失败${NC}"
        return 1
    fi
}

# 构建数据源服务
echo -e "${BLUE}🏗️  开始构建微服务镜像...${NC}"
build_optimized_service "data-source" "services/data-source"

# 构建数据采集服务
build_optimized_service "data-collection" "services/data-collection"

# 显示构建结果
echo ""
echo -e "${BLUE}📊 构建完成的镜像:${NC}"
echo "----------------------------------------"
docker images "$REGISTRY_PREFIX-*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -15

# 计算镜像大小
echo ""
echo -e "${BLUE}📏 镜像大小对比:${NC}"
echo "----------------------------------------"
for image in data-source data-collection; do
    if docker image inspect "$REGISTRY_PREFIX-$image:latest" > /dev/null 2>&1; then
        size=$(docker images "$REGISTRY_PREFIX-$image:latest" --format "{{.Size}}")
        echo "  $image (生产): $size"
    fi
    if docker image inspect "$REGISTRY_PREFIX-$image:dev" > /dev/null 2>&1; then
        dev_size=$(docker images "$REGISTRY_PREFIX-$image:dev" --format "{{.Size}}")
        echo "  $image (开发): $dev_size"
    fi
done

echo ""
echo -e "${GREEN}🎉 优化镜像构建完成!${NC}"
echo -e "${BLUE}💡 下一步: 运行 ./scripts/push-optimized-images.sh 推送到Docker Hub${NC}"
echo -e "${BLUE}🧪 或运行: ./scripts/test-optimized-images.sh 本地测试${NC}"