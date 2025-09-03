#!/bin/bash

# 历史文本项目 - Docker Hub镜像验证脚本
# 用于验证镜像是否成功上传到Docker Hub

set -e

echo "🐳 验证Docker Hub镜像上传状态"
echo "=================================="

# 镜像信息
REGISTRY="lhgray/historical-projects"
IMAGES=("data-source-latest" "data-collection-latest")

echo "📊 本地镜像状态:"
echo "------------------------"
docker images "$REGISTRY" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
echo "🔍 验证Docker Hub镜像可用性:"
echo "------------------------"

for image in "${IMAGES[@]}"; do
    echo "验证镜像: $REGISTRY:$image"
    
    # 删除本地镜像（如果存在）
    if docker image inspect "$REGISTRY:$image" >/dev/null 2>&1; then
        echo "  删除本地镜像..."
        docker rmi "$REGISTRY:$image" >/dev/null 2>&1
    fi
    
    # 从Docker Hub拉取镜像
    echo "  从Docker Hub拉取..."
    if docker pull "$REGISTRY:$image" >/dev/null 2>&1; then
        echo "  ✅ $image 拉取成功"
        
        # 获取镜像大小
        size=$(docker images "$REGISTRY:$image" --format "{{.Size}}")
        echo "  📏 镜像大小: $size"
    else
        echo "  ❌ $image 拉取失败"
    fi
    echo ""
done

echo "🎉 验证完成！"
echo ""
echo "📝 使用方法:"
echo "  docker pull $REGISTRY:data-source-latest"
echo "  docker pull $REGISTRY:data-collection-latest"