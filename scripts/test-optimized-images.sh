#!/bin/bash

# 历史文本项目 - 优化镜像本地测试脚本
# 使用本地构建的优化镜像进行集成测试

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🧪 历史文本项目 - 优化镜像集成测试${NC}"
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

# 创建临时的优化测试compose文件
TEMP_COMPOSE="docker-compose.optimized-test.yml"
cat > "$TEMP_COMPOSE" << 'EOF'
version: '3.8'

services:
  # 基础设施服务
  postgres:
    image: postgres:15-alpine
    container_name: historical_postgres_test
    environment:
      POSTGRES_DB: historical_text_test
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin123
      POSTGRES_HOST_AUTH_METHOD: md5
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d historical_text_test"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - historical-net

  redis:
    image: redis:7-alpine
    container_name: historical_redis_test
    ports:
      - "6380:6379"
    command: redis-server --requirepass redis123
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - historical-net

  rabbitmq:
    image: rabbitmq:3-management
    container_name: historical_rabbitmq_test
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin123
    ports:
      - "5673:5672"
      - "15673:15672"
    healthcheck:
      test: rabbitmq-diagnostics -q ping
      interval: 30s
      timeout: 10s
      retries: 5
    networks:
      - historical-net

  minio:
    image: minio/minio:latest
    container_name: historical_minio_test
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: admin123456
    ports:
      - "9001:9000"
      - "9002:9001"
    command: server /data --console-address ":9001"
    volumes:
      - minio_test_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - historical-net

  # 优化的微服务
  data-source:
    image: historicaltext/historical-text-data-source:latest
    container_name: historical_data_source_optimized
    environment:
      - DATABASE_URL=postgresql://admin:admin123@postgres:5432/historical_text_test
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://admin:admin123@rabbitmq:5672/
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=admin
      - MINIO_SECRET_KEY=admin123456
      - LOG_LEVEL=INFO
    ports:
      - "8001:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    networks:
      - historical-net

  data-collection:
    image: historicaltext/historical-text-data-collection:latest
    container_name: historical_data_collection_optimized
    environment:
      - DATABASE_URL=postgresql://admin:admin123@postgres:5432/historical_text_test
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://admin:admin123@rabbitmq:5672/
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=admin
      - MINIO_SECRET_KEY=admin123456
      - DATA_SOURCE_URL=http://data-source:8000
      - LOG_LEVEL=INFO
    ports:
      - "8003:8002"
    depends_on:
      data-source:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    networks:
      - historical-net

volumes:
  postgres_test_data:
  minio_test_data:

networks:
  historical-net:
    driver: bridge
EOF

echo -e "${BLUE}📋 测试配置:${NC}"
echo "  优化镜像版本: latest"
echo "  数据库端口: 5433"
echo "  Redis端口: 6380"
echo "  RabbitMQ端口: 5673"
echo "  MinIO端口: 9001/9002"
echo "  数据源服务: 8001"
echo "  数据采集服务: 8003"
echo ""

# 清理现有容器
echo -e "${YELLOW}🧹 清理现有测试容器...${NC}"
docker-compose -f "$TEMP_COMPOSE" down --remove-orphans --volumes 2>/dev/null || true

# 启动优化版本集成测试环境
echo -e "${YELLOW}🚀 启动优化版本集成测试环境...${NC}"
docker-compose -f "$TEMP_COMPOSE" up -d

# 等待服务启动
echo -e "${YELLOW}⏱️  等待服务完全启动...${NC}"
sleep 30

# 检查服务状态
echo -e "${BLUE}📊 检查服务状态:${NC}"
docker-compose -f "$TEMP_COMPOSE" ps

echo ""
echo -e "${YELLOW}🔍 运行集成测试...${NC}"

# 运行集成测试
if [ -f "tests/integration_test_runner.py" ]; then
    python3 tests/integration_test_runner.py
else
    echo -e "${RED}❌ 集成测试脚本不存在${NC}"
    exit 1
fi

# 显示容器资源使用情况
echo ""
echo -e "${BLUE}📈 容器资源使用情况:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
    $(docker-compose -f "$TEMP_COMPOSE" ps -q)

echo ""
echo -e "${GREEN}🎉 优化镜像测试完成!${NC}"
echo -e "${BLUE}💡 查看详细日志: docker-compose -f $TEMP_COMPOSE logs${NC}"
echo -e "${BLUE}🛑 停止测试环境: docker-compose -f $TEMP_COMPOSE down${NC}"

# 清理临时文件
# rm -f "$TEMP_COMPOSE"