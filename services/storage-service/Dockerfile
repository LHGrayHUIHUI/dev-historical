# 数据采集服务Docker镜像 - 多阶段构建
# 第一阶段：构建阶段
FROM python:3.11-slim as builder

# 构建参数
ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION=latest

# 设置标签
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$GIT_COMMIT \
      org.opencontainers.image.title="Historical Text Data Collection Service" \
      org.opencontainers.image.description="数据采集与存储微服务" \
      org.opencontainers.image.vendor="Historical Text Project"

# 设置工作目录
WORKDIR /build

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    libmagic1 \
    libmagic-dev \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖到用户目录
RUN pip install --user --no-cache-dir -r requirements.txt

# 第二阶段：运行阶段
FROM python:3.11-slim as runner

# 从构建阶段复制依赖
COPY --from=builder /root/.local /root/.local

# 设置环境变量
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8002

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    libmagic1 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建应用目录和用户
RUN useradd --create-home --shell /bin/bash --uid 1001 app \
    && mkdir -p /app/logs /app/data /app/temp \
    && chown -R app:app /app

# 设置工作目录
WORKDIR /app

# 切换到非root用户
USER app

# 复制应用代码
COPY --chown=app:app src/ ./src/
COPY --chown=app:app alembic.ini .
COPY --chown=app:app migrations/ ./migrations/
COPY --chown=app:app .env.example .env

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# 暴露端口
EXPOSE ${SERVICE_PORT}

# 启动命令
CMD ["python", "-m", "src.main"]

# 测试阶段（可选）
FROM runner as test

# 切换回root安装测试依赖
USER root

# 安装测试依赖
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    pytest-mock==3.12.0 \
    httpx==0.25.2

# 复制测试文件
COPY --chown=app:app tests/ ./tests/

# 切换回app用户
USER app

# 运行测试
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]