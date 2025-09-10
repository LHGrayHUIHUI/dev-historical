# 智能分类服务Dockerfile
# 基于Python 3.11的多阶段构建，优化镜像大小和安全性

# 构建阶段
FROM python:3.11-slim-bookworm as builder

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（构建时需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /build

# 复制依赖文件
COPY requirements.txt .

# 创建虚拟环境并安装Python依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip并安装依赖
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# 运行阶段
FROM python:3.11-slim-bookworm as runtime

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# 创建非root用户
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置工作目录
WORKDIR /app

# 创建必要的目录并设置权限
RUN mkdir -p /app/logs /tmp/intelligent-classification /tmp/models /tmp/cache && \
    chown -R appuser:appgroup /app /tmp/intelligent-classification /tmp/models /tmp/cache

# 复制应用代码
COPY --chown=appuser:appgroup . .

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# 暴露端口
EXPOSE 8007

# 设置启动命令
CMD ["python", "-m", "src.main"]