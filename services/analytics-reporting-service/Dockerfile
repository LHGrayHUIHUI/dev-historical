# 分析报告服务多阶段Docker构建

# 第一阶段：基础环境
FROM python:3.11-slim as base

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 第二阶段：依赖安装
FROM base as dependencies

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# 第三阶段：开发环境
FROM dependencies as development

# 安装开发依赖
RUN pip install pytest pytest-cov pytest-asyncio black isort flake8 mypy

# 复制源代码
COPY . .

# 创建必要目录
RUN mkdir -p logs reports models temp

# 暴露端口
EXPOSE 8099

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8099/health || exit 1

# 开发环境启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8099", "--reload"]

# 第四阶段：生产环境
FROM dependencies as production

# 创建非root用户
RUN groupadd -r analytics && useradd -r -g analytics analytics

# 设置工作目录
WORKDIR /app

# 复制源代码（排除开发文件）
COPY --chown=analytics:analytics src/ src/
COPY --chown=analytics:analytics requirements.txt .
COPY --chown=analytics:analytics README.md .

# 创建必要目录并设置权限
RUN mkdir -p logs reports models temp && \
    chown -R analytics:analytics /app

# 切换到非root用户
USER analytics

# 暴露端口
EXPOSE 8099

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8099/health || exit 1

# 生产环境启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "4", "--log-level", "info"]