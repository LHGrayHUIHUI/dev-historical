# 知识图谱构建服务 Docker 镜像
# 基于Python 3.11的多阶段构建，优化镜像大小和安全性

# ============ 构建阶段 ============
FROM python:3.11-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 创建虚拟环境并安装Python依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip并安装依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============ 运行阶段 ============
FROM python:3.11-slim as runtime

# 设置标签
LABEL maintainer="Historical Text Project Team"
LABEL description="Knowledge Graph Construction Service for Historical Texts"
LABEL version="1.0.0"

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 创建必要的目录
RUN mkdir -p /app/logs /app/temp /app/cache && \
    chown -R appuser:appuser /app

# 复制应用代码
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser *.txt ./

# 下载spaCy中文模型
RUN python -m spacy download zh_core_web_sm || echo "spaCy模型下载失败，运行时需要手动下载"

# 设置环境变量
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 服务配置
ENV SERVICE_NAME="knowledge-graph-service"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8006"
ENV ENVIRONMENT="production"
ENV LOG_LEVEL="INFO"

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8006/health || exit 1

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8006

# 启动命令
CMD ["python", "-m", "src.main"]