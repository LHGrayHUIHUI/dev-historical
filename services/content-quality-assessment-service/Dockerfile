# Content Quality Assessment Service Dockerfile
# 内容质量评估服务容器配置

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 下载spaCy中文模型
RUN python -m spacy download zh_core_web_sm

# 初始化jieba分词
RUN python -c "import jieba; jieba.initialize()"

# 复制应用代码
COPY src/ ./src/

# 创建必要的目录
RUN mkdir -p /app/logs /app/temp /app/models

# 创建非root用户
RUN useradd --create-home --shell /bin/bash assessor && \
    chown -R assessor:assessor /app
USER assessor

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8012/health || exit 1

# 暴露端口
EXPOSE 8012

# 启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8012"]