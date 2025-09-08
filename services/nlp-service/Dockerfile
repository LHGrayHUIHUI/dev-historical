# NLP服务Docker镜像
# 无状态NLP文本处理微服务，专注于自然语言处理算法
# 数据存储通过storage-service完成

# 使用多阶段构建优化镜像大小
FROM python:3.11-slim as builder

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖和构建工具
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # NLP和机器学习依赖
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # 中文字体支持
    fonts-noto-cjk \
    # 清理
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip和安装基础工具
RUN pip install --upgrade pip setuptools wheel

# 复制requirements文件
COPY requirements.txt /tmp/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 下载spaCy中文模型
RUN python -m spacy download zh_core_web_sm || echo "spaCy中文模型下载失败，将在运行时处理"

# 第二阶段：运行时镜像
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y \
    # NLP和机器学习依赖
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # 中文字体支持
    fonts-noto-cjk \
    # 网络工具
    curl \
    # 清理
    && rm -rf /var/lib/apt/lists/*

# 从builder阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 创建应用用户
RUN groupadd -r nlp && useradd -r -g nlp nlp

# 创建应用目录
WORKDIR /app

# 创建必要的目录（无状态架构，仅需临时目录和模型目录）
RUN mkdir -p /app/temp \
             /app/models \
             /app/dictionaries \
    && chown -R nlp:nlp /app

# 复制应用代码
COPY --chown=nlp:nlp . /app/

# 预下载NLP模型以提高启动速度（可选）
USER nlp

# 尝试预下载模型（失败不影响构建）
RUN python -c "import spacy; spacy.load('zh_core_web_sm')" || echo "spaCy模型将在运行时加载"
RUN python -c "import jieba; jieba.lcut('测试')" || echo "jieba初始化"
RUN python -c "from transformers import pipeline; print('transformers ready')" || echo "transformers将在运行时加载"

# 设置工作目录权限
USER root
RUN chown -R nlp:nlp /app

# 切换到应用用户
USER nlp

# 暴露端口（使用8004端口，避免与其他服务冲突）
EXPOSE 8004

# 环境变量
ENV NLP_SERVICE_API_PORT=8004 \
    NLP_SERVICE_TEMP_DIR=/app/temp \
    NLP_SERVICE_MODELS_DIR=/app/models \
    NLP_SERVICE_CUSTOM_DICT_DIR=/app/dictionaries \
    NLP_SERVICE_ENVIRONMENT=production

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# 启动命令
CMD ["python", "-m", "src.main"]