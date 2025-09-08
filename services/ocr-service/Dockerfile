# OCR服务Docker镜像
# 无状态OCR文本识别微服务，专注于图像文本识别算法
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
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
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

# 第二阶段：运行时镜像
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y \
    # OCR相关依赖
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-eng \
    # 图像处理依赖
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # 网络工具
    curl \
    # 清理
    && rm -rf /var/lib/apt/lists/*

# 从builder阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 创建应用用户
RUN groupadd -r ocr && useradd -r -g ocr ocr

# 创建应用目录
WORKDIR /app

# 创建必要的目录（无状态架构，仅需临时目录）
RUN mkdir -p /app/temp \
    && chown -R ocr:ocr /app

# 复制应用代码
COPY --chown=ocr:ocr . /app/

# 预下载OCR模型以提高启动速度（可选）
USER ocr
RUN python -c "import paddleocr; paddleocr.PaddleOCR(use_angle_cls=True, lang='ch')" || true
RUN python -c "import easyocr; easyocr.Reader(['ch_sim', 'en'])" || true

# 设置工作目录权限
USER root
RUN chown -R ocr:ocr /app

# 切换到应用用户
USER ocr

# 暴露端口（更新为8003，避免与storage-service冲突）
EXPOSE 8003

# 环境变量
ENV OCR_API_PORT=8003 \
    OCR_TEMP_DIR=/app/temp \
    OCR_ENVIRONMENT=production

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# 启动命令
CMD ["python", "-m", "src.main"]