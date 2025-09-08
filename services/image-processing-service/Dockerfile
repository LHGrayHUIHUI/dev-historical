# 图像处理服务Docker配置
# 使用Python 3.11基础镜像，包含系统级图像处理库
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=Asia/Shanghai

# 安装系统依赖 - 图像处理所需的系统库
RUN apt-get update && apt-get install -y \
    # OpenCV依赖
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 图像处理库
    libimage-exiftool-perl \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # 构建工具
    gcc \
    g++ \
    make \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY *.py ./

# 创建必要的目录并设置权限
RUN mkdir -p /app/logs /app/temp /app/uploads && \
    chown -R appuser:appuser /app

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8005/health')" || exit 1

# 暴露端口
EXPOSE 8005

# 启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8005"]