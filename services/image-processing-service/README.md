# 图像处理服务 (Image Processing Service)

历史文本项目的图像处理微服务，专注于历史文档图像的增强、去噪、倾斜校正、尺寸调整和质量评估。

## 🚀 功能特性

### 核心图像处理能力
- **图像增强** - 亮度、对比度、锐度优化
- **去噪处理** - 多种算法去除图像噪声
- **倾斜校正** - 自动检测并校正文档倾斜
- **尺寸调整** - 支持多种插值算法的图像缩放
- **格式转换** - 支持主流图像格式间的转换
- **质量评估** - 多维度图像质量分析

### 高级特性
- **批量处理** - 支持大规模图像批处理
- **智能自动增强** - 基于质量评估的自动优化
- **多引擎支持** - OpenCV、Pillow、scikit-image、PyTorch
- **异步处理** - 支持长时间处理任务的异步执行
- **任务管理** - 完整的任务状态跟踪和管理

## 🏗️ 架构设计

### 服务架构
- **完全无状态设计** - 不直接连接数据库或缓存系统，通过storage-service管理所有数据
- **RESTful API** - 标准HTTP接口
- **异步处理** - 支持同步和异步两种处理模式
- **微服务通信** - 通过HTTP与storage-service通信
- **统一缓存策略** - 不使用独立Redis实例，遵循项目架构一致性原则

### 端口分配
- **服务端口**: 8005
- **健康检查**: `/health`
- **API文档**: `/docs`
- **就绪探针**: `/ready`

## 📦 快速开始

### 环境要求
- Python 3.11+
- Docker 20.10+
- 至少2GB可用内存
- 系统级图像处理库支持

### Docker部署（推荐）

1. **克隆项目并进入目录**
```bash
cd services/image-processing-service
```

2. **配置环境变量**
```bash
cp .env.example .env
# 根据需要修改配置
```

3. **使用Docker Compose启动**
```bash
docker-compose up -d
```

4. **验证服务状态**
```bash
curl http://localhost:8005/health
```

### 开发环境部署

1. **安装系统依赖（Ubuntu/Debian）**
```bash
sudo apt-get update && sudo apt-get install -y \
    libopencv-dev libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **启动服务**
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8005 --reload
```

## 🔧 配置说明

### 核心配置
```env
# 服务配置
SERVICE_NAME=image-processing-service
PORT=8005
DEBUG=false
LOG_LEVEL=INFO

# 文件处理限制
MAX_FILE_SIZE=52428800  # 50MB
SUPPORTED_IMAGE_FORMATS=["jpg", "jpeg", "png", "tiff", "bmp", "webp"]
MAX_CONCURRENT_TASKS=10
TASK_TIMEOUT=300

# Storage Service通信
STORAGE_SERVICE_URL=http://localhost:8002
STORAGE_SERVICE_TIMEOUT=30
STORAGE_SERVICE_RETRIES=3
```

### 图像处理参数
```env
# 处理引擎
DEFAULT_PROCESSING_ENGINE=opencv
OPENCV_THREAD_COUNT=4

# 质量评估阈值
QUALITY_BRIGHTNESS_MIN=50
QUALITY_BRIGHTNESS_MAX=200
QUALITY_CONTRAST_MIN=30
QUALITY_SHARPNESS_MIN=100

# 增强参数
ENHANCE_BRIGHTNESS_FACTOR=1.1
ENHANCE_CONTRAST_FACTOR=1.2
ENHANCE_SHARPNESS_FACTOR=1.1
```

## 📖 API使用指南

### 单图像处理

#### 同步处理
```bash
curl -X POST "http://localhost:8005/api/v1/images/process" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sample.jpg" \
  -F "processing_type=enhance" \
  -F "engine=opencv"
```

#### 异步处理
```bash
curl -X POST "http://localhost:8005/api/v1/images/process-async" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/image.jpg",
    "processing_type": "auto_enhance",
    "config": {
      "brightness_factor": 1.2,
      "contrast_factor": 1.3
    }
  }'
```

### 批量处理
```bash
curl -X POST "http://localhost:8005/api/v1/images/batch-process" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": [
      "/path/to/image1.jpg",
      "/path/to/image2.png"
    ],
    "processing_type": "denoise",
    "engine": "opencv",
    "config": {
      "denoise_strength": 15
    }
  }'
```

### 质量评估
```bash
curl -X POST "http://localhost:8005/api/v1/images/assess-quality" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@document.jpg"
```

### 任务管理
```bash
# 获取任务状态
curl "http://localhost:8005/api/v1/tasks/{task_id}/status"

# 获取任务列表
curl "http://localhost:8005/api/v1/tasks?status=completed&limit=10"

# 删除任务
curl -X DELETE "http://localhost:8005/api/v1/tasks/{task_id}"
```

## 🎯 处理类型和参数

### 支持的处理类型

1. **enhance** - 图像增强
   ```json
   {
     "brightness_factor": 1.1,
     "contrast_factor": 1.2,
     "sharpness_factor": 1.1
   }
   ```

2. **denoise** - 去噪处理
   ```json
   {
     "denoise_strength": 10,
     "method": "bilateral"
   }
   ```

3. **deskew** - 倾斜校正
   ```json
   {
     "angle_threshold": 1.0,
     "method": "hough_lines"
   }
   ```

4. **resize** - 尺寸调整
   ```json
   {
     "width": 1024,
     "height": 768,
     "interpolation": "lanczos",
     "maintain_aspect_ratio": true
   }
   ```

5. **format_convert** - 格式转换
   ```json
   {
     "output_format": "png",
     "quality": 95,
     "optimize": true
   }
   ```

6. **auto_enhance** - 智能自动增强
   ```json
   {
     "target_brightness": 128,
     "target_contrast": 50,
     "adaptive": true
   }
   ```

### 质量评估指标
- **brightness** - 亮度 (0-255)
- **contrast** - 对比度
- **sharpness** - 锐度
- **noise_level** - 噪声水平 (0-1)
- **blur_metric** - 模糊度
- **skew_angle** - 倾斜角度 (度)

## 🔍 监控和日志

### 健康检查端点
- `GET /health` - 服务健康状态
- `GET /ready` - Kubernetes就绪探针
- `GET /info` - 服务详细信息

### 统计信息
```bash
curl "http://localhost:8005/api/v1/statistics?engine=opencv&date_from=2024-01-01"
```

### 引擎信息
```bash
curl "http://localhost:8005/api/v1/engines"
```

### 日志配置
服务使用loguru进行结构化日志记录：
- **DEBUG** - 详细调试信息
- **INFO** - 一般操作信息
- **WARNING** - 警告信息
- **ERROR** - 错误信息

## 🚀 部署和扩展

### Kubernetes部署
```bash
# 应用部署配置
kubectl apply -f k8s-deployment.yaml

# 查看服务状态
kubectl get pods -l app=image-processing-service

# 查看服务日志
kubectl logs -l app=image-processing-service -f
```

### 水平扩展
服务支持水平扩展，通过HPA自动调整实例数量：
- **CPU阈值**: 70%
- **内存阈值**: 80%
- **最小副本数**: 2
- **最大副本数**: 10

### 资源要求
- **CPU**: 0.5-2.0 cores
- **内存**: 1-4 GB
- **存储**: 临时文件存储空间
- **网络**: 与storage-service的通信

## 🛠️ 开发指南

### 项目结构
```
services/image-processing-service/
├── src/
│   ├── main.py              # FastAPI应用入口
│   ├── config/
│   │   └── settings.py      # 配置管理
│   ├── controllers/
│   │   └── image_controller.py  # API控制器
│   ├── services/
│   │   └── image_processing_service.py  # 核心业务逻辑
│   ├── clients/
│   │   └── storage_client.py    # Storage Service客户端
│   └── schemas/
│       └── image_schemas.py     # 数据模型
├── tests/                   # 测试文件
├── Dockerfile              # Docker配置
├── docker-compose.yml      # 本地开发环境
├── k8s-deployment.yaml     # Kubernetes部署
├── requirements.txt        # Python依赖
└── README.md              # 本文档
```

### 添加新的处理算法
1. 在`image_processing_service.py`中添加处理方法
2. 在`image_schemas.py`中定义相关数据模型
3. 在`image_controller.py`中添加API端点
4. 更新配置文件中的支持类型列表

### 测试
```bash
# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

## 🔧 故障排除

### 常见问题

1. **服务启动失败**
   - 检查端口8005是否被占用
   - 验证Python环境和依赖安装
   - 检查系统级图像处理库

2. **图像处理失败**
   - 验证图像格式是否支持
   - 检查文件大小限制
   - 查看详细错误日志

3. **与Storage Service通信失败**
   - 检查storage-service是否运行
   - 验证网络连接和URL配置
   - 检查超时和重试设置

4. **内存使用过高**
   - 调整批处理大小
   - 减少并发任务数量
   - 优化图像处理参数

### 调试模式
```bash
# 启用调试模式
DEBUG=true python -m uvicorn src.main:app --reload

# 查看详细日志
LOG_LEVEL=DEBUG python -m uvicorn src.main:app
```

## 📞 支持和贡献

### 获取帮助
- 查看API文档: `http://localhost:8005/docs`
- 检查服务状态: `http://localhost:8005/health`
- 查看服务信息: `http://localhost:8005/info`

### 开发贡献
1. Fork项目仓库
2. 创建功能分支
3. 添加测试用例
4. 提交代码变更
5. 创建Pull Request

---

**版本**: 1.0.0  
**最后更新**: 2024年12月  
**维护者**: Historical Text Project Team