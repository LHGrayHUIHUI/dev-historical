# Story 2.3: 图像处理服务

## 基本信息
- **Story ID**: 2.3
- **Epic**: Epic 2 - 数据处理和智能分类微服务
- **标题**: 图像处理服务
- **优先级**: 中
- **状态**: 待开发
- **预估工期**: 6-8天

## 用户故事
**作为** 图像处理专员  
**我希望** 有一个专业的图像处理服务  
**以便** 对历史文档图像进行增强、修复、格式转换等处理，提高图像质量和OCR识别效果

## 需求描述
开发专业的图像处理服务，支持图像增强、噪声去除、倾斜校正、尺寸调整、格式转换、图像修复等功能，为OCR识别和AI分析提供高质量的图像数据。

## 技术实现

### 核心技术栈
- **后端框架**: FastAPI 0.104+ (Python)
- **图像处理**: 
  - OpenCV 4.8+ (主要图像处理库)
  - Pillow 10.1+ (图像操作)
  - scikit-image 0.22+ (科学图像处理)
  - ImageIO 2.31+ (图像读写)
- **深度学习图像处理**: 
  - PyTorch 2.1+ (深度学习框架)
  - torchvision 0.16+ (计算机视觉)
  - ESRGAN (超分辨率)
  - Real-ESRGAN (实用超分辨率)
- **图像增强**: 
  - albumentations 1.3+ (图像增强库)
  - imgaug 0.4+ (图像增强)
- **文档图像处理**: 
  - deskew (倾斜校正)
  - textdistance (文本相似度)
- **数据库**: 
  - PostgreSQL (处理记录存储)
  - Redis (缓存和队列)
- **对象存储**: MinIO
- **消息队列**: RabbitMQ 3.12+

### 数据模型设计

#### 图像处理任务表 (image_processing_tasks)
```sql
CREATE TABLE image_processing_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES datasets(id),
    original_image_path VARCHAR(500) NOT NULL,
    processed_image_path VARCHAR(500),
    processing_type VARCHAR(50) NOT NULL, -- enhance, denoise, deskew, resize, format_convert, super_resolution
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    processing_config JSONB, -- 处理配置参数
    original_size JSONB, -- {"width": 1920, "height": 1080}
    processed_size JSONB,
    file_size_before BIGINT, -- 原始文件大小(字节)
    file_size_after BIGINT, -- 处理后文件大小
    quality_metrics JSONB, -- 图像质量指标
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time FLOAT,
    error_message TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 图像质量评估表 (image_quality_assessments)
```sql
CREATE TABLE image_quality_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES image_processing_tasks(id),
    image_path VARCHAR(500) NOT NULL,
    assessment_type VARCHAR(50), -- before, after
    brightness_score FLOAT, -- 亮度评分
    contrast_score FLOAT, -- 对比度评分
    sharpness_score FLOAT, -- 清晰度评分
    noise_level FLOAT, -- 噪声水平
    blur_level FLOAT, -- 模糊程度
    skew_angle FLOAT, -- 倾斜角度
    text_region_ratio FLOAT, -- 文本区域占比
    overall_quality FLOAT, -- 整体质量评分(0-1)
    assessment_method VARCHAR(100), -- 评估方法
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 图像处理配置表 (image_processing_configs)
```sql
CREATE TABLE image_processing_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    processing_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL, -- 处理配置参数
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 服务架构

#### 图像处理服务主类
```python
# src/services/image_processing_service.py
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from skimage import filters, morphology, measure, restoration
from skimage.metrics import structural_similarity as ssim
import albumentations as A
from pathlib import Path
import uuid
import time
import math
import io
import base64

class ImageProcessingService:
    def __init__(self):
        # 初始化深度学习模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.super_resolution_model = None  # 延迟加载
        
        # 数据库和存储
        self.db = DatabaseManager()
        self.storage = MinIOClient()
        self.message_queue = RabbitMQClient()
        
        # 图像处理配置
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        self.max_image_size = 50 * 1024 * 1024  # 50MB
    
    async def process_image(self, 
                          image_path: str, 
                          task_id: str,
                          processing_type: str,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理图像
        
        Args:
            image_path: 图像文件路径
            task_id: 任务ID
            processing_type: 处理类型
            config: 处理配置
            
        Returns:
            图像处理结果
        """
        try:
            start_time = time.time()
            
            # 更新任务状态
            await self._update_task_status(task_id, 'processing')
            
            # 加载图像
            original_image = await self._load_image(image_path)
            if original_image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 获取原始图像信息
            original_info = self._get_image_info(original_image)
            
            # 处理前质量评估
            quality_before = await self._assess_image_quality(original_image, 'before')
            
            # 根据处理类型执行相应处理
            if processing_type == 'enhance':
                processed_image = await self._enhance_image(original_image, config)
            elif processing_type == 'denoise':
                processed_image = await self._denoise_image(original_image, config)
            elif processing_type == 'deskew':
                processed_image = await self._deskew_image(original_image, config)
            elif processing_type == 'resize':
                processed_image = await self._resize_image(original_image, config)
            elif processing_type == 'format_convert':
                processed_image = await self._convert_format(original_image, config)
            elif processing_type == 'super_resolution':
                processed_image = await self._super_resolution(original_image, config)
            elif processing_type == 'auto_enhance':
                processed_image = await self._auto_enhance(original_image, config)
            else:
                raise ValueError(f"不支持的处理类型: {processing_type}")
            
            # 处理后质量评估
            quality_after = await self._assess_image_quality(processed_image, 'after')
            
            # 保存处理后的图像
            processed_path = await self._save_processed_image(
                processed_image, task_id, processing_type, config
            )
            
            # 获取处理后图像信息
            processed_info = self._get_image_info(processed_image)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 保存处理结果
            result = await self._save_processing_result(
                task_id=task_id,
                original_info=original_info,
                processed_info=processed_info,
                processed_path=processed_path,
                quality_before=quality_before,
                quality_after=quality_after,
                processing_time=processing_time
            )
            
            # 更新任务状态
            await self._update_task_status(task_id, 'completed')
            
            return {
                'success': True,
                'task_id': task_id,
                'processed_image_path': processed_path,
                'original_size': original_info['size'],
                'processed_size': processed_info['size'],
                'quality_improvement': quality_after['overall_quality'] - quality_before['overall_quality'],
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            await self._update_task_status(task_id, 'failed', str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像数组
        """
        try:
            # 支持多种加载方式
            if image_path.startswith('http'):
                # 从URL加载
                import requests
                response = requests.get(image_path)
                image = Image.open(io.BytesIO(response.content))
            else:
                # 从本地文件加载
                image = Image.open(image_path)
            
            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            logger.error(f"加载图像失败: {str(e)}")
            return None
    
    def _get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """
        获取图像信息
        
        Args:
            image: 图像数组
            
        Returns:
            图像信息字典
        """
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'size': {'width': width, 'height': height},
            'channels': channels,
            'dtype': str(image.dtype),
            'file_size': image.nbytes
        }
    
    async def _assess_image_quality(self, 
                                  image: np.ndarray, 
                                  assessment_type: str) -> Dict[str, float]:
        """
        评估图像质量
        
        Args:
            image: 图像数组
            assessment_type: 评估类型
            
        Returns:
            质量评估结果
        """
        # 转换为灰度图像进行分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 亮度评分
        brightness_score = np.mean(gray) / 255.0
        
        # 对比度评分
        contrast_score = np.std(gray) / 255.0
        
        # 清晰度评分（拉普拉斯方差）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # 归一化到0-1
        
        # 噪声水平评估
        noise_level = self._estimate_noise_level(gray)
        
        # 模糊程度评估
        blur_level = self._estimate_blur_level(gray)
        
        # 倾斜角度检测
        skew_angle = self._detect_skew_angle(gray)
        
        # 文本区域占比
        text_region_ratio = self._estimate_text_region_ratio(gray)
        
        # 整体质量评分
        overall_quality = (
            brightness_score * 0.15 +
            contrast_score * 0.2 +
            sharpness_score * 0.3 +
            (1 - noise_level) * 0.2 +
            (1 - blur_level) * 0.15
        )
        
        return {
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'sharpness_score': sharpness_score,
            'noise_level': noise_level,
            'blur_level': blur_level,
            'skew_angle': skew_angle,
            'text_region_ratio': text_region_ratio,
            'overall_quality': overall_quality
        }
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """
        估计图像噪声水平
        
        Args:
            image: 灰度图像
            
        Returns:
            噪声水平(0-1)
        """
        # 使用高斯滤波后的差异来估计噪声
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        noise = cv2.absdiff(image, blurred)
        noise_level = np.mean(noise) / 255.0
        
        return min(noise_level, 1.0)
    
    def _estimate_blur_level(self, image: np.ndarray) -> float:
        """
        估计图像模糊程度
        
        Args:
            image: 灰度图像
            
        Returns:
            模糊程度(0-1)
        """
        # 使用Sobel算子检测边缘强度
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(sobelx**2 + sobely**2)
        
        # 边缘强度越低，模糊程度越高
        blur_level = 1.0 - min(np.mean(edge_strength) / 100.0, 1.0)
        
        return blur_level
    
    def _detect_skew_angle(self, image: np.ndarray) -> float:
        """
        检测图像倾斜角度
        
        Args:
            image: 灰度图像
            
        Returns:
            倾斜角度(度)
        """
        # 边缘检测
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:20]:  # 只取前20条线
                angle = theta * 180 / np.pi - 90
                if abs(angle) < 45:  # 只考虑小角度倾斜
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
        
        return 0.0
    
    def _estimate_text_region_ratio(self, image: np.ndarray) -> float:
        """
        估计文本区域占比
        
        Args:
            image: 灰度图像
            
        Returns:
            文本区域占比(0-1)
        """
        # 二值化
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作连接文本区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 计算文本像素占比
        text_pixels = np.sum(processed == 0)  # 假设文本是黑色
        total_pixels = processed.size
        
        return text_pixels / total_pixels
    
    async def _enhance_image(self, 
                           image: np.ndarray, 
                           config: Dict[str, Any]) -> np.ndarray:
        """
        图像增强
        
        Args:
            image: 输入图像
            config: 增强配置
            
        Returns:
            增强后的图像
        """
        enhanced = image.copy()
        
        # 亮度调整
        if config.get('adjust_brightness', False):
            brightness_factor = config.get('brightness_factor', 1.0)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=brightness_factor, beta=0)
        
        # 对比度调整
        if config.get('adjust_contrast', False):
            contrast_factor = config.get('contrast_factor', 1.0)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_factor, beta=0)
        
        # 伽马校正
        if config.get('gamma_correction', False):
            gamma = config.get('gamma', 1.0)
            enhanced = self._gamma_correction(enhanced, gamma)
        
        # 直方图均衡化
        if config.get('histogram_equalization', False):
            if len(enhanced.shape) == 3:
                # 彩色图像：在LAB空间进行
                lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # 灰度图像
                enhanced = cv2.equalizeHist(enhanced)
        
        # CLAHE (对比度限制自适应直方图均衡化)
        if config.get('clahe', False):
            clip_limit = config.get('clahe_clip_limit', 2.0)
            tile_grid_size = config.get('clahe_tile_size', (8, 8))
            
            if len(enhanced.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced = clahe.apply(enhanced)
        
        # 锐化
        if config.get('sharpen', False):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        伽马校正
        
        Args:
            image: 输入图像
            gamma: 伽马值
            
        Returns:
            校正后的图像
        """
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # 应用伽马校正
        return cv2.LUT(image, table)
    
    async def _denoise_image(self, 
                           image: np.ndarray, 
                           config: Dict[str, Any]) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image: 输入图像
            config: 去噪配置
            
        Returns:
            去噪后的图像
        """
        method = config.get('denoise_method', 'bilateral')
        
        if method == 'bilateral':
            # 双边滤波
            d = config.get('bilateral_d', 9)
            sigma_color = config.get('bilateral_sigma_color', 75)
            sigma_space = config.get('bilateral_sigma_space', 75)
            
            if len(image.shape) == 3:
                denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            else:
                denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        elif method == 'gaussian':
            # 高斯滤波
            kernel_size = config.get('gaussian_kernel_size', 5)
            sigma = config.get('gaussian_sigma', 1.0)
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        elif method == 'median':
            # 中值滤波
            kernel_size = config.get('median_kernel_size', 5)
            denoised = cv2.medianBlur(image, kernel_size)
        
        elif method == 'nlm':
            # 非局部均值去噪
            h = config.get('nlm_h', 10)
            template_window_size = config.get('nlm_template_size', 7)
            search_window_size = config.get('nlm_search_size', 21)
            
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, None, h, h, template_window_size, search_window_size
                )
            else:
                denoised = cv2.fastNlMeansDenoising(
                    image, None, h, template_window_size, search_window_size
                )
        
        else:
            raise ValueError(f"不支持的去噪方法: {method}")
        
        return denoised
    
    async def _deskew_image(self, 
                          image: np.ndarray, 
                          config: Dict[str, Any]) -> np.ndarray:
        """
        图像倾斜校正
        
        Args:
            image: 输入图像
            config: 校正配置
            
        Returns:
            校正后的图像
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 检测倾斜角度
        angle = self._detect_skew_angle(gray)
        
        # 角度阈值
        angle_threshold = config.get('angle_threshold', 0.5)
        if abs(angle) < angle_threshold:
            return image  # 不需要校正
        
        # 旋转图像
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新的边界框
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # 调整旋转矩阵的平移部分
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # 应用旋转
        deskewed = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return deskewed
    
    async def _resize_image(self, 
                          image: np.ndarray, 
                          config: Dict[str, Any]) -> np.ndarray:
        """
        图像尺寸调整
        
        Args:
            image: 输入图像
            config: 调整配置
            
        Returns:
            调整后的图像
        """
        resize_method = config.get('resize_method', 'scale')
        
        if resize_method == 'scale':
            # 按比例缩放
            scale_factor = config.get('scale_factor', 1.0)
            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        
        elif resize_method == 'fixed_size':
            # 固定尺寸
            new_width = config.get('target_width', image.shape[1])
            new_height = config.get('target_height', image.shape[0])
        
        elif resize_method == 'max_dimension':
            # 最大尺寸限制
            max_dimension = config.get('max_dimension', 2048)
            height, width = image.shape[:2]
            
            if max(height, width) > max_dimension:
                if height > width:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                else:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
            else:
                return image  # 不需要调整
        
        else:
            raise ValueError(f"不支持的调整方法: {resize_method}")
        
        # 插值方法
        interpolation_method = config.get('interpolation', 'cubic')
        interpolation_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        interpolation = interpolation_map.get(interpolation_method, cv2.INTER_CUBIC)
        
        # 调整尺寸
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        return resized
    
    async def _convert_format(self, 
                            image: np.ndarray, 
                            config: Dict[str, Any]) -> np.ndarray:
        """
        图像格式转换
        
        Args:
            image: 输入图像
            config: 转换配置
            
        Returns:
            转换后的图像
        """
        target_format = config.get('target_format', 'RGB')
        
        if target_format == 'grayscale':
            if len(image.shape) == 3:
                converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                converted = image
        
        elif target_format == 'binary':
            # 转换为二值图像
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            threshold_method = config.get('threshold_method', 'otsu')
            
            if threshold_method == 'otsu':
                _, converted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_method == 'adaptive':
                converted = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            else:
                threshold_value = config.get('threshold_value', 127)
                _, converted = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        elif target_format == 'RGB':
            if len(image.shape) == 2:
                converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                converted = image
        
        else:
            converted = image
        
        return converted
    
    async def _super_resolution(self, 
                              image: np.ndarray, 
                              config: Dict[str, Any]) -> np.ndarray:
        """
        超分辨率处理
        
        Args:
            image: 输入图像
            config: 超分辨率配置
            
        Returns:
            超分辨率图像
        """
        # 简化版本：使用双三次插值进行放大
        scale_factor = config.get('scale_factor', 2)
        
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 使用双三次插值
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 可以在这里集成Real-ESRGAN等深度学习超分辨率模型
        # if self.super_resolution_model is not None:
        #     upscaled = self._apply_sr_model(upscaled)
        
        return upscaled
    
    async def _auto_enhance(self, 
                          image: np.ndarray, 
                          config: Dict[str, Any]) -> np.ndarray:
        """
        自动图像增强
        
        Args:
            image: 输入图像
            config: 增强配置
            
        Returns:
            增强后的图像
        """
        enhanced = image.copy()
        
        # 评估图像质量
        quality = await self._assess_image_quality(image, 'before')
        
        # 根据质量评估结果自动调整参数
        auto_config = {
            'adjust_brightness': quality['brightness_score'] < 0.3 or quality['brightness_score'] > 0.8,
            'adjust_contrast': quality['contrast_score'] < 0.2,
            'clahe': quality['contrast_score'] < 0.3,
            'denoise_method': 'bilateral' if quality['noise_level'] > 0.3 else None,
            'sharpen': quality['sharpness_score'] < 0.5
        }
        
        # 亮度调整
        if auto_config['adjust_brightness']:
            if quality['brightness_score'] < 0.3:
                brightness_factor = 1.3
            else:
                brightness_factor = 0.8
            enhanced = cv2.convertScaleAbs(enhanced, alpha=brightness_factor, beta=0)
        
        # 对比度调整
        if auto_config['adjust_contrast']:
            contrast_factor = 1.5
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_factor, beta=0)
        
        # CLAHE
        if auto_config['clahe']:
            if len(enhanced.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(enhanced)
        
        # 去噪
        if auto_config['denoise_method']:
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 锐化
        if auto_config['sharpen']:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
```

### API设计

#### 图像处理控制器
```python
# controllers/image_processing_controller.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import json
import uuid
from datetime import datetime

from ..services.image_processing_service import ImageProcessingService
from ..models.image_processing_models import (
    ProcessingType, ImageProcessingRequest, ImageBatchRequest,
    ImageProcessingResponse, ImageTaskResponse, ImageQualityResponse,
    ImageTaskListResponse, ImageProcessingConfig
)
from ..dependencies import get_image_processing_service, get_current_user, get_optional_user
from ..models.user_models import User

router = APIRouter(prefix="/api/v1/image-processing", tags=["图像处理"])

@router.post("/process", response_model=ImageProcessingResponse)
async def process_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="待处理的图像文件"),
    processing_type: ProcessingType = Form(..., description="处理类型"),
    config: str = Form(default="{}", description="处理配置JSON字符串"),
    image_service: ImageProcessingService = Depends(get_image_processing_service),
    current_user: User = Depends(get_current_user)
):
    """
    单图像处理
    
    Args:
        background_tasks: 后台任务管理器
        image: 上传的图像文件
        processing_type: 处理类型
        config: 处理配置
        image_service: 图像处理服务
        current_user: 当前用户
        
    Returns:
        图像处理响应
    """
    try:
        # 验证文件类型
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 验证文件大小 (50MB)
        if image.size and image.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小不能超过50MB")
        
        # 解析配置
        try:
            processing_config = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="配置格式错误")
        
        # 创建处理任务
        task_id = str(uuid.uuid4())
        
        # 异步处理图像
        background_tasks.add_task(
            image_service.process_image_async,
            task_id=task_id,
            image_file=image,
            processing_type=processing_type,
            config=processing_config,
            user_id=current_user.id
        )
        
        return ImageProcessingResponse(
            success=True,
            task_id=task_id,
            message="图像处理任务已提交",
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/batch", response_model=ImageProcessingResponse)
async def batch_process_images(
    request: ImageBatchRequest,
    background_tasks: BackgroundTasks,
    image_service: ImageProcessingService = Depends(get_image_processing_service),
    current_user: User = Depends(get_current_user)
):
    """
    批量图像处理
    
    Args:
        request: 批量处理请求
        background_tasks: 后台任务管理器
        image_service: 图像处理服务
        current_user: 当前用户
        
    Returns:
        批量处理响应
    """
    try:
        # 验证图像路径数量
        if len(request.image_paths) > 100:
            raise HTTPException(status_code=400, detail="单次批量处理不能超过100张图像")
        
        # 创建批量任务
        batch_id = str(uuid.uuid4())
        
        # 异步处理批量图像
        background_tasks.add_task(
            image_service.batch_process_images_async,
            batch_id=batch_id,
            image_paths=request.image_paths,
            processing_type=request.processing_type,
            config=request.config.dict() if request.config else {},
            user_id=current_user.id,
            dataset_id=request.dataset_id
        )
        
        return ImageProcessingResponse(
            success=True,
            task_id=batch_id,
            message=f"批量图像处理任务已提交，共{len(request.image_paths)}张图像",
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")

@router.get("/tasks/{task_id}", response_model=ImageTaskResponse)
async def get_task_status(
    task_id: str,
    image_service: ImageProcessingService = Depends(get_image_processing_service),
    current_user: User = Depends(get_current_user)
):
    """
    获取处理任务状态
    
    Args:
        task_id: 任务ID
        image_service: 图像处理服务
        current_user: 当前用户
        
    Returns:
        任务状态响应
    """
    try:
        task = await image_service.get_task_status(task_id, current_user.id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return ImageTaskResponse(
            success=True,
            task=task
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@router.get("/tasks", response_model=ImageTaskListResponse)
async def get_task_list(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    processing_type: Optional[ProcessingType] = None,
    image_service: ImageProcessingService = Depends(get_image_processing_service),
    current_user: User = Depends(get_current_user)
):
    """
    获取任务列表
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 状态过滤
        processing_type: 处理类型过滤
        image_service: 图像处理服务
        current_user: 当前用户
        
    Returns:
        任务列表响应
    """
    try:
        tasks, total = await image_service.get_task_list(
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            status=status,
            processing_type=processing_type
        )
        
        return ImageTaskListResponse(
            success=True,
            tasks=tasks,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")

@router.post("/assess-quality", response_model=ImageQualityResponse)
async def assess_image_quality(
    image: UploadFile = File(..., description="待评估的图像文件"),
    image_service: ImageProcessingService = Depends(get_image_processing_service),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    图像质量评估
    
    Args:
        image: 上传的图像文件
        image_service: 图像处理服务
        current_user: 当前用户（可选）
        
    Returns:
        质量评估响应
    """
    try:
        # 验证文件类型
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 评估图像质量
        quality_metrics = await image_service.assess_image_quality_from_file(image)
        
        return ImageQualityResponse(
            success=True,
            quality_metrics=quality_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")

@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    image_service: ImageProcessingService = Depends(get_image_processing_service),
    current_user: User = Depends(get_current_user)
):
    """
    删除处理任务
    
    Args:
        task_id: 任务ID
        image_service: 图像处理服务
        current_user: 当前用户
        
    Returns:
        删除结果
    """
    try:
        success = await image_service.delete_task(task_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或无权限删除")
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "任务删除成功"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")

@router.post("/tasks/{task_id}/retry", response_model=ImageProcessingResponse)
async def retry_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    image_service: ImageProcessingService = Depends(get_image_processing_service),
    current_user: User = Depends(get_current_user)
):
    """
    重试失败的处理任务
    
    Args:
        task_id: 任务ID
        background_tasks: 后台任务管理器
        image_service: 图像处理服务
        current_user: 当前用户
        
    Returns:
        重试响应
    """
    try:
        # 异步重试任务
        background_tasks.add_task(
            image_service.retry_task_async,
            task_id=task_id,
            user_id=current_user.id
        )
        
        return ImageProcessingResponse(
            success=True,
            task_id=task_id,
            message="任务重试已提交",
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重试任务失败: {str(e)}")
```

### Pydantic模型定义

```python
# models/image_processing_models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime
import uuid

class ProcessingType(str, Enum):
    """
    图像处理类型枚举
    """
    ENHANCE = "enhance"  # 图像增强
    DENOISE = "denoise"  # 噪声去除
    DESKEW = "deskew"  # 倾斜校正
    RESIZE = "resize"  # 尺寸调整
    FORMAT_CONVERT = "format_convert"  # 格式转换
    SUPER_RESOLUTION = "super_resolution"  # 超分辨率
    AUTO_ENHANCE = "auto_enhance"  # 自动增强

class DenoiseMethod(str, Enum):
    """
    去噪方法枚举
    """
    BILATERAL = "bilateral"  # 双边滤波
    GAUSSIAN = "gaussian"  # 高斯滤波
    MEDIAN = "median"  # 中值滤波
    NLM = "nlm"  # 非局部均值

class ResizeMethod(str, Enum):
    """
    尺寸调整方法枚举
    """
    SCALE = "scale"  # 按比例缩放
    FIXED_SIZE = "fixed_size"  # 固定尺寸
    MAX_DIMENSION = "max_dimension"  # 最大尺寸限制

class InterpolationMethod(str, Enum):
    """
    插值方法枚举
    """
    NEAREST = "nearest"  # 最近邻
    LINEAR = "linear"  # 线性插值
    CUBIC = "cubic"  # 三次插值
    LANCZOS = "lanczos"  # Lanczos插值

class ThresholdMethod(str, Enum):
    """
    阈值方法枚举
    """
    OTSU = "otsu"  # OTSU自动阈值
    ADAPTIVE = "adaptive"  # 自适应阈值
    FIXED = "fixed"  # 固定阈值

class TaskStatus(str, Enum):
    """
    任务状态枚举
    """
    PENDING = "pending"  # 等待中
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消

class ImageSize(BaseModel):
    """
    图像尺寸模型
    """
    width: int = Field(..., description="图像宽度")
    height: int = Field(..., description="图像高度")

class QualityMetrics(BaseModel):
    """
    图像质量指标模型
    """
    brightness_score: float = Field(..., description="亮度评分(0-1)")
    contrast_score: float = Field(..., description="对比度评分(0-1)")
    sharpness_score: float = Field(..., description="清晰度评分(0-1)")
    noise_level: float = Field(..., description="噪声水平(0-1)")
    blur_level: float = Field(..., description="模糊程度(0-1)")
    skew_angle: float = Field(..., description="倾斜角度(度)")
    text_region_ratio: float = Field(..., description="文本区域占比(0-1)")
    overall_quality: float = Field(..., description="整体质量评分(0-1)")

class EnhanceConfig(BaseModel):
    """
    图像增强配置
    """
    adjust_brightness: bool = Field(default=False, description="是否调整亮度")
    brightness_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="亮度因子")
    adjust_contrast: bool = Field(default=False, description="是否调整对比度")
    contrast_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="对比度因子")
    gamma_correction: bool = Field(default=False, description="是否进行伽马校正")
    gamma: float = Field(default=1.0, ge=0.1, le=3.0, description="伽马值")
    histogram_equalization: bool = Field(default=False, description="是否进行直方图均衡化")
    clahe: bool = Field(default=False, description="是否使用CLAHE")
    clahe_clip_limit: float = Field(default=2.0, ge=1.0, le=10.0, description="CLAHE裁剪限制")
    clahe_tile_size: tuple = Field(default=(8, 8), description="CLAHE瓦片大小")
    sharpen: bool = Field(default=False, description="是否锐化")

class DenoiseConfig(BaseModel):
    """
    去噪配置
    """
    denoise_method: DenoiseMethod = Field(default=DenoiseMethod.BILATERAL, description="去噪方法")
    # 双边滤波参数
    bilateral_d: int = Field(default=9, ge=3, le=15, description="双边滤波邻域直径")
    bilateral_sigma_color: float = Field(default=75, ge=10, le=200, description="颜色空间标准差")
    bilateral_sigma_space: float = Field(default=75, ge=10, le=200, description="坐标空间标准差")
    # 高斯滤波参数
    gaussian_kernel_size: int = Field(default=5, ge=3, le=15, description="高斯核大小")
    gaussian_sigma: float = Field(default=1.0, ge=0.1, le=5.0, description="高斯标准差")
    # 中值滤波参数
    median_kernel_size: int = Field(default=5, ge=3, le=15, description="中值滤波核大小")
    # 非局部均值参数
    nlm_h: float = Field(default=10, ge=1, le=30, description="滤波强度")
    nlm_template_size: int = Field(default=7, ge=3, le=15, description="模板窗口大小")
    nlm_search_size: int = Field(default=21, ge=7, le=35, description="搜索窗口大小")

class DeskewConfig(BaseModel):
    """
    倾斜校正配置
    """
    angle_threshold: float = Field(default=0.5, ge=0.1, le=5.0, description="角度阈值")
    auto_detect: bool = Field(default=True, description="是否自动检测角度")
    manual_angle: Optional[float] = Field(default=None, description="手动指定角度")

class ResizeConfig(BaseModel):
    """
    尺寸调整配置
    """
    resize_method: ResizeMethod = Field(default=ResizeMethod.SCALE, description="调整方法")
    scale_factor: float = Field(default=1.0, ge=0.1, le=5.0, description="缩放因子")
    target_width: Optional[int] = Field(default=None, ge=1, description="目标宽度")
    target_height: Optional[int] = Field(default=None, ge=1, description="目标高度")
    max_dimension: int = Field(default=2048, ge=100, description="最大尺寸")
    interpolation: InterpolationMethod = Field(default=InterpolationMethod.CUBIC, description="插值方法")
    maintain_aspect_ratio: bool = Field(default=True, description="是否保持宽高比")

class FormatConvertConfig(BaseModel):
    """
    格式转换配置
    """
    target_format: str = Field(default="RGB", description="目标格式")
    threshold_method: ThresholdMethod = Field(default=ThresholdMethod.OTSU, description="阈值方法")
    threshold_value: int = Field(default=127, ge=0, le=255, description="阈值")
    quality: int = Field(default=95, ge=1, le=100, description="输出质量")

class SuperResolutionConfig(BaseModel):
    """
    超分辨率配置
    """
    scale_factor: int = Field(default=2, ge=2, le=4, description="放大倍数")
    model_name: str = Field(default="ESRGAN", description="模型名称")
    use_gpu: bool = Field(default=True, description="是否使用GPU")

class ImageProcessingConfig(BaseModel):
    """
    图像处理配置
    """
    enhance: Optional[EnhanceConfig] = Field(default=None, description="增强配置")
    denoise: Optional[DenoiseConfig] = Field(default=None, description="去噪配置")
    deskew: Optional[DeskewConfig] = Field(default=None, description="校正配置")
    resize: Optional[ResizeConfig] = Field(default=None, description="调整配置")
    format_convert: Optional[FormatConvertConfig] = Field(default=None, description="转换配置")
    super_resolution: Optional[SuperResolutionConfig] = Field(default=None, description="超分辨率配置")

class ImageProcessingRequest(BaseModel):
    """
    图像处理请求模型
    """
    processing_type: ProcessingType = Field(..., description="处理类型")
    config: Optional[ImageProcessingConfig] = Field(default=None, description="处理配置")
    priority: int = Field(default=5, ge=1, le=10, description="任务优先级")
    callback_url: Optional[str] = Field(default=None, description="回调URL")

class ImageBatchRequest(BaseModel):
    """
    批量图像处理请求模型
    """
    dataset_id: Optional[str] = Field(default=None, description="数据集ID")
    image_paths: List[str] = Field(..., min_items=1, max_items=100, description="图像路径列表")
    processing_type: ProcessingType = Field(..., description="处理类型")
    config: Optional[ImageProcessingConfig] = Field(default=None, description="处理配置")
    priority: int = Field(default=5, ge=1, le=10, description="任务优先级")
    callback_url: Optional[str] = Field(default=None, description="回调URL")

class ImageTask(BaseModel):
    """
    图像处理任务模型
    """
    id: str = Field(..., description="任务ID")
    user_id: str = Field(..., description="用户ID")
    dataset_id: Optional[str] = Field(default=None, description="数据集ID")
    processing_type: ProcessingType = Field(..., description="处理类型")
    status: TaskStatus = Field(..., description="任务状态")
    config: Dict[str, Any] = Field(default_factory=dict, description="处理配置")
    original_image_path: str = Field(..., description="原始图像路径")
    processed_image_path: Optional[str] = Field(default=None, description="处理后图像路径")
    original_size: Optional[ImageSize] = Field(default=None, description="原始尺寸")
    processed_size: Optional[ImageSize] = Field(default=None, description="处理后尺寸")
    quality_before: Optional[QualityMetrics] = Field(default=None, description="处理前质量")
    quality_after: Optional[QualityMetrics] = Field(default=None, description="处理后质量")
    processing_time: Optional[float] = Field(default=None, description="处理时间(秒)")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    priority: int = Field(default=5, description="任务优先级")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")

class ImageProcessingResponse(BaseModel):
    """
    图像处理响应模型
    """
    success: bool = Field(..., description="是否成功")
    task_id: str = Field(..., description="任务ID")
    message: str = Field(..., description="响应消息")
    status: str = Field(..., description="任务状态")
    processed_image_url: Optional[str] = Field(default=None, description="处理后图像URL")
    quality_improvement: Optional[float] = Field(default=None, description="质量提升")
    processing_time: Optional[float] = Field(default=None, description="处理时间")
    original_size: Optional[ImageSize] = Field(default=None, description="原始尺寸")
    processed_size: Optional[ImageSize] = Field(default=None, description="处理后尺寸")

class ImageTaskResponse(BaseModel):
    """
    图像任务响应模型
    """
    success: bool = Field(..., description="是否成功")
    task: ImageTask = Field(..., description="任务信息")

class ImageTaskListResponse(BaseModel):
    """
    图像任务列表响应模型
    """
    success: bool = Field(..., description="是否成功")
    tasks: List[ImageTask] = Field(..., description="任务列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="页码")
    page_size: int = Field(..., description="每页大小")

class ImageQualityResponse(BaseModel):
    """
    图像质量评估响应模型
    """
    success: bool = Field(..., description="是否成功")
    quality_metrics: QualityMetrics = Field(..., description="质量指标")

class ImageProcessingSettings(BaseModel):
    """
    图像处理服务设置
    """
    # 模型配置
    super_resolution_model_path: str = Field(default="models/ESRGAN", description="超分辨率模型路径")
    enable_gpu: bool = Field(default=True, description="是否启用GPU")
    
    # 处理限制
    max_image_size: int = Field(default=50 * 1024 * 1024, description="最大图像大小(字节)")
    max_dimension: int = Field(default=8192, description="最大图像尺寸")
    max_batch_size: int = Field(default=100, description="最大批量处理数量")
    
    # 性能设置
    max_concurrent_tasks: int = Field(default=5, description="最大并发任务数")
    task_timeout: int = Field(default=300, description="任务超时时间(秒)")
    
    # 存储设置
    storage_backend: str = Field(default="minio", description="存储后端")
    temp_dir: str = Field(default="/tmp/image_processing", description="临时目录")
    
    # 缓存设置
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    cache_ttl: int = Field(default=3600, description="缓存TTL(秒)")
    
    # 质量评估设置
    quality_assessment_enabled: bool = Field(default=True, description="是否启用质量评估")
    auto_enhance_threshold: float = Field(default=0.6, description="自动增强阈值")
```

### 依赖注入配置

```python
# dependencies/image_processing_deps.py
from functools import lru_cache
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta

from services.image_processing_service import ImageProcessingService
from models.image_processing_models import ImageProcessingSettings
from core.config import get_settings
from core.database import get_database
from core.storage import get_storage_client
from core.message_queue import get_message_queue

security = HTTPBearer()

@lru_cache()
def get_image_processing_service() -> ImageProcessingService:
    """
    获取图像处理服务单例实例
    
    Returns:
        ImageProcessingService: 图像处理服务实例
    """
    settings = get_settings()
    image_settings = ImageProcessingSettings()
    
    # 初始化数据库连接
    database = get_database()
    
    # 初始化存储客户端
    storage_client = get_storage_client()
    
    # 初始化消息队列
    message_queue = get_message_queue()
    
    # 创建图像处理服务实例
    service = ImageProcessingService(
        settings=image_settings,
        database=database,
        storage_client=storage_client,
        message_queue=message_queue
    )
    
    return service

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    获取当前认证用户信息
    
    Args:
        credentials: JWT认证凭据
        
    Returns:
        dict: 用户信息字典
        
    Raises:
        HTTPException: 认证失败时抛出401错误
    """
    try:
        # 解码JWT token
        settings = get_settings()
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # 检查token是否过期
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 获取用户ID
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "user_id": user_id,
            "username": payload.get("username"),
            "email": payload.get("email"),
            "roles": payload.get("roles", [])
        }
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    获取可选的用户信息（允许匿名访问）
    
    Args:
        credentials: 可选的JWT认证凭据
        
    Returns:
        Optional[dict]: 用户信息字典或None
    """
    if not credentials:
        return None
    
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None

def validate_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """
    验证管理员用户权限
    
    Args:
        current_user: 当前用户信息
        
    Returns:
        dict: 用户信息字典
        
    Raises:
        HTTPException: 权限不足时抛出403错误
    """
    user_roles = current_user.get("roles", [])
    if "admin" not in user_roles and "super_admin" not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足，需要管理员权限"
        )
    
    return current_user
```

### 应用程序入口点

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import time
import traceback
from typing import Dict, Any

from controllers.image_processing_controller import router as image_processing_router
from dependencies.image_processing_deps import get_image_processing_service
from core.config import get_settings
from core.database import init_database, close_database
from core.logging import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用程序生命周期管理
    
    Args:
        app: FastAPI应用实例
    """
    # 启动时初始化
    logger.info("正在启动图像处理服务...")
    
    try:
        # 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")
        
        # 初始化图像处理服务
        image_service = get_image_processing_service()
        await image_service.initialize()
        logger.info("图像处理服务初始化完成")
        
        # 预热模型（如果启用GPU）
        settings = get_settings()
        if hasattr(settings, 'enable_gpu') and settings.enable_gpu:
            logger.info("正在预热深度学习模型...")
            await image_service.warmup_models()
            logger.info("模型预热完成")
        
        logger.info("图像处理服务启动成功")
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭图像处理服务...")
    
    try:
        # 清理图像处理服务
        image_service = get_image_processing_service()
        await image_service.cleanup()
        logger.info("图像处理服务清理完成")
        
        # 关闭数据库连接
        await close_database()
        logger.info("数据库连接已关闭")
        
        logger.info("图像处理服务关闭完成")
        
    except Exception as e:
        logger.error(f"服务关闭时出错: {str(e)}")
        logger.error(traceback.format_exc())

# 创建FastAPI应用
app = FastAPI(
    title="历史文本项目 - 图像处理服务",
    description="提供图像增强、去噪、校正等处理功能的微服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置GZip压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    记录HTTP请求日志
    
    Args:
        request: HTTP请求对象
        call_next: 下一个中间件或路由处理器
        
    Returns:
        Response: HTTP响应对象
    """
    start_time = time.time()
    
    # 记录请求信息
    logger.info(
        f"请求开始: {request.method} {request.url} "
        f"客户端: {request.client.host if request.client else 'unknown'}"
    )
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 记录响应信息
    logger.info(
        f"请求完成: {request.method} {request.url} "
        f"状态码: {response.status_code} "
        f"处理时间: {process_time:.3f}s"
    )
    
    # 添加处理时间到响应头
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    全局异常处理器
    
    Args:
        request: HTTP请求对象
        exc: 异常对象
        
    Returns:
        JSONResponse: 错误响应
    """
    logger.error(
        f"未处理的异常: {request.method} {request.url} "
        f"错误: {str(exc)}"
    )
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "服务器内部错误",
            "error_type": "InternalServerError",
            "timestamp": time.time()
        }
    )

# 请求验证异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """
    请求验证异常处理器
    
    Args:
        request: HTTP请求对象
        exc: 验证异常对象
        
    Returns:
        JSONResponse: 验证错误响应
    """
    logger.warning(
        f"请求验证失败: {request.method} {request.url} "
        f"错误: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "请求参数验证失败",
            "error_type": "ValidationError",
            "details": exc.errors(),
            "timestamp": time.time()
        }
    )

# 注册路由
app.include_router(
    image_processing_router,
    prefix="/api/v1/image-processing",
    tags=["图像处理"]
)

# 健康检查端点
@app.get("/health", tags=["系统"])
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点
    
    Returns:
        Dict[str, Any]: 服务健康状态
    """
    try:
        # 检查图像处理服务状态
        image_service = get_image_processing_service()
        service_status = await image_service.health_check()
        
        return {
            "status": "healthy",
            "service": "image-processing-service",
            "version": "1.0.0",
            "timestamp": time.time(),
            "components": {
                "image_processing": service_status,
                "database": "healthy",  # 可以添加数据库健康检查
                "storage": "healthy",   # 可以添加存储健康检查
                "message_queue": "healthy"  # 可以添加消息队列健康检查
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "image-processing-service",
                "error": str(e),
                "timestamp": time.time()
            }
        )

# 服务信息端点
@app.get("/info", tags=["系统"])
async def service_info() -> Dict[str, Any]:
    """
    获取服务信息
    
    Returns:
        Dict[str, Any]: 服务详细信息
    """
    settings = get_settings()
    
    return {
        "service": "image-processing-service",
        "version": "1.0.0",
        "description": "历史文本项目图像处理服务",
        "features": [
            "图像增强",
            "噪声去除",
            "倾斜校正",
            "尺寸调整",
            "格式转换",
            "超分辨率",
            "自动增强",
            "质量评估",
            "批量处理"
        ],
        "supported_formats": [
            "JPEG", "PNG", "TIFF", "BMP", "WEBP"
        ],
        "max_image_size": "50MB",
        "max_batch_size": 100,
        "gpu_enabled": getattr(settings, 'enable_gpu', False),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    
    # 开发环境运行配置
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )
```

### 前端集成

#### Vue3 图像处理组件
```vue
<!-- components/ImageProcessor.vue -->
<template>
  <div class="image-processor">
    <el-card class="config-card">
      <template #header>
        <span>图像处理配置</span>
      </template>
      
      <el-form :model="processingConfig" label-width="120px">
        <el-form-item label="处理类型">
          <el-select v-model="processingType">
            <el-option label="图像增强" value="enhance" />
            <el-option label="噪声去除" value="denoise" />
            <el-option label="倾斜校正" value="deskew" />
            <el-option label="尺寸调整" value="resize" />
            <el-option label="格式转换" value="format_convert" />
            <el-option label="超分辨率" value="super_resolution" />
            <el-option label="自动增强" value="auto_enhance" />
          </el-select>
        </el-form-item>
        
        <!-- 增强配置 -->
        <div v-if="processingType === 'enhance'">
          <el-form-item label="亮度调整">
            <el-switch v-model="processingConfig.adjust_brightness" />
            <el-slider 
              v-if="processingConfig.adjust_brightness"
              v-model="processingConfig.brightness_factor" 
              :min="0.5" 
              :max="2.0" 
              :step="0.1" 
              show-input
            />
          </el-form-item>
          
          <el-form-item label="对比度调整">
            <el-switch v-model="processingConfig.adjust_contrast" />
            <el-slider 
              v-if="processingConfig.adjust_contrast"
              v-model="processingConfig.contrast_factor" 
              :min="0.5" 
              :max="2.0" 
              :step="0.1" 
              show-input
            />
          </el-form-item>
          
          <el-form-item label="CLAHE增强">
            <el-switch v-model="processingConfig.clahe" />
          </el-form-item>
          
          <el-form-item label="锐化">
            <el-switch v-model="processingConfig.sharpen" />
          </el-form-item>
        </div>
        
        <!-- 去噪配置 -->
        <div v-if="processingType === 'denoise'">
          <el-form-item label="去噪方法">
            <el-select v-model="processingConfig.denoise_method">
              <el-option label="双边滤波" value="bilateral" />
              <el-option label="高斯滤波" value="gaussian" />
              <el-option label="中值滤波" value="median" />
              <el-option label="非局部均值" value="nlm" />
            </el-select>
          </el-form-item>
        </div>
        
        <!-- 尺寸调整配置 -->
        <div v-if="processingType === 'resize'">
          <el-form-item label="调整方法">
            <el-select v-model="processingConfig.resize_method">
              <el-option label="按比例缩放" value="scale" />
              <el-option label="固定尺寸" value="fixed_size" />
              <el-option label="最大尺寸限制" value="max_dimension" />
            </el-select>
          </el-form-item>
          
          <el-form-item v-if="processingConfig.resize_method === 'scale'" label="缩放比例">
            <el-input-number v-model="processingConfig.scale_factor" :min="0.1" :max="5" :step="0.1" />
          </el-form-item>
        </div>
      </el-form>
    </el-card>
    
    <el-card class="upload-card">
      <template #header>
        <span>图像上传</span>
      </template>
      
      <el-upload
        ref="uploadRef"
        class="image-upload"
        drag
        :action="uploadUrl"
        :headers="uploadHeaders"
        :data="uploadData"
        :on-success="handleUploadSuccess"
        :on-error="handleUploadError"
        :before-upload="beforeUpload"
        accept="image/*"
        :limit="1"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽图像到此处或 <em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 JPG、PNG、TIFF 等格式，单个文件不超过50MB
          </div>
        </template>
      </el-upload>
    </el-card>
    
    <!-- 处理结果 -->
    <el-card v-if="processingResult" class="result-card">
      <template #header>
        <span>处理结果</span>
      </template>
      
      <el-row :gutter="20">
        <el-col :span="12">
          <h4>原始图像</h4>
          <div class="image-container">
            <img :src="originalImageUrl" alt="原始图像" class="result-image" />
          </div>
          
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="尺寸">
              {{ processingResult.original_size.width }} × {{ processingResult.original_size.height }}
            </el-descriptions-item>
          </el-descriptions>
        </el-col>
        
        <el-col :span="12">
          <h4>处理后图像</h4>
          <div class="image-container">
            <img :src="processingResult.processed_image_url" alt="处理后图像" class="result-image" />
          </div>
          
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="尺寸">
              {{ processingResult.processed_size.width }} × {{ processingResult.processed_size.height }}
            </el-descriptions-item>
            <el-descriptions-item label="质量提升">
              <el-tag :type="getQualityImprovementType(processingResult.quality_improvement)">
                {{ (processingResult.quality_improvement * 100).toFixed(1) }}%
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="处理时间">
              {{ processingResult.processing_time.toFixed(1) }}秒
            </el-descriptions-item>
          </el-descriptions>
          
          <div class="action-buttons">
            <el-button type="primary" @click="downloadProcessedImage">
              下载处理后图像
            </el-button>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import type { UploadFile } from 'element-plus'

interface ProcessingResult {
  processed_image_url: string
  quality_improvement: number
  processing_time: number
  original_size: { width: number; height: number }
  processed_size: { width: number; height: number }
}

const authStore = useAuthStore()
const uploadRef = ref()
const processingType = ref('enhance')
const processingResult = ref<ProcessingResult | null>(null)
const originalImageUrl = ref('')

const processingConfig = reactive({
  // 增强配置
  adjust_brightness: false,
  brightness_factor: 1.0,
  adjust_contrast: false,
  contrast_factor: 1.0,
  clahe: true,
  sharpen: false,
  
  // 去噪配置
  denoise_method: 'bilateral',
  
  // 尺寸调整配置
  resize_method: 'scale',
  scale_factor: 1.0,
  target_width: 1920,
  target_height: 1080,
  max_dimension: 2048
})

const uploadUrl = computed(() => 
  `${import.meta.env.VITE_API_BASE_URL}/api/v1/image-processing/process`
)

const uploadHeaders = computed(() => ({
  'Authorization': `Bearer ${authStore.accessToken}`
}))

const uploadData = computed(() => ({
  processing_type: processingType.value,
  config: JSON.stringify(processingConfig)
}))

/**
 * 上传前验证
 * @param file 上传文件
 */
const beforeUpload = (file: UploadFile) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp']
  const maxSize = 50 * 1024 * 1024 // 50MB
  
  if (!allowedTypes.includes(file.type || '')) {
    ElMessage.error('不支持的图像格式')
    return false
  }
  
  if (file.size! > maxSize) {
    ElMessage.error('图像文件不能超过50MB')
    return false
  }
  
  // 保存原始图像URL用于显示
  originalImageUrl.value = URL.createObjectURL(file.raw!)
  
  return true
}

/**
 * 上传成功处理
 * @param response 响应数据
 * @param file 文件信息
 */
const handleUploadSuccess = (response: any, file: UploadFile) => {
  if (response.success) {
    processingResult.value = response
    ElMessage.success('图像处理完成')
  } else {
    ElMessage.error('图像处理失败')
  }
}

/**
 * 上传失败处理
 * @param error 错误信息
 * @param file 文件信息
 */
const handleUploadError = (error: any, file: UploadFile) => {
  ElMessage.error(`图像处理失败: ${error.message || '未知错误'}`)
}

/**
 * 下载处理后的图像
 */
const downloadProcessedImage = () => {
  if (!processingResult.value) return
  
  const link = document.createElement('a')
  link.href = processingResult.value.processed_image_url
  link.download = 'processed_image.jpg'
  link.click()
}

/**
 * 获取质量提升类型
 * @param improvement 质量提升值
 */
const getQualityImprovementType = (improvement: number) => {
  if (improvement > 0.1) return 'success'
  if (improvement > 0) return 'warning'
  return 'danger'
}
</script>

<style scoped>
.image-processor {
  padding: 20px;
}

.config-card,
.upload-card,
.result-card {
  margin-bottom: 20px;
}

.image-upload {
  width: 100%;
}

.image-container {
  text-align: center;
  margin-bottom: 15px;
}

.result-image {
  max-width: 100%;
  max-height: 300px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.action-buttons {
  margin-top: 15px;
  text-align: center;
}
</style>
```

## 验收标准

### 功能验收
- [ ] 支持多种图像增强算法
- [ ] 噪声去除效果明显
- [ ] 倾斜校正准确率 > 95%
- [ ] 尺寸调整保持图像质量
- [ ] 格式转换功能正常
- [ ] 自动增强效果良好
- [ ] 批量处理功能稳定

### 性能验收
- [ ] 单张图像处理时间 < 10秒
- [ ] 并发处理能力 > 5个任务
- [ ] 内存使用稳定
- [ ] 支持大尺寸图像处理

### 质量验收
- [ ] 图像质量评估准确
- [ ] 处理后图像质量提升明显
- [ ] 无明显处理伪影
- [ ] 文本区域保持清晰

## 业务价值
- 提高历史文档图像质量，改善OCR识别效果
- 自动化图像预处理，减少人工干预
- 为AI分析提供高质量的图像数据
- 支持文档数字化的标准化处理

## 依赖关系
- **前置条件**: Story 1.3 (数据采集服务)
- **后续依赖**: Story 2.1 (OCR文本识别服务)

## 风险与缓解
- **风险**: 处理后图像失真
- **缓解**: 质量评估机制 + 参数优化
- **风险**: 处理速度慢
- **缓解**: GPU加速 + 算法优化

## 开发任务分解
1. 基础图像处理算法集成 (2天)
2. 图像质量评估系统开发 (2天)
3. 高级处理功能开发 (2天)
4. 批量处理和队列管理 (1天)
5. 前端图像处理组件开发 (1天)
6. 性能优化和测试 (1天)