"""
图像处理核心服务类
无状态图像处理算法，专注于图像处理计算
数据存储通过storage-service完成
"""

import asyncio
import time
import math
import io
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
from pathlib import Path
from loguru import logger

# 图像处理核心库
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage import filters, morphology, measure, restoration, transform, exposure
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import torch
import torchvision.transforms as transforms

from ..config.settings import settings, get_processing_engine_config, get_enhancement_config, get_denoise_config
from ..schemas.image_schemas import *


class ImageProcessingService:
    """图像处理核心服务类（无状态架构）"""
    
    def __init__(self):
        self.is_initialized = False
        self.models = {}
        self.engine_config = get_processing_engine_config()
        self.enhancement_config = get_enhancement_config()
        self.denoise_config = get_denoise_config()
        
        # 本地缓存（内存缓存）
        self._cache = {}
        self._cache_max_size = settings.cache_max_size
        
        # 支持的处理操作
        self.supported_operations = [
            "enhance", "denoise", "deskew", "resize", 
            "format_convert", "super_resolution", "auto_enhance", "quality_assessment"
        ]
        
        logger.info("图像处理服务初始化开始...")
    
    async def initialize(self):
        """初始化图像处理服务"""
        if self.is_initialized:
            return
        
        try:
            # 检查OpenCV版本和功能
            logger.info(f"OpenCV版本: {cv2.__version__}")
            
            # 检查GPU支持（如果启用）
            if settings.use_gpu:
                await self._check_gpu_support()
            
            # 初始化深度学习模型（如果需要）
            if settings.enable_super_resolution:
                await self._init_super_resolution_models()
            
            self.is_initialized = True
            logger.info("图像处理服务初始化完成")
            
        except Exception as e:
            logger.error(f"图像处理服务初始化失败: {str(e)}")
            raise
    
    async def _check_gpu_support(self):
        """检查GPU支持"""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"检测到 {device_count} 个GPU设备，主设备: {device_name}")
            else:
                logger.warning("未检测到GPU设备，将使用CPU处理")
                settings.use_gpu = False
        except Exception as e:
            logger.warning(f"GPU检查失败: {str(e)}，将使用CPU处理")
            settings.use_gpu = False
    
    async def _init_super_resolution_models(self):
        """初始化超分辨率模型"""
        try:
            # 这里可以加载预训练的超分辨率模型
            # 例如 ESRGAN, Real-ESRGAN 等
            logger.info("超分辨率模型初始化（占位符）")
        except Exception as e:
            logger.warning(f"超分辨率模型加载失败: {str(e)}")
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取结果"""
        if not settings.enable_cache:
            return None
        return self._cache.get(key)
    
    def _set_cache(self, key: str, value: Any):
        """设置缓存"""
        if not settings.enable_cache:
            return
        
        # 简单的LRU缓存实现
        if len(self._cache) >= self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    # ============ 图像加载和保存 ============
    
    def load_image(self, image_path_or_data: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """加载图像"""
        try:
            if isinstance(image_path_or_data, str):
                # 从文件路径加载
                image = cv2.imread(image_path_or_data)
                if image is None:
                    raise ValueError(f"无法加载图像: {image_path_or_data}")
                # OpenCV默认是BGR，转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_path_or_data, bytes):
                # 从字节数据加载
                nparr = np.frombuffer(image_path_or_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("无法解码图像数据")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_path_or_data, np.ndarray):
                # 直接使用numpy数组
                image = image_path_or_data
            else:
                raise ValueError(f"不支持的图像数据类型: {type(image_path_or_data)}")
            
            return image
            
        except Exception as e:
            logger.error(f"图像加载失败: {str(e)}")
            raise
    
    def save_image(
        self,
        image: np.ndarray,
        output_path: str,
        format: str = "jpeg",
        quality: int = 95
    ) -> bool:
        """保存图像"""
        try:
            # 确保图像数据类型正确
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # 转换颜色空间（RGB到BGR）
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 保存参数
            save_params = []
            
            if format.lower() in ["jpg", "jpeg"]:
                save_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif format.lower() == "png":
                compression = int((100 - quality) / 10)
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
            elif format.lower() == "webp":
                save_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            
            # 保存图像
            success = cv2.imwrite(output_path, image, save_params)
            
            if not success:
                raise ValueError(f"图像保存失败: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"图像保存失败: {str(e)}")
            return False
    
    def get_image_info(self, image: np.ndarray) -> ImageInfo:
        """获取图像基本信息"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # 估算文件大小（近似值）
        file_size = image.nbytes
        
        # 确定颜色模式
        if channels == 1:
            color_mode = "grayscale"
        elif channels == 3:
            color_mode = "RGB"
        elif channels == 4:
            color_mode = "RGBA"
        else:
            color_mode = f"{channels}channel"
        
        return ImageInfo(
            size=ImageSize(width=width, height=height),
            channels=channels,
            format="array",  # 内存中的数组格式
            file_size=file_size,
            color_mode=color_mode
        )
    
    # ============ 图像质量评估 ============
    
    async def assess_image_quality(
        self,
        image: np.ndarray,
        reference_image: Optional[np.ndarray] = None
    ) -> QualityMetrics:
        """评估图像质量"""
        
        cache_key = f"quality_{hash(image.tobytes())}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
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
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # 噪声水平评估
            noise_level = self._estimate_noise_level(gray)
            
            # 模糊程度评估
            blur_level = self._estimate_blur_level(gray)
            
            # 倾斜角度检测
            skew_angle = self._detect_skew_angle(gray)
            
            # 文本区域占比
            text_region_ratio = self._estimate_text_region_ratio(gray)
            
            # 整体质量评分
            quality_weights = settings.overall_quality_weights
            overall_quality = (
                brightness_score * quality_weights["brightness"] +
                contrast_score * quality_weights["contrast"] +
                sharpness_score * quality_weights["sharpness"] +
                (1 - noise_level) * quality_weights["noise_level"] +
                (1 - blur_level) * quality_weights["blur_level"]
            )
            
            result = QualityMetrics(
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                sharpness_score=sharpness_score,
                noise_level=noise_level,
                blur_level=blur_level,
                skew_angle=skew_angle,
                text_region_ratio=text_region_ratio,
                overall_quality=overall_quality
            )
            
            # 缓存结果
            self._set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"图像质量评估失败: {str(e)}")
            raise
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """估计图像噪声水平"""
        # 使用高斯滤波后的差异来估计噪声
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        noise = cv2.absdiff(image, blurred)
        noise_level = np.mean(noise) / 255.0
        return min(noise_level, 1.0)
    
    def _estimate_blur_level(self, image: np.ndarray) -> float:
        """估计图像模糊程度"""
        # 使用Sobel算子检测边缘强度
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(sobelx**2 + sobely**2)
        
        # 边缘强度越低，模糊程度越高
        blur_level = 1.0 - min(np.mean(edge_strength) / 100.0, 1.0)
        return blur_level
    
    def _detect_skew_angle(self, image: np.ndarray) -> float:
        """检测图像倾斜角度"""
        try:
            # 边缘检测
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # 霍夫变换检测直线
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=settings.hough_threshold)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:20]:  # 只取前20条线
                    angle = theta * 180 / np.pi - 90
                    if abs(angle) < settings.deskew_max_angle:
                        angles.append(angle)
                
                if angles:
                    return np.median(angles)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_text_region_ratio(self, image: np.ndarray) -> float:
        """估计文本区域占比"""
        try:
            # 二值化
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作连接文本区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 计算文本像素占比
            text_pixels = np.sum(processed == 0)  # 假设文本是黑色
            total_pixels = processed.size
            
            return text_pixels / total_pixels
            
        except Exception:
            return 0.0
    
    # ============ 图像增强功能 ============
    
    async def enhance_image(
        self,
        image: np.ndarray,
        config: EnhanceConfig
    ) -> np.ndarray:
        """图像增强"""
        enhanced = image.copy()
        
        try:
            # 亮度调整
            if config.adjust_brightness:
                enhanced = self._adjust_brightness(enhanced, config.brightness_factor)
            
            # 对比度调整
            if config.adjust_contrast:
                enhanced = self._adjust_contrast(enhanced, config.contrast_factor)
            
            # 伽马校正
            if config.gamma_correction:
                enhanced = self._gamma_correction(enhanced, config.gamma)
            
            # 直方图均衡化
            if config.histogram_equalization:
                enhanced = self._histogram_equalization(enhanced)
            
            # CLAHE
            if config.clahe:
                enhanced = self._apply_clahe(enhanced, config.clahe_clip_limit, config.clahe_tile_size)
            
            # 锐化
            if config.sharpen:
                enhanced = self._sharpen_image(enhanced, config.sharpen_strength)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"图像增强失败: {str(e)}")
            raise
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整亮度"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整对比度"""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """伽马校正"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """直方图均衡化"""
        if len(image.shape) == 3:
            # 彩色图像：在LAB空间进行
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # 灰度图像
            return cv2.equalizeHist(image)
    
    def _apply_clahe(self, image: np.ndarray, clip_limit: float, tile_size: Tuple[int, int]) -> np.ndarray:
        """应用CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            return clahe.apply(image)
    
    def _sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """图像锐化"""
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
        return cv2.filter2D(image, -1, kernel)
    
    # ============ 去噪功能 ============
    
    async def denoise_image(
        self,
        image: np.ndarray,
        config: DenoiseConfig
    ) -> np.ndarray:
        """图像去噪"""
        try:
            if config.method == DenoiseMethod.BILATERAL:
                return self._bilateral_filter(image, config)
            elif config.method == DenoiseMethod.GAUSSIAN:
                return self._gaussian_filter(image, config)
            elif config.method == DenoiseMethod.MEDIAN:
                return self._median_filter(image, config)
            elif config.method == DenoiseMethod.NLM:
                return self._nlm_filter(image, config)
            else:
                raise ValueError(f"不支持的去噪方法: {config.method}")
                
        except Exception as e:
            logger.error(f"图像去噪失败: {str(e)}")
            raise
    
    def _bilateral_filter(self, image: np.ndarray, config: DenoiseConfig) -> np.ndarray:
        """双边滤波去噪"""
        return cv2.bilateralFilter(image, config.bilateral_d, 
                                  config.bilateral_sigma_color, 
                                  config.bilateral_sigma_space)
    
    def _gaussian_filter(self, image: np.ndarray, config: DenoiseConfig) -> np.ndarray:
        """高斯滤波去噪"""
        kernel_size = (config.gaussian_kernel_size, config.gaussian_kernel_size)
        return cv2.GaussianBlur(image, kernel_size, config.gaussian_sigma)
    
    def _median_filter(self, image: np.ndarray, config: DenoiseConfig) -> np.ndarray:
        """中值滤波去噪"""
        return cv2.medianBlur(image, config.median_kernel_size)
    
    def _nlm_filter(self, image: np.ndarray, config: DenoiseConfig) -> np.ndarray:
        """非局部均值去噪"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, config.nlm_h, config.nlm_h,
                config.nlm_template_size, config.nlm_search_size
            )
        else:
            return cv2.fastNlMeansDenoising(
                image, None, config.nlm_h,
                config.nlm_template_size, config.nlm_search_size
            )
    
    # ============ 倾斜校正功能 ============
    
    async def deskew_image(
        self,
        image: np.ndarray,
        config: DeskewConfig
    ) -> Tuple[np.ndarray, float]:
        """图像倾斜校正"""
        try:
            if config.auto_detect:
                # 自动检测倾斜角度
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                angle = self._detect_skew_angle(gray)
            else:
                # 使用手动指定的角度
                angle = config.manual_angle or 0.0
            
            # 角度阈值检查
            if abs(angle) < config.angle_threshold:
                return image, 0.0
            
            # 执行旋转校正
            corrected = self._rotate_image(image, angle, config.interpolation)
            
            return corrected, angle
            
        except Exception as e:
            logger.error(f"图像倾斜校正失败: {str(e)}")
            raise
    
    def _rotate_image(self, image: np.ndarray, angle: float, interpolation: InterpolationMethod) -> np.ndarray:
        """旋转图像"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # 获取插值方法
        interp_map = self.engine_config["opencv"]["interpolation_methods"]
        cv_interpolation = interp_map.get(interpolation.value, cv2.INTER_CUBIC)
        
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
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv_interpolation,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    # ============ 尺寸调整功能 ============
    
    async def resize_image(
        self,
        image: np.ndarray,
        config: ResizeConfig
    ) -> np.ndarray:
        """图像尺寸调整"""
        try:
            height, width = image.shape[:2]
            
            if config.method == ResizeMethod.SCALE:
                new_width = int(width * config.scale_factor)
                new_height = int(height * config.scale_factor)
            elif config.method == ResizeMethod.FIXED_SIZE:
                new_width = config.target_width or width
                new_height = config.target_height or height
                
                if config.maintain_aspect_ratio:
                    # 保持宽高比
                    aspect_ratio = width / height
                    if new_width / new_height > aspect_ratio:
                        new_width = int(new_height * aspect_ratio)
                    else:
                        new_height = int(new_width / aspect_ratio)
            elif config.method == ResizeMethod.MAX_DIMENSION:
                max_dim = max(height, width)
                if max_dim > config.max_dimension:
                    scale_factor = config.max_dimension / max_dim
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                else:
                    return image  # 不需要调整
            else:
                raise ValueError(f"不支持的尺寸调整方法: {config.method}")
            
            # 获取插值方法
            interp_map = self.engine_config["opencv"]["interpolation_methods"]
            cv_interpolation = interp_map.get(config.interpolation.value, cv2.INTER_CUBIC)
            
            # 调整尺寸
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv_interpolation)
            
            return resized
            
        except Exception as e:
            logger.error(f"图像尺寸调整失败: {str(e)}")
            raise
    
    # ============ 格式转换功能 ============
    
    async def convert_format(
        self,
        image: np.ndarray,
        config: FormatConvertConfig
    ) -> np.ndarray:
        """图像格式转换"""
        try:
            if config.target_format == ImageFormat.JPEG:
                # JPEG不支持透明通道
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = image[:,:,:3]
                return image
            elif config.target_format == ImageFormat.PNG:
                # PNG支持透明通道，保持原样
                return image
            elif config.target_format == ImageFormat.TIFF:
                return image
            elif config.target_format == ImageFormat.BMP:
                # BMP不支持透明通道
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = image[:,:,:3]
                return image
            elif config.target_format == ImageFormat.WEBP:
                return image
            else:
                # 灰度或二值化转换
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                if config.enable_binarization:
                    if config.threshold_method == ThresholdMethod.OTSU:
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    elif config.threshold_method == ThresholdMethod.ADAPTIVE:
                        binary = cv2.adaptiveThreshold(
                            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                        )
                    else:
                        _, binary = cv2.threshold(gray, config.threshold_value, 255, cv2.THRESH_BINARY)
                    return binary
                else:
                    return gray
                    
        except Exception as e:
            logger.error(f"格式转换失败: {str(e)}")
            raise
    
    # ============ 自动增强功能 ============
    
    async def auto_enhance_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """自动图像增强"""
        try:
            # 评估图像质量
            quality = await self.assess_image_quality(image)
            
            # 根据质量评估结果自动确定增强参数
            enhance_config = EnhanceConfig()
            
            # 亮度调整
            if quality.brightness_score < 0.3:
                enhance_config.adjust_brightness = True
                enhance_config.brightness_factor = 1.3
            elif quality.brightness_score > 0.8:
                enhance_config.adjust_brightness = True
                enhance_config.brightness_factor = 0.8
            
            # 对比度调整
            if quality.contrast_score < 0.2:
                enhance_config.adjust_contrast = True
                enhance_config.contrast_factor = 1.5
            
            # CLAHE增强
            if quality.contrast_score < 0.3:
                enhance_config.clahe = True
            
            # 锐化
            if quality.sharpness_score < 0.5:
                enhance_config.sharpen = True
                enhance_config.sharpen_strength = 1.2
            
            # 应用增强
            enhanced = await self.enhance_image(image, enhance_config)
            
            # 如果需要，还可以应用去噪
            if quality.noise_level > 0.3:
                denoise_config = DenoiseConfig(method=DenoiseMethod.BILATERAL)
                enhanced = await self.denoise_image(enhanced, denoise_config)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"自动图像增强失败: {str(e)}")
            raise
    
    # ============ 批量处理功能 ============
    
    async def batch_process_images(
        self,
        image_paths: List[str],
        processing_type: ProcessingType,
        config: Optional[ProcessingConfig] = None
    ) -> List[Dict[str, Any]]:
        """批量图像处理"""
        results = []
        semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
        
        async def process_single_image(image_path: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    # 加载图像
                    image = self.load_image(image_path)
                    
                    # 根据处理类型执行相应操作
                    if processing_type == ProcessingType.ENHANCE and config and config.enhance:
                        result_image = await self.enhance_image(image, config.enhance)
                    elif processing_type == ProcessingType.DENOISE and config and config.denoise:
                        result_image = await self.denoise_image(image, config.denoise)
                    elif processing_type == ProcessingType.DESKEW and config and config.deskew:
                        result_image, angle = await self.deskew_image(image, config.deskew)
                    elif processing_type == ProcessingType.RESIZE and config and config.resize:
                        result_image = await self.resize_image(image, config.resize)
                    elif processing_type == ProcessingType.FORMAT_CONVERT and config and config.format_convert:
                        result_image = await self.convert_format(image, config.format_convert)
                    elif processing_type == ProcessingType.AUTO_ENHANCE:
                        result_image = await self.auto_enhance_image(image)
                    elif processing_type == ProcessingType.QUALITY_ASSESSMENT:
                        quality = await self.assess_image_quality(image)
                        return {"success": True, "image_path": image_path, "quality": quality.dict()}
                    else:
                        raise ValueError(f"不支持的处理类型: {processing_type}")
                    
                    return {
                        "success": True,
                        "image_path": image_path,
                        "result_image": result_image,
                        "original_info": self.get_image_info(image).dict(),
                        "processed_info": self.get_image_info(result_image).dict()
                    }
                    
                except Exception as e:
                    logger.error(f"处理图像失败 {image_path}: {str(e)}")
                    return {
                        "success": False,
                        "image_path": image_path,
                        "error": str(e)
                    }
        
        # 并发处理所有图像
        results = await asyncio.gather(
            *[process_single_image(path) for path in image_paths],
            return_exceptions=True
        )
        
        return results
    
    def get_available_engines(self) -> List[ProcessingEngineInfo]:
        """获取可用的图像处理引擎信息"""
        engines = []
        
        # OpenCV引擎
        engines.append(ProcessingEngineInfo(
            name="opencv",
            version=cv2.__version__,
            supported_formats=settings.supported_input_formats,
            supported_operations=self.supported_operations,
            gpu_support=settings.use_gpu,
            description="开源计算机视觉库，功能全面"
        ))
        
        # PIL引擎
        engines.append(ProcessingEngineInfo(
            name="pillow",
            version="10.1.0",
            supported_formats=settings.supported_input_formats,
            supported_operations=["enhance", "resize", "format_convert"],
            gpu_support=False,
            description="Python图像处理库，简单易用"
        ))
        
        # scikit-image引擎
        engines.append(ProcessingEngineInfo(
            name="skimage",
            version="0.22.0",
            supported_formats=settings.supported_input_formats,
            supported_operations=["enhance", "denoise", "quality_assessment"],
            gpu_support=False,
            description="基于numpy的科学图像处理库"
        ))
        
        return engines
    
    async def cleanup(self):
        """清理资源"""
        logger.info("图像处理服务清理资源...")
        
        # 清理模型
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'close'):
                    await model.close()
                elif hasattr(model, '__del__'):
                    del model
            except:
                pass
        
        self.models.clear()
        self._cache.clear()
        
        logger.info("图像处理服务资源清理完成")