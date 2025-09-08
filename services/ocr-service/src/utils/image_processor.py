"""
图像预处理工具模块

提供古籍文档图像的专业化预处理功能，包括降噪、增强对比度、
去倾斜、二值化等操作，专门针对古代汉字识别进行优化。

主要功能：
- 自适应图像去噪和增强
- 古籍文档倾斜校正
- 多种二值化算法支持
- 文字区域检测和优化
- 批量图像预处理

Author: OCR开发团队
Created: 2025-01-15
Version: 1.0.0
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import Tuple, Optional, Dict, Any, Union
import logging
from enum import Enum
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from skimage import morphology, segmentation, measure, filters
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import math

# 配置日志
logger = logging.getLogger(__name__)


class BinarizationMethod(str, Enum):
    """二值化方法枚举"""
    OTSU = "otsu"                    # OTSU自适应阈值
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"  # 自适应高斯阈值
    ADAPTIVE_MEAN = "adaptive_mean"          # 自适应均值阈值
    SAUVOLA = "sauvola"             # Sauvola局部阈值
    NIBLACK = "niblack"             # Niblack局部阈值
    TRIANGLE = "triangle"            # Triangle阈值


class DenoiseMethod(str, Enum):
    """降噪方法枚举"""
    GAUSSIAN = "gaussian"           # 高斯模糊
    MEDIAN = "median"              # 中值滤波
    BILATERAL = "bilateral"        # 双边滤波
    NON_LOCAL_MEANS = "nlm"       # 非局部均值去噪
    MORPHOLOGICAL = "morphological" # 形态学去噪
    WIENER = "wiener"             # 维纳滤波


class ImageProcessor:
    """
    图像预处理器
    
    专门为古籍OCR优化的图像预处理工具类，支持各种
    图像增强、去噪、校正等功能。
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化图像处理器
        
        Args:
            max_workers: 线程池最大工作线程数
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 预设的处理参数配置
        self.default_params = {
            'denoise': {
                'method': DenoiseMethod.BILATERAL,
                'gaussian_kernel': 5,
                'median_kernel': 5,
                'bilateral_d': 9,
                'bilateral_sigma_color': 75,
                'bilateral_sigma_space': 75,
                'nlm_h': 10,
                'nlm_template_window': 7,
                'nlm_search_window': 21
            },
            'contrast': {
                'alpha': 1.2,  # 对比度增益
                'beta': 10,    # 亮度偏移
                'gamma': 1.0,  # 伽马校正
                'histogram_equalization': False
            },
            'deskew': {
                'angle_threshold': 0.1,  # 角度阈值(度)
                'line_detection_threshold': 100,
                'min_line_length': 100,
                'max_line_gap': 10
            },
            'binarization': {
                'method': BinarizationMethod.OTSU,
                'block_size': 15,
                'c_value': 2,
                'sauvola_k': 0.2,
                'sauvola_r': 128
            },
            'morphology': {
                'kernel_size': (3, 3),
                'erosion_iterations': 1,
                'dilation_iterations': 1
            }
        }
    
    async def process_image_async(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        异步图像预处理主函数
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）
            config: 处理配置参数
            
        Returns:
            Tuple[处理后图像数组, 处理统计信息]
        """
        try:
            logger.info(f"开始异步处理图像: {image_path}")
            
            # 合并配置参数
            processing_config = self._merge_config(config or {})
            
            # 异步读取图像
            image = await self._load_image_async(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 记录原始图像信息
            original_shape = image.shape
            processing_stats = {
                'original_size': original_shape,
                'processing_steps': [],
                'processing_time': {},
                'quality_metrics': {}
            }
            
            # 执行预处理步骤
            processed_image = await self._execute_processing_pipeline(
                image, processing_config, processing_stats
            )
            
            # 保存处理后的图像
            if output_path:
                await self._save_image_async(processed_image, output_path)
                logger.info(f"处理后图像已保存至: {output_path}")
            
            logger.info(f"图像处理完成，处理步骤: {len(processing_stats['processing_steps'])}")
            return processed_image, processing_stats
            
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}")
            raise
    
    async def _load_image_async(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        异步加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像数组或None
        """
        def _load_sync():
            try:
                # 支持多种图像格式
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is not None:
                    # 转换颜色空间从BGR到RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image
                
                # 尝试使用PIL加载
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3 and image.shape[2] == 4:
                    # 处理RGBA图像
                    image = image[:, :, :3]
                
                return image
                
            except Exception as e:
                logger.error(f"图像加载失败: {str(e)}")
                return None
        
        # 在线程池中执行同步IO操作
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_sync
        )
    
    async def _save_image_async(
        self, 
        image: np.ndarray, 
        output_path: Union[str, Path]
    ) -> bool:
        """
        异步保存图像文件
        
        Args:
            image: 图像数组
            output_path: 输出路径
            
        Returns:
            是否保存成功
        """
        def _save_sync():
            try:
                # 确保输出目录存在
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存图像
                if len(image.shape) == 3:
                    # RGB图像转换为BGR保存
                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), bgr_image)
                else:
                    # 灰度图像直接保存
                    cv2.imwrite(str(output_path), image)
                
                return True
                
            except Exception as e:
                logger.error(f"图像保存失败: {str(e)}")
                return False
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _save_sync
        )
    
    async def _execute_processing_pipeline(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        执行完整的图像处理流水线
        
        Args:
            image: 输入图像
            config: 处理配置
            stats: 统计信息记录
            
        Returns:
            处理后的图像
        """
        processed_image = image.copy()
        
        # 1. 转换为灰度图像
        if config.get('grayscale', True):
            processed_image = await self._to_grayscale_async(processed_image)
            stats['processing_steps'].append('grayscale')
        
        # 2. 图像去噪
        if config.get('denoise', True):
            processed_image = await self._denoise_async(
                processed_image, config.get('denoise_params', {})
            )
            stats['processing_steps'].append('denoise')
        
        # 3. 增强对比度
        if config.get('enhance_contrast', True):
            processed_image = await self._enhance_contrast_async(
                processed_image, config.get('contrast_params', {})
            )
            stats['processing_steps'].append('enhance_contrast')
        
        # 4. 去倾斜校正
        if config.get('deskew', True):
            processed_image, skew_angle = await self._deskew_async(
                processed_image, config.get('deskew_params', {})
            )
            stats['processing_steps'].append('deskew')
            stats['skew_angle'] = skew_angle
        
        # 5. 二值化
        if config.get('binarize', False):
            processed_image = await self._binarize_async(
                processed_image, config.get('binarization_params', {})
            )
            stats['processing_steps'].append('binarize')
        
        # 6. 形态学操作
        if config.get('morphological_operations', False):
            processed_image = await self._morphological_operations_async(
                processed_image, config.get('morphology_params', {})
            )
            stats['processing_steps'].append('morphological')
        
        # 7. 图像缩放
        if config.get('resize', False):
            scale_factor = config.get('scale_factor', 1.0)
            if scale_factor != 1.0:
                processed_image = await self._resize_async(processed_image, scale_factor)
                stats['processing_steps'].append('resize')
                stats['scale_factor'] = scale_factor
        
        return processed_image
    
    async def _to_grayscale_async(self, image: np.ndarray) -> np.ndarray:
        """
        异步转换为灰度图像
        
        Args:
            image: 输入图像
            
        Returns:
            灰度图像
        """
        def _grayscale_sync():
            if len(image.shape) == 3:
                # 使用加权平均转换为灰度
                # 对中文文字优化的权重
                weights = np.array([0.299, 0.587, 0.114])
                return np.dot(image[..., :3], weights).astype(np.uint8)
            return image
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _grayscale_sync
        )
    
    async def _denoise_async(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        异步图像去噪
        
        Args:
            image: 输入图像
            params: 去噪参数
            
        Returns:
            去噪后图像
        """
        def _denoise_sync():
            merged_params = {**self.default_params['denoise'], **params}
            method = merged_params.get('method', DenoiseMethod.BILATERAL)
            
            if method == DenoiseMethod.GAUSSIAN:
                kernel_size = merged_params['gaussian_kernel']
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            elif method == DenoiseMethod.MEDIAN:
                kernel_size = merged_params['median_kernel']
                return cv2.medianBlur(image, kernel_size)
            
            elif method == DenoiseMethod.BILATERAL:
                return cv2.bilateralFilter(
                    image,
                    merged_params['bilateral_d'],
                    merged_params['bilateral_sigma_color'],
                    merged_params['bilateral_sigma_space']
                )
            
            elif method == DenoiseMethod.NON_LOCAL_MEANS:
                if len(image.shape) == 3:
                    return cv2.fastNlMeansDenoising(
                        image,
                        None,
                        merged_params['nlm_h'],
                        merged_params['nlm_template_window'],
                        merged_params['nlm_search_window']
                    )
                else:
                    return cv2.fastNlMeansDenoising(
                        image,
                        None,
                        merged_params['nlm_h'],
                        merged_params['nlm_template_window'],
                        merged_params['nlm_search_window']
                    )
            
            else:
                # 默认使用双边滤波
                return cv2.bilateralFilter(image, 9, 75, 75)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _denoise_sync
        )
    
    async def _enhance_contrast_async(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        异步增强图像对比度
        
        Args:
            image: 输入图像
            params: 对比度增强参数
            
        Returns:
            增强后图像
        """
        def _enhance_sync():
            merged_params = {**self.default_params['contrast'], **params}
            
            # 基础对比度和亮度调整
            alpha = merged_params['alpha']
            beta = merged_params['beta']
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # 伽马校正
            gamma = merged_params['gamma']
            if gamma != 1.0:
                inv_gamma = 1.0 / gamma
                table = np.array([
                    ((i / 255.0) ** inv_gamma) * 255 
                    for i in np.arange(0, 256)
                ]).astype("uint8")
                enhanced = cv2.LUT(enhanced, table)
            
            # 直方图均衡化
            if merged_params.get('histogram_equalization', False):
                if len(enhanced.shape) == 3:
                    # 彩色图像在YUV空间进行直方图均衡化
                    yuv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YUV)
                    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
                else:
                    # 灰度图像直接均衡化
                    enhanced = cv2.equalizeHist(enhanced)
            
            return enhanced
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _enhance_sync
        )
    
    async def _deskew_async(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, float]:
        """
        异步图像去倾斜校正
        
        Args:
            image: 输入图像
            params: 去倾斜参数
            
        Returns:
            Tuple[校正后图像, 倾斜角度]
        """
        def _deskew_sync():
            merged_params = {**self.default_params['deskew'], **params}
            
            # 转换为灰度图像用于边缘检测
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # 边缘检测
            edges = canny(gray, low_threshold=50, high_threshold=150)
            
            # 霍夫直线变换检测文本行
            tested_angles = np.linspace(-np.pi / 4, np.pi / 4, 360, endpoint=False)
            h, theta, d = hough_line(edges, theta=tested_angles)
            
            # 找到最强的直线
            hough_peaks = hough_line_peaks(h, theta, d, num_peaks=20)
            
            if len(hough_peaks[1]) == 0:
                return image, 0.0
            
            # 计算平均倾斜角度
            angles = hough_peaks[1]
            most_common_angle = np.median(angles)
            skew_angle = np.degrees(most_common_angle)
            
            # 检查角度是否超过阈值
            angle_threshold = merged_params['angle_threshold']
            if abs(skew_angle) < angle_threshold:
                return image, skew_angle
            
            # 执行旋转校正
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -skew_angle, 1.0)
            
            # 计算新的边界框
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))
            
            # 调整旋转矩阵的平移部分
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # 应用旋转
            rotated = cv2.warpAffine(
                image, 
                rotation_matrix, 
                (new_width, new_height), 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return rotated, skew_angle
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _deskew_sync
        )
    
    async def _binarize_async(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        异步图像二值化
        
        Args:
            image: 输入图像
            params: 二值化参数
            
        Returns:
            二值化图像
        """
        def _binarize_sync():
            merged_params = {**self.default_params['binarization'], **params}
            method = merged_params.get('method', BinarizationMethod.OTSU)
            
            # 确保是灰度图像
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            if method == BinarizationMethod.OTSU:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
            
            elif method == BinarizationMethod.ADAPTIVE_GAUSSIAN:
                block_size = merged_params['block_size']
                c_value = merged_params['c_value']
                return cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, block_size, c_value
                )
            
            elif method == BinarizationMethod.ADAPTIVE_MEAN:
                block_size = merged_params['block_size']
                c_value = merged_params['c_value']
                return cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, block_size, c_value
                )
            
            elif method == BinarizationMethod.TRIANGLE:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
                return binary
            
            else:
                # 默认使用OTSU
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _binarize_sync
        )
    
    async def _morphological_operations_async(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        异步形态学操作
        
        Args:
            image: 输入图像
            params: 形态学操作参数
            
        Returns:
            处理后图像
        """
        def _morphology_sync():
            merged_params = {**self.default_params['morphology'], **params}
            
            kernel_size = merged_params['kernel_size']
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            
            # 腐蚀操作
            erosion_iter = merged_params['erosion_iterations']
            if erosion_iter > 0:
                image_processed = cv2.erode(image, kernel, iterations=erosion_iter)
            else:
                image_processed = image.copy()
            
            # 膨胀操作
            dilation_iter = merged_params['dilation_iterations']
            if dilation_iter > 0:
                image_processed = cv2.dilate(image_processed, kernel, iterations=dilation_iter)
            
            return image_processed
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _morphology_sync
        )
    
    async def _resize_async(
        self, 
        image: np.ndarray, 
        scale_factor: float
    ) -> np.ndarray:
        """
        异步图像缩放
        
        Args:
            image: 输入图像
            scale_factor: 缩放因子
            
        Returns:
            缩放后图像
        """
        def _resize_sync():
            height, width = image.shape[:2]
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            return cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_CUBIC
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _resize_sync
        )
    
    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并用户配置与默认配置
        
        Args:
            user_config: 用户提供的配置
            
        Returns:
            合并后的配置
        """
        merged_config = {}
        
        # 合并各个步骤的参数
        for param_key in ['denoise', 'contrast', 'deskew', 'binarization', 'morphology']:
            param_name = f"{param_key}_params"
            if param_name in user_config:
                merged_config[param_name] = {
                    **self.default_params[param_key],
                    **user_config[param_name]
                }
            else:
                merged_config[param_name] = self.default_params[param_key]
        
        # 合并其他配置选项
        config_keys = [
            'grayscale', 'denoise', 'enhance_contrast', 'deskew', 
            'binarize', 'morphological_operations', 'resize', 'scale_factor'
        ]
        
        for key in config_keys:
            if key in user_config:
                merged_config[key] = user_config[key]
            else:
                # 设置默认值
                defaults = {
                    'grayscale': True,
                    'denoise': True,
                    'enhance_contrast': True,
                    'deskew': True,
                    'binarize': False,
                    'morphological_operations': False,
                    'resize': False,
                    'scale_factor': 1.0
                }
                merged_config[key] = defaults.get(key, False)
        
        return merged_config
    
    async def batch_process_images(
        self,
        image_paths: list,
        output_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 4
    ) -> Dict[str, Any]:
        """
        批量处理图像文件
        
        Args:
            image_paths: 图像文件路径列表
            output_dir: 输出目录
            config: 处理配置
            max_concurrent: 最大并发处理数
            
        Returns:
            批量处理统计信息
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_single(image_path):
            async with semaphore:
                try:
                    output_file = output_path / f"processed_{Path(image_path).name}"
                    processed_image, stats = await self.process_image_async(
                        image_path, output_file, config
                    )
                    return {
                        'input_path': str(image_path),
                        'output_path': str(output_file),
                        'success': True,
                        'stats': stats
                    }
                except Exception as e:
                    logger.error(f"批量处理失败 {image_path}: {str(e)}")
                    return {
                        'input_path': str(image_path),
                        'output_path': None,
                        'success': False,
                        'error': str(e)
                    }
        
        # 并发处理所有图像
        tasks = [process_single(path) for path in image_paths]
        results = await asyncio.gather(*tasks)
        
        # 统计处理结果
        success_count = sum(1 for result in results if result['success'])
        total_count = len(results)
        
        batch_stats = {
            'total_images': total_count,
            'successful_processed': success_count,
            'failed_processed': total_count - success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'results': results
        }
        
        logger.info(f"批量处理完成: {success_count}/{total_count} 成功")
        return batch_stats
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# 便捷函数
async def preprocess_image(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    便捷的图像预处理函数
    
    Args:
        image_path: 输入图像路径
        output_path: 输出路径（可选）
        **kwargs: 处理配置参数
        
    Returns:
        Tuple[处理后图像, 处理统计信息]
    """
    processor = ImageProcessor()
    try:
        return await processor.process_image_async(image_path, output_path, kwargs)
    finally:
        del processor


def create_ancient_text_config() -> Dict[str, Any]:
    """
    创建专门用于古籍文本识别的图像处理配置
    
    Returns:
        古籍优化配置字典
    """
    return {
        'grayscale': True,
        'denoise': True,
        'enhance_contrast': True,
        'deskew': True,
        'binarize': True,
        'morphological_operations': True,
        'denoise_params': {
            'method': DenoiseMethod.BILATERAL,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75
        },
        'contrast_params': {
            'alpha': 1.3,
            'beta': 15,
            'histogram_equalization': True
        },
        'binarization_params': {
            'method': BinarizationMethod.ADAPTIVE_GAUSSIAN,
            'block_size': 21,
            'c_value': 8
        },
        'morphology_params': {
            'kernel_size': (2, 2),
            'erosion_iterations': 1,
            'dilation_iterations': 1
        }
    }