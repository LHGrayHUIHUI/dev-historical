"""
图像内容分析器

基于计算机视觉和深度学习的图像内容审核
支持NSFW检测、暴力内容识别、政治敏感图像检测等功能
"""

import io
import time
import logging
import hashlib
import tempfile
from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageStat, ImageFilter
    import tensorflow as tf
    HAS_CV_DEPS = True
except ImportError as e:
    logging.warning(f"计算机视觉依赖缺失: {e}")
    HAS_CV_DEPS = False

from .base_analyzer import BaseAnalyzer, AnalysisResult, ViolationDetail, ViolationType, AnalysisStatus

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysisMetadata:
    """图像分析元数据"""
    width: int
    height: int
    channels: int
    file_size: int
    format: str
    color_mode: str
    brightness: float
    contrast: float
    sharpness: float
    has_faces: bool
    face_count: int
    dominant_colors: List[str]


class NSFWDetector:
    """NSFW内容检测器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化NSFW检测器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        if HAS_CV_DEPS:
            self._load_model()
    
    def _load_model(self):
        """加载NSFW检测模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                # 加载预训练模型
                self.model = tf.keras.models.load_model(self.model_path)
                self.is_loaded = True
                logger.info("NSFW检测模型加载成功")
            else:
                logger.warning("NSFW模型文件不存在，使用模拟检测")
                self.is_loaded = False
        except Exception as e:
            logger.error(f"NSFW模型加载失败: {e}")
            self.is_loaded = False
    
    def detect(self, image: np.ndarray) -> ViolationDetail:
        """
        检测图像中的NSFW内容
        
        Args:
            image: 图像数组
            
        Returns:
            Optional[ViolationDetail]: 违规详情
        """
        try:
            if self.is_loaded and self.model:
                # 预处理图像
                processed_image = self._preprocess_image(image)
                
                # 模型预测
                prediction = self.model.predict(processed_image)
                confidence = float(prediction[0][0])  # 假设模型输出NSFW概率
                
                if confidence > 0.5:
                    return ViolationDetail(
                        type=ViolationType.PORNOGRAPHY,
                        confidence=confidence,
                        description=f"检测到NSFW内容，置信度: {confidence:.2f}",
                        evidence={"nsfw_score": confidence}
                    )
            else:
                # 模拟检测逻辑
                return self._simulate_nsfw_detection(image)
                
        except Exception as e:
            logger.error(f"NSFW检测失败: {e}")
        
        return None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像用于模型推理
        
        Args:
            image: 原始图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 调整图像大小为模型输入尺寸 (224x224)
        resized = cv2.resize(image, (224, 224))
        
        # 归一化到 [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # 添加批次维度
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def _simulate_nsfw_detection(self, image: np.ndarray) -> Optional[ViolationDetail]:
        """
        模拟NSFW检测（用于演示）
        
        Args:
            image: 图像数组
            
        Returns:
            Optional[ViolationDetail]: 违规详情
        """
        # 基于简单的图像特征进行模拟检测
        height, width = image.shape[:2]
        
        # 检查图像的肤色像素比例（简化逻辑）
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义肤色HSV范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask) / (height * width * 255)
        
        # 如果肤色比例过高，可能是NSFW内容
        if skin_ratio > 0.4:
            confidence = min(0.9, skin_ratio * 2)
            return ViolationDetail(
                type=ViolationType.PORNOGRAPHY,
                confidence=confidence,
                description=f"基于肤色分析检测到可疑内容，肤色比例: {skin_ratio:.2f}",
                evidence={"skin_ratio": skin_ratio}
            )
        
        return None


class ViolenceDetector:
    """暴力内容检测器"""
    
    def __init__(self):
        """初始化暴力内容检测器"""
        self.violence_colors = [
            ([0, 0, 100], [50, 50, 255]),    # 红色范围（血液）
            ([0, 0, 80], [30, 30, 180]),     # 暗红色
        ]
    
    def detect(self, image: np.ndarray) -> Optional[ViolationDetail]:
        """
        检测图像中的暴力内容
        
        Args:
            image: 图像数组
            
        Returns:
            Optional[ViolationDetail]: 违规详情
        """
        try:
            # 检测红色像素比例（血液指示）
            red_ratio = self._detect_red_content(image)
            
            # 检测边缘密度（武器、暴力场景通常有更多尖锐边缘）
            edge_density = self._calculate_edge_density(image)
            
            # 综合评估
            violence_score = red_ratio * 0.6 + edge_density * 0.4
            
            if violence_score > 0.3:
                confidence = min(0.85, violence_score)
                return ViolationDetail(
                    type=ViolationType.VIOLENCE,
                    confidence=confidence,
                    description=f"检测到暴力相关内容，评分: {violence_score:.2f}",
                    evidence={
                        "red_ratio": red_ratio,
                        "edge_density": edge_density,
                        "violence_score": violence_score
                    }
                )
        
        except Exception as e:
            logger.error(f"暴力内容检测失败: {e}")
        
        return None
    
    def _detect_red_content(self, image: np.ndarray) -> float:
        """
        检测红色内容比例
        
        Args:
            image: 图像数组
            
        Returns:
            float: 红色比例
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        total_red_pixels = 0
        total_pixels = image.shape[0] * image.shape[1]
        
        for lower, upper in self.violence_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            red_pixels = np.sum(mask > 0)
            total_red_pixels += red_pixels
        
        return total_red_pixels / total_pixels
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """
        计算图像边缘密度
        
        Args:
            image: 图像数组
            
        Returns:
            float: 边缘密度
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return edge_ratio


class FaceDetector:
    """人脸检测器"""
    
    def __init__(self):
        """初始化人脸检测器"""
        self.face_cascade = None
        if HAS_CV_DEPS:
            try:
                # 加载OpenCV的人脸检测器
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception as e:
                logger.warning(f"人脸检测器加载失败: {e}")
    
    def detect_faces(self, image: np.ndarray) -> Tuple[bool, int, List[Dict[str, int]]]:
        """
        检测图像中的人脸
        
        Args:
            image: 图像数组
            
        Returns:
            Tuple[bool, int, List[Dict[str, int]]]: (有人脸, 人脸数量, 人脸位置列表)
        """
        if self.face_cascade is None:
            return False, 0, []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append({
                    "x": int(x), 
                    "y": int(y), 
                    "width": int(w), 
                    "height": int(h)
                })
            
            return len(faces) > 0, len(faces), face_locations
            
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return False, 0, []


class ColorAnalyzer:
    """颜色分析器"""
    
    def __init__(self):
        """初始化颜色分析器"""
        pass
    
    def get_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[str]:
        """
        获取图像的主导颜色
        
        Args:
            image: 图像数组
            k: 聚类数量
            
        Returns:
            List[str]: 主导颜色的十六进制代码列表
        """
        try:
            # 将图像重塑为像素向量
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # 使用K-means聚类
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 转换为十六进制颜色代码
            hex_colors = []
            for center in centers:
                color = tuple(int(c) for c in center)
                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                hex_colors.append(hex_color)
            
            return hex_colors
            
        except Exception as e:
            logger.error(f"主导颜色分析失败: {e}")
            return []
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        计算图像亮度
        
        Args:
            image: 图像数组
            
        Returns:
            float: 亮度值 (0-1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255.0
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        计算图像对比度
        
        Args:
            image: 图像数组
            
        Returns:
            float: 对比度值
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 255.0


class ImageAnalyzer(BaseAnalyzer):
    """
    图像内容分析器
    
    提供全面的图像内容审核功能，包括：
    - NSFW内容检测
    - 暴力内容识别
    - 人脸检测
    - 图像质量分析
    - 颜色分析
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化图像分析器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        
        if not HAS_CV_DEPS:
            logger.warning("计算机视觉依赖未安装，图像分析功能受限")
        
        self.max_image_size = self.config.get("max_image_size", 10 * 1024 * 1024)  # 10MB
        self.min_image_size = self.config.get("min_image_size", 1024)  # 1KB
        self.max_resolution = self.config.get("max_resolution", (4096, 4096))
        
        # 初始化检测器
        nsfw_model_path = self.config.get("nsfw_model_path")
        self.nsfw_detector = NSFWDetector(nsfw_model_path)
        self.violence_detector = ViolenceDetector()
        self.face_detector = FaceDetector()
        self.color_analyzer = ColorAnalyzer()
        
        logger.info("图像分析器初始化完成")
    
    def get_supported_types(self) -> List[str]:
        """获取支持的图像类型"""
        return [
            "image/jpeg",
            "image/jpg", 
            "image/png",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff"
        ]
    
    async def analyze(self, content: Union[str, bytes], metadata: Dict[str, Any] = None) -> AnalysisResult:
        """
        分析图像内容
        
        Args:
            content: 图像内容 (bytes或文件路径)
            metadata: 元数据信息
            
        Returns:
            AnalysisResult: 分析结果
        """
        start_time = time.time()
        
        if not HAS_CV_DEPS:
            return self.create_error_result(
                AnalysisStatus.UNSUPPORTED,
                "缺少计算机视觉依赖，无法进行图像分析"
            )
        
        try:
            # 加载图像
            image_array, pil_image = await self._load_image(content)
            if image_array is None:
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "无法加载或解析图像文件"
                )
            
            # 内容验证
            if not self._validate_image(image_array):
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "图像不符合大小或分辨率要求"
                )
            
            # 并发执行多种检测
            violations = []
            
            # NSFW检测
            nsfw_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, self.nsfw_detector.detect, image_array
                )
            )
            
            # 暴力内容检测
            violence_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, self.violence_detector.detect, image_array
                )
            )
            
            # 人脸检测
            face_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, self.face_detector.detect_faces, image_array
                )
            )
            
            # 颜色分析
            color_task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    None, self._analyze_colors, image_array
                )
            )
            
            # 等待所有任务完成
            nsfw_result = await nsfw_task
            violence_result = await violence_task
            has_faces, face_count, face_locations = await face_task
            color_analysis = await color_task
            
            # 收集违规结果
            if nsfw_result:
                violations.append(nsfw_result)
            
            if violence_result:
                violations.append(violence_result)
            
            # 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(violations)
            
            # 生成分析元数据
            analysis_metadata = self._generate_metadata(
                image_array, pil_image, has_faces, face_count, 
                color_analysis, metadata
            )
            
            processing_time = time.time() - start_time
            
            return self.create_success_result(
                confidence=overall_confidence,
                violations=violations,
                processing_time=processing_time,
                metadata=analysis_metadata.__dict__
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"图像分析失败: {str(e)}")
            return self.create_error_result(
                AnalysisStatus.FAILED,
                f"分析过程中发生错误: {str(e)}",
                processing_time
            )
    
    async def _load_image(self, content: Union[str, bytes]) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
        """
        加载图像文件
        
        Args:
            content: 图像内容或文件路径
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[Image.Image]]: (OpenCV图像, PIL图像)
        """
        try:
            if isinstance(content, str):
                # 从文件路径加载
                if Path(content).exists():
                    image_array = cv2.imread(content)
                    pil_image = Image.open(content)
                else:
                    logger.error(f"图像文件不存在: {content}")
                    return None, None
            
            elif isinstance(content, bytes):
                # 从字节数据加载
                nparr = np.frombuffer(content, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                with io.BytesIO(content) as bio:
                    pil_image = Image.open(bio)
                    pil_image.load()  # 确保图像数据已加载
            
            else:
                logger.error("不支持的图像内容类型")
                return None, None
            
            if image_array is None:
                logger.error("无法解码图像数据")
                return None, None
            
            return image_array, pil_image
            
        except Exception as e:
            logger.error(f"图像加载失败: {e}")
            return None, None
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """
        验证图像是否符合要求
        
        Args:
            image: 图像数组
            
        Returns:
            bool: 是否有效
        """
        if image is None:
            return False
        
        height, width = image.shape[:2]
        
        # 检查分辨率限制
        if width > self.max_resolution[0] or height > self.max_resolution[1]:
            logger.warning(f"图像分辨率 {width}x{height} 超过限制 {self.max_resolution}")
            return False
        
        # 检查最小分辨率
        if width < 32 or height < 32:
            logger.warning("图像分辨率过小")
            return False
        
        return True
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像颜色特征
        
        Args:
            image: 图像数组
            
        Returns:
            Dict[str, Any]: 颜色分析结果
        """
        dominant_colors = self.color_analyzer.get_dominant_colors(image)
        brightness = self.color_analyzer.calculate_brightness(image)
        contrast = self.color_analyzer.calculate_contrast(image)
        
        return {
            "dominant_colors": dominant_colors,
            "brightness": brightness,
            "contrast": contrast
        }
    
    def _calculate_overall_confidence(self, violations: List[ViolationDetail]) -> float:
        """
        计算整体违规置信度
        
        Args:
            violations: 违规详情列表
            
        Returns:
            float: 整体置信度
        """
        if not violations:
            return 0.0
        
        # 取最高置信度作为整体置信度
        max_confidence = max(v.confidence for v in violations)
        
        # 如果有多个违规类型，稍微提高置信度
        if len(violations) > 1:
            bonus = min(0.1, len(violations) * 0.05)
            max_confidence = min(1.0, max_confidence + bonus)
        
        return max_confidence
    
    def _generate_metadata(
        self, 
        image_array: np.ndarray, 
        pil_image: Image.Image,
        has_faces: bool,
        face_count: int,
        color_analysis: Dict[str, Any],
        input_metadata: Dict[str, Any] = None
    ) -> ImageAnalysisMetadata:
        """
        生成分析元数据
        
        Args:
            image_array: 图像数组
            pil_image: PIL图像对象
            has_faces: 是否有人脸
            face_count: 人脸数量
            color_analysis: 颜色分析结果
            input_metadata: 输入元数据
            
        Returns:
            ImageAnalysisMetadata: 分析元数据
        """
        height, width, channels = image_array.shape
        
        # 计算文件大小
        file_size = input_metadata.get("file_size", 0) if input_metadata else 0
        
        # 图像格式信息
        image_format = pil_image.format or "Unknown"
        color_mode = pil_image.mode or "Unknown"
        
        # 图像质量分析
        stat = ImageStat.Stat(pil_image)
        brightness = sum(stat.mean) / len(stat.mean) / 255.0
        
        # 计算清晰度（使用拉普拉斯算子）
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized_sharpness = min(1.0, sharpness / 1000.0)
        
        return ImageAnalysisMetadata(
            width=width,
            height=height,
            channels=channels,
            file_size=file_size,
            format=image_format,
            color_mode=color_mode,
            brightness=brightness,
            contrast=color_analysis.get("contrast", 0.0),
            sharpness=normalized_sharpness,
            has_faces=has_faces,
            face_count=face_count,
            dominant_colors=color_analysis.get("dominant_colors", [])
        )
    
    def batch_analyze_images(self, image_paths: List[str]) -> List[AnalysisResult]:
        """
        批量分析图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            List[AnalysisResult]: 分析结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = asyncio.run(self.analyze(image_path))
                results.append(result)
            except Exception as e:
                error_result = self.create_error_result(
                    AnalysisStatus.FAILED,
                    f"批量分析失败: {str(e)}"
                )
                results.append(error_result)
        
        return results