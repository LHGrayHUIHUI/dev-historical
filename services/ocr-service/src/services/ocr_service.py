"""
OCR文本识别服务核心类

无状态OCR服务，专注于图像文本识别算法。
所有数据存储操作通过storage-service完成。

主要功能：
- 支持PaddleOCR、Tesseract、EasyOCR多种引擎
- 完整的图像预处理流水线
- 高精度的古代汉字识别
- 无状态计算服务设计

Author: OCR开发团队
Created: 2025-01-15
Version: 2.0.0 (无状态架构)
"""

from typing import Dict, Any, Optional, List
import asyncio
import time
import tempfile
import logging
from pathlib import Path
import uuid

# 图像处理相关导入
import cv2
import numpy as np
from PIL import Image
import io

# OCR引擎导入
try:
    import paddleocr
except ImportError:
    paddleocr = None
    
try:
    import pytesseract
except ImportError:
    pytesseract = None
    
try:
    import easyocr
except ImportError:
    easyocr = None

# 文本处理导入
try:
    import jieba
except ImportError:
    jieba = None
    
try:
    import opencc
except ImportError:
    opencc = None

from ..clients.storage_client import StorageServiceClient, OCRTaskRequest, StorageClientError
from ..utils.image_processor import ImageProcessor
from ..utils.text_processor import TextProcessor
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OCREngineError(Exception):
    """OCR引擎异常"""
    pass


class ImageProcessingError(Exception):
    """图像处理异常"""
    pass


class OCRService:
    """OCR文本识别服务主类
    
    无状态的OCR计算服务，专注于图像文本识别算法。
    不直接连接数据库，所有数据操作通过storage-service完成。
    
    支持的OCR引擎：
    - PaddleOCR: 主要引擎，适合中文识别
    - Tesseract: 备用引擎，适合英文识别
    - EasyOCR: 多语言支持引擎
    
    主要特性：
    - 异步处理，支持并发任务
    - 完整的错误处理和重试机制
    - 无状态设计，支持水平扩展
    - 详细的处理日志和性能监控
    """
    
    def __init__(self):
        """初始化OCR服务"""
        # 图像和文本处理器
        self._image_processor = ImageProcessor()
        self._text_processor = TextProcessor()
        
        # OCR引擎实例
        self._paddle_ocr = None
        self._easy_ocr = None
        
        # 初始化标志
        self._initialized = False
        
        logger.info("OCR服务实例已创建")
    
    async def initialize(self) -> None:
        """初始化OCR服务
        
        加载OCR模型，预热缓存等准备工作。
        
        Raises:
            OCREngineError: OCR引擎初始化失败
        """
        if self._initialized:
            logger.warning("OCR服务已经初始化过了")
            return
            
        try:
            logger.info("开始初始化OCR服务...")
            
            # 获取引擎配置
            default_engine = settings.ocr.DEFAULT_ENGINE
            paddle_config = settings.get_engine_config("paddleocr")
            tesseract_config = settings.get_engine_config("tesseract")
            easyocr_config = settings.get_engine_config("easyocr")
            
            # 初始化PaddleOCR引擎
            if paddleocr and (default_engine == "paddleocr" or "paddleocr" in default_engine):
                logger.info("初始化PaddleOCR引擎...")
                self._paddle_ocr = paddleocr.PaddleOCR(
                    use_angle_cls=paddle_config.get("use_angle_cls", True),
                    lang=paddle_config.get("lang", "ch"),
                    use_gpu=paddle_config.get("use_gpu", True),
                    show_log=False
                )
                logger.info("PaddleOCR引擎初始化完成")
            else:
                logger.warning("PaddleOCR未安装或未配置，该引擎将不可用")
            
            # 初始化EasyOCR引擎
            if easyocr and ("easyocr" in default_engine):
                logger.info("初始化EasyOCR引擎...")
                self._easy_ocr = easyocr.Reader(
                    easyocr_config.get("lang_list", ['ch_sim', 'en']),
                    gpu=easyocr_config.get("gpu", True)
                )
                logger.info("EasyOCR引擎初始化完成")
            else:
                logger.warning("EasyOCR未安装或未配置，该引擎将不可用")
            
            # 验证Tesseract是否可用
            if pytesseract:
                try:
                    # 测试Tesseract命令是否可用
                    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
                    pytesseract.image_to_string(test_image, lang=tesseract_config.get("lang", "chi_sim+eng"))
                    logger.info("Tesseract引擎验证通过")
                except Exception as e:
                    logger.warning(f"Tesseract引擎不可用: {e}")
            else:
                logger.warning("Tesseract未安装，该引擎将不可用")
            
            self._initialized = True
            logger.info("OCR服务初始化成功")
            
        except Exception as e:
            logger.error(f"OCR服务初始化失败: {e}")
            raise OCREngineError(f"OCR服务初始化失败: {str(e)}")
    
    async def recognize_image(
        self,
        image_content: bytes,
        engine: str = "paddleocr",
        confidence_threshold: float = 0.8,
        language_codes: str = "zh,en",
        enable_preprocessing: bool = True,
        enable_postprocessing: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """识别图像中的文字
        
        这是主要的OCR处理方法，执行完整的识别流程：
        1. 图像预处理
        2. OCR识别
        3. 结果后处理
        
        Args:
            image_content: 图像字节内容
            engine: OCR引擎类型
            confidence_threshold: 置信度阈值
            language_codes: 语言代码
            enable_preprocessing: 启用预处理
            enable_postprocessing: 启用后处理
            metadata: 额外元数据
            
        Returns:
            处理结果字典
            
        Raises:
            OCREngineError: OCR处理失败
            ImageProcessingError: 图像处理失败
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始OCR识别，引擎: {engine}")
            
            # 验证引擎支持
            if not self._is_engine_available(engine):
                raise OCREngineError(f"OCR引擎不可用: {engine}")
            
            # 转换为OpenCV格式
            nparr = np.frombuffer(image_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ImageProcessingError("图像文件格式不支持或已损坏")
            
            # 获取原始图像信息
            height, width = image.shape[:2]
            image_info = {
                "width": width,
                "height": height,
                "channels": len(image.shape),
                "size_bytes": len(image_content)
            }
            
            # 图像预处理
            preprocessed_image = image
            if enable_preprocessing:
                preprocessing_config = {
                    "grayscale": True,
                    "denoise": True,
                    "enhance_contrast": True,
                    "deskew": True
                }
                preprocessed_image = await self._preprocess_image(image, preprocessing_config)
            
            # 选择并执行OCR识别
            ocr_config = {
                "engine": engine,
                "confidence_threshold": confidence_threshold,
                "language_codes": language_codes.split(","),
                "preprocessing": enable_preprocessing
            }
            
            if engine == "paddleocr":
                result = await self._paddle_ocr_recognize(preprocessed_image, ocr_config)
            elif engine == "tesseract":
                result = await self._tesseract_recognize(preprocessed_image, ocr_config)
            elif engine == "easyocr":
                result = await self._easy_ocr_recognize(preprocessed_image, ocr_config)
            else:
                raise OCREngineError(f"不支持的OCR引擎: {engine}")
            
            # 文本后处理
            processed_result = result
            if enable_postprocessing:
                post_processing_config = {
                    "remove_whitespace": True,
                    "normalize_punctuation": True,
                    "traditional_to_simplified": True
                }
                processed_result = await self._post_process_result(result, post_processing_config)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 构建最终结果
            final_result = {
                "success": True,
                "text_content": processed_result["text"],
                "confidence": processed_result["confidence"],
                "bounding_boxes": processed_result.get("bounding_boxes", []),
                "text_blocks": processed_result.get("text_blocks", []),
                "language_detected": processed_result.get("language_detected", "unknown"),
                "word_count": len(processed_result["text"].split()),
                "char_count": len(processed_result["text"]),
                "processing_time": processing_time,
                "metadata": {
                    "engine": engine,
                    "original_image_info": image_info,
                    "preprocessing_enabled": enable_preprocessing,
                    "postprocessing_enabled": enable_postprocessing,
                    "confidence_threshold": confidence_threshold,
                    "language_codes": language_codes,
                    "custom_metadata": metadata or {}
                }
            }
            
            logger.info(f"OCR识别完成，耗时: {processing_time:.2f}秒，字符数: {final_result['char_count']}")
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            
            logger.error(f"OCR识别失败: {error_message}, 耗时: {processing_time:.2f}秒")
            
            return {
                "success": False,
                "error_message": error_message,
                "processing_time": processing_time,
                "metadata": {
                    "engine": engine,
                    "error_type": type(e).__name__
                }
            }
    
    async def process_batch_images(
        self,
        images_data: List[Dict[str, Any]],
        engine: str = "paddleocr",
        confidence_threshold: float = 0.8,
        language_codes: str = "zh,en",
        enable_preprocessing: bool = True,
        enable_postprocessing: bool = True
    ) -> List[Dict[str, Any]]:
        """批量处理多个图像
        
        Args:
            images_data: 图像数据列表，每个包含 {"image_content": bytes, "image_id": str}
            engine: OCR引擎类型
            confidence_threshold: 置信度阈值
            language_codes: 语言代码
            enable_preprocessing: 启用预处理
            enable_postprocessing: 启用后处理
            
        Returns:
            处理结果列表
        """
        if not images_data:
            return []
        
        logger.info(f"开始批量OCR处理，数量: {len(images_data)}")
        
        # 限制并发数量
        max_concurrent = min(settings.ocr.MAX_CONCURRENT_TASKS, len(images_data))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_image(image_data):
            async with semaphore:
                try:
                    result = await self.recognize_image(
                        image_content=image_data["image_content"],
                        engine=engine,
                        confidence_threshold=confidence_threshold,
                        language_codes=language_codes,
                        enable_preprocessing=enable_preprocessing,
                        enable_postprocessing=enable_postprocessing,
                        metadata={"image_id": image_data.get("image_id")}
                    )
                    result["image_id"] = image_data.get("image_id")
                    return result
                except Exception as e:
                    logger.error(f"批量处理中单个图像失败: {e}")
                    return {
                        "success": False,
                        "image_id": image_data.get("image_id"),
                        "error_message": str(e)
                    }
        
        # 并发处理所有图像
        tasks = [process_single_image(image_data) for image_data in images_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "success": False,
                    "image_id": images_data[i].get("image_id"),
                    "error_message": str(result)
                })
            else:
                final_results.append(result)
        
        successful_count = sum(1 for r in final_results if r.get("success"))
        logger.info(f"批量OCR处理完成，成功: {successful_count}/{len(images_data)}")
        
        return final_results
    
    async def _preprocess_image(
        self, 
        image: np.ndarray, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        """图像预处理
        
        对输入图像进行各种预处理操作，提高OCR识别准确率。
        
        Args:
            image: 原始图像
            config: 预处理配置
            
        Returns:
            预处理后的图像
        """
        processed_image = image.copy()
        
        try:
            # 灰度转换
            if config.get("grayscale", True):
                if len(processed_image.shape) == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            
            # 去噪
            if config.get("denoise", True):
                if len(processed_image.shape) == 2:
                    processed_image = cv2.fastNlMeansDenoising(processed_image)
                else:
                    processed_image = cv2.fastNlMeansDenoisingColored(processed_image)
            
            # 对比度增强
            if config.get("enhance_contrast", True):
                if len(processed_image.shape) == 2:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    processed_image = clahe.apply(processed_image)
            
            # 倾斜校正
            if config.get("deskew", True):
                processed_image = self._image_processor.deskew_image(processed_image)
            
            # 二值化
            if config.get("binarize", False):
                if len(processed_image.shape) == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                _, processed_image = cv2.threshold(
                    processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            
            # 尺寸调整
            if config.get("resize", False):
                scale_factor = config.get("scale_factor", 2.0)
                if 0.1 <= scale_factor <= 5.0:
                    height, width = processed_image.shape[:2]
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    processed_image = cv2.resize(
                        processed_image, 
                        (new_width, new_height), 
                        interpolation=cv2.INTER_CUBIC
                    )
            
            return processed_image
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return image  # 返回原始图像
    
    async def _paddle_ocr_recognize(
        self, 
        image: np.ndarray, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用PaddleOCR进行文字识别
        
        Args:
            image: 预处理后的图像
            config: OCR配置
            
        Returns:
            识别结果字典
            
        Raises:
            OCREngineError: PaddleOCR识别失败
        """
        if not self._paddle_ocr:
            raise OCREngineError("PaddleOCR引擎未初始化")
        
        try:
            # 执行OCR识别
            results = self._paddle_ocr.ocr(image, cls=True)
            
            if not results or not results[0]:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "text_blocks": [],
                    "bounding_boxes": []
                }
            
            # 解析结果
            text_blocks = []
            bounding_boxes = []
            all_text = []
            confidences = []
            
            min_confidence = config.get("confidence_threshold", 0.8)
            
            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    
                    # 过滤低置信度结果
                    if confidence >= min_confidence:
                        all_text.append(text)
                        confidences.append(confidence)
                        
                        # 保存边界框信息
                        bounding_boxes.append({
                            "coordinates": bbox,
                            "text": text,
                            "confidence": confidence
                        })
                        
                        # 保存文本块信息
                        text_blocks.append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox
                        })
            
            return {
                "text": "\n".join(all_text),
                "confidence": np.mean(confidences) if confidences else 0.0,
                "text_blocks": text_blocks,
                "bounding_boxes": bounding_boxes,
                "language_detected": "zh"  # PaddleOCR主要用于中文
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            raise OCREngineError(f"PaddleOCR识别失败: {str(e)}")
    
    async def _tesseract_recognize(
        self, 
        image: np.ndarray, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用Tesseract进行文字识别
        
        Args:
            image: 预处理后的图像
            config: OCR配置
            
        Returns:
            识别结果字典
            
        Raises:
            OCREngineError: Tesseract识别失败
        """
        if not pytesseract:
            raise OCREngineError("Tesseract引擎未安装")
        
        try:
            # 配置Tesseract参数
            lang = "+".join(config.get("language_codes", ["chi_sim", "eng"]))
            custom_config = r'--oem 3 --psm 6'
            
            # 执行文字识别
            text = pytesseract.image_to_string(
                image, 
                lang=lang, 
                config=custom_config
            )
            
            # 获取详细信息
            data = pytesseract.image_to_data(
                image, 
                lang=lang, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # 处理结果
            text_blocks = []
            bounding_boxes = []
            confidences = []
            min_confidence = config.get("confidence_threshold", 0.8)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    confidence = int(data['conf'][i]) / 100.0
                    word_text = data['text'][i].strip()
                    
                    if word_text and confidence >= min_confidence:
                        confidences.append(confidence)
                        
                        # 构建边界框
                        bbox = [
                            [data['left'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                            [data['left'][i], data['top'][i] + data['height'][i]]
                        ]
                        
                        bounding_boxes.append({
                            "coordinates": bbox,
                            "text": word_text,
                            "confidence": confidence
                        })
                        
                        text_blocks.append({
                            "text": word_text,
                            "confidence": confidence,
                            "bbox": bbox
                        })
            
            return {
                "text": text.strip(),
                "confidence": np.mean(confidences) if confidences else 0.0,
                "text_blocks": text_blocks,
                "bounding_boxes": bounding_boxes,
                "language_detected": "mixed" if "chi_sim" in lang and "eng" in lang else lang
            }
            
        except Exception as e:
            logger.error(f"Tesseract识别失败: {e}")
            raise OCREngineError(f"Tesseract识别失败: {str(e)}")
    
    async def _easy_ocr_recognize(
        self, 
        image: np.ndarray, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用EasyOCR进行文字识别
        
        Args:
            image: 预处理后的图像
            config: OCR配置
            
        Returns:
            识别结果字典
            
        Raises:
            OCREngineError: EasyOCR识别失败
        """
        if not self._easy_ocr:
            raise OCREngineError("EasyOCR引擎未初始化")
        
        try:
            # 执行OCR识别
            results = self._easy_ocr.readtext(image)
            
            if not results:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "text_blocks": [],
                    "bounding_boxes": []
                }
            
            # 解析结果
            text_blocks = []
            bounding_boxes = []
            all_text = []
            confidences = []
            min_confidence = config.get("confidence_threshold", 0.8)
            
            for bbox, text, confidence in results:
                if confidence >= min_confidence:
                    all_text.append(text)
                    confidences.append(confidence)
                    
                    # EasyOCR返回的是四个角点坐标
                    coordinates = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
                    
                    bounding_boxes.append({
                        "coordinates": coordinates,
                        "text": text,
                        "confidence": confidence
                    })
                    
                    text_blocks.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": coordinates
                    })
            
            return {
                "text": "\n".join(all_text),
                "confidence": np.mean(confidences) if confidences else 0.0,
                "text_blocks": text_blocks,
                "bounding_boxes": bounding_boxes,
                "language_detected": "mixed"  # EasyOCR支持多语言
            }
            
        except Exception as e:
            logger.error(f"EasyOCR识别失败: {e}")
            raise OCREngineError(f"EasyOCR识别失败: {str(e)}")
    
    async def _post_process_result(
        self, 
        result: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """OCR结果后处理
        
        对识别结果进行后处理优化，包括繁简转换、
        空白处理、标点符号规范化、错别字纠正等。
        
        Args:
            result: 原始OCR结果
            config: 后处理配置
            
        Returns:
            优化后的结果
        """
        try:
            text = result["text"]
            
            # 使用文本处理器进行后处理
            processed_text = await self._text_processor.process_text(text, config)
            
            # 更新结果
            result["text"] = processed_text
            
            return result
            
        except Exception as e:
            logger.error(f"OCR结果后处理失败: {e}")
            return result  # 返回原始结果
    
    def _is_engine_available(self, engine: str) -> bool:
        """检查OCR引擎是否可用
        
        Args:
            engine: 引擎名称
            
        Returns:
            引擎是否可用
        """
        if engine == "paddleocr":
            return self._paddle_ocr is not None
        elif engine == "tesseract":
            return pytesseract is not None
        elif engine == "easyocr":
            return self._easy_ocr is not None
        else:
            return False
    
    async def get_available_engines(self) -> List[str]:
        """获取可用的OCR引擎列表
        
        Returns:
            可用引擎名称列表
        """
        engines = []
        if self._paddle_ocr:
            engines.append("paddleocr")
        if pytesseract:
            engines.append("tesseract")
        if self._easy_ocr:
            engines.append("easyocr")
        return engines
    
    async def health_check(self) -> Dict[str, Any]:
        """OCR服务健康检查
        
        Returns:
            健康状态信息
        """
        try:
            status = {
                "service": "ocr",
                "status": "healthy",
                "initialized": self._initialized,
                "engines": {
                    "paddleocr": self._paddle_ocr is not None,
                    "tesseract": pytesseract is not None,
                    "easyocr": self._easy_ocr is not None
                },
                "available_engines": await self.get_available_engines()
            }
            
            # 检查是否至少有一个OCR引擎可用
            if not any(status["engines"].values()):
                status["status"] = "degraded"
                status["warning"] = "没有可用的OCR引擎"
            
            return status
            
        except Exception as e:
            logger.error(f"OCR服务健康检查失败: {e}")
            return {
                "service": "ocr",
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """清理OCR服务资源
        
        释放模型内存等。
        """
        try:
            logger.info("开始清理OCR服务资源...")
            
            # 清理OCR引擎实例
            self._paddle_ocr = None
            self._easy_ocr = None
                
            self._initialized = False
            logger.info("OCR服务资源清理完成")
            
        except Exception as e:
            logger.error(f"OCR服务资源清理失败: {e}")


# 全局OCR服务实例
_ocr_service: Optional[OCRService] = None


async def get_ocr_service() -> OCRService:
    """获取OCR服务实例（单例模式）
    
    Returns:
        OCR服务实例
    """
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
        await _ocr_service.initialize()
    return _ocr_service