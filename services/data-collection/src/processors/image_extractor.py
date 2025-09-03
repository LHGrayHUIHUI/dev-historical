"""
图像OCR文本提取器

使用OCR技术从图像中提取文本内容
"""

import logging
from typing import Dict, List, Any, Optional

from .base import TextExtractor

logger = logging.getLogger(__name__)


class ImageExtractor(TextExtractor):
    """图像OCR文本提取器
    
    支持从各种图像格式中使用OCR技术提取文本
    """
    
    SUPPORTED_TYPES = {
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/tiff',
        'image/tif',
        'image/bmp',
        'image/webp',
        'image/gif'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化图像提取器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.tesseract_path = config.get('tesseract_path', '/usr/bin/tesseract')
        self.languages = config.get('languages', ['chi_sim', 'chi_tra', 'eng'])
        self.preprocess_image = config.get('preprocess_image', True)
        self.confidence_threshold = config.get('confidence_threshold', 30)
        self.psm_mode = config.get('psm_mode', 6)  # 统一文本块模式
    
    def supports_file_type(self, file_type: str) -> bool:
        """检查是否支持指定文件类型"""
        return file_type in self.SUPPORTED_TYPES
    
    async def extract(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """从图像文件提取文本内容
        
        Args:
            file_path: 图像文件路径
            **kwargs: 其他参数
            
        Returns:
            提取的文本内容列表
        """
        if not self.validate_file(file_path):
            raise ValueError(f"文件验证失败: {file_path}")
        
        try:
            # 进行OCR识别
            contents = await self._extract_with_tesseract(file_path)
            
            self.logger.info(f"图像OCR提取完成: {file_path}, 文本块数: {len(contents)}")
            return contents
            
        except Exception as e:
            self.logger.error(f"图像OCR提取失败: {file_path}, 错误: {str(e)}")
            raise
    
    async def _extract_with_tesseract(self, file_path: str) -> List[Dict[str, Any]]:
        """使用Tesseract进行OCR识别
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            提取的文本内容列表
        """
        contents = []
        
        try:
            from PIL import Image
            import pytesseract
            
            # 设置Tesseract路径（如果需要）
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            
            # 打开并预处理图像
            image = Image.open(file_path)
            
            if self.preprocess_image:
                image = self._preprocess_image(image)
            
            # 构建语言字符串
            lang_string = '+'.join(self.languages)
            
            # 配置Tesseract选项
            custom_config = f'--oem 3 --psm {self.psm_mode} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u4e00-\u9fff'
            
            # 基础OCR识别
            text = pytesseract.image_to_string(
                image,
                lang=lang_string,
                config=custom_config
            )
            
            if text and text.strip():
                # 获取详细数据（包含置信度）
                data = pytesseract.image_to_data(
                    image,
                    lang=lang_string,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # 处理OCR结果
                processed_contents = self._process_ocr_data(text, data)
                contents.extend(processed_contents)
            
            # 尝试不同的PSM模式（如果第一次结果不好）
            if not contents or self._is_poor_quality_result(contents):
                alternative_contents = await self._try_alternative_psm_modes(image, lang_string)
                if alternative_contents and len(alternative_contents) > len(contents):
                    contents = alternative_contents
            
            return contents
            
        except ImportError:
            self.logger.error("缺少依赖库: PIL 或 pytesseract")
            raise ImportError("需要安装 Pillow 和 pytesseract 库")
        except Exception as e:
            self.logger.error(f"Tesseract OCR失败: {str(e)}")
            raise
    
    def _preprocess_image(self, image: 'Image') -> 'Image':
        """预处理图像以提高OCR质量
        
        Args:
            image: PIL图像对象
            
        Returns:
            预处理后的图像
        """
        try:
            import cv2
            import numpy as np
            from PIL import ImageEnhance, ImageFilter
            
            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # 增强清晰度
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # 转换为numpy数组进行OpenCV处理
            img_array = np.array(image)
            
            # 转换为灰度图
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 高斯模糊去噪
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 形态学操作去噪
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 转换回PIL图像
            processed_image = Image.fromarray(cleaned)
            
            return processed_image
            
        except ImportError:
            self.logger.warning("OpenCV未安装，跳过图像预处理")
            return image
        except Exception as e:
            self.logger.warning(f"图像预处理失败: {str(e)}")
            return image
    
    def _process_ocr_data(self, text: str, data: Dict[str, List]) -> List[Dict[str, Any]]:
        """处理OCR识别数据
        
        Args:
            text: OCR识别的文本
            data: Tesseract输出的详细数据
            
        Returns:
            处理后的内容列表
        """
        contents = []
        
        # 按行分组处理
        lines = {}
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > self.confidence_threshold:
                line_num = data['line_num'][i]
                word_text = data['text'][i].strip()
                
                if word_text:
                    if line_num not in lines:
                        lines[line_num] = {
                            'text': [],
                            'confidences': [],
                            'coords': []
                        }
                    
                    lines[line_num]['text'].append(word_text)
                    lines[line_num]['confidences'].append(data['conf'][i])
                    lines[line_num]['coords'].append({
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    })
        
        # 为每行创建内容项
        for line_num, line_data in lines.items():
            if line_data['text']:
                line_text = ' '.join(line_data['text'])
                avg_confidence = sum(line_data['confidences']) / len(line_data['confidences'])
                
                content_item = self.create_text_content(
                    content=line_text,
                    page_number=line_num,
                    title=f"Line {line_num}",
                    confidence=round(avg_confidence / 100, 3),
                    line_number=line_num,
                    coordinates=line_data['coords'],
                    extraction_method='tesseract_ocr'
                )
                
                contents.append(content_item)
        
        # 如果没有按行分组的结果，创建一个整体内容项
        if not contents and text.strip():
            overall_confidence = self._estimate_overall_confidence(data)
            content_item = self.create_text_content(
                content=text,
                page_number=1,
                title="OCR Text",
                confidence=overall_confidence,
                extraction_method='tesseract_ocr'
            )
            contents.append(content_item)
        
        return contents
    
    async def _try_alternative_psm_modes(self, image: 'Image', lang_string: str) -> List[Dict[str, Any]]:
        """尝试不同的PSM模式
        
        Args:
            image: PIL图像对象
            lang_string: 语言字符串
            
        Returns:
            最佳结果的内容列表
        """
        import pytesseract
        
        # 尝试的PSM模式
        psm_modes = [3, 4, 6, 8, 11, 13]  # 不同的页面分割模式
        best_result = []
        best_confidence = 0
        
        for psm in psm_modes:
            if psm == self.psm_mode:
                continue  # 跳过已经尝试过的模式
            
            try:
                custom_config = f'--oem 3 --psm {psm}'
                
                text = pytesseract.image_to_string(
                    image,
                    lang=lang_string,
                    config=custom_config
                )
                
                if text and text.strip():
                    # 获取置信度数据
                    data = pytesseract.image_to_data(
                        image,
                        lang=lang_string,
                        config=custom_config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    avg_confidence = self._estimate_overall_confidence(data)
                    
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_result = self._process_ocr_data(text, data)
            
            except Exception as e:
                self.logger.debug(f"PSM模式 {psm} 失败: {str(e)}")
                continue
        
        return best_result
    
    def _estimate_overall_confidence(self, data: Dict[str, List]) -> float:
        """估算整体置信度
        
        Args:
            data: Tesseract输出数据
            
        Returns:
            整体置信度 (0-1)
        """
        confidences = [conf for conf in data['conf'] if conf > 0]
        if not confidences:
            return 0.0
        
        avg_conf = sum(confidences) / len(confidences)
        return round(avg_conf / 100, 3)
    
    def _is_poor_quality_result(self, contents: List[Dict[str, Any]]) -> bool:
        """判断OCR结果质量是否较差
        
        Args:
            contents: OCR结果列表
            
        Returns:
            是否为低质量结果
        """
        if not contents:
            return True
        
        # 检查平均置信度
        confidences = [item.get('confidence', 0) for item in contents]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        if avg_confidence < 0.5:
            return True
        
        # 检查文本长度
        total_chars = sum(item.get('char_count', 0) for item in contents)
        if total_chars < 10:
            return True
        
        return False
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的OCR语言列表
        
        Returns:
            支持的语言列表
        """
        try:
            import pytesseract
            return pytesseract.get_languages()
        except Exception as e:
            self.logger.warning(f"获取支持语言列表失败: {str(e)}")
            return self.languages