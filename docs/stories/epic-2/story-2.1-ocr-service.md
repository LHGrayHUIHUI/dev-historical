# Story 2.1: OCR文本识别服务

## 基本信息
- **Story ID**: 2.1
- **Epic**: Epic 2 - 数据处理和智能分类微服务
- **标题**: OCR文本识别服务
- **优先级**: 高
- **状态**: 待开发
- **预估工期**: 6-8天

## 用户故事
**作为** 数据处理专员  
**我希望** 有一个高精度的OCR文本识别服务  
**以便** 从历史文档图像中准确提取文字内容，支持古代汉字和多种字体识别

## 需求描述
开发专业的OCR文本识别服务，支持古代汉字、繁体字、多种字体的识别，具备图像预处理、文字检测、字符识别、后处理优化等完整功能。

## 技术实现

### 核心技术栈
- **后端框架**: FastAPI 0.104+ (Python)
- **OCR引擎**: 
  - PaddleOCR 2.7+ (主要引擎)
  - Tesseract 5.3+ (备用引擎)
  - EasyOCR 1.7+ (多语言支持)
- **图像处理**: 
  - OpenCV 4.8+ (图像预处理)
  - Pillow 10.1+ (图像操作)
  - scikit-image 0.22+ (图像增强)
- **深度学习**: 
  - PyTorch 2.1+ (模型推理)
  - ONNX Runtime (模型优化)
- **文本处理**: 
  - jieba 0.42+ (中文分词)
  - opencc 1.1+ (繁简转换)
- **数据库**: 
  - PostgreSQL (识别结果存储)
  - Redis (缓存和队列)
- **消息队列**: RabbitMQ 3.12+
- **对象存储**: MinIO

### 数据模型设计

#### OCR任务表 (ocr_tasks)
```sql
CREATE TABLE ocr_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES datasets(id),
    image_path VARCHAR(500) NOT NULL,
    image_size JSONB, -- {"width": 1920, "height": 1080}
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    ocr_engine VARCHAR(50), -- paddleocr, tesseract, easyocr
    confidence_threshold FLOAT DEFAULT 0.8,
    language_codes VARCHAR(100) DEFAULT 'zh,en', -- 支持的语言
    preprocessing_config JSONB, -- 预处理配置
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### OCR结果表 (ocr_results)
```sql
CREATE TABLE ocr_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES ocr_tasks(id),
    text_content TEXT NOT NULL,
    confidence_score FLOAT,
    bounding_boxes JSONB, -- 文字区域坐标
    text_blocks JSONB, -- 文本块信息
    language_detected VARCHAR(50),
    word_count INTEGER,
    char_count INTEGER,
    processing_time FLOAT, -- 处理时间(秒)
    metadata JSONB, -- 额外元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 全文搜索索引
    text_vector tsvector GENERATED ALWAYS AS (to_tsvector('chinese', text_content)) STORED
);

-- 创建索引
CREATE INDEX idx_ocr_results_task ON ocr_results(task_id);
CREATE INDEX idx_ocr_results_search ON ocr_results USING GIN(text_vector);
CREATE INDEX idx_ocr_results_confidence ON ocr_results(confidence_score);
```

#### OCR配置表 (ocr_configs)
```sql
CREATE TABLE ocr_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    engine VARCHAR(50) NOT NULL,
    config JSONB NOT NULL, -- 引擎配置参数
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 服务架构

#### OCR服务主类
```python
# src/services/ocr_service.py
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from typing import List, Optional, Dict, Any
import asyncio
import cv2
import numpy as np
from PIL import Image
import paddleocr
import pytesseract
import easyocr
from pathlib import Path
import uuid
import time

class OCRService:
    def __init__(self):
        self.paddle_ocr = paddleocr.PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=True if torch.cuda.is_available() else False
        )
        self.easy_ocr = easyocr.Reader(['ch_sim', 'ch_tra', 'en'])
        self.db = DatabaseManager()
        self.storage = MinIOClient()
        self.message_queue = RabbitMQClient()
        
    async def process_image(self, 
                          image_path: str, 
                          task_id: str,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理图像OCR识别
        
        Args:
            image_path: 图像文件路径
            task_id: 任务ID
            config: OCR配置参数
            
        Returns:
            OCR识别结果
        """
        try:
            start_time = time.time()
            
            # 更新任务状态
            await self._update_task_status(task_id, 'processing')
            
            # 加载和预处理图像
            image = await self._load_and_preprocess_image(image_path, config)
            
            # 选择OCR引擎
            engine = config.get('engine', 'paddleocr')
            
            if engine == 'paddleocr':
                result = await self._paddle_ocr_recognize(image, config)
            elif engine == 'tesseract':
                result = await self._tesseract_recognize(image, config)
            elif engine == 'easyocr':
                result = await self._easy_ocr_recognize(image, config)
            else:
                raise ValueError(f"不支持的OCR引擎: {engine}")
            
            # 后处理优化
            result = await self._post_process_result(result, config)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 保存结果到数据库
            ocr_result = await self._save_ocr_result(
                task_id=task_id,
                result=result,
                processing_time=processing_time
            )
            
            # 更新任务状态
            await self._update_task_status(task_id, 'completed')
            
            return {
                'success': True,
                'task_id': task_id,
                'result_id': ocr_result.id,
                'text_content': result['text'],
                'confidence': result['confidence'],
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"OCR处理失败: {str(e)}")
            await self._update_task_status(task_id, 'failed', str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _load_and_preprocess_image(self, 
                                       image_path: str, 
                                       config: Dict[str, Any]) -> np.ndarray:
        """
        加载和预处理图像
        
        Args:
            image_path: 图像路径
            config: 预处理配置
            
        Returns:
            预处理后的图像数组
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 图像预处理
        preprocessing = config.get('preprocessing', {})
        
        # 灰度转换
        if preprocessing.get('grayscale', True):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 噪声去除
        if preprocessing.get('denoise', True):
            image = cv2.fastNlMeansDenoising(image)
        
        # 对比度增强
        if preprocessing.get('enhance_contrast', True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
        
        # 二值化
        if preprocessing.get('binarize', False):
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 倾斜校正
        if preprocessing.get('deskew', True):
            image = self._deskew_image(image)
        
        # 尺寸调整
        if preprocessing.get('resize', False):
            scale = preprocessing.get('scale_factor', 2.0)
            height, width = image.shape[:2]
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        return image
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像倾斜校正
        
        Args:
            image: 输入图像
            
        Returns:
            校正后的图像
        """
        # 边缘检测
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # 计算平均角度
            angles = []
            for rho, theta in lines[:10]:  # 只取前10条线
                angle = theta * 180 / np.pi - 90
                angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                
                # 旋转图像
                if abs(avg_angle) > 0.5:  # 只有角度大于0.5度才校正
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    async def _paddle_ocr_recognize(self, 
                                  image: np.ndarray, 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用PaddleOCR进行文字识别
        
        Args:
            image: 预处理后的图像
            config: OCR配置
            
        Returns:
            识别结果
        """
        # PaddleOCR识别
        results = self.paddle_ocr.ocr(image, cls=True)
        
        if not results or not results[0]:
            return {
                'text': '',
                'confidence': 0.0,
                'text_blocks': [],
                'bounding_boxes': []
            }
        
        text_blocks = []
        bounding_boxes = []
        all_text = []
        confidences = []
        
        for line in results[0]:
            if line:
                bbox, (text, confidence) = line
                
                # 过滤低置信度结果
                min_confidence = config.get('confidence_threshold', 0.8)
                if confidence >= min_confidence:
                    all_text.append(text)
                    confidences.append(confidence)
                    
                    # 保存边界框信息
                    bounding_boxes.append({
                        'coordinates': bbox,
                        'text': text,
                        'confidence': confidence
                    })
                    
                    # 保存文本块信息
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return {
            'text': '\n'.join(all_text),
            'confidence': np.mean(confidences) if confidences else 0.0,
            'text_blocks': text_blocks,
            'bounding_boxes': bounding_boxes
        }
    
    async def _tesseract_recognize(self, 
                                 image: np.ndarray, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用Tesseract进行文字识别
        
        Args:
            image: 预处理后的图像
            config: OCR配置
            
        Returns:
            识别结果
        """
        # Tesseract配置
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十'
        
        # 文字识别
        text = pytesseract.image_to_string(image, lang='chi_sim+eng', config=custom_config)
        
        # 获取详细信息
        data = pytesseract.image_to_data(image, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
        
        # 处理结果
        text_blocks = []
        bounding_boxes = []
        confidences = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                confidence = int(data['conf'][i]) / 100.0
                word_text = data['text'][i].strip()
                
                if word_text and confidence >= config.get('confidence_threshold', 0.8):
                    confidences.append(confidence)
                    
                    bbox = [
                        [data['left'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                        [data['left'][i], data['top'][i] + data['height'][i]]
                    ]
                    
                    bounding_boxes.append({
                        'coordinates': bbox,
                        'text': word_text,
                        'confidence': confidence
                    })
                    
                    text_blocks.append({
                        'text': word_text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return {
            'text': text.strip(),
            'confidence': np.mean(confidences) if confidences else 0.0,
            'text_blocks': text_blocks,
            'bounding_boxes': bounding_boxes
        }
    
    async def _post_process_result(self, 
                                 result: Dict[str, Any], 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        OCR结果后处理
        
        Args:
            result: 原始OCR结果
            config: 后处理配置
            
        Returns:
            优化后的结果
        """
        text = result['text']
        
        post_processing = config.get('post_processing', {})
        
        # 繁简转换
        if post_processing.get('traditional_to_simplified', True):
            import opencc
            converter = opencc.OpenCC('t2s')
            text = converter.convert(text)
        
        # 去除多余空白
        if post_processing.get('remove_extra_whitespace', True):
            import re
            text = re.sub(r'\s+', ' ', text).strip()
        
        # 标点符号规范化
        if post_processing.get('normalize_punctuation', True):
            text = text.replace('，', ',')
            text = text.replace('。', '.')
            text = text.replace('；', ';')
            text = text.replace('：', ':')
        
        # 错别字纠正(简单版本)
        if post_processing.get('spell_check', False):
            text = self._simple_spell_check(text)
        
        result['text'] = text
        return result
    
    def _simple_spell_check(self, text: str) -> str:
        """
        简单的错别字纠正
        
        Args:
            text: 输入文本
            
        Returns:
            纠正后的文本
        """
        # 常见错别字映射
        corrections = {
            '0': '○',  # 数字0替换为圆圈
            '1': '一',  # 数字1替换为汉字一
            '2': '二',
            '3': '三',
            # 可以添加更多映射规则
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text
```

### API设计

#### OCRController实现

```python
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
from uuid import UUID
import asyncio

from ..services.ocr_service import OCRService
from ..models.schemas import (
    OCRTaskResponse, OCRResultResponse, OCRTaskListResponse,
    OCRConfigRequest, OCRBatchRequest
)
from ..dependencies import get_current_user, get_ocr_service
from ..models.database import User

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR"])

@router.post("/recognize", response_model=OCRTaskResponse)
async def recognize_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = Form(default="paddleocr"),
    confidence_threshold: float = Form(default=0.8),
    language_codes: str = Form(default="zh,en"),
    preprocessing_config: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    单个图像OCR识别
    
    Args:
        background_tasks: 后台任务管理器
        file: 上传的图像文件
        engine: OCR引擎选择
        confidence_threshold: 置信度阈值
        language_codes: 支持的语言代码
        preprocessing_config: 预处理配置JSON字符串
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        OCR任务响应
    """
    try:
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图像文件")
        
        # 验证文件大小 (50MB)
        if file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小不能超过50MB")
        
        # 解析预处理配置
        import json
        preprocessing = {}
        if preprocessing_config:
            try:
                preprocessing = json.loads(preprocessing_config)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="预处理配置格式错误")
        
        # 创建OCR任务
        task = await ocr_service.create_task(
            file=file,
            user_id=current_user.id,
            engine=engine,
            confidence_threshold=confidence_threshold,
            language_codes=language_codes,
            preprocessing_config=preprocessing
        )
        
        # 添加后台处理任务
        background_tasks.add_task(
            ocr_service.process_image_async,
            task.id,
            file,
            {
                'engine': engine,
                'confidence_threshold': confidence_threshold,
                'language_codes': language_codes.split(','),
                'preprocessing': preprocessing
            }
        )
        
        return OCRTaskResponse(
            success=True,
            task_id=str(task.id),
            message="OCR任务已创建，正在处理中",
            status=task.processing_status
        )
        
    except Exception as e:
        logger.error(f"OCR识别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=OCRTaskListResponse)
async def batch_recognize(
    request: OCRBatchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    批量图像OCR识别
    
    Args:
        request: 批量OCR请求
        background_tasks: 后台任务管理器
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        OCR任务列表响应
    """
    try:
        # 验证批量大小
        if len(request.image_paths) > 100:
            raise HTTPException(status_code=400, detail="批量处理最多支持100个文件")
        
        # 创建批量任务
        tasks = await ocr_service.create_batch_tasks(
            image_paths=request.image_paths,
            user_id=current_user.id,
            dataset_id=request.dataset_id,
            config=request.config.dict() if request.config else {}
        )
        
        # 添加后台处理任务
        for task in tasks:
            background_tasks.add_task(
                ocr_service.process_image_by_path,
                task.id,
                task.image_path,
                request.config.dict() if request.config else {}
            )
        
        return OCRTaskListResponse(
            success=True,
            tasks=[OCRTask.from_orm(task) for task in tasks],
            total=len(tasks),
            message=f"已创建{len(tasks)}个OCR任务"
        )
        
    except Exception as e:
        logger.error(f"批量OCR处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}", response_model=OCRTaskResponse)
async def get_task_status(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    获取OCR任务状态
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        OCR任务响应
    """
    try:
        task = await ocr_service.get_task(task_id, current_user.id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        # 获取OCR结果
        result = None
        if task.processing_status == 'completed':
            result = await ocr_service.get_result_by_task_id(task_id)
        
        return OCRTaskResponse(
            success=True,
            task=OCRTask.from_orm(task),
            result=OCRResult.from_orm(result) if result else None
        )
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks", response_model=OCRTaskListResponse)
async def list_tasks(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    dataset_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    获取OCR任务列表
    
    Args:
        page: 页码
        page_size: 每页大小
        status: 任务状态过滤
        dataset_id: 数据集ID过滤
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        OCR任务列表响应
    """
    try:
        tasks, total = await ocr_service.list_tasks(
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            status=status,
            dataset_id=dataset_id
        )
        
        return OCRTaskListResponse(
            success=True,
            tasks=[OCRTask.from_orm(task) for task in tasks],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    删除OCR任务
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        删除结果
    """
    try:
        success = await ocr_service.delete_task(task_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或无权限删除")
        
        return {"success": True, "message": "任务删除成功"}
        
    except Exception as e:
        logger.error(f"删除任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/{task_id}/retry")
async def retry_task(
    task_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    重试失败的OCR任务
    
    Args:
        task_id: 任务ID
        background_tasks: 后台任务管理器
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        重试结果
    """
    try:
        task = await ocr_service.get_task(task_id, current_user.id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.processing_status != 'failed':
            raise HTTPException(status_code=400, detail="只能重试失败的任务")
        
        # 重置任务状态
        await ocr_service.reset_task(task_id)
        
        # 重新添加处理任务
        background_tasks.add_task(
            ocr_service.process_image_by_path,
            task_id,
            task.image_path,
            task.preprocessing_config or {}
        )
        
        return {"success": True, "message": "任务重试已启动"}
        
    except Exception as e:
        logger.error(f"重试任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{result_id}", response_model=OCRResultResponse)
async def get_result(
    result_id: UUID,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    获取OCR识别结果
    
    Args:
        result_id: 结果ID
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        OCR结果响应
    """
    try:
        result = await ocr_service.get_result(result_id, current_user.id)
        if not result:
            raise HTTPException(status_code=404, detail="结果不存在")
        
        return OCRResultResponse(
            success=True,
            result=OCRResult.from_orm(result)
        )
        
    except Exception as e:
        logger.error(f"获取OCR结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_text(
    query: str,
    page: int = 1,
    page_size: int = 20,
    dataset_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    全文搜索OCR结果
    
    Args:
        query: 搜索关键词
        page: 页码
        page_size: 每页大小
        dataset_id: 数据集ID过滤
        current_user: 当前用户
        ocr_service: OCR服务实例
        
    Returns:
        搜索结果
    """
    try:
        results, total = await ocr_service.search_text(
            query=query,
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            dataset_id=dataset_id
        )
        
        return {
            "success": True,
            "results": [OCRResult.from_orm(result) for result in results],
            "total": total,
            "page": page,
            "page_size": page_size,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"文本搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Pydantic模型定义

```python
# models/ocr_models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid

class OCREngine(str, Enum):
    """
    OCR引擎枚举
    """
    PADDLEOCR = "paddleocr"  # PaddleOCR
    TESSERACT = "tesseract"  # Tesseract
    EASYOCR = "easyocr"      # EasyOCR

class TaskStatus(str, Enum):
    """
    任务状态枚举
    """
    PENDING = "pending"      # 等待中
    PROCESSING = "processing" # 处理中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"   # 已取消

class BoundingBox(BaseModel):
    """
    边界框模型
    """
    coordinates: List[List[float]] = Field(..., description="边界框坐标")
    text: str = Field(..., description="文本内容")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")

class TextBlock(BaseModel):
    """
    文本块模型
    """
    text: str = Field(..., description="文本内容")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    bbox: List[List[float]] = Field(..., description="边界框坐标")
    language: Optional[str] = Field(default=None, description="检测到的语言")

class PreprocessingConfig(BaseModel):
    """
    图像预处理配置
    """
    grayscale: bool = Field(default=True, description="是否转换为灰度")
    denoise: bool = Field(default=True, description="是否去噪")
    enhance_contrast: bool = Field(default=True, description="是否增强对比度")
    binarize: bool = Field(default=False, description="是否二值化")
    deskew: bool = Field(default=True, description="是否倾斜校正")
    resize: bool = Field(default=False, description="是否调整尺寸")
    scale_factor: float = Field(default=2.0, ge=0.5, le=5.0, description="缩放因子")

class PostProcessingConfig(BaseModel):
    """
    文本后处理配置
    """
    traditional_to_simplified: bool = Field(default=True, description="繁体转简体")
    remove_extra_whitespace: bool = Field(default=True, description="去除多余空白")
    normalize_punctuation: bool = Field(default=True, description="标点符号规范化")
    spell_check: bool = Field(default=False, description="错别字纠正")

class OCRConfig(BaseModel):
    """
    OCR配置模型
    """
    engine: OCREngine = Field(default=OCREngine.PADDLEOCR, description="OCR引擎")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="置信度阈值")
    language_codes: List[str] = Field(default=["zh", "en"], description="支持的语言代码")
    preprocessing: Optional[PreprocessingConfig] = Field(default=None, description="预处理配置")
    post_processing: Optional[PostProcessingConfig] = Field(default=None, description="后处理配置")
    use_gpu: bool = Field(default=True, description="是否使用GPU")
    batch_size: int = Field(default=1, ge=1, le=32, description="批处理大小")

class OCRRequest(BaseModel):
    """
    OCR请求模型
    """
    config: Optional[OCRConfig] = Field(default=None, description="OCR配置")
    priority: int = Field(default=5, ge=1, le=10, description="任务优先级")
    callback_url: Optional[str] = Field(default=None, description="回调URL")

class OCRBatchRequest(BaseModel):
    """
    批量OCR请求模型
    """
    dataset_id: Optional[str] = Field(default=None, description="数据集ID")
    image_paths: List[str] = Field(..., min_items=1, max_items=100, description="图像路径列表")
    config: Optional[OCRConfig] = Field(default=None, description="OCR配置")
    priority: int = Field(default=5, ge=1, le=10, description="任务优先级")
    callback_url: Optional[str] = Field(default=None, description="回调URL")

class OCRTask(BaseModel):
    """
    OCR任务模型
    """
    id: str = Field(..., description="任务ID")
    dataset_id: Optional[str] = Field(default=None, description="数据集ID")
    image_path: str = Field(..., description="图像路径")
    image_size: Optional[Dict[str, int]] = Field(default=None, description="图像尺寸")
    processing_status: TaskStatus = Field(..., description="处理状态")
    ocr_engine: OCREngine = Field(..., description="OCR引擎")
    confidence_threshold: float = Field(..., description="置信度阈值")
    language_codes: str = Field(..., description="支持的语言")
    preprocessing_config: Optional[Dict[str, Any]] = Field(default=None, description="预处理配置")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    created_by: str = Field(..., description="创建者ID")
    created_at: datetime = Field(..., description="创建时间")

class OCRResult(BaseModel):
    """
    OCR结果模型
    """
    id: str = Field(..., description="结果ID")
    task_id: str = Field(..., description="任务ID")
    text_content: str = Field(..., description="识别的文本内容")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="整体置信度")
    bounding_boxes: List[BoundingBox] = Field(default=[], description="边界框列表")
    text_blocks: List[TextBlock] = Field(default=[], description="文本块列表")
    language_detected: Optional[str] = Field(default=None, description="检测到的语言")
    word_count: int = Field(..., ge=0, description="词数")
    char_count: int = Field(..., ge=0, description="字符数")
    processing_time: float = Field(..., ge=0.0, description="处理时间(秒)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="额外元数据")
    created_at: datetime = Field(..., description="创建时间")

class OCRTaskResponse(BaseModel):
    """
    OCR任务响应模型
    """
    success: bool = Field(..., description="是否成功")
    task_id: Optional[str] = Field(default=None, description="任务ID")
    message: str = Field(..., description="响应消息")
    status: Optional[str] = Field(default=None, description="任务状态")
    task: Optional[OCRTask] = Field(default=None, description="任务详情")
    result: Optional[OCRResult] = Field(default=None, description="OCR结果")

class OCRTaskListResponse(BaseModel):
    """
    OCR任务列表响应模型
    """
    success: bool = Field(..., description="是否成功")
    tasks: List[OCRTask] = Field(..., description="任务列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="页码")
    page_size: int = Field(..., description="每页大小")
    message: Optional[str] = Field(default=None, description="响应消息")

class OCRResultResponse(BaseModel):
    """
    OCR结果响应模型
    """
    success: bool = Field(..., description="是否成功")
    result: OCRResult = Field(..., description="OCR结果")

class OCRSettings(BaseModel):
    """
    OCR服务设置
    """
    # 引擎配置
    default_engine: OCREngine = Field(default=OCREngine.PADDLEOCR, description="默认OCR引擎")
    enable_gpu: bool = Field(default=True, description="是否启用GPU")
    
    # 处理限制
    max_image_size: int = Field(default=50 * 1024 * 1024, description="最大图像大小(字节)")
    max_batch_size: int = Field(default=100, description="最大批量处理数量")
    
    # 性能设置
    max_concurrent_tasks: int = Field(default=10, description="最大并发任务数")
    task_timeout: int = Field(default=600, description="任务超时时间(秒)")
    
    # 存储设置
    storage_backend: str = Field(default="minio", description="存储后端")
    temp_dir: str = Field(default="/tmp/ocr", description="临时目录")
    
    # 缓存设置
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    cache_ttl: int = Field(default=3600, description="缓存TTL(秒)")
    
    # 模型路径
    paddleocr_model_dir: str = Field(default="models/paddleocr", description="PaddleOCR模型目录")
    tesseract_data_dir: str = Field(default="/usr/share/tesseract-ocr/5/tessdata", description="Tesseract数据目录")
    
    # 质量控制
     min_confidence_threshold: float = Field(default=0.5, description="最小置信度阈值")
     max_confidence_threshold: float = Field(default=0.95, description="最大置信度阈值")
```

### 依赖注入配置

```python
# dependencies.py
from functools import lru_cache
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import jwt
from datetime import datetime

from .database import get_db_session
from .services.ocr_service import OCRService
from .models.user import User
from .core.config import settings

security = HTTPBearer()

@lru_cache()
def get_ocr_service() -> OCRService:
    """
    获取OCR服务单例实例
    
    Returns:
        OCR服务实例
    """
    return OCRService()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> User:
    """
    获取当前认证用户
    
    Args:
        credentials: JWT认证凭据
        db: 数据库会话
        
    Returns:
        当前用户对象
        
    Raises:
        HTTPException: 认证失败时抛出
    """
    try:
        # 解码JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 检查token是否过期
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="认证凭据已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 从数据库获取用户
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户账户已禁用",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> Optional[User]:
    """
    获取可选的当前用户（允许匿名访问）
    
    Args:
        credentials: 可选的JWT认证凭据
        db: 数据库会话
        
    Returns:
        当前用户对象或None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None

async def validate_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    验证管理员用户权限
    
    Args:
        current_user: 当前用户
        
    Returns:
        管理员用户对象
        
    Raises:
        HTTPException: 权限不足时抛出
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
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
import uvicorn
from typing import Dict, Any

from .core.config import settings
from .database import init_database, close_database
from .services.ocr_service import OCRService
from .controllers.ocr_controller import router as ocr_router
from .dependencies import get_ocr_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用程序生命周期管理
    
    Args:
        app: FastAPI应用实例
    """
    # 启动时初始化
    logger.info("正在启动OCR服务...")
    
    try:
        # 初始化数据库
        await init_database()
        logger.info("数据库初始化完成")
        
        # 初始化OCR服务
        ocr_service = get_ocr_service()
        await ocr_service.initialize()
        logger.info("OCR服务初始化完成")
        
        # 预热模型
        await ocr_service.warmup_models()
        logger.info("OCR模型预热完成")
        
        logger.info("OCR服务启动成功")
        
    except Exception as e:
        logger.error(f"OCR服务启动失败: {str(e)}")
        raise
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭OCR服务...")
    
    try:
        # 清理OCR服务资源
        ocr_service = get_ocr_service()
        await ocr_service.cleanup()
        logger.info("OCR服务资源清理完成")
        
        # 关闭数据库连接
        await close_database()
        logger.info("数据库连接关闭完成")
        
        logger.info("OCR服务关闭成功")
        
    except Exception as e:
        logger.error(f"OCR服务关闭时出错: {str(e)}")

# 创建FastAPI应用
app = FastAPI(
    title="历史文本OCR识别服务",
    description="基于深度学习的古籍文本OCR识别服务",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
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
        HTTP响应
    """
    start_time = time.time()
    
    # 记录请求信息
    logger.info(
        f"请求开始: {request.method} {request.url} - "
        f"客户端: {request.client.host if request.client else 'unknown'}"
    )
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 记录响应信息
    logger.info(
        f"请求完成: {request.method} {request.url} - "
        f"状态码: {response.status_code} - "
        f"处理时间: {process_time:.3f}s"
    )
    
    # 添加处理时间到响应头
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# 全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理请求验证异常
    
    Args:
        request: HTTP请求对象
        exc: 验证异常
        
    Returns:
        错误响应
    """
    logger.warning(f"请求验证失败: {request.url} - {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "请求参数验证失败",
            "errors": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    处理全局异常
    
    Args:
        request: HTTP请求对象
        exc: 异常对象
        
    Returns:
        错误响应
    """
    logger.error(f"未处理的异常: {request.url} - {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "服务器内部错误",
            "error": str(exc) if settings.DEBUG else "Internal server error"
        }
    )

# 注册路由
app.include_router(ocr_router, prefix="/api/v1/ocr", tags=["OCR识别"])

# 健康检查端点
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    健康检查端点
    
    Returns:
        服务健康状态
    """
    try:
        # 检查OCR服务状态
        ocr_service = get_ocr_service()
        ocr_status = await ocr_service.health_check()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "ocr": ocr_status,
                "database": "healthy",  # 可以添加数据库健康检查
            }
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

# 服务信息端点
@app.get("/info")
async def service_info() -> Dict[str, Any]:
    """
    获取服务信息
    
    Returns:
        服务基本信息
    """
    return {
        "name": "历史文本OCR识别服务",
        "version": "1.0.0",
        "description": "基于深度学习的古籍文本OCR识别服务",
        "supported_engines": ["paddleocr", "tesseract", "easyocr"],
        "supported_languages": ["zh", "en", "ja", "ko"],
        "max_image_size": "50MB",
        "max_batch_size": 100
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS
    )
```
    image: UploadFile = File(...),
    engine: str = Form("paddleocr"),
    confidence_threshold: float = Form(0.8),
    language: str = Form("zh,en"),
    preprocessing: str = Form("{}"),
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    单张图像OCR识别
    
    Args:
        background_tasks: 后台任务管理器
        image: 上传的图像文件
        engine: OCR引擎类型 (paddleocr/tesseract/easyocr)
        confidence_threshold: 置信度阈值 (0.0-1.0)
        language: 语言代码，逗号分隔
        preprocessing: 预处理选项JSON字符串
        current_user: 当前用户
        ocr_service: OCR服务实例
    
    Returns:
        OCRTaskResponse: 任务创建响应
    """
    try:
        # 验证文件类型
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图像文件")
        
        # 验证文件大小 (10MB)
        if image.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图像文件不能超过10MB")
        
        # 解析预处理选项
        import json
        preprocessing_options = json.loads(preprocessing) if preprocessing else {}
        
        # 创建OCR任务
        task_id = await ocr_service.create_task(
            user_id=current_user.id,
            image_file=image,
            engine=engine,
            confidence_threshold=confidence_threshold,
            language=language,
            preprocessing_options=preprocessing_options
        )
        
        # 添加后台处理任务
        background_tasks.add_task(
            ocr_service.process_image_async,
            task_id=task_id
        )
        
        return OCRTaskResponse(
            success=True,
            task_id=task_id,
            message="OCR任务已提交"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR任务创建失败: {str(e)}")

@router.post("/batch", response_model=OCRTaskResponse)
async def recognize_batch(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    engine: str = Form("paddleocr"),
    confidence_threshold: float = Form(0.8),
    language: str = Form("zh,en"),
    preprocessing: str = Form("{}"),
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    批量图像OCR识别
    
    Args:
        background_tasks: 后台任务管理器
        images: 上传的图像文件列表
        engine: OCR引擎类型
        confidence_threshold: 置信度阈值
        language: 语言代码
        preprocessing: 预处理选项JSON字符串
        current_user: 当前用户
        ocr_service: OCR服务实例
    
    Returns:
        OCRTaskResponse: 批量任务创建响应
    """
    try:
        # 验证批量上传限制
        if len(images) > 50:
            raise HTTPException(status_code=400, detail="批量上传最多支持50个文件")
        
        # 验证每个文件
        for image in images:
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"文件 {image.filename} 不是图像格式")
            if image.size > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"文件 {image.filename} 超过10MB限制")
        
        # 解析预处理选项
        import json
        preprocessing_options = json.loads(preprocessing) if preprocessing else {}
        
        # 创建批量OCR任务
        task_ids = await ocr_service.create_batch_tasks(
            user_id=current_user.id,
            image_files=images,
            engine=engine,
            confidence_threshold=confidence_threshold,
            language=language,
            preprocessing_options=preprocessing_options
        )
        
        # 添加后台处理任务
        for task_id in task_ids:
            background_tasks.add_task(
                ocr_service.process_image_async,
                task_id=task_id
            )
        
        return OCRTaskResponse(
            success=True,
            task_ids=task_ids,
            message=f"批量OCR任务已提交，共{len(task_ids)}个任务"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量OCR任务创建失败: {str(e)}")

@router.get("/tasks/{task_id}", response_model=OCRResultResponse)
async def get_task_status(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    获取OCR任务状态和结果
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        ocr_service: OCR服务实例
    
    Returns:
        OCRResultResponse: 任务状态和结果
    """
    try:
        task_info = await ocr_service.get_task_status(task_id, current_user.id)
        
        if not task_info:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return OCRResultResponse(
            success=True,
            task=task_info['task'],
            result=task_info.get('result')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")

@router.get("/results", response_model=OCRTaskListResponse)
async def get_ocr_results(
    page: int = 1,
    size: int = 20,
    status: Optional[str] = None,
    engine: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    获取OCR结果列表
    
    Args:
        page: 页码
        size: 每页大小
        status: 状态筛选
        engine: 引擎筛选
        current_user: 当前用户
        ocr_service: OCR服务实例
    
    Returns:
        OCRTaskListResponse: 分页的OCR结果列表
    """
    try:
        # 验证分页参数
        if page < 1 or size < 1 or size > 100:
            raise HTTPException(status_code=400, detail="无效的分页参数")
        
        results = await ocr_service.get_user_results(
            user_id=current_user.id,
            page=page,
            size=size,
            status=status,
            engine=engine
        )
        
        return OCRTaskListResponse(
            success=True,
            data=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取OCR结果失败: {str(e)}")

@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: UUID,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    删除OCR任务
    
    Args:
        task_id: 任务ID
        current_user: 当前用户
        ocr_service: OCR服务实例
    
    Returns:
        删除结果
    """
    try:
        success = await ocr_service.delete_task(task_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或无权限删除")
        
        return {"success": True, "message": "任务删除成功"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")

@router.post("/tasks/{task_id}/retry")
async def retry_task(
    task_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    ocr_service: OCRService = Depends(get_ocr_service)
):
    """
    重试失败的OCR任务
    
    Args:
        task_id: 任务ID
        background_tasks: 后台任务管理器
        current_user: 当前用户
        ocr_service: OCR服务实例
    
    Returns:
        重试结果
    """
    try:
        success = await ocr_service.retry_task(task_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在或无法重试")
        
        # 添加后台处理任务
        background_tasks.add_task(
            ocr_service.process_image_async,
            task_id=task_id
        )
        
        return {"success": True, "message": "任务重试已提交"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重试任务失败: {str(e)}")
```

#### 配置管理

```python
# src/config/ocr_config.py
from pydantic import BaseSettings
from typing import Dict, Any, List
import os

class OCRConfig(BaseSettings):
    """OCR服务配置"""
    
    # 基础配置
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_batch_size: int = 50
    supported_formats: List[str] = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
    
    # PaddleOCR配置
    paddle_use_gpu: bool = True
    paddle_lang: str = 'ch'
    paddle_use_angle_cls: bool = True
    paddle_det_model_dir: str = None
    paddle_rec_model_dir: str = None
    paddle_cls_model_dir: str = None
    
    # Tesseract配置
    tesseract_cmd: str = 'tesseract'
    tesseract_data_path: str = None
    tesseract_lang: str = 'chi_sim+eng'
    
    # EasyOCR配置
    easyocr_gpu: bool = True
    easyocr_lang_list: List[str] = ['ch_sim', 'ch_tra', 'en']
    
    # 预处理默认配置
    default_preprocessing: Dict[str, Any] = {
        'grayscale': True,
        'denoise': True,
        'enhance_contrast': True,
        'deskew': True,
        'binarize': False,
        'resize': False,
        'scale_factor': 2.0
    }
    
    # 后处理默认配置
    default_postprocessing: Dict[str, Any] = {
        'traditional_to_simplified': True,
        'remove_extra_whitespace': True,
        'normalize_punctuation': True,
        'spell_check': False
    }
    
    # 性能配置
    max_workers: int = 4
    task_timeout: int = 300  # 5分钟
    result_cache_ttl: int = 3600  # 1小时
    
    class Config:
        env_prefix = "OCR_"
        case_sensitive = False

# 全局配置实例
ocr_config = OCRConfig()
```

#### 应用入口点

```python
# src/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn

from .controllers.ocr_controller import router as ocr_router
from .services.ocr_service import OCRService
from .config.ocr_config import ocr_config
from .middleware.auth import AuthMiddleware
from .middleware.logging import LoggingMiddleware
from .database import init_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    await init_database()
    
    # 初始化OCR服务
    ocr_service = OCRService()
    await ocr_service.initialize()
    app.state.ocr_service = ocr_service
    
    yield
    
    # 关闭时清理
    await ocr_service.cleanup()

# 创建FastAPI应用
app = FastAPI(
    title="历史文档智能分析系统 - OCR服务",
    description="提供高精度的OCR文本识别服务，支持古代汉字和多种字体识别",
    version="1.0.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)

# 注册路由
app.include_router(ocr_router)

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ocr"}

# 服务信息
@app.get("/info")
async def service_info():
    return {
        "service": "OCR文本识别服务",
        "version": "1.0.0",
        "supported_engines": ["paddleocr", "tesseract", "easyocr"],
        "max_file_size": ocr_config.max_file_size,
        "max_batch_size": ocr_config.max_batch_size,
        "supported_formats": ocr_config.supported_formats
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
```

#### Pydantic模式定义

```python
# src/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from enum import Enum

class OCREngine(str, Enum):
    """OCR引擎枚举"""
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TextBlock(BaseModel):
    """文本块模型"""
    text: str = Field(..., description="文本内容")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    bbox: List[List[float]] = Field(..., description="边界框坐标")
    language: Optional[str] = Field(None, description="检测到的语言")

class OCRResult(BaseModel):
    """OCR识别结果模型"""
    text: str = Field(..., description="完整识别文本")
    confidence: float = Field(..., ge=0, le=1, description="平均置信度")
    char_count: int = Field(..., ge=0, description="字符数")
    word_count: int = Field(..., ge=0, description="词数")
    processing_time: float = Field(..., ge=0, description="处理时间(秒)")
    text_blocks: List[TextBlock] = Field(default=[], description="文本块列表")
    metadata: Dict[str, Any] = Field(default={}, description="额外元数据")

class OCRTask(BaseModel):
    """OCR任务模型"""
    id: UUID = Field(..., description="任务ID")
    user_id: UUID = Field(..., description="用户ID")
    filename: str = Field(..., description="文件名")
    status: TaskStatus = Field(..., description="任务状态")
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比")
    engine: OCREngine = Field(..., description="使用的OCR引擎")
    confidence_threshold: float = Field(..., ge=0, le=1, description="置信度阈值")
    language: str = Field(..., description="语言设置")
    preprocessing_options: Dict[str, Any] = Field(default={}, description="预处理选项")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误信息")

class OCRTaskResponse(BaseModel):
    """OCR任务响应模型"""
    success: bool = Field(..., description="是否成功")
    task_id: Optional[UUID] = Field(None, description="单个任务ID")
    task_ids: Optional[List[UUID]] = Field(None, description="批量任务ID列表")
    message: str = Field(..., description="响应消息")

class OCRResultResponse(BaseModel):
    """OCR结果响应模型"""
    success: bool = Field(..., description="是否成功")
    task: OCRTask = Field(..., description="任务信息")
    result: Optional[OCRResult] = Field(None, description="识别结果")

class OCRTaskListResponse(BaseModel):
    """OCR任务列表响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Dict[str, Any] = Field(..., description="分页数据")

class OCRConfigRequest(BaseModel):
    """OCR配置请求模型"""
    engine: OCREngine = Field(default=OCREngine.PADDLEOCR, description="OCR引擎")
    confidence_threshold: float = Field(default=0.8, ge=0, le=1, description="置信度阈值")
    language: str = Field(default="zh,en", description="语言设置")
    preprocessing_options: Dict[str, Any] = Field(default={}, description="预处理选项")

class OCRBatchRequest(BaseModel):
    """OCR批量请求模型"""
    config: OCRConfigRequest = Field(..., description="OCR配置")
    image_paths: List[str] = Field(..., min_items=1, max_items=50, description="图像路径列表")
```

#### 依赖注入配置

```python
# src/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from .database import get_db_session
from .services.ocr_service import OCRService
from .services.auth_service import AuthService
from .models.database import User

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends()
) -> User:
    """
    获取当前认证用户
    
    Args:
        credentials: JWT凭证
        db: 数据库会话
        auth_service: 认证服务
    
    Returns:
        User: 当前用户对象
    
    Raises:
        HTTPException: 认证失败时抛出401错误
    """
    try:
        # 验证JWT令牌
        payload = auth_service.verify_token(credentials.credentials)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的认证令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 从数据库获取用户信息
        user = await auth_service.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证失败",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_ocr_service() -> OCRService:
    """
    获取OCR服务实例
    
    Returns:
        OCRService: OCR服务实例
    """
    # 这里可以实现服务的单例模式或依赖注入
    return OCRService()

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db_session),
    auth_service: AuthService = Depends()
) -> Optional[User]:
    """
    获取可选的当前用户（用于公开接口）
    
    Args:
        credentials: 可选的JWT凭证
        db: 数据库会话
        auth_service: 认证服务
    
    Returns:
        Optional[User]: 当前用户对象或None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db, auth_service)
    except HTTPException:
        return None
```

### 前端集成

#### Vue3 OCR组件
```vue
<!-- components/OCRProcessor.vue -->
<template>
  <div class="ocr-processor">
    <el-card class="config-card">
      <template #header>
        <span>OCR配置</span>
      </template>
      
      <el-form :model="ocrConfig" label-width="120px">
        <el-form-item label="OCR引擎">
          <el-select v-model="ocrConfig.engine">
            <el-option label="PaddleOCR" value="paddleocr" />
            <el-option label="Tesseract" value="tesseract" />
            <el-option label="EasyOCR" value="easyocr" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="置信度阈值">
          <el-slider 
            v-model="ocrConfig.confidence_threshold" 
            :min="0.1" 
            :max="1" 
            :step="0.1" 
            show-input
          />
        </el-form-item>
        
        <el-form-item label="预处理选项">
          <el-checkbox-group v-model="preprocessingOptions">
            <el-checkbox label="grayscale">灰度转换</el-checkbox>
            <el-checkbox label="denoise">噪声去除</el-checkbox>
            <el-checkbox label="enhance_contrast">对比度增强</el-checkbox>
            <el-checkbox label="deskew">倾斜校正</el-checkbox>
            <el-checkbox label="binarize">二值化</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
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
        multiple
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽图像到此处或 <em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 JPG、PNG、TIFF 等格式，单个文件不超过10MB
          </div>
        </template>
      </el-upload>
    </el-card>
    
    <!-- OCR任务列表 -->
    <el-card v-if="ocrTasks.length > 0" class="tasks-card">
      <template #header>
        <span>OCR任务</span>
      </template>
      
      <el-table :data="ocrTasks" style="width: 100%">
        <el-table-column prop="filename" label="文件名" width="200" />
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="150">
          <template #default="{ row }">
            <el-progress :percentage="row.progress" :status="getProgressStatus(row.status)" />
          </template>
        </el-table-column>
        <el-table-column prop="confidence" label="置信度" width="100">
          <template #default="{ row }">
            <span v-if="row.confidence">{{ (row.confidence * 100).toFixed(1) }}%</span>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column prop="processing_time" label="处理时间" width="120">
          <template #default="{ row }">
            <span v-if="row.processing_time">{{ row.processing_time.toFixed(1) }}s</span>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button 
              v-if="row.status === 'completed'" 
              type="primary" 
              size="small" 
              @click="viewResult(row)"
            >
              查看结果
            </el-button>
            <el-button 
              v-if="row.status === 'failed'" 
              type="danger" 
              size="small" 
              @click="retryTask(row)"
            >
              重试
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
    
    <!-- OCR结果对话框 -->
    <el-dialog v-model="resultDialogVisible" title="OCR识别结果" width="80%">
      <div v-if="currentResult" class="result-content">
        <el-row :gutter="20">
          <el-col :span="12">
            <h4>识别文本</h4>
            <el-input 
              v-model="currentResult.text" 
              type="textarea" 
              :rows="15" 
              readonly
            />
          </el-col>
          <el-col :span="12">
            <h4>统计信息</h4>
            <el-descriptions :column="1" border>
              <el-descriptions-item label="置信度">
                {{ (currentResult.confidence * 100).toFixed(1) }}%
              </el-descriptions-item>
              <el-descriptions-item label="字符数">
                {{ currentResult.char_count }}
              </el-descriptions-item>
              <el-descriptions-item label="词数">
                {{ currentResult.word_count }}
              </el-descriptions-item>
              <el-descriptions-item label="处理时间">
                {{ currentResult.processing_time }}秒
              </el-descriptions-item>
            </el-descriptions>
            
            <h4 style="margin-top: 20px;">文本块信息</h4>
            <el-table :data="currentResult.text_blocks" size="small" max-height="300">
              <el-table-column prop="text" label="文本" width="200" show-overflow-tooltip />
              <el-table-column prop="confidence" label="置信度" width="100">
                <template #default="{ row }">
                  {{ (row.confidence * 100).toFixed(1) }}%
                </template>
              </el-table-column>
            </el-table>
          </el-col>
        </el-row>
      </div>
      
      <template #footer>
        <el-button @click="resultDialogVisible = false">关闭</el-button>
        <el-button type="primary" @click="exportResult">导出结果</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import type { UploadFile } from 'element-plus'

interface OCRTask {
  id: string
  filename: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  confidence?: number
  processing_time?: number
  error_message?: string
}

interface OCRResult {
  text: string
  confidence: number
  char_count: number
  word_count: number
  processing_time: number
  text_blocks: Array<{
    text: string
    confidence: number
    bbox: number[][]
  }>
}

const authStore = useAuthStore()
const uploadRef = ref()
const ocrTasks = ref<OCRTask[]>([])
const currentResult = ref<OCRResult | null>(null)
const resultDialogVisible = ref(false)
const pollingInterval = ref<number | null>(null)

const ocrConfig = reactive({
  engine: 'paddleocr',
  confidence_threshold: 0.8,
  language: 'zh,en'
})

const preprocessingOptions = ref(['grayscale', 'denoise', 'enhance_contrast', 'deskew'])

const uploadUrl = computed(() => `${import.meta.env.VITE_API_BASE_URL}/api/v1/ocr/recognize`)
const uploadHeaders = computed(() => ({
  'Authorization': `Bearer ${authStore.accessToken}`
}))

const uploadData = computed(() => ({
  engine: ocrConfig.engine,
  confidence_threshold: ocrConfig.confidence_threshold,
  language: ocrConfig.language,
  preprocessing: Object.fromEntries(
    preprocessingOptions.value.map(option => [option, true])
  )
}))

/**
 * 上传前验证
 * @param file 上传文件
 */
const beforeUpload = (file: UploadFile) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp']
  const maxSize = 10 * 1024 * 1024 // 10MB
  
  if (!allowedTypes.includes(file.type || '')) {
    ElMessage.error('不支持的图像格式')
    return false
  }
  
  if (file.size! > maxSize) {
    ElMessage.error('图像文件不能超过10MB')
    return false
  }
  
  return true
}

/**
 * 上传成功处理
 * @param response 响应数据
 * @param file 文件信息
 */
const handleUploadSuccess = (response: any, file: UploadFile) => {
  if (response.success) {
    // 添加到任务列表
    ocrTasks.value.push({
      id: response.task_id,
      filename: file.name,
      status: 'processing',
      progress: 0
    })
    
    ElMessage.success(`${file.name} 上传成功，开始OCR识别`)
    
    // 开始轮询任务状态
    startPolling()
  } else {
    ElMessage.error('OCR任务提交失败')
  }
}

/**
 * 上传失败处理
 * @param error 错误信息
 * @param file 文件信息
 */
const handleUploadError = (error: any, file: UploadFile) => {
  ElMessage.error(`${file.name} 上传失败`)
}

/**
 * 开始轮询任务状态
 */
const startPolling = () => {
  if (pollingInterval.value) return
  
  pollingInterval.value = window.setInterval(async () => {
    await updateTaskStatuses()
  }, 2000) // 每2秒轮询一次
}

/**
 * 停止轮询
 */
const stopPolling = () => {
  if (pollingInterval.value) {
    clearInterval(pollingInterval.value)
    pollingInterval.value = null
  }
}

/**
 * 更新任务状态
 */
const updateTaskStatuses = async () => {
  const pendingTasks = ocrTasks.value.filter(task => 
    task.status === 'pending' || task.status === 'processing'
  )
  
  if (pendingTasks.length === 0) {
    stopPolling()
    return
  }
  
  for (const task of pendingTasks) {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/v1/ocr/tasks/${task.id}`,
        {
          headers: {
            'Authorization': `Bearer ${authStore.accessToken}`
          }
        }
      )
      
      if (response.ok) {
        const data = await response.json()
        
        task.status = data.task.status
        task.progress = data.task.progress || 0
        
        if (data.result) {
          task.confidence = data.result.confidence
          task.processing_time = data.result.processing_time
        }
        
        if (data.task.error_message) {
          task.error_message = data.task.error_message
        }
      }
    } catch (error) {
      console.error('获取任务状态失败:', error)
    }
  }
}

/**
 * 查看OCR结果
 * @param task 任务信息
 */
const viewResult = async (task: OCRTask) => {
  try {
    const response = await fetch(
      `${import.meta.env.VITE_API_BASE_URL}/api/v1/ocr/tasks/${task.id}`,
      {
        headers: {
          'Authorization': `Bearer ${authStore.accessToken}`
        }
      }
    )
    
    if (response.ok) {
      const data = await response.json()
      currentResult.value = data.result
      resultDialogVisible.value = true
    } else {
      ElMessage.error('获取OCR结果失败')
    }
  } catch (error) {
    ElMessage.error('获取OCR结果失败')
  }
}

/**
 * 重试任务
 * @param task 任务信息
 */
const retryTask = async (task: OCRTask) => {
  // 实现重试逻辑
  ElMessage.info('重试功能开发中')
}

/**
 * 导出结果
 */
const exportResult = () => {
  if (!currentResult.value) return
  
  const blob = new Blob([currentResult.value.text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'ocr_result.txt'
  link.click()
  URL.revokeObjectURL(url)
}

/**
 * 获取状态类型
 * @param status 状态
 */
const getStatusType = (status: string) => {
  const typeMap = {
    'pending': 'info',
    'processing': 'warning',
    'completed': 'success',
    'failed': 'danger'
  }
  return typeMap[status as keyof typeof typeMap] || 'info'
}

/**
 * 获取状态文本
 * @param status 状态
 */
const getStatusText = (status: string) => {
  const textMap = {
    'pending': '等待中',
    'processing': '处理中',
    'completed': '已完成',
    'failed': '失败'
  }
  return textMap[status as keyof typeof textMap] || status
}

/**
 * 获取进度状态
 * @param status 状态
 */
const getProgressStatus = (status: string) => {
  if (status === 'completed') return 'success'
  if (status === 'failed') return 'exception'
  return undefined
}

onMounted(() => {
  // 组件挂载时可以加载历史任务
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped>
.ocr-processor {
  padding: 20px;
}

.config-card,
.upload-card,
.tasks-card {
  margin-bottom: 20px;
}

.image-upload {
  width: 100%;
}

.result-content {
  max-height: 600px;
  overflow-y: auto;
}
</style>
```

## 验收标准

### 功能验收
- [ ] 支持多种OCR引擎切换
- [ ] 图像预处理功能正常
- [ ] 古代汉字识别准确率 > 85%
- [ ] 现代汉字识别准确率 > 95%
- [ ] 英文识别准确率 > 98%
- [ ] 批量处理功能正常
- [ ] 结果导出功能完整

### 性能验收
- [ ] 单张图像处理时间 < 5秒
- [ ] 并发处理能力 > 10个任务
- [ ] 内存使用稳定，无内存泄漏
- [ ] GPU加速效果明显

### 准确性验收
- [ ] 置信度评估准确
- [ ] 文字定位精确
- [ ] 文本块分割合理
- [ ] 后处理优化有效

## 业务价值
- 自动化文字识别，提高数据处理效率
- 支持古代文献的数字化处理
- 为后续AI分析提供高质量文本数据
- 减少人工录入成本和错误率

## 依赖关系
- **前置条件**: Story 1.3 (数据采集服务)
- **后续依赖**: Story 2.2 (NLP文本处理), Story 2.3 (图像处理)

## 风险与缓解
- **风险**: 古代汉字识别准确率低
- **缓解**: 训练专门的古文字识别模型
- **风险**: 处理速度慢影响用户体验
- **缓解**: GPU加速 + 异步处理队列

## 开发任务分解
1. OCR引擎集成和配置 (2天)
2. 图像预处理算法开发 (2天)
3. OCR服务API开发 (1天)
4. 批量处理和队列管理 (1天)
5. 前端OCR组件开发 (1天)
6. 性能优化和测试 (1天)