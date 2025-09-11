"""
视频内容分析器

基于帧提取和深度学习的视频内容审核
支持NSFW检测、暴力内容识别、关键帧分析等功能
"""

import os
import time
import logging
import tempfile
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image
    HAS_CV_DEPS = True
except ImportError as e:
    logging.warning(f"计算机视觉依赖缺失: {e}")
    HAS_CV_DEPS = False

from .base_analyzer import BaseAnalyzer, AnalysisResult, ViolationDetail, ViolationType, AnalysisStatus
from .image_analyzer import ImageAnalyzer, NSFWDetector, ViolenceDetector, FaceDetector

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisMetadata:
    """视频分析元数据"""
    duration: float  # 秒
    fps: float
    total_frames: int
    width: int
    height: int
    file_size: int
    codec: str
    bitrate: int
    analyzed_frames: int
    keyframes_count: int
    face_detection_frames: int
    average_brightness: float
    scene_changes: int


@dataclass
class FrameAnalysisResult:
    """单帧分析结果"""
    frame_number: int
    timestamp: float
    violations: List[ViolationDetail]
    has_faces: bool
    face_count: int
    brightness: float
    is_keyframe: bool


class FrameExtractor:
    """视频帧提取器"""
    
    def __init__(self, sampling_rate: int = 1):
        """
        初始化帧提取器
        
        Args:
            sampling_rate: 采样率 (每秒提取帧数)
        """
        self.sampling_rate = sampling_rate
    
    def extract_frames(self, video_path: str, max_frames: int = 100) -> List[Tuple[int, float, np.ndarray]]:
        """
        从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数
            
        Returns:
            List[Tuple[int, float, np.ndarray]]: (帧号, 时间戳, 图像数据) 列表
        """
        if not HAS_CV_DEPS:
            logger.error("缺少OpenCV依赖，无法提取视频帧")
            return []
        
        frames = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 计算采样间隔
            frame_interval = max(1, int(fps / self.sampling_rate))
            max_frame_count = min(max_frames, total_frames // frame_interval)
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔采样
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    frames.append((frame_count, timestamp, frame.copy()))
                    extracted_count += 1
                
                frame_count += 1
            
            logger.info(f"从视频中提取了 {len(frames)} 帧")
            return frames
            
        except Exception as e:
            logger.error(f"视频帧提取失败: {e}")
            return []
        
        finally:
            if cap is not None:
                cap.release()
    
    def extract_keyframes(self, video_path: str, threshold: float = 30.0) -> List[Tuple[int, float, np.ndarray]]:
        """
        提取视频关键帧（基于场景变化）
        
        Args:
            video_path: 视频文件路径
            threshold: 场景变化阈值
            
        Returns:
            List[Tuple[int, float, np.ndarray]]: 关键帧列表
        """
        if not HAS_CV_DEPS:
            logger.error("缺少OpenCV依赖，无法提取关键帧")
            return []
        
        keyframes = []
        cap = None
        prev_frame = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为灰度图像进行比较
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # 计算帧间差异
                    diff = cv2.absdiff(prev_frame, gray_frame)
                    diff_mean = np.mean(diff)
                    
                    # 如果差异超过阈值，认为是关键帧
                    if diff_mean > threshold:
                        timestamp = frame_count / fps
                        keyframes.append((frame_count, timestamp, frame.copy()))
                else:
                    # 第一帧总是关键帧
                    timestamp = frame_count / fps
                    keyframes.append((frame_count, timestamp, frame.copy()))
                
                prev_frame = gray_frame
                frame_count += 1
            
            logger.info(f"提取了 {len(keyframes)} 个关键帧")
            return keyframes
            
        except Exception as e:
            logger.error(f"关键帧提取失败: {e}")
            return []
        
        finally:
            if cap is not None:
                cap.release()


class VideoMetadataExtractor:
    """视频元数据提取器"""
    
    def __init__(self):
        """初始化元数据提取器"""
        pass
    
    def extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        提取视频元数据
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Dict[str, Any]: 元数据信息
        """
        if not HAS_CV_DEPS:
            return {}
        
        metadata = {}
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return metadata
            
            # 基本属性
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
            metadata['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['duration'] = metadata['total_frames'] / metadata['fps'] if metadata['fps'] > 0 else 0
            
            # 编码信息
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            metadata['codec'] = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # 文件大小
            if Path(video_path).exists():
                metadata['file_size'] = Path(video_path).stat().st_size
            
            # 比特率估算
            if metadata.get('duration', 0) > 0:
                metadata['bitrate'] = int(metadata.get('file_size', 0) * 8 / metadata['duration'] / 1000)  # kbps
            
            return metadata
            
        except Exception as e:
            logger.error(f"视频元数据提取失败: {e}")
            return {}
        
        finally:
            if cap is not None:
                cap.release()


class VideoSceneDetector:
    """视频场景检测器"""
    
    def __init__(self, threshold: float = 30.0):
        """
        初始化场景检测器
        
        Args:
            threshold: 场景变化阈值
        """
        self.threshold = threshold
    
    def detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        检测视频中的场景切换
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Tuple[float, float]]: 场景时间段列表 (开始时间, 结束时间)
        """
        if not HAS_CV_DEPS:
            return []
        
        scenes = []
        cap = None
        prev_frame = None
        scene_start = 0.0
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return scenes
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp = frame_count / fps
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray_frame)
                    diff_mean = np.mean(diff)
                    
                    if diff_mean > self.threshold:
                        # 场景切换
                        scenes.append((scene_start, timestamp))
                        scene_start = timestamp
                
                prev_frame = gray_frame
                frame_count += 1
            
            # 添加最后一个场景
            if frame_count > 0:
                final_timestamp = frame_count / fps
                scenes.append((scene_start, final_timestamp))
            
            return scenes
            
        except Exception as e:
            logger.error(f"场景检测失败: {e}")
            return []
        
        finally:
            if cap is not None:
                cap.release()


class VideoAnalyzer(BaseAnalyzer):
    """
    视频内容分析器
    
    提供全面的视频内容审核功能，包括：
    - 关键帧提取和分析
    - NSFW内容检测
    - 暴力内容识别
    - 人脸检测统计
    - 场景变化检测
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化视频分析器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        
        if not HAS_CV_DEPS:
            logger.warning("计算机视觉依赖未安装，视频分析功能受限")
        
        self.max_video_size = self.config.get("max_video_size", 500 * 1024 * 1024)  # 500MB
        self.max_duration = self.config.get("max_duration", 3600)  # 1小时
        self.sampling_rate = self.config.get("sampling_rate", 1)  # 每秒1帧
        self.max_analyzed_frames = self.config.get("max_analyzed_frames", 100)
        
        # 初始化组件
        self.frame_extractor = FrameExtractor(self.sampling_rate)
        self.metadata_extractor = VideoMetadataExtractor()
        self.scene_detector = VideoSceneDetector()
        
        # 复用图像分析器的检测器
        nsfw_model_path = self.config.get("nsfw_model_path")
        self.nsfw_detector = NSFWDetector(nsfw_model_path)
        self.violence_detector = ViolenceDetector()
        self.face_detector = FaceDetector()
        
        logger.info("视频分析器初始化完成")
    
    def get_supported_types(self) -> List[str]:
        """获取支持的视频类型"""
        return [
            "video/mp4",
            "video/avi",
            "video/mov",
            "video/mkv",
            "video/wmv",
            "video/flv",
            "video/webm"
        ]
    
    async def analyze(self, content: Union[str, bytes], metadata: Dict[str, Any] = None) -> AnalysisResult:
        """
        分析视频内容
        
        Args:
            content: 视频内容 (bytes或文件路径)
            metadata: 元数据信息
            
        Returns:
            AnalysisResult: 分析结果
        """
        start_time = time.time()
        
        if not HAS_CV_DEPS:
            return self.create_error_result(
                AnalysisStatus.UNSUPPORTED,
                "缺少计算机视觉依赖，无法进行视频分析"
            )
        
        temp_file_path = None
        
        try:
            # 处理输入内容
            video_path = await self._prepare_video_file(content)
            if not video_path:
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "无法处理视频文件"
                )
            
            temp_file_path = video_path if isinstance(content, bytes) else None
            
            # 提取视频元数据
            video_metadata = self.metadata_extractor.extract_metadata(video_path)
            if not video_metadata:
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "无法读取视频元数据"
                )
            
            # 验证视频
            if not self._validate_video(video_metadata):
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "视频不符合大小或时长要求"
                )
            
            # 提取关键帧
            frames = self.frame_extractor.extract_frames(video_path, self.max_analyzed_frames)
            if not frames:
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "无法从视频中提取帧"
                )
            
            # 分析每一帧
            frame_results = await self._analyze_frames(frames)
            
            # 场景检测
            scenes = self.scene_detector.detect_scenes(video_path)
            
            # 汇总分析结果
            violations = self._aggregate_violations(frame_results)
            overall_confidence = self._calculate_overall_confidence(violations, frame_results)
            
            # 生成分析元数据
            analysis_metadata = self._generate_metadata(
                video_metadata, frame_results, scenes, metadata
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
            logger.error(f"视频分析失败: {str(e)}")
            return self.create_error_result(
                AnalysisStatus.FAILED,
                f"分析过程中发生错误: {str(e)}",
                processing_time
            )
        
        finally:
            # 清理临时文件
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
    
    async def _prepare_video_file(self, content: Union[str, bytes]) -> Optional[str]:
        """
        准备视频文件用于分析
        
        Args:
            content: 视频内容或文件路径
            
        Returns:
            Optional[str]: 视频文件路径
        """
        if isinstance(content, str):
            # 文件路径
            if Path(content).exists():
                return content
            else:
                logger.error(f"视频文件不存在: {content}")
                return None
        
        elif isinstance(content, bytes):
            # 字节数据，保存到临时文件
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_file.write(content)
                    return temp_file.name
            except Exception as e:
                logger.error(f"保存临时视频文件失败: {e}")
                return None
        
        else:
            logger.error("不支持的视频内容类型")
            return None
    
    def _validate_video(self, metadata: Dict[str, Any]) -> bool:
        """
        验证视频是否符合要求
        
        Args:
            metadata: 视频元数据
            
        Returns:
            bool: 是否有效
        """
        # 检查时长限制
        duration = metadata.get('duration', 0)
        if duration > self.max_duration:
            logger.warning(f"视频时长 {duration} 秒超过限制 {self.max_duration} 秒")
            return False
        
        # 检查文件大小
        file_size = metadata.get('file_size', 0)
        if file_size > self.max_video_size:
            logger.warning(f"视频文件大小 {file_size} 字节超过限制 {self.max_video_size} 字节")
            return False
        
        # 检查基本参数
        if metadata.get('width', 0) < 64 or metadata.get('height', 0) < 64:
            logger.warning("视频分辨率过小")
            return False
        
        return True
    
    async def _analyze_frames(self, frames: List[Tuple[int, float, np.ndarray]]) -> List[FrameAnalysisResult]:
        """
        分析视频帧
        
        Args:
            frames: 帧数据列表
            
        Returns:
            List[FrameAnalysisResult]: 帧分析结果列表
        """
        results = []
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(5)  # 最多5个并发分析
        
        async def analyze_single_frame(frame_data):
            async with semaphore:
                frame_number, timestamp, image = frame_data
                return await self._analyze_single_frame(frame_number, timestamp, image)
        
        # 并发分析所有帧
        tasks = [analyze_single_frame(frame_data) for frame_data in frames]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]
    
    async def _analyze_single_frame(self, frame_number: int, timestamp: float, image: np.ndarray) -> Optional[FrameAnalysisResult]:
        """
        分析单个视频帧
        
        Args:
            frame_number: 帧号
            timestamp: 时间戳
            image: 图像数据
            
        Returns:
            Optional[FrameAnalysisResult]: 帧分析结果
        """
        try:
            violations = []
            
            # NSFW检测
            nsfw_result = await asyncio.get_event_loop().run_in_executor(
                None, self.nsfw_detector.detect, image
            )
            if nsfw_result:
                violations.append(nsfw_result)
            
            # 暴力内容检测
            violence_result = await asyncio.get_event_loop().run_in_executor(
                None, self.violence_detector.detect, image
            )
            if violence_result:
                violations.append(violence_result)
            
            # 人脸检测
            has_faces, face_count, _ = await asyncio.get_event_loop().run_in_executor(
                None, self.face_detector.detect_faces, image
            )
            
            # 计算亮度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            
            return FrameAnalysisResult(
                frame_number=frame_number,
                timestamp=timestamp,
                violations=violations,
                has_faces=has_faces,
                face_count=face_count,
                brightness=brightness,
                is_keyframe=True  # 简化：所有采样帧都认为是关键帧
            )
            
        except Exception as e:
            logger.error(f"帧分析失败 (帧号: {frame_number}): {e}")
            return None
    
    def _aggregate_violations(self, frame_results: List[FrameAnalysisResult]) -> List[ViolationDetail]:
        """
        汇总所有帧的违规结果
        
        Args:
            frame_results: 帧分析结果列表
            
        Returns:
            List[ViolationDetail]: 汇总的违规详情
        """
        violation_stats = {}
        
        # 统计每种违规类型的出现情况
        for frame_result in frame_results:
            for violation in frame_result.violations:
                if violation.type not in violation_stats:
                    violation_stats[violation.type] = {
                        'count': 0,
                        'total_confidence': 0.0,
                        'max_confidence': 0.0,
                        'timestamps': []
                    }
                
                stats = violation_stats[violation.type]
                stats['count'] += 1
                stats['total_confidence'] += violation.confidence
                stats['max_confidence'] = max(stats['max_confidence'], violation.confidence)
                stats['timestamps'].append(frame_result.timestamp)
        
        # 生成汇总违规详情
        aggregated_violations = []
        
        for violation_type, stats in violation_stats.items():
            # 计算平均置信度
            avg_confidence = stats['total_confidence'] / stats['count']
            
            # 如果违规帧数超过阈值，认为是确定违规
            violation_ratio = stats['count'] / len(frame_results)
            
            if violation_ratio > 0.1:  # 超过10%的帧违规
                # 使用最大置信度和违规比例的加权平均
                final_confidence = (stats['max_confidence'] * 0.7 + violation_ratio * 0.3)
                final_confidence = min(1.0, final_confidence)
                
                violation = ViolationDetail(
                    type=violation_type,
                    confidence=final_confidence,
                    description=f"在{len(frame_results)}帧中检测到{stats['count']}帧违规内容",
                    evidence={
                        "violation_frames": stats['count'],
                        "total_frames": len(frame_results),
                        "violation_ratio": violation_ratio,
                        "avg_confidence": avg_confidence,
                        "max_confidence": stats['max_confidence']
                    },
                    location={
                        "timestamps": stats['timestamps'][:10]  # 最多记录10个时间戳
                    }
                )
                aggregated_violations.append(violation)
        
        return aggregated_violations
    
    def _calculate_overall_confidence(self, violations: List[ViolationDetail], frame_results: List[FrameAnalysisResult]) -> float:
        """
        计算整体违规置信度
        
        Args:
            violations: 违规详情列表
            frame_results: 帧分析结果列表
            
        Returns:
            float: 整体置信度
        """
        if not violations:
            return 0.0
        
        # 考虑违规的严重程度和连续性
        max_confidence = max(v.confidence for v in violations)
        
        # 计算违规帧的分布情况
        violation_frame_count = sum(
            1 for frame in frame_results if len(frame.violations) > 0
        )
        
        if violation_frame_count > 0:
            violation_distribution = violation_frame_count / len(frame_results)
            # 如果违规帧分布广泛，提高置信度
            distribution_bonus = min(0.2, violation_distribution * 0.3)
            max_confidence = min(1.0, max_confidence + distribution_bonus)
        
        return max_confidence
    
    def _generate_metadata(
        self,
        video_metadata: Dict[str, Any],
        frame_results: List[FrameAnalysisResult],
        scenes: List[Tuple[float, float]],
        input_metadata: Dict[str, Any] = None
    ) -> VideoAnalysisMetadata:
        """
        生成分析元数据
        
        Args:
            video_metadata: 视频基础元数据
            frame_results: 帧分析结果
            scenes: 场景列表
            input_metadata: 输入元数据
            
        Returns:
            VideoAnalysisMetadata: 分析元数据
        """
        # 统计人脸检测结果
        face_detection_frames = sum(1 for fr in frame_results if fr.has_faces)
        
        # 计算平均亮度
        if frame_results:
            average_brightness = sum(fr.brightness for fr in frame_results) / len(frame_results)
        else:
            average_brightness = 0.0
        
        # 统计关键帧
        keyframes_count = sum(1 for fr in frame_results if fr.is_keyframe)
        
        return VideoAnalysisMetadata(
            duration=video_metadata.get('duration', 0.0),
            fps=video_metadata.get('fps', 0.0),
            total_frames=video_metadata.get('total_frames', 0),
            width=video_metadata.get('width', 0),
            height=video_metadata.get('height', 0),
            file_size=video_metadata.get('file_size', 0),
            codec=video_metadata.get('codec', 'Unknown'),
            bitrate=video_metadata.get('bitrate', 0),
            analyzed_frames=len(frame_results),
            keyframes_count=keyframes_count,
            face_detection_frames=face_detection_frames,
            average_brightness=average_brightness,
            scene_changes=len(scenes)
        )
    
    def batch_analyze_videos(self, video_paths: List[str]) -> List[AnalysisResult]:
        """
        批量分析视频
        
        Args:
            video_paths: 视频文件路径列表
            
        Returns:
            List[AnalysisResult]: 分析结果列表
        """
        results = []
        for video_path in video_paths:
            try:
                result = asyncio.run(self.analyze(video_path))
                results.append(result)
            except Exception as e:
                error_result = self.create_error_result(
                    AnalysisStatus.FAILED,
                    f"批量分析失败: {str(e)}"
                )
                results.append(error_result)
        
        return results