"""
音频内容分析器

基于语音识别和音频处理的音频内容审核
支持语音转文本、敏感内容检测、音频质量分析等功能
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
    import librosa
    import numpy as np
    import soundfile as sf
    HAS_AUDIO_DEPS = True
except ImportError as e:
    logging.warning(f"音频处理依赖缺失: {e}")
    HAS_AUDIO_DEPS = False

from .base_analyzer import BaseAnalyzer, AnalysisResult, ViolationDetail, ViolationType, AnalysisStatus
from .text_analyzer import TextAnalyzer, SensitiveWordDetector

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysisMetadata:
    """音频分析元数据"""
    duration: float  # 秒
    sample_rate: int
    channels: int
    file_size: int
    format: str
    bitrate: int
    segments_analyzed: int
    speech_segments: int
    silence_ratio: float
    average_volume: float
    peak_volume: float
    spectral_centroid: float
    zero_crossing_rate: float


@dataclass
class AudioSegment:
    """音频片段"""
    start_time: float
    end_time: float
    audio_data: np.ndarray
    sample_rate: int
    is_speech: bool
    volume: float


class SpeechDetector:
    """语音检测器"""
    
    def __init__(self, min_speech_length: float = 0.5):
        """
        初始化语音检测器
        
        Args:
            min_speech_length: 最小语音长度(秒)
        """
        self.min_speech_length = min_speech_length
        self.energy_threshold = 0.01  # 能量阈值
        self.zcr_threshold = 0.1      # 过零率阈值
    
    def detect_speech_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        检测音频中的语音片段
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            List[AudioSegment]: 语音片段列表
        """
        if not HAS_AUDIO_DEPS:
            logger.error("缺少音频处理依赖，无法检测语音片段")
            return []
        
        try:
            # 计算短时能量
            frame_length = int(0.025 * sample_rate)  # 25ms窗口
            hop_length = int(0.010 * sample_rate)    # 10ms跳跃
            
            # 短时能量
            energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 过零率
            zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 语音检测
            speech_frames = (energy > self.energy_threshold) & (zcr < self.zcr_threshold)
            
            # 连接相邻的语音帧
            segments = self._merge_speech_frames(speech_frames, hop_length, sample_rate, audio_data)
            
            return segments
            
        except Exception as e:
            logger.error(f"语音检测失败: {e}")
            return []
    
    def _merge_speech_frames(self, speech_frames: np.ndarray, hop_length: int, sample_rate: int, audio_data: np.ndarray) -> List[AudioSegment]:
        """
        合并相邻的语音帧
        
        Args:
            speech_frames: 语音帧标记
            hop_length: 跳跃长度
            sample_rate: 采样率
            audio_data: 原始音频数据
            
        Returns:
            List[AudioSegment]: 合并后的语音片段
        """
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # 语音开始
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                # 语音结束
                end_frame = i
                start_time = start_frame * hop_length / sample_rate
                end_time = end_frame * hop_length / sample_rate
                
                # 检查片段长度
                if end_time - start_time >= self.min_speech_length:
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    
                    segment_audio = audio_data[start_sample:end_sample]
                    volume = np.sqrt(np.mean(segment_audio**2))
                    
                    segment = AudioSegment(
                        start_time=start_time,
                        end_time=end_time,
                        audio_data=segment_audio,
                        sample_rate=sample_rate,
                        is_speech=True,
                        volume=volume
                    )
                    segments.append(segment)
                
                in_speech = False
        
        # 处理最后一个片段
        if in_speech and len(speech_frames) > 0:
            end_frame = len(speech_frames) - 1
            start_time = start_frame * hop_length / sample_rate
            end_time = end_frame * hop_length / sample_rate
            
            if end_time - start_time >= self.min_speech_length:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                segment_audio = audio_data[start_sample:end_sample]
                volume = np.sqrt(np.mean(segment_audio**2))
                
                segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    audio_data=segment_audio,
                    sample_rate=sample_rate,
                    is_speech=True,
                    volume=volume
                )
                segments.append(segment)
        
        return segments


class MockSpeechToTextService:
    """模拟语音转文本服务"""
    
    def __init__(self):
        """初始化模拟语音转文本服务"""
        # 模拟一些常见的语音转文本结果
        self.mock_texts = [
            "这是一段正常的语音内容",
            "欢迎使用我们的产品",
            "今天天气不错",
            "请注意安全",
            "谢谢您的支持"
        ]
    
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        模拟语音转文本
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            str: 转换后的文本
        """
        # 模拟转换时间
        await asyncio.sleep(0.1)
        
        # 基于音频特征选择模拟文本
        duration = len(audio_data) / sample_rate
        volume = np.sqrt(np.mean(audio_data**2))
        
        if volume < 0.01:
            return ""  # 静音
        elif duration > 5.0:
            return f"这是一段长语音内容，持续时间约{duration:.1f}秒"
        else:
            import random
            return random.choice(self.mock_texts)


class AudioQualityAnalyzer:
    """音频质量分析器"""
    
    def __init__(self):
        """初始化音频质量分析器"""
        pass
    
    def analyze_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        分析音频质量
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            Dict[str, float]: 质量分析结果
        """
        if not HAS_AUDIO_DEPS:
            return {}
        
        try:
            # 音量统计
            rms_energy = np.sqrt(np.mean(audio_data**2))
            peak_volume = np.max(np.abs(audio_data))
            
            # 静音比例
            silence_threshold = 0.01
            silence_frames = np.sum(np.abs(audio_data) < silence_threshold)
            silence_ratio = silence_frames / len(audio_data)
            
            # 频谱质心
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            
            # 过零率
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            return {
                "average_volume": float(rms_energy),
                "peak_volume": float(peak_volume),
                "silence_ratio": float(silence_ratio),
                "spectral_centroid": float(spectral_centroid),
                "zero_crossing_rate": float(zcr)
            }
            
        except Exception as e:
            logger.error(f"音频质量分析失败: {e}")
            return {}


class AudioAnalyzer(BaseAnalyzer):
    """
    音频内容分析器
    
    提供全面的音频内容审核功能，包括：
    - 语音检测和分割
    - 语音转文本
    - 文本内容审核
    - 音频质量分析
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化音频分析器
        
        Args:
            config: 配置参数
        """
        super().__init__(config)
        
        if not HAS_AUDIO_DEPS:
            logger.warning("音频处理依赖未安装，音频分析功能受限")
        
        self.max_audio_size = self.config.get("max_audio_size", 100 * 1024 * 1024)  # 100MB
        self.max_duration = self.config.get("max_duration", 1800)  # 30分钟
        self.segment_length = self.config.get("segment_length", 30.0)  # 30秒片段
        
        # 初始化组件
        self.speech_detector = SpeechDetector()
        self.stt_service = MockSpeechToTextService()
        self.quality_analyzer = AudioQualityAnalyzer()
        
        # 文本分析器用于分析转换后的文本
        self.text_analyzer = TextAnalyzer(self.config.get("text_analyzer_config", {}))
        
        logger.info("音频分析器初始化完成")
    
    def get_supported_types(self) -> List[str]:
        """获取支持的音频类型"""
        return [
            "audio/mp3",
            "audio/wav",
            "audio/aac",
            "audio/ogg",
            "audio/flac",
            "audio/m4a",
            "audio/wma"
        ]
    
    async def analyze(self, content: Union[str, bytes], metadata: Dict[str, Any] = None) -> AnalysisResult:
        """
        分析音频内容
        
        Args:
            content: 音频内容 (bytes或文件路径)
            metadata: 元数据信息
            
        Returns:
            AnalysisResult: 分析结果
        """
        start_time = time.time()
        
        if not HAS_AUDIO_DEPS:
            return self.create_error_result(
                AnalysisStatus.UNSUPPORTED,
                "缺少音频处理依赖，无法进行音频分析"
            )
        
        temp_file_path = None
        
        try:
            # 处理输入内容
            audio_path = await self._prepare_audio_file(content)
            if not audio_path:
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "无法处理音频文件"
                )
            
            temp_file_path = audio_path if isinstance(content, bytes) else None
            
            # 加载音频数据
            audio_data, sample_rate = await self._load_audio(audio_path)
            if audio_data is None:
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "无法读取音频数据"
                )
            
            # 验证音频
            if not self._validate_audio(audio_data, sample_rate):
                return self.create_error_result(
                    AnalysisStatus.FAILED,
                    "音频不符合大小或时长要求"
                )
            
            # 语音检测和分割
            speech_segments = self.speech_detector.detect_speech_segments(audio_data, sample_rate)
            
            # 语音转文本和内容分析
            violations = []
            transcribed_texts = []
            
            for segment in speech_segments:
                # 语音转文本
                text = await self.stt_service.transcribe(segment.audio_data, segment.sample_rate)
                
                if text.strip():
                    transcribed_texts.append({
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "text": text
                    })
                    
                    # 文本内容审核
                    text_result = await self.text_analyzer.analyze(text)
                    
                    if text_result.is_violation:
                        # 为每个违规添加时间位置信息
                        for violation in text_result.violations:
                            violation.location = violation.location or {}
                            violation.location["audio_segment"] = {
                                "start_time": segment.start_time,
                                "end_time": segment.end_time
                            }
                            violations.append(violation)
            
            # 音频质量分析
            quality_analysis = self.quality_analyzer.analyze_quality(audio_data, sample_rate)
            
            # 检查音频中的异常模式
            audio_violations = self._detect_audio_patterns(audio_data, sample_rate, quality_analysis)
            violations.extend(audio_violations)
            
            # 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(violations)
            
            # 生成分析元数据
            analysis_metadata = self._generate_metadata(
                audio_data, sample_rate, speech_segments, quality_analysis, 
                transcribed_texts, metadata
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
            logger.error(f"音频分析失败: {str(e)}")
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
    
    async def _prepare_audio_file(self, content: Union[str, bytes]) -> Optional[str]:
        """
        准备音频文件用于分析
        
        Args:
            content: 音频内容或文件路径
            
        Returns:
            Optional[str]: 音频文件路径
        """
        if isinstance(content, str):
            # 文件路径
            if Path(content).exists():
                return content
            else:
                logger.error(f"音频文件不存在: {content}")
                return None
        
        elif isinstance(content, bytes):
            # 字节数据，保存到临时文件
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(content)
                    return temp_file.name
            except Exception as e:
                logger.error(f"保存临时音频文件失败: {e}")
                return None
        
        else:
            logger.error("不支持的音频内容类型")
            return None
    
    async def _load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[int]]: (音频数据, 采样率)
        """
        try:
            # 使用librosa加载音频
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"音频加载失败: {e}")
            return None, None
    
    def _validate_audio(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """
        验证音频是否符合要求
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            bool: 是否有效
        """
        # 检查时长限制
        duration = len(audio_data) / sample_rate
        if duration > self.max_duration:
            logger.warning(f"音频时长 {duration} 秒超过限制 {self.max_duration} 秒")
            return False
        
        # 检查采样率
        if sample_rate < 8000:  # 最低8kHz
            logger.warning(f"音频采样率 {sample_rate} Hz 过低")
            return False
        
        # 检查数据完整性
        if len(audio_data) == 0:
            logger.warning("音频数据为空")
            return False
        
        return True
    
    def _detect_audio_patterns(self, audio_data: np.ndarray, sample_rate: int, quality_analysis: Dict[str, Any]) -> List[ViolationDetail]:
        """
        检测音频中的异常模式
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            quality_analysis: 质量分析结果
            
        Returns:
            List[ViolationDetail]: 违规详情列表
        """
        violations = []
        
        try:
            # 检查音频质量异常
            silence_ratio = quality_analysis.get("silence_ratio", 0.0)
            
            # 如果静音比例过高，可能是无效内容
            if silence_ratio > 0.9:
                violation = ViolationDetail(
                    type=ViolationType.SPAM,
                    confidence=0.6,
                    description=f"音频静音比例过高: {silence_ratio:.2f}",
                    evidence={"silence_ratio": silence_ratio}
                )
                violations.append(violation)
            
            # 检查音量异常
            average_volume = quality_analysis.get("average_volume", 0.0)
            peak_volume = quality_analysis.get("peak_volume", 0.0)
            
            # 音量过低可能是无效内容
            if average_volume < 0.001 and peak_volume < 0.01:
                violation = ViolationDetail(
                    type=ViolationType.SPAM,
                    confidence=0.5,
                    description="音频音量异常低，可能是无效内容",
                    evidence={
                        "average_volume": average_volume,
                        "peak_volume": peak_volume
                    }
                )
                violations.append(violation)
            
            # 检查频谱异常
            spectral_centroid = quality_analysis.get("spectral_centroid", 0.0)
            
            # 频谱质心异常可能表示非正常语音
            if spectral_centroid > sample_rate * 0.3:  # 超过奈奎斯特频率的30%
                violation = ViolationDetail(
                    type=ViolationType.SPAM,
                    confidence=0.4,
                    description="音频频谱特征异常，可能包含噪声或非语音内容",
                    evidence={"spectral_centroid": spectral_centroid}
                )
                violations.append(violation)
        
        except Exception as e:
            logger.error(f"音频模式检测失败: {e}")
        
        return violations
    
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
        
        # 使用最高置信度作为基础
        max_confidence = max(v.confidence for v in violations)
        
        # 如果有多种违规类型，稍微提高置信度
        violation_types = set(v.type for v in violations)
        if len(violation_types) > 1:
            type_bonus = min(0.1, len(violation_types) * 0.03)
            max_confidence = min(1.0, max_confidence + type_bonus)
        
        return max_confidence
    
    def _generate_metadata(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speech_segments: List[AudioSegment],
        quality_analysis: Dict[str, Any],
        transcribed_texts: List[Dict[str, Any]],
        input_metadata: Dict[str, Any] = None
    ) -> AudioAnalysisMetadata:
        """
        生成分析元数据
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            speech_segments: 语音片段
            quality_analysis: 质量分析结果
            transcribed_texts: 转录文本
            input_metadata: 输入元数据
            
        Returns:
            AudioAnalysisMetadata: 分析元数据
        """
        duration = len(audio_data) / sample_rate
        channels = 1  # 假设单声道
        
        # 文件信息
        file_size = input_metadata.get("file_size", 0) if input_metadata else 0
        audio_format = input_metadata.get("format", "Unknown") if input_metadata else "Unknown"
        
        # 比特率估算
        if duration > 0:
            bitrate = int(file_size * 8 / duration / 1000)  # kbps
        else:
            bitrate = 0
        
        return AudioAnalysisMetadata(
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            file_size=file_size,
            format=audio_format,
            bitrate=bitrate,
            segments_analyzed=len(speech_segments),
            speech_segments=len([s for s in speech_segments if s.is_speech]),
            silence_ratio=quality_analysis.get("silence_ratio", 0.0),
            average_volume=quality_analysis.get("average_volume", 0.0),
            peak_volume=quality_analysis.get("peak_volume", 0.0),
            spectral_centroid=quality_analysis.get("spectral_centroid", 0.0),
            zero_crossing_rate=quality_analysis.get("zero_crossing_rate", 0.0)
        )
    
    def batch_analyze_audio(self, audio_paths: List[str]) -> List[AnalysisResult]:
        """
        批量分析音频
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            List[AnalysisResult]: 分析结果列表
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = asyncio.run(self.analyze(audio_path))
                results.append(result)
            except Exception as e:
                error_result = self.create_error_result(
                    AnalysisStatus.FAILED,
                    f"批量分析失败: {str(e)}"
                )
                results.append(error_result)
        
        return results