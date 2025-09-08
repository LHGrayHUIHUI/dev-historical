"""
图像处理服务测试配置
Pytest fixtures和测试工具
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io

# 设置测试环境变量
os.environ["TESTING"] = "true"
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["STORAGE_SERVICE_URL"] = "http://localhost:8002"

from src.main import app
from src.config.settings import settings


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """FastAPI测试客户端"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """临时目录fixture"""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture(scope="function")
def sample_image() -> bytes:
    """创建测试用的简单图像"""
    # 创建一个简单的RGB图像
    image = Image.new('RGB', (100, 100), color='white')
    
    # 添加一些内容使其更真实
    pixels = np.array(image)
    pixels[20:80, 20:80] = [0, 0, 0]  # 黑色方块
    pixels[30:70, 30:70] = [255, 255, 255]  # 白色方块
    
    image = Image.fromarray(pixels)
    
    # 转换为字节
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


@pytest.fixture(scope="function")
def sample_image_file(temp_dir: Path, sample_image: bytes) -> Path:
    """保存测试图像到临时文件"""
    image_path = temp_dir / "test_image.jpg"
    with open(image_path, "wb") as f:
        f.write(sample_image)
    return image_path


@pytest.fixture(scope="function") 
def noisy_image() -> bytes:
    """创建有噪声的测试图像"""
    # 创建基础图像
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    # 添加高斯噪声
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = np.clip(image + noise, 0, 255)
    
    # 转换为PIL图像并保存为字节
    pil_image = Image.fromarray(noisy)
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


@pytest.fixture(scope="function")
def skewed_image() -> bytes:
    """创建倾斜的测试图像"""
    # 创建文档样式的图像
    image = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # 添加一些文本线条
    for i in range(50, 150, 20):
        image[i:i+5, 50:250] = [0, 0, 0]
    
    # 将图像旋转5度
    pil_image = Image.fromarray(image)
    rotated = pil_image.rotate(5, expand=True, fillcolor=(255, 255, 255))
    
    img_bytes = io.BytesIO()
    rotated.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


@pytest.fixture(scope="function")
def large_image() -> bytes:
    """创建大尺寸测试图像"""
    image = Image.new('RGB', (2000, 1500), color='lightblue')
    
    # 添加一些图案
    pixels = np.array(image)
    for i in range(0, 2000, 100):
        pixels[:, i:i+50] = [255, 0, 0]  # 红色条纹
    
    image = Image.fromarray(pixels)
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG', quality=90)
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


@pytest.fixture(scope="function")
def processing_config():
    """标准处理配置"""
    return {
        "brightness_factor": 1.1,
        "contrast_factor": 1.2,
        "sharpness_factor": 1.0,
        "denoise_strength": 10,
        "resize_width": 800,
        "resize_height": 600,
        "output_format": "png",
        "quality": 95
    }


@pytest.fixture(scope="function")
def batch_image_paths(temp_dir: Path, sample_image: bytes) -> list[str]:
    """创建批量处理测试用的图像路径列表"""
    paths = []
    for i in range(3):
        image_path = temp_dir / f"batch_image_{i}.jpg"
        with open(image_path, "wb") as f:
            f.write(sample_image)
        paths.append(str(image_path))
    return paths


class MockStorageClient:
    """Mock Storage Service客户端用于测试"""
    
    def __init__(self):
        self.tasks = {}
        self.results = {}
        self.task_counter = 1
    
    async def create_image_processing_task(self, **kwargs):
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        task_data = {
            "task_id": task_id,
            "status": "pending",
            "created_at": "2024-01-01T00:00:00Z",
            **kwargs
        }
        self.tasks[task_id] = task_data
        return {"data": task_data}
    
    async def update_task_status(self, task_id: str, status: str, **kwargs):
        if task_id in self.tasks:
            self.tasks[task_id].update({
                "status": status,
                **kwargs
            })
        return {"data": self.tasks.get(task_id)}
    
    async def get_task(self, task_id: str):
        return {"data": self.tasks.get(task_id)}
    
    async def save_processing_result(self, **kwargs):
        result_id = f"result_{len(self.results) + 1}"
        self.results[result_id] = kwargs
        return {"data": {"result_id": result_id, **kwargs}}
    
    async def upload_processed_image(self, image_data: bytes, filename: str, **kwargs):
        return {
            "data": {
                "file_id": f"file_{len(self.results) + 1}",
                "filename": filename,
                "size": len(image_data),
                "path": f"/uploads/{filename}"
            }
        }


@pytest.fixture(scope="function")
def mock_storage_client():
    """Mock storage client fixture"""
    return MockStorageClient()


# 测试标记
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow