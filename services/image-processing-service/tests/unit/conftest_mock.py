"""
图像处理服务Mock测试配置
独立的测试配置，避免依赖问题
"""

import pytest
import os
from pathlib import Path

# 设置测试环境变量
os.environ["TESTING"] = "true"
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "ERROR"


@pytest.fixture(scope="function")
def sample_image_data():
    """示例图像数据"""
    return b"fake_image_data_for_testing" * 1000


@pytest.fixture(scope="function")
def large_image_data():
    """大尺寸图像数据"""
    return b"large_fake_image_data" * 5000


@pytest.fixture(scope="function")
def batch_image_paths():
    """批量处理测试路径"""
    return [
        "/test/batch_image_1.jpg",
        "/test/batch_image_2.png", 
        "/test/batch_image_3.tiff"
    ]


# 测试标记
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.mock = pytest.mark.mock