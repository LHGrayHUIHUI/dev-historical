"""
CORE-UNIT-001: 通用工具函数单元测试
优先级: P0 - 基础设施稳定性
"""

import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


class CommonUtils:
    """通用工具类 - 模拟实现"""
    
    @staticmethod
    def generate_id(prefix: str = "") -> str:
        """生成唯一ID"""
        import uuid
        base_id = str(uuid.uuid4()).replace('-', '')
        return f"{prefix}{base_id}" if prefix else base_id
    
    @staticmethod
    def safe_get(data: Dict, key: str, default: Any = None) -> Any:
        """安全获取字典值"""
        try:
            keys = key.split('.')
            result = data
            for k in keys:
                if isinstance(result, dict) and k in result:
                    result = result[k]
                else:
                    return default
            return result
        except Exception:
            return default
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def calculate_file_hash(content: bytes, algorithm: str = "md5") -> str:
        """计算文件哈希值"""
        if algorithm == "md5":
            return hashlib.md5(content).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(content).hexdigest()
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本内容"""
        if not text:
            return ""
        
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text.strip())
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return text
    
    @staticmethod
    def parse_date_range(date_str: str) -> Optional[Dict[str, datetime]]:
        """解析日期范围"""
        try:
            if 'to' in date_str or '至' in date_str:
                parts = date_str.replace('至', 'to').split('to')
                start_date = datetime.fromisoformat(parts[0].strip())
                end_date = datetime.fromisoformat(parts[1].strip())
                return {"start": start_date, "end": end_date}
            else:
                single_date = datetime.fromisoformat(date_str.strip())
                return {"start": single_date, "end": single_date}
        except ValueError:
            return None


class TestCommonUtils:
    """通用工具函数测试套件"""
    
    def test_generate_id(self):
        """测试ID生成功能
        
        测试场景: CORE-UNIT-001-001
        验证点: ID生成的唯一性和格式
        """
        # 测试无前缀ID生成
        id1 = CommonUtils.generate_id()
        id2 = CommonUtils.generate_id()
        
        assert id1 != id2, "生成的ID应该是唯一的"
        assert len(id1) == 32, "无前缀ID长度应该是32"
        
        # 测试带前缀ID生成
        prefixed_id = CommonUtils.generate_id("test_")
        assert prefixed_id.startswith("test_"), "带前缀ID应该以指定前缀开始"
        assert len(prefixed_id) == 37, "带前缀ID长度应该正确"
        
        print("✅ CORE-UNIT-001-001: ID生成测试通过")
    
    def test_safe_get(self):
        """测试安全字典取值
        
        测试场景: CORE-UNIT-001-002
        验证点: 安全访问嵌套字典数据
        """
        test_data = {
            "user": {
                "profile": {
                    "name": "张三",
                    "age": 30
                },
                "settings": {
                    "language": "zh-CN"
                }
            },
            "config": {
                "debug": True
            }
        }
        
        # 测试正常访问
        name = CommonUtils.safe_get(test_data, "user.profile.name")
        assert name == "张三"
        
        age = CommonUtils.safe_get(test_data, "user.profile.age")
        assert age == 30
        
        debug = CommonUtils.safe_get(test_data, "config.debug")
        assert debug is True
        
        # 测试不存在的键
        nonexistent = CommonUtils.safe_get(test_data, "user.profile.email", "default@example.com")
        assert nonexistent == "default@example.com"
        
        # 测试深层不存在的键
        deep_nonexistent = CommonUtils.safe_get(test_data, "user.preferences.theme", "light")
        assert deep_nonexistent == "light"
        
        print("✅ CORE-UNIT-001-002: 安全字典取值测试通过")
    
    def test_validate_email(self):
        """测试邮箱验证
        
        测试场景: CORE-UNIT-001-003
        验证点: 邮箱格式验证准确性
        """
        # 测试有效邮箱
        valid_emails = [
            "user@example.com",
            "test.email@domain.co.uk",
            "user+tag@example.org",
            "user123@test-domain.com"
        ]
        
        for email in valid_emails:
            assert CommonUtils.validate_email(email), f"{email} 应该是有效邮箱"
        
        # 测试无效邮箱
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user@domain",
            "user..email@domain.com",
            ""
        ]
        
        for email in invalid_emails:
            assert not CommonUtils.validate_email(email), f"{email} 应该是无效邮箱"
        
        print("✅ CORE-UNIT-001-003: 邮箱验证测试通过")
    
    def test_calculate_file_hash(self):
        """测试文件哈希计算
        
        测试场景: CORE-UNIT-001-004
        验证点: 哈希值计算准确性和一致性
        """
        test_content = "这是一个测试文件内容".encode('utf-8')
        
        # 测试MD5哈希
        md5_hash1 = CommonUtils.calculate_file_hash(test_content, "md5")
        md5_hash2 = CommonUtils.calculate_file_hash(test_content, "md5")
        assert md5_hash1 == md5_hash2, "相同内容的MD5哈希应该一致"
        assert len(md5_hash1) == 32, "MD5哈希长度应该是32"
        
        # 测试SHA256哈希
        sha256_hash1 = CommonUtils.calculate_file_hash(test_content, "sha256")
        sha256_hash2 = CommonUtils.calculate_file_hash(test_content, "sha256")
        assert sha256_hash1 == sha256_hash2, "相同内容的SHA256哈希应该一致"
        assert len(sha256_hash1) == 64, "SHA256哈希长度应该是64"
        
        # 测试不同内容产生不同哈希
        different_content = "不同的内容".encode('utf-8')
        different_hash = CommonUtils.calculate_file_hash(different_content, "md5")
        assert different_hash != md5_hash1, "不同内容应该产生不同哈希"
        
        # 测试不支持的算法
        try:
            CommonUtils.calculate_file_hash(test_content, "sha1")
            assert False, "应该抛出不支持算法的异常"
        except ValueError as e:
            assert "不支持的哈希算法" in str(e)
        
        print("✅ CORE-UNIT-001-004: 文件哈希计算测试通过")
    
    def test_format_file_size(self):
        """测试文件大小格式化
        
        测试场景: CORE-UNIT-001-005
        验证点: 文件大小的人类可读格式化
        """
        # 测试各种文件大小
        test_cases = [
            (0, "0 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024 * 1024, "1.0 MB"),
            (1.5 * 1024 * 1024, "1.5 MB"),
            (1024 * 1024 * 1024, "1.0 GB"),
            (2.5 * 1024 * 1024 * 1024, "2.5 GB")
        ]
        
        for size_bytes, expected in test_cases:
            result = CommonUtils.format_file_size(int(size_bytes))
            assert result == expected, f"大小 {size_bytes} 格式化结果应该是 {expected}，实际是 {result}"
        
        print("✅ CORE-UNIT-001-005: 文件大小格式化测试通过")
    
    def test_clean_text(self):
        """测试文本清理
        
        测试场景: CORE-UNIT-001-006
        验证点: 文本内容清理和标准化
        """
        # 测试多余空白清理
        messy_text = "  这是   一个   有很多    空白的    文本  "
        cleaned = CommonUtils.clean_text(messy_text)
        assert cleaned == "这是 一个 有很多 空白的 文本", f"清理后应该是单个空格，实际是: '{cleaned}'"
        
        # 测试特殊字符清理
        special_chars_text = "文本!@#$%包含&*()特殊+=[]字符{}|\\:;\"'<>?/.,~`"
        cleaned_special = CommonUtils.clean_text(special_chars_text)
        assert "!" not in cleaned_special, "特殊字符应该被清理"
        assert "文本" in cleaned_special, "中文字符应该保留"
        
        # 测试空文本
        assert CommonUtils.clean_text("") == ""
        assert CommonUtils.clean_text(None) == ""
        
        print("✅ CORE-UNIT-001-006: 文本清理测试通过")
    
    def test_parse_date_range(self):
        """测试日期范围解析
        
        测试场景: CORE-UNIT-001-007
        验证点: 日期字符串解析功能
        """
        # 测试单个日期
        single_date_result = CommonUtils.parse_date_range("2025-09-09")
        assert single_date_result is not None
        assert single_date_result["start"].year == 2025
        assert single_date_result["start"] == single_date_result["end"]
        
        # 测试日期范围 (英文)
        range_result = CommonUtils.parse_date_range("2025-09-01 to 2025-09-30")
        assert range_result is not None
        assert range_result["start"].day == 1
        assert range_result["end"].day == 30
        
        # 测试日期范围 (中文)
        chinese_range = CommonUtils.parse_date_range("2025-09-01 至 2025-09-30")
        assert chinese_range is not None
        assert chinese_range["start"].day == 1
        assert chinese_range["end"].day == 30
        
        # 测试无效日期
        invalid_result = CommonUtils.parse_date_range("invalid-date")
        assert invalid_result is None
        
        print("✅ CORE-UNIT-001-007: 日期范围解析测试通过")


if __name__ == "__main__":
    # 直接运行测试
    test_utils = TestCommonUtils()
    
    print("🛠️ 开始执行通用工具函数单元测试...")
    test_utils.test_generate_id()
    test_utils.test_safe_get()
    test_utils.test_validate_email()
    test_utils.test_calculate_file_hash()
    test_utils.test_format_file_size()
    test_utils.test_clean_text()
    test_utils.test_parse_date_range()
    
    print("✅ 通用工具函数单元测试全部通过！")