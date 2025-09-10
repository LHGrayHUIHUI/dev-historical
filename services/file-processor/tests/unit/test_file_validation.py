"""
FP-UNIT-002: 文件格式验证单元测试
优先级: P0 - 安全防护的关键算法
"""

import os
import tempfile
from pathlib import Path


class FileValidator:
    """文件验证器 - 模拟实现"""
    
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.html', '.jpg', '.png', '.gif'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> dict:
        """验证文件扩展名"""
        file_ext = Path(filename).suffix.lower()
        
        return {
            "valid": file_ext in cls.ALLOWED_EXTENSIONS,
            "extension": file_ext,
            "supported_formats": list(cls.ALLOWED_EXTENSIONS)
        }
    
    @classmethod
    def validate_file_size(cls, file_size: int) -> dict:
        """验证文件大小"""
        return {
            "valid": file_size <= cls.MAX_FILE_SIZE,
            "size": file_size,
            "max_allowed": cls.MAX_FILE_SIZE,
            "size_mb": round(file_size / (1024 * 1024), 2)
        }
    
    @classmethod
    def validate_file_content(cls, file_content: bytes) -> dict:
        """验证文件内容（简单的魔术字节检查）"""
        # PDF文件魔术字节
        if file_content.startswith(b'%PDF'):
            return {"valid": True, "type": "pdf", "confidence": 0.95}
        
        # DOCX文件魔术字节 (ZIP格式)
        if file_content.startswith(b'PK\x03\x04'):
            return {"valid": True, "type": "docx", "confidence": 0.8}
        
        # JPEG文件魔术字节
        if file_content.startswith(b'\xff\xd8\xff'):
            return {"valid": True, "type": "jpg", "confidence": 0.95}
        
        # PNG文件魔术字节
        if file_content.startswith(b'\x89PNG\r\n\x1a\n'):
            return {"valid": True, "type": "png", "confidence": 0.95}
        
        # 文本文件（简单检查）
        try:
            file_content.decode('utf-8')
            return {"valid": True, "type": "text", "confidence": 0.7}
        except UnicodeDecodeError:
            pass
        
        return {"valid": False, "type": "unknown", "confidence": 0.0}


class TestFileValidation:
    """文件验证测试套件"""
    
    def test_valid_file_extensions(self):
        """测试有效文件扩展名
        
        测试场景: FP-UNIT-002-001
        验证点: 支持的文件格式识别
        """
        valid_files = [
            "document.pdf",
            "report.docx", 
            "image.jpg",
            "scan.png",
            "data.txt",
            "webpage.html"
        ]
        
        for filename in valid_files:
            result = FileValidator.validate_file_extension(filename)
            assert result["valid"] is True, f"文件 {filename} 应该被认为是有效的"
            assert result["extension"] in FileValidator.ALLOWED_EXTENSIONS
        
        print("✅ FP-UNIT-002-001: 有效文件扩展名验证通过")
    
    def test_invalid_file_extensions(self):
        """测试无效文件扩展名
        
        测试场景: FP-UNIT-002-002
        验证点: 不支持格式的拒绝
        """
        invalid_files = [
            "malware.exe",
            "script.bat",
            "archive.zip",
            "code.py",
            "database.db"
        ]
        
        for filename in invalid_files:
            result = FileValidator.validate_file_extension(filename)
            assert result["valid"] is False, f"文件 {filename} 应该被拒绝"
            assert result["extension"] not in FileValidator.ALLOWED_EXTENSIONS
        
        print("✅ FP-UNIT-002-002: 无效文件扩展名拒绝通过")
    
    def test_file_size_validation(self):
        """测试文件大小验证
        
        测试场景: FP-UNIT-002-003
        验证点: 文件大小限制检查
        """
        # 测试有效大小
        valid_sizes = [
            1024,  # 1KB
            1024 * 1024,  # 1MB
            10 * 1024 * 1024,  # 10MB
            FileValidator.MAX_FILE_SIZE  # 最大允许大小
        ]
        
        for size in valid_sizes:
            result = FileValidator.validate_file_size(size)
            assert result["valid"] is True, f"大小 {size} 字节应该是有效的"
            assert result["size"] == size
        
        # 测试无效大小
        invalid_sizes = [
            FileValidator.MAX_FILE_SIZE + 1,  # 超过最大限制
            100 * 1024 * 1024  # 100MB
        ]
        
        for size in invalid_sizes:
            result = FileValidator.validate_file_size(size)
            assert result["valid"] is False, f"大小 {size} 字节应该被拒绝"
        
        print("✅ FP-UNIT-002-003: 文件大小验证通过")
    
    def test_file_content_validation(self):
        """测试文件内容验证
        
        测试场景: FP-UNIT-002-004
        验证点: 文件魔术字节检查
        """
        # 测试PDF文件
        pdf_content = b'%PDF-1.4\x01\x02\x03...'
        result = FileValidator.validate_file_content(pdf_content)
        assert result["valid"] is True
        assert result["type"] == "pdf"
        assert result["confidence"] > 0.9
        
        # 测试JPEG文件  
        jpg_content = b'\xff\xd8\xff\xe0\x00\x10JFIF...'
        result = FileValidator.validate_file_content(jpg_content)
        assert result["valid"] is True
        assert result["type"] == "jpg"
        
        # 测试PNG文件
        png_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...'
        result = FileValidator.validate_file_content(png_content)
        assert result["valid"] is True
        assert result["type"] == "png"
        
        # 测试文本文件
        text_content = "这是一个测试文本文件".encode('utf-8')
        result = FileValidator.validate_file_content(text_content)
        assert result["valid"] is True
        assert result["type"] == "text"
        
        # 测试未知文件（使用真正的二进制内容，不能被误认为文本）
        unknown_content = bytes([0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE])
        result = FileValidator.validate_file_content(unknown_content)
        assert result["valid"] is False, f"未知文件应该被拒绝，但得到: {result}"
        assert result["type"] == "unknown"
        
        print("✅ FP-UNIT-002-004: 文件内容验证通过")
    
    def test_comprehensive_file_validation(self):
        """测试综合文件验证
        
        测试场景: FP-UNIT-002-005
        验证点: 完整的文件验证流程
        """
        # 模拟一个有效的PDF文件
        test_file = {
            "filename": "test_document.pdf",
            "size": 2 * 1024 * 1024,  # 2MB
            "content": b'%PDF-1.4\x01\x02\x03...'
        }
        
        # 执行完整验证
        ext_result = FileValidator.validate_file_extension(test_file["filename"])
        size_result = FileValidator.validate_file_size(test_file["size"])
        content_result = FileValidator.validate_file_content(test_file["content"])
        
        # 综合验证结果
        overall_valid = all([
            ext_result["valid"],
            size_result["valid"], 
            content_result["valid"]
        ])
        
        assert overall_valid is True, "综合验证应该通过"
        assert content_result["type"] == "pdf"
        
        validation_report = {
            "filename": test_file["filename"],
            "overall_valid": overall_valid,
            "extension_check": ext_result["valid"],
            "size_check": size_result["valid"],
            "content_check": content_result["valid"],
            "detected_type": content_result["type"],
            "confidence": content_result["confidence"]
        }
        
        print("✅ FP-UNIT-002-005: 综合文件验证通过")
        return validation_report


if __name__ == "__main__":
    # 直接运行测试
    test_validator = TestFileValidation()
    
    print("🛡️ 开始执行文件验证单元测试...")
    test_validator.test_valid_file_extensions()
    test_validator.test_invalid_file_extensions()
    test_validator.test_file_size_validation()
    test_validator.test_file_content_validation()
    report = test_validator.test_comprehensive_file_validation()
    
    print("✅ 文件验证单元测试全部通过！")
    print(f"📊 验证报告示例: {report}")