"""
数据加密服务

提供敏感数据的加密和解密功能
使用AES-256加密算法保护OAuth令牌和其他敏感信息
"""

import base64
import json
import logging
import secrets
from typing import Any, Dict, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

from ..config.settings import settings
from ..utils.exceptions import EncryptionError, DecryptionError

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    加密服务类
    
    提供AES-256加密算法的数据加密和解密功能
    专门用于保护OAuth令牌、用户凭据等敏感信息
    """
    
    def __init__(self):
        """初始化加密服务"""
        self.master_key = settings.encryption_key.encode('utf-8')
        self._fernet_key = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """初始化加密组件"""
        try:
            # 验证主密钥长度
            if len(self.master_key) < 32:
                # 使用PBKDF2扩展密钥
                salt = b'historical-text-salt-2024'
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )
                derived_key = kdf.derive(self.master_key)
            else:
                derived_key = self.master_key[:32]
            
            # 创建Fernet实例
            fernet_key = base64.urlsafe_b64encode(derived_key)
            self._fernet_key = Fernet(fernet_key)
            
            logger.info("加密服务初始化成功")
            
        except Exception as e:
            logger.error(f"加密服务初始化失败: {e}")
            raise EncryptionError(f"加密服务初始化失败: {str(e)}")
    
    def encrypt_string(self, plaintext: str) -> str:
        """
        加密字符串
        
        Args:
            plaintext: 需要加密的明文字符串
            
        Returns:
            str: Base64编码的加密字符串
            
        Raises:
            EncryptionError: 加密失败
        """
        try:
            if not plaintext:
                return ""
            
            # 使用Fernet加密
            plaintext_bytes = plaintext.encode('utf-8')
            encrypted_bytes = self._fernet_key.encrypt(plaintext_bytes)
            
            # Base64编码返回
            encrypted_str = base64.urlsafe_b64encode(encrypted_bytes).decode('utf-8')
            
            logger.debug("字符串加密成功")
            return encrypted_str
            
        except Exception as e:
            logger.error(f"字符串加密失败: {e}")
            raise EncryptionError(f"字符串加密失败: {str(e)}")
    
    def decrypt_string(self, encrypted_str: str) -> str:
        """
        解密字符串
        
        Args:
            encrypted_str: Base64编码的加密字符串
            
        Returns:
            str: 解密后的明文字符串
            
        Raises:
            DecryptionError: 解密失败
        """
        try:
            if not encrypted_str:
                return ""
            
            # Base64解码
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_str.encode('utf-8'))
            
            # 使用Fernet解密
            plaintext_bytes = self._fernet_key.decrypt(encrypted_bytes)
            plaintext = plaintext_bytes.decode('utf-8')
            
            logger.debug("字符串解密成功")
            return plaintext
            
        except Exception as e:
            logger.error(f"字符串解密失败: {e}")
            raise DecryptionError(f"字符串解密失败: {str(e)}")
    
    def encrypt_json(self, data: Dict[str, Any]) -> str:
        """
        加密JSON数据
        
        Args:
            data: 需要加密的字典数据
            
        Returns:
            str: Base64编码的加密JSON字符串
            
        Raises:
            EncryptionError: 加密失败
        """
        try:
            if not data:
                return ""
            
            # 序列化为JSON字符串
            json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
            
            # 加密JSON字符串
            encrypted_json = self.encrypt_string(json_str)
            
            logger.debug("JSON数据加密成功")
            return encrypted_json
            
        except json.JSONEncodeError as e:
            logger.error(f"JSON序列化失败: {e}")
            raise EncryptionError(f"JSON序列化失败: {str(e)}")
        except Exception as e:
            logger.error(f"JSON加密失败: {e}")
            raise EncryptionError(f"JSON加密失败: {str(e)}")
    
    def decrypt_json(self, encrypted_json: str) -> Dict[str, Any]:
        """
        解密JSON数据
        
        Args:
            encrypted_json: Base64编码的加密JSON字符串
            
        Returns:
            Dict: 解密后的字典数据
            
        Raises:
            DecryptionError: 解密失败
        """
        try:
            if not encrypted_json:
                return {}
            
            # 解密JSON字符串
            json_str = self.decrypt_string(encrypted_json)
            
            # 反序列化JSON数据
            data = json.loads(json_str)
            
            logger.debug("JSON数据解密成功")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON反序列化失败: {e}")
            raise DecryptionError(f"JSON反序列化失败: {str(e)}")
        except Exception as e:
            logger.error(f"JSON解密失败: {e}")
            raise DecryptionError(f"JSON解密失败: {str(e)}")
    
    def encrypt_oauth_credentials(self, credentials: Dict[str, Any]) -> str:
        """
        加密OAuth凭据
        
        Args:
            credentials: OAuth凭据字典，包含access_token、refresh_token等
            
        Returns:
            str: 加密后的凭据字符串
            
        Raises:
            EncryptionError: 加密失败
        """
        try:
            # 提取敏感字段
            sensitive_fields = {
                'access_token': credentials.get('access_token'),
                'refresh_token': credentials.get('refresh_token'),
                'client_secret': credentials.get('client_secret'),
                'api_key': credentials.get('api_key'),
                'api_secret': credentials.get('api_secret')
            }
            
            # 过滤空值
            filtered_credentials = {
                k: v for k, v in sensitive_fields.items() 
                if v is not None and v != ""
            }
            
            # 添加时间戳
            filtered_credentials['encrypted_at'] = int(
                __import__('time').time()
            )
            
            encrypted_credentials = self.encrypt_json(filtered_credentials)
            
            logger.debug("OAuth凭据加密成功")
            return encrypted_credentials
            
        except Exception as e:
            logger.error(f"OAuth凭据加密失败: {e}")
            raise EncryptionError(f"OAuth凭据加密失败: {str(e)}")
    
    def decrypt_oauth_credentials(self, encrypted_credentials: str) -> Dict[str, Any]:
        """
        解密OAuth凭据
        
        Args:
            encrypted_credentials: 加密后的凭据字符串
            
        Returns:
            Dict: 解密后的OAuth凭据字典
            
        Raises:
            DecryptionError: 解密失败
        """
        try:
            credentials = self.decrypt_json(encrypted_credentials)
            
            # 移除时间戳
            credentials.pop('encrypted_at', None)
            
            logger.debug("OAuth凭据解密成功")
            return credentials
            
        except Exception as e:
            logger.error(f"OAuth凭据解密失败: {e}")
            raise DecryptionError(f"OAuth凭据解密失败: {str(e)}")
    
    def generate_salt(self, length: int = 32) -> str:
        """
        生成随机盐值
        
        Args:
            length: 盐值长度(字节)
            
        Returns:
            str: Base64编码的盐值
        """
        salt_bytes = os.urandom(length)
        salt_str = base64.urlsafe_b64encode(salt_bytes).decode('utf-8')
        
        logger.debug(f"生成 {length} 字节盐值")
        return salt_str
    
    def hash_password(self, password: str, salt: str = None) -> tuple[str, str]:
        """
        使用PBKDF2哈希密码
        
        Args:
            password: 明文密码
            salt: 可选的盐值，如果不提供则生成新盐值
            
        Returns:
            tuple: (哈希值, 盐值)
            
        Raises:
            EncryptionError: 哈希失败
        """
        try:
            if salt is None:
                salt = self.generate_salt()
            
            # Base64解码盐值
            salt_bytes = base64.urlsafe_b64decode(salt.encode('utf-8'))
            
            # 使用PBKDF2哈希
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
                backend=default_backend()
            )
            
            password_bytes = password.encode('utf-8')
            hashed_bytes = kdf.derive(password_bytes)
            hashed_str = base64.urlsafe_b64encode(hashed_bytes).decode('utf-8')
            
            logger.debug("密码哈希成功")
            return hashed_str, salt
            
        except Exception as e:
            logger.error(f"密码哈希失败: {e}")
            raise EncryptionError(f"密码哈希失败: {str(e)}")
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        验证密码
        
        Args:
            password: 明文密码
            hashed_password: 哈希密码
            salt: 盐值
            
        Returns:
            bool: 密码是否匹配
        """
        try:
            # 使用相同的盐值重新哈希
            new_hash, _ = self.hash_password(password, salt)
            
            # 比较哈希值
            is_valid = secrets.compare_digest(hashed_password, new_hash)
            
            logger.debug(f"密码验证结果: {is_valid}")
            return is_valid
            
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False
    
    def encrypt_file_content(self, content: bytes) -> bytes:
        """
        加密文件内容
        
        Args:
            content: 文件内容字节
            
        Returns:
            bytes: 加密后的内容
            
        Raises:
            EncryptionError: 加密失败
        """
        try:
            if not content:
                return b""
            
            # 使用Fernet加密
            encrypted_content = self._fernet_key.encrypt(content)
            
            logger.debug("文件内容加密成功")
            return encrypted_content
            
        except Exception as e:
            logger.error(f"文件内容加密失败: {e}")
            raise EncryptionError(f"文件内容加密失败: {str(e)}")
    
    def decrypt_file_content(self, encrypted_content: bytes) -> bytes:
        """
        解密文件内容
        
        Args:
            encrypted_content: 加密后的内容
            
        Returns:
            bytes: 解密后的文件内容
            
        Raises:
            DecryptionError: 解密失败
        """
        try:
            if not encrypted_content:
                return b""
            
            # 使用Fernet解密
            content = self._fernet_key.decrypt(encrypted_content)
            
            logger.debug("文件内容解密成功")
            return content
            
        except Exception as e:
            logger.error(f"文件内容解密失败: {e}")
            raise DecryptionError(f"文件内容解密失败: {str(e)}")
    
    def rotate_key(self, new_key: str) -> None:
        """
        轮换加密密钥
        
        Args:
            new_key: 新的加密密钥
            
        Raises:
            EncryptionError: 密钥轮换失败
        """
        try:
            old_fernet = self._fernet_key
            
            # 设置新密钥
            self.master_key = new_key.encode('utf-8')
            self._initialize_encryption()
            
            logger.info("加密密钥轮换成功")
            
        except Exception as e:
            # 恢复原密钥
            self._fernet_key = old_fernet
            logger.error(f"密钥轮换失败: {e}")
            raise EncryptionError(f"密钥轮换失败: {str(e)}")
    
    def is_encrypted(self, data: str) -> bool:
        """
        检测数据是否已加密
        
        Args:
            data: 待检测的数据
            
        Returns:
            bool: 数据是否已加密
        """
        try:
            if not data:
                return False
            
            # 尝试解密，如果成功则说明已加密
            self.decrypt_string(data)
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def generate_encryption_key(length: int = 32) -> str:
        """
        生成新的加密密钥
        
        Args:
            length: 密钥长度(字节)
            
        Returns:
            str: Base64编码的加密密钥
        """
        key_bytes = os.urandom(length)
        key_str = base64.urlsafe_b64encode(key_bytes).decode('utf-8')
        
        logger.info(f"生成 {length} 字节加密密钥")
        return key_str
    
    def get_key_info(self) -> Dict[str, Any]:
        """
        获取密钥信息
        
        Returns:
            Dict: 密钥信息
        """
        return {
            'algorithm': 'AES-256',
            'mode': 'Fernet',
            'key_length': len(self.master_key),
            'initialized': self._fernet_key is not None
        }