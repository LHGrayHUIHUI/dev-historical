"""
病毒扫描服务

提供文件病毒检测功能，集成ClamAV
"""

import logging
from typing import Dict, Any, Optional

from fastapi import UploadFile

logger = logging.getLogger(__name__)


class ScanResult:
    """病毒扫描结果"""
    
    def __init__(self, is_clean: bool, threat_name: Optional[str] = None):
        self.is_clean = is_clean
        self.threat_name = threat_name
    
    def __repr__(self) -> str:
        if self.is_clean:
            return "ScanResult(clean)"
        return f"ScanResult(threat={self.threat_name})"


class VirusScanner:
    """病毒扫描服务
    
    集成ClamAV进行文件病毒检测
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化病毒扫描器
        
        Args:
            config: 配置字典，包含ClamAV连接信息
        """
        self.enabled = config.get('enabled', True)
        self.clamav_host = config.get('host', 'localhost')
        self.clamav_port = config.get('port', 3310)
        self.timeout = config.get('timeout', 30)
        
        logger.info(
            f"病毒扫描器初始化完成",
            extra={
                "enabled": self.enabled,
                "host": self.clamav_host,
                "port": self.clamav_port
            }
        )
    
    async def scan_file(self, file: UploadFile) -> ScanResult:
        """扫描文件是否包含病毒
        
        Args:
            file: 要扫描的文件
            
        Returns:
            ScanResult: 扫描结果
        """
        if not self.enabled:
            logger.debug("病毒扫描已禁用，跳过扫描")
            return ScanResult(is_clean=True, threat_name=None)
        
        try:
            # 尝试导入pyclamd
            import pyclamd
            
            # 创建ClamAV客户端连接
            cd = pyclamd.ClamdNetworkSocket(self.clamav_host, self.clamav_port)
            
            # 测试连接
            if not cd.ping():
                logger.warning("无法连接到ClamAV服务器，跳过病毒扫描")
                return ScanResult(is_clean=True, threat_name=None)
            
            # 重置文件指针
            await file.seek(0)
            file_content = await file.read()
            await file.seek(0)  # 重置指针供后续使用
            
            # 扫描文件内容
            scan_result = cd.scan_stream(file_content)
            
            if scan_result is None:
                # 文件干净
                logger.debug(f"文件 {file.filename} 病毒扫描通过")
                return ScanResult(is_clean=True, threat_name=None)
            else:
                # 发现威胁
                threat_name = scan_result.get('stream', ['Unknown'])[1] if isinstance(scan_result.get('stream'), list) and len(scan_result.get('stream', [])) > 1 else 'Unknown'
                logger.warning(f"文件 {file.filename} 发现威胁: {threat_name}")
                return ScanResult(is_clean=False, threat_name=threat_name)
                
        except ImportError:
            logger.warning("pyclamd 模块未安装，跳过病毒扫描")
            return ScanResult(is_clean=True, threat_name=None)
        except Exception as e:
            logger.warning(f"病毒扫描失败: {str(e)}，允许文件通过")
            # 扫描失败时允许文件通过，但记录警告
            return ScanResult(is_clean=True, threat_name=None)
    
    async def scan_file_path(self, file_path: str) -> ScanResult:
        """扫描本地文件路径
        
        Args:
            file_path: 本地文件路径
            
        Returns:
            ScanResult: 扫描结果
        """
        if not self.enabled:
            logger.debug("病毒扫描已禁用，跳过扫描")
            return ScanResult(is_clean=True, threat_name=None)
        
        try:
            import pyclamd
            
            # 创建ClamAV客户端连接
            cd = pyclamd.ClamdNetworkSocket(self.clamav_host, self.clamav_port)
            
            # 测试连接
            if not cd.ping():
                logger.warning("无法连接到ClamAV服务器，跳过病毒扫描")
                return ScanResult(is_clean=True, threat_name=None)
            
            # 扫描文件
            scan_result = cd.scan_file(file_path)
            
            if scan_result is None:
                # 文件干净
                logger.debug(f"文件 {file_path} 病毒扫描通过")
                return ScanResult(is_clean=True, threat_name=None)
            else:
                # 发现威胁
                threat_name = scan_result.get(file_path, ['Unknown'])[1] if isinstance(scan_result.get(file_path), list) and len(scan_result.get(file_path, [])) > 1 else 'Unknown'
                logger.warning(f"文件 {file_path} 发现威胁: {threat_name}")
                return ScanResult(is_clean=False, threat_name=threat_name)
                
        except ImportError:
            logger.warning("pyclamd 模块未安装，跳过病毒扫描")
            return ScanResult(is_clean=True, threat_name=None)
        except Exception as e:
            logger.warning(f"病毒扫描失败: {str(e)}，允许文件通过")
            return ScanResult(is_clean=True, threat_name=None)
    
    def get_version(self) -> Optional[str]:
        """获取ClamAV版本信息
        
        Returns:
            版本字符串，失败返回None
        """
        if not self.enabled:
            return None
            
        try:
            import pyclamd
            cd = pyclamd.ClamdNetworkSocket(self.clamav_host, self.clamav_port)
            return cd.version()
        except Exception as e:
            logger.error(f"获取ClamAV版本失败: {e}")
            return None
    
    def reload_database(self) -> bool:
        """重新加载病毒数据库
        
        Returns:
            是否重新加载成功
        """
        if not self.enabled:
            return False
            
        try:
            import pyclamd
            cd = pyclamd.ClamdNetworkSocket(self.clamav_host, self.clamav_port)
            cd.reload()
            logger.info("ClamAV病毒数据库重新加载成功")
            return True
        except Exception as e:
            logger.error(f"重新加载ClamAV数据库失败: {e}")
            return False