"""
SS-UNIT-001: 内容CRUD操作逻辑单元测试
优先级: P0 - 数据完整性保障
"""

from datetime import datetime
from typing import Dict, List, Optional
import uuid


class ContentModel:
    """内容模型 - 模拟实现"""
    
    def __init__(self, title: str, content: str, content_type: str = "text"):
        self.id = str(uuid.uuid4())
        self.title = title
        self.content = content
        self.content_type = content_type
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.tags = []
        self.metadata = {}
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


class ContentCRUDService:
    """内容CRUD服务 - 模拟实现"""
    
    def __init__(self):
        self._storage: Dict[str, ContentModel] = {}
    
    def create(self, title: str, content: str, content_type: str = "text", 
               tags: List[str] = None, metadata: Dict = None) -> Dict:
        """创建内容"""
        if not title or not content:
            return {"success": False, "error": "标题和内容不能为空"}
        
        content_obj = ContentModel(title, content, content_type)
        
        if tags:
            content_obj.tags = tags
        if metadata:
            content_obj.metadata = metadata
        
        self._storage[content_obj.id] = content_obj
        
        return {
            "success": True,
            "data": content_obj.to_dict(),
            "message": "内容创建成功"
        }
    
    def read(self, content_id: str) -> Dict:
        """读取内容"""
        if not content_id:
            return {"success": False, "error": "内容ID不能为空"}
        
        content_obj = self._storage.get(content_id)
        if not content_obj:
            return {"success": False, "error": "内容不存在"}
        
        return {
            "success": True,
            "data": content_obj.to_dict(),
            "message": "内容获取成功"
        }
    
    def update(self, content_id: str, title: str = None, content: str = None,
               tags: List[str] = None, metadata: Dict = None) -> Dict:
        """更新内容"""
        if not content_id:
            return {"success": False, "error": "内容ID不能为空"}
        
        content_obj = self._storage.get(content_id)
        if not content_obj:
            return {"success": False, "error": "内容不存在"}
        
        if title:
            content_obj.title = title
        if content:
            content_obj.content = content
        if tags is not None:
            content_obj.tags = tags
        if metadata:
            content_obj.metadata.update(metadata)
        
        content_obj.updated_at = datetime.now()
        
        return {
            "success": True,
            "data": content_obj.to_dict(),
            "message": "内容更新成功"
        }
    
    def delete(self, content_id: str) -> Dict:
        """删除内容"""
        if not content_id:
            return {"success": False, "error": "内容ID不能为空"}
        
        if content_id not in self._storage:
            return {"success": False, "error": "内容不存在"}
        
        del self._storage[content_id]
        
        return {
            "success": True,
            "message": "内容删除成功"
        }
    
    def list_all(self, limit: int = 10, offset: int = 0) -> Dict:
        """列出所有内容"""
        all_content = list(self._storage.values())
        total = len(all_content)
        
        # 简单分页
        paginated = all_content[offset:offset + limit]
        
        return {
            "success": True,
            "data": {
                "items": [content.to_dict() for content in paginated],
                "total": total,
                "limit": limit,
                "offset": offset
            },
            "message": "内容列表获取成功"
        }


class TestContentCRUD:
    """内容CRUD测试套件"""
    
    def setup_method(self):
        """测试前置设置"""
        self.service = ContentCRUDService()
    
    def test_create_content_success(self):
        """测试成功创建内容
        
        测试场景: SS-UNIT-001-001
        验证点: 基础内容创建功能
        """
        result = self.service.create(
            title="测试文档",
            content="这是一个测试文档的内容",
            content_type="text",
            tags=["测试", "文档"],
            metadata={"source": "unit_test"}
        )
        
        assert result["success"] is True
        assert result["data"]["title"] == "测试文档"
        assert result["data"]["content"] == "这是一个测试文档的内容"
        assert "测试" in result["data"]["tags"]
        assert result["data"]["metadata"]["source"] == "unit_test"
        assert result["data"]["id"] is not None
        
        print("✅ SS-UNIT-001-001: 内容创建测试通过")
        return result["data"]["id"]
    
    def test_create_content_validation(self):
        """测试内容创建验证
        
        测试场景: SS-UNIT-001-002
        验证点: 输入验证和错误处理
        """
        # 测试空标题
        result = self.service.create("", "内容")
        assert result["success"] is False
        assert "标题" in result["error"]
        
        # 测试空内容
        result = self.service.create("标题", "")
        assert result["success"] is False
        assert "内容" in result["error"]
        
        print("✅ SS-UNIT-001-002: 内容创建验证测试通过")
    
    def test_read_content_success(self):
        """测试成功读取内容
        
        测试场景: SS-UNIT-001-003
        验证点: 内容检索功能
        """
        # 先创建内容
        create_result = self.service.create("读取测试", "读取测试内容")
        content_id = create_result["data"]["id"]
        
        # 读取内容
        result = self.service.read(content_id)
        
        assert result["success"] is True
        assert result["data"]["id"] == content_id
        assert result["data"]["title"] == "读取测试"
        assert result["data"]["content"] == "读取测试内容"
        
        print("✅ SS-UNIT-001-003: 内容读取测试通过")
    
    def test_read_content_not_found(self):
        """测试读取不存在的内容
        
        测试场景: SS-UNIT-001-004  
        验证点: 错误处理和异常情况
        """
        # 使用不存在的ID
        result = self.service.read("nonexistent-id")
        
        assert result["success"] is False
        assert "不存在" in result["error"]
        
        # 测试空ID
        result = self.service.read("")
        assert result["success"] is False
        assert "不能为空" in result["error"]
        
        print("✅ SS-UNIT-001-004: 内容不存在处理测试通过")
    
    def test_update_content_success(self):
        """测试成功更新内容
        
        测试场景: SS-UNIT-001-005
        验证点: 内容更新功能
        """
        # 先创建内容
        create_result = self.service.create("原标题", "原内容")
        content_id = create_result["data"]["id"]
        original_updated_at = create_result["data"]["updated_at"]
        
        # 更新内容
        result = self.service.update(
            content_id,
            title="新标题",
            content="新内容",
            tags=["更新", "测试"],
            metadata={"updated_by": "test"}
        )
        
        assert result["success"] is True
        assert result["data"]["title"] == "新标题"
        assert result["data"]["content"] == "新内容"
        assert "更新" in result["data"]["tags"]
        assert result["data"]["metadata"]["updated_by"] == "test"
        assert result["data"]["updated_at"] != original_updated_at
        
        print("✅ SS-UNIT-001-005: 内容更新测试通过")
    
    def test_delete_content_success(self):
        """测试成功删除内容
        
        测试场景: SS-UNIT-001-006
        验证点: 内容删除功能
        """
        # 先创建内容
        create_result = self.service.create("删除测试", "待删除内容")
        content_id = create_result["data"]["id"]
        
        # 确认内容存在
        read_result = self.service.read(content_id)
        assert read_result["success"] is True
        
        # 删除内容
        delete_result = self.service.delete(content_id)
        assert delete_result["success"] is True
        
        # 确认内容已删除
        read_after_delete = self.service.read(content_id)
        assert read_after_delete["success"] is False
        assert "不存在" in read_after_delete["error"]
        
        print("✅ SS-UNIT-001-006: 内容删除测试通过")
    
    def test_list_content_pagination(self):
        """测试内容列表和分页
        
        测试场景: SS-UNIT-001-007
        验证点: 列表查询和分页功能
        """
        # 创建多个内容
        content_ids = []
        for i in range(15):
            result = self.service.create(f"内容{i}", f"内容{i}的详细信息")
            content_ids.append(result["data"]["id"])
        
        # 测试默认分页
        result = self.service.list_all()
        assert result["success"] is True
        assert len(result["data"]["items"]) == 10  # 默认limit
        assert result["data"]["total"] == 15
        
        # 测试自定义分页
        result = self.service.list_all(limit=5, offset=10)
        assert len(result["data"]["items"]) == 5
        assert result["data"]["offset"] == 10
        
        print("✅ SS-UNIT-001-007: 内容列表分页测试通过")


if __name__ == "__main__":
    # 直接运行测试
    test_crud = TestContentCRUD()
    
    print("📊 开始执行内容CRUD单元测试...")
    test_crud.setup_method()
    test_crud.test_create_content_success()
    
    test_crud.setup_method()
    test_crud.test_create_content_validation()
    
    test_crud.setup_method()
    test_crud.test_read_content_success()
    
    test_crud.setup_method()
    test_crud.test_read_content_not_found()
    
    test_crud.setup_method()
    test_crud.test_update_content_success()
    
    test_crud.setup_method()
    test_crud.test_delete_content_success()
    
    test_crud.setup_method()
    test_crud.test_list_content_pagination()
    
    print("✅ 内容CRUD单元测试全部通过！")