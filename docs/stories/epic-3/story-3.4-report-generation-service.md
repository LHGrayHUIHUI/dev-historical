# Story 3.4: 报告生成服务

## 用户故事描述

**作为** 历史文献研究人员和学者  
**我希望** 能够自动生成专业的研究报告和分析文档  
**以便于** 快速整理研究成果，生成标准化的学术报告，提高研究工作的效率和质量

## 核心技术栈

### 后端技术
- **Web框架**: FastAPI (Python)
- **报告引擎**: ReportLab, WeasyPrint, Jinja2
- **文档处理**: python-docx, openpyxl, PyPDF2
- **图表生成**: Matplotlib, Plotly, Seaborn
- **模板引擎**: Jinja2, Mako
- **数据处理**: Pandas, NumPy
- **任务队列**: Celery, RQ
- **文件存储**: MinIO, AWS S3
- **数据库**: PostgreSQL, MongoDB
- **缓存**: Redis
- **消息队列**: RabbitMQ

### 前端技术
- **框架**: Vue 3 + TypeScript
- **富文本编辑**: Quill.js, TinyMCE
- **PDF预览**: PDF.js, vue-pdf
- **文件上传**: vue-upload-component
- **UI组件**: Element Plus
- **状态管理**: Pinia
- **图表库**: ECharts, Chart.js

## 数据模型设计

### 报告模板模型
```python
class ReportTemplate(BaseModel):
    """报告模板模型"""
    id: str = Field(default_factory=lambda: f"template_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="模板名称")
    description: Optional[str] = Field(None, description="模板描述")
    category: str = Field(..., description="模板分类")
    template_type: str = Field(..., description="模板类型")
    template_content: Dict[str, Any] = Field(..., description="模板内容")
    variables: List[Dict[str, Any]] = Field(default_factory=list, description="模板变量")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="报告章节")
    styles: Dict[str, Any] = Field(default_factory=dict, description="样式配置")
    output_formats: List[str] = Field(default_factory=list, description="支持的输出格式")
    is_public: bool = Field(default=False, description="是否公开")
    created_by: str = Field(..., description="创建者ID")
    created_time: datetime = Field(default_factory=datetime.utcnow)
    updated_time: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0", description="模板版本")
```

### 报告生成任务模型
```python
class ReportTask(BaseModel):
    """报告生成任务模型"""
    id: str = Field(default_factory=lambda: f"report_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="报告名称")
    template_id: str = Field(..., description="使用的模板ID")
    user_id: str = Field(..., description="创建用户ID")
    data_sources: List[str] = Field(default_factory=list, description="数据源列表")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="生成参数")
    variables: Dict[str, Any] = Field(default_factory=dict, description="模板变量值")
    output_format: str = Field(default="pdf", description="输出格式")
    status: str = Field(default="pending", description="任务状态")
    progress: float = Field(default=0.0, description="生成进度")
    file_path: Optional[str] = Field(None, description="生成文件路径")
    file_size: Optional[int] = Field(None, description="文件大小")
    error_message: Optional[str] = Field(None, description="错误信息")
    created_time: datetime = Field(default_factory=datetime.utcnow)
    started_time: Optional[datetime] = Field(None, description="开始时间")
    completed_time: Optional[datetime] = Field(None, description="完成时间")
    expires_at: Optional[datetime] = Field(None, description="文件过期时间")
```

### 报告章节模型
```python
class ReportSection(BaseModel):
    """报告章节模型"""
    id: str = Field(default_factory=lambda: f"section_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="章节名称")
    section_type: str = Field(..., description="章节类型")
    order: int = Field(..., description="章节顺序")
    content_template: str = Field(..., description="内容模板")
    data_query: Optional[Dict[str, Any]] = Field(None, description="数据查询配置")
    chart_config: Optional[Dict[str, Any]] = Field(None, description="图表配置")
    style_config: Dict[str, Any] = Field(default_factory=dict, description="样式配置")
    is_required: bool = Field(default=True, description="是否必需")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="显示条件")
```

### 报告数据源模型
```python
class ReportDataSource(BaseModel):
    """报告数据源模型"""
    id: str = Field(default_factory=lambda: f"datasource_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="数据源名称")
    source_type: str = Field(..., description="数据源类型")
    connection_config: Dict[str, Any] = Field(..., description="连接配置")
    query_template: str = Field(..., description="查询模板")
    cache_duration: Optional[int] = Field(None, description="缓存时长(秒)")
    is_active: bool = Field(default=True, description="是否激活")
    created_time: datetime = Field(default_factory=datetime.utcnow)
    updated_time: datetime = Field(default_factory=datetime.utcnow)
```

## 服务架构设计

### 报告生成服务核心类
```python
class ReportGenerationService:
    """报告生成服务"""
    
    def __init__(self):
        self.db = get_database()
        self.redis = get_redis_client()
        self.file_storage = get_file_storage()
        self.template_engine = TemplateEngine()
        self.data_processor = ReportDataProcessor()
        self.chart_generator = ChartGenerator()
        self.document_generator = DocumentGenerator()
    
    async def create_template(self, template_data: dict) -> str:
        """创建报告模板"""
        template = ReportTemplate(**template_data)
        
        # 验证模板内容
        await self._validate_template(template)
        
        # 保存模板
        await self.db.report_templates.insert_one(template.dict())
        
        return template.id
    
    async def generate_report(self, task_data: dict) -> str:
        """生成报告"""
        task = ReportTask(**task_data)
        
        # 保存任务信息
        await self.db.report_tasks.insert_one(task.dict())
        
        # 异步生成报告
        await self._generate_report_async(task.id)
        
        return task.id
    
    async def get_report_status(self, task_id: str) -> dict:
        """获取报告生成状态"""
        task = await self.db.report_tasks.find_one({"id": task_id})
        if not task:
            raise ValueError(f"Report task {task_id} not found")
        
        return {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "file_path": task.get("file_path"),
            "error_message": task.get("error_message"),
            "created_time": task["created_time"],
            "completed_time": task.get("completed_time")
        }
    
    async def download_report(self, task_id: str) -> bytes:
        """下载报告文件"""
        task = await self.db.report_tasks.find_one({"id": task_id})
        if not task or task["status"] != "completed":
            raise ValueError(f"Report {task_id} not ready for download")
        
        file_path = task["file_path"]
        if not file_path:
            raise ValueError(f"Report file not found for task {task_id}")
        
        # 从文件存储获取文件
        file_content = await self.file_storage.get_file(file_path)
        
        return file_content
    
    async def preview_report(self, template_id: str, variables: dict) -> dict:
        """预览报告"""
        template = await self.db.report_templates.find_one({"id": template_id})
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # 生成预览内容
        preview_content = await self._generate_preview_content(
            template, variables
        )
        
        return {
            "template_id": template_id,
            "preview_content": preview_content,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def _generate_report_async(self, task_id: str):
        """异步生成报告"""
        try {
            # 更新任务状态
            await self._update_task_status(task_id, "processing", 0)
            
            # 获取任务信息
            task = await self.db.report_tasks.find_one({"id": task_id})
            template = await self.db.report_templates.find_one(
                {"id": task["template_id"]}
            )
            
            # 收集数据
            await self._update_task_status(task_id, "processing", 20)
            report_data = await self._collect_report_data(
                task["data_sources"], 
                task["parameters"]
            )
            
            # 渲染内容
            await self._update_task_status(task_id, "processing", 50)
            rendered_content = await self._render_report_content(
                template, 
                task["variables"], 
                report_data
            )
            
            # 生成文档
            await self._update_task_status(task_id, "processing", 80)
            file_path = await self._generate_document(
                task_id,
                rendered_content,
                task["output_format"],
                template["styles"]
            )
            
            # 完成任务
            await self._update_task_status(
                task_id, 
                "completed", 
                100, 
                file_path=file_path
            )
            
        } catch Exception as e:
            logger.error(f"Report generation failed for task {task_id}: {str(e)}")
            await self._update_task_status(
                task_id, 
                "failed", 
                error_message=str(e)
            )
    
    async def _collect_report_data(self, data_sources: List[str], 
                                 parameters: dict) -> dict:
        """收集报告数据"""
        report_data = {}
        
        for source_id in data_sources:
            source_config = await self.db.report_data_sources.find_one(
                {"id": source_id}
            )
            
            if source_config:
                data = await self.data_processor.get_data(
                    source_config["source_type"],
                    source_config["connection_config"],
                    source_config["query_template"],
                    parameters
                )
                
                report_data[source_id] = data
        
        return report_data
    
    async def _render_report_content(self, template: dict, variables: dict, 
                                   data: dict) -> dict:
        """渲染报告内容"""
        rendered_sections = []
        
        for section_config in template["sections"]:
            # 检查显示条件
            if not self._check_section_conditions(
                section_config.get("conditions", []), 
                variables, 
                data
            ):
                continue
            
            # 渲染章节内容
            section_content = await self._render_section(
                section_config, 
                variables, 
                data
            )
            
            rendered_sections.append({
                "id": section_config["id"],
                "name": section_config["name"],
                "type": section_config["section_type"],
                "order": section_config["order"],
                "content": section_content
            })
        
        # 按顺序排序
        rendered_sections.sort(key=lambda x: x["order"])
        
        return {
            "title": self.template_engine.render(
                template["template_content"].get("title", ""), 
                variables
            ),
            "sections": rendered_sections,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "template_id": template["id"],
                "template_version": template["version"]
            }
        }
    
    async def _render_section(self, section_config: dict, variables: dict, 
                            data: dict) -> dict:
        """渲染章节内容"""
        section_type = section_config["section_type"]
        
        if section_type == "text":
            return await self._render_text_section(
                section_config, variables, data
            )
        elif section_type == "table":
            return await self._render_table_section(
                section_config, variables, data
            )
        elif section_type == "chart":
            return await self._render_chart_section(
                section_config, variables, data
            )
        elif section_type == "image":
            return await self._render_image_section(
                section_config, variables, data
            )
        else:
            raise ValueError(f"Unsupported section type: {section_type}")
    
    async def _render_text_section(self, section_config: dict, 
                                 variables: dict, data: dict) -> dict:
        """渲染文本章节"""
        content_template = section_config["content_template"]
        
        # 准备模板上下文
        context = {
            **variables,
            "data": data,
            "section": section_config
        }
        
        # 渲染内容
        rendered_content = self.template_engine.render(
            content_template, context
        )
        
        return {
            "type": "text",
            "content": rendered_content,
            "style": section_config.get("style_config", {})
        }
    
    async def _render_table_section(self, section_config: dict, 
                                  variables: dict, data: dict) -> dict:
        """渲染表格章节"""
        data_query = section_config.get("data_query")
        if not data_query:
            raise ValueError("Table section requires data_query configuration")
        
        # 获取表格数据
        table_data = await self.data_processor.query_data(
            data, data_query
        )
        
        # 处理表格格式
        table_config = section_config.get("table_config", {})
        formatted_table = self._format_table_data(
            table_data, table_config
        )
        
        return {
            "type": "table",
            "data": formatted_table,
            "style": section_config.get("style_config", {})
        }
    
    async def _render_chart_section(self, section_config: dict, 
                                  variables: dict, data: dict) -> dict:
        """渲染图表章节"""
        chart_config = section_config.get("chart_config")
        if not chart_config:
            raise ValueError("Chart section requires chart_config")
        
        # 生成图表
        chart_data = await self.chart_generator.generate_chart(
            chart_config["chart_type"],
            data,
            chart_config.get("data_config", {}),
            chart_config.get("options", {})
        )
        
        # 保存图表图片
        chart_image_path = await self._save_chart_image(
            chart_data, section_config["id"]
        )
        
        return {
            "type": "chart",
            "image_path": chart_image_path,
            "chart_data": chart_data,
            "style": section_config.get("style_config", {})
        }
    
    async def _generate_document(self, task_id: str, content: dict, 
                               output_format: str, styles: dict) -> str:
        """生成文档文件"""
        if output_format == "pdf":
            return await self.document_generator.generate_pdf(
                task_id, content, styles
            )
        elif output_format == "docx":
            return await self.document_generator.generate_docx(
                task_id, content, styles
            )
        elif output_format == "html":
            return await self.document_generator.generate_html(
                task_id, content, styles
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    async def _update_task_status(self, task_id: str, status: str, 
                                progress: float = None, 
                                file_path: str = None,
                                error_message: str = None):
        """更新任务状态"""
        update_data = {
            "status": status,
            "updated_time": datetime.utcnow()
        }
        
        if progress is not None:
            update_data["progress"] = progress
        
        if file_path:
            update_data["file_path"] = file_path
            update_data["completed_time"] = datetime.utcnow()
        
        if error_message:
            update_data["error_message"] = error_message
        
        if status == "processing" and progress == 0:
            update_data["started_time"] = datetime.utcnow()
        
        await self.db.report_tasks.update_one(
            {"id": task_id},
            {"$set": update_data}
        )
    
    def _check_section_conditions(self, conditions: List[dict], 
                                variables: dict, data: dict) -> bool:
        """检查章节显示条件"""
        if not conditions:
            return True
        
        for condition in conditions:
            condition_type = condition.get("type")
            
            if condition_type == "variable":
                var_name = condition["variable"]
                operator = condition["operator"]
                expected_value = condition["value"]
                actual_value = variables.get(var_name)
                
                if not self._evaluate_condition(
                    actual_value, operator, expected_value
                ):
                    return False
            
            elif condition_type == "data":
                data_path = condition["data_path"]
                operator = condition["operator"]
                expected_value = condition["value"]
                
                actual_value = self._get_nested_value(data, data_path)
                
                if not self._evaluate_condition(
                    actual_value, operator, expected_value
                ):
                    return False
        
        return True
    
    def _evaluate_condition(self, actual_value, operator: str, 
                          expected_value) -> bool:
        """评估条件"""
        if operator == "eq":
            return actual_value == expected_value
        elif operator == "ne":
            return actual_value != expected_value
        elif operator == "gt":
            return actual_value > expected_value
        elif operator == "gte":
            return actual_value >= expected_value
        elif operator == "lt":
            return actual_value < expected_value
        elif operator == "lte":
            return actual_value <= expected_value
        elif operator == "in":
            return actual_value in expected_value
        elif operator == "contains":
            return expected_value in str(actual_value)
        elif operator == "exists":
            return actual_value is not None
        else:
            return False
    
    def _get_nested_value(self, data: dict, path: str):
        """获取嵌套值"""
        keys = path.split(".")
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
```

### 模板引擎
```python
class TemplateEngine:
    """模板引擎"""
    
    def __init__(self):
        self.jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # 注册自定义过滤器
        self.jinja_env.filters['format_date'] = self._format_date
        self.jinja_env.filters['format_number'] = self._format_number
        self.jinja_env.filters['format_currency'] = self._format_currency
        self.jinja_env.filters['truncate_text'] = self._truncate_text
    
    def render(self, template_string: str, context: dict) -> str:
        """渲染模板"""
        try:
            template = self.jinja_env.from_string(template_string)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Template rendering failed: {str(e)}")
            raise ValueError(f"Template rendering error: {str(e)}")
    
    def _format_date(self, date_value, format_string='%Y-%m-%d'):
        """格式化日期"""
        if isinstance(date_value, str):
            date_value = datetime.fromisoformat(date_value)
        elif isinstance(date_value, datetime):
            pass
        else:
            return str(date_value)
        
        return date_value.strftime(format_string)
    
    def _format_number(self, number_value, decimal_places=2):
        """格式化数字"""
        try:
            return f"{float(number_value):.{decimal_places}f}"
        except (ValueError, TypeError):
            return str(number_value)
    
    def _format_currency(self, amount, currency='¥'):
        """格式化货币"""
        try:
            return f"{currency}{float(amount):,.2f}"
        except (ValueError, TypeError):
            return str(amount)
    
    def _truncate_text(self, text, max_length=100, suffix='...'):
        """截断文本"""
        if len(str(text)) <= max_length:
            return str(text)
        return str(text)[:max_length - len(suffix)] + suffix
```

### 文档生成器
```python
class DocumentGenerator:
    """文档生成器"""
    
    def __init__(self):
        self.file_storage = get_file_storage()
    
    async def generate_pdf(self, task_id: str, content: dict, 
                         styles: dict) -> str:
        """生成PDF文档"""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        # 创建临时文件
        temp_file = f"/tmp/report_{task_id}.pdf"
        
        # 创建PDF文档
        doc = SimpleDocTemplate(
            temp_file,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # 获取样式
        styles_sheet = getSampleStyleSheet()
        story = []
        
        # 添加标题
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles_sheet['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # 居中
        )
        
        story.append(Paragraph(content["title"], title_style))
        story.append(Spacer(1, 12))
        
        # 添加章节
        for section in content["sections"]:
            await self._add_pdf_section(story, section, styles_sheet)
        
        # 构建PDF
        doc.build(story)
        
        # 上传到文件存储
        file_path = f"reports/{task_id}.pdf"
        with open(temp_file, 'rb') as f:
            await self.file_storage.upload_file(file_path, f.read())
        
        # 删除临时文件
        os.remove(temp_file)
        
        return file_path
    
    async def generate_docx(self, task_id: str, content: dict, 
                          styles: dict) -> str:
        """生成DOCX文档"""
        from docx import Document
        from docx.shared import Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        # 创建文档
        doc = Document()
        
        # 添加标题
        title = doc.add_heading(content["title"], 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # 添加章节
        for section in content["sections"]:
            await self._add_docx_section(doc, section)
        
        # 保存到临时文件
        temp_file = f"/tmp/report_{task_id}.docx"
        doc.save(temp_file)
        
        # 上传到文件存储
        file_path = f"reports/{task_id}.docx"
        with open(temp_file, 'rb') as f:
            await self.file_storage.upload_file(file_path, f.read())
        
        # 删除临时文件
        os.remove(temp_file)
        
        return file_path
    
    async def generate_html(self, task_id: str, content: dict, 
                          styles: dict) -> str:
        """生成HTML文档"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: 'Microsoft YaHei', Arial, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                    color: #333;
                }
                .title {
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 30px;
                    color: #2c3e50;
                }
                .section {
                    margin-bottom: 30px;
                }
                .section-title {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #34495e;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                }
                .section-content {
                    margin-left: 20px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .chart-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .chart-container img {
                    max-width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <div class="title">{{ title }}</div>
            
            {% for section in sections %}
            <div class="section">
                <div class="section-title">{{ section.name }}</div>
                <div class="section-content">
                    {% if section.content.type == 'text' %}
                        {{ section.content.content | safe }}
                    {% elif section.content.type == 'table' %}
                        <table>
                            {% if section.content.data.headers %}
                            <thead>
                                <tr>
                                    {% for header in section.content.data.headers %}
                                    <th>{{ header }}</th>
                                    {% endfor %}
                                </tr>
                            </thead>
                            {% endif %}
                            <tbody>
                                {% for row in section.content.data.rows %}
                                <tr>
                                    {% for cell in row %}
                                    <td>{{ cell }}</td>
                                    {% endfor %}
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    {% elif section.content.type == 'chart' %}
                        <div class="chart-container">
                            <img src="{{ section.content.image_path }}" alt="Chart">
                        </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        # 渲染HTML
        template = Template(html_template)
        html_content = template.render(
            title=content["title"],
            sections=content["sections"]
        )
        
        # 保存HTML文件
        file_path = f"reports/{task_id}.html"
        await self.file_storage.upload_file(
            file_path, 
            html_content.encode('utf-8')
        )
        
        return file_path
    
    async def _add_pdf_section(self, story, section, styles_sheet):
        """添加PDF章节"""
        from reportlab.platypus import Paragraph, Spacer, Table, Image
        from reportlab.lib.styles import ParagraphStyle
        
        # 添加章节标题
        section_title_style = ParagraphStyle(
            'SectionTitle',
            parent=styles_sheet['Heading2'],
            fontSize=14,
            spaceAfter=12
        )
        
        story.append(Paragraph(section["name"], section_title_style))
        
        # 添加章节内容
        content = section["content"]
        
        if content["type"] == "text":
            story.append(Paragraph(content["content"], styles_sheet['Normal']))
        
        elif content["type"] == "table":
            table_data = content["data"]
            if "headers" in table_data:
                data = [table_data["headers"]] + table_data["rows"]
            else:
                data = table_data["rows"]
            
            table = Table(data)
            table.setStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            
            story.append(table)
        
        elif content["type"] == "chart":
            if content.get("image_path"):
                # 这里需要处理图片路径
                image_path = content["image_path"]
                story.append(Image(image_path, width=6*inch, height=4*inch))
        
        story.append(Spacer(1, 12))
    
    async def _add_docx_section(self, doc, section):
        """添加DOCX章节"""
        from docx.shared import Inches
        
        # 添加章节标题
        doc.add_heading(section["name"], level=2)
        
        # 添加章节内容
        content = section["content"]
        
        if content["type"] == "text":
            doc.add_paragraph(content["content"])
        
        elif content["type"] == "table":
            table_data = content["data"]
            
            if "headers" in table_data and "rows" in table_data:
                table = doc.add_table(
                    rows=len(table_data["rows"]) + 1, 
                    cols=len(table_data["headers"])
                )
                
                # 添加表头
                hdr_cells = table.rows[0].cells
                for i, header in enumerate(table_data["headers"]):
                    hdr_cells[i].text = str(header)
                
                # 添加数据行
                for i, row in enumerate(table_data["rows"]):
                    row_cells = table.rows[i + 1].cells
                    for j, cell_value in enumerate(row):
                        row_cells[j].text = str(cell_value)
        
        elif content["type"] == "chart":
            if content.get("image_path"):
                # 添加图片
                doc.add_picture(content["image_path"], width=Inches(6))
```

## API设计

### 报告模板API

#### 1. 创建报告模板
```http
POST /api/v1/reports/templates
```

**请求体:**
```json
{
  "name": "历史文献分析报告模板",
  "description": "用于生成历史文献研究分析报告",
  "category": "research",
  "template_type": "analysis",
  "template_content": {
    "title": "{{ project_name }}研究分析报告",
    "subtitle": "基于{{ data_period }}的数据分析"
  },
  "variables": [
    {
      "name": "project_name",
      "type": "string",
      "label": "项目名称",
      "required": true,
      "default_value": ""
    },
    {
      "name": "data_period",
      "type": "string",
      "label": "数据时期",
      "required": true,
      "default_value": ""
    }
  ],
  "sections": [
    {
      "name": "项目概述",
      "section_type": "text",
      "order": 1,
      "content_template": "本报告分析了{{ project_name }}项目在{{ data_period }}期间的研究成果...",
      "is_required": true
    },
    {
      "name": "数据统计",
      "section_type": "table",
      "order": 2,
      "data_query": {
        "source": "documents",
        "fields": ["category", "count"],
        "group_by": "category"
      },
      "is_required": true
    }
  ],
  "styles": {
    "font_family": "Microsoft YaHei",
    "font_size": 12,
    "line_height": 1.5,
    "margin": {
      "top": 72,
      "bottom": 72,
      "left": 72,
      "right": 72
    }
  },
  "output_formats": ["pdf", "docx", "html"]
}
```

#### 2. 生成报告
```http
POST /api/v1/reports/generate
```

**请求体:**
```json
{
  "name": "明清历史文献分析报告",
  "template_id": "template_abc123",
  "data_sources": ["datasource_1", "datasource_2"],
  "variables": {
    "project_name": "明清历史文献研究",
    "data_period": "1368-1644年",
    "researcher_name": "张三",
    "report_date": "2024-01-20"
  },
  "parameters": {
    "date_range": {
      "start": "1368-01-01",
      "end": "1644-12-31"
    },
    "categories": ["政治", "经济", "文化"],
    "include_charts": true
  },
  "output_format": "pdf"
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "report_xyz789",
    "status": "pending",
    "estimated_time": 120
  }
}
```

#### 3. 获取报告状态
```http
GET /api/v1/reports/tasks/{task_id}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "task_id": "report_xyz789",
    "status": "completed",
    "progress": 100,
    "file_path": "reports/report_xyz789.pdf",
    "file_size": 2048576,
    "created_time": "2024-01-20T10:00:00Z",
    "completed_time": "2024-01-20T10:02:30Z"
  }
}
```

#### 4. 下载报告
```http
GET /api/v1/reports/download/{task_id}
```

**响应**: 文件流（PDF/DOCX/HTML）

#### 5. 预览报告
```http
POST /api/v1/reports/preview
```

**请求体:**
```json
{
  "template_id": "template_abc123",
  "variables": {
    "project_name": "测试项目",
    "data_period": "2024年"
  },
  "sample_data": {
    "documents": [
      {"category": "政治", "count": 150},
      {"category": "经济", "count": 120}
    ]
  }
}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "template_id": "template_abc123",
    "preview_content": {
      "title": "测试项目研究分析报告",
      "sections": [
        {
          "name": "项目概述",
          "type": "text",
          "content": "本报告分析了测试项目在2024年期间的研究成果..."
        }
      ]
    },
    "generated_at": "2024-01-20T15:30:00Z"
  }
}
```

#### 6. 获取模板列表
```http
GET /api/v1/reports/templates?category={category}&page={page}&size={size}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "templates": [
      {
        "id": "template_abc123",
        "name": "历史文献分析报告模板",
        "description": "用于生成历史文献研究分析报告",
        "category": "research",
        "template_type": "analysis",
        "created_time": "2024-01-15T09:00:00Z",
        "is_public": true
      }
    ],
    "total": 1,
    "page": 1,
    "size": 10
  }
}
```

#### 7. 获取报告历史
```http
GET /api/v1/reports/history?user_id={user_id}&page={page}&size={size}
```

**响应示例:**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "reports": [
      {
        "task_id": "report_xyz789",
        "name": "明清历史文献分析报告",
        "template_name": "历史文献分析报告模板",
        "status": "completed",
        "output_format": "pdf",
        "file_size": 2048576,
        "created_time": "2024-01-20T10:00:00Z",
        "completed_time": "2024-01-20T10:02:30Z"
      }
    ],
    "total": 1,
    "page": 1,
    "size": 10
  }
}
```

## 前端集成

### Vue3 报告生成组件

```vue
<template>
  <div class="report-generator">
    <!-- 报告生成向导 -->
    <el-card class="wizard-card">
      <template #header>
        <div class="card-header">
          <span>报告生成向导</span>
          <el-button 
            type="primary" 
            @click="showTemplateDialog = true"
            :disabled="generating"
          >
            选择模板
          </el-button>
        </div>
      </template>
      
      <!-- 步骤指示器 -->
      <el-steps :active="currentStep" finish-status="success">
        <el-step title="选择模板" />
        <el-step title="配置参数" />
        <el-step title="预览报告" />
        <el-step title="生成报告" />
      </el-steps>
      
      <!-- 步骤内容 -->
      <div class="step-content">
        <!-- 步骤1: 模板选择 -->
        <div v-if="currentStep === 0" class="template-selection">
          <div v-if="selectedTemplate" class="selected-template">
            <el-card>
              <h3>{{ selectedTemplate.name }}</h3>
              <p>{{ selectedTemplate.description }}</p>
              <el-tag>{{ selectedTemplate.category }}</el-tag>
              <el-button 
                type="text" 
                @click="showTemplateDialog = true"
              >
                更换模板
              </el-button>
            </el-card>
          </div>
          <div v-else class="no-template">
            <el-empty description="请选择报告模板" />
          </div>
        </div>
        
        <!-- 步骤2: 参数配置 -->
        <div v-if="currentStep === 1" class="parameter-config">
          <el-form 
            ref="paramForm" 
            :model="reportConfig" 
            :rules="paramRules"
            label-width="120px"
          >
            <!-- 基本信息 -->
            <el-form-item label="报告名称" prop="name">
              <el-input 
                v-model="reportConfig.name" 
                placeholder="请输入报告名称"
              />
            </el-form-item>
            
            <el-form-item label="输出格式" prop="output_format">
              <el-select v-model="reportConfig.output_format">
                <el-option 
                  v-for="format in outputFormats" 
                  :key="format.value"
                  :label="format.label" 
                  :value="format.value"
                />
              </el-select>
            </el-form-item>
            
            <!-- 模板变量 -->
            <div v-if="templateVariables.length > 0">
              <h4>模板变量</h4>
              <el-form-item 
                v-for="variable in templateVariables" 
                :key="variable.name"
                :label="variable.label"
                :prop="`variables.${variable.name}`"
                :rules="variable.required ? [{ required: true, message: `请输入${variable.label}` }] : []"
              >
                <el-input 
                  v-if="variable.type === 'string'"
                  v-model="reportConfig.variables[variable.name]"
                  :placeholder="variable.default_value"
                />
                <el-input-number 
                  v-else-if="variable.type === 'number'"
                  v-model="reportConfig.variables[variable.name]"
                  :placeholder="variable.default_value"
                />
                <el-date-picker 
                  v-else-if="variable.type === 'date'"
                  v-model="reportConfig.variables[variable.name]"
                  type="date"
                  placeholder="选择日期"
                />
                <el-select 
                  v-else-if="variable.type === 'select'"
                  v-model="reportConfig.variables[variable.name]"
                  placeholder="请选择"
                >
                  <el-option 
                    v-for="option in variable.options" 
                    :key="option.value"
                    :label="option.label" 
                    :value="option.value"
                  />
                </el-select>
              </el-form-item>
            </div>
            
            <!-- 数据源配置 -->
            <div v-if="dataSources.length > 0">
              <h4>数据源</h4>
              <el-form-item label="选择数据源">
                <el-checkbox-group v-model="reportConfig.data_sources">
                  <el-checkbox 
                    v-for="source in dataSources" 
                    :key="source.id"
                    :label="source.id"
                  >
                    {{ source.name }}
                  </el-checkbox>
                </el-checkbox-group>
              </el-form-item>
            </div>
            
            <!-- 高级参数 -->
            <el-collapse>
              <el-collapse-item title="高级参数" name="advanced">
                <el-form-item label="日期范围">
                  <el-date-picker
                    v-model="dateRange"
                    type="daterange"
                    range-separator="至"
                    start-placeholder="开始日期"
                    end-placeholder="结束日期"
                    @change="updateDateRange"
                  />
                </el-form-item>
                
                <el-form-item label="包含图表">
                  <el-switch v-model="reportConfig.parameters.include_charts" />
                </el-form-item>
                
                <el-form-item label="数据过滤">
                  <el-input 
                    v-model="reportConfig.parameters.filter_conditions"
                    type="textarea"
                    placeholder="输入过滤条件（JSON格式）"
                  />
                </el-form-item>
              </el-collapse-item>
            </el-collapse>
          </el-form>
        </div>
        
        <!-- 步骤3: 报告预览 -->
        <div v-if="currentStep === 2" class="report-preview">
          <div class="preview-actions">
            <el-button @click="generatePreview" :loading="previewing">
              生成预览
            </el-button>
            <el-button 
              v-if="previewContent" 
              @click="showPreviewDialog = true"
            >
              查看预览
            </el-button>
          </div>
          
          <div v-if="previewContent" class="preview-summary">
            <el-card>
              <h3>{{ previewContent.title }}</h3>
              <p>章节数量: {{ previewContent.sections.length }}</p>
              <p>预览生成时间: {{ formatDate(previewContent.generated_at) }}</p>
            </el-card>
          </div>
        </div>
        
        <!-- 步骤4: 生成报告 -->
        <div v-if="currentStep === 3" class="report-generation">
          <div v-if="!generating && !generationResult" class="generation-ready">
            <el-card>
              <h3>准备生成报告</h3>
              <p>报告名称: {{ reportConfig.name }}</p>
              <p>输出格式: {{ getFormatLabel(reportConfig.output_format) }}</p>
              <p>预计生成时间: 1-3分钟</p>
            </el-card>
          </div>
          
          <div v-if="generating" class="generation-progress">
            <el-card>
              <h3>正在生成报告...</h3>
              <el-progress 
                :percentage="generationProgress" 
                :status="generationStatus"
              />
              <p>{{ generationMessage }}</p>
            </el-card>
          </div>
          
          <div v-if="generationResult" class="generation-result">
            <el-card>
              <h3>报告生成完成</h3>
              <p>文件大小: {{ formatFileSize(generationResult.file_size) }}</p>
              <p>生成时间: {{ formatDate(generationResult.completed_time) }}</p>
              <div class="result-actions">
                <el-button 
                  type="primary" 
                  @click="downloadReport"
                >
                  下载报告
                </el-button>
                <el-button @click="resetGenerator">
                  生成新报告
                </el-button>
              </div>
            </el-card>
          </div>
        </div>
      </div>
      
      <!-- 操作按钮 -->
      <div class="step-actions">
        <el-button 
          v-if="currentStep > 0" 
          @click="prevStep"
          :disabled="generating"
        >
          上一步
        </el-button>
        <el-button 
          v-if="currentStep < 3" 
          type="primary" 
          @click="nextStep"
          :disabled="!canProceed || generating"
        >
          下一步
        </el-button>
        <el-button 
          v-if="currentStep === 3 && !generating && !generationResult" 
          type="primary" 
          @click="generateReport"
        >
          生成报告
        </el-button>
      </div>
    </el-card>
    
    <!-- 报告历史 -->
    <el-card class="history-card">
      <template #header>
        <div class="card-header">
          <span>报告历史</span>
          <el-button @click="loadReportHistory">
            刷新
          </el-button>
        </div>
      </template>
      
      <el-table :data="reportHistory" v-loading="loadingHistory">
        <el-table-column prop="name" label="报告名称" />
        <el-table-column prop="template_name" label="模板" />
        <el-table-column prop="output_format" label="格式" width="80" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag 
              :type="getStatusType(row.status)"
              size="small"
            >
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_time" label="创建时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.created_time) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button 
              v-if="row.status === 'completed'"
              type="text" 
              size="small"
              @click="downloadHistoryReport(row.task_id)"
            >
              下载
            </el-button>
            <el-button 
              type="text" 
              size="small"
              @click="deleteReport(row.task_id)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <el-pagination
        v-if="historyTotal > 0"
        :current-page="historyPage"
        :page-size="historySize"
        :total="historyTotal"
        @current-change="handleHistoryPageChange"
        layout="total, prev, pager, next"
      />
    </el-card>
    
    <!-- 模板选择对话框 -->
    <el-dialog 
      v-model="showTemplateDialog" 
      title="选择报告模板" 
      width="800px"
    >
      <div class="template-selector">
        <div class="template-filters">
          <el-select 
            v-model="templateFilter.category" 
            placeholder="选择分类"
            clearable
            @change="loadTemplates"
          >
            <el-option 
              v-for="category in templateCategories" 
              :key="category.value"
              :label="category.label" 
              :value="category.value"
            />
          </el-select>
          
          <el-input 
            v-model="templateFilter.keyword"
            placeholder="搜索模板"
            @input="debounceSearch"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </div>
        
        <div class="template-list">
          <el-row :gutter="16">
            <el-col 
              v-for="template in templates" 
              :key="template.id"
              :span="12"
            >
              <el-card 
                class="template-card"
                :class="{ selected: selectedTemplate?.id === template.id }"
                @click="selectTemplate(template)"
              >
                <h4>{{ template.name }}</h4>
                <p>{{ template.description }}</p>
                <div class="template-meta">
                  <el-tag size="small">{{ template.category }}</el-tag>
                  <span class="template-type">{{ template.template_type }}</span>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>
        
        <el-pagination
          v-if="templateTotal > 0"
          :current-page="templatePage"
          :page-size="templateSize"
          :total="templateTotal"
          @current-change="handleTemplatePageChange"
          layout="total, prev, pager, next"
        />
      </div>
      
      <template #footer>
        <el-button @click="showTemplateDialog = false">
          取消
        </el-button>
        <el-button 
          type="primary" 
          @click="confirmTemplate"
          :disabled="!selectedTemplate"
        >
          确定
        </el-button>
      </template>
    </el-dialog>
    
    <!-- 预览对话框 -->
    <el-dialog 
      v-model="showPreviewDialog" 
      title="报告预览" 
      width="900px"
      class="preview-dialog"
    >
      <div v-if="previewContent" class="preview-content">
        <h2>{{ previewContent.title }}</h2>
        
        <div 
          v-for="section in previewContent.sections" 
          :key="section.id"
          class="preview-section"
        >
          <h3>{{ section.name }}</h3>
          
          <div v-if="section.content.type === 'text'" class="text-content">
            <div v-html="section.content.content"></div>
          </div>
          
          <div v-else-if="section.content.type === 'table'" class="table-content">
            <el-table :data="section.content.data.rows" border>
              <el-table-column 
                v-for="(header, index) in section.content.data.headers" 
                :key="index"
                :prop="index.toString()"
                :label="header"
              />
            </el-table>
          </div>
          
          <div v-else-if="section.content.type === 'chart'" class="chart-content">
            <img 
              :src="section.content.image_path" 
              alt="Chart"
              style="max-width: 100%; height: auto;"
            />
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import { debounce } from 'lodash-es'
import { reportApi } from '@/api/report'
import { formatDate, formatFileSize } from '@/utils/format'

// 响应式数据
const currentStep = ref(0)
const generating = ref(false)
const previewing = ref(false)
const loadingHistory = ref(false)
const showTemplateDialog = ref(false)
const showPreviewDialog = ref(false)

// 模板相关
const templates = ref([])
const selectedTemplate = ref(null)
const templateVariables = ref([])
const templateCategories = ref([
  { label: '研究分析', value: 'research' },
  { label: '数据统计', value: 'statistics' },
  { label: '项目总结', value: 'summary' }
])
const templateFilter = reactive({
  category: '',
  keyword: ''
})
const templatePage = ref(1)
const templateSize = ref(10)
const templateTotal = ref(0)

// 报告配置
const reportConfig = reactive({
  name: '',
  template_id: '',
  data_sources: [],
  variables: {},
  parameters: {
    include_charts: true,
    filter_conditions: ''
  },
  output_format: 'pdf'
})

const dateRange = ref([])
const dataSources = ref([])
const outputFormats = ref([
  { label: 'PDF', value: 'pdf' },
  { label: 'Word文档', value: 'docx' },
  { label: 'HTML', value: 'html' }
])

// 表单验证规则
const paramRules = {
  name: [
    { required: true, message: '请输入报告名称', trigger: 'blur' }
  ],
  output_format: [
    { required: true, message: '请选择输出格式', trigger: 'change' }
  ]
}

// 生成相关
const generationProgress = ref(0)
const generationStatus = ref('')
const generationMessage = ref('')
const generationResult = ref(null)
const currentTaskId = ref('')

// 预览相关
const previewContent = ref(null)

// 历史记录
const reportHistory = ref([])
const historyPage = ref(1)
const historySize = ref(10)
const historyTotal = ref(0)

// 计算属性
const canProceed = computed(() => {
  switch (currentStep.value) {
    case 0:
      return selectedTemplate.value !== null
    case 1:
      return reportConfig.name && reportConfig.output_format
    case 2:
      return previewContent.value !== null
    default:
      return true
  }
})

// 方法
const loadTemplates = async () => {
  try {
    const response = await reportApi.getTemplates({
      category: templateFilter.category,
      keyword: templateFilter.keyword,
      page: templatePage.value,
      size: templateSize.value
    })
    
    templates.value = response.data.templates
    templateTotal.value = response.data.total
  } catch (error) {
    ElMessage.error('加载模板失败')
  }
}

const selectTemplate = (template) => {
  selectedTemplate.value = template
  templateVariables.value = template.variables || []
  
  // 初始化变量默认值
  template.variables?.forEach(variable => {
    if (variable.default_value) {
      reportConfig.variables[variable.name] = variable.default_value
    }
  })
}

const confirmTemplate = () => {
  if (selectedTemplate.value) {
    reportConfig.template_id = selectedTemplate.value.id
    showTemplateDialog.value = false
    
    if (currentStep.value === 0) {
      nextStep()
    }
  }
}

const nextStep = () => {
  if (canProceed.value && currentStep.value < 3) {
    currentStep.value++
  }
}

const prevStep = () => {
  if (currentStep.value > 0) {
    currentStep.value--
  }
}

const updateDateRange = (range) => {
  if (range && range.length === 2) {
    reportConfig.parameters.date_range = {
      start: range[0].toISOString().split('T')[0],
      end: range[1].toISOString().split('T')[0]
    }
  }
}

const generatePreview = async () => {
  previewing.value = true
  
  try {
    const response = await reportApi.previewReport({
      template_id: reportConfig.template_id,
      variables: reportConfig.variables,
      sample_data: {}
    })
    
    previewContent.value = response.data.preview_content
    ElMessage.success('预览生成成功')
  } catch (error) {
    ElMessage.error('预览生成失败')
  } finally {
    previewing.value = false
  }
}

const generateReport = async () => {
  generating.value = true
  generationProgress.value = 0
  generationStatus.value = ''
  generationMessage.value = '正在初始化...'
  
  try {
    const response = await reportApi.generateReport(reportConfig)
    currentTaskId.value = response.data.task_id
    
    // 开始轮询任务状态
    pollTaskStatus()
  } catch (error) {
    ElMessage.error('报告生成失败')
    generating.value = false
  }
}

const pollTaskStatus = async () => {
  try {
    const response = await reportApi.getReportStatus(currentTaskId.value)
    const taskData = response.data
    
    generationProgress.value = taskData.progress
    
    switch (taskData.status) {
      case 'pending':
        generationMessage.value = '任务排队中...'
        setTimeout(pollTaskStatus, 2000)
        break
      case 'processing':
        generationMessage.value = '正在生成报告...'
        setTimeout(pollTaskStatus, 1000)
        break
      case 'completed':
        generationMessage.value = '报告生成完成'
        generationStatus.value = 'success'
        generationResult.value = taskData
        generating.value = false
        loadReportHistory()
        break
      case 'failed':
        generationMessage.value = `生成失败: ${taskData.error_message}`
        generationStatus.value = 'exception'
        generating.value = false
        break
      default:
        setTimeout(pollTaskStatus, 2000)
    }
  } catch (error) {
    ElMessage.error('获取任务状态失败')
    generating.value = false
  }
}

const downloadReport = async () => {
  try {
    const response = await reportApi.downloadReport(currentTaskId.value)
    
    // 创建下载链接
    const blob = new Blob([response.data])
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${reportConfig.name}.${reportConfig.output_format}`
    link.click()
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('报告下载成功')
  } catch (error) {
    ElMessage.error('报告下载失败')
  }
}

const downloadHistoryReport = async (taskId: string) => {
  try {
    const response = await reportApi.downloadReport(taskId)
    
    // 创建下载链接
    const blob = new Blob([response.data])
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `report_${taskId}.pdf`
    link.click()
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('报告下载成功')
  } catch (error) {
    ElMessage.error('报告下载失败')
  }
}

const loadReportHistory = async () => {
  loadingHistory.value = true
  
  try {
    const response = await reportApi.getReportHistory({
      page: historyPage.value,
      size: historySize.value
    })
    
    reportHistory.value = response.data.reports
    historyTotal.value = response.data.total
  } catch (error) {
    ElMessage.error('加载报告历史失败')
  } finally {
    loadingHistory.value = false
  }
}

const deleteReport = async (taskId: string) => {
  try {
    await ElMessageBox.confirm('确定要删除这个报告吗？', '确认删除', {
      type: 'warning'
    })
    
    await reportApi.deleteReport(taskId)
    ElMessage.success('报告删除成功')
    loadReportHistory()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('报告删除失败')
    }
  }
}

const resetGenerator = () => {
  currentStep.value = 0
  generating.value = false
  generationResult.value = null
  previewContent.value = null
  currentTaskId.value = ''
  
  // 重置配置
  Object.assign(reportConfig, {
    name: '',
    template_id: '',
    data_sources: [],
    variables: {},
    parameters: {
      include_charts: true,
      filter_conditions: ''
    },
    output_format: 'pdf'
  })
}

const handleTemplatePageChange = (page: number) => {
  templatePage.value = page
  loadTemplates()
}

const handleHistoryPageChange = (page: number) => {
  historyPage.value = page
  loadReportHistory()
}

const getStatusType = (status: string) => {
  const statusMap = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return statusMap[status] || 'info'
}

const getStatusText = (status: string) => {
  const statusMap = {
    pending: '等待中',
    processing: '生成中',
    completed: '已完成',
    failed: '失败'
  }
  return statusMap[status] || status
}

const getFormatLabel = (format: string) => {
  const formatMap = {
    pdf: 'PDF',
    docx: 'Word文档',
    html: 'HTML'
  }
  return formatMap[format] || format
}

// 防抖搜索
const debounceSearch = debounce(() => {
  templatePage.value = 1
  loadTemplates()
}, 500)

// 生命周期
onMounted(() => {
  loadTemplates()
  loadReportHistory()
})

// 监听模板选择
watch(selectedTemplate, (newTemplate) => {
  if (newTemplate) {
    templateVariables.value = newTemplate.variables || []
  }
})
</script>

<style scoped>
.report-generator {
  padding: 20px;
}

.wizard-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.step-content {
  margin: 30px 0;
  min-height: 300px;
}

.step-actions {
  text-align: center;
  margin-top: 20px;
}

.template-selection .no-template {
  text-align: center;
  padding: 50px;
}

.selected-template {
  margin-bottom: 20px;
}

.parameter-config {
  max-width: 600px;
}

.preview-actions {
  margin-bottom: 20px;
}

.generation-ready,
.generation-progress,
.generation-result {
  text-align: center;
}

.result-actions {
  margin-top: 20px;
}

.template-selector {
  max-height: 500px;
  overflow-y: auto;
}

.template-filters {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.template-list {
  margin-bottom: 20px;
}

.template-card {
  cursor: pointer;
  transition: all 0.3s;
  margin-bottom: 16px;
}

.template-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.template-card.selected {
  border-color: #409eff;
  box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
}

.template-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 10px;
}

.template-type {
  font-size: 12px;
  color: #999;
}

.preview-dialog .preview-content {
  max-height: 600px;
  overflow-y: auto;
}

.preview-section {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.preview-section:last-child {
  border-bottom: none;
}

.text-content {
  line-height: 1.6;
}

.table-content {
  margin: 15px 0;
}

.chart-content {
  text-align: center;
  margin: 20px 0;
}

.history-card {
  margin-top: 20px;
}
</style>
```

## 验收标准

### 功能性验收标准

1. **模板管理**
   - ✅ 支持创建、编辑、删除报告模板
   - ✅ 模板支持变量替换和条件渲染
   - ✅ 支持模板分类和搜索
   - ✅ 模板预览功能正常

2. **报告生成**
   - ✅ 支持PDF、DOCX、HTML格式输出
   - ✅ 报告生成任务异步处理
   - ✅ 生成进度实时显示
   - ✅ 支持批量数据处理

3. **数据集成**
   - ✅ 支持多种数据源接入
   - ✅ 数据查询和过滤功能
   - ✅ 图表自动生成
   - ✅ 数据格式化和转换

4. **用户界面**
   - ✅ 报告生成向导流程清晰
   - ✅ 模板选择和配置界面友好
   - ✅ 报告预览功能完整
   - ✅ 历史记录管理便捷

### 性能验收标准

1. **生成性能**
   - ✅ 小型报告（<10页）生成时间 < 30秒
   - ✅ 中型报告（10-50页）生成时间 < 2分钟
   - ✅ 大型报告（50-200页）生成时间 < 10分钟
   - ✅ 支持并发生成任务数 ≥ 10个

2. **系统性能**
   - ✅ API响应时间 < 500ms
   - ✅ 模板加载时间 < 2秒
   - ✅ 文件下载速度 > 1MB/s
   - ✅ 系统内存使用率 < 80%

3. **存储性能**
   - ✅ 支持单个报告文件大小 ≤ 100MB
   - ✅ 模板存储容量 ≥ 1GB
   - ✅ 报告历史保存期限 ≥ 6个月
   - ✅ 文件压缩率 ≥ 30%

### 安全性验收标准

1. **访问控制**
   - ✅ 用户身份验证和授权
   - ✅ 模板访问权限控制
   - ✅ 报告下载权限验证
   - ✅ 敏感数据脱敏处理

2. **数据安全**
   - ✅ 报告文件加密存储
   - ✅ 传输过程HTTPS加密
   - ✅ 临时文件自动清理
   - ✅ 数据备份和恢复

3. **系统安全**
   - ✅ SQL注入防护
   - ✅ XSS攻击防护
   - ✅ 文件上传安全检查
   - ✅ 操作日志记录

## 业务价值

### 直接价值

1. **提升效率**
   - 自动化报告生成，减少人工编写时间90%
   - 标准化模板确保报告格式一致性
   - 批量处理能力提升数据分析效率
   - 可视化图表自动生成节省设计时间

2. **降低成本**
   - 减少报告制作人力成本
   - 降低格式错误和数据错误率
   - 提高报告质量和专业度
   - 减少重复性工作投入

3. **增强能力**
   - 支持复杂数据分析报告生成
   - 提供多格式输出满足不同需求
   - 历史数据对比分析能力
   - 自定义模板适应业务变化

### 间接价值

1. **决策支持**
   - 快速生成分析报告支持决策制定
   - 数据可视化提升信息理解效率
   - 历史趋势分析预测未来发展
   - 多维度数据对比发现业务机会

2. **知识管理**
   - 标准化报告模板积累业务知识
   - 历史报告形成知识库资源
   - 分析方法和指标体系沉淀
   - 经验传承和知识共享

3. **业务创新**
   - 新的数据分析维度和方法
   - 自动化流程释放创新时间
   - 数据驱动的业务优化
   - 智能化报告推荐和建议

## 依赖关系

### 技术依赖

1. **后端服务依赖**
   - 用户认证服务（用户身份验证）
   - 文档管理服务（数据源接入）
   - 数据分析服务（统计计算）
   - 文件存储服务（报告文件存储）

2. **数据库依赖**
   - PostgreSQL（模板和任务数据）
   - MongoDB（文档数据）
   - Redis（缓存和任务队列）
   - MinIO/S3（文件存储）

3. **第三方库依赖**
   - ReportLab（PDF生成）
   - python-docx（Word文档生成）
   - Jinja2（模板引擎）
   - Celery（异步任务处理）

### 业务依赖

1. **数据依赖**
   - 历史文献数据完整性
   - 分类标签数据准确性
   - 用户行为数据可用性
   - 统计分析结果可靠性

2. **流程依赖**
   - 数据预处理流程完成
   - 分析模型训练完成
   - 用户权限配置完成
   - 模板审核流程建立

### 环境依赖

1. **开发环境**
   - Python 3.9+
   - Node.js 16+
   - Docker容器化环境
   - Git版本控制

2. **部署环境**
   - Kubernetes集群
   - 负载均衡器
   - 监控和日志系统
   - 备份和恢复系统

## 风险评估

### 技术风险

1. **性能风险**
   - **风险**：大型报告生成时间过长
   - **影响**：用户体验下降，系统资源占用过高
   - **缓解措施**：
     - 实施分页生成策略
     - 优化模板渲染算法
     - 增加缓存机制
     - 设置生成超时限制

2. **兼容性风险**
   - **风险**：不同格式输出兼容性问题
   - **影响**：报告在不同设备上显示异常
   - **缓解措施**：
     - 建立格式测试用例
     - 使用标准化库和工具
     - 定期兼容性测试
     - 提供格式转换工具

3. **稳定性风险**
   - **风险**：模板解析错误导致生成失败
   - **影响**：报告生成中断，数据丢失
   - **缓解措施**：
     - 模板语法验证
     - 异常处理和重试机制
     - 生成状态监控
     - 错误日志记录

### 业务风险

1. **数据质量风险**
   - **风险**：源数据质量问题影响报告准确性
   - **影响**：报告结论错误，决策失误
   - **缓解措施**：
     - 数据质量检查机制
     - 异常数据标记和处理
     - 数据来源追溯
     - 质量评估指标

2. **安全风险**
   - **风险**：敏感数据泄露
   - **影响**：数据安全事故，合规风险
   - **缓解措施**：
     - 数据脱敏处理
     - 访问权限控制
     - 加密存储和传输
     - 安全审计日志

3. **合规风险**
   - **风险**：报告格式不符合行业标准
   - **影响**：无法满足监管要求
   - **缓解措施**：
     - 标准模板库建设
     - 合规性检查工具
     - 专家评审机制
     - 持续更新维护

### 运营风险

1. **容量风险**
   - **风险**：用户量增长超出系统承载能力
   - **影响**：系统性能下降，服务中断
   - **缓解措施**：
     - 弹性扩容机制
     - 负载均衡配置
     - 性能监控告警
     - 容量规划评估

2. **维护风险**
   - **风险**：系统维护期间服务不可用
   - **影响**：业务中断，用户体验差
   - **缓解措施**：
     - 滚动更新策略
     - 蓝绿部署方案
     - 维护窗口规划
     - 应急恢复预案

## 开发任务分解

### 后端开发任务

#### Phase 1: 核心服务开发（2周）

1. **模板管理模块**
   - 模板CRUD操作API
   - 模板变量解析引擎
   - 模板验证和预览
   - 模板分类和搜索
   - **工作量**：5人天

2. **报告生成引擎**
   - PDF生成服务
   - DOCX生成服务
   - HTML生成服务
   - 模板渲染引擎
   - **工作量**：8人天

3. **任务管理系统**
   - 异步任务队列
   - 任务状态跟踪
   - 进度监控机制
   - 错误处理和重试
   - **工作量**：5人天

#### Phase 2: 数据集成（1.5周）

1. **数据源接入**
   - 数据库连接器
   - API数据获取
   - 文件数据导入
   - 数据格式转换
   - **工作量**：4人天

2. **图表生成服务**
   - 统计图表生成
   - 图表样式配置
   - 图表数据绑定
   - 图表格式输出
   - **工作量**：4人天

3. **数据处理管道**
   - 数据清洗和预处理
   - 统计计算引擎
   - 数据聚合和分组
   - 结果缓存机制
   - **工作量**：3人天

#### Phase 3: 高级功能（1周）

1. **批量处理**
   - 批量报告生成
   - 并发任务管理
   - 资源调度优化
   - 结果合并处理
   - **工作量**：3人天

2. **性能优化**
   - 生成算法优化
   - 内存使用优化
   - 缓存策略实施
   - 并发控制机制
   - **工作量**：2人天

3. **监控和日志**
   - 性能指标监控
   - 错误日志记录
   - 用户行为跟踪
   - 系统健康检查
   - **工作量**：2人天

### 前端开发任务

#### Phase 1: 基础界面（1.5周）

1. **报告生成向导**
   - 步骤导航组件
   - 模板选择界面
   - 参数配置表单
   - 进度显示组件
   - **工作量**：4人天

2. **模板管理界面**
   - 模板列表展示
   - 模板搜索过滤
   - 模板预览功能
   - 模板操作按钮
   - **工作量**：3人天

3. **报告预览功能**
   - 预览内容渲染
   - 章节导航
   - 格式化显示
   - 预览操作控制
   - **工作量**：3人天

#### Phase 2: 交互功能（1周）

1. **参数配置界面**
   - 动态表单生成
   - 数据源选择
   - 高级参数设置
   - 表单验证机制
   - **工作量**：3人天

2. **历史记录管理**
   - 历史列表展示
   - 状态标识显示
   - 操作按钮组
   - 分页和搜索
   - **工作量**：2人天

3. **文件下载功能**
   - 下载进度显示
   - 多格式下载
   - 批量下载支持
   - 下载状态反馈
   - **工作量**：2人天

#### Phase 3: 用户体验优化（0.5周）

1. **响应式设计**
   - 移动端适配
   - 布局优化
   - 交互体验改进
   - 加载状态优化
   - **工作量**：2人天

2. **错误处理**
   - 错误信息展示
   - 用户操作引导
   - 异常状态处理
   - 重试机制界面
   - **工作量**：1人天

### 测试任务

#### 单元测试（1周）

1. **后端单元测试**
   - 模板解析测试
   - 报告生成测试
   - 数据处理测试
   - API接口测试
   - **工作量**：3人天

2. **前端单元测试**
   - 组件功能测试
   - 用户交互测试
   - 数据流测试
   - 界面渲染测试
   - **工作量**：2人天

3. **集成测试**
   - 端到端流程测试
   - 数据一致性测试
   - 性能基准测试
   - 兼容性测试
   - **工作量**：2人天

#### 系统测试（0.5周）

1. **功能测试**
   - 完整功能验证
   - 边界条件测试
   - 异常场景测试
   - 用户场景测试
   - **工作量**：2人天

2. **性能测试**
   - 负载测试
   - 压力测试
   - 并发测试
   - 资源使用测试
   - **工作量**：1人天

### 部署任务

#### 环境准备（0.5周）

1. **基础设施**
   - 容器镜像构建
   - 配置文件准备
   - 依赖服务部署
   - 网络配置
   - **工作量**：1人天

2. **监控配置**
   - 监控指标配置
   - 告警规则设置
   - 日志收集配置
   - 健康检查设置
   - **工作量**：1人天

3. **部署验证**
   - 部署流程验证
   - 功能验证测试
   - 性能验证测试
   - 回滚机制测试
   - **工作量**：1人天

### 总体时间安排

- **总开发周期**：6.5周
- **后端开发**：4.5周
- **前端开发**：3周
- **测试阶段**：1.5周
- **部署上线**：0.5周
- **并行开发**：前后端可并行进行
- **关键路径**：报告生成引擎 → 数据集成 → 系统测试

### 人力资源需求

- **后端开发工程师**：2人
- **前端开发工程师**：2人
- **测试工程师**：1人
- **DevOps工程师**：1人
- **项目经理**：1人
- **总人力投入**：7人 × 6.5周 = 45.5人周