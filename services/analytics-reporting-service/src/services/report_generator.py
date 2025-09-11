"""
报告生成服务模块

负责各种类型的数据分析报告生成和导出功能：
- PDF报告生成
- Excel报告导出
- 图表可视化
- 报告模板管理
- 批量报告生成
"""

import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from io import BytesIO
import json
import base64

# 报告生成相关库
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus.flowables import PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.colors import HexColor

# Excel导出相关库
import pandas as pd
import xlsxwriter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import LineChart, BarChart, PieChart, Reference

# 图表生成相关库
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio

from ..models import (
    get_db, get_redis,
    ReportTemplate, GeneratedReport,
    AnalysisTask, AnalysisTaskType,
    ContentPerformance, PlatformComparison, 
    TrendAnalysis, UserBehaviorInsights,
    ChartConfig, ExportFormat
)
from ..config.settings import settings

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """
    报告生成器核心类
    
    提供完整的报告生成功能：
    - 多格式报告生成 (PDF, Excel, HTML)
    - 数据可视化图表生成
    - 报告模板管理和应用
    - 批量报告生成和调度
    - 报告分享和权限管理
    """
    
    def __init__(self):
        self.redis_client = None
        self.temp_dir = tempfile.mkdtemp()
        self.chart_cache = {}

    async def initialize(self):
        """初始化报告生成器"""
        try:
            self.redis_client = await get_redis()
            
            # 确保报告目录存在
            os.makedirs(settings.report.report_storage_path, exist_ok=True)
            
            logger.info("ReportGenerator 初始化完成")
        except Exception as e:
            logger.error(f"ReportGenerator 初始化失败: {e}")
            raise

    async def generate_analysis_report(
        self,
        task_id: uuid.UUID,
        user_id: str,
        template_id: Optional[uuid.UUID] = None,
        export_format: ExportFormat = ExportFormat.PDF,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成分析报告
        
        Args:
            task_id: 分析任务ID
            user_id: 用户ID
            template_id: 报告模板ID
            export_format: 导出格式
            custom_config: 自定义配置
            
        Returns:
            生成的报告信息
        """
        try:
            logger.info(f"开始生成分析报告 - 任务: {task_id}, 格式: {export_format}")
            
            # 获取分析任务数据
            analysis_task = await self._get_analysis_task(task_id)
            if not analysis_task:
                raise ValueError(f"分析任务 {task_id} 不存在")
            
            # 获取报告模板
            template = None
            if template_id:
                template = await self._get_report_template(template_id)
            
            # 获取分析结果数据
            analysis_data = await self._get_analysis_data(task_id)
            
            # 生成报告内容
            report_content = await self._build_report_content(
                analysis_task, analysis_data, template, custom_config
            )
            
            # 生成图表
            chart_data = await self._generate_charts(analysis_data, template)
            
            # 根据格式生成报告文件
            if export_format == ExportFormat.PDF:
                file_path = await self._generate_pdf_report(
                    report_content, chart_data, analysis_task
                )
            elif export_format == ExportFormat.EXCEL:
                file_path = await self._generate_excel_report(
                    report_content, chart_data, analysis_task
                )
            elif export_format == ExportFormat.JSON:
                file_path = await self._generate_json_report(
                    report_content, analysis_task
                )
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            # 保存报告记录
            report_record = await self._save_report_record(
                analysis_task, template_id, file_path, export_format, user_id
            )
            
            report_info = {
                'report_id': str(report_record.id),
                'file_path': file_path,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'format': export_format.value,
                'generated_at': datetime.now(),
                'download_url': f"/api/v1/reports/{report_record.id}/download"
            }
            
            logger.info(f"报告生成完成 - ID: {report_record.id}")
            return report_info
            
        except Exception as e:
            logger.error(f"生成分析报告失败: {e}")
            raise

    async def generate_dashboard_report(
        self,
        user_id: str,
        date_range: Tuple[datetime, datetime],
        platforms: List[str],
        export_format: ExportFormat = ExportFormat.PDF
    ) -> Dict[str, Any]:
        """
        生成仪表板报告
        
        Args:
            user_id: 用户ID
            date_range: 日期范围
            platforms: 平台列表
            export_format: 导出格式
            
        Returns:
            生成的报告信息
        """
        try:
            logger.info(f"开始生成仪表板报告 - 用户: {user_id}")
            
            start_date, end_date = date_range
            
            # 收集仪表板数据
            dashboard_data = await self._collect_dashboard_data(
                user_id, start_date, end_date, platforms
            )
            
            # 生成综合分析内容
            report_content = await self._build_dashboard_content(dashboard_data, date_range)
            
            # 生成可视化图表
            chart_data = await self._generate_dashboard_charts(dashboard_data)
            
            # 生成报告文件
            report_title = f"数据分析仪表板报告_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            if export_format == ExportFormat.PDF:
                file_path = await self._generate_dashboard_pdf(
                    report_content, chart_data, report_title
                )
            elif export_format == ExportFormat.EXCEL:
                file_path = await self._generate_dashboard_excel(
                    report_content, chart_data, report_title
                )
            else:
                raise ValueError(f"仪表板报告不支持格式: {export_format}")
            
            # 创建报告记录
            report_record = await self._create_dashboard_report_record(
                user_id, file_path, export_format, date_range
            )
            
            report_info = {
                'report_id': str(report_record.id),
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'format': export_format.value,
                'generated_at': datetime.now(),
                'download_url': f"/api/v1/reports/{report_record.id}/download"
            }
            
            logger.info(f"仪表板报告生成完成 - ID: {report_record.id}")
            return report_info
            
        except Exception as e:
            logger.error(f"生成仪表板报告失败: {e}")
            raise

    async def generate_custom_report(
        self,
        user_id: str,
        report_config: Dict[str, Any],
        export_format: ExportFormat = ExportFormat.PDF
    ) -> Dict[str, Any]:
        """
        生成自定义报告
        
        Args:
            user_id: 用户ID
            report_config: 报告配置
            export_format: 导出格式
            
        Returns:
            生成的报告信息
        """
        try:
            logger.info(f"开始生成自定义报告 - 用户: {user_id}")
            
            # 解析报告配置
            report_type = report_config.get('type', 'custom')
            data_sources = report_config.get('data_sources', [])
            chart_configs = report_config.get('charts', [])
            filters = report_config.get('filters', {})
            
            # 收集数据
            report_data = await self._collect_custom_report_data(
                user_id, data_sources, filters
            )
            
            # 生成内容
            report_content = await self._build_custom_content(
                report_data, report_config
            )
            
            # 生成图表
            chart_data = await self._generate_custom_charts(
                report_data, chart_configs
            )
            
            # 生成文件
            report_title = report_config.get('title', f'自定义报告_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            
            if export_format == ExportFormat.PDF:
                file_path = await self._generate_custom_pdf(
                    report_content, chart_data, report_title
                )
            elif export_format == ExportFormat.EXCEL:
                file_path = await self._generate_custom_excel(
                    report_content, chart_data, report_title
                )
            else:
                raise ValueError(f"自定义报告不支持格式: {export_format}")
            
            # 保存记录
            report_record = await self._create_custom_report_record(
                user_id, file_path, export_format, report_config
            )
            
            report_info = {
                'report_id': str(report_record.id),
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'format': export_format.value,
                'generated_at': datetime.now(),
                'download_url': f"/api/v1/reports/{report_record.id}/download"
            }
            
            logger.info(f"自定义报告生成完成 - ID: {report_record.id}")
            return report_info
            
        except Exception as e:
            logger.error(f"生成自定义报告失败: {e}")
            raise

    # ===== PDF报告生成 =====

    async def _generate_pdf_report(
        self,
        report_content: Dict[str, Any],
        chart_data: Dict[str, Any],
        analysis_task: Any
    ) -> str:
        """生成PDF格式报告"""
        
        try:
            # 创建PDF文件路径
            filename = f"analysis_report_{analysis_task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            file_path = os.path.join(settings.report.report_storage_path, filename)
            
            # 创建PDF文档
            doc = SimpleDocTemplate(
                file_path,
                pagesize=A4,
                rightMargin=72, leftMargin=72,
                topMargin=72, bottomMargin=18
            )
            
            # 创建样式
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # 居中
            )
            
            # 构建PDF内容
            story = []
            
            # 标题
            story.append(Paragraph(report_content.get('title', '数据分析报告'), title_style))
            story.append(Spacer(1, 20))
            
            # 报告摘要
            if 'summary' in report_content:
                story.append(Paragraph('报告摘要', styles['Heading2']))
                story.append(Paragraph(report_content['summary'], styles['Normal']))
                story.append(Spacer(1, 20))
            
            # 添加图表
            for chart_name, chart_path in chart_data.items():
                if os.path.exists(chart_path):
                    story.append(Paragraph(f'{chart_name}', styles['Heading3']))
                    
                    # 调整图片大小
                    img = Image(chart_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
            
            # 详细分析
            if 'sections' in report_content:
                for section in report_content['sections']:
                    story.append(Paragraph(section['title'], styles['Heading2']))
                    story.append(Paragraph(section['content'], styles['Normal']))
                    
                    # 添加表格数据
                    if 'table_data' in section:
                        table = Table(section['table_data'])
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 14),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(table)
                    
                    story.append(Spacer(1, 20))
            
            # 生成PDF
            doc.build(story)
            
            logger.info(f"PDF报告生成完成: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"生成PDF报告失败: {e}")
            raise

    # ===== Excel报告生成 =====

    async def _generate_excel_report(
        self,
        report_content: Dict[str, Any],
        chart_data: Dict[str, Any],
        analysis_task: Any
    ) -> str:
        """生成Excel格式报告"""
        
        try:
            # 创建Excel文件路径
            filename = f"analysis_report_{analysis_task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            file_path = os.path.join(settings.report.report_storage_path, filename)
            
            # 创建Excel工作簿
            workbook = xlsxwriter.Workbook(file_path)
            
            # 定义样式
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 18,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            header_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'bg_color': '#D9E2F3',
                'border': 1
            })
            
            data_format = workbook.add_format({
                'border': 1,
                'align': 'center'
            })
            
            # 创建摘要工作表
            summary_sheet = workbook.add_worksheet('报告摘要')
            
            # 写入标题
            summary_sheet.merge_range('A1:F1', report_content.get('title', '数据分析报告'), title_format)
            
            # 写入摘要信息
            row = 2
            if 'summary' in report_content:
                summary_sheet.write(row, 0, '报告摘要:', header_format)
                summary_sheet.merge_range(row, 1, row, 5, report_content['summary'])
                row += 2
            
            # 写入基本信息
            basic_info = [
                ['分析任务ID', str(analysis_task.id)],
                ['任务类型', analysis_task.task_type.value],
                ['生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['分析周期', f"{analysis_task.start_date} 至 {analysis_task.end_date}"]
            ]
            
            for info_row in basic_info:
                summary_sheet.write(row, 0, info_row[0], header_format)
                summary_sheet.write(row, 1, info_row[1], data_format)
                row += 1
            
            # 创建数据工作表
            if 'sections' in report_content:
                for i, section in enumerate(report_content['sections']):
                    sheet_name = f"数据{i+1}"
                    data_sheet = workbook.add_worksheet(sheet_name)
                    
                    # 写入节标题
                    data_sheet.merge_range('A1:F1', section['title'], title_format)
                    
                    # 写入表格数据
                    if 'table_data' in section:
                        table_data = section['table_data']
                        
                        # 写入表头
                        for col, header in enumerate(table_data[0]):
                            data_sheet.write(2, col, header, header_format)
                        
                        # 写入数据
                        for row_idx, row_data in enumerate(table_data[1:], 3):
                            for col_idx, cell_data in enumerate(row_data):
                                data_sheet.write(row_idx, col_idx, cell_data, data_format)
                        
                        # 添加图表
                        if section['title'] in chart_data:
                            chart = workbook.add_chart({'type': 'line'})
                            chart.add_series({
                                'name': section['title'],
                                'categories': [sheet_name, 3, 0, len(table_data), 0],
                                'values': [sheet_name, 3, 1, len(table_data), 1],
                            })
                            chart.set_title({'name': section['title']})
                            data_sheet.insert_chart('H3', chart)
            
            # 关闭工作簿
            workbook.close()
            
            logger.info(f"Excel报告生成完成: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"生成Excel报告失败: {e}")
            raise

    # ===== JSON报告生成 =====

    async def _generate_json_report(
        self,
        report_content: Dict[str, Any],
        analysis_task: Any
    ) -> str:
        """生成JSON格式报告"""
        
        try:
            # 创建JSON文件路径
            filename = f"analysis_report_{analysis_task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(settings.report.report_storage_path, filename)
            
            # 准备JSON数据
            json_report = {
                'report_metadata': {
                    'report_id': str(analysis_task.id),
                    'title': report_content.get('title', '数据分析报告'),
                    'generated_at': datetime.now().isoformat(),
                    'analysis_type': analysis_task.task_type.value,
                    'date_range': {
                        'start': analysis_task.start_date.isoformat() if analysis_task.start_date else None,
                        'end': analysis_task.end_date.isoformat() if analysis_task.end_date else None
                    }
                },
                'report_content': report_content,
                'analysis_results': analysis_task.result_data if hasattr(analysis_task, 'result_data') else {}
            }
            
            # 写入JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"JSON报告生成完成: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"生成JSON报告失败: {e}")
            raise

    # ===== 图表生成 =====

    async def _generate_charts(
        self,
        analysis_data: Dict[str, Any],
        template: Optional[Any] = None
    ) -> Dict[str, str]:
        """生成分析图表"""
        
        try:
            chart_paths = {}
            
            # 趋势图表
            if 'trend_data' in analysis_data:
                trend_chart_path = await self._create_trend_chart(analysis_data['trend_data'])
                if trend_chart_path:
                    chart_paths['趋势分析图'] = trend_chart_path
            
            # 平台对比图表
            if 'platform_data' in analysis_data:
                platform_chart_path = await self._create_platform_chart(analysis_data['platform_data'])
                if platform_chart_path:
                    chart_paths['平台对比图'] = platform_chart_path
            
            # 内容表现图表
            if 'content_performance' in analysis_data:
                content_chart_path = await self._create_content_chart(analysis_data['content_performance'])
                if content_chart_path:
                    chart_paths['内容表现图'] = content_chart_path
            
            # 用户行为图表
            if 'user_behavior' in analysis_data:
                behavior_chart_path = await self._create_behavior_chart(analysis_data['user_behavior'])
                if behavior_chart_path:
                    chart_paths['用户行为图'] = behavior_chart_path
            
            return chart_paths
            
        except Exception as e:
            logger.error(f"生成图表失败: {e}")
            return {}

    async def _create_trend_chart(self, trend_data: List[Dict[str, Any]]) -> Optional[str]:
        """创建趋势分析图表"""
        
        try:
            if not trend_data:
                return None
            
            # 准备数据
            dates = []
            values = []
            
            for item in trend_data:
                if 'data_points' in item:
                    for point in item['data_points']:
                        dates.append(point['timestamp'])
                        values.append(point['value'])
            
            if not dates or not values:
                return None
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            plt.plot(dates, values, marker='o', linewidth=2, markersize=6)
            plt.title('数据趋势分析', fontsize=16, fontweight='bold')
            plt.xlabel('时间')
            plt.ylabel('数值')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            chart_filename = f"trend_chart_{uuid.uuid4().hex[:8]}.png"
            chart_path = os.path.join(self.temp_dir, chart_filename)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"创建趋势图表失败: {e}")
            return None

    async def _create_platform_chart(self, platform_data: List[Dict[str, Any]]) -> Optional[str]:
        """创建平台对比图表"""
        
        try:
            if not platform_data:
                return None
            
            # 准备数据
            platforms = []
            engagement_rates = []
            total_views = []
            
            for platform in platform_data:
                platforms.append(platform.get('platform', '未知平台'))
                engagement_rates.append(platform.get('avg_engagement_rate', 0))
                total_views.append(platform.get('total_views', 0))
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 参与度对比
            ax1.bar(platforms, engagement_rates, color='skyblue', alpha=0.8)
            ax1.set_title('平台参与度对比')
            ax1.set_ylabel('平均参与度')
            ax1.tick_params(axis='x', rotation=45)
            
            # 浏览量对比
            ax2.bar(platforms, total_views, color='lightcoral', alpha=0.8)
            ax2.set_title('平台浏览量对比')
            ax2.set_ylabel('总浏览量')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            chart_filename = f"platform_chart_{uuid.uuid4().hex[:8]}.png"
            chart_path = os.path.join(self.temp_dir, chart_filename)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"创建平台图表失败: {e}")
            return None

    async def _create_content_chart(self, content_data: List[Dict[str, Any]]) -> Optional[str]:
        """创建内容表现图表"""
        
        try:
            if not content_data:
                return None
            
            # 准备数据 - 取前10个内容
            content_data = content_data[:10]
            content_titles = [item.get('title', f'内容{i+1}')[:20] for i, item in enumerate(content_data)]
            performance_scores = [item.get('performance_score', 0) for item in content_data]
            
            # 创建横向条形图
            plt.figure(figsize=(12, 8))
            bars = plt.barh(content_titles, performance_scores, color='lightgreen', alpha=0.8)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}', ha='left', va='center')
            
            plt.title('内容表现评分排行')
            plt.xlabel('表现评分')
            plt.gca().invert_yaxis()  # 反转y轴，最高分在顶部
            plt.tight_layout()
            
            # 保存图表
            chart_filename = f"content_chart_{uuid.uuid4().hex[:8]}.png"
            chart_path = os.path.join(self.temp_dir, chart_filename)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"创建内容图表失败: {e}")
            return None

    async def _create_behavior_chart(self, behavior_data: Dict[str, Any]) -> Optional[str]:
        """创建用户行为图表"""
        
        try:
            if not behavior_data:
                return None
            
            # 创建饼图显示行为分布
            behavior_patterns = behavior_data.get('behavior_patterns', {})
            action_distribution = behavior_patterns.get('action_distribution', {})
            
            if not action_distribution:
                return None
            
            # 准备数据
            labels = list(action_distribution.keys())
            sizes = list(action_distribution.values())
            colors = plt.cm.Set3(range(len(labels)))
            
            # 创建饼图
            plt.figure(figsize=(10, 8))
            wedges, texts, autotexts = plt.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90
            )
            
            plt.title('用户行为分布', fontsize=16, fontweight='bold')
            plt.axis('equal')
            
            # 保存图表
            chart_filename = f"behavior_chart_{uuid.uuid4().hex[:8]}.png"
            chart_path = os.path.join(self.temp_dir, chart_filename)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"创建行为图表失败: {e}")
            return None

    # ===== 辅助方法 =====

    async def _get_analysis_task(self, task_id: uuid.UUID) -> Optional[Any]:
        """获取分析任务"""
        
        try:
            async with get_db() as db:
                query = "SELECT * FROM analysis_tasks WHERE id = :task_id"
                result = await db.execute(text(query), {"task_id": str(task_id)})
                return result.fetchone()
        except Exception as e:
            logger.error(f"获取分析任务失败: {e}")
            return None

    async def _get_report_template(self, template_id: uuid.UUID) -> Optional[Any]:
        """获取报告模板"""
        
        try:
            async with get_db() as db:
                query = "SELECT * FROM report_templates WHERE id = :template_id"
                result = await db.execute(text(query), {"template_id": str(template_id)})
                return result.fetchone()
        except Exception as e:
            logger.error(f"获取报告模板失败: {e}")
            return None

    async def _get_analysis_data(self, task_id: uuid.UUID) -> Dict[str, Any]:
        """获取分析数据"""
        
        try:
            # 从Redis缓存获取分析结果
            cache_key = f"analysis_result:{task_id}"
            cached_result = await self.redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            # 从数据库获取
            async with get_db() as db:
                query = "SELECT result_data FROM analysis_tasks WHERE id = :task_id"
                result = await db.execute(text(query), {"task_id": str(task_id)})
                row = result.fetchone()
                
                if row and row.result_data:
                    return row.result_data
            
            return {}
            
        except Exception as e:
            logger.error(f"获取分析数据失败: {e}")
            return {}

    async def _build_report_content(
        self,
        analysis_task: Any,
        analysis_data: Dict[str, Any],
        template: Optional[Any] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """构建报告内容"""
        
        try:
            report_content = {
                'title': f'{analysis_task.title} - 分析报告',
                'summary': '本报告基于数据分析任务生成，包含详细的分析结果和洞察。',
                'sections': []
            }
            
            # 添加基本信息部分
            basic_section = {
                'title': '基本信息',
                'content': f'分析任务类型：{analysis_task.task_type.value}\n'
                          f'分析周期：{analysis_task.start_date} 至 {analysis_task.end_date}\n'
                          f'任务状态：{analysis_task.status.value}',
                'table_data': [
                    ['项目', '内容'],
                    ['任务ID', str(analysis_task.id)],
                    ['创建时间', analysis_task.created_at.strftime('%Y-%m-%d %H:%M:%S')],
                    ['完成时间', analysis_task.completed_at.strftime('%Y-%m-%d %H:%M:%S') if analysis_task.completed_at else '进行中']
                ]
            }
            report_content['sections'].append(basic_section)
            
            # 根据分析类型添加特定内容
            if analysis_task.task_type == AnalysisTaskType.CONTENT_PERFORMANCE:
                await self._add_content_performance_section(report_content, analysis_data)
            elif analysis_task.task_type == AnalysisTaskType.PLATFORM_COMPARISON:
                await self._add_platform_comparison_section(report_content, analysis_data)
            elif analysis_task.task_type == AnalysisTaskType.TREND_ANALYSIS:
                await self._add_trend_analysis_section(report_content, analysis_data)
            elif analysis_task.task_type == AnalysisTaskType.USER_BEHAVIOR:
                await self._add_user_behavior_section(report_content, analysis_data)
            
            return report_content
            
        except Exception as e:
            logger.error(f"构建报告内容失败: {e}")
            return {'title': '报告生成错误', 'summary': str(e), 'sections': []}

    async def _add_content_performance_section(
        self,
        report_content: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ):
        """添加内容表现分析部分"""
        
        content_performance = analysis_data.get('content_performance', [])
        
        if content_performance:
            # 统计信息
            total_content = len(content_performance)
            avg_performance = sum(item.get('performance_score', 0) for item in content_performance) / total_content
            best_content = max(content_performance, key=lambda x: x.get('performance_score', 0))
            
            section = {
                'title': '内容表现分析',
                'content': f'本期共分析了 {total_content} 个内容，平均表现评分为 {avg_performance:.2f}。\n'
                          f'表现最佳的内容是：{best_content.get("title", "未知")}，评分：{best_content.get("performance_score", 0):.2f}',
                'table_data': [
                    ['内容标题', '浏览量', '点赞数', '评论数', '分享数', '表现评分']
                ]
            }
            
            # 添加前10个内容的数据
            for item in content_performance[:10]:
                section['table_data'].append([
                    item.get('title', ''),
                    item.get('views', 0),
                    item.get('likes', 0),
                    item.get('comments', 0),
                    item.get('shares', 0),
                    f"{item.get('performance_score', 0):.2f}"
                ])
            
            report_content['sections'].append(section)

    async def _add_platform_comparison_section(
        self,
        report_content: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ):
        """添加平台对比分析部分"""
        
        platform_data = analysis_data.get('platform_data', [])
        
        if platform_data:
            best_platform = max(platform_data, key=lambda x: x.get('total_engagement', 0))
            
            section = {
                'title': '平台对比分析',
                'content': f'本次分析涵盖了 {len(platform_data)} 个平台。\n'
                          f'表现最佳的平台是：{best_platform.get("platform", "未知")}，总互动数：{best_platform.get("total_engagement", 0)}',
                'table_data': [
                    ['平台', '内容数', '总浏览量', '总互动数', '平均参与度']
                ]
            }
            
            for platform in platform_data:
                section['table_data'].append([
                    platform.get('platform', ''),
                    platform.get('total_content', 0),
                    platform.get('total_views', 0),
                    platform.get('total_engagement', 0),
                    f"{platform.get('avg_engagement_rate', 0):.4f}"
                ])
            
            report_content['sections'].append(section)

    async def _add_trend_analysis_section(
        self,
        report_content: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ):
        """添加趋势分析部分"""
        
        trend_data = analysis_data.get('trend_data', [])
        
        if trend_data:
            section = {
                'title': '趋势分析',
                'content': f'本次趋势分析包含 {len(trend_data)} 个指标的变化情况。',
                'table_data': [
                    ['指标名称', '趋势方向', '增长率(%)', '数据点数量']
                ]
            }
            
            for trend in trend_data:
                section['table_data'].append([
                    trend.get('metric_name', ''),
                    trend.get('trend_direction', ''),
                    f"{trend.get('growth_rate', 0):.2f}",
                    len(trend.get('data_points', []))
                ])
            
            report_content['sections'].append(section)

    async def _add_user_behavior_section(
        self,
        report_content: Dict[str, Any],
        analysis_data: Dict[str, Any]
    ):
        """添加用户行为分析部分"""
        
        behavior_data = analysis_data.get('user_behavior', {})
        
        if behavior_data:
            section = {
                'title': '用户行为分析',
                'content': f'活跃用户数：{behavior_data.get("active_users", 0)}\n'
                          f'新用户数：{behavior_data.get("new_users", 0)}\n'
                          f'回访用户数：{behavior_data.get("returning_users", 0)}\n'
                          f'平均会话时长：{behavior_data.get("avg_session_duration", 0):.2f}分钟',
                'table_data': [
                    ['指标', '数值'],
                    ['活跃用户', behavior_data.get('active_users', 0)],
                    ['新用户', behavior_data.get('new_users', 0)],
                    ['回访用户', behavior_data.get('returning_users', 0)],
                    ['平均会话时长(分钟)', f"{behavior_data.get('avg_session_duration', 0):.2f}"]
                ]
            }
            
            report_content['sections'].append(section)

    async def _save_report_record(
        self,
        analysis_task: Any,
        template_id: Optional[uuid.UUID],
        file_path: str,
        export_format: ExportFormat,
        user_id: str
    ) -> GeneratedReport:
        """保存报告记录"""
        
        try:
            async with get_db() as db:
                report = GeneratedReport(
                    analysis_task_id=analysis_task.id,
                    template_id=template_id,
                    title=f"{analysis_task.title} - 分析报告",
                    user_id=user_id,
                    file_path=file_path,
                    file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    file_format=export_format.value,
                    generation_status='completed'
                )
                
                db.add(report)
                await db.commit()
                await db.refresh(report)
                
                return report
                
        except Exception as e:
            logger.error(f"保存报告记录失败: {e}")
            raise

    # ===== 其他生成方法的占位符 =====

    async def _collect_dashboard_data(self, user_id: str, start_date: datetime, end_date: datetime, platforms: List[str]) -> Dict[str, Any]:
        """收集仪表板数据"""
        return {}

    async def _build_dashboard_content(self, dashboard_data: Dict[str, Any], date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """构建仪表板内容"""
        return {}

    async def _generate_dashboard_charts(self, dashboard_data: Dict[str, Any]) -> Dict[str, str]:
        """生成仪表板图表"""
        return {}

    async def _generate_dashboard_pdf(self, report_content: Dict[str, Any], chart_data: Dict[str, str], title: str) -> str:
        """生成仪表板PDF"""
        return ""

    async def _generate_dashboard_excel(self, report_content: Dict[str, Any], chart_data: Dict[str, str], title: str) -> str:
        """生成仪表板Excel"""
        return ""

    async def _create_dashboard_report_record(self, user_id: str, file_path: str, export_format: ExportFormat, date_range: Tuple[datetime, datetime]) -> GeneratedReport:
        """创建仪表板报告记录"""
        async with get_db() as db:
            report = GeneratedReport(
                title=f"仪表板报告_{datetime.now().strftime('%Y%m%d')}",
                user_id=user_id,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                file_format=export_format.value,
                generation_status='completed'
            )
            db.add(report)
            await db.commit()
            await db.refresh(report)
            return report

    # 其他方法的简化实现...
    async def _collect_custom_report_data(self, user_id: str, data_sources: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def _build_custom_content(self, report_data: Dict[str, Any], report_config: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    async def _generate_custom_charts(self, report_data: Dict[str, Any], chart_configs: List[Dict[str, Any]]) -> Dict[str, str]:
        return {}

    async def _generate_custom_pdf(self, report_content: Dict[str, Any], chart_data: Dict[str, str], title: str) -> str:
        return ""

    async def _generate_custom_excel(self, report_content: Dict[str, Any], chart_data: Dict[str, str], title: str) -> str:
        return ""

    async def _create_custom_report_record(self, user_id: str, file_path: str, export_format: ExportFormat, report_config: Dict[str, Any]) -> GeneratedReport:
        async with get_db() as db:
            report = GeneratedReport(
                title=report_config.get('title', '自定义报告'),
                user_id=user_id,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                file_format=export_format.value,
                generation_status='completed'
            )
            db.add(report)
            await db.commit()
            await db.refresh(report)
            return report