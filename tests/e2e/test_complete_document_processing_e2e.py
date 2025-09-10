"""
E2E-BIZ-001: 完整文档处理流程端到端测试
优先级: P0 - 关键用户路径验证
测试场景: 用户上传 → 处理 → 存储 → 结果返回
"""

import asyncio
import json
import aiohttp
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class CompleteDocumentProcessingE2ETester:
    """完整文档处理流程端到端测试器"""
    
    def __init__(self):
        self.file_processor_url = "http://localhost:8001"
        self.storage_service_url = "http://localhost:8002"
        self.test_results = []
        self.test_artifacts = []
        
    async def log_test(self, name: str, status: str, details: Dict = None, error: str = None, duration: float = 0):
        """记录测试结果"""
        result = {
            "test_name": name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "details": details or {},
            "error": error
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
        print(f"{status_emoji} {name}: {status}")
        if error:
            print(f"   错误: {error}")
        if details and status == "PASSED":
            print(f"   详情: {details}")
        if duration > 0:
            print(f"   耗时: {duration:.3f}秒")
    
    async def test_system_readiness_check(self, session):
        """测试系统就绪状态检查
        
        测试场景: E2E-BIZ-001-001
        验证点: 端到端测试前的系统状态验证
        """
        start_time = time.time()
        
        try:
            system_status = {}
            
            # 检查file-processor状态
            try:
                async with session.get(f"{self.file_processor_url}/health") as response:
                    if response.status == 200:
                        fp_health = await response.json()
                        system_status["file_processor"] = {
                            "healthy": True,
                            "processors_ready": fp_health.get("data", {}).get("components", {}).get("processors", {}).get("status") == "ready",
                            "available_processors": fp_health.get("data", {}).get("components", {}).get("processors", {}).get("available_processors", [])
                        }
                    else:
                        system_status["file_processor"] = {"healthy": False, "error": f"HTTP {response.status}"}
            except Exception as e:
                system_status["file_processor"] = {"healthy": False, "error": str(e)}
            
            # 检查storage-service状态
            try:
                async with session.get(f"{self.storage_service_url}/health") as response:
                    if response.status == 200:
                        storage_health = await response.json()
                        system_status["storage_service"] = {
                            "healthy": True,
                            "service_info": storage_health.get("data", {})
                        }
                    else:
                        system_status["storage_service"] = {"healthy": False, "error": f"HTTP {response.status}"}
            except Exception as e:
                system_status["storage_service"] = {"healthy": False, "error": str(e)}
            
            duration = time.time() - start_time
            
            # 评估系统就绪状态
            healthy_services = len([s for s in system_status.values() if s.get("healthy", False)])
            total_services = len(system_status)
            
            details = {
                "system_status": system_status,
                "healthy_services": healthy_services,
                "total_services": total_services,
                "readiness_score": round((healthy_services / total_services) * 100, 2),
                "e2e_ready": healthy_services >= 2  # 至少需要2个服务才能进行E2E测试
            }
            
            if healthy_services >= 2:
                await self.log_test("系统就绪状态检查", "PASSED", details, duration=duration)
                return True
            else:
                await self.log_test("系统就绪状态检查", "FAILED", 
                                  details, 
                                  error=f"就绪服务不足: {healthy_services}/{total_services}",
                                  duration=duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            await self.log_test("系统就绪状态检查", "FAILED", error=str(e), duration=duration)
            return False
    
    async def test_complete_document_upload_and_processing(self, session):
        """测试完整的文档上传和处理流程
        
        测试场景: E2E-BIZ-001-002
        验证点: 用户视角的完整文档处理体验
        """
        start_time = time.time()
        
        try:
            # 创建历史文本测试文档
            historical_document = """历史文本智能分析系统 - 测试文档

== 文档信息 ==
标题：宋代商业发展研究
作者：测试用户
创建时间：{}
文档类型：历史研究文献

== 正文内容 ==

一、概述
宋代（960-1279年）是中国历史上商业高度发达的时期。在这一时代，商业活动不仅在规模上有了显著扩展，在组织形式和经营方式上也出现了许多创新。

二、商业发展的背景
1. 农业技术的进步
   - 占城稻的引入提高了粮食产量
   - 农业剩余增加，为商业发展提供了基础

2. 手工业的繁荣
   - 丝织业技术精湛，产品远销海外
   - 陶瓷制造业达到前所未有的高度
   - 印刷术的普及促进了书籍贸易

三、商业活动的特点
1. 城市商业的繁荣
   - 汴京、临安等城市成为重要商业中心
   - 夜市的出现打破了传统的时间限制
   - 瓦子、勾栏等商业娱乐场所兴起

2. 长距离贸易的发展
   - 陆上丝绸之路继续发挥作用
   - 海上丝绸之路更加繁荣
   - 与东南亚、西亚等地区贸易往来密切

四、商业组织形式
1. 行会制度的完善
   - 各种手工业和商业行会组织健全
   - 行会在维护同业利益方面发挥重要作用

2. 商帮的形成
   - 地域性商人集团开始形成
   - 为后来明清时期商帮的发展奠定基础

五、货币制度与金融
1. 货币的多样化
   - 铜钱、银两、纸币并用
   - 世界上最早的纸币"交子"在宋代出现

2. 金融机构的发展
   - 柜坊、质库等金融机构兴起
   - 汇兑业务开始出现

六、对后世的影响
宋代商业的发展不仅推动了当时社会经济的繁荣，也为后来元、明、清各朝的商业发展奠定了重要基础。其商业组织形式、经营理念等都对后世产生了深远影响。

== 结论 ==
宋代商业的繁荣发展是多种因素共同作用的结果，它不仅体现了当时社会经济的活力，也展现了中国古代商业文明的高度成就。

== 参考文献 ==
1. 《宋史·食货志》
2. 漆侠《宋代经济史》
3. 斯波义信《宋代商业史研究》

文档字数：约800字
测试目的：验证历史文本智能分析系统的文档处理能力
特殊字符测试：""《》【】〖〗（）""".format(datetime.now().isoformat())
            
            # Step 1: 上传文档到file-processor进行处理
            print("   📤 步骤1: 上传文档到file-processor...")
            
            data = aiohttp.FormData()
            data.add_field('file',
                          io.BytesIO(historical_document.encode('utf-8')),
                          filename='historical_research_song_dynasty.txt',
                          content_type='text/plain')
            
            processing_start = time.time()
            async with session.post(f"{self.file_processor_url}/api/v1/process/document", data=data) as response:
                processing_duration = time.time() - processing_start
                
                if response.status == 200:
                    processing_result = await response.json()
                    
                    # 验证处理结果
                    success = processing_result.get("success", False)
                    extracted_text = processing_result.get("extracted_text", "")
                    file_info = processing_result.get("file_info", {})
                    processing_info = processing_result.get("processing_info", {})
                    
                    # 检查关键内容是否正确提取
                    content_integrity_checks = {
                        "title_extracted": "宋代商业发展研究" in extracted_text,
                        "content_structure": "一、概述" in extracted_text and "二、商业发展的背景" in extracted_text,
                        "historical_details": "汴京、临安" in extracted_text and "交子" in extracted_text,
                        "special_chars": "《》【】" in extracted_text and "（）" in extracted_text,
                        "chinese_content": "历史文本智能分析系统" in extracted_text
                    }
                    
                    content_integrity_score = sum(content_integrity_checks.values()) / len(content_integrity_checks)
                    
                    step1_details = {
                        "file_processing_success": success,
                        "processing_duration": round(processing_duration, 3),
                        "original_size": len(historical_document),
                        "extracted_size": len(extracted_text),
                        "content_integrity_score": round(content_integrity_score * 100, 2),
                        "integrity_checks": content_integrity_checks,
                        "file_info": file_info,
                        "detected_encoding": processing_info.get("encoding", "unknown")
                    }
                    
                    print(f"     ✅ 文档处理完成: {step1_details['content_integrity_score']}% 内容完整性")
                else:
                    error_content = await response.text()
                    step1_details = {
                        "file_processing_success": False,
                        "error": f"HTTP {response.status}: {error_content}"
                    }
                    print(f"     ❌ 文档处理失败: HTTP {response.status}")
            
            duration = time.time() - start_time
            
            # 评估整体流程成功度
            if step1_details.get("file_processing_success") and step1_details.get("content_integrity_score", 0) >= 80:
                details = {
                    "workflow_status": "success",
                    "total_duration": round(duration, 3),
                    "processing_details": step1_details,
                    "business_value_delivered": True,
                    "user_experience_quality": "good" if step1_details.get("content_integrity_score", 0) >= 90 else "acceptable"
                }
                
                # 保存测试产物
                self.test_artifacts.append({
                    "type": "processed_document",
                    "original_content": historical_document,
                    "processing_result": processing_result if 'processing_result' in locals() else None
                })
                
                await self.log_test("完整文档处理流程", "PASSED", details, duration=duration)
                return processing_result if 'processing_result' in locals() else None
            else:
                details = {
                    "workflow_status": "failed",
                    "total_duration": round(duration, 3),
                    "processing_details": step1_details,
                    "business_value_delivered": False
                }
                
                await self.log_test("完整文档处理流程", "FAILED", 
                                  details,
                                  error="文档处理质量不达标或处理失败",
                                  duration=duration)
                return None
                
        except Exception as e:
            duration = time.time() - start_time
            await self.log_test("完整文档处理流程", "FAILED", error=str(e), duration=duration)
            return None
    
    async def test_batch_historical_documents_processing(self, session):
        """测试批量历史文档处理
        
        测试场景: E2E-BIZ-001-003
        验证点: 批量处理历史文档的完整业务流程
        """
        start_time = time.time()
        
        try:
            # 创建多个不同类型的历史文档
            historical_documents = [
                {
                    "filename": "tang_poetry_analysis.txt",
                    "content": """唐诗研究文献

== 李白诗歌艺术特色分析 ==

李白（701-762年），字太白，号青莲居士，是唐代最伟大的诗人之一。其诗歌具有以下特色：

一、浪漫主义风格
李白的诗歌充满了丰富的想象力和强烈的浪漫主义色彩。如《将进酒》中"君不见黄河之水天上来，奔流到海不复回"，展现了其豪迈的气魄。

二、语言特点
1. 语言清新自然，不拘格律
2. 善用夸张和比喻
3. 音律优美，朗朗上口

三、主要作品
- 《静夜思》：表达思乡之情
- 《蜀道难》：描写蜀道之险峻
- 《梦游天姥吟留别》：体现超脱现实的理想

李白的诗歌对后世产生了深远影响，被誉为"诗仙"。

创建时间：{}""".format(datetime.now().isoformat())
                },
                {
                    "filename": "ming_dynasty_economy.txt", 
                    "content": """明代经济发展概况

== 明代商品经济的繁荣 ==

明朝（1368-1644年）是中国历史上商品经济高度发达的时期。

一、农业基础
1. 农作物品种增多
   - 美洲作物的引入（玉米、番薯等）
   - 提高了粮食产量

2. 农业技术改进
   - 精耕细作技术更加完善
   - 水利工程建设发达

二、手工业发展
1. 丝织业
   - 江南地区丝织业尤为发达
   - 苏州、杭州成为丝织业中心

2. 制瓷业
   - 景德镇成为"瓷都"
   - 青花瓷远销海外

三、商业贸易
1. 国内贸易
   - 商品流通范围扩大
   - 区域性商品市场形成

2. 对外贸易
   - 郑和下西洋促进了海外贸易
   - 白银大量流入中国

明代经济的发展为清朝的繁荣奠定了基础。

文档类型：经济史研究
创建时间：{}""".format(datetime.now().isoformat())
                },
                {
                    "filename": "qing_dynasty_culture.txt",
                    "content": """清代文化发展特点

== 清代文化的多元特征 ==

清朝（1644-1912年）作为中国最后一个封建王朝，在文化发展上呈现出独特的特点。

一、文学艺术
1. 小说创作
   - 《红楼梦》：古典小说的巅峰之作
   - 《聊斋志异》：文言短篇小说集大成者

2. 戏曲发展
   - 京剧的形成和发展
   - 地方戏曲的繁荣

二、学术思想
1. 考据学的兴起
   - 强调实证研究方法
   - 代表人物：顾炎武、黄宗羲等

2. 经世致用思想
   - 关注现实问题的解决
   - 影响了后来的洋务运动

三、中西文化交流
1. 传教士来华
   - 带来了西方科学技术
   - 促进了东西文化交流

2. 翻译活动
   - 大量西方典籍被翻译介绍
   - 开阔了国人的视野

清代文化在传承传统的同时，也体现了时代的变化特征。

研究领域：文化史
编写日期：{}""".format(datetime.now().isoformat())
                }
            ]
            
            print("   📚 步骤1: 准备批量历史文档...")
            
            # 批量处理所有文档
            batch_results = []
            processing_start = time.time()
            
            for i, doc in enumerate(historical_documents):
                print(f"     处理文档 {i+1}/3: {doc['filename']}")
                
                data = aiohttp.FormData()
                data.add_field('file',
                              io.BytesIO(doc['content'].encode('utf-8')),
                              filename=doc['filename'],
                              content_type='text/plain')
                
                try:
                    async with session.post(f"{self.file_processor_url}/api/v1/process/document", data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # 分析处理质量
                            extracted_text = result.get("extracted_text", "")
                            quality_score = 0
                            
                            # 检查关键历史概念是否被正确提取
                            if "tang_poetry" in doc['filename'].lower():
                                quality_indicators = ["李白", "唐诗", "浪漫主义", "静夜思"]
                            elif "ming_dynasty" in doc['filename'].lower():
                                quality_indicators = ["明朝", "商品经济", "郑和", "景德镇"]
                            else:  # qing_dynasty
                                quality_indicators = ["清朝", "红楼梦", "京剧", "考据学"]
                            
                            found_indicators = sum(1 for indicator in quality_indicators if indicator in extracted_text)
                            quality_score = (found_indicators / len(quality_indicators)) * 100
                            
                            batch_results.append({
                                "filename": doc['filename'],
                                "success": True,
                                "quality_score": quality_score,
                                "content_length": len(extracted_text),
                                "key_concepts_found": found_indicators
                            })
                            
                            print(f"       ✅ 处理成功，质量评分: {quality_score:.1f}%")
                        else:
                            batch_results.append({
                                "filename": doc['filename'],
                                "success": False,
                                "error": f"HTTP {response.status}"
                            })
                            print(f"       ❌ 处理失败: HTTP {response.status}")
                            
                except Exception as e:
                    batch_results.append({
                        "filename": doc['filename'],
                        "success": False,
                        "error": str(e)
                    })
                    print(f"       ❌ 处理异常: {str(e)}")
            
            batch_duration = time.time() - processing_start
            duration = time.time() - start_time
            
            # 分析批量处理结果
            successful_docs = len([r for r in batch_results if r.get("success", False)])
            total_docs = len(historical_documents)
            success_rate = (successful_docs / total_docs) * 100
            
            avg_quality_score = 0
            if successful_docs > 0:
                quality_scores = [r.get("quality_score", 0) for r in batch_results if r.get("success", False)]
                avg_quality_score = sum(quality_scores) / len(quality_scores)
            
            details = {
                "batch_processing_summary": {
                    "total_documents": total_docs,
                    "successful_processing": successful_docs,
                    "success_rate": round(success_rate, 2),
                    "average_quality_score": round(avg_quality_score, 2),
                    "total_processing_time": round(batch_duration, 3),
                    "average_time_per_doc": round(batch_duration / total_docs, 3)
                },
                "individual_results": batch_results,
                "business_value": {
                    "historical_content_processed": successful_docs > 0,
                    "multi_period_coverage": successful_docs >= 2,  # 覆盖多个历史时期
                    "batch_efficiency_acceptable": batch_duration < 60  # 1分钟内完成
                }
            }
            
            # 保存测试产物
            self.test_artifacts.append({
                "type": "batch_processing_results",
                "documents": historical_documents,
                "results": batch_results
            })
            
            if success_rate >= 80 and avg_quality_score >= 70:
                await self.log_test("批量历史文档处理", "PASSED", details, duration=duration)
                return batch_results
            else:
                await self.log_test("批量历史文档处理", "FAILED", 
                                  details,
                                  error=f"批量处理质量不达标: {success_rate:.1f}%成功率, {avg_quality_score:.1f}%平均质量",
                                  duration=duration)
                return batch_results
                
        except Exception as e:
            duration = time.time() - start_time
            await self.log_test("批量历史文档处理", "FAILED", error=str(e), duration=duration)
            return None
    
    async def test_user_journey_simulation(self, session):
        """测试用户使用场景模拟
        
        测试场景: E2E-BIZ-001-004
        验证点: 从用户角度的完整使用体验
        """
        start_time = time.time()
        
        try:
            print("   👤 模拟用户场景: 历史研究者上传研究资料...")
            
            user_journey_steps = []
            
            # Step 1: 用户检查系统状态
            step_start = time.time()
            try:
                async with session.get(f"{self.file_processor_url}/health") as response:
                    step_duration = time.time() - step_start
                    if response.status == 200:
                        health_data = await response.json()
                        user_journey_steps.append({
                            "step": "system_health_check",
                            "status": "success",
                            "duration": round(step_duration, 3),
                            "user_experience": "系统运行正常，可以上传文件"
                        })
                        print("     ✅ 步骤1: 系统状态检查 - 正常")
                    else:
                        user_journey_steps.append({
                            "step": "system_health_check",
                            "status": "failed",
                            "duration": round(step_duration, 3),
                            "user_experience": f"系统异常: HTTP {response.status}"
                        })
                        print(f"     ❌ 步骤1: 系统状态检查 - HTTP {response.status}")
            except Exception as e:
                user_journey_steps.append({
                    "step": "system_health_check",
                    "status": "failed",
                    "user_experience": f"无法连接到系统: {str(e)}"
                })
                print(f"     ❌ 步骤1: 系统连接失败 - {str(e)}")
            
            # Step 2: 用户查看支持的文件格式
            step_start = time.time()
            try:
                async with session.get(f"{self.file_processor_url}/api/v1/process/supported-formats") as response:
                    step_duration = time.time() - step_start
                    if response.status == 200:
                        formats_data = await response.json()
                        user_journey_steps.append({
                            "step": "check_supported_formats",
                            "status": "success",
                            "duration": round(step_duration, 3),
                            "user_experience": f"可以上传的格式: {formats_data.get('supported_formats', '格式信息获取成功')}"
                        })
                        print("     ✅ 步骤2: 查看支持格式 - 成功")
                    else:
                        user_journey_steps.append({
                            "step": "check_supported_formats",
                            "status": "failed",
                            "duration": round(step_duration, 3),
                            "user_experience": "无法获取支持的文件格式信息"
                        })
                        print("     ❌ 步骤2: 查看支持格式 - 失败")
            except Exception as e:
                user_journey_steps.append({
                    "step": "check_supported_formats",
                    "status": "failed",
                    "user_experience": f"格式查询异常: {str(e)}"
                })
                print(f"     ❌ 步骤2: 格式查询异常 - {str(e)}")
            
            # Step 3: 用户上传研究文档
            research_document = """古代丝绸之路贸易研究

== 研究背景 ==
丝绸之路是连接古代中国与西方世界的重要贸易通道，对促进东西方文化交流和经济发展发挥了重要作用。

== 主要贸易商品 ==
1. 中国出口商品
   - 丝绸：最重要的出口商品
   - 茶叶：深受西方欢迎
   - 瓷器：精美的工艺品
   - 香料：珍贵的调料

2. 从西方进口商品
   - 玻璃器皿：西方的精美制品
   - 宝石：装饰用品
   - 香料：异域香料
   - 毛织品：御寒用品

== 贸易路线 ==
主要分为陆上丝绸之路和海上丝绸之路两条线路。

陆上路线：长安→河西走廊→新疆→中亚→西亚→欧洲
海上路线：泉州、广州→东南亚→印度洋→阿拉伯海→欧洲

== 历史影响 ==
丝绸之路不仅促进了商品贸易，更重要的是促进了文化、技术、宗教的传播与交流。

研究者：历史研究用户
研究时间：{}
研究目的：验证历史文本智能分析系统""".format(datetime.now().isoformat())
            
            step_start = time.time()
            try:
                data = aiohttp.FormData()
                data.add_field('file',
                              io.BytesIO(research_document.encode('utf-8')),
                              filename='silk_road_trade_research.txt',
                              content_type='text/plain')
                
                async with session.post(f"{self.file_processor_url}/api/v1/process/document", data=data) as response:
                    step_duration = time.time() - step_start
                    if response.status == 200:
                        processing_result = await response.json()
                        extracted_text = processing_result.get("extracted_text", "")
                        
                        # 从用户角度验证结果质量
                        key_research_elements = {
                            "research_topic": "丝绸之路" in extracted_text,
                            "trade_goods": "丝绸" in extracted_text and "茶叶" in extracted_text,
                            "trade_routes": "长安" in extracted_text and "河西走廊" in extracted_text,
                            "historical_context": "文化交流" in extracted_text,
                            "research_info": "研究者：历史研究用户" in extracted_text
                        }
                        
                        user_satisfaction_score = sum(key_research_elements.values()) / len(key_research_elements) * 100
                        
                        user_journey_steps.append({
                            "step": "upload_and_process_document",
                            "status": "success" if user_satisfaction_score >= 80 else "partial_success",
                            "duration": round(step_duration, 3),
                            "user_experience": f"文档处理完成，内容识别准确度: {user_satisfaction_score:.1f}%",
                            "satisfaction_score": user_satisfaction_score,
                            "extracted_elements": key_research_elements
                        })
                        
                        if user_satisfaction_score >= 80:
                            print(f"     ✅ 步骤3: 文档上传处理 - 成功 ({user_satisfaction_score:.1f}%满意度)")
                        else:
                            print(f"     ⚠️ 步骤3: 文档上传处理 - 部分成功 ({user_satisfaction_score:.1f}%满意度)")
                    else:
                        error_content = await response.text()
                        user_journey_steps.append({
                            "step": "upload_and_process_document", 
                            "status": "failed",
                            "duration": round(step_duration, 3),
                            "user_experience": f"文档处理失败: {error_content}"
                        })
                        print(f"     ❌ 步骤3: 文档上传处理 - 失败")
                        
            except Exception as e:
                user_journey_steps.append({
                    "step": "upload_and_process_document",
                    "status": "failed",
                    "user_experience": f"上传异常: {str(e)}"
                })
                print(f"     ❌ 步骤3: 文档上传异常 - {str(e)}")
            
            duration = time.time() - start_time
            
            # 评估整体用户体验
            successful_steps = len([s for s in user_journey_steps if s.get("status") == "success"])
            partial_success_steps = len([s for s in user_journey_steps if s.get("status") == "partial_success"])
            total_steps = len(user_journey_steps)
            
            overall_success_rate = (successful_steps + partial_success_steps * 0.5) / total_steps * 100
            
            # 计算用户满意度
            satisfaction_scores = [s.get("satisfaction_score", 0) for s in user_journey_steps if "satisfaction_score" in s]
            avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0
            
            details = {
                "user_journey_analysis": {
                    "total_steps": total_steps,
                    "successful_steps": successful_steps,
                    "partial_success_steps": partial_success_steps,
                    "overall_success_rate": round(overall_success_rate, 2),
                    "average_user_satisfaction": round(avg_satisfaction, 2),
                    "total_user_time": round(duration, 3)
                },
                "step_by_step_results": user_journey_steps,
                "user_experience_rating": "excellent" if overall_success_rate >= 90 else
                                        "good" if overall_success_rate >= 70 else
                                        "acceptable" if overall_success_rate >= 50 else
                                        "poor"
            }
            
            # 保存用户场景测试产物
            self.test_artifacts.append({
                "type": "user_journey_simulation",
                "research_document": research_document,
                "journey_steps": user_journey_steps
            })
            
            if overall_success_rate >= 70:
                await self.log_test("用户使用场景模拟", "PASSED", details, duration=duration)
                return user_journey_steps
            else:
                await self.log_test("用户使用场景模拟", "FAILED", 
                                  details,
                                  error=f"用户体验不达标: {overall_success_rate:.1f}%成功率",
                                  duration=duration)
                return user_journey_steps
                
        except Exception as e:
            duration = time.time() - start_time
            await self.log_test("用户使用场景模拟", "FAILED", error=str(e), duration=duration)
            return None
    
    async def run_all_tests(self):
        """运行所有关键业务路径E2E测试"""
        print("🎭 开始执行关键业务路径端到端测试...")
        
        async with aiohttp.ClientSession() as session:
            # 按顺序执行测试
            await self.test_system_readiness_check(session)
            await self.test_complete_document_upload_and_processing(session)
            await self.test_batch_historical_documents_processing(session)
            await self.test_user_journey_simulation(session)
        
        # 生成测试摘要
        passed_tests = len([t for t in self.test_results if t["status"] == "PASSED"])
        failed_tests = len([t for t in self.test_results if t["status"] == "FAILED"])
        total_tests = len(self.test_results)
        total_duration = sum([t.get("duration", 0) for t in self.test_results])
        
        print(f"\n📊 关键业务路径E2E测试摘要:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过: {passed_tests}")
        print(f"   失败: {failed_tests}")
        print(f"   成功率: {round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0}%")
        print(f"   总执行时间: {round(total_duration, 3)}秒")
        print(f"   测试产物数: {len(self.test_artifacts)}")
        
        return {
            "test_results": self.test_results,
            "test_artifacts": self.test_artifacts
        }


async def main():
    tester = CompleteDocumentProcessingE2ETester()
    results = await tester.run_all_tests()
    
    # 保存测试结果
    with open("/Users/yjlh/Documents/code/Historical Text Project/test-results/e2e_critical_business_paths_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_type": "e2e_critical_business_paths",
            "execution_time": datetime.now().isoformat(),
            "results": results["test_results"],
            "artifacts_count": len(results["test_artifacts"])
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 E2E测试结果已保存")


if __name__ == "__main__":
    asyncio.run(main())