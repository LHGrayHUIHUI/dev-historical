#!/usr/bin/env python3
"""
智能分类服务独立测试脚本
测试服务的基本功能（不依赖storage-service）
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# 添加服务根目录到Python路径
script_dir = os.path.dirname(__file__)
service_root = os.path.join(script_dir, '../../services/intelligent-classification-service')
sys.path.insert(0, service_root)
sys.path.insert(0, os.path.join(service_root, 'src'))

class StandaloneTest:
    """独立测试类"""
    
    def __init__(self):
        self.test_results = {
            "start_time": datetime.now().isoformat(),
            "service": "intelligent-classification-service",
            "test_type": "standalone",
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": []
            }
        }
    
    def add_test_result(self, name: str, passed: bool, duration: float, details: dict = None, error: str = None):
        """添加测试结果"""
        self.test_results["tests"].append({
            "name": name,
            "status": "PASSED" if passed else "FAILED",
            "duration": duration,
            "details": details or {},
            "error": error
        })
        
        if passed:
            self.test_results["summary"]["passed"] += 1
        else:
            self.test_results["summary"]["failed"] += 1
            if error:
                self.test_results["summary"]["errors"].append(error)
        
        self.test_results["summary"]["total"] += 1
    
    def test_imports(self):
        """测试模块导入"""
        print("📦 测试模块导入...")
        start_time = time.time()
        
        try:
            # 测试配置模块
            from config.settings import settings
            assert settings.service_name == "intelligent-classification-service"
            print(f"  ✅ 配置模块导入成功: {settings.service_name}")
            
            # 测试schema模块
            from schemas.classification_schemas import ClassificationRequest, BaseResponse
            print("  ✅ Schema模块导入成功")
            
            # 测试工具模块
            from utils.text_preprocessing import ChineseTextPreprocessor
            from utils.feature_extraction import TfidfFeatureExtractor
            print("  ✅ 工具模块导入成功")
            
            # 测试服务模块
            from services.model_trainer import ModelTrainer
            from services.classification_service import ClassificationService
            print("  ✅ 服务模块导入成功")
            
            duration = time.time() - start_time
            self.add_test_result("模块导入测试", True, duration, {
                "modules_imported": ["config", "schemas", "utils", "services"]
            })
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"模块导入失败: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.add_test_result("模块导入测试", False, duration, error=error_msg)
    
    def test_text_preprocessing(self):
        """测试文本预处理功能"""
        print("🔤 测试文本预处理...")
        start_time = time.time()
        
        try:
            from utils.text_preprocessing import ChineseTextPreprocessor, PreprocessingConfig
            
            # 创建预处理器
            preprocessor = ChineseTextPreprocessor()
            
            # 测试文本
            test_text = "漢武帝時期，國力強盛，多次出征匈奴，建立了強大的漢帝國。"
            
            # 预处理测试
            tokens = preprocessor.preprocess(test_text, return_tokens=True)
            processed_text = preprocessor.preprocess(test_text, return_tokens=False)
            
            # 获取统计信息
            stats = preprocessor.get_text_statistics(test_text)
            
            print(f"  ✅ 原文本: {test_text}")
            print(f"  ✅ 处理后: {processed_text}")
            print(f"  ✅ 分词结果: {tokens}")
            print(f"  ✅ 统计信息: 原长度={stats['original_length']}, 词数={stats['token_count']}")
            
            duration = time.time() - start_time
            self.add_test_result("文本预处理测试", True, duration, {
                "original_text": test_text,
                "processed_text": processed_text,
                "token_count": len(tokens),
                "statistics": stats
            })
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"文本预处理失败: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.add_test_result("文本预处理测试", False, duration, error=error_msg)
    
    def test_feature_extraction(self):
        """测试特征提取功能"""
        print("🔍 测试特征提取...")
        start_time = time.time()
        
        try:
            from utils.feature_extraction import TfidfFeatureExtractor
            from utils.text_preprocessing import ChineseTextPreprocessor
            
            # 准备测试数据
            texts = [
                "汉武帝时期国力强盛",
                "唐朝诗歌艺术繁荣",
                "宋朝商业贸易发达",
                "明朝军事技术先进"
            ]
            
            # 创建TF-IDF提取器
            extractor = TfidfFeatureExtractor()
            
            # 训练和提取特征
            features = extractor.fit_transform(texts)
            
            print(f"  ✅ 训练文本数量: {len(texts)}")
            print(f"  ✅ 特征矩阵形状: {features.shape}")
            print(f"  ✅ 词汇表大小: {len(extractor.get_feature_names())}")
            
            # 获取重要词汇
            top_words = extractor.get_top_words(10)
            print(f"  ✅ 前10个重要词汇: {[word for word, score in top_words[:5]]}")
            
            duration = time.time() - start_time
            self.add_test_result("特征提取测试", True, duration, {
                "text_count": len(texts),
                "feature_shape": list(features.shape),
                "vocab_size": len(extractor.get_feature_names()),
                "top_words": top_words[:5]
            })
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"特征提取失败: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.add_test_result("特征提取测试", False, duration, error=error_msg)
    
    def test_model_trainer_initialization(self):
        """测试模型训练器初始化"""
        print("🤖 测试模型训练器...")
        start_time = time.time()
        
        try:
            from services.model_trainer import ModelTrainer, TrainingConfig
            from schemas.classification_schemas import ModelType, FeatureExtractorType
            
            # 创建训练配置
            config = TrainingConfig(
                test_size=0.3,
                cv_folds=3,
                random_state=42
            )
            
            # 创建训练器
            trainer = ModelTrainer(config)
            
            print(f"  ✅ 模型训练器初始化成功")
            print(f"  ✅ 配置测试集大小: {config.test_size}")
            print(f"  ✅ 交叉验证折数: {config.cv_folds}")
            
            # 测试模型注册表
            model_types = [ModelType.SVM, ModelType.RANDOM_FOREST]
            for model_type in model_types:
                if model_type in trainer._model_registry:
                    print(f"  ✅ 支持的模型类型: {model_type}")
                else:
                    print(f"  ⚠️  模型类型未注册: {model_type}")
            
            duration = time.time() - start_time
            self.add_test_result("模型训练器测试", True, duration, {
                "trainer_initialized": True,
                "test_size": config.test_size,
                "cv_folds": config.cv_folds,
                "supported_models": len(trainer._model_registry)
            })
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"模型训练器初始化失败: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.add_test_result("模型训练器测试", False, duration, error=error_msg)
    
    def test_classification_schemas(self):
        """测试分类数据模型"""
        print("📋 测试数据模型...")
        start_time = time.time()
        
        try:
            from schemas.classification_schemas import (
                ClassificationRequest, 
                BaseResponse,
                TrainingDataCreate,
                ModelTrainingRequest,
                ClassificationType
            )
            
            # 测试分类请求模型
            request_data = {
                "project_id": "test-project-123",
                "text_content": "这是一个测试文本",
                "return_probabilities": True,
                "return_explanation": True
            }
            
            request = ClassificationRequest(**request_data)
            print(f"  ✅ 分类请求模型创建成功: {request.project_id}")
            
            # 测试基础响应模型
            response_data = {
                "success": True,
                "message": "测试成功",
                "data": {"result": "test"}
            }
            
            response = BaseResponse(**response_data)
            print(f"  ✅ 基础响应模型创建成功: {response.message}")
            
            # 测试训练数据模型
            training_data = {
                "project_id": "test-project-123",
                "text_content": "训练文本示例",
                "true_label": "政治",
                "label_confidence": 1.0,
                "data_source": "test"
            }
            
            training_request = TrainingDataCreate(**training_data)
            print(f"  ✅ 训练数据模型创建成功: {training_request.true_label}")
            
            # 测试分类类型枚举
            classification_types = [t.value for t in ClassificationType]
            print(f"  ✅ 支持的分类类型: {classification_types}")
            
            duration = time.time() - start_time
            self.add_test_result("数据模型测试", True, duration, {
                "models_tested": ["ClassificationRequest", "BaseResponse", "TrainingDataCreate"],
                "classification_types": classification_types
            })
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"数据模型测试失败: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.add_test_result("数据模型测试", False, duration, error=error_msg)
    
    def test_configuration(self):
        """测试配置系统"""
        print("⚙️ 测试配置系统...")
        start_time = time.time()
        
        try:
            from config.settings import settings
            
            # 测试基本配置
            assert settings.service_name == "intelligent-classification-service"
            assert settings.api_port == 8007
            
            print(f"  ✅ 服务名称: {settings.service_name}")
            print(f"  ✅ 服务端口: {settings.api_port}")
            print(f"  ✅ 环境: {settings.environment}")
            
            # 测试预定义标签
            topic_labels = settings.get_classification_labels("topic")
            print(f"  ✅ 主题分类标签数量: {len(topic_labels)}")
            print(f"  ✅ 主题标签示例: {topic_labels[:3]}")
            
            # 测试模型配置
            rf_config = settings.get_model_config("random_forest")
            print(f"  ✅ 随机森林配置: n_estimators={rf_config.get('n_estimators')}")
            
            # 测试特征配置
            tfidf_config = settings.get_feature_config("tfidf")
            print(f"  ✅ TF-IDF配置: max_features={tfidf_config.get('max_features')}")
            
            duration = time.time() - start_time
            self.add_test_result("配置系统测试", True, duration, {
                "service_name": settings.service_name,
                "api_port": settings.api_port,
                "topic_labels_count": len(topic_labels),
                "has_model_config": bool(rf_config),
                "has_feature_config": bool(tfidf_config)
            })
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"配置系统测试失败: {str(e)}"
            print(f"  ❌ {error_msg}")
            self.add_test_result("配置系统测试", False, duration, error=error_msg)
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始智能分类服务独立测试...")
        print("=" * 60)
        
        # 运行各项测试
        self.test_imports()
        self.test_configuration()
        self.test_classification_schemas()
        self.test_text_preprocessing()
        self.test_feature_extraction()
        self.test_model_trainer_initialization()
        
        # 完成测试
        self.test_results["end_time"] = datetime.now().isoformat()
        self.test_results["total_duration"] = sum(
            test.get("duration", 0) for test in self.test_results["tests"]
        )
        
        # 计算成功率
        passed = self.test_results["summary"]["passed"]
        total = self.test_results["summary"]["total"]
        self.test_results["summary"]["success_rate"] = (passed / total * 100) if total > 0 else 0
        
        return self.test_results

async def main():
    """主函数"""
    print("🧪 智能分类服务独立功能测试")
    print("=" * 60)
    print("📝 说明: 此测试不依赖外部服务，仅测试核心功能模块")
    print()
    
    # 创建测试实例
    tester = StandaloneTest()
    
    # 运行测试
    results = await tester.run_all_tests()
    
    # 打印测试结果摘要
    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    print(f"🎯 总测试数: {results['summary']['total']}")
    print(f"✅ 通过: {results['summary']['passed']}")
    print(f"❌ 失败: {results['summary']['failed']}")
    print(f"📈 成功率: {results['summary']['success_rate']:.1f}%")
    print(f"⏱️  总耗时: {results['total_duration']:.2f}秒")
    
    if results['summary']['errors']:
        print(f"\n❗ 错误列表:")
        for i, error in enumerate(results['summary']['errors'], 1):
            print(f"  {i}. {error}")
    
    # 保存测试结果
    result_file = "standalone_test_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细测试结果已保存到: {result_file}")
    print("🏁 独立测试完成")
    
    return results

if __name__ == "__main__":
    # 切换到智能分类服务目录
    script_dir = Path(__file__).parent
    service_dir = script_dir.parent.parent / "services" / "intelligent-classification-service"
    
    if service_dir.exists():
        os.chdir(service_dir)
        print(f"📁 切换到服务目录: {service_dir}")
    else:
        # 如果在智能分类服务目录中
        if Path("src").exists() and Path("requirements.txt").exists():
            print(f"📁 当前目录: {os.getcwd()}")
        else:
            print("❌ 无法找到智能分类服务目录")
            sys.exit(1)
    
    asyncio.run(main())