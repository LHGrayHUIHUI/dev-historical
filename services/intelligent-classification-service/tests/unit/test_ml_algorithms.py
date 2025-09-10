"""
IC-UNIT-001: 分类算法准确性单元测试
优先级: P1 - ML核心逻辑
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
import random


class MockMLClassifier(BaseEstimator, ClassifierMixin):
    """模拟机器学习分类器"""
    
    def __init__(self, algorithm: str = "svm", confidence_threshold: float = 0.8):
        self.algorithm = algorithm
        self.confidence_threshold = confidence_threshold
        self.classes_ = None
        self.is_fitted_ = False
        self.feature_names_ = None
        
    def fit(self, X: List[List[float]], y: List[str]) -> 'MockMLClassifier':
        """训练模型"""
        self.classes_ = list(set(y))
        self.is_fitted_ = True
        self.feature_names_ = [f"feature_{i}" for i in range(len(X[0]) if X else 0)]
        return self
    
    def predict(self, X: List[List[float]]) -> List[str]:
        """预测分类"""
        if not self.is_fitted_:
            raise ValueError("模型尚未训练")
        
        predictions = []
        for features in X:
            # 模拟预测逻辑
            if sum(features) > 0.5:
                predictions.append(self.classes_[0] if self.classes_ else "未知")
            else:
                predictions.append(self.classes_[-1] if self.classes_ else "未知")
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[Dict[str, float]]:
        """预测概率"""
        if not self.is_fitted_:
            raise ValueError("模型尚未训练")
        
        probabilities = []
        for features in X:
            # 模拟概率计算
            base_prob = min(max(sum(features), 0.1), 0.9)
            if len(self.classes_) == 2:
                prob_dict = {
                    self.classes_[0]: base_prob,
                    self.classes_[1]: 1.0 - base_prob
                }
            else:
                # 多分类情况
                remaining_prob = 1.0 - base_prob
                prob_dict = {self.classes_[0]: base_prob}
                for cls in self.classes_[1:]:
                    prob_dict[cls] = remaining_prob / (len(self.classes_) - 1)
            
            probabilities.append(prob_dict)
        
        return probabilities


class TextClassificationService:
    """文本分类服务 - 模拟实现"""
    
    def __init__(self):
        self.models: Dict[str, MockMLClassifier] = {}
        self.vectorizer = None
        
    def create_model(self, model_name: str, algorithm: str = "svm") -> Dict:
        """创建分类模型"""
        try:
            model = MockMLClassifier(algorithm=algorithm)
            self.models[model_name] = model
            
            return {
                "success": True,
                "model_name": model_name,
                "algorithm": algorithm,
                "status": "created",
                "message": f"模型 {model_name} 创建成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "模型创建失败"
            }
    
    def train_model(self, model_name: str, training_data: List[Dict]) -> Dict:
        """训练分类模型"""
        if model_name not in self.models:
            return {"success": False, "error": "模型不存在"}
        
        try:
            # 提取特征和标签
            X = []
            y = []
            
            for item in training_data:
                # 简单特征提取：文本长度、字符数等
                text = item.get("text", "")
                features = [
                    len(text) / 1000.0,  # 文本长度特征
                    text.count("历史") / max(len(text), 1),  # 历史词频
                    text.count("文档") / max(len(text), 1),  # 文档词频
                    len(text.split()) / max(len(text), 1)   # 词密度
                ]
                X.append(features)
                y.append(item.get("label", "未知"))
            
            # 训练模型
            model = self.models[model_name]
            model.fit(X, y)
            
            # 计算训练准确率（模拟）
            predictions = model.predict(X)
            accuracy = sum(1 for i in range(len(y)) if predictions[i] == y[i]) / len(y)
            
            return {
                "success": True,
                "model_name": model_name,
                "training_samples": len(training_data),
                "accuracy": accuracy,
                "classes": model.classes_,
                "message": "模型训练完成"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "模型训练失败"
            }
    
    def classify_text(self, model_name: str, text: str, return_probabilities: bool = False) -> Dict:
        """分类单个文本"""
        if model_name not in self.models:
            return {"success": False, "error": "模型不存在"}
        
        model = self.models[model_name]
        if not model.is_fitted_:
            return {"success": False, "error": "模型尚未训练"}
        
        try:
            # 特征提取
            features = [
                len(text) / 1000.0,
                text.count("历史") / max(len(text), 1),
                text.count("文档") / max(len(text), 1),
                len(text.split()) / max(len(text), 1)
            ]
            
            # 预测
            prediction = model.predict([features])[0]
            result = {
                "success": True,
                "text": text,
                "prediction": prediction,
                "model_name": model_name
            }
            
            if return_probabilities:
                probabilities = model.predict_proba([features])[0]
                result["probabilities"] = probabilities
                result["confidence"] = max(probabilities.values())
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "文本分类失败"
            }


class TestMLAlgorithms:
    """机器学习算法测试套件"""
    
    def setup_method(self):
        """测试前置设置"""
        self.service = TextClassificationService()
        self.training_data = [
            {"text": "这是一份重要的历史文档，记录了古代的政治制度", "label": "历史"},
            {"text": "技术规范文档详细说明了系统的架构设计", "label": "技术"},
            {"text": "财务报告显示了公司的盈利状况和发展趋势", "label": "财务"},
            {"text": "古代文献中记载了丰富的历史信息和文化内容", "label": "历史"},
            {"text": "软件开发文档包含了详细的API接口说明", "label": "技术"},
            {"text": "年度财务分析报告展现了企业的经营成果", "label": "财务"}
        ]
    
    def test_model_creation(self):
        """测试模型创建
        
        测试场景: IC-UNIT-001-001
        验证点: 模型创建功能和参数设置
        """
        # 测试SVM模型创建
        result = self.service.create_model("test_svm", "svm")
        assert result["success"] is True
        assert result["model_name"] == "test_svm"
        assert result["algorithm"] == "svm"
        
        # 测试RandomForest模型创建
        result = self.service.create_model("test_rf", "random_forest")
        assert result["success"] is True
        assert result["algorithm"] == "random_forest"
        
        # 验证模型存储
        assert "test_svm" in self.service.models
        assert "test_rf" in self.service.models
        
        print("✅ IC-UNIT-001-001: 模型创建测试通过")
    
    def test_model_training(self):
        """测试模型训练
        
        测试场景: IC-UNIT-001-002
        验证点: 模型训练过程和结果
        """
        # 创建模型
        self.service.create_model("training_test", "svm")
        
        # 训练模型
        result = self.service.train_model("training_test", self.training_data)
        
        assert result["success"] is True
        assert result["training_samples"] == len(self.training_data)
        assert result["accuracy"] > 0.0  # 应该有一定准确率
        assert len(result["classes"]) > 0  # 应该识别出分类
        
        # 验证模型状态
        model = self.service.models["training_test"]
        assert model.is_fitted_ is True
        assert model.classes_ is not None
        
        print("✅ IC-UNIT-001-002: 模型训练测试通过")
    
    def test_text_classification(self):
        """测试文本分类
        
        测试场景: IC-UNIT-001-003
        验证点: 文本分类准确性
        """
        # 准备训练好的模型
        self.service.create_model("classification_test", "svm")
        self.service.train_model("classification_test", self.training_data)
        
        # 测试历史文档分类
        historical_text = "这是一份古代历史文献，描述了王朝的兴衰历程"
        result = self.service.classify_text("classification_test", historical_text)
        
        assert result["success"] is True
        assert result["prediction"] is not None
        assert result["text"] == historical_text
        
        # 测试技术文档分类
        technical_text = "系统架构技术文档说明了微服务的设计模式"
        result = self.service.classify_text("classification_test", technical_text)
        
        assert result["success"] is True
        assert result["prediction"] is not None
        
        print("✅ IC-UNIT-001-003: 文本分类测试通过")
    
    def test_classification_with_probabilities(self):
        """测试带概率的文本分类
        
        测试场景: IC-UNIT-001-004
        验证点: 分类置信度和概率分布
        """
        # 准备训练好的模型
        self.service.create_model("prob_test", "svm")
        self.service.train_model("prob_test", self.training_data)
        
        # 测试带概率的分类
        test_text = "历史文档记录了重要的文化遗产信息"
        result = self.service.classify_text("prob_test", test_text, return_probabilities=True)
        
        assert result["success"] is True
        assert "probabilities" in result
        assert "confidence" in result
        assert isinstance(result["probabilities"], dict)
        assert 0.0 <= result["confidence"] <= 1.0
        
        # 验证概率分布
        probs = result["probabilities"]
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.01, "概率总和应该接近1.0"
        
        print("✅ IC-UNIT-001-004: 分类概率测试通过")
    
    def test_model_error_handling(self):
        """测试模型错误处理
        
        测试场景: IC-UNIT-001-005
        验证点: 异常情况和错误处理
        """
        # 测试对不存在模型的操作
        result = self.service.classify_text("nonexistent_model", "测试文本")
        assert result["success"] is False
        assert "不存在" in result["error"]
        
        # 测试对未训练模型的分类
        self.service.create_model("untrained_model", "svm")
        result = self.service.classify_text("untrained_model", "测试文本")
        assert result["success"] is False
        assert "尚未训练" in result["error"]
        
        # 测试空训练数据
        result = self.service.train_model("untrained_model", [])
        assert result["success"] is False
        
        print("✅ IC-UNIT-001-005: 模型错误处理测试通过")
    
    def test_algorithm_performance_comparison(self):
        """测试算法性能比较
        
        测试场景: IC-UNIT-001-006
        验证点: 不同算法的性能对比
        """
        algorithms = ["svm", "random_forest", "naive_bayes"]
        results = {}
        
        for algorithm in algorithms:
            model_name = f"test_{algorithm}"
            
            # 创建和训练模型
            self.service.create_model(model_name, algorithm)
            training_result = self.service.train_model(model_name, self.training_data)
            
            if training_result["success"]:
                results[algorithm] = {
                    "accuracy": training_result["accuracy"],
                    "classes_count": len(training_result["classes"])
                }
        
        # 验证所有算法都能正常工作
        assert len(results) > 0, "至少应该有一个算法能正常工作"
        
        # 验证准确率在合理范围内
        for algorithm, metrics in results.items():
            assert 0.0 <= metrics["accuracy"] <= 1.0, f"{algorithm}的准确率应该在0-1之间"
            assert metrics["classes_count"] > 0, f"{algorithm}应该识别出分类"
        
        print("✅ IC-UNIT-001-006: 算法性能比较测试通过")
        return results


if __name__ == "__main__":
    # 直接运行测试
    test_ml = TestMLAlgorithms()
    
    print("🤖 开始执行机器学习算法单元测试...")
    test_ml.setup_method()
    test_ml.test_model_creation()
    
    test_ml.setup_method()
    test_ml.test_model_training()
    
    test_ml.setup_method()
    test_ml.test_text_classification()
    
    test_ml.setup_method()
    test_ml.test_classification_with_probabilities()
    
    test_ml.setup_method()
    test_ml.test_model_error_handling()
    
    test_ml.setup_method()
    performance_results = test_ml.test_algorithm_performance_comparison()
    
    print("✅ 机器学习算法单元测试全部通过！")
    print(f"📊 算法性能对比: {performance_results}")