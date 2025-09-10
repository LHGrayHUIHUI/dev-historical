#!/usr/bin/env python3
"""
智能分类服务核心功能测试
简化版本，测试核心组件而不依赖复杂的模块结构
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 基础测试功能
def test_basic_imports():
    """测试基础模块导入"""
    try:
        import jieba
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        print("✅ 基础依赖库导入成功")
        return True, "所有基础依赖库导入成功"
    except Exception as e:
        print(f"❌ 基础依赖库导入失败: {e}")
        return False, str(e)

def test_jieba_functionality():
    """测试中文分词功能"""
    try:
        import jieba
        test_text = "汉武帝时期国力强盛，多次出征匈奴，建立了强大的汉帝国。"
        tokens = list(jieba.cut(test_text))
        
        print(f"✅ 中文分词成功")
        print(f"  📝 原文本: {test_text}")
        print(f"  🔤 分词结果: {tokens[:10]}...")  # 只显示前10个词
        print(f"  📊 词汇数量: {len(tokens)}")
        
        return True, {
            "original_text": test_text,
            "tokens": tokens,
            "token_count": len(tokens)
        }
    except Exception as e:
        print(f"❌ 中文分词测试失败: {e}")
        return False, str(e)

def test_tfidf_functionality():
    """测试TF-IDF特征提取"""
    try:
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        # 测试文档
        docs = [
            "汉武帝时期国力强盛",
            "唐朝诗歌艺术繁荣", 
            "宋朝商业贸易发达",
            "明朝军事技术先进"
        ]
        
        # 使用jieba分词
        processed_docs = []
        for doc in docs:
            tokens = list(jieba.cut(doc))
            processed_docs.append(" ".join(tokens))
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=100, token_pattern=r'(?u)\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        print("✅ TF-IDF特征提取成功")
        print(f"  📊 文档数量: {len(docs)}")
        print(f"  📈 特征矩阵形状: {tfidf_matrix.shape}")
        print(f"  🔤 特征词汇数: {len(feature_names)}")
        print(f"  📝 部分特征词: {list(feature_names[:10])}")
        
        return True, {
            "docs_count": len(docs),
            "feature_matrix_shape": tfidf_matrix.shape,
            "vocab_size": len(feature_names),
            "sample_features": list(feature_names[:10])
        }
    except Exception as e:
        print(f"❌ TF-IDF特征提取失败: {e}")
        return False, str(e)

def test_ml_classifier():
    """测试机器学习分类器"""
    try:
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # 模拟训练数据
        texts = [
            "汉武帝征战匈奴建立功勋",
            "唐太宗治理国家政绩显著", 
            "李白杜甫诗歌传世经典",
            "王维孟浩然山水诗优美",
            "宋朝商人贸易海外发达",
            "明朝郑和下西洋贸易",
            "春秋战国军事思想发展",
            "孙子兵法军事战略经典"
        ]
        
        labels = [
            "政治", "政治", "文学", "文学", 
            "经济", "经济", "军事", "军事"
        ]
        
        # 文本预处理
        processed_texts = []
        for text in texts:
            tokens = list(jieba.cut(text))
            processed_texts.append(" ".join(tokens))
        
        # 特征提取
        vectorizer = TfidfVectorizer(max_features=50, token_pattern=r'(?u)\b\w+\b')
        X = vectorizer.fit_transform(processed_texts).toarray()
        y = np.array(labels)
        
        # 训练分类器
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier.fit(X, y)
        
        # 预测测试
        test_text = "诗人创作优秀作品"
        test_tokens = list(jieba.cut(test_text))
        test_processed = " ".join(test_tokens)
        test_features = vectorizer.transform([test_processed])
        prediction = classifier.predict(test_features)[0]
        probabilities = classifier.predict_proba(test_features)[0]
        
        print("✅ 机器学习分类器测试成功")
        print(f"  📊 训练样本数: {len(texts)}")
        print(f"  🎯 分类类别: {list(set(labels))}")
        print(f"  🔮 测试文本: {test_text}")
        print(f"  📈 预测结果: {prediction}")
        print(f"  🎲 预测概率: {dict(zip(classifier.classes_, probabilities))}")
        
        return True, {
            "training_samples": len(texts),
            "classes": list(set(labels)),
            "test_text": test_text,
            "prediction": prediction,
            "probabilities": dict(zip(classifier.classes_, probabilities))
        }
    except Exception as e:
        print(f"❌ 机器学习分类器测试失败: {e}")
        return False, str(e)

def test_pydantic_models():
    """测试Pydantic数据模型"""
    try:
        from pydantic import BaseModel, Field
        from typing import Optional, List
        from datetime import datetime
        
        # 定义测试模型
        class TestClassificationRequest(BaseModel):
            project_id: str = Field(..., description="项目ID")
            text_content: str = Field(..., min_length=1, description="文本内容")
            return_probabilities: bool = Field(True, description="返回概率")
        
        class TestResponse(BaseModel):
            success: bool = Field(..., description="是否成功")
            prediction: str = Field(..., description="预测结果")
            probabilities: Optional[dict] = Field(None, description="概率分布")
            timestamp: datetime = Field(default_factory=datetime.now)
        
        # 测试模型创建
        request = TestClassificationRequest(
            project_id="test-project-001",
            text_content="这是一个测试文本内容",
            return_probabilities=True
        )
        
        response = TestResponse(
            success=True,
            prediction="测试分类",
            probabilities={"测试分类": 0.85, "其他": 0.15}
        )
        
        print("✅ Pydantic数据模型测试成功")
        print(f"  📝 请求模型: {request.model_dump()}")
        print(f"  📤 响应模型: {response.model_dump()}")
        
        return True, {
            "request_model": request.model_dump(),
            "response_model": response.model_dump()
        }
    except Exception as e:
        print(f"❌ Pydantic数据模型测试失败: {e}")
        return False, str(e)

def main():
    """主测试函数"""
    print("🧪 智能分类服务核心功能测试")
    print("=" * 60)
    print("📝 说明: 测试核心ML和NLP功能模块")
    print()
    
    test_results = {
        "start_time": datetime.now().isoformat(),
        "service": "intelligent-classification-service",
        "test_type": "core_functionality",
        "tests": [],
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
    }
    
    tests = [
        ("基础依赖导入", test_basic_imports),
        ("中文分词功能", test_jieba_functionality),
        ("TF-IDF特征提取", test_tfidf_functionality), 
        ("机器学习分类器", test_ml_classifier),
        ("Pydantic数据模型", test_pydantic_models)
    ]
    
    for test_name, test_func in tests:
        print(f"🔧 测试{test_name}...")
        start_time = time.time()
        
        try:
            success, details = test_func()
            duration = time.time() - start_time
            
            test_results["tests"].append({
                "name": test_name,
                "status": "PASSED" if success else "FAILED",
                "duration": duration,
                "details": details,
                "error": None if success else details
            })
            
            if success:
                test_results["summary"]["passed"] += 1
            else:
                test_results["summary"]["failed"] += 1
                test_results["summary"]["errors"].append(details)
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{test_name}异常: {str(e)}"
            print(f"❌ {error_msg}")
            
            test_results["tests"].append({
                "name": test_name,
                "status": "ERROR",
                "duration": duration,
                "details": {},
                "error": error_msg
            })
            
            test_results["summary"]["failed"] += 1
            test_results["summary"]["errors"].append(error_msg)
        
        test_results["summary"]["total"] += 1
        print()
    
    # 完成测试
    test_results["end_time"] = datetime.now().isoformat()
    test_results["total_duration"] = sum(test.get("duration", 0) for test in test_results["tests"])
    test_results["summary"]["success_rate"] = (
        test_results["summary"]["passed"] / test_results["summary"]["total"] * 100
        if test_results["summary"]["total"] > 0 else 0
    )
    
    # 打印摘要
    print("=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    print(f"🎯 总测试数: {test_results['summary']['total']}")
    print(f"✅ 通过: {test_results['summary']['passed']}")
    print(f"❌ 失败: {test_results['summary']['failed']}")
    print(f"📈 成功率: {test_results['summary']['success_rate']:.1f}%")
    print(f"⏱️  总耗时: {test_results['total_duration']:.2f}秒")
    
    if test_results['summary']['errors']:
        print(f"\n❗ 错误列表:")
        for i, error in enumerate(test_results['summary']['errors'], 1):
            print(f"  {i}. {error}")
    
    # 保存测试结果
    result_file = "core_functionality_test_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细测试结果已保存到: {result_file}")
    print("🏁 核心功能测试完成")
    
    return test_results

if __name__ == "__main__":
    results = main()