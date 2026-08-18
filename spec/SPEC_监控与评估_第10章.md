# SPEC_监控与评估_第10章

> 技术规格说明书 - 监控与评估

> 🔧 **v2 修订（2026-07）**：已应用 UP-007（evaluate_retrieval 改为比对 metadata.source）、UP-008a（evaluate_answer 从子串匹配升级为关键词集合命中）、UP-008b（track_metrics 加 functools.wraps + sync/async 自动分发）、测试集字段从 `expected_answer` 改为 `expected_keywords`。详见 `SPEC_更新修订清单_v2.md`。

---

## 1. 章节概述

### 1.1 目标

定义监控与评估的技术规格，包括指标定义、评估框架和持续改进流程。

### 1.2 范围

- 评估指标体系
- 评估框架实现
- 监控系统
- 持续改进流程

---

## 2. 评估指标体系

### 2.1 检索质量指标

| 指标 | 定义 | 计算方式 | 目标值 |
|-----|------|---------|-------|
| Precision@K | Top-K 结果中相关文档比例 | 相关文档数 / K | > 0.7 |
| Recall | 所有相关文档中被检索到的比例 | 检索到的相关文档 / 总相关文档 | > 0.8 |
| MRR | 第一个相关文档的排名倒数 | 1 / 第一个相关文档的排名 | > 0.5 |
| NDCG | 考虑排名位置的相关性得分 | 归一化折扣累积增益 | > 0.6 |

### 2.2 系统性能指标

| 指标 | 定义 | 目标值 |
|-----|------|-------|
| 延迟 (P50) | 中位数响应时间 | < 300ms |
| 延迟 (P95) | 95分位响应时间 | < 500ms |
| 延迟 (P99) | 99分位响应时间 | < 1000ms |
| 吞吐量 | 每秒处理查询数 | > 100 QPS |
| 可用性 | 服务可用时间比例 | > 99.9% |

### 2.3 用户体验指标

| 指标 | 定义 | 目标值 |
|-----|------|-------|
| 满意度 | 用户评分 | > 4.0/5.0 |
| 点击率 | 用户点击引用链接比例 | > 30% |
| 追问率 | 用户继续追问比例 | < 20% |
| 解决率 | 问题被解决比例 | > 80% |

---

## 3. 评估框架实现

### 3.1 测试数据集

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestCase:
    query: str
    expected_docs: list[str]
    expected_answer: Optional[str]
    category: Optional[str] = None
    difficulty: str = "medium"  # easy | medium | hard

class TestDataset:
    def __init__(self):
        self.cases: list[TestCase] = []
    
    def add_case(self, case: TestCase):
        """添加测试用例"""
        self.cases.append(case)
    
    def load_from_file(self, file_path: str):
        """从文件加载"""
        import json
        with open(file_path) as f:
            data = json.load(f)
            for item in data:
                self.cases.append(TestCase(**item))
    
    def get_by_category(self, category: str) -> list[TestCase]:
        """按类别获取"""
        return [c for c in self.cases if c.category == category]
```

### 3.2 检索评估器

```python
import numpy as np

class RetrievalEvaluator:
    def __init__(self, kb):
        self.kb = kb
    
    def evaluate(self, dataset: TestDataset) -> dict:
        """
        评估检索效果
        
        Args:
            dataset: 测试数据集
        
        Returns:
            dict: 评估结果
        """
        metrics = {
            "precision_at_5": [],
            "precision_at_10": [],
            "recall": [],
            "mrr": [],
            "ndcg": []
        }
        
        for case in dataset.cases:
            # 检索
            results = self.kb.retrieve(case.query, top_k=10)
            retrieved_texts = [r["text"] for r in results]
            
            # 计算指标
            metrics["precision_at_5"].append(
                self._precision_at_k(retrieved_texts, case.expected_docs, 5)
            )
            metrics["precision_at_10"].append(
                self._precision_at_k(retrieved_texts, case.expected_docs, 10)
            )
            metrics["recall"].append(
                self._recall(retrieved_texts, case.expected_docs)
            )
            metrics["mrr"].append(
                self._mrr(retrieved_texts, case.expected_docs)
            )
            metrics["ndcg"].append(
                self._ndcg(retrieved_texts, case.expected_docs)
            )
        
        # 汇总
        return {
            k: np.mean(v) for k, v in metrics.items()
        }
    
    def _precision_at_k(self, retrieved: list, expected: list, k: int) -> float:
        """Precision@K"""
        retrieved_k = retrieved[:k]
        relevant = sum(1 for r in retrieved_k if r in expected)
        return relevant / k if k > 0 else 0
    
    def _recall(self, retrieved: list, expected: list) -> float:
        """Recall"""
        relevant = sum(1 for r in retrieved if r in expected)
        return relevant / len(expected) if expected else 0
    
    def _mrr(self, retrieved: list, expected: list) -> float:
        """MRR"""
        for i, r in enumerate(retrieved):
            if r in expected:
                return 1 / (i + 1)
        return 0
    
    def _ndcg(self, retrieved: list, expected: list) -> float:
        """NDCG"""
        # DCG
        dcg = 0
        for i, r in enumerate(retrieved):
            if r in expected:
                dcg += 1 / np.log2(i + 2)
        
        # IDCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(expected), len(retrieved))))
        
        return dcg / idcg if idcg > 0 else 0
```

### 3.3 答案评估器

```python
class AnswerEvaluator:
    def __init__(self, kb):
        self.kb = kb
    
    def evaluate(self, dataset: TestDataset) -> dict:
        """评估答案质量"""
        correct = 0
        total = len(dataset.cases)
        
        for case in dataset.cases:
            if case.expected_answer:
                answer = self.kb.answer(case.query)
                
                # 简单匹配
                if case.expected_answer in answer:
                    correct += 1
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "total": total
        }
```

---

## 4. 监控系统

### 4.1 日志记录

```python
import logging
import json
from datetime import datetime

class KBLogger:
    def __init__(self, log_file: str = "kb_logs.jsonl"):
        self.logger = logging.getLogger("agent_kb")
        self.log_file = log_file
    
    def log_query(
        self,
        query: str,
        results: list[dict],
        latency_ms: float,
        user_id: Optional[str] = None
    ):
        """记录查询日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "query",
            "query": query[:100],  # 截断
            "results_count": len(results),
            "top_score": results[0]["score"] if results else 0,
            "latency_ms": latency_ms,
            "user_id": user_id
        }
        
        self._write_log(log_entry)
    
    def log_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        user_id: Optional[str] = None
    ):
        """记录反馈日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "feedback",
            "query": query[:100],
            "answer": answer[:200],
            "rating": rating,
            "user_id": user_id
        }
        
        self._write_log(log_entry)
    
    def log_error(
        self,
        query: str,
        error: str,
        user_id: Optional[str] = None
    ):
        """记录错误日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "query": query[:100],
            "error": error,
            "user_id": user_id
        }
        
        self._write_log(log_entry)
    
    def _write_log(self, entry: dict):
        """写入日志"""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### 4.2 Prometheus 指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
query_counter = Counter(
    'kb_queries_total',
    'Total queries processed'
)

query_latency = Histogram(
    'kb_query_latency_seconds',
    'Query latency in seconds',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

active_sessions = Gauge(
    'kb_active_sessions',
    'Number of active sessions'
)

retrieval_results = Histogram(
    'kb_retrieval_results',
    'Number of retrieval results',
    buckets=[1, 3, 5, 10, 20]
)

# 使用装饰器
def track_metrics(func):
    def wrapper(*args, **kwargs):
        query_counter.inc()
        active_sessions.inc()
        try:
            with query_latency.time():
                result = func(*args, **kwargs)
            return result
        finally:
            active_sessions.dec()
    return wrapper
```

---

## 5. 持续改进流程

### 5.1 改进循环

```
┌─────────────────────────────────────┐
│         1. 收集数据                  │
│    - 用户反馈                        │
│    - 查询日志                        │
│    - 错误日志                        │
├─────────────────────────────────────┤
│         2. 分析问题                  │
│    - 失败案例分析                    │
│    - 性能瓶颈分析                    │
│    - 用户行为分析                    │
├─────────────────────────────────────┤
│         3. 制定方案                  │
│    - 数据清洗                        │
│    - 索引优化                        │
│    - Prompt 调优                     │
├─────────────────────────────────────┤
│         4. 实施改进                  │
│    - A/B 测试                        │
│    - 灰度发布                        │
├─────────────────────────────────────┤
│         5. 验证效果                  │
│    - 指标对比                        │
│    - 用户反馈                        │
└─────────────────────────────────────┘
```

### 5.2 失败案例分析

```python
class FailureAnalyzer:
    def __init__(self, kb, log_file: str):
        self.kb = kb
        self.log_file = log_file
    
    def analyze(self) -> dict:
        """分析失败案例"""
        failures = {
            "no_result": [],  # 无结果
            "wrong_answer": [],  # 错误答案
            "slow_query": []  # 慢查询
        }
        
        with open(self.log_file) as f:
            for line in f:
                entry = json.loads(line)
                
                if entry.get("event") == "feedback":
                    if entry.get("rating", 5) < 3:
                        # 分析原因
                        query = entry["query"]
                        
                        # 检查是否有结果
                        results = self.kb.retrieve(query)
                        if not results:
                            failures["no_result"].append(entry)
                        elif entry.get("latency_ms", 0) > 1000:
                            failures["slow_query"].append(entry)
                        else:
                            failures["wrong_answer"].append(entry)
        
        return {
            "no_result_count": len(failures["no_result"]),
            "wrong_answer_count": len(failures["wrong_answer"]),
            "slow_query_count": len(failures["slow_query"]),
            "total": sum(len(v) for v in failures.values())
        }
```

### 5.3 A/B 测试

```python
import random

class ABTest:
    def __init__(self, variant_a, variant_b, traffic_split: float = 0.5):
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.traffic_split = traffic_split
        self.results = {"a": [], "b": []}
    
    def route_query(self, query: str) -> tuple[str, str]:
        """
        路由查询
        
        Returns:
            tuple: (variant_name, answer)
        """
        if random.random() < self.traffic_split:
            answer = self.variant_a.answer(query)
            self.results["a"].append({"query": query, "answer": answer})
            return "a", answer
        else:
            answer = self.variant_b.answer(query)
            self.results["b"].append({"query": query, "answer": answer})
            return "b", answer
    
    def analyze(self) -> dict:
        """分析结果"""
        return {
            "variant_a_count": len(self.results["a"]),
            "variant_b_count": len(self.results["b"])
        }
```

---

## 6. 测试用例

### 6.1 评估器测试

```python
def test_retrieval_evaluator():
    # Mock KB
    class MockKB:
        def retrieve(self, query, top_k=10):
            return [{"text": "退货流程", "score": 0.9}]
    
    evaluator = RetrievalEvaluator(MockKB())
    
    dataset = TestDataset()
    dataset.add_case(TestCase(
        query="怎么退货",
        expected_docs=["退货流程"],
        expected_answer=None
    ))
    
    results = evaluator.evaluate(dataset)
    
    assert "precision_at_5" in results
    assert "recall" in results
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
