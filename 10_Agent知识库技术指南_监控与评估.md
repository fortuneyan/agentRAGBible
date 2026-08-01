# 第 10 章：监控与评估

> 没有量化就没有优化。你不知道效果好不好，怎么改进？

---

## 关键指标

### 检索质量

| 指标 | 定义 | 目标 |
|-----|------|-----|
| Precision@K | top K 结果中相关文档的比例 | > 0.7 |
| Recall | 所有相关文档中被检索到的比例 | > 0.8 |
| MRR | 第一个相关文档的排名倒数 | > 0.5 |
| NDCG | 考虑排名位置的相关性得分 | > 0.6 |

### 系统性能

| 指标 | 定义 | 目标 |
|-----|------|-----|
| 延迟 | 从查询到返回的时间 | < 500ms |
| 吞吐量 | 每秒处理的查询数 | > 10 QPS |
| 准确率 | 回答正确的比例 | > 85% |

### 用户体验

| 指标 | 定义 | 目标 |
|-----|------|-----|
| 满意度 | 用户评分 | > 4.0/5.0 |
| 点击率 | 用户点击引用链接的比例 | > 30% |
| 追问率 | 用户继续追问的比例 | < 20% |

---

## 评估框架

### 测试数据集

```python
test_cases = [
    {
        "query": "怎么退货？",
        "expected_docs": ["退货流程.pdf"],           # 检索评估用：比对 metadata.source
        "expected_keywords": ["7天", "无理由", "上传照片"]  # 回答评估用：关键词命中
    },
    {
        "query": "会员积分怎么算？",
        "expected_docs": ["会员规则.pdf"],
        "expected_keywords": ["1元", "1积分"]
    }
]
```

### 自动评估

> ⚠️ **早期版本的两个坑**（UP-007 / UP-008a）：
> - `evaluate_retrieval` 把文件名 `expected_docs`（如 `"退货流程.pdf"`）拿去和正文 `doc["text"]` 做 `in` 匹配，
>   **永远命中不了**——应该比 `doc["metadata"]["source"]`。
> - `evaluate_answer` 用 `if expected in answer` 子串匹配，`"7天"` 是 `"27天"` 的子串，宽松到无意义。

```python
class KBEvaluator:
    def __init__(self, kb: AgentKnowledgeBase):
        self.kb = kb

    def evaluate_retrieval(self, test_cases: list[dict]) -> dict:
        """
        检索质量评估。expected_docs 是【来源文件名】，
        所以要比对 doc['metadata']['source']，而不是 doc['text']（见 UP-007）。
        """
        total = len(test_cases)
        if total == 0:
            return {"precision@k": 0.0, "mrr": 0.0, "total": 0}

        hits = 0
        mrr = 0.0

        for case in test_cases:
            query = case["query"]
            expected = case["expected_docs"]  # 例如 ["退货流程.pdf"]

            retrieved = self.kb.retrieve(query, top_k=10)
            # ✅ 比 source，不是比正文
            retrieved_sources = [
                doc.get("metadata", {}).get("source", "") for doc in retrieved
            ]

            for i, src in enumerate(retrieved_sources):
                # 用子串匹配应对 "退货流程.pdf" vs "/data/退货流程.pdf"
                if any(exp in src for exp in expected):
                    hits += 1
                    mrr += 1 / (i + 1)
                    break

        return {
            "precision@k": hits / total,
            "mrr": mrr / total,
            "total": total,
        }

    def evaluate_answer(self, test_cases: list[dict]) -> dict:
        """
        回答质量评估。
        简单子串匹配过宽（UP-008a）：这里改用【关键词集合命中】，
        生产环境建议进一步升级为 LLM-as-judge。
        """
        total = len(test_cases)
        if total == 0:
            return {"accuracy": 0.0, "total": 0}

        correct = 0
        for case in test_cases:
            query = case["query"]
            expected_keywords = case["expected_keywords"]  # 例 ["7天","无理由","上传照片"]

            answer = self.kb.answer(query)

            # 关键词命中比例，设阈值（如 0.6）判正确
            if not expected_keywords:
                continue
            hit = sum(1 for kw in expected_keywords if kw in answer)
            if hit / len(expected_keywords) >= 0.6:
                correct += 1

        return {
            "accuracy": correct / total,
            "total": total,
        }
```

---

## 日志记录

```python
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_kb")

class KBLogger:
    def __init__(self, log_file: str = "kb_logs.jsonl"):
        self.log_file = log_file
    
    def log_query(self, query: str, results: list[dict], latency: float):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results_count": len(results),
            "top_score": results[0]["score"] if results else 0,
            "latency_ms": latency
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Query: {query[:50]}... | Latency: {latency:.0f}ms | Results: {len(results)}")
    
    def log_feedback(self, query: str, answer: str, rating: int):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer[:200],
            "rating": rating
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
```

---

## 监控仪表盘

### Prometheus 指标

```python
# requires: prometheus-client>=0.17
import asyncio
import functools
import inspect
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
query_counter = Counter('kb_queries_total', 'Total queries')
query_latency = Histogram('kb_query_latency_seconds', 'Query latency')
active_sessions = Gauge('kb_active_sessions', 'Active sessions')

# 使用（修正点 UP-008b：加 functools.wraps，并自动适配 sync / async 函数）
def track_metrics(func):
    """
    统一装饰器：自动判断被装饰函数是 sync 还是 async。
    - 保留 __name__ / __doc__（functools.wraps）
    - async 函数用 await + async with 计时，否则计时器无法覆盖 await 期间
    """
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            query_counter.inc()
            with query_latency.time():
                return await func(*args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            query_counter.inc()
            with query_latency.time():
                return func(*args, **kwargs)
        return sync_wrapper

# 用法（sync 与 async 都能正确装饰）
@track_metrics
def answer(query: str) -> str: ...

@track_metrics
async def answer_async(query: str) -> str: ...

# CLI 工具调用指标（行动 Agent）
tool_calls_total = Counter('agent_tool_calls_total', 'Total tool calls', ['tool', 'status'])
tool_call_duration = Histogram('agent_tool_duration_seconds', 'Tool execution duration', ['tool'])
tool_call_errors = Counter('agent_tool_errors_total', 'Tool errors', ['tool', 'error_type'])

# 意图分布与人工介入
intent_distribution = Counter('agent_intent_distribution', 'Intent types', ['intent'])
human_intervention_total = Counter('agent_human_intervention_total', 'Human intervention triggers', ['reason'])
```

> 这些指标配合第 14 章的执行器与控制器使用：可观测"哪些工具被调用最多、失败率如何、哪些意图触发了人工确认"，是行动 Agent 上生产的必备监控。

### Grafana 看板

```json
{
  "panels": [
    {
      "title": "查询量",
      "type": "graph",
      "targets": [{"expr": "rate(kb_queries_total[5m])"}]
    },
    {
      "title": "延迟分布",
      "type": "heatmap",
      "targets": [{"expr": "histogram_quantile(0.95, kb_query_latency_seconds)"}]
    }
  ]
}
```

---

## A/B 测试

```python
class ABTest:
    def __init__(self, variant_a: AgentKnowledgeBase, variant_b: AgentKnowledgeBase):
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.results = {"a": [], "b": []}
    
    def route_query(self, query: str) -> str:
        # 随机分流
        import random
        variant = random.choice(["a", "b"])
        
        if variant == "a":
            answer = self.variant_a.answer(query)
        else:
            answer = self.variant_b.answer(query)
        
        self.results[variant].append({
            "query": query,
            "answer": answer
        })
        
        return answer
    
    def analyze(self) -> dict:
        # 简单分析（实际应用中需要更复杂的统计）
        return {
            "variant_a_count": len(self.results["a"]),
            "variant_b_count": len(self.results["b"])
        }
```

---

## 持续改进流程

```
收集用户反馈
    ↓
分析失败案例
    ↓
识别问题类型
    ↓
针对性优化
    ↓
A/B 测试验证
    ↓
上线新版本
```

### 失败案例分析

```python
def analyze_failures(log_file: str) -> dict:
    failures = {"no_result": 0, "wrong_answer": 0, "slow": 0}
    
    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)
            
            if entry.get("rating", 5) < 3:
                if entry.get("results_count", 0) == 0:
                    failures["no_result"] += 1
                elif entry.get("latency_ms", 0) > 1000:
                    failures["slow"] += 1
                else:
                    failures["wrong_answer"] += 1
    
    return failures
```

---

## 本章小结

- 评估指标要分层：检索质量、系统性能、用户体验
- 自动评估 + 人工评估结合
- 日志是改进的基础
- A/B 测试验证优化效果
- 持续改进是长期过程

下一章，我们讲常见坑与解决方案。

---

*监控不是为了好看，是为了发现问题。我之前没做监控，出了问题都不知道哪里错。*
