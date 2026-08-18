# SPEC_常见坑与解决方案_第11章

> 技术规格说明书 - 常见坑与解决方案

---

## 1. 章节概述

### 1.1 目标

定义常见问题的诊断方法和解决方案规范。

### 1.2 范围

- 问题分类体系
- 诊断方法
- 解决方案
- 预防措施

---

## 2. 问题分类体系

### 2.1 问题分类

| 类别 | 严重程度 | 影响范围 |
|-----|---------|---------|
| 检索质量 | P1 | 用户体验 |
| 系统性能 | P1 | 服务可用性 |
| 数据质量 | P2 | 内容准确性 |
| 配置错误 | P2 | 功能异常 |
| 资源问题 | P1 | 服务稳定性 |

### 2.2 问题清单

| 问题 ID | 问题描述 | 类别 | 严重程度 |
|--------|---------|-----|---------|
| P001 | 检索结果答非所问 | 检索质量 | P1 |
| P002 | 回答过时 | 数据质量 | P2 |
| P003 | 成本爆炸 | 资源问题 | P1 |
| P004 | 中文检索效果差 | 检索质量 | P2 |
| P005 | PDF 解析乱码 | 数据质量 | P2 |
| P006 | 多语言混杂 | 检索质量 | P2 |
| P007 | 检索速度慢 | 系统性能 | P1 |
| P008 | 回答幻觉 | 检索质量 | P1 |
| P009 | 数据导入失败 | 数据质量 | P2 |
| P010 | 生产环境崩溃 | 系统性能 | P1 |

---

## 3. 问题诊断规范

### 3.1 诊断流程

```
1. 问题报告
   ↓
2. 问题复现
   ↓
3. 日志分析
   ↓
4. 根因定位
   ↓
5. 解决方案制定
   ↓
6. 方案实施
   ↓
7. 效果验证
   ↓
8. 文档更新
```

### 3.2 诊断工具

```python
class DiagnosticTool:
    def __init__(self, kb):
        self.kb = kb
    
    def diagnose_query(self, query: str) -> dict:
        """诊断查询问题"""
        results = {}
        
        # 1. 检查检索结果
        retrieval_results = self.kb.retrieve(query)
        results["retrieval_count"] = len(retrieval_results)
        results["top_score"] = retrieval_results[0]["score"] if retrieval_results else 0
        
        # 2. 检查重排序
        if retrieval_results:
            reranked = self.kb.reranker.rerank(query, retrieval_results)
            results["rerank_improvement"] = reranked[0]["rerank_score"] - retrieval_results[0]["score"]
        
        # 3. 检查生成
        answer = self.kb.answer(query)
        results["answer_length"] = len(answer)
        results["has_citation"] = "[" in answer
        
        return results
    
    def diagnose_performance(self) -> dict:
        """诊断性能问题"""
        import time
        import psutil
        
        results = {}
        
        # 1. 内存使用
        results["memory_percent"] = psutil.virtual_memory().percent
        
        # 2. 延迟测试
        query = "测试查询"
        
        start = time.time()
        self.kb.retrieve(query)
        results["retrieval_latency_ms"] = (time.time() - start) * 1000
        
        start = time.time()
        self.kb.answer(query)
        results["total_latency_ms"] = (time.time() - start) * 1000
        
        return results
```

---

## 4. 解决方案规范

### 4.1 P001: 检索结果答非所问

**症状**：用户问"怎么退货"，返回的是"退货政策"全文，但没回答具体流程。

**诊断方法**：
```python
def diagnose_p001(query: str, kb):
    # 1. 检查分块大小
    chunk_sizes = [len(r["text"]) for r in kb.retrieve(query)]
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
    
    # 2. 检查重排序效果
    results = kb.retrieve(query, top_k=10)
    reranked = kb.reranker.rerank(query, results)
    
    return {
        "avg_chunk_size": avg_chunk_size,
        "rerank_improvement": reranked[0]["rerank_score"] - results[0]["score"]
    }
```

**解决方案**：
```python
# 1. 减小分块大小
config.pipeline.chunk_size = 256  # 从 512 减小到 256

# 2. 启用重排序
config.reranker.top_k = 3

# 3. Query 改写
def rewrite_query(query: str) -> str:
    if "退货" in query:
        return "退货流程 步骤"
    return query
```

### 4.2 P003: 成本爆炸

**症状**：Embedding 成本飙升，LLM 费用暴涨。

**诊断方法**：
```python
def diagnose_p003():
    import psutil
    
    return {
        "memory_percent": psutil.virtual_memory().percent,
        "cache_hit_rate": get_cache_hit_rate(),
        "avg_context_length": get_avg_context_length()
    }
```

**解决方案**：
```python
# 1. 实现缓存
embedding_cache = EmbeddingCache(max_size=10000)
semantic_cache = SemanticCache(threshold=0.95)

# 2. 限制上下文长度
config.generator.max_context_length = 4000

# 3. 批量处理
batch_processor = BatchProcessor(batch_size=100)
```

### 4.3 P007: 检索速度慢

**症状**：查询延迟超过 1 秒。

**诊断方法**：
```python
def diagnose_p007(query: str, kb):
    import time
    
    # 1. 向量化延迟
    start = time.time()
    kb.embedder.embed_single(query)
    embed_latency = (time.time() - start) * 1000
    
    # 2. 检索延迟
    start = time.time()
    kb.retrieve(query)
    retrieval_latency = (time.time() - start) * 1000
    
    # 3. 重排序延迟
    start = time.time()
    results = kb.retrieve(query)
    kb.reranker.rerank(query, results)
    rerank_latency = (time.time() - start) * 1000
    
    return {
        "embed_latency_ms": embed_latency,
        "retrieval_latency_ms": retrieval_latency,
        "rerank_latency_ms": rerank_latency
    }
```

**解决方案**：
```python
# 1. 优化索引
index_params = {
    "M": 32,
    "ef_construction": 400,
    "ef": 128
}

# 2. 实现缓存
semantic_cache = SemanticCache(threshold=0.95)

# 3. 分片检索
shard_manager = ShardManager()
```

### 4.4 P008: 回答幻觉

**症状**：LLM 编造了知识库里没有的信息。

**诊断方法**：
```python
def diagnose_p008(query: str, kb):
    # 1. 获取回答
    answer = kb.answer(query)
    
    # 2. 检查引用
    has_citation = "[" in answer
    
    # 3. 检查关键词匹配
    results = kb.retrieve(query)
    context = " ".join([r["text"] for r in results])
    
    keywords = extract_keywords(answer)
    in_context = sum(1 for k in keywords if k in context)
    
    return {
        "has_citation": has_citation,
        "keyword_match_rate": in_context / len(keywords) if keywords else 0
    }
```

**解决方案**：
```python
# 1. 强化 Prompt 约束
system_prompt = """规则：
1. 只根据提供的参考资料回答，绝对不要编造
2. 如果资料中没有相关信息，请直接说"我无法从现有资料中找到答案"
3. 回答时必须标注来源"""

# 2. 降低 Temperature
config.generator.temperature = 0

# 3. 加置信度判断
def check_confidence(answer: str, context: str) -> float:
    keywords = extract_keywords(answer)
    in_context = sum(1 for k in keywords if k in context)
    return in_context / len(keywords) if keywords else 0
```

---

## 5. 预防措施

### 5.1 代码审查清单

```python
PREVENTION_CHECKLIST = {
    "data_quality": [
        "分块大小是否合理？",
        "元数据是否完整？",
        "数据清洗是否完成？"
    ],
    "performance": [
        "是否实现缓存？",
        "批量处理是否启用？",
        "索引参数是否优化？"
    ],
    "reliability": [
        "异常处理是否完善？",
        "降级策略是否实现？",
        "监控是否配置？"
    ]
}
```

### 5.2 监控告警规则

```python
ALERT_RULES = {
    "high_latency": {
        "condition": "latency_p95 > 500ms",
        "severity": "warning",
        "action": "检查索引和缓存"
    },
    "low_accuracy": {
        "condition": "accuracy < 0.8",
        "severity": "critical",
        "action": "检查数据质量和检索策略"
    },
    "high_error_rate": {
        "condition": "error_rate > 0.01",
        "severity": "critical",
        "action": "检查系统日志"
    }
}
```

---

## 6. 测试用例

### 6.1 诊断工具测试

```python
def test_diagnostic_tool():
    kb = AgentKnowledgeBase()
    tool = DiagnosticTool(kb)
    
    results = tool.diagnose_query("怎么退货")
    
    assert "retrieval_count" in results
    assert "top_score" in results
```

### 6.2 解决方案验证

```python
def test_solution_p001():
    # 原始配置
    config1 = KBConfig()
    kb1 = AgentKnowledgeBase(config1)
    
    # 优化后配置
    config2 = KBConfig()
    config2.pipeline.chunk_size = 256
    kb2 = AgentKnowledgeBase(config2)
    
    # 对比效果
    results1 = kb1.retrieve("怎么退货")
    results2 = kb2.retrieve("怎么退货")
    
    # 验证优化效果
    assert len(results2) > 0
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
