# 第 6 章：重排序技术

> 检索回来的候选文档，直接喂给 LLM 效果一般。加个 Reranker 能提升 10-15%。

---

## 为什么需要 Reranker？

向量检索是"双塔模型"，query 和 document 分开编码，速度快但精度有限。

Reranker 是"交叉编码器"，query 和 document 一起编码，速度慢但精度高。

```
向量检索（召回）：query → embedding → 搜索 → top 10
重排序（精排）：(query, doc) → score → 排序 → top 3
```

---

## 模型选择

| 模型 | 类型 | 效果 | 推荐 |
|-----|------|-----|-----|
| bge-reranker-large | 交叉编码器 | ⭐⭐⭐⭐⭐ | 追求效果 |
| bge-reranker-v2-m3 | 交叉编码器 | ⭐⭐⭐⭐ | 多语言 |
| Cohere Rerank | API | ⭐⭐⭐⭐ | 不想自建 |
| LLM-based Rerank | LLM | ⭐⭐⭐ | 已有 LLM |

---

## 实现代码

### 基础实现

```python
from sentence_transformers import CrossEncoder

# 加载模型
reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank(query: str, documents: list[str], top_k: int = 3) -> list[str]:
    # 构造 query-document 对
    pairs = [(query, doc) for doc in documents]
    
    # 计算相关性得分
    scores = reranker.predict(pairs)
    
    # 按得分排序
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:top_k]]
```

### 带元数据的重排序

> ⚠️ 修正点（UP-004）：早期版本直接 `docs.sort(...)` 并原地写 `doc["rerank_score"]`，
> 会**破坏调用方传入的 list 及其元素**。生产代码应返回新列表、不改入参。

```python
import copy

def rerank_with_metadata(
    query: str,
    docs: list[dict],
    top_k: int = 3
) -> list[dict]:
    if not docs:
        return []

    texts = [doc["text"] for doc in docs]
    pairs = [(query, text) for text in texts]
    scores = reranker.predict(pairs)

    # 深拷贝后写分数 + 排序，绝不污染入参
    scored = []
    for doc, score in zip(docs, scores):
        new_doc = copy.deepcopy(doc)
        new_doc["rerank_score"] = float(score)
        scored.append(new_doc)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
```

---

## Cohere Rerank API

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

def rerank_cohere(query: str, documents: list[str], top_k: int = 3):
    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_k,
        model="rerank-english-v3.0"
    )
    
    return [documents[r.index] for r in results.results]
```

---

## LLM-based Rerank

如果没有专门的 Reranker 模型，可以用 LLM：

```python
def rerank_with_llm(query: str, documents: list[str], top_k: int = 3):
    prompt = f"""请对以下文档与查询的相关性打分（0-10分）。

查询：{query}

文档：
"""
    for i, doc in enumerate(documents):
        prompt += f"\n{i+1}. {doc}"
    
    prompt += "\n\n请输出每个文档的得分，格式为：1:分数, 2:分数, ..."
    
    response = call_llm(prompt)
    scores = parse_scores(response)
    
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:top_k]]
```

---

## 重排序策略

### 简单重排序

```python
# 直接用 Reranker 分数排序
results = rerank(query, candidates, top_k=3)
```

### 混合重排序

```python
def hybrid_rerank(
    query: str,
    vector_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 3
):
    # 先融合
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    
    # 取 top 20 候选
    candidate_ids = [doc_id for doc_id, _ in fused[:20]]
    candidates = [get_doc_by_id(doc_id) for doc_id in candidate_ids]
    
    # 重排序
    reranked = rerank(query, candidates, top_k=top_k)
    
    return reranked
```

---

## 性能优化

### 缓存 Reranker 结果

> ⚠️ 修正点（UP-105）：`lru_cache` 返回的是**同一个对象的引用**。
> 早期版本返回 `list[str]`（可变），调用方一旦修改就污染了缓存里后续所有命中。
> 下面返回 `tuple`（不可变）规避。

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_rerank(query: str, doc_ids: tuple) -> tuple[str, ...]:
    """
    返回 tuple 而非 list——lru_cache 会复用同一个返回对象，
    tuple 不可变，避免调用方修改污染缓存（UP-105）。
    """
    documents = [get_doc_by_id(doc_id) for doc_id in doc_ids]
    # rerank 返回 list，这里转成 tuple 再缓存
    return tuple(rerank(query, documents, top_k=3))

# 调用方需要 list 时再转一次（新的 list，不影响缓存）
result = list(cached_rerank(query, tuple(doc_ids)))
```

### 批量处理

```python
def batch_rerank(queries: list[str], documents_per_query: list[list[str]], top_k: int = 3):
    # 构造所有 pairs
    all_pairs = []
    for query, docs in zip(queries, documents_per_query):
        for doc in docs:
            all_pairs.append((query, doc))
    
    # 批量预测
    all_scores = reranker.predict(all_pairs)
    
    # 按 query 分组
    results = []
    idx = 0
    for query, docs in zip(queries, documents_per_query):
        scores = all_scores[idx:idx+len(docs)]
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        results.append([doc for doc, _ in scored_docs[:top_k]])
        idx += len(docs)
    
    return results
```

---

## 本章小结

- Reranker 能显著提升准确率
- 推荐用 bge-reranker-large
- Cohere Rerank 是省心的选择
- 缓存 + 批量处理能优化性能

下一章，我们讲上下文拼接与 LLM 生成。

---

*Reranker 不是必须的，但加上之后效果提升明显。我的建议是：先跑通，再优化。*
