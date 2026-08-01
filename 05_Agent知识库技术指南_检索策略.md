# 第 5 章：检索策略

> 光有向量检索不够，我实测下来，纯向量检索的准确率大概 60-70%。加上混合检索能提到 80%+。

---

## 向量检索

### 基本原理

把问题和文档都转成向量，计算相似度，找最接近的。

```python
# 简单实现
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def vector_search(query_embedding, doc_embeddings, top_k=5):
    similarities = [cosine_similarity(query_embedding, doc) for doc in doc_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices
```

### 索引类型

| 索引类型 | 原理 | 适用场景 |
|---------|------|---------|
| HNSW | 图搜索 | 通用，推荐 |
| IVF | 倒排索引 | 大规模数据 |
| PQ | 量化压缩 | 内存受限 |
| FLAT | 暴力搜索 | 小规模，精确 |

```python
# HNSW 参数
index_params = {
    "M": 16,  # 每个节点的连接数
    "efConstruction": 200  # 构建时的搜索范围
}

# 检索时参数
search_params = {
    "ef": 64  # 检索时的搜索范围
}
```

---

## 关键词检索（BM25）

### 基本原理

基于词频和文档频率的统计方法。

```python
from rank_bm25 import BM25Okapi
import jieba

# 分词
tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# 检索
query_tokens = list(jieba.cut("怎么退货"))
scores = bm25.get_scores(query_tokens)
top_indices = np.argsort(scores)[-5:][::-1]
```

### 优点

- 不需要向量化
- 对精确匹配效果好
- 可解释性强

### 缺点

- 不理解语义
- 对同义词效果差
- 依赖分词质量

---

## 混合检索

### 为什么要混合？

- 向量检索：理解语义，但可能漏掉精确匹配
- BM25：精确匹配好，但不理解语义
- 混合：取长补短

### 融合策略

#### 方法一：加权融合

```python
def weighted_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> list[dict]:
    # 归一化分数
    vector_max = max(r["score"] for r in vector_results)
    bm25_max = max(r["score"] for r in bm25_results)
    
    scores = {}
    for r in vector_results:
        doc_id = r["id"]
        scores[doc_id] = vector_weight * (r["score"] / vector_max)
    
    for r in bm25_results:
        doc_id = r["id"]
        if doc_id in scores:
            scores[doc_id] += bm25_weight * (r["score"] / bm25_max)
        else:
            scores[doc_id] = bm25_weight * (r["score"] / bm25_max)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### 方法二：RRF（Reciprocal Rank Fusion）

```python
def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    fused_scores = {}
    
    for rank, r in enumerate(vector_results):
        doc_id = r["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    for rank, r in enumerate(bm25_results):
        doc_id = r["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

---

## Query 改写

用户问的问题往往很模糊，直接拿去检索效果很差。

### 方法一：简单规则改写

```python
def rewrite_query(query: str) -> str:
    # 去掉口语化表达
    query = query.replace("怎么", "如何")
    query = query.replace("咋", "如何")
    query = query.replace("啥", "什么")
    
    # 补充关键词
    if "退货" in query and "流程" not in query:
        query += " 流程"
    
    return query
```

### 方法二：用 LLM 改写

```python
def rewrite_query_with_llm(query: str) -> list[str]:
    prompt = f"""请将以下用户问题改写成3个更适合搜索引擎检索的查询语句。

原始问题：{query}

输出格式（每行一个）：
1. ...
2. ...
3. ..."""
    
    response = call_llm(prompt)
    return parse_queries(response)
```

### 方法三：HyDE（假设文档嵌入）

```python
def hyde_rewrite(query: str) -> str:
    """让 LLM 先生成一个假设的答案，用这个答案去检索"""
    prompt = f"""请回答以下问题，假设你正在编写一份内部文档的答案：

问题：{query}

答案："""
    
    hypothetical_answer = call_llm(prompt)
    return hypothetical_answer
```

**HyDE 的原理**：

用户问题 → LLM 生成假设答案 → 用假设答案检索 → 找到真实文档

因为假设答案和真实文档更接近，所以检索效果更好。

---

## 检索过滤

### 元数据过滤

```python
# 只检索特定来源的文档
results = vector_db.search(
    query_vector=query_embedding,
    filter={
        "must": [
            {"key": "doc_type", "match": {"value": "policy"}},
            {"key": "last_updated", "range": {"gte": "2024-01-01"}}
        ]
    },
    limit=5
)
```

### 时间衰减

```python
import math
from datetime import datetime

def time_decay_score(score: float, doc_time: datetime, half_life_days: int = 30) -> float:
    days_ago = (datetime.now() - doc_time).days
    decay = math.exp(-0.693 * days_ago / half_life_days)
    return score * decay
```

---

## 多路召回

```python
def multi_recall(query: str) -> list[dict]:
    # 路径1：向量检索
    vector_results = vector_search(query_embedding, top_k=10)
    
    # 路径2：BM25
    bm25_results = bm25_search(query, top_k=10)
    
    # 路径3：知识图谱（如果有）
    kg_results = knowledge_graph_search(query, top_k=5)
    
    # 融合
    all_results = vector_results + bm25_results + kg_results
    fused = reciprocal_rank_fusion(all_results)
    
    return fused[:10]
```

---

## 意图分类与工具路由

当系统同时支持"知识问答"和"工具操作"两类能力时，检索前要先判断用户**想要什么**，再决定走哪条链路。

**意图分类器**（在检索之前执行）：

```python
class IntentClassifier:
    def classify(self, query: str) -> dict:
        # 返回: {"intent": "knowledge_qa" | "tool_query" | "tool_execute" | "multi_step", ...}
        ...
```

**路由逻辑**：

| 意图类型 | 路由目标 | 说明 |
|---------|---------|-----|
| `knowledge_qa` | 标准 RAG 流程（原第 5–7 章） | 纯问答 |
| `tool_query` | 检索 `type=tool` 的文档块 | 问"怎么用" |
| `tool_execute` | 进入新增的**行动 Agent 流程**（第 14 章） | 直接执行 |
| `multi_step` | 进入新增的**多步规划流程**（第 14 章） | 复合任务 |

> 该分类器是行动 Agent（第 14 章）的第一环；它本身不执行任何操作，只做"该走 RAG 还是该走 CLI"的判决。

---

## 本章小结

- 向量检索 + BM25 是标配
- RRF 是最稳健的融合策略
- Query 改写能显著提升效果
- HyDE 效果好但有额外开销
- 检索过滤可以缩小范围，提升效率

下一章，我们讲重排序——怎么对检索结果精排。

---

*我之前觉得检索就够用了，加 Reranker 多此一举。后来测了一下，准确率从 72% 提到 87%。真香。*
