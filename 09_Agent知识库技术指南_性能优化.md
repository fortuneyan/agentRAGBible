# 第 9 章：性能优化

> 知识库搭起来只是开始，跑得快才是真本事。

---

## 缓存策略

### Embedding 缓存

> ⚠️ **早期版本的坑**（UP-001）：原来的 `set` 注释写"删掉一半"，实际删的是字典前 N//2 个 key
> ——可能是刚刚写入的热数据，且不是 LRU。读者照抄进生产会导致缓存命中率骤降。
> 下面是修正后的 **OrderedDict LRU** 实现。

```python
import hashlib
from collections import OrderedDict

class EmbeddingCache:
    """
    基于 OrderedDict 的 LRU（Least Recently Used）缓存。
    - get 命中时把 key 移到队尾（最近使用）
    - set 满时从队首淘汰（最久未使用）
    """
    def __init__(self, max_size: int = 10000):
        self.cache: OrderedDict[str, list[float]] = OrderedDict()
        self.max_size = max_size

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)  # 命中，提升为最近使用
            return self.cache[key]
        return None

    def set(self, text: str, embedding: list[float]):
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)  # 已存在，提升
        else:
            # 满了，从队首淘汰最久未访问的项（不是最早写入的！）
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = embedding

    def clear(self):
        self.cache.clear()

# 使用
cache = EmbeddingCache()

def embed_with_cache(texts: list[str]) -> list[list[float]]:
    """
    批量带缓存 embed。
    修正点（UP-002）：早期版本用 to_embed[to_embed_indices.index(idx)] 反查文本，
    是 O(n) 查找且重复 idx 会取错；这里直接并行 zip 三个顺序一致的列表。
    """
    results: list[list[float] | None] = [None] * len(texts)
    to_embed: list[str] = []
    to_embed_indices: list[int] = []

    # 1. 先查缓存，命中的直接填，未命中的收集起来
    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            results[i] = cached
        else:
            to_embed.append(text)
            to_embed_indices.append(i)

    # 2. 未命中的批量算
    if to_embed:
        new_embeddings = embedder.embed(to_embed)
        # ✅ 三个列表顺序天然一致，直接并行 zip，无需 .index() 反查
        for idx, text, embedding in zip(to_embed_indices, to_embed, new_embeddings):
            results[idx] = embedding
            cache.set(text, embedding)

    # 3. 到这里 results 里不应该再有 None
    return results  # type: ignore[return-value]
```

### 语义缓存

```python
import numpy as np

class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self.cache = {}  # embedding -> (answer, timestamp)
        self.threshold = threshold
    
    def get(self, query_embedding: list[float]) -> str:
        query_emb = np.array(query_embedding)
        
        for cached_emb, (answer, _) in self.cache.items():
            similarity = np.dot(query_emb, cached_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
            )
            if similarity > self.threshold:
                return answer
        
        return None
    
    def set(self, query_embedding: list[float], answer: str):
        self.cache[tuple(query_embedding)] = (answer, time.time())

# 使用
semantic_cache = SemanticCache(threshold=0.95)

def answer_with_cache(query: str) -> str:
    query_embedding = embedder.embed_single(query)
    
    # 先查缓存
    cached = semantic_cache.get(query_embedding)
    if cached:
        return cached
    
    # 没有缓存，正常生成
    answer = kb.answer(query)
    
    # 存入缓存
    semantic_cache.set(query_embedding, answer)
    
    return answer
```

### 检索结果缓存

```python
from functools import lru_cache
import json

@lru_cache(maxsize=1000)
def cached_retrieve(query: str, top_k: int = 10) -> str:
    """缓存检索结果"""
    results = vector_store.search(query_embedding, top_k)
    return json.dumps(results)

def retrieve_with_cache(query: str, top_k: int = 10) -> list[dict]:
    cached = cached_retrieve(query, top_k)
    return json.loads(cached)
```

### CLI 结果缓存（行动 Agent 只读操作）

对幂等、只读的 CLI 调用（如 `ls`、`cat`、`--json` 查询），结果可缓存以避免重复执行与重复开销。

```python
import hashlib, json, time

class CLIResultCache:
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.cache = {}  # key -> (result, timestamp)
        self.ttl = ttl
        self.max_size = max_size

    def get(self, tool_name: str, params: dict):
        key = hashlib.md5(f"{tool_name}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        if key in self.cache and time.time() - self.cache[key]["ts"] < self.ttl:
            return self.cache[key]["result"]
        return None

    def set(self, tool_name: str, params: dict, result):
        self.cache[hashlib.md5(f"{tool_name}:{json.dumps(params, sort_keys=True)}".encode()).hexdigest()] = {
            "result": result, "ts": time.time()
        }
```

**执行器配置（超时与重试，写入 `config.py`）**：

```python
from dataclasses import dataclass, field

@dataclass
class ExecutorConfig:
    default_timeout: int = 30          # 单条命令超时秒数
    max_retries: int = 3               # 失败重试次数
    retry_backoff: float = 1.0         # 指数退避基数
    safe_paths: list = field(default_factory=lambda: ["/data", "/tmp/agent_workspace"])
    banned_commands: list = field(default_factory=lambda: ["rm", "dd", "mkfs", "curl", "wget"])
```

> 这两个组件是行动 Agent 执行层（第 14 章）的性能与稳定性基础：缓存减少重复执行，超时/重试防止单条命令挂死整个链路。

---

## 批量处理

### 批量导入

```python
def batch_ingest(file_paths: list[str], batch_size: int = 100):
    all_chunks = []
    for path in file_paths:
        chunks = pipeline.process(path)
        all_chunks.extend(chunks)
    
    # 批量处理
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        
        embeddings = embedder.embed(texts)
        vector_store.add(texts, embeddings, metadatas)
        
        print(f"已处理 {i+len(batch)}/{len(all_chunks)}")
```

### 批量查询

```python
def batch_query(queries: list[str], top_k: int = 5) -> list[list[dict]]:
    # 批量向量化
    query_embeddings = embedder.embed(queries)
    
    results = []
    for embedding in query_embeddings:
        docs = vector_store.search(embedding, top_k)
        results.append(docs)
    
    return results
```

---

## 异步化

### 异步检索

```python
import asyncio
from qdrant_client import AsyncQdrantClient

class AsyncVectorStore:
    def __init__(self, config: Config):
        self.client = AsyncQdrantClient(host=config.vector_db_host, port=config.vector_db_port)
    
    async def search(self, query_embedding: list[float], top_k: int = 10):
        return await self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k
        )

async def async_retrieve(query: str) -> list[dict]:
    query_embedding = embedder.embed_single(query)
    return await vector_store.search(query_embedding)
```

### 异步生成

```python
import openai

async def async_generate(query: str, context: str) -> str:
    client = openai.AsyncOpenAI()
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query}
        ]
    )
    
    return response.choices[0].message.content
```

### 并发处理

```python
import asyncio

async def process_multiple_queries(queries: list[str]):
    tasks = [async_retrieve(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

---

## 索引优化

### HNSW 参数调优

```python
# 构建时参数（更精确，但更慢）
index_params = {
    "M": 32,  # 每个节点的连接数，越大越精确
    "efConstruction": 400  # 构建时的搜索范围
}

# 检索时参数（更精确，但更慢）
search_params = {
    "ef": 128  # 检索时的搜索范围
}
```

### 量化压缩

```python
# PQ 量化（牺牲精度换空间）
index_params = {
    "metric_type": "COSINE",
    "index_type": "PQ",
    "params": {"nbits": 8}
}
```

---

## 分片策略

```python
# 按文档类型分片
collections = {
    "policy": "agent_kb_policy",
    "product": "agent_kb_product",
    "faq": "agent_kb_faq"
}

def search_by_type(query: str, doc_type: str):
    collection = collections[doc_type]
    return vector_store.search(query_embedding, collection=collection)
```

---

## 本章小结

- 缓存是性价比最高的优化手段
- 批量处理减少网络开销
- 异步化提升吞吐量
- 索引参数要根据场景调优
- 分片可以缩小检索范围

下一章，我们讲监控与评估。

---

*性能优化是个持续的过程，别指望一次搞定。先跑起来，看瓶颈在哪，再针对性优化。*
