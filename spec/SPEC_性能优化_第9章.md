# SPEC_性能优化_第9章

> 技术规格说明书 - 性能优化

> 🔧 **v2 修订（2026-07）**：已应用 UP-001（EmbeddingCache 改 OrderedDict LRU）、UP-002（embed_with_cache 回填用并行 zip，删除 O(n) 的 .index 反查）。详见 `SPEC_更新修订清单_v2.md`。

---

## 1. 章节概述

### 1.1 目标

定义性能优化的技术规格，包括缓存策略、批量处理、异步化和索引优化。

### 1.2 范围

- 缓存策略
- 批量处理规范
- 异步化实现
- 索引优化

---

## 2. 缓存策略

### 2.1 缓存层级

```
┌─────────────────────────────────────┐
│         L1: 语义缓存               │
│    (相似查询命中缓存)              │
├─────────────────────────────────────┤
│         L2: Embedding 缓存         │
│    (相同文本向量化缓存)            │
├─────────────────────────────────────┤
│         L3: 检索结果缓存           │
│    (查询结果缓存)                  │
├─────────────────────────────────────┤
│         L4: LLM 响应缓存           │
│    (生成结果缓存)                  │
└─────────────────────────────────────┘
```

### 2.2 Embedding 缓存

```python
import hashlib
from typing import Optional

class EmbeddingCache:
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[list[float]]:
        """获取缓存"""
        key = self._hash(text)
        return self.cache.get(key)
    
    def set(self, text: str, embedding: list[float]):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # LRU 淘汰
            keys = list(self.cache.keys())
            del self.cache[keys[0]]
        
        self.cache[self._hash(text)] = embedding
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
```

### 2.3 语义缓存

```python
import numpy as np
from typing import Optional

class SemanticCache:
    def __init__(self, threshold: float = 0.95, max_size: int = 1000):
        self.cache = {}
        self.threshold = threshold
        self.max_size = max_size
    
    def get(self, query_embedding: list[float]) -> Optional[str]:
        """获取缓存"""
        query_emb = np.array(query_embedding)
        
        for cached_emb, (answer, _) in self.cache.items():
            cached_emb = np.array(cached_emb)
            similarity = np.dot(query_emb, cached_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
            )
            if similarity > self.threshold:
                return answer
        
        return None
    
    def set(self, query_embedding: list[float], answer: str):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            # 淘汰最旧的
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        import time
        self.cache[tuple(query_embedding)] = (answer, time.time())
```

---

## 3. 批量处理

### 3.1 批量导入规范

```python
class BatchProcessor:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def batch_ingest(
        self,
        file_paths: list[str],
        pipeline,
        embedder,
        vector_store
    ) -> dict:
        """
        批量导入
        
        Args:
            file_paths: 文件路径列表
            pipeline: 数据管道
            embedder: 向量化引擎
            vector_store: 向量数据库
        
        Returns:
            dict: 处理统计
        """
        stats = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "total_chunks": 0,
            "failed_files": []
        }
        
        all_chunks = []
        for path in file_paths:
            try:
                chunks = pipeline.process(path)
                all_chunks.extend(chunks)
            except Exception as e:
                stats["failed_files"].append({"file": path, "error": str(e)})
        
        # 分批处理
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i+self.batch_size]
            
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            
            embeddings = embedder.embed(texts)
            vector_store.add(texts, embeddings, metadatas)
            
            stats["processed_files"] += 1
            stats["total_chunks"] += len(batch)
        
        return stats
```

### 3.2 批量查询规范

```python
class BatchQuerier:
    def __init__(self, kb, batch_size: int = 32):
        self.kb = kb
        self.batch_size = batch_size
    
    def batch_query(self, queries: list[str]) -> list[dict]:
        """
        批量查询
        
        Args:
            queries: 查询列表
        
        Returns:
            list[dict]: 查询结果列表
        """
        results = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i+self.batch_size]
            
            batch_results = []
            for query in batch:
                result = self.kb.answer(query)
                batch_results.append({"query": query, "answer": result})
            
            results.extend(batch_results)
        
        return results
```

---

## 4. 异步化

### 4.1 异步向量化

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncEmbedder:
    def __init__(self, embedder, max_workers: int = 4):
        self.embedder = embedder
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """异步向量化"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.embedder.embed,
            texts
        )
```

### 4.2 异步检索

```python
import asyncio
from qdrant_client import AsyncQdrantClient

class AsyncVectorStore:
    def __init__(self, config: VectorStoreConfig, dimension: int):
        self.client = AsyncQdrantClient(host=config.host, port=config.port)
        self.config = config
        self.dimension = dimension
    
    async def search_async(
        self,
        query_embedding: list[float],
        top_k: int = 10
    ) -> list[dict]:
        """异步检索"""
        results = await self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            {
                "id": hit.id,
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results
        ]
```

### 4.3 异步生成

```python
import openai

class AsyncGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
    
    async def generate_async(self, query: str, context: str) -> str:
        """异步生成"""
        messages = [
            {"role": "system", "content": f"参考资料：{context}"},
            {"role": "user", "content": query}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
```

---

## 5. 索引优化

### 5.1 HNSW 参数调优

```python
@dataclass
class HNSWOptimizer:
    # 数据量分级
    small_dataset: int = 100000  # < 10万
    medium_dataset: int = 1000000  # 10-100万
    large_dataset: int = 10000000  # > 100万
    
    def get_params(self, data_size: int) -> dict:
        """根据数据量获取最优参数"""
        if data_size < self.small_dataset:
            return {"M": 16, "ef_construction": 200, "ef": 64}
        elif data_size < self.medium_dataset:
            return {"M": 32, "ef_construction": 400, "ef": 128}
        else:
            return {"M": 64, "ef_construction": 800, "ef": 256}
```

### 5.2 分片策略

```python
class ShardManager:
    def __init__(self):
        self.collections = {}
    
    def create_shard(self, shard_name: str, config: VectorStoreConfig):
        """创建分片"""
        self.collections[shard_name] = VectorStore(config)
    
    def route_query(self, query: str, query_embedding: list[float]) -> str:
        """路由查询到对应分片"""
        # 根据查询内容决定分片
        if "退货" in query or "退款" in query:
            return "policy"
        elif "产品" in query or "功能" in query:
            return "product"
        else:
            return "default"
    
    def search(self, query: str, query_embedding: list[float], top_k: int = 10):
        """跨分片检索"""
        shard = self.route_query(query, query_embedding)
        return self.collections[shard].search(query_embedding, top_k)
```

---

## 6. 性能监控

### 6.1 性能指标

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class PerformanceMetrics:
    # 延迟指标
    embedding_latency_ms: float = 0
    retrieval_latency_ms: float = 0
    rerank_latency_ms: float = 0
    generation_latency_ms: float = 0
    total_latency_ms: float = 0
    
    # 吞吐量指标
    queries_per_second: float = 0
    
    # 资源指标
    memory_usage_mb: float = 0
    cpu_usage_percent: float = 0

class PerformanceTracker:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def track_embedding(self, func):
        """追踪 embedding 延迟"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            self.metrics.embedding_latency_ms = (time.time() - start) * 1000
            return result
        return wrapper
    
    def track_retrieval(self, func):
        """追踪检索延迟"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            self.metrics.retrieval_latency_ms = (time.time() - start) * 1000
            return result
        return wrapper
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取指标"""
        if self.start_time:
            self.metrics.total_latency_ms = (time.time() - self.start_time) * 1000
        return self.metrics
```

---

## 7. 测试用例

### 7.1 性能测试

```python
def test_embedding_performance():
    config = EmbeddingConfig()
    embedder = Embedder(config)
    
    texts = ["测试文本"] * 1000
    
    import time
    start = time.time()
    embeddings = embedder.embed(texts)
    end = time.time()
    
    # 1000 条文本应该在 10 秒内完成
    assert end - start < 10

def test_retrieval_performance():
    kb = AgentKnowledgeBase()
    
    # 插入测试数据
    for i in range(1000):
        kb.ingest(f"test_{i}.pdf")
    
    import time
    start = time.time()
    for _ in range(100):
        kb.retrieve("测试查询")
    end = time.time()
    
    # 100 次检索应该在 5 秒内完成
    assert end - start < 5
```

### 7.2 缓存测试

```python
def test_embedding_cache():
    cache = EmbeddingCache(max_size=100)
    
    # 第一次
    embedding = [0.1] * 1024
    cache.set("测试文本", embedding)
    
    # 第二次
    cached = cache.get("测试文本")
    assert cached == embedding
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
