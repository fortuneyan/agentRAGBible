# SPEC_向量数据库选型_第4章

> 技术规格说明书 - 向量数据库选型

---

## 1. 章节概述

### 1.1 目标

定义向量数据库的选型标准、配置规范和接口标准。

### 1.2 范围

- 数据库对比评估
- 配置参数标准
- 部署架构
- 接口规范

---

## 2. 数据库对比

### 2.1 评估矩阵

| 维度 | Chroma | Qdrant | Milvus | Weaviate | pgvector |
|-----|--------|--------|--------|----------|----------|
| 部署难度 | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 性能 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 生态 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 扩展性 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 运维成本 | 低 | 中 | 高 | 中 | 低 |

### 2.2 推荐场景

| 场景 | 推荐 | 理由 |
|-----|------|-----|
| 开发测试 | Chroma | 一分钟上手 |
| 小规模生产 | Qdrant | 性能和易用性平衡 |
| 大规模生产 | Milvus | 分布式支持 |
| 已有 PostgreSQL | pgvector | 无需额外组件 |

---

## 3. Qdrant 配置规范

### 3.1 服务配置

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC
    volumes:
      - ./qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__SERVICE__HTTP_PORT=6333
    deploy:
      resources:
        limits:
          memory: 4G
```

### 3.2 集合配置

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    OptimizersConfigDiff,
    HnswConfigDiff
)

class QdrantConfig:
    def __init__(self):
        self.host = "localhost"
        self.port = 6333
        self.collection_name = "agent_kb"
        self.vector_dim = 1024

def create_collection(client: QdrantClient, config: QdrantConfig):
    """创建集合"""
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(
            size=config.vector_dim,
            distance=Distance.COSINE,
            on_disk=False  # 是否存储在磁盘
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=20000
        ),
        hnsw_config=HnswConfigDiff(
            m=16,  # 每个节点的连接数
            ef_construct=200,  # 构建时的搜索范围
            full_scan_threshold=10000
        )
    )
```

### 3.3 检索配置

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def search_with_filter(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    top_k: int = 10,
    filters: dict = None
) -> list[dict]:
    """带过滤的检索"""
    query_filter = None
    
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        query_filter = Filter(must=conditions)
    
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter
    )
    
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload
        }
        for hit in results
    ]
```

---

## 4. Milvus 配置规范

### 4.1 服务配置

```yaml
# docker-compose.yml
version: '3.8'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ./etcd_data:/etcd
    
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    
  milvus:
    image: milvusdb/milvus:v2.3.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
```

### 4.2 集合配置

```python
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

class MilvusConfig:
    def __init__(self):
        self.host = "localhost"
        self.port = "19530"
        self.collection_name = "agent_kb"
        self.vector_dim = 1024
        self.metric_type = "COSINE"

def create_milvus_collection(config: MilvusConfig):
    """创建 Milvus 集合"""
    connections.connect(host=config.host, port=config.port)
    
    # 定义字段
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=config.vector_dim
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=2000
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=200
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON
        )
    ]
    
    # 创建 schema
    schema = CollectionSchema(
        fields=fields,
        description="Agent knowledge base"
    )
    
    # 创建集合
    collection = Collection(
        name=config.collection_name,
        schema=schema
    )
    
    # 创建索引
    index_params = {
        "metric_type": config.metric_type,
        "index_type": "HNSW",
        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }
    
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    return collection
```

---

## 5. 接口规范

### 5.1 统一接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict

class BaseVectorStore(ABC):
    @abstractmethod
    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict]
    ) -> list[str]:
        """添加文档"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict = None
    ) -> list[SearchResult]:
        """检索"""
        pass
    
    @abstractmethod
    def delete(self, ids: list[str]) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """获取文档数量"""
        pass
```

### 5.2 Qdrant 实现

```python
class QdrantVectorStore(BaseVectorStore):
    def __init__(self, config: QdrantConfig):
        self.client = QdrantClient(host=config.host, port=config.port)
        self.config = config
    
    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict]
    ) -> list[str]:
        from qdrant_client.models import PointStruct
        import uuid, hashlib

        def _make_point_id(text: str, metadata: dict) -> str:
            # 确定性 ID（UP-003 修复）：同内容同来源永远得到同一 ID，
            # 保证幂等 upsert，并支持第 2 章"按 source 增量更新"
            source = metadata.get("source", "unknown")
            h = hashlib.sha256((source + text.strip()).encode("utf-8")).hexdigest()[:16]
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{h}"))

        points = [
            PointStruct(
                id=_make_point_id(text, metadata),
                vector=embedding,
                payload={
                    "text": text,
                    **metadata
                }
            )
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]

        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )

        return [p.id for p in points]
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict = None
    ) -> list[SearchResult]:
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            SearchResult(
                id=str(hit.id),
                score=hit.score,
                text=hit.payload.get("text", ""),
                metadata={
                    k: v for k, v in hit.payload.items() 
                    if k != "text"
                }
            )
            for hit in results
        ]
    
    def delete(self, ids: list[str]) -> bool:
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=ids
        )
        return True
    
    def count(self) -> int:
        info = self.client.get_collection(self.config.collection_name)
        return info.points_count
```

---

## 6. 性能优化

### 6.1 HNSW 参数调优指南

HNSW 是目前最常用的向量索引算法，参数调优对性能影响很大。

**参数说明**：

| 参数 | 作用 | 默认值 | 推荐范围 |
|-----|------|-------|---------|
| M | 每个节点的连接数 | 16 | 16-64 |
| ef_construction | 构建时的搜索范围 | 100 | 100-500 |
| ef | 检索时的搜索范围 | 64 | 32-256 |

**参数与数据量的关系**：

```python
def get_recommended_hnsw_params(data_size: int) -> dict:
    """
    根据数据量推荐 HNSW 参数
    
    Args:
        data_size: 数据量
    
    Returns:
        dict: 推荐参数
    """
    if data_size < 100000:
        return {"M": 16, "ef_construction": 200, "ef": 64}
    elif data_size < 500000:
        return {"M": 32, "ef_construction": 300, "ef": 96}
    elif data_size < 1000000:
        return {"M": 48, "ef_construction": 400, "ef": 128}
    else:
        return {"M": 64, "ef_construction": 500, "ef": 192}
```

**参数调优实验**：

```python
def tune_hnsw_parameters(
    vector_store,
    test_queries: list[list[float]],
    ground_truth: list[set],
    param_grid: dict
) -> dict:
    """
    网格搜索最优 HNSW 参数
    
    Args:
        vector_store: 向量存储
        test_queries: 测试查询
        ground_truth: 真实相关文档
        param_grid: 参数网格
    
    Returns:
        dict: 最优参数
    """
    best_params = None
    best_recall = 0
    
    for m in param_grid.get("M", [16, 32]):
        for ef_construction in param_grid.get("ef_construction", [200, 400]):
            for ef in param_grid.get("ef", [64, 128]):
                # 更新索引参数
                vector_store.update_hnsw_params(m, ef_construction, ef)
                
                # 测试召回率
                recalls = []
                for query, expected in zip(test_queries, ground_truth):
                    results = vector_store.search(query, top_k=10, ef=ef)
                    retrieved_ids = set(r["id"] for r in results)
                    recall = len(retrieved_ids & expected) / len(expected)
                    recalls.append(recall)
                
                avg_recall = np.mean(recalls)
                
                if avg_recall > best_recall:
                    best_recall = avg_recall
                    best_params = {
                        "M": m,
                        "ef_construction": ef_construction,
                        "ef": ef,
                        "recall": avg_recall
                    }
    
    return best_params
```

### 6.2 量化压缩指南

量化可以在内存和精度之间权衡。

**量化方案对比**：

| 方案 | 压缩比 | 精度损失 | 内存占用 | 适用场景 |
|-----|-------|---------|---------|---------|
| 无量化 | 1x | 0% | 4KB/向量 | 内存充足 |
| FP16 | 2x | ~0% | 2KB/向量 | 需要高精度 |
| INT8 | 4x | <1% | 1KB/向量 | 通用场景 |
| INT4 | 8x | 1-3% | 0.5KB/向量 | 内存受限 |
| PQ | 8-32x | 1-5% | 0.125-0.5KB | 极端内存限制 |

**量化配置示例**：

```python
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    QuantizationConfig, 
    ScalarQuantization
)

# INT8 量化（推荐）
quantization_config = QuantizationConfig(
    scalar=ScalarQuantization(
        type="int8",
        always_ram=True
    )
)

# 创建集合
client.create_collection(
    collection_name="agent_kb",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    quantization_config=quantization_config
)
```

### 6.3 分片策略指南

分片可以提升查询效率和管理灵活性。

**分片策略选择**：

| 策略 | 适用场景 | 优势 | 劣势 |
|-----|---------|-----|------|
| 按时间 | 数据有时效性 | 近期数据查询快 | 跨时间查询复杂 |
| 按类型 | 不同类型查询模式不同 | 精准路由 | 需要意图识别 |
| 按大小 | 数据量不均衡 | 负载均衡 | 实现复杂 |

**分片实现**：

```python
class ShardManager:
    def __init__(self):
        self.shards = {}
    
    def create_shard(self, name: str, vector_store):
        """创建分片"""
        self.shards[name] = vector_store
    
    def route_query(self, query: str, query_embedding: list[float]) -> str:
        """路由查询到对应分片"""
        if "退货" in query or "退款" in query:
            return "policy"
        elif "产品" in query or "功能" in query:
            return "product"
        else:
            return "default"
    
    def search(
        self, 
        query: str, 
        query_embedding: list[float], 
        top_k: int = 10
    ) -> list[dict]:
        """跨分片检索"""
        shard_name = self.route_query(query, query_embedding)
        return self.shards[shard_name].search(query_embedding, top_k)
```

### 6.4 性能基准测试

建立性能基准，用于调优前后对比。

```python
import time
import numpy as np

class VectorDBBenchmark:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def benchmark_insert(self, num_docs: int = 10000) -> dict:
        """测试插入性能"""
        texts = [f"测试文档{i}" for i in range(num_docs)]
        embeddings = [list(np.random.random(1024)) for _ in range(num_docs)]
        metadatas = [{"index": i} for i in range(num_docs)]
        
        start = time.time()
        self.vector_store.add(texts, embeddings, metadatas)
        end = time.time()
        
        return {
            "docs_per_second": num_docs / (end - start),
            "total_time": end - start
        }
    
    def benchmark_search(self, num_queries: int = 100) -> dict:
        """测试查询性能"""
        queries = [list(np.random.random(1024)) for _ in range(num_queries)]
        
        latencies = []
        for q in queries:
            start = time.time()
            self.vector_store.search(q, top_k=10)
            end = time.time()
            latencies.append((end - start) * 1000)
        
        return {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "qps": 1000 / np.mean(latencies)
        }
```

---

## 7. 测试用例

### 7.1 功能测试

```python
def test_add_and_search():
    store = QdrantVectorStore(QdrantConfig())
    
    # 添加
    texts = ["测试文本1", "测试文本2"]
    embeddings = [[0.1] * 1024, [0.2] * 1024]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    
    ids = store.add(texts, embeddings, metadatas)
    assert len(ids) == 2
    
    # 检索
    results = store.search([0.1] * 1024, top_k=2)
    assert len(results) == 2
    assert results[0].score > 0.9
```

### 7.2 性能测试

```python
def test_search_performance():
    store = QdrantVectorStore(QdrantConfig())
    
    # 插入 10000 条数据
    texts = [f"测试文档{i}" for i in range(10000)]
    embeddings = [[0.1] * 1024 for _ in range(10000)]
    metadatas = [{"index": i} for i in range(10000)]
    
    store.add(texts, embeddings, metadatas)
    
    # 测试检索性能
    import time
    start = time.time()
    
    for _ in range(100):
        store.search([0.1] * 1024, top_k=10)
    
    end = time.time()
    avg_latency = (end - start) / 100
    
    # 平均延迟应该小于 50ms
    assert avg_latency < 0.05
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：修复 QdrantVectorStore.add 的 `id=i` 自增 bug（UP-003 P0），改为确定性 uuid5 幂等 ID；与正文第 2/8 章对齐*
