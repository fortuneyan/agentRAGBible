# 第 4 章：向量数据库选型

> Chroma 一分钟上手，但数据量大了就卡；Milvus 能扛千万级，但运维复杂。选型要根据你的实际场景来。

---

## 主流向量数据库对比

| 数据库 | 部署难度 | 性能 | 生态 | 推荐指数 |
|-------|---------|-----|-----|---------|
| Milvus | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 生产首选 |
| Qdrant | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 轻量生产 |
| Weaviate | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 功能丰富 |
| Chroma | ⭐ | ⭐⭐ | ⭐⭐ | 开发测试 |
| Pinecone | ⭐（云服务） | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 不想运维 |
| pgvector | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 已有 PG |

---

## Qdrant（推荐起步用）

### 安装

```bash
# Docker 一键部署
docker run -p 6333:6333 qdrant/qdrant

# 或用 docker-compose
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

### Python 客户端

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# 连接
client = QdrantClient(host="localhost", port=6333)

# 创建集合
client.create_collection(
    collection_name="agent_kb",
    vectors_config=VectorParams(
        size=1024,  # 跟 embedding 模型维度一致
        distance=Distance.COSINE
    )
)

# 插入数据
points = [
    PointStruct(
        id=1,
        vector=[0.1] * 1024,
        payload={
            "text": "退货流程：7天无理由，需上传照片",
            "source": "policy.pdf",
            "page": 15
        }
    )
]

client.upsert(collection_name="agent_kb", points=points)

# 检索
results = client.search(
    collection_name="agent_kb",
    query_vector=[0.1] * 1024,
    limit=5
)

for result in results:
    print(f"score: {result.score}, text: {result.payload['text']}")
```

### 优点

- 部署简单（Docker 一键）
- 性能好（支持 HNSW 索引）
- 支持过滤检索
- Rust 编写，内存安全

### 缺点

- 分布式支持需要付费版
- 社区不如 Milvus 活跃

---

## Milvus（生产首选）

### 安装

```bash
# Docker Compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

docker-compose up -d
```

### Python 客户端

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 连接
connections.connect(host="localhost", port="19530")

# 定义 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
]

schema = CollectionSchema(fields=fields, description="Agent knowledge base")
collection = Collection(name="agent_kb", schema=schema)

# 创建索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 插入数据
data = [
    [0.1] * 1024,  # embedding
    ["退货流程：7天无理由"],  # text
    ["policy.pdf"],  # source
]
collection.insert(data)

# 检索
collection.load()
results = collection.search(
    data=[[0.1] * 1024],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    output_fields=["text", "source"]
)
```

### 优点

- 支持大规模数据（亿级）
- 分布式部署
- 丰富的索引类型
- 社区活跃

### 缺点

- 部署相对复杂
- 资源占用较大

---

## Chroma（开发测试）

```python
import chromadb

# 创建客户端（数据存本地）
client = chromadb.Client()

# 创建集合
collection = client.create_collection(
    name="agent_kb",
    metadata={"hnsw:space": "cosine"}
)

# 插入数据
collection.add(
    documents=["退货流程：7天无理由", "会员积分规则"],
    ids=["1", "2"],
    metadatas=[
        {"source": "policy.pdf", "page": 15},
        {"source": "member.pdf", "page": 3}
    ]
)

# 检索
results = collection.query(
    query_texts=["怎么退货"],
    n_results=2
)

print(results)
```

### 优点

- 一分钟上手
- 无需服务器
- 适合原型开发

### 缺点

- 数据量大了会卡
- 不支持分布式
- 生产环境不推荐

---

## pgvector（已有 PostgreSQL）

```sql
-- 安装扩展
CREATE EXTENSION vector;

-- 创建表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding vector(1024),
    metadata JSONB
);

-- 创建索引
CREATE INDEX ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 插入数据
INSERT INTO documents (text, embedding, metadata)
VALUES ('退货流程：7天无理由', '[0.1, 0.2, ...]', '{"source": "policy.pdf"}');

-- 检索
SELECT text, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

### 优点

- 已有 PostgreSQL 可直接用
- 支持混合查询（向量 + SQL）
- 运维简单

### 缺点

- 向量检索性能一般
- 大规模场景不推荐

---

## 选型建议

| 场景 | 推荐 |
|-----|-----|
| 开发测试 | Chroma |
| 小规模生产 | Qdrant |
| 大规模生产 | Milvus |
| 已有 PostgreSQL | pgvector |
| 不想运维 | Pinecone |

---

## 参数调优指南

### HNSW 参数调优

HNSW（Hierarchical Navigable Small World）是目前最常用的向量索引算法。

**核心参数**：

```python
# M: 每个节点的连接数
# 越大 → 召回率越高，但内存占用越大，索引越慢
# 推荐范围: 16-64

# ef_construction: 构建时的搜索范围
# 越大 → 索引质量越好，但构建越慢
# 推荐范围: 100-500

# ef: 检索时的搜索范围
# 越大 → 召回率越高，但检索越慢
# 推荐范围: 32-256
```

**调优策略**：

| 数据量 | M | ef_construction | ef | 预期召回率 |
|-------|---|-----------------|-----|----------|
| < 10万 | 16 | 200 | 64 | 95%+ |
| 10-50万 | 32 | 300 | 96 | 97%+ |
| 50-100万 | 48 | 400 | 128 | 98%+ |
| > 100万 | 64 | 500 | 192 | 99%+ |

**调优代码**：

```python
def tune_hnsw_params(data_size: int, recall_target: float = 0.95):
    """
    根据数据量和目标召回率，推荐 HNSW 参数
    
    Args:
        data_size: 数据量
        recall_target: 目标召回率
    
    Returns:
        dict: 推荐参数
    """
    if data_size < 100000:
        base_m = 16
        base_ef_construction = 200
        base_ef = 64
    elif data_size < 500000:
        base_m = 32
        base_ef_construction = 300
        base_ef = 96
    elif data_size < 1000000:
        base_m = 48
        base_ef_construction = 400
        base_ef = 128
    else:
        base_m = 64
        base_ef_construction = 500
        base_ef = 192
    
    # 根据召回率目标调整
    if recall_target > 0.98:
        base_m = min(base_m * 2, 128)
        base_ef_construction = min(base_ef_construction * 2, 1000)
        base_ef = min(base_ef * 2, 512)
    
    return {
        "M": base_m,
        "ef_construction": base_ef_construction,
        "ef": base_ef
    }
```

### 量化压缩策略

当内存受限时，可以用量化压缩减少内存占用。

**PQ（Product Quantization）量化**：

```python
# 原理: 将高维向量分成多个子空间，每个子空间用聚类中心表示
# 优势: 内存占用减少 4-32 倍
# 劣势: 精度略有损失

# Qdrant 量化配置
from qdrant_client.models import VectorParams, Distance, QuantizationConfig, ScalarQuantization

# 启用标量量化
quantization_config = QuantizationConfig(
    scalar=ScalarQuantization(
        type="int8",  # 量化类型: int8 | float16
        always_ram=True  # 是否始终加载到内存
    )
)

# 创建集合时启用量化
client.create_collection(
    collection_name="agent_kb",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    ),
    quantization_config=quantization_config
)
```

**量化方案对比**：

| 方案 | 内存压缩比 | 精度损失 | 适用场景 |
|-----|-----------|---------|---------|
| 无量化 | 1x | 0% | 内存充足 |
| INT8 量化 | 4x | < 1% | 通用场景 |
| FP16 量化 | 2x | 0% | 需要高精度 |
| PQ 量化 | 8-32x | 1-3% | 内存受限 |

### 分片策略

当数据量很大时，可以通过分片提升性能。

**按时间分片**：

```python
# 适用于: 数据有时效性，近期数据查询频繁
collections = {
    "2024Q1": "kb_2024_q1",
    "2024Q2": "kb_2024_q2",
    "2024Q3": "kb_2024_q3",
    "2024Q4": "kb_2024_q4"
}

# 查询时优先查近期分片
def search_with_time_shard(query, query_embedding):
    # 先查最近一个季度
    results = search_in_collection("kb_2024_q4", query_embedding)
    if results:
        return results
    
    # 没找到，查更早的
    return search_in_collection("kb_2024_q3", query_embedding)
```

**按文档类型分片**：

```python
# 适用于: 不同类型文档的查询模式不同
collections = {
    "policy": "kb_policy",      # 政策文档
    "product": "kb_product",    # 产品文档
    "faq": "kb_faq",           # FAQ
    "manual": "kb_manual"      # 操作手册
}

# 查询时根据意图路由到对应分片
def search_with_type_shard(query, query_embedding):
    # 意图识别
    if "退货" in query or "退款" in query:
        return search_in_collection("kb_policy", query_embedding)
    elif "功能" in query or "使用" in query:
        return search_in_collection("kb_product", query_embedding)
    else:
        # 默认搜所有分片
        return search_all_collections(query_embedding)
```

### 性能基准测试

在调优之前，先建立性能基准。

```python
import time
import numpy as np

class Benchmark:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def benchmark_insert(self, num_docs: int = 10000) -> dict:
        """测试插入性能"""
        texts = [f"测试文档{i}" for i in range(num_docs)]
        embeddings = [[np.random.random() for _ in range(1024)] for _ in range(num_docs)]
        metadatas = [{"index": i} for i in range(num_docs)]
        
        start = time.time()
        self.vector_store.add(texts, embeddings, metadatas)
        end = time.time()
        
        return {
            "docs_per_second": num_docs / (end - start),
            "total_time": end - start
        }
    
    def benchmark_search(self, num_queries: int = 100, top_k: int = 10) -> dict:
        """测试检索性能"""
        # 准备查询
        queries = [[np.random.random() for _ in range(1024)] for _ in range(num_queries)]
        
        # 预热
        for q in queries[:10]:
            self.vector_store.search(q, top_k)
        
        # 正式测试
        latencies = []
        for q in queries:
            start = time.time()
            self.vector_store.search(q, top_k)
            end = time.time()
            latencies.append((end - start) * 1000)
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "qps": 1000 / np.mean(latencies)
        }
    
    def benchmark_recall(
        self, 
        test_queries: list[dict],
        ground_truth: dict
    ) -> dict:
        """测试召回率"""
        recalls = {"recall@5": [], "recall@10": [], "recall@20": []}
        
        for query_info in test_queries:
            query_embedding = self.embedder.embed_single(query_info["query"])
            results = self.vector_store.search(query_embedding, top_k=20)
            
            retrieved_ids = set(r["id"] for r in results)
            expected_ids = set(ground_truth[query_info["query"]])
            
            for k in [5, 10, 20]:
                retrieved_at_k = set(list(retrieved_ids)[:k])
                recall = len(retrieved_at_k & expected_ids) / len(expected_ids)
                recalls[f"recall@{k}"].append(recall)
        
        return {k: np.mean(v) for k, v in recalls.items()}
```

### 调优检查清单

在调优之前，先检查以下项目：

```
□ 1. 数据质量
  - 分块大小是否合理？
  - 是否有过多噪声数据？
  - 元数据是否完整？

□ 2. 索引配置
  - 索引类型是否合适？
  - HNSW 参数是否匹配数据量？
  - 是否需要启用量化？

□ 3. 查询优化
  - 是否实现了 Query 改写？
  - 是否需要混合检索？
  - 检索结果是否经过重排序？

□ 4. 缓存策略
  - 是否实现了 Embedding 缓存？
  - 是否实现了语义缓存？
  - 缓存大小是否合理？

□ 5. 监控告警
  - 是否配置了延迟监控？
  - 是否配置了召回率监控？
  - 是否配置了错误率告警？
```

---

## 本章小结

- Chroma 适合原型，Milvus 适合生产
- Qdrant 是中间选择，性能和易用性平衡
- 选型要考虑数据规模、运维能力、成本
- HNSW 参数要根据数据量调优
- 量化压缩可以在内存和精度之间权衡
- 分片策略可以提升查询效率

下一章，我们讲检索策略——怎么找到最相关的文档。

---

*我之前用 Chroma 跑了个 demo，效果不错，直接上生产。结果数据量到 10 万就开始卡。换 Qdrant 之后才稳住。血泪教训。*
