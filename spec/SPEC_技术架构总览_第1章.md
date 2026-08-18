# SPEC_技术架构总览_第1章

> 技术规格说明书 - Agent 知识库技术指南

---

## 1. 章节概述

### 1.1 目标

定义 Agent 知识库系统的完整技术架构，包括数据流、核心组件、接口规范和技术选型依据。

### 1.2 范围

- 系统架构设计
- 数据流定义
- 组件接口规范
- 技术选型矩阵

### 1.3 术语表

| 术语 | 定义 |
|-----|------|
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| Embedding | 向量化，将文本转换为高维向量 |
| Chunk | 文档分块，将长文档切分为可检索的片段 |
| Vector DB | 向量数据库，专门存储和检索向量的数据库 |
| Reranker | 重排序模型，对检索结果精排 |

---

## 2. 算法原理

### 2.1 向量检索算法原理

#### HNSW (Hierarchical Navigable Small World)

**核心思想**：构建多层图结构，每层是一个小世界图，通过层级跳转快速定位目标。

```
层级结构:
L3:  A ─── D ─── H
     │     │     │
L2:  A ─── B ─── D ─── F ─── H
     │     │     │     │     │
L1:  A ─── B ─── C ─── D ─── E ─── F ─── G ─── H
     │     │     │     │     │     │     │     │
L0:  A ─ B ─ C ─ D ─ E ─ F ─ G ─ H (全连接)

搜索过程:
1. 从最高层(L3)的入口点开始
2. 在当前层贪心搜索，找到最近的节点
3. 如果当前层不是最底层，下降到下一层，以上一步的节点为起点
4. 重复直到到达最底层(L0)
5. 在最底层执行精确搜索
```

**参数影响**：

| 参数 | 增大效果 | 减小效果 |
|-----|---------|---------|
| M (连接数) | 召回率↑, 内存↑, 构建慢 | 召回率↓, 内存↓, 构建快 |
| ef_construction | 索引质量↑, 构建慢 | 索引质量↓, 构建快 |
| ef (搜索) | 召回率↑, 搜索慢 | 召回率↓, 搜索快 |

#### IVF (Inverted File Index)

**核心思想**：将向量空间用聚类算法划分成若干区域，每个区域维护一个倒排索引。

```
训练阶段:
  1. 用 K-Means 将所有向量聚成 nlist 个簇
  2. 每个簇有一个中心点 (centroid)

搜索阶段:
  1. 计算查询向量与所有中心点的距离
  2. 找到最近的 nprobe 个簇
  3. 只在这些簇中搜索（而非全量搜索）
  
  nprobe 越大 → 召回率越高，但搜索越慢
```

**参数配置**：

```python
# IVF 参数
index_params = {
    "index_type": "IVF_FLAT",
    "params": {
        "nlist": 1024  # 聚类中心数量，一般为 sqrt(数据量)
    }
}

# 搜索参数
search_params = {
    "params": {
        "nprobe": 32  # 搜索的簇数量，越大召回率越高
    }
}
```

#### PQ (Product Quantization)

**核心思想**：将高维向量分成多个子空间，每个子空间用聚类中心表示，实现压缩。

```
原始向量: [0.1, 0.3, -0.2, 0.5, 0.8, -0.1, ...] (1024维)

分组 (m=32, 每组32维):
  子空间1: [0.1, 0.3, ...] → 聚类中心 C1
  子空间2: [-0.2, 0.5, ...] → 聚类中心 C2
  ...
  子空间32: [...] → 聚类中心 C32

存储: 只存聚类中心的ID (32个字节 vs 4096字节)
压缩比: 128倍
```

### 2.2 相似度度量

#### 余弦相似度 (Cosine Similarity)

```python
def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度
    
    公式: cos(θ) = (a·b) / (||a|| × ||b||)
    
    特点:
    - 范围: [-1, 1]
    - 只关注方向，不关注长度
    - 适合文本相似度计算
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

#### 内积 (Inner Product / Dot Product)

```python
def inner_product(a, b):
    """
    计算两个向量的内积
    
    公式: a·b = Σ(a[i] * b[i])
    
    特点:
    - 范围: [-∞, +∞]
    - 关注方向和长度
    - 适合推荐系统
    """
    return np.dot(a, b)
```

#### 欧氏距离 (L2 Distance)

```python
def l2_distance(a, b):
    """
    计算两个向量的欧氏距离
    
    公式: ||a - b|| = sqrt(Σ(a[i] - b[i])²)
    
    特点:
    - 范围: [0, +∞]
    - 关注绝对距离
    - 适合聚类
    """
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))
```

#### 度量选择指南

| 度量 | 适用场景 | 不适用场景 |
|-----|---------|-----------|
| 余弦相似度 | 文本检索、语义相似度 | 需要长度信息的场景 |
| 内积 | 推荐系统、注意力机制 | 向量未归一化时 |
| 欧氏距离 | 聚类、异常检测 | 高维稀疏向量 |

## 3. 系统架构

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户接口层                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ REST API │  │ WebSocket│  │ CLI      │  │ Web UI   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      业务逻辑层                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Query Processor                         │  │
│  │  - Intent Recognition                                │  │
│  │  - Query Rewriting                                   │  │
│  │  - Query Expansion                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Retrieval Engine                         │  │
│  │  - Vector Search                                     │  │
│  │  - BM25 Search                                       │  │
│  │  - Hybrid Fusion                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Reranker                                 │  │
│  │  - Cross-Encoder Scoring                             │  │
│  │  - Score Normalization                               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Generator                                │  │
│  │  - Context Assembly                                  │  │
│  │  - Prompt Construction                               │  │
│  │  - Response Generation                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      数据存储层                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Vector DB│  │ Cache    │  │ Metadata │  │ Logs     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 组件职责

| 组件 | 职责 | 输入 | 输出 |
|-----|------|-----|------|
| Query Processor | 处理用户查询 | 原始查询 | 标准化查询 |
| Retrieval Engine | 检索相关文档 | 标准化查询 | 候选文档列表 |
| Reranker | 精排候选文档 | 候选文档列表 | 排序后文档 |
| Generator | 生成最终回答 | 排序后文档 + 查询 | 回答文本 |

---

## 3. 数据流规范

### 3.1 查询处理流程

```
User Query
    ↓
[Validation] - 参数校验
    ↓
[Rewriting] - Query 改写
    ↓
[Embedding] - 向量化
    ↓
[Retrieval] - 多路召回
    ↓
[Fusion] - 结果融合
    ↓
[Reranking] - 精排
    ↓
[Context Assembly] - 上下文拼接
    ↓
[Generation] - LLM 生成
    ↓
[Post Processing] - 后处理（引用标注等）
    ↓
Response
```

### 3.2 数据导入流程

```
Document
    ↓
[Format Detection] - 格式检测
    ↓
[Loading] - 文档加载
    ↓
[Cleaning] - 数据清洗
    ↓
[Chunking] - 文档分块
    ↓
[Metadata Extraction] - 元数据提取
    ↓
[Embedding] - 向量化
    ↓
[Indexing] - 索引入库
    ↓
[Validation] - 导入验证
```

---

## 4. 接口规范

### 4.1 Query Processor 接口

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class QueryRequest:
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    filters: Optional[dict] = None
    top_k: int = 10

@dataclass
class ProcessedQuery:
    original: str
    rewritten: str
    expanded: list[str]
    embedding: list[float]
    filters: dict

class QueryProcessor:
    def process(self, request: QueryRequest) -> ProcessedQuery:
        """
        处理用户查询
        
        Args:
            request: 查询请求
            
        Returns:
            ProcessedQuery: 处理后的查询
        """
        pass
```

### 4.2 Retrieval Engine 接口

```python
@dataclass
class RetrievalResult:
    doc_id: str
    text: str
    score: float
    metadata: dict
    source: str

class RetrievalEngine:
    def retrieve(
        self, 
        query: ProcessedQuery, 
        top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 处理后的查询
            top_k: 返回数量
            
        Returns:
            list[RetrievalResult]: 检索结果
        """
        pass
```

### 4.3 Reranker 接口

```python
@dataclass
class RerankedResult:
    doc_id: str
    text: str
    original_score: float
    rerank_score: float
    metadata: dict

class Reranker:
    def rerank(
        self, 
        query: str, 
        results: list[RetrievalResult],
        top_k: int = 3
    ) -> list[RerankedResult]:
        """
        重排序
        
        Args:
            query: 原始查询
            results: 检索结果
            top_k: 返回数量
            
        Returns:
            list[RerankedResult]: 排序后结果
        """
        pass
```

### 4.4 Generator 接口

```python
@dataclass
class GenerationRequest:
    query: str
    context: list[RerankedResult]
    system_prompt: Optional[str] = None
    temperature: float = 0
    max_tokens: int = 1024

@dataclass
class GenerationResponse:
    answer: str
    sources: list[str]
    confidence: float
    metadata: dict

class Generator:
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        生成回答
        
        Args:
            request: 生成请求
            
        Returns:
            GenerationResponse: 生成结果
        """
        pass
    
    def generate_stream(self, request: GenerationRequest):
        """
        流式生成回答
        
        Args:
            request: 生成请求
            
        Yields:
            str: 生成的文本片段
        """
        pass
```

---

## 5. 配置参数

### 5.1 系统配置

```python
@dataclass
class SystemConfig:
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "logs/agent_kb.log"
    
    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 秒
    cache_max_size: int = 10000
```

### 5.2 检索配置

```python
@dataclass
class RetrievalConfig:
    # 向量检索
    vector_top_k: int = 10
    vector_score_threshold: float = 0.6
    
    # BM25 检索
    bm25_top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # 融合策略
    fusion_method: str = "rrf"  # rrf | weighted
    rrf_k: int = 60
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    
    # 重排序
    rerank_enabled: bool = True
    rerank_top_k: int = 3
    rerank_model: str = "BAAI/bge-reranker-large"
```

### 5.3 生成配置

```python
@dataclass
class GenerationConfig:
    model: str = "gpt-4"
    temperature: float = 0
    max_tokens: int = 1024
    stream: bool = True
    
    # 上下文配置
    max_context_length: int = 4000
    include_sources: bool = True
    source_format: str = "[{index}] {text}\n来源：{source}"
```

---

## 6. 非功能需求

### 6.1 性能要求

| 指标 | 目标值 |
|-----|-------|
| 查询延迟（P95） | < 500ms |
| 吞吐量 | > 100 QPS |
| 导入速度 | > 1000 docs/min |

### 6.2 可用性要求

| 指标 | 目标值 |
|-----|-------|
| 可用性 | > 99.9% |
| 故障恢复时间 | < 5min |
| 数据持久性 | > 99.999% |

### 6.3 扩展性要求

| 指标 | 目标值 |
|-----|-------|
| 数据容量 | > 10M docs |
| 并发连接 | > 1000 |
| 水平扩展 | 支持 |

---

## 7. 依赖项

### 7.1 外部依赖

| 依赖 | 版本 | 用途 |
|-----|------|-----|
| Python | >= 3.9 | 运行环境 |
| sentence-transformers | >= 2.2 | Embedding |
| qdrant-client | >= 1.3 | 向量数据库 |
| openai | >= 1.0 | LLM 调用 |
| langchain | >= 0.0 | 文档处理 |

### 7.2 基础设施依赖

| 组件 | 用途 | 部署方式 |
|-----|------|---------|
| Qdrant | 向量存储 | Docker / K8s |
| Redis | 缓存 | Docker / K8s |
| PostgreSQL | 元数据存储 | Docker / K8s |

---

## 8. 测试要求

### 8.1 单元测试

- 每个组件独立测试
- 覆盖率 > 80%
- 边界条件测试

### 8.2 集成测试

- 组件间交互测试
- 端到端流程测试
- 性能基准测试

### 8.3 验收标准

- 所有测试通过
- 性能指标达标
- 无 P0/P1 级 bug

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
