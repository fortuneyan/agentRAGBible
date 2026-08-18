# SPEC_检索策略_第5章

> 技术规格说明书 - 检索策略

---

## 1. 章节概述

### 1.1 目标

定义检索策略的技术规格，包括向量检索、BM25 检索、混合检索和 Query 改写。

### 1.2 范围

- 向量检索算法
- BM25 检索实现
- 混合检索融合
- Query 改写策略

---

## 2. 向量检索

### 2.1 索引类型规范

| 索引类型 | 原理 | 适用场景 | 精度 | 速度 |
|---------|------|---------|-----|------|
| HNSW | 分层可导航小世界图 | 通用场景 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| IVF | 倒排文件索引 | 大规模数据 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| PQ | 乘积量化 | 内存受限 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| FLAT | 暴力搜索 | 小规模精确 | ⭐⭐⭐⭐⭐ | ⭐ |

### 2.2 HNSW 参数配置

```python
@dataclass
class HNSWConfig:
    # 构建参数
    m: int = 16  # 每个节点的连接数
    ef_construction: int = 200  # 构建时的搜索范围
    
    # 检索参数
    ef: int = 64  # 检索时的搜索范围
    
    # 优化建议
    # - 数据量 < 10万: m=16, ef_construction=200, ef=64
    # - 数据量 10-100万: m=32, ef_construction=400, ef=128
    # - 数据量 > 100万: m=64, ef_construction=800, ef=256
```

### 2.3 向量检索实现

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class VectorSearchResult:
    id: str
    score: float
    text: str
    metadata: dict

class VectorRetriever:
    def __init__(self, vector_store, config: HNSWConfig):
        self.vector_store = vector_store
        self.config = config
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict = None
    ) -> list[VectorSearchResult]:
        """
        向量检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filters: 过滤条件
        
        Returns:
            list[VectorSearchResult]: 检索结果
        """
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        return [
            VectorSearchResult(
                id=r.id,
                score=r.score,
                text=r.text,
                metadata=r.metadata
            )
            for r in results
        ]
```

---

## 3. BM25 检索

### 3.1 BM25 参数配置

```python
@dataclass
class BM25Config:
    k1: float = 1.5  # 词频饱和参数
    b: float = 0.75  # 文档长度归一化参数
    
    # 分词配置
    use_jieba: bool = True  # 是否使用 jieba 分词
    custom_dict: Optional[str] = None  # 自定义词典
```

### 3.2 BM25 实现

```python
from rank_bm25 import BM25Okapi
import jieba
from typing import Optional

class BM25Retriever:
    def __init__(self, config: BM25Config):
        self.config = config
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def build_index(self, documents: list[dict]):
        """
        构建 BM25 索引
        
        Args:
            documents: 文档列表，每个文档包含 id 和 text
        """
        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]
        
        # 分词
        tokenized_corpus = [
            self._tokenize(doc["text"]) 
            for doc in documents
        ]
        
        # 构建索引
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.config.k1,
            b=self.config.b
        )
    
    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        if self.config.use_jieba:
            # 加载自定义词典
            if self.config.custom_dict:
                jieba.load_userdict(self.config.custom_dict)
            return list(jieba.cut(text))
        else:
            return text.split()
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> list[VectorSearchResult]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            list[VectorSearchResult]: 检索结果
        """
        # 查询分词
        query_tokens = self._tokenize(query)
        
        # 计算得分
        scores = self.bm25.get_scores(query_tokens)
        
        # 排序
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(VectorSearchResult(
                    id=self.doc_ids[idx],
                    score=float(scores[idx]),
                    text=self.documents[idx]["text"],
                    metadata=self.documents[idx].get("metadata", {})
                ))
        
        return results
```

---

## 4. 混合检索

### 4.1 融合策略

```python
@dataclass
class HybridConfig:
    # 融合方法
    fusion_method: str = "rrf"  # rrf | weighted
    
    # RRF 参数
    rrf_k: int = 60
    
    # 加权融合参数
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        config: HybridConfig
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config
    
    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10
    ) -> list[VectorSearchResult]:
        """
        混合检索
        
        Args:
            query: 查询文本
            query_embedding: 查询向量
            top_k: 返回数量
        
        Returns:
            list[VectorSearchResult]: 检索结果
        """
        # 向量检索
        vector_results = self.vector_retriever.search(
            query_embedding, top_k=top_k * 2
        )
        
        # BM25 检索
        bm25_results = self.bm25_retriever.search(
            query, top_k=top_k * 2
        )
        
        # 融合
        if self.config.fusion_method == "rrf":
            fused = self._rrf_fusion(vector_results, bm25_results)
        else:
            fused = self._weighted_fusion(vector_results, bm25_results)
        
        return fused[:top_k]
    
    def _rrf_fusion(
        self,
        vector_results: list[VectorSearchResult],
        bm25_results: list[VectorSearchResult]
    ) -> list[VectorSearchResult]:
        """RRF 融合"""
        k = self.config.rrf_k
        fused_scores = {}
        
        # 向量检索得分
        for rank, result in enumerate(vector_results):
            doc_id = result.id
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # BM25 检索得分
        for rank, result in enumerate(bm25_results):
            doc_id = result.id
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        
        # 排序
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # 构建结果
        all_results = {r.id: r for r in vector_results + bm25_results}
        
        return [
            VectorSearchResult(
                id=doc_id,
                score=fused_scores[doc_id],
                text=all_results[doc_id].text,
                metadata=all_results[doc_id].metadata
            )
            for doc_id in sorted_ids
            if doc_id in all_results
        ]
    
    def _weighted_fusion(
        self,
        vector_results: list[VectorSearchResult],
        bm25_results: list[VectorSearchResult]
    ) -> list[VectorSearchResult]:
        """加权融合"""
        # 归一化得分
        vector_max = max(r.score for r in vector_results) if vector_results else 1
        bm25_max = max(r.score for r in bm25_results) if bm25_results else 1
        
        scores = {}
        
        for r in vector_results:
            scores[r.id] = self.config.vector_weight * (r.score / vector_max)
        
        for r in bm25_results:
            if r.id in scores:
                scores[r.id] += self.config.bm25_weight * (r.score / bm25_max)
            else:
                scores[r.id] = self.config.bm25_weight * (r.score / bm25_max)
        
        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 构建结果
        all_results = {r.id: r for r in vector_results + bm25_results}
        
        return [
            VectorSearchResult(
                id=doc_id,
                score=scores[doc_id],
                text=all_results[doc_id].text,
                metadata=all_results[doc_id].metadata
            )
            for doc_id in sorted_ids
            if doc_id in all_results
        ]
```

---

## 5. Query 改写

### 5.1 改写策略

```python
from abc import ABC, abstractmethod

class BaseQueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> str:
        """改写查询"""
        pass

class RuleBasedRewriter(BaseQueryRewriter):
    """基于规则的改写"""
    
    def __init__(self):
        self.synonyms = {
            "退货": ["退款", "退换", "退回"],
            "积分": ["points", "score"],
            "怎么": ["如何", "咋"],
            "啥": ["什么"]
        }
    
    def rewrite(self, query: str) -> str:
        # 替换口语化表达
        for word, replacements in self.synonyms.items():
            if word in query:
                query = query.replace(word, replacements[0])
        
        return query

class LLMQueryRewriter(BaseQueryRewriter):
    """基于 LLM 的改写"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def rewrite(self, query: str) -> str:
        prompt = f"""请将以下用户问题改写成更适合搜索引擎检索的形式。

原始问题：{query}

改写后的查询："""
        
        response = self.llm.generate(prompt)
        return response.strip()
    
    def rewrite_multiple(self, query: str, n: int = 3) -> list[str]:
        """生成多个改写版本"""
        prompt = f"""请将以下用户问题改写成 {n} 个不同的查询语句，用于搜索引擎检索。

原始问题：{query}

输出格式（每行一个）：
1. ...
2. ...
3. ..."""
        
        response = self.llm.generate(prompt)
        lines = response.strip().split("\n")
        
        queries = []
        for line in lines:
            # 去除编号
            if ". " in line:
                line = line.split(". ", 1)[1]
            queries.append(line.strip())
        
        return queries[:n]

class HyDERewriter(BaseQueryRewriter):
    """HyDE（假设文档嵌入）改写"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def rewrite(self, query: str) -> str:
        prompt = f"""请回答以下问题，假设你正在编写一份内部文档的答案。

问题：{query}

答案："""
        
        hypothetical_answer = self.llm.generate(prompt)
        return hypothetical_answer
```

---

## 6. 检索过滤

### 6.1 过滤条件

```python
@dataclass
class SearchFilter:
    # 文档类型
    doc_type: Optional[str] = None
    
    # 时间范围
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # 来源
    source: Optional[str] = None
    
    # 自定义字段
    custom: Optional[dict] = None

class FilteredRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[SearchFilter] = None
    ) -> list[VectorSearchResult]:
        """带过滤的检索"""
        filter_dict = self._build_filter(filters)
        
        return self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filter_dict
        )
    
    def _build_filter(self, filters: Optional[SearchFilter]) -> Optional[dict]:
        """构建过滤条件"""
        if not filters:
            return None
        
        conditions = []
        
        if filters.doc_type:
            conditions.append({"key": "doc_type", "match": {"value": filters.doc_type}})
        
        if filters.source:
            conditions.append({"key": "source", "match": {"value": filters.source}})
        
        if filters.start_date:
            conditions.append({"key": "updated_at", "range": {"gte": filters.start_date}})
        
        if filters.end_date:
            conditions.append({"key": "updated_at", "range": {"lte": filters.end_date}})
        
        if not conditions:
            return None
        
        return {"must": conditions}
```

---

## 7. 测试用例

### 7.1 单元测试

```python
def test_vector_search():
    retriever = VectorRetriever(vector_store, HNSWConfig())
    
    results = retriever.search(
        query_embedding=[0.1] * 1024,
        top_k=5
    )
    
    assert len(results) <= 5
    assert all(r.score > 0 for r in results)

def test_bm25_search():
    bm25_retriever = BM25Retriever(BM25Config())
    bm25_retriever.build_index([
        {"id": "1", "text": "退货流程：7天无理由"},
        {"id": "2", "text": "会员积分规则"}
    ])
    
    results = bm25_retriever.search("退货", top_k=1)
    
    assert len(results) == 1
    assert results[0].id == "1"
```

### 7.2 集成测试

```python
def test_hybrid_search():
    hybrid = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        HybridConfig()
    )
    
    results = hybrid.search(
        query="怎么退货",
        query_embedding=embedder.embed_single("怎么退货"),
        top_k=5
    )
    
    assert len(results) <= 5
    assert results[0].score > 0
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
