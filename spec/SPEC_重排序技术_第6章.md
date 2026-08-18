# SPEC_重排序技术_第6章

> 技术规格说明书 - 重排序技术

> 🔧 **v2 修订（2026-07）**：已应用 UP-004（rerank_with_metadata 不修改入参，返回新列表）、UP-105（cached_rerank 返回 tuple 而非 list，避免 lru_cache 复用引用被污染）。详见 `SPEC_更新修订清单_v2.md`。

---

## 1. 章节概述

### 1.1 目标

定义重排序（Reranker）的技术规格，包括模型选择、实现接口和性能优化。

### 1.2 范围

- Reranker 模型规范
- 实现接口
- 性能优化
- 集成方案

---

## 2. Reranker 模型规范

### 2.1 模型对比

| 模型 | 类型 | 效果 | 速度 | 内存 |
|-----|------|-----|------|------|
| bge-reranker-large | 交叉编码器 | ⭐⭐⭐⭐⭐ | 中 | 1.3GB |
| bge-reranker-v2-m3 | 交叉编码器 | ⭐⭐⭐⭐ | 中 | 2.2GB |
| Cohere Rerank | API | ⭐⭐⭐⭐ | 快 | N/A |
| LLM-based | LLM | ⭐⭐⭐ | 慢 | 取决于模型 |

### 2.2 配置规范

```python
from dataclasses import dataclass
from enum import Enum

class RerankerProvider(Enum):
    LOCAL = "local"
    COHERE = "cohere"
    LLM = "llm"

@dataclass
class RerankerConfig:
    # 模型配置
    provider: RerankerProvider = RerankerProvider.LOCAL
    model_name: str = "BAAI/bge-reranker-large"
    
    # API 配置
    api_key: Optional[str] = None
    
    # 本地部署配置
    device: str = "cpu"  # cpu | cuda
    
    # 性能配置
    batch_size: int = 32
    max_length: int = 512
    
    # 结果配置
    top_k: int = 3
    score_threshold: float = 0.0
```

---

## 3. 实现接口

### 3.1 基础接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RerankResult:
    doc_id: str
    text: str
    original_score: float
    rerank_score: float
    metadata: dict

class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3
    ) -> list[RerankResult]:
        """
        重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回数量
        
        Returns:
            list[RerankResult]: 排序后结果
        """
        pass
```

### 3.2 本地 Reranker 实现

```python
from sentence_transformers import CrossEncoder

class LocalReranker(BaseReranker):
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = CrossEncoder(
            config.model_name,
            max_length=config.max_length
        )
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3
    ) -> list[RerankResult]:
        """
        重排序
        """
        if not documents:
            return []
        
        # 构造 query-document 对
        texts = [doc.get("text", "") for doc in documents]
        pairs = [(query, text) for text in texts]
        
        # 批量预测
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size
        )
        
        # 构建结果
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                doc_id=doc.get("id", str(i)),
                text=doc.get("text", ""),
                original_score=doc.get("score", 0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {})
            ))
        
        # 排序
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return results[:top_k]
```

### 3.3 Cohere Reranker 实现

```python
import cohere

class CohereReranker(BaseReranker):
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.client = cohere.Client(config.api_key)
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3
    ) -> list[RerankResult]:
        """
        重排序
        """
        if not documents:
            return []
        
        texts = [doc.get("text", "") for doc in documents]
        
        response = self.client.rerank(
            query=query,
            documents=texts,
            top_n=top_k,
            model="rerank-english-v3.0"
        )
        
        results = []
        for r in response.results:
            doc = documents[r.index]
            results.append(RerankResult(
                doc_id=doc.get("id", str(r.index)),
                text=doc.get("text", ""),
                original_score=doc.get("score", 0),
                rerank_score=r.relevance_score,
                metadata=doc.get("metadata", {})
            ))
        
        return results
```

### 3.4 LLM-based Reranker 实现

```python
class LLMReranker(BaseReranker):
    def __init__(self, config: RerankerConfig, llm_client):
        self.config = config
        self.llm = llm_client
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3
    ) -> list[RerankResult]:
        """
        重排序
        """
        if not documents:
            return []
        
        # 构造 prompt
        docs_text = "\n".join([
            f"{i+1}. {doc.get('text', '')}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""请对以下文档与查询的相关性打分（0-10分）。

查询：{query}

文档：
{docs_text}

请输出每个文档的得分，格式为：1:分数, 2:分数, ..."""
        
        response = self.llm.generate(prompt)
        
        # 解析分数
        scores = self._parse_scores(response, len(documents))
        
        # 构建结果
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                doc_id=doc.get("id", str(i)),
                text=doc.get("text", ""),
                original_score=doc.get("score", 0),
                rerank_score=score,
                metadata=doc.get("metadata", {})
            ))
        
        # 排序
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return results[:top_k]
    
    def _parse_scores(self, response: str, expected_count: int) -> list[float]:
        """解析 LLM 返回的分数"""
        import re
        
        scores = []
        pattern = r'(\d+):(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, response)
        
        for match in matches:
            scores.append(float(match[1]))
        
        # 补齐缺失的分数
        while len(scores) < expected_count:
            scores.append(0.0)
        
        return scores[:expected_count]
```

---

## 4. 混合重排序

### 4.1 多路召回 + 重排序

```python
class HybridReranker:
    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        reranker: BaseReranker,
        fusion_method: str = "rrf"
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.fusion_method = fusion_method
    
    def search_and_rerank(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 3,
        candidate_size: int = 20
    ) -> list[RerankResult]:
        """
        多路召回 + 重排序
        """
        # 向量检索
        vector_results = self.vector_retriever.search(
            query_embedding, top_k=candidate_size
        )
        
        # BM25 检索
        bm25_results = self.bm25_retriever.search(
            query, top_k=candidate_size
        )
        
        # 融合
        if self.fusion_method == "rrf":
            fused = self._rrf_fusion(vector_results, bm25_results)
        else:
            fused = self._weighted_fusion(vector_results, bm25_results)
        
        # 取 top candidate_size 个候选
        candidates = fused[:candidate_size]
        
        # 转换为 dict 格式
        candidate_dicts = [
            {
                "id": r.id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in candidates
        ]
        
        # 重排序
        return self.reranker.rerank(query, candidate_dicts, top_k)
```

---

## 5. 性能优化

### 5.1 缓存

```python
from functools import lru_cache
import hashlib

class CachedReranker(BaseReranker):
    def __init__(self, reranker: BaseReranker, cache_size: int = 1000):
        self.reranker = reranker
        self.cache = {}
        self.cache_size = cache_size
    
    def _hash(self, query: str, doc_ids: tuple) -> str:
        """生成缓存键"""
        content = f"{query}:{','.join(doc_ids)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3
    ) -> list[RerankResult]:
        """带缓存的重排序"""
        # 生成缓存键
        doc_ids = tuple(doc.get("id", "") for doc in documents)
        cache_key = self._hash(query, doc_ids)
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 执行重排序
        results = self.reranker.rerank(query, documents, top_k)
        
        # 存入缓存
        if len(self.cache) >= self.cache_size:
            # 简单淘汰：清空一半
            keys = list(self.cache.keys())[:self.cache_size // 2]
            for k in keys:
                del self.cache[k]
        
        self.cache[cache_key] = results
        
        return results
```

### 5.2 批量处理

```python
class BatchReranker:
    def __init__(self, reranker: BaseReranker):
        self.reranker = reranker
    
    def batch_rerank(
        self,
        queries: list[str],
        documents_per_query: list[list[dict]],
        top_k: int = 3
    ) -> list[list[RerankResult]]:
        """
        批量重排序
        """
        results = []
        
        for query, docs in zip(queries, documents_per_query):
            result = self.reranker.rerank(query, docs, top_k)
            results.append(result)
        
        return results
```

---

## 6. 测试用例

### 6.1 功能测试

```python
def test_local_reranker():
    config = RerankerConfig(provider=RerankerProvider.LOCAL)
    reranker = LocalReranker(config)
    
    documents = [
        {"id": "1", "text": "退货流程：7天无理由"},
        {"id": "2", "text": "会员积分规则"},
        {"id": "3", "text": "如何申请退款"}
    ]
    
    results = reranker.rerank("怎么退货？", documents, top_k=2)
    
    assert len(results) == 2
    assert results[0].rerank_score >= results[1].rerank_score
```

### 6.2 性能测试

```python
def test_reranker_performance():
    config = RerankerConfig(provider=RerankerProvider.LOCAL)
    reranker = LocalReranker(config)
    
    documents = [
        {"id": str(i), "text": f"测试文档{i}"}
        for i in range(100)
    ]
    
    import time
    start = time.time()
    results = reranker.rerank("测试查询", documents, top_k=10)
    end = time.time()
    
    # 100 个文档重排序应该在 1 秒内完成
    assert end - start < 1
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
