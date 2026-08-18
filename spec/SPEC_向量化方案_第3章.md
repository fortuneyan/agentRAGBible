# SPEC_向量化方案_第3章

> 技术规格说明书 - 向量化方案

---

## 1. 章节概述

### 1.1 目标

定义文本向量化（Embedding）的技术规格，包括模型选择、维度配置、部署方式和质量优化。

### 1.2 范围

- Embedding 模型规范
- 向量维度标准
- 部署架构
- 质量优化策略

---

## 2. Embedding 模型规范

### 2.1 模型对比矩阵

| 模型 | 维度 | 中文效果 | 速度 | 内存 | 推荐场景 |
|-----|------|---------|-----|------|---------|
| text-embedding-3-small | 1536 | ⭐⭐⭐ | 快 | N/A | 国际化项目 |
| text-embedding-3-large | 3072 | ⭐⭐⭐⭐ | 中 | N/A | 追求效果 |
| bge-large-zh-v1.5 | 1024 | ⭐⭐⭐⭐⭐ | 中 | 1.3GB | 中文首选 |
| bge-m3 | 1024 | ⭐⭐⭐⭐⭐ | 慢 | 2.2GB | 多语言 |
| m3e-base | 768 | ⭐⭐⭐ | 快 | 400MB | 轻量部署 |

### 2.2 模型配置

```python
from dataclasses import dataclass
from enum import Enum

class EmbeddingProvider(Enum):
    LOCAL = "local"
    OPENAI = "openai"
    COHERE = "cohere"

@dataclass
class EmbeddingConfig:
    # 模型配置
    provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    model_name: str = "BAAI/bge-large-zh-v1.5"
    dimension: int = 1024
    
    # 本地部署配置
    device: str = "cpu"  # cpu | cuda
    batch_size: int = 32
    max_seq_length: int = 512
    
    # API 配置
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # 归一化
    normalize: bool = True
    
    # 缓存
    cache_enabled: bool = True
    cache_size: int = 10000
```

### 2.3 模型加载

```python
from sentence_transformers import SentenceTransformer
import openai

class EmbeddingEngine:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.config.provider == EmbeddingProvider.LOCAL:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
        elif self.config.provider == EmbeddingProvider.OPENAI:
            openai.api_key = self.config.api_key
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量向量化
        
        Args:
            texts: 文本列表
        
        Returns:
            list[list[float]]: 向量列表
        """
        if self.config.provider == EmbeddingProvider.LOCAL:
            return self._embed_local(texts)
        elif self.config.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(texts)
    
    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """本地模型向量化"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=True
        )
        return embeddings.tolist()
    
    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """OpenAI API 向量化"""
        # 分批处理（API 限制）
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = openai.embeddings.create(
                model=self.config.model_name,
                input=batch
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
```

---

## 3. 向量维度规范

### 3.1 维度标准

| 模型 | 维度 | 存储大小/向量 | 推荐索引类型 |
|-----|------|--------------|-------------|
| m3e-base | 768 | 3 KB | HNSW |
| bge-large-zh-v1.5 | 1024 | 4 KB | HNSW |
| text-embedding-3-small | 1536 | 6 KB | HNSW |
| text-embedding-3-large | 3072 | 12 KB | IVF + PQ |

### 3.2 维度选择策略

```python
def select_dimension(
    data_size: int,
    memory_limit: int,
    quality_requirement: str
) -> tuple[str, int]:
    """
    根据场景选择维度
    
    Args:
        data_size: 数据量
        memory_limit: 内存限制（GB）
        quality_requirement: 质量要求（low | medium | high）
    
    Returns:
        tuple: (模型名, 维度)
    """
    if quality_requirement == "high" and memory_limit > 8:
        return ("BAAI/bge-large-zh-v1.5", 1024)
    elif quality_requirement == "medium" or memory_limit > 4:
        return ("m3e-base", 768)
    else:
        return ("m3e-base", 768)
```

---

## 4. 部署架构

### 4.1 本地部署

```python
# 本地模型服务
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    embeddings = engine.embed(request.texts)
    return EmbedResponse(
        embeddings=embeddings,
        dimension=len(embeddings[0])
    )
```

### 4.2 Docker 部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
RUN pip install sentence-transformers fastapi uvicorn

# 下载模型
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.3 Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding
        image: embedding-service:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: embedding-service
spec:
  selector:
    app: embedding-service
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

---

## 5. 质量优化策略

### 5.1 文本预处理

```python
class TextPreprocessor:
    def preprocess(self, text: str, max_length: int = 512) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            max_length: 最大长度
        
        Returns:
            str: 处理后文本
        """
        # 1. 去除多余空白
        text = " ".join(text.split())
        
        # 2. 截断过长文本
        if len(text) > max_length:
            text = text[:max_length]
        
        # 3. 添加任务前缀（针对特定模型）
        text = f"为这个句子生成表示以用于检索中文文档：{text}"
        
        return text
```

### 5.2 批量处理优化

```python
class BatchEmbedder:
    def __init__(self, engine: EmbeddingEngine, batch_size: int = 32):
        self.engine = engine
        self.batch_size = batch_size
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量向量化（带进度显示）
        
        Args:
            texts: 文本列表
        
        Returns:
            list[list[float]]: 向量列表
        """
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, self.batch_size):
            batch = texts[i:i+self.batch_size]
            embeddings = self.engine.embed(batch)
            all_embeddings.extend(embeddings)
            
            progress = min(i + self.batch_size, total)
            print(f"进度: {progress}/{total} ({progress/total*100:.1f}%)")
        
        return all_embeddings
```

### 5.3 质量评估

```python
class EmbeddingEvaluator:
    def evaluate(
        self,
        queries: list[str],
        expected_docs: list[list[str]],
        embeddings: list[list[float]]
    ) -> dict:
        """
        评估 embedding 质量
        
        Args:
            queries: 查询列表
            expected_docs: 期望检索到的文档
            embeddings: 文档 embeddings
        
        Returns:
            dict: 评估指标
        """
        # 计算相似度矩阵
        query_embeddings = self.engine.embed(queries)
        
        # 计算 Recall@K
        recall_at_5 = 0
        recall_at_10 = 0
        
        for i, query_emb in enumerate(query_embeddings):
            similarities = [
                self._cosine_similarity(query_emb, doc_emb)
                for doc_emb in embeddings
            ]
            
            # 排序
            ranked_indices = sorted(
                range(len(similarities)),
                key=lambda x: similarities[x],
                reverse=True
            )
            
            # 计算召回率
            expected_set = set(expected_docs[i])
            
            # Recall@5
            top_5 = set([ranked_indices[j] for j in range(min(5, len(ranked_indices)))])
            if expected_set.intersection(top_5):
                recall_at_5 += 1
            
            # Recall@10
            top_10 = set([ranked_indices[j] for j in range(min(10, len(ranked_indices)))])
            if expected_set.intersection(top_10):
                recall_at_10 += 1
        
        return {
            "recall_at_5": recall_at_5 / len(queries),
            "recall_at_10": recall_at_10 / len(queries)
        }
```

---

## 6. 接口规范

### 6.1 EmbeddingEngine 接口

```python
from abc import ABC, abstractmethod

class BaseEmbeddingEngine(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """向量化"""
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """单条向量化"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取维度"""
        pass
```

---

## 7. 测试用例

### 7.1 功能测试

```python
def test_embed_single():
    engine = EmbeddingEngine(EmbeddingConfig())
    
    embedding = engine.embed_single("测试文本")
    
    assert len(embedding) == 1024
    assert all(isinstance(x, float) for x in embedding)

def test_embed_batch():
    engine = EmbeddingEngine(EmbeddingConfig())
    
    texts = ["文本1", "文本2", "文本3"]
    embeddings = engine.embed(texts)
    
    assert len(embeddings) == 3
    assert all(len(e) == 1024 for e in embeddings)
```

### 7.2 性能测试

```python
def test_embed_performance():
    engine = EmbeddingEngine(EmbeddingConfig())
    
    texts = ["测试文本"] * 1000
    
    import time
    start = time.time()
    embeddings = engine.embed(texts)
    end = time.time()
    
    # 1000 条文本应该在 10 秒内完成
    assert end - start < 10
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
