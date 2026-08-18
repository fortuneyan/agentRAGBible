# SPEC_完整实现示例_第8章

> 技术规格说明书 - 完整实现示例

> 🔧 **v2 修订（2026-07）**：已应用 UP-003（VectorStore 改 UUID5 幂等 ID + 新增 delete_by_source）、UP-004（Reranker 不修改入参）、UP-005（Prompt 用 replace 而非 format）。详见 `SPEC_更新修订清单_v2.md`。

---

## 1. 章节概述

### 1.1 目标

定义 Agent 知识库完整实现的代码规范、模块结构和集成方式。

### 1.2 范围

- 项目结构规范
- 模块接口定义
- 集成实现
- 配置管理

---

## 2. 项目结构

### 2.1 目录结构

```
agent_kb/
├── __init__.py
├── config.py              # 配置定义
├── pipeline.py            # 数据处理管道
├── embedder.py            # 向量化引擎
├── vector_store.py        # 向量数据库
├── retriever.py           # 检索引擎
├── reranker.py            # 重排序
├── generator.py           # 生成器
├── kb.py                  # 知识库主类
├── main.py                # 入口
└── tests/                 # 测试
    ├── __init__.py
    ├── test_pipeline.py
    ├── test_embedder.py
    ├── test_retriever.py
    └── test_kb.py
```

### 2.2 模块依赖关系

```
config.py
    ↓
pipeline.py → embedder.py → vector_store.py
    ↓
retriever.py → reranker.py → generator.py
    ↓
kb.py → main.py
```

---

## 3. 配置规范

### 3.1 配置类

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-zh-v1.5"
    dimension: int = 1024
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True

@dataclass
class VectorStoreConfig:
    provider: str = "qdrant"  # qdrant | milvus | chroma
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "agent_kb"

@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-large"
    device: str = "cpu"
    top_k: int = 3

@dataclass
class GeneratorConfig:
    model: str = "gpt-4"
    temperature: float = 0
    max_tokens: int = 1024
    api_key: Optional[str] = None

@dataclass
class PipelineConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    clean_text: bool = True

@dataclass
class KBConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
```

---

## 4. 模块实现

### 4.1 数据管道

```python
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def process(self, file_path: str) -> list[dict]:
        """处理单个文件"""
        # 加载
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # 分块
        chunks = self.splitter.split_documents(documents)
        
        # 处理
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "source": file_path,
                    "chunk_index": i
                }
            })
        
        return result
```

### 4.2 向量化引擎

```python
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = SentenceTransformer(
            config.model_name,
            device=config.device
        )
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize
        )
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> list[float]:
        """单条向量化"""
        return self.embed([text])[0]
```

### 4.3 向量数据库

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class VectorStore:
    def __init__(self, config: VectorStoreConfig, dimension: int):
        self.config = config
        self.client = QdrantClient(host=config.host, port=config.port)
        self.dimension = dimension
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保集合存在"""
        try:
            self.client.get_collection(self.config.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
    
    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        """添加文档"""
        points = [
            PointStruct(
                id=i,
                vector=embedding,
                payload={"text": text, **metadata}
            )
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas))
        ]
        self.client.upsert(collection_name=self.config.collection_name, points=points)
    
    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        """检索"""
        results = self.client.search(
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

### 4.4 检索引擎

```python
class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """检索"""
        query_embedding = self.embedder.embed_single(query)
        return self.vector_store.search(query_embedding, top_k)
```

### 4.5 重排序

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = CrossEncoder(config.model_name)
    
    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        """重排序"""
        if not documents:
            return []
        
        texts = [doc["text"] for doc in documents]
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs)
        
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        return documents[:top_k]
```

### 4.6 生成器

```python
import openai

class Generator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        if config.api_key:
            openai.api_key = config.api_key
    
    def generate(self, query: str, context: str) -> str:
        """生成回答"""
        system_prompt = f"""你是一个专业的客服助手。请根据以下参考资料回答用户的问题。

规则：
1. 只根据提供的参考资料回答
2. 如果没有相关信息，请说明
3. 保持简洁

参考资料：
{context}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户问题：{query}"}
        ]
        
        response = openai.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
```

---

## 5. 知识库主类

```python
class AgentKnowledgeBase:
    def __init__(self, config: KBConfig = None):
        self.config = config or KBConfig()
        
        # 初始化组件
        self.pipeline = DataPipeline(self.config.pipeline)
        self.embedder = Embedder(self.config.embedding)
        self.vector_store = VectorStore(
            self.config.vector_store,
            self.config.embedding.dimension
        )
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.reranker = Reranker(self.config.reranker)
        self.generator = Generator(self.config.generator)
    
    def ingest(self, file_path: str):
        """导入文档"""
        chunks = self.pipeline.process(file_path)
        
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(texts, embeddings, metadatas)
        
        print(f"导入完成：{len(chunks)} 个文档块")
    
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """检索"""
        return self.retriever.retrieve(query, top_k)
    
    def answer(self, query: str) -> str:
        """完整问答"""
        # 检索
        docs = self.retrieve(query)
        
        # 重排序
        docs = self.reranker.rerank(query, docs, self.config.reranker.top_k)
        
        # 拼接上下文
        context = "\n\n".join([
            f"[{i+1}] {d['text']}"
            for i, d in enumerate(docs)
        ])
        
        # 生成
        return self.generator.generate(query, context)
```

---

## 6. 入口文件

```python
def main():
    # 初始化
    kb = AgentKnowledgeBase()
    
    # 导入文档
    kb.ingest("产品手册.pdf")
    kb.ingest("FAQ.md")
    
    # 问答循环
    while True:
        query = input("\n用户：")
        if query == "quit":
            break
        
        answer = kb.answer(query)
        print(f"\n助手：{answer}")

if __name__ == "__main__":
    main()
```

---

## 7. 测试规范

### 7.1 单元测试

```python
import pytest

def test_pipeline():
    config = PipelineConfig()
    pipeline = DataPipeline(config)
    
    chunks = pipeline.process("test.pdf")
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)

def test_embedder():
    config = EmbeddingConfig()
    embedder = Embedder(config)
    
    embedding = embedder.embed_single("测试文本")
    assert len(embedding) == 1024
```

### 7.2 集成测试

```python
def test_kb_e2e():
    kb = AgentKnowledgeBase()
    
    # 导入
    kb.ingest("test.pdf")
    
    # 检索
    docs = kb.retrieve("测试问题")
    assert len(docs) > 0
    
    # 问答
    answer = kb.answer("测试问题")
    assert len(answer) > 0
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
