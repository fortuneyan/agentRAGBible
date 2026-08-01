# 第 8 章：完整实现示例

> 把前面所有东西串起来，给你一个能直接跑的代码。

---

## 项目结构

```
agent_kb/
├── __init__.py
├── config.py          # 配置
├── pipeline.py        # 数据处理管道
├── embedder.py        # 向量化
├── vector_store.py    # 向量数据库
├── retriever.py       # 检索
├── reranker.py        # 重排序
├── generator.py       # 生成
├── kb.py              # 知识库主类
└── main.py            # 入口
├── controller.py      # 新增：意图分类、规划、执行调度（第 14 章）
├── executor.py        # 新增：安全 CLI 执行器（第 14 章）
└── tools/             # 新增：工具注册目录
    ├── registry.json  # 工具白名单与元数据
    └── schemas/       # 各工具的详细 Schema
```

> 上表中的 `controller.py` / `executor.py` / `tools/` 是接入"自然语言驱动 CLI 操作"（第 14 章）时新增的。原有 RAG 链路（pipeline → embedder → vector_store → retriever → reranker → generator）保持不变。

新增依赖（行动 Agent，更新 `requirements.txt`）：

```text
# 原有依赖保持不变，新增以下
pyyaml>=6.0
jsonschema>=4.17
```

---

## 完整代码

### config.py

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Config:
    # Embedding
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024

    # Vector DB
    vector_db_host: str = "localhost"
    vector_db_port: int = 6333
    collection_name: str = "agent_kb"

    # 文档 ID 策略
    #   "uuid5"     : 基于文本内容的稳定 UUID，幂等导入，重导不膨胀（推荐）
    #   "sequential": 自增整数，仅用于一次性导入的 demo（重导会覆盖！）
    id_strategy: Literal["uuid5", "sequential"] = "uuid5"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-large"

    # LLM
    llm_model: str = "gpt-4"
    llm_temperature: float = 0

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_k: int = 3
```

> ⚠️ **重要**：早期版本 `VectorStore.add` 用自增 `id=i`，每次 ingest 都从 0 开始编号，第二次导入会**静默覆盖**第一次的数据（见 UP-003）。现已改为基于内容 hash 的稳定 UUID，重导幂等。

### pipeline.py

```python
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

class DataPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def process(self, file_path: str) -> list[dict]:
        # 加载
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # 分块
        chunks = self.splitter.split_documents(documents)
        
        # 处理
        processed = []
        for i, chunk in enumerate(chunks):
            processed.append({
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "source": file_path,
                    "chunk_index": i
                }
            })
        
        return processed
```

### embedder.py

```python
from sentence_transformers import SentenceTransformer
from config import Config

class Embedder:
    def __init__(self, config: Config):
        self.model = SentenceTransformer(config.embedding_model)
        self.dim = config.embedding_dim
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

### vector_store.py

```python
# requires: qdrant-client>=1.7
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config import Config

# 固定命名空间，保证同一文本 → 同一 UUID（幂等去重）
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "agent-kb")

class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.client = QdrantClient(host=config.vector_db_host, port=config.vector_db_port)
        self.collection = config.collection_name
        self.dim = config.embedding_dim

        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )

    def _make_id(self, text: str, seq: int) -> str | int:
        """根据 id_strategy 生成稳定 ID（幂等）或自增 ID（仅一次性 demo）"""
        if self.config.id_strategy == "uuid5":
            # 同一文本永远映射到同一 UUID → 重导不会膨胀，也不会覆盖别人
            return str(uuid.uuid5(_UUID_NAMESPACE, text))
        return seq  # sequential：调用方需自行保证不重复导入

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        points = [
            PointStruct(
                id=self._make_id(text, i),
                vector=embedding,
                payload={"text": text, **metadata}
            )
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def delete_by_source(self, source: str):
        """按来源文件删除——文档更新/增量重建时用（见 UP-201）"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k
        )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results
        ]
```

### reranker.py

```python
import copy
from sentence_transformers import CrossEncoder
from config import Config

class Reranker:
    def __init__(self, config: Config):
        self.model = CrossEncoder(config.reranker_model)

    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
        """
        重排序。注意：不修改入参 documents（避免副作用，见 UP-004）。
        返回的是带 rerank_score 的新列表。
        """
        if not documents:
            return []

        texts = [doc["text"] for doc in documents]
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs)

        # 深拷贝后再写分数 + 排序，绝不污染调用方的 list
        scored = []
        for doc, score in zip(documents, scores):
            new_doc = copy.deepcopy(doc)
            new_doc["rerank_score"] = float(score)
            scored.append(new_doc)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
```

### generator.py

```python
# requires: openai>=1.10
import openai
from config import Config

SYSTEM_PROMPT = """你是一个专业的客服助手。请根据以下参考资料回答用户的问题。

规则：
1. 只根据提供的参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，请直接说"我无法从现有资料中找到答案"
3. 回答时标注信息来源，格式为 [来源：文档名]
4. 保持回答简洁，控制在 200 字以内

参考资料：
__CONTEXT__"""

class Generator:
    def __init__(self, config: Config):
        self.model = config.llm_model
        self.temperature = config.llm_temperature

    def _build_system_content(self, context: str) -> str:
        # ⚠️ 不用 str.format：检索结果里常含 { }（JSON/代码），format 会报 KeyError（见 UP-005）。
        # 用 replace 安全替换命名占位符。
        return SYSTEM_PROMPT.replace("__CONTEXT__", context)

    def generate(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": self._build_system_content(context)},
            {"role": "user", "content": f"用户问题：{query}"}
        ]

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return response.choices[0].message.content

    def generate_stream(self, query: str, context: str):
        messages = [
            {"role": "system", "content": self._build_system_content(context)},
            {"role": "user", "content": f"用户问题：{query}"}
        ]

        stream = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### kb.py

```python
from config import Config
from pipeline import DataPipeline
from embedder import Embedder
from vector_store import VectorStore
from reranker import Reranker
from generator import Generator

class AgentKnowledgeBase:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.pipeline = DataPipeline(self.config)
        self.embedder = Embedder(self.config)
        self.vector_store = VectorStore(self.config)
        self.reranker = Reranker(self.config)
        self.generator = Generator(self.config)
    
    def ingest(self, file_path: str):
        """导入文档"""
        chunks = self.pipeline.process(file_path)
        
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(texts, embeddings, metadatas)
        
        print(f"导入完成：{len(chunks)} 个文档块")
    
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """检索"""
        top_k = top_k or self.config.retrieval_top_k
        query_embedding = self.embedder.embed_single(query)
        return self.vector_store.search(query_embedding, top_k)
    
    def answer(self, query: str) -> str:
        """完整问答"""
        # 检索
        docs = self.retrieve(query)
        
        # 重排序
        docs = self.reranker.rerank(query, docs, self.config.rerank_top_k)
        
        # 拼接上下文
        context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(docs)])
        
        # 生成
        return self.generator.generate(query, context)
    
    def answer_stream(self, query: str):
        """流式问答"""
        docs = self.retrieve(query)
        docs = self.reranker.rerank(query, docs, self.config.rerank_top_k)
        context = "\n\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(docs)])
        
        return self.generator.generate_stream(query, context)
```

### main.py

```python
from kb import AgentKnowledgeBase

def main():
    # 初始化
    kb = AgentKnowledgeBase()
    
    # 导入文档
    kb.ingest("产品手册.pdf")
    kb.ingest("FAQ.md")
    
    # 问答
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

## 使用示例

```python
# 初始化知识库
kb = AgentKnowledgeBase()

# 导入文档
kb.ingest("产品手册.pdf")
kb.ingest("FAQ.md")
kb.ingest("退货政策.docx")

# 问答
answer = kb.answer("怎么退货？")
print(answer)

# 流式输出
for chunk in kb.answer_stream("会员积分怎么算？"):
    print(chunk, end="", flush=True)
```

---

## 本章小结

- 这是一个最小可用的实现
- 生产环境还需要加错误处理、日志、监控
- 代码结构清晰，方便扩展
- **v2 修订要点**：文档 ID 用内容 hash（UUID5）保证幂等（UP-003）；Reranker 不修改入参（UP-004）；Prompt 模板用 `replace` 而非 `format`，避免检索结果里的 `{}` 导致崩溃（UP-005）

下一章，我们讲性能优化。

---

*代码能跑只是开始，跑得好才是真本事。*
