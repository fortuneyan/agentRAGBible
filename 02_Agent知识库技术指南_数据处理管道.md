# 第 2 章：数据处理管道

> 这是整个系统的地基。我之前按固定 512 tokens 切分，结果一个完整的 API 文档被切成三段，检索出来全是断章取义。

---

## 数据源接入

最常见的数据源：

| 数据源类型 | 难度 | 推荐工具 |
|-----------|------|---------|
| PDF | ⭐⭐⭐ | PyMuPDF, pdfplumber, Unstructured |
| Word | ⭐⭐ | python-docx, Unstructured |
| HTML | ⭐⭐ | BeautifulSoup, Readability |
| Markdown | ⭐ | 直接读 |
| 数据库 | ⭐⭐ | SQLAlchemy + 自定义 chunking |
| API 文档 | ⭐⭐ | Swagger 解析 + 分块 |

### 工具 / SOP 数据源（行动 Agent 专用）

当知识库要支撑"用自然语言驱动 CLI 操作"的行动 Agent 时，除了文档，还需要把**工具的用法与操作流程**也作为可检索知识入库。

| 数据源类型 | 内容说明 | 入库策略 |
|-----------|---------|---------|
| CLI 工具 `--help` | 命令用法、参数列表、示例 | 结构化解析 + 元数据标注 |
| 操作流程文档（SOP） | 多步骤操作的标准流程 | 分块 + 步骤顺序元数据 |
| 工具注册表（YAML/JSON） | 工具名称、描述、风险等级、白名单路径 | 直接入库为结构化元数据 |

**工具元数据 Schema（入库必须携带）**：

```python
# 所有工具入库时必须携带此结构
TOOL_SCHEMA = {
    "type": "tool",                      # 固定值，用于检索过滤
    "tool_id": "generate_report",
    "name": "generate_report",
    "description": "生成月度销售报表",
    "command_template": "python report.py --data {data} --month {month}",
    "parameters": [
        {"name": "data", "type": "file_path", "required": True, "allowed_paths": ["/data/"]},
        {"name": "month", "type": "string", "required": True, "pattern": r"^\d{4}-\d{2}$"}
    ],
    "risk_level": "low",                 # low | medium | high
    "requires_confirmation": False,
    "output_format": "json",             # 强制要求 CLI 支持 --json
    "timeout_seconds": 30,
    "tags": ["reporting", "internal"]
}
```

> 这部分与第 14 章（行动 Agent）直接联动：工具块通过 `type=tool` 被检索出来，供意图路由与行动规划使用。

### PDF 处理的坑

PDF 是最头疼的格式。我踩过的坑：

1. **扫描版 PDF**：文字是图片，需要 OCR
2. **表格 PDF**：普通解析器会把表格拆成碎片
3. **双栏排版**：解析顺序会乱

推荐方案：

```python
# 简单场景：PyMuPDF
import fitz  # PyMuPDF

def extract_pdf(file_path: str) -> list[str]:
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return pages

# 复杂场景：Unstructured
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("document.pdf")
```

### 多格式解析细节

PDF 之外，企业知识库最常见的是网页、Word、表格和结构化数据。解析质量直接决定下游分块与检索的上限——**解析错了，后面再怎么调都没用**。

**HTML / 网页**：原始 HTML 里混着导航、广告、页脚、脚本。直接 `get_text()` 会污染知识库。优先用 `trafilatura` 或 `readability-lxml` 抽取"正文主干"，再取文本：

```python
import trafilatura

def extract_html(url_or_path: str) -> str:
    # 支持本地文件或 URL；自动去导航/广告/页脚，保留段落结构
    raw = trafilatura.load(url_or_path)
    return trafilatura.extract(raw, include_comments=False,
                               include_tables=True) or ""
```

**Word（.docx）**：注意表格和批注。`python-docx` 能按段落、表格分别提取，避免把表格拆散：

```python
from docx import Document

def extract_docx(path: str) -> list[str]:
    doc = Document(path)
    blocks = []
    for p in doc.paragraphs:
        if p.text.strip():
            blocks.append(p.text)
    for table in doc.tables:                  # 表格单独成块，保留结构
        rows = [" | ".join(c.text for c in r.cells) for r in table.rows]
        blocks.append("\n".join(rows))
    return blocks
```

**Excel / CSV**：按"一行或一张表"作为语义单元，别按字符硬切——否则一条记录被劈成两半：

```python
import pandas as pd

def extract_excel(path: str) -> list[str]:
    chunks = []
    for sheet_name, df in pd.read_excel(path, sheet_name=None).items():
        # 每 N 行合并成一个块，保留表头
        for i in range(0, len(df), 20):
            chunk = df.iloc[i:i+20].to_markdown(index=False)
            chunks.append(f"【{sheet_name}】\n{chunk}")
    return chunks
```

**JSON / 结构化数据**：把字段展开成可读文本，关键字段写入元数据便于过滤：

```python
import json

def extract_json(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    # 把对象拍平为 "字段: 值" 文本，原 key 进元数据
    chunks = []
    for item in (data if isinstance(data, list) else [data]):
        text = "\n".join(f"{k}: {v}" for k, v in item.items())
        chunks.append({"text": text, "metadata": {"keys": list(item.keys())}})
    return chunks
```

**扫描版 PDF / 图片型文档**：先用文字密度检测（见下），命中就走 OCR（PaddleOCR / tesseract）。注意 OCR 会引入错字，金融/医疗等高精度场景建议人工复核或上版面识别模型：

```python
import fitz

def is_scanned_pdf(path: str, threshold: float = 0.1) -> bool:
    """文字密度（字符数/页面面积）低于阈值视为扫描版"""
    doc = fitz.open(path)
    for page in doc:
        density = len(page.get_text()) / (page.rect.width * page.rect.height)
        if density < threshold:
            return True
    return False
```

**表格提取（PDF/HTML）**：表格是知识库里信息密度最高的部分，普通解析器会把它拆成碎片。用 `camelot` / `pdfplumber` 抽 PDF 表格、用 `pandas.read_html` 抽网页表格，统一转成 Markdown 表格保留行列结构。

**版面感知解析（复杂排版救星）**：双栏、图文混排、带公式的技术手册，传统按阅读顺序解析会乱。可上 **MinerU / Marker** 这类版面感知工具（或多模态大模型），自动还原标题层级、表格、公式与阅读顺序，再交给分块。这是目前复杂中文文档解析效果最好的路线。

**解析质量校验（别跳过）**：入库前用几个简单指标拦掉"假解析"：

- 提取字符数 / 页数 是否过低（可能整本都是图）；
- 空页比例、乱码率（`\ufffd` 占比）；
- 关键字段（如"退货"）是否真的命中。
校验不通过的文档隔离告警，而不是默默进库。

---

## 文档分块策略

**这是最容易踩坑的地方。**

### 方法一：固定长度分块

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
)
```

**优点**：简单，可预测
**缺点**：可能切断语义

### 方法二：按文档结构分块

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
```

**优点**：保持语义完整
**缺点**：需要文档有明确结构

### 方法三：语义分块

```python
# 用 embedding 找语义断点
import numpy as np

def semantic_split(text: str, embedder, threshold: float = 0.5):
    sentences = text.split("。")
    embeddings = embedder.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < threshold:
            chunks.append("。".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    chunks.append("。".join(current_chunk))
    return chunks
```

**优点**：语义最完整
**缺点**：计算成本高

### 我的建议

- 有明确结构的文档（Markdown、HTML）：用结构分块
- 通用文档：用 RecursiveCharacterTextSplitter
- 追求极致效果：用语义分块

---

## 分块参数调优

### chunk_size

```python
# 太小：信息碎片化
chunk_size = 128  # 可能切断完整句子

# 太大：检索不精准
chunk_size = 2048  # 一个块包含太多信息

# 推荐范围
chunk_size = 512  # 中文
chunk_size = 1024  # 英文
```

### chunk_overlap

```python
# 没有重叠：跨块信息丢失
chunk_overlap = 0  # 不推荐

# 重叠太大：重复信息
chunk_overlap = 256  # 浪费存储

# 推荐
chunk_overlap = 64  # 约 10-20% 的 chunk_size
```

---

## 元数据标注

分块的时候，顺手把元数据加上：

```python
chunk_metadata = {
    "source": "product_doc_v2.pdf",
    "page": 15,
    "section": "退货流程",
    "last_updated": "2024-01-15",
    "author": "张三",
    "doc_type": "policy"  # 可以用来做过滤
}
```

**为什么加元数据？**

- 后面做检索过滤
- 引用标注（告诉用户信息来源）
- 版本控制
- 数据审计

---

## 完整的数据处理流程

```python
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataPipeline:
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def process(self, file_path: str) -> list[dict]:
        # 1. 加载文档
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # 2. 分块
        chunks = self.splitter.split_documents(documents)
        
        # 3. 添加元数据
        processed = []
        for i, chunk in enumerate(chunks):
            processed.append({
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return processed

# 使用
pipeline = DataPipeline()
chunks = pipeline.process("产品手册.pdf")
print(f"生成 {len(chunks)} 个文档块")
```

---

## 数据清洗

别忽略这步。我之前直接导入脏数据，结果检索效果很差。但清洗也有"坑"——**洗太狠会把表格、代码、公式洗没**。目标不是"越干净越好"，而是"去掉噪声、保留语义结构"。

### 基础清洗

```python
import re

def clean_text(text: str) -> str:
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 去除页眉页脚
    text = re.sub(r'第\d+页', '', text)
    text = re.sub(r'©.*?\d{4}', '', text)

    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff。！？，、；：""''（）]', '', text)

    return text.strip()
```

### 深度清洗要点

1. **保留表格与代码块**：清洗前先用占位符把 ` ```代码块``` `、Markdown 表格、LaTeX 公式整体保护起来，清洗完再还原——否则分块会把一段 SQL 或一张配置表切碎，检索出来全是半句。
2. **跨文档 / 块级去重**：同一份制度文件在多个部门各存一份、或爬虫重复抓取，会产生大量 near-duplicate chunk。入库前用 **SimHash / MinHash** 或 embedding 相似度做近重复检测，合并或丢弃，既省存储又避免检索时同一意思反复命中。
3. **编码与乱码**：统一 `utf-8`；检测并修复 `GBK`/`BIG5` 误读；剔除替换字符 `\ufffd` 占比过高的"假解析"片段；必要时做全角/半角、繁简归一。
4. **噪声模板**：目录、页眉页脚、水印、批注、免责声明、"我们使用 Cookie" 横幅、导航链接——这些对问答毫无价值，应识别后剔除。
5. **Markdown 规范化**：统一标题层级、列表、链接格式，让下游分块与引用更稳定。

```python
import re

class TextCleaner:
    _CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)   # 代码块
    _TABLE = re.compile(r"\|.*\|", re.MULTILINE)          # 简易表格行
    _REPL = "�"                                            # 乱码替换符

    def clean(self, text: str) -> str:
        # 1) 先保护代码块与表格
        codes = self._CODE_FENCE.findall(text)
        tables = self._TABLE.findall(text)
        text = self._CODE_FENCE.sub("[CODE]", text)
        text = self._TABLE.sub("[TABLE]", text)

        # 2) 去噪声
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'第\d+页', '', text)
        text = re.sub(r'©.*?\d{4}', '', text)
        text = re.sub(r'我们使用 Cookie.*?确定', '', text)   # Cookie 横幅
        text = re.sub(r'[^\w\s\u4e00-\u9fff。！？，、；：""''（）\[\]【】]', '', text)

        # 3) 还原结构（保持语义完整）
        for c in codes:
            text = text.replace("[CODE]", c, 1)
        for t in tables:
            text = text.replace("[TABLE]", t, 1)
        return text.strip()

    def is_garbled(self, text: str, ratio: float = 0.05) -> bool:
        """乱码率超阈值则判定为假解析，应隔离告警"""
        return text.count(self._REPL) / max(len(text), 1) > ratio
```

> **隐私挂钩**：如果文档可能含个人信息（客户名单、合同、简历），清洗阶段就要调用脱敏（见第 13 章 13.2）。把 `mask_pii()` 串在 `TextCleaner.clean` 之后，是性价比最高的合规防线——**数据还没进向量库，PII 就已经被掩码了**。

---

## 入库（Ingestion）工程

前面做完"加载 → 解析 → 清洗 → 分块 → 标元数据"，还差**最后一步：把 chunk 写进向量库**。这一步正文里一直没讲，但它决定了知识库"能不能稳定更新、会不会越存越乱"。

> 注意：向量化（调 Embedding 模型）在第 3 章。入库 = **分块文本 → 调 Embedding 拿到向量 → 带元数据 upsert 进向量库**。

### 去重与幂等写入

同一份文件被重复导入，绝不能产生重复向量。用"内容哈希 + 稳定 ID"保证幂等：

- `content_hash = sha256(归一化后的文本)`
- `chunk_id = uuid5(namespace, source + page + content_hash)`

这样同一内容无论导入几次，ID 都一致，向量库做 upsert（存在则覆盖）即可。

```python
import hashlib, uuid

def make_chunk_id(source: str, page: int, text: str) -> str:
    norm = text.strip().replace("\s+", " ")
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{page}#{h}"))
```

### 增量更新（文档改了怎么办）

这是生产环境的日常，但最容易写错。正确姿势是按 `source` 做"先删后插"：

1. 文档内容变化 → 算出该 `source` 下所有新 chunk_id；
2. 删除向量库中 `source == 该文件` 的旧 chunk（**按 source 维度删，不是全表清空**）；
3. 写入新 chunk；
4. 打 `version` 元数据、清掉相关查询缓存（见第 9 章）。

> 对应修订清单 **UP-201**。软删（标记 `deleted_at`）还是物理删，取决于你是否需要"历史版本回溯"——合规审计场景建议软删。

### 批量写入与限流

Embedding API 有 QPS 限制。把 chunk 分批（如每批 100 条）调用 Embedding，再批量 upsert 进向量库，避免触发限流与超时：

```python
def ingest(chunks: list[dict], embedder, vector_db, batch=100):
    for i in range(0, len(chunks), batch):
        group = chunks[i:i+batch]
        texts = [c["text"] for c in group]
        vectors = embedder.encode(texts)           # 批量化，省 API 调用
        payloads = [{
            "text": c["text"],
            "source": c["metadata"].get("source"),
            "chunk_id": make_chunk_id(c["metadata"]["source"],
                                      c["metadata"].get("page", 0), c["text"]),
            **c["metadata"],
        } for c in group]
        vector_db.upsert(vectors=vectors, payloads=payloads)  # 向量库原生 batch upsert
```

### 失败重试与断点续传

大批量入库一定会遇到网络抖动。记录"已成功入库的文件/offset"，失败重跑时跳过已完成的，避免重复劳动：

- 用一张 `ingestion_state` 表记录 `{file, status, chunks_done}`；
- 异常捕获后 `continue`，并把失败文件写入告警队列人工排查；
- 支持 `--resume` 从断点继续。

### 写入校验（别写完就走）

入库后必须回读校验，否则"以为进库了其实没进"：

- 条数：`len(入库chunk)` == 向量库该 `source` 下 count；
- 维度：每个向量的 dim == Embedding 模型输出维度；
- payload：抽样回读，确认 `text` 与关键元数据（source/page）完整无空。

```python
def verify(source: str, expected: int, vector_db) -> None:
    got = vector_db.count(filter={"source": source})
    assert got == expected, f"入库条数不符: 期望 {expected}, 实际 {got}"
```

### 完整入库流程串起来

```python
class IngestionPipeline:
    def __init__(self, cleaner, chunker, embedder, vector_db):
        self.cleaner, self.chunker = cleaner, chunker
        self.embedder, self.vector_db = embedder, vector_db

    def run(self, file_path: str) -> int:
        # 1) 解析 + 清洗（含隐私脱敏，见第 13 章）
        text = self.cleaner.clean(load_text(file_path))
        # 2) 分块
        chunks = self.chunker.chunk(text, {"source": file_path})
        # 3) 去重后入库（先按 source 删旧，再批量 upsert）
        self.vector_db.delete(filter={"source": file_path})
        ingest(chunks, self.embedder, self.vector_db)
        verify(file_path, len(chunks), self.vector_db)
        return len(chunks)
```

---

## 本章小结

- 数据处理是地基，别跳过
- **解析按格式选型**：PDF/Word 用专用库，网页用正文抽取，表格/复杂排版上版面感知解析（MinerU/Marker），并用质量校验拦掉假解析
- 分块策略要根据文档类型选择（结构分块 / Recursive / 语义分块）
- chunk_size 512、chunk_overlap 64 是不错的起点
- 元数据一定要加
- **数据清洗要"保结构"**：保留代码块与表格，做近重复检测、编码修复、噪声过滤；含个人信息的文档在清洗阶段就脱敏（见第 13 章）
- **入库是独立工程**：去重幂等、增量更新（按 source 先删后插）、批量限流、失败重试断点续传、写入校验，一步都不能省

下一章，我们讲向量化——怎么把文本变成计算机能"理解"的向量。

---

*说实话，数据处理这步很枯燥，但效果提升最明显。我之前花两周调检索算法，不如花两天清洗数据效果好。*
