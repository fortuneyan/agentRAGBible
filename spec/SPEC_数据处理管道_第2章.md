# SPEC_数据处理管道_第2章

> 技术规格说明书 - 数据处理管道

---

## 1. 章节概述

### 1.1 目标

定义文档数据处理管道的技术规格，包括文档加载、分块策略、元数据提取和数据清洗。

### 1.2 范围

- 文档格式支持
- 分块算法规范
- 元数据标准
- 数据清洗流程

---

## 2. 文档格式支持

### 2.1 支持格式矩阵

| 格式 | 扩展名 | 解析库 | 优先级 |
|-----|--------|--------|--------|
| PDF | .pdf | PyMuPDF / Unstructured | P0 |
| Word | .docx | python-docx | P0 |
| Markdown | .md | 直接解析 | P1 |
| HTML | .html | BeautifulSoup | P1 |
| 纯文本 | .txt | 直接读取 | P1 |
| JSON | .json | 自定义解析 | P2 |
| CSV | .csv | pandas | P2 |

### 2.2 PDF 解析规范

```python
@dataclass
class PDFParsingConfig:
    # 解析方式
    use_ocr: bool = False  # 是否使用 OCR
    ocr_lang: str = "ch"  # OCR 语言
    
    # 页面处理
    extract_images: bool = False  # 是否提取图片
    extract_tables: bool = True  # 是否提取表格
    
    # 文本清理
    remove_headers: bool = True  # 移除页眉
    remove_footers: bool = True  # 移除页脚
    remove_page_numbers: bool = True  # 移除页码
    
    # 表格处理
    table_format: str = "markdown"  # markdown | csv | json
```

### 2.3 检测扫描版 PDF

```python
import fitz

def is_scanned_pdf(file_path: str, threshold: float = 0.1) -> bool:
    """
    检测是否为扫描版 PDF
    
    Args:
        file_path: PDF 文件路径
        threshold: 文字密度阈值（文字字符数 / 页面面积）
    
    Returns:
        bool: 是否为扫描版
    """
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text()
        page_area = page.rect.width * page.rect.height
        text_density = len(text) / page_area
        
        if text_density < threshold:
            return True
    return False
```

---

## 3. 分块策略规范

### 3.1 分块参数标准

```python
@dataclass
class ChunkingConfig:
    # 基础参数
    chunk_size: int = 512  # 目标块大小（tokens）
    chunk_overlap: int = 64  # 重叠大小（tokens）
    
    # 分隔符优先级
    separators: list[str] = field(default_factory=lambda: [
        "\n\n",  # 段落
        "\n",    # 换行
        "。",    # 中文句号
        "！",    # 中文感叹号
        "？",    # 中文问号
        ".",     # 英文句号
        "!",     # 英文感叹号
        "?",     # 英文问号
        "；",    # 中文分号
        ";",     # 英文分号
        "，",    # 中文逗号
        ",",     # 英文逗号
        " ",     # 空格
    ])
    
    # 长度控制
    min_chunk_size: int = 100  # 最小块大小
    max_chunk_size: int = 1024  # 最大块大小
```

### 3.2 分块算法

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )
    
    def chunk(self, text: str, metadata: dict = None) -> list[dict]:
        """
        文档分块
        
        Args:
            text: 文档文本
            metadata: 文档元数据
        
        Returns:
            list[dict]: 分块结果
        """
        chunks = self.splitter.split_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            if len(chunk) < self.config.min_chunk_size:
                continue
            
            result.append({
                "text": chunk,
                "index": i,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return result
```

### 3.3 语义分块

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticChunker:
    def __init__(
        self, 
        embedder: SentenceTransformer,
        threshold: float = 0.5,
        min_chunk_size: int = 100
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str) -> list[str]:
        """
        语义分块：在语义断点处分割
        
        Args:
            text: 文档文本
        
        Returns:
            list[str]: 分块结果
        """
        # 按句子分割
        sentences = text.split("。")
        
        if len(sentences) <= 1:
            return [text]
        
        # 生成 embedding
        embeddings = self.embedder.encode(sentences)
        
        # 计算相邻句子相似度
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(
                embeddings[i-1], 
                embeddings[i]
            )
            
            if similarity < self.threshold:
                chunk_text = "。".join(current_chunk) + "。"
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        # 添加最后一个块
        if current_chunk:
            chunk_text = "。".join(current_chunk) + "。"
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 4. 元数据规范

### 4.1 标准元数据字段

```python
@dataclass
class DocumentMetadata:
    # 必需字段
    source: str  # 来源文件
    chunk_index: int  # 块索引
    total_chunks: int  # 总块数
    
    # 可选字段
    page: Optional[int] = None  # 页码
    section: Optional[str] = None  # 章节
    title: Optional[str] = None  # 标题
    author: Optional[str] = None  # 作者
    created_at: Optional[str] = None  # 创建时间
    updated_at: Optional[str] = None  # 更新时间
    
    # 自定义字段
    doc_type: Optional[str] = None  # 文档类型
    tags: Optional[list[str]] = None  # 标签
    version: Optional[str] = None  # 版本号
```

### 4.2 元数据提取规则

```python
class MetadataExtractor:
    def extract_from_file(self, file_path: str) -> dict:
        """从文件提取元数据"""
        return {
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "created_at": datetime.fromtimestamp(
                os.path.getctime(file_path)
            ).isoformat(),
            "updated_at": datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat(),
        }
    
    def extract_from_text(self, text: str, file_path: str) -> dict:
        """从文本内容提取元数据"""
        metadata = self.extract_from_file(file_path)
        
        # 尝试提取标题（第一行非空文本）
        lines = text.strip().split("\n")
        for line in lines:
            if line.strip():
                metadata["title"] = line.strip()[:100]
                break
        
        return metadata
```

---

## 5. 数据清洗规范

### 5.1 清洗规则

```python
import re

class TextCleaner:
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def clean(self, text: str) -> str:
        """
        文本清洗
        
        Args:
            text: 原始文本
        
        Returns:
            str: 清洗后文本
        """
        # 1. 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 2. 去除页眉页脚
        text = self._remove_headers_footers(text)
        
        # 3. 去除特殊字符
        text = self._remove_special_chars(text)
        
        # 4. 统一标点符号
        text = self._normalize_punctuation(text)
        
        # 5. 去除重复内容
        text = self._remove_duplicates(text)
        
        return text.strip()
    
    def _remove_headers_footers(self, text: str) -> str:
        """移除页眉页脚"""
        # 移除页码
        text = re.sub(r'第\d+页', '', text)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # 移除版权信息
        text = re.sub(r'©.*?\d{4}', '', text)
        
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        # 保留中文、英文、数字、常用标点
        pattern = r'[^\w\s\u4e00-\u9fff。！？，、；：""''（）\[\]【】]'
        return re.sub(pattern, '', text)
    
    def _normalize_punctuation(self, text: str) -> str:
        """统一标点符号"""
        replacements = {
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "；": ";",
            "：": ":",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _remove_duplicates(self, text: str) -> str:
        """移除重复内容"""
        sentences = text.split(". ")
        seen = set()
        unique = []
        
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique.append(sentence)
        
        return ". ".join(unique)
```

---

## 6. 数据管道接口

### 6.1 主接口

```python
class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.chunker = DocumentChunker(config.chunking)
        self.cleaner = TextCleaner(config.cleaning)
        self.extractor = MetadataExtractor()
    
    def process(self, file_path: str) -> list[dict]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            list[dict]: 处理结果
        """
        # 1. 加载文档
        text = self._load_document(file_path)
        
        # 2. 清洗
        text = self.cleaner.clean(text)
        
        # 3. 提取元数据
        metadata = self.extractor.extract_from_text(text, file_path)
        
        # 4. 分块
        chunks = self.chunker.chunk(text, metadata)
        
        return chunks
    
    def process_batch(self, file_paths: list[str]) -> list[dict]:
        """
        批量处理文件
        
        Args:
            file_paths: 文件路径列表
        
        Returns:
            list[dict]: 所有处理结果
        """
        results = []
        for path in file_paths:
            try:
                chunks = self.process(path)
                results.extend(chunks)
            except Exception as e:
                logger.error(f"处理失败: {path}, 错误: {e}")
        
        return results
```

---

## 7. 测试用例

### 7.1 单元测试

```python
def test_chunker_basic():
    config = ChunkingConfig(chunk_size=512, chunk_overlap=64)
    chunker = DocumentChunker(config)
    
    text = "这是一段测试文本。" * 100
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 0
    assert all(len(c["text"]) <= 1024 for c in chunks)

def test_cleaner():
    cleaner = TextCleaner()
    
    text = "  这是  测试  文本  \n\n\n"
    cleaned = cleaner.clean(text)
    
    assert cleaned == "这是 测试 文本"
```

### 7.2 集成测试

```python
def test_pipeline_e2e():
    config = PipelineConfig()
    pipeline = DataPipeline(config)
    
    chunks = pipeline.process("test.pdf")
    
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("metadata" in c for c in chunks)
```

---

## 8. 入库（Ingestion）工程规格

> 同步正文第 2 章新增的"入库"章节（v2）。规格定义去重、增量更新、批量限流、写入校验的接口契约。

### 8.1 入库配置

```python
@dataclass
class IngestionConfig:
    # 去重 / 幂等
    dedup_by_content: bool = True        # 内容哈希保证幂等
    id_namespace: str = "kb-chunk"        # uuid5 命名空间

    # 增量更新
    incremental_by_source: bool = True    # 按 source 先删后插

    # 批量与限流
    batch_size: int = 100                 # 每批 embedding + upsert 条数
    embed_concurrency: int = 4            # embedding 并发

    # 写入校验
    verify_count: bool = True             # 入库条数回读校验
    verify_dim: bool = True               # 向量维度校验
```

### 8.2 入库接口契约

```python
class IngestionPipeline:
    def run(self, file_path: str) -> int:
        """
        解析 → 清洗(含隐私脱敏) → 分块 → 按 source 删旧 → 批量 upsert → 校验
        Returns: 实际入库 chunk 数
        """

    def _make_chunk_id(self, source: str, page: int, text: str) -> str:
        """sha256(归一化文本) + uuid5，保证幂等"""

    def _delete_by_source(self, source: str) -> None:
        """增量更新：删除该 source 下全部旧 chunk（软删或物理删）"""

    def _verify(self, source: str, expected: int) -> None:
        """回读校验：count 与 dim 匹配，否则抛异常"""
```

### 8.3 隐私脱敏挂钩

```python
# 清洗阶段即脱敏（见第 13 章 13.2.2），数据未进向量库前 PII 已掩码
def process(self, file_path: str) -> list[dict]:
    text = self.cleaner.clean(self._load_document(file_path))
    text = mask_pii(text)                 # PII 前置脱敏
    metadata = self.extractor.extract_from_text(text, file_path)
    return self.chunker.chunk(text, metadata)
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：新增第 8 章入库工程规格（去重/增量/批量/校验）与隐私脱敏挂钩，对齐正文第 2 章 v2*
