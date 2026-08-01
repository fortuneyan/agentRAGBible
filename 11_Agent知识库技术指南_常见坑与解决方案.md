# 第 11 章：常见坑与解决方案

> 血泪教训总结，希望你别再踩一遍。

---

## 坑1：检索结果"答非所问"

**症状**：用户问"怎么退货"，返回的是"退货政策"全文，但没回答具体流程。

**原因**：
- 分块太大
- 没有重排序
- Query 没改写

**解决方案**：

```python
# 1. 减小 chunk_size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,  # 从 512 减小到 256
    chunk_overlap=32
)

# 2. 加 Reranker
reranker = CrossEncoder("BAAI/bge-reranker-large")
docs = reranker.rerank(query, docs, top_k=3)

# 3. Query 改写
def rewrite_query(query: str) -> str:
    if "退货" in query:
        return "退货流程 步骤"
    return query
```

---

## 坑2：回答"过时"

**症状**：知识库更新了，但 Agent 还是返回旧信息。

**原因**：
- 缓存没清理
- 索引没更新
- 版本控制缺失

**解决方案**：

```python
# 1. 更新时清理缓存
def update_document(file_path: str):
    # 删除旧数据
    delete_by_source(file_path)
    
    # 导入新数据
    kb.ingest(file_path)
    
    # 清理缓存
    cache.clear()
    semantic_cache.clear()

# 2. 版本控制
def add_with_version(text: str, metadata: dict):
    metadata["version"] = datetime.now().isoformat()
    vector_store.add(text, metadata)

# 3. 过滤最新版本
def search_latest(query: str):
    return vector_store.search(
        query_embedding,
        filter={"must": [{"key": "version", "range": {"gte": "2024-01-01"}}]}
    )
```

---

## 坑3：成本爆炸

**症状**：Embedding 成本飙升，LLM 费用暴涨。

**原因**：
- 没有缓存，重复计算
- 检索结果太多，上下文太长
- 频繁调用

**解决方案**：

```python
# 1. Embedding 缓存
embedding_cache = EmbeddingCache(max_size=10000)

# 2. 限制上下文长度
def build_context(docs: list[dict], max_length: int = 4000) -> str:
    context = ""
    for doc in docs:
        if len(context) + len(doc["text"]) > max_length:
            break
        context += doc["text"] + "\n\n"
    return context

# 3. 语义缓存
semantic_cache = SemanticCache(threshold=0.95)
```

---

## 坑4：中文检索效果差

**症状**：英文问题检索准确率高，中文问题效果差。

**原因**：
- 用了英文优化的 Embedding 模型
- 中文分词不好
- 同义词处理缺失

**解决方案**：

```python
# 1. 换用中文模型
embedding_model = "BAAI/bge-large-zh-v1.5"

# 2. 中文分词
import jieba

def tokenize_chinese(text: str) -> list[str]:
    return list(jieba.cut(text))

# 3. 同义词扩展
synonyms = {
    "退货": ["退款", "退换", "退回"],
    "积分": ["points", "score"]
}

def expand_query(query: str) -> str:
    for word, syns in synonyms.items():
        if word in query:
            query += " " + " ".join(syns)
    return query
```

---

## 坑5：PDF 解析乱码

**症状**：PDF 提取的文字乱码、表格错位。

**原因**：
- 扫描版 PDF（图片）
- 复杂排版
- 特殊字体

**解决方案**：

```python
# 1. 检测是否为扫描版
import fitz

def is_scanned_pdf(file_path: str) -> bool:
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text()
        if len(text.strip()) < 10:  # 几乎没文字
            return True
    return False

# 2. 扫描版用 OCR
from paddleocr import PaddleOCR

def ocr_pdf(file_path: str) -> list[str]:
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    doc = fitz.open(file_path)
    
    texts = []
    for page in doc:
        img = page.get_pixmap()
        result = ocr.ocr(img.tobytes())
        texts.append("\n".join([line[1][0] for line in result[0]]))
    
    return texts

# 3. 用 Unstructured 处理复杂排版
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("complex.pdf")
```

---

## 坑6：多语言混杂

**症状**：中英文混合的文档，检索效果差。

**原因**：
- Embedding 模型对混合语言支持不好
- 分词器不认识英文

**解决方案**：

```python
# 1. 用多语言模型
embedding_model = "BAAI/bge-m3"

# 2. 语言检测 + 分词
from langdetect import detect

def smart_tokenize(text: str) -> list[str]:
    lang = detect(text)
    if lang == "zh":
        return list(jieba.cut(text))
    else:
        return text.split()
```

---

## 坑7：检索速度慢

**症状**：查询延迟超过 1 秒。

**原因**：
- 数据量大，索引没优化
- 没有缓存
- 网络延迟

**解决方案**：

```python
# 1. 优化索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 32, "efConstruction": 400}
}

# 2. 分片检索
collections = {
    "recent": "kb_recent",  # 最近 3 个月
    "archive": "kb_archive"  # 历史数据
}

def smart_search(query: str):
    # 先搜近期
    results = search_in_collection(query, "recent")
    if results:
        return results
    
    # 再搜历史
    return search_in_collection(query, "archive")

# 3. 异步 + 并发
async def batch_search(queries: list[str]):
    tasks = [async_search(q) for q in queries]
    return await asyncio.gather(*tasks)
```

---

## 坑8：回答"幻觉"

**症状**：LLM 编造了知识库里没有的信息。

**原因**：
- 没有约束 LLM 行为
- 上下文不够相关
- Temperature 太高

**解决方案**：

```python
# 1. 强化 Prompt 约束
SYSTEM_PROMPT = """规则：
1. 只根据提供的参考资料回答，绝对不要编造
2. 如果资料中没有相关信息，请直接说"我无法从现有资料中找到答案"
3. 回答时必须标注来源"""

# 2. 降低 Temperature
response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0  # 设为 0
)

# 3. 加置信度判断
def check_confidence(answer: str, context: str) -> float:
    # 简单检查：回答中的关键词是否在上下文中
    keywords = extract_keywords(answer)
    in_context = sum(1 for k in keywords if k in context)
    return in_context / len(keywords) if keywords else 0
```

---

## 坑9：数据导入失败

**症状**：导入文档时报错，或导入后数据不全。

**原因**：
- 文件格式不支持
- 文件损坏
- 权限问题

**解决方案**：

```python
# 1. 文件格式检测
import os

def detect_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    format_map = {
        ".pdf": "pdf",
        ".docx": "word",
        ".md": "markdown",
        ".txt": "text"
    }
    return format_map.get(ext, "unknown")

# 2. 异常处理
def safe_ingest(file_path: str):
    try:
        kb.ingest(file_path)
    except Exception as e:
        logger.error(f"导入失败: {file_path}, 错误: {e}")
        # 记录失败，后续重试
        save_failed_file(file_path, str(e))

# 3. 断点续传
def batch_ingest_with_checkpoint(file_paths: list[str]):
    checkpoint = load_checkpoint()
    
    for path in file_paths:
        if path in checkpoint.get("completed", []):
            continue
        
        try:
            kb.ingest(path)
            save_checkpoint(path)
        except Exception as e:
            logger.error(f"跳过: {path}")
```

---

## 坑10：生产环境崩溃

**症状**：上线后服务不可用。

**原因**：
- 内存溢出
- 并发处理不当
- 没有降级策略

**解决方案**：

```python
# 1. 内存监控
import psutil

def check_memory():
    if psutil.virtual_memory().percent > 80:
        logger.warning("内存使用过高")
        # 触发 GC
        import gc
        gc.collect()

# 2. 限流
from functools import wraps
import time

def rate_limit(max_calls: int, period: int):
    def decorator(func):
        calls = []
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if c > now - period]
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 3. 降级策略
def fallback_answer(query: str) -> str:
    try:
        return kb.answer(query)
    except Exception:
        # 降级到简单搜索
        return "抱歉，系统暂时繁忙，请稍后再试。"
```

---

## 坑11：命令注入攻击

**症状**：用户输入 `; rm -rf /`，Agent 拼接到命令中直接执行。

**方案**：永远使用 `subprocess.run(cmd_array, shell=False)`，并启用参数白名单校验（见第 14 章 14.3 安全底线）。

## 坑12：Agent 自我循环

**症状**：命令输出被当作新命令再次执行，无限循环。

**方案**：引入 `LoopDetector`，记录最近 10 次操作哈希，发现重复立即中断。

## 坑13：CLI 输出格式不统一

**症状**：部分工具输出 JSON，部分输出纯文本，解析失败。

**方案**：强制所有可调用 CLI 支持 `--json` 参数，并在工具 Schema 中声明 `output_format`（见第 2 章工具元数据 Schema）。

## 坑14：工具版本漂移

**症状**：知识库存的是旧版参数，实际执行新版，参数不匹配。

**方案**：入库时记录 `tool_version`，执行前先调用 `--version` 校验，不匹配则触发重新入库告警。

---

## 本章小结

- 坑是正常的，关键是知道怎么解决
- 生产环境要多做防御性编程
- 监控 + 日志是发现问题的基础
- 降级策略保证服务可用性

下一章，我们讲技术栈推荐组合。

---

*踩坑不可怕，可怕的是同一个坑踩两次。写下来，分享出去。*
