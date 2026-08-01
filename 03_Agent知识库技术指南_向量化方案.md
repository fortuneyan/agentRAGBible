# 第 3 章：向量化方案

> 之前用 bge-large（1024 维）生成向量，后来换成 OpenAI（1536 维），结果索引要重建。提前想好用哪个模型，别中途换。

---

## 模型选择

我自己测过几个：

| 模型 | 维度 | 中文效果 | 速度 | 推荐场景 |
|-----|------|---------|-----|---------|
| text-embedding-3-small | 1536 | ⭐⭐⭐ | 快 | 国际化项目 |
| text-embedding-3-large | 3072 | ⭐⭐⭐⭐ | 中 | 追求效果 |
| bge-large-zh-v1.5 | 1024 | ⭐⭐⭐⭐⭐ | 中 | 中文场景首选 |
| bge-m3 | 1024 | ⭐⭐⭐⭐⭐ | 慢 | 多语言场景 |
| m3e-base | 768 | ⭐⭐⭐ | 快 | 轻量部署 |

**个人推荐**：

- 纯中文场景：**bge-large-zh-v1.5**，效果好，社区活跃
- 多语言：**bge-m3**，支持 100+ 种语言
- 追求极致效果：**text-embedding-3-large**（OpenAI），但贵

---

## 向量维度选择

```python
# OpenAI embedding
dimension = 1536  # text-embedding-3-small
dimension = 3072  # text-embedding-3-large

# BGE
dimension = 1024  # bge-large-zh-v1.5

# 本地部署（Ollama）
dimension = 768   # m3e-base
```

**踩坑记录**：

维度一旦确定，后面所有环节都要对齐：
- 向量数据库的索引维度
- 检索时的查询向量维度
- 缓存的向量维度

换模型 = 重建索引 = 花时间花钱。

---

## 本地部署 vs API 调用

### 本地部署

```python
from sentence_transformers import SentenceTransformer

# 首次会下载模型，约 1.3GB
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")

def embed_local(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
```

**优点**：
- 无调用费用
- 数据不出内网
- 无限调用次数

**缺点**：
- 需要 GPU（CPU 也能跑，但慢）
- 首次加载慢
- 模型更新需要手动

### API 调用

```python
import openai

def embed_openai(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
```

**优点**：
- 无需 GPU
- 效果有保障
- 自动更新

**缺点**：
- 有费用（约 $0.02/1M tokens）
- 数据要发到外部
- 有调用限制

### 我的建议

- 开发测试：本地模型
- 小规模生产：API 调用
- 大规模生产：本地模型 + GPU

---

## Embedding 质量优化

### 归一化

```python
import numpy as np

def normalize(embedding: list[float]) -> list[float]:
    norm = np.linalg.norm(embedding)
    return (np.array(embedding) / norm).tolist()
```

**为什么归一化？**

余弦相似度计算时，归一化后可以直接用点积，速度更快。

### 文本预处理

```python
def preprocess_for_embedding(text: str) -> str:
    # 去除多余空白
    text = " ".join(text.split())
    
    # 截断过长文本（模型有 token 限制）
    if len(text) > 500:
        text = text[:500]
    
    return text
```

### 批量处理

```python
def batch_embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings
```

---

## 多模态 Embedding

有些场景需要处理图片、音频：

```python
# 图片 + 文本联合 Embedding
from CLIP import clip

def embed_multimodal(image_path: str, text: str):
    image = preprocess_image(image_path)
    image_embedding = clip.encode_image(image)
    text_embedding = clip.encode_text(text)
    
    # 融合
    combined = (image_embedding + text_embedding) / 2
    return combined
```

这个在电商、医疗等场景有用，但普通知识库用不到。

---

## 本章小结

- 中文场景推荐 bge-large-zh-v1.5
- 维度一旦确定不要换
- 本地部署适合数据敏感场景，API 适合快速起步
- 批量处理 + 归一化是基本操作

下一章，我们讲向量数据库——怎么存、怎么查。

---

*选 embedding 模型就像选女朋友，没有最好的，只有最适合你的。别听别人吹，自己测了才知道。*
