# SPEC_更新修订清单_v2

> 《Agent 知识库技术指南》勘误与工程化修订规格
> 基于"架构师 + 高级程序员"审核报告，汇总所有需要更新的内容。
> 本文档是后续章节修订的**唯一权威依据**，每项修订均带唯一编号（UP-xxx），便于追溯。

---

## 0. 修订总览

| 类别 | 数量 | 状态 |
|-----|------|-----|
| P0 代码正确性（必修，阻塞生产） | 8 | 待修 |
| P1 工程化缺陷（强烈建议） | 6 | 待修 |
| P2 架构补全（建议新增） | 4 | 待新增 |
| 文档一致性（版本标注等） | 1 | 待补 |

**修订原则**：
1. 不改变章节叙事风格与目录结构，只修正错误与补强工程化。
2. 所有代码修订必须同时更新对应正文章节（`NN_*.md`）与 SPEC（`spec/SPEC_*.md`）。
3. 第三方 API 代码统一在代码块首行标注 `# requires: <pkg><op><ver>`。
4. 修订后的代码必须满足：可运行、无副作用、有空集/除零保护、有类型注解。

---

## 1. P0 级修订（代码正确性）

### UP-001 【第9章】EmbeddingCache 淘汰逻辑错误
- **文件**：`09_Agent知识库技术指南_性能优化.md`、`spec/SPEC_性能优化_第9章.md`
- **问题**：`EmbeddingCache.set` 注释写"删掉一半"，实际删除字典前 N//2 个 key（可能是热数据），且非 LRU。读者照抄进生产会导致缓存命中率骤降。
- **修订方案**：改用 `collections.OrderedDict` 实现 LRU 语义，`get` 时 `move_to_end`，`set` 满时 `popitem(last=False)` 淘汰最久未访问项。
- **验收**：单元测试——写入 `max_size` 条后，第 `max_size+1` 条触发淘汰，被淘汰的是最早访问（非最早写入）的项。

### UP-002 【第9章】embed_with_cache 回填逻辑错误
- **文件**：同 UP-001
- **问题**：
  ```python
  for idx, embedding in zip(to_embed_indices, new_embeddings):
      results[idx] = embedding
      cache.set(to_embed[to_embed_indices.index(idx)], embedding)  # O(n) 查找，重复 idx 时取错
  ```
- **修订方案**：直接 `for idx, text, embedding in zip(to_embed_indices, to_embed, new_embeddings)`，三个列表顺序天然一致，无需 `.index()`。
- **验收**：含重复文本的批量调用，每条 cache key 与 embedding 一一对应。

### UP-003 【第8章】VectorStore.add 用自增 int id 导致数据覆盖
- **文件**：`08_Agent知识库技术指南_完整实现示例.md`、`spec/SPEC_完整实现示例_第8章.md`
- **问题**：
  ```python
  points = [PointStruct(id=i, ...) for i, (...) in enumerate(...)]
  ```
  每次 ingest 都从 0 开始，第二次导入 upsert 覆盖第一次的数据。
- **修订方案**：
  1. ID 改为基于内容的稳定 hash（`uuid.uuid5(NAMESPACE_URL, text)`），保证幂等。
  2. Qdrant collection 支持 `PointStruct(id=<uuid str>)`（Qdrant 支持 UUID/整型/字符串）。
  3. 配置项增加 `id_strategy: Literal["uuid5","sequential"]`，默认 `uuid5`。
- **验收**：同一文档 ingest 两次，数据量不翻倍也不丢失；不同文档不冲突。

### UP-004 【第8章/第6章】Reranker.rerank 原地排序副作用
- **文件**：`08_完整实现示例.md`、`06_重排序技术.md`
- **问题**：`documents.sort(...)` 直接修改入参 list，调用方原列表被破坏。
- **修订方案**：对 `documents` 的浅拷贝排序：`sorted(documents, key=..., reverse=True)`，并从拷贝上设置 `rerank_score` 后返回新列表（不修改入参元素的副作用也需避免——返回带 `rerank_score` 的新 dict）。
- **验收**：调用前后，入参 list 的元素与顺序不变。

### UP-005 【第7章/第8章】SYSTEM_PROMPT.format 对 `{}` 脆弱
- **文件**：`07_上下文与LLM生成.md`、`08_完整实现示例.md`
- **问题**：检索结果常含 JSON/代码片段（`{`、`}`），`.format(context=context)` 抛 `KeyError`/`IndexError`。
- **修订方案**：
  1. SYSTEM_PROMPT 占位符从 `{context}` 改为 `{{context}}` 转义不可行（会破坏可读性），改用 **`str.replace`** 方案：
     ```python
     content = SYSTEM_PROMPT.replace("{context}", context)
     ```
  2. 或采用更严格的命名占位 `__CONTEXT__` + replace。
  3. 全书统一一种方案，推荐 `replace`。
- **验收**：context 含 `{"key":"val"}` 或 `{0}` 时不再报错。

### UP-006 【第7章】本地模型 pipeline task 类型错误
- **文件**：`07_上下文与LLM生成.md`
- **问题**：
  ```python
  generator = pipeline("text2text-generation", model="Qwen/Qwen-7B-Chat")
  ```
  Qwen 是 decoder-only causal LM，应为 `text-generation`；且未指定 `device_map`/`torch_dtype`，消费级显卡 OOM。
- **修订方案**：
  ```python
  generator = pipeline(
      "text-generation",
      model="Qwen/Qwen-7B-Chat",
      device_map="auto",
      torch_dtype="auto",
      trust_remote_code=True,
  )
  ```
  并补注：7B 模型至少需 14GB 显存（FP16）；CPU 推理建议换 Qwen-1.8B。
- **验收**：代码块标注显存要求；task 类型与模型架构匹配。

### UP-007 【第10章】evaluate_retrieval 比对对象错误
- **文件**：`10_监控与评估.md`、`spec/SPEC_监控与评估_第10章.md`
- **问题**：`expected_docs` 是文件名（`["退货流程.pdf"]`），却用 `any(exp in doc["text"] for exp in expected)` 去正文里匹配，永远不命中。
- **修订方案**：改为比对 `doc["metadata"]["source"]`：
  ```python
  retrieved_sources = [doc["metadata"].get("source", "") for doc in retrieved]
  for i, src in enumerate(retrieved_sources):
      if any(exp in src for exp in expected):
          hits += 1; mrr += 1/(i+1); break
  ```
- **验收**：用样例测试集能跑出非零 precision/mrr。

### UP-008 【第10章】evaluate_answer 子串匹配过宽 + track_metrics 装饰器缺陷
- **文件**：`10_监控与评估.md`
- **问题 a**：`if expected in answer` 子串匹配（"7天" 是 "27天" 子串）。
- **问题 b**：`track_metrics` 装饰器缺 `functools.wraps`，且对 async 函数失效。
- **修订方案**：
  - a：改为关键词集合匹配（Jaccard）或 LLM-as-judge，至少对数字/实体做边界校验。
  - b：用 `functools.wraps`；提供 sync 与 async 两个版本，或用 `inspect.iscoroutinefunction` 自动分发。
- **验收**：装饰器可同时装饰普通函数与 `async def`；wraps 不丢失 `__name__`。

---

## 2. P1 级修订（工程化缺陷）

### UP-101 【全书】第三方 API 缺版本标注
- **范围**：所有含第三方调用的代码块
- **修订**：代码块首行加 `# requires: pymilvus>=2.3,<2.5; qdrant-client>=1.7; langchain>=0.1; openai>=1.10`
- **重点章节**：第4章（Milvus/Qdrant）、第8章（langchain PyMuPDFLoader 路径迁移到 `langchain-community`）

### UP-102 【第4章/第9章】PQ 与标量量化概念混淆
- **问题**：第4章讲 PQ 原理却用 `ScalarQuantization` 示例；第9章 Milvus `PQ` 缺 `m` 参数。
- **修订**：明确三档量化（Scalar INT8 / Scalar FP16 / Product Quantization / Binary），各自独立代码示例，标注压缩比与精度损失。

### UP-103 【第5章】multi_recall 融合函数签名不匹配
- **问题**：`reciprocal_rank_fusion(all_results)` 传单个 list，但签名是两个 list。
- **修订**：统一 RRF 签名为 `rrf(*ranked_lists, k=60)`，接收可变多路。

### UP-104 【第5章】weighted_fusion 缺空集/除零保护
- **修订**：`vector_max`/`bm25_max` 为 0 时跳过该路；空 list 直接返回空。

### UP-105 【第6章】cached_rerank 返回可变对象污染缓存
- **问题**：`lru_cache` 返回同一 list 引用，调用方修改污染缓存。
- **修订**：返回 `tuple`，或在函数内对结果做浅拷贝。

### UP-106 【第2章】semantic_split 引用未定义 cosine_similarity + 阈值魔法数
- **修订**：补内部 `_cosine_similarity`；说明 threshold 应基于相邻距离百分位（推荐 LangChain `SemanticChunker` 的 percentile 模式）。

---

## 3. P2 级修订（架构补全 / 新增内容）

### UP-201 【新增小节】增量更新与删除
- **目标章节**：第9章（性能优化）或第11章（常见坑）
- **要点**：
  - 文档更新 = 删旧 chunk（按 `source` 过滤删除）+ 插入新 chunk + 清缓存
  - 幂等 ID（见 UP-003）保证重导不膨胀
  - 软删除标记 vs 物理删除的取舍

### UP-202 【新增小节】多轮对话的 Query 独立化
- **目标章节**：第7章
- **要点**：用历史把"那它的价格呢"改写为"iPhone 15 的价格"，再检索；与第5章 query 改写打通。

### UP-203 【新增章节/小节】Agentic RAG（检索即工具）
- **目标**：补足书名"Agent"维度
- **要点**：把检索封装为 tool，由 LLM Agent 自主决定是否检索、检索几次（多跳）；ReAct / OpenAI function calling 示例。

### UP-204 【新增小节】安全：Prompt 注入与数据权限
- **目标章节**：第11章
- **要点**：用户输入清洗、system prompt 防护、行级权限（payload 过滤实现多租户）。

---

## 4. 文档一致性

### UP-301 【全书】版本号与日期
- 所有 SPEC 文件页脚统一从 `v1.0 / 2024-01` 升级为 `v2.0 / 2026-07`，本次修订涉及的章节在 SPEC 顶部注明"已应用 UP-xxx"。

---

## 5. 修订执行顺序（任务编排）

| 任务 | 涉及修订项 | 预计影响文件 |
|-----|----------|------------|
| **任务1：P0 代码修复** | UP-001 ~ UP-008 | 第6/7/8/9/10章正文 + 对应 SPEC |
| 任务2：P1 工程化 | UP-101 ~ UP-106 | 全书代码块 + 第4/5/6/9章 |
| 任务3：P2 架构补全 | UP-201 ~ UP-204 | 新增小节/章节 |
| 任务4：一致性扫描 | UP-301 | 所有 SPEC 页脚 |

---

*文档版本：2.0*
*更新日期：2026-07*
*审核人：架构师 + 高级程序员视角*
