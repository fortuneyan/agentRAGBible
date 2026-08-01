# 第 7 章：上下文拼接与 LLM 生成

> 检索做得再好，Prompt 写得烂，回答照样拉垮。这一章讲怎么把检索结果喂给 LLM。

---

## Prompt 设计原则

### 1. 明确角色

```python
SYSTEM_PROMPT = """你是一个专业的客服助手。请根据以下参考资料回答用户的问题。"""
```

### 2. 约束行为

```python
SYSTEM_PROMPT = """规则：
1. 只根据提供的参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，请直接说"我无法从现有资料中找到答案"
3. 回答时标注信息来源，格式为 [来源：文档名]
4. 保持回答简洁，控制在 200 字以内"""
```

### 3. 提供上下文

```python
# 注意：占位符用 __CONTEXT__，配合 .replace() 注入（见 UP-005）
SYSTEM_PROMPT = """参考资料：
__CONTEXT__"""
```

---

## 上下文拼接

### 基础拼接

```python
def build_context(retrieved_docs: list[dict]) -> str:
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc["metadata"]["source"]
        text = doc["text"]
        context_parts.append(f"[{i}] {text}\n来源：{source}")
    return "\n\n".join(context_parts)
```

### 带分数的拼接

```python
def build_context_with_scores(retrieved_docs: list[dict]) -> str:
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc["metadata"]["source"]
        score = doc.get("score", 0)
        text = doc["text"]
        context_parts.append(f"[{i}] (相关度：{score:.2f}) {text}\n来源：{source}")
    return "\n\n".join(context_parts)
```

### 分块拼接

```python
def build_context_chunked(retrieved_docs: list[dict], max_length: int = 4000) -> str:
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(retrieved_docs, 1):
        text = doc["text"]
        if current_length + len(text) > max_length:
            break
        context_parts.append(f"[{i}] {text}")
        current_length += len(text)
    
    return "\n\n".join(context_parts)
```

---

## LLM 调用

### OpenAI API

```python
import openai

def build_system_content(context: str) -> str:
    # ⚠️ 不要用 SYSTEM_PROMPT.format(context=context)！
    # 检索到的文档里经常含 { }（JSON 片段、代码块），format 会抛 KeyError（见 UP-005）。
    # 用 replace 安全替换命名占位符 __CONTEXT__。
    return SYSTEM_PROMPT.replace("__CONTEXT__", context)

def generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": build_system_content(context)},
        {"role": "user", "content": f"用户问题：{query}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content
```

### 流式输出

```python
async def stream_answer(query: str, context: str):
    messages = [
        {"role": "system", "content": build_system_content(context)},
        {"role": "user", "content": f"用户问题：{query}"}
    ]

    stream = await openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### 本地模型

> ⚠️ **易错点**（见 UP-006）：
> - Qwen / LLaMA / GLM 都是 **decoder-only causal LM**，task 必须是 `text-generation`，
>   不是 `text2text-generation`（后者只适用于 T5 / FLAN 这类 encoder-decoder）。
> - 7B 模型 FP16 至少需要 **14GB 显存**；不指定 `device_map="auto"` 会在单卡上 OOM，
>   CPU 环境请降到 Qwen-1.8B 之类的轻量模型。

```python
# requires: transformers>=4.40; torch>=2.1
from transformers import pipeline

# 全局只加载一次（7B 模型加载耗时几十秒，不要每次推理都 new）
_generator = pipeline(
    "text-generation",                  # ✅ decoder-only 用 text-generation
    model="Qwen/Qwen-7B-Chat",
    device_map="auto",                  # ✅ 自动分卡 / 卸载到 CPU
    torch_dtype="auto",                 # ✅ FP16，省一半显存
    trust_remote_code=True,
)

def generate_local(query: str, context: str) -> str:
    prompt = f"""根据以下资料回答问题。

资料：
{context}

问题：{query}

回答："""

    # decoder-only 模型会把 prompt 原样吐回来，取 [0]["generated_text"] 后去掉前缀
    result = _generator(
        prompt,
        max_new_tokens=512,             # ✅ 用 max_new_tokens，而不是 max_length（后者含 prompt）
        do_sample=False,                # temperature=0 的等价写法
    )
    generated = result[0]["generated_text"]
    return generated[len(prompt):].strip()  # 去掉 prompt 前缀
```

---

## 引用标注

```python
def generate_with_citations(query: str, retrieved_docs: list[dict]) -> dict:
    context = build_context(retrieved_docs)
    
    prompt = f"""请根据以下参考资料回答用户的问题，并在回答中标注引用来源。

参考资料：
{context}

用户问题：{query}

要求：
1. 回答时使用 [1]、[2] 等标注引用来源
2. 如果某个信息来自多个来源，标注主要来源
3. 如果参考资料中没有相关信息，请说明

回答："""
    
    response = call_llm(prompt)
    
    return {
        "answer": response,
        "sources": [doc["metadata"]["source"] for doc in retrieved_docs]
    }
```

---

## 多轮对话

> ⚠️ 多轮对话里最常见的坑：直接拿当前消息去检索。
> 用户说"那它的价格呢"，`retrieve("那它的价格呢")` 检索效果极差——代词"它"指向上文。
> 正确做法是**先用历史把当前消息独立化（query rewriting with history）再检索**（见 UP-202）。

```python
def build_system_content(context: str) -> str:
    # 同样用 replace 注入，避免 format 崩溃（UP-005）
    return SYSTEM_PROMPT.replace("__CONTEXT__", context)

class ConversationManager:
    def __init__(self, kb, llm=None):
        self.kb = kb
        self.llm = llm  # 用于 query 独立化；为 None 时退化为直接检索
        self.history = []

    def _standalone_query(self, user_message: str) -> str:
        """
        多轮 query 独立化：用历史把"那它的价格呢"改写成"iPhone 15 的价格"。
        没有 LLM 时退化为直接返回当前消息。
        """
        if not self.history or self.llm is None:
            return user_message

        recent = "\n".join(
            f"用户：{h['user']}\n助手：{h['assistant']}"
            for h in self.history[-3:]
        )
        prompt = (
            "请根据对话历史，把用户的最后一句话改写成一个独立、完整、可直接用于检索的查询。"
            "只输出改写后的查询，不要解释。\n\n"
            f"对话历史：\n{recent}\n\n"
            f"用户最后一句话：{user_message}\n\n改写后的查询："
        )
        return self.llm(prompt).strip() or user_message

    def chat(self, user_message: str) -> str:
        # 1. 先独立化 query，再检索（多轮场景的关键）
        retrieval_query = self._standalone_query(user_message)
        docs = self.kb.retrieve(retrieval_query)
        context = build_context(docs)

        # 2. 构造消息（用 replace 而非 format，UP-005）
        messages = [
            {"role": "system", "content": build_system_content(context)}
        ]

        # 3. 添加历史对话（只保留最近 5 轮控制 token）
        for h in self.history[-5:]:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["assistant"]})

        messages.append({"role": "user", "content": user_message})

        # 4. 生成回答
        response = call_llm(messages)

        # 5. 记录历史
        self.history.append({
            "user": user_message,
            "assistant": response
        })

        return response
```

---

## 输出格式化

### JSON 输出

```python
def generate_structured(query: str, context: str) -> dict:
    prompt = f"""请根据以下资料回答问题，并以 JSON 格式输出。

资料：
{context}

问题：{query}

输出格式：
{{
    "answer": "回答内容",
    "confidence": 0.95,
    "sources": ["来源1", "来源2"],
    "need_more_info": false
}}"""
    
    response = call_llm(prompt)
    return json.loads(response)
```

### Markdown 输出

```python
def generate_markdown(query: str, context: str) -> str:
    prompt = f"""请根据以下资料回答问题，使用 Markdown 格式输出。

资料：
{context}

问题：{query}

要求：
- 使用标题、列表、加粗等格式
- 标注引用来源
- 结构清晰"""
    
    return call_llm(prompt)
```

---

## 行动规划 Prompt 模板（供第 14 章）

当 Agent 需要把"用户需求"转成"可执行计划"时，用一套**独立的规划器 System Prompt**（与问答 Prompt 解耦）：

```python
PLANNER_SYSTEM_PROMPT = """你是一个操作规划器。根据用户需求和可用工具列表，输出 JSON 格式的行动计划。

可用工具：
{available_tools}

输出格式：
{
    "action_type": "single" | "multi_step",
    "steps": [
        {
            "step": 1,
            "tool": "工具名",
            "parameters": {"参数": "值"},
            "description": "步骤说明",
            "requires_confirmation": false
        }
    ],
    "fallback": "备选方案"
}

规则：
1. 所有参数必须从用户输入或上下文中提取，不得编造。
2. 缺失必要参数时，在参数值位置填 "MISSING_REQUIRED_PARAM"。
3. 高风险操作（删除、修改配置、网络请求）必须标记 requires_confirmation=true。
"""
```

> 注意：规划器输出是**机器可读的 JSON**，不是自然语言。把它当作"数据"解析，再交给执行器，避免再次走 LLM 让它"自由发挥"。

---

## 本章小结

- Prompt 要明确角色、约束行为、提供上下文
- 流式输出提升用户体验
- 引用标注增加可信度
- 多轮对话需要维护历史
- 输出格式要根据场景选择

下一章，我们把所有部分串起来，给一个完整的实现示例。

---

*Prompt 工程是个玄学，同样的检索结果，换个 Prompt 效果可能差很多。多试，多测。*
