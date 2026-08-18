# SPEC_上下文与LLM生成_第7章

> 技术规格说明书 - 上下文拼接与 LLM 生成

> 🔧 **v2 修订（2026-07）**：已应用 UP-005（Prompt 模板占位符改为 `__CONTEXT__` + `replace`，避免检索结果里的 `{}` 触发 format 崩溃）、UP-006（本地模型 pipeline 改为 `text-generation` + `device_map="auto"` + `max_new_tokens`）、UP-202（ConversationManager 增加 `_standalone_query` 多轮 query 独立化）。详见 `SPEC_更新修订清单_v2.md`。

---

## 1. 章节概述

### 1.1 目标

定义上下文拼接和 LLM 生成的技术规格，包括 Prompt 设计、上下文管理和输出格式化。

### 1.2 范围

- Prompt 模板规范
- 上下文拼接策略
- LLM 调用接口
- 输出格式化

---

## 2. Prompt 模板规范

### 2.1 System Prompt 模板

```python
# v2：占位符用 __CONTEXT__，配合 .replace() 注入，避免检索结果里的 { } 触发 format 崩溃（UP-005）
DEFAULT_SYSTEM_PROMPT = """你是一个专业的客服助手。请根据以下参考资料回答用户的问题。

规则：
1. 只根据提供的参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，请直接说"我无法从现有资料中找到答案"
3. 回答时标注信息来源，格式为 [来源：文档名]
4. 保持回答简洁，控制在 200 字以内

参考资料：
__CONTEXT__"""
```

### 2.2 Prompt 配置

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PromptConfig:
    # 模板配置
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = "用户问题：{query}"
    
    # 输出配置
    max_tokens: int = 1024
    temperature: float = 0
    
    # 上下文配置
    max_context_length: int = 4000
    context_separator: str = "\n\n"
    
    # 引用配置
    include_citations: bool = True
    citation_format: str = "[{index}] {text}\n来源：{source}"
```

---

## 3. 上下文拼接

### 3.1 上下文构建器

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContextItem:
    index: int
    text: str
    source: str
    score: float
    metadata: dict

class ContextBuilder:
    def __init__(self, config: PromptConfig):
        self.config = config
    
    def build(self, documents: list[dict]) -> str:
        """
        构建上下文字符串
        
        Args:
            documents: 检索结果列表
        
        Returns:
            str: 格式化的上下文
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            score = doc.get("score", 0)
            
            # 截断过长文本
            if current_length + len(text) > self.config.max_context_length:
                remaining = self.config.max_context_length - current_length
                if remaining > 100:
                    text = text[:remaining] + "..."
                else:
                    break
            
            # 格式化
            if self.config.include_citations:
                part = f"[{i}] {text}\n来源：{source}"
            else:
                part = text
            
            context_parts.append(part)
            current_length += len(part)
        
        return self.config.context_separator.join(context_parts)
    
    def build_with_scores(self, documents: list[dict]) -> str:
        """构建带分数的上下文"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            score = doc.get("score", 0)
            
            part = f"[{i}] (相关度：{score:.2f}) {text}\n来源：{source}"
            context_parts.append(part)
        
        return self.config.context_separator.join(context_parts)
```

---

## 4. LLM 调用接口

### 4.1 基础接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict
    metadata: dict

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """生成回答"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        """流式生成"""
        pass
```

### 4.2 OpenAI 实现

```python
import openai

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """生成回答"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            metadata={}
        )
    
    async def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0,
        max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        """流式生成"""
        client = openai.AsyncOpenAI(api_key=self.client.api_key)
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

---

## 5. 生成器实现

### 5.1 基础生成器

```python
class Generator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        context_builder: ContextBuilder,
        config: PromptConfig
    ):
        self.llm = llm_client
        self.context_builder = context_builder
        self.config = config
    
    def generate(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """
        生成回答
        
        Args:
            query: 用户问题
            documents: 检索结果
        
        Returns:
            dict: 生成结果
        """
        # 构建上下文
        context = self.context_builder.build(documents)
        
        # 构建消息
        messages = [
            {"role": "system", "content": self.config.system_prompt.format(context=context)},
            {"role": "user", "content": self.config.user_template.format(query=query)}
        ]
        
        # 调用 LLM
        response = self.llm.generate(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # 提取来源
        sources = [doc.get("source", "unknown") for doc in documents]
        
        return {
            "answer": response.content,
            "sources": sources,
            "usage": response.usage,
            "metadata": {
                "model": response.model,
                "documents_count": len(documents)
            }
        }
    
    async def generate_stream(
        self,
        query: str,
        documents: list[dict]
    ) -> AsyncIterator[str]:
        """流式生成"""
        context = self.context_builder.build(documents)
        
        messages = [
            {"role": "system", "content": self.config.system_prompt.format(context=context)},
            {"role": "user", "content": self.config.user_template.format(query=query)}
        ]
        
        async for chunk in self.llm.generate_stream(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        ):
            yield chunk
```

### 5.2 带引用的生成器

```python
class CitationGenerator(Generator):
    def generate_with_citations(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """带引用的生成"""
        context = self.context_builder.build(documents)
        
        prompt = f"""请根据以下资料回答用户的问题，并在回答中标注引用来源。

资料：
{context}

用户问题：{query}

要求：
1. 回答时使用 [1]、[2] 等标注引用来源
2. 如果某个信息来自多个来源，标注主要来源
3. 如果参考资料中没有相关信息，请说明

回答："""
        
        messages = [
            {"role": "system", "content": self.config.system_prompt.format(context="")},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate(messages)
        
        return {
            "answer": response.content,
            "sources": [doc.get("source", "unknown") for doc in documents],
            "usage": response.usage
        }
```

---

## 6. 多轮对话

### 6.1 对话管理器

```python
from collections import deque

class ConversationManager:
    def __init__(
        self,
        generator: Generator,
        max_history: int = 5
    ):
        self.generator = generator
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
    
    def chat(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """
        多轮对话
        
        Args:
            query: 用户问题
            documents: 检索结果
        
        Returns:
            dict: 回答
        """
        # 构建上下文
        context = self.generator.context_builder.build(documents)
        
        # 构建历史消息
        messages = [
            {"role": "system", "content": self.generator.config.system_prompt.format(context=context)}
        ]
        
        # 添加历史对话
        for h in self.history:
            messages.append({"role": "user", "content": h["query"]})
            messages.append({"role": "assistant", "content": h["answer"]})
        
        # 添加当前问题
        messages.append({"role": "user", "content": query})
        
        # 生成回答
        response = self.generator.llm.generate(messages)
        
        # 记录历史
        self.history.append({
            "query": query,
            "answer": response.content
        })
        
        return {
            "answer": response.content,
            "sources": [doc.get("source", "unknown") for doc in documents],
            "usage": response.usage
        }
    
    def clear_history(self):
        """清空历史"""
        self.history.clear()
```

---

## 7. 输出格式化

### 7.1 JSON 输出

```python
import json

class StructuredGenerator(Generator):
    def generate_structured(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """生成结构化输出"""
        context = self.context_builder.build(documents)
        
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
        
        messages = [
            {"role": "system", "content": "你是一个专业的助手，请以 JSON 格式输出。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate(messages)
        
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "answer": response.content,
                "confidence": 0.5,
                "sources": [doc.get("source", "unknown") for doc in documents],
                "need_more_info": False
            }
        
        return result
```

---

## 8. 测试用例

### 8.1 功能测试

```python
def test_context_builder():
    config = PromptConfig()
    builder = ContextBuilder(config)
    
    documents = [
        {"text": "退货流程", "source": "policy.pdf", "score": 0.9},
        {"text": "会员规则", "source": "member.pdf", "score": 0.8}
    ]
    
    context = builder.build(documents)
    
    assert "退货流程" in context
    assert "policy.pdf" in context

def test_generator():
    # Mock LLM client
    class MockLLM(BaseLLMClient):
        def generate(self, messages, **kwargs):
            return LLMResponse(
                content="测试回答",
                model="mock",
                usage={},
                metadata={}
            )
    
    config = PromptConfig()
    generator = Generator(MockLLM(), ContextBuilder(config), config)
    
    result = generator.generate("怎么退货？", [
        {"text": "退货流程", "source": "policy.pdf"}
    ])
    
    assert result["answer"] == "测试回答"
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
