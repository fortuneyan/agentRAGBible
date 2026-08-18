# Agent 知识库技术指南

> 面向工程落地的 RAG（检索增强生成）知识库完整技术指南 —— 从文档解析到 LLM 生成，从性能优化到安全合规。

## 项目简介

本仓库是一份从"数据到回答"的完整 RAG 知识库技术指南，覆盖以下全链路：

- **数据处理管道**：多格式文档解析（PDF/Word/PPTX/CSV/HTML/Markdown）、数据清洗（11 项深度清洗 + LLM 辅助清洗）、分块策略（7 种方法含 Late Chunking / Contextual Retrieval / Agentic Chunking）、入库工程
- **向量化方案**：Embedding 模型选型、维度选择、本地部署 vs API 调用
- **向量数据库选型**：Qdrant / Milvus / Chroma / pgvector 对比与调优
- **检索策略**：混合检索（向量 + BM25）、Query 改写（含 HyDE）、多路召回
- **重排序技术**：Cross-Encoder Reranker、Cohere API、LLM-based 重排
- **上下文与生成**：Prompt 工程、引用标注、多轮对话、输出格式化
- **性能优化**：缓存策略、批量处理、异步化、索引优化
- **监控与评估**：检索质量指标、Prometheus + Grafana 看板、A/B 测试
- **安全合规与伦理**：PII 脱敏、提示注入防护、多租户权限、上线前合规清单
- **行动 Agent（RA-A）**：在 RAG 之上叠加 CLI 行动层，含命令注入防护与审计日志

## 章节目录

| 篇章 | 章节 | 文件 |
|------|------|------|
| 前言 | 这份指南讲什么 | [00_前言.md](00_Agent知识库技术指南_前言.md) |
| 基础篇 | 第 1 章：技术架构总览 | [01_技术架构总览.md](01_Agent知识库技术指南_技术架构总览.md) |
| 基础篇 | 第 2 章：数据处理管道 | [02_数据处理管道.md](02_Agent知识库技术指南_数据处理管道.md) |
| 基础篇 | 第 3 章：向量化方案 | [03_向量化方案.md](03_Agent知识库技术指南_向量化方案.md) |
| 基础篇 | 第 4 章：向量数据库选型 | [04_向量数据库选型.md](04_Agent知识库技术指南_向量数据库选型.md) |
| 进阶篇 | 第 5 章：检索策略 | [05_检索策略.md](05_Agent知识库技术指南_检索策略.md) |
| 进阶篇 | 第 6 章：重排序技术 | [06_重排序技术.md](06_Agent知识库技术指南_重排序技术.md) |
| 进阶篇 | 第 7 章：上下文拼接与 LLM 生成 | [07_上下文与LLM生成.md](07_Agent知识库技术指南_上下文与LLM生成.md) |
| 实战篇 | 第 8 章：完整实现示例 | [08_完整实现示例.md](08_Agent知识库技术指南_完整实现示例.md) |
| 实战篇 | 第 9 章：性能优化 | [09_性能优化.md](09_Agent知识库技术指南_性能优化.md) |
| 实战篇 | 第 10 章：监控与评估 | [10_监控与评估.md](10_Agent知识库技术指南_监控与评估.md) |
| 附录篇 | 第 11 章：常见坑与解决方案 | [11_常见坑与解决方案.md](11_Agent知识库技术指南_常见坑与解决方案.md) |
| 附录篇 | 第 12 章：技术栈推荐组合 | [12_技术栈推荐组合.md](12_Agent知识库技术指南_技术栈推荐组合.md) |
| 附录篇 | 第 13 章：安全、合规与伦理 | [13_安全合规与伦理.md](13_Agent知识库技术指南_安全合规与伦理.md) |
| 附录篇 | 第 14 章：接入 LLM 与 CLI 行动 Agent | [14_接入LLM与CLI行动Agent.md](14_Agent知识库技术指南_接入LLM与CLI行动Agent.md) |

> 想一口气读完？查看 [合并版](Agent知识库技术指南_合并版.md) · [导航索引](Agent知识库技术指南_导航索引.md) · [EPUB 电子书](Agent知识库技术指南.epub)

## 阅读建议

- **第一次接触 RAG**：按顺序阅读第 1-4 章，建立基础概念
- **已有经验想快速上手**：直接看第 8 章（完整实现示例），然后按需补充
- **生产环境遇到问题**：第 11 章（常见坑与解决方案）和第 13 章（安全合规）
- **要搭行动 Agent**：第 14 章（RA-A 架构：检索 → 增强 → 行动）

## 技术栈覆盖

```
文档解析:  PyMuPDF / MinerU 3.4 / Docling / Marker / Unstructured
分块策略:  RecursiveCharacterTextSplitter / 语义分块 / Late Chunking / Contextual Retrieval / Agentic Chunking
向量化:    BGE-M3 / Stella-V5 / OpenAI text-embedding-3 / Cohere
向量库:    Qdrant / Milvus / Chroma / pgvector
检索:      向量检索 + BM25 混合检索 / HyDE / 多路召回
重排序:    BGE-Reranker / Cohere Rerank / LLM-based Rerank
生成:      OpenAI / Claude / 本地模型 (Ollama)
监控:      Prometheus / Grafana / RAGAS 评估框架
```

## 仓库结构

```
agentRAGBible/
├── 00_Agent知识库技术指南_前言.md          # 前言与阅读指南
├── 01_Agent知识库技术指南_技术架构总览.md   # 第 1 章
├── 02_Agent知识库技术指南_数据处理管道.md   # 第 2 章（2026 最新更新）
├── 03_Agent知识库技术指南_向量化方案.md     # 第 3 章
├── 04_Agent知识库技术指南_向量数据库选型.md # 第 4 章
├── 05_Agent知识库技术指南_检索策略.md       # 第 5 章
├── 06_Agent知识库技术指南_重排序技术.md     # 第 6 章
├── 07_Agent知识库技术指南_上下文与LLM生成.md# 第 7 章
├── 08_Agent知识库技术指南_完整实现示例.md   # 第 8 章
├── 09_Agent知识库技术指南_性能优化.md       # 第 9 章
├── 10_Agent知识库技术指南_监控与评估.md     # 第 10 章
├── 11_Agent知识库技术指南_常见坑与解决方案.md# 第 11 章
├── 12_Agent知识库技术指南_技术栈推荐组合.md # 第 12 章
├── 13_Agent知识库技术指南_安全合规与伦理.md # 第 13 章
├── 14_Agent知识库技术指南_接入LLM与CLI行动Agent.md # 第 14 章
├── Agent知识库技术指南_合并版.md            # 全文合并版
├── Agent知识库技术指南_导航索引.md          # 章节锚点导航
├── Agent知识库技术指南.epub                 # EPUB 电子书
├── SUMMARY.md                              # 章节导读索引
├── 文档审核报告_Agent知识库技术指南.md       # 文档质量审核报告
├── spec/                                   # 各章规格说明文档
│   ├── SPEC_技术架构总览_第1章.md
│   ├── SPEC_数据处理管道_第2章.md
│   └── ... (共 13 个 spec 文件)
└── .gitignore
```

## 许可证

[MIT License](LICENSE) - 可自由复制、修改、分发，请注明出处。

## 贡献

欢迎通过 Issue 或 Pull Request 提交改进建议、纠错或新增内容。

---

*本指南持续更新，追踪 2025-2026 年 RAG 领域最新技术进展。*
