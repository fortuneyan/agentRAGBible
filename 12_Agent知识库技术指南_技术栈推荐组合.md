# 第 12 章：技术栈推荐组合

> 没有最好的方案，只有最适合你的方案。

---

## 轻量级方案（个人/小团队）

### 适用场景

- 数据量 < 10 万条
- 并发 < 10 QPS
- 预算有限
- 快速验证

### 技术栈

| 组件 | 推荐 | 理由 |
|-----|------|-----|
| Embedding | bge-large-zh-v1.5（本地） | 中文效果好，免费 |
| 向量数据库 | Chroma | 一分钟上手 |
| Reranker | bge-reranker-large（本地） | 效果好 |
| LLM | GPT-4 或本地模型 | 灵活选择 |

### 部署方式

```bash
# Docker Compose
version: '3.8'
services:
  chroma:
    image: chromadb/chroma
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
  
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - CHROMA_HOST=chroma
      - EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
```

### 优缺点

**优点**：
- 快速启动（30 分钟）
- 成本低（几乎免费）
- 维护简单

**缺点**：
- 数据量大了会卡
- 不支持分布式
- 性能有限

---

## 标准方案（中型团队）

### 适用场景

- 数据量 10-100 万条
- 并发 10-100 QPS
- 中等预算
- 生产环境

### 技术栈

| 组件 | 推荐 | 理由 |
|-----|------|-----|
| Embedding | bge-large-zh-v1.5（本地） | 性价比高 |
| 向量数据库 | Qdrant | 性能好，易部署 |
| Reranker | bge-reranker-large（本地） | 效果好 |
| LLM | GPT-4 | 效果稳定 |
| 缓存 | Redis | 提升性能 |

### 部署方式

```bash
# Docker Compose
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  app:
    build: .
    ports:
      - "5000:5000"
    depends_on:
      - qdrant
      - redis
```

### 优缺点

**优点**：
- 性能好
- 可扩展
- 成本适中

**缺点**：
- 需要一定运维能力
- 组件较多

---

## 生产级方案（大型团队）

### 适用场景

- 数据量 > 100 万条
- 并发 > 100 QPS
- 预算充足
- 高可用要求

### 技术栈

| 组件 | 推荐 | 理由 |
|-----|------|-----|
| Embedding | text-embedding-3-large（API） | 效果最好 |
| 向量数据库 | Milvus 集群 | 大规模支持 |
| Reranker | Cohere Rerank（API） | 省心 |
| LLM | GPT-4 + 本地模型备用 | 高可用 |
| 缓存 | Redis 集群 | 高性能 |
| 监控 | Prometheus + Grafana | 可观测 |
| 日志 | ELK | 日志分析 |

### 部署方式

```bash
# Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-kb
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-kb
  template:
    spec:
      containers:
      - name: app
        image: agent-kb:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### 优缺点

**优点**：
- 高可用
- 高性能
- 可扩展

**缺点**：
- 成本高
- 运维复杂
- 需要专业团队

---

## 方案对比

| 维度 | 轻量级 | 标准 | 生产级 |
|-----|-------|-----|-------|
| 启动时间 | 30 分钟 | 2 小时 | 1 天 |
| 数据容量 | 10 万 | 100 万 | 无限制 |
| 并发 | 10 QPS | 100 QPS | 1000+ QPS |
| 月成本 | ¥0 | ¥500 | ¥5000+ |
| 运维难度 | 简单 | 中等 | 复杂 |

---

## 选型建议

### 如果你是...

**学生/个人开发者**：轻量级方案
- 先跑通，再优化
- 用免费工具
- 积累经验

**小团队（2-5 人）**：标准方案
- 性能和成本平衡
- Qdrant 是不错的选择
- 逐步优化

**中型团队（5-20 人）**：标准方案或生产级方案
- 根据业务需求定
- 考虑长期维护成本

**大型团队（20+ 人）**：生产级方案
- 高可用是必须的
- 专业运维团队
- 完善的监控体系

---

## 渐进式演进

最好的方式是从简单开始，逐步演进：

```
阶段1：原型验证
    Chroma + 本地模型
    ↓
阶段2：小规模生产
    Qdrant + 本地模型 + Redis
    ↓
阶段3：大规模生产
    Milvus 集群 + API 模型 + 完整监控
```

---

## 本章小结

- 没有最好的方案，只有最适合的方案
- 从简单开始，逐步演进
- 根据数据量、并发、预算来选型
- 生产环境要考虑高可用

---

*技术选型不是一锤子买卖，要根据业务发展持续调整。*
