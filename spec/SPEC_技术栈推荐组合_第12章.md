# SPEC_技术栈推荐组合_第12章

> 技术规格说明书 - 技术栈推荐组合

---

## 1. 章节概述

### 1.1 目标

定义不同场景下的技术栈选型标准和部署规范。

### 1.2 范围

- 方案分级标准
- 技术栈配置
- 部署架构
- 成本估算

---

## 2. 方案分级标准

### 2.1 分级维度

| 维度 | 轻量级 | 标准 | 生产级 |
|-----|-------|-----|-------|
| 数据量 | < 10万 | 10-100万 | > 100万 |
| 并发 | < 10 QPS | 10-100 QPS | > 100 QPS |
| 预算 | ¥0-500/月 | ¥500-5000/月 | > ¥5000/月 |
| 团队 | 1-2人 | 3-10人 | > 10人 |
| 可用性 | 99% | 99.9% | 99.99% |

### 2.2 场景匹配

| 场景 | 推荐方案 |
|-----|---------|
| 学生/个人开发者 | 轻量级 |
| 小团队 MVP | 轻量级 |
| 小团队生产环境 | 标准 |
| 中型团队 | 标准 |
| 大型团队 | 生产级 |
| 企业级应用 | 生产级 |

---

## 3. 轻量级方案

### 3.1 技术栈配置

```python
@dataclass
class LightweightConfig:
    # Embedding
    embedding_model: str = "m3e-base"
    embedding_dim: int = 768
    embedding_device: str = "cpu"
    
    # Vector Store
    vector_store: str = "chroma"
    vector_store_path: str = "./chroma_data"
    
    # Reranker
    reranker_enabled: bool = False
    
    # LLM
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
```

### 3.2 部署架构

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./chroma_data:/app/chroma_data
    environment:
      - EMBEDDING_MODEL=m3e-base
      - VECTOR_STORE=chroma
```

### 3.3 成本估算

| 组件 | 月成本 |
|-----|-------|
| 服务器 | ¥0（本地） |
| LLM API | ¥100-500 |
| 存储 | ¥0（本地） |
| **总计** | **¥100-500** |

### 3.4 优缺点

**优点**：
- 快速启动（30 分钟）
- 成本低（几乎免费）
- 维护简单

**缺点**：
- 数据量大了会卡
- 不支持分布式
- 性能有限

---

## 4. 标准方案

### 4.1 技术栈配置

```python
@dataclass
class StandardConfig:
    # Embedding
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"
    
    # Vector Store
    vector_store: str = "qdrant"
    vector_store_host: str = "localhost"
    vector_store_port: int = 6333
    
    # Reranker
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-large"
    
    # LLM
    llm_model: str = "gpt-4"
    llm_temperature: float = 0
    
    # Cache
    cache_enabled: bool = True
    cache_size: int = 10000
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
```

### 4.2 部署架构

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
    deploy:
      resources:
        limits:
          memory: 4G
  
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
    environment:
      - VECTOR_STORE_HOST=qdrant
      - REDIS_HOST=redis
```

### 4.3 成本估算

| 组件 | 月成本 |
|-----|-------|
| 服务器（4核8G） | ¥200-400 |
| LLM API | ¥500-2000 |
| 存储 | ¥50-100 |
| **总计** | **¥750-2500** |

### 4.4 优缺点

**优点**：
- 性能好
- 可扩展
- 成本适中

**缺点**：
- 需要一定运维能力
- 组件较多

---

## 5. 生产级方案

### 5.1 技术栈配置

```python
@dataclass
class ProductionConfig:
    # Embedding
    embedding_provider: str = "openai"  # local | openai
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    
    # Vector Store
    vector_store: str = "milvus"
    vector_store_cluster: bool = True
    vector_store_nodes: int = 3
    
    # Reranker
    reranker_provider: str = "cohere"
    reranker_model: str = "rerank-english-v3.0"
    
    # LLM
    llm_model: str = "gpt-4"
    llm_fallback: str = "qwen-72b"
    llm_temperature: float = 0
    
    # Cache
    cache_provider: str = "redis"
    cache_cluster: bool = True
    cache_size: int = 100000
    
    # Monitoring
    monitoring_enabled: bool = True
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
```

### 5.2 部署架构

```yaml
# kubernetes deployment
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
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        ports:
        - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: agent-kb
spec:
  selector:
    app: agent-kb
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### 5.3 成本估算

| 组件 | 月成本 |
|-----|-------|
| Kubernetes 集群 | ¥2000-5000 |
| Milvus 集群 | ¥1000-3000 |
| LLM API | ¥2000-10000 |
| Redis 集群 | ¥500-1000 |
| 监控系统 | ¥500-1000 |
| **总计** | **¥6000-20000** |

### 5.4 优缺点

**优点**：
- 高可用
- 高性能
- 可扩展

**缺点**：
- 成本高
- 运维复杂
- 需要专业团队

---

## 6. 方案对比

### 6.1 功能对比

| 功能 | 轻量级 | 标准 | 生产级 |
|-----|-------|-----|-------|
| 向量检索 | ✓ | ✓ | ✓ |
| BM25 检索 | ✗ | ✓ | ✓ |
| 混合检索 | ✗ | ✓ | ✓ |
| 重排序 | ✗ | ✓ | ✓ |
| 缓存 | ✗ | ✓ | ✓ |
| 监控 | ✗ | 基础 | 完整 |
| 高可用 | ✗ | ✗ | ✓ |
| 水平扩展 | ✗ | ✗ | ✓ |

### 6.2 性能对比

| 指标 | 轻量级 | 标准 | 生产级 |
|-----|-------|-----|-------|
| 延迟 (P95) | < 2s | < 500ms | < 200ms |
| 吞吐量 | 1 QPS | 50 QPS | 500+ QPS |
| 数据容量 | 10万 | 100万 | 无限制 |

---

## 7. 渐进式演进

### 7.1 演进路径

```
阶段1：原型验证
    轻量级方案
    ↓ 验证可行性
阶段2：小规模生产
    标准方案
    ↓ 业务增长
阶段3：大规模生产
    生产级方案
```

### 7.2 迁移指南

```python
class MigrationGuide:
    def migrate_lightweight_to_standard(self):
        """从轻量级迁移到标准方案"""
        steps = [
            "1. 部署 Qdrant 和 Redis",
            "2. 迁移数据到 Qdrant",
            "3. 配置 Reranker",
            "4. 更新应用配置",
            "5. 验证功能",
            "6. 切换流量"
        ]
        return steps
    
    def migrate_standard_to_production(self):
        """从标准迁移到生产级方案"""
        steps = [
            "1. 部署 Kubernetes 集群",
            "2. 部署 Milvus 集群",
            "3. 配置监控系统",
            "4. 迁移数据到 Milvus",
            "5. 配置高可用",
            "6. 灰度发布",
            "7. 全量切换"
        ]
        return steps
```

---

## 8. 测试用例

### 8.1 配置验证

```python
def test_lightweight_config():
    config = LightweightConfig()
    
    assert config.embedding_model == "m3e-base"
    assert config.vector_store == "chroma"

def test_standard_config():
    config = StandardConfig()
    
    assert config.embedding_model == "BAAI/bge-large-zh-v1.5"
    assert config.vector_store == "qdrant"
    assert config.reranker_enabled == True
```

---

*文档版本：2.0*
*更新日期：2026-07*
*变更：版本对齐正文 v2（单一事实来源，删除冗余合集 00 后确立）；内容无变更，仅版本升级*
