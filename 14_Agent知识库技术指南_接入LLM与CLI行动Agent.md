# 第 14 章：接入 LLM 与 CLI —— 行动 Agent 完整实现

> 本章实现"自然语言 → 安全 CLI 操作"的完整链路，作为原有 RAG 知识库的**行动层（Action Layer）**。与 1–13 章完全兼容，可渐进式实施：先有能问答的 RAG（阶段 0），再逐步接入只读 CLI、业务 CLI、多步规划与监控审计（阶段 1–4）。

---

## 14.1 整体架构：RA-A（检索-增强-行动）

在 RAG（检索增强生成）之上，增加"行动"一环，构成 **RA-A（Retrieve-Augment-Act）**：

```
用户输入
   ↓
┌─────────────────────────────────────────┐
│  1. 意图分类（IntentClassifier）         │
│     → knowledge_qa / tool_query /       │
│       tool_execute / multi_step          │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  2. 路由分发                            │
│     - 知识问答 → 原有 RAG 流程（1–7章） │
│     - 工具操作 → 进入下方行动流程        │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  3. 工具/SOP 检索（扩展检索器）           │
│     - 向量检索 type=tool 的文档块        │
│     - 返回工具 Schema + 使用示例         │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  4. 行动规划（ActionPlanner）            │
│     → 生成 JSON 行动计划                 │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  5. 参数补全与确认（ParameterCompleter） │
│     → 缺失参数则追问用户                 │
│     → 高风险操作请求确认                 │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  6. 安全执行（CLIExecutor）              │
│     → 白名单 + 路径校验 + 沙箱执行       │
│     → 超时/重试 + 结果捕获               │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│  7. 结果解读与回答生成（Generator）      │
│     → 将 stdout/stderr 转为自然语言      │
│     → 附上执行摘要与来源引用             │
└─────────────────────────────────────────┘
   ↓
最终输出（回答 + 执行结果）
```

---

## 14.2 核心组件完整代码实现

### 14.2.1 意图分类器（`controller/intent.py`）

```python
import json
from typing import Dict, Any

class IntentClassifier:
    def __init__(self, llm):
        self.llm = llm

    def classify(self, query: str) -> Dict[str, Any]:
        prompt = f"""分析用户意图，仅输出 JSON。

用户输入：{query}

意图类型：
- knowledge_qa: 纯问答，不涉及操作
- tool_query: 询问工具用法（如"怎么用xxx"）
- tool_execute: 要求执行具体操作（如"帮我生成报表"）
- multi_step: 需要多步组合操作

输出格式：
{{"intent": "类型", "confidence": 0.95, "entities": {{"tool": "推测工具名", "params": {{}}}}}}
"""
        resp = self.llm.generate(prompt)
        try:
            return json.loads(resp)
        except Exception:
            return {"intent": "knowledge_qa", "confidence": 0.5, "entities": {}}
```

### 14.2.2 行动规划器（`controller/planner.py`）

```python
import json
from typing import List, Dict

class ActionPlanner:
    def __init__(self, llm, kb):
        self.llm = llm
        self.kb = kb

    def plan(self, query: str, tool_docs: List[Dict]) -> Dict:
        tools_summary = "\n".join(
            f"- {doc['metadata']['tool_id']}: {doc['metadata']['description']}"
            for doc in tool_docs
        )
        prompt = f"""用户需求：{query}

可用工具：
{tools_summary}

请输出 JSON 行动计划（严格按以下格式）：
{{
    "action_type": "single" | "multi_step",
    "steps": [
        {{"step": 1, "tool": "工具ID", "parameters": {{"param": "value"}},
          "description": "步骤说明", "requires_confirmation": false}}
    ],
    "fallback": "失败备选方案"
}}

规则：
1. 参数值必须从用户输入提取，缺失则填 "MISSING_REQUIRED_PARAM"。
2. 涉及删除、修改配置、外发数据的步骤，requires_confirmation 设为 true。
"""
        resp = self.llm.generate(prompt)
        try:
            return json.loads(resp)
        except Exception:
            return {"action_type": "single", "steps": [], "fallback": "规划失败，请重试"}
```

### 14.2.3 安全 CLI 执行器（`controller/executor.py`）

```python
import subprocess
import os
import tempfile
import json
from typing import Dict, Any

class CLIExecutor:
    def __init__(self, config):
        self.config = config
        self.whitelist_paths = config.safe_paths
        self.banned_commands = config.banned_commands

    def safe_execute(self, tool_name: str, params: Dict) -> Dict:
        # 1. 工具白名单校验
        tool = self._get_tool_registry(tool_name)
        if not tool:
            return {"success": False, "error": f"工具 {tool_name} 未注册或不在白名单"}

        # 2. 参数消毒与校验
        sanitized = self._sanitize_params(params, tool["parameters"])
        if not sanitized["valid"]:
            return {"success": False, "error": f"参数校验失败: {sanitized['errors']}"}

        # 3. 构造命令数组（禁止 shell=True）
        cmd_parts = self._build_command_array(tool["command_template"], sanitized["values"])

        # 4. 命令注入防护
        if not self._check_command_safety(cmd_parts):
            return {"success": False, "error": "命令包含危险字符或被禁止的命令"}

        # 5. 在沙箱中执行
        try:
            with tempfile.TemporaryDirectory() as workdir:
                env = os.environ.copy()
                env["PATH"] = self.config.safe_paths_env

                result = subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    text=True,
                    timeout=self.config.default_timeout,
                    cwd=workdir,
                    env=env,
                    shell=False  # 绝对禁止
                )

                # 6. 解析输出（优先 JSON）
                stdout = result.stdout.strip()
                if tool.get("output_format") == "json":
                    try:
                        data = json.loads(stdout)
                    except Exception:
                        data = {"raw": stdout, "parse_error": True}
                else:
                    data = {"text": stdout}

                return {
                    "success": result.returncode == 0,
                    "return_code": result.returncode,
                    "stdout": stdout,
                    "stderr": result.stderr,
                    "data": data,
                    "duration_ms": 0  # 可由调用方填充
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"命令超时（>{self.config.default_timeout}s）"}
        except Exception as e:
            return {"success": False, "error": f"执行异常: {str(e)}"}

    def _check_command_safety(self, cmd_parts: list) -> bool:
        dangerous_chars = set("\\;&|`$(){}[]<>!*?")
        for part in cmd_parts:
            if any(c in dangerous_chars for c in part):
                return False
        if cmd_parts[0].lower() in self.banned_commands:
            return False
        return True

    def _sanitize_params(self, params: dict, schema: list) -> dict:
        errors = []
        validated = {}
        for p in schema:
            name = p["name"]
            value = params.get(name)
            if p.get("required") and value is None:
                errors.append(f"缺少必填参数: {name}")
                continue
            if value is not None:
                # 路径白名单
                if p["type"] == "file_path":
                    allowed = p.get("allowed_paths", [])
                    if allowed and not any(
                        os.path.realpath(value).startswith(os.path.realpath(a)) for a in allowed
                    ):
                        errors.append(f"路径 {value} 不在白名单 {allowed}")
                        continue
                # 正则校验
                if "pattern" in p:
                    import re
                    if not re.match(p["pattern"], str(value)):
                        errors.append(f"参数 {name} 格式不匹配")
                        continue
                validated[name] = value
        return {"valid": len(errors) == 0, "errors": errors, "values": validated}
```

### 14.2.4 控制器主类（`controller/agent_controller.py`）

```python
class AgentController:
    def __init__(self, config, kb, llm):
        self.config = config
        self.kb = kb
        self.llm = llm
        self.classifier = IntentClassifier(llm)
        self.planner = ActionPlanner(llm, kb)
        self.executor = CLIExecutor(config.executor)
        self.conversation = ConversationManager()  # 复用第 7 章的多轮管理

    def process(self, user_input: str) -> str:
        # 1. 意图分类
        intent = self.classifier.classify(user_input)

        # 2. 路由
        if intent["intent"] == "knowledge_qa":
            return self.kb.answer(user_input)  # 原 RAG 流程

        # 3. 工具/SOP 检索
        tool_docs = self.kb.retrieve_tools(user_input, top_k=5)

        # 4. 检查是否有待补全计划（多轮）
        pending = self.conversation.get_pending_plan()
        if pending:
            return self._continue_plan(user_input, pending, tool_docs)

        # 5. 生成行动计划
        plan = self.planner.plan(user_input, tool_docs)
        if not plan.get("steps"):
            return "无法生成有效的执行计划，请提供更具体的描述。"

        # 6. 参数补全
        completion = self._complete_params(plan)
        if completion["status"] == "need_info":
            self.conversation.set_pending_plan(plan)
            return completion["question"]

        # 7. 执行计划
        results = []
        for step in plan["steps"]:
            if step.get("requires_confirmation", False):
                confirm = self._ask_user_confirm(step["description"])
                if not confirm:
                    return f"已取消操作：{step['description']}"
            result = self.executor.safe_execute(step["tool"], step["parameters"])
            results.append({"step": step["step"], "description": step["description"], "result": result})
            if not result["success"]:
                break

        # 8. 生成最终回答
        self.conversation.clear_pending_plan()
        return self._format_final_answer(user_input, plan, results)

    def _format_final_answer(self, query, plan, results):
        context = f"执行计划：{json.dumps(plan, ensure_ascii=False)}\n执行结果：{json.dumps(results, ensure_ascii=False)}"
        prompt = f"根据执行结果，用自然语言告知用户。成功则说明完成情况，失败则解释原因和建议。\n{context}"
        return self.llm.generate(prompt)
```

---

## 14.3 安全底线（必须配置）

| 安全项 | 配置方式 |
|-------|---------|
| 命令白名单 | `executor_config.allowed_commands = ["python", "bash", "ls", "cat", ...]` |
| 路径白名单 | `executor_config.safe_paths = ["/data", "/tmp/agent"]` |
| 禁止 Shell | 强制 `subprocess.run(..., shell=False)` |
| 参数消毒 | 正则校验 + 路径规范化（`os.path.realpath`） |
| 审计日志 | 所有调用写入 `audit.log`，高风险操作实时告警 |
| 超时强制 | 每条命令必须设 `timeout`，防止阻塞 |

> 详见第 13 章 13.5（提示注入与越权）、13.5.1（命令注入防护）、13.5.2（操作审计日志）。安全底线不是"上线后补"，而是行动 Agent 的设计前提。

---

## 14.4 从零到一实施路线图

| 阶段 | 目标 | 产出 |
|------|------|------|
| **阶段0**（已有） | 标准 RAG 知识库（1–13 章） | 能问答 |
| **阶段1** | 接入一个只读 CLI（如 `ls`、`cat`） | 能执行简单查询 |
| **阶段2** | 接入业务 CLI（如 `generate_report`） | 能执行领域操作 |
| **阶段3** | 完善多步规划 + 高危确认 | 能处理复杂任务 |
| **阶段4** | 接入监控 + 审计 | 生产就绪 |

---

## 14.5 与原有指南的衔接清单

| 原有章节 | 对应新增内容 | 是否必须 |
|---------|------------|---------|
| 第 2 章（数据处理） | 工具/SOP 入库 Schema | 必须 |
| 第 5 章（检索） | 意图分类 + 工具路由 | 必须 |
| 第 7 章（生成） | 规划器 Prompt 模板 | 必须 |
| 第 8 章（实现） | Controller/Executor 代码 | 必须 |
| 第 9 章（性能） | CLI 缓存 + 超时配置 | 推荐 |
| 第 10 章（监控） | 工具调用指标 | 生产必备 |
| 第 11 章（坑） | CLI 注入/循环/版本漂移 | 建议阅读 |
| 第 13 章（安全） | 命令防护 + 审计日志 | 生产必备 |

---

## 快速启动检查清单（新增章节后）

- [ ] 已安装新增依赖（`pyyaml`、`jsonschema`）
- [ ] 已创建 `tools/registry.json` 并注册第一个 CLI 工具
- [ ] 已配置 `ExecutorConfig` 的白名单路径与禁止命令
- [ ] 已实现至少一个支持 `--json` 输出的测试 CLI
- [ ] 已运行 `AgentController.process(...)` 验证单步执行
- [ ] 已配置审计日志落盘路径
- [ ] 已测试高危操作确认流程（`requires_confirmation=True`）

---

## 本章小结

- RA-A = RAG + 行动层：意图分类 → 路由 → 工具检索 → 规划 → 确认 → 安全执行 → 结果解读
- 安全是前提：`shell=False` + 命令/路径白名单 + 审计日志，缺一不可
- 与 1–13 章完全兼容，按阶段 0→4 渐进实施，不要一上来就放开写操作
- 规划器输出是机器可读 JSON，执行器只认数组命令，全程不把用户输入当指令
