# Lineage Agent

基于 LangGraph 的最小 Agent Loop：接收一次用户输入，调用兼容 OpenAI Responses API 的模型，将结果写入状态并结束。当前使用异步、非流式调用，每次运行独立，不保存会话或执行检查点。

## 运行

开发基线为 Python 3.12，依赖由 uv 管理，首个运行环境为 Windows。项目配置声明 Python >=3.12，`.python-version` 选择 3.12；其他 Python 版本和操作系统尚未验证。

准备 Python 3.12 和 uv，在项目根目录用 PowerShell 执行：

```powershell
uv sync --locked
Copy-Item .env.example .env
```

已有 `.env` 时保留原文件。填写其中的模型连接信息：

```dotenv
BASE_URL=https://your-provider.example/v1
API_KEY=your-api-key
MODEL_NAME=your-model-name
TIMEOUT_SECONDS=60
```

`BASE_URL` 是支持 Responses 协议的 API 基础地址，按供应商要求包含版本前缀，不追加 `/responses`。仅支持 Chat Completions 的地址不满足本项目要求。`BASE_URL`、`API_KEY`、`MODEL_NAME` 必填，超时默认为 60 秒且必须为有限正数。

配置通过 `python-dotenv` 显式读取当前工作目录的 `.env`，因此运行命令应位于项目根目录；已有进程环境变量优先。真实 `.env` 已加入忽略规则，密钥仅用于创建模型客户端，不进入 Agent State 或应用日志。

```powershell
uv run lineage-agent "请用一句话介绍你自己"
```

也可以使用模块入口：

```powershell
uv run python -m lineage_agent "请用一句话介绍你自己"
```

正常执行时输出模型回答并退出 `0`；参数或配置错误退出 `2`，模型调用失败退出 `1`。模型响应失败、不完整或缺少文本时明确报错。SDK 自动重试已关闭，本步不进行应用层重试。

## 模块职责

```text
src/lineage_agent/
├── __main__.py           # 转发到 CLI
├── entrypoints/
│   └── cli.py            # 参数、运行调用和结果展示
├── agent/
│   ├── graph.py          # 节点、执行边和图构建
│   ├── state.py          # AgentState
│   └── runtime.py        # 依赖组装、状态初始化和客户端生命周期
├── models/
│   ├── responses.py      # Responses 请求与响应解析
│   └── types.py          # 模型返回值与可调用对象类型
└── config/
    └── settings.py       # dotenv 加载与配置校验
```

各目录均为 Python 包；上图省略 `__init__.py`。

`build_graph(call_model)` 接收异步模型调用函数，构建 `START → model → END`。模型节点只读取输入并返回状态更新，不构造 SDK 请求。`run_agent(user_input, settings)` 管理一次执行及客户端关闭，返回最终 State；CLI 负责调用该函数和显示结果。

## State

| 字段 | 用途 | 更新方式 |
| --- | --- | --- |
| `user_input: str` | 本次用户输入 | 初始化时写入；模型节点只读 |
| `result: NotRequired[ModelResult]` | 本次模型结果 | 初始不存在；节点成功后一次写入 |

`ModelResult.text` 用于展示；`raw_response` 保留本次 Responses 响应的 JSON 可序列化数据，包括实际返回的输出协议项和消耗信息。缺失的消耗数据不会被估算或补齐。原始响应只保存在本次返回状态中，不持久化，也不直接作为后续模型输入。

节点返回 `{"result": result}`，由 LangGraph 默认规则更新字段，保留原始 `user_input`。到达 `END` 即正常结束，不另存状态标记或循环计数。异常向调用方传播，不生成伪成功结果。

## 后续扩展位置

Tools 可以通过新增执行节点和条件边接入；Skills 可以在模型输入构造处扩展。当前不提供工具调度、Skills 加载、多轮对话、持久化或恢复机制，也不包含测试代码和测试依赖。

项目设计依据见 [技术选型](docs/技术选型.md) 和 [开发阶段划分](docs/开发阶段划分.md)。
