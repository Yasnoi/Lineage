# Lineage Agent

基于 LangGraph 和 Responses API 的最小 ReAct 循环。Agent 可以直接回答，也可以请求工具，在工具结果回传后继续调用模型，直到给出最终回答或达到模型调用上限。

每次 `run_agent` 处理一次用户请求，使用异步、非流式模型调用，状态只在本次运行中保留。CLI 默认不提供工具；配置 `PROJECT_ROOT` 后，可以只读列举该项目目录的直接子项。目录名称和结构不能证明已经理解代码，当前不读取文件正文。

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
MAX_MODEL_CALLS=5
LOG_DIR=logs
PROJECT_ROOT=
DIRECTORY_MAX_ENTRIES=100
```

`BASE_URL` 是支持 Responses 协议的 API 基础地址，按供应商要求包含版本前缀，不追加 `/responses`。仅支持 Chat Completions 的地址不满足本项目要求。`BASE_URL`、`API_KEY`、`MODEL_NAME` 必填；超时默认为 60 秒且必须为有限正数，模型调用上限默认为 5 次且必须为正整数。

`PROJECT_ROOT` 可选，未配置或留空时保持无工具问答。`DIRECTORY_MAX_ENTRIES` 默认为 100，必须为正整数；根目录和条目上限均由调用方控制，模型不能修改。

配置通过 `python-dotenv` 显式读取当前工作目录的 `.env`，因此运行命令应位于项目根目录；已有进程环境变量优先。真实 `.env` 已加入忽略规则，密钥仅用于创建模型客户端，不进入 Agent State 或应用日志。

```powershell
uv run lineage-agent "请用一句话介绍你自己"
```

启用目录工具时，在 `.env` 中将 `PROJECT_ROOT` 设置为要分析的目录，例如 `C:/path/to/project`，再运行：

```powershell
uv run lineage-agent "请先列举项目根目录，再列举其中的 src 子目录，只报告实际看到的文件路径"
```

被分析项目可以与 Lineage Agent 自身目录不同。相对 `PROJECT_ROOT` 以 dotenv 文件所在目录为基准；无效根目录在调用模型前报告配置错误。恢复无工具 CLI 时，将配置中的 `PROJECT_ROOT` 留空，并确保进程环境没有覆盖该值。

也可以使用模块入口：

```powershell
uv run python -m lineage_agent "请用一句话介绍你自己"
```

CLI 仅保留一个用户输入参数，将最终模型回答写入 stdout，将运行标识、日志位置、记录完整性和实际工具处理统计写入 stderr。退出码 `0` 表示得到最终回答，`1` 表示模型、协议或日志错误，`2` 表示参数或配置错误，`3` 表示模型调用次数耗尽、请求尚未完成；Ctrl-C 退出 `130`。达到上限时不会把最近一轮的中间文本当作最终回答。

SDK 自动重试已关闭，不使用服务端会话或 `previous_response_id`。每次请求都使用本地当前上下文。

## 模块职责

```text
src/lineage_agent/
├── __main__.py           # 转发到 CLI
├── entrypoints/
│   └── cli.py            # 参数、运行调用和结果展示
├── agent/
│   ├── graph.py          # 条件路由、调用计数与本次请求的 usage 累计
│   ├── state.py          # AgentState
│   └── runtime.py        # 依赖组装、状态初始化和客户端生命周期
├── models/
│   ├── responses.py      # Responses 请求与响应解析
│   └── types.py          # 模型返回值与可调用对象类型
├── tools/
│   ├── execution.py      # 串行执行、参数解析和工具结果/错误转换
│   ├── types.py          # 执行器接口与参数错误类型
│   └── filesystem/
│       ├── paths.py      # 根目录绑定和访问范围检查
│       └── directory.py  # 目录工具 Schema、字段校验和直接子项列举
├── recording/
│   ├── recorder.py       # 每次运行的 JSONL 文件、事件和步骤标识
│   └── summaries.py      # 确定性的参数与结果摘要
└── config/
    └── settings.py       # dotenv 加载与配置校验
```

各目录均为 Python 包；上图省略 `__init__.py`。

执行图为 `START → model → tools → model`，模型给出最终回答、模型失败、达到上限或日志故障时转入 `END`。不设置单独的“思考节点”。

`build_graph(call_model, *, tools=None, execute_tool=None, max_model_calls=5, recorder=None)` 接收已绑定连接和工具定义的异步模型函数，负责状态更新与路由。`run_agent` 创建本次运行独享的记录器并注入图及工具执行层；模型解析和 SDK 调用集中在 models 模块。

程序入口为 `await run_agent(input, settings, *, tools=None, execute_tool=None)`，返回最终 State，并在成功或失败后关闭客户端。`input` 可以是字符串，也可以是包含消息、工具请求、工具结果及 reasoning 等内容的完整 Responses 协议项列表；传入列表会被复制，调用方的数据不被原地修改。

初始上下文中的 `function_call` 与 `function_call_output` 必须使用非空字符串 `call_id`，请求先于结果且一一对应；缺失、重复或无法配对时返回错误终态，模型调用次数为零。其他协议项原样保留。模型响应不得夹带由应用生成的 `function_call_output`，否则按协议错误终止；字典形式的响应同样深拷贝，避免不同运行共享嵌套数据。

模型接口为 `await generate_response(client, model_name, input, *, tools=None)`。工具定义使用 Responses 的 `type: "function"` 格式及 JSON Schema，不使用 Chat Completions 的工具格式。

## 目录列举

`list_directory` 只接受 `{"path": "."}` 形式的参数；`path` 必填且为字符串，不接受额外字段。`.` 表示绑定根目录，也可以传入 `src`、`src/package` 等相对路径。Schema 使用严格模式，工具仍自行执行字段和路径校验。

Python 运行接口 `run_agent` 不自动根据配置启用工具。调用方可以显式绑定目录，再通过现有接口注入；省略 `tools` 和 `execute_tool` 即为无工具运行：

```python
import asyncio

from lineage_agent.agent.runtime import run_agent
from lineage_agent.config.settings import load_settings
from lineage_agent.tools.filesystem.directory import (
    DIRECTORY_TOOL, create_directory_executor,
)

settings = load_settings()
executor = create_directory_executor("C:/path/to/project", max_entries=100)
state = asyncio.run(run_agent(
    "请列举项目根目录", settings,
    tools=[DIRECTORY_TOOL], execute_tool=executor,
))
```

直接构造配置或执行器时，相对根目录以当前工作目录为基准。执行器不读取配置、不改变工作目录，根目录绑定于该执行器；不同项目应各自创建执行器。配置读取只绝对化路径，访问校验集中在 filesystem 模块，绑定时才检查根目录并规范化。

执行器成功返回一个 JSON 字符串，再由通用执行层封装到 `{"ok": true, "result": "..."}`。内层结构为：

```json
{
  "path": ".",
  "entries": [{"name": "src", "path": "src", "type": "directory"}],
  "limit": 100,
  "truncated": false
}
```

所有返回路径均相对于项目根目录，使用 `/` 分隔。类型包括 `file`、`directory`、`symlink`、`junction`、`reparse_point` 和 `other`。不读取正文，也不展开链接目标；不返回绝对路径或链接的外部目标。

仅使用 Python 标准库 `os.scandir`，按文件系统枚举顺序最多消费上限加一项，保留前 N 项后按 `(name.casefold(), name)` 排序。超出上限时 `truncated=true`，未返回项不能视为不存在；排序仅针对已保留子集，不保证是全目录排序的前 N 项，重复列举也不保证得到同一子集。空目录成功返回空列表且 `truncated=false`。本轮限制条目数，不提供分页、总字节预算或严格执行耗时上限。

不应用隐藏项或 `.gitignore` 过滤，`.git` 等目录项也包含在列举范围内。链接条目可以显示自身名称和类型，但不能作为目标或中间路径进入，即使链接指向根目录内部也会拒绝。其他特殊条目仅报告类型，不展开。

路径接受 `/` 和 `\` 分隔，拒绝任何 `..` 组件、绝对路径、盘符相对路径、UNC/设备路径、ADS、控制字符、Windows 保留字符和设备名，以及组件末尾的点或空格。不展开环境变量或 `~`。根目录自身及祖先不得经过符号链接或 Windows reparse point；每次访问重新检查路径组件，并使用解析后的路径进行范围核对。

字段、路径范围、链接和非目录目标通过 `ToolArgumentError` 计为拒绝；不存在、权限不足及其他 I/O 错误计为执行失败。枚举或条目分类中途发生异常时，整次调用失败，不返回部分成功结果；同批后续请求仍按原有规则处理。成功列举不是文件系统快照，枚举期间的新增或删除可能影响结果。

这些是应用层的路径访问检查，检查与实际操作并非原子，不能抵御恶意并发替换目录的所有情况，不构成操作系统安全沙箱。日志继续只记录参数和结果的类型、长度等安全摘要，不新增原始路径或文件名预览。

## State

| 字段 | 用途 | 更新方式 |
| --- | --- | --- |
| `input` | 本次请求的完整上下文，替代原 `user_input` | 初始化用户项；每轮只追加新增协议项 |
| `run_id` | 本次用户请求的 UUID | 初始化一次，运行期间不变 |
| `log_path` | 本次 JSONL 文件的绝对路径 | 初始化一次，失败时也可定位预期文件 |
| `log_complete` | 本次运行记录是否完整 | 初始为 `False`；结束事件成功写入且文件成功关闭后才为 `True` |
| `result` | 最近一次模型调用结果 | 每次模型尝试替换；无响应的失败设为 `None` |
| `model_calls` | 本次请求内的模型调用尝试数 | 每次尝试加一，包括失败 |
| `tool_requests` | 本次请求内收到的工具请求数 | 按条数增加，包括未知工具、无效参数及因上限未执行的请求 |
| `tool_calls` | 本次请求内实际调用执行器的尝试数 | 进入执行器前增加；执行器异常也计数 |
| `tool_outcomes` | 实际成功、校验拒绝、执行失败和未执行数量 | 本次运行内累加 `succeeded/rejected/failed/skipped`，不统计调用方传入的历史结果 |
| `usage_totals` | 本次 Agent 处理一次用户请求期间的已知 Token 小计 | 每次模型尝试后累加实际报告的数据 |
| `stop_reason` | `completed`、`max_model_calls` 或 `error` | 运行中为 `None`，结束时写入 |
| `error` | 导致整个循环终止的错误 | 正常运行为空；可恢复工具错误保存在工具结果中 |

`ModelResult` 包含 `text`、`raw_response`、`output_items`、`tool_calls` 和 `usage`。工具请求保留 `call_id`、名称及原始参数字符串；原始响应保留可序列化的协议数据。下一轮上下文仅追加 `output_items`，包括 reasoning 和端点扩展字段，不追加整个响应对象，也不重复追加历史。

`result.usage` **仅表示最近一次模型调用**的输入、输出和总 Token。未报告 usage 时为 `None`；部分字段缺失时相应值为 `None`，不沿用上一轮。

`state.usage_totals` **仅统计一次 `run_agent` 处理一次用户请求**，包括该请求内部全部模型调用。每次新运行都重新初始化，与其他请求无关：

- `input_tokens`、`output_tokens`、`total_tokens` 初始均为 `None`，收到实际数值后才累加；缺失或无效字段不补零、不推算。
- `missing_usage_calls` 记录没有完整 usage 的模型尝试数，包括未收到响应的失败。大于零时，累计值只是已知小计，不能视为完整消耗。
- 失败响应若包含可用 usage，仍累计其中已知数据；最近结果和累计数据各自只更新一次。

## 工具调用与终止

执行器是异步可调用对象：`execute_tool(name, arguments) -> str`。提供工具定义时须同时注入执行器；工具定义、执行器及调用上限作为运行依赖，不存入 State。

纯工具请求是有效模型响应；文本与工具请求并存时先处理工具。每批按请求顺序串行执行，所有结果按 `call_id` 回传后再调用模型。工具名称必须属于本次提供的定义，参数必须可解析为 JSON 对象；具体字段由执行器校验，不提供通用 JSON Schema 校验引擎。

参数中的每个 JSON 对象都必须具有唯一键名，包括嵌套对象和数组中的对象。按 JSON 解码后的键名精确比较，例如 `key` 与 `\u006bey` 视为同名；不做大小写折叠或 Unicode 归一化。不同对象各自使用同名字段允许通过。重复键在 `argument_parsing` 阶段返回 `invalid_arguments`，计为拒绝且不进入执行器；同批其他请求继续处理。此前重复键会静默保留后值，现在明确拒绝，错误说明不包含原始参数或键名。Schema 的 `strict: true` 不能替代这些解析检查或工具自身的字段与业务校验。

工具结果使用 `function_call_output`，其 `output` 为 JSON 字符串。成功返回 `{"ok": true, "result": "..."}`；失败返回 `{"ok": false, "error": {"code": "...", "message": "..."}}`。错误码为 `unknown_tool`、`invalid_arguments` 或 `execution_error`。参数语义错误由执行器抛出 `ToolArgumentError`；其他执行器异常使用安全的通用错误消息。

工具错误会回传模型，继续处理同批其他请求，并允许模型在剩余次数内修正。未知工具和 JSON 解析失败不会进入执行器，不增加 `tool_calls`；执行器内部校验失败或执行异常仍计一次。得到最终回答不代表此前失败的工具操作已经成功。

`ToolArgumentError` 计入 `rejected`，即使它发生在执行器内部；其他执行异常计入 `failed`。`completed` 仅表示模型给出了最终回答，不表示所有工具成功或业务目标已验证。CLI 始终展示程序统计；模型在工具失败后输出“成功”不会覆盖日志和失败计数。

执行器返回非字符串同样计为一次执行尝试和一次失败，返回 `execution_error`；不自动将返回值转换成字符串或 JSON。同批成功、失败和拒绝结果均按原请求顺序关联，全部处理后才发起下一次模型请求。若写入拒绝或失败事件时日志故障，已有统计及已生成结果仍留在 State，剩余请求跳过且不再调用模型；State 中存在错误结果并不表示它已经回传。

第 N 次模型调用若已产生最终回答，仍可正常完成；若继续请求工具，则以 `max_model_calls` 结束，不执行该批工具。模型或协议错误以 `error` 结束，返回已产生的上下文、计数、可取得的响应及 usage，而不是丢失失败节点的统计。配置或输入错误在调用模型前报告。

当前只保证模型调用轮数有界，不对单次工具执行设置超时。单个异步工具如果长期不返回，模型轮数上限不能终止它。

## 运行记录

`LOG_DIR` 默认为 `logs`，每次运行创建独立的 `<run_id>.jsonl`。通过 `load_settings` 加载时，相对路径以 dotenv 文件所在目录为基准；直接构造 `ModelSettings` 时相对当前工作目录解析。调用方从返回 State 的 `run_id`、`log_path` 和 `log_complete` 获取记录信息。默认日志目录被 Git 忽略；自定义目录需自行安排保留与清理。

每行是 UTF-8 JSON，写入后立即 flush。记录器按运行绑定，不使用共享的可变运行标识，日志也不进入模型上下文。通用字段包括 UTC `timestamp`、运行内递增的 `event_seq`、`event` 和 `run_id`。每次模型或工具节点执行分配 `step-0001` 形式的 `step_id`，并记录 `node`；同批工具共享步骤，运行级事件省略步骤。

工具事件保留原 `call_id`、`tool_name` 和从 1 开始的 `request_index`（本轮响应中的工具请求序号）。缺失标识不补造；重复标识结合步骤和请求序号定位。

| 事件 | 记录的事实 |
| --- | --- |
| `run.started` / `run.ended` | 开始与终止；结束时含调用统计、工具处理统计、最近 usage 和本次请求累计 usage |
| `run.input_rejected` | 初始工具关联校验失败，含协议项位置与错误码 |
| `model.started` | 模型调用即将发起，含实际输入中全部、有序的 `input_tool_result_call_ids` |
| `model.returned` / `model.failed` | 可取得的响应摘要与 usage，或固定错误码及失败阶段 |
| `tool.rejected` / `tool.started` / `tool.succeeded` / `tool.failed` | 名称、JSON、字段校验或实际执行的结果 |
| `tool.result_created` | 对应 `call_id` 的协议结果已经生成，与执行成功分开记录 |
| `tool.skipped` | 因模型上限、协议错误或日志故障而未执行 |

核对回传时，先找到 `tool.result_created`，再检查下一次 `model.started` 的关联列表；工具成功本身不能证明结果已进入后续输入。该列表包括当次输入的全部历史工具结果，同一个结果可出现在多个后续模型请求中。调用开始记录表示已进入请求边界，不表示远端已成功接收；模型返回或失败事件给出后续结果。

参数摘要仅允许出现工具 Schema 中的字段名，敏感字段名显示为 `[redacted]`，未知字段名显示为 `[unknown]`。不预览任何参数值；只记录类型、字符串长度或集合条数，嵌套对象不展开。模型文本、工具返回值和无效 JSON 也只记录类型与大小。每份摘要最多 16 个字段、字段名最多 64 字符、序列化后最多 1 KiB；敏感信息处理先于截断，省略项明确标记。异常仅记录固定错误码和阶段，不包含原始异常文本、配置环境、凭据、完整 State 或原始模型响应。摘要不修改工具实际参数和回传内容。

日志创建或写入失败会停止后续模型、工具调用，以 `error` 结束并设置 `log_complete=False`。已发生的调用、结果和 usage 保留；未启动的剩余工具计入 `skipped`。开始事件写入失败时不启动对应调用；结束写入或关闭失败也不会宣称记录完整。故障可能使文件不存在或只含部分记录，以返回 State 和 CLI 的完整性标志为准。配置异常或主动取消仍沿用原异常/取消行为，已打开的记录尝试写入 `run.aborted` 并关闭，不伪装正常结束。

日志不属于可恢复检查点，也不提供进程崩溃后的任务恢复或写入持久性保证。

## 独立验证

从项目根目录执行：

```powershell
# 确定性响应替身：协议、控制流、异常、计数、usage 与运行记录
uv run --locked python scripts/check_react.py

# 当前 .env 端点：无工具回归及固定结果工具的完整循环
uv run --locked python scripts/verify_function_calling.py

# 真实临时目录与确定性模型：目录边界、截断、失败和实际回传
uv run --locked python scripts/check_filesystem.py

# 当前 .env 端点：在非敏感临时目录中列举根目录、子目录并回答
uv run --locked python scripts/verify_directory.py
```

验证代码保留在独立脚本中，生产模块不导入脚本，CLI 不增加验证参数。`check_react.py` 和 `verify_function_calling.py` 自带临时固定结果工具；目录检查使用生产目录工具和真实临时目录。确定性检查使用标准库断言和模型替身，不需要测试框架或真实凭据。真实固定结果验证需至少两次模型调用额度；目录验证需至少三次额度，检查随机临时文件路径进入最终回答，不要求固定的实际调用次数，也不将一次回答视为具备代码理解能力。

确定性检查将日志保留在 `logs/checks/<检查标识>/`，逐场景输出运行、步骤、工具调用和日志路径。检查将日志中的关联列表与替身实际捕获的模型请求逐项比对，并覆盖工具失败后修正或误报成功、嵌套数据与并发运行隔离、摘要处理和日志 I/O 故障。真实脚本同时显示运行标识、工具处理统计和日志完整性。

检查还覆盖同批部分失败、执行器非字符串返回、各层级重复参数键，以及 `tool.rejected`、`tool.failed`、`model.failed` 写入时再次发生故障。故障场景联合核对返回 State、实际调用记录和日志完整性；未落盘事件的定位信息来自故障注入器观察到的写入尝试，并单独标明来源，不视为已经持久化的日志。

目录检查覆盖 Unicode 名称、上下级目录、空目录、条目上限、参数及路径拒绝、同批失败后继续、不同根目录运行隔离、配置和 CLI。权限与中途 I/O 故障使用标注的注入；Windows symlink/junction 尝试创建真实样例，无法构造时单独报告跳过原因。工具真实返回值与后续 SDK 请求逐项比对，日志关联和安全摘要分别核对。

这两类结果分别报告，确定性检查不能代替真实端点的 Function Calling 兼容性验证。

当前只提供目录直接子项列举，不实现文本读取、文件搜索、文件修改或命令执行，不包括工具自动发现、Skills、流式输出、并行执行、持久化或恢复机制。

项目设计依据见 [技术选型](docs/技术选型.md) 和 [开发阶段划分](docs/开发阶段划分.md)。
