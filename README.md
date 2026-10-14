# Lineage Agent

Lineage Agent 面向需要长期理解和持续迭代的科研或软件项目。
接手一个项目时，需求、研究记录、代码和验证结果往往分散在不同材料中；
项目希望帮助使用者理清已有成果对应哪些目标、设计依据是什么，以及下一步如何推进。

长期目标是把项目理解、问题研究和代码迭代连接起来：
先建立需求与成果之间的对应关系，再围绕具体问题研究方案，经用户确认后继续开发。
另一条主线是保存探索路径和任务状态，让代码、依据与上下文能够一同回访、分叉和恢复。

当前仓库提供这套流程的命令行基础，使用 Python、LangGraph 和 OpenAI 兼容的 Responses API，
已实现模型与工具循环、受限文件读取、进程内多轮会话、上下文摘要及运行记录。
这些基础用于继续开发和检验 Agent 行为，完整的长期项目工作流仍在建设中。

## 当前能力边界

| 能力 | 当前范围 |
| --- | --- |
| 模型交互 | 单次请求和多轮对话；模型可请求工具，程序执行后继续调用模型 |
| 项目材料 | 在配置的根目录内列举目录、读取限额内的 UTF-8 普通文本文件 |
| 会话 | 在同一进程内新建、继续和切换会话 |
| 上下文 | 按本地估算检查请求预算，必要时摘要压缩，并提供原文检索与回读工具 |
| 记录 | 保存运行日志、原始交互和较大的工具结果；可选启用 LangSmith 追踪 |
| 验证 | 离线工程检查，以及需单独配置的项目任务 Benchmark |

读取源码和文档使模型具备分析这些材料的输入条件，不等于已经建立完整的需求映射或验证了项目结论。
目录列举不递归；文本读取受单文件和单次运行总量限制，超限时拒绝读取，不截取文件冒充完整正文。
目前不提供 PDF 解析。

摘要是有损的上下文整理，可能遗漏细节，也可能保留错误判断。
原文检索与回读用于补充依据，不能保证模型总能召回所需信息；本地预算估算也不等于端点的精确 Token 计数。

当前尚无 Web 界面、联网检索与 Deep Research、目标项目的命令执行和代码修改能力。
Git 版本与研究路径的联动、成果节点管理、跨进程任务恢复也尚未实现。

## 快速开始

准备 Python 3.12 和 uv。项目声明 Python >=3.12，当前开发基线由 `.python-version` 固定为 3.12；
已验证的运行环境为 Windows，其他 Python 版本和操作系统尚未验证。
获取仓库后，在仓库根目录执行以下 PowerShell 命令；依赖版本由 `uv.lock` 锁定。

```powershell
uv sync --locked
```

首次使用时创建 `.env`。如果已经有个人配置，保留并编辑原文件；以下命令不会覆盖已有 `.env`：

```powershell
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
```

在 `.env` 中填写三个必要的模型连接配置：

```dotenv
BASE_URL=https://your-api.example/v1
API_KEY=your-api-key
MODEL_NAME=your-model-name
```

| 配置 | 含义 |
| --- | --- |
| `BASE_URL` | 模型服务的 API 基础地址；需要时包含 `/v1` 等前缀 |
| `API_KEY` | 对应服务的访问凭据 |
| `MODEL_NAME` | 该服务实际提供的模型名称 |

端点必须支持 `POST /responses`，并兼容应用使用的 Responses 参数及工具调用。
`BASE_URL` 不要填写完整的 `/responses` 请求地址；仅支持 Chat Completions 的端点不能直接使用。
启用项目读取时，还需要模型支持函数工具调用。

程序默认读取启动目录中的 `.env`，已存在的进程环境变量优先于文件配置。
完整的可配置项及默认示例见 [`.env.example`](.env.example)，包括超时、模型调用上限、读取限额和上下文配置。
LangSmith 追踪默认关闭；按样例启用后，会上传经过过滤的对话和运行信息。
`.env` 不进入版本管理，不应将真实密钥填入 `.env.example`。

完成配置后，可先发送一个不需要项目文件的问题：

```powershell
uv run --locked lineage-agent "请简要介绍你可以如何协助阅读项目材料。"
```

该命令会调用配置的在线模型服务。

## 使用方式

### 绑定项目材料

将 `PROJECT_ROOT` 设置为准备交给 Agent 阅读的目录，可以与 Lineage Agent 仓库分开：

```dotenv
PROJECT_ROOT=../your-project
```

相对路径以 `.env` 所在目录为基准，也可以填写绝对路径。
留空时不注册项目文件工具，仍可进行普通对话以及使用当前运行或会话的内置原文召回工具。
修改配置后重新启动命令，以建立新的文件访问范围。

项目工具只接受根目录内的相对路径，拒绝越界路径、符号链接和 Windows 重解析点访问；
根目录及其祖先也不能通过这些链接间接绑定。应用保留的 `.lineage` 等存储位置不向项目工具开放。
这些检查构成应用层访问边界，不是对抗并发路径替换的操作系统沙箱。

目录列举包含隐藏项，`.gitignore` 不作为文件访问过滤器，读取仍受运行进程的文件系统权限约束。
根目录内可访问的源码、配置和其他文本可能被发送给模型，因此应选择已获准发送的材料范围。
例如，将包含密钥的 `.env` 放进读取根目录，并不会因其被 Git 忽略而自动阻止读取。

### 单次请求

传入一个问题时，程序完成本次运行后退出：

```powershell
uv run --locked lineage-agent "请先查看项目根目录，再阅读 README，说明项目用途及依据。"
```

默认日志目录为 `logs/`，可通过 `LOG_DIR` 调整。
终端会分别显示模型回答和程序记录的运行标识、工具处理情况、日志位置等信息；
模型对执行结果的描述应结合这些记录核对。

### 交互会话

省略问题进入交互模式：

```powershell
uv run --locked lineage-agent
```

在提示符后直接输入问题，后续输入继续当前会话。可用命令如下：

| 命令 | 用途 |
| --- | --- |
| `/new` | 新建并切换到一个会话 |
| `/switch <会话ID>` | 切换到本次交互进程中已经创建的会话 |
| `/exit` | 退出交互模式 |

会话的执行状态和可继续使用的检查点只存在于当前进程，退出后不能通过旧会话 ID 恢复任务。
摘要、History 查询范围、工具结果登记及逐次运行事实保存在
`.lineage/<会话ID>/conversation.sqlite`，原始交互和工具结果正文分别保存在同目录的
`history/` 与 `artifacts/` 中。业务归档可凭明确的已确认资源视图重新建立读取模块，
无需历史检查点；运行日志也继续保留在磁盘上。重新启动仍不会自动载入或继续旧会话。

## 源码阅读入口

可从命令入口沿实际调用顺序阅读，再按关注点查看工具和记录：

- [`entrypoints/cli.py`](src/lineage_agent/entrypoints/cli.py)：命令参数、单次请求及交互会话入口。
- [`config/settings.py`](src/lineage_agent/config/settings.py)：配置读取、默认值和输入校验。
- [`agent/runtime.py`](src/lineage_agent/agent/runtime.py) 与 [`agent/graph.py`](src/lineage_agent/agent/graph.py)：运行初始化及模型、工具循环。
- [`agent/sessions.py`](src/lineage_agent/agent/sessions.py) 与 [`agent/compaction.py`](src/lineage_agent/agent/compaction.py)：会话延续和上下文压缩。
- [`models/`](src/lineage_agent/models/)：Responses 请求、响应解析和上下文预算。
- [`tools/filesystem/`](src/lineage_agent/tools/filesystem/)：目录列举、文本读取与路径访问边界。
- [`observations/`](src/lineage_agent/observations/) 与 [`artifacts/`](src/lineage_agent/artifacts/)：工具结果入模处理、完整结果保存与召回。
- [`recording/`](src/lineage_agent/recording/)：本地运行记录与可选追踪。
- [`archive/`](src/lineage_agent/archive/)：独立业务归档、资源视图及完整性校验；[`agent/records.py`](src/lineage_agent/agent/records.py) 提供受视图约束的 History 读取。

## 验证与资料入口

在仓库根目录运行离线工程检查：

```powershell
uv run --locked python -m benchmarks run agent-contracts
uv run --locked python -m unittest discover -s benchmarks/tests -v
```

`agent-contracts` 使用受控材料检查工程行为，不调用真实模型或外部追踪服务。
工程检查和单元测试用于验证实现约定；它们不代表模型回答质量、项目理解效果或产品需求已经验收通过。
真实模型任务及其评审需要另外配置，执行方式见 [Benchmark 使用与扩展](benchmarks/README.md)。

以下资料均随仓库维护：

- [配置样例](.env.example) 与 [项目依赖声明](pyproject.toml)：运行配置和环境要求。
- [Agent 源码](src/lineage_agent/)：当前行为与能力边界的实现依据。
- [Benchmark 使用与扩展](benchmarks/README.md)：工程检查、项目任务与结果阅读方式。
- [Benchmark 覆盖映射](benchmarks/COVERAGE.md)：工程检查覆盖的行为及其对应实现。
