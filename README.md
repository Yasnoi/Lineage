# Lineage Agent

Lineage Agent 面向需要长期理解和持续迭代的科研或软件项目。
接手一个项目时，需求、研究记录、代码和验证结果往往分散在不同材料中；
项目希望帮助使用者理清已有成果对应哪些目标、设计依据是什么，以及下一步如何推进。

长期目标是把项目理解、问题研究和代码迭代连接起来：
先建立需求与成果之间的对应关系，再围绕具体问题研究方案，经用户确认后继续开发。
另一条主线是保存探索路径和任务状态，让代码、依据与上下文能够一同回访、分叉和恢复。

当前仓库提供这套流程的命令行基础，使用 Python、LangGraph 和 OpenAI 兼容的 Responses API，
已实现模型与工具循环、受限文件读取、多轮会话及跨进程打开、只读任务中断恢复、上下文摘要及运行记录。
这些基础用于继续开发和检验 Agent 行为，完整的长期项目工作流仍在建设中。

## 当前能力边界

| 能力 | 当前范围 |
| --- | --- |
| 模型交互 | 单次请求和多轮对话；模型可请求工具，程序执行后继续调用模型 |
| 项目材料 | 在配置的根目录内列举目录、读取限额内的 UTF-8 普通文本文件 |
| 会话 | 新建、切换会话；列出并打开已发布的对话快照，在新进程中提交新轮次 |
| 任务恢复 | 明确选择后，从已确认执行边界继续同一只读 Run，复用已保存结果并保留额度 |
| 上下文 | 按本地估算检查请求预算，必要时摘要压缩，并提供原文检索与回读工具 |
| 记录 | 保存运行日志、原始交互和较大的工具结果；可选启用 LangSmith 追踪 |
| 验证 | 离线工程检查，以及需单独配置的项目任务 Benchmark |

持久化与恢复由三项能力组成，各自保留明确的访问和执行边界：

| 能力 | 保存和恢复的内容 | 边界 |
| --- | --- | --- |
| 独立业务归档 | 摘要、History、Artifact、运行详情及认可的资源范围 | 业务读取不依赖旧的内存 Checkpoint；文件存在不等于允许访问 |
| 跨进程对话打开 | 最后已发布轮末快照中的有效上下文、配置和资源关系 | `/open` 建立会话，提交新输入后创建新 Run；不会续跑中断轮次 |
| 同一只读 Run 执行恢复 | 已确认执行位置、固定请求、可信结果和剩余额度 | `/resume` 经明确确认后继续原 Run；未知操作仅按安全策略有界重试 |

读取源码和文档使模型具备分析这些材料的输入条件，不等于已经建立完整的需求映射或验证了项目结论。
目录列举不递归；文本读取受单文件和单次运行总量限制，超限时拒绝读取，不截取文件冒充完整正文。
目前不提供 PDF 解析。

摘要是有损的上下文整理，可能遗漏细节，也可能保留错误判断。
原文检索与回读用于补充依据，不能保证模型总能召回所需信息；本地预算估算也不等于端点的精确 Token 计数。

当前尚无 Web 界面、联网检索与 Deep Research、目标项目的命令执行和代码修改能力。
Git 版本与研究路径的联动、成果节点管理及通用写工具恢复尚未实现。启动或打开会话不会自动续跑中断轮次。

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

新建对话并调用模型前，在 `.env` 中填写三个必要的模型连接配置：

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
可恢复会话的地址不能包含 URL 用户信息、查询参数或片段；凭据只通过 `API_KEY` 提供。
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
终端分别输出模型回答和程序统计：回答使用 stdout，工具处理、模型调用及本地上下文预算使用 stderr。
正常完成时省略 Run ID、活动上下文版本和日志路径；运行失败、预算终止或保存未确认时显示诊断日志位置、
记录完整性及文件是否可读取。日志仍按 Run 保存，记录完整不代表对话快照已发布。

### 交互会话

省略问题进入交互模式：

```powershell
uv run --locked lineage-agent
```

进入时可以尚未选择会话。直接输入问题会新建会话，后续输入继续当前会话；也可以先列出或打开旧会话，无需配置当前 `MODEL_NAME` 和 `BASE_URL`。可用命令如下：

| 命令 | 用途 |
| --- | --- |
| `/new` | 新建并切换到一个会话 |
| `/list` | 列出当前存储根目录中登记的会话及可恢复状态 |
| `/open <会话ID>` | 校验并打开已保存的会话 |
| `/inspect-run <会话ID> <RunID>` | 检查未完成轮次的执行位置、结果及额度，不执行任务 |
| `/resume <会话ID> <RunID>` | 展示恢复说明，明确确认后继续原 Run |
| `/switch <会话ID>` | 切换到本次进程中已新建或打开的会话 |
| `/close [会话ID]` | 关闭指定会话，省略 ID 时关闭当前会话；保留持久记录 |
| `/exit` | 退出交互模式 |

输入区由 `prompt_toolkit` 管理，Enter 提交整条消息，自动折行不会插入换行。
主动换行时，先按 Esc，再按 Enter；两次按键之间可以停顿。
退出输入时恢复终端模式，不修改 Windows Terminal 或输入法配置。
多行粘贴留在编辑区等待提交，换行和缩进保留；只有完整单行输入会被识别为管理命令，
多行中的 `/exit`、`/open` 等属于正文。Ctrl+C 取消并退出；Windows 空输入时 Ctrl+Z 后 Enter 结束输入。

编辑器仅在输入、输出和诊断流都连接终端时启用，不保存磁盘输入历史。
管道或输出重定向保持逐行读取，单次请求不启动编辑器。真实终端中的中文输入法、首次跨行和窗口重绘
需要在使用环境中单独检查，协议及缓冲区自动测试不能代替屏幕验收。

也可直接列出或打开：

```powershell
uv run --locked lineage-agent --list
uv run --locked lineage-agent --open <会话ID>
uv run --locked lineage-agent --open <会话ID> "基于之前的资料继续分析。"
```

`--application-root <目录>` 指定会话存储根目录；默认使用启动目录。该选项不改变 `.env` 的查找位置，也不改变相对 `PROJECT_ROOT`、`LOG_DIR` 的解释方式。
对话快照、非敏感配置、摘要、History 范围、Artifact 登记及运行详情保存在
`.lineage/<会话ID>/conversation.sqlite`，正文位于同目录的 `history/` 和 `artifacts/`。
轮末快照成功发布后，该轮才成为可跨进程打开的恢复点；内存检查点或日志保存成功不代表快照发布成功。即使发布后尚未显示结果便退出，新进程仍能识别已发布快照。

打开沿用保存的模型地址、名称、推理设置和数值额度；额度与当前配置不同时会提示，并继续使用保存值。`API_KEY`、日志目录及目录访问权限取自当前环境。
包含内置文件工具的会话要求当前 `PROJECT_ROOT` 覆盖保存的工具根目录，重建后的工具仍使用保存的根目录，不把旧相对路径指向新目录。必要归档缺失、损坏或格式不兼容时明确拒绝打开。

若存在未封存轮次，`/open` 会显示其 ID 和最后已发布的快照，只有输入 `yes` 才从该快照开始新的交互。该选择将未完成轮次标为放弃，保留其记录与产物，此后不能再续跑它。没有已发布快照时不能通过 `/open` 恢复。同一会话同一时间只允许一个打开实例。

### 继续中断的只读任务

恢复原任务使用独立入口，保留原 Conversation ID、Run ID、有效上下文和剩余额度：

```powershell
uv run --locked lineage-agent --inspect-run <会话ID> <RunID>
uv run --locked lineage-agent --resume <会话ID> <RunID>
```

程序先展示已确认阶段、可复用结果、可能重试或暂停的操作，以及已知消耗和未知预留；交互终端中明确输入 `yes` 后才执行。非交互输入只显示检查结果；自动化调用可使用 `SessionManager.inspect_run()`、`open_run()` 和 `resume_run()`，传递检查返回的完整选择。检查和打开执行状态不会请求模型或调用工具，过期或重复选择不会重复派发。

已完整保存的模型响应、工具结果和摘要候选优先复用。结果未知的同一逻辑操作最多补试一次，跨进程累计，真实重试占用原额度；未知 usage 保持未知。文件读取在发送前预留最大可能字节，未知尝试保留预留，重试可读取当前文件的新内容。自定义工具需要应用提供兼容的恢复绑定与策略，否则需要执行时暂停。恢复仍检查当前目录授权，不恢复项目版本或历史权限，也不保证外部请求恰好执行一次。

首次轮次尚无轮末快照，但已有可信执行点时也可恢复。已封存、明确放弃、必要资源损坏或无首个执行点的 Run 不可续跑。保存失败会阻断原实例，需关闭后通过显式恢复入口重新检查。History 续写到同一 Run 的新分段文件，旧尾部保留且不自动授权；运行日志按执行次数分段，旧日志不覆盖。

当前数据库格式为 `3`，恢复契约版本为 `2`；旧格式明确拒绝，本次没有迁移入口。轮末快照与执行恢复点分开确认，InMemorySaver 仍只服务当前进程调度。

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

检查 Windows Terminal 输入时，在新启动的交互窗口中保持宽度，先连续输入中文跨过右边界，
不要先用英文跨行或调整窗口。随后检查中英混排、边界删除与光标移动，再缩窄和放宽窗口；文本应完整且不覆盖行首。
分别尝试提示中可用的换行键，以及粘贴含缩进、`/exit` 和 `/open` 行的多行文本：
提交前内容应留在编辑区，按 Enter 后只产生一个轮次。最后检查 Ctrl+C、Ctrl+Z 后 Enter 和 `/exit` 后的终端状态，
以及正常完成后退出、重新打开该会话并继续。提交问题会调用已配置模型；可先只编辑并用 Ctrl+C 取消来检查渲染。

以下资料均随仓库维护：

- [配置样例](.env.example) 与 [项目依赖声明](pyproject.toml)：运行配置和环境要求。
- [Agent 源码](src/lineage_agent/)：当前行为与能力边界的实现依据。
- [Benchmark 使用与扩展](benchmarks/README.md)：工程检查、项目任务与结果阅读方式。
- [Benchmark 覆盖映射](benchmarks/COVERAGE.md)：工程检查覆盖的行为及其对应实现。
