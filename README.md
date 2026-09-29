# Lineage Agent

基于 LangGraph 和 Responses API 的最小 ReAct 循环。Agent 可以直接回答，也可以请求工具，在工具结果回传后继续调用模型，直到给出最终回答、达到模型调用上限或无法在上下文预算内继续。会话历史达到阈值时，可以为较早的完整轮次生成有界摘要；候选通过校验并确认保存后，才使用新活动版本继续。

每次 `run_agent` 处理一次独立用户请求，使用异步、非流式模型调用。`SessionManager` 使用 LangGraph 原生内存检查点保留同一会话的上下文，让后续问题使用之前的消息和工具资料；保存寿命仅限当前进程。CLI 默认不提供工具；配置 `PROJECT_ROOT` 后，可以列举该项目目录的直接子项，并读取限额内的 UTF-8 普通文本文件（包括源码和配置）。目录结构或单个文件正文都不能证明已经理解整个项目。

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
TIMEOUT_SECONDS=300
MAX_MODEL_CALLS=30
CONTEXT_BUDGET_TOKENS=250000
MAX_OUTPUT_TOKENS=16384
CONTEXT_SAFETY_MARGIN_TOKENS=8192
OPAQUE_FALLBACK_TOKENS=4096
CONTEXT_KEEP_RECENT_TURNS=2
SUMMARY_TARGET_PERCENT=40
LOG_DIR=logs
PROJECT_ROOT=
DIRECTORY_MAX_ENTRIES=1000
TEXT_MAX_BYTES=131072
TEXT_TOTAL_MAX_BYTES=1048576
```

`BASE_URL` 是支持 Responses 协议的 API 基础地址，按供应商要求包含版本前缀，不追加 `/responses`。仅支持 Chat Completions 的地址不满足本项目要求。`BASE_URL`、`API_KEY`、`MODEL_NAME` 必填；请求超时默认为 300 秒且必须为有限正数，业务模型调用上限默认为 30 次且必须为正整数。超时沿用 SDK 的请求超时语义，不是 Agent 整轮时限；本项目没有整轮时限配置。连接地址、模型名称及超时由配置对象共同校验，直接构造与 dotenv 加载遵循相同规则；超时不接受布尔值。

`PROJECT_ROOT` 可选，未配置或留空时保持无工具问答。`DIRECTORY_MAX_ENTRIES` 默认为每页 1000 条，`TEXT_MAX_BYTES` 默认为 131,072 字节（128 KiB），`TEXT_TOTAL_MAX_BYTES` 为每次运行成功读取的累计额度，默认 1,048,576 字节（1 MiB），均必须为正整数；程序构造配置或执行器时也不接受布尔值。根目录和限额由调用方控制，模型不能修改。

累计读取许可不等于单次请求能够容纳全部资料。即使单文件不超过 128 KiB，包装、已有历史和当前轮其他工具结果也可能使后续请求超限。当前轮工具事实完整保留，按现有预算检查和 Compact 恢复或停止；不截断文件、丢弃结果或分块压缩，不能据此保证任意大文件任务完成。

配置通过 `python-dotenv` 显式读取当前工作目录的 `.env`，因此运行命令应位于项目根目录；优先顺序为进程环境、`.env`、生产默认值。直接构造 `ModelSettings` 或调用模型与工具便捷接口时，以显式参数为准，不自动读取环境。各生产入口共用预算默认值；升级不会覆盖已有 `.env` 或进程环境中的显式值。真实 `.env` 已加入忽略规则，连接配置本身不进入 Agent State 或应用日志；但根目录内的文件正文另受下述文件访问授权规则约束。

```powershell
uv run lineage-agent "请用一句话介绍你自己"
```

启用目录和读取工具时，在 `.env` 中将 `PROJECT_ROOT` 设置为要分析的目录，例如 `C:/path/to/project`，再运行：

```powershell
uv run lineage-agent "请先分页列举项目根目录，再选择其中的源码或配置文件读取；根据实际正文回答，并标明来源路径"
```

被分析项目可以与 Lineage Agent 自身目录不同。相对 `PROJECT_ROOT` 以 dotenv 文件所在目录为基准；无效根目录在调用模型前报告配置错误。恢复无工具 CLI 时，将配置中的 `PROJECT_ROOT` 留空，并确保进程环境没有覆盖该值。

也可以使用模块入口：

```powershell
uv run python -m lineage_agent "请用一句话介绍你自己"
```

CLI 只有一个可选的用户输入参数，将最终模型回答写入 stdout，将运行标识、日志位置、记录完整性、实际工具处理统计、业务与摘要调用次数、压缩阶段、活动版本和最近预算判断写入 stderr。提供输入参数时，退出码 `0` 表示得到最终回答，`1` 表示模型、协议、计量、日志或客户端关闭错误，`2` 表示参数或配置错误，`3` 表示模型调用次数耗尽，`4` 表示上下文预算达到触发阈值、请求尚未完成；Ctrl-C 退出 `130`。达到上限时不会把最近一轮的中间文本当作最终回答。若已得到最终回答但客户端关闭失败，仍展示该回答，同时报告关闭失败并退出 `1`；若收尾前未完成，则不展示中间文本。日志或关闭故障覆盖预算终态时，仍展示最近预算判断，同时按实际错误退出。

SDK 自动重试已关闭，不使用服务端会话或 `previous_response_id`。每次请求都使用本地当前上下文。

## 连续对话与状态保存

省略问题参数进入交互模式。配置和文件访问权限仍来自启动时的 `.env`，没有额外的验证或权限参数：

```powershell
uv run lineage-agent
```

每行输入一个问题。`/new` 新建并切换到独立会话，`/switch <会话ID>` 切换到本次交互已创建的会话，`/exit` 或 EOF 正常退出 `0`；Ctrl-C 关闭运行资源后退出 `130`。每轮展示会话标识、独立运行标识、工具处理统计、预算判断、日志及最终检查点确认。普通模型或工具错误、调用额度耗尽及预算拒绝后仍可提交新问题；检查点故障会阻断该会话，需新建或切换会话。交互模式正常退出不表示中途每一轮都成功，应查看每轮结果。

明确发生在执行前的同步配置拒绝（例如客户端工厂因缺少 `API_KEY` 而拒绝创建）只拒绝当前请求，保留原检查点和会话可用性。程序调用仍抛出 `ConfigurationError`，修正后可在同一会话继续；交互模式报告本轮错误并继续接收输入。启动配置错误和单次请求的配置错误仍退出 `2`。客户端进入阶段或执行后的未处理异常、取消及保存不确定继续阻断会话，不能仅凭模型调用次数为零判断可以安全重试。

程序调用可直接选择工具权限；以下在同一进程中创建两轮无工具问答：

```python
import asyncio

from lineage_agent.agent.sessions import SessionManager
from lineage_agent.config.settings import load_settings


async def main():
    sessions = SessionManager()
    session_id = sessions.create_session(load_settings())
    first = await sessions.run_turn(session_id, "本次记录中的纸船是蓝色。请记住这个事实。")
    if not first.checkpoint_complete:
        raise RuntimeError(first.checkpoint_error)
    second = await sessions.run_turn(session_id, "刚才纸船是什么颜色？")
    print(second.state["result"]["text"] if second.state["result"] else second.state["error"])
    print(second.session_id, second.checkpoint_id, second.checkpoint_complete)


asyncio.run(main())
```

需要目录与读取能力时，把 `tools=[DIRECTORY_TOOL, TEXT_TOOL]` 和已绑定可信根目录的 `execute_tool=create_filesystem_executor(root)` 传入 `create_session`；这些对象与单次调用使用同一接口。会话创建后，后续输入不能更换配置、项目根目录或工具定义。未注入工具时，即使配置含 `PROJECT_ROOT`，程序接口也不自动启用工具。

每个会话 UUID 映射为 LangGraph `thread_id`，与每轮新建的 `run_id` 分开。`SessionTurn` 提供 `session_id`、`state`、`checkpoint_id`、`checkpoint_complete` 和固定安全错误 `checkpoint_error`；只有最终保存确认成功才返回检查点 ID。同一会话的重叠提交立即抛出带固定错误码的 `SessionError`，不排队或交错合并历史；未知或已阻断的会话同样明确拒绝。不同会话可以交错运行。

检查点由管理器拥有的 `InMemorySaver` 保存，不在项目中创建状态文件，也不另建一份完整历史数据库。每轮从检查点加载可继承的活动协议上下文，通过原生 `Overwrite` 写入“有效历史＋新输入”，同一活动版本的后续节点只追加本次新增项；成功压缩另建立并保存新版本。调用方输入和返回 State 与已保存内容相互隔离。

`history_end` 指向上下文中可供下一轮使用的前缀边界，不是预算截断位置。首次预算放行并实际发起模型调用的合法输入、通过协议检查的最终回答，以及工具请求和结果完整配对的批次可以保留；工具拒绝或执行失败的关联结果也是有效资料。无效模型输出、最后一轮未执行的工具请求或未完成批次不进入下一轮，也不会补造结果。原轮返回 State 保留已发生事实，检查点与日志的保留程度分别取决于实际保存和写入结果。一个批次只有全部关联结果完成后才推进边界，不提供逐工具恢复。

新增结构化输入先经过本地消息及已知工具项的必要形状检查。确定无效的输入以 `invalid_input` 结束本轮，不调用模型；失败 State 和安全条目位置可以保存，但不会推进可继承历史边界。下一轮正常问题仍使用此前的有效历史。未知扩展格式继续由端点验证；不会仅因 HTTP 400、连接或服务错误便回退历史，也不会根据异常文本猜测输入错误原因。

每个新问题重新初始化结果、终态、错误、模型与工具计数、`usage_totals`、运行日志和读取额度。`result.usage` 仍是最近一次业务模型调用；`usage_totals` 是本轮业务与摘要调用的合计，`summary_usage_totals` 单列摘要小计，两者都不是会话累计。前一轮失败或耗尽次数不会耗用下一轮额度。文件正文历史表示当时的读取结果，不自动反映文件当前版本；每轮读取额度也不限制整个会话不断增长的上下文。

### 保存故障与收尾

图使用同步检查点提交，内存后端的 pending writes 故障在下一次提交前阻断后续步骤。创建、读取或写入失败均明确报告并封锁该会话，不自动重试、回退或继续原请求；已有调用、工具结果和可取得的 usage 仍保留在返回 State，无法确认的事实不补造。

检查点故障缓存只保留首次故障的固定错误码，每次传播新建安全异常，不缓存已经抛出的异常及其调用栈。收尾保护覆盖客户端关闭、记录器关闭、检查点与记录器解绑及 Tracing 关闭；各项清理独立尝试，已有主错误不被后续清理错误覆盖。

收尾先关闭模型客户端、完成并关闭 JSONL，再用原生 `aupdate_state` 保存最终 State；该操作不执行图节点。此前没有检查点错误时，会话的 `run.ended` 明确标为等待最终保存，记录此前已确认的检查点 ID；已有检查点错误时标为失败，不再尝试保存。最终保存状态由返回的 `SessionTurn` 和 CLI 展示，不重开日志补写确认。因此 `log_complete=true` 只证明本地日志收尾成功，不能证明最后的检查点提交成功。保存故障与执行结果分别显示；已经产生的回答和统计不会被伪装成没有发生。

检查点不序列化运行依赖中的连接配置、API 密钥、客户端、文件句柄、记录器或执行器。消息和工具正文属于 State，会原样保存在进程内；若用户或获准读取的文件本身包含敏感内容，这些材料也会进入历史。Tracing 只接收经过过滤的观察数据，不自动上传完整 State，也不是检查点成功的凭据。退出进程或释放管理器后历史丢失；本轮没有跨进程加载、任意中断恢复、历史分叉或重放接口。新问题不是恢复原请求，不能以重置计数重新运行来宣称恢复成功。

## 模型请求的上下文预算

每次业务模型调用前检查实际准备的请求，包括首次调用、工具回传后的调用、工具定义，以及适用的固定收尾说明和 `tool_choice`。普通业务请求省略 `instructions` 和 `tool_choice`，不播报剩余调用次数或读取额度。检查通过时沿用同一请求副本发送，不改变历史的内容、顺序或工具关联；摘要选源和候选检查也复用相同业务请求选项。实际请求设置 `max_output_tokens=MAX_OUTPUT_TOKENS` 和 `truncation="disabled"`，不请求端点自动截断或压缩；输出额度同时用于预留可见回答和推理输出空间。指令与历史稳定不保证供应商 Prompt Cache 命中，最后回答轮仍会切换为明确的收尾策略。

本地方法名为 `utf8_json_opaque_v1`，采用透明内容估算加不透明（opaque）内容风险预留，不调用 Provider 计数服务，也不依赖 tokenizer：

- 消息文本、函数参数和结果、可识别的 reasoning 文本、应用指令及工具定义，按紧凑 JSON 的 UTF-8 字节数计量，包含结构开销；1 字节计为 1 个估算预算单位。
- `encrypted_content`、未知类型协议项及非文本内容块分别作为 opaque 单元，不按其载荷字节换算 Token。同一不透明子树不重复计量，其可识别的兄弟内容继续估算；工具结果字符串不被重新解析以猜测 opaque 内容。
- 模型产生的每个 opaque 单元使用产生它的那次响应中有效、正整数的 `output_tokens` 作为风险预留；多个单元各自使用该数值，不平摊。缺失、无效或零值使用 `OPAQUE_FALLBACK_TOKENS`。调用方输入没有可信来源时同样使用回退值，不信任输入中自带的 usage。

预算计算使用应用估算单位，名称中的 `TOKENS` 不表示精确测得的输入 Token 数；`MAX_OUTPUT_TOKENS` 同时按 Responses 协议作为实际输出 Token 上限发送：

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `CONTEXT_BUDGET_TOKENS` | 250000 | 十进制 250K 应用工作窗口，不能从模型别名推断为端点实际窗口 |
| `MAX_OUTPUT_TOKENS` | 16384 | 输出预留及实际请求的输出上限，包含推理与可见回答 |
| `CONTEXT_SAFETY_MARGIN_TOKENS` | 8192 | 协议转换及估算误差的安全余量 |
| `OPAQUE_FALLBACK_TOKENS` | 4096 | 缺少可信来源 usage 时，每个 opaque 单元的预留 |

这些值可通过环境变量或 `ModelSettings` 显式设置；额度为正整数，安全余量允许零，均拒绝布尔值。工作窗口必须大于业务输出预留与安全余量之和。

默认组合的可用输入预算为 225424，输入中的 opaque 风险仍按既有规则计入。预算选值面向多步只读研究：30 次业务调用保留发现、读取和核对空间，250K 工作窗口与 16K 输出主动限制请求规模；读取额度和请求等待分别控制，不按同一倍数扩张。服务商最大值不直接成为应用默认值，真实任务完成率和消耗仍需另行评测。

```text
所需预算 = 透明内容估算 + opaque 风险预留 + 输出预留 + 安全余量
可用输入预算 I = 工作窗口 W − 当前请求输出预留 O − 安全余量 M
所需预算 <= W：放行；所需预算 > W：尝试符合条件的压缩，无法完成则停止
```

这是一套保守的应用策略，不是精确窗口保证。UTF-8 字节和 Token 不等价，opaque 输出 usage 也不能证明同一项再次输入时的实际占用。兼容端点的 tokenizer、协议转换和窗口限制可能不同，配置应以供应商实际限制为依据；一次成功请求不能消除估算误差。进入预算检查后，不可序列化、循环引用、非有限数值或其他计量失败以 `context_measurement_error` 停止，不按零放行。会话模式的原生初始检查点先于模型节点写入；若原提交本身无法被检查点保存，可能先返回 `checkpoint_write_error`，此时预算尚未执行、模型未调用，保存失败与会话阻断按既有规则处理，不宣称该提交已保存。上一轮 usage 和本轮累计 usage 都不替代下一次请求计量。

超出预算且不能完成符合下述规则的压缩时，以 `context_budget_exceeded` 结束本轮，不发送对应业务模型请求，不增加该业务调用的尝试或缺失 usage 次数；之前的模型及工具调用、结果和消耗保留。首次调用前被拒绝的新输入仍留在本轮 State 和确认保存的检查点中，但不进入可继承历史，随后较小且预算允许的问题仍可使用此前有效历史。工具执行后超限则保留并继承完整配对的请求和实际结果，明确结果尚未进入后续模型调用；下一轮可能再次超限，不静默删除资料、不重复执行工具，也不声称可以自动恢复。预算拒绝本身不永久阻断会话，日志和检查点故障仍单独报告。

`context_budget` 保存最近检查的组成、决定、步骤及请求是否发起；每轮初始化，每次检查替换。旧诊断字段 `trigger_percent/trigger_tokens` 保留为 `100/W` 的兼容投影，不参与独立比例触发。`context_risks` 与 `input` 对齐，只保存 opaque 数量、预留依据和来源运行/模型调用，不重复保存正文；透明项为空。该字段整体替换，历史继承时同步裁取，避免检查点恢复重复追加或用后续响应的 usage 改写旧风险依据。请求副本、State 和已保存历史相互隔离。

### 有界摘要与活动版本

自动压缩只使用 `SessionManager` 已确认结束的旧轮次。单次 `run_agent` 不从任意结构化输入猜测轮次边界；没有旧轮可压缩时仍明确停止。正常预算内不调用摘要器、不重新排序或筛选历史。同一版本持续追加，只有候选通过校验并确认保存才切换版本。

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `CONTEXT_KEEP_RECENT_TURNS` | 2 | 原样保留最近的完整有效用户轮次，不包含当前交互 |
| `SUMMARY_TARGET_PERCENT` | 40 | 扣除全部保留内容后，摘要可用空间的软目标百分比，允许 `1–100` |

上述配置不接受布尔值。旧的 `CONTEXT_TRIGGER_PERCENT`、`CONTEXT_TARGET_PERCENT`、`SUMMARY_MAX_OUTPUT_TOKENS`、`SUMMARY_MAX_BYTES` 已移除；环境或 `ModelSettings` 构造仍提供这些字段时明确报错，应删除并按新语义配置 `SUMMARY_TARGET_PERCENT`。

对实际请求分别测量空历史开销 `F` 和全部保留历史的增量 `P`，计算摘要可用空间 `B = I − F − P`，软目标 `T = floor(B × SUMMARY_TARGET_PERCENT / 100)`。消息包装、来源标识、JSON 转义及插入分隔符均计入摘要占用；输出预留和安全余量只扣一次。摘要自己的请求也要独立计量，计入源资料、指令、输出预留和安全余量，超限时停止，不自动分块。

从最早尚未覆盖的旧轮选择连续前缀，选择使 `T` 能容纳至少一条带合法来源的非空摘要结构的最短范围。已有活动摘要随此前缀一起压缩。当前问题、当前已产生的工具交互及近期完整轮次始终保留。未知协议项、非文本资料或含 system/developer 输入的轮次是不可越过的屏障；不跳过这些轮次任意筛选后文。完整旧轮中的加密 reasoning 可以退出活动上下文，但不当作可读文本送入摘要器，并标记未总结的资料。保留区的工具关联及 opaque 风险来源维持原样。

摘要采用同一端点、模型和客户端的独立 Responses 调用，设置 `reasoning.effort="none"`、`tool_choice="none"`，不提供工具，不改变业务请求的推理设置。动态提示要求完整摘要插入开销尽量不超过 `T`，实际 API `max_output_tokens=T`；这一数值映射不是字节与供应商 token 的精确换算。端点不支持参数时明确失败，不回退为推理模式。第一次输入为上一份摘要（若有）及本次新覆盖的原文，包含原角色、固定来源引用、工具结果和安全终态；当前新问题、无效尾部与未配对批次不作为摘要来源。

`MAX_COMPACT_PASSES=2` 限制同一个待发送业务请求最多两次摘要调用，包含第一次。完整合法候选即使超过软目标，只要完整业务请求预算通过且确实缩减上下文即可接受。首次候选仍超预算时，第二次仅压缩已保存但未启用的候选及其必要来源信息，重新计算动态目标；不重新附上原文，不扩展可引用来源。第二次仍失败则停止，没有第三次调用。拒答、不完整响应、结构或来源非法、模型请求失败、计量或保存故障均不重试。后续不同业务请求可以分别触发恢复，累计 `summary_calls` 与本次 `pass_no` 分别记录。

摘要输出为固定 JSON，分列有效要求、事实与进展、建议/推断、纠正与否决及未解决事项，每条带来源标识。应用检查形状、重复键、来源范围、完整响应、拒答和工具请求，再计量完整候选业务请求；不直接截断文本或 JSON 字段。校验只能确认结构与边界，不能证明语义无损；摘要可能遗漏信息或误解用户纠正。候选作为明确的派生 assistant 资料进入上下文，不提升为 system/developer 指令，不改变工具权限。模型返回的 reasoning 或其他非摘要内容不作为新活动摘要。

摘要调用发生、候选生成、候选校验、版本保存和业务请求实际使用是五个不同事实。摘要节点先保存候选及已发生的消耗；需要第二次时保存 `retry_pending`，确认后才调用摘要，两个候选均可沿检查点查询。最终候选通过验证并保存活动版本后，后续业务节点再次检查实际请求，才能发起调用。激活保存失败时返回状态仍指向旧活动上下文，同时保留候选及已发生消耗，不继续业务调用。摘要调用本身不批准当前用户输入进入下轮历史；`history_end` 仍表示协议准入边界，切换时只调整它在新活动列表中的位置。已经生成的工具结果不会通过重跑工具重建。

无可压缩历史、保留内容过大、摘要请求超限、无压缩效果或两次恢复耗尽，以预算不足停止，CLI 退出 `4`。摘要服务失败、无效响应或计量失败以错误停止，退出 `1`。日志与保存故障按原有规则独立报告，保存不确定则阻断会话。实际摘要调用与可取得的 usage 始终计入本轮统计，即使候选最终未启用。

### 原始记录与来源查询

原始记录复用原生检查点，轻量轮次目录以 `run_id` 指向该轮最后确认的检查点，不复制一份完整历史数据库。原文引用绑定固定检查点和其中的位置，不使用会随压缩改变的当前列表下标。轮次中发生的提交、协议交互、业务及摘要响应和终态通过检查点链取回；无效或未获准继承的提交也保留为该轮事实，但不会通过摘要重新进入活动历史。

```python
record = await sessions.get_record(session_id, first.state["run_id"])
```

`get_record` 返回独立副本，不执行或恢复图。摘要记录上一份摘要引用及新增覆盖轮次；原文、派生摘要和活动版本各有明确职责。`active_context` 标明当前版本、摘要引用、保留轮次及本轮起点，`compaction` 描述最近一次恢复的阶段与确认结果。只有确认保存的记录可声称能通过引用取回，保存失败后的未确认事实仍由返回 State 交付。JSONL 与过滤后的 Tracing 不替代原始事实或完整 SDK 请求档案；内存检查点也不是跨进程持久化。退出进程或释放管理器后，原文引用、摘要和版本随之失效。

压缩选源时，每个候选轮次只读取其固定的最终检查点，提取已准入交互、终态和工具处理统计，不回访整条调用记录链。`get_record` 仍沿检查点链返回完整轮次记录。两条路径共用轮次标识和边界校验，返回的数据均与已保存内容隔离；原生检查点后端仍需反序列化该最终检查点。

## 可选 LangSmith Tracing

Tracing 用于在 LangSmith 网页查看运行、模型请求和逐条工具处理记录，不需要 Studio 或 Agent Server。默认关闭；关闭时不会创建 LangSmith 客户端或访问其服务，也不要求 LangSmith 账号或密钥。

先在 [LangSmith](https://smith.langchain.com) 注册或登录账号，选择对应 Workspace，在设置中创建 API key。把以下配置加入本机 `.env`；LangSmith 密钥与模型 `API_KEY` 是两个独立凭据：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=lineage-agent-dev
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_WORKSPACE_ID=
```

上述 endpoint 对应 GCP US。其他区域使用账号对应的官方 API endpoint，例如 GCP EU 为 `https://eu.api.smith.langchain.com`；区域必须与账号匹配。需要指定 Workspace（尤其密钥可访问多个 Workspace）时填写其 ID。未填写时不继承本机 LangSmith profile 的 Workspace。配置沿用 dotenv 规则：已有进程环境变量优先。

启用后仍使用原命令：

```powershell
uv run lineage-agent "请列举项目根目录，只报告实际看到的条目"
```

在网页的 Tracing Projects 中打开配置的项目，按 metadata 中的 `run_id` 筛选 CLI 显示的运行标识。根记录与本地运行使用相同 UUID；模型记录包含 `step_id` 和 `model_call`，每条工具记录包含 `step_id`、`call_id` 和批内 `request_index`，可与 JSONL 对照。工具拒绝和失败标为错误，未执行请求标为 `skipped`；生成了协议结果不等于工具执行成功。模型记录保留可取得的 token usage，缺失值不会补零；兼容端点不保证提供 usage 或可计算的价格。

关闭时将 `LANGSMITH_TRACING` 设为 `false`，并检查进程环境中是否存在覆盖值。发行给其他用户时保留此默认值，只提交 `.env.example`；不附带真实 `.env`、账号凭据或 `logs/`，也不依赖开发者本机 profile。

### 上传范围

开启即允许把本次对话与模型回复的可读文本、模型名称、usage、运行统计和时间上传到指定项目。目录工具允许上传 `path`、`after` 参数及返回的相对路径、条目名称/类型、条目上限、截断标志和 `after/next_after`。模型记录还含剩余调用数、最后回答轮标志、读取预算快照和拒答标志；运行收尾失败保留安全错误码。读取工具只允许上传 `path` 参数及结果中的 `path`、`encoding`、`bytes_read`、`complete`，以 `content_omitted=true` 标明正文被省略。失败结果保留错误大类及应用固定表中的安全原因，忽略外部传入的错误说明。独立工具记录和模型上下文中的工具结果使用相同过滤规则；未知工具、未知字段、无效 JSON 只保留摘要。超限数据明确标注省略，网页显示不一定包含完整输入。

摘要模型独立记录仅上传来源引用、数量/长度、预算、usage 及状态，不上传请求材料、摘要正文、reasoning 或工具请求参数。后续业务请求中的摘要项按应用内部来源标记省略正文，不依靠文本前缀识别；不会因为工具正文被放入普通摘要消息就重新上传。来源列表最多展示 32 条并标记省略，其他容量限制继续适用。最终业务回答引用摘要或文件事实时仍按现有模型文本规则追踪，因此不能声称资料永不进入外部追踪。

上下文预算只上传固定方法名、安全步骤、数值组成、判断和请求是否发起，不上传用于计量的完整请求或 `context_risks` 来源数据。根记录保留最近一次预算判断，实际模型记录关联该次检查；预算拒绝不创建虚假的模型调用记录，根结果显示 `stopped`。计量失败、日志失败和检查点失败仍可区分。Trace 的截断和展示上限独立于模型预算，不能用于计量模型实际请求。

上传前在独立副本中移除已知凭据、掩码常见凭据字段和文本格式、替换绝对项目根路径。已知凭据跨消息或文本分段时也先匹配再截断；引号内的逗号、分号和转义不会提前结束凭据遮盖。未闭合引号会遮盖剩余逻辑文本，非结构化的未加引号凭据值保守遮盖到行尾。完整 State、原始响应对象、reasoning 内容、供应商扩展、配置环境、异常堆栈和文件附件不上传。读取结果中的完整正文会发送给配置的模型端点；若模型回答引用正文，该引用仍按模型文本规则进入 Tracing，主动粘贴的正文也属于对话范围。因此不能保证文件内容绝不会进入外部追踪服务。过滤不能识别任意未知秘密，开启前应确认对话、文件名和可能引用的内容可以上传。

追踪文本最多保留 4,000 字符，协议项、文本分段、目录条目和工具定义列表分别最多保留 128 项，每条应用上传载荷最多 64 KiB。超限上下文优先保留开头的 system/developer 指令、初始用户请求，以及最近可容纳的完整轮次；同批工具请求和对应结果作为一组保留，单组自身超限时记录整组摘要。`trace_omissions` 独立记录原始数量、保留数量和省略区间，区间使用从 1 开始、包含两端的原始列表索引。需要详细核对工具时，还应查看该请求的独立工具记录。

总容量不足时先减少较旧输入，再压缩输入内容，最后才压缩输出；有效 usage 与运行关联、状态和耗时单独保留。模型文本只在结构化 `output` 中记录一次，根运行保留最终回答摘要。Trace 是经过清洗和容量限制的受控视图，不能当成完整模型请求副本。这些规则独立于 JSONL 摘要，不修改模型实际输入、工具参数、结果或 State。LangGraph 的自动 State 追踪在运行期间关闭，只提交应用显式构造的允许字段。

工具结果的 JSON 超出追踪解析上限时，读取元数据也可能降级为带省略标记的摘要；不会为了展示元数据而取消容量保护，模型实际收到的完整正文不受影响。

### 故障与收尾

每次运行使用有界队列和单个后台 daemon 线程发送已过滤记录。SDK 自动批次与上传重试关闭；连接和读取超时分别为 1 秒和 2 秒。异常、队列满或过滤失败仅输出固定安全提示，不改变 Agent 结果、退出码或调用次数，不触发模型/工具重试；JSONL 原有故障即停止策略保持不变。

运行结束、异常或取消时最多等待 5 秒收尾，超时后不再等待，并提示追踪可能不完整。进程崩溃或强制终止可能丢失未上传记录。CLI 正常完成、HTTP 提交成功或队列排空都不代表远端记录已经完整可见，应在网页核对根记录及各个子记录。配置错误或不支持的 SDK 后台上传开关会安全关闭本次追踪。

官方参考：[LangGraph 追踪](https://docs.langchain.com/langsmith/trace-with-langgraph)、[数据过滤](https://docs.langchain.com/langsmith/mask-inputs-outputs)、[显式记录](https://docs.langchain.com/langsmith/trace-with-api)、[LLM 输入输出与 usage](https://docs.langchain.com/langsmith/log-llm-trace)。

## 模块职责

```text
src/lineage_agent/
├── __main__.py           # 转发到 CLI
├── entrypoints/
│   └── cli.py            # 参数、运行调用和结果展示
├── agent/
│   ├── graph.py          # 条件路由、调用计数与本次请求的 usage 累计
│   ├── state.py          # AgentState
│   ├── runtime.py        # 依赖组装、状态初始化和客户端生命周期
│   ├── sessions.py       # 会话协调、跨轮边界、记录查询和独立运行
│   ├── context.py        # 原始记录引用、活动版本及压缩范围选择
│   ├── compaction.py     # 摘要候选、校验与版本激活
│   └── checkpointing.py  # 内存检查点及保存故障边界
├── models/
│   ├── responses.py      # Responses 请求与响应解析
│   ├── requests.py       # 构造供计量与实际发送共用的请求副本
│   ├── budget.py         # 透明内容估算、opaque 风险与预算判断
│   ├── summarization.py  # 摘要请求构造及固定输出校验
│   └── types.py          # 模型返回值与可调用对象类型
├── tools/
│   ├── execution.py      # 串行执行、参数解析和工具结果/错误转换
│   ├── types.py          # 执行器接口与安全错误类型
│   └── filesystem/
│       ├── paths.py      # 根目录绑定和访问范围检查
│       ├── directory.py  # 目录工具 Schema、字段校验和直接子项列举
│       ├── text.py       # 读取工具 Schema、限额、类型和编码检查
│       └── executor.py   # 同一根目录下两个工具的轻量名称分发
├── recording/
│   ├── recorder.py       # 每次运行的 JSONL 文件、事件和步骤标识
│   ├── summaries.py      # 确定性的参数与结果摘要
│   ├── tracing.py        # 可选追踪、关联和有限后台上传
│   └── trace_data.py     # 上传白名单与数据清洗
└── config/
    └── settings.py       # dotenv 加载与配置校验
```

各目录均为 Python 包；上图省略 `__init__.py`。

执行图为 `START → model → tools → model`，模型给出最终回答、模型失败、达到调用或上下文预算边界、计量失败或日志故障时转入 `END`。预算检查在模型节点内、实际调用前执行，不设置单独的“思考节点”。

`build_graph` 接收请求准备函数、可信预算配置和发送已准备请求的异步模型函数，负责预算决定、状态更新与路由。`run_agent` 绑定这些依赖，创建本次运行独享的记录器并注入图及工具执行层；请求准备、计量、模型解析和 SDK 调用集中在 models 模块。

程序入口为 `await run_agent(input, settings, *, tools=None, execute_tool=None)`，返回最终 State，并在成功或失败后关闭客户端。`input` 可以是字符串，也可以是包含消息、工具请求、工具结果及 reasoning 等内容的完整 Responses 协议项列表；传入列表会被复制，调用方的数据不被原地修改。

消息角色只接受 `user`、`assistant`、`system` 或 `developer`，`content` 必须是字符串或内容部件列表；列表部件必须具有有效 `type`，已知文本与拒答部件的对应字段必须是字符串。函数请求的名称和参数必须是字符串，函数结果必须使用支持的输出容器。参数字符串即使曾被工具拒绝，仍可与其错误结果一并保留。其他带类型的扩展项及扩展字段继续透传；这些检查不等同于完整的 Responses 或端点私有格式校验器。

初始上下文中的 `function_call` 与 `function_call_output` 必须使用非空字符串 `call_id`，请求先于结果且一一对应；缺失、重复或无法配对时返回错误终态，模型调用次数为零。其他协议项原样保留。模型响应不得夹带由应用生成的 `function_call_output`，否则按协议错误终止；字典形式的响应同样深拷贝，避免不同运行共享嵌套数据。输出消息只接受 `type="message"`、`role="assistant"`，其他角色、缺失角色、伪装输入消息或非消息项夹带角色均按协议错误拒绝，不进入后续上下文。合法 `refusal` 内容与 `output_text` 按顺序合为展示文本；纯拒答是有效回答，与请求失败区分。

独立模型接口为 `await generate_response(client, model_name, input, *, tools=None, instructions=None, tool_choice=None, max_output_tokens=16384)`（可另注入追踪器）；该兼容入口只负责单次请求，完整的调用前预算检查通过 `run_agent` 或 `SessionManager.run_turn` 使用。图按普通或收尾策略准备请求，适用时包含 `instructions` 和 `tool_choice`，完成预算检查后调用 `send_prepared_response` 发送同一份参数。工具定义使用 Responses 的 `type: "function"` 格式及 JSON Schema，不使用 Chat Completions 的工具格式。

## 目录列举

`list_directory` 使用 `{"path": ".", "after": null}`；`path` 必填且为字符串，`after` 为上一页最后一个条目的名称，第一页为 `null`。严格 Schema 要求这两个字段，应用兼容旧调用只传 `path`。不接受额外字段；游标拒绝非字符串、空字符串及路径分隔符，只用于排序定位，不参与路径拼接。`.` 表示绑定根目录，也可以传入 `src`、`src/package` 等相对路径。Schema 使用严格模式，工具仍自行执行字段和路径校验。

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
  "truncated": false,
  "after": null,
  "next_after": null
}
```

所有返回路径均相对于项目根目录，使用 `/` 分隔。类型包括 `file`、`directory`、`symlink`、`junction`、`reparse_point` 和 `other`。不读取正文，也不展开链接目标；不返回绝对路径或链接的外部目标。

使用 Python 标准库 `os.scandir` 和有界候选堆，每页扫描整个目录，按 `(name.casefold(), name)` 选出游标之后的前 N 项，最多保留 N+1 个候选用于判断是否还有下一页。`truncated=true` 时以返回的 `next_after` 请求同一路径的下一页。稳定目录连续翻页不遗漏、不重复；`truncated=false` 只表示当前游标之后没有更多条目，不证明此前页面已被读取。

空目录或游标之后没有条目时返回空列表、`truncated=false`、`next_after=null`。目录变化时不提供快照保证：插入到游标之前的条目需从第一页重新发现。分页限制候选内存和返回条目数，不限制扫描总量、耗时或结果总字节数。

不应用隐藏项或 `.gitignore` 过滤，`.git` 等目录项也包含在列举范围内。链接条目可以显示自身名称和类型，但不能作为目标或中间路径进入，即使链接指向根目录内部也会拒绝。其他特殊条目仅报告类型，不展开。

路径接受 `/` 和 `\` 分隔，拒绝任何 `..` 组件、绝对路径、盘符相对路径、UNC/设备路径、ADS、控制字符、Windows 保留字符和设备名，以及组件末尾的点或空格。不展开环境变量或 `~`。根目录自身及祖先不得经过符号链接或 Windows reparse point；每次访问重新检查路径组件，并使用解析后的路径进行范围核对。

字段、路径范围、链接和非目录目标通过 `ToolArgumentError` 计为拒绝；不存在、权限不足及其他 I/O 错误计为执行失败，分别使用 `file_not_found`、`permission_denied` 和 `io_error` 安全原因码。枚举或条目分类中途发生异常时，整次调用失败，不返回部分成功结果；同批后续请求仍按原有规则处理。成功列举不是文件系统快照，枚举期间的新增或删除可能影响结果。

这些是应用层的路径访问检查，检查与实际操作并非原子，不能抵御恶意并发替换目录的所有情况，不构成操作系统安全沙箱。日志继续只记录参数和结果的类型、长度等安全摘要，不新增原始路径或文件名预览。

## 纯文本读取

`read_text_file` 只接受 `{"path": "docs/说明.md"}` 形式的参数。路径必填且为字符串，不接受额外字段；沿用目录工具的根目录、相对路径和禁止链接规则。目标必须是普通文件，不按扩展名限制；源码、配置、隐藏文件以及 Dockerfile、Makefile 等无扩展名文本均可读取。项目根目录就是文件访问授权边界，不额外设置敏感文件黑名单。根内 `.env`、凭据配置等也可能被读取并发送到模型；绑定根目录即授权其中符合访问和文本规则的文件。

通过程序同时开放两个工具：

```python
from lineage_agent.tools.filesystem.directory import DIRECTORY_TOOL
from lineage_agent.tools.filesystem.text import TEXT_TOOL, create_text_executor
from lineage_agent.tools.filesystem.executor import create_filesystem_executor

executor = create_filesystem_executor(
    "C:/path/to/project", max_entries=100, max_bytes=65536, max_total_bytes=262144,
)
state = asyncio.run(run_agent(
    "先分页列举目录，再读取实际发现的源码或配置文件，依据正文回答并标明来源",
    settings, tools=[DIRECTORY_TOOL, TEXT_TOOL], execute_tool=executor,
))
```

此例沿用上一节的 `asyncio`、`run_agent` 和 `settings`。只开放读取时，使用 `create_text_executor(root, max_bytes=65536, max_total_bytes=262144)` 和 `[TEXT_TOOL]`；只开放目录时，原 `create_directory_executor` 用法不变。`run_agent` 本身不根据配置自动启用工具。

完整成功结果的内层 JSON 为：

```json
{
  "path": "docs/说明.md",
  "content": "示例",
  "encoding": "utf-8",
  "bytes_read": 6,
  "complete": true
}
```

`bytes_read` 为原始文件字节数，包含 BOM；有 UTF-8 BOM 时 `encoding` 为 `utf-8-sig`，正文去除开头的 BOM。严格解码，不忽略或替换错误字节；保留 CRLF、LF、CR，不归一化换行。空文件返回空正文、零字节及 `complete=true`。C0 控制字符除 TAB、LF、CR 外均拒绝，DEL 也拒绝；这是基础筛查，不是完整二进制格式识别。

打开前检查类型和大小，打开后通过文件描述符再次检查。读取采用无缓冲二进制模式，累计最多读取 `min(max_bytes, 本次运行剩余成功读取额度) + 1` 字节；额外字节仅用于识别超限。已知大小超限时不读正文，读取过程中发现超限也不返回部分内容。不存在分页或截断成功状态；中途 I/O 或解码失败不会伪装为空文件。成功结果是本次读取到 EOF 的完整内容，不是文件版本快照。

| 情况 | 分类 | 安全原因码 |
| --- | --- | --- |
| 字段错误 | 拒绝 / `invalid_arguments` | `invalid_text_argument_shape` |
| 路径或链接拒绝 | 拒绝 / `invalid_arguments` | 沿用公共路径原因码 |
| 非普通文件 | 拒绝 / `invalid_arguments` | `target_not_regular_file` |
| 单文件超限、累计额度不足、异常控制字符 | 拒绝 / `invalid_arguments` | `file_too_large`、`read_budget_exceeded`、`non_text_content` |
| 非 UTF-8 | 失败 / `execution_error` | `decode_error` |
| 不存在、权限不足、其他 I/O | 失败 / `execution_error` | `file_not_found`、`permission_denied`、`io_error` |

成功读取按 `bytes_read` 累计，包含 BOM，重复读取重复计量。完整读取、校验和结果序列化成功后才扣减，失败不占成功正文额度；随后日志失败也不撤销已发生的扣减。余额不足拒绝当前文件，但同批较小文件仍可执行，空文件不消耗额度。

内置文本/组合执行器为可调用对象，提供 `bind_run()`；每次 `run_agent` 自动获得独立预算实例，同一执行器连续或交错复用不串用计量。直接调用执行器时预算持续到该实例结束，需显式 `bind_run()` 才能开启另一份预算。组合执行器只绑定一次根目录，每次访问仍重新校验路径。只读 `read_budget` 快照含 `limit_bytes/used_bytes/remaining_bytes`，供运行记录和 Tracing 观测使用，不转换为模型额度提示，也不在 State 重复存储。包装执行器时应转发 `bind_run()` 和 `read_budget`；普通异步函数仍可注入，但自定义输出没有内置读取预算。

累计额度仅限制本次新返回的文件正文字节数，不保证整个模型上下文的 Token 或序列化字节数有界，也不限制历史输入、目录结果或任意自定义工具输出。文件 I/O 仍同步执行，慢磁盘可能占用事件循环。

原因通过固定表进入模型错误结果、JSONL 和 Tracing，不携带原始异常或正文。正文只作为关联原 `call_id` 的工具结果发送给模型，不提升为 system/developer 指令；不自动跟随链接、读取引用文件或执行其中命令。模型之后请求其他工具仍受绑定根目录和参数检查限制。文件正文会发送给配置的模型端点，选择根目录和输入文件时应考虑这一点。

## State

| 字段 | 用途 | 更新方式 |
| --- | --- | --- |
| `input` | 本次请求使用的协议上下文，包含会话继承的有效历史 | 独立运行初始化用户项；会话新轮用有效历史和新输入覆盖，节点只追加新增项 |
| `history_end` | 可以继承到下一轮的上下文前缀长度 | 合法输入、有效最终回答或完整关联的工具批次推进边界 |
| `run_id` | 本次用户请求的 UUID | 初始化一次，运行期间不变 |
| `log_path` | 本次 JSONL 文件的绝对路径 | 初始化一次，失败时也可定位预期文件 |
| `log_complete` | 本次运行记录是否完整 | 初始为 `False`；结束事件成功写入且文件成功关闭后才为 `True` |
| `result` | 最近一次业务模型调用结果 | 每次模型尝试替换；无响应的失败设为 `None` |
| `model_calls` | 本次请求内的业务模型调用尝试数 | 每次尝试加一，包括失败 |
| `tool_requests` | 本次请求内收到的工具请求数 | 按条数增加，包括未知工具、无效参数及因上限未执行的请求 |
| `tool_calls` | 本次请求内实际调用执行器的尝试数 | 进入执行器前增加；执行器异常也计数 |
| `tool_outcomes` | 实际成功、校验拒绝、执行失败和未执行数量 | 本次运行内累加 `succeeded/rejected/failed/skipped`，不统计调用方传入的历史结果 |
| `usage_totals` | 本次用户请求内业务与摘要调用的已知 Token 合计 | 每次实际调用后累加报告的数据，各计一次 |
| `summary_calls` / `summary_result` / `summary_usage_totals` | 本轮摘要尝试数、最近摘要结果和消耗小计 | 每轮重置；失败尝试仍计数，缺失 usage 不补零 |
| `active_context` | 活动版本、摘要与保留轮次引用、本轮起点 | 平时稳定追加；确认压缩时切换版本 |
| `compaction` | 最近恢复过程的来源、候选、阶段及确认信息 | 与预算恢复及保存阶段同步更新，不等同于业务回答 |
| `context_budget` | 最近检查的预算组成、决定与发起事实 | 每轮初始化、每次检查替换；拒绝不产生模型尝试或 usage |
| `context_risks` | 与 `input` 对齐的 opaque 风险标注 | 整体替换更新，随协议项同步继承与追加，不重复正文 |
| `stop_reason` | `completed`、`max_model_calls`、`context_budget_exceeded` 或 `error` | 运行中为 `None`，结束时写入 |
| `error` | 导致整个循环终止的安全说明 | 正常运行为空；可恢复工具错误保存在工具结果中 |
| `error_code` | 终止的固定错误码 | 正常为空；区分模型、协议、日志与关闭失败 |
| `cleanup_error` | 收尾错误及收尾前终止原因 | 正常为空；关闭失败保留 `code=client_close_error`、`previous_stop_reason` |

`ModelResult` 包含 `text`、`has_refusal`、`raw_response`、`output_items`、`tool_calls` 和 `usage`。工具请求保留 `call_id`、名称及原始参数字符串；原始响应保留可序列化的协议数据。业务响应在下一轮上下文中仅追加 `output_items`，包括 reasoning 和端点扩展字段，不追加整个响应对象，也不重复追加历史；摘要响应先通过独立校验，只有派生摘要消息随已确认的活动版本启用。

`result.usage` **仅表示最近一次业务模型调用**的输入、输出和总 Token。未报告 usage 时为 `None`；部分字段缺失时相应值为 `None`，不沿用上一轮。

`state.usage_totals` **仅统计一次 `run_agent` 或 `run_turn` 处理一次用户请求**，包括该请求内部业务与摘要的全部实际模型调用；`summary_usage_totals` 是其中摘要调用的小计，不能再次加到总计中。`summary_result.usage` 只表示最近一次摘要调用。每次新运行都重新初始化，与其他请求无关：

- `input_tokens`、`output_tokens`、`total_tokens` 初始均为 `None`，收到实际数值后才累加；缺失或无效字段不补零、不推算。
- `missing_usage_calls` 记录没有完整 usage 的模型尝试数，包括未收到响应的失败。大于零时，累计值只是已知小计，不能视为完整消耗。
- 失败响应若包含可用 usage，仍累计其中已知数据；最近结果和累计数据各自只更新一次。

## 工具调用与终止

执行器是异步可调用对象：`execute_tool(name, arguments) -> str`。提供工具定义时须同时注入执行器；工具定义、执行器及调用上限作为运行依赖，不存入 State。

执行器可选提供 `bind_run()` 和 `read_budget`：前者为新请求创建独立额度，后者仅提供 `limit_bytes/used_bytes/remaining_bytes` 的观测快照。普通异步函数无需实现这两项；包装内置执行器时，应按需要转发这两个能力。没有 `read_budget` 属性或明确返回 `None` 均表示未提供额度元数据，属于正常情况；提供了无效的非 `None` 快照，或属性读取、快照验证发生普通异常时，以 `read_budget_unavailable` 安全诊断降级为未知，不补零或沿用旧值，不改变业务结果、保存结果或工具内部的实际额度约束。取消等控制异常仍按原规则传播。收尾只读取一次额度，JSONL 和 Tracing 使用同一份观测。

纯工具请求是有效模型响应；文本与工具请求并存时先处理工具。每批按请求顺序串行执行，所有结果按 `call_id` 回传后再调用模型。工具名称必须属于本次提供的定义，参数必须可解析为 JSON 对象；具体字段由执行器校验，不提供通用 JSON Schema 校验引擎。

参数中的每个 JSON 对象都必须具有唯一键名，包括嵌套对象和数组中的对象。按 JSON 解码后的键名精确比较，例如 `key` 与 `\u006bey` 视为同名；不做大小写折叠或 Unicode 归一化。不同对象各自使用同名字段允许通过。重复键在 `argument_parsing` 阶段返回 `invalid_arguments`，计为拒绝且不进入执行器；同批其他请求继续处理。此前重复键会静默保留后值，现在明确拒绝，错误说明不包含原始参数或键名。Schema 的 `strict: true` 不能替代这些解析检查或工具自身的字段与业务校验。

工具结果使用 `function_call_output`，其 `output` 为 JSON 字符串。成功返回 `{"ok": true, "result": "..."}`；失败返回 `{"ok": false, "error": {"code": "...", "message": "..."}}`。错误码为 `unknown_tool`、`invalid_arguments` 或 `execution_error`。参数语义错误由执行器抛出 `ToolArgumentError`，已分类执行失败使用 `ToolExecutionError`；这两类在 `error` 中另带 `reason_code`、`reason_message`，均由应用固定表生成。其他执行器异常仍使用安全的通用错误消息，未知原因码退回对应的安全默认值。

工具错误会回传模型，继续处理同批其他请求，并允许模型在剩余次数内修正。未知工具和 JSON 解析失败不会进入执行器，不增加 `tool_calls`；执行器内部校验失败或执行异常仍计一次。得到最终回答不代表此前失败的工具操作已经成功。

`ToolArgumentError` 计入 `rejected`，即使它发生在执行器内部；其他执行异常计入 `failed`。`completed` 仅表示模型给出了最终回答，不表示所有工具成功或业务目标已验证。CLI 始终展示程序统计；模型在工具失败后输出“成功”不会覆盖日志和失败计数。

执行器返回非字符串同样计为一次执行尝试和一次失败，返回 `execution_error`；不自动将返回值转换成字符串或 JSON。同批成功、失败和拒绝结果均按原请求顺序关联，全部处理后才发起下一次模型请求。若写入拒绝或失败事件时日志故障，已有统计及已生成结果仍留在 State，剩余请求跳过且不再调用模型；State 中存在错误结果并不表示它已经回传。

剩余调用次数包含即将发出的本次业务调用。大于一次时，普通业务请求不注入额度指令；只剩一次时，`instructions` 固定为“本次仅根据已有资料回答；资料不足时明确说明，不再请求工具。”，同时设置 `tool_choice="none"`。默认最多 30 次，`MAX_MODEL_CALLS=1` 时从第一次起就采用该收尾策略。调用次数、读取额度、用量及预算判断由 Agent 代码管理，并保留在 JSONL 和 Tracing 中，不改写消息或工具定义来逐轮播报。读取拒绝及失败仍作为关联原调用的工具事实进入模型上下文，不会随额度提示一并删除。

调用次数耗尽后不发起额外请求。若兼容端点在末轮仍返回工具请求，则保留请求事实，以 `max_model_calls` 结束，不执行该批工具，也不发起第 N+1 次请求。模型错误说明和阶段由固定错误码生成，超时、连接、HTTP 与协议错误可区分，不展示任意异常文本。模型或协议错误以 `error` 结束，返回已产生的上下文、计数、可取得的响应及 usage，而不是丢失失败节点的统计。配置或输入错误在调用模型前报告。摘要模型的固定任务指令和独立调用统计不受这一业务请求策略影响。

客户端关闭只尝试一次。已有最终 State 时关闭失败保留回答、上下文、统计及 usage，并记录 `client_close_error`；已有主要错误不被关闭错误覆盖，取消或配置异常继续传播。日志仍尝试完成，`log_complete` 只描述日志完整性，Tracing 上传失败仍不阻断业务。

当前只保证模型调用轮数有界，不对单次工具执行设置超时。单个异步工具如果长期不返回，模型轮数上限不能终止它。

## 运行记录

`LOG_DIR` 默认为 `logs`，每次运行创建独立的 `<run_id>.jsonl`。通过 `load_settings` 加载时，相对路径以 dotenv 文件所在目录为基准；直接构造 `ModelSettings` 时相对当前工作目录解析。调用方从返回 State 的 `run_id`、`log_path` 和 `log_complete` 获取记录信息。默认日志目录被 Git 忽略；自定义目录需自行安排保留与清理。

每行是 UTF-8 JSON，写入后立即 flush。记录器按运行绑定，不使用共享的可变运行标识，日志也不进入模型上下文。通用字段包括 UTC `timestamp`、运行内递增的 `event_seq`、`event` 和 `run_id`。每次模型或工具节点执行分配 `step-0001` 形式的 `step_id`，并记录 `node`；同批工具共享步骤，运行级事件省略步骤。

会话运行另带 `session_id`，可将多次独立 `run_id` 关联到同一会话；过滤后的 Tracing 同样保留该关联。检查点故障、此前已确认的检查点及最终保存待确认状态与运行结果分别记录，最终确认以 `SessionTurn` 为准。

工具事件保留原 `call_id`、`tool_name` 和从 1 开始的 `request_index`（本轮响应中的工具请求序号）。缺失标识不补造；重复标识结合步骤和请求序号定位。

| 事件 | 记录的事实 |
| --- | --- |
| `run.started` / `run.ended` | 开始与终止；结束时含调用统计、工具处理统计、最近及累计 usage、读取预算快照、拒答与收尾错误 |
| `run.cleanup_failed` | 客户端关闭失败；已有主要错误、回答与统计保留，日志完整性另行判断 |
| `run.input_rejected` | 初始消息或工具项形状、工具关联校验失败，含安全协议项位置与错误码 |
| `context_budget.checked` | 候选请求的估算方式、预算组成、阈值、判断及工具结果关联；该记录本身不证明请求已发出 |
| `compaction.*` | 压缩范围、摘要候选及版本保存/使用阶段；只含引用、预算与安全诊断，不含正文 |
| `summary.*` | 摘要请求计量、实际发起、返回或失败及可取得的 usage；不等于版本已启用 |
| `model.started` | 模型调用即将发起，含剩余调用次数、最后回答轮标志、读取预算及实际输入中全部、有序的 `input_tool_result_call_ids` |
| `model.returned` / `model.failed` | 可取得的响应摘要与 usage，或固定错误码及失败阶段 |
| `tool.rejected` / `tool.started` / `tool.succeeded` / `tool.failed` | 名称、JSON、字段校验或实际执行的结果 |
| `tool.result_created` | 对应 `call_id` 的协议结果已经生成，与执行成功分开记录 |
| `tool.skipped` | 因模型上限、协议错误或日志故障而未执行 |

核对回传时，先找到 `tool.result_created`，再检查下一次 `model.started` 的关联列表；工具成功本身不能证明结果已进入后续输入。该列表包括当次输入的全部历史工具结果，同一个结果可出现在多个后续模型请求中。调用开始记录表示已进入请求边界，不表示远端已成功接收；模型返回或失败事件给出后续结果。

每个实际模型调用前先记录预算判断。检查未通过，或放行后开始事件写入失败时，不调用模型；最近预算决定仍保留在 State。预算诊断只含安全方法名、规模、配置额度、步骤及关联信息，不能从被过滤的 Tracing 展示反推完整请求。日志结束记录中的预算和调用事实、`SessionTurn` 的最终保存确认分别核对，日志完整不等于检查点保存成功。

参数摘要仅允许出现工具 Schema 中的字段名，敏感字段名显示为 `[redacted]`，未知字段名显示为 `[unknown]`。不预览任何参数值；只记录类型、字符串长度或集合条数，嵌套对象不展开。模型文本、工具返回值和无效 JSON 也只记录类型与大小。每份摘要最多 16 个字段、字段名最多 64 字符、序列化后最多 1 KiB；敏感信息处理先于截断，省略项明确标记。异常仅记录固定错误码和阶段，不包含原始异常文本、配置环境、凭据、完整 State 或原始模型响应。摘要不修改工具实际参数和回传内容。

日志创建或写入失败会停止后续模型、工具调用，以 `error` 结束并设置 `log_complete=False`。已发生的调用、结果和 usage 保留；未启动的剩余工具计入 `skipped`。开始事件写入失败时不启动对应调用；结束写入或关闭失败也不会宣称记录完整。故障可能使文件不存在或只含部分记录，以返回 State 和 CLI 的完整性标志为准。配置异常或主动取消仍沿用原异常/取消行为，已打开的记录尝试写入 `run.aborted` 并关闭，不伪装正常结束。

日志不属于可恢复检查点，也不提供进程崩溃后的任务恢复或写入持久性保证。

工具拒绝和已分类执行失败另含 `reason_code` 和固定 `reason_message`，例如 `target_not_directory`、`link_or_reparse_forbidden`、`decode_error`。这些应用诊断同时进入工具错误结果，关闭追踪也保留；错误大类和计数不变，未知原因码使用安全默认值，不记录异常原文。旧日志未保存的信息不能据此恢复。

## Benchmark

Benchmark 将工程约定、项目任务和验证方式分别组织。查看可用集合并执行离线工程回归：

```powershell
uv run --locked python -m benchmarks list
uv run --locked python -m benchmarks run agent-contracts
uv run --locked python -m unittest discover -s benchmarks/tests -v
```

无参数显示帮助，不执行检查。`agent-contracts` 覆盖五个确定性验证模块、15 条隔离行为链、141 个行为组；模块依据执行、文件系统、会话、上下文和记录约定划分，不作为五个样例项目。成熟的检查已迁入 `benchmarks/checks/`，原 `scripts/check_*.py`、`--suite`、`control-*` 与 `--research-profile` 入口已撤销。

`wechat-reporting` 是固定项目任务集，包含八类任务、九个实例、19 个计划轮次，沿需求覆盖、代码与需求的反向对应、缺陷/性能分析和改进方案组织，并覆盖多轮纠正、读取受限和摘要后继续研究。真实运行使用显式启用的通用配置：

```powershell
uv run --locked python -m benchmarks run wechat-reporting --config logs/run-config.json
uv run --locked python -m benchmarks review logs/benchmarks/<运行ID> --file logs/review-filled.json
```

配置样例默认禁用。运行配置 v2 的 `agent_source: "project_env"` 复用当前生产配置，优先级为任务覆盖 > `model_overrides` > 进程环境 > 项目根 `.env` > 默认值；v1 的显式模型与 `credential_env` 语义保持不变。v2 可用 `units` 指定任务顺序，用 `gate_first_sample` 核对首例可信完成、本地日志/检查点和回答产物后再继续；此门槛不评价回答质量。外部追踪默认关闭，仅对 `tracing.unit_ids` 指定且显式启用的实例上传经过过滤的数据。

仅运行工程集不会加载 `.env`、传递真实凭据、调用模型或外部追踪。研究材料取自固定 Git 对象，每个实例独立复制；评分资料不进入 Agent 可访问目录。共享入口与执行器不包含项目用例分支，新增项目只需定义材料、任务和 Rubric 并登记到固定目录表。

结果使用 v3，继续只读校验、渲染 v1/v2；执行状态、评测判定和 Agent 终态分别记录。JSON、Markdown、逐组事件、输出和回执存入独占的 `logs/benchmarks/<运行ID>/`；组、任务、Agent 运行和模型消耗分别统计，未知不补零，不合成总分。机械规则、检查项清单、人工 Rubric 与 LLM 建议适配分别组织；未经人工核对的模型建议保持待评审。

退出优先级为 `2 > 1 > 3 > 0`：2 表示执行/证据/保存故障、阻塞或适用覆盖缺口，1 表示有效未通过，3 表示仍待评审，0 表示选定范围完整通过且成功保存。断言失败后保留组事实和依赖未执行项，其他独立行为链可继续；记录或收尾不确定时停止。标签与失败环节不自动证明根因。

运行、配置、结构、扩展和评审方法见 [Benchmark 使用说明](benchmarks/README.md)，原检查去向见 [覆盖映射](benchmarks/COVERAGE.md)，项目事实与评分边界见 [WeChat Reporting 任务说明](benchmarks/suites/wechat_reporting/README.md)。语义评审完成前，真实运行结果保持待评审；执行和追踪完成均不能证明 Research 质量通过。

生产预算与批次上限分别管理。[研究运行样例](benchmarks/entrypoints/run.example.json) 不再固定旧的每轮 8 次调用、60 秒请求、32768 窗口和 4096 输出组合，未覆盖的值按 v2 配置优先级读取。读取受限例仍使用 4 KiB；摘要例 v2.3 使用 65536 窗口、6 KiB 背景和至多 32 KiB 压力，保留最近两轮、40% 动态摘要目标，其他任务沿用当前配置。执行器硬上限为 400 次实际请求、7200 秒累计、每实例 1200 秒；首轮计划九个实例各一次、19 轮，采用更低的 200 次请求、5400 秒累计和每实例 1200 秒。业务、摘要及失败请求均计数；费用无账单时为未知，单次观察不构成稳定性结论。

## 独立端点诊断

`scripts/verify_*.py` 保留为按需调用的真实端点诊断，使用当前本机配置；不会由 Benchmark 的列举或工程回归自动运行：

```powershell
uv run --locked python scripts/verify_function_calling.py
uv run --locked python scripts/verify_directory.py
uv run --locked python scripts/verify_text.py
uv run --locked python scripts/verify_review_fixes.py
uv run --locked python scripts/verify_sessions.py
uv run --locked python scripts/verify_compaction.py
uv run --locked python scripts/verify_tracing.py
```

这些脚本使用非敏感临时材料，检查 Function Calling、目录列举、读取与回答、末轮额度、历史或摘要的真实服务表现。读取诊断需要至少三次模型调用额度；固定结果工具诊断需要至少两次。它们比较实际 SDK 参数、工具结果与最终回答，不要求固定调用顺序。跨轮验证不重复提供旧正文，保留会话、运行与日志标识；只验证进程内连续性，不证明跨进程恢复。

Tracing 诊断会访问已配置的真实模型及 LangSmith，查询远端记录并给出私有 Trace 链接，之后仍需登录核对网页展示。usage 分类只使用最终投影中有效字段，零值有效、缺失不补零；缓存字段仅作端点观测，不推断提示优化收益。其 usage 对比公共函数与确定性回归复用，避免维护两份断言。离线 Tracing 检查则使用真实 SDK 序列化与替身 HTTP，不能证明真实服务接收或网页显示正确。
