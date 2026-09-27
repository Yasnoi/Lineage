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
TIMEOUT_SECONDS=60
MAX_MODEL_CALLS=5
CONTEXT_BUDGET_TOKENS=32768
MAX_OUTPUT_TOKENS=4096
CONTEXT_SAFETY_MARGIN_TOKENS=1024
OPAQUE_FALLBACK_TOKENS=4096
CONTEXT_TRIGGER_PERCENT=90
CONTEXT_KEEP_RECENT_TURNS=2
CONTEXT_TARGET_PERCENT=70
SUMMARY_MAX_OUTPUT_TOKENS=2048
SUMMARY_MAX_BYTES=8192
LOG_DIR=logs
PROJECT_ROOT=
DIRECTORY_MAX_ENTRIES=100
TEXT_MAX_BYTES=65536
TEXT_TOTAL_MAX_BYTES=262144
```

`BASE_URL` 是支持 Responses 协议的 API 基础地址，按供应商要求包含版本前缀，不追加 `/responses`。仅支持 Chat Completions 的地址不满足本项目要求。`BASE_URL`、`API_KEY`、`MODEL_NAME` 必填；超时默认为 60 秒且必须为有限正数，模型调用上限默认为 5 次且必须为正整数。连接地址、模型名称及超时由配置对象共同校验，直接构造与 dotenv 加载遵循相同规则；超时不接受布尔值。

`PROJECT_ROOT` 可选，未配置或留空时保持无工具问答。`DIRECTORY_MAX_ENTRIES` 默认为 100，`TEXT_MAX_BYTES` 默认为 65,536 字节（64 KiB），`TEXT_TOTAL_MAX_BYTES` 为每次运行成功读取的累计额度，默认 262,144 字节（256 KiB），均必须为正整数；程序构造配置或执行器时也不接受布尔值。根目录和限额由调用方控制，模型不能修改。

配置通过 `python-dotenv` 显式读取当前工作目录的 `.env`，因此运行命令应位于项目根目录；已有进程环境变量优先。真实 `.env` 已加入忽略规则，连接配置本身不进入 Agent State 或应用日志；但根目录内的文件正文另受下述文件访问授权规则约束。

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
| `CONTEXT_BUDGET_TOKENS` | 32768 | 应用总预算，不能从模型别名推断为端点实际窗口 |
| `MAX_OUTPUT_TOKENS` | 4096 | 输出预留及实际请求的输出上限 |
| `CONTEXT_SAFETY_MARGIN_TOKENS` | 1024 | 协议转换及估算误差的安全余量 |
| `OPAQUE_FALLBACK_TOKENS` | 4096 | 缺少可信来源 usage 时，每个 opaque 单元的预留 |
| `CONTEXT_TRIGGER_PERCENT` | 90 | 触发压缩或停止的阈值占应用总预算的百分比 |

这些值可通过环境变量或 `ModelSettings` 显式设置；额度为正整数，安全余量允许零，比例为 `1–100`，均拒绝布尔值。触发阈值必须大于输出预留与安全余量之和。

```text
所需预算 = 透明内容估算 + opaque 风险预留 + 输出预留 + 安全余量
触发阈值 = floor(应用总预算 × 触发比例 / 100)
所需预算 < 触发阈值：放行；所需预算 >= 触发阈值：尝试符合条件的压缩，无法完成则停止
```

这是一套保守的应用策略，不是精确窗口保证。UTF-8 字节和 Token 不等价，opaque 输出 usage 也不能证明同一项再次输入时的实际占用。兼容端点的 tokenizer、协议转换和窗口限制可能不同，配置应以供应商实际限制为依据；一次成功请求不能消除估算误差。进入预算检查后，不可序列化、循环引用、非有限数值或其他计量失败以 `context_measurement_error` 停止，不按零放行。会话模式的原生初始检查点先于模型节点写入；若原提交本身无法被检查点保存，可能先返回 `checkpoint_write_error`，此时预算尚未执行、模型未调用，保存失败与会话阻断按既有规则处理，不宣称该提交已保存。上一轮 usage 和本轮累计 usage 都不替代下一次请求计量。

达到阈值且不能完成符合下述规则的压缩时，以 `context_budget_exceeded` 结束本轮，不发送对应业务模型请求，不增加该业务调用的尝试或缺失 usage 次数；之前的模型及工具调用、结果和消耗保留。首次调用前被拒绝的新输入仍留在本轮 State 和确认保存的检查点中，但不进入可继承历史，随后较小且预算允许的问题仍可使用此前有效历史。工具执行后超限则保留并继承完整配对的请求和实际结果，明确结果尚未进入后续模型调用；下一轮可能再次超限，不静默删除资料、不重复执行工具，也不声称可以自动恢复。预算拒绝本身不永久阻断会话，日志和检查点故障仍单独报告。

`context_budget` 保存最近检查的组成、阈值、决定、步骤及请求是否发起；每轮初始化，每次检查替换。`context_risks` 与 `input` 对齐，只保存 opaque 数量、预留依据和来源运行/模型调用，不重复保存正文；透明项为空。该字段整体替换，历史继承时同步裁取，避免检查点恢复重复追加或用后续响应的 usage 改写旧风险依据。请求副本、State 和已保存历史相互隔离。

### 有界摘要与活动版本

自动压缩只使用 `SessionManager` 已确认结束的旧轮次。单次 `run_agent` 不从任意结构化输入猜测轮次边界；没有旧轮可压缩时仍明确停止。正常预算内不调用摘要器、不重新排序或筛选历史。同一版本持续追加，只有候选通过校验并确认保存才切换版本。

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `CONTEXT_KEEP_RECENT_TURNS` | 2 | 原样保留最近的完整有效用户轮次，不包含当前交互 |
| `CONTEXT_TARGET_PERCENT` | 70 | 切换后完整业务请求须低于的应用预算比例，小于触发比例 |
| `SUMMARY_MAX_OUTPUT_TOKENS` | 2048 | 摘要请求的输出预留及实际输出上限，包含推理输出 |
| `SUMMARY_MAX_BYTES` | 8192 | 摘要完整协议消息的紧凑 JSON UTF-8 字节上限 |

上述配置均为正整数，不接受布尔值；目标预算必须容纳业务输出预留和安全余量。摘要请求继续使用原总预算、触发比例与安全余量，不绕过估算保护，不提高限额，也不启用供应商自动截断。

从最早尚未覆盖的旧轮选择连续前缀，按摘要消息上限预留空间，选择足以将完整候选业务请求降至目标以下的范围。当前问题、当前已产生的工具交互及近期完整轮次始终保留。未知协议项、非文本资料或含 system/developer 输入的轮次是不可越过的屏障；不跳过这些轮次任意筛选后文。完整旧轮中的加密 reasoning 可以退出活动上下文，但不当作可读文本送入摘要器，并标记未总结的资料。保留区的工具关联及 opaque 风险来源维持原样。

摘要采用同一端点、模型和客户端的独立 Responses 调用，职责、入口与统计独立，不提供工具。每个业务请求的预算恢复过程最多发起一次摘要调用，不递归摘要或自动重试。实际业务调用以后，后续请求因增长可以再次触发；业务调用上限仍限制整个流程。摘要输入为上一份摘要（若有）及本次新覆盖的原文，包含原角色、固定来源引用、工具结果和安全终态；当前新问题、无效尾部与未配对批次不作为摘要来源。

摘要输出为固定 JSON，分列有效要求、事实与进展、建议/推断、纠正与否决及未解决事项，每条带来源标识。应用检查形状、重复键、来源范围、长度、拒答和工具请求，随后重新构造并检查完整业务请求。校验只能确认结构与边界，不能证明语义无损；摘要可能遗漏信息或误解用户纠正。候选作为明确的派生 assistant 资料进入上下文，不提升为 system/developer 指令，不改变工具权限。模型返回的 reasoning 或其他非摘要内容不作为新活动摘要。

摘要调用发生、候选生成、候选校验、版本保存和业务请求实际使用是五个不同事实。摘要节点先保存候选及已发生的消耗；激活节点验证后替换活动输入与风险标注，等待同步检查点确认；后续业务节点再次检查实际请求，才能发起调用。未成功保存时停止，不自动回退执行；候选失败不破坏旧已确认版本。摘要调用本身不批准当前用户输入进入下轮历史；`history_end` 仍表示协议准入边界，切换时只调整它在新活动列表中的位置。已经生成的工具结果不会通过重跑工具重建。

无可压缩历史、摘要请求超限、摘要过大、无压缩效果或候选未达目标以预算不足停止，CLI 退出 `4`。摘要服务失败、无效响应或计量失败以错误停止，退出 `1`。日志与保存故障按原有规则独立报告，保存不确定则阻断会话。实际摘要调用与可取得的 usage 始终计入本轮统计，即使候选最终未启用。

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

独立模型接口为 `await generate_response(client, model_name, input, *, tools=None, instructions=None, tool_choice=None, max_output_tokens=4096)`（可另注入追踪器）；该兼容入口只负责单次请求，完整的调用前预算检查通过 `run_agent` 或 `SessionManager.run_turn` 使用。图按普通或收尾策略准备请求，适用时包含 `instructions` 和 `tool_choice`，完成预算检查后调用 `send_prepared_response` 发送同一份参数。工具定义使用 Responses 的 `type: "function"` 格式及 JSON Schema，不使用 Chat Completions 的工具格式。

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

剩余调用次数包含即将发出的本次业务调用。大于一次时，普通业务请求不注入额度指令；只剩一次时，`instructions` 固定为“本次仅根据已有资料回答；资料不足时明确说明，不再请求工具。”，同时设置 `tool_choice="none"`。默认最多 5 次，`MAX_MODEL_CALLS=1` 时从第一次起就采用该收尾策略。调用次数、读取额度、用量及预算判断由 Agent 代码管理，并保留在 JSONL 和 Tracing 中，不改写消息或工具定义来逐轮播报。读取拒绝及失败仍作为关联原调用的工具事实进入模型上下文，不会随额度提示一并删除。

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

Benchmark 是独立评测入口，由评测端选择检查并形成判定。当前控制回归定义覆盖 14 个原脚本入口，组织为 **5 个主评测集、12 个隔离执行单元、138 个检查组**。这些数字描述可选择的覆盖范围，实际执行、失败、未执行及环境覆盖缺口以每次报告为准；确定性控制回归不代表真实 Research 质量通过。

| 主评测集 | 主要职责 | 隔离单元数 | 定义检查组数 |
| --- | --- | ---: | ---: |
| `control-execution` | 执行与工具调度 | 2 | 30 |
| `control-filesystem` | 文件系统访问 | 2 | 18 |
| `control-sessions` | 会话与保存 | 2 | 25 |
| `control-context` | 上下文预算与摘要 | 2 | 33 |
| `control-recording` | 记录与追踪 | 4 | 32 |

原入口、稳定组 ID、行为、主要归属、跨模块标签与间接覆盖见[控制回归覆盖映射](benchmarks/COVERAGE.md)。评测集按单元主要职责选择，跨模块组合场景保留完整链路；标签用于说明覆盖，不会增加执行次数，也不能据此断言已运行其他入口内的同类场景。

使用前述 uv 环境，从项目根目录执行：

```powershell
uv run --locked python -m benchmarks --list

# 默认仍只运行 control-react，即原 ReAct 脚本的 20 组检查
uv run --locked python -m benchmarks
uv run --locked python -m benchmarks --suite control-react

# 单集、多个评测集或完整控制回归
uv run --locked python -m benchmarks --suite control-filesystem
uv run --locked python -m benchmarks --suite control-execution --suite control-sessions
uv run --locked python -m benchmarks --suite control-all

# 框架自身的故障注入与契约检查，不调用真实端点
uv run --locked python -m unittest discover -s benchmarks/tests -v
```

`control-react` 保留为 `react-script` 的兼容选择，`control-all` 选择全部 12 个单元。`--suite` 可重复指定；重复或重叠选择按单元 ID 去重，依固定定义顺序运行一次，并记录选择来源。未知评测集在启动前拒绝，不回退执行其他检查。未选单元单独列出，不计为通过。

每个单元在独立子进程中执行原脚本完整的 `__main__` 入口，包括同步、异步及入口后的检查：`check_sessions.py` 和 `check_session_review_fixes.py` 的异步 `main()` 后 CLI 检查仍执行；`check_tracing.py` 验证卡住上传的 CLI 子进程仍由原检查启动。`check_trace_data.py` 的 2 组及 `check_trace_usage.py` 的 4 组已由 Tracing 主入口完整调用，全量回归不再次安排独立单元。原脚本命令继续可用，例如 `uv run --locked python scripts/check_react.py`；完整列表见下方“独立验证”。

Benchmark 不读取 `.env`，子进程只继承筛选后的运行环境，不传递模型或追踪凭据，并确保 Python 断言启用。父环境的追踪开关关闭；离线 Tracing 检查仍在受控范围内使用合成配置开启追踪，保留真实 SDK 序列化、替身 HTTP、上传失败及有限收尾断言，不访问外部模型或追踪服务。每单元默认超时为 120 秒，超时请求终止进程树。

每次运行使用独立的 `logs/benchmarks/<UTC时间>-<UUID>/`。执行前保存 `plan.json`，随后输出 `result.json`、`report.md`；单元目录保存 `stdout.txt`、`stderr.txt`、`receipt.json` 和逐条落盘的 `groups.jsonl`。回执绑定运行与单元标识、源码版本、组计划及事件摘要，并与进程退出行为核对。报告只依据有效组事件和回执判断，不解析成功文字补判通过。原始 Agent JSONL 及安全检查产物仍保留原位置，按显式观测的标识或路径关联，不因导入辅助模块产生目录而认定该组已执行。

新结果使用 `schema_version="2"`。`units` 保留执行状态、评测判定和 Agent 终态分离的契约，新增 `groups`，记录计划序号、开始与结束、状态、失败环节、证据、观测及覆盖缺口。结果校验和报告仍接受 v1，按其原有脚本粒度呈现，不补造组明细。`baseline` 记录 Agent 提交、工作区修改状态、Python/实际依赖版本及锁文件摘要；定义、共享检查代码、配置和固定材料以对应内容摘要标识。修改中的工作区不能仅凭提交号复现，本框架不保存源码快照。缺失值使用 `availability=unknown` 与原因，不适用使用 `not_applicable`，不补零；耗时实测，替身 usage 不作为真实模型消耗。

| 情况 | 执行状态 | 评测判定 |
| --- | --- | --- |
| 原检查完整通过 | `completed` | `passed` |
| 原检查断言失败 | `completed` | `failed` |
| 启动、执行、超时或回执错误 | `failed` | `pending_review` |
| 检查入口等前置条件缺失 | `blocked` | `pending_review` |
| 已选择但未执行 | `not_run` | `pending_review` |
| 其余断言已完成，但适用环境样例无法构造 | `completed` | `pending_review`，另记覆盖缺口 |
| 研究执行完成但尚未评分（仅契约支持） | `completed` | `pending_review` |

Benchmark 中一个组断言失败后，保留已完成组，将当前脚本剩余组标为 `not_run`，不在状态不确定的同进程中强行继续；其他隔离单元可以继续。原 Review 脚本独立运行时仍保留其收集多个失败的行为，Benchmark 观察器激活后采用组级遇错停止。组执行异常或入口提前退出不伪造剩余组完成；中断、组记录故障或进程树收尾无法确认时停止后续调度。事件损坏时只保留有效前缀，缺失结束事件的组不判通过。环境能力不足的适用样例记录原因和覆盖缺口，不计为已验证；明确不适用的样例另行标记。

汇总分别报告所选评测集、单元、检查组的状态数量，以及组的计划、开始和完成数量，不能相加为“测试总数”。实际 Agent 运行按显式观测的 `run_id` 去重，标记 `completeness=partial`：取消等没有返回 State 的运行可能仅有日志引用。直接工具、SDK、检查点或追踪 harness 的观测单列，不把构造的状态与 UUID 当作 Agent 运行；模型请求数量与断言数量也不充当组数。Agent `completed` 不自动等于评测通过，预期的 Agent `error` 也可能是检查通过的证据。

退出码为：`0` 全部通过且报告保存成功；`1` 存在有效断言失败，其导致的同脚本后续组 `not_run` 不升级为框架故障；`2` 选择、执行、适配、记录或保存故障，单元阻塞/未执行，或存在适用环境覆盖缺口；`3` 执行完成但尚待评审。混合结果优先级为 `2 > 1 > 3 > 0`。保存失败不抹去已取得的结论，尽力保留产物并向 stderr 报错；仅确认保存的文件才显示 `Saved`。

定位失败时结合组事件、适用的运行/会话/步骤/调用/检查点标识、原日志及单元完整输出；组输出区间之外的 worker 异常堆栈仍可在完整 `stderr.txt` 中查看。`issues` 与覆盖标签说明发生环节和涉及能力，不自动证明根本原因。JSONL、经过过滤的 Tracing 与检查点用途不同，都不能默认作为完整模型请求或完整资料副本。

实现位于 [benchmarks](benchmarks/)，通过明确映射接入，无自动发现或任意命令执行接口。契约保留 `turns` 的多轮会话、检查点、回答、过程与摘要版本引用，以及 `dimensions` 的质量维度；[契约样例](benchmarks/tests/contract_examples.py) 只验证表达能力。真实研究执行器、评分器及效果基线尚未实现；后续参考答案与评分资料须与 Agent 可访问材料隔离。

## 独立验证

从项目根目录执行：

```powershell
# 确定性响应替身：协议、控制流、异常、计数、usage 与运行记录
uv run --locked python scripts/check_react.py

# 当前 .env 端点：无工具回归及固定结果工具的完整循环
uv run --locked python scripts/verify_function_calling.py

# 真实临时目录与确定性模型：目录边界、截断、失败和实际回传
uv run --locked python scripts/check_filesystem.py

# 真实临时文件与确定性模型：限额读取、编码、失败、隔离和实际回传
uv run --locked python scripts/check_text.py

# 确定性追踪检查：真实 SDK 序列化、替身 HTTP、清洗和 CLI 有限退出
uv run --locked python scripts/check_tracing.py

# 读取工具追踪：正文省略、模型引用、元数据和上传故障
uv run --locked python scripts/check_text_tracing.py

# Review 回归：协议、收尾、配置、分页、读取预算和最后回答轮
uv run --locked python scripts/check_review_fixes.py

# 会话检查：连续追问、状态隔离、历史边界、预算重置及检查点故障
uv run --locked python scripts/check_sessions.py

# 会话 Tracing：实际会话集成、过滤范围、上传和保存故障
uv run --locked python scripts/check_session_tracing.py

# 会话 Review 回归：异常缓存、额度元数据、输入准入和配置拒绝
uv run --locked python scripts/check_session_review_fixes.py

# 上下文预算：边界、opaque 来源、超限历史、计量与保存故障
uv run --locked python scripts/check_context_budget.py

# 会话摘要：范围引用、候选校验、版本保存、实际请求和故障边界
uv run --locked python scripts/check_compaction.py

# 摘要压缩：真实 SDK 序列化下来源/正文省略和上传故障
uv run --locked python scripts/check_compaction_tracing.py

# 当前 .env 端点：压缩较早的非敏感资料，再从摘要追问事实与来源
uv run --locked python scripts/verify_compaction.py

# 当前 .env 端点：读取非敏感临时资料，再在同一会话中追问事实和来源
uv run --locked python scripts/verify_sessions.py

# 当前 .env 端点：分页发现、源码/配置读取及最后回答轮兼容性
uv run --locked python scripts/verify_review_fixes.py

# 当前 .env 端点：在非敏感临时目录中列举根目录、子目录并回答
uv run --locked python scripts/verify_directory.py

# 当前 .env 端点：发现随机文件、读取正文并引用事实和来源
uv run --locked python scripts/verify_text.py

# 已配置 LangSmith 后：非敏感临时目录、真实模型、远端 Trace 查询
uv run --locked python scripts/verify_tracing.py
```

验证代码保留在独立脚本中，生产模块不导入脚本，CLI 不增加验证参数。`check_react.py` 和 `verify_function_calling.py` 自带临时固定结果工具；目录检查使用生产目录工具和真实临时目录。确定性检查使用标准库断言和模型替身，不需要测试框架或真实凭据。真实固定结果验证需至少两次模型调用额度；目录验证需至少三次额度，检查随机临时文件路径进入最终回答，不要求固定的实际调用次数，也不将一次回答视为具备代码理解能力。

确定性检查将日志保留在 `logs/checks/<检查标识>/`，逐场景输出运行、步骤、工具调用和日志路径。检查将日志中的关联列表与替身实际捕获的模型请求逐项比对，并覆盖工具失败后修正或误报成功、嵌套数据与并发运行隔离、摘要处理和日志 I/O 故障。真实脚本同时显示运行标识、工具处理统计和日志完整性。

检查还覆盖同批部分失败、执行器非字符串返回、各层级重复参数键，以及 `tool.rejected`、`tool.failed`、`model.failed` 写入时再次发生故障。故障场景联合核对返回 State、实际调用记录和日志完整性；未落盘事件的定位信息来自故障注入器观察到的写入尝试，并单独标明来源，不视为已经持久化的日志。

目录检查覆盖 Unicode 名称、上下级目录、空目录、条目上限、参数及路径拒绝、同批失败后继续、不同根目录运行隔离、配置和 CLI。权限与中途 I/O 故障使用标注的注入；Windows symlink/junction 尝试创建真实样例，无法构造时单独报告跳过原因。工具真实返回值与后续 SDK 请求逐项比对，日志关联和安全摘要分别核对。

这两类结果分别报告，确定性检查不能代替真实端点的 Function Calling 兼容性验证。

`check_tracing.py` 不使用真实凭据或网络，核对实际 SDK 序列化载荷、跨消息重组后的凭据遮盖、上下文成组保留、遗漏计数、容量限制、同批逐条记录和开关前后的业务等价性；也覆盖连接失败、HTTP 错误、初始化/清洗失败、队列溢出、取消和上传卡住时的 CLI 退出。边界检查由 `check_trace_data.py` 和 `check_trace_usage.py` 提供，并由该脚本统一执行。

`verify_tracing.py` 使用 `.env` 中的真实模型与 LangSmith 配置，仅访问临时样例目录；它查询远端记录、输出私有 Trace 链接和本地运行标识，并分别报告同批请求、拒绝是否在该次真实运行中出现。usage 按最终投影与远端一致的字段报告每次调用的 `none`、`partial` 或 `full` 及汇总数量；合法零值有效，缺失字段不补零，`usage_covered=true` 仅表示至少一次调用有有效 token 字段。之后仍需登录网页核对参数、结果、调用关联和耗时。确定性检查不能证明真实服务接收或网页显示正确；真实远端与网页验收仍待执行。

`check_review_fixes.py` 另核对输出角色与拒答、直接配置校验、关闭失败叠加错误、分页候选内存、累计预算及执行器复用隔离、最后回答轮和追踪字段。`verify_review_fixes.py` 使用随机非敏感 Python/TOML 文件验证分页、正文与来源引用，另发起一次仅允许回答的真实调用。脚本捕获实际 SDK 请求，检查普通请求省略额度指令、末轮使用固定收尾说明及 `tool_choice="none"`、上下文稳定追加，以及工具定义不随额度变化；读取额度仍与运行记录核对。

真实验证单独报告端点实际返回的 `usage.input_tokens_details.cached_tokens`。只有有效的非负整数才作为缓存观测，缺失或无效标为不可观测，不推算为零。该观测不证明提示词调整带来缓存收益；一次命中或未命中也不能替代功能验收。

`check_text.py` 使用真实临时文件，检查字节边界及实际读取量、短读和增长、UTF-8/BOM/换行、控制字符、路径拒绝、批次继续、隔离和文件未修改。权限、读取中途故障及元数据变化采用明确标注的注入；真实链接无法创建时报告跳过原因。`check_text_tracing.py` 使用真实 SDK 序列化和替身 HTTP，区分被省略的工具正文与允许按模型文本规则处理的回答引用，比较开关及上传故障下的实际模型请求和统计，不代表远端网页验收。

`verify_text.py` 需要至少三次模型调用额度，并在不含敏感信息的临时目录内验证“列举 → 读取 → 回答”。文件名和正文识别码不预先出现在提示中，检查真实模型请求携带完整正文，最终回答引用内容与来源；实际调用次数不固定。

`check_sessions.py` 将实际 SDK 请求与历史消息、真实临时文件结果逐项比较，检查历史不重复、不同会话隔离、新问题计数重置及同会话重叠拒绝。保存故障使用明确的注入，结合返回 State、实际调用与客户端关闭判断，不把缺失记录补造为成功；逐场景报告会话、运行、适用检查点和日志。`verify_sessions.py` 在第一轮发现并读取随机文件，第二轮不重供正文事实，分别核对真实请求携带的历史及模型回答引用的事实和来源，不固定实际调用次数。两者均只验证同进程保存，不证明跨进程加载或任意中断恢复。

`check_session_review_fixes.py` 检查故障缓存不保留异常调用栈、额度观测异常不改变工具限额或收尾、无效新增输入不污染后续历史，以及执行前配置拒绝后同会话可继续。它对照实际 SDK 请求、返回 State、检查点和安全记录，并将元数据故障与日志或保存故障组合验证；不会把未知 HTTP 错误当作输入格式拒绝。

`check_context_budget.py` 对照计量使用的准备请求与替身实际捕获的 SDK 参数，检查阈值两侧及相等边界、应用指令和工具占用、opaque 来源 usage 与回退、首次拒绝和工具后拒绝的历史差异。它还验证连续追问不改写历史、计量故障不产生虚假调用、对象隔离、日志/保存故障、CLI 和 Tracing，逐场景输出会话、运行、步骤、适用检查点及日志定位。既有大载荷检查显式配置符合其测试目的的预算，保留原有安全断言。真实端点只运行正常两轮会话回归，不发送巨大请求探测窗口，也不以一次真实成功证明估算精确。

`check_compaction.py` 对照实际捕获的摘要与业务 SDK 请求，核对原文引用、连续覆盖范围、近期协议交互、候选保存和版本实际使用；确定性替身用于控制流、统计和故障边界，不证明摘要语义无损。`check_compaction_tracing.py` 使用真实 LangSmith SDK 序列化和替身 HTTP，验证摘要材料与正文省略、内部来源标记、容量上限、最终业务回答引用以及上传故障不影响模型结果。其独立追踪测试记录明确标为 trace harness，不声称存在原生图检查点。

压缩检查还将轻量选源与完整记录查询逐项比较，报告实际检查点读取次数，核对多次摘要、失败轮次、无效尾部和副本隔离。压缩阶段及可选字段由类型声明描述，运行时保存校验和 Tracing 的独立允许字段规则仍保留；检查覆盖各阶段的记录、CLI 展示及正文省略。

`verify_compaction.py` 通过当前端点，在可控预算内压缩非敏感临时资料后继续追问；关键事实来自被压缩的旧记录，不在近期原文或新提示重复提供。分别检查实际请求中的摘要来源与版本，以及模型是否保留关键事实、纠正、失败限制和来源。调用次数取决于实际端点表现，不固定次数，也不使用巨大请求探测窗口；真实结果仍不能保证未来摘要无损或估算精确。

该真实脚本会打印样例使用的预算配置：总预算仍为 32768，摘要输出预留为 8192，摘要消息上限为 4096 字节，业务输出预留为 1024，安全余量为 256。这些是验证样例的显式覆盖，不修改应用默认值。摘要输出额度也包含推理输出；默认 2048 在某些模型上可能不足，此时明确停止，不自动提高额度或重试。摘要还可能遗漏事实；结构与来源校验通过不等于语义完整。

当前提供目录直接子项分页列举、限额内 UTF-8 普通文本读取和进程内连续会话，不实现 PDF/Word 转换、正文分页、文件搜索、文件修改或命令执行，不包括工具自动发现、Skills、流式输出、并行执行、磁盘持久化或任务恢复。两类文件工具均不提供严格耗时保证，也不构成操作系统安全沙箱。

项目设计依据见 [技术选型](docs/技术选型.md) 和 [开发阶段划分](docs/开发阶段划分.md)。
