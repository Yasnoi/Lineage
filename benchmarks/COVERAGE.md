# 控制逻辑回归覆盖映射

本文面向维护控制逻辑回归的开发者，说明原独立入口、Benchmark 隔离执行单元和检查组之间的对应关系。当前定义包含 **14 个原脚本入口、12 个隔离执行单元、138 个检查组**。这些数字描述受版本管理的覆盖定义，不表示某次运行已经执行或通过；实际完成、失败、阻塞、未执行和证据位置，以该次运行的 `result.json` 与 `report.md` 为准。

定义保存在 [execution.json](definitions/execution.json)、[sessions_context.json](definitions/sessions_context.json) 和 [recording.json](definitions/recording.json)，由 [catalog.py](catalog.py) 显式列举。增加或修改检查时，需要同步维护定义和本表。框架不从目录中自动发现评测，不接受任意命令作为评测入口。

## 选择与计数

从项目根目录运行；以下命令采用 PowerShell：

```powershell
uv run --locked python -m benchmarks --list
uv run --locked python -m benchmarks --suite control-filesystem
uv run --locked python -m benchmarks --suite control-execution --suite control-sessions
uv run --locked python -m benchmarks --suite control-all
```

省略 `--suite` 时仍选择 `control-react`，只运行原 ReAct 脚本。`control-react` 是 `react-script` 的兼容选择，`control-all` 选择全部 12 个单元。重复选择或选择重叠评测集时，每个单元按固定定义顺序只运行一次；保留触发该单元的选择来源，不把同一次执行计为多份覆盖。

| 主评测集 | 职责 | 隔离单元数 | 定义检查组数 |
|---|---|---:|---:|
| `control-execution` | 执行与工具调度 | 2 | 30 |
| `control-filesystem` | 文件系统访问 | 2 | 18 |
| `control-sessions` | 会话与保存 | 2 | 25 |
| `control-context` | 上下文预算与摘要 | 2 | 33 |
| `control-recording` | 记录与追踪 | 4 | 32 |

评测集按隔离单元的主要职责选择，组级主要归属与标签用于说明真实行为。`review-fixes-script` 保留在执行集，其中分页、读取额度和追踪组仍分别标注文件系统、记录职责。会话和摘要入口内的追踪组合场景同样保留原链路。单独运行某个主评测集，不表示运行了其他入口内所有同标签场景；完整控制回归使用 `control-all`。

评测集、隔离单元、检查组、实际 Agent 运行分别计数。一个组可以调用 Agent 多次，也可以完全不运行 Agent；标签数量、模型请求数量、断言数量不加入检查组总数。纯投影、直接 SDK、工具执行器或追踪 harness 的构造状态和标识不计为实际 Agent 运行。实际运行数量只依据明确观测到的调用身份，缺失观测不补零、不由日志文件个数或旧脚本汇总文字推算。

## 原入口与执行单元

每个隔离单元运行原脚本的完整入口，保留同步、异步、CLI 检查及 fixture 收尾。原独立命令 `uv run --locked python scripts/<脚本名>.py` 仍可使用。

| 原入口 | Benchmark 执行单元 | 主评测集 | 定义检查组数 | 接入关系 |
|---|---|---|---:|---|
| [check_react.py](../scripts/check_react.py) | `react-script` | `control-execution` | 20 | 完整独立入口 |
| [check_filesystem.py](../scripts/check_filesystem.py) | `filesystem-script` | `control-filesystem` | 9 | 8 个原函数组及最后的 fixture 完整性组 |
| [check_text.py](../scripts/check_text.py) | `text-script` | `control-filesystem` | 9 | 完整独立入口 |
| [check_review_fixes.py](../scripts/check_review_fixes.py) | `review-fixes-script` | `control-execution` | 10 | 保留共享 fixture 与跨组依赖；组级记录实际职责 |
| [check_sessions.py](../scripts/check_sessions.py) | `sessions-script` | `control-sessions` | 13 | 异步 main 及入口外追加的同步 CLI 检查 |
| [check_session_review_fixes.py](../scripts/check_session_review_fixes.py) | `session-review-fixes-script` | `control-sessions` | 12 | 完整独立入口 |
| [check_context_budget.py](../scripts/check_context_budget.py) | `context-budget-script` | `control-context` | 14 | 完整独立入口 |
| [check_compaction.py](../scripts/check_compaction.py) | `compaction-script` | `control-context` | 19 | 完整独立入口 |
| [check_tracing.py](../scripts/check_tracing.py) | `tracing-script` | `control-recording` | 17 | 完整入口；包含 trace_data 的 2 组、trace_usage 的 4 组 |
| [check_text_tracing.py](../scripts/check_text_tracing.py) | `text-tracing-script` | `control-recording` | 5 | 原行为组及最后的 fixture 完整性组 |
| [check_session_tracing.py](../scripts/check_session_tracing.py) | `session-tracing-script` | `control-recording` | 6 | 原行为组及最后的 fixture 完整性组 |
| [check_compaction_tracing.py](../scripts/check_compaction_tracing.py) | `compaction-tracing-script` | `control-recording` | 4 | 按 4 个完整行为组记录，包含 5 次追踪 harness exercise |
| [check_trace_data.py](../scripts/check_trace_data.py) | `tracing-script`，间接调用 | `control-recording` | 2，已包含在该单元 17 组中 | 原入口的两个函数由 tracing 入口完整调用，不再安排第二个单元 |
| [check_trace_usage.py](../scripts/check_trace_usage.py) | `tracing-script`，间接调用 | `control-recording` | 4，已包含在该单元 17 组中 | 原 main 的四个函数由 tracing 入口完整调用，不重复执行和计数 |

原脚本之间导入替身、`audit_recording`、读取探针、`capture` 等辅助函数，不代表间接执行了被导入脚本的全部检查。只有表中 trace_data 和 trace_usage 的明确调用构成间接覆盖；其他同名或相似函数按各自输入、配置与失败边界保留。

新增的三个 `fixture_integrity` 组分别归属于 filesystem、text-tracing 和 session-tracing 单元，只将原入口末尾已有断言显式列入计划；没有增加生产行为要求。text 单元的每组 fixture 不变断言仍计入对应组，完成该断言后才记录组通过。

compaction-tracing 原输出的“6 groups”口径没有六个独立行为边界。当前四组分别为正常／关闭／上传失败的对照链、摘要意外工具调用、不完整摘要、来源引用与明确派生关系；前一组包含三个相互对照的 exercise，加上后两个摘要 exercise 共五次。所有原断言保留，三个等价性 exercise 不拆开计为三个独立组。

## 执行事实与诊断边界

组计划在执行前确定，结构化事件分别记录开始和结束。未到达的组保留为未执行；已开始但未正常结束的组不能判为通过。Benchmark 中组断言或异常会停止当前脚本的后续组，其他隔离单元可以继续；不能在共享状态或清理状态不确定时强行继续同进程检查。Review 原独立入口前四组的异常聚合行为保留，Benchmark 激活组观察时遇错停止。

文件系统链接样例受环境能力影响：适用样例创建失败须记录覆盖缺口和原因，不计为已验证；非 Windows 的 junction 样例标不适用。通过部分样例不代表整个链接边界已覆盖。原 stdout 的 `PASS` 或旧汇总不会覆盖结构化结果中的阻塞、失败和未执行事实。

组证据包括本次 stdout/stderr 的对应位置、实际返回状态中的身份、明确取得的运行日志以及追踪或故障注入产物。会话、步骤、调用、检查点、摘要和活动版本标识仅在适用且已取得时关联。辅助模块导入产生的目录或路径对象，不自动成为该组执行证据；经过过滤的 JSONL 与追踪也不代表完整模型请求或全部资料副本。

失败环节依据组异常、断言位置和实际证据定位，覆盖标签仅说明受影响行为。标签不能证明故障根本原因；根因仍需分析或对照实验。本映射只组织确定性工程回归，不提供真实模型 Research 质量结论。

## 检查组明细

每行是一项计划执行的组，顺序与对应单元的原入口一致。`source symbol` 中的 `main:名称` 表示原 main 内的连续行为块，`main` 表示其收尾断言；没有对应函数时不虚构函数入口。标签使用定义中的稳定枚举：`execution`、`filesystem`、`sessions`、`context`、`recording`。

### `react-script`

主评测集：`control-execution`；入口：[scripts/check_react.py](../scripts/check_react.py)；计划 20 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_react/check_direct_answer` | 直接回答、工具缺省与用量统计 | [check_direct_answer](../scripts/check_react.py) | 执行与工具调度 | `execution` |
| `check_react/check_unified_model_input` | 字符串与结构化输入实际 SDK 参数 | [check_unified_model_input](../scripts/check_react.py) | 执行与工具调度 | `execution` |
| `check_react/check_tool_loop` | 工具调用结果关联、回传及累计用量 | [check_tool_loop](../scripts/check_react.py) | 执行与工具调度 | `execution`, `recording` |
| `check_react/check_multiple_requests_and_context` | 同批串行执行及历史协议项完整保留 | [check_multiple_requests_and_context](../scripts/check_react.py) | 执行与工具调度 | `execution`, `context` |
| `check_react/check_limits` | 模型调用额度与末轮工具跳过 | [check_limits](../scripts/check_react.py) | 执行与工具调度 | `execution`, `recording` |
| `check_react/check_tool_errors` | 工具名称、参数及执行错误正确回传 | [check_tool_errors](../scripts/check_react.py) | 执行与工具调度 | `execution` |
| `check_react/check_model_failures` | 模型请求失败、不完整响应与无效协议 | [check_model_failures](../scripts/check_react.py) | 执行与工具调度 | `execution`, `recording` |
| `check_react/check_usage_gaps_and_run_isolation` | 用量缺失、真实零值与运行统计隔离 | [check_usage_gaps_and_run_isolation](../scripts/check_react.py) | 执行与工具调度 | `execution`, `recording` |
| `check_react/check_failure_history_and_final_display` | 失败历史后继续及 CLI 回答与工具事实显示 | [check_failure_history_and_final_display](../scripts/check_react.py) | 执行与工具调度 | `execution`, `recording` |
| `check_react/check_protocol_protection` | 拒绝伪造结果、重复调用标识及非法历史 | [check_protocol_protection](../scripts/check_react.py) | 执行与工具调度 | `execution`, `context` |
| `check_react/check_nested_and_concurrent_isolation` | 嵌套输入响应副本及并发运行隔离 | [check_nested_and_concurrent_isolation](../scripts/check_react.py) | 执行与工具调度 | `execution`, `context`, `recording` |
| `check_react/check_safe_summaries` | 大载荷的有界安全日志摘要 | [check_safe_summaries](../scripts/check_react.py) | 记录与追踪 | `recording`, `execution` |
| `check_react/check_safe_rejection_reasons` | 工具拒绝原因白名单与敏感信息过滤 | [check_safe_rejection_reasons](../scripts/check_react.py) | 记录与追踪 | `recording`, `execution` |
| `check_react/check_safe_execution_reasons` | 工具执行错误原因白名单与敏感信息过滤 | [check_safe_execution_reasons](../scripts/check_react.py) | 记录与追踪 | `recording`, `execution` |
| `check_react/check_tool_trace_hooks` | 工具追踪钩子及日志失败后的结果与收尾 | [check_tool_trace_hooks](../scripts/check_react.py) | 记录与追踪 | `recording`, `execution` |
| `check_react/check_recording_failures` | 记录创建、写入、刷新与关闭故障边界 | [check_recording_failures](../scripts/check_react.py) | 记录与追踪 | `recording`, `execution` |
| `check_react/check_mixed_batches_and_return_types` | 混合批次部分失败后继续及非字符串结果拒绝 | [check_mixed_batches_and_return_types](../scripts/check_react.py) | 执行与工具调度 | `execution`, `recording` |
| `check_react/check_duplicate_argument_keys` | 嵌套与转义重复参数键拒绝及合法相似键接受 | [check_duplicate_argument_keys](../scripts/check_react.py) | 执行与工具调度 | `execution`, `recording` |
| `check_react/check_failure_event_recording_failures` | 工具或模型失败后失败事件再次写入故障 | [check_failure_event_recording_failures](../scripts/check_react.py) | 记录与追踪 | `recording`, `execution` |
| `check_react/check_cli_exit_codes` | 构造 Agent 终态与配置错误的 CLI 退出语义 | [check_cli_exit_codes](../scripts/check_react.py) | 执行与工具调度 | `execution` |

### `filesystem-script`

主评测集：`control-filesystem`；入口：[scripts/check_filesystem.py](../scripts/check_filesystem.py)；计划 9 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_filesystem/check_listing` | 真实目录列举、排序与正文读取隔离 | [check_listing](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_filesystem/check_limits` | 有界目录输出、实际枚举及资源关闭 | [check_limits](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_filesystem/check_arguments` | 目录参数拒绝与同批后续请求继续 | [check_arguments](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_filesystem/check_paths` | 目录路径访问与根边界限制 | [check_paths](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_filesystem/check_io_failures` | 真实枚举与分类故障后的结果回传 | [check_io_failures](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_filesystem/check_links` | 真实符号链接与 junction 的列举和访问拒绝 | [check_links](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_filesystem/check_isolation` | 不同目录根并发运行、上下文与用量隔离 | [check_isolation](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_filesystem/check_configuration_and_cli` | 目录配置、CLI 工具启用及非法根提前拒绝 | [check_configuration_and_cli](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_filesystem/fixture_integrity` | 全部目录检查后输入资料与相邻目录保持不变 | [main](../scripts/check_filesystem.py) | 文件系统访问 | `filesystem` |

### `text-script`

主评测集：`control-filesystem`；入口：[scripts/check_text.py](../scripts/check_text.py)；计划 9 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_text/check_reading` | 发现后读取真实 UTF-8、BOM、换行及空文件 | [check_reading](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_text/check_limits` | 有界实际读取、短读、增长与文件描述符大小注入 | [check_limits](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_text/check_arguments` | 文本参数拒绝与后续请求继续 | [check_arguments](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_text/check_paths_and_formats` | 文本路径、文件类型、编码与控制字符限制 | [check_paths_and_formats](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_text/check_io_and_logging_failures` | 读取失败及失败事件写入再次故障的停止边界 | [check_io_and_logging_failures](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_text/check_links` | 真实文件与目录链接及 junction 的直接和祖先访问拒绝 | [check_links](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_text/check_untrusted_content` | 资料中的伪指令不能扩大目录与工具权限 | [check_untrusted_content](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution`, `context` |
| `check_text/check_isolation` | 不同资料根并发读取、输入副本及统计隔离 | [check_isolation](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_text/check_configuration_and_cli` | 文本额度配置及 CLI 列举后读取真实内容 | [check_configuration_and_cli](../scripts/check_text.py) | 文件系统访问 | `filesystem`, `execution` |

### `review-fixes-script`

主评测集：`control-execution`；入口：[scripts/check_review_fixes.py](../scripts/check_review_fixes.py)；计划 10 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_review_fixes/check_output_roles` | 无效输出角色与输入形状拒绝及合法扩展项保留 | [check_output_roles](../scripts/check_review_fixes.py) | 执行与工具调度 | `execution`, `context` |
| `check_review_fixes/check_refusal` | 纯拒绝、混合拒绝与工具调用及无效拒绝内容 | [check_refusal](../scripts/check_review_fixes.py) | 执行与工具调度 | `execution` |
| `check_review_fixes/check_direct_config` | 直接与 dotenv 配置边界及安全模型错误 | [check_direct_config](../scripts/check_review_fixes.py) | 执行与工具调度 | `execution`, `filesystem`, `recording` |
| `check_review_fixes/check_close_failure` | 客户端关闭故障保留结果、原错误与日志失败事实 | [check_close_failure](../scripts/check_review_fixes.py) | 执行与工具调度 | `execution`, `recording` |
| `check_review_fixes/check_file_scope` | 代码、配置与无扩展名普通文本读取和扣账 | [check_file_scope](../scripts/check_review_fixes.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_review_fixes/check_pagination` | 稳定排序分页、堆空间、真实结果回传及变化语义 | [check_pagination](../scripts/check_review_fixes.py) | 文件系统访问 | `filesystem`, `execution` |
| `check_review_fixes/check_read_budgets` | 跨轮读取累计额度、增长短读及成功后日志失败 | [check_read_budgets](../scripts/check_review_fixes.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_review_fixes/check_budget_isolation` | 复用执行器顺序与并发运行的读取额度隔离 | [check_budget_isolation](../scripts/check_review_fixes.py) | 文件系统访问 | `filesystem`, `execution`, `recording` |
| `check_review_fixes/check_review_tracing` | 额度与游标追踪投影、上传失败等价及拒绝和收尾脱敏 | [check_review_tracing](../scripts/check_review_fixes.py) | 记录与追踪 | `recording`, `filesystem`, `execution` |
| `check_review_fixes/check_final_answer_allowance` | 末次请求回答指令、工具禁用及用户额度文字原样保留 | [check_final_answer_allowance](../scripts/check_review_fixes.py) | 执行与工具调度 | `execution`, `context`, `recording` |

### `sessions-script`

主评测集：`control-sessions`；入口：[scripts/check_sessions.py](../scripts/check_sessions.py)；计划 13 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_sessions/check_history_and_copy_isolation` | 跨轮历史、返回值拷贝和检查点隔离 | [check_history_and_copy_isolation](../scripts/check_sessions.py) | 会话与保存 | `sessions` |
| `check_sessions/check_file_history_and_budget` | 真实文件历史继承与每轮读取额度重置 | [check_file_history_and_budget](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `filesystem`, `recording` |
| `check_sessions/check_failure_history_boundaries` | 模型、协议、工具失败后的历史准入及次轮重置 | [check_failure_history_boundaries](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `execution` |
| `check_sessions/check_partial_batch_and_usage` | 半批工具日志失败后的历史边界及缺失用量 | [check_partial_batch_and_usage](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `execution`, `recording` |
| `check_sessions/check_invalid_new_input` | 非法工具关联输入不发送且不继承 | [check_invalid_new_input](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `execution` |
| `check_sessions/check_concurrency` | 同会话重入拒绝与跨会话交错运行隔离 | [check_concurrency](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `filesystem` |
| `check_sessions/check_checkpoint_write_failures` | 六类真实检查点写入失败与会话阻塞 | [check_checkpoint_write_failures](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `execution`, `recording` |
| `check_sessions/check_checkpoint_and_skipped_log_failure` | 检查点失败叠加工具跳过日志失败保留事实 | [check_checkpoint_and_skipped_log_failure](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `execution`, `recording` |
| `check_sessions/check_checkpoint_read_failure` | 检查点读取失败保留原存储并阻塞会话 | [check_checkpoint_read_failure](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `recording` |
| `check_sessions/check_creation_and_start_log_failure` | 检查点创建失败与开始日志失败后的历史恢复 | [check_creation_and_start_log_failure](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `recording` |
| `check_sessions/check_finalization_combinations` | 最终保存、客户端关闭和结束日志组合失败 | [check_finalization_combinations](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `recording` |
| `check_sessions/check_tracing_sessions` | 会话追踪开关、过滤、上传故障及业务等价性 | [check_tracing_sessions](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `recording`, `filesystem` |
| `check_sessions/check_cli` | 完整同步 CLI 交互、会话切换、单次命令及保存失败恢复 | [check_cli](../scripts/check_sessions.py) | 会话与保存 | `sessions`, `execution` |

### `session-review-fixes-script`

主评测集：`control-sessions`；入口：[scripts/check_session_review_fixes.py](../scripts/check_session_review_fixes.py)；计划 12 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_session_review_fixes/check_cache_does_not_keep_frames` | 检查点失败缓存不保留异常栈和已关闭资源 | [check_cache_does_not_keep_frames](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `recording` |
| `check_session_review_fixes/check_budget_helper_boundary` | 读取额度观察异常与取消传播边界 | [check_budget_helper_boundary](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `filesystem`, `recording` |
| `check_session_review_fixes/check_metadata_stages_and_tracing` | 三阶段读取额度异常及日志追踪共享观察 | [check_metadata_stages_and_tracing](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `filesystem`, `recording` |
| `check_session_review_fixes/check_metadata_keeps_real_budget` | 元数据不可用时真实文件读取额度仍然生效 | [check_metadata_keeps_real_budget](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `filesystem`, `execution`, `recording` |
| `check_session_review_fixes/check_metadata_failure_combinations` | 额度元数据与日志、检查点组合失败保留结果 | [check_metadata_failure_combinations](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `filesystem`, `recording` |
| `check_session_review_fixes/check_invalid_input_history` | 非法新输入拒绝且不污染可继承历史 | [check_invalid_input_history](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `execution` |
| `check_session_review_fixes/check_valid_protocol_and_http_history` | 合法协议扩展历史与 HTTP 错误准入边界 | [check_valid_protocol_and_http_history](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `execution` |
| `check_session_review_fixes/check_preexecution_config_retry` | 执行前配置失败保留历史并允许重试 | [check_preexecution_config_retry](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `execution`, `recording` |
| `check_session_review_fixes/check_execution_started_still_blocks` | 客户端进入失败后会话阻塞及资源关闭 | [check_execution_started_still_blocks](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `execution`, `recording` |
| `check_session_review_fixes/check_finally_and_primary_error` | 收尾取消传播、资源关闭及主错误保留 | [check_finally_and_primary_error](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `recording` |
| `check_session_review_fixes/check_prepare_and_final_save_secondary_errors` | 准备和最终保存错误不被次级追踪异常覆盖 | [check_prepare_and_final_save_secondary_errors](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `recording` |
| `check_session_review_fixes/check_cli_preconfiguration_retry` | 同步 CLI 配置失败重试及错误退出语义 | [check_cli_preconfiguration_retry](../scripts/check_session_review_fixes.py) | 会话与保存 | `sessions`, `execution` |

### `context-budget-script`

主评测集：`control-context`；入口：[scripts/check_context_budget.py](../scripts/check_context_budget.py)；计划 14 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_context_budget/check_configuration` | 上下文预算默认值、配置校验及显式环境加载 | [check_configuration](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context` |
| `check_context_budget/check_transparent_request_and_boundary` | 实际请求透明计量与精确预算阈值 | [check_transparent_request_and_boundary](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `execution` |
| `check_context_budget/check_stable_history_and_rejected_input` | 超预算新输入拒绝后的历史稳定性 | [check_stable_history_and_rejected_input](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `sessions` |
| `check_context_budget/check_tool_result_limit_and_real_file` | 真实文件结果触发下一请求预算并保留事实 | [check_tool_result_limit_and_real_file](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `filesystem`, `sessions`, `execution` |
| `check_context_budget/check_opaque_counting` | 不透明内容风险计量、来源用量与后备值 | [check_opaque_counting](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context` |
| `check_context_budget/check_opaque_provenance_and_isolation` | 不透明风险来源跨轮保留及会话隔离 | [check_opaque_provenance_and_isolation](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `sessions` |
| `check_context_budget/check_file_sessions_isolation` | 文件根、预算、历史及新轮额度共同隔离 | [check_file_sessions_isolation](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `filesystem`, `sessions` |
| `check_context_budget/check_every_call_and_usage` | 每次实际请求先计量及部分用量累计 | [check_every_call_and_usage](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `execution`, `recording` |
| `check_context_budget/check_measurement_failure` | 不可计量输入与请求准备失败阻止发送 | [check_measurement_failure](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `execution` |
| `check_context_budget/check_session_invalid_storage_boundaries` | 实际检查点序列化与请求计量失败边界 | [check_session_invalid_storage_boundaries](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `sessions`, `recording` |
| `check_context_budget/check_budget_recording_failures` | 预算日志失败及已执行工具事实保留 | [check_budget_recording_failures](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `execution`, `recording` |
| `check_context_budget/check_budget_checkpoint_failures` | 预算停止和工具完成后的检查点保存失败 | [check_budget_checkpoint_failures](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `sessions`, `execution`, `recording` |
| `check_context_budget/check_tracing_diagnostics` | 预算追踪诊断过滤及上传故障等价性 | [check_tracing_diagnostics](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `recording` |
| `check_context_budget/check_cli` | 预算停止展示与交互拒绝后继续 | [check_cli](../scripts/check_context_budget.py) | 上下文预算与摘要 | `context`, `sessions`, `execution` |

### `compaction-script`

主评测集：`control-context`；入口：[scripts/check_compaction.py](../scripts/check_compaction.py)；计划 19 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_compaction/check_configuration` | 摘要和历史保留配置边界 | [check_configuration](../scripts/check_compaction.py) | 上下文预算与摘要 | `context` |
| `check_compaction/check_stable_history_and_records` | 未压缩活动版本与运行记录拷贝隔离 | [check_stable_history_and_records](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions` |
| `check_compaction/check_selection_read_path` | 轻量历史选择与完整记录等价及源读取失败 | [check_selection_read_path](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions`, `recording` |
| `check_compaction/check_compaction_status_consumers` | 压缩状态的 CLI 展示与追踪投影 | [check_compaction_status_consumers](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `recording` |
| `check_compaction/check_activation_and_stable_reuse` | 摘要保存确认、活动版本切换与后续复用 | [check_activation_and_stable_reuse](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions`, `recording` |
| `check_compaction/check_request_options_through_compaction` | 请求选项贯穿选择、候选、真实保存和 SDK | [check_request_options_through_compaction](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions`, `execution` |
| `check_compaction/check_refusal_boundaries` | 无可压缩历史或当前请求过大时保持旧版本 | [check_refusal_boundaries](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions` |
| `check_compaction/check_summary_failure_cases` | 摘要服务、协议、来源及大小失败停止业务请求 | [check_summary_failure_cases](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `execution` |
| `check_compaction/check_usage_missing` | 摘要与业务用量缺失分计及新轮重置 | [check_usage_missing](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `recording`, `sessions` |
| `check_compaction/check_summary_and_candidate_budget_boundaries` | 摘要请求及候选预算和准备失败边界 | [check_summary_and_candidate_budget_boundaries](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `execution` |
| `check_compaction/check_opaque_barriers_and_omissions` | 受保护历史阻止压缩及加密内容省略 | [check_opaque_barriers_and_omissions](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions` |
| `check_compaction/check_real_file_and_failure_source` | 真实文件及工具失败事实进入摘要来源 | [check_real_file_and_failure_source](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `filesystem`, `execution` |
| `check_compaction/check_repeated_compaction` | 同轮两次恢复的来源去重和工具结果保留 | [check_repeated_compaction](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions`, `execution` |
| `check_compaction/check_unattempted_recovery_record` | 未尝试恢复与已调用摘要记录保持区分 | [check_unattempted_recovery_record](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `recording` |
| `check_compaction/check_session_isolation` | 并发摘要、来源、版本及记录查询会话隔离 | [check_session_isolation](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions` |
| `check_compaction/check_runtime_tracing` | 摘要正文过滤、追踪开关和上传失败业务等价性 | [check_runtime_tracing](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `recording` |
| `check_compaction/check_history_admission_and_cli` | 失败历史摘要准入及 CLI 压缩后连续交互 | [check_history_admission_and_cli](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions`, `execution` |
| `check_compaction/check_checkpoint_boundaries` | 候选、激活、最终保存及工具后恢复保存失败 | [check_checkpoint_boundaries](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `sessions`, `execution`, `recording` |
| `check_compaction/check_log_boundaries` | 摘要和版本日志失败及模型失败的组合边界 | [check_log_boundaries](../scripts/check_compaction.py) | 上下文预算与摘要 | `context`, `execution`, `recording` |

### `tracing-script`

主评测集：`control-recording`；入口：[scripts/check_tracing.py](../scripts/check_tracing.py)；计划 17 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_tracing/check_projection` | 字段投影、密钥及路径过滤、原对象不变与载荷上限 | [check_projection](../scripts/check_tracing.py) | 记录与追踪 | `recording` |
| `check_trace_data/check_redaction_boundaries` | 引号、转义、跨文本及截断边界的凭据过滤与幂等性 | [check_trace_data.py:check_redaction_boundaries](../scripts/check_trace_data.py) | 记录与追踪 | `recording` |
| `check_trace_data/check_capacity_and_rounds` | 容量与遗漏元数据、完整工具批次保留、回答和用量投影 | [check_trace_data.py:check_capacity_and_rounds](../scripts/check_trace_data.py) | 记录与追踪 | `recording`, `execution` |
| `check_trace_usage/check_usage_categories` | 缺失、无效、零值和部分用量的覆盖分类 | [check_trace_usage.py:check_usage_categories](../scripts/check_trace_usage.py) | 记录与追踪 | `recording` |
| `check_trace_usage/check_mixed_run_and_empty_run` | 多次调用混合用量覆盖及空运行 | [check_trace_usage.py:check_mixed_run_and_empty_run](../scripts/check_trace_usage.py) | 记录与追踪 | `recording` |
| `check_trace_usage/check_remote_mismatches` | 拒绝本地与远端最终用量投影不一致 | [check_trace_usage.py:check_remote_mismatches](../scripts/check_trace_usage.py) | 记录与追踪 | `recording` |
| `check_trace_usage/check_final_projection_only` | 最终投影缺失用量时不使用原始用量补判覆盖 | [check_trace_usage.py:check_final_projection_only](../scripts/check_trace_usage.py) | 记录与追踪 | `recording` |
| `check_tracing/check_serialized_redaction` | 真实 SDK 序列化后重组过滤凭据，业务对象与请求不变 | [check_serialized_redaction](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |
| `check_tracing/check_serialized_capacity` | 超大输入的上传限额不丢失小回答及实际投影用量 | [check_serialized_capacity](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `context` |
| `check_tracing/check_disabled_and_equivalence` | 追踪开关的业务等价、混合工具结果、调用关联与统计 | [check_disabled_and_equivalence](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution`, `filesystem` |
| `check_tracing/check_failures_and_shutdown` | 上传、配置、初始化及过滤失败的隔离、不重试和有限收尾 | [check_failures_and_shutdown](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |
| `check_tracing/check_concurrency_and_skips` | 并发运行隔离、模型额度导致的工具跳过及关联 | [check_concurrency_and_skips](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |
| `check_tracing/check_local_recording_failure` | 本地记录失败后工具跳过事实与追踪保持一致 | [check_local_recording_failure](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |
| `check_tracing/check_late_protocol_failure` | 后续重复调用标识的协议错误只标记对应模型调用 | [check_late_protocol_failure](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |
| `check_tracing/check_missing_usage_and_model_error` | 缺失用量和模型异常的安全追踪投影 | [check_missing_usage_and_model_error](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |
| `check_tracing/check_cancellation_and_queue_overflow` | 模型与工具取消、资源关闭及有界上传队列 | [check_cancellation_and_queue_overflow](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |
| `check_tracing/check_cli_exit_deadline` | 真实 CLI 子进程在上传卡住时仍有限退出 | [check_cli_exit_deadline](../scripts/check_tracing.py) | 记录与追踪 | `recording`, `execution` |

### `text-tracing-script`

主评测集：`control-recording`；入口：[scripts/check_text_tracing.py](../scripts/check_text_tracing.py)；计划 5 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_text_tracing/check_real_read_and_equivalence` | 真实文件读取在追踪开关及五类上传失败下保持业务等价 | [check_real_read_and_equivalence](../scripts/check_text_tracing.py) | 记录与追踪 | `recording`, `filesystem` |
| `check_text_tracing/check_listing_and_model_quotation` | 目录和 BOM 读取元数据隐藏正文，同时保留模型对资料的引用 | [check_listing_and_model_quotation](../scripts/check_text_tracing.py) | 记录与追踪 | `recording`, `filesystem`, `execution` |
| `check_text_tracing/check_failure_reasons` | 同批读取拒绝、限制、权限、中途 I/O 与成功结果在模型、JSONL 和追踪中一致 | [check_failure_reasons](../scripts/check_text_tracing.py) | 记录与追踪 | `recording`, `filesystem`, `execution` |
| `check_text_tracing/check_projection_boundaries` | 文本结果容量、原对象隔离和不可信原因及元数据过滤 | [check_projection_boundaries](../scripts/check_text_tracing.py) | 记录与追踪 | `recording`, `filesystem` |
| `check_text_tracing/fixture_integrity` | 全部检查结束后文件内容、目录成员及当前目录保持不变 | [main:fixture_integrity](../scripts/check_text_tracing.py) | 记录与追踪 | `recording`, `filesystem` |

### `session-tracing-script`

主评测集：`control-recording`；入口：[scripts/check_session_tracing.py](../scripts/check_session_tracing.py)；计划 6 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_session_tracing/checkpoint_trace_equivalence` | 检查点追踪 harness 在关闭、开启及上传失败下请求和结果等价 | [main:checkpoint_trace_equivalence](../scripts/check_session_tracing.py) | 记录与追踪 | `recording`, `sessions` |
| `check_session_tracing/checkpoint_error_metadata` | 三类检查点错误的追踪元数据及安全投影，与正常 harness 业务结果等价 | [main:checkpoint_error_metadata](../scripts/check_session_tracing.py) | 记录与追踪 | `recording`, `sessions` |
| `check_session_tracing/session_history_trace_equivalence` | 真实多轮会话、文件历史与检查点在追踪开关及上传失败下保持等价 | [main:session_history_trace_equivalence](../scripts/check_session_tracing.py) | 记录与追踪 | `recording`, `sessions`, `filesystem` |
| `check_session_tracing/session_final_save_failure` | 第二轮最终保存失败的追踪归属与已取得业务结果保留 | [main:session_final_save_failure](../scripts/check_session_tracing.py) | 记录与追踪 | `recording`, `sessions` |
| `check_session_tracing/check_skipped_after_save_failure` | 模型结果保存失败后停止工具执行，保留用量、跳过及关联事实 | [check_skipped_after_save_failure](../scripts/check_session_tracing.py) | 记录与追踪 | `recording`, `sessions`, `execution`, `filesystem` |
| `check_session_tracing/fixture_integrity` | 全部会话检查结束后实际来源文件保持不变 | [main:fixture_integrity](../scripts/check_session_tracing.py) | 记录与追踪 | `recording`, `sessions`, `filesystem` |

### `compaction-tracing-script`

主评测集：`control-recording`；入口：[scripts/check_compaction_tracing.py](../scripts/check_compaction_tracing.py)；计划 4 组。

| 稳定组 ID | 行为 | 原 source symbol | 主要归属 | 覆盖标签 |
|---|---|---|---|---|
| `check_compaction_tracing/summary_trace_equivalence` | 摘要追踪 harness 的正常、关闭及上传失败保持请求结果等价，摘要正文与派生输入被省略 | [main:summary_trace_equivalence](../scripts/check_compaction_tracing.py) | 记录与追踪 | `recording`, `context` |
| `check_compaction_tracing/summary_tool_request` | 摘要意外工具调用的失败投影保持正文省略 | [main:summary_tool_request](../scripts/check_compaction_tracing.py) | 记录与追踪 | `recording`, `context` |
| `check_compaction_tracing/summary_incomplete` | 未完成摘要的追踪仍不暴露摘要正文 | [main:summary_incomplete](../scripts/check_compaction_tracing.py) | 记录与追踪 | `recording`, `context` |
| `check_compaction_tracing/bounded_lineage_provenance` | 摘要来源引用有界且只依据明确来源隐藏派生内容，不按文本前缀猜测 | [main:bounded_lineage_provenance](../scripts/check_compaction_tracing.py) | 记录与追踪 | `recording`, `context` |
