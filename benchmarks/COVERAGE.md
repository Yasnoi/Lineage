# 工程约定与迁移覆盖

本文面向维护确定性检查的开发者。工程行为组的可执行清单为 [contracts.json](suites/agent_contracts/contracts.json)，模块按被测约定划分，不代表不同项目评测集。全量入口为 `python -m benchmarks run agent-contracts`。当前 17 条隔离行为链包含 184 个行为组；实际是否执行及结果以运行报告为准。`benchmarks/tests/` 另有可通过 `unittest` 执行的框架和专项检查，部分与工程行为组复用同一场景，不重复累加为覆盖数量。

## 模块与判断依据

| 模块 | 行为组 | 约定与指标 |
| --- | ---: | --- |
| `execution` | 19 | 输入/响应协议、批次继续、结果关联、调用额度和终态统计；直接比较实际请求、工具返回和 State。 |
| `filesystem` | 36 | 目录与文本路径边界、读取额度、真实 I/O，以及 Artifact 快照、有界召回与会话隔离；核对实际文件和累计预算。 |
| `sessions` | 25 | 历史准入、隔离、重置、并发拒绝、保存和关闭；核对原生检查点及后续请求。 |
| `context` | 65 | 请求预算、风险计量、H/R 选源、三组保护、原始记录回读、保存屏障和活动版本；对照实际请求参数与记录。 |
| `recording` | 39 | JSONL、过滤、usage 投影、序列化、上传异常和有限收尾；对照实际 SDK 载荷与故障观察。 |

一次跨模块场景保留完整执行链，只按主要模块计一次；标签不作为根因结论。响应耗时、真实 token 与缓存字段由真实任务轨迹另外观测，替身 token 不算成本。没有统一总分。

## 迁移与等价性

来源基线为 `c97467ed41bcd3a0232033f9160d15199b30069b`。14 个原 `scripts/check_*.py` 已移除，其检查函数和实际公共替身归入 Benchmark。ReAct 的执行检查与记录检查分开；Review 的输入协议、文件系统预算、追踪投影按职责分开，未拆断单个故障链。Session 两处异步入口后的 CLI 检查、fixture 收尾和五次摘要 Tracing exercise 均保留。data/usage 组经 recording.transport 调用，不另建隔离单元或重复计数。

此前脚本迁移核对中，基线 1,444 条 assert 有 1,443 条业务/边界断言的 AST 保持不变。原 Review `assert not failures` 是独立脚本聚合器收尾，现由组记录及失败停止策略承担。远端 usage 对比公共函数从 `verify_tracing.py` 提取，其一条既有断言归入公共代码，供诊断脚本和工程检查共同复用。该历史核对覆盖当时的 138 组，不能解释为当前改造仍逐条保持 AST 不变，断言数量也不是覆盖率。后续动态恢复、Artifact 和观测治理扩展至 155 组；本次将旧 Compact 的 22 组替换为当前前缀、H/R、原始记录及保存边界的 51 组，合计 184 组。当前证据以实际事件、请求、状态及故障观察为准。

Artifact 检查验证完整原文保存、正文字符位置、精确 UTF-8 响应上限、关键词分页、会话及缓存路径隔离、关闭与故障行为，并用离线模型替身走通两轮工具调用及记录过滤。观测治理检查另行验证自动归档、结构化提示、默认召回工具和治理故障；两类检查均不以真实模型的上下文下降或回答质量作为结论。

框架回归保留选择拒绝、状态分离、回执一致性、事件有效前缀、资源收尾、保存故障、会话/材料隔离、预算与评分绑定。旧 v1 worker 和 control-* 别名专用执行测试随接口撤销，其仍适用的断言/启动/源码/优化保护/保存等边界由组 worker 与通用框架测试承接；旧结果读取测试独立保留。新增第二个项目注册、构造材料、LLM 建议绑定与离线网络边界测试。

## Compact 与历史召回的检查来源

当前 Run 前缀恢复及原始记录回读使用下列 `unittest` 场景，入口为 `python -m unittest discover -s benchmarks/tests -v`。其中前缀 21 组、保存与请求边界 6 组、原始记录 13 组、配置 3 组和摘要协议 8 组也由 `context.compaction` 调用，共 51 个工程组；两种入口复用场景，不能视为两份独立覆盖。其余传输、追踪和 Benchmark 检查单独执行。

| 检查来源 | 覆盖范围 |
| --- | --- |
| [test_prefix_compaction.py](tests/test_prefix_compaction.py) | 独立 Run、会话第一轮、H/R 整体启用与失败隔离、当前三组保护和历史全量选择、重复压缩、原始调用身份、最多两次尝试及四次摘要调用、源版本和保存故障、压缩后搜索并读取遗漏细节。 |
| [test_compaction_boundaries.py](tests/test_compaction_boundaries.py) | 完整请求检查与实际 SDK 参数一致、低于旧空闲门槛但预算放行且有净回收仍提交、独立 Run 最终保存失败、激活确认与日志故障，以及短任务无摘要调用。 |
| [test_history.py](tests/test_history.py) | 不可变原始记录、固定快照、缺失与冲突身份、未被摘要引用的记录、会话隔离、Unicode 与转义后的 8 KiB 分页、查询绑定游标、原生召回不重复归档，以及执行失败和中断事实。 |
| [test_compact_settings.py](tests/test_compact_settings.py)、[test_summary_contracts.py](tests/test_summary_contracts.py)、[test_compact_transport.py](tests/test_compact_transport.py) | 当前三组与分路输出上限、旧软目标和余量配置明确迁移拒绝、普通文字响应与最小校验、H/R 实际发送和预算上限一致、完整协议组及已知加密 reasoning 边界。 |
| [test_compact_tracing.py](tests/test_compact_tracing.py) | H/R 和旧 R 按可信摘要位置过滤、普通伪造标记不触发过滤、元数据截断不遗漏正文过滤、两路指标与 CLI 阶段。 |
| [test_compact_evidence.py](tests/test_compact_evidence.py)、[test_compact_entrypoint.py](tests/test_compact_entrypoint.py) | Benchmark 历史召回证据、存储指标与请求开销、统一运行入口和配置限制。 |

新增文件持久化与第二阶段规则由以下专项测试补充，仍通过上述 unittest 入口执行，不重复计入 184 个工程组：

- `test_record_journal.py`、`test_persistent_storage.py`：JSONL 受限编码、完整批次写入、确认边界、索引重建、短写及 fsync 故障、关闭后保留文件、Artifact 多绑定和 `.lineage` 路径隔离。
- `test_record_runtime.py`：初始化、模型／工具返回后写入、终态写入和最终检查点确认故障；保留真实执行事实、不重跑工具、不发布未确认历史。
- `test_context_stage2.py`：正常结束即登记历史、当前 0～3 组独立保护、旧轮次全部进入 H、无重复登记与 R 背景限制。
- `test_native_context_material.py`：确认快照和文件原文一致、原生消息顺序与 opaque 风险继承、已有摘要原位正文替换、来源范围继承、多项原始背景及对象隔离；历史工具可读投影保持原样。

摘要传输检查按原生协议对象核对预算请求与 HTTP 请求，包括含 JSON、Unicode、换行的结果字符串和原生 reasoning；不再以旧包装的解包还原作为生产验收。Benchmark 通过仅程序侧来源信息、确认文件水位和实际请求联合取证；旧摘要包装仅由 `legacy_summary.py` 读取历史证据，不能用于新请求回退。

`recording.compaction` 继续验证追踪关闭、开启和上传失败时的业务等价性；两路摘要正文实际进入业务请求，但 H、R 和历史轮次的 R 都按程序维护的 `summary_spans` 隐去派生正文。用于过滤的完整索引与上传的有界来源元数据分别处理，普通用户消息中的仿造标记不能改变过滤范围。

这些确定性检查证明控制流、协议和保存约定，不能证明摘要事实准确。真实 `defect-reference`、`history-recall` 和 `compaction-continuation` 的任务完成、条件保留、失败状态及来源支持需单独评审，来源编号合法不替代语义核查。

`test_summary_contracts.py` 另验证 H/R 与第二次缩短指令的独立路由，原生输入不变且用途识别仍要求完整程序策略。`test_summary_prompt_comparison.py` 的十项独立框架检查验证固定输入 A/B 的 HTTP 一致性、旧摘要和 opaque 风险不变、预算拒绝零发送、实际失败不重试、认证及证据故障停止、旧账本不能重启，以及提示词占用计量。这些检查由 unittest 入口执行，不增加工程行为链或用关键词替代摘要语义评审。

## 逐组对应关系

下表由固定清单整理。原入口列指基线提交中的文件，不是保留的执行路径；当前文件的 check_* 函数承接原断言，main/execute 保留顺序、异步和清理作用域。函数内可以包含多个 Agent 运行。

| 当前行为组 | 主模块 | 行为与标签 | 原入口 / 检查 | 当前来源 |
| --- | --- | --- | --- | --- |
| `execution.react.direct_answer` | execution | 直接回答、工具缺省与用量统计；execution | `check_react.py:check_direct_answer` | [check_direct_answer](checks/execution/react.py) |
| `execution.react.unified_model_input` | execution | 字符串与结构化输入实际 SDK 参数；execution | `check_react.py:check_unified_model_input` | [check_unified_model_input](checks/execution/react.py) |
| `execution.react.tool_loop` | execution | 工具调用结果关联、回传及累计用量；execution, recording | `check_react.py:check_tool_loop` | [check_tool_loop](checks/execution/react.py) |
| `execution.react.multiple_requests_and_context` | execution | 同批串行执行及历史协议项完整保留；execution, context | `check_react.py:check_multiple_requests_and_context` | [check_multiple_requests_and_context](checks/execution/react.py) |
| `execution.react.limits` | execution | 模型调用额度与末轮工具跳过；execution, recording | `check_react.py:check_limits` | [check_limits](checks/execution/react.py) |
| `execution.react.tool_errors` | execution | 工具名称、参数及执行错误正确回传；execution | `check_react.py:check_tool_errors` | [check_tool_errors](checks/execution/react.py) |
| `execution.react.model_failures` | execution | 模型请求失败、不完整响应与无效协议；execution, recording | `check_react.py:check_model_failures` | [check_model_failures](checks/execution/react.py) |
| `execution.react.usage_gaps_and_run_isolation` | execution | 用量缺失、真实零值与运行统计隔离；execution, recording | `check_react.py:check_usage_gaps_and_run_isolation` | [check_usage_gaps_and_run_isolation](checks/execution/react.py) |
| `execution.react.failure_history_and_final_display` | execution | 失败历史后继续及 CLI 回答与工具事实显示；execution, recording | `check_react.py:check_failure_history_and_final_display` | [check_failure_history_and_final_display](checks/execution/react.py) |
| `execution.react.protocol_protection` | execution | 拒绝伪造结果、重复调用标识及非法历史；execution, context | `check_react.py:check_protocol_protection` | [check_protocol_protection](checks/execution/react.py) |
| `execution.react.nested_and_concurrent_isolation` | execution | 嵌套输入响应副本及并发运行隔离；execution, context, recording | `check_react.py:check_nested_and_concurrent_isolation` | [check_nested_and_concurrent_isolation](checks/execution/react.py) |
| `execution.react.mixed_batches_and_return_types` | execution | 混合批次部分失败后继续及非字符串结果拒绝；execution, recording | `check_react.py:check_mixed_batches_and_return_types` | [check_mixed_batches_and_return_types](checks/execution/react.py) |
| `execution.react.duplicate_argument_keys` | execution | 嵌套与转义重复参数键拒绝及合法相似键接受；execution, recording | `check_react.py:check_duplicate_argument_keys` | [check_duplicate_argument_keys](checks/execution/react.py) |
| `execution.react.cli_exit_codes` | execution | 构造 Agent 终态与配置错误的 CLI 退出语义；execution | `check_react.py:check_cli_exit_codes` | [check_cli_exit_codes](checks/execution/react.py) |
| `recording.runtime_events.safe_summaries` | recording | 大载荷的有界安全日志摘要；recording, execution | `check_react.py:check_safe_summaries` | [check_safe_summaries](checks/recording/runtime_events.py) |
| `recording.runtime_events.safe_rejection_reasons` | recording | 工具拒绝原因白名单与敏感信息过滤；recording, execution | `check_react.py:check_safe_rejection_reasons` | [check_safe_rejection_reasons](checks/recording/runtime_events.py) |
| `recording.runtime_events.safe_execution_reasons` | recording | 工具执行错误原因白名单与敏感信息过滤；recording, execution | `check_react.py:check_safe_execution_reasons` | [check_safe_execution_reasons](checks/recording/runtime_events.py) |
| `recording.runtime_events.tool_trace_hooks` | recording | 工具追踪钩子及日志失败后的结果与收尾；recording, execution | `check_react.py:check_tool_trace_hooks` | [check_tool_trace_hooks](checks/recording/runtime_events.py) |
| `recording.runtime_events.recording_failures` | recording | 记录创建、写入、刷新与关闭故障边界；recording, execution | `check_react.py:check_recording_failures` | [check_recording_failures](checks/recording/runtime_events.py) |
| `recording.runtime_events.failure_event_recording_failures` | recording | 工具或模型失败后失败事件再次写入故障；recording, execution | `check_react.py:check_failure_event_recording_failures` | [check_failure_event_recording_failures](checks/recording/runtime_events.py) |
| `filesystem.directory.listing` | filesystem | 真实目录列举、排序与正文读取隔离；filesystem, execution, recording | `check_filesystem.py:check_listing` | [check_listing](checks/filesystem/directory.py) |
| `filesystem.directory.limits` | filesystem | 有界目录输出、实际枚举及资源关闭；filesystem, execution | `check_filesystem.py:check_limits` | [check_limits](checks/filesystem/directory.py) |
| `filesystem.directory.arguments` | filesystem | 目录参数拒绝与同批后续请求继续；filesystem, execution | `check_filesystem.py:check_arguments` | [check_arguments](checks/filesystem/directory.py) |
| `filesystem.directory.paths` | filesystem | 目录路径访问与根边界限制；filesystem, execution | `check_filesystem.py:check_paths` | [check_paths](checks/filesystem/directory.py) |
| `filesystem.directory.io_failures` | filesystem | 真实枚举与分类故障后的结果回传；filesystem, execution, recording | `check_filesystem.py:check_io_failures` | [check_io_failures](checks/filesystem/directory.py) |
| `filesystem.directory.links` | filesystem | 真实符号链接与 junction 的列举和访问拒绝；filesystem, execution | `check_filesystem.py:check_links` | [check_links](checks/filesystem/directory.py) |
| `filesystem.directory.isolation` | filesystem | 不同目录根并发运行、上下文与用量隔离；filesystem, execution, recording | `check_filesystem.py:check_isolation` | [check_isolation](checks/filesystem/directory.py) |
| `filesystem.directory.configuration_and_cli` | filesystem | 目录配置、CLI 工具启用及非法根提前拒绝；filesystem, execution | `check_filesystem.py:check_configuration_and_cli` | [check_configuration_and_cli](checks/filesystem/directory.py) |
| `filesystem.directory.fixture_integrity` | filesystem | 全部目录检查后输入资料与相邻目录保持不变；filesystem | `check_filesystem.py:main` | [main](checks/filesystem/directory.py) |
| `filesystem.text.reading` | filesystem | 发现后读取真实 UTF-8、BOM、换行及空文件；filesystem, execution, recording | `check_text.py:check_reading` | [check_reading](checks/filesystem/text.py) |
| `filesystem.text.limits` | filesystem | 有界实际读取、短读、增长与文件描述符大小注入；filesystem, execution | `check_text.py:check_limits` | [check_limits](checks/filesystem/text.py) |
| `filesystem.text.arguments` | filesystem | 文本参数拒绝与后续请求继续；filesystem, execution | `check_text.py:check_arguments` | [check_arguments](checks/filesystem/text.py) |
| `filesystem.text.paths_and_formats` | filesystem | 文本路径、文件类型、编码与控制字符限制；filesystem, execution | `check_text.py:check_paths_and_formats` | [check_paths_and_formats](checks/filesystem/text.py) |
| `filesystem.text.io_and_logging_failures` | filesystem | 读取失败及失败事件写入再次故障的停止边界；filesystem, execution, recording | `check_text.py:check_io_and_logging_failures` | [check_io_and_logging_failures](checks/filesystem/text.py) |
| `filesystem.text.links` | filesystem | 真实文件与目录链接及 junction 的直接和祖先访问拒绝；filesystem, execution | `check_text.py:check_links` | [check_links](checks/filesystem/text.py) |
| `filesystem.text.untrusted_content` | filesystem | 资料中的伪指令不能扩大目录与工具权限；filesystem, execution, context | `check_text.py:check_untrusted_content` | [check_untrusted_content](checks/filesystem/text.py) |
| `filesystem.text.isolation` | filesystem | 不同资料根并发读取、输入副本及统计隔离；filesystem, execution, recording | `check_text.py:check_isolation` | [check_isolation](checks/filesystem/text.py) |
| `filesystem.text.configuration_and_cli` | filesystem | 文本额度配置及 CLI 列举后读取真实内容；filesystem, execution | `check_text.py:check_configuration_and_cli` | [check_configuration_and_cli](checks/filesystem/text.py) |
| `execution.input_boundaries.output_roles` | execution | 无效输出角色与输入形状拒绝及合法扩展项保留；execution, context | `check_review_fixes.py:check_output_roles` | [check_output_roles](checks/execution/input_boundaries.py) |
| `execution.input_boundaries.refusal` | execution | 纯拒绝、混合拒绝与工具调用及无效拒绝内容；execution | `check_review_fixes.py:check_refusal` | [check_refusal](checks/execution/input_boundaries.py) |
| `execution.input_boundaries.direct_config` | execution | 直接与 dotenv 配置边界及安全模型错误；execution, filesystem, recording | `check_review_fixes.py:check_direct_config` | [check_direct_config](checks/execution/input_boundaries.py) |
| `execution.input_boundaries.close_failure` | execution | 客户端关闭故障保留结果、原错误与日志失败事实；execution, recording | `check_review_fixes.py:check_close_failure` | [check_close_failure](checks/execution/input_boundaries.py) |
| `execution.input_boundaries.final_answer_allowance` | execution | 末次请求回答指令、工具禁用及用户额度文字原样保留；execution, context, recording | `check_review_fixes.py:check_final_answer_allowance` | [check_final_answer_allowance](checks/execution/input_boundaries.py) |
| `filesystem.access_budget.file_scope` | filesystem | 代码、配置与无扩展名普通文本读取和扣账；filesystem, execution, recording | `check_review_fixes.py:check_file_scope` | [check_file_scope](checks/filesystem/access_budget.py) |
| `filesystem.access_budget.pagination` | filesystem | 稳定排序分页、堆空间、真实结果回传及变化语义；filesystem, execution | `check_review_fixes.py:check_pagination` | [check_pagination](checks/filesystem/access_budget.py) |
| `filesystem.access_budget.read_budgets` | filesystem | 跨轮读取累计额度、增长短读及成功后日志失败；filesystem, execution, recording | `check_review_fixes.py:check_read_budgets` | [check_read_budgets](checks/filesystem/access_budget.py) |
| `filesystem.access_budget.budget_isolation` | filesystem | 复用执行器顺序与并发运行的读取额度隔离；filesystem, execution, recording | `check_review_fixes.py:check_budget_isolation` | [check_budget_isolation](checks/filesystem/access_budget.py) |
| `filesystem.artifacts.saving` | filesystem | 原始字符串完整保存、已知正文解码与受控降级；filesystem | 新增约定 | [check_saving](checks/filesystem/artifacts.py) |
| `filesystem.artifacts.reading` | filesystem | 字符范围、有界序列化与超长单行连续读取；filesystem, execution | 新增约定 | [check_reading](checks/filesystem/artifacts.py) |
| `filesystem.artifacts.searching` | filesystem | 区分大小写的字面 OR、重叠命中与有界分页；filesystem, execution | 新增约定 | [check_searching](checks/filesystem/artifacts.py) |
| `filesystem.artifacts.isolation` | filesystem | 会话与 Store 隔离、非法引用及文件工具旁路拒绝；filesystem, sessions | 新增约定 | [check_isolation](checks/filesystem/artifacts.py) |
| `filesystem.artifacts.lifecycle` | filesystem | 显式关闭、幂等清理与其他 Store 保留；filesystem, sessions | 新增约定 | [check_lifecycle](checks/filesystem/artifacts.py) |
| `filesystem.artifacts.faults` | filesystem | 写入与读取故障、完整性失败和安全诊断；filesystem, execution | 新增约定 | [check_faults](checks/filesystem/artifacts.py) |
| `filesystem.artifacts.integration` | filesystem | 同会话跨 Run 召回、默认工具注册与业务工具兼容；filesystem, sessions, execution | 新增约定 | [check_integration](checks/filesystem/artifacts.py) |
| `filesystem.artifacts.recording` | filesystem | 有界正文仅进模型上下文、日志及 Tracing 过滤；filesystem, recording | 新增约定 | [check_recording](checks/filesystem/artifacts.py) |
| `recording.runtime_projection.review_tracing` | recording | 额度与游标追踪投影、上传失败等价及拒绝和收尾脱敏；recording, filesystem, execution | `check_review_fixes.py:check_review_tracing` | [check_review_tracing](checks/recording/runtime_projection.py) |
| `filesystem.observations.pipeline` | filesystem | 统一观测入口、小结果内联、大结果自动保存与有界召回；filesystem, execution, context | 新增约定 | [check_pipeline](checks/filesystem/observations.py) |
| `filesystem.observations.hints` | filesystem | 结构化观测提示、参数拆分及关键词预览降级；filesystem, execution, context | 新增约定 | [check_hints](checks/filesystem/observations.py) |
| `filesystem.observations.failures` | filesystem | 保存与治理失败安全返回，保留工具执行事实；filesystem, execution, context | 新增约定 | [check_failures](checks/filesystem/observations.py) |
| `filesystem.observations.sessions` | filesystem | 会话隔离、跨 Run 引用及默认工具注册；filesystem, execution, context | 新增约定 | [check_sessions](checks/filesystem/observations.py) |
| `filesystem.observations.privacy` | filesystem | 观测正文与提示值不进入日志及 Tracing；filesystem, execution, context | 新增约定 | [check_privacy](checks/filesystem/observations.py) |
| `filesystem.observations.lifecycle` | filesystem | 运行及会话存储所有权、关闭与失败清理；filesystem, execution, context | 新增约定 | [check_lifecycle](checks/filesystem/observations.py) |
| `sessions.history.history_and_copy_isolation` | sessions | 跨轮历史、返回值拷贝和检查点隔离；sessions | `check_sessions.py:check_history_and_copy_isolation` | [check_history_and_copy_isolation](checks/sessions/history.py) |
| `sessions.history.file_history_and_budget` | sessions | 真实文件历史继承与每轮读取额度重置；sessions, filesystem, recording | `check_sessions.py:check_file_history_and_budget` | [check_file_history_and_budget](checks/sessions/history.py) |
| `sessions.history.failure_history_boundaries` | sessions | 模型、协议、工具失败后的历史准入及次轮重置；sessions, execution | `check_sessions.py:check_failure_history_boundaries` | [check_failure_history_boundaries](checks/sessions/history.py) |
| `sessions.history.partial_batch_and_usage` | sessions | 半批工具日志失败后的历史边界及缺失用量；sessions, execution, recording | `check_sessions.py:check_partial_batch_and_usage` | [check_partial_batch_and_usage](checks/sessions/history.py) |
| `sessions.history.invalid_new_input` | sessions | 非法工具关联输入不发送且不继承；sessions, execution | `check_sessions.py:check_invalid_new_input` | [check_invalid_new_input](checks/sessions/history.py) |
| `sessions.history.concurrency` | sessions | 同会话重入拒绝与跨会话交错运行隔离；sessions, filesystem | `check_sessions.py:check_concurrency` | [check_concurrency](checks/sessions/history.py) |
| `sessions.history.checkpoint_write_failures` | sessions | 六类真实检查点写入失败与会话阻塞；sessions, execution, recording | `check_sessions.py:check_checkpoint_write_failures` | [check_checkpoint_write_failures](checks/sessions/history.py) |
| `sessions.history.checkpoint_and_skipped_log_failure` | sessions | 检查点失败叠加工具跳过日志失败保留事实；sessions, execution, recording | `check_sessions.py:check_checkpoint_and_skipped_log_failure` | [check_checkpoint_and_skipped_log_failure](checks/sessions/history.py) |
| `sessions.history.checkpoint_read_failure` | sessions | 检查点读取失败保留原存储并阻塞会话；sessions, recording | `check_sessions.py:check_checkpoint_read_failure` | [check_checkpoint_read_failure](checks/sessions/history.py) |
| `sessions.history.creation_and_start_log_failure` | sessions | 检查点创建失败与开始日志失败后的历史恢复；sessions, recording | `check_sessions.py:check_creation_and_start_log_failure` | [check_creation_and_start_log_failure](checks/sessions/history.py) |
| `sessions.history.finalization_combinations` | sessions | 最终保存、客户端关闭和结束日志组合失败；sessions, recording | `check_sessions.py:check_finalization_combinations` | [check_finalization_combinations](checks/sessions/history.py) |
| `sessions.history.tracing_sessions` | sessions | 会话追踪开关、过滤、上传故障及业务等价性；sessions, recording, filesystem | `check_sessions.py:check_tracing_sessions` | [check_tracing_sessions](checks/sessions/history.py) |
| `sessions.history.cli` | sessions | 完整同步 CLI 交互、会话切换、单次命令及保存失败恢复；sessions, execution | `check_sessions.py:check_cli` | [check_cli](checks/sessions/history.py) |
| `sessions.admission.cache_does_not_keep_frames` | sessions | 检查点失败缓存不保留异常栈和已关闭资源；sessions, recording | `check_session_review_fixes.py:check_cache_does_not_keep_frames` | [check_cache_does_not_keep_frames](checks/sessions/admission.py) |
| `sessions.admission.budget_helper_boundary` | sessions | 读取额度观察异常与取消传播边界；sessions, filesystem, recording | `check_session_review_fixes.py:check_budget_helper_boundary` | [check_budget_helper_boundary](checks/sessions/admission.py) |
| `sessions.admission.metadata_stages_and_tracing` | sessions | 三阶段读取额度异常及日志追踪共享观察；sessions, filesystem, recording | `check_session_review_fixes.py:check_metadata_stages_and_tracing` | [check_metadata_stages_and_tracing](checks/sessions/admission.py) |
| `sessions.admission.metadata_keeps_real_budget` | sessions | 元数据不可用时真实文件读取额度仍然生效；sessions, filesystem, execution, recording | `check_session_review_fixes.py:check_metadata_keeps_real_budget` | [check_metadata_keeps_real_budget](checks/sessions/admission.py) |
| `sessions.admission.metadata_failure_combinations` | sessions | 额度元数据与日志、检查点组合失败保留结果；sessions, filesystem, recording | `check_session_review_fixes.py:check_metadata_failure_combinations` | [check_metadata_failure_combinations](checks/sessions/admission.py) |
| `sessions.admission.invalid_input_history` | sessions | 非法新输入拒绝且不污染可继承历史；sessions, execution | `check_session_review_fixes.py:check_invalid_input_history` | [check_invalid_input_history](checks/sessions/admission.py) |
| `sessions.admission.valid_protocol_and_http_history` | sessions | 合法协议扩展历史与 HTTP 错误准入边界；sessions, execution | `check_session_review_fixes.py:check_valid_protocol_and_http_history` | [check_valid_protocol_and_http_history](checks/sessions/admission.py) |
| `sessions.admission.preexecution_config_retry` | sessions | 执行前配置失败保留历史并允许重试；sessions, execution, recording | `check_session_review_fixes.py:check_preexecution_config_retry` | [check_preexecution_config_retry](checks/sessions/admission.py) |
| `sessions.admission.execution_started_still_blocks` | sessions | 客户端进入失败后会话阻塞及资源关闭；sessions, execution, recording | `check_session_review_fixes.py:check_execution_started_still_blocks` | [check_execution_started_still_blocks](checks/sessions/admission.py) |
| `sessions.admission.finally_and_primary_error` | sessions | 收尾取消传播、资源关闭及主错误保留；sessions, recording | `check_session_review_fixes.py:check_finally_and_primary_error` | [check_finally_and_primary_error](checks/sessions/admission.py) |
| `sessions.admission.prepare_and_final_save_secondary_errors` | sessions | 准备和最终保存错误不被次级追踪异常覆盖；sessions, recording | `check_session_review_fixes.py:check_prepare_and_final_save_secondary_errors` | [check_prepare_and_final_save_secondary_errors](checks/sessions/admission.py) |
| `sessions.admission.cli_preconfiguration_retry` | sessions | 同步 CLI 配置失败重试及错误退出语义；sessions, execution | `check_session_review_fixes.py:check_cli_preconfiguration_retry` | [check_cli_preconfiguration_retry](checks/sessions/admission.py) |
| `context.budget.configuration` | context | 上下文预算默认值、配置校验及显式环境加载；context | `check_context_budget.py:check_configuration` | [check_configuration](checks/context/budget.py) |
| `context.budget.transparent_request_and_boundary` | context | 实际请求透明计量与精确预算阈值；context, execution | `check_context_budget.py:check_transparent_request_and_boundary` | [check_transparent_request_and_boundary](checks/context/budget.py) |
| `context.budget.stable_history_and_rejected_input` | context | 超预算新输入拒绝后的历史稳定性；context, sessions | `check_context_budget.py:check_stable_history_and_rejected_input` | [check_stable_history_and_rejected_input](checks/context/budget.py) |
| `context.budget.tool_result_limit_and_real_file` | context | 真实文件结果触发下一请求预算并保留事实；context, filesystem, sessions, execution | `check_context_budget.py:check_tool_result_limit_and_real_file` | [check_tool_result_limit_and_real_file](checks/context/budget.py) |
| `context.budget.opaque_counting` | context | 不透明内容风险计量、来源用量与后备值；context | `check_context_budget.py:check_opaque_counting` | [check_opaque_counting](checks/context/budget.py) |
| `context.budget.opaque_provenance_and_isolation` | context | 不透明风险来源跨轮保留及会话隔离；context, sessions | `check_context_budget.py:check_opaque_provenance_and_isolation` | [check_opaque_provenance_and_isolation](checks/context/budget.py) |
| `context.budget.file_sessions_isolation` | context | 文件根、预算、历史及新轮额度共同隔离；context, filesystem, sessions | `check_context_budget.py:check_file_sessions_isolation` | [check_file_sessions_isolation](checks/context/budget.py) |
| `context.budget.every_call_and_usage` | context | 每次实际请求先计量及部分用量累计；context, execution, recording | `check_context_budget.py:check_every_call_and_usage` | [check_every_call_and_usage](checks/context/budget.py) |
| `context.budget.measurement_failure` | context | 不可计量输入与请求准备失败阻止发送；context, execution | `check_context_budget.py:check_measurement_failure` | [check_measurement_failure](checks/context/budget.py) |
| `context.budget.session_invalid_storage_boundaries` | context | 实际检查点序列化与请求计量失败边界；context, sessions, recording | `check_context_budget.py:check_session_invalid_storage_boundaries` | [check_session_invalid_storage_boundaries](checks/context/budget.py) |
| `context.budget.budget_recording_failures` | context | 预算日志失败及已执行工具事实保留；context, execution, recording | `check_context_budget.py:check_budget_recording_failures` | [check_budget_recording_failures](checks/context/budget.py) |
| `context.budget.budget_checkpoint_failures` | context | 预算停止和工具完成后的检查点保存失败；context, sessions, execution, recording | `check_context_budget.py:check_budget_checkpoint_failures` | [check_budget_checkpoint_failures](checks/context/budget.py) |
| `context.budget.tracing_diagnostics` | context | 预算追踪诊断过滤及上传故障等价性；context, recording | `check_context_budget.py:check_tracing_diagnostics` | [check_tracing_diagnostics](checks/context/budget.py) |
| `context.budget.cli` | context | 预算停止展示与交互拒绝后继续；context, sessions, execution | `check_context_budget.py:check_cli` | [check_cli](checks/context/budget.py) |
| `context.compaction.prefix_independent_long_run_recovers_current_prefix_and_protects_three_groups` | context | 独立长 Run 压缩当前前缀并保护最近三组；context, sessions, recording | 当前 Compact 约定 | [check_prefix_independent_long_run_recovers_current_prefix_and_protects_three_groups](checks/context/compaction.py) |
| `context.compaction.prefix_session_first_long_run_originals_and_r_inherit_with_old_turn` | context | 会话首轮原始记录完整，旧 R 随所属轮次继承；context, sessions, recording | 当前 Compact 约定 | [check_prefix_session_first_long_run_originals_and_r_inherit_with_old_turn](checks/context/compaction.py) |
| `context.compaction.prefix_original_call_ids_remain_reserved_after_prefix_compaction` | context | 移出前缀后的原工具调用身份仍被保留；context, sessions, recording | 当前 Compact 约定 | [check_prefix_original_call_ids_remain_reserved_after_prefix_compaction](checks/context/compaction.py) |
| `context.compaction.prefix_summary_failure_keeps_active_originals_and_does_not_replay_tools` | context | 摘要失败保留旧活动正文且不重放业务工具；context, sessions, recording | 当前 Compact 约定 | [check_prefix_summary_failure_keeps_active_originals_and_does_not_replay_tools](checks/context/compaction.py) |
| `context.compaction.prefix_history_and_current_prefix_generate_h_then_r_and_commit_together` | context | 历史 H 与当前 R 顺序生成并整体提交；context, sessions, recording | 当前 Compact 约定 | [check_prefix_history_and_current_prefix_generate_h_then_r_and_commit_together](checks/context/compaction.py) |
| `context.compaction.prefix_h_success_r_failure_keeps_both_original_views` | context | H 成功而 R 失败时保留两路原活动视图；context, sessions, recording | 当前 Compact 约定 | [check_prefix_h_success_r_failure_keeps_both_original_views](checks/context/compaction.py) |
| `context.compaction.prefix_failed_first_model_call_history_keeps_error_outcome_when_compacted` | context | 首个业务模型失败的历史仍保留错误终态；context, sessions, recording | 当前 Compact 约定 | [check_prefix_failed_first_model_call_history_keeps_error_outcome_when_compacted](checks/context/compaction.py) |
| `context.compaction.prefix_current_protection_never_preserves_historical_turns` | context | 旧轮次全部进入 H，不参与当前三组保护；context, sessions, recording | 当前 Compact 约定 | [check_prefix_current_protection_never_preserves_historical_turns](checks/context/compaction.py) |
| `context.compaction.prefix_model_searches_and_reads_detail_missing_from_r_without_tool_replay` | context | 模型搜索并回读 R 遗漏细节且不重放工具；context, sessions, recording | 当前 Compact 约定 | [check_prefix_model_searches_and_reads_detail_missing_from_r_without_tool_replay](checks/context/compaction.py) |
| `context.compaction.prefix_oversized_valid_candidate_is_shortened_once_without_original_bodies` | context | 过大合法候选只缩短一次，不重新附上原文；context, sessions, recording | 当前 Compact 约定 | [check_prefix_oversized_valid_candidate_is_shortened_once_without_original_bodies](checks/context/compaction.py) |
| `context.compaction.prefix_empty_summary_stops_after_one_call_and_keeps_original_view` | context | 空摘要在一次调用后停止并保留旧视图；context, sessions, recording | 当前 Compact 约定 | [check_prefix_empty_summary_stops_after_one_call_and_keeps_original_view](checks/context/compaction.py) |
| `context.compaction.prefix_checkpoint_failures_keep_old_activity_and_actual_tool_and_summary_facts` | context | 保存故障保留原活动正文和已发生的工具、摘要事实；context, sessions, recording | 当前 Compact 约定 | [check_prefix_checkpoint_failures_keep_old_activity_and_actual_tool_and_summary_facts](checks/context/compaction.py) |
| `context.compaction.prefix_two_complete_attempts_can_call_both_lanes_but_never_a_fifth_summary` | context | 两次完整尝试可调用两路，绝不发起第五次摘要；context, sessions, recording | 当前 Compact 约定 | [check_prefix_two_complete_attempts_can_call_both_lanes_but_never_a_fifth_summary](checks/context/compaction.py) |
| `context.compaction.prefix_second_oversized_candidate_stops_without_reinitializing_attempts` | context | 第二个候选仍超限时停止，不重置恢复额度；context, sessions, recording | 当前 Compact 约定 | [check_prefix_second_oversized_candidate_stops_without_reinitializing_attempts](checks/context/compaction.py) |
| `context.compaction.prefix_source_version_change_preserves_latest_confirmed_view` | context | 源版本变化时保持最近已确认的活动视图；context, sessions, recording | 当前 Compact 约定 | [check_prefix_source_version_change_preserves_latest_confirmed_view](checks/context/compaction.py) |
| `context.compaction.prefix_protected_context_too_large_stops_without_reducing_three_groups` | context | 保护区过大时停止，不减少三组保护；context, sessions, recording | 当前 Compact 约定 | [check_prefix_protected_context_too_large_stops_without_reducing_three_groups](checks/context/compaction.py) |
| `context.compaction.prefix_summary_request_preflight_refuses_without_sending_or_recursive_recovery` | context | 摘要预检拒绝不发起调用，也不递归恢复；context, sessions, recording | 当前 Compact 约定 | [check_prefix_summary_request_preflight_refuses_without_sending_or_recursive_recovery](checks/context/compaction.py) |
| `context.compaction.prefix_multiple_tool_calls_and_legal_failures_remain_in_whole_groups` | context | 同响应多工具与合法失败保持完整交互组；context, sessions, recording | 当前 Compact 约定 | [check_prefix_multiple_tool_calls_and_legal_failures_remain_in_whole_groups](checks/context/compaction.py) |
| `context.compaction.prefix_unclosed_multi_tool_group_is_excluded_from_confirmed_prefix_selection` | context | 未闭合多工具组不进入已确认前缀来源；context, sessions, recording | 当前 Compact 约定 | [check_prefix_unclosed_multi_tool_group_is_excluded_from_confirmed_prefix_selection](checks/context/compaction.py) |
| `context.compaction.prefix_previous_h_is_reused_without_call_or_target_allocation` | context | 无新增来源直接复用旧 H，不进行目标分配；context, sessions, recording | 当前 Compact 约定 | [check_prefix_previous_h_is_reused_without_call_or_target_allocation](checks/context/compaction.py) |
| `context.compaction.prefix_actual_summary_request_larger_than_window_is_not_sent` | context | 真实摘要候选超过窗口时不发送；context, sessions, recording | 当前 Compact 约定 | [check_prefix_actual_summary_request_larger_than_window_is_not_sent](checks/context/compaction.py) |
| `context.compaction.boundaries_checked_requests_equal_actual_sdk_requests_including_final_call_options` | context | 预算检查副本与实际 SDK 请求相同，含末轮收尾选项；context | 当前 Compact 约定 | [check_boundaries_checked_requests_equal_actual_sdk_requests_including_final_call_options](checks/context/compaction.py) |
| `context.compaction.boundaries_allowed_candidate_below_old_headroom_commits_on_first_attempt` | context | 候选预算放行且净回收为正时提交，不设置空闲比例门槛；context | 当前 Compact 约定 | [check_boundaries_allowed_candidate_below_old_headroom_commits_on_first_attempt](checks/context/compaction.py) |
| `context.compaction.boundaries_private_final_save_failure_cannot_report_completed` | context | 独立 Run 最终保存失败不能报告完成；context | 当前 Compact 约定 | [check_boundaries_private_final_save_failure_cannot_report_completed](checks/context/compaction.py) |
| `context.compaction.boundaries_activation_confirmation_failure_keeps_old_activity_and_paid_facts` | context | 激活确认失败保持原视图和已付调用事实；context | 当前 Compact 约定 | [check_boundaries_activation_confirmation_failure_keeps_old_activity_and_paid_facts](checks/context/compaction.py) |
| `context.compaction.boundaries_summary_logging_boundaries_do_not_send_or_enable_unsaved_work` | context | 摘要日志故障不发送或启用未确认工作；context | 当前 Compact 约定 | [check_boundaries_summary_logging_boundaries_do_not_send_or_enable_unsaved_work](checks/context/compaction.py) |
| `context.compaction.boundaries_short_run_has_no_summary_or_history_tools_executed` | context | 短任务不触发摘要或历史工具执行；context | 当前 Compact 约定 | [check_boundaries_short_run_has_no_summary_or_history_tools_executed](checks/context/compaction.py) |
| `context.compaction.history_records_survive_active_input_replacement_and_fixed_snapshot` | context | 替换活动正文后原始记录和固定快照仍可读取；context | 当前 Compact 约定 | [check_history_records_survive_active_input_replacement_and_fixed_snapshot](checks/context/compaction.py) |
| `context.compaction.history_original_record_creation_preserves_invalid_input_for_admission` | context | 原始记录保留无效输入事实供准入判断；context | 当前 Compact 约定 | [check_history_original_record_creation_preserves_invalid_input_for_admission](checks/context/compaction.py) |
| `context.compaction.history_native_saved_non_json_opaque_values_do_not_poison_later_records` | context | 原生保存的非 JSON 不透明值不破坏后续记录；context | 当前 Compact 约定 | [check_history_native_saved_non_json_opaque_values_do_not_poison_later_records](checks/context/compaction.py) |
| `context.compaction.history_missing_sequences_and_conflicting_immutable_identity_fail` | context | 记录序号缺失或不可变身份冲突时拒绝读取；context | 当前 Compact 约定 | [check_history_missing_sequences_and_conflicting_immutable_identity_fail](checks/context/compaction.py) |
| `context.compaction.history_unconfirmed_checkpoints_and_released_cache_are_unreadable` | context | 未确认检查点和已释放缓存不可读取；context | 当前 Compact 约定 | [check_history_unconfirmed_checkpoints_and_released_cache_are_unreadable](checks/context/compaction.py) |
| `context.compaction.history_scope_is_immutable_and_contains_unreferenced_originals` | context | 检索范围不可变且包含未被摘要引用的原文；context | 当前 Compact 约定 | [check_history_scope_is_immutable_and_contains_unreferenced_originals](checks/context/compaction.py) |
| `context.compaction.history_read_unicode_escape_pagination_matches_fixed_projection` | context | Unicode 与转义分页与固定可读视图一致；context | 当前 Compact 约定 | [check_history_read_unicode_escape_pagination_matches_fixed_projection](checks/context/compaction.py) |
| `context.compaction.history_search_pages_have_no_omissions_and_cursors_are_bound` | context | 关键词分页无遗漏，游标绑定范围及查询；context | 当前 Compact 约定 | [check_history_search_pages_have_no_omissions_and_cursors_are_bound](checks/context/compaction.py) |
| `context.compaction.history_cross_session_reference_is_rejected` | context | 跨会话来源引用被拒绝；context | 当前 Compact 约定 | [check_history_cross_session_reference_is_rejected](checks/context/compaction.py) |
| `context.compaction.history_native_history_recall_has_no_artifact_or_file_budget` | context | 原生历史召回不创建 Artifact 或消耗文件读取额度；context | 当前 Compact 约定 | [check_history_native_history_recall_has_no_artifact_or_file_budget](checks/context/compaction.py) |
| `context.compaction.history_opaque_projection_omits_encrypted_and_unknown_content` | context | 可读投影省略加密字段和未知不透明内容；context | 当前 Compact 约定 | [check_history_opaque_projection_omits_encrypted_and_unknown_content](checks/context/compaction.py) |
| `context.compaction.history_tool_facts_distinguish_execution_and_observation_failure` | context | 工具执行失败与观测失败事实分别记录；context | 当前 Compact 约定 | [check_history_tool_facts_distinguish_execution_and_observation_failure](checks/context/compaction.py) |
| `context.compaction.history_interruption_keeps_actual_facts_without_fabricating_results` | context | 中断保留实际事实，不编造终态结果；context | 当前 Compact 约定 | [check_history_interruption_keeps_actual_facts_without_fabricating_results](checks/context/compaction.py) |
| `context.compaction.settings_defaults_and_limits` | context | 当前组保护与 H/R 输出上限默认值及正整数边界；context | 当前 Compact 约定 | [check_settings_defaults_and_limits](checks/context/compaction.py) |
| `context.compaction.settings_old_turn_setting_is_rejected_in_constructor_and_environment` | context | 构造和环境加载均拒绝旧轮次保护配置；context | 当前 Compact 约定 | [check_settings_old_turn_setting_is_rejected_in_constructor_and_environment](checks/context/compaction.py) |
| `context.compaction.settings_new_environment_values_are_loaded_without_old_field` | context | 新环境配置正确加载且不再使用旧字段；context | 当前 Compact 约定 | [check_settings_new_environment_values_are_loaded_without_old_field](checks/context/compaction.py) |
| `context.compaction.summary_lane_caps_are_independent_of_business_output_allowance` | context | H/R 输出上限独立于业务上限并与实际请求一致；context | 当前 Compact 约定 | [check_summary_lane_caps_are_independent_of_business_output_allowance](checks/context/compaction.py) |
| `context.compaction.summary_invalid_lane_or_attempt_never_prepares_a_request` | context | 无效路别或尝试编号不构造请求；context | 当前 Compact 约定 | [check_summary_invalid_lane_or_attempt_never_prepares_a_request](checks/context/compaction.py) |
| `context.compaction.summary_plain_text_is_preserved_without_schema_or_reference_parsing` | context | 普通文字无需 JSON 或逐条引用解析；context | 当前 Compact 约定 | [check_summary_plain_text_is_preserved_without_schema_or_reference_parsing](checks/context/compaction.py) |
| `context.compaction.summary_invalid_completion_refusal_tool_request_and_empty_text_are_rejected` | context | 拒绝空白、错误、拒绝、截断及工具请求；context | 当前 Compact 约定 | [check_summary_invalid_completion_refusal_tool_request_and_empty_text_are_rejected](checks/context/compaction.py) |
| `context.compaction.summary_summary_budget_uses_lane_cap_without_changing_business_settings` | context | 摘要预算只使用所属路别的输出上限；context | 当前 Compact 约定 | [check_summary_summary_budget_uses_lane_cap_without_changing_business_settings](checks/context/compaction.py) |
| `context.compaction.summary_complete_multi_tool_group_including_failure_is_allowed` | context | 包含合法失败的完整多工具组允许压缩；context | 当前 Compact 约定 | [check_summary_complete_multi_tool_group_including_failure_is_allowed](checks/context/compaction.py) |
| `context.compaction.summary_known_encrypted_reasoning_is_whole_group_only` | context | 已知加密 reasoning 仅随完整交互组移出；context | 当前 Compact 约定 | [check_summary_known_encrypted_reasoning_is_whole_group_only](checks/context/compaction.py) |
| `context.compaction.summary_unrecognized_nontext_authority_and_incomplete_items_are_barriers` | context | 未知协议、多模态、权限消息与未闭合项阻止压缩；context | 当前 Compact 约定 | [check_summary_unrecognized_nontext_authority_and_incomplete_items_are_barriers](checks/context/compaction.py) |
| `recording.transport.projection` | recording | 字段投影、密钥及路径过滤、原对象不变与载荷上限。；recording | `check_tracing.py:check_projection` | [check_projection](checks/recording/transport.py) |
| `recording.projection.redaction_boundaries` | recording | 引号、转义、跨文本及截断边界的凭据过滤与幂等性。；recording | `check_trace_data.py:check_redaction_boundaries` | [check_redaction_boundaries](checks/recording/projection.py) |
| `recording.projection.capacity_and_rounds` | recording | 容量与遗漏元数据、完整工具批次保留、回答和用量投影。；recording, execution | `check_trace_data.py:check_capacity_and_rounds` | [check_capacity_and_rounds](checks/recording/projection.py) |
| `recording.usage.usage_categories` | recording | 缺失、无效、零值和部分用量的覆盖分类。；recording | `check_trace_usage.py:check_usage_categories` | [check_usage_categories](checks/recording/usage.py) |
| `recording.usage.mixed_run_and_empty_run` | recording | 多次调用混合用量覆盖及空运行。；recording | `check_trace_usage.py:check_mixed_run_and_empty_run` | [check_mixed_run_and_empty_run](checks/recording/usage.py) |
| `recording.usage.remote_mismatches` | recording | 拒绝本地与远端最终用量投影不一致。；recording | `check_trace_usage.py:check_remote_mismatches` | [check_remote_mismatches](checks/recording/usage.py) |
| `recording.usage.final_projection_only` | recording | 最终投影缺失用量时不使用原始用量补判覆盖。；recording | `check_trace_usage.py:check_final_projection_only` | [check_final_projection_only](checks/recording/usage.py) |
| `recording.transport.serialized_redaction` | recording | 真实 SDK 序列化后重组过滤凭据，业务对象与请求不变。；recording, execution | `check_tracing.py:check_serialized_redaction` | [check_serialized_redaction](checks/recording/transport.py) |
| `recording.transport.serialized_capacity` | recording | 超大输入的上传限额不丢失小回答及实际投影用量。；recording, context | `check_tracing.py:check_serialized_capacity` | [check_serialized_capacity](checks/recording/transport.py) |
| `recording.transport.disabled_and_equivalence` | recording | 追踪开关的业务等价、混合工具结果、调用关联与统计。；recording, execution, filesystem | `check_tracing.py:check_disabled_and_equivalence` | [check_disabled_and_equivalence](checks/recording/transport.py) |
| `recording.transport.failures_and_shutdown` | recording | 上传、配置、初始化及过滤失败的隔离、不重试和有限收尾。；recording, execution | `check_tracing.py:check_failures_and_shutdown` | [check_failures_and_shutdown](checks/recording/transport.py) |
| `recording.transport.concurrency_and_skips` | recording | 并发运行隔离、模型额度导致的工具跳过及关联。；recording, execution | `check_tracing.py:check_concurrency_and_skips` | [check_concurrency_and_skips](checks/recording/transport.py) |
| `recording.transport.local_recording_failure` | recording | 本地记录失败后工具跳过事实与追踪保持一致。；recording, execution | `check_tracing.py:check_local_recording_failure` | [check_local_recording_failure](checks/recording/transport.py) |
| `recording.transport.late_protocol_failure` | recording | 后续重复调用标识的协议错误只标记对应模型调用。；recording, execution | `check_tracing.py:check_late_protocol_failure` | [check_late_protocol_failure](checks/recording/transport.py) |
| `recording.transport.missing_usage_and_model_error` | recording | 缺失用量和模型异常的安全追踪投影。；recording, execution | `check_tracing.py:check_missing_usage_and_model_error` | [check_missing_usage_and_model_error](checks/recording/transport.py) |
| `recording.transport.cancellation_and_queue_overflow` | recording | 模型与工具取消、资源关闭及有界上传队列。；recording, execution | `check_tracing.py:check_cancellation_and_queue_overflow` | [check_cancellation_and_queue_overflow](checks/recording/transport.py) |
| `recording.transport.cli_exit_deadline` | recording | 真实 CLI 子进程在上传卡住时仍有限退出。；recording, execution | `check_tracing.py:check_cli_exit_deadline` | [check_cli_exit_deadline](checks/recording/transport.py) |
| `recording.files.real_read_and_equivalence` | recording | 真实文件读取在追踪开关及五类上传失败下保持业务等价。；recording, filesystem | `check_text_tracing.py:check_real_read_and_equivalence` | [check_real_read_and_equivalence](checks/recording/files.py) |
| `recording.files.listing_and_model_quotation` | recording | 目录和 BOM 读取元数据隐藏正文，同时保留模型对资料的引用。；recording, filesystem, execution | `check_text_tracing.py:check_listing_and_model_quotation` | [check_listing_and_model_quotation](checks/recording/files.py) |
| `recording.files.failure_reasons` | recording | 同批读取拒绝、限制、权限、中途 I/O 与成功结果在模型、JSONL 和追踪中一致。；recording, filesystem, execution | `check_text_tracing.py:check_failure_reasons` | [check_failure_reasons](checks/recording/files.py) |
| `recording.files.projection_boundaries` | recording | 文本结果容量、原对象隔离和不可信原因及元数据过滤。；recording, filesystem | `check_text_tracing.py:check_projection_boundaries` | [check_projection_boundaries](checks/recording/files.py) |
| `recording.files.fixture_integrity` | recording | 全部检查结束后文件内容、目录成员及当前目录保持不变。；recording, filesystem | `check_text_tracing.py:main:fixture_integrity` | [main:fixture_integrity](checks/recording/files.py) |
| `recording.sessions.checkpoint_trace_equivalence` | recording | 检查点追踪 harness 在关闭、开启及上传失败下请求和结果等价。；recording, sessions | `check_session_tracing.py:main:checkpoint_trace_equivalence` | [main:checkpoint_trace_equivalence](checks/recording/sessions.py) |
| `recording.sessions.checkpoint_error_metadata` | recording | 三类检查点错误的追踪元数据及安全投影，与正常 harness 业务结果等价。；recording, sessions | `check_session_tracing.py:main:checkpoint_error_metadata` | [main:checkpoint_error_metadata](checks/recording/sessions.py) |
| `recording.sessions.session_history_trace_equivalence` | recording | 真实多轮会话、文件历史与检查点在追踪开关及上传失败下保持等价。；recording, sessions, filesystem | `check_session_tracing.py:main:session_history_trace_equivalence` | [main:session_history_trace_equivalence](checks/recording/sessions.py) |
| `recording.sessions.session_final_save_failure` | recording | 第二轮最终保存失败的追踪归属与已取得业务结果保留。；recording, sessions | `check_session_tracing.py:main:session_final_save_failure` | [main:session_final_save_failure](checks/recording/sessions.py) |
| `recording.sessions.skipped_after_save_failure` | recording | 模型结果保存失败后停止工具执行，保留用量、跳过及关联事实。；recording, sessions, execution, filesystem | `check_session_tracing.py:check_skipped_after_save_failure` | [check_skipped_after_save_failure](checks/recording/sessions.py) |
| `recording.sessions.fixture_integrity` | recording | 全部会话检查结束后实际来源文件保持不变。；recording, sessions, filesystem | `check_session_tracing.py:main:fixture_integrity` | [main:fixture_integrity](checks/recording/sessions.py) |
| `recording.compaction.summary_trace_equivalence` | recording | 摘要追踪 harness 的正常、关闭及上传失败保持请求结果等价，摘要正文与派生输入被省略。；recording, context | `check_compaction_tracing.py:main:summary_trace_equivalence` | [main:summary_trace_equivalence](checks/recording/compaction.py) |
| `recording.compaction.summary_tool_request` | recording | 摘要意外工具调用的失败投影保持正文省略。；recording, context | `check_compaction_tracing.py:main:summary_tool_request` | [main:summary_tool_request](checks/recording/compaction.py) |
| `recording.compaction.summary_incomplete` | recording | 未完成摘要的追踪仍不暴露摘要正文。；recording, context | `check_compaction_tracing.py:main:summary_incomplete` | [main:summary_incomplete](checks/recording/compaction.py) |
| `recording.compaction.bounded_lineage_provenance` | recording | 摘要来源引用有界且只依据明确来源隐藏派生内容，不按文本前缀猜测。；recording, context | `check_compaction_tracing.py:main:bounded_lineage_provenance` | [main:bounded_lineage_provenance](checks/recording/compaction.py) |
