# 工程约定与迁移覆盖

本文面向维护确定性检查的开发者。唯一可执行清单为 [contracts.json](suites/agent_contracts/contracts.json)，模块按被测约定划分，不代表不同项目评测集。全量入口为 `python -m benchmarks run agent-contracts`。当前 17 条隔离行为链包含 155 个行为组（原 138 组、动态摘要恢复新增 3 组、Artifact 新增 8 组及观测治理新增 6 组）；实际是否执行及结果以运行报告为准。

## 模块与判断依据

| 模块 | 行为组 | 约定与指标 |
| --- | ---: | --- |
| `execution` | 19 | 输入/响应协议、批次继续、结果关联、调用额度和终态统计；直接比较实际请求、工具返回和 State。 |
| `filesystem` | 36 | 目录与文本路径边界、读取额度、真实 I/O，以及 Artifact 快照、有界召回与会话隔离；核对实际文件和累计预算。 |
| `sessions` | 25 | 历史准入、隔离、重置、并发拒绝、保存和关闭；核对原生检查点及后续请求。 |
| `context` | 36 | 请求预算、风险计量、摘要选择、保存屏障和活动版本；对照真实请求参数与记录。 |
| `recording` | 39 | JSONL、过滤、usage 投影、序列化、上传异常和有限收尾；对照实际 SDK 载荷与故障观察。 |

一次跨模块场景保留完整执行链，只按主要模块计一次；标签不作为根因结论。响应耗时、真实 token 与缓存字段由真实任务轨迹另外观测，替身 token 不算成本。没有统一总分。

## 迁移与等价性

来源基线为 `c97467ed41bcd3a0232033f9160d15199b30069b`。14 个原 `scripts/check_*.py` 已移除，其检查函数和实际公共替身归入 Benchmark。ReAct 的执行检查与记录检查分开；Review 的输入协议、文件系统预算、追踪投影按职责分开，未拆断单个故障链。Session 两处异步入口后的 CLI 检查、fixture 收尾和五次摘要 Tracing exercise 均保留。data/usage 组经 recording.transport 调用，不另建隔离单元或重复计数。

此前脚本迁移核对中，基线 1,444 条 assert 有 1,443 条业务/边界断言的 AST 保持不变。原 Review `assert not failures` 是独立脚本聚合器收尾，现由组记录及失败停止策略承担。远端 usage 对比公共函数从 `verify_tracing.py` 提取，其一条既有断言归入公共代码，供诊断脚本和工程检查共同复用。该历史核对覆盖 138 组；断言数量不是覆盖率。动态 Compact 改造保留这些行为组，调整已被替代的比例阈值及固定摘要上限断言，并增加下表中的三个恢复行为组。Artifact 的八组为新增约定，不属于旧脚本迁移；当前结果以 155 组的实际事件、请求、状态及故障证据为准。

Artifact 检查验证完整原文保存、正文字符位置、精确 UTF-8 响应上限、关键词分页、会话及缓存路径隔离、关闭与故障行为，并用离线模型替身走通两轮工具调用及记录过滤。观测治理检查另行验证自动归档、结构化提示、默认召回工具和治理故障；两类检查均不以真实模型的上下文下降或回答质量作为结论。

框架回归保留选择拒绝、状态分离、回执一致性、事件有效前缀、资源收尾、保存故障、会话/材料隔离、预算与评分绑定。旧 v1 worker 和 control-* 别名专用执行测试随接口撤销，其仍适用的断言/启动/源码/优化保护/保存等边界由组 worker 与通用框架测试承接；旧结果读取测试独立保留。新增第二个项目注册、构造材料、LLM 建议绑定与离线网络边界测试。

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
| `context.compaction.configuration` | context | 摘要和历史保留配置边界；context | `check_compaction.py:check_configuration` | [check_configuration](checks/context/compaction.py) |
| `context.compaction.stable_history_and_records` | context | 未压缩活动版本与运行记录拷贝隔离；context, sessions | `check_compaction.py:check_stable_history_and_records` | [check_stable_history_and_records](checks/context/compaction.py) |
| `context.compaction.selection_read_path` | context | 轻量历史选择与完整记录等价及源读取失败；context, sessions, recording | `check_compaction.py:check_selection_read_path` | [check_selection_read_path](checks/context/compaction.py) |
| `context.compaction.compaction_status_consumers` | context | 压缩状态的 CLI 展示与追踪投影；context, recording | `check_compaction.py:check_compaction_status_consumers` | [check_compaction_status_consumers](checks/context/compaction.py) |
| `context.compaction.activation_and_stable_reuse` | context | 摘要保存确认、活动版本切换与后续复用；context, sessions, recording | `check_compaction.py:check_activation_and_stable_reuse` | [check_activation_and_stable_reuse](checks/context/compaction.py) |
| `context.compaction.request_options_through_compaction` | context | 请求选项贯穿选择、候选、真实保存和 SDK；context, sessions, execution | `check_compaction.py:check_request_options_through_compaction` | [check_request_options_through_compaction](checks/context/compaction.py) |
| `context.compaction.refusal_boundaries` | context | 无可压缩历史或当前请求过大时保持旧版本；context, sessions | `check_compaction.py:check_refusal_boundaries` | [check_refusal_boundaries](checks/context/compaction.py) |
| `context.compaction.summary_failure_cases` | context | 摘要服务、协议、来源及大小失败停止业务请求；context, execution | `check_compaction.py:check_summary_failure_cases` | [check_summary_failure_cases](checks/context/compaction.py) |
| `context.compaction.usage_missing` | context | 摘要与业务用量缺失分计及新轮重置；context, recording, sessions | `check_compaction.py:check_usage_missing` | [check_usage_missing](checks/context/compaction.py) |
| `context.compaction.summary_and_candidate_budget_boundaries` | context | 摘要请求及候选预算和准备失败边界；context, execution | `check_compaction.py:check_summary_and_candidate_budget_boundaries` | [check_summary_and_candidate_budget_boundaries](checks/context/compaction.py) |
| `context.compaction.dynamic_summary_budgets` | context | 实际请求的 W/O/F/P 动态目标及超过软目标的合法摘要；context, execution | 动态摘要恢复新增 | [check_dynamic_summary_budgets](checks/context/compaction.py) |
| `context.compaction.bounded_recovery_passes` | context | 同请求最多两次摘要、候选来源与请求归档及二次预检；context, sessions, recording | 动态摘要恢复新增 | [check_bounded_recovery_passes](checks/context/compaction.py) |
| `context.compaction.second_pass_checkpoint_boundaries` | context | 重试与第二候选保存故障保持旧上下文及已发生消耗；context, sessions, recording | 动态摘要恢复新增 | [check_second_pass_checkpoint_boundaries](checks/context/compaction.py) |
| `context.compaction.opaque_barriers_and_omissions` | context | 受保护历史阻止压缩及加密内容省略；context, sessions | `check_compaction.py:check_opaque_barriers_and_omissions` | [check_opaque_barriers_and_omissions](checks/context/compaction.py) |
| `context.compaction.real_file_and_failure_source` | context | 真实文件及工具失败事实进入摘要来源；context, filesystem, execution | `check_compaction.py:check_real_file_and_failure_source` | [check_real_file_and_failure_source](checks/context/compaction.py) |
| `context.compaction.repeated_compaction` | context | 同轮两次恢复的来源去重和工具结果保留；context, sessions, execution | `check_compaction.py:check_repeated_compaction` | [check_repeated_compaction](checks/context/compaction.py) |
| `context.compaction.unattempted_recovery_record` | context | 未尝试恢复与已调用摘要记录保持区分；context, recording | `check_compaction.py:check_unattempted_recovery_record` | [check_unattempted_recovery_record](checks/context/compaction.py) |
| `context.compaction.session_isolation` | context | 并发摘要、来源、版本及记录查询会话隔离；context, sessions | `check_compaction.py:check_session_isolation` | [check_session_isolation](checks/context/compaction.py) |
| `context.compaction.runtime_tracing` | context | 摘要正文过滤、追踪开关和上传失败业务等价性；context, recording | `check_compaction.py:check_runtime_tracing` | [check_runtime_tracing](checks/context/compaction.py) |
| `context.compaction.history_admission_and_cli` | context | 失败历史摘要准入及 CLI 压缩后连续交互；context, sessions, execution | `check_compaction.py:check_history_admission_and_cli` | [check_history_admission_and_cli](checks/context/compaction.py) |
| `context.compaction.checkpoint_boundaries` | context | 候选、激活、最终保存及工具后恢复保存失败；context, sessions, execution, recording | `check_compaction.py:check_checkpoint_boundaries` | [check_checkpoint_boundaries](checks/context/compaction.py) |
| `context.compaction.log_boundaries` | context | 摘要和版本日志失败及模型失败的组合边界；context, execution, recording | `check_compaction.py:check_log_boundaries` | [check_log_boundaries](checks/context/compaction.py) |
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
