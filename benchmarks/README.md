# Benchmark 使用与扩展

Benchmark 面向维护 Agent 的开发者，区分工程约定、项目任务和验证方式。评测端选择任务、执行并保存证据，Agent 不选择评测或判断自己通过。当前提供离线工程回归和 WeChat Reporting 项目任务；真实执行、机制覆盖与语义评审分别记录，运行完成不能代替质量判定。

## 选择与执行

从项目根目录运行：

```powershell
uv run --locked python -m benchmarks list
uv run --locked python -m benchmarks list wechat-reporting
uv run --locked python -m benchmarks list agent-contracts
uv run --locked python -m benchmarks run agent-contracts
uv run --locked python -m unittest discover -s benchmarks/tests -v
```

无参数时显示帮助，不执行检查。每次只选一个评测集；未知标识会拒绝。`agent-contracts` 是工程约定集合，其五个模块不是五个项目评测集。它用受控 fixture 验证实现，不依赖 WeChat 项目，也不调用模型或外部 Tracing。原 `--suite`、`control-*`、`--research-profile` 和已迁移 `scripts/check_*.py` 命令已撤销；没有保留另一份断言实现。`scripts/verify_*.py` 仍是显式的真实端点诊断入口，不属于默认评测范围。

`list <评测集>` 列出可填写的任务 ID、标题和轮次；工程集显示行为链及组数。列举不会读取 `.env`、准备材料或调用模型。

项目任务使用相同的 `run` 指令。首次使用时复制模板；已有个人配置时直接编辑，不重复覆盖：

```powershell
New-Item -ItemType Directory -Force benchmarks/local | Out-Null
Copy-Item benchmarks/entrypoints/run.example.json benchmarks/local/run.json
# 编辑 source_repository、units、预算和追踪范围；确认后设置 enabled=true。
uv run --locked python -m benchmarks run wechat-reporting --config benchmarks/local/run.json
```

[运行配置样例](entrypoints/run.example.json) 默认禁用，不含凭据。配置版本与结果版本分别管理；运行配置 v2 使用 `schema_version: "2"` 和 `agent_source: "project_env"`，在父进程通过生产配置入口读取进程环境及项目根目录 `.env`。模型字段的优先级为：任务 `settings_overrides` > 运行配置 `model_overrides` > 进程环境 > `.env` > 生产默认值。`project_root`、`log_dir` 由执行器为每个实例设置，不能通过 `model_overrides` 指向其他材料或日志。连接凭据沿用生产 `API_KEY` 来源，不写入配置、结果或子进程命令行；缺失时阻塞，不切换模型。

运行配置仅支持 v2，旧版会明确拒绝；请按当前模板重新填写，不会自动转换。只列举集合或运行 `agent-contracts` 不会加载研究配置或 `.env`。配置错误会指出字段、错误码、约束与修正提示，不输出凭据或原始配置值。

| 文件位置 | 职责与版本管理 |
| --- | --- |
| `benchmarks/entrypoints/*.example.json` | 受版本管理的通用模板，默认禁用，不放个人路径和凭据 |
| `benchmarks/local/` | 被 Git 忽略的个人运行、Judge 配置；不是执行产物，不会被自动迁移或改写 |
| `logs/benchmark-materials/`、`logs/benchmarks/` | 被 Git 忽略的准备材料、运行、报告和评审产物 |

`--config` 仍接受其他显式文件路径，包括之前放在 `logs/` 中的当前格式配置；推荐的新位置不改变解析规则。密钥放在生产 `.env` 或规定的进程环境变量中，不写入 JSON 配置。

v2 的 `units` 是必填的非空任务 ID 有序列表，按列出顺序执行，未知或重复 ID 被拒绝；`repeats` 明确每个任务的执行次数。`gate_first_sample: true` 要求第一个实例返回可信的 Agent `completed`，本地日志及适用检查点完整，回答文件与摘要可回访，才继续后续实例。该技术门槛不要求语义评分、摘要机制或远端 Tracing 通过。每个实例使用独立子进程与材料副本；同一实例的多轮共用会话，后续轮次只在前轮可信完成并确认检查点后继续。

外部追踪须由 v2 的 `tracing.enabled: true` 显式开启，并通过 `tracing.unit_ids` 指定本次已选任务中的上传范围；默认关闭，未列入的实例即使项目 `.env` 开启追踪也不上传。上传继续使用生产安全过滤，完整请求只在内存中用于机制核对。追踪关联和只读状态查询用于观察证据，不改变 Agent 的执行、摘要或回答策略。

配置须显式声明 `limits.requests`、`limits.execution_seconds`、`limits.sample_seconds`；每次业务、摘要及失败请求开始前均持久化记账，不自动重试。防误操作上限为 400 次请求、7,200 秒累计执行、每实例 1,200 秒、最多 20 次重复；这些是执行器上限，单次运行仍按获准配置的较低额度执行。每个启用追踪的实例最多进行三轮远端证据查询，合计不超过 30 秒，并计入批次累计时间。配置、凭据或材料缺失会产生 `blocked/pending_review` 记录。普通 Agent 终态与框架故障分别保存；全局凭据错误、记录失败、中断或预算耗尽停止后续调度。

## 工程结构

```text
benchmarks/
  __main__.py                    薄命令入口
  entrypoints/                   CLI 与通用配置样例
  core/                          注册、调度、结果、材料、预算、进程和证据
    executors/                   工程行为链和 Agent 任务两种执行方式
  checks/                        按约定组织的确定性检查
    execution/ filesystem/ sessions/ context/ recording/
    support/                     共用替身、fixture、观察与断言
  suites/
    agent_contracts/             工程约定清单、覆盖映射
    wechat_reporting/            项目任务、材料清单、构造变体、Rubric、转写
  evaluators/                    机械规则、检查项清单、人工评审与模型评审适配
  reporting/                     结果保存与人工可读报告
  local/                         个人配置目录，Git 忽略
  tests/                         框架自身的离线故障与契约检查
```

任务数据不包含模型、密钥或执行入口分支。`core/registry.py` 维护明确、受版本管理的评测集映射，运行时用标识选择；没有自动发现、任意命令接口或插件平台。新增项目时在 `suites/<project>/` 添加任务、材料与 Rubric，登记加载函数即可复用 Agent 执行器。只有新增不同的执行机制时才需扩展执行器；不需要修改 CLI 或用项目名称分支。框架测试用未注册到正式目录的第二个合成项目验证这条扩展路径。

## 验证方式

| 方式 | 目前用途 | 判定边界 |
| --- | --- | --- |
| 确定性验证 | 工程断言、读取失败/摘要实际发生、结构和版本绑定 | 不用关键词判断结论正确性 |
| 检查项清单 | 按任务适用条件聚合人工判断 | 缺失判断保持待评审，不适用项不进分母 |
| Rubric + LLM | 独立模型依据固定规则和材料生成结构化评审建议 | 当前未经真实校准，不自动形成质量通过结论 |
| 配对比较 | 固定摘要输入的 Instructions A/B 专用入口；通用任务配对未实现 | 描述性比较，不代替业务任务完成或泛化验证 |

真实执行结束后，运行目录的 `review.md` 和 `review-template.json` 提供回答、条件、严重错误和证据引用。复制模板为个人填写文件，填写评审者、时间、各项依据和支持关系，再导入：

```powershell
uv run --locked python -m benchmarks review logs/benchmarks/<运行ID> --file logs/benchmarks/<运行ID>/review-filled.json
```

导入校验原结果、回答摘要、材料与评分版本，保存独立的评分修订。引用存在并不意味着支持结论；严重错误可以否定质量通过，机械机制缺口不能被语义高分抵消。原执行结果和以前的修订保持原样。

[评判配置样例](entrypoints/judge.example.json) 也默认禁用。Judge 的 `credential_env` 只读取**当前进程环境**，不会自动加载项目 `.env`，也不继承研究加载器临时解析的凭据。首次复制并编辑模型、预算，确认后启用；在同一个 PowerShell 7 会话设置密钥再执行：

```powershell
New-Item -ItemType Directory -Force benchmarks/local | Out-Null
Copy-Item benchmarks/entrypoints/judge.example.json benchmarks/local/judge.json
# 编辑并启用 Judge 配置。输入的密钥不写入命令历史或配置文件。
$env:BENCHMARK_JUDGE_KEY = Read-Host 'Judge API key' -MaskInput
uv run --locked python -m benchmarks review logs/benchmarks/<运行ID> --config benchmarks/local/judge.json
Remove-Item Env:BENCHMARK_JUDGE_KEY
```

Judge 输入必须是原始未评审的当前 v3 结果，不能传入 `reviews/<修订ID>/`。无效结果在读取凭据、创建客户端和请求前拒绝。任务、回答和源码均作为不可信数据，输入字节、请求次数及时间有界，先记账再发送。

Rubric 的 `judge_scope` 指定 `model_suggestion` 或 `human_only`。Judge 只为回答、源码及可见对话提供建议；执行一致性、实际查阅证据、读取失败条件和摘要机制留给人工。严重错误中只有无依据扩大结论可以建议，执行真实性及含禁止行为的判断留给人工。历史判断只评价可见回答是否落实范围与约束，不证明记忆方式或摘要来源。代码拒绝越权的判定和证据引用；人工项保持待评审。缺少范围标注会明确报错，不自动套用当前规则；有效规则全部为人工项时不调用模型。

结果保存在 `judge-proposals/`，包含范围策略、绑定、逐项建议及独立 usage。建议需人工核对、补齐人工项并填写评审者后导入，不自动形成质量通过。当前模型评审未经真实校准。

### 重新生成评审包

评审包生成发生在执行结果保存之后。若模板或说明写入失败，命令退出 `2` 并列出已保存文件，不改写其绑定的原执行结果。可针对同一运行重新生成：

```powershell
uv run --locked python -m benchmarks review logs/benchmarks/<运行ID> --prepare
```

每次输出到独立的 `review-packets/<时间>-<UUID>/`，绑定原运行目录的 `result.json`；不会覆盖根目录模板、已有包或填写内容。使用新模板填写后，仍对原运行目录执行 `review --file <填写文件>`。此操作不调用模型。

## 结果、证据与失败

每次运行创建独占的 `logs/benchmarks/<UTC时间>-<UUID>/`，先保存 `plan.json`，再记录进度、`result.json` 和 `report.md`。实现、定义和转写进入 Git；运行材料位于 `logs/benchmark-materials/`，本机执行与评审产物沿用被忽略的 `logs/`。

结果仅支持 v3。`units` 表达隔离行为链或任务实例，`groups` 表达工程检查组，`turns` 表达 Agent 请求轮次，`dimensions` 和 `assessments` 表达适用维度及验证方式。`conditions` 绑定任务、模型、配置、材料和 Rubric。旧 v1/v2 的解析与渲染已移除；旧 JSON、Markdown、日志和历史定义保留，可直接查看，不自动转换或重评。

`report.md` 先展示状态、问题和覆盖缺口，再给出单元证据与简明基线。完整配置、轮次、评审结构和源码指纹见 `result.json`；完整源码清单仅在运行基线保存一次，不逐行为链复制。报告中的“评测结果退出码”只表示所保存执行结果；后续评审包操作失败仍使命令退出 `2`，不改变原执行事实。

执行状态 `completed/failed/blocked/not_run`、判定 `passed/failed/pending_review` 和 Agent 终态分别记录。退出码按 `2 > 1 > 3 > 0` 聚合：2 表示执行/证据/保存故障、阻塞、预算停止或适用机制缺口；1 表示有效未通过；3 表示仍待评审；0 表示选定范围完整通过且保存成功。一个断言失败会结束当前行为链，后续组保留 `not_run`，并继续下一独立进程；这些由断言导致的依赖未执行不另外升级为框架故障。记录通道失效、中断或收尾不确定则停止调度。

组事件、回执身份、定义摘要和进程退出需一致；自由文本中的成功提示不参与补判。损坏事件仅保留有效前缀。每组关联已观察到的运行/会话/步骤/调用/检查点标识及原日志，不把导入产生的目录当作已执行。标签说明受影响能力与失败环节，不证明根因。保存失败保留已有事实，仅公布确认保存的文件。

追踪证据分别记录安全的本地关联、提交状态及远端核验状态，关联本次样本、Agent 运行、会话、trace、实际请求 ID 和请求用途，记录已生成、入队、尝试提交、提交完成、失败、丢弃及尚未确认的 span 数量。`disabled` 表示未启用，`submission_completed` 仅表示本地提交收尾完成，`incomplete` 表示本地收尾不完整；只有实际远端查询取得匹配记录才能标为已核验。网络故障、尚未出现或查询预算不足均保留实际状态，不把本地提交成功当成远端接收证明，也不把追踪证据完整当成研究结论正确。

任务实例、轮次、工程组、实际 Agent 运行分别计数，不合并为“测试总数”。耗时与真实 SDK usage 单列，业务与摘要总量不重复相加；替身 usage 不算真实成本。缓存只报告端点实际提供的字段，完整且一致的输入/缓存计数才能计算缓存输入占比，它不等于请求命中率。无数据或账单时为未知，不补零或估算金额；评判模型消耗与被测模型分开保存。

## 维护与覆盖

摘要提示词对照使用独立入口 `python -m benchmarks.core.summary_prompt_comparison`，不执行 Agent 业务循环。先用 `freeze <sources.json> <新目录>` 固定历史正式提示词 A、当前提示词 B、原生输入、风险依据和来源文件哈希；验证并提交候选后，显式运行 `run <目录>`。冻结资料是本地验收输入，完整消息与响应只保存到被忽略的证据目录，不进入生产追踪。

该入口最多接受十份固定上下文，每份 A/B 各两次，第一轮 A→B、第二轮 B→A；总上限 40 次 SDK 请求、1200 秒，每次请求最多 120 秒，自动重试关闭。每次只替换 Instructions，按原预算器检查后发送并核对 HTTP 正文。失败占用原样本，不补跑；认证、记账或证据保存故障停止。已有账本禁止重启，未用额度关闭。结果保留未运行项、失败、实际用量与缓存字段；匿名输出文件供来源复核，但不自动给出语义通过结论。固定旧摘要不会替换成本次新摘要，因此此比较不验证连续压缩链、业务任务完成率或历史召回性能。

[逐组覆盖映射](COVERAGE.md) 说明原检查去向、模块职责和完整链路。添加检查时同步更新固定清单、稳定组 ID 与针对性故障测试，保留直接核对 SDK 请求、State、文件和检查点的断言。组开始/结束由显式记录上下文产生，正常清理和关联后置断言完成后才记录通过。

[项目任务说明](suites/wechat_reporting/README.md) 描述当前案例的 Research 目标、固定事实范围和机制条件。当前评测流程使用一个项目；更多项目可用于验证不同语言、规模与文档质量下的表现。当前任务不测试联网、代码执行修改、PDF 解析或跨进程恢复，也不把项目上的一次结果推广为通用能力。

工程约定集包含 17 条隔离行为链、155 个行为组。工具调用检查对照实际模型请求中的 `observation`：小结果保留原文，大结果验证有界片段及 Artifact 引用；原始文件 I/O 和读取额度仍独立核对。Artifact 召回工具默认注册，完整请求预算检查继续生效。详细覆盖见 [COVERAGE.md](COVERAGE.md)。

### 摘要 effort 与指令结构对照

`summary_prompt_comparison` 的 `freeze-effort` 接收已捕获的基线目录（`baseline.json`、`p0-manifest.json`、`p0-results.json`），将上一轮 B 冻结为 P0。P1 使用当前指令与首条范围说明标签，其余原生输入和风险保持不变。运行前须提交候选并校验来源哈希。

```powershell
.\.venv\Scripts\python.exe -m benchmarks.core.summary_prompt_comparison freeze-effort <baseline-directory> <new-output-directory>
.\.venv\Scripts\python.exe -m benchmarks.core.summary_prompt_comparison run <new-output-directory>
```

新实验交错运行 P0/high、P0/max、P1/high、P1/max；10 份固定上下文各重复两次，共 80 个请求槽。总上限 80 次 SDK 请求、4800 秒，每次 120 秒；不重试或续跑同一账本。旧 P0/none 仅作历史参考，其缓存与延迟不能视为同条件随机对照。输出上限包含推理和可见文本，原始 usage 保留供核对；缓存占比应按输入 token 加权计算。协议接受、语义质量与完整 Agent 任务完成分别评价，标签与大纲作为组合改动比较，不声称各自的独立贡献。
