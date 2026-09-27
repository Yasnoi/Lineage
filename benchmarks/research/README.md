# WeChat Reporting 研究评测

研究入口显式调用真实模型，用固定材料评估当前只读 Research 的回答与历史使用。它不测试代码修改、联网、PDF 解析或跨进程恢复，不与控制回归合成总分。

## 固定条件与运行

从项目根目录执行 `uv run --locked python -m benchmarks --suite research-wechat --research-profile wechat-v1`。默认从同级 `WeChat Reporting` 仓库读取以下 Git 对象，既不切换分支也不读取未提交修改：

- 代码：`214b38ef4d813c7ae7ab2defb5406bc4165e8b9b`，50 个白名单文件。
- 资料：`917efa1c4355f741486265ecd3d33db3cb0982e3`，三个 Markdown 文件及两份 PDF 的核对转写。

白名单和原路径见 [materials.py](materials.py)，转写见 [transcripts](transcripts/)。准备结果位于 `logs/benchmark-materials/<批次ID>/`；每个样本复制到自己的 `materials/material/`，这是唯一的 Agent 项目根目录。`manifest.json` 保留源提交、源文件、Git blob、材料路径、字节数与 SHA-256。`SOURCE_INDEX.md` 是中性来源索引。摘要例额外包含明确标识为评测构造的 12 KiB 背景文件。验收说明、评分规则与参考事实不在此目录中。

`wechat-v1` 只接受 `deepseek-flash` 和 `https://api.deepseek.com`。连接值来自进程环境或项目 `.env`（环境优先）；只向研究子进程传递必要连接字段与凭据，禁止外部 Tracing。固定数值配置不从 `.env` 扩大：每轮最多八次业务调用，请求超时 60 秒、上下文额度 32768、业务输出上限 4096；各例覆盖设置见 [definitions.py](definitions.py)。应用上下文计量不是提供商 tokenizer，不能用它估算账单。

每批先运行六个试运行样本，再以固定题目顺序重复三次，正式分母为 18 个样本。最多 56 个计划轮次、400 次请求开始记账、120 分钟累计执行；单样本 300 秒。业务、摘要及失败请求均占用预算，不自动重试。不保证预算内全部样本都能完成。试运行只核对执行链路与记录完整性；错误回答不会被筛掉，改变定义或评分必须保留旧版结果。

## 用例与机制

| 用例 | 轮次 | 观察范围 |
| --- | ---: | --- |
| wechat-capabilities | 1 | 离线采集、签名权威性与模板范围 |
| wechat-security | 1 | 加密、脱敏要求的确认依据与实现 |
| wechat-capacity-ocr | 2 | 容量证据、OCR、来源追问与验证建议 |
| wechat-correction | 3 | 从最终需求切回模板承诺，继承范围与格式纠正 |
| wechat-read-limit | 2 | 4 KiB 单文件限额导致真实读取失败后的判断 |
| wechat-compaction | 5 | 摘要后旧事实、来源与有效约束的使用 |

后续轮次仅在前轮正常完成且会话检查点确认后继续。Agent 错误或额度终态独立保存，其依赖轮次未执行；它们不能被改写成模型回答质量通过。

读取失败必须由实际工具结果证实。摘要例使用原签名规则源码，第一轮以后不允许再次读材料；第四轮按固定预算公式产生最多 16 KiB 的无事实填充。框架核对摘要来源、保存与活动版本、后续 SDK 输入、原文退出及工具调用情况。近期回答有无复述关键事实须由用户判断；结构证据通过不能代替语义评审。未发生摘要、发生重读或证据不足时记录覆盖缺口，不临时提高预算帮助通过。

## 产物与人工评分

批次与每个样本都有独立运行 ID，结果位于 `logs/benchmarks/<ID>/`。批次的计划先于执行保存；样本保留原始回答、实际提示、逐轮过程摘要、安全生产日志、请求与工具观测、worker 回执，以及 JSON/Markdown 结果。完整 SDK 请求只在进程内用于机制比对，不保存完整请求、State 或原始响应。摘要文档与必要安全引用作为明确证据另存。

`review.md` 汇集回答与证据入口，`review-template.json` 是人工评审表。保留绑定字段，填写评审者、每个适用条件的判定及具体依据；不能仅因存在引用就判证据充分。条件、严重错误与原文定位定义在 [rubric.json](rubric.json)。未评分、证据不足和不适用分别表示；不由被测模型或脚本关键词替代人工判断。

填写后从项目根目录运行：

```powershell
uv run --locked python -m benchmarks --review-result 'logs/benchmarks/<批次ID>/result.json' --review-file '<已填写评审JSON路径>'
```

导入校验运行身份、回答摘要、材料和评分版本，修订写入新目录，原始执行记录不变。回答文件改变或引用不匹配时拒绝导入。待人工评分前只能称为待评审执行结果，不能宣称效果基线已经完成。

退出优先级延续 `2 > 1 > 3 > 0`：执行、证据、保存、预算停止或机制覆盖缺口为 2；有效质量失败为 1；正常执行待评审为 3；有效评审全部通过且无缺口为 0。试运行与正式样本分开报告，五维条件、Agent 终态、耗时和消耗分别呈现。三次重复仅支持这个固定项目上的小样本观察，不能证明普遍稳定性。

## 离线验证

`uv run --locked python -m unittest discover -s benchmarks/tests -v` 包括研究执行器、真实 SDK 参数原样转发、会话隔离、请求账本与保存故障、材料隔离、评分绑定和机制负例的确定性检查。它们不调用真实端点，也不构成 Research 效果结果。`uv run --locked python -m benchmarks --suite control-all` 继续执行原 138 个控制组。
