# Release Proof 不可替代清单

TensorRtSharp4.0 的发布材料已经非常丰富：runbook、template、collection package、input draft、build report、sidecar、sample evidence、release bundle 和 dashboard 都已经存在。材料多了以后，最重要的是防止它们互相越级。本文列出常见“看起来像 proof，但不能替代 proof”的材料。

## 总原则

Release proof 必须满足四个条件：

1. 真实执行。
2. 真实环境。
3. 真实日志和 hash。
4. 对应 validator 通过。

缺任意一项，都只能作为 guidance、precheck、diagnostics 或 handoff。

## 统一非替代清单

所有 release-facing 脚本、artifact 和文档必须至少列出并拒绝以下 non-substitute proof kinds：

- template
- draft
- runbook
- collection package
- input package
- local feed
- ProjectReference
- build-only
- parse-only
- sidecar-only
- dependency-probe-only
- Skipped=True
- blocked-by-cuda-driver
- bridge-only package consumer log
- bridge-only wrapper surface
- WrapperSurfaceEvidenceKind=compile-surface-proof
- IsRuntimeExecutionProof=False
- mismatched log SHA256
- Parser/ParserRefitter diagnostic snapshots
- copied managed diagnostic snapshot
- managed-readiness
- managed-readiness-only
- callback-allocator-readiness-snapshot
- CallbackAllocatorReadinessSnapshot
- TensorRtCallbackAllocatorReadinessSnapshot
- precheck-only
- dry-run-only
- schema-only
- template-only release issue close record
- schema-only release issue close record
- preflight-only release issue close record
- release-issue-close-record-template.json
- Windows handoff for Linux proof

这些条目可以出现在 runbook、diagnostic snapshot、wrapper surface、input draft 或 collection package 中，但它们都不能让 `canPromoteRuntimeProof`、`canPublishPublicly` 或 `canCloseReleaseIssue` 变为 true；在真实 close record 通过前，`canCloseReleaseIssue=false` 必须保持。

## 不可替代材料矩阵

| 材料 | 可以证明 | 不能替代 |
| --- | --- | --- |
| helper script | 命令入口存在 | owner authorization |
| template | 字段结构清楚 | 真实 proof record |
| draft | owner 填写起点 | post-publish verification |
| runbook | 执行顺序清楚 | runtime execution proof |
| collection package | 命令和输入聚合 | package-consumer-runtime |
| input package | 字段收集完整 | real-model-runtime |
| local feed | 本地包可 restore | 不能替代真实渠道 package proof |
| ProjectReference | 源码可构建 | 不能替代 package consumer proof |
| dependency probe | native dependency 可诊断 | runtime smoke |
| managed-readiness | managed wrapper readiness 可聚合 | package-consumer-runtime 或 real callback runtime proof |
| CallbackAllocatorReadinessSnapshot | callback/allocator 边界准备状态可审计 | 真实 TensorRT callback invocation |
| precheck-only | 前置条件和阻塞项清楚 | smoke passed |
| dry-run-only | lifecycle 或 owner 设计可演练 | 兼容主机 runtime execution |
| schema-only | 字段结构和 marker 可约束 | 真实日志、hash 和 validator 通过 |
| template-only release issue close record | 最终关闭记录字段结构清楚 | release issue close proof |
| preflight-only release issue close record | 聚合 blocker 状态清楚 | owner final close decision |
| build-only | engine 构建路径可审计 | inference proof |
| parse-only | CLI 参数可解析和报告 | native TensorRT 行为已实现 |
| sidecar-only | 模型和报告 metadata 可追踪 | runtime proof |
| `blocked-by-cuda-driver` | 当前主机不兼容 | smoke passed |

## 具体风险

### build-only 被误写成 proof

TensorRtExec 的 build-only 报告非常有价值：它记录 ONNX、engine、shape profile、precision、workspace、normalized command 和 sidecar。但它不执行真实输入推理，也不验证输出语义。因此它不能替代 `real-model-runtime`，更不能替代 `package-consumer-runtime`。

### ProjectReference 被误写成 consumer proof

ProjectReference consumer 能证明源码工程配合运行，不证明 NuGet 包消费者路径。package consumer proof 必须在仓库外 clean consumer 中 restore/build/smoke，并记录 managed/runtime nupkg SHA256。

### managed-readiness 被误写成 runtime proof

`TensorRtCallbackAllocatorReadinessSnapshot` 和 `RuntimeEvidenceKind=managed-readiness` 能证明 managed 层聚合了 logger/profiler/progress monitor、allocator ledger、output allocator、debug listener 等 readiness，但它不表示 TensorRT 在兼容主机真实回调过托管对象。没有真实 package consumer runtime smoke、callback invocation、日志 hash 和 validator 通过时，它只能保留为 readiness evidence。

### runbook 被误写成 Linux runner proof

Windows 上生成 Linux handoff 或 runbook 可以帮助 owner 在 Linux 上执行命令，但 Linux proof 必须来自真实 Linux runner。`validationState=template-only` 不能写成 Linux runner proof。

### sidecar 被误写成 real-model-runtime

sidecar 可以记录模型来源、hash、license、TensorRtExec report 和 sample runner 期望字段。只有真实模型、真实输入、真实 sample log 和 sample-run-evidence validator 一起通过后，才可以晋级为 `real-model-runtime`。

### release issue close record 模板被误写成 close proof

`release-issue-close-record-template.json` 和 `release-issue-close-record-validation=blocked-template-only` 只能说明最终关闭记录的 schema 和 validator 存在。真实关闭前必须先让 release close preflight、post-publish verification、stale claim audit 和 evidence bundle hash 全部对齐，再由 owner 填入 rollback plan 与 final close decision，并通过 `Test-ReleaseIssueCloseRecord.ps1 -FailOnNotCloseReady`。缺任一项都不能让 `canCloseReleaseIssue=true`。

## 文档写法建议

推荐写法：

- “当前需要 owner 在兼容主机补齐 `package-consumer-runtime`。”
- “该报告是 build-only evidence，不是 release proof record。”
- “`blocked-by-cuda-driver` 表示当前主机无法执行 runtime smoke。”
- “post-publish verification 需要真实渠道发布后执行。”

避免写法：

- 把 runtime proof 写成 complete。
- 不要把当前状态写成公开发布就绪。
- 把 `package-consumer-runtime` 写成 passed。
- 把 `real-model-runtime` 写成 passed。
- 把 post-publish verification 写成 verified。
- 把 `release-issue-close-record-template.json` 写成 release issue 可关闭。
- 把 `build-only` 写成 release proof。
- 把 `sidecar-only` 写成 runtime proof。

## 和 stale claim audit 的关系

`Test-StaleReleaseClaims.ps1` 会扫描这些高风险句式。它不是文案审美工具，而是发布可信度门禁。新增文章、README、release note 或 issue 模板时，如果必须提到高风险短语，应把它们放在禁止示例、规则说明或边界说明里。

最终发布前，所有材料都应能回答同一个问题：这条声明背后是真实 proof，还是只是帮助 owner 执行 proof 的材料？
