# 发布前 stale claim 自查

发布前最危险的问题不一定是 build 失败，而是文档、报告或清单把“准备动作”写成“已经通过”。TensorRtSharp4.0 已经有大量 evidence 文件、runbook、collection bundle、sample manifest 和工具报告；它们让项目更可审计，但也更容易出现 stale claim。

本文给出一套发布前自查方法，目标是把过度声明挡在发布之前：`blocked-by-cuda-driver` 不是 smoke passed，`build-only` 不是 inference proof，`ready-needs-manual-approval` 不是 public release approval，`package-consumer-runtime` belongs to release proof records。

## stale claim 是什么

stale claim 指文档或产物中的声明已经超过实际证据等级。例如：

- 把 `blocked-by-cuda-driver` 写成 runtime smoke passed。
- 把 TensorRtExec build report 写成真实模型推理通过。
- 把 sample-run-evidence 写成 `package-consumer-runtime`。
- 把 runbook 或 collection bundle 写成外部主机 proof。
- 把 `ready-needs-manual-approval` 写成 public release approved。
- 把 template/example/draft 写成 release owner 的真实输入。
- 把 `parse-only` 参数覆盖写成真实 TensorRT 行为已经实现。
- 把 evidence sidecar 写成 runtime proof。

这些问题不会总是让编译失败，但会破坏发布可信度。发布候选材料必须区分“能指导 owner 做什么”和“已经真实执行了什么”。

## 第一层：全仓关键词搜索

先从高风险词开始：

```powershell
rg -n "package-consumer-runtime|blocked-by-cuda-driver|Passed=True|ready-needs-manual-approval|build-only|dependency-probe-only|collection bundle|runbook|template-only|example-not-for-publication" .\docs .\samples .\applications .\eng .\tests
```

逐条检查上下文。出现这些词本身不是问题；问题是它们是否被用来声明“已通过”。

建议采用以下判断：

| 关键词 | 合理上下文 | 不合理上下文 |
| --- | --- | --- |
| `blocked-by-cuda-driver` | 环境兼容性 blocker、owner action、兼容主机待执行 | smoke passed、runtime proof 已完成 |
| `build-only` | parser/builder/build report 证据 | inference proof、real model proof、release proof |
| `parse-only` | CLI/parser/report 已接收参数 | native TensorRT 行为已经实现 |
| `sidecar-only` | build report 旁的模型来源、hash、license、日志摘要记录 | runtime proof、package consumer proof |
| `sample-run-evidence` | sample-level `real-model-runtime` 回填 | package consumer runtime proof |
| `collection bundle` | owner guidance、命令收集清单 | runtime execution evidence |
| `ready-needs-manual-approval` | dry-run 输出、等待 owner 审批 | public release approved |

## 第二层：运行 stale claim 审计脚本

项目已有脚本：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1
```

输出：

```text
artifacts/final-release/stale-release-claims-audit.json
artifacts/final-release/stale-release-claims-audit.md
```

脚本通过表示当前规则没有发现已知过度声明。它不是 runtime proof，也不是发布批准；它只是证明文本层面没有踩到当前审计规则。

如果脚本失败，不要删除规则来制造 green。应修正文档或产物，把不准确的 “passed/approved/proof” 改回真实状态。

## 第三层：检查 release readiness

缺少真实外部 runtime proof 时，以下命令应保留 blocker：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1
```

在当前主机上，预期状态是缺少真实 `artifacts/final-release/external-runtime-proof-record.json`，并且 `runtimeProofStatus=blocked-by-cuda-driver`。这不是失败的发布修饰词，而是真实的 release gate。

如果需要查看 dry run 汇总，可以运行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-FinalReleaseDryRun.ps1 -AllowRuntimeSmokeBlocked -WarnOnly
```

`-AllowRuntimeSmokeBlocked` 记录的是 dry-run 意图，不会把 blocker 改成 proof。

## 第四层：样例证据自查

Classification 和 YoloVision 的常见误写是把 asset template 或 build report 写成真实模型通过。正确路径应是：

1. TensorRtExec build-only report 只证明构建边界。
2. evidence sidecar 只补模型来源、hash、许可证和摘要。
3. sample-run-evidence 连接真实 runner log 和 manifest。
4. manifest audit 检查 hash、sampleName 和 evidence record。
5. user acceptance catalog 汇总 sample-level 状态。

只有真实模型、labels、输入图片、hash、许可证、runner log 和 validator 全部对齐后，才可以写 `real-model-runtime`。即使如此，它仍不是 `package-consumer-runtime`。

## 第五层：人工审阅句式

发布前建议优先替换这些危险句式：

| 不推荐 | 推荐 |
| --- | --- |
| “runtime proof 已通过，当前只是 driver blocker” | “runtime proof 仍被 `blocked-by-cuda-driver` 阻塞，需要兼容 host 回填” |
| “collection bundle 已完成 package proof” | “collection bundle 是 owner-action guidance，不是 runtime proof” |
| “build-only 报告证明模型可运行” | “build-only 报告证明 parser/builder 到达构建边界” |
| “parse-only 参数已经实现 TensorRT 行为” | “parse-only 表示 parser/report 已接收，真实 TensorRT 行为仍需 native 实现和 smoke” |
| “sidecar 证明 runtime 已通过” | “sidecar 只记录模型来源、hash、许可证和日志摘要，不能晋级 proof” |
| “ready-needs-manual-approval 表示可以发布” | “ready-needs-manual-approval 表示还需要 release owner 明确审批” |
| “sample-run-evidence 证明包消费端运行” | “sample-run-evidence 最多证明 sample-level real-model-runtime” |

## 发布前推荐清单

```powershell
rg -n "YoloVision|package-consumer-runtime|blocked-by-cuda-driver|ready-needs-manual-approval|Passed=True" .\README.md .\README.zh-CN.md .\docs .\samples .\applications .\eng

pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-StaleReleaseClaims.ps1

dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~TechnicalArticleRoadmap|FullyQualifiedName~ReleaseCandidateReadiness"
```

如果搜索命中旧 YOLO 项目名，要判断它是不是检测类型名或 stale guard 上下文。当前样例项目名应使用 `YoloVision`。

## 小结

stale claim 自查不是文案洁癖，而是发布可信度的一部分。真实项目可以有 blocker，可以有 owner action，可以有尚未完成的 external runtime proof；不能有把 blocker 写成通过的宣传材料。只要证据等级清楚，项目就能稳步接近发布，而不是靠删除风险提示制造表面完成度。
