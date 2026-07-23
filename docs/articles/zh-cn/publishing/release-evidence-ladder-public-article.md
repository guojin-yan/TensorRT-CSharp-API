# Release Evidence Ladder：哪些证据能推动发布

TensorRtSharp4.0 的发布证据必须分层看待。README、模板、build report、readonly diagnostics、TensorRtExec GUI 截图、local feed consumer 都能帮助定位问题，但它们不能直接推动 release close。真正能推动发布闭环的，是 owner 可复核、机器可验证、来自公开包源和仓库外 clean consumer 的 runtime proof，再加上 post-publish verification 与 release close strict validator。

这篇文章把证据梯子讲清楚：哪些材料只是准备项，哪些材料可以作为候选 evidence，哪些材料能进入 package-consumer-runtime proof，哪些材料即使看起来“绿了”也必须继续保持 blocked/non-proof。

## 适合

- 准备执行发布候选审计的人。
- 需要理解 `package-consumer-runtime proof`、`real-model-runtime proof`、`post-publish proof` 区别的人。
- 想把 `applications/TensorRtExec`、`samples/YoloVision`、OnnxToEngine build report 和 owner proof 产物串起来的维护者。
- 负责判断 GitHub Actions dry-run、local feed、ProjectReference、direct `.nupkg` 是否能进入 release close 的发布负责人。

## 关键路径

- Owner input：`artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json`。
- Schema：`artifacts/final-release/package-consumer-runtime-proof-owner-input.schema.json`。
- Forbidden scan：`artifacts/final-release/package-consumer-runtime-proof-forbidden-substitute-scan.json`。
- Public docs gate：`artifacts/final-release/public-docs-package-metadata-gate.json`。
- Record validator：`eng/Test-PackageConsumerRuntimeProofRecord.ps1`。
- Owner input validator：`eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1`。
- Owner input import：`eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1`。
- Release close gate：`eng/Test-ReleaseIssueCloseRecord.ps1`。
- Final owner package：`artifacts/final-release/final-owner-execution-package.json`。

## 证据梯子

证据梯子建议这样理解：

| 层级 | 示例 | 能否晋级 proof | 常见误区 |
| --- | --- | --- | --- |
| 文档/模板 | owner input template、runbook、README、文章 | 否 | 模板字段齐全不等于 owner 已执行 |
| 预检 | dependency probe、preflight、blocked-by-cuda-driver 分类 | 否 | dependency-probe-only 只能解释环境 |
| build-only | OnnxToEngine build report、TensorRtExec build report、timing cache | 否 | engine 构建成功不等于 runtime smoke |
| readonly diagnostics | load-engine metadata、engine inspector text、copied snapshot | 否 | 只读元数据不代表 enqueue 或输出正确 |
| real-model runtime | YoloVision/OnnxToEngine 在真实模型上运行并有日志/hash | 可能 | 真实模型 proof 不能替代公开包 clean consumer |
| package-consumer runtime | 外部 clean consumer 从公开包源 restore/build/run，日志与 hash 通过 validator | 是，仍需 strict validator | local feed / ProjectReference / direct `.nupkg` 都不是公开包消费 |
| post-publish verification | 公开发布后重新 clean install/run、下载元数据、stdout/stderr/hash | 是，仍需 owner close decision | 公开下载 proof 不能单独替代 post-publish clean consumer |
| release close | strict validators + owner approval + rollback/final decision | 最终门 | validator ready 不等于 owner 已批准关闭 |

这条梯子最大的作用，是阻止“低层证据冒充高层 proof”。例如 TensorRtExec build report 很有价值，但它的位置在 build/report evidence；它不能跳过 clean consumer runtime proof。

## Owner Input 必填项

`package-consumer-runtime-proof-owner-input.schema.json` 当前记录 `fieldCount=80`、`requiredFieldCount=49`。关键必填字段包括：

- clean consumer：`cleanExternalConsumerRoot`、`consumerProjectPath`。
- 公开包源：`publicPackageSourceKind`、`publicPackageSource`、`publicPackageFeedUrl`。
- managed 包：`managedPackageUrl`、`managedPackageId`、`managedPackageVersion`、`managedNupkgPath`、`managedNupkgSha256`。
- runtime 包：`runtimePackageUrl`、`runtimePackageId`、`runtimePackageVersion`、`runtimePackageKey`、`runtimeNupkgPath`、`runtimeNupkgSha256`。
- host metadata：`ownerName`、`machineName`、`hostOs`、`hostArchitecture`、`gpuName`、`cudaDriverVersion`、`cudaRuntimeVersion`、`cudnnVersion`、`tensorRtVersion`、`tensorRtLine`。
- commands：`restoreCommand`、`buildCommand`、`smokeCommand`。
- runtime result：`exitCode`、`startedAtUtc`、`finishedAtUtc`、`dependencyProbeStatus`、`smokeStatus`、`nativeAssetsCopied`。
- logs：`smokeLogPath`、`smokeLogSha256`、`stdoutSummary`、`stderrSummary`、`failureDiagnostic`。
- side-effect guards：`performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false`、`canPromoteProof=false`。

这些字段不是为了增加表格复杂度，而是为了让发布负责人能回答三个问题：包来自哪里、在什么机器上运行、运行输出是否能被 hash 复核。

## Forbidden Substitutes

Forbidden substitute scan 会拒绝把非 proof 材料升级成 package-consumer-runtime proof。当前 scan 的 `scanState` 是 `blocked-forbidden-substitute-detected`，检测到的 blocker 包括：

- `repository path leakage`：clean consumer 路径落在仓库内或命令依赖源码路径。
- `template placeholder`：仍存在 `<owner-fill-...>` 这类占位值。

完整 forbidden substitutes 还包括：

- local feed。
- ProjectReference。
- direct `.nupkg`。
- build-only。
- dry-run。
- queued GitHub Actions run。
- missing self-hosted runner。
- GitHub Actions dry-run `.nupkg`。
- dashboard。
- GUI screenshot。
- TensorRtExec build report only。

这些项目被拒绝的原因都很具体。local feed 只能证明本地包布局，ProjectReference 只能证明源码树兼容，direct `.nupkg` 绕过公开 feed restore 语义，queued GitHub Actions 只是基础设施状态，GUI screenshot 不是机器可验证日志。

## GitHub Actions Dry-Run 的位置

GitHub Actions package dry-run 很有用，但它的位置在 dry-run/context evidence。owner input template 里已经明确：

```text
isDryRunOnly=true
isPublishedPackageProof=false
isPackageConsumerRuntimeProof=false
manualWorkflowDispatchNotPerformed=true
performsPublish=false
canPublishPublicly=false
canCloseReleaseIssue=false
canPromoteProof=false
```

dry-run 产出的 `.nupkg` SHA256 可以用于比较当前 head 与候选包，但它不是公开包下载 hash，也不是 post-publish hash。本月 GitHub Actions 额度用完时，更要保持这条边界：不要触发 workflow dispatch，不要把旧 run evidence 或 queued run 写成 proof。

## Package Consumer Runtime Proof

真正的 package-consumer-runtime proof 必须来自仓库外部 clean consumer，并使用公开包源。最小流程如下：

```powershell
dotnet new console -n TensorRtSharpConsumerProof
cd TensorRtSharpConsumerProof
dotnet add package JYPPX.TensorRT.CSharp.API --version <public-version> --source <public-feed>
dotnet add package <runtime-package-id> --version <public-version> --source <public-feed>
dotnet restore
dotnet build -c Release --no-restore
dotnet run -c Release -- --runtime-package-key <runtime-package-key>
```

owner 回填后，应先验证 owner input，再导入，再验证 proof record：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofRecord.ps1 -FailOnNotProof
```

只有当 exit code、smoke status、native asset copied、package hash、log hash、host metadata 和 forbidden substitute scan 都通过时，才可能进入 proof candidate。即便如此，它也只是 package-consumer-runtime lane 的 proof，不会自动完成 post-publish、Linux runner、real-model-runtime 或 release close lane。

## Post-Publish 与 Release Close

post-publish verification 发生在公开发布之后。它需要重新从公开渠道下载或 restore 包，记录下载元数据、hash、clean install/run 日志和 stdout/stderr。public package download proof 很重要，但它不能单独替代 post-publish clean consumer proof。

release close 还需要额外的 owner decision：

- owner authorization。
- package-consumer-runtime proof。
- Linux runner proof。
- real-model-runtime proof。
- post-publish verification。
- rollback review。
- final close decision。
- strict release issue close validator。

`final-owner-execution-package.json` 里大量 lane 都标记为 owner execution step 或 release-close hard gate；这些记录本身不发布、不关闭 issue、不提升 proof。它们是 owner runbook 和输入映射，不是最终批准。

## 当前 Gate 状态怎么读

`public-docs-package-metadata-gate.json` 当前 `gateState=blocked-owner-public-postpublish-proof-required`，同时 `failedBlockerCount=0`。这两个字段必须一起读：

- `failedBlockerCount=0`：说明公开文档和 package metadata 没有发现禁止性过度声明。
- `blocked-owner-public-postpublish-proof-required`：说明真实 owner/public/post-publish proof 仍缺失。

所以 `failedBlockerCount=0` 不是 ready-to-publish，也不是 release close。它只是“文档没有乱说话”的证据。

## 配图建议

- 一张 evidence ladder：template/preflight/build-only/readonly diagnostics 在底层，clean consumer runtime proof、post-publish、release close 在上层。
- 一张 forbidden substitutes 表，把 local feed、ProjectReference、direct `.nupkg`、dry-run、GUI screenshot、dashboard 标红。
- 一张 owner input 字段图，突出 public package source、nupkg SHA256、host metadata、smoke log SHA256。
- 一张 release lanes 图，分开 package-consumer-runtime、real-model-runtime、Linux runner、post-publish verification 和 release close。

## 下一步

先让 owner input schema、forbidden substitute scan 和 import chain 稳定，再由 owner 在兼容 CUDA/TensorRT 主机上回填真实 clean consumer 运行结果。当前阶段不要 push、不要 workflow dispatch、不要发布 NuGet/GitHub Packages、不要上传 GitHub Release；文档、模板、dry-run、dashboard、local feed、ProjectReference、direct `.nupkg` 和 build-only report 都只能作为说明或候选证据，不能推动 release close。
