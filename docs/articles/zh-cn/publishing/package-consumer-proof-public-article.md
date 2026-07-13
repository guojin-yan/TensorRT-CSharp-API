# Package Consumer Runtime Proof：为什么本地包不能证明发布可用

发布一个 TensorRT C# 项目，最危险的误区是“仓库里能 build，所以 NuGet 包一定可用”。事实上，本地源码、ProjectReference、local feed 和 direct `.nupkg` 都会绕过真实用户会遇到的包解析、runtime asset 复制和依赖加载路径。

Package Consumer Runtime Proof 的目标就是补上这段证据。

## 适合谁阅读

- 准备公开发布 NuGet 包的 owner。
- 需要判断 local feed、ProjectReference、direct `.nupkg` 是否能作为 proof 的评审者。
- 负责外部 clean consumer 验证、smoke 和 release issue close 的维护者。

## 它证明什么

它证明一个干净外部 consumer 项目可以：

1. 从公开包源 restore 指定版本。
2. build 成功。
3. 复制 native runtime assets。
4. 在目标机器执行 dependency probe 或 smoke。
5. 产生 exitCode=0、stdout/stderr、host metadata 和验证记录。

这不是项目内部自测，也不是 dry-run。

## 禁止替代项

以下内容不能作为 package-consumer-runtime proof：

- local feed。
- ProjectReference。
- direct `.nupkg`。
- template。
- draft。
- dry-run。
- build-only report。
- TensorRtExec GUI 截图。
- YoloVision candidate template。

这些材料有价值，但它们属于开发证据、教程证据或准备材料，不是发布 proof。

## Owner 需要回填什么

owner 输入至少应包含：

- ownerName、machineName、gpuName。
- cudaDriverSupportedRuntime、cudnnVersion、tensorRtLine。
- restoreCommand、buildCommand。
- exitCode、startedAtUtc、finishedAtUtc。
- dependencyProbeStatus、smokeStatus、nativeAssetsCopied。
- stdout/stderr 摘要和 failureDiagnostic。

严格 validator 会检查这些字段是否真实、可解析、非 placeholder，并确保没有本地替代项。

## 导入与校验脚手架

Owner 拿到真实 clean consumer 结果后，不应该直接改 release proof record，而应该先导入 owner input：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PackageConsumerRuntimeProofOwnerInput.ps1 `
  -InputPath .\artifacts\final-release\package-consumer-runtime-proof-owner-input.template.json `
  -Strict
```

导入脚手架会生成：

```text
artifacts/final-release/package-consumer-runtime-proof-owner-input.imported.json
artifacts/final-release/package-consumer-runtime-proof-owner-input-import.json
artifacts/final-release/package-consumer-runtime-proof-owner-input-import.md
artifacts/final-release/package-consumer-runtime-proof-owner-input-validation.json
artifacts/final-release/package-consumer-runtime-proof-record.json
artifacts/final-release/package-consumer-runtime-proof-record-validation.json
```

这一步只复制、校验和投影 owner 输入；它不会执行 `dotnet nuget push`，不会关闭 release issue，也不会把 template、dry-run、build-only、local feed、ProjectReference 或 direct `.nupkg` 晋级为 proof。

## 与 YoloVision / TensorRtExec 的关系

YoloVision 可以产生 real-model-runtime proof 的候选材料；TensorRtExec 可以产生 build report 和 conversion diagnostics。但 package-consumer-runtime proof 必须来自外部 consumer 使用公开包。

## 配图建议

- 内部 build、real-model-runtime、package-consumer-runtime 三层证据对比图。
- clean consumer 项目的 restore/build/smoke 流程图。
- owner input JSON 字段标注图。

## 下一步

发布前应冻结公开包版本、hash、runtime package key 和 clean consumer 记录，再由 owner 手动执行发布或关闭 release issue。自动化脚本可以辅助验证，但不应替 owner 做公开发布决定。
