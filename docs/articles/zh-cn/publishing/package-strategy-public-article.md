# 包策略：从本地构建到公开消费验证

TensorRtSharp4.0 的包策略分成两层：托管 API 包和 runtime split 包。这样做是为了让 C# 项目引用清晰，同时让 CUDA/TensorRT/cuDNN 相关 native 资产按目标平台分发、审计和替换。

## 适合

- 准备用 NuGet 或 GitHub Packages 分发 TensorRtSharp4.0 的维护者。
- 需要理解 runtime split 包和 managed 包关系的用户。
- 负责 release close、post-publish verification 和 package-consumer-runtime proof 的发布负责人。

## 包结构

托管侧核心包面向 C# API：

```text
JYPPX.TensorRT.CSharp.API
JYPPX.CudaSharp
JYPPX.TensorRtSharp
```

runtime 侧关注平台和 SDK 组合，例如：

```text
win-x64-trt11.0-cuda13.2-cudnn9.22
```

本地可以用这些脚本收集和检查 runtime 资产：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Collect-SplitRuntimeAssets.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1
```

但本地 feed、direct `.nupkg` 和 ProjectReference 都只是开发便利，不是 post-publish proof。

## 真实 proof 输入

发布闭环现在以 owner 回填结果为准。关键输入文件是：

```text
artifacts/final-release/owner-external-proof-execution-result.input.template.json
artifacts/final-release/owner-external-proof-execution-result.input.json
```

其中每个 `resultInputs[]` 项至少要包含：

```text
resultInputId
packageIdentity.nupkgPath
packageIdentity.nupkgSha256
stdoutPath / stdoutSha256
stderrPath / stderrSha256
mergedTranscriptPath / mergedTranscriptSha256
validatorOutputPath / validatorOutputSha256
exitCode
passed
ownerReviewer
ownerReviewTimestampUtc
nonSubstituteConfirmations
```

导入和验证命令：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-OwnerExternalProofExecutionResult.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerExternalProofExecutionResultImport.ps1 -Strict
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-RealProofRecordCandidateFromOwnerResultImport.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-RealProofRecordCandidateFromOwnerResultImport.ps1 -Strict
```

## proof 边界

package-consumer-runtime proof 必须来自仓库外部的 clean consumer，并且使用公开包源。local feed、ProjectReference、direct `.nupkg`、candidate、draft、dashboard、dry-run、blocked-by-cuda-driver 和 build-only 都不能替代真实 proof。

## 配图建议

- 一张 managed package 与 runtime split package 的关系图。
- 一张 owner result input JSON 截图，突出 `resultInputs[]` 和 SHA256 字段。
- 一张 release evidence ladder，标出本地构建、candidate、strict validator 和 post-publish proof 的分层。

## 下一步

包策略收口后，继续执行 clean external consumer 与 post-publish verification。只有严格 validator 接受真实 owner result 后，才能把候选记录交给 release close bridge；文章、截图和 dashboard 只能作为说明材料，不能单独推动发布。
