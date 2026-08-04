# Runtime Package Matrix

runtime package matrix 是发布候选的跨版本证据表。它把 TensorRT、CUDA、cuDNN、RID、package ID、validation state、build preset 和 consumer evidence 放在同一张表里，避免只看某一个 `.nupkg` 就误判整条发布线已经完成。

## 生成方式

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateReadiness.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -AllowRuntimeSmokeBlocked
```

输出：

- `artifacts/release-candidate/runtime-package-matrix.json`
- `artifacts/release-candidate/runtime-package-matrix.md`

## 状态含义

- `materialized-current`：当前 runtime key 在本机已有 package/readiness/consumer 证据。
- `materialized-historical`：manifest 标记为 `local-validated`，代表历史本地验证组合。
- `planned-dry-run`：Linux hosted 或 container 发布线已建模，但当前本机没有真实 native runtime 消费证据。
- `pending-local-validation`：包线存在，但还需要兼容环境完成 runtime smoke 或更完整验证。

当前重点组合是 `win-x64-trt11.0-cuda13.2-cudnn9.22`。它已经完成 native build、runtime package、bridge consumer、full package consumer restore/build/native-copy 和本地 feed consumer；full runtime smoke 在当前机器被 CUDA error 35 阻塞。发布时可以在 Owner 明确 opt-in 后将它作为“环境受限、运行未验证”的 warning 发布，但不能把它写成 runtime validated。

矩阵中的 `Runtime proof` 列来自 `runtimeProofStatus`。当前 Windows TRT11/CUDA13.2 行应显示 `blocked-by-cuda-driver`，这表示包和消费端链路已具备证据，但真实 runtime execution proof 仍需兼容 CUDA driver/runtime 环境补齐。`-AllowRuntimeSmokeBlocked` 只有在本页披露块完整时才会把门禁降为 warning。

## 发布必读披露

下面的固定字段由 release readiness validator 检查。它们是用户可见的限制声明，不是运行成功证明：

```text
runtimePackageKey=win-x64-trt11.0-cuda13.2-cudnn9.22
validationDisposition=build-package-validated-runtime-unverified
runtimeProofStatus=blocked-by-cuda-driver
isRuntimeExecutionEvidence=false
isDependencyProbeOnly=true
isRealCallbackRuntimeProof=false
vendorDependencies=user-installed
```

发布说明必须同时写明：

- TensorRT、CUDA、cuDNN 和 NVRTC 由使用者自行安装，项目包只包含项目自有 bridge。
- TensorRT 11.0 + CUDA 13.2 在当前验证主机未完成真实 runtime smoke；用户应使用匹配的 NVIDIA 驱动、CUDA、cuDNN 和 TensorRT 组合后自行复验。
- `native-copy`、dependency probe、build-only report 和本页矩阵都不能替代 runtime execution proof 或 callback proof。

## 与 callback proof 的关系

matrix 只证明包线的构建和消费状态。`IDebugListener::processDebugTensor` 的真实回调 proof 仍以 full package consumer 的 `IsRealCallbackRuntimeProof=True` 且 `InvocationCount>0` 为唯一提升条件。`overallStatus=ready` 或 `ready-with-warnings` 都不能替代这个条件。
