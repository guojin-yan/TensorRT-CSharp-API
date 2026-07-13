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

当前重点组合是 `win-x64-trt11.0-cuda13.2-cudnn9.22`。它已经完成 native build、runtime package、bridge consumer、full package consumer restore/build/native-copy 和本地 feed consumer；full runtime smoke 在当前机器被 CUDA error 35 阻塞，所以仍记录为环境 warning。

矩阵中的 `Runtime proof` 列来自 `runtimeProofStatus`。当前 Windows TRT11/CUDA13.2 行应显示 `blocked-by-cuda-driver`，这表示包和消费端链路已具备证据，但真实 runtime execution proof 仍需兼容 CUDA driver/runtime 环境补齐。

## 与 callback proof 的关系

matrix 只证明包线的构建和消费状态。`IDebugListener::processDebugTensor` 的真实回调 proof 仍以 full package consumer 的 `IsRealCallbackRuntimeProof=True` 且 `InvocationCount>0` 为唯一提升条件。`overallStatus=ready` 或 `ready-with-warnings` 都不能替代这个条件。
