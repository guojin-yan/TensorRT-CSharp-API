# Runtime Deserialization Dependency Diagnostics

`runtime-deserialization-dependency-diagnostics` 是 `runtime-deserialization-boundary-precheck` 之后的一层发布诊断。它把 managed `TensorRtRuntime.Deserialize(...)` 的安全边界、full package consumer report、dependency-probe-only 状态和 `blocked-by-cuda-driver` 分类放在同一个结构里，但它不是 runtime execution proof。

源码职责分为 `TensorRtRuntimeDeserializationDependencyDiagnostics.cs` 与
`TensorRtRuntimeDeserializationDependencyDiagnosticsResult.cs`：前者只拥有 evaluation 与 blocker helper，后者
拥有 result 构造、公开属性、分类逻辑、诊断和 `ToString`。该归类不改变 dependency-diagnostics/non-proof 边界。

## 输出字段

典型字段包括：

- `EvidenceKind=runtime-deserialization-dependency-diagnostics`
- `RuntimeEvidenceKind=dependency-diagnostics`
- `IsRuntimeExecutionEvidence=False`
- `IsRuntimeExecutionProof=False`
- `PrecheckReady=True`
- `ManagedDeserializeSurfaceReady=True`
- `FullPackageConsumerReportPresent`
- `FullPackageConsumerSmokeRequested`
- `FullPackageConsumerSmokeResult`
- `DependencyProbeOnly`
- `BlockedByCudaDriver`
- `DriverRuntimeMismatchClassified`
- `PackageConsumerEvidenceClassification`
- `RuntimeProofBlockerCategory`
- `RuntimeProofOwnerActionRequired`
- `ExternalRuntimeProofRequired`
- `PackageConsumerRuntimeProofPresent=False`
- `PluginLibraryDependencyDiagnosticsComplete=False`
- `LoadRuntimeOwnershipModeled=False`
- `WhyNotRuntimeProof`
- `NextOwnerAction`
- `CanAttemptRuntimeProof=False`
- `CanPromoteRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `DeferredRowsStillRequired=True`

这些字段帮助 release owner 看清楚：当前证据到底是 package layout、dependency probe、driver blocker，还是可晋级的 package-consumer-runtime proof。

其中 `PackageConsumerEvidenceClassification` 和 `RuntimeProofBlockerCategory` 是面向发布审查的归类字段。常见组合包括：

| 字段 | 典型值 | 含义 |
| --- | --- | --- |
| `PackageConsumerEvidenceClassification` | `dependency-probe-only` | 只证明 native dependency probe，不证明 TensorRT engine runtime execution。 |
| `PackageConsumerEvidenceClassification` | `runtime-smoke-driver-blocked` | smoke 已到达 CUDA runtime 边界，但被 driver/runtime 兼容性阻塞。 |
| `RuntimeProofBlockerCategory` | `full-package-consumer-report-missing` | 还没有 full package consumer report。 |
| `RuntimeProofBlockerCategory` | `runtime-smoke-not-requested` | 尚未请求 full package consumer smoke。 |
| `RuntimeProofBlockerCategory` | `cuda-driver-runtime-compatibility` | 需要在兼容 NVIDIA driver / CUDA runtime 的主机上重新跑 smoke。 |
| `RuntimeProofBlockerCategory` | `plugin-library-dependency-diagnostics-incomplete` | plugin/library dependency 诊断仍不完整。 |

`WhyNotRuntimeProof` 是给 release owner 的边界说明；`NextOwnerAction` 是下一步动作建议。它们都是 pointer-free 字符串诊断，不会调用 native runtime，也不会改变 `CanPromoteRuntimeProof=False`。

## 与 precheck 的关系

`runtime-deserialization-boundary-precheck` 证明的是 C# public surface 不暴露 serialized buffer、engine pointer 或 borrowed native pointer。`runtime-deserialization-dependency-diagnostics` 进一步把 full package consumer smoke 分类接进来，但仍不调用 `IRuntime::loadRuntime`，也不加载 plugin host code。

因此 direct `IRuntime::deserializeCudaEngine` 已由 scoped-buffer bridge 覆盖；`IRuntime::deserializeCudaEngineV2` 和 `IRuntime::loadRuntime` 行仍然保留 deferred。`PluginLibraryDependencyDiagnosticsComplete=False` 和 `LoadRuntimeOwnershipModeled=False` 是有意保留的发布边界。

## blocked-by-cuda-driver 怎么读

当 full package consumer smoke 输出 `blocked-by-cuda-driver` 时，说明 packaged runtime 已经走到 CUDA runtime 边界，但当前机器的 NVIDIA driver/runtime 组合不兼容目标 CUDA runtime。它不是：

- smoke passed。
- runtime execution proof。
- callback proof。
- API 缺失。
- 可以删除 deferred 行的依据。

正确的 owner action 是在兼容 CUDA driver/GPU host 上重新运行 package consumer smoke，并用 external runtime proof record 回填可验证证据。

## 建议命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22 -SplitPackageRole bridge
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-BridgePackageRuntimeConsumer.ps1 -SourceRuntimeKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ExternalRuntimeProofRecordTemplate.ps1 -RuntimePackageKey win-x64-trt11.0-cuda13.2-cudnn9.22
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1
```

只有兼容主机上的真实 package-consumer-runtime 证据可以推动 `CanPromoteRuntimeProof=True`。dependency diagnostics、precheck、build-only、dependency-probe-only 和 `blocked-by-cuda-driver` 都不能晋级。
