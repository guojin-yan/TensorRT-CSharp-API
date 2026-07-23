# Runtime Deserialization Deferred Boundary Audit

本审计收口 `runtime-deserialization-boundary` 设计组中仍处于 `deferred-only`
的 5 条候选：

| Interface | TRT line | Manifest | Decision |
| --- | --- | --- | --- |
| `IRuntime::deserializeCudaEngineV2` | 10 | `trt10-runtime-deserialize-cuda-engine-v2-deferred` | keep-deferred |
| `IRuntime::deserializeCudaEngineV2` | 11 | `trt11-runtime-deserialize-cuda-engine-v2-deferred` | keep-deferred |
| `IRuntime::loadRuntime` | 8 | `trt8-runtime-load-runtime-deferred` | keep-deferred |
| `IRuntime::loadRuntime` | 10 | `trt10-runtime-load-runtime-deferred` | keep-deferred |
| `IRuntime::loadRuntime` | 11 | `trt11-runtime-load-runtime-deferred` | keep-deferred |

## Boundary

`IRuntime::deserializeCudaEngine` 已由 scoped-buffer bridge 和 managed owner wrapper 覆盖，
但 `deserializeCudaEngineV2` 引入 stream reader / reader V2 callback state，`loadRuntime`
引入外部 lean runtime、plugin host code、returned runtime owner 和 library provenance。
这些不是 scalar getter 或 count/copy snapshot，不能通过 manifest alias 或 build-only smoke
晋级。

当前可用的安全替代面是：

- `TensorRtRuntimeDeserializationBoundaryPrecheck.EvaluateKnownSurface`
- `TensorRtRuntimeDeserializationDependencyDiagnostics.EvaluateKnownSurface`
- `TensorRtStreamIoInterfaceInfoDesignGate.EvaluateKnownSurface`

这些 surface 都是 pointer-free diagnostics/design gate，不暴露 `IRuntime*`、`ICudaEngine*`、
`IStreamReader*`、`IStreamReaderV2*`、`IStreamWriter*`、`IntPtr`、`nint` 或 `SafeHandle`。

## Promotion Prerequisites

在以下前置项全部闭环前，本组继续 deferred：

- bridge-owned returned runtime lifetime for `IRuntime::loadRuntime`
- plugin host-code and external lean runtime dependency diagnostics
- managed-owned stream reader/writer callback handles with no-throw native vtables
- detach-before-release ordering for any callback-owned serialization object
- serialized buffer copy or pin policy that cannot escape the native call
- full package consumer runtime smoke on a compatible NVIDIA driver/runtime host
- strict classification evidence that no local build-only or dependency-probe-only output is promoted as runtime proof

## Decision

本批不修改 native ABI surface，不新增 entrypoint，不删除 deferred manifest，不触发 GitHub Actions，
也不执行 NuGet/GitHub Packages/GitHub Release 发布。结论是
`runtime-deserialization-deferred-boundary-audit`：5 条 medium-risk 候选保持
`deferred-only`，现有 managed deserialization precheck 和 dependency diagnostics 只作为
source-quality/design-gate evidence。
