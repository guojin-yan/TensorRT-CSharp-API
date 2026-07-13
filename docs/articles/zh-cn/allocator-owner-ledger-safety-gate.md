# Allocator Owner Ledger Safety Gate

> 状态：safety gate
> readiness marker：`allocator-owner-ledger-safety-gate`
> evidence kind：`ledger-safety-gate`
> 当前结论：只证明 allocator owner ledger 的 copied diagnostics 已可汇总，不是真实 TensorRT callback runtime proof；this is not proof。

## 目标

`allocator-owner-ledger-safety-gate` 用于把 allocator owner 的几个安全前置条件合到一个 public、pointer-free 的结果里：

- 托管 owner keep-alive：`GCHandle`、delegate pinning、`InFlightCallbackCount`、`ReleaseHookCount`。
- native owner state ledger dry-run：`StateTransitionCount`、`LedgerAllocationCount`、`LedgerReleaseCount`、`LedgerFailureCount`。
- dispose release 诊断：`DisposeRequested`、`CallbackStatePinned=False`、`DelegatePinned=False`、`DisposeReleaseReady`。
- 下一阶段 runtime proof 阻塞项：attach/detach、device pointer ledger、stream/async、full package consumer smoke evidence。

该 gate 不调用 `setGpuAllocator`，不注册 native owner 到 TensorRT，不产生或返回 device pointer，也不解除任何 callback deferred row。

## Public API

当前 public surface：

- `TensorRtAllocatorLedgerSafetyGate`
- `TensorRtAllocatorLedgerSafetyGateResult`
- `TensorRtAllocatorLedgerSafetyGate.Evaluate`
- `TensorRtAllocatorLedgerSafetyGate.GetSnapshot`

`Evaluate` 会触发 internal sync allocator prototype 诊断，并尝试运行 native state ledger dry-run。native bridge 或 vendor runtime 不可用时，错误会复制到 `NativeLedgerDiagnostic`，不会升级为 real callback proof。

`GetSnapshot` 只复制当前托管 owner 生命周期状态，适合在 `Dispose` 后验证 release hook。它不会运行 native ledger。

## 输出字段

核心 marker 和字段：

- `EvidenceKind=allocator-owner-ledger-safety-gate`
- `RuntimeEvidenceKind=ledger-safety-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `ManagedKeepAliveReady`
- `DisposeReleaseReady`
- `NativeLedgerDesignReady`
- `PointerFreeSurfaceReady`
- `LineSpecificAttachDetachReady=False`
- `DevicePointerLedgerRuntimeReady=False`
- `StreamLifetimeReady=False`
- `FullPackageConsumerRuntimeEvidenceReady=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `BlockedPrerequisiteCount`

`ManagedKeepAliveReady` 和 `DisposeReleaseReady` 通常来自生命周期的不同时间点：前者说明 owner 仍可安全保持 callback state，后者说明 dispose 后 release hook 已释放 keep-alive handles。因此该 gate 会输出多条 smoke evidence，而不是把单个 snapshot 伪装成完整生命周期 proof。

## 与现有证据的关系

| 证据 | 作用 | 是否 proof |
| --- | --- | --- |
| `allocator-owner-dry-run-diagnostics` | 验证 managed handler skeleton、异常吞吐和计数。 | 否 |
| `allocator-owner-native-dry-run-controls` | 验证短生命周期 native diagnostic owner 可创建、复制状态、释放。 | 否 |
| `allocator-owner-state-ledger-dry-run-controls` | 验证 synthetic attach/allocation/release/detach intent 可以复制。 | 否 |
| `allocator-owner-internal-runtime-prototype` | 验证 internal prototype 的 pinning、in-flight counter、release hook 和 exception-to-status。 | 否 |
| `allocator-owner-ledger-safety-gate` | 汇总上述 copied diagnostics，列出 runtime proof 阻塞项。 | 否 |

readiness 中 `allocatorOwnerLedgerSafetyGate.isRealCallbackRuntimeProof=false` 必须保持。只有 full package consumer smoke 输出完整 `EvidenceKind=real-callback-runtime`、`RealCallbackRuntime=True` 和 required counters 后，才能考虑提升 `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true`。

## Deferred rows

在真实 runtime proof 前，以下 rows 必须继续保留 direct deferred：

- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`
