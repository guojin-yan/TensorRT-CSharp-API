# OutputAllocator Runtime Proof Precheck

> 状态：runtime-gate-precheck / precheck-ready
> readiness marker：`output-allocator-runtime-proof-precheck`
> runtime evidence：`RuntimeEvidenceKind=runtime-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：进入真实 `IOutputAllocator::notifyShape` / `IOutputAllocator::reallocateOutput` runtime proof 前的前置条件检查。

## 目标

`output-allocator-runtime-proof-precheck` 消费 `output-allocator-callback-owner-design` 的 copied snapshot、[OutputAllocator Attach/Detach Design Gate](output-allocator-attach-detach-design-gate.md) 的 copied result，以及 [OutputBuffer Ownership Safety Gate](output-buffer-ownership-safety-gate.md) 的 copied result，输出下一阶段真实 OutputAllocator callback runtime proof 还缺哪些条件。

公开 API：

- `TensorRtOutputAllocatorRuntimeProofPrecheck`
- `TensorRtOutputAllocatorRuntimeProofPrecheckResult`
- `Evaluate`

该 precheck 不调用 TensorRT `setOutputAllocator`，不 attach 到 execution context，不经过 build/enqueue，不返回 output buffer/device pointer，也不证明真实 `IOutputAllocator::notifyShape` 或 `IOutputAllocator::reallocateOutput` 已被 TensorRT 调用。它只报告 copied diagnostics 和阻塞项。

## 当前能证明什么

`TensorRtOutputAllocatorRuntimeProofPrecheck.Evaluate` 会检查：

| 字段 | 说明 |
| --- | --- |
| `OwnerDesignReady` | owner design snapshot 是干净的 `output-allocator-callback-owner-design` 证据。 |
| `NativeLedgerDesignReady` | native allocator owner ledger dry-run 证据存在且 allocation/release 配对干净；dependency-probe-only 环境可能为 false。 |
| `DisposeReleaseReady` | dispose 后 release hook、GCHandle/delegate unpin 和 in-flight drain 证据存在。 |
| `PointerFreeSurfaceReady` | public API 未暴露或产生 output buffer pointer。 |
| `AttachDetachDesignGateReady` | `output-allocator-attach-detach-design-gate` 已有可供 precheck 消费的 copied evidence。 |
| `DetachClearControlAvailable` | TRT8/TRT10/TRT11 已有 `setOutputAllocator(nullptr)` 清理控制。 |
| `AttachControlAvailable` | 当前固定为 `False`，non-null OutputAllocator attach bridge 尚未实现。 |
| `ManagedOwnerStateMachineReady` | 来自 attach/detach design gate，表示 dispose/release/unpin/drain 证据干净。 |
| `StableNativeOwnerAddressReady` / `NoThrowNativeVTableReady` | 当前固定为 `False`，native owner stable address 与 no-throw vtable 尚未实现。 |
| `NativeVTableReady` | 当前固定为 `False`，因为 native owner 与 no-throw vtable 均未齐备。 |
| `OutputBufferOwnershipSafetyGateReady` | `output-buffer-ownership-safety-gate` 已有可供 precheck 消费的 copied evidence。 |
| `OutputBufferOwnershipRuntimeReady` | 当前固定为 `False`，output buffer ownership、current-memory reuse、borrowed/owned pointer 规则尚未实现。 |
| `CurrentMemoryReusePolicyReady` | 当前固定为 `False`，`currentMemory` reuse policy 尚未由真实 TensorRT callback 证明。 |
| `BorrowedPointerEscapeBlocked` | 当前为 `True`，public API 没有 borrowed pointer 逃逸通道。 |
| `OwnedDevicePointerReleasePolicyReady` | 当前固定为 `False`，owned device pointer release policy 尚未实现。 |
| `ShapeNotificationOrderingReady` | 当前固定为 `False`，`notifyShape` / `reallocateOutput` ordering 尚未 runtime proof。 |
| `ReallocateOutputRuntimeReady` | 当前固定为 `False`，`IOutputAllocator::reallocateOutput` runtime callback 尚未实现。 |
| `BlockedPrerequisiteCount` | 仍阻塞真实 runtime proof 的前置项数量。 |
| `RuntimeProofBlocked` | 当前固定为 `True`，因为还没有真实 attach/runtime ledger/stream/smoke 证据。 |

smoke 输出必须保留：

- `EvidenceKind=output-allocator-runtime-proof-precheck`
- `RuntimeEvidenceKind=runtime-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `AttachDetachDesignGateReady=True`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `AttachControlAvailable=False`
- `DetachClearControlAvailable=True`
- `LineSpecificAttachDetachReady=False`
- `StableNativeOwnerAddressReady=False`
- `NoThrowNativeVTableReady=False`
- `NativeVTableReady=False`
- `DevicePointerLedgerRuntimeReady=False`
- `StreamLifetimeReady=False`
- `OutputBufferOwnershipSafetyGateReady=True`
- `OutputBufferOwnershipRuntimeReady=False`
- `CurrentMemoryReusePolicyReady=False`
- `BorrowedPointerEscapeBlocked=True`
- `OwnedDevicePointerReleasePolicyReady=False`
- `ShapeNotificationOrderingReady=False`
- `ReallocateOutputRuntimeReady=False`
- `FullPackageConsumerRuntimeEvidenceReady=False`

## 当前明确阻塞项

真实 OutputAllocator runtime proof 仍被这些条件阻塞：

1. line-specific execution context `setOutputAllocator(non-null)` attach bridge 尚未实现。
2. native OutputAllocator owner stable address 尚未实现。
3. native `IOutputAllocator` no-throw vtable trampoline 尚未实现。
4. runtime device pointer ownership ledger 尚未实现。
5. CUDA stream lifetime 和 async allocation semantics 尚未实现。
6. `output-buffer-ownership-safety-gate` 已能复制 `currentMemory`/shape/request metadata 并阻断 pointer 逃逸，但真实 output buffer ownership、current-memory reuse、owned device pointer release 和 `reallocateOutput` runtime callback 尚未实现。
7. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

TRT8/TRT10/TRT11 的 `setOutputAllocator(nullptr)` clear/detach 安全控制已可诊断，但它只能清理可能存在的 allocator，不能证明 non-null attach、native vtable 或真实 TensorRT callback 已经启用。

因此：

- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `IsRealCallbackRuntimeProof=False`

## 不能证明什么

该 precheck 是 runtime gate precheck，not proof。它不能解除以下 deferred rows：

- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

`Test-PackageConsumer.ps1` 必须把 `output-allocator-runtime-proof-precheck` 归类为非 proof callback evidence。只有 full package consumer smoke 输出完整 `real-callback-runtime` 字段，并且 readiness 将 `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true`，才能说明真实 TensorRT callback runtime 已触发。
