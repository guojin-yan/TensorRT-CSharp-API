# OutputAllocator Callback Owner Design

> 状态：owner-design-gate
> readiness marker：`output-allocator-callback-owner-design`
> runtime evidence：`RuntimeEvidenceKind=not-present`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：`IOutputAllocator::notifyShape` / `IOutputAllocator::reallocateOutput` 真实 callback 前的 owner 生命周期门禁。

## 目标

`output-allocator-callback-owner-design` 用于把 OutputAllocator 的高层 C# owner 形状固定下来。它把已有的 `TensorRtOutputAllocatorRuntimeGate` 与 native allocator owner state ledger dry-run 组合成一个 public wrapper：

- `TensorRtOutputAllocatorCallbackOwner`
- `TensorRtOutputAllocatorCallbackRequest`
- `TensorRtOutputAllocatorCallbackOwnerSnapshot`
- `RunDesignDiagnostic`

该 wrapper 只返回 copied snapshot，不返回 raw `IntPtr` / `nint`、native owner handle、borrowed TensorRT pointer、output buffer pointer 或 device pointer ownership。

## 当前能证明什么

`RunDesignDiagnostic` 会记录以下 copied diagnostics：

| 字段 | 说明 |
| --- | --- |
| `TensorName` / `ShapeRank` / `ShapeSummary` | 复制 output tensor name 与 shape metadata。 |
| `RequestedSize` / `Alignment` | 复制 `reallocateOutput` 请求形状。 |
| `NotifyShapeCount` / `ReallocateOutputCount` | synthetic notify/reallocate 门禁计数。 |
| `InFlightCallbackCount` / `MaxInFlightCallbackCount` | 托管 gate 的 callback in-flight 诊断。 |
| `StateTransitionCount` | native owner attach/detach intent 复制计数。 |
| `LedgerAllocationCount` / `LedgerReleaseCount` | synthetic output buffer ledger allocation/release intent 计数。 |
| `LedgerFailureCount` | ledger 失败计数。 |
| `NativeLedgerAvailable` | native bridge 可用时为 true；本机环境缺失时写入 diagnostic，不解释成 API 缺失。 |
| `OutputBufferPointerExposed` | 固定为 `False`。 |
| `OutputBufferPointerProduced` | 固定为 `False`。 |

smoke 输出必须保留：

- `EvidenceKind=output-allocator-callback-owner-design`
- `RuntimeEvidenceKind=not-present`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `CallbackKind=output-allocator-prototype`
- `NativeLedgerAvailable`
- `StateTransitionCount`
- `LedgerAllocationCount`
- `LedgerReleaseCount`
- `OutputBufferPointerExposed=False`
- `OutputBufferPointerProduced=False`

## 不能证明什么

该 owner design gate 不调用 TensorRT `setOutputAllocator`，不 attach 到真实 execution context，不经过真实 build/enqueue，也不执行 TensorRT vtable callback。

因此它是 design gate，not proof，不能解除以下 deferred rows：

- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

## 下一步门禁

当前已新增 [OutputAllocator Attach/Detach Design Gate](output-allocator-attach-detach-design-gate.md)：它以 `output-allocator-attach-detach-design-gate` 输出 `RuntimeEvidenceKind=design-gate`、`AttachControlAvailable=False`、`DetachClearControlAvailable=True`、`LineSpecificAttachDetachReady=False`、`NativeVTableReady=False`、`OutputBufferOwnershipRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，用于固定 non-null attach 前的生命周期门禁。它仍是 not proof。

下一层是 [OutputBuffer Ownership Safety Gate](output-buffer-ownership-safety-gate.md)：它以 `output-buffer-ownership-safety-gate` 复制 `currentMemory` 是否存在、shape、size、alignment 等 metadata，并固定 `CurrentMemoryReusePolicyReady=False`、`BorrowedPointerEscapeBlocked=True`、`OwnedDevicePointerReleasePolicyReady=False`、`ShapeNotificationOrderingReady=False` 和 `ReallocateOutputRuntimeReady=False`。它只证明 ownership safety gate 可审计，不证明真实 `IOutputAllocator::reallocateOutput` 已运行。

当前还包括 [OutputAllocator Runtime Proof Precheck](output-allocator-runtime-proof-precheck.md)：它以 `output-allocator-runtime-proof-precheck` 消费 attach/detach design gate，并输出 `RuntimeEvidenceKind=runtime-gate`、`AttachDetachDesignGateReady=True`、`NativeLedgerDesignReady`、`AttachControlAvailable=False`、`DetachClearControlAvailable=True`、`NativeVTableReady=False`、`DevicePointerLedgerRuntimeReady=False`、`OutputBufferOwnershipRuntimeReady=False`、`CanAttemptRuntimeProof=False` 和 `RuntimeProofBlocked=True`，用于固定进入真实 runtime proof 前的阻塞项。它仍是 not proof。

进入真实 OutputAllocator runtime proof 前仍必须完成：

1. line-specific execution context attach/detach。
2. native owner stable address 与 no-throw vtable。
3. TensorRT 可能保存 callback pointer 时的 dispose 顺序。
4. output buffer device pointer ledger 的 owned/borrowed 区分。
5. `notifyShape` 与 `reallocateOutput` exception-to-status 映射。
6. full package consumer smoke 由真实 TensorRT build/enqueue 触发 callback，并输出 `EvidenceKind=real-callback-runtime`。

在这些条件完成前，readiness 中 `outputAllocatorCallbackOwnerDesign.isRealCallbackRuntimeProof=false`，`realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=false`。
