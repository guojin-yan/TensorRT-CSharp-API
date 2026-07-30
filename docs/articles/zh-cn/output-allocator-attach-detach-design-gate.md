# OutputAllocator Attach/Detach Design Gate

> 状态：design-gate / design-gate-ready
> readiness marker：`output-allocator-attach-detach-design-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：进入真实 `IOutputAllocator::notifyShape` / `IOutputAllocator::reallocateOutput` runtime proof 前的 attach/detach 生命周期门禁。

## 目标

`output-allocator-attach-detach-design-gate` 用于把 OutputAllocator 的 attach/detach 生命周期拆成可诊断字段，避免把 `setOutputAllocator(nullptr)` 清理能力、owner design snapshot 或 bridge-only wrapper surface 误当成真实 callback runtime。

公开 API：

- `TensorRtOutputAllocatorAttachDetachDesignGate`
- `TensorRtOutputAllocatorAttachDetachDesignGateResult`
- `Evaluate`

源码按职责拆分为两份：`TensorRtOutputAllocatorAttachDetachDesignGate.cs` 只拥有 `Evaluate`，
`TensorRtOutputAllocatorAttachDetachDesignGateResult.cs` 拥有 result constructor、公开属性、诊断和 `ToString`。
这只是源码归类，不改变 public surface、pointer-free 边界或 design-gate/non-proof 分类。

该 gate 不调用 TensorRT `setOutputAllocator` 的 non-null attach，不返回 raw `IntPtr` / `nint`、native allocator pointer、output buffer pointer、device pointer 或 CUDA stream handle。

## 当前能证明什么

`TensorRtOutputAllocatorAttachDetachDesignGate.Evaluate` 只消费 copied `TensorRtOutputAllocatorCallbackOwnerSnapshot`，输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `LineSupportsOutputAllocator` | `True` for TRT8/TRT10/TRT11 | 当前三条 TensorRT line 均有 OutputAllocator API support。 |
| `OwnerDesignReady` | 来自 owner snapshot | 必须是干净的 `output-allocator-callback-owner-design` 证据。 |
| `ManagedOwnerStateMachineReady` | `True` when dispose/release/unpin/drain evidence exists | 证明托管 owner 释放顺序可复制诊断。 |
| `PointerFreeSurfaceReady` | `True` | public API 不暴露 output buffer pointer 或 device pointer。 |
| `DetachClearControlAvailable` | `True` on TRT8/TRT10/TRT11 | 表示已有 `setOutputAllocator(nullptr)` 清理控制。 |
| `AttachControlAvailable` | `False` | non-null OutputAllocator attach bridge 尚未实现。 |
| `LineSpecificAttachDetachReady` | `False` | attach/detach 成对 readiness 仍被 non-null attach 阻塞。 |
| `StableNativeOwnerAddressReady` | `False` | native owner stable address 尚未实现。 |
| `NoThrowNativeVTableReady` | `False` | native no-throw vtable trampoline 尚未实现。 |
| `NativeVTableReady` | `False` | stable address 与 no-throw vtable 均未齐备。 |
| `OutputBufferOwnershipRuntimeReady` | `False` | output buffer ownership、current-memory reuse、borrowed/owned device pointer 规则尚未 runtime proof。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=output-allocator-attach-detach-design-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `AttachControlAvailable=False`
- `DetachClearControlAvailable=True`
- `LineSpecificAttachDetachReady=False`
- `NativeVTableReady=False`
- `OutputBufferOwnershipRuntimeReady=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 OutputAllocator runtime proof 仍被以下条件阻塞：

1. line-specific `setOutputAllocator(non-null)` attach bridge 尚未实现。
2. native OutputAllocator owner stable address 尚未实现。
3. native `IOutputAllocator` no-throw vtable trampoline 尚未实现。
4. output buffer ownership、current-memory reuse、borrowed/owned device pointer 规则尚未实现。
5. full package consumer smoke 尚未输出完整 `EvidenceKind=real-callback-runtime`。

`setOutputAllocator(nullptr)` detach/clear 已有安全控制，但它只能清除可能存在的 allocator，不能证明 non-null attach、native vtable 或真实 TensorRT callback 已经启用。

## 与 Runtime Proof Precheck 的关系

[OutputAllocator Runtime Proof Precheck](output-allocator-runtime-proof-precheck.md) 会消费本 gate 的 copied result，并把以下字段纳入 precheck：

- `AttachDetachDesignGateReady`
- `AttachControlAvailable`
- `DetachClearControlAvailable`
- `ManagedOwnerStateMachineReady`
- `LineSpecificAttachDetachReady`
- `StableNativeOwnerAddressReady`
- `NoThrowNativeVTableReady`
- `NativeVTableReady`
- `OutputBufferOwnershipRuntimeReady`

当前还新增了 [OutputBuffer Ownership Safety Gate](output-buffer-ownership-safety-gate.md)：它以 `output-buffer-ownership-safety-gate` 固定 `currentMemory` reuse、borrowed pointer escape、owned device pointer release、`notifyShape`/`reallocateOutput` ordering 和 `ReallocateOutputRuntimeReady=False` 等字段。precheck 可以更具体地说明：当前不是“OutputAllocator 不存在”，而是 detach clear 已有安全控制，non-null attach、native vtable、output buffer ownership runtime 与 full package consumer runtime evidence 仍缺失。

## 不能证明什么

该 design gate 是 not proof。它不能解除以下 deferred rows：

- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IDebugListener::processDebugTensor`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

只有 full package consumer smoke 输出完整 `real-callback-runtime` 字段，并且 readiness 将 `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true`，才能说明真实 TensorRT callback runtime 已触发。
