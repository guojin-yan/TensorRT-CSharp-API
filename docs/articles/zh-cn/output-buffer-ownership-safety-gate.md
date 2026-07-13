# OutputBuffer Ownership Safety Gate

> 状态：design-gate / safety-gate-ready
> readiness marker：`output-buffer-ownership-safety-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：进入真实 `IOutputAllocator::reallocateOutput` runtime proof 前，先固定 output buffer ownership、`currentMemory` reuse 和 borrowed/owned device pointer 规则。

## 目标

`output-buffer-ownership-safety-gate` 把 OutputAllocator 最危险的一段边界拆出来单独审计：TensorRT 会把 `currentMemory`、size、alignment 和 tensor shape 传给 `IOutputAllocator::reallocateOutput`，但 C# public API 不能把这些 native/device pointer 直接交给用户，也不能在 ownership 不清楚时让 borrowed pointer 逃逸。

公开 API：

- `TensorRtOutputBufferOwnershipSafetyGate`
- `TensorRtOutputBufferOwnershipSafetyGateResult`
- `Evaluate`

该 gate 只消费 copied `TensorRtOutputAllocatorCallbackOwnerSnapshot` 和可选的 `TensorRtOutputAllocatorAttachDetachDesignGateResult`。它不分配 device memory，不消费或返回 `currentMemory` 指针值，不返回 `IntPtr` / `nint` / native owner pointer / output buffer pointer / device pointer。

## 当前能证明什么

`TensorRtOutputBufferOwnershipSafetyGate.Evaluate` 输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `AttachDetachDesignGateReady` | 来自 attach/detach gate | attach/detach lifecycle copied evidence 可供 ownership gate 消费。 |
| `OwnerDesignReady` | 来自 owner snapshot | 必须是干净的 `output-allocator-callback-owner-design` 证据。 |
| `PointerFreeSurfaceReady` | `True` | public API 不暴露 output buffer pointer 或 device pointer。 |
| `CopiedCurrentMemoryMetadataReady` | `True` when owner snapshot clean | 只复制是否存在 currentMemory 的元数据，不复制 pointer 值。 |
| `CopiedShapeMetadataReady` | `True` | output tensor name / shape rank 等元数据可诊断。 |
| `CopiedRequestMetadataReady` | `True` | size / alignment 请求元数据可诊断。 |
| `BorrowedPointerEscapeBlocked` | `True` | public API 没有 borrowed pointer 逃逸通道。 |
| `SafetyGateReady` | `True` | copied metadata 与 pointer-free surface 已可审计。 |
| `OutputBufferOwnershipRuntimeReady` | `False` | 真实 output buffer ownership 还未 runtime proof。 |
| `CurrentMemoryReusePolicyReady` | `False` | `currentMemory` 复用策略未经过真实 TensorRT callback 证明。 |
| `OwnedDevicePointerReleasePolicyReady` | `False` | owned device pointer release policy 尚未实现。 |
| `ShapeNotificationOrderingReady` | `False` | `notifyShape` 先于 `reallocateOutput` 的真实调用顺序未实证。 |
| `ReallocateOutputRuntimeReady` | `False` | `IOutputAllocator::reallocateOutput` runtime callback 尚未实现。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=output-buffer-ownership-safety-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `SafetyGateReady=True`
- `OutputBufferOwnershipRuntimeReady=False`
- `CurrentMemoryReusePolicyReady=False`
- `BorrowedPointerEscapeBlocked=True`
- `OwnedDevicePointerReleasePolicyReady=False`
- `ShapeNotificationOrderingReady=False`
- `ReallocateOutputRuntimeReady=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 OutputAllocator output buffer ownership proof 仍被这些条件阻塞：

1. `currentMemory` reuse policy 尚未经过真实 TensorRT callback 验证。
2. owned device pointer release policy 尚未实现。
3. `notifyShape` 与 `reallocateOutput` 的真实调用顺序尚未实证。
4. `IOutputAllocator::reallocateOutput` runtime callback 尚未实现。
5. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

这意味着当前只能证明 public C# surface 不把 pointer 泄漏出去，不能证明真实 output buffer 分配、复用或释放已经安全。

## 与 Attach/Detach Gate 和 Precheck 的关系

[OutputAllocator Attach/Detach Design Gate](output-allocator-attach-detach-design-gate.md) 负责生命周期边界：`DetachClearControlAvailable=True`，但 `AttachControlAvailable=False`、`NativeVTableReady=False`。

本 gate 负责 output buffer ownership 边界：复制 `currentMemory` 是否存在、shape、size、alignment 等元数据，并把 ownership/runtime blocker 展开为独立字段。

[OutputAllocator Runtime Proof Precheck](output-allocator-runtime-proof-precheck.md) 会消费本 gate，并纳入：

- `OutputBufferOwnershipSafetyGateReady`
- `OutputBufferOwnershipRuntimeReady`
- `CurrentMemoryReusePolicyReady`
- `BorrowedPointerEscapeBlocked`
- `OwnedDevicePointerReleasePolicyReady`
- `ShapeNotificationOrderingReady`
- `ReallocateOutputRuntimeReady`

因此 precheck 可以区分 ownership safety gate 已可审计和真实 output buffer runtime ownership 尚未完成。

## 不能证明什么

该 safety gate 是 not proof。它不能解除以下 deferred rows：

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
