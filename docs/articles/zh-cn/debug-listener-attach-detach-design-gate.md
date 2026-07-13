# DebugListener Attach/Detach Design Gate

> 状态：design-gate / design-gate-ready
> readiness marker：`debug-listener-attach-detach-design-gate`
> runtime evidence：`RuntimeEvidenceKind=design-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：进入真实 `IDebugListener::processDebugTensor` runtime proof 前的 attach/detach 生命周期门禁。

## 目标

`debug-listener-attach-detach-design-gate` 用于把 DebugListener 的 attach/detach 生命周期拆成可诊断字段，避免把 `setDebugListener(nullptr)` 清理能力、owner design snapshot 或 bridge-only wrapper surface 误当成真实 callback runtime。

公开 API：

- `TensorRtDebugListenerAttachDetachDesignGate`
- `TensorRtDebugListenerAttachDetachDesignGateResult`
- `Evaluate`

该 gate 不调用 TensorRT `setDebugListener` 的 non-null attach，不返回 raw `IntPtr` / `nint`、native listener pointer、borrowed debug tensor pointer、debug tensor data pointer 或 CUDA stream handle。

## 当前能证明什么

`TensorRtDebugListenerAttachDetachDesignGate.Evaluate` 只消费 copied `TensorRtDebugListenerCallbackOwnerSnapshot`，输出 pointer-free diagnostics：

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `LineSupportsDebugListener` | `True` for TRT10/TRT11 | TRT8 不具备相同 DebugListener attach 路径。 |
| `OwnerDesignReady` | 来自 owner snapshot | 必须是干净的 `debug-listener-callback-owner-design` 证据。 |
| `ManagedOwnerStateMachineReady` | `True` when dispose/release/unpin/drain evidence exists | 证明托管 owner 释放顺序可复制诊断。 |
| `DebugTensorMetadataCopied` | `True` when copied metadata exists | 只表示 name/type/location/shape 等元数据已复制。 |
| `PointerFreeSurfaceReady` | `True` | public API 不暴露 borrowed debug tensor pointer。 |
| `DetachClearControlAvailable` | `True` on TRT10/TRT11 | 表示已有 `setDebugListener(nullptr)` 清理控制。 |
| `AttachControlAvailable` | `False` | non-null listener attach bridge 尚未实现。 |
| `LineSpecificAttachDetachReady` | `False` | attach/detach 成对 readiness 仍被 non-null attach 阻塞。 |
| `StableNativeOwnerAddressReady` | `False` | native owner stable address 尚未实现。 |
| `NoThrowNativeVTableReady` | `False` | native no-throw vtable trampoline 尚未实现。 |
| `NativeVTableReady` | `False` | stable address 与 no-throw vtable 均未齐备。 |
| `BorrowedDebugTensorLifetimeReady` | `False` | borrowed debug tensor/data lifetime 规则尚未实现。 |
| `CanAttemptRuntimeProof` | `False` | 不允许进入真实 runtime proof。 |
| `RuntimeProofBlocked` | `True` | deferred rows 必须继续保留。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-attach-detach-design-gate`
- `RuntimeEvidenceKind=design-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `AttachControlAvailable=False`
- `DetachClearControlAvailable=True`
- `LineSpecificAttachDetachReady=False`
- `NativeVTableReady=False`
- `BorrowedDebugTensorLifetimeReady=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`

## 当前明确阻塞项

真实 DebugListener runtime proof 仍被以下条件阻塞：

1. line-specific `setDebugListener(non-null)` attach bridge 尚未实现。
2. native DebugListener owner stable address 尚未实现。
3. native `IDebugListener` no-throw vtable trampoline 尚未实现。
4. borrowed debug tensor pointer 与 data buffer lifetime 规则尚未实现。
5. full package consumer smoke 尚未输出完整 `EvidenceKind=real-callback-runtime`。

`setDebugListener(nullptr)` detach/clear 已有安全控制，但它只能清除可能存在的 listener，不能证明 non-null attach、native vtable 或真实 TensorRT callback 已经启用。

## 与 Runtime Proof Precheck 的关系

[DebugListener Borrowed Tensor Safety Gate](debug-listener-borrowed-tensor-safety-gate.md) 会消费本 gate 的 copied result，并把以下字段纳入 borrowed tensor/data lifetime 安全门禁：

- `AttachDetachDesignGateReady`
- `DebugTensorMetadataCopied`
- `PointerFreeSurfaceReady`
- `BorrowedDebugTensorPointerEscapeBlocked`
- `BorrowedDebugTensorLifetimeReady`
- `BorrowedDebugTensorDataLifetimeReady`
- `ProcessDebugTensorRuntimeReady`

该 `debug-listener-borrowed-tensor-safety-gate` 是 design gate，not proof；它只说明 borrowed pointer 不从 public API 逃逸，不能证明真实 TensorRT debug tensor/data lifetime。

[DebugListener Attach/VTable Safety Gate](debug-listener-attach-vtable-safety-gate.md) 会继续消费本 gate 与 borrowed tensor safety gate，并把 non-null attach、stable native owner address、no-throw vtable 和 exception-to-status mapping 展开成独立安全字段。该 `debug-listener-attach-vtable-safety-gate` 同样是 design gate，not proof；当前保持 `AttachControlAvailable=False`、`StableNativeOwnerAddressReady=False`、`NoThrowNativeVTableReady=False`、`ExceptionToStatusMappingReady=False` 和 `RuntimeProofBlocked=True`。

[DebugListener Runtime Proof Precheck](debug-listener-runtime-proof-precheck.md) 会消费本 gate 的 copied result，并把以下字段纳入 precheck：

- `AttachDetachDesignGateReady`
- `AttachControlAvailable`
- `DetachClearControlAvailable`
- `ManagedOwnerStateMachineReady`
- `LineSpecificAttachDetachReady`
- `StableNativeOwnerAddressReady`
- `NoThrowNativeVTableReady`
- `NativeVTableReady`
- `ExceptionToStatusMappingReady`
- `BorrowedDebugTensorLifetimeReady`
- `BorrowedDebugTensorDataLifetimeReady`
- `ProcessDebugTensorRuntimeReady`

因此 precheck 可以更具体地说明：当前不是“DebugListener 不存在”，而是 detach clear 已有安全控制，non-null attach、native vtable、borrowed tensor lifetime 与 full package consumer runtime evidence 仍缺失。

## 不能证明什么

该 design gate 是 not proof。它不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

只有 full package consumer smoke 输出完整 `real-callback-runtime` 字段，并且 readiness 将 `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true`，才能说明真实 TensorRT callback runtime 已触发。
