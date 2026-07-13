# DebugListener Runtime Proof Precheck

> 状态：runtime-gate-precheck / precheck-ready
> readiness marker：`debug-listener-runtime-proof-precheck`
> runtime evidence：`RuntimeEvidenceKind=runtime-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：进入真实 `IDebugListener::processDebugTensor` runtime proof 前的前置条件检查。

## 目标

`debug-listener-runtime-proof-precheck` 消费 `debug-listener-callback-owner-design` 的 copied snapshot、`debug-listener-attach-detach-design-gate` 的 copied result、`debug-listener-borrowed-tensor-safety-gate` 的 copied result、`debug-listener-attach-vtable-safety-gate` 的 copied result、`debug-listener-native-attach-nothrow-preflight` 的 copied result、`debug-listener-native-owner-address-design-gate` 的 copied result、`debug-listener-native-nothrow-vtable-design-gate` 的 copied result、`debug-listener-native-attach-entry-design-gate` 的 copied result、`debug-listener-native-detach-before-release-design-gate` 的 copied result、`debug-listener-native-owner-lifecycle-dry-run` 的 copied result、`debug-listener-native-attach-entry-runtime-scaffold` 的 copied result、`debug-listener-native-owner-stable-identity` 的 copied result、`debug-listener-native-owner-noncopyable-storage` 的 storage-gate result、`debug-listener-native-nothrow-destructor` 的 destructor-gate result、`debug-listener-native-owner-lifecycle-gate` 的 lifecycle-gate result、`debug-listener-native-attach-bridge-shape-gate` 的 attach bridge shape gate result、`debug-listener-exception-status-mapping-gate` 的 exception/status gate result、`debug-listener-inflight-accounting-gate` 的 in-flight accounting gate result 和 `debug-listener-native-nothrow-vtable-scaffold-gate` 的 vtable scaffold gate result，输出下一阶段真实 DebugListener callback runtime proof 还缺哪些条件。

公开 API：

- `TensorRtDebugListenerRuntimeProofPrecheck`
- `TensorRtDebugListenerRuntimeProofPrecheckResult`
- `Evaluate`

该 precheck 不调用 TensorRT `setDebugListener`，不 attach 到 execution context，不经过 build/enqueue，也不触发真实 `IDebugListener::processDebugTensor`。它只报告 copied diagnostics 和阻塞项。

## 当前能证明什么

`TensorRtDebugListenerRuntimeProofPrecheck.Evaluate` 会检查：

| 字段 | 说明 |
| --- | --- |
| `OwnerDesignReady` | owner design snapshot 是干净的 `debug-listener-callback-owner-design` 证据。 |
| `DebugTensorMetadataCopied` | debug tensor name、类型、位置和 shape metadata 已被复制。 |
| `DisposeReleaseReady` | dispose 后 release hook、GCHandle/delegate unpin 和 in-flight drain 证据存在。 |
| `PointerFreeSurfaceReady` | public API 未暴露 debug tensor pointer 或 data pointer。 |
| `AttachDetachDesignGateReady` | attach/detach design gate 有足够 copied evidence 可供 precheck 消费。 |
| `AttachControlAvailable` | 当前固定为 `False`，non-null listener attach bridge 尚未实现。 |
| `DetachClearControlAvailable` | TRT10/TRT11 `setDebugListener(nullptr)` 清理控制可用。 |
| `ManagedOwnerStateMachineReady` | 托管 owner dispose/release/unpin/in-flight drain 证据干净。 |
| `StableNativeOwnerAddressReady` / `NoThrowNativeVTableReady` | 当前固定为 `False`，native owner 与 vtable 尚未实现。 |
| `ExceptionToStatusMappingReady` | 当前固定为 `False`，native vtable bridge 尚未证明 managed exception 到 status 的转换。 |
| `BorrowedTensorSafetyGateReady` | borrowed tensor safety gate 已有 copied evidence 可供 precheck 消费。 |
| `AttachVTableSafetyGateReady` | attach/vtable safety gate 已有 copied evidence 可供 precheck 消费。 |
| `NativeAttachNoThrowPreflightReady` | native attach/no-throw preflight 已有 copied evidence 可供 precheck 消费。 |
| `NativeOwnerAddressDesignGateReady` | native owner address design gate 已有 copied evidence 可供 precheck 消费。 |
| `NativeNoThrowVTableDesignGateReady` | native no-throw vtable design gate 已有 copied evidence 可供 precheck 消费。 |
| `NativeAttachEntryDesignGateReady` | native attach entry design gate 已有 copied evidence 可供 precheck 消费。 |
| `NativeDetachBeforeReleaseDesignGateReady` | native detach-before-release design gate 已有 copied evidence 可供 precheck 消费。 |
| `NativeOwnerLifecycleDryRunReady` | native owner lifecycle dry-run 已有 copied evidence 可供 precheck 消费。 |
| `NativeAttachEntryRuntimeScaffoldReady` | native attach entry runtime scaffold 已有 copied evidence 可供 precheck 消费。 |
| `NativeOwnerStableIdentityReady` | stable owner identity gate 已有 copied evidence 可供 precheck 消费；这不是 stable native address。 |
| `OwnerIdentityDiagnosticsReady` | copied owner id、last status、last diagnostic 和 release diagnostic 可审计。 |
| `OwnerIdentityPointerFree` | owner identity diagnostics surface 不暴露 raw pointer。 |
| `NativeOwnerNonCopyableStorageReady` | storage gate 已有 source-visible no-copy/no-move scaffold evidence 可供 precheck 消费。 |
| `NativeOwnerCopyBlocked` / `NativeOwnerMoveBlocked` | 当前为 `True`，native storage scaffold 删除 copy/move 构造与赋值。 |
| `NativeOwnerAddressExposed` / `NativeOwnerPointerProduced` | 当前为 `False`，storage gate 不暴露也不创建 native owner pointer。 |
| `NativeNoThrowDestructorGateReady` | destructor gate 已有 source-visible no-throw destructor scaffold evidence 可供 precheck 消费。 |
| `DestructorNoThrowScaffoldReady` / `DestructorExceptionEscapeBlocked` | 当前为 `True`，表示析构 scaffold 是 no-throw 且不会让异常跨 ABI。 |
| `DestructorAddressExposed` / `DestructorPointerProduced` | 当前为 `False`，destructor gate 不暴露地址也不产生 pointer。 |
| `NativeOwnerLifecycleGateReady` | lifecycle gate 已有 source-visible lifecycle scaffold evidence 可供 precheck 消费。 |
| `ManagedDisposeSnapshotReady` | dispose 后 copied snapshot 对 release hook、in-flight 和 pin 状态是干净的。 |
| `LifecycleScaffoldReady` | native lifecycle scaffold 已 source-visible。 |
| `ReleaseHookOrderingGateReady` / `DisposeIdempotencyGateReady` | release ordering 与 dispose idempotency 目前是 gate scaffold evidence。 |
| `InFlightDrainGateReady` | in-flight drain 目前是 gate scaffold evidence。 |
| `CallbackStateUnpinAfterDetachGateReady` / `DelegateUnpinAfterDetachGateReady` | post-detach unpin 目前是 gate scaffold evidence。 |
| `LifecycleAddressExposed` / `LifecyclePointerProduced` | 当前为 `False`，lifecycle gate 不暴露地址也不产生 pointer。 |
| `NativeAttachBridgeShapeGateReady` | attach bridge shape gate 已有 source-visible scaffold evidence 可供 precheck 消费。 |
| `AttachBridgeShapeReady` / `AttachBridgeNoThrowBoundaryReady` | 当前为 `True`，表示 attach bridge 参数形状与 no-throw/status boundary 已结构化。 |
| `AttachBridgeVersionGuardReady` / `AttachBridgeOwnershipDiagnosticsReady` | 当前为 `True`，表示 TRT10/TRT11 guard 与 ownership diagnostics 已结构化。 |
| `AttachBridgePointerFree` / `NonNullAttachStillDisabled` | 当前为 `True`，public API 不返回 pointer，且 non-null attach 仍被禁用。 |
| `ExceptionStatusMappingGateReady` | exception/status mapping gate 已有 source-visible scaffold evidence 可供 precheck 消费。 |
| `NativeCallbackExceptionCaptureReady` / `CallbackStatusMappingGateReady` | 当前为 `True`，表示 native exception capture 与 status mapping scaffold 已结构化。 |
| `ExceptionEscapeBlocked` / `DiagnosticCopyReady` | 当前为 `True`，表示异常不跨 ABI，diagnostic 只复制到 pointer-free status records。 |
| `InFlightAccountingGateReady` | in-flight accounting gate 已有 source-visible scaffold evidence 可供 precheck 消费。 |
| `CallbackEnterAccountingGateReady` / `CallbackLeaveAccountingGateReady` | 当前为 `True`，表示 copied diagnostics 能审计 enter/leave 和 drain。 |
| `CallbackInFlightNeverNegativeReady` / `ReleaseAfterDrainGateReady` / `CallbackStateUnpinAfterDrainGateReady` | 当前为 `True`，表示 copied in-flight counter、release-after-drain 和 unpin-after-drain gate 已结构化。 |
| `NativeNoThrowVTableScaffoldGateReady` | native no-throw vtable scaffold gate 已可供 precheck 消费。 |
| `NoThrowVTableScaffoldReady` / `VTableDestructorNoThrowReady` / `ProcessDebugTensorCallbackStubNoThrowReady` | 当前为 `True`，表示 vtable scaffold、destructor 和 callback stub 的 no-throw 形状已 source-visible。 |
| `VTableAddressExposed` / `VTablePointerProduced` | 当前为 `False`，vtable scaffold gate 不暴露地址也不产生 pointer。 |
| `AttachEntryParameterShapeReady` | 当前为 `True`，attach entry 参数 shape 已结构化，但不代表 native attach entry 已实现。 |
| `AttachEntryNoThrowBoundaryReady` | 当前为 `True`，C ABI no-throw/status mapping 预期已结构化。 |
| `AttachEntryOwnershipDiagnosticsReady` | 当前为 `True`，ownership diagnostics 已结构化且仍保持 pointer-free。 |
| `NativeAttachEntryLocated` | 当前固定为 `False`，non-null attach entry 尚未实现。 |
| `NativeDetachEntryLocated` | 当前为 `True`，detach/clear entry 已有 copied 证据。 |
| `LineSpecificAttachEntryDesignReady` / `AttachEntryNoThrowReady` | 当前固定为 `False`，line-specific attach entry 设计与 no-throw boundary 尚未实现。 |
| `AttachEntryVersionGuardReady` / `AttachEntryOwnershipReady` | 当前固定为 `False`，version guard 与 ownership contract 尚未实现。 |
| `DetachBeforeReleaseReady` | 当前固定为 `False`，detach-before-release ordering 尚未实现。 |
| `ReleaseHookOrderingReady` / `DisposeIdempotencyReady` | 当前固定为 `False`，release hook 顺序与 dispose 幂等性尚未实现。 |
| `InFlightDrainBeforeReleaseReady` | 当前固定为 `False`，release 前 in-flight callback drain 尚未实现。 |
| `CallbackStateUnpinAfterDetachReady` / `DelegateUnpinAfterDetachReady` | 当前固定为 `False`，callback state 和 delegate pinning 尚未保证 detach 完成后再释放。 |
| `StableNativeOwnerAddressDesignReady` | 当前固定为 `False`，stable owner address 尚未实现。 |
| `NativeOwnerNonCopyableReady` | 当前为 `True`，但只表示 source-visible storage scaffold 已阻止 copy/move；不代表 native owner lifecycle 完成。 |
| `NativeOwnerDisposeOrderReady` / `NativeOwnerReleaseHookReady` / `NativeOwnerInFlightDrainReady` | 当前固定为 `False`，native owner dispose/release/drain 生命周期尚未实现。 |
| `NoThrowNativeDestructorReady` | 当前为 `True`，只表示 source-visible no-throw destructor scaffold evidence ready，不表示 native owner lifecycle 完成。 |
| `NativeOwnerLifecycleReady` | 当前固定为 `False`，native owner attach/detach/release/drain 生命周期尚未实现。 |
| `NoThrowVTableDesignReady` / `ExceptionToStatusMappingDesignReady` | 当前固定为 `False`，native vtable no-throw 与 exception mapping 尚未实现。 |
| `NativeVTableTrampolineReady` / `CallbackExceptionCaptureReady` | 当前固定为 `False`，native vtable trampoline 与 callback exception capture 尚未实现。 |
| `CallbackStatusMappingReady` / `CallbackInFlightAccountingReady` | 当前固定为 `False`，callback failure status mapping 与 in-flight accounting 尚未实现。 |
| `CanImplementNativeAttach` | 当前固定为 `False`，不允许实施 non-null attach bridge。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | public API 阻止 borrowed debug tensor pointer 逃逸。 |
| `BorrowedDebugTensorDataLifetimeReady` | 当前固定为 `False`，debug tensor data buffer lifetime 尚未 runtime proof。 |
| `ProcessDebugTensorRuntimeReady` | 当前固定为 `False`，`IDebugListener::processDebugTensor` runtime callback 尚未实现。 |
| `BlockedPrerequisiteCount` | 仍阻塞真实 runtime proof 的前置项数量。 |
| `RuntimeProofBlocked` | 当前固定为 `True`，因为还没有真实 attach/vtable/lifetime/smoke 证据。 |

smoke 输出必须保留：

- `EvidenceKind=debug-listener-runtime-proof-precheck`
- `RuntimeEvidenceKind=runtime-gate`
- `RealCallbackRuntime=False`
- `IsRealCallbackRuntimeProof=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `AttachDetachDesignGateReady=True`
- `AttachControlAvailable=False`
- `DetachClearControlAvailable=True`
- `ManagedOwnerStateMachineReady=True`
- `LineSpecificAttachDetachReady=False`
- `StableNativeOwnerAddressReady=False`
- `NoThrowNativeVTableReady=False`
- `NativeVTableReady=False`
- `ExceptionToStatusMappingReady=False`
- `BorrowedTensorSafetyGateReady=True`
- `AttachVTableSafetyGateReady=True`
- `NativeAttachNoThrowPreflightReady=True`
- `NativeOwnerAddressDesignGateReady=True`
- `NativeNoThrowVTableDesignGateReady=True`
- `NativeAttachEntryDesignGateReady=True`
- `NativeDetachBeforeReleaseDesignGateReady=True`
- `NativeOwnerLifecycleDryRunReady=True`
- `NativeAttachEntryRuntimeScaffoldReady=True`
- `NativeOwnerStableIdentityReady=True`
- `OwnerIdentityDiagnosticsReady=True`
- `OwnerIdentityPointerFree=True`
- `NativeOwnerNonCopyableStorageReady=True`
- `NativeOwnerCopyBlocked=True`
- `NativeOwnerMoveBlocked=True`
- `NativeOwnerAddressExposed=False`
- `NativeOwnerPointerProduced=False`
- `NativeNoThrowDestructorGateReady=True`
- `DestructorNoThrowScaffoldReady=True`
- `DestructorExceptionEscapeBlocked=True`
- `DestructorAddressExposed=False`
- `DestructorPointerProduced=False`
- `NativeOwnerLifecycleGateReady=True`
- `ManagedDisposeSnapshotReady=True`
- `LifecycleScaffoldReady=True`
- `ReleaseHookOrderingGateReady=True`
- `DisposeIdempotencyGateReady=True`
- `InFlightDrainGateReady=True`
- `CallbackStateUnpinAfterDetachGateReady=True`
- `DelegateUnpinAfterDetachGateReady=True`
- `LifecycleAddressExposed=False`
- `LifecyclePointerProduced=False`
- `NativeAttachBridgeShapeGateReady=True`
- `AttachBridgeShapeReady=True`
- `AttachBridgeNoThrowBoundaryReady=True`
- `AttachBridgeVersionGuardReady=True`
- `AttachBridgeOwnershipDiagnosticsReady=True`
- `AttachBridgePointerFree=True`
- `NonNullAttachStillDisabled=True`
- `ExceptionStatusMappingGateReady=True`
- `NativeCallbackExceptionCaptureReady=True`
- `CallbackStatusMappingGateReady=True`
- `ExceptionEscapeBlocked=True`
- `DiagnosticCopyReady=True`
- `InFlightAccountingGateReady=True`
- `CallbackEnterAccountingGateReady=True`
- `CallbackLeaveAccountingGateReady=True`
- `CallbackInFlightNeverNegativeReady=True`
- `ReleaseAfterDrainGateReady=True`
- `CallbackStateUnpinAfterDrainGateReady=True`
- `NativeNoThrowVTableScaffoldGateReady=True`
- `NoThrowVTableScaffoldReady=True`
- `VTableDestructorNoThrowReady=True`
- `ProcessDebugTensorCallbackStubNoThrowReady=True`
- `VTableAddressExposed=False`
- `VTablePointerProduced=False`
- `NativeAttachEntryLocated=False`
- `NativeDetachEntryLocated=True`
- `AttachEntryParameterShapeReady=True`
- `LineSpecificAttachEntryDesignReady=False`
- `AttachEntryNoThrowReady=False`
- `AttachEntryNoThrowBoundaryReady=True`
- `AttachEntryVersionGuardReady=False`
- `AttachEntryOwnershipReady=False`
- `AttachEntryOwnershipDiagnosticsReady=True`
- `DetachBeforeReleaseReady=False`
- `ReleaseHookOrderingReady=False`
- `DisposeIdempotencyReady=False`
- `InFlightDrainBeforeReleaseReady=False`
- `CallbackStateUnpinAfterDetachReady=False`
- `DelegateUnpinAfterDetachReady=False`
- `StableNativeOwnerAddressDesignReady=False`
- `NativeOwnerNonCopyableReady=True`
- `NativeOwnerDisposeOrderReady=False`
- `NativeOwnerReleaseHookReady=False`
- `NativeOwnerInFlightDrainReady=False`
- `NoThrowNativeDestructorReady=True`
- `NativeOwnerLifecycleReady=False`
- `NoThrowVTableDesignReady=False`
- `ExceptionToStatusMappingDesignReady=False`
- `NativeVTableTrampolineReady=False`
- `CallbackExceptionCaptureReady=False`
- `CallbackStatusMappingReady=False`
- `CallbackInFlightAccountingReady=False`
- `CanImplementNativeAttach=False`
- `BorrowedDebugTensorPointerEscapeBlocked=True`
- `BorrowedDebugTensorLifetimeReady=False`
- `BorrowedDebugTensorDataLifetimeReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `FullPackageConsumerRuntimeEvidenceReady=False`

## 当前明确阻塞项

真实 DebugListener runtime proof 仍被这些条件阻塞：

1. line-specific execution context `setDebugListener(non-null)` attach bridge 尚未实现。
2. native `IDebugListener` owner stable address 与 no-throw vtable 尚未实现。
3. native DebugListener vtable 的 exception-to-status mapping 尚未证明。
4. `debug-listener-native-attach-nothrow-preflight` 已把 native attach/no-throw 设计 blocker 结构化，但 `CanImplementNativeAttach=False`。
5. `debug-listener-native-owner-address-design-gate` 已把 stable owner address、不可复制 owner、dispose order、release hook、in-flight drain 和 no-throw destructor blocker 结构化，但 `NativeOwnerLifecycleReady=False`。
6. `debug-listener-native-nothrow-vtable-design-gate` 已把 no-throw vtable、exception capture、status mapping 和 in-flight accounting blocker 结构化，但 `NativeVTableTrampolineReady=False`、`CallbackExceptionCaptureReady=False`、`CallbackStatusMappingReady=False` 和 `CallbackInFlightAccountingReady=False`。
7. `debug-listener-native-attach-entry-design-gate` 已把 line-specific attach entry、no-throw boundary、version guard、ownership contract 和 detach-before-release blocker 结构化，但 `NativeAttachEntryLocated=False`、`LineSpecificAttachEntryDesignReady=False`、`AttachEntryNoThrowReady=False`、`AttachEntryVersionGuardReady=False`、`AttachEntryOwnershipReady=False` 和 `DetachBeforeReleaseReady=False`。
8. `debug-listener-native-detach-before-release-design-gate` 已把 release hook ordering、dispose idempotency、in-flight drain 和 post-detach unpin blocker 结构化，但 `ReleaseHookOrderingReady=False`、`DisposeIdempotencyReady=False`、`InFlightDrainBeforeReleaseReady=False`、`CallbackStateUnpinAfterDetachReady=False` 和 `DelegateUnpinAfterDetachReady=False`。
9. `debug-listener-native-owner-lifecycle-dry-run` 已把 stable owner identity、non-copyable storage、release hook ordering、dispose idempotency、in-flight drain、post-detach unpin 和 no-throw destructor blocker 结构化，但 `NativeOwnerLifecycleReady=False`、`CanImplementNativeAttach=False`。
10. `debug-listener-native-attach-entry-runtime-scaffold` 已把 attach entry 参数 shape、TRT10/TRT11 version guard、no-throw/status mapping 和 ownership diagnostics 结构化，但 `NativeAttachEntryLocated=False`、`StableNativeOwnerIdentityReady=False`、`NativeOwnerNonCopyableReady=False`、`NoThrowNativeDestructorReady=False`。
11. `debug-listener-native-owner-stable-identity` 已把 owner id / diagnostic identity 做成 pointer-free evidence，但该 gate 自身仍保持 `NativeOwnerNonCopyableReady=False`。
12. `debug-listener-native-owner-noncopyable-storage` 已把 source-visible no-copy/no-move storage scaffold 做成 storage-gate evidence，因此 precheck 中 `NativeOwnerNonCopyableStorageReady=True`、`NativeOwnerCopyBlocked=True`、`NativeOwnerMoveBlocked=True`、`NativeOwnerNonCopyableReady=True`、`NativeOwnerAddressExposed=False` 和 `NativeOwnerPointerProduced=False`。
13. `debug-listener-native-nothrow-destructor` 已把 source-visible no-throw destructor scaffold 做成 destructor-gate evidence，因此 precheck 中 `NativeNoThrowDestructorGateReady=True`、`DestructorNoThrowScaffoldReady=True`、`DestructorExceptionEscapeBlocked=True`、`DestructorAddressExposed=False`、`DestructorPointerProduced=False` 和 `NoThrowNativeDestructorReady=True`；但这仍不代表 `NativeOwnerLifecycleReady=True`。
14. `debug-listener-native-owner-lifecycle-gate` 已把 source-visible lifecycle scaffold 证据做成 lifecycle-gate evidence，因此 precheck 中 `NativeOwnerLifecycleGateReady=True`、`ManagedDisposeSnapshotReady=True`、`LifecycleScaffoldReady=True`、`ReleaseHookOrderingGateReady=True`、`DisposeIdempotencyGateReady=True`、`InFlightDrainGateReady=True`、`CallbackStateUnpinAfterDetachGateReady=True`、`DelegateUnpinAfterDetachGateReady=True`、`LifecycleAddressExposed=False` 和 `LifecyclePointerProduced=False`；但这仍不代表 `NativeOwnerLifecycleReady=True`。
15. `debug-listener-native-attach-bridge-shape-gate` 已把 attach bridge 参数形状、TRT10/TRT11 version guard、no-throw boundary 和 ownership diagnostics 做成 pointer-free evidence；但 `SetDebugListenerNonNullEnabled=False`、`NativeAttachEntryLocated=False`，仍不代表可以 attach。
16. `debug-listener-exception-status-mapping-gate` 已把 exception capture、status mapping、exception escape blocking 和 diagnostic copy 做成 pointer-free evidence；但它不安装 native vtable。
17. `debug-listener-inflight-accounting-gate` 已把 callback enter/leave、in-flight drain、release-after-drain 和 unpin-after-drain 做成 pointer-free evidence；但它不证明 TensorRT 曾经调用 callback。
18. `debug-listener-native-nothrow-vtable-scaffold-gate` 已把 source-visible no-throw vtable scaffold、callback stub、exception/status mapping 和 in-flight accounting 汇总为 vtable scaffold gate；但 `NativeVTableDesignReady=False`、`NativeAttachEntryLocated=False`。
19. `debug-listener-borrowed-tensor-safety-gate` 已阻止 borrowed pointer 从 public API 逃逸，但 borrowed debug tensor pointer lifetime 尚未实证。
20. borrowed debug tensor data buffer lifetime 规则尚未由真实 TensorRT callback 证明。
21. `IDebugListener::processDebugTensor` runtime callback 尚未实现。
22. full package consumer smoke 尚未输出 `EvidenceKind=real-callback-runtime`。

因此：

- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `IsRealCallbackRuntimeProof=False`

## 不能证明什么

该 precheck 是 runtime gate precheck，not proof。它不能解除以下 deferred rows：

- `IDebugListener::processDebugTensor`
- `IOutputAllocator::notifyShape`
- `IOutputAllocator::reallocateOutput`
- `IGpuAllocator::allocate`
- `IGpuAllocator::free`
- `IGpuAllocator::deallocate`
- `IGpuAllocator::reallocate`
- `IGpuAsyncAllocator::allocateAsync`
- `IGpuAsyncAllocator::deallocateAsync`

`Test-PackageConsumer.ps1` 必须把 `debug-listener-runtime-proof-precheck` 归类为非 proof callback evidence。只有 full package consumer smoke 输出完整 `real-callback-runtime` 字段，并且 readiness 将 `realCallbackRuntimeEvidence.isRealCallbackRuntimeProof=true`，才能说明真实 TensorRT callback runtime 已触发。
