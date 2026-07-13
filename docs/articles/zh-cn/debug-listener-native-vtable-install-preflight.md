# DebugListener Native VTable Install Preflight

> 状态：native-vtable-install-preflight / native-vtable-install-preflight-ready
> readiness marker：`debug-listener-native-vtable-install-preflight`
> runtime evidence：`RuntimeEvidenceKind=native-vtable-install-preflight`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`

`debug-listener-native-vtable-install-preflight` 位于 borrowed debug tensor metadata gate 之后、runtime proof precheck 之前。它把 native owner lifecycle gate、attach bridge shape gate、native no-throw vtable scaffold gate 和 borrowed debug tensor metadata gate 汇总为一个安装前诊断面，用来回答“是否已经具备安装 native `IDebugListener` vtable 的形状、版本守卫、no-throw 边界和 pointer-free 证据”。

## Public Surface

- `TensorRtDebugListenerNativeVTableInstallPreflight`
- `TensorRtDebugListenerNativeVTableInstallPreflightResult`

关键字段：

- `NativeVTableInstallPreflightReady`
- `VTableInstallShapeReady`
- `VTableInstallVersionGuardReady`
- `VTableInstallNoThrowBoundaryReady`
- `VTableInstallOwnershipDiagnosticsReady`
- `VTableInstallPointerFree`
- `SetDebugListenerNonNullEnabled=False`
- `NativeVTableInstalled=False`
- `NativeVTableInstallRuntimeReady=False`
- `CanEnableSetDebugListenerNonNull=False`
- `CanInstallNativeVTable=False`
- `CanCallProcessDebugTensorRuntime=False`
- `RuntimeProofBlocked=True`
- `ReasonNativeVTableInstallStillBlocked`

## Native Scaffold

native 侧提供 `debug_listener_native_vtable_install_preflight.inc`，其中 `DebugListenerNativeVTableInstallPreflight final` 固定：

- copy/move deleted
- destructor noexcept
- `configure_api_line`
- `configure_preflight`
- `preflight_shape_ready`
- `version_guard_ready`
- `no_throw_install_boundary_ready`
- `ownership_diagnostics_ready`
- `pointer_free`
- `can_enable_set_debug_listener_non_null`
- `can_install_native_vtable`
- `native_vtable_installed`
- `process_debug_tensor_runtime_ready`

这些函数只提供 source-visible scaffold 和静态 no-throw 约束，不安装真实 vtable，也不返回 borrowed/native pointer。

## Smoke And Readiness

`CallbackAllocatorSafeControlsSmokeRunner` 输出：

- `DebugListenerNativeVTableInstallPreflight=...`
- `EvidenceKind=debug-listener-native-vtable-install-preflight`
- `RuntimeEvidenceKind=native-vtable-install-preflight`
- `NativeVTableInstallPreflightReady=True`
- `VTableInstallPointerFree=True`
- `SetDebugListenerNonNullEnabled=False`
- `NativeVTableInstalled=False`
- `CanInstallNativeVTable=False`
- `RuntimeProofBlocked=True`

`Test-RuntimePackageReadiness.ps1` 在 JSON/Markdown 中报告 `debugListenerNativeVTableInstallPreflight`，并继续要求 `IDebugListener::processDebugTensor` deferred row 保留。

## Boundary

该 preflight 是 native-vtable-install-preflight，not proof。它不能被解释为：

- `setDebugListener(non-null)` 已启用。
- native `IDebugListener` vtable 已安装。
- borrowed debug tensor 或 data pointer 生命周期已由真实 TensorRT callback 证明。
- `IDebugListener::processDebugTensor` 已经有 runtime callback invocation。
- `real-callback-runtime` 可以被 promotion。

下一阶段的受控 native owner/vtable install 实验入口见 [DebugListener Native Owner VTable Install Experiment](debug-listener-native-owner-vtable-install-experiment.md)。它仍必须让 public API 保持 pointer-free，并且只有 full package consumer smoke 产出 `RuntimeEvidenceKind=real-callback-runtime` 与 `IsRealCallbackRuntimeProof=True` 后，才允许把结果归类为真实 callback runtime proof。
