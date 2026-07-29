# DebugListener Native Owner VTable Install Experiment

> 状态：native-owner-vtable-install-experiment / native-owner-vtable-install-experiment-ready
> readiness marker：`debug-listener-native-owner-vtable-install-experiment`
> runtime evidence：`RuntimeEvidenceKind=native-owner-vtable-install-experiment`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`

`debug-listener-native-owner-vtable-install-experiment` 位于 native vtable install preflight 之后、runtime proof precheck 之前。它把 native owner lifecycle、attach bridge shape、no-throw vtable scaffold、borrowed debug tensor metadata gate 和 native vtable install preflight 汇总为一个 disabled-by-default 的实验诊断面，用来回答“如果未来要尝试安装 native `IDebugListener` vtable，当前是否已经具备 shape、guard、rollback、detach-before-release、failure/status mapping 和 pointer-free 证据”。

## Public Surface

- `TensorRtDebugListenerNativeOwnerVTableInstallExperiment`
- `TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult`

evaluator 位于 `src/JYPPX.TensorRtSharp/Callbacks/Debugging/TensorRtDebugListenerNativeOwnerVTableInstallExperiment.cs`，
只读 result model 位于同目录的 `TensorRtDebugListenerNativeOwnerVTableInstallExperimentResult.cs`。完整 experiment
evidence consumer 必须读取两份源码。

关键字段：

- `ExperimentShapeReady`
- `InstallAttemptGuardReady`
- `NonNullAttachEnabled=False`
- `RuntimeProofEnabled=False`
- `NativeVTableInstallAttempted=False`
- `NativeVTableInstalled=False`
- `RollbackReady`
- `DetachBeforeReleaseReady`
- `FailureStatusMappingReady`
- `PointerFree`
- `CanEnableSetDebugListenerNonNull=False`
- `CanInstallNativeVTable=False`
- `CanCallProcessDebugTensorRuntime=False`
- `RuntimeProofBlocked=True`
- `ReasonNativeOwnerVTableInstallStillBlocked`

## Native Scaffold

native 侧提供 `debug_listener_native_owner_vtable_install_experiment.inc`，其中 `DebugListenerNativeOwnerVTableInstallExperiment final` 固定：

- copy/move deleted
- destructor noexcept
- `configure_api_line`
- `configure_prerequisites`
- `configure_disabled_reason`
- `experiment_shape_ready`
- `install_attempt_guard_ready`
- `non_null_attach_enabled`
- `native_vtable_install_attempted`
- `native_vtable_installed`
- `rollback_ready`
- `detach_before_release_ready`
- `failure_status_mapping_ready`
- `pointer_free`
- `process_debug_tensor_runtime_ready`
- `can_attempt_runtime_proof`

这些函数只提供 source-visible scaffold 和静态 no-throw 约束，不安装真实 vtable，也不返回 borrowed/native pointer。

## Smoke And Readiness

`CallbackAllocatorSafeControlsSmokeRunner` 输出：

- `DebugListenerNativeOwnerVTableInstallExperiment=...`
- `EvidenceKind=debug-listener-native-owner-vtable-install-experiment`
- `RuntimeEvidenceKind=native-owner-vtable-install-experiment`
- `ExperimentShapeReady=True`
- `InstallAttemptGuardReady=True`
- `NonNullAttachEnabled=False`
- `RuntimeProofEnabled=False`
- `NativeVTableInstallAttempted=False`
- `NativeVTableInstalled=False`
- `PointerFree=True`
- `CanInstallNativeVTable=False`
- `RuntimeProofBlocked=True`

`Test-RuntimePackageReadiness.ps1` 在 JSON/Markdown 中报告 `debugListenerNativeOwnerVTableInstallExperiment`，并继续要求 `IDebugListener::processDebugTensor` deferred row 保留。

## Boundary

该 experiment 是 native-owner-vtable-install-experiment，not proof。它不能被解释为：

- `setDebugListener(non-null)` 已启用。
- native `IDebugListener` vtable install 已尝试。
- native `IDebugListener` vtable 已安装。
- borrowed debug tensor 或 data pointer 生命周期已由真实 TensorRT callback 证明。
- `IDebugListener::processDebugTensor` 已经有 runtime callback invocation。
- `real-callback-runtime` 可以被 promotion。

下一阶段如果要继续推进，应先实现真实 non-null attach runtime smoke 的更强前置条件和回滚策略。只有 full package consumer smoke 产出 `RuntimeEvidenceKind=real-callback-runtime` 与 `IsRealCallbackRuntimeProof=True` 后，才允许把结果归类为真实 callback runtime proof。
