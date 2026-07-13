# ErrorRecorder Diagnostics Design Gate

`error-recorder-diagnostics-design-gate` 用来收口 `IErrorRecorder` 中风险 deferred 候选：项目已经可以通过 owner 对象复制诊断信息，但仍不能把原生 recorder 当作 public ownership 对象交给 C# 用户。

## 当前结论

- `RuntimeEvidenceKind=design-gate`。
- `CopiedDiagnosticsReady=True`，Runtime 和 Refitter 可以通过 `TensorRtErrorRecorderSnapshot` 读取 copied diagnostics。
- `RequiredOutputMode=owner-scoped copied diagnostics and interface metadata snapshot`。
- `CandidateMethods=IErrorRecorder::getInterfaceInfo, IErrorRecorder::getNbErrors, IErrorRecorder::getErrorCode, IErrorRecorder::getErrorDesc, IErrorRecorder::hasOverflowed, IErrorRecorder::incRefCount, IErrorRecorder::decRefCount`。
- `PointerFreeSurfaceReady=True`，public API 不暴露、返回或保存 recorder pointer。
- `RecorderPointerExposed=False`，`RecorderPointerProduced=False`，`BorrowedRecorderPointerEscaped=False`。
- `RefCountPublicOwnershipControl=False`，`InterfaceInfoPublicOwnershipControl=False`。
- `DirectRecorderOwnershipDeferred=True`，direct `IErrorRecorder*` ownership、ref-count 和 interface-info 行继续 deferred。
- `CanPromoteWithoutRuntimeProof=False`，`RuntimeProofBlocked=True`，`DeferredRowsStillRequired=True`，该门禁是 not proof。

## 已允许的 public 边界

Runtime 和 Refitter 使用 copied snapshot：

- `TensorRtRuntime.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)`。
- `TensorRtRefitter.TryGetErrorRecorderSnapshot(out TensorRtErrorRecorderSnapshot snapshot)`。
- `TensorRtErrorRecorderSnapshot.Records` 是托管集合，记录项为 `TensorRtErrorRecord`。

Builder、Engine、ExecutionContext、NetworkDefinition、EngineInspector 使用 presence / clear 边界：

- `HasErrorRecorder` 只返回是否附加 recorder。
- `ClearErrorRecorder()` 只清除 owner 上的外部 recorder 绑定，不销毁 recorder，不接管生命周期。

Plugin registry inventory 也只使用 copied/presence 语义，不暴露 registry 或 recorder borrowed pointer。

## 继续 deferred 的内容

以下内容不能因为 design gate ready 而晋级：

- `IErrorRecorder::incRefCount` / `decRefCount`。
- direct `IErrorRecorder::getInterfaceInfo` ownership。
- 任何 `IErrorRecorder*` 或裸 `IntPtr` public API。
- 任何需要用户持有 borrowed recorder lifetime 的 API。

这些接口如果后续要提升，必须先有独立 ownership 模型、跨 ABI no-throw 约束、release ordering 和 package-consumer runtime proof。

下一轮如果只能证明 owner-scoped snapshot，则应继续扩展 `TensorRtErrorRecorderSnapshot` 证据链，而不是新增 direct `IErrorRecorder` public wrapper。

## 验证信号

`CallbackAllocatorSafeControlsSmokeRunner --dependency-probe-only` 会输出：

- `SafeControlSurface=error-recorder-diagnostics-design-gate;...`
- `ErrorRecorderDiagnosticsDesignGate=error-recorder-diagnostics-design-gate;...`
- `CopiedDiagnosticsReady=True`
- `PointerFreeSurfaceReady=True`
- `RefCountPublicOwnershipControl=False`
- `RuntimeProofBlocked=True`

这些信号只说明 design gate 存在，不是 full runtime smoke passed，也不是 package-consumer runtime proof。
