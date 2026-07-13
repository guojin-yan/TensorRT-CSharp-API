# DebugListener Borrowed Debug Tensor Metadata Runtime Gate

> 状态：borrowed-debug-tensor-metadata-gate / borrowed-debug-tensor-metadata-gate-ready
> readiness marker：`debug-listener-borrowed-debug-tensor-metadata-runtime-gate`
> runtime evidence：`RuntimeEvidenceKind=borrowed-debug-tensor-metadata-gate`，`RealCallbackRuntime=False`，`IsRealCallbackRuntimeProof=False`
> 适用范围：在真实 native `IDebugListener` vtable 和 `processDebugTensor` runtime 尚未启用前，审计 borrowed debug tensor metadata copy、lifetime blocker 和 no-pointer-escape 边界。

## 目标

`debug-listener-borrowed-debug-tensor-metadata-runtime-gate` 消费 `debug-listener-borrowed-tensor-safety-gate` 和 `debug-listener-nothrow-vtable-callback-stub`。它把 DebugListener owner snapshot 中的 tensor name、data type、location、rank/shape、`isInput`、`isOutput`、`isShapeTensor`、`isExecutionTensor` 固定为 copied metadata evidence。

公开 API：

- `TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGate`
- `TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult`
- `Evaluate`

native source-visible gate：

- `native/src/tensorrt/common/debug_listener_borrowed_debug_tensor_metadata_runtime_gate.inc`
- `DebugListenerBorrowedDebugTensorMetadataRuntimeGate final`
- `copy_metadata`
- `metadata_copy_ready`
- `borrowed_tensor_pointer_escape_blocked`
- `borrowed_tensor_data_pointer_escape_blocked`
- `borrowed_tensor_lifetime_runtime_ready`
- `borrowed_tensor_data_lifetime_runtime_ready`
- `process_debug_tensor_runtime_ready`

## 当前能证明什么

| 字段 | 当前状态 | 说明 |
| --- | --- | --- |
| `BorrowedTensorSafetyGateReady` | `True` | 已消费 borrowed tensor safety gate。 |
| `CallbackStubGateReady` | `True` | 已消费 no-throw vtable callback stub gate。 |
| `MetadataGateReady` | `True` | copied metadata gate 可审计。 |
| `TensorNameCopied` | `True` | tensor name 已复制到托管字符串。 |
| `TensorNameLength` | `>0` | copied name 长度可诊断。 |
| `TensorTypeCopied` | `True` | data type enum 已复制。 |
| `TensorLocationCopied` | `True` | tensor location enum 已复制。 |
| `TensorShapeCopied` | `True` | rank/shape summary 已复制。 |
| `TensorShapeRank` | `>=0` | copied rank 可诊断。 |
| `TensorFlagsCopied` | `True` | input/output/shape/execution flags 已复制。 |
| `BorrowedDebugTensorMetadataCopyReady` | `True` | borrowed debug tensor 只以 metadata 形式进入 gate。 |
| `BorrowedDebugTensorPointerEscapeBlocked` | `True` | debug tensor pointer 不从 public API 逃逸。 |
| `BorrowedDebugTensorDataPointerEscapeBlocked` | `True` | data pointer 不从 public API 逃逸。 |
| `DebugTensorPointerExposed` | `False` | 不暴露 debug tensor pointer。 |
| `DebugTensorDataPointerExposed` | `False` | 不暴露 debug tensor data pointer。 |
| `BorrowedDebugTensorLifetimeReady` | `False` | 真实 TensorRT callback 生命周期仍未证明。 |
| `BorrowedDebugTensorDataLifetimeReady` | `False` | data buffer 生命周期仍未证明。 |
| `SetDebugListenerNonNullEnabled` | `False` | 不启用 `setDebugListener(non-null)`。 |
| `NativeVTableInstalled` | `False` | native IDebugListener vtable 未安装。 |
| `ProcessDebugTensorRuntimeReady` | `False` | 真实 runtime callback 仍未实现。 |
| `CanCallProcessDebugTensorRuntime` | `False` | 当前不能让 TensorRT 调用 callback。 |
| `CanAttemptRuntimeProof` | `False` | 当前不能启动 runtime proof。 |
| `RuntimeProofBlocked` | `True` | direct callback deferred rows 必须保留。 |

smoke/readiness marker 必须包含：

- `DebugListenerBorrowedDebugTensorMetadataRuntimeGate=`
- `TensorRtDebugListenerBorrowedDebugTensorMetadataRuntimeGateResult`
- `RuntimeEvidenceKind=borrowed-debug-tensor-metadata-gate`
- `MetadataGateReady=True`
- `TensorNameCopied=True`
- `TensorTypeCopied=True`
- `TensorLocationCopied=True`
- `TensorShapeCopied=True`
- `TensorFlagsCopied=True`
- `BorrowedDebugTensorMetadataCopyReady=True`
- `BorrowedDebugTensorPointerEscapeBlocked=True`
- `BorrowedDebugTensorDataPointerEscapeBlocked=True`
- `DebugTensorPointerExposed=False`
- `DebugTensorDataPointerExposed=False`
- `BorrowedDebugTensorLifetimeReady=False`
- `BorrowedDebugTensorDataLifetimeReady=False`
- `ProcessDebugTensorRuntimeReady=False`
- `CanCallProcessDebugTensorRuntime=False`
- `CanAttemptRuntimeProof=False`
- `RuntimeProofBlocked=True`
- `ReasonMetadataRuntimeStillBlocked`

## 不能证明什么

该 gate 是 borrowed-debug-tensor-metadata-gate，not proof。copied metadata 不等于 borrowed debug tensor lifetime ready；copied shape/name/type/location/flags 不等于 TensorRT 已经调用 `IDebugListener::processDebugTensor`；也不能解除 `IDebugListener::processDebugTensor` deferred row。

下一步如果要接近真实 runtime，必须先解决 native owner/vtable install、non-null attach、in-flight drain、borrowed tensor lifetime 和 full package consumer real callback smoke。
