# ErrorRecorder Snapshot Guide

ErrorRecorder snapshot 的目标是把 TensorRT 内部错误信息复制到托管侧安全结构，而不是把 TensorRT 内部对象或 borrowed pointer 暴露给 public API。

## 当前覆盖

- Runtime、Refitter、Builder、Engine、ExecutionContext、NetworkDefinition、EngineInspector 已提供 copied snapshot API。
- 这些 owner 的 snapshot 都只复制错误记录，不暴露 recorder 指针；`HasErrorRecorder` / `ClearErrorRecorder` 仍保持独立的安全边界。
- Direct `IErrorRecorder*` ownership、引用计数和 interface info 仍保持 deferred。

## Snapshot 内容

各 owner snapshot 会复制：

- error code。
- error description。
- error count。
- has overflowed 状态。

如果 TensorRT 返回的字符串生命周期不由调用方持有，native bridge 必须使用 caller buffer 或复制模式，不能直接返回悬空指针。

覆盖矩阵把 `IErrorRecorder::getNbErrors`、`getErrorCode`、`getErrorDesc`、`hasOverflowed` 和 TRT10+ 的 copied interface metadata 归并到各 owner snapshot 的安全替代实现；这只是表示只读诊断语义已通过 owner 对象复制出来，不表示 public API 可以取得或持有裸 `IErrorRecorder*`。

## 质量门禁

测试应覆盖：

- 无错误时返回空 snapshot。
- 有错误时复制 code 和 message。
- 对象释放后 snapshot 仍可读。
- 跨 ABI 不抛异常。

## 边界

ErrorRecorder snapshot 是诊断能力，不等同于 runtime proof。它帮助解释失败原因，但不能把失败或 blocked 状态写成通过。

## 设计门状态

`error-recorder-diagnostics-design-gate` 已把当前可用能力收口为 design gate：`CopiedDiagnosticsReady=True`、`PointerFreeSurfaceReady=True`、`RecorderPointerExposed=False`、`RefCountPublicOwnershipControl=False`、`RuntimeProofBlocked=True`。详见 [ErrorRecorder Diagnostics Design Gate](error-recorder-diagnostics-design-gate.md)。
