# Execution Context Auxiliary Stream 生命周期

`TensorRtExecutionContext.SetAuxStreams` 和 `ClearAuxStreams` 支持 TensorRT 8、10、11。调用方仍拥有传入的 `CudaStream`，但 TensorRT execution context 会在可能跨 enqueue 借用这些 stream 的整个期间持有对应 `SafeHandle` lease。

## 生命周期规则

- `SetAuxStreams` 拒绝 `null`、已释放、default/invalid 和重复 stream。
- 新 lease 会先逐个执行 `DangerousAddRef`；native set 成功后才替换旧 lease，失败会回滚新引用。
- 即使调用方在 assignment 存续期间对 `CudaStream` 调用 `Dispose`，native bridge stream 也会保持到 context clear 或 dispose 之后才真正销毁。
- `ClearAuxStreams` 只有在 native clear 成功后才释放旧 lease。
- `TensorRtExecutionContext.Dispose` 先尝试 native clear，再释放 native context，最后对 lease 执行 `DangerousRelease`；它不会替调用方 dispose `CudaStream` 对象。
- 调用方仍负责保证 enqueue 已完成并进行必要的 stream 同步，然后再 clear、替换 assignment 或释放 context。

## 安全边界

`GetAuxiliaryStreamAssignmentSnapshot()` 只返回版本线、已分配数量、clear 状态、managed lease 状态和文本诊断。`NativeStreamPointerExposed` 与 `BorrowedHandleEscaped` 恒为 `false`，public API 不返回 `IntPtr`、`nint`、`SafeHandle` 或 CUDA stream pointer。

native ABI 使用 count/array 输入，拒绝 default 和重复 native stream，并在 C++ exception 与 Windows SEH guard 内调用 TensorRT。TRT8、TRT10、TRT11 继续使用独立 version guard。

## 验证范围

`NetworkBuilderSmokeRunner` 在三条版本线上调用真实 `ClearAuxStreams` entry，并输出 pointer-free assignment snapshot。当前 identity engine 的 `AuxiliaryStreamCount=0`，因此 runtime smoke 不伪造非空 assignment；非空路径由 managed lifetime 专项测试和无 ProjectReference package consumer 编译验证覆盖。
