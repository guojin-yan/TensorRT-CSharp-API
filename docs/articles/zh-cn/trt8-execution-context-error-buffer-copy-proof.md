# TRT8 Execution Context Error Buffer Copy Proof

本批补齐 TensorRT 8 `IExecutionContext::getErrorBuffer` 的低所有权风险诊断路径。
TensorRT vendor API 返回 borrowed `const char*`，因此 native bridge 只在调用期间复制到
caller-owned buffer，再由托管层解码为 `string`。

公共 API 为 `TensorRtExecutionContext.TryGetErrorBuffer(...)`，不会暴露 `IntPtr`、`nint`、
`SafeHandle` 或 TensorRT-owned pointer。TensorRT 10 和 TensorRT 11 没有对应 legacy query，
会返回受控 unsupported diagnostic。

coverage 同时保留旧 deferred manifest，因此矩阵状态为
`implemented-with-deferred-history`。smoke 输出的是 copied-boundary marker，不是 runtime、
clean package-consumer 或 release proof。GitHub Actions、NuGet 和 Release 在本批保持冻结。
