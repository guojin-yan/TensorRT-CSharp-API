# TRT8 Execution Context Error Buffer Deferred Diagnostic Evidence

跨版本重建确认，本 bridge 使用的标准 `nvinfer1::IExecutionContext` 类型在当前支持的
TensorRT 8.6 与 TensorRT 10 vendor headers 中均不提供 `getErrorBuffer`。早先将 safe-runtime
header 中的同名方法映射到标准 execution context、并标记为已实现，是不成立的证据声明。

为保持 ABI 与托管兼容性，`jyppx_trt8_execution_context_get_error_buffer_copy` 和
`TensorRtExecutionContext.TryGetErrorBuffer(...)` 继续保留。native 入口不会访问 vendor
对象或 borrowed pointer，而是清零 required size 并返回明确的 `NotImplemented` 诊断；
托管 `Try...` API 返回 `false`、空字符串和该诊断。

manifest 中的兼容入口和原始历史记录都以 deferred 身份保留，coverage 状态恢复为
`deferred-only`。smoke 输出的是 deferred diagnostic marker，不是 vendor query、runtime、
clean package-consumer 或 release proof。若未来接入 TensorRT safe runtime，必须建立独立的
safe execution-context owner/lifetime 证明，不能复用标准 context 指针作推断。
