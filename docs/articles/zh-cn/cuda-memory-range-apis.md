# CUDA Memory Range APIs

CUDA memory range API 用于查询和设置一段内存的访问属性，例如 prefetch、advise、accessed-by 和 range attribute。它们适合诊断 managed memory 行为，也适合为多 GPU 或 host/device 访问模式做准备。

## 典型能力

- 查询 range attributes。
- 设置 read mostly、preferred location、accessed by。
- 对 managed memory 做 prefetch。
- 记录设备 id、范围大小和返回状态。

## 使用建议

memory range API 应与明确的 memory owner 配合使用。调用方需要知道当前指针对应的分配类型、长度和目标 device，避免对 borrowed pointer 或未知范围执行属性设置。

## 测试重点

质量测试应覆盖：

- 空范围和非法 device 的错误映射。
- managed memory 的 advise/prefetch 调用路径。
- attribute 查询的 count/copy 或 caller buffer 模式。
- CUDA driver/runtime 不兼容时的 blocked 状态记录。

## 边界

这些 API 不能替代 runtime smoke。当前环境出现 `blocked-by-cuda-driver` 时，应记录为环境阻塞，而不是把调用链写成 smoke passed。
