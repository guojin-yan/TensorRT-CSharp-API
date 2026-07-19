# Builder Scalar Cross-Version Proof

本矩阵来自同一台 compatible host 上当前源码 bridge 的实际 TensorRtExec 执行。所有 report 均通过 strict validator，failed blocker 为 0。

| TensorRT | Max tactics | Tiling | L2 limit | Quantization flags | 证据状态 |
| --- | --- | --- | --- | --- | --- |
| 8.6.1 / CUDA 12.1 | unsupported | unsupported | unsupported | applied/readback match | dependency-probe-only；缺少 cuDNN8 parser DLL |
| 10.11.0 / CUDA 12.9 | applied/readback match | applied/readback match | 3MiB applied；256MiB rejected/readback 3MiB | applied/readback match | build-only copied-readback |
| 11.0.0 / CUDA 12.9 | applied/readback match | applied/readback match | 3MiB applied；256MiB rejected/readback 3MiB | removed-by-vendor | build-only copied-readback |

## 关键结论

- `AppliedOptions` 只包含 `Applied=True` 且 `ReadbackMatch=True` 的 scalar。
- TRT10/11 对有效的 `--l2LimitForTiling 3MiB` 均成功；`256MiB` 超过本机可接受范围时，setter 返回 false，实际 readback 保持 3MiB，报告进入 parse-only。
- TRT8 真实调用 quantization flags setter/readback；现代 tactics/tiling/L2 API 按版本 guard 输出 controlled unsupported。
- TRT11 对 quantization flags 输出 `RemovedByTensorRT11`，没有调用已移除 vendor API。

## 边界

TRT10/11 报告是 local compatible-host build-only；TRT8 是 dependency-probe-only。没有 inference、output comparison、real external model 或 clean package consumer，因此三者均不是 real-model-runtime、package-consumer-runtime 或 release proof。旧 deferred history 保留。
