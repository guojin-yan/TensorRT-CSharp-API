# TensorRtSharp 4.0.0 RC 已知限制

这份限制清单用于发布候选审计。它不降低质量门禁，只是把当前仍需外部环境或下一阶段实现的事项明确写出来。

## 环境限制

- 当前机器的 NVIDIA driver/runtime 组合无法完成 CUDA 13 runtime smoke，`cudaRuntimeGetVersion` 返回 CUDA error 35。
- CUDA error 35 代表驱动/runtime 兼容性阻塞，不代表 managed wrapper、native asset copy 或 package layout 失败。
- 需要一台 CUDA 13-capable driver 的 Windows x64 机器完成 `win-x64-trt11.0-cuda13.2-cudnn9.22` 的普通 runtime smoke。

## API 边界限制

- `IDebugListener::processDebugTensor` 仍保留 deferred row。
- `setDebugListener(non-null)` 默认不启用。
- native `IDebugListener` vtable 不默认安装。
- `InvocationCount=0` 时不能提升为 real callback runtime proof。
- public API 不能暴露 native owner pointer、vtable pointer、debug tensor pointer 或 data pointer。

## 发布限制

- runtime package 中包含 NVIDIA TensorRT/CUDA/cuDNN 二进制资产，公开发布前需要复核再分发许可。
- 本地签名脚本用于开发机 WDAC/application-control 排障，不等同于正式发布证书。
- Linux x64 hosted/container 包线已经建模，但 ARM/SBSA、Jetson/L4T 和非 Ubuntu Linux 需要独立 package ID、runner 和依赖策略。
