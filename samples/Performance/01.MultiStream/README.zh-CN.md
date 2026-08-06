# CUDA 多流案例

[English](README.md) | 简体中文

该案例使用两个 non-blocking CUDA Stream 和 Event，演示异步设备内存操作、pinned host memory 读回、Event record/synchronize 以及跨流等待顺序。

## 运行

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = '1'
dotnet .\samples\Performance\01.MultiStream\bin\Release\net8.0\MultiStream.dll
```

成功运行会输出流和 Event 的关键阶段、读回结果及最终通过标记。完整流程、真实 Windows Terminal 截图和同步语义说明见 [CUDA 多流与 Event 同步教程](../../../docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md)。

该案例用于解释正确性和资源所有权，不将一次运行耗时作为跨机器性能基准。正式基准测试还需要预热、重复采样、固定输入和完整主机环境信息。
