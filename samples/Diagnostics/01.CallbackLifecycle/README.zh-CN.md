# Callback 生命周期

[English](README.md) | 简体中文

本示例通过公开的 `JYPPX.TensorRT.CSharp.API` 包，在一个最小 TensorRT 流程中同时运行 Logger、ProgressMonitor、Profiler 和 DebugListener。重点是所有权：TensorRT 借用 callback 期间 owner 必须保持存活；可解除的 callback 要显式 clear；DebugListener 的托管代码只能接收复制后的元数据。

离线帮助不会加载 CUDA 或 TensorRT：

```powershell
dotnet run --project .\samples\Diagnostics\01.CallbackLifecycle -- --help
```

运行 TensorRT 10 synthetic 流程：

```powershell
$env:JYPPX_ENABLE_DEVELOPMENT_PROBING = "1"
dotnet run --project .\samples\Diagnostics\01.CallbackLifecycle -- --tensor-rt-line 10 --output-json .\artifacts\callback-lifecycle.json
```

样例在 ProgressMonitor 绑定期间构建一个 `1x1` 卷积，构建完成后清除 monitor；随后把 Profiler 和 DebugListener 绑定到 execution context，执行一次 `1x1x2x2` FP32 推理，同步 CUDA stream、复制输出、读取 callback 快照，并在释放 callback owner 前清除两个运行时 callback。

JSON 报告包含 callback 次数、失败次数、attach/detach 状态、复制的 debug tensor 元数据、`borrowedPointerExposed=false`、输出一致性和生命周期顺序。Logger 会一直存活到所有 builder/runtime borrower 都释放。报告使用 `proofClassification=synthetic-input-runtime`，不是外部真实模型或 `package-consumer-runtime` 发布证明。

本组合样例明确拒绝 TensorRT 8，因为 ProgressMonitor 与 DebugListener 路径要求 TensorRT 10 或 11。
