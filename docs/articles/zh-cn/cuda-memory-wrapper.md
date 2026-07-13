# CUDA Memory Wrapper

CUDA memory wrapper 的目标是让 .NET 调用方用明确的 owner 对象管理 device、pinned、managed 和 pitched memory，减少裸指针在 public API 中漂移。

## 常见内存类型

- Device memory：GPU 设备内存，适合 tensor 输入输出。
- Pinned host memory：页锁定主机内存，适合异步拷贝。
- Managed memory：统一内存，适合部分原型和诊断场景。
- Pitched memory：二维布局，适合图像或行对齐场景。

## 推荐使用方式

应用侧应优先使用托管 wrapper 持有生命周期，再把地址传给 TensorRT execution context。释放顺序应保持外层推理对象不再使用该地址后再释放 memory owner。

对于跨 stream copy，优先结合 `CudaStream` 和 `CudaEvent`，用事件记录和等待表达顺序。

## 证据入口

- `smoke/CudaSmokeRunner`
- `samples/MultiStream`
- `docs/articles/zh-cn/cuda-stream-event-multistream-tutorial.md`

## 边界

本文讨论的是 memory wrapper 使用方式，不表示 allocator callback 已具备真实 runtime proof。`IGpuAllocator`、`IGpuAsyncAllocator`、`IOutputAllocator` 的 callback 边界仍要按 owner ledger 和 runtime proof gate 推进。

## 第二批正文门禁

### 适用读者

本文适合正在把 TensorRT 输入输出从裸地址迁移到 C# owner wrapper 的工程师，也适合需要审计 CUDA buffer 生命周期、stream copy 和 package consumer proof 边界的发布负责人。

### 解决问题

CUDA memory wrapper 要解决的是生命周期和诊断问题：用户应看到 `CudaMemory`、`CudaPinnedMemory`、`CudaManagedMemory` 这类 owner，而不是在 public API 中直接传递无语义地址。这样后续 TensorRT enqueue、OnnxToEngine report、TensorRtExec report 和 YoloVision matrix 才能清楚标记哪些是 smoke，哪些才可能进入 runtime proof。

### 核心思路

核心思路是把 memory owner、copy API、stream/event ordering 和释放顺序拆开记录。build-only、dry-run、template、local feed、ProjectReference、direct `.nupkg`、TensorRtExec report、YoloVision matrix、OnnxToEngine report、readonly diagnostics 都不是 runtime proof；它们只能作为定位环境和打包问题的辅助证据。

### 操作路径

先用同步 copy 验证最小 device/pinned memory 路径，再用 `CudaStream` 和 `CudaEvent` 扩展到异步 copy。进入 TensorRT 推理时，必须额外记录 engine、binding、shape、stdout/stderr summary、hash、host metadata 和 validator 结果。

### 边界说明

本文是 CUDA memory owner 使用教程，不证明 allocator callback、output allocator 或 clean consumer runtime proof 已完成。callback 和外部资源 ownership 必须继续走 owner ledger、strict validator 和真实兼容主机执行记录。

### 下一步

下一步把本文与 stream/event 教程、inference binding 教程串成 CUDA 到 TensorRT 的基础系列，并在真实兼容主机 proof 回填后再更新发布状态。
