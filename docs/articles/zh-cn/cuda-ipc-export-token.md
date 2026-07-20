# CUDA IPC 导出 Token：只复制，不暴露指针

`CudaEvent.ExportIpcToken()` 与 `CudaMemory.ExportIpcToken()` 会把 CUDA 的 opaque IPC
导出 handle 复制成不可变托管值。公开 API 不返回 CUDA event handle、device pointer、
`IntPtr`、`UIntPtr` 或 `SafeHandle`。

## 导出 Event Token

CUDA 要求 IPC event 同时带有两个标志：

```csharp
using CudaEvent cudaEvent = new CudaEvent(
    CudaEventCreationFlags.Interprocess | CudaEventCreationFlags.DisableTiming);

CudaIpcExportToken token = cudaEvent.ExportIpcToken();
byte[] serialized = token.ToArray();
Console.WriteLine($"Kind={token.Kind} Length={token.Length}");
```

另一个进程使用 token 期间，源 `CudaEvent` 必须保持存活。CUDA 文档明确说明：先销毁
exported event，再操作 imported event，行为未定义。

## 导出 Device Memory Token

```csharp
using CudaMemory memory = new CudaMemory(4096);
CudaIpcExportToken token = memory.ExportIpcToken();
byte[] serialized = token.ToArray();
```

只有 `CudaMemory(int)` 同步路径得到的 `cudaMalloc` 基址可以导出。managed memory、
`cudaMallocAsync` 与 memory-pool allocation 会 fail closed。所有 imported mapping 停止使用前，
源 `CudaMemory` 必须保持存活。

`ToArray()` 每次返回新副本。`ToString()` 只显示 kind 与 length；日志中不打印 token
内容，也不要把 token 当成普通诊断字段上传。

## 有意保留的边界

本批不实现 `cudaIpcOpenEventHandle`、`cudaIpcOpenMemHandle` 或
`cudaIpcCloseMemHandle`。import 会创建当前进程内的资源，需要另行设计 device affinity、
peer access、引用计数、失败恢复和 cleanup。export 成功不等于跨进程 runtime proof，也不等于
package-consumer proof。

NVIDIA 将 Windows CUDA IPC 定义为兼容用途支持，但不建议用于性能敏感设计。实际部署前应
检查设备 IPC 能力，并在目标机器上验证真实多进程流程。
