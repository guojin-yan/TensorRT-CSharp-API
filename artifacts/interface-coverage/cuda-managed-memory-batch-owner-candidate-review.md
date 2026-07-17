# CUDA managed-memory batch owner candidate review

更新时间：2026-07-17

## 本批结论

本批仅提升 CUDA 13 新增的 3 个 managed-memory batch 接口：

| CUDA 接口 | bridge 方案 | managed 方案 |
| --- | --- | --- |
| `cudaMemPrefetchBatchAsync` | `JYPPX_CudaManagedMemoryBatchRange[]` 临时输入数组 | `CudaManagedMemoryPrefetchRange` + `CudaManagedMemoryBatch.PrefetchAsync` |
| `cudaMemDiscardBatchAsync` | 同上，destination 字段忽略 | `CudaManagedMemoryRange` + `CudaManagedMemoryBatch.DiscardAsync` |
| `cudaMemDiscardAndPrefetchBatchAsync` | 同上 | `CudaManagedMemoryPrefetchRange` + `CudaManagedMemoryBatch.DiscardAndPrefetchAsync` |

三条接口只接受现有 `CudaManagedMemory` owner 与 `CudaStream` owner。公开 API 不返回或接收
`IntPtr`、`nint`、`SafeHandle`、device pointer 或裸 CUDA stream。internal interop 在调用期间对每个
`SafeCudaMemoryHandle` 执行 `DangerousAddRef`，固定 managed descriptor 数组，调用结束后逆序释放租约。

## Native 安全边界

- `MemoryObject::is_managed` 区分 `cudaMallocManaged` 与普通、异步、memory-pool device allocation。
- native 在构造 `void*` / size / location / location-index 临时数组前检查 owner 类型、非零范围、越界与设备序号。
- flags 固定为 CUDA 当前要求的 `0ULL`。
- CUDA 13 使用独立的 `CUDART_VERSION >= 13000` guard；CUDA 11/12 返回 `NotSupported`。
- `std::bad_alloc`、其他 C++ exception 与 Windows SEH 均在 C ABI 内转换为 bridge status。
- 调用是异步提交。调用方必须让全部 `CudaManagedMemory` owner 和 `CudaStream` 存活到 stream 同步完成。

## Runtime 证明

`CudaSmokeRunner` 在 CUDA 12.x bridge 上实际提交一次 typed batch 请求并要求得到
`VersionGuard=NotSupported`。在 CUDA 13.x 上，只有系统中所有设备均报告
`ConcurrentManagedAccess` 时才依次运行 prefetch、discard、discard-and-prefetch，并在每次操作后同步。
这遵守 CUDA 对 batch 接口的整机 capability 前置条件，也避免把环境缺少能力误记为实现失败。

## Deferred 保留

旧 deferred manifest 全部保留，用于记录历史缺口；coverage 通过显式 real alias 优先匹配，将 CUDA 13.2
对应行解析为 `implemented-with-deferred-history`。CUDA 11/12 的 vendor header 不包含这些接口，因此不产生
伪 implemented 行。

本批不改变以下 deferred 边界：callback trampoline、external/IPC pointer、resource tagged union、
allocator/resource acquire/release、裸 device/function/plugin pointer，以及 ownership 不明确的通用 graph 参数。
