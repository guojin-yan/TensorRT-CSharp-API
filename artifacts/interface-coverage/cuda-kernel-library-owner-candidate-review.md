# CUDA Kernel Library 与 Execution Context 候选审查

生成日期：2026-07-17

本轮从最新 CUDA `deferred-only` 行中复核 20 个 library、kernel 与 execution-context 候选。上一轮将整个 CUDA library/kernel/resource 区域统一保留 deferred；本轮只提升能够建立 bridge-owned library、且 borrowed `cudaKernel_t` 不离开 native 调用栈的元数据子集。

## 本轮实现

| 官方 API | 决策 | 安全边界 |
| --- | --- | --- |
| `cudaLibraryLoadData` | 实现 | native owner 保留输入 code 的独立副本直到 unload |
| `cudaLibraryLoadFromFile` | 实现 | UTF-8 路径仅在调用栈内使用，返回 bridge-owned library |
| `cudaLibraryUnload` | 实现 | `SafeCudaKernelLibraryHandle` 唯一释放 bridge-owned library |
| `cudaLibraryGetKernelCount` | 实现 | 返回复制型 `uint32_t` |
| `cudaLibraryEnumerateKernels` | 实现 | borrowed kernel 数组仅在 native 栈内校验，返回 count/completeness 快照 |
| `cudaLibraryGetKernel` | 实现 | 按名称查询存在性，borrowed kernel handle 不跨 ABI |

## 继续 Deferred

| 候选 | 原因 |
| --- | --- |
| `cudaLibraryGetGlobal` | 返回 device pointer，缺少 allocation/size owner |
| `cudaLibraryGetManaged` | 返回 managed/device pointer，生命周期依赖 library/context |
| `cudaLibraryGetUnifiedFunction` | 返回可调用 function pointer，不能安全公开 |
| `cudaKernelSetAttributeForDevice` | mutating kernel API 需要 owner-bound kernel capability 与 device policy |
| `cudaGetKernel` | 从 host symbol 获取 borrowed kernel，host symbol/function pointer 不可公开 |
| `cudaGetFuncBySymbol` | 返回 borrowed function handle，缺少 owner 模型 |
| `cudaDeviceGetExecutionCtx` | 返回 driver-owned execution context，需要独立 bridge owner/borrow contract |
| `cudaExecutionCtxDestroy` | 只有完成 create/get ownership 区分后才能安全释放 |
| `cudaExecutionCtxGetDevice` | 依赖尚未建模的 execution-context owner |
| `cudaExecutionCtxGetId` | scalar 本身安全，但 owner 前置条件尚未满足 |
| `cudaExecutionCtxSynchronize` | 会同步 context work，必须先明确异步生命周期 |
| `cudaExecutionCtxStreamCreate` | 返回与 context 绑定的 stream，需要双 owner lease |
| `cudaDeviceGetDevResource` | 返回 resource handle，acquire/release 语义未建模 |
| `cudaGreenCtxCreate` | 涉及 device resource 与 execution context 组合 ownership |

旧 deferred manifest 全部保留。coverage 只有在显式 real alias 优先命中后才将目标行记为 `implemented-with-deferred-history`。公开 C# API 不暴露 `cudaLibrary_t`、`cudaKernel_t`、`IntPtr`、`nint`、`SafeHandle`、device pointer 或 function pointer。
