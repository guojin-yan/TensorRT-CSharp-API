# CUDA Primary Execution Context Owner 候选审查

生成日期：2026-07-17

本轮从 CUDA 13.2 的 87 条 `deferred-only` 行中复核 21 个 execution context、device resource、graph 与 kernel-library 候选。官方头文件明确规定 `cudaDeviceGetExecutionCtx` 返回设备主上下文，并明确禁止把该句柄传给 `cudaExecutionCtxDestroy`。因此本轮只建立 bridge-owned、non-destroying primary wrapper；green context 与 resource descriptor 链不进入此 owner。

## 本轮实现

| 官方 API | 决策 | 安全边界 |
| --- | --- | --- |
| `cudaDeviceGetExecutionCtx` | 实现 | 返回 bridge-owned wrapper；底层设备主上下文仍由 CUDA runtime 拥有 |
| `cudaExecutionCtxGetDevice` | 实现 | 仅复制 `int32_t` device ordinal |
| `cudaExecutionCtxGetId` | 实现 | 仅复制进程内唯一 `uint64_t` id |
| `cudaExecutionCtxSynchronize` | 实现 | 同步调用只接受 owner wrapper，不暴露 context handle |
| `cudaExecutionCtxStreamCreate` | 实现 | 返回现有 bridge-owned `CudaStream`；主上下文不会由 wrapper 销毁 |
| `cudaExecutionCtxRecordEvent` | 实现 | event 使用现有 SafeHandle，仅在 native 调用栈内借用 |
| `cudaExecutionCtxWaitEvent` | 实现 | event 使用现有 SafeHandle，仅在 native 调用栈内借用 |

bridge wrapper 的 release entry 只删除 `ExecutionContextObject`，不会调用 `cudaExecutionCtxDestroy`，也不映射为该官方 API 的实现。

## 继续 Deferred

| 候选 | 原因 |
| --- | --- |
| `cudaExecutionCtxDestroy` | 对 `cudaDeviceGetExecutionCtx` 返回的主上下文调用属于未定义行为；green context owner 尚未建立 |
| `cudaDeviceGetDevResource` | 返回 device resource union，resource/descriptor 生命周期尚未建模 |
| `cudaExecutionCtxGetDevResource` | 返回 context resource union，不能并入 non-destroying primary wrapper |
| `cudaDevResourceGenerateDesc` | descriptor 依赖 resource 集合与后续 green-context ownership |
| `cudaDevSmResourceSplit` | 产生可组合 resource 集合，涉及 partition ownership |
| `cudaDevSmResourceSplitByCount` | 同上，count/copy 不能替代 resource owner |
| `cudaGreenCtxCreate` | 需要 owning execution context、descriptor lease 和 stream teardown 顺序 |
| `cudaStreamGetDevResource` | 返回 stream-bound resource union，缺少可释放 owner |
| `cudaGraphNodeGetParams` | tagged union 中包含 driver-owned pointers；需独立复制型 discriminated snapshot |
| `cudaGraphAddNode` | tagged union 可能包含 out pointer 与不可复制 handle |
| `cudaKernelSetAttributeForDevice` | 需要 owner-bound kernel 与 device policy，当前只允许 library 内 named existence |
| `cudaLibraryGetGlobal` | 返回 device pointer，缺少 allocation/size owner |
| `cudaLibraryGetManaged` | 返回 managed/device pointer，生命周期依赖 library/context |
| `cudaLibraryGetUnifiedFunction` | 返回可调用 function pointer，不能安全公开 |

旧 deferred manifest 全部保留。coverage 通过显式 real alias 优先匹配并合并 deferred history；只有 7 个真实官方调用可变为 `implemented-with-deferred-history`。公开 C# API 不暴露 `cudaExecutionContext_t`、`IntPtr`、`nint`、`SafeHandle`、device pointer 或 function pointer。
