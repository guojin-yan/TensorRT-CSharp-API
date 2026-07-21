# CUDA IPC Import Owner-Safe Candidate Audit

本批选择 `cudaIpcOpenEventHandle`、`cudaIpcOpenMemHandle` 与
`cudaIpcCloseMemHandle`，与上一批 export token 组成完整跨进程生命周期。

| 证据 | 11.6 | 11.8 | 12.1 | 12.3 | 12.9 | 13.2 |
| --- | --- | --- | --- | --- | --- | --- |
| header declaration | present | present | present | present | present | present |
| `cudart.lib` symbol | present | present | present | present | present | present |
| installed runtime DLL export | present | present | present | present | present | DLL not separately installed |

## Owner Model

- event import 返回 bridge-owned event wrapper，并由 `cudaEventDestroy` 释放。
- memory import 返回 bridge-owned memory wrapper，并且只由 `cudaIpcCloseMemHandle` 释放。
- memory open flags 固定为 `cudaIpcMemLazyEnablePeerAccess`，不向 public API 暴露随意 flags。
- token 必须是 64 字节不可变副本；memory token 与精确 allocation size 作为一个不可变
  transport descriptor 传递。
- 普通 `cudaFree` 与 `cudaFreeAsync` 路径遇到 imported mapping 都 fail closed。
- importer 存活期间 exporter 必须保持源 event/memory owner 存活。

## Boundary

旧 deferred manifest 全部保留。public surface 不暴露 raw/device pointer、`IntPtr`、`UIntPtr`
或 `SafeHandle`。TRT10/CUDA12.9 的真实父子进程 smoke 已完成，状态为
`passed-real-cross-process-smoke`，独立证据见
`cuda-ipc-import-cross-process-runtime-evidence.{json,md}`。该本地 ProjectReference smoke 仍不是
package-consumer runtime、公开发布批准或 release issue close 证据。
