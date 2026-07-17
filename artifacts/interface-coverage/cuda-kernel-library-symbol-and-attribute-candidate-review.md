# CUDA Kernel Library Symbol 与 Attribute 候选审查

生成日期：2026-07-17

## 本批结论

本批复核上一阶段保留 deferred 的 4 个 CUDA 12.9/13.2 Kernel Library API。在已有 bridge-owned `CudaKernelLibrary` 生命周期成立后，将它们收敛为不暴露 pointer 的 size、existence 与 owner-bound control：

| 官方 API | 本批公开语义 | pointer 边界 |
| --- | --- | --- |
| `cudaLibraryGetGlobal` | `TryGetGlobalSymbolSize` | native 将 `dptr` 传 null，只请求 copied size |
| `cudaLibraryGetManaged` | `TryGetManagedSymbolSize` | native 将 `dptr` 传 null，只请求 copied size |
| `cudaLibraryGetUnifiedFunction` | `ContainsUnifiedFunction` | function pointer 只在 native 栈内判空，不跨 ABI |
| `cudaKernelSetAttributeForDevice` | `SetAttributeForDevice` | 通过 library owner + kernel name 临时取得 borrowed kernel，调用后立即丢弃 |

## 安全约束

- global/managed symbol query 只返回 `bool` 与 `ulong sizeInBytes`，不构造或返回 device/managed pointer。
- unified function query 只返回存在性；成功时只检查 vendor pointer 非空，pointer 不保存、不调用、不返回。
- kernel attribute 只接受官方文档列出的 7 个可变 `cudaFuncAttribute`，device ordinal 必须非负。
- 名称由 managed UTF-8 scope 提供，仅在同步 native 调用期间有效。
- `cudaErrorSymbolNotFound` 被消费后调用 `cudaGetLastError` 清理 sticky error，再返回 `false` 与 size 0。
- 本机 CUDA 12.9 对 raw PTX library 的不存在 unified function 返回 `cudaErrorInvalidValue`，而非文档列出的 `cudaErrorSymbolNotFound`；高层保留该诊断异常，smoke 受控消费 last-error 后继续，不伪造 `false`。
- CUDA 12.9/13.2 调用真实 vendor API；更早版本返回 `NotSupported`。C++ exception 与 Windows SEH 不跨 ABI。

## 继续 Deferred

- 任何返回或读写 global/managed device pointer 的 API。
- unified function pointer 调用、kernel launch pointer、raw symbol copy。
- library code mutation、跨 library borrowed kernel 保存、kernel/device resource ownership。
- callback trampoline、external/IPC resource、allocator/resource acquire/release 与 generic tagged union。

旧 deferred manifest 全部保留。coverage 通过显式 real alias 优先，将支持版本标记为 `implemented-with-deferred-history`；不得删除 deferred history 改写统计。

## 实施后验证

- generator：179 manifests / 3916 records，连续生成与幂等检查通过。
- CUDA 12.9：239 implemented / 68 deferred-only；CUDA 13.2：257 / 73。
- 四个目标函数在 CUDA 12.9/13.2 的 8 行全部为 `implemented-with-deferred-history`。
- CUDA 12.9 runtime：global size 为 4，missing global/managed 为 false，attribute setter 成功，unified-function vendor invalid-argument 被保留并受控清理，最终 last error 为 0。
- CUDA 11.8 runtime：Kernel Library 按 guard 返回 `NotSupported`。
- ProjectQuality 受影响分片 65/65；累计类覆盖 378/378、missing 0。
- 五套 native、managed package、三个 bridge-only package 与三个纯 PackageReference consumer 均完成验证。
