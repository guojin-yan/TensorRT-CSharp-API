# CUDA Graph Memory Allocation Candidate Audit

Date: 2026-07-20

## Decision

`cudaGraphAddMemAllocNode` and `cudaGraphAddMemFreeNode` are promoted through a graph-bound,
pointer-free wrapper. Historical deferred manifests remain present and coverage resolves the real
and historical records as `implemented-with-deferred-history`.

| API | Public ownership model | Decision |
| --- | --- | --- |
| `cudaGraphAddMemAllocNode` | `CudaGraphMemoryAllocation` keeps its owning `CudaGraph` alive; the device address never leaves native code | promote |
| `cudaGraphAddMemFreeNode` | the same graph may add one matching free node after the last consumer | promote |
| `cudaGraphMemAllocNodeGetParams` | copied diagnostics already exist | no duplicate promotion |
| `cudaGraphMemFreeNodeGetParams` | copied diagnostics already exist | no duplicate promotion |
| `cudaGraphAddHostNode` | callback trampoline and lifetime are unresolved | deferred |
| external semaphore graph nodes | imported resource ownership is unresolved | deferred |

## Vendor Evidence

| CUDA toolkit | Header declarations | `cudart.lib` symbols | Runtime DLL exports | Result |
| --- | --- | --- | --- | --- |
| 11.8 | both | both | both in `cudart64_110.dll` | eligible |
| 12.1 | both | both | both in `cudart64_12.dll` | eligible |
| 12.9 | both | both | both in `cudart64_12.dll` | eligible and runtime-tested |
| 13.2 | both | both | no standalone cudart DLL is installed on this host | native build only |

The independent compile guard is `CUDART_VERSION >= 11040`. TRT10/CUDA12.9 and
TRT11/CUDA13.2 native presets both build. CUDA 13.2 runtime remains outside this host's installed
driver/runtime compatibility and is not reported as a runtime pass.

## Safety Boundary

- No public `IntPtr`, `nint`, `UIntPtr`, `SafeHandle`, device address, or native allocation node token.
- Memset/copy/free helpers automatically include the allocation node as a dependency.
- Cross-graph use, a second free node, and graph destruction with an active wrapper fail closed.
- Generic node removal refuses memory-allocation and memory-free nodes so callers cannot bypass the wrapper.
- The local smoke is ProjectReference runtime evidence only. It is not package-consumer runtime,
  post-publish, or release-close proof.
