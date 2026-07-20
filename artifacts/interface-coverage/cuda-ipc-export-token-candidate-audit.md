# CUDA IPC Export Token Candidate Audit

Generated: 2026-07-20

## Selected

| Official API | Public shape | Owner and lifetime boundary |
| --- | --- | --- |
| `cudaIpcGetEventHandle` | immutable copied `CudaIpcExportToken` | source `CudaEvent` must use `Interprocess | DisableTiming` and remain alive while another process uses the token |
| `cudaIpcGetMemHandle` | immutable copied `CudaIpcExportToken` | source must be the base of a synchronous `cudaMalloc` allocation and remain alive while imported mappings are used |

Both APIs return fixed-size opaque bytes through a caller-owned buffer. No event, memory, device,
or vendor pointer crosses the bridge ABI. Keep the source event or memory allocation alive while
another process uses the exported token.

## Kept Deferred

- `cudaIpcOpenEventHandle`
- `cudaIpcOpenMemHandle`
- `cudaIpcCloseMemHandle`

Those APIs create or release imported resources and require a separate cross-process ownership,
device-affinity, peer-access, and failure-recovery design. This batch does not add them.

## Evidence Boundary

The implementation may produce local runtime smoke evidence on a compatible CUDA host. It is not
package-consumer runtime proof, cross-process interoperability proof, redistribution approval, or
release-close approval. Historical deferred records remain present.
