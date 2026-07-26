# TensorRT Execution Context NVTX Verbosity Candidate Audit

生成日期：2026-07-26

本批选择 `IExecutionContext::getNvtxVerbosity` 与
`IExecutionContext::setNvtxVerbosity` 作为一个成对的 scalar diagnostics
工作包。两项都已经有真实 TRT8/TRT10/TRT11 native entry、manifest、托管
interop、pointer-free wrapper 和 smoke 调用；本轮补齐的是显式 deferred-history
alias 与可复核的候选审计，不删除历史 deferred manifest。

| API | vendor header | import library/DLL | ownership | version guard | decision |
|---|---|---|---|---|---|
| `IExecutionContext::getNvtxVerbosity` | `NvInferRuntime.h` declares scalar enum getter in TRT8/10/11 | Existing native ABI/export parity records cover `jyppx_trt{8,10,11}_execution_context_get_nvtx_verbosity`; no vendor pointer crosses ABI | caller-owned scalar output | `JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8/10/11` | promote real entry; retain TRT8/TRT10 deferred history |
| `IExecutionContext::setNvtxVerbosity` | `NvInferRuntime.h` declares scalar enum setter in TRT8/10/11 | Existing native ABI/export parity records cover `jyppx_trt{8,10,11}_execution_context_set_nvtx_verbosity`; return is copied boolean | caller-owned scalar input/output | `JYPPX_TENSORRT_VERSION_MAJOR_NUM == 8/10/11` | promote real entry; retain TRT8/TRT10 deferred history |

## Safety Decision

- No callback, allocator, plugin lifecycle, device pointer, borrowed object, or
  external resource is involved.
- Native code validates the context and output pointer, converts the enum locally,
  and returns only `int32_t`/boolean values.
- The public wrapper exposes `TensorRtProfilingVerbosity` and `bool`; it does not
  expose `IntPtr`, `nint`, `UIntPtr`, `SafeHandle`, or a TensorRT object pointer.
- `trt8-...-nvtx-verbosity-deferred` and `trt10-...-nvtx-verbosity-deferred`
  records remain in the deferred manifests. TRT11 has no matching historical
  deferred record. The coverage exporter now records the existing history through
  explicit aliases so the matrix cannot silently lose provenance.

## Proof Boundary

The smoke call is a bounded diagnostic round-trip (`get -> set(same value) -> get`).
It is not a model runtime proof, package-consumer runtime proof, release proof, or
permission to publish. `canDeleteDeferredRecords` remains false.
