# CUDA Stream Capture To Graph Candidate Audit

Decision: `promote-by-explicit-alias-history`

This batch promotes the CUDA 12.3+ stream-to-existing-graph capture route only
after checking vendor declarations, import-library linkage, runtime exports and
the managed owner contract. The earlier deferred records remain in the
inventory and are represented as history aliases.

## Candidate Review

| Official function | Header evidence | Import library / DLL evidence | Guard | Ownership decision |
| --- | --- | --- | --- | --- |
| `cudaStreamBeginCaptureToGraph` | Declared by CUDA 12.3, 12.9 and 13.2 `cuda_runtime_api.h` | Symbol links in the CUDA 12.9 and 13.2 native presets; runtime smoke succeeds on TRT10/CUDA12.9 | `JYPPX_HAS_CUDA_TOOLKIT && CUDART_VERSION >= 12030` | Promote through an owner-scoped session that retains the managed stream and graph until End |
| `cudaStreamEndCapture` | Existing CUDA runtime declaration across the supported toolkit lines | Existing bridge entry and native link are reused; End verifies the returned graph is the supplied graph | `JYPPX_HAS_CUDA_TOOLKIT` | Keep as the session terminator; never create a second managed graph wrapper |

## Safety Contract

- Begin copies dependency node tokens and `CudaGraphEdgeData` into native call
  storage; the managed array does not escape the synchronous vendor call.
- The session increments capture-use counts on both `CudaStream` and
  `CudaGraph`. `Dispose()` is rejected while a session is active, preventing a
  stream or graph wrapper from being released before CUDA ends capture.
- End calls `cudaStreamEndCapture`, checks the returned handle against the
  original graph owner, then releases the session counts. A mismatched handle
  is an error rather than a second owner transfer.
- C++ allocation/exception and Windows SEH paths are translated to bridge
  status inside native code. No exception crosses the ABI.
- Public C# APIs expose typed `CudaStream`, `CudaGraph` and a disposable
  `CudaStreamCaptureToGraphSession`; they do not expose `IntPtr`, `nint`,
  `UIntPtr`, `SafeHandle`, device pointers or borrowed vendor pointers.

## Cross-Version Result

| Toolkit | Begin symbol | End route | Coverage |
| --- | --- | --- | --- |
| CUDA 11.8 / 12.1 | unavailable | existing `cudaStreamEndCapture` only | deferred; no fabricated API |
| CUDA 12.3 / 12.9 | available | owner-safe session | `implemented-with-deferred-history` |
| CUDA 13.2 | available | owner-safe session | `implemented-with-deferred-history` |

The previous deferred record
`cuda-thirty-fifth-batch-stream-device-boundaries.manifest.json` is retained.
The real IDs are explicit aliases in `eng/Export-InterfaceCoverageMatrix.ps1`.

## Verification

- Binding generator: `189 manifests / 3947 API records`; output validation and
  idempotence passed.
- Coverage rows for CUDA 12.3, 12.9 and 13.2 report
  `implemented-with-deferred-history`.
- Native ABI declaration and PE export parity: TRT8, TRT10 and TRT11 all
  report `MissingDeclarations=0 MissingExports=0`.
- Native Release presets: TRT8/CUDA11.8, TRT8/CUDA12.1,
  TRT10/CUDA12.9 and TRT11/CUDA13.2 built successfully.
- The TRT10/CUDA12.9 `CudaGraphSmokeRunner` emitted
  `ToGraph=True Nodes=1` and completed the graph round trip.
- Focused ProjectQuality: `13/13`; bridge-only package consumers for three
  runtime lines built with zero warnings and zero errors.

These results are source/build/compatible-host smoke evidence. They are not
clean public package-consumer runtime proof, post-publish proof, publication
approval or release-close approval.
