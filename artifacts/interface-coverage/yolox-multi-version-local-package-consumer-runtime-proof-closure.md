# YOLOX Multi-version Local Package Consumer Runtime Proof Closure

- state: `passed-path-free-matrix-proof-closure`
- requested / passed / blocked: `3 / 2 / 1`
- package surface findings: `0`
- C-drive test/assets matches: `0 / 0`
- package-consumer runtime proof: `False`
- publish executed: `False`

| Runtime key | TRT | State | Predictions | Elapsed ms | Runtime root | Diagnostic |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `win-x64-trt8.6-cuda12.1-cudnn8.9` | 8 | `blocked-runtime-attempt` | 0 | 0 | `resolved-runtime-root` | TensorRT ONNX parser library was not found at build time. |
| `win-x64-trt10.11-cuda12.9-cudnn9.22` | 10 | `passed-local-package-consumer-runtime` | 5 | 14.36 | `resolved-runtime-root` | Runtime passed with required package, bridge-version, inference, and output markers. |
| `win-x64-trt11.0-cuda12.9-cudnn9.22` | 11 | `passed-local-package-consumer-runtime` | 5 | 11.679 | `existing-assembled-runtime` | Runtime passed with required package, bridge-version, inference, and output markers. |

TRT10 and TRT11 are real local-file-feed PackageReference YOLOX runtimes on this host. TRT8 is blocked because the available cuDNN 8 developer root has no cudnn64_8 runtime DLL, so its bridge intentionally excludes the ONNX parser. No row proves public download, owner redistribution approval, post-publish verification, or release closure.
