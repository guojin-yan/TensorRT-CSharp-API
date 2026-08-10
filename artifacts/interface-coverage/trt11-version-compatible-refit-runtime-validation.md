# TRT11 Version-Compatible Refit Runtime Validation

- State: `passed`
- Checks: `45`
- Failures: `0`

| Check | Passed | Detail |
|---|---:|---|
| `schema` | `True` | `trt11-version-compatible-refit-runtime-evidence.v1` |
| `source-tree-boundary` | `True` | `sourceTreeDirtyAtExecution=False` |
| `sameProcess-command` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/command.txt/2185/ad8036f57f72ebe06e45c741faebeb4b3235f76c65e0b31f9e2db523e77aa4fd` |
| `sameProcess-stdout` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/stdout.log/11895/2274ed475e603b866892f31daca4efa1ad8f19f9b1c2ba31ddc17818355117c8` |
| `sameProcess-stderr` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/stderr.log/0/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `sameProcess-exit-code` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/exit-code.txt/3/13bf7b3039c63bf5a50491fa3cfd8eb4e699d1ba1436315aef9cbe5711530354` |
| `sameProcess-report` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/report.json/48246/0079a89488019ba67a9f5ce265cf4b6ed486f7a1f0f204f563fc1335e7674b60` |
| `sameProcess-strict-validation` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/strict-validation.json/14079/4fb57213fc33edd2d0a2ed8267cf5bed624496c976b6388dfd32eb321d304e2e` |
| `sameProcess-output` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/output.json/5185/fea059533ad53958834234e2ba7cf10bcf656b4f44cc8852aa6ce138c6e01274` |
| `secondProcess-command` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/second-process-reload/command.txt/1776/9e3b69f1f4b3e11c7054b161c08e82f3727cb054c96122adcd5bc9949ac0e9c4` |
| `secondProcess-stdout` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/second-process-reload/stdout.log/7358/7fac7131f35344b9adf579b67b43474bc5210a65f329dd351ff9ee91de153dcd` |
| `secondProcess-stderr` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/second-process-reload/stderr.log/0/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `secondProcess-exit-code` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/second-process-reload/exit-code.txt/3/13bf7b3039c63bf5a50491fa3cfd8eb4e699d1ba1436315aef9cbe5711530354` |
| `secondProcess-report` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/second-process-reload/report.json/35812/445165e9d0df7071995c203f7c122e1fb275c0b7b2bd216681b5e10d6639f876` |
| `secondProcess-strict-validation` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/second-process-reload/strict-validation.json/14125/9e41d965d073e5be6f1d9fcc73e4407a51a9cf86c8e4cbb2d609697926f0f9cb` |
| `secondProcess-output` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/second-process-reload/output.json/5780/66972236a3f2b513f12c9608a03559da1fb2113503f6138ad173aed7c77a347a` |
| `asset-model` | `True` | `../models/OnnxToEngine/MNIST/nvidia-tensorrt-10.11/mnist.onnx/26454/2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf` |
| `asset-input` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7-input-f32.bin/3136/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564` |
| `asset-reference` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.reference.json/329/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef` |
| `asset-nativeBridge` | `True` | `build-out/win-x64-trt11-cuda12-release/bin/Release/jyppxtrtbridge.dll/921600/a94d1e5fe4454c9402979ae050f7a64d74c7f51bd6bb2d74936531f608a9ef6f` |
| `asset-vendorRuntime` | `True` | `../downloads/trt11-cuda12.9-v4.0.6156/assembled-runtime/bin/nvinfer_11.dll/414402672/7359c70c65f4a9f138d4bef6d5945a2c4a69c37a7db8dd44ca001af1d701acf7` |
| `asset-cudaRuntime` | `True` | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/cudart64_12.dll/580608/cf68ac7d47c621988db3343b8a211188e6f94cccf30dec3b15dd52d69fdcf512` |
| `stripped-plan` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/mnist-trt11-version-compatible-stripped.plan/41682452/63534e5f0120be93c90c373fa2479850f8ef1c2baba93faa36eb8a472668ea51` |
| `refitted-plan` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/mnist-trt11-version-compatible-refitted-full-weight.plan/106612/961db2fba8cf396bf9bb9cb1f9f6fdda37ae61c739a960c8f1773785db058af6` |
| `environment` | `True` | `artifacts/real-case/tensorrtexec-trt11-version-compatible-refit-runtime-20260810-111142/environment-dependencies.json/3472/19778f45e84e04ff14e297fd42ff0d3caa09f09415b55c7baeeab78753b75532` |
| `main-exit` | `True` | `exit=0` |
| `second-exit` | `True` | `exit=0` |
| `main-runtime-state` | `True` | `external-onnx-refit-reload-reference-validated-runtime/synthetic-input-runtime` |
| `second-runtime-state` | `True` | `load-engine-reference-validated-runtime/synthetic-input-runtime` |
| `main-preflight` | `True` | `runtime=True/builder=True/trt=11.0.0` |
| `main-version-compatible-applied` | `True` | `VersionCompatible set/readback` |
| `main-refit-applied` | `True` | `refit lifecycle completed without parser errors` |
| `main-refit-inventory` | `True` | `missingBefore=0/missingAfter=0` |
| `main-persistence` | `True` | `persisted reload and distinct full-weight artifact` |
| `main-host-code` | `True` | `build host-code policy readback` |
| `main-reference` | `True` | `enqueue/reference mismatch=0` |
| `main-output-hash` | `True` | `a632b881db2328e9103bbdbfb9205c988577a146125faa8ae20e51977b6877c8/40` |
| `second-load-diagnostics` | `True` | `independent deserialize diagnostics` |
| `second-host-code` | `True` | `load-engine host-code policy readback` |
| `second-reference` | `True` | `independent enqueue/reference mismatch=0` |
| `output-hash-stable` | `True` | `same-process=a632b881db2328e9103bbdbfb9205c988577a146125faa8ae20e51977b6877c8/second-process=a632b881db2328e9103bbdbfb9205c988577a146125faa8ae20e51977b6877c8` |
| `strict-main` | `True` | `strict report checks=69 failures=0` |
| `strict-second` | `True` | `strict report checks=69 failures=0` |
| `plan-hash-record` | `True` | `stripped/refitted hashes are distinct and recorded` |
| `proof-boundary` | `True` | `synthetic runtime and release boundaries remain explicit` |
