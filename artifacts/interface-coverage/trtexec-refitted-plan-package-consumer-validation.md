# TensorRtExec Refitted Plan Package Consumer Validation

- strict: `True`
- checks: `46`
- passed: `46`
- failed: `0`

| Check | Passed | Actual |
| --- | --- | --- |
| `schema` | `True` | `trtexec-refitted-plan-package-consumer-evidence.v1` |
| `state` | `True` | `local-package-consumer-refitted-plan-runtime-passed` |
| `classification` | `True` | `local-package-consumer-refitted-plan-runtime` |
| `trt10-runtime-key` | `True` | `win-x64-trt10.11-cuda12.9-cudnn9.22` |
| `two-declared-local-sources` | `True` | `2` |
| `nuget-org-disabled` | `True` | `False` |
| `package-reference-only` | `True` | `True` |
| `no-project-reference` | `True` | `False` |
| `no-manual-managed-load` | `True` | `False` |
| `isolated-restore-cache` | `True` | `True` |
| `target-packages-resolved` | `True` | `True/True` |
| `managed-package-id` | `True` | `JYPPX.TensorRT.CSharp.API` |
| `bridge-package-id` | `True` | `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge` |
| `package-hashes` | `True` | `9afd8486da2094027fc73bc222e61df277e406d295559c136637b5824b6125da/24dfbdb294de81b418f9e72aaa1c265968a2f8d2b2706cd7bd39b623ffb6829d` |
| `consumer-outside-repository` | `True` | `True` |
| `consumer-off-system-drive` | `True` | `False` |
| `consumer-workspace-removed` | `True` | `True/False` |
| `managed-assembly-isolated` | `True` | `True/True` |
| `bridge-isolated` | `True` | `True` |
| `consumer-source-hashes` | `True` | `cdcd409783018b1ffeafeb5408e34b5ee301e18d6f0f71b13fff2529a4c84a41/21d77cd5b5fe7be6e8d00dbfcadf2b95c4398c54c2840438eb6ff644467338c0` |
| `relative-source-artifacts` | `True` | `artifacts/real-case/trtexec-refitted-plan-persistence/mnist-refitted-persisted.plan/artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7-input-f32.bin` |
| `plan-copy-distinct` | `True` | `True` |
| `plan-length-cross-check` | `True` | `408876` |
| `plan-hash-cross-check` | `True` | `5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb/5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb` |
| `input-copy-distinct` | `True` | `True` |
| `input-copy-hash` | `True` | `3136/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564` |
| `output-length` | `True` | `40` |
| `output-hash-cross-check` | `True` | `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041/6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041` |
| `output-exact-match` | `True` | `True` |
| `process-exits` | `True` | `0/0/0` |
| `runtime-passed` | `True` | `True` |
| `execution-log-hashes` | `True` | `8d08a45c975c757cb69083627f46452df33829db54f1bdfdfd674579cbf05b72/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855/8d08a45c975c757cb69083627f46452df33829db54f1bdfdfd674579cbf05b72` |
| `command-shape` | `True` | `dotnet run --project <consumer-project> -c Release --no-build -- <copied-plan> <copied-input> <raw-output> <expected-output-sha256>` |
| `full-weight-refittable-fact` | `True` | `False` |
| `engine-metadata` | `True` | `2/5/1` |
| `input-contract` | `True` | `Input3/[1,1,28,28]/784` |
| `output-contract` | `True` | `Plus214_Output_0/[1,10]/10` |
| `enqueue-gates` | `True` | `True/True/True/True` |
| `mnist-output-index` | `True` | `7` |
| `host-metadata` | `True` | `X64/NVIDIA GeForce RTX 3060 Laptop GPU` |
| `native-inventory` | `True` | `bridge-package-output,tensor-rt-runtime,cuda-runtime` |
| `c-drive-clean` | `True` | `False/0` |
| `local-runtime-boundary` | `True` | `True` |
| `no-public-package-proof` | `True` | `False/False/False` |
| `no-release-side-effect` | `True` | `False/False/False` |
| `compact-evidence-path-free` | `True` | `absolute-windows-path-present=False` |
