# TensorRtExec Refitted Plan Package Consumer Validation

- strict: `True`
- checks: `53`
- passed: `53`
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
| `package-hashes` | `True` | `9401f9cbd2c513ea5c3954e0b0895ed55e6151662e2d83488ab4b4ff58ca369f/0a4cb23b0175abdde7ff7b6fe16c9094829d0b17304cc5822baf477064e9e032` |
| `consumer-outside-repository` | `True` | `True` |
| `consumer-off-system-drive` | `True` | `False` |
| `consumer-workspace-removed` | `True` | `True/False` |
| `managed-assembly-isolated` | `True` | `True/True` |
| `bridge-isolated` | `True` | `True` |
| `consumer-source-hashes` | `True` | `e956ed02713cde582bef68b8e27c10162ea796a5af7132bf1d75e4e7cc697c93/f1e7960e6e7a0c50008a66eb62010e07fab0c9ae9ac31ae588f23855606c7481` |
| `relative-source-artifacts` | `True` | `artifacts/real-case/trtexec-refitted-plan-persistence/mnist-refitted-persisted.plan/artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7-input-f32.bin/artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.reference.json` |
| `plan-copy-distinct` | `True` | `True` |
| `plan-length-cross-check` | `True` | `408876` |
| `plan-hash-cross-check` | `True` | `5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb/5594817d8b152a9478a57ae73ec2a8ed19e8c9e0794b448b89bcd738d99441eb` |
| `input-copy-distinct` | `True` | `True` |
| `input-copy-hash` | `True` | `3136/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564` |
| `reference-copy-distinct` | `True` | `True` |
| `reference-copy-hash` | `True` | `07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef` |
| `reference-contract` | `True` | `Plus214_Output_0/1,10/10/repository-mnist-runtime-output-derived-unreviewed` |
| `reference-policy` | `True` | `0.0001/0.0001/reject/exact` |
| `output-length` | `True` | `40` |
| `output-hash-cross-check` | `True` | `6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041/6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041` |
| `output-exact-match` | `True` | `True` |
| `process-exits` | `True` | `0/0/0` |
| `runtime-passed` | `True` | `True` |
| `execution-log-hashes` | `True` | `77ece0695db91787005b91ca1cf8b1d06c983326d06f9a309dbebc4855004796/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855/1aa4d422d517979c617fa448bf3aa9b0f380f5305cf8fe44edc928c9fb165936` |
| `command-shape` | `True` | `dotnet run --project <consumer-project> -c Release --no-build -- <copied-plan> <copied-input> <raw-output> <expected-output-sha256> <copied-reference-json> <abs-tolerance> <rel-tolerance> <nan-policy> <infinity-policy>` |
| `full-weight-refittable-fact` | `True` | `False` |
| `engine-metadata` | `True` | `2/5/1` |
| `input-contract` | `True` | `Input3/[1,1,28,28]/784` |
| `output-contract` | `True` | `Plus214_Output_0/[1,10]/10` |
| `enqueue-gates` | `True` | `True/True/True/True` |
| `reference-validation-gates` | `True` | `True/True/10/0/-1` |
| `reference-validation-errors` | `True` | `0/0` |
| `mnist-output-index` | `True` | `7` |
| `host-metadata` | `True` | `X64/NVIDIA GeForce RTX 3060 Laptop GPU` |
| `native-inventory` | `True` | `bridge-package-output,tensor-rt-runtime,cuda-runtime` |
| `c-drive-clean` | `True` | `False/0` |
| `local-runtime-boundary` | `True` | `True` |
| `no-public-package-proof` | `True` | `False/False/False` |
| `no-release-side-effect` | `True` | `False/False/False` |
| `unreviewed-reference-boundary` | `True` | `This proves a copied full-weight refitted plan can be restored and executed by a PackageReference-only consumer from declared local feeds on one compatible host, with the recorded structured reference comparison. The reference is an unreviewed same-runtime MNIST regression baseline, so this is not independent model-accuracy, Owner-approved golden output, public-feed, post-publish, or release-close proof.` |
| `compact-evidence-path-free` | `True` | `absolute-windows-path-present=False` |
