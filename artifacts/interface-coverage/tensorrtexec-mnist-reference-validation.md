# TensorRtExec MNIST Reference Validation

- strict: `True`
- runtime artifacts required: `True`
- checks: `44`
- passed: `44`
- failed: `0`

| Check | Passed | Actual |
| --- | --- | --- |
| `schema` | `True` | `tensorrtexec-mnist-reference-validation-evidence.v1` |
| `state` | `True` | `mnist-reference-regression-runtime-passed-owner-review-required` |
| `classification` | `True` | `real-model-reference-candidate-runtime` |
| `runtime-key` | `True` | `win-x64-trt10.11-cuda12.9-cudnn9.22/10.11.0/12.9` |
| `model-identity` | `True` | `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf/ONNX Model Zoo vision/classification/mnist, as recorded by the TensorRT data/mnist README` |
| `license-review-boundary` | `True` | `owner-review-required/False` |
| `input-contract` | `True` | `Input3/1,1,28,28/784` |
| `reference-identity` | `True` | `Plus214_Output_0/1,10/10` |
| `reference-source-boundary` | `True` | `repository-mnist-runtime-output-derived-unreviewed/False/False` |
| `reference-policy` | `True` | `0.0001/0.0001/reject/exact` |
| `source-tree-build` | `True` | `external-onnx-reference-validated-runtime/True/0` |
| `independent-load-engine` | `True` | `load-engine-reference-validated-runtime/True/0` |
| `build-load-engine-identity` | `True` | `14044b5d345a68bebe7b01c3a48ce10c1665bde40088fbbfd61ae236f1221a89/14044b5d345a68bebe7b01c3a48ce10c1665bde40088fbbfd61ae236f1221a89` |
| `build-load-raw-output` | `True` | `0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5/0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5` |
| `build-error-within-policy` | `True` | `9.536743E-07/1.3443339E-06` |
| `load-error-within-policy` | `True` | `9.536743E-07/1.3443339E-06` |
| `runtime-artifact-hashes` | `True` | `1e93d4705c08a3e0fee620393d3517b274af849b3913ca76b4b2eebffac5a4dc/a9ad565a5549dd194088f384de81dea06b1a45974009e4bcba1a8db56f6d1605/29577756e0ff9ae289f2ad37120d71e5c5515225a9915f2d47850c17bbaed919/070f70cd67fe1d48946032e0cbaf114c4f0d168b5e6ec5fcd4f50b44834b4252` |
| `consumer-state` | `True` | `local-package-consumer-refitted-plan-runtime-passed/local-package-consumer-refitted-plan-runtime/True` |
| `consumer-isolation` | `True` | `True/False/False/True` |
| `consumer-reference` | `True` | `True/True/10/0/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef` |
| `consumer-not-public-proof` | `True` | `False` |
| `owner-review-open` | `True` | `not-provided/False/False/False` |
| `proof-boundary` | `True` | `The existing TensorRT MNIST model ran on a real GPU and the same structured reference passed source-tree build, independent load-engine, and isolated local PackageReference consumer paths. Because the reference was copied from an earlier TensorRT output and model/license/redistribution/golden-reference Owner review is not provided, this is a real-model reference regression candidate only, not independent numerical correctness, Owner-accepted real-model, public-package, post-publish, or release proof.` |
| `path-free-compact-evidence` | `True` | `absolute-windows-path-present=False` |
| `reference-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.reference.json/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef` |
| `reference-sidecar-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.reference.sidecar.json/d117ad49fb5a70136b03e4579c92c9e02c41f10de310623b171f1ede6aaaf935` |
| `reference-file-contract` | `True` | `1/Plus214_Output_0/1,10/10/repository-mnist-runtime-output-derived-unreviewed` |
| `sidecar-owner-boundary` | `True` | `real-model-reference-candidate-owner-review-required/not-provided/False/False` |
| `package-evidence-cross-check` | `True` | `local-package-consumer-refitted-plan-runtime-passed/True/True/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef/6f5771d6c5b056406c190a59e725cf9bb13c1f148c1ef06f99ed8acfb11b9041` |
| `model-file-hash` | `True` | `third_party/nvidia/TensorRT-10.11.0.33-cuda 12.9/data/mnist/mnist.onnx/2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf` |
| `model-source-readme-hash` | `True` | `third_party/nvidia/TensorRT-10.11.0.33-cuda 12.9/data/mnist/README.md/b80c70931e11b2edc517bcd081cfdbafa222f2c860819ee0806f4eaed472a475` |
| `model-license-readme-hash` | `True` | `third_party/nvidia/TensorRT-10.11.0.33-cuda 12.9/samples/sampleOnnxMNIST/README.md/696b89fdf1046390156bd3eea37b75c7bfee320a699cb6cbb78b754d7b3f7057` |
| `source-input-file-hash` | `True` | `third_party/nvidia/TensorRT-10.11.0.33-cuda 12.9/data/mnist/7.pgm/880e75f93fe00ab6f5c4e8ab00ff695c61e7e30bdf0d967ff8b34de1f5a94634` |
| `input-tensor-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7-input-f32.bin/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564` |
| `source-output-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7-output.json/37b234e2f4cd8583699238822e8ebdc419c6dd43bcd71a87e8590e9e5e1d9f3e` |
| `build-engine-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/reference-validation/mnist-build-reference.plan/14044b5d345a68bebe7b01c3a48ce10c1665bde40088fbbfd61ae236f1221a89` |
| `build-output-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/reference-validation/mnist-build-reference-output.json/1e93d4705c08a3e0fee620393d3517b274af849b3913ca76b4b2eebffac5a4dc` |
| `build-report-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/reference-validation/mnist-build-reference-report.json/a9ad565a5549dd194088f384de81dea06b1a45974009e4bcba1a8db56f6d1605` |
| `build-raw-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/reference-validation/mnist-build-reference.raw/0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5` |
| `load-output-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/reference-validation/mnist-load-reference-output.json/29577756e0ff9ae289f2ad37120d71e5c5515225a9915f2d47850c17bbaed919` |
| `load-report-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/reference-validation/mnist-load-reference-report.json/070f70cd67fe1d48946032e0cbaf114c4f0d168b5e6ec5fcd4f50b44834b4252` |
| `load-raw-file-hash` | `True` | `artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/reference-validation/mnist-load-reference.raw/0202efd6a92fbb38f066b9d392de290b5fe5e61d07604da6561b9f22441b61d5` |
| `build-runtime-report-contract` | `True` | `True/True/True/True/True` |
| `load-runtime-report-contract` | `True` | `True/True/True/True/True` |
