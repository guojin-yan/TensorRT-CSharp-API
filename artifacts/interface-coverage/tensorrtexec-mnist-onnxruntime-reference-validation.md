# TensorRtExec MNIST ONNX Runtime Reference Validation

- Strict: `True`
- Runtime artifact checks required: `True`
- Passed: `51/51`
- Failures: `0`

| Check | Passed | Actual |
| --- | ---: | --- |
| schema | True | tensorrtexec-mnist-onnxruntime-reference-evidence.v1 |
| state | True | independent-onnxruntime-cpu-reference-runtime-passed-owner-review-required |
| classification | True | independent-framework-reference-candidate-runtime |
| runtime-identity | True | ONNX Runtime/1.23.2/CPUExecutionProvider |
| provider-available | True | AzureExecutionProvider,CPUExecutionProvider |
| provider-profile-only-cpu | True | True/CPUExecutionProvider |
| runtime-repository-commit | True | a83fc4d58cb48eb68890dd689f94f28288cf2278 |
| package-count | True | 4 |
| package-inventory | True | microsoft.ml.onnxruntime,microsoft.ml.onnxruntime.managed,system.memory,system.numerics.tensors |
| package-versions | True | microsoft.ml.onnxruntime/1.23.2;microsoft.ml.onnxruntime.managed/1.23.2;system.numerics.tensors/9.0.0;system.memory/4.5.5 |
| package-hashes | True | 25fe172f7fcdf34f2b5b02c8b997f6b88abc282956849d4af4335af23d0c4e4a,6e29d09318b9ec1a3207fd97fe0dfebfcccec15b936c85bdff2f6c8080e91836,b750243c36002a62b28b1ac5d3fbc284ad340ba1494cc36aca110611a0b1f959,10f43da352a29fb2b3188e4edd4dcf5100194c8b526e4f61fe2e2b5623775a22 |
| runtime-binary-hashes | True | 4c6985df4aa2f810ba8a2b37c3c791e619d8fe126954a6bd28cffd3f4adeba95/dec964ab1ee36cc9b0ae247d13b376627992fc57dec0454354017ab8fd84f1ea |
| model-contract | True | 2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf/False |
| model-source-boundary | True | b80c70931e11b2edc517bcd081cfdbafa222f2c860819ee0806f4eaed472a475/696b89fdf1046390156bd3eea37b75c7bfee320a699cb6cbb78b754d7b3f7057 |
| input-contract | True | 81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564/Input3/1,1,28,28 |
| reference-identity | True | 1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571/Plus214_Output_0/1,10 |
| reference-source-boundary | True | onnxruntime-cpu-1.23.2-derived-unreviewed/True/7 |
| reference-artifact-hashes | True | 9d624fa545260fd80e01573071159264506c3981f8b015553b59a4647b06fdb0/a20932857fb2d51f5f0b79daa211140fce631b3f67787b69e0a23f33c8817d75 |
| comparison-reference | True | 07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef |
| comparison-result | True | True/10/0/-1 |
| comparison-errors | True | 5.722046E-06/5.7323444E-07/0.0001/0.0001 |
| offline-restore-boundary | True | True/True/True/True |
| workspace-cleanup | True | False/True |
| process-exits | True | 0/0/0 |
| source-and-log-hashes | True | e0eeb924bd08b1ac206abd29d0f1eedec8532e6af9e9865ef80d1a5956ec8ba9/672b9d166d6618e3760ac7b4fda5c6064591f8ae6975872e580587b66557d09f/18825de054c07d0627e840b37f5f3e8ad8517d21bf89fdab4d6cd2560ba8a124/8b8575a48da5984dd76dc5904b066fe5f187195dc4f0a6410f7116b0d986a427 |
| relative-runtime-artifact-paths | True | relative-only |
| owner-review-open | True | not-provided/False/False/False |
| independent-execution-boundary | True | True/True/False |
| no-release-promotion | True | False/False/False/False/False |
| proof-statement | True | Two deterministic ONNX Runtime CPU runs and CPUExecutionProvider profiling provide an execution path independent from TensorRT. Owner review for model/license/redistribution/reference acceptance remains absent, so this is an independent-framework reference candidate, not accepted real-model, public-package, post-publish, or release proof. |
| path-free-compact-evidence | True | absolute-windows-path-present=False |
| reference-file-contract | True | 1/Plus214_Output_0/1,10/10/onnxruntime-cpu-1.23.2-derived-unreviewed |
| sidecar-contract | True | tensorrtexec-mnist-onnxruntime-reference-sidecar.v1/independent-onnxruntime-cpu-reference-candidate-owner-review-required/1.23.2/CPUExecutionProvider |
| sidecar-cross-check | True | 1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571/a20932857fb2d51f5f0b79daa211140fce631b3f67787b69e0a23f33c8817d75/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef/True |
| reference-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.onnxruntime-cpu.reference.json/1babfa81c0d277a3d483fdc6288285b26008ba5b382868272f1e2187338bd571 |
| sidecar-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.onnxruntime-cpu.reference.sidecar.json/9d624fa545260fd80e01573071159264506c3981f8b015553b59a4647b06fdb0 |
| prior-mnist-cross-check | True | 2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef |
| model-file-hash | True | ../models/OnnxToEngine/MNIST/nvidia-tensorrt-10.11/mnist.onnx/2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf |
| model-source-readme-hash | True | &lt;user-tensorrt-root&gt;/data/mnist/README.md/b80c70931e11b2edc517bcd081cfdbafa222f2c860819ee0806f4eaed472a475 |
| model-license-readme-hash | True | &lt;user-tensorrt-root&gt;/samples/sampleOnnxMNIST/README.md/696b89fdf1046390156bd3eea37b75c7bfee320a699cb6cbb78b754d7b3f7057 |
| input-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7-input-f32.bin/81f2cd7784dca5c8f02a9887ae89a6a3582880a27d2eba95a53f845c63187564 |
| tensorrt-reference-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/mnist-trt10-7.reference.json/07969a96f76f9dc69a777770ca581df64ad9cb2d1ea320f97bdfbb7cc697beef |
| raw-output-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/onnxruntime-cpu-reference/mnist-onnxruntime-cpu-output.raw/a20932857fb2d51f5f0b79daa211140fce631b3f67787b69e0a23f33c8817d75 |
| stdout-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/onnxruntime-cpu-reference/mnist-onnxruntime-cpu.stdout.log/01d18529254ee04035183e6e19f2e860e064e069e4750357c2177c1b8b05b0cb |
| stderr-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/onnxruntime-cpu-reference/mnist-onnxruntime-cpu.stderr.log/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| run-report-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/onnxruntime-cpu-reference/mnist-onnxruntime-cpu-run-report.json/18825de054c07d0627e840b37f5f3e8ad8517d21bf89fdab4d6cd2560ba8a124 |
| profile-file-hash | True | artifacts/real-case/onnx-to-engine-mnist-trt10-runtime/digit-7/onnxruntime-cpu-reference/onnxruntime_profile__2026-07-29_10-02-09.json/8b8575a48da5984dd76dc5904b066fe5f187195dc4f0a6410f7116b0d986a427 |
| cached-package-microsoft.ml.onnxruntime | True | 1.23.2/25fe172f7fcdf34f2b5b02c8b997f6b88abc282956849d4af4335af23d0c4e4a |
| cached-package-microsoft.ml.onnxruntime.managed | True | 1.23.2/6e29d09318b9ec1a3207fd97fe0dfebfcccec15b936c85bdff2f6c8080e91836 |
| cached-package-system.numerics.tensors | True | 9.0.0/b750243c36002a62b28b1ac5d3fbc284ad340ba1494cc36aca110611a0b1f959 |
| cached-package-system.memory | True | 4.5.5/10f43da352a29fb2b3188e4edd4dcf5100194c8b526e4f61fe2e2b5623775a22 |

CPUExecutionProvider profiling proves an execution path independent from TensorRT. It does not supply Owner model/license/redistribution/golden acceptance, public-package proof, post-publish proof, or release authorization.
