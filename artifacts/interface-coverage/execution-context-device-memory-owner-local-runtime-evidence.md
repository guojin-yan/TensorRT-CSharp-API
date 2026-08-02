# Execution Context Device Memory Owner Local Runtime Evidence

Classification: `project-reference-local-runtime`

The pointer-free `NetworkConvolutionScaleSmokeRunner` was executed on an NVIDIA GeForce RTX 3060 Laptop GPU with driver 576.02. It constructs its network through TensorRT APIs and does not read an external model.

| TensorRT / CUDA | Result | Relevant marker |
| --- | --- | --- |
| 8.6.1 / 12.1 | passed | `Set=True V2=False ... CallerWrappersDisposed=True RetainedLeases=2`; `ConvolutionScalePaddingOutputMatch=True` |
| 10.11.0 / 12.9 | passed | `Set=True V2=True ... CallerWrappersDisposed=True RetainedLeases=2`; clear left two retired leases; output matched |
| 11.0.0 / 13.2 | blocked before device-memory binding | `createInferRuntime returned a null TensorRT object.` |

TRT10 and TRT11 Release native bridge builds succeeded. TRT8 native code was unchanged, so its existing bridge was used for the compatibility runtime check. The JSON companion stores the bridge and smoke assembly SHA256 values.

This record contains no device address and no full smoke log. It sets `isAllSupportedLinesRuntimeProof=false`, `isPackageConsumerRuntimeProof=false`, `isPostPublishProof=false`, and `canPublishPublicly=false`. It is not TRT11 runtime, clean consumer, Linux, post-publish, or release-close evidence.
