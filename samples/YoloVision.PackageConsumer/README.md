# YoloVision Local Package Consumer

This template is copied into a clean E-drive workspace by
`eng/Test-YoloVisionLocalPackageConsumer.ps1`. It contains only `PackageReference` entries and
calls the reusable `YoloVisionCommand` from `JYPPX.TensorRT.CSharp.API.YoloVision`.

The validation script supplies three local packages: the managed API, YoloVision, and one
runtime-key-selected TRT8, TRT10, or TRT11 bridge-only component. TensorRT, CUDA, and cuDNN remain
system-installed or explicitly selected local dependencies. The official YOLOX assets remain under
the outer E-drive `downloads/yolox-apache` workspace. A successful run also checks that the bridge
build's TensorRT major version matches the selected runtime package key.

Passing this sample is `local-package-consumer-runtime` engineering evidence. It is not proof that
the packages were downloaded from a public feed, not public redistribution approval, and not
post-publish or release-close proof.
