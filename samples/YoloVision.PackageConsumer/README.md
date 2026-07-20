# YoloVision Local Package Consumer

This template is copied into a clean E-drive workspace by
`eng/Test-YoloVisionLocalPackageConsumer.ps1`. It contains only `PackageReference` entries and
calls the reusable `YoloVisionCommand` from `JYPPX.TensorRT.CSharp.API.YoloVision`.

The validation script supplies three local packages: the managed API, YoloVision, and the TRT10
bridge-only runtime component. TensorRT and CUDA remain system-installed dependencies. The official
YOLOX assets remain under the outer E-drive `downloads/yolox-apache` workspace.

Passing this sample is `local-package-consumer-runtime` engineering evidence. It is not proof that
the packages were downloaded from a public feed, not public redistribution approval, and not
post-publish or release-close proof.
