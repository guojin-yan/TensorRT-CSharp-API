# YoloVision Local Package Consumer

This template is copied into a clean E-drive workspace by
`eng/Test-YoloVisionLocalPackageConsumer.ps1`. It contains only `PackageReference` entries and
calls the reusable `YoloVisionCommand` from `JYPPX.TensorRT.CSharp.API.YoloVision`.

The validation script supplies three local packages: the managed API, YoloVision, and one
runtime-key-selected TRT8, TRT10, or TRT11 bridge-only component. TensorRT, CUDA, and cuDNN remain
system-installed or explicitly selected local dependencies. The official YOLOX assets and the
locally licensed YOLOv8n-seg assets remain under the outer E-drive `downloads` workspace. The
official torchvision LRASPP ONNX remains under the outer E-drive `models` workspace for a future
separately governed Model Zoo. A
successful run also checks that the bridge build's TensorRT major version matches the selected
runtime package key.

`eng/Test-YoloVisionLocalPackageConsumer.ps1` defaults to the YOLOX detection case.
`eng/Test-YoloVisionSegmentationLocalPackageConsumer.ps1` selects the strict YOLOv8n-seg case,
which also requires two raw tensor references, source-image mask artifacts, an independent
Ultralytics/PyTorch CPU comparison, and controlled raw-reference and mask-integrity failures.
`eng/Test-YoloVisionSemanticLocalPackageConsumer.ps1` selects the LRASPP semantic case, compares
the full raw tensor and 102,400-pixel class-index map, and requires raw-reference plus artifact
integrity mutations to fail closed.

Passing this sample is `local-package-consumer-runtime` engineering evidence. It is not proof that
the packages were downloaded from a public feed, not public redistribution approval, and not
post-publish or release-close proof.
