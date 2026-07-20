# YOLOX Local Package Consumer Runtime Proof Closure

- classification: `local-package-consumer-runtime`
- consumer: `samples/YoloVision.PackageConsumer`
- package sources: three local file feeds
- ProjectReference: `0`
- restored project libraries: `0`
- workspace/cache drive: `E:`
- managed/YoloVision/bridge packages: `4.0.0`
- native bridge: copied from the bridge-only NuGet package
- TensorRT/CUDA: system-installed `10.11.0.33` / `12.9`
- runtime: 5 detections, `bicycle=0.954854`, `dog=0.913407`, `14.238 ms`
- passed marker: `YoloVision Passed=True`
- workspace removed after validation: `True`

## Boundary

This closure proves a real official-YOLOX runtime from a clean, ProjectReference-free consumer that
restored three local NuGet packages. The packages were not downloaded from a public feed. This is
not public `package-consumer-runtime` proof, redistribution approval, post-publish proof, publish
approval, release-close approval, or a package push.
