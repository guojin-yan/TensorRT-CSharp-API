# YoloVision Managed Package Consumer

This template is materialized by `eng/Test-YoloVisionManagedPackageDryRun.ps1` in a clean workspace outside the repository.
It restores only `JYPPX.TensorRT.CSharp.API` and `JYPPX.TensorRT.CSharp.API.YoloVision` from the selected local package directory,
then builds and runs pointer-free YoloVision capability checks.

The smoke intentionally does not load the native bridge or NVIDIA runtime. It proves managed package dependency, restore, build,
and execution shape only; it is not TensorRT runtime, public-feed, post-publish, redistribution, or release proof.
