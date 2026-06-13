# Smoke

This directory contains validation-oriented runner projects that we use for:

- package-consumer smoke
- release gates
- native loading checks
- TensorRT / CUDA feature regression coverage

These projects are intentionally different from the user-facing examples under `samples/`:

- `samples/` should focus on common adoption scenarios and readable reference cases.
- `smoke/` should focus on deterministic diagnostics, compact output, and CI-friendly validation.

## Current Smoke Projects

- `CudaSmokeRunner`
- `CudaGraphSmokeRunner`
- `TensorRtSmokeRunner`
- `LifecycleSmokeRunner`
- `InferenceBindingsSmokeRunner`
- `OnnxToEngineSmokeRunner`
- `RefitWeightsSmokeRunner`
- `NetworkBuilderSmokeRunner`
- `NetworkLayersSmokeRunner`
- `NetworkShapeOpsSmokeRunner`
- `NetworkConcatSliceSmokeRunner`
- `NetworkSoftmaxTopKSmokeRunner`
- `NetworkActivationPoolingResizeSmokeRunner`
- `NetworkMatrixFillSelectSmokeRunner`
- `NetworkConvolutionScaleSmokeRunner`
- `NetworkDeconvolutionSmokeRunner`
- `NetworkLrnSmokeRunner`
- `NetworkQuantizeDequantizeSmokeRunner`
- `NetworkCompatLayerMetadataSmokeRunner`
- `NetworkTrt11ModernLayersSmokeRunner`
- `NetworkTrt11ModernLayerMetadataRunner`
- `NetworkTrt11AdvancedLayersSmokeRunner`

## Notes

- Release and workflow documents should reference `smoke/` when describing validation runners.
- New end-user examples should go to `samples/`, not `smoke/`.
