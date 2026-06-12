# Classification

Status: documented asset-dependent sample.

This directory is reserved for an image-classification demo that uses a real ONNX classifier and input assets. The repository does not currently bundle those assets because model files, label files, and sample images add licensing and size constraints that are separate from API validation.

Use these runnable samples first:

```powershell
dotnet .\samples\DynamicShape\bin\Debug\net8.0\DynamicShape.dll --tensor-rt-line 10 --batch 3
dotnet .\samples\OnnxToEngineSmokeRunner\bin\Debug\net8.0\OnnxToEngineSmokeRunner.dll --tensor-rt-line 10
```

They validate the same deployment foundation a classification demo needs: TensorRT builder/runtime creation, dynamic profile setup, tensor binding, enqueue, and output readback.

## Required Assets For A Full Demo

- `model.onnx`: an image-classification ONNX model with documented input tensor name, layout, data type, and normalization.
- `labels.txt`: one label per output class.
- `input.*`: a small image with redistribution rights.
- Preprocessing metadata: resize policy, crop policy, RGB/BGR order, scale, mean, and standard deviation.

## Roadmap

- Add an asset loader that fails with a clear message when model or image files are missing.
- Add CPU preprocessing first; add GPU preprocessing only after safe CUDA kernel/module wrappers exist.
- Reuse `TensorRtInferenceBindings` for input/output binding.
- Print stable evidence lines such as `Top1=...`, `OutputCount=...`, and `Classification Passed=True`.
