# YoloDet

Status: documented asset-dependent sample.

This directory is reserved for a YOLO-style object-detection demo. A complete detector sample needs a redistributable ONNX model, class labels, input images, preprocessing rules, and postprocessing rules. Some exported YOLO graphs also depend on TensorRT plugins such as NMS variants, so a fake bundled demo would be misleading.

Use these runnable samples first:

```powershell
dotnet .\samples\OnnxToEngineSmokeRunner\bin\Debug\net8.0\OnnxToEngineSmokeRunner.dll --tensor-rt-line 10
dotnet .\samples\DynamicShape\bin\Debug\net8.0\DynamicShape.dll --tensor-rt-line 10 --batch 1
```

They validate ONNX parsing, dynamic-shape profile setup, tensor binding, enqueue, and output readback without requiring a third-party detector asset.

## Required Assets For A Full Demo

- `model.onnx`: a YOLO-family model with documented input tensor name, output tensor names, layout, and supported image sizes.
- `labels.txt`: class label list matching the model.
- `input.*`: one or more redistributable test images.
- Postprocessing metadata: confidence threshold, IoU threshold, output decoding format, and NMS/plugin requirements.

## Roadmap

- Add asset preflight that reports missing model, labels, or image files without crashing.
- Add CPU preprocessing and postprocessing first.
- Add optional TensorRT plugin diagnostics when the selected model requires plugin layers.
- Print stable evidence lines such as `Detections=...`, `TopDetection=...`, and `YoloDet Passed=True`.
