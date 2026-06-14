# YoloDet

This sample runs a user-provided single-input float YOLO-family ONNX model through TensorRT, then decodes common output layouts:

- `[1, 84, 8400]` style channel-first output
- `[1, 8400, 84]` style box-first output

The repository does not bundle detector models, label files, or images because those assets have separate licensing and size constraints. The sample uses a synthetic input tensor by default, so detections are useful as pipeline evidence rather than as image-quality evidence.

```powershell
dotnet run --project .\samples\YoloDet -- `
  --model .\models\yolo.onnx `
  --labels .\models\coco.names `
  --input-shape 1x3x640x640 `
  --tensor-rt-line 10 `
  --layout auto `
  --has-objectness auto `
  --confidence 0.25
```

## Required Assets For A Full Demo

- `model.onnx`: a YOLO-family model with documented input tensor name, output tensor names, layout, and supported image sizes.
- `labels.txt`: class label list matching the model.
- `input.*`: one or more redistributable test images.
- Postprocessing metadata: confidence threshold, IoU threshold, output decoding format, and NMS/plugin requirements.

## Evidence Lines

- `YoloDet TensorRtLine=...`
- `Input=... Output=...`
- `Detection Class=... Score=... BoxCxCyWh=...` or `Detections=0 ...`
- `YoloDet Passed=True`
