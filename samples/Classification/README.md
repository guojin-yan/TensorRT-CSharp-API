# Classification

This sample runs a user-provided single-input float ONNX classifier through TensorRT and prints Top-K scores.

The repository does not bundle model, label, or image assets because those files have separate licensing and size constraints. The sample uses a synthetic input tensor by default, so it validates the deployment pipeline. Replace the input generation with your own preprocessing when you wire it into an application.

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --labels .\models\labels.txt `
  --input-shape 1x3x224x224 `
  --tensor-rt-line 10 `
  --top-k 5
```

For dynamic classifiers, provide profile bounds:

```powershell
dotnet run --project .\samples\Classification -- `
  --model .\models\classifier.onnx `
  --input-shape 1x3x224x224 `
  --min-shape 1x3x224x224 `
  --opt-shape 4x3x224x224 `
  --max-shape 8x3x224x224
```

## Required Assets For A Full Demo

- `model.onnx`: an image-classification ONNX model with documented input tensor name, layout, data type, and normalization.
- `labels.txt`: one label per output class.
- `input.*`: a small image with redistribution rights.
- Preprocessing metadata: resize policy, crop policy, RGB/BGR order, scale, mean, and standard deviation.

## Evidence Lines

- `Classification TensorRtLine=...`
- `Input=... Output=...`
- `TopK Index=... Label=... Score=...`
- `Classification Passed=True`
