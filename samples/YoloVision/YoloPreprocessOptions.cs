using System;

namespace YoloVisionSample;

public sealed class YoloPreprocessOptions
{
    public YoloPreprocessOptions(
        string tensorLayout,
        string colorOrder,
        string resizeMode,
        float scale,
        bool normalize,
        bool preserveAspectRatio)
    {
        TensorLayout = string.IsNullOrWhiteSpace(tensorLayout) ? "NCHW" : tensorLayout;
        ColorOrder = string.IsNullOrWhiteSpace(colorOrder) ? "RGB" : colorOrder;
        ResizeMode = string.IsNullOrWhiteSpace(resizeMode) ? "letterbox" : resizeMode;
        Scale = scale > 0 ? scale : throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be positive.");
        Normalize = normalize;
        PreserveAspectRatio = preserveAspectRatio;
    }

    public string TensorLayout { get; }

    public string ColorOrder { get; }

    public string ResizeMode { get; }

    public float Scale { get; }

    public bool Normalize { get; }

    public bool PreserveAspectRatio { get; }

    public static YoloPreprocessOptions Default { get; } = new YoloPreprocessOptions(
        "NCHW",
        "RGB",
        "letterbox",
        1.0f / 255.0f,
        normalize: true,
        preserveAspectRatio: true);
}
