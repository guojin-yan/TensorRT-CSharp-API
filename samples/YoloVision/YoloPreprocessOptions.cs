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
        bool preserveAspectRatio,
        string letterboxAlignment)
        : this(
            tensorLayout,
            colorOrder,
            resizeMode,
            scale,
            normalize,
            preserveAspectRatio,
            letterboxAlignment,
            resizeShorterSide: 0)
    {
    }

    public YoloPreprocessOptions(
        string tensorLayout,
        string colorOrder,
        string resizeMode,
        float scale,
        bool normalize,
        bool preserveAspectRatio,
        string letterboxAlignment,
        int resizeShorterSide)
    {
        TensorLayout = string.IsNullOrWhiteSpace(tensorLayout) ? "NCHW" : tensorLayout;
        ColorOrder = string.IsNullOrWhiteSpace(colorOrder) ? "RGB" : colorOrder;
        ResizeMode = string.IsNullOrWhiteSpace(resizeMode) ? "letterbox" : resizeMode;
        Scale = scale > 0 ? scale : throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be positive.");
        Normalize = normalize;
        PreserveAspectRatio = preserveAspectRatio;
        LetterboxAlignment = string.IsNullOrWhiteSpace(letterboxAlignment) ? "center" : letterboxAlignment;
        ResizeShorterSide = resizeShorterSide >= 0
            ? resizeShorterSide
            : throw new ArgumentOutOfRangeException(nameof(resizeShorterSide), "Resize shorter side must be zero or positive.");
    }

    public string TensorLayout { get; }

    public string ColorOrder { get; }

    public string ResizeMode { get; }

    public float Scale { get; }

    public bool Normalize { get; }

    public bool PreserveAspectRatio { get; }

    public string LetterboxAlignment { get; }

    public int ResizeShorterSide { get; }

    public static YoloPreprocessOptions Default { get; } = new YoloPreprocessOptions(
        "NCHW",
        "RGB",
        "letterbox",
        1.0f / 255.0f,
        normalize: true,
        preserveAspectRatio: true,
        letterboxAlignment: "center");
}
