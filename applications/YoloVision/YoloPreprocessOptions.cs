using System;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

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
            resizeShorterSide: 0,
            mean: new float[3],
            standardDeviation: new[] { 1.0f, 1.0f, 1.0f })
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
        : this(
            tensorLayout,
            colorOrder,
            resizeMode,
            scale,
            normalize,
            preserveAspectRatio,
            letterboxAlignment,
            resizeShorterSide,
            mean: new float[3],
            standardDeviation: new[] { 1.0f, 1.0f, 1.0f })
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
        int resizeShorterSide,
        float[] mean,
        float[] standardDeviation)
    {
        TensorLayout = string.IsNullOrWhiteSpace(tensorLayout) ? "NCHW" : tensorLayout;
        ColorOrder = string.IsNullOrWhiteSpace(colorOrder) ? "RGB" : colorOrder;
        ResizeMode = string.IsNullOrWhiteSpace(resizeMode) ? "letterbox" : resizeMode;
        Scale = float.IsFinite(scale) && scale > 0
            ? scale
            : throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be finite and positive.");
        Normalize = normalize;
        PreserveAspectRatio = preserveAspectRatio;
        LetterboxAlignment = string.IsNullOrWhiteSpace(letterboxAlignment) ? "center" : letterboxAlignment;
        ResizeShorterSide = resizeShorterSide >= 0
            ? resizeShorterSide
            : throw new ArgumentOutOfRangeException(nameof(resizeShorterSide), "Resize shorter side must be zero or positive.");
        Mean = CopyTriplet(mean, nameof(mean), requirePositive: false);
        StandardDeviation = CopyTriplet(standardDeviation, nameof(standardDeviation), requirePositive: true);
    }

    public string TensorLayout { get; }

    public string ColorOrder { get; }

    public string ResizeMode { get; }

    public float Scale { get; }

    public bool Normalize { get; }

    public bool PreserveAspectRatio { get; }

    public string LetterboxAlignment { get; }

    public int ResizeShorterSide { get; }

    public float[] Mean { get; }

    public float[] StandardDeviation { get; }

    public string ContractSha256 => Convert.ToHexString(
        SHA256.HashData(Encoding.UTF8.GetBytes(ToCanonicalString()))).ToLowerInvariant();

    public string ToCanonicalString()
    {
        return string.Join(
            ";",
            "tensorLayout=" + TensorLayout,
            "colorOrder=" + ColorOrder,
            "resizeMode=" + ResizeMode,
            "scale=" + Scale.ToString("R", CultureInfo.InvariantCulture),
            "normalize=" + Normalize.ToString(CultureInfo.InvariantCulture).ToLowerInvariant(),
            "mean=" + FormatTriplet(Mean),
            "std=" + FormatTriplet(StandardDeviation),
            "preserveAspectRatio=" + PreserveAspectRatio.ToString(CultureInfo.InvariantCulture).ToLowerInvariant(),
            "letterboxAlignment=" + LetterboxAlignment,
            "resizeShorterSide=" + ResizeShorterSide.ToString(CultureInfo.InvariantCulture),
            "interpolation=bilinear-half-pixel");
    }

    public static YoloPreprocessOptions Default { get; } = new YoloPreprocessOptions(
        "NCHW",
        "RGB",
        "letterbox",
        1.0f / 255.0f,
        normalize: true,
        preserveAspectRatio: true,
        letterboxAlignment: "center");

    private static float[] CopyTriplet(float[] values, string name, bool requirePositive)
    {
        if (values == null || values.Length != 3)
        {
            throw new ArgumentException("Value must contain exactly three channels.", name);
        }

        float[] copy = (float[])values.Clone();
        foreach (float value in copy)
        {
            if (!float.IsFinite(value) || (requirePositive && value <= 0.0f))
            {
                throw new ArgumentOutOfRangeException(
                    name,
                    requirePositive
                        ? "Every channel must be finite and positive."
                        : "Every channel must be finite.");
            }
        }

        return copy;
    }

    private static string FormatTriplet(float[] values)
    {
        return string.Join(",", Array.ConvertAll(values, value => value.ToString("R", CultureInfo.InvariantCulture)));
    }
}
