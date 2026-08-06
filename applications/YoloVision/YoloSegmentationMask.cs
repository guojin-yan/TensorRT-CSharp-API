using System;

namespace YoloVisionSample;

public enum YoloSegmentationMaskValueKind
{
    RawLogits,
    Probability
}

public sealed class YoloSegmentationMask
{
    public const float DefaultThreshold = 0.5f;

    public YoloSegmentationMask(int width, int height, float[] values)
        : this(width, height, values, YoloSegmentationMaskValueKind.Probability, DefaultThreshold)
    {
    }

    public YoloSegmentationMask(
        int width,
        int height,
        float[] values,
        YoloSegmentationMaskValueKind valueKind,
        float threshold)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length != checked(width * height))
        {
            throw new ArgumentException("Mask value count must match width * height.", nameof(values));
        }

        if (!float.IsFinite(threshold) || threshold < 0.0f || threshold > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Mask threshold must be in [0, 1].");
        }

        if (valueKind is not YoloSegmentationMaskValueKind.RawLogits and not YoloSegmentationMaskValueKind.Probability)
        {
            throw new ArgumentOutOfRangeException(nameof(valueKind));
        }

        Width = width;
        Height = height;
        Values = values;
        ValueKind = valueKind;
        Threshold = threshold;
    }

    public int Width { get; }

    public int Height { get; }

    public float[] Values { get; }

    public YoloSegmentationMaskValueKind ValueKind { get; }

    public float Threshold { get; }

    public float GetProbability(int index)
    {
        if (index < 0 || index >= Values.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        float probability = ValueKind == YoloSegmentationMaskValueKind.RawLogits
            ? YoloMaskComposer.Sigmoid(Values[index])
            : Math.Clamp(Values[index], 0.0f, 1.0f);
        return float.IsNaN(probability) ? 0.0f : probability;
    }

    public int CountPixelsAtOrAboveThreshold()
    {
        int count = 0;
        for (int index = 0; index < Values.Length; index++)
        {
            if (GetProbability(index) >= Threshold)
            {
                count++;
            }
        }

        return count;
    }
}
