using System;

namespace YoloVisionSample;

public sealed class YoloSegmentationMask
{
    public YoloSegmentationMask(int width, int height, float[] values)
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

        Width = width;
        Height = height;
        Values = values;
    }

    public int Width { get; }

    public int Height { get; }

    public float[] Values { get; }
}
