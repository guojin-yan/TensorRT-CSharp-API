using System;

namespace YoloVisionSample;

public sealed class YoloSemanticMap
{
    public YoloSemanticMap(int classCount, int width, int height, float[] values)
    {
        if (classCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(classCount));
        }

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

        if (values.Length != checked(classCount * width * height))
        {
            throw new ArgumentException("Semantic map value count must match classCount * width * height.", nameof(values));
        }

        ClassCount = classCount;
        Width = width;
        Height = height;
        Values = values;
    }

    public int ClassCount { get; }

    public int Width { get; }

    public int Height { get; }

    public float[] Values { get; }
}
