using System;
using System.Collections.Generic;

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

        for (int index = 0; index < values.Length; index++)
        {
            if (!float.IsFinite(values[index]))
            {
                throw new ArgumentException(
                    $"Semantic map values must be finite; index {index} is not finite.",
                    nameof(values));
            }
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

    public int GetClassIndex(int x, int y)
    {
        if ((uint)x >= (uint)Width)
        {
            throw new ArgumentOutOfRangeException(nameof(x));
        }
        if ((uint)y >= (uint)Height)
        {
            throw new ArgumentOutOfRangeException(nameof(y));
        }

        int pixelOffset = y * Width + x;
        int planeSize = checked(Width * Height);
        int bestClass = 0;
        float bestValue = Values[pixelOffset];
        for (int classIndex = 1; classIndex < ClassCount; classIndex++)
        {
            float candidate = Values[classIndex * planeSize + pixelOffset];
            if (candidate > bestValue)
            {
                bestValue = candidate;
                bestClass = classIndex;
            }
        }

        return bestClass;
    }

    public int[] GetClassIndexMap()
    {
        int[] classIndices = new int[checked(Width * Height)];
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                classIndices[y * Width + x] = GetClassIndex(x, y);
            }
        }

        return classIndices;
    }

    public int[] GetClassHistogram()
    {
        int[] histogram = new int[ClassCount];
        foreach (int classIndex in GetClassIndexMap())
        {
            histogram[classIndex]++;
        }

        return histogram;
    }
}
