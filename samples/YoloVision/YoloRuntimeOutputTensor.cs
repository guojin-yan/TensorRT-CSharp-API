using System;

namespace YoloVisionSample;

public sealed class YoloRuntimeOutputTensor
{
    public YoloRuntimeOutputTensor(string name, YoloOutputTensorRole role, float[] values, int[] shape)
    {
        Name = name ?? string.Empty;
        Role = role;
        Values = values ?? throw new ArgumentNullException(nameof(values));
        Shape = shape ?? throw new ArgumentNullException(nameof(shape));
    }

    public string Name { get; }

    public YoloOutputTensorRole Role { get; }

    public float[] Values { get; }

    public int[] Shape { get; }

    public int ElementCount => Values.Length;

    public override string ToString()
    {
        return $"{Role}:{Name}[{string.Join(",", Shape)}] Values={ElementCount}";
    }
}
