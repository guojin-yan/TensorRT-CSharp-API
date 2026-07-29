using System;

namespace JYPPX.TensorRtSharp.Tools;

/// <summary>
/// One binary PGM image used by the TensorRT MNIST sample.
/// TensorRT MNIST 样例使用的一张二进制 PGM 图像。
/// </summary>
public sealed class MnistPgmImage
{
    public MnistPgmImage(int width, int height, int maxValue, byte[] pixels)
    {
        Width = width;
        Height = height;
        MaxValue = maxValue;
        Pixels = pixels ?? throw new ArgumentNullException(nameof(pixels));
    }

    public int Width { get; }

    public int Height { get; }

    public int MaxValue { get; }

    public byte[] Pixels { get; }
}
