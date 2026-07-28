using System;
using JYPPX.CudaSharp.Internal.Interop;

namespace JYPPX.CudaSharp;

/// <summary>Defines a non-zero CUDA grid or block extent. 定义非零 CUDA grid 或 block 范围。</summary>
public readonly struct CudaDim3
{
    /// <summary>Creates a CUDA dimension. 创建 CUDA 维度。</summary>
    public CudaDim3(uint x, uint y = 1, uint z = 1)
    {
        if (x == 0) throw new ArgumentOutOfRangeException(nameof(x));
        if (y == 0) throw new ArgumentOutOfRangeException(nameof(y));
        if (z == 0) throw new ArgumentOutOfRangeException(nameof(z));
        X = x;
        Y = y;
        Z = z;
    }

    /// <summary>Gets the X dimension. 获取 X 维度。</summary>
    public uint X { get; }

    /// <summary>Gets the Y dimension. 获取 Y 维度。</summary>
    public uint Y { get; }

    /// <summary>Gets the Z dimension. 获取 Z 维度。</summary>
    public uint Z { get; }

    internal NativeCudaDim3 ToNative()
    {
        Validate();
        return new NativeCudaDim3 { X = X, Y = Y, Z = Z };
    }

    internal void Validate()
    {
        if (X == 0 || Y == 0 || Z == 0)
        {
            throw new InvalidOperationException("CUDA dimensions must all be greater than zero.");
        }
    }
}
