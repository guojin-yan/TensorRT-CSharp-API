using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents TensorRT 11 dimensions whose extents are reported as 64-bit values.
/// 表示 TensorRT 11 中以 64 位整数报告的维度信息。
/// </summary>
/// <remarks>
/// TensorRT 11 can report larger dimension extents and unknown-rank shapes. This type keeps that information intact instead of narrowing values to <see cref="int"/>.
/// TensorRT 11 可能报告更大的维度 extent，也可能返回 unknown-rank shape；此类型会保留这些信息，而不是强制压缩为 <see cref="int"/>。
/// </remarks>
public sealed class TensorRtDims64
{
    private TensorRtDims64(long[] values, bool isUnknownRank)
    {
        Values = values;
        IsUnknownRank = isUnknownRank;
    }

    /// <summary>
    /// Gets a singleton value that represents TensorRT's unknown-rank shape.
    /// 获取表示 TensorRT unknown-rank shape 的单例值。
    /// </summary>
    public static TensorRtDims64 UnknownRank { get; } = new TensorRtDims64(Array.Empty<long>(), isUnknownRank: true);

    /// <summary>
    /// Creates a rank-known TensorRT 64-bit dimension value.
    /// 创建一个 rank 已知的 TensorRT 64 位维度值。
    /// </summary>
    /// <param name="values">Dimension extents. 维度 extent 数组。</param>
    public TensorRtDims64(long[] values)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length > NativeTensorRtDims64.MaxDimensionCount)
        {
            throw new ArgumentOutOfRangeException(nameof(values), "TensorRT dimensions support at most 8 entries.");
        }

        Values = (long[])values.Clone();
        IsUnknownRank = false;
    }

    /// <summary>
    /// Gets whether TensorRT reported an unknown rank instead of a concrete dimension vector.
    /// 获取 TensorRT 是否报告 unknown-rank，而不是具体的维度向量。
    /// </summary>
    public bool IsUnknownRank { get; }

    /// <summary>
    /// Gets the rank. Unknown-rank shapes return <c>-1</c>.
    /// 获取 rank；unknown-rank shape 返回 <c>-1</c>。
    /// </summary>
    public int Rank => IsUnknownRank ? -1 : Values.Length;

    /// <summary>
    /// Gets a copy-safe 64-bit dimension extent array.
    /// 获取可安全读取的 64 位维度 extent 数组。
    /// </summary>
    public long[] Values { get; }

    internal static TensorRtDims64 FromNative(NativeTensorRtDims64 native)
    {
        if (native.NbDims < 0)
        {
            return UnknownRank;
        }

        if (native.NbDims > NativeTensorRtDims64.MaxDimensionCount)
        {
            throw new ArgumentOutOfRangeException(nameof(native), "Native TensorRT 64-bit dimensions contain an invalid rank.");
        }

        long[] values = new long[native.NbDims];
        if (native.NbDims > 0 && native.D != null)
        {
            Array.Copy(native.D, values, native.NbDims);
        }

        return new TensorRtDims64(values);
    }

    /// <summary>
    /// Returns a compact diagnostic string.
    /// 返回紧凑的诊断字符串。
    /// </summary>
    public override string ToString()
    {
        if (IsUnknownRank)
        {
            return "<unknown-rank>";
        }

        return Values.Length == 0 ? "[]" : "[" + string.Join(", ", Values) + "]";
    }
}
