using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

/// <summary>
/// Represents a TensorRT dimension vector.
/// 表示一个 TensorRT 维度向量。
/// </summary>
public sealed class TensorRtDims
{
    /// <summary>
    /// Initializes a TensorRT dimension vector.
    /// 初始化一个 TensorRT 维度向量。
    /// </summary>
    /// <param name="values">The dimension values in TensorRT order. 按 TensorRT 顺序排列的维度值。</param>
    public TensorRtDims(int[] values)
    {
        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Length > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(values), "TensorRT dimensions support at most 8 entries.");
        }

        Values = (int[])values.Clone();
    }

    /// <summary>
    /// Gets the number of dimensions.
    /// 获取维度数量。
    /// </summary>
    public int Rank => Values.Length;

    /// <summary>
    /// Gets the dimension values.
    /// 获取维度值。
    /// </summary>
    public int[] Values { get; }

    internal NativeTensorRtDims ToNative()
    {
        int[] dims = new int[8];
        Array.Copy(Values, dims, Values.Length);
        return new NativeTensorRtDims
        {
            NbDims = Values.Length,
            D = dims
        };
    }

    internal static TensorRtDims FromNative(NativeTensorRtDims native)
    {
        int rank = native.NbDims;
        if (rank < 0 || rank > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(native), "Native TensorRT dimensions contain an invalid rank.");
        }

        int[] values = new int[rank];
        if (rank > 0 && native.D != null)
        {
            Array.Copy(native.D, values, rank);
        }

        return new TensorRtDims(values);
    }

    /// <summary>
    /// Formats the dimension vector for diagnostics.
    /// 将维度向量格式化为便于诊断的字符串。
    /// </summary>
    public override string ToString()
    {
        return Values.Length == 0 ? "[]" : "[" + string.Join(", ", Values) + "]";
    }
}
