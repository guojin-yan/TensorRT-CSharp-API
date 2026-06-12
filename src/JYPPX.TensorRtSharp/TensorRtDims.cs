using System;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed class TensorRtDims
{
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

    public int Rank => Values.Length;

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

    public override string ToString()
    {
        return Values.Length == 0 ? "[]" : "[" + string.Join(", ", Values) + "]";
    }
}
