using System;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtDebugListenerCallbackOwner
{
    private static long GetDimension(long[] dimensions, int index)
    {
        return index >= 0 && index < dimensions.Length ? dimensions[index] : 0L;
    }

    private static string FormatShape(
        int rank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7)
    {
        if (rank <= 0)
        {
            return "[]";
        }

        long[] dimensions = new[] { dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7 };
        int copiedRank = Math.Min(rank, MaxShapeRank);
        string[] values = new string[copiedRank];
        for (int index = 0; index < copiedRank; index++)
        {
            values[index] = dimensions[index].ToString(CultureInfo.InvariantCulture);
        }

        return "[" + string.Join("x", values) + "]";
    }

}
