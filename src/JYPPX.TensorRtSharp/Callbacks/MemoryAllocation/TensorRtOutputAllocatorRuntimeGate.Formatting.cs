using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal sealed partial class TensorRtOutputAllocatorRuntimeGate
{
    private static string OperationName(int operation)
    {
        return operation switch
        {
            NotifyShapeOperation => "notify-shape",
            ReallocateOutputOperation => "reallocate-output",
            _ => "unknown"
        };
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
