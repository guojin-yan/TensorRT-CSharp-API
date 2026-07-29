using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using JYPPX.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Interop;

namespace JYPPX.TensorRtSharp;

internal readonly struct TensorRtOutputAllocatorRuntimeGateRequest
{
    internal TensorRtOutputAllocatorRuntimeGateRequest(
        string tensorName,
        ulong requestedSize,
        ulong alignment,
        long[] shapeDimensions,
        string reason = "",
        bool hasCurrentMemory = false)
    {
        TensorName = tensorName ?? string.Empty;
        RequestedSize = requestedSize;
        Alignment = alignment;
        ShapeDimensions = shapeDimensions == null ? Array.Empty<long>() : (long[])shapeDimensions.Clone();
        Reason = reason ?? string.Empty;
        HasCurrentMemory = hasCurrentMemory;
    }

    public string TensorName { get; }

    public ulong RequestedSize { get; }

    public ulong Alignment { get; }

    public int ShapeRank => ShapeDimensions.Length;

    public string Reason { get; }

    public bool HasCurrentMemory { get; }

    internal long[] ShapeDimensions { get; }

    internal long GetDimension(int index)
    {
        return index >= 0 && index < ShapeDimensions.Length ? ShapeDimensions[index] : 0L;
    }
}
