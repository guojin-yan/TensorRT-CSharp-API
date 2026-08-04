using System;
using System.Runtime.InteropServices;
using System.Text;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp;

public sealed partial class TensorRtOutputAllocatorCallbackOwner
{
    private const int MaxRuntimeShapeRank = 8;
    private const ulong MaxRuntimeTensorNameBytes = 4096UL;

    [ThreadStatic]
    private static int s_runtimeCallbackDepth;

    internal static bool IsExecutingRuntimeCallbackOnCurrentThread => s_runtimeCallbackDepth > 0;

    private static BridgeStatusCode InvokeManagedOutputAllocator(
        uint line,
        int callbackKind,
        IntPtr tensorName,
        UIntPtr tensorNameLength,
        ulong requestedSize,
        ulong alignment,
        int hasCurrentMemory,
        int hasStream,
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7,
        out int shouldAllocate,
        IntPtr userState)
    {
        shouldAllocate = 0;
        s_runtimeCallbackDepth++;
        try
        {
            if (userState == IntPtr.Zero)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            RuntimeCallbackState? state = GCHandle.FromIntPtr(userState).Target as RuntimeCallbackState;
            if (state == null || line != (uint)state.Line)
            {
                return BridgeStatusCode.InvalidState;
            }

            TensorRtOutputAllocatorCallbackKind kind = Enum.IsDefined(
                typeof(TensorRtOutputAllocatorCallbackKind),
                callbackKind)
                ? (TensorRtOutputAllocatorCallbackKind)callbackKind
                : TensorRtOutputAllocatorCallbackKind.Unknown;
            if (kind == TensorRtOutputAllocatorCallbackKind.Unknown)
            {
                return BridgeStatusCode.InvalidArgument;
            }

            string copiedName = DecodeRuntimeTensorName(tensorName, tensorNameLength);
            long[] dimensions = CopyRuntimeShape(
                shapeRank,
                dim0,
                dim1,
                dim2,
                dim3,
                dim4,
                dim5,
                dim6,
                dim7);
            TensorRtOutputAllocatorCallbackRequest request = new TensorRtOutputAllocatorCallbackRequest(
                kind,
                copiedName,
                requestedSize,
                alignment,
                dimensions,
                kind == TensorRtOutputAllocatorCallbackKind.NotifyShape
                    ? "TensorRT IOutputAllocator::notifyShape callback."
                    : "TensorRT IOutputAllocator::reallocateOutput callback.",
                hasCurrentMemory != 0,
                hasStream != 0);

            bool accepted = state.Handler(request);
            shouldAllocate = kind == TensorRtOutputAllocatorCallbackKind.NotifyShape || accepted ? 1 : 0;
            return BridgeStatusCode.Ok;
        }
        catch
        {
            shouldAllocate = 0;
            return BridgeStatusCode.InvalidState;
        }
        finally
        {
            s_runtimeCallbackDepth--;
        }
    }

    private static string DecodeRuntimeTensorName(IntPtr value, UIntPtr length)
    {
        if (value == IntPtr.Zero || length == UIntPtr.Zero)
        {
            return string.Empty;
        }

        ulong byteLength = length.ToUInt64();
        if (byteLength > MaxRuntimeTensorNameBytes)
        {
            throw new InvalidOperationException("TensorRT output tensor name exceeded the copied metadata limit.");
        }

        byte[] buffer = new byte[checked((int)byteLength)];
        Marshal.Copy(value, buffer, 0, buffer.Length);
        return Encoding.UTF8.GetString(buffer);
    }

    private static long[] CopyRuntimeShape(
        int shapeRank,
        long dim0,
        long dim1,
        long dim2,
        long dim3,
        long dim4,
        long dim5,
        long dim6,
        long dim7)
    {
        if (shapeRank < 0 || shapeRank > MaxRuntimeShapeRank)
        {
            throw new InvalidOperationException("TensorRT output allocator reported an invalid shape rank.");
        }

        long[] allDimensions = { dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7 };
        long[] dimensions = new long[shapeRank];
        Array.Copy(allDimensions, dimensions, shapeRank);
        return dimensions;
    }

    private sealed class RuntimeCallbackState
    {
        public RuntimeCallbackState(TensorRtApiLine line, TensorRtOutputAllocatorHandler handler)
        {
            Line = line;
            Handler = handler;
        }

        public TensorRtApiLine Line { get; }

        public TensorRtOutputAllocatorHandler Handler { get; }
    }
}
