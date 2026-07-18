using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using JYPPX.CudaSharp.Internal.Handles;

namespace JYPPX.CudaSharp.Internal.Interop;

internal static partial class NativeCudaApi
{
    public static CudaStreamCaptureScalarInfo GetStreamCaptureInfoPtzs(SafeCudaStreamHandle stream)
    {
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_get_capture_info_ptsz_copied_scalars_safe(
            stream,
            out int status,
            out ulong captureId));
        return new CudaStreamCaptureScalarInfo((CudaStreamCaptureStatus)status, captureId);
    }

    public static void UpdateStreamCaptureDependenciesPtzs(
        SafeCudaStreamHandle stream,
        IReadOnlyList<CudaGraphNode> dependencies,
        CudaStreamCaptureDependencyMode mode)
    {
        UIntPtr[] tokens = ToDependencyTokens(dependencies);
        CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_update_capture_dependencies_ptsz_safe(
            stream,
            tokens,
            new UIntPtr((uint)tokens.Length),
            (uint)mode));
    }

    public static void UpdateStreamCaptureDependenciesV2(
        SafeCudaStreamHandle stream,
        IReadOnlyList<CudaGraphNodeDependency> dependencies,
        CudaStreamCaptureDependencyMode mode)
    {
        if (dependencies.Count > 1000000)
        {
            throw new ArgumentOutOfRangeException(nameof(dependencies));
        }

        UIntPtr[] tokens = new UIntPtr[dependencies.Count];
        NativeCudaGraphEdgeData[] edgeData = new NativeCudaGraphEdgeData[dependencies.Count];
        for (int index = 0; index < dependencies.Count; index++)
        {
            tokens[index] = dependencies[index].Node.Token;
            edgeData[index] = dependencies[index].EdgeData.ToNative();
        }

        GCHandle edgeDataHandle = default;
        try
        {
            IntPtr edgeDataPointer = IntPtr.Zero;
            if (edgeData.Length != 0)
            {
                edgeDataHandle = GCHandle.Alloc(edgeData, GCHandleType.Pinned);
                edgeDataPointer = edgeDataHandle.AddrOfPinnedObject();
            }

            CudaNativeStatus.ThrowIfFailed(NativeMethodsCuda.jyppx_cuda_stream_update_capture_dependencies_v2_safe(
                stream,
                tokens,
                edgeDataPointer,
                new UIntPtr((uint)tokens.Length),
                (uint)mode));
        }
        finally
        {
            if (edgeDataHandle.IsAllocated)
            {
                edgeDataHandle.Free();
            }
        }
    }

    private static UIntPtr[] ToDependencyTokens(IReadOnlyList<CudaGraphNode> dependencies)
    {
        if (dependencies == null)
        {
            throw new ArgumentNullException(nameof(dependencies));
        }

        if (dependencies.Count > 1000000)
        {
            throw new ArgumentOutOfRangeException(nameof(dependencies));
        }

        UIntPtr[] tokens = new UIntPtr[dependencies.Count];
        for (int index = 0; index < dependencies.Count; index++)
        {
            tokens[index] = dependencies[index].Token;
        }

        return tokens;
    }
}
