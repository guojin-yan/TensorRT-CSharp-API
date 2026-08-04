using System;
using JYPPX.TensorRtSharp.Internal.Handles;
using JYPPX.TensorRtSharp.Shared.Interop;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateGpuAllocatorOwner(
        TensorRtApiLine line,
        TensorRtGpuAllocatorNativeCallback callback,
        IntPtr userState)
    {
        SafeTensorRtObjectHandle owner;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_gpu_allocator_owner_create(callback, userState, out owner),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_gpu_allocator_owner_create(callback, userState, out owner),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_gpu_allocator_owner_create(callback, userState, out owner),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return owner;
    }

    public static bool AttachGpuAllocatorOwnerToRuntime(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        SafeTensorRtObjectHandle runtime)
    {
        int attached;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_gpu_allocator_owner_attach_runtime(owner, runtime, out attached),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_gpu_allocator_owner_attach_runtime(owner, runtime, out attached),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_gpu_allocator_owner_attach_runtime(owner, runtime, out attached),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return attached != 0;
    }

    public static bool AttachGpuAllocatorOwnerToBuilder(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        SafeTensorRtObjectHandle builder)
    {
        int attached;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_gpu_allocator_owner_attach_builder(owner, builder, out attached),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_gpu_allocator_owner_attach_builder(owner, builder, out attached),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_gpu_allocator_owner_attach_builder(owner, builder, out attached),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return attached != 0;
    }

    public static bool DetachGpuAllocatorOwner(TensorRtApiLine line, SafeTensorRtObjectHandle owner)
    {
        int detached;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_gpu_allocator_owner_detach(owner, out detached),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_gpu_allocator_owner_detach(owner, out detached),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_gpu_allocator_owner_detach(owner, out detached),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return detached != 0;
    }

    public static NativeTensorRtGpuAllocatorOwnerInfo GetGpuAllocatorOwnerInfo(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner)
    {
        NativeTensorRtGpuAllocatorOwnerInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_gpu_allocator_owner_get_info(owner, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_gpu_allocator_owner_get_info(owner, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_gpu_allocator_owner_get_info(owner, out info),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }
}
