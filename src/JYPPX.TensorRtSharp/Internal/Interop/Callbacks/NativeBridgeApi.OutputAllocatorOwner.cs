using System;
using JYPPX.TensorRtSharp.Shared.Interop;
using JYPPX.TensorRtSharp.Internal.Handles;

namespace JYPPX.TensorRtSharp.Internal.Interop;

internal static partial class NativeBridgeApi
{
    public static SafeTensorRtObjectHandle CreateOutputAllocatorOwner(
        TensorRtApiLine line,
        TensorRtOutputAllocatorNativeCallback callback,
        IntPtr userState)
    {
        SafeTensorRtObjectHandle owner;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_output_allocator_owner_create(callback, userState, out owner),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_output_allocator_owner_create(callback, userState, out owner),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_output_allocator_owner_create(callback, userState, out owner),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return owner;
    }

    public static bool AttachOutputAllocatorOwner(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner,
        SafeTensorRtObjectHandle context,
        string tensorName)
    {
        ValidateTensorName(tensorName);
        using Utf8Interop.Utf8StringScope tensorNameUtf8 = Utf8Interop.ToNativeString(tensorName);
        int attached;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_output_allocator_owner_attach(owner, context, tensorNameUtf8.Pointer, out attached),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_output_allocator_owner_attach(owner, context, tensorNameUtf8.Pointer, out attached),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_output_allocator_owner_attach(owner, context, tensorNameUtf8.Pointer, out attached),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return attached != 0;
    }

    public static bool DetachOutputAllocatorOwner(TensorRtApiLine line, SafeTensorRtObjectHandle owner)
    {
        int detached;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_output_allocator_owner_detach(owner, out detached),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_output_allocator_owner_detach(owner, out detached),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_output_allocator_owner_detach(owner, out detached),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return detached != 0;
    }

    public static NativeTensorRtOutputAllocatorOwnerInfo GetOutputAllocatorOwnerInfo(
        TensorRtApiLine line,
        SafeTensorRtObjectHandle owner)
    {
        NativeTensorRtOutputAllocatorOwnerInfo info;
        BridgeStatusCode status = line switch
        {
            TensorRtApiLine.TensorRt8 => NativeMethodsTensorRt.jyppx_trt8_output_allocator_owner_get_info(owner, out info),
            TensorRtApiLine.TensorRt10 => NativeMethodsTensorRt.jyppx_trt10_output_allocator_owner_get_info(owner, out info),
            TensorRtApiLine.TensorRt11 => NativeMethodsTensorRt.jyppx_trt11_output_allocator_owner_get_info(owner, out info),
            _ => throw UnsupportedLine()
        };

        NativeStatus.ThrowIfFailed(status);
        return info;
    }
}
